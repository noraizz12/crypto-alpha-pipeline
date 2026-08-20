"""
Tardis data downloader and processor for cryptocurrency market data.

This module provides functionality to:
1. Download historical market data from Tardis.dev API
2. Process raw data into 1-minute OHLCV bars
3. Extract funding rate information for perpetual futures
4. Upload processed data to S3 for archival
5. Generate pre-bar data for further processing

The Tardis class handles:
- Downloading trades, order book snapshots, and derivative ticker data
- Aggregating tick data into 1-minute bars with volume, spread, and microstructure metrics
- Managing data availability windows for different symbols
- Parallel processing for efficient data downloads
- S3 integration for data storage and retrieval
"""

import json
import logging
import os
import shutil
from datetime import datetime as dt, date
from datetime import timedelta as td
from multiprocessing.dummy import Pool
from typing import List, Optional, Literal

import pandas as pd
import requests
from tardis_dev import datasets

from lib.util import set_index
from lib.util.aws import BUCKET, get_bucket, upload_file
from lib.external.binance_utils import get_exchange_info
from lib.calcs import Calcs
from lib.data.dataloader import DataLoader
from lib.util.dataframes import concat, get_min_max_ts, merge_on_index, shrink_floats, make_symbol_venue
from lib.util.files import safe_mkdir
from lib.util.time_util import beginning_of_day, date_range, date_to_str, to_datetime, today_date, date_str_to_date, date_to_start_dt
from lib.util.directory import DATA_DIR, dir_manager
from lib.util.util import TARDIS_EXCHANGE, log_and_raise, unique_list, SYMBOL_PAIR, SYMBOL_VENUE, SYMBOL_BASE, symbol_venue_to_symbol
from lib.universe import Universe

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Tardis data type constants
# accepted data types - 'datasets.symbols[].dataTypes' field in https://api.tardis.dev/v1/exchanges/deribit
DATA_TYPES = ['trades', 'book_snapshot_5', 'derivative_ticker']  # Default data types to download
FILE_FORMAT = 'csv'  # Tardis file format
IDX = ['ts', 'symbol', 'venue']  # Standard index columns for bar data
UNDEFINED_AVAILABILITY_TO = '2100-01-01T00:00:00.000Z'  # Default end date for undefined availability
TARDIS_BAR_START_DATE = date_str_to_date("20230101")

class Tardis:
    """
    Handles downloading and processing market data from Tardis.dev API.
    
    This class manages the entire workflow of downloading raw market data,
    processing it into standardized bars, and optionally uploading to S3.
    """

    def __init__(
            self,
            api_key: str,
            config: dict,
            s3: bool,
            pool_size: int = 1,
            symbols: Optional[List[str]] = None,
            overwrite: bool = False,
            debug: bool = False,
            backfill: bool = False,
            force: bool = False,
            data_types: Optional[List[str]] = None,
            funding_output_dir: str = dir_manager.FUNDING_DIR
            ):
        """
        Initialize Tardis data downloader.
        
        Args:
            api_key: Tardis API key for authentication
            config: Configuration dictionary with trading parameters
            s3: Whether to upload/download data from S3
            pool_size: Number of parallel download workers (default: 1)
            universe: List of perpetual futures symbols to download (default: load from universe)
            overwrite: Whether to overwrite existing files (default: False)
            debug: Enable debug mode with verbose output (default: False)
            backfill: Skip metadata and exchange info updates during backfill (default: False)
            force: Continue processing even if some downloads fail (default: False)
            data_types: List of data types to download (default: trades, book_snapshot_5, derivative_ticker)
            funding_output_dir: Directory to save funding rate data (default: configured funding directory)
        """
        self.api_key = api_key
        self.debug = debug
        self.config = config
        self.backfill = backfill
        self.data_loader = DataLoader(config=self.config)
        self.universe = Universe(config=self.config, debug=self.debug)
        self.symbols = symbols
        self.calcs = Calcs(config)
        self.s3 = s3
        self.bucket = get_bucket(BUCKET) if s3 else None
        self.pool_size = pool_size
        self.overwrite = overwrite
        self.force = force
        self.availability_dict = {}
        self.metadata_dict = {}
        self.data_types = DATA_TYPES if data_types is None else data_types
        self.funding_output_dir = funding_output_dir
        self.symbol_funding_dfs = []
        if self.symbols is not None:
            logger.info(f'Downloading tardis data for {len(self.symbols)} symbols for {self.data_types}')

        if self.backfill:
            assert self.symbols is not None

    @staticmethod
    def _tardis_data_file_name(exchange: str, data_type: str, date: date, symbol: str, file_format: str = FILE_FORMAT) -> str:
        """Generate standardized filename for Tardis data files.
        
        Creates consistent filename format for storing and retrieving Tardis data.
        
        Args:
            exchange: Exchange name (e.g., 'binance-futures')
            data_type: Type of data ('trades', 'book_snapshot_5', 'derivative_ticker')
            date: Date of the data
            symbol: Trading symbol (e.g., 'BTCUSDT')
            file_format: File format, default 'csv'
            
        Returns:
            Filename string in format: {data_type}.{symbol}.{exchange}.{date}.{format}.gz
            
        Example:
            >>> filename = Tardis._tardis_data_file_name(
            ...     'binance-futures', 'trades', date(2024, 1, 1), 'BTCUSDT'
            ... )
            >>> # Returns: 'trades.BTCUSDT.binance-futures.20240101.csv.gz'
        """
        return f"{data_type}.{symbol}.{exchange}.{date_to_str(date)}.{file_format}.gz"

    def _download_symbol(self, symbol: str, data_type: str, exchange: str, download_date: date, tardis_dir: str) -> bool:
        """Download raw data for a symbol from Tardis API.
        
        Handles downloading from Tardis API with S3 caching. Checks S3 first,
        falls back to local files, and downloads from Tardis if needed.
        
        Args:
            symbol: Trading symbol to download
            data_type: Type of data to download ('trades', 'book_snapshot_5', etc.)
            exchange: Exchange name
            download_date: Date to download data for
            tardis_dir: Local directory for storing raw data
            
        Returns:
            True if download successful or file already exists, False on error
            
        Side Effects:
            - Downloads file from Tardis API if not cached
            - Uploads to S3 if s3 mode enabled
            - Updates availability dict on API errors
        """
        download_date_str = date_to_str(download_date)
        tardis_data_dir = f"{DATA_DIR}/{tardis_dir}"
        safe_mkdir(tardis_data_dir)
        filename = self._tardis_data_file_name(exchange=exchange, data_type=data_type, date=download_date, symbol=symbol)
        full_filename = f"{tardis_data_dir}/{filename}"
        s3_name = f"{tardis_dir}/{filename}"

        if self.s3 and not self.overwrite:
            missing_s3 = False
            logger.info(f"Downloading {s3_name} from S3")
            try:
                self.bucket.download_file(s3_name, full_filename)
            except Exception as e:
                logger.warning(e)
                logger.info(f"Could not download {s3_name} from s3")
                missing_s3 = True

            local_file_found = os.path.isfile(full_filename)
            logger.info(f"Looking for local file {full_filename}, local: {local_file_found} s3: {not missing_s3}")
            if missing_s3 and local_file_found:
                logger.info(f"Tardis file local, but not on s3! Uploading to {BUCKET}")
                upload_file(full_filename, s3_name, BUCKET)

        if os.path.isfile(full_filename) and not self.overwrite:
            logger.info(f"Already downloaded {full_filename}")
            return True

        logger.info(f"Downloading from tardis {symbol} {exchange} {data_type} on {download_date_str}")
        try:
            datasets.download(
                exchange=exchange,
                data_types=[data_type],
                from_date=download_date_str,
                to_date=download_date_str,
                symbols=[symbol],
                api_key=self.api_key,
                download_dir=tardis_data_dir,
                format=FILE_FORMAT,
                get_filename=self._tardis_data_file_name,
            )
        except Exception as e:
            logger.warning(e)
            logger.warning(f"download failed for {symbol} for {download_date_str}")
            self._update_availability_list(symbol, str(e))
            return False

        if self.s3:
            logger.info(f"Uploading to {BUCKET}")
            upload_file(full_filename, s3_name, BUCKET)

        return True


    def _get_prebar_filename(self, symbol: str, exchange: str, download_date: date) -> str:
        dir_name = f"{dir_manager.PREBAR_DIR}/tardis/{date_to_str(download_date)}/{exchange}"
        full_filename = os.path.join(dir_name, f"prebars.tardis.{exchange}.{date_to_str(download_date)}.{symbol}.parquet")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        return full_filename


    def _dump_prebars_data(self, bar_df: pd.DataFrame, symbol: str, exchange: str, download_date: date) -> None:
        """Save pre-bar data for debugging and analysis.
        
        Stores intermediate bar data before final processing. Useful for
        debugging data issues and analyzing raw bar generation.
        
        Args:
            bar_df: DataFrame with generated bar data
            symbol: Trading symbol
            exchange: Exchange name
            download_date: Date of the data
            
        File Structure:
            {PREBAR_DIR}/tardis/{YYYYMMDD}/{exchange}/prebars.tardis.{exchange}.{date}.{symbol}.parquet
        """

        bar_df = bar_df.reset_index()
        bar_df = bar_df.drop(columns=['symbol', 'venue'])
        bar_df = bar_df.set_index('ts')

        bar_cnt = len(bar_df)
        if bar_cnt != 1440:
            if bar_cnt == 0:
                raise log_and_raise(f"Empty Dataframe passed")
            elif bar_cnt > 1440:
                raise log_and_raise(f"Dataframe passed has more than 1440 rows", df=bar_df)
            else:
                bar_df = self._reindex(bar_df)

        full_filename = self._get_prebar_filename(symbol, exchange, download_date)
        logger.info(f"Writing {full_filename}")
        bar_df.to_parquet(full_filename)


    def _update_availability_list(self, symbol: str, msg: str) -> None:
        """Update symbol availability based on Tardis API error messages.
        
        Parses Tardis API error responses to extract data availability windows.
        This prevents repeated failed requests for symbols outside their data range.
        
        Args:
            symbol: Trading symbol
            msg: Error message from Tardis API containing availability info
            
        Side Effects:
            Updates self.availability_dict with (start_date, end_date) tuple
            
        Note:
            Expects JSON error format with datasetInfo.availableSince/To fields
        """
        try:
            jsonstr = msg[msg.index('{'):]
            msg = json.loads(jsonstr)
            available_at = to_datetime(msg['datasetInfo']['availableSince']).astimezone(None).date()
            available_to = to_datetime(msg['datasetInfo']['availableTo']).astimezone(None).date()
            logger.info(f"not requesting {symbol} outside {available_at} - {available_to}")
            self.availability_dict[symbol] = (available_at, available_to)
        except Exception as e:
            logger.error(f"Could not parse tardis json... {e}")

    def _download_metadata_info(self, symbol: str, exchange: str) -> None:
        """Download instrument metadata from Tardis API.
        
        Fetches detailed instrument information including availability dates,
        contract specifications, and other metadata for perpetual futures.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            
        Side Effects:
            - Updates self.availability_dict with availability window
            - Updates self.metadata_dict with full metadata
            - Logs errors but doesn't raise (for resilience)
            
        API Filter:
            - type: 'perpetual'
            - quoteCurrency: 'USDT'
        """
        filters = {"type": "perpetual", 'quoteCurrency': 'USDT'}
        encoded_filters = requests.utils.quote(json.dumps(filters))
        headers = {'Authorization': f'Bearer {self.api_key}'}
        url = f"https://api.tardis.dev/v1/instruments/{exchange}/{symbol}?filter={encoded_filters}"

        try:
            result = requests.get(url, headers=headers, timeout=5)
        except requests.exceptions.RequestException as e:
            logger.error(f"fail to request fetching metadata for {symbol}: {e}")
            return

        try:
            response = result.json()
        except json.JSONDecodeError as e:
            logger.error(f"fail to decode JSON response for {symbol}: {e}")
            return

        if response.get("code"):
            logger.error(f"fail to get instrument meta response for {exchange} {symbol} since {response}")
        else:
            available_at = dt.strptime(response['availableSince'], "%Y-%m-%dT%H:%M:%S.%f%z").astimezone(None).date()
            # since it's non inclusive, we minus one day to make it fit our convention as inclusive, while we still save the raw response in metadata_dict
            available_to = dt.strptime(response.get('availableTo', UNDEFINED_AVAILABILITY_TO), "%Y-%m-%dT%H:%M:%S.%f%z").astimezone(None).date() - td(days=1)
            logger.info(f"getting {symbol} availability {available_at} - {available_to}")
            self.availability_dict[symbol] = (available_at, available_to)
            self.metadata_dict[symbol] = response


    def _merge_and_dump_funding_data(self, funding_date: date) -> None:
        """Merge and save funding rate data for all symbols.
        
        Combines individual symbol funding DataFrames into a single file
        for the given date. Used for funding rate analysis and PnL calculations.
        
        Args:
            funding_date: Date to merge funding data for
            
        Raises:
            RuntimeError: If no funding data available
            
        Output Format:
            {funding_dir}/funding.{YYYYMMDD}.parquet
            
        Note:
            Expects exactly 1440 minutes (24 hours) of data per symbol
        """
        if len(self.symbol_funding_dfs) == 0:
            raise log_and_raise(f"no funding dataframe generated for {funding_date}, no funding 1_min files?")

        funding_df = concat(self.symbol_funding_dfs, check=False).sort_index()
        assert len(funding_df) % 1440 == 0
        if self.debug:
            print(funding_df.head().to_markdown())
            return
        date_str = date_to_str(funding_date)
        funding_df = shrink_floats(funding_df)
        funding_df = set_index(make_symbol_venue(funding_df.reset_index()))

        filename = f"{self.funding_output_dir}/funding.{date_str}.parquet"
        logger.info(f"Writing {filename}")

        funding_df.to_parquet(filename)

    def _download_and_dump_exchange_info(self) -> None:
        """Download and save current Binance exchange information.
        
        Fetches exchange info including trading rules, filters, and contract
        specifications. Saved with current date for historical tracking.
        
        Output:
            {EXCH_INFO_DIR}/exchange.{YYYYMMDD}.parquet
            
        Note:
            Used to track changes in exchange rules and contract specs over time
        """
        if not os.path.exists(dir_manager.EXCH_INFO_DIR):
            os.makedirs(dir_manager.EXCH_INFO_DIR)
        today_str = date_to_str()
        exch_info_file = os.path.join(dir_manager.EXCH_INFO_DIR, f'exchange.{today_str}.parquet')
        logger.info(f"Downloading Exchange Info for {today_str}")
        exch_info_dict = get_exchange_info()
        exch_info_df = pd.DataFrame(exch_info_dict['symbols'])
        exch_info_df.to_parquet(exch_info_file)
        logger.info(f"Exchange Info successfully saved to {exch_info_file}")

    def _download_and_dump_current_binance_metadata(self, exchange: str, symbols: List[str]) -> None:
        """Download metadata for all symbols from Tardis.
        
        Fetches availability windows and instrument details for all symbols
        in the universe. Retries failed requests up to 3 times.
        
        Args:
            exchange: Exchange name
            
        Raises:
            RuntimeError: If metadata download fails after retries (unless force=True)
            
        Output:
            {BINANCE_META_DIR}/meta.{YYYYMMDD}.parquet
            
        Note:
            With force=True, continues processing even if some symbols fail
        """
        if not os.path.exists(dir_manager.BINANCE_META_DIR):
            os.makedirs(dir_manager.BINANCE_META_DIR)

        for symbol in symbols:
            logger.info(f"Downloading Binance Metadata for {symbol}")
            max_retries = 3
            attempt = 0
            error_msg = ''
            while attempt < max_retries:
                try:
                    self._download_metadata_info(symbol, exchange)
                    attempt = max_retries + 1
                except Exception as e:
                    error_msg += str(e) + ', '
                    attempt += 1
            if attempt == max_retries:
                # force allows it to not fail and continue processing
                if self.force:
                    logger.error(f"fail to download metadata {max_retries} times for {symbol}: {error_msg}")
                else:
                    raise log_and_raise(f"fail to download metadata {max_retries} times for {symbol}: {error_msg}")

        metadata_df = pd.DataFrame.from_dict(self.metadata_dict, orient='index')
        metadata_df = metadata_df.reset_index().rename(columns={'index': 'symbol'})
        metadata_filename = os.path.join(dir_manager.BINANCE_META_DIR, f'meta.{date_to_str()}.parquet')
        metadata_df.to_parquet(metadata_filename, index=False)
        logger.info(f"Metadata successfully saved to {metadata_filename}")

    def _generate_1m_fundings_for_symbol(self, symbol: str, exchange: str, download_date: date, tardis_dir: str) -> None:
        """Generate 1-minute funding rate data from derivative ticker files.
        
        Processes raw derivative ticker data to extract funding rates, open interest,
        and index prices. Resamples to 1-minute intervals with forward fill.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            download_date: Date to process
            tardis_dir: Directory containing raw data
            
        Side Effects:
            Appends processed funding DataFrame to self.symbol_funding_dfs
            
        Aggregated Fields:
            - funding_timestamp: Next funding time
            - funding_rate: Current funding rate
            - open_interest: Open interest in contracts
            - last_price: Last traded price
            - index_price: Underlying index price
            - mark_price: Mark price for margining
        """
        logger.info(f"Generating funding bars from raw data for {symbol} {exchange}, {download_date}")

        tardis_data_dir = f"{DATA_DIR}/{tardis_dir}"
        funding_file = self._tardis_data_file_name(exchange=exchange, data_type='derivative_ticker', symbol=symbol, date=download_date)

        tardis_funding_file = f"{tardis_data_dir}/{funding_file}"
        if not os.path.isfile(tardis_funding_file):
            logger.warning(f"{tardis_funding_file} not found....")
            return

        if os.path.getsize(tardis_funding_file) == 0:
            logger.warning(f"{tardis_funding_file} size is 0")
            return

        try:
            funding_df = pd.read_csv(tardis_funding_file, dtype=
                                     {'local_timestamp': 'Int64',
                                       'timestamp': 'Int64',
                                       'funding_timestamp': 'Int64',
                                       'funding_rate': 'float32',
                                       'open_interest': 'float32',
                                       'last_price': 'float32',
                                       'index_price': 'float32',
                                       'mark_price': 'float32',
                                       'symbol': 'string',
                                       'exchange': 'string'
                                     })
        except pd.errors.EmptyDataError:
            logger.info(f"{symbol=},No data to parse in file {tardis_funding_file}. The file may be empty or corrupted.")
            return
        except Exception as e:
            logger.info(f"{symbol=},An unexpected error occurred while processing file {tardis_funding_file}: {e}")
            return

        funding_df['local_timestamp'] = to_datetime(funding_df['local_timestamp'], unit='us')
        funding_df['timestamp'] = to_datetime(funding_df['timestamp'], unit='us')
        funding_df['funding_timestamp'] = to_datetime(funding_df['funding_timestamp'], unit='us')
        funding_df['ts'] = funding_df['local_timestamp']

        funding_min_df = funding_df.set_index('ts') \
            .resample('60s', label='right') \
            .agg(funding_timestamp=('funding_timestamp', 'last'),
                 funding_rate=('funding_rate', 'last'),
                 open_interest=('open_interest', 'last'),
                 last_price=('last_price', 'last'),
                 index_price=('index_price', 'last'),
                 mark_price=('mark_price', 'last'),
                 symbol=('symbol', 'last'),
                 venue=('exchange', 'last')) \
            .reset_index().set_index(IDX)


        #XXX replace with fillin_idx from dataframes.py
        min_ts, _ = get_min_max_ts(funding_min_df)
        start = beginning_of_day(min_ts) + td(minutes=1)
        end = beginning_of_day(min_ts) + td(days=1)
        rng = pd.date_range(start=start, end=end, freq='1Min', tz='UTC')
        funding_min_df = funding_min_df.reindex(
            pd.MultiIndex.from_product([rng, funding_min_df.index.levels[1], funding_min_df.index.levels[2]], names=['ts', 'symbol', 'venue'])
        )
        self.symbol_funding_dfs.append(funding_min_df)


    def _reindex(self, df: pd.DataFrame) -> pd.DataFrame:
        row_cnt = len(df)
        start_ts, end_ts = get_min_max_ts(df)
        idx_start_ts = date_to_start_dt(start_ts.date()) + td(minutes=1)
        idx_end_ts = date_to_start_dt(end_ts.date() + td(days=1))

        logger.info(f"Reindexing from {idx_start_ts} -> {idx_end_ts}")

        expected_index = pd.date_range(
            start=idx_start_ts,
            end=idx_end_ts,
            freq='1min',
            inclusive='both',  # Include both start and end
            tz='UTC'
        )

        #XXX a bit dangerous!!!  not really right but i want my 1440
        df = df.reindex(expected_index, method='ffill')

        logger.info(f"Reindexing dataframe {row_cnt} -> {len(df)}")
        assert len(df) % 1440 == 0
        return df

    def _generate_prebars_for_symbol(self, symbol: str, exchange: str, download_date: date, tardis_dir: str, barfile_dir: str):
        """
        Generate 1-minute OHLCV bars from raw Tardis trades and order book data.

        This method:
        1. Loads trades and book snapshot CSV files
        2. Aggregates tick data into 1-minute bars
        3. Calculates OHLCV, volume, spread, and microstructure metrics
        4. Saves bars as parquet files and optionally uploads to S3

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            exchange: Exchange name (e.g., 'binance-futures')
            download_date: Date to process
            tardis_dir: Directory containing raw Tardis data
            barfile_dir: Directory to save processed bar files
        """
        logger.info(f"Generating bars from raw data for {symbol} {exchange}, {download_date}")

        tardis_data_dir = f"{DATA_DIR}/{tardis_dir}"

        book_file = self._tardis_data_file_name(exchange=exchange, data_type='book_snapshot_5', symbol=symbol, date=download_date)
        trades_file = self._tardis_data_file_name(exchange=exchange, data_type='trades', symbol=symbol, date=download_date)

        logger.info(f"Aggregating book data for {symbol} {exchange} {download_date}")
        tardis_book_file = f"{tardis_data_dir}/{book_file}"
        if not os.path.isfile(tardis_book_file):
            logger.warning(f"{tardis_book_file} not found....")
            return
        if os.path.getsize(tardis_book_file) == 0:
            logger.warning(f"{tardis_book_file} size is 0")
            return

        try:
            book_df = pd.read_csv(tardis_book_file)
        except pd.errors.EmptyDataError:
            logger.error(f"{symbol=},No data to parse in file {tardis_book_file}. The file may be empty or corrupted.")
            return
        except Exception as e:
            logger.error(f"{symbol=},An unexpected error occurred while processing file {tardis_book_file}: {e}")
            return

        book_df['local_timestamp'] = to_datetime(book_df['local_timestamp'], unit='us')
        book_df['timestamp'] = to_datetime(book_df['timestamp'], unit='us')
        book_df['latency'] = (book_df['local_timestamp'] - book_df['timestamp']).dt.total_seconds() * 1_000_000

        #this is a hard choice.  it looks like tardis relies on local_timestamp though
        book_df['ts'] = book_df['local_timestamp']
        book_df['mid'] = .5 * (book_df['bids[0].price'] + book_df['asks[0].price'])

        # not just inside market!
        book_df['bid_sz'] = book_df['bids[0].amount'] + book_df['bids[1].amount'] + book_df['bids[2].amount'] + book_df['bids[3].amount'] + book_df['bids[4].amount']
        book_df['ask_sz'] = book_df['asks[0].amount'] + book_df['asks[1].amount'] + book_df['asks[2].amount'] + book_df['asks[3].amount'] + book_df['asks[4].amount']
        book_df['spread'] = book_df['asks[0].price'] - book_df['bids[0].price']
        book_df['close_wgt_mid'] = (book_df['bids[0].price'] * book_df['bid_sz'] + book_df['asks[0].price'] * book_df['ask_sz']) / (book_df['bid_sz'] + book_df['ask_sz'])

        book_bar_df = book_df.set_index('ts') \
            .resample('60s', label='right') \
            .agg(mid_std=('mid', 'std'),
                 open_mid=('mid', 'first'),
                 close_mid=('mid', 'last'),
                 high_mid=('mid', 'max'),
                 low_mid=('mid', 'min'),
                 update_cnt=('mid', 'count'),
                 spread_avg=('spread', 'mean'),
                 bid_sz_avg=('bid_sz', 'mean'),
                 ask_sz_avg=('ask_sz', 'mean'),
                 close_wgt_mid=('close_wgt_mid', 'last'),
                 book_latency=('latency', 'mean'),
                 symbol=('symbol', 'last'),
                 venue=('exchange', 'last')) \
            .reset_index().set_index(IDX)

        if len(book_bar_df) < 1440:
            symbol = book_bar_df.index.get_level_values('symbol')[0]
            venue = book_bar_df.index.get_level_values('venue')[0]

            # Drop extra levels and reindex
            book_bar_df = book_bar_df.droplevel(['symbol', 'venue'])

            start_ts = book_bar_df.index[0]
            start_of_day = start_ts.normalize() + td(minutes=1)
            end_of_day = start_ts.normalize() + td(days=1)
            full_index = pd.date_range(start=start_of_day, end=end_of_day, freq='60s')

            book_bar_df = book_bar_df.reindex(full_index)
            book_bar_df.index.name = 'ts'

            # Restore the MultiIndex
            book_bar_df['symbol'] = symbol
            book_bar_df['venue'] = venue
            book_bar_df = book_bar_df.set_index(['symbol', 'venue'], append=True)
            book_bar_df = book_bar_df.reorder_levels(['ts', 'symbol', 'venue'])

        if len(book_bar_df) != 1440:
            raise log_and_raise(f"Bad row count in dataframe {len(book_bar_df)}", df=book_df)

        if len(book_bar_df.dropna(subset=['close_mid'])) == 0:
            logger.warning(f"No non-nan price data for {symbol} {download_date}")
            return

        book_bar_df = book_bar_df[['open_mid', 'close_mid', 'mid_std', 'update_cnt', 'spread_avg', 'bid_sz_avg', 'ask_sz_avg', 'high_mid', 'low_mid', 'close_wgt_mid', 'book_latency']]
        book_bar_df = self.calcs.calc_logret(book_bar_df)

        logger.info(f"Aggregating trades for {symbol} {exchange} {download_date}")
        tardis_trades_file = f"{tardis_data_dir}/{trades_file}"
        try:
            trades_df = pd.read_csv(tardis_trades_file)
        except Exception as e:
            logger.error(f"Could not read trades file for {symbol}: {tardis_trades_file}")
            logger.error(e)
            return

        trades_df['local_timestamp'] = to_datetime(trades_df['local_timestamp'], unit='us')
        trades_df['timestamp'] = to_datetime(trades_df['timestamp'], unit='us')
        trades_df['latency'] = (trades_df['local_timestamp'] - trades_df['timestamp']).dt.total_seconds() * 1_000_000
        trades_df['ts'] = trades_df['timestamp']
        trades_df['notional'] = trades_df['price'] * trades_df['amount']
        trades_df = pd.concat([book_df[['ts', 'mid']], trades_df]).sort_values(by='ts')
        trades_df['mid'] = trades_df['mid'].ffill()
        trades_df = trades_df[~trades_df['notional'].isna()]
        trades_df['bid_trade_dollars'] = 0.0
        trades_df['ask_trade_dollars'] = 0.0
        trades_df.loc[trades_df['price'] <= trades_df['mid'], 'bid_trade_dollars'] = trades_df['notional']
        trades_df.loc[trades_df['price'] > trades_df['mid'], 'ask_trade_dollars'] = trades_df['notional']

        trades_bar_df = trades_df.set_index('ts') \
            .resample('60s', label='right') \
            .agg(high_trade=('price', 'max'),
                 low_trade=('price', 'min'),
                 close_trade=('price', 'last'),
                 dvolume=('notional', 'sum'),
                 trade_cnt=('price', 'count'),
                 volume=('amount', 'sum'),
                 bid_trade_dollars=('bid_trade_dollars', 'sum'),
                 ask_trade_dollars=('ask_trade_dollars', 'sum'),
                 trade_latency=('latency', 'mean'),
                 symbol=('symbol', 'last'),
                 venue=('exchange', 'last'),
                 ).reset_index().set_index(IDX)
        trades_bar_df = self.calcs.calc_vwap(trades_bar_df)
        bar_df = merge_on_index(book_bar_df, trades_bar_df)
        assert len(bar_df) % 1440 == 0

        bar_df['update_cnt'] = bar_df['update_cnt'].astype('Int32')
        bar_df['trade_cnt'] = bar_df['trade_cnt'].astype('Int32')
        for col in ['dvolume', 'volume', 'bid_trade_dollars', 'ask_trade_dollars', 'trade_cnt', 'update_cnt']:
            bar_df[col] = bar_df[col].fillna(0)

        if self.debug:
            print(bar_df)
            return

        bar_df = shrink_floats(bar_df)
        self._dump_prebars_data(bar_df, symbol, exchange, download_date)

    def _download_files_and_generate_bars(self, symbol_venue: str, download_date: date, barfile_dir: str, exchange: str, tardis_dir: str) -> None:
        """Download raw data and generate bars for a single symbol.
        
        Coordinates downloading all required data types for a symbol and
        generating the corresponding bar and funding data. Respects
        availability windows to avoid unnecessary API calls.
        
        Args:
            symbol_venue: Trading symbol to process
            download_date: Date to download and process
            barfile_dir: Directory for output bar files
            exchange: Exchange name
            tardis_dir: Directory for raw Tardis data
            
        Process Flow:
            1. Check availability window, skip if outside range
            2. Download each data type (trades, book_snapshot_5, derivative_ticker)
            3. Generate 1-minute bars if trades and book data available
            4. Generate funding data if derivative_ticker available
            
        Note:
            Stops downloading remaining data types if any download fails
        """
        logger.info(f"Downloading tardis data and generating bars for {symbol_venue}")

        availability = self.availability_dict.get(symbol_venue)
        if availability is not None:
            available_at, available_to = availability
            if download_date > available_to:
                logger.info(f"Skipping {symbol_venue} after {available_to}")
                return

            if download_date < available_at:
                logger.info(f"Skipping {symbol_venue} until {available_at}")
                return

            logger.info(f"Removing {symbol_venue} from availability blacklist...")
            del self.availability_dict[symbol_venue]

        symbol = symbol_venue_to_symbol(symbol_venue)
        for data_type in self.data_types:
            success = self._download_symbol(symbol=symbol, data_type=data_type, exchange=exchange, download_date=download_date, tardis_dir=tardis_dir)
            if not success:
                logger.warning(f"Failed to download {data_type}, not attempting other types...")
                break

        if {'trades', 'book_snapshot_5'}.issubset(self.data_types):
            self._generate_prebars_for_symbol(symbol=symbol_venue, exchange=exchange, download_date=download_date, tardis_dir=tardis_dir, barfile_dir=barfile_dir)
        if 'derivative_ticker' in self.data_types:
            self._generate_1m_fundings_for_symbol(symbol=symbol_venue, exchange=exchange, download_date=download_date, tardis_dir=tardis_dir)


    def _get_big_universe(self, download_date: date, symbol_type: Literal[SYMBOL_BASE, SYMBOL_PAIR, SYMBOL_VENUE] ) -> List[str]:
        logger.info(f"Getting big universe for {symbol_type}")
        symbols = []
        mktcap_symbol_venues = self.universe.load_universe_symbols(
            universe_source='marketcap',
            universe_date=download_date,
            symbol_type=symbol_type
        )
        symbols += mktcap_symbol_venues
        previous_symbol_venues = self.universe.load_universe_symbols(
            universe_source='file',
            universe_date=None, #get latest
            symbol_type=symbol_type
        )
        symbols += previous_symbol_venues
        symbols = sorted(unique_list(symbols))
        return symbols


    def download_files(self, start_date: date, end_date: date, exchange: str = TARDIS_EXCHANGE) -> None:
        """Main entry point to download and process Tardis data.

        Orchestrates the complete data pipeline:
        1. Downloads metadata (unless backfill mode)
        2. Downloads raw data files for each symbol and date
        3. Generates 1-minute bars from trades and book data
        4. Processes funding rate data
        5. Optionally uploads to S3 and cleans up local files

        Args:
            start_date: First date to download (inclusive)
            end_date: Last date to download (inclusive)
            exchange: Exchange to download from (default: configured exchange)

        Raises:
            RuntimeError: If end_date < start_date

        Note:
            - Uses multiprocessing if pool_size > 1
            - Removes local files after S3 upload to save space
            - Skips metadata updates in backfill mode for efficiency
        """
        if end_date < start_date:
            raise log_and_raise("end_date before start_date!")

        symbols = self._get_big_universe(today_date(), symbol_type=SYMBOL_PAIR) if self.symbols is None else self.symbols
        if not self.backfill:
            self._download_and_dump_current_binance_metadata(exchange, symbols=symbols)
            self._download_and_dump_exchange_info()

        for download_date in date_range(start_date, end_date):
            logger.info(f"Downloading files for {download_date} with {len(symbols)} symbols")

            self.symbol_funding_dfs = []
            date_str = date_to_str(download_date)
            barfile_dir = f"{dir_manager.TARDIS_PREBAR_DIR}/{date_str}/{exchange}"
            safe_mkdir(barfile_dir)
            tardis_dir = f"tardis/{date_str}"

            if self.pool_size == 1:
                for symbol in symbols:
                    self._download_files_and_generate_bars(symbol_venue=symbol, download_date=download_date, barfile_dir=barfile_dir, exchange=exchange, tardis_dir=tardis_dir)
            else:
                pool = Pool(processes=self.pool_size)
                for _ in pool.starmap(
                        self._download_files_and_generate_bars,
                        [(symbol, download_date, barfile_dir, exchange, tardis_dir) for symbol in symbols]
                ):
                    pass
                pool.close()
                pool.join()

            if 'derivative_ticker' in self.data_types:
                self._merge_and_dump_funding_data(download_date)

            # remove local tardis if s3 or only derivative_ticker for funding data backfill
            if self.s3 or (len(self.data_types) == 1 and self.data_types[0] == 'derivative_ticker'):
                tardis_data_dir = f"{DATA_DIR}/{tardis_dir}"
                if os.path.exists(tardis_data_dir):
                    logger.info(f"Removing local tardis files from {tardis_data_dir}")
                    shutil.rmtree(tardis_data_dir)
