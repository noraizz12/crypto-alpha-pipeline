import logging
import os
from collections import defaultdict
from datetime import date
from datetime import timedelta as td
from typing import List, Optional

import numpy as np
import pandas as pd

from lib.bars.bar_resampler import BarResampler
from lib.calcs.calcs import Calcs
from lib.data import dump_parquet_files
from lib.data.dataloader import DataLoader
from lib.util.config import extract_horizons
from lib.util.dataframes import check_df, concat, get_min_max_ts, merge_on_index, get_trailing_nan_mask, read_parquet, safe_del, make_date, safe_int32_cast
from lib.util.directory import DirectoryManager, dir_manager
from lib.util.logging_util import KeyLogger
from lib.util.time_util import compute_lookback_days, date_range, date_to_start_dt, date_to_str, today_date
from lib.util.util import TARDIS_EXCHANGE, unique_list, log_and_raise, MKT_SYMBOL

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)

BAR_FLDS_AGG_DICT = {
    'bid_sz_avg': ['mean'],
    'ask_sz_avg': ['mean'],
    'low_mid': ['min'],
    'high_mid': ['max'],
    'dvolume': ['sum'],
    'trade_cnt': ['sum'],
    'update_cnt': ['sum'],
    'high_trade': ['max'],
    'low_trade': ['min'],
    'volume': ['sum'],
    'bid_trade_dollars': ['sum'],
    'ask_trade_dollars': ['sum'],
    'spread_avg': ['mean'],
    'logret': ['sum'],
    'logret_resid_eqmkt': ['sum'],
    'logret_resid_wgtmkt': ['sum'],
    'logret_funding_adj': ['sum'],
    'logret_funding_adj_resid_eqmkt': ['sum'],
    'logret_funding_adj_resid_wgtmkt': ['sum'],
    'book_latency': ['mean'],
    'trade_latency': ['mean'],
    'open_interest': ['mean'],
    'index_price_dvolume': ['sum'],
    'index_mid_price_diff': ['mean', 'std'],
    'last_funding_rate': ['mean', 'std'],
    'twap': ['mean'],
}

FIXED_COLS_TO_PERSIST = [
    'close_trade',
    'close_mid',
    'close_wgt_mid',
    'priceable',
    'advp',
    'index_price',
    'estimated_settle_price',
    'last_funding_rate',
    'next_funding_time'
]
ALLOWABLE_NANS_ROLLING = 5


class BarGenerator:
    def __init__(
            self,
            config: dict,
            venue: str,
            start_date: date,
            end_date: date,
            debug: bool = False,
            horizons: Optional[List[int]] = None,
            chunk_days: Optional[int] = None,
            pool_size: int = 6,
            backfill: bool = False,
            bars_dir_manager: Optional[DirectoryManager] = None,
            output_dir: Optional[str] = None,
            symbols: Optional[List[str]] = None
    ):
        self.venue = venue
        self.debug = debug
        self.config = config
        self.chunk_days = chunk_days
        self.backfill = backfill
        self.bars_dir_manager = bars_dir_manager if bars_dir_manager is not None else dir_manager
        self.data_loader = DataLoader(self.config, self.bars_dir_manager, debug=self.debug)

        if output_dir is not None:
            self.output_dir = output_dir
        else:
            # Always use new format directory
            self.output_dir = self.bars_dir_manager.BAR_DIR

        self.horizons = extract_horizons(config) if horizons is None else horizons

        assert end_date <= today_date()
        assert end_date >= start_date

        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        
        self.adv_lookback_days = self.config['ADV_LOOKBACK_DAYS']
        self.min_bars_dvolume = self.config['MIN_ADVP_BARS']
        self.daily_bars_df = None
        self.calcs = Calcs(config)
        self.bar_resampler = BarResampler(
            debug=debug,
            venue=self.venue,
            data_loader=self.data_loader,
            pool_size=pool_size,
            dir_manager=self.bars_dir_manager
        )

        self.advp_map = self.data_loader.load_prebar_advp_map(start_date=self.start_date, end_date=self.end_date)
        self.symbol_cache = {}
        logger.info(f'Generating bars for {start_date} to {end_date}')

    def _calculate_rolling_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling aggregations for all bar fields across configured horizons.
        
        Computes rolling statistics (sum, mean, min, max, std) for various bar fields
        over multiple time horizons. Handles NaN values properly by forward filling
        internal gaps while preserving trailing NaNs.
        
        Args:
            df: Bar DataFrame with fields to aggregate
            
        Returns:
            DataFrame with rolling fields added as new columns
            
        Notes:
            - Uses BAR_FLDS_AGG_DICT to determine which aggregations to compute
            - Column naming: '{field}_{horizon}' for single aggregation
            - Column naming: '{field}_{method}_{horizon}' for multiple aggregations
            - Efficiently processes fields with same aggregation method together
            - Preserves trailing NaN patterns in the data
        """
        check_df(df)
        # Pre-compute method groups
        method_groups = defaultdict(list)
        for fld, methods in BAR_FLDS_AGG_DICT.items():
            for method in methods:
                method_groups[method].append(fld)
        # Delete all potential columns at once
        for fld, methods in BAR_FLDS_AGG_DICT.items():
            for horizon in self.horizons:
                safe_del(df, f"{fld}_{horizon}")
                for method in methods:
                    safe_del(df, f"{fld}_{method}_{horizon}")

        for method, fields in method_groups.items():
            # Process multiple fields with same method at once
            flds_df = df[fields].unstack()
            trailing_nan_mask = get_trailing_nan_mask(flds_df)
            for horizon in self.horizons:
                logger.info(f"Generating rolling values for {fields} at {horizon=}, {method=}")
                # 1. Do rolling calculation first
                # 2. Forward fill internal NaNs from rolling result
                # 3. Keep consecutive NaNs at the end
                rolling_result = (flds_df
                                  .rolling(f"{horizon}min", min_periods=1)
                                  .agg(method)
                                  .ffill()
                                  .mask(trailing_nan_mask)
                                  .astype(np.float32)
                                  .stack(future_stack=True))

                suffixed_columns = [
                    f"{fld}_{horizon}" if len(BAR_FLDS_AGG_DICT[fld]) == 1
                    else f"{fld}_{method}_{horizon}"
                    for fld in fields
                ]
                rolling_result.columns = suffixed_columns
                df = merge_on_index(df, rolling_result)

        return df

    def _load_mkt_ret_parquets(self, mkt_ret_dir: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Load market return data from parquet files for date range.
        
        Loads market index (MKTIDX) files used for calculating market-adjusted returns.
        
        Args:
            mkt_ret_dir: Directory containing market return files
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            DataFrame with market return data
            
        Notes:
            - Files expected in format: {date}/bars.1.{exchange}.{date}.MKTIDX.parquet
            - Removes duplicate indices keeping first occurrence
            - Concatenates multiple days into single DataFrame
        """
        dfs = []
        while start_date <= end_date:
            date_str = date_to_str(start_date)
            filename = f"{mkt_ret_dir}/{date_str}/bars.1.{TARDIS_EXCHANGE}.{date_str}.{MKT_SYMBOL}.parquet"
            df = read_parquet(filename)
            dfs.append(df)
            start_date += td(days=1)
        df = concat(dfs, fast=True)
        df = df[~df.index.duplicated(keep='first')]
        df = df.reset_index()
        return df


    def _get_valid_symbols_for_date(self, bar_date: date) -> List[str]:
        """Filter symbols to only those available on the specified date.
        
        Removes symbols that have been delisted or are not yet available.
        
        Args:
            bar_date: Date to check symbol availability for
            
        Returns:
            List of symbols that are valid/tradeable on the given date
            
        Notes:
            - Uses availability_date_dict loaded from metadata
            - Symbols with availableTo date before bar_date are excluded
        """
        if self.symbols is not None:
            return self.symbols

        if self.symbol_cache.get(bar_date) is not None:
            return self.symbol_cache[bar_date]

        symbols = []
        advp_sum = 0
        for symbol, advp in self.advp_map[bar_date].items():
            advp_sum += advp
            add_symbol = advp > self.min_bars_dvolume
            logger.info(f"bar filter: {symbol}:{bar_date} {advp=} pass:{add_symbol}")
            if add_symbol:
                symbols.append(symbol)

        if advp_sum == 0 or len(symbols) == 0:
            logger.error(self.advp_map[bar_date])
            raise log_and_raise(f"No dvolume data on {bar_date}")

        symbols = sorted(symbols)
        logger.info(f"Found {len(symbols)} symbols available on {bar_date} with advp > {self.min_bars_dvolume:.2f}:\n{symbols}")
        self.symbol_cache[bar_date] = symbols
        return symbols

    def _generate_bars(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Generate 1-minute bars for all symbols across date range.
        
        Core method that creates bars for each date and combines them.
        
        Args:
            start_date: Start date for bar generation
            end_date: End date for bar generation
            
        Returns:
            Combined DataFrame with all bars for the date range,
            or None if no bars could be generated
            
        Raises:
            Exception: If no bar files generated for any date
            
        Notes:
            - Filters symbols by availability for each date
            - Uses bar_resampler to create bars for each date
            - Cleans cache after processing to free memory
        """
        logger.info(f"Generating 1 min bars from {date_to_str(start_date)} to {date_to_str(end_date)}")

        bars_dfs = []
        for bar_date in date_range(start_date, end_date):
            #ensure that we have a history for everything we want to add on the first data and from there can drop
            symbol_date = max(bar_date, self.start_date)
            bar_date_symbols = self._get_valid_symbols_for_date(symbol_date)
            logging.info(f"Generating bars with {len(bar_date_symbols)} symbols on {bar_date}")
            bars_df = self.bar_resampler.create_resampled_bars(bar_date=bar_date, bar_date_symbols=bar_date_symbols)
            if bars_df is None:
                raise log_and_raise(f"No bar files generated for {bar_date}")

            check_df(bars_df)
            min_ts, max_ts = get_min_max_ts(bars_df)
            if (max_ts - td(minutes=1)).date() != bar_date:
                raise  log_and_raise(f"resampled bars for {bar_date} go from {min_ts} - {max_ts}")

            bars_dfs.append(bars_df)

        if len(bars_dfs) == 0:
            logger.error(f"No bar files generated! {start_date=}, {end_date=}", key="fail to generate bar files")
            return None

        bars_df = concat(bars_dfs)
        min_ts, max_ts = get_min_max_ts(bars_df)
        logger.info(f"Returning bars from {min_ts} to {max_ts}")

        self.bar_resampler.clean_cache(start_date)
        return bars_df

    def run(self) -> None:
        """Execute the bar generation process for configured date range and horizons.

        Main entry point that orchestrates the entire bar generation workflow:
        1. Generates 1-minute bars with appropriate lookback
        2. Calculates returns and market-adjusted returns
        3. Computes rolling statistics for all configured horizons
        4. Saves results to appropriate output format/location

        Notes:
            - Processes data in chunks if chunk_days is specified
            - Calculates ADVP (average daily volume) for liquidity filtering
            - Handles market return calculation for BARS_TYPE_NEW
            - Outputs to parquet files organized by date and horizon
            - Automatically handles lookback periods for rolling calculations
        """
        max_rolling_lookback_days = compute_lookback_days(max(self.horizons), lags=0)

        #not sure why this had adv_loomback_days removed
        total_lookback_days = max_rolling_lookback_days + self.adv_lookback_days

        chunk_start_date = self.start_date
        if self.chunk_days is not None:
            chunk_end_date = min(chunk_start_date + td(days=self.chunk_days + 1), self.end_date)
        else:
            chunk_end_date = self.end_date

        logger.info(f"Generating bars with {total_lookback_days=} from {date_to_str(chunk_start_date)} to {date_to_str(chunk_end_date)}")

        while chunk_start_date <= self.end_date:
            buffer_start_date = chunk_start_date - td(days=total_lookback_days)
            logger.info(f"Generating chunk of bars with lookback from {chunk_start_date} to {chunk_start_date} with lookback to {buffer_start_date}")
            chunk_bars_df = self._generate_bars(start_date=buffer_start_date, end_date=chunk_end_date)
            if chunk_bars_df is None:
                continue

            check_df(chunk_bars_df)

            chunk_bars_df = self.calcs.calculate_advp(chunk_bars_df, lookback_days=self.adv_lookback_days)
            rolling_lookback_dt = date_to_start_dt(chunk_start_date - td(days=max_rolling_lookback_days))

            MKT_RET_FILE = f'{self.output_dir}/1/{TARDIS_EXCHANGE}/{date_to_str(chunk_start_date)}/bars.1.{TARDIS_EXCHANGE}.{date_to_str(chunk_start_date)}.{MKT_SYMBOL}.parquet'
            mkt_ret_dir = f'{self.output_dir}/1/{TARDIS_EXCHANGE}'
            if self.backfill and os.path.isfile(MKT_RET_FILE):
                logger.info(f'Calculating bars return with existing market return {mkt_ret_dir}')
                market_ret_df = self._load_mkt_ret_parquets(mkt_ret_dir=mkt_ret_dir, start_date=chunk_start_date, end_date=chunk_end_date)
                chunk_bars_df = self.calcs.calc_returns(chunk_bars_df, start_dt=rolling_lookback_dt, market_ret_df=market_ret_df)
            else:
                logger.info(f'Calculating bars return and dump market return to {mkt_ret_dir}')
                chunk_start_dt = date_to_start_dt(chunk_start_date)
                chunk_bars_df = self.calcs.calc_returns(chunk_bars_df, start_dt=rolling_lookback_dt, chunk_start_dt=chunk_start_dt, mkt_ret_dir=mkt_ret_dir)

            chunk_bars_df = chunk_bars_df[chunk_bars_df.index.get_level_values('ts') >= rolling_lookback_dt]
            chunk_bars_df['twap'] = chunk_bars_df['close_mid']
            chunk_bars_df = self._calculate_rolling_fields(chunk_bars_df)
            chunk_bars_df = chunk_bars_df[chunk_bars_df.index.get_level_values('ts') > date_to_start_dt(chunk_start_date)]
            chunk_bars_df = make_date(chunk_bars_df)
            for horizon in self.horizons:
                chunk_bars_df[f'vwap_{horizon}'] = chunk_bars_df[f'dvolume_{horizon}'] / chunk_bars_df[f'volume_{horizon}']
                chunk_bars_df[f'index_price_vwap_{horizon}'] = chunk_bars_df[f'index_price_dvolume_{horizon}'] / chunk_bars_df[f'volume_{horizon}']
                bar_flds_list = [
                    f"{col}_{horizon}" if len(methods) == 1 else f"{col}_{method}_{horizon}"
                    for col, methods in BAR_FLDS_AGG_DICT.items()
                    for method in methods
                ]
                for fld in [f'trade_cnt_{horizon}', f'update_cnt_{horizon}']:
                    chunk_bars_df[fld] = safe_int32_cast(chunk_bars_df[fld])
                cols = unique_list(FIXED_COLS_TO_PERSIST + bar_flds_list + [f"vwap_{horizon}", "close_trade", f'index_price_vwap_{horizon}'])

                for ddate in sorted(list(chunk_bars_df['date'].unique())):
                    symbols_for_date = list([ss for ss, dv in self.advp_map[ddate.date()].items() if dv > self.min_bars_dvolume])
                    date_chunk_df = chunk_bars_df[(chunk_bars_df['date'] == ddate) & chunk_bars_df['symbol'].isin(symbols_for_date)]
                    logger.info(f"Bars {horizon} on {ddate} count {len(date_chunk_df)/1440}")
                    dump_parquet_files(file_type='bars', df=date_chunk_df, directory=f"{self.output_dir}/{horizon}/{TARDIS_EXCHANGE}", name=f"bars.{horizon}.{TARDIS_EXCHANGE}", cols=cols, start_date=chunk_start_date, debug=self.debug)

            chunk_start_date = chunk_end_date + td(days=1)
            chunk_end_date = min(chunk_end_date + td(days=self.chunk_days), self.end_date)
