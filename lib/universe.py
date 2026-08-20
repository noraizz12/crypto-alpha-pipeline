"""Universe management for cryptocurrency trading system.

This module handles the dynamic selection and filtering of tradeable cryptocurrency
symbols based on market capitalization, liquidity metrics, and trading activity.
It maintains an evolving universe of instruments that meet specific criteria for:
- Model fitting (sufficient liquidity for reliable signals)
- Trading execution (adequate volume for order execution)
- Position expansion (high liquidity for larger positions)

The universe is updated daily based on:
- Market capitalization rankings from CoinGecko
- Average daily volume (ADVP) calculations
- Trading activity metrics (update counts, volume)
- Delisting status and warnings

Key concepts:
- Fittable: Symbols with enough liquidity for model training
- Tradeable: Symbols meeting minimum liquidity for trading
- Expandable: High-liquidity symbols suitable for larger positions
"""
import datetime
import glob
import logging
from datetime import timedelta as td, date
from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Literal, Any

import numpy as np
import pandas as pd

from lib.calcs import Calcs
from lib.util import date_range
from lib.util.dataframes import make_date, merge_on_index, make_symbol, set_index, make_symbol_venue
from lib.data.dataloader import DataLoader
from lib.util.logging_util import KeyLogger
from lib.data.loaders import load_metadata, load_positions
from lib.util.directory import dir_manager, DirectoryManager
from lib.util.time_util import date_to_str, today_date, date_to_end_dt, to_datetime
from lib.util.util import TARDIS_EXCHANGE, get_config_universe, symbols_to_symbol_venues, log_and_raise, symbols_to_pairs, SYMBOL_BASE, SYMBOL_PAIR, SYMBOL_VENUE, unique_list
from lib.util.files import find_latest_universe_date
from lib.data.loaders import load_mktcap_and_groupings

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)

# Symbols to exclude from universe (stablecoins, etc.)
IGNORED_SYMBOLS = ["USDC"]
DEFAULT_TOP_MKTCAP = 200
DEFAULT_BAR_FREQUENCY = 1440


def filter_universe(init_universe: List[str], bars_df: pd.DataFrame, positions_df: Optional[pd.DataFrame], date: datetime.date) -> List[str]:
    """Filter trading universe based on liquidity and delisting status.

    Applies various filters to ensure we only trade liquid, active instruments
    while protecting existing positions.

    Args:
        init_universe: Initial list of symbols to consider
        bars_df: Bar data with liquidity metrics
        positions_df: Current positions (can be None)
        date: Current date for delisting check

    Returns:
        Filtered list of tradeable symbols

    Notes:
        - Excludes delisting instruments with buffer period
        - Excludes low liquidity instruments
        - Always includes symbols with existing positions
    """
    universe_change_msg = f"Update universe on {date} from init universe {len(init_universe)}, "
    max_1_min_bar_ts = bars_df.index.get_level_values('ts').max()
    latest_bars_df = bars_df[bars_df.index.get_level_values('ts') == max_1_min_bar_ts]
    starting_universe = list(latest_bars_df.index.get_level_values('symbol_venue').unique())
    universe_change_msg += f"starting universe {len(starting_universe)}, "
    logger.info(f"Live bar universe: sorted({starting_universe})")

    dead_idx = (latest_bars_df['update_cnt'].fillna(0) == 0) & (latest_bars_df['volume'].fillna(0) == 0)
    dead_df = latest_bars_df[dead_idx]
    latest_bars_df = latest_bars_df[~dead_idx]
    
    live_universe = set(latest_bars_df.index.get_level_values('symbol_venue').unique())
    universe_change_msg += f"live universe {len(live_universe)}, "

    dead_secs = list(set(init_universe) - live_universe)
    if len(dead_secs):
        logger.error(f"Starting with {len(starting_universe)}, Removing {dead_secs} from universe due to no quote updates", key='missing quotes in universe')
        logger.warning(dead_df[['update_cnt', 'volume', 'close_mid']])

    fittable_universe = set(latest_bars_df[latest_bars_df['fittable']].index.get_level_values('symbol_venue').unique())
    universe_change_msg += f"fittable universe {len(fittable_universe)}, "

    position_universe = set(positions_df[positions_df['qty'] != 0].index.get_level_values('symbol_venue').unique()) if positions_df is not None else set()
    universe_change_msg += f"position universe {len(position_universe)}, "

    fittable_universe = fittable_universe | position_universe
    removed_secs = live_universe - fittable_universe

    fittable_universe = list(fittable_universe)

    logger.info(f"Removing unfittable securities: {removed_secs}")
    logger.info(f"Trading universe of {len(fittable_universe)} securities")
    logger.info(f"Trading universe: {sorted([cc.split('_')[0] for cc in fittable_universe])}")

    universe_change_msg += f"latest universe {len(fittable_universe)}"
    logger.info(universe_change_msg)
    return fittable_universe


class Universe:
    """Manages the dynamic trading universe based on market cap and liquidity.
    
    The Universe class maintains a curated list of tradeable cryptocurrency symbols
    that evolve over time based on market conditions. It combines market cap data
    from external sources with internal liquidity calculations to determine which
    symbols are suitable for different trading operations.
    
    Attributes:
        config: Trading system configuration dict
        debug: If True, prints data instead of saving files
        data_loader: DataLoader instance for accessing market data
        calcs: Calcs instance for liquidity calculations
        output_dir: Directory for saving universe files
        adv_lookback_days: Days of history for ADV calculation
    """

    def __init__(
            self,
            config: dict,
            debug: bool = False,
            output_dir: Optional[str] = None,
            universe_dir_manager: Optional[DirectoryManager] = None,
            top_n: int = DEFAULT_TOP_MKTCAP,
            prod: bool = False
    ):
        self.config = config
        self.debug = debug
        self.top_n = top_n
        self.prod = prod
        self.dir_manager = universe_dir_manager if universe_dir_manager is not None else dir_manager
        self.data_loader = DataLoader(config=config, data_loader_dir_manager=self.dir_manager)
        self.calcs = Calcs(config=self.config, frequency=DEFAULT_BAR_FREQUENCY)
        self.output_dir = output_dir if output_dir is not None else self.dir_manager.UNIVERSE_DIR
        self.adv_lookback_days = config['ADV_LOOKBACK_DAYS']
        self.featureable_hist_days = config['DEFAULT_FEATURE_LOOKBACK_PERIODS']
        self.dynamic_universe = config['DYNAMIC_UNIVERSE']

        logger.info(f"Universe with {self.adv_lookback_days=}")

    def _dump_universe(self, df: pd.DataFrame, universe_date: date) -> None:
        """Save universe data to parquet file.

        Writes the universe DataFrame to a date-stamped parquet file and logs
        summary statistics about liquidity filters.

        Args:
            df: Universe DataFrame with liquidity flags
            universe_date: Date for filename

        Output format: universe.{YYYYMMDD}.parquet

        Side Effects:
            - Creates parquet file in output_dir
            - Logs counts of fittable/tradeable/expandable symbols
        """
        universe_file = f'{self.output_dir}/universe.{date_to_str(universe_date)}.parquet'
        logger.info(f"Dumping universe data to {universe_file}, fittable {df['fittable'].sum()} tradeable {df['tradeable'].sum()} expandable_cnt {df['expandable'].sum()}")
        df.to_parquet(universe_file, index=False)

    def _create_updated_universe(self, universe_date: date, marketcap_df: Optional[pd.DataFrame], pos_df: Optional[pd.DataFrame]) -> Tuple[List[str], List[str]]:
        """Create updated universe by incorporating market cap rankings.
        
        Updates the trading universe by:
        1. Loading the previous day's universe
        2. Getting top N symbols by market cap
        3. Adding new high market cap symbols to the universe
        4. Keeping existing symbols (never removes based on market cap alone)
        
        Args:
            universe_date: Date for which to create universe
            marketcap_df: DataFrame with symbol and market_cap columns from CoinGecko
            
        Returns:
            Tuple of:
                - previous_uni: List of symbols from previous day
                - new_symbols: List of newly added symbols
                
        Notes:
            - Symbols are only added, never removed based on market cap
            - Uses previous day's universe as base
            - Filters out IGNORED_SYMBOLS (stablecoins)
        """
        previous_universe_date = universe_date - td(days=1)
        previous_uni_pairs = self.load_universe_symbols(universe_source='file', universe_date=previous_universe_date, symbol_type=SYMBOL_PAIR)
        if previous_uni_pairs is not None:
            previous_uni_pairs = set(previous_uni_pairs)
        else:
            previous_uni_pairs = set()

        if pos_df is not None:
            pos_symbols = set(pos_df['symbol'].unique())
            logger.info(f"Adding position symbols {len(pos_symbols)} ")
        else:
            pos_symbols = set()

        # update universe by taking marketcap from previous_universe_date
        if marketcap_df is None:
            logger.warning(f"Could not update universe on {universe_date} since no marketcap data, size {len(previous_uni_pairs)}")
            return list(previous_uni_pairs), []

        marketcap_pairs = marketcap_df.sort_values(by='marketcap', ascending=False)['symbol'].to_list()[:self.top_n]
        marketcap_pairs = set([ss for ss in marketcap_pairs if ss not in IGNORED_SYMBOLS])
        new_symbols = marketcap_pairs - previous_uni_pairs

        logger.info(f"Marketcap symbols: {sorted(marketcap_pairs)}")
        logger.info(f"New symbols: {sorted(new_symbols)}")
        logger.info(f"Previous symbols: {sorted(previous_uni_pairs)}")
        logger.info(f"Position symbols: {sorted(pos_symbols)}")

        new_uni = previous_uni_pairs | new_symbols | pos_symbols

        previous_uni = sorted(list(previous_uni_pairs))
        new_symbols = sorted(list(new_symbols))
        logger.info(f"Universe on {universe_date} going from {len(previous_uni_pairs)} -> {len(new_uni)}")
        if len(new_uni) > 0:
            logger.info(f"Adding {new_symbols}")
        return previous_uni, new_symbols

    def _load_bars(self, asof_date: date) -> Optional[pd.DataFrame]:
        """Load bar data for ADVP calculation and fittable filter.
        
        Loads historical bar data needed to calculate average daily volume
        and determine which symbols meet liquidity thresholds for model fitting.
        
        Args:
            asof: Date to calculate metrics for
            universe: List of symbols to load data for
            
        Returns:
            DataFrame with bars and calculated ADVP/fittable flags,
            or None if no data available
            
        Data loaded:
            - dvolume_1: Dollar volume for ADVP calculation
            - update_cnt_1: Order book updates for activity check
            - close_mid: Mid price for value calculations
        """
        # dvolume for advp calculation, update_cnt and close_mid used for fittable filter
        bars_df = self.data_loader.load_bars(
            horizon=DEFAULT_BAR_FREQUENCY,
            start_date=asof_date - td(days=self.adv_lookback_days),
            end_date=asof_date,
            cols=['dvolume_1440', 'update_cnt_1440', 'close_mid', 'advp', 'priceable'],
        )
        if bars_df is None:
            return None

        logger.info(f"Filtering to bars as of {asof_date}")
        bars_df = make_date(bars_df)
        bars_df = bars_df.loc[bars_df['date'] == to_datetime(asof_date)]

        return bars_df

    def _load_features(self, asof_date: date, bars_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Load feature data for tradeable and expandable filters.
        
        Enhances bar data with feature-based metrics to determine trading
        and position expansion eligibility.
        
        Args:
            asof: Date to load features for
            bars_df: Bar data to merge with features
            
        Returns:
            DataFrame with all liquidity flags (fittable, tradeable, expandable)
            
        Processing steps:
            1. Load dvolume_1440_trmean for trailing volume
            2. Merge with bar data
            3. Add delisting dates for risk management
            4. Calculate tradeable/expandable flags
            5. Extract end-of-day snapshot
        """

        symbol_venues = list(bars_df.index.get_level_values('symbol_venue').unique())

        # dvolume_1440_trmean used for tradeable filter
        features_df = self.data_loader.load_features(
            horizons=[1440],
            start_date=asof_date,
            end_date=asof_date,
            cols=['dvolume_1440_trmean', 'mdvp'],
            symbol_venues=symbol_venues
        )
        if features_df is None:
            return None

        feature_symbols = set(features_df.index.get_level_values('symbol_venue').unique())
        bar_symbols = set(symbol_venues)
        no_feature_symbols = sorted(bar_symbols - feature_symbols)
        logger.info(f"No feature symbols {no_feature_symbols}")

        features_df = merge_on_index(features_df, bars_df, how='outer')

        features_df = self.data_loader.add_delisting_dates(features_df, start_date=asof_date, end_date=asof_date)
        features_df = self.calcs.calculate_fittable_filter(features_df, error_threshold=0.0)

        # get the last piece of data at t-2
        features_df = features_df.loc[features_df.index.get_level_values('ts') == date_to_end_dt(asof_date)]
        features_df = make_symbol(features_df)
        # Reset index to ensure symbol is a column and deduplicate
        features_df = features_df.reset_index()
        features_df = features_df.drop_duplicates(subset=['symbol'], keep='first')
        return features_df

    def generate_universe_file(self, universe_date: date) -> pd.DataFrame:
        """Generate the trading universe for a given date.
        
        Main entry point that orchestrates universe generation by:
        1. Loading market cap data from CoinGecko (t-1)
        2. Creating updated symbol list based on market cap
        3. Loading historical bars/features (t-2) for liquidity metrics
        4. Combining all data into final universe DataFrame
        
        Args:
            universe_date: Date to generate universe for
            
        Returns:
            DataFrame with columns:
                - date: Universe date
                - symbol: Cryptocurrency symbol (e.g., 'BTCUSDT')
                - marketcap: Market capitalization from CoinGecko
                - advp: Average daily volume in USD
                - fittable: Boolean - sufficient liquidity for model fitting
                - tradeable: Boolean - meets minimum liquidity for trading
                - expandable: Boolean - high liquidity for larger positions
                
        Notes:
            - Uses t-1 market cap data (most recent available)
            - Uses t-2 bar/feature data (ensures data completeness)
            - Sorts by ADVP descending (most liquid first)
        """

        # for bars/feature load, the data of t-1 should be ready at that time
        previous_date = universe_date # - td(days=1)
        #so will only work after bars are generated for tardis

        #maybe t-1 or 2
        prebars_df = self.data_loader.load_prebar_files(
            start_date=previous_date,
            end_date=previous_date,
            bars_type='tardis'
        )
        if prebars_df is None:
            raise log_and_raise(f"Could not get prebars for {previous_date}")
        prebars_df = prebars_df[prebars_df.index.get_level_values('ts') == prebars_df.index.get_level_values('ts').max()]

        # Reset index and extract symbol from symbol_venue
        prebars_df = prebars_df.reset_index()
        prebars_df = make_symbol(prebars_df)
        # Keep only one row per symbol (in case of multiple venues)
        prebars_df = prebars_df.drop_duplicates(subset=['symbol'], keep='first')[['symbol', 'close_mid']]

        prebar_symbols = prebars_df['symbol'].unique()
        logger.info(f"Prebar symbols ({len(prebar_symbols)}): {prebar_symbols}")

        # we generate universe t at the beginning of t, get the latest universe and market cap from t-1
        marketcap_df = load_mktcap_and_groupings(universe_date - td(days=1), mktcap_dir=self.dir_manager.MARKETCAP_DIR)
        if marketcap_df is None:
            raise log_and_raise(f"Not generating universe without mktcap data")
        marketcap_df = marketcap_df[['symbol', 'marketcap']]

        if self.prod:
            pos_df = None
            pdate = universe_date
            while pos_df is None:
                pos_df = load_positions(mode='latest', pos_dir=self.dir_manager.POSITION_DIR, start_date=pdate, end_date=pdate)
                pdate = pdate - td(days=1)

            logger.info(f"Using positions for {pdate}")
            pos_df = pos_df[pos_df['value'] != 0][['symbol', 'value']]
            pos_df = pos_df.rename(columns={'value': 'position_value'})
        else:
            pos_df = None

        prev_universe, new_symbols = self._create_updated_universe(universe_date=universe_date, marketcap_df=marketcap_df, pos_df=pos_df)
        universe = unique_list(prev_universe + new_symbols)
        df = pd.DataFrame({'symbol': universe})

        # get full symbol universe
        if marketcap_df is not None:
            marketcap_df = marketcap_df[['symbol', 'marketcap']]
            df = pd.merge(df, marketcap_df, how='left', on='symbol')
        else:
            df['marketcap'] = np.nan

        if pos_df is not None:
            df = pd.merge(df, pos_df, how='left', on='symbol')
            df['position_value'] = df['position_value'].fillna(0)
        else:
            df['position_value'] = np.nan

        df = pd.merge(prebars_df, df, how='outer', on='symbol')
        df['dead'] = df['close_mid'].isna()

        bars_df = self._load_bars(asof_date=previous_date)
        if bars_df is None:
            raise log_and_raise(f"Could not load bars for universe...")

        bars_df = make_symbol(bars_df)
        bar_symbols = sorted(bars_df['symbol'].unique())
        logger.info(f"Bar symbols ({len(bar_symbols)}): {bar_symbols}")
        bars_df = set_index(pd.merge(bars_df.reset_index(), df, on='symbol', how='left', suffixes=('', '_prebar' )))

        features_df = self._load_features(asof_date=previous_date, bars_df=bars_df)
        if features_df is None:
            raise log_and_raise(f"Could not load features for universe...")

        feature_symbols = sorted(features_df['symbol'].unique())
        logger.info(f"Feature symbols ({len(feature_symbols)}): {feature_symbols}")

        df = pd.merge(df, features_df[['symbol', 'advp', 'mdvp', 'dvolume_1440_trmean', 'priceable', 'fittable', 'symbol_venue']], on='symbol', how='left')

        df = make_symbol_venue(df).set_index(['symbol_venue'])
        df = self.calcs.calculate_tradeable_filter(df)
        df = self.calcs.calculate_expandable_filter(df)
        df = df.reset_index()

        df['date'] = universe_date
        for fld in ['priceable', 'fittable', 'tradeable', 'expandable']:
            if fld in df.columns:
                df[fld] = df[fld].fillna(value=False).astype('bool')
            else:
                logger.warning(f"{fld} not in df, setting to False")
                df[fld] = False
        for vol_col in ['advp', 'mdvp', 'dvolume_1440_trmean']:
            if vol_col not in df.columns:
                logger.warning(f"{vol_col} not in df, setting to zero")
                df[vol_col] = np.float32(0.0)

        df = df.reset_index()
        df = df[['date', 'symbol', 'marketcap', 'advp', 'mdvp', 'dvolume_1440_trmean', 'position_value', 'dead', 'priceable', 'fittable', 'tradeable', 'expandable']].sort_values(by=['dead', 'advp', 'marketcap'], ascending=[True, False, False])

        if self.debug:
            print(df)
        else:
            self._dump_universe(df=df, universe_date=universe_date)
        return df

    def load_latest_availablility(self) -> Dict[str, Dict[str, Any]]:
        metadata_df = load_metadata(latest=True, metadata_dir=self.dir_manager.BINANCE_META_DIR)
        if metadata_df is None:
            raise log_and_raise("no latest metadata found!")
        metadata_df = metadata_df[['baseCurrency', 'availableTo', 'availableSince']]
        availability_map = metadata_df.set_index('baseCurrency').to_dict('index')
        return availability_map

    def symbols_available_at_date(self, adate: date) -> List[str]:
        symbols = []
        availability_map = self.load_latest_availablility()
        for symbol, availability in availability_map.items():
            available_to = availability['availableTo']
            available_since = availability['availableSince']
            if available_to is not None and pd.notna(available_to) and pd.Timestamp(available_to).date() < adate:
                continue
            if pd.notna(available_since) and pd.Timestamp(available_since).date() > adate:
                continue
            # Return base currency without USDT suffix for consistency with config
            symbols.append(symbol)
        return symbols

    def get_todays_new_symbols(self, check_bars: bool = False) -> List[str]:
        """Identify newly added symbols in today's universe.

        Compares today's universe against yesterday's data files to find new listings.

        Args:
            check_bars: If True, checks against bar files; if False, checks against prebar files

        Returns:
            List of newly added symbol names

        Notes:
            - Used to identify new listings that need special handling
            - Helps avoid trading new symbols before sufficient data is available
        """
        current_date = today_date()
        previous_date_str = date_to_str(current_date - td(days=1))
        current_symbols = self.load_universe_symbols(universe_source='file', universe_date=current_date)

        if current_symbols is None:
            return []

        # i don't like this...brittle
        if check_bars:
            bars_file_match = f"{self.dir_manager.BAR_DIR}/1/{TARDIS_EXCHANGE}/{previous_date_str}/bars.1.{TARDIS_EXCHANGE}.{previous_date_str}.*.parquet"
        else:
            bars_file_match = f"{self.dir_manager.PREBAR_DIR}/tardis/{previous_date_str}/{TARDIS_EXCHANGE}/prebars.tardis.{TARDIS_EXCHANGE}.{previous_date_str}.*.parquet"

        previous_symbols = symbols_to_symbol_venues([file.split('.')[-2] for file in sorted(glob.glob(bars_file_match))])

        logger.info(f"previous symbols: {previous_symbols}")
        logger.info(f"current symbols: {current_symbols}")

        new_symbols = [symbol for symbol in current_symbols if symbol not in previous_symbols]
        logger.info(f'Get new symbols {new_symbols} for {current_date}, {check_bars=}')
        return new_symbols

    def filter_config_symbols(self, symbols: List[str], symbol_type: Literal[SYMBOL_BASE, SYMBOL_PAIR, SYMBOL_VENUE]) -> List[str]:
        config_symbols = get_config_universe(self.config, symbol_type=symbol_type)
        symbols = list(set(symbols) & set(config_symbols))
        return symbols

    def load_universe_symbols(
            self,
            universe_source: Literal['available', 'marketcap', 'bars', 'file'],
            universe_date: Optional[date] = None,
            symbol_type: Literal[SYMBOL_BASE, SYMBOL_PAIR, SYMBOL_VENUE] = 'symbol_venue',
            filter: Optional[Literal['fittable', 'tradeable']] = None,
            frequency: Optional[int] = None
    ) -> Optional[List[str]]:
        """Load list of symbols or symbol_venue strings from universe.

        Convenience method to get just the symbol list from universe data based on
        the specified universe type and filtering criteria.

        Args:
            symbol_type:
            filter:
            universe_source: Type of universe to load:
                - 'available': Symbols available at the specified date based on metadata
                - 'bars': Symbols present in bar data for the date
                - 'file': Symbols from saved universe file for the date
            universe_date: Date to load universe for. Required for all types except 'all'

        Returns:
            List of symbol or symbol_venue strings

        Notes:
            - Falls back to config universe if universe file not found
            - Useful for quickly getting tradeable symbol lists
        """
        logger.info(f"Loading universe for {universe_date} with type: {universe_source} / {filter=} {symbol_type=}")
        symbols = []

        if universe_source == 'available':
            assert filter is None
            assert universe_date is not None
            symbols = self.symbols_available_at_date(adate=universe_date)
            if symbol_type in (SYMBOL_PAIR, SYMBOL_VENUE):
                symbols = symbols_to_pairs(symbols)
            if symbol_type == SYMBOL_VENUE:
                symbols = symbols_to_symbol_venues(symbols)
        elif universe_source == 'marketcap':
            mktcap_df = load_mktcap_and_groupings(universe_date=universe_date - td(days=1), mktcap_dir=self.dir_manager.MARKETCAP_DIR)
            if mktcap_df is None:
                return []
            if symbol_type in (SYMBOL_PAIR, SYMBOL_VENUE):
                symbols = list(mktcap_df['symbol'].unique())
            if symbol_type == SYMBOL_VENUE:
                symbols = symbols_to_symbol_venues(symbols)

        #loads universe from bars WHERE WE HAVE ENOUGH HISTORY
        elif universe_source == 'bars':
            assert universe_date is not None
            frequency = max(frequency, 1440)
            yesterday_universe_date = universe_date - td(days=1)
            lookback = min(90, self.featureable_hist_days * frequency / 1440)
            start_date = yesterday_universe_date - td(lookback-1)
            history_day_map = defaultdict(int)
            for ddate in date_range(start_date, yesterday_universe_date):
                datestr = date_to_str(ddate)
                date_file_pattern = f"{self.dir_manager.BAR_DIR}/{frequency}/{TARDIS_EXCHANGE}/{datestr}/bars.{frequency}.{TARDIS_EXCHANGE}.{datestr}.*.parquet"
                bar_files = glob.glob(date_file_pattern)
                logger.info(f"Loading bar data for {ddate}, loading {len(bar_files)} files")
                for bfile in bar_files:
                    symbol = bfile.split('.')[-2]
                    history_day_map[symbol] += 1

            threshhold = lookback * .90
            avg_hist = sum(history_day_map.values()) / len(history_day_map)
            logger.info(f"threshold day count {threshhold}, with avg hist length of {avg_hist} at frequency {frequency}")
            symbols = symbols_to_symbol_venues([ss for ss, cnt in history_day_map.items() if cnt > threshhold])

            if symbol_type == SYMBOL_BASE:
                raise log_and_raise(f"Return type {symbol_type} not supported for {universe_source}")

            if symbol_type == SYMBOL_PAIR:
                raise log_and_raise(f"Return type {symbol_type} not supported for {universe_source}")

        elif universe_source == 'file':
            if universe_date is None:
                universe_date = find_latest_universe_date(self.dir_manager.UNIVERSE_DIR)
                if universe_date is None:
                    logger.warning("No universe files found")
                    return None

            universe_df = self.data_loader.load_universe_df(universe_date=universe_date, filter_dead=False)
            if universe_df is None:
                logger.warning(f"No universe file found on {universe_date}")
                return None

            if filter == 'fittable':
                universe_df = universe_df[universe_df['fittable']]
            elif filter == 'tradeable':
                universe_df = universe_df[universe_df['tradeable']]

            if symbol_type == SYMBOL_BASE:
                symbol_pairs = universe_df['symbol'].to_list()
                symbols = [s.replace("USDT", "") for s in symbol_pairs]

            if symbol_type == SYMBOL_PAIR:
                symbols = universe_df['symbol'].to_list()

            if symbol_type == SYMBOL_VENUE:
                symbols = universe_df[SYMBOL_VENUE].to_list()
        else:
            raise log_and_raise(f"Unknown universe type {universe_source}")

        if len(symbols) == 0:
            raise log_and_raise(f"No symbols found on {universe_date} for source {universe_source}")

        if not self.dynamic_universe:
            logger.info(f"Not Dynamic Universe, Restricting universe to config symbols")
            symbols = self.filter_config_symbols(symbols, symbol_type=symbol_type)
            if len(symbols) == 0:
                raise log_and_raise(f"No symbols found on {universe_date} after restricting to config symbols")

        symbols = sorted(symbols)

        logger.info(f"Loaded universe for {universe_date} with {len(symbols)} symbols")
        return symbols
