"""Bar resampler for processing and aggregating 1-minute bar data.

This module provides functionality to resample minute-level bar data from multiple
sources (TARDIS, live) into consolidated datasets. It handles:
- Loading 1-minute bar files from local storage or S3
- Merging funding rate data with bar data
- Caching loaded data for performance
- Parallel processing of multiple symbols
"""

import logging
from datetime import datetime as dt, date
from typing import Dict, List, Optional, Literal

import numpy as np
import pandas as pd

from lib.data.dataloader import DataLoader
from lib.util import DirectoryManager
from lib.util.dataframes import carry_forward, merge_on_index, safe_del
from lib.util.logging_util import KeyLogger
from lib.util.util import symbols_to_symbol_venues
from lib.data.loaders import load_funding_rates

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


class BarResampler:
    """Handles resampling and consolidation of 1-minute bar data.
    
    This class manages the loading, processing, and caching of minute-level bar data
    from multiple sources (TARDIS, live). It supports downloading missing files from S3,
    merging funding rate data, and parallel processing for efficiency.
    
    Attributes:
        debug: Debug mode flag (prevents directory creation)
        venue: Trading venue (e.g., 'binance-futures')
        data_loader: DataLoader instance for accessing funding rates
        pool_size: Number of parallel workers for file loading
        min_bar_cache: Cache of loaded bar data by date
        funding_rates_df: Current funding rates DataFrame
    """

    def __init__(
            self,
            debug: bool,
            venue: str,
            data_loader: DataLoader,
            dir_manager: DirectoryManager,
            pool_size: int = 6,
            bars_type: Literal['tardis', 'live', 'merged', 'tardis_prefer', 'tardis_or_live'] = 'tardis'
    ):
        """Initialize BarResampler with configuration.
        
        Args:
            debug: Debug mode flag
            venue: Trading venue to process
            data_loader: DataLoader instance for funding rates
            pool_size: Number of parallel workers
        """
        self.debug = debug
        self.venue = venue
        self.pool_size = pool_size
        self.min_bar_cache: Dict[date, pd.DataFrame] = {}
        self.dir_manager = dir_manager
        self.data_loader = data_loader
        self.bars_type = bars_type

    def clean_cache(self, start_date: date):
        """Remove cached bar data for dates before the specified date.
        
        Helps manage memory by removing old cached data that is no longer needed.
        
        Args:
            start_date: Remove cache entries for all dates before this date
            
        Notes:
            - Cache is stored in self.min_bar_cache dictionary keyed by date
            - Useful when processing data in chunks to free memory
        """
        for bar_date in list(self.min_bar_cache):
            if bar_date < start_date:
                logger.info(f"Removing bar cache for {bar_date}")
                del self.min_bar_cache[bar_date]

    def _load_funding_rates_for_bars(self, bar_date: date) -> Optional[pd.DataFrame]:
        """Load and prepare funding rate data for the specified date.
        
        Loads funding rates from data source and formats them for merging with bar data.
        
        Args:
            bar_date: Date to load funding rates for
            
        Returns:
            DataFrame with funding rate data indexed by (ts, symbol, venue),
            or None if no funding data available
            
        Notes:
            - Renames columns to match bar data format
            - Forward fills funding rates, index prices, and open interest
            - Adds venue information for proper merging
        """
        funding_rates_df = load_funding_rates(start_date=bar_date, end_date=bar_date, fast=True, funding_dir=self.dir_manager.FUNDING_DIR)
        if funding_rates_df is None:
            return None

        funding_rates_df = funding_rates_df.rename(columns={'funding_rate': 'last_funding_rate', 'funding_timestamp': 'next_funding_time'})
        funding_rates_df = carry_forward(funding_rates_df, ['last_funding_rate', 'index_price', 'open_interest'])
        funding_rates_df['next_funding_time'] = funding_rates_df['next_funding_time'].astype('datetime64[ns, UTC]')
        return funding_rates_df

    def _merge_funding_for_bars(self, bars_df: pd.DataFrame, funding_rates_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Merge funding rate data into TARDIS bar DataFrame.
        
        Adds funding-related columns to bar data including funding rates, index prices,
        and calculated fields like index-mid price difference.
        
        Args:
            bars_df: Bar DataFrame to add funding data to
            
        Returns:
            Bar DataFrame with funding columns added
            
        Notes:
            - Adds estimated_settle_price as NaN (not available from TARDIS)
            - Calculates index_price_dvolume as index_price * volume
            - Calculates index_mid_price_diff as index_price - close_mid
            - Sets all funding columns to NaN if no funding data available
        """

        bars_df['estimated_settle_price'] = np.float32(np.nan)
        if funding_rates_df is not None:
            logger.info("Merging bars and funding data")
            funding_flds = ['last_funding_rate', 'index_price', 'next_funding_time', 'open_interest']
            for fld in funding_flds:
                safe_del(bars_df, fld)
            bars_df = merge_on_index(bars_df, funding_rates_df[funding_flds], how='left')
            bars_df['index_price_dvolume'] = bars_df['index_price'] * bars_df['volume']
            bars_df['index_mid_price_diff'] = bars_df['index_price'] - bars_df['close_mid']
            logger.info("Done merge")
        else:
            logger.warning("Setting binance funding rates to nan")
            for col in ['last_funding_rate', 'index_price', 'open_interest', 'index_price_dvolume', 'index_mid_price_diff']:
                bars_df[col] = np.float32(np.nan)
            bars_df['next_funding_time'] = pd.NaT
        return bars_df

    def create_resampled_bars(self, bar_date: date, bar_date_symbols: List[str]) -> Optional[pd.DataFrame]:
        """Create consolidated 1-minute bars for all symbols on a given date.
        
        Loads individual symbol files, processes them in parallel if configured,
        and combines into a single DataFrame.
        
        Args:
            bar_date: Date to create bars for
            bar_date_symbols: List of symbols to include
            
        Returns:
            Combined DataFrame with all symbols' 1-minute bars for the date,
            or None if no files available
            
        Raises:
            Exception: If no bar dataframes could be generated
            
        Notes:
            - Uses cache if date was previously loaded
            - Updates funding rates before processing bars
            - Can use multiprocessing pool for parallel file loading
            - Caches result for future use
        """
        logger.info(f"Creating 1 minute bars on {bar_date}")

        if bar_date in self.min_bar_cache:
            logger.info(f"Loading {bar_date} from cache")
            return self.min_bar_cache[bar_date]

        bar_date_symbol_venues = symbols_to_symbol_venues(bar_date_symbols)
        df = self.data_loader.load_prebar_files(start_date=bar_date, end_date=bar_date, bars_type=self.bars_type, symbol_venues=bar_date_symbol_venues)
        if df is None:
            return None
        funding_rates_df = self._load_funding_rates_for_bars(bar_date=bar_date)
        df = self._merge_funding_for_bars(bars_df=df, funding_rates_df=funding_rates_df)
        self.min_bar_cache[bar_date] = df
        return df
