"""Live bars converter for generating bars from live data.

This module provides functionality to convert live market data into bar format,
which is used for backtesting and analysis. It handles:
- Loading live bar data from disk
- Converting to standard bar format with required columns
- Handling missing data and funding time adjustments
- Generating minute bars and aggregated bars
- Supporting both legacy and new format outputs

The LiveBarsConverter class processes live trading data into standardized bar formats.
"""
import logging
import os
from datetime import date
from datetime import timedelta as td
from typing import List, Optional

import numpy as np
import pandas as pd

from lib.data.live_bars import LiveBars
from lib.util.dataframes import shrink_floats, fillin_index_df, safe_int32_cast
from lib.calcs import Calcs
from lib.calcs.calc_util import adjust_next_funding_time_jumps_in_live_bars
from lib.util.dataframes import set_index
from lib.util.directory import DirectoryManager, dir_manager
from lib.util.util import symbol_venue_to_symbol
from lib.util.time_util import date_range, date_to_start_dt, date_to_str, convert_to_datetime_series

logger = logging.getLogger(__name__)

# Minimum required columns for bar data
MIN_BAR_COLUMNS = [
    'close_mid', 'update_cnt', 'spread_avg', 'bid_sz_avg', 'ask_sz_avg', 'high_mid', 'low_mid', 'close_wgt_mid',
    'book_latency', 'logret', 'high_trade', 'low_trade', 'close_trade', 'dvolume', 'trade_cnt', 'volume',
    'bid_trade_dollars', 'ask_trade_dollars', 'trade_latency', 'vwap', 'estimated_settle_price', 'index_mid_price_diff',
    'index_price', 'index_price_dvolume', 'last_funding_rate', 'next_funding_time', 'open_interest',
]

# Columns that may be missing from live data and need to be filled with NaN
MISSING_BAR_COLUMNS = [
    'close_wgt_mid', 'low_mid', 'high_mid', 'book_latency', 'trade_latency', 'bid_trade_dollars', 'ask_trade_dollars', 'estimated_settle_price',
]


class LiveBarsConverter:
    """Convert live market data into standardized live bar format.
    
    This class handles the conversion of live trading data collected in real-time
    into the standardized bar format used throughout the trading system. It supports
    multiple generation options and output formats.
    
    Attributes:
        config: Configuration dictionary with trading parameters
        start_date: Start date for bar generation
        end_date: End date for bar generation (inclusive)
        venue: Trading venue (e.g., 'binance-futures')
        generate_option: Type of generation ('full', '1min', 'bars', 'prebars')
        pool_size: Number of parallel workers for processing
        debug: Debug mode flag
        horizons: List of time horizons to generate bars for
        universe: List of symbols to process
        calcs: Calculator instance for derived metrics
        dir_manager: Directory manager for file paths
        live_bars: LiveBars instance for loading data
    """
    
    def __init__(
            self,
            config: dict,
            start_date: date,
            end_date: date,
            venue: str,
            bars_dir_manager: DirectoryManager = dir_manager,
            pool_size: int = 6,
            debug: bool = False,
            horizons: Optional[List[int]] = None,
            use_new: bool = True
    ):
        """Initialize LiveBarsConverter with configuration and parameters.

        Args:
            config: Trading system configuration
            start_date: Start date for processing
            end_date: End date for processing
            venue: Trading venue to process
            bars_dir_manager: Directory manager for file paths
            pool_size: Number of parallel workers
            debug: Enable debug mode (no file writes)
            horizons: List of time horizons for bar aggregation
        """
        self.config = config
        self.use_new = use_new
        self.start_date = start_date
        self.end_date = end_date
        self.venue = venue
        self.pool_size = pool_size
        self.debug = debug
        self.horizons = horizons
        self.dir_manager = bars_dir_manager

        self.calcs = Calcs(config=self.config, prod=True)
        self.live_bars = LiveBars(live_bar_dir=self.dir_manager.LIVE_DATA_NEW_DIR, use_new=self.use_new)

    def _load_live_bars(self, bar_date: date) -> Optional[pd.DataFrame]:
        """Load and process live bar data for a specific date.
        
        Loads raw live data and processes it to ensure all required columns exist,
        handles missing data, adjusts funding times, and calculates log returns.
        
        Args:
            bar_date: Date to load data for

        """
        end_date = bar_date + td(days=1)
        logger.info(f'Loading live bar for {bar_date}')
        live_bars_df = self.live_bars.load_live_bars(
            start_dt=date_to_start_dt(bar_date), 
            end_dt=date_to_start_dt(end_date),
        )
        if live_bars_df is None:
            logger.warning(f'No live bar found for {bar_date}')
            return None
            
        # Fill missing columns with NaN
        for col in MISSING_BAR_COLUMNS:
            if col not in live_bars_df.columns:
                live_bars_df[col] = np.float32(np.nan)
                
        # Handle funding time column
        if 'next_funding_time' not in live_bars_df.columns:
            live_bars_df['next_funding_time'] = pd.NaT
        else:
            # Use the utility function to handle various datetime formats
            live_bars_df['next_funding_time'] = convert_to_datetime_series(
                live_bars_df['next_funding_time'],
                unit='ms',
                context='next_funding_time'
            )
            live_bars_df = adjust_next_funding_time_jumps_in_live_bars(live_bars_df)
            
        live_bars_df = self.calcs.calc_logret(live_bars_df)
        
        # Restructure index for standard format
        live_bars_df = live_bars_df.reset_index()
        live_bars_df = set_index(live_bars_df)

        #i dont know about this may cover up problems.....
        # live_bars_df = fillin_index_df(live_bars_df)

        live_bars_df = live_bars_df[MIN_BAR_COLUMNS]
        for fld in ['update_cnt', 'trade_cnt']:
            live_bars_df[fld] = safe_int32_cast(live_bars_df[fld])
        live_bars_df = shrink_floats(live_bars_df)
        return live_bars_df

    def _dump_prebars(self, live_bars_df: pd.DataFrame, bar_date: date) -> None:
        """Save bars to disk in new standardized format.
        
        Saves each symbol's data with standardized naming convention.
        New format: {dir_name}/{file_prefix}.{symbol}.parquet
        
        Args:
            dir_name: Directory to save files to
            file_prefix: Prefix for filenames (includes type and date)
        """

        dir_name = f"{self.dir_manager.PREBAR_DIR}/live/{date_to_str(bar_date)}/binance-futures"
        file_prefix = f"prebars.live.binance-futures.{date_to_str(bar_date)}"

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        for symbol_venue in sorted(live_bars_df.index.get_level_values('symbol_venue').unique()):
            symbol_idx = live_bars_df.index.get_level_values('symbol_venue') == symbol_venue
            symbol_live_bars_df = live_bars_df.loc[symbol_idx]
            symbol_live_bars_df = symbol_live_bars_df.reset_index().set_index('ts')

            symbol = symbol_venue_to_symbol(symbol_venue)
            file_path = os.path.join(dir_name, f"{file_prefix}.{symbol}.parquet")
            if self.debug:
                logger.info(f"Not dumping {file_path}")
            else:
                logger.info(f"Dumping live data to live bar to {file_path}")
                symbol_live_bars_df.to_parquet(file_path)


    def run(self):
        """Execute the bar generation based on configured option.

        Processes data according to the generation option:
        - 'prebars': Generates prebar format files for each date
        """
        for bar_date in date_range(self.start_date, self.end_date):
            live_bars_df = self._load_live_bars(bar_date)
            if live_bars_df is None:
                logger.warning(f"No live bars loaded for {bar_date}")
                continue
            self._dump_prebars(live_bars_df, bar_date)
