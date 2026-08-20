"""Forward returns calculation module for the statistical arbitrage trading system.

This module handles the computation of forward-looking returns across multiple time horizons
for use in alpha generation and model training. It calculates various types of forward returns
including raw, market-residualized, and funding-adjusted returns.

Key functionality:
- Calculates forward returns over multiple prediction horizons (15min to daily)
- Supports multiple lag periods for time-series modeling
- Handles both historical and live data sources
- Computes market-adjusted returns (equal-weighted and volume-weighted)
- Incorporates funding rate adjustments for futures contracts
- Provides scaled returns normalized by volatility

The forward returns are essential for training predictive models and evaluating
trading signal performance in backtesting and live trading scenarios.
"""

import logging.config
from datetime import datetime as dt, date
from datetime import timedelta as td
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from lib.data.live_bars import LiveBars
from lib.calcs.calcs import Calcs
from lib.util.config import extract_max_lags
from lib.data import find_latest_forward_file_date, dump_parquet_files
from lib.data.dataloader import DataLoader
from lib.universe import Universe
from lib.util.dataframes import carry_forward, concat, get_trailing_nan_mask, merge_on_index
from lib.util.time_util import compute_lookback_days, date_to_end_dt, date_to_start_dt, date_to_str, today_date, yesterday_date, date_range
from lib.util.directory import DirectoryManager, dir_manager
from lib.util.util import log_and_raise, SYMBOL_VENUE
from lib.util.logging_util import KeyLogger

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)

FOWARD_HORIZON_FROM_LIVE = [15]  # Horizons that use live data in update mode


class Forwards:
    """Manages the calculation and generation of forward returns for model training and evaluation.
    
    This class handles the computation of forward-looking returns across multiple time horizons,
    supporting both historical data processing and real-time updates from live market data.
    It calculates various return types including raw, market-residualized, and funding-adjusted
    returns that serve as target variables for predictive models.
    
    The forward returns are essential for:
    - Training alpha generation models with future price targets
    - Backtesting strategy performance with realistic look-ahead data
    - Evaluating model predictions against actual realized returns
    
    Attributes:
        config: System configuration dictionary containing model parameters
        update: Boolean flag indicating whether to use live data updates
        debug: Debug mode flag for additional logging and validation
        horizons: List of forward-looking time horizons in minutes
        horizon_to_max_lags: Mapping of horizons to maximum lag periods
        dir_manager: Directory manager for data storage paths
        data_loader: Data loader instance for retrieving market data
        calcs: Calculations module for computing returns and metrics
        symbol_venues: Universe of tradeable symbol-venue pairs
    """

    def __init__(
            self,
            config: dict,
            update: bool,
            horizons: List[int],
            debug: bool = False,
            forwards_dir_manager: DirectoryManager = dir_manager,
            output_dir: Optional[str] = None
    ):
        """Initialize the Forwards calculation engine.
        
        Args:
            config: System configuration dictionary containing:
                - Model parameters and horizons
                - Universe selection criteria
                - Risk and portfolio settings
            update: If True, incorporates live market data for recent periods
            horizons: List of forward-looking horizons in minutes (e.g., [15, 60, 1440])
            debug: Enable debug mode for additional logging and validation
            forwards_dir_manager: Directory manager for organizing output files
        
        The initialization sets up data loading infrastructure and determines
        the appropriate universe of instruments based on the configuration.
        """
        self.config = config
        self.update = update
        self.debug = debug
        self.horizons = horizons
        self.horizon_to_max_lags = extract_max_lags(self.config)
        self.dir_manager = forwards_dir_manager
        self.output_dir = output_dir
        self.data_loader = DataLoader(self.config, self.dir_manager)
        self.horizon2lag = extract_max_lags(self.config)
        self.universe = Universe(config=self.config, debug=self.debug, universe_dir_manager=self.dir_manager)
        self.calcs = Calcs(config=config, prod=True)

    def _calculate_forward_returns_from_live_bars(self, horizon: int, start_date: date, end_date: date, symbol_venues: List[str]) -> pd.DataFrame:
        """Calculate forward returns by combining historical bars with live market data.
        
        This method is used when update mode is enabled to incorporate the most recent
        market data that may not yet be in the historical database. It loads minute bars
        and live data, then calculates rolling forward returns over the specified horizon.
        
        Args:
            horizon: Time horizon in minutes for forward return calculation
            start_date: Start date for historical data loading
            end_date: End date for historical data (live data starts after this)
        
        Returns:
            DataFrame with columns for each return type suffixed by horizon:
            - logret_{horizon}: Raw log returns
            - logret_resid_eqmkt_{horizon}: Equal-weighted market residualized returns
            - logret_resid_wgtmkt_{horizon}: Volume-weighted market residualized returns
            - logret_funding_adj_{horizon}: Funding-adjusted returns
            - logret_funding_adj_resid_eqmkt_{horizon}: Funding-adjusted + eq-weighted residual
            - logret_funding_adj_resid_wgtmkt_{horizon}: Funding-adjusted + vol-weighted residual
        
        The method ensures continuity between historical and live data by:
        1. Loading minute bars up to end_date
        2. Loading live bars from end_date + 1 day forward
        3. Concatenating and forward-filling metadata fields
        4. Calculating all return types using rolling windows
        """
        min_bars_df = self.data_loader.load_bars(
            horizon=1,
            start_date=start_date,
            end_date=end_date,
            cols=['close_mid', 'advp', 'update_cnt_1', 'volume_1', 'dvolume_1', 'last_funding_rate', 'next_funding_time'],
            symbol_venues=symbol_venues,
        )
        min_bars_df = min_bars_df.rename(columns={'update_cnt_1': 'update_cnt', 'dvolume_1': 'dvolume', 'volume_1': 'volume'})
        live_bars = LiveBars(live_bar_dir=self.dir_manager.LIVE_DATA_DIR, universe=symbol_venues, use_new=True)
        live_bars_df = live_bars.load_live_bars(start_dt=date_to_start_dt(end_date) + td(days=1), end_dt=date_to_start_dt(end_date) + td(days=1) + td(minutes=2 * horizon + 5))
        live_bars_df = live_bars_df[['close_mid', 'update_cnt', 'volume', 'dvolume', 'last_funding_rate', 'next_funding_time']]

        min_bars_df = concat([min_bars_df, live_bars_df])
        min_bars_df = carry_forward(min_bars_df, ['advp'])
        min_bars_df = self.calcs.calc_returns(df=min_bars_df)
        fields = ['logret', 'logret_resid_eqmkt', 'logret_resid_wgtmkt', 'logret_funding_adj', 'logret_funding_adj_resid_eqmkt', 'logret_funding_adj_resid_wgtmkt']

        flds_df = min_bars_df[fields].unstack()
        trailing_nan_mask = get_trailing_nan_mask(flds_df)
        rolling_result = (
            flds_df
            .rolling(f"{horizon}min", min_periods=1)
            .agg('sum')
            .ffill()
            .mask(trailing_nan_mask)
            .astype(np.float32)
            .stack(future_stack=True)
        )
        suffixed_columns = [f"{fld}_{horizon}" for fld in fields]
        rolling_result.columns = suffixed_columns
        return rolling_result

    def _calculate_forward_returns(
            self,
            bars_df: pd.DataFrame,
            alpha_lags: int,
            horizon: int,
            lookback_days: int,
            end_date: Optional[date] = None,
            produce_scaled_fields: bool = True,
            funding_adjusted: bool = False,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Calculate multi-period forward returns for model training targets.
        
        This static method computes forward-looking returns over multiple lag periods,
        creating target variables for time-series prediction models. It generates returns
        in three forms: raw, equal-weighted market residualized, and volume-weighted
        market residualized.
        
        Args:
            bars_df: DataFrame containing historical bar data with return columns
            alpha_lags: Number of forward periods to calculate (e.g., 5 means y1 to y5)
            horizon: Time horizon in minutes for each forward period
            lookback_days: Number of days used for volatility scaling calculations
            end_date: Maximum date for forward return calculations
            produce_scaled_fields: If True, creates volatility-scaled return fields
            funding_adjusted: If True, uses funding-adjusted returns as base
        
        Returns:
            Tuple containing:
            - DataFrame with original data plus forward return columns
            - List of column names that were added
        
        The method creates columns with naming convention:
        - y{funding_adj}_raw{lag}_{horizon}: Cumulative raw returns
        - y{funding_adj}_resid_eqmkt{lag}_{horizon}: Cumulative eq-weighted residual returns
        - y{funding_adj}_resid_wgtmkt{lag}_{horizon}: Cumulative vol-weighted residual returns
        - y{funding_adj}_scaled_raw{lag}_{horizon}: Volatility-scaled returns (if enabled)
        
        Example:
            For horizon=60 and alpha_lags=3:
            - y_raw1_60: 60-minute forward return
            - y_raw2_60: 120-minute cumulative forward return
            - y_raw3_60: 180-minute cumulative forward return
        
        Raises:
            Exception: If all forward returns are NaN, indicating data issues
        """

        logger.info(f"Calculating forward returns with {alpha_lags=} at {horizon=} and {lookback_days=} and {funding_adjusted=}")
        funding_adjusted_str = '_funding_adj' if funding_adjusted else ''

        raw_logret_col = f'logret{funding_adjusted_str}_{horizon}'
        resid_logret_col = f'logret{funding_adjusted_str}_resid_eqmkt_{horizon}'
        wgt_resid_logret_col = f'logret{funding_adjusted_str}_resid_wgtmkt_{horizon}'

        raw_col = f'y{funding_adjusted_str}_raw1_{horizon}'
        resid_col = f'y{funding_adjusted_str}_resid_eqmkt1_{horizon}'
        wgt_resid_col = f'y{funding_adjusted_str}_resid_wgtmkt1_{horizon}'
        cols_to_add = [raw_col, resid_col, wgt_resid_col]

        logger.info(f"Calculating forward return 1 for horizon {horizon}")
        raw_unstacked_df = bars_df[[raw_logret_col]].unstack()
        resid_unstacked_df = bars_df[[resid_logret_col]].unstack()
        wgt_resid_unstacked_df = bars_df[[wgt_resid_logret_col]].unstack()

        fwd_ret_raw_df = raw_unstacked_df.shift(-horizon).stack(future_stack=True).rename(columns={raw_logret_col: raw_col}).astype(np.float32)
        fwd_ret_resid_df = resid_unstacked_df.shift(-horizon).stack(future_stack=True).rename(columns={resid_logret_col: resid_col}).astype(np.float32)
        fwd_ret_wgt_resid_df = wgt_resid_unstacked_df.shift(-horizon).stack(future_stack=True).rename(columns={wgt_resid_logret_col: wgt_resid_col}).astype(np.float32)
        ret_df = pd.concat([fwd_ret_raw_df, fwd_ret_resid_df, fwd_ret_wgt_resid_df], axis=1)
        df = merge_on_index(bars_df, ret_df)

        if (len(df[~df[raw_logret_col].isna()]) == 0) or (len(df[~df[resid_logret_col].isna()]) == 0):
            raise log_and_raise(f"All nans generated for forwards at {horizon=}, lag: 1, exiting...")

        for ii in range(2, alpha_lags + 2):
            logger.info(f"Calculating forward return {ii} for horizon {horizon}")
            raw_col = f'y{funding_adjusted_str}_raw{ii}_{horizon}'
            resid_col = f'y{funding_adjusted_str}_resid_eqmkt{ii}_{horizon}'
            wgt_resid_col = f'y{funding_adjusted_str}_resid_wgtmkt{ii}_{horizon}'

            horizon_shift = -horizon * ii

            fwd_ret_raw_df = raw_unstacked_df.shift(horizon_shift).stack(future_stack=True).rename(columns={raw_logret_col: raw_col}).astype(np.float32)
            fwd_ret_resid_df = resid_unstacked_df.shift(horizon_shift).stack(future_stack=True).rename(columns={resid_logret_col: resid_col}).astype(np.float32)
            fwd_ret_wgt_resid_df = wgt_resid_unstacked_df.shift(horizon_shift).stack(future_stack=True).rename(columns={wgt_resid_logret_col: wgt_resid_col}).astype(np.float32)
            ret_df = pd.concat([fwd_ret_raw_df, fwd_ret_resid_df, fwd_ret_wgt_resid_df], axis=1)

            ret_nona_df = ret_df.dropna(subset=[raw_col, resid_col])
            if len(ret_nona_df) == 0:
                print(horizon_shift)
                print(raw_unstacked_df)
                print(raw_unstacked_df.shift(horizon_shift))
                log_and_raise(f"No rows after dropping NAs on {raw_col} and {resid_col}...", df=ret_df)
            ret_df = ret_nona_df

            df = merge_on_index(df, ret_df)

            if (len(df[~df[raw_col].isna()]) == 0) or (len(df[~df[resid_col].isna()]) == 0):
                logger.error(f"All nans generated for forwards at {horizon=}, lag: {ii}, exiting...", key="forwards generated with all nan")
                continue

            df[raw_col] = (df[f'y{funding_adjusted_str}_raw{ii - 1}_{horizon}'].fillna(0) + df[raw_col]).astype(np.float32)
            df[resid_col] = (df[f'y{funding_adjusted_str}_resid_eqmkt{ii - 1}_{horizon}'].fillna(0) + df[resid_col]).astype(np.float32)
            df[wgt_resid_col] = (df[f'y{funding_adjusted_str}_resid_wgtmkt{ii - 1}_{horizon}'].fillna(0) + df[wgt_resid_col]).astype(np.float32)

            cols_to_add += [raw_col, resid_col, wgt_resid_col]

        if produce_scaled_fields:
            for ii in range(1, alpha_lags + 1):
                scaled_col = f'y{funding_adjusted_str}_scaled_raw{ii}_{horizon}'
                logger.info(f"Calculating {scaled_col}")
                # might need fix if scaled by logret trailing std for funding adjusted forwards, however, produce_scaled_fields is set to False now so should be fine
                df[scaled_col] = df[f'y{funding_adjusted_str}_raw{ii}_{horizon}'].div(df[f'logret_{horizon}_trstd'] * np.sqrt(ii))
                cols_to_add += [scaled_col]

        # clean up df if it's calculating return horizon from live and at the funding adjusted stage
        if funding_adjusted and horizon in FOWARD_HORIZON_FROM_LIVE and end_date is not None:
            df = df[df.index.get_level_values('ts') <= date_to_end_dt(end_date)]

        # check df shape except for return horizon from live and not at the funding adjusted stage
        if not (horizon in FOWARD_HORIZON_FROM_LIVE and not funding_adjusted):
            assert len(df) % 1440 == 0
        return df, cols_to_add


    def generate_horizon_forwards(self, horizon: int, start_date: date, end_date: date, update: bool) -> None:
        lags = self.horizon_to_max_lags[horizon] + 1
        lookback_days = max(compute_lookback_days(horizon, lags), 1)

        if update:
            forward_start_date = find_latest_forward_file_date(horizon)
            forward_end_date = end_date - td(days=lookback_days)
            forward_start_date = min(forward_start_date, forward_end_date)
            logger.info(f"Found latest file for {horizon} at {date_to_str(forward_start_date)}")
        else:
            forward_start_date = start_date
            forward_end_date = end_date
            assert forward_start_date + td(days=lookback_days) < today_date()

        logger.info(f"Generating forwards at {horizon} with {lags=} {lookback_days=} {date_to_str(forward_start_date)} - {date_to_str(forward_end_date)}")
        assert forward_start_date <= forward_end_date

        for bar_start_date in date_range(forward_start_date, forward_end_date):
            logger.info(f"Forwards for {horizon} as of {date_to_str(bar_start_date)}")

            bar_end_date = bar_start_date + td(days=lookback_days)

            symbol_venues = self.universe.load_universe_symbols(
                universe_source='file',
                universe_date=bar_start_date - td(days=1),
                symbol_type=SYMBOL_VENUE,
                filter='fittable'
            )
            logger.info(f"Calculating forward returns at {horizon} {lookback_days=} with bar data from {bar_start_date} - {bar_end_date}")
            logger.info(symbol_venues)

            if bar_end_date >= today_date():
                log_and_raise(f"Not enough look forward for {horizon=}: desired {date_to_str(bar_end_date)}")

            bars_cols = [
                f'logret_{horizon}',
                f'logret_resid_eqmkt_{horizon}',
                f'logret_resid_wgtmkt_{horizon}',
                f'logret_funding_adj_{horizon}',
                f'logret_funding_adj_resid_eqmkt_{horizon}',
                f'logret_funding_adj_resid_wgtmkt_{horizon}'
            ]

            # only using live bars for 15 min horizon in update mode and fdate equals to end_date
            if self.update and horizon in FOWARD_HORIZON_FROM_LIVE and bar_start_date == end_date:
                logger.info(f"Calculating forward returns from live bars for {horizon=}, {bar_start_date=} - {bar_end_date=}")
                bars_df = self._calculate_forward_returns_from_live_bars(
                    horizon=horizon,
                    start_date=bar_start_date,
                    end_date=bar_end_date,
                    symbol_venues=symbol_venues
                )
            else:
                bars_df = self.data_loader.load_bars(
                    horizon=horizon,
                    start_date=bar_start_date,
                    end_date=bar_end_date,
                    cols=bars_cols,
                    symbol_venues=symbol_venues,
                )

            forwards_df, y_cols_to_dump_non_funding = self._calculate_forward_returns(
                bars_df=bars_df,
                alpha_lags=lags,
                horizon=horizon,
                lookback_days=lookback_days,
                produce_scaled_fields=False,
                funding_adjusted=False,
            )
            forwards_df, y_cols_to_dump_funding = self._calculate_forward_returns(
                bars_df=forwards_df,
                alpha_lags=lags,
                horizon=horizon,
                lookback_days=lookback_days,
                end_date=bar_end_date,
                produce_scaled_fields=False,
                funding_adjusted=True,
            )

            max_ts = min(date_to_end_dt(end_date), forwards_df.index.get_level_values('ts').max() - td(days=lookback_days))
            # could keep one more day when we calculate return from live
            if self.update and horizon in FOWARD_HORIZON_FROM_LIVE:
                max_ts += td(days=1)
            forwards_df = forwards_df[forwards_df.index.get_level_values('ts') <= max_ts]
            y_cols_to_dump = y_cols_to_dump_non_funding + y_cols_to_dump_funding
            output_dir = self.output_dir if self.output_dir is not None else self.dir_manager.FORWARDS_DIR
            if self.debug:
                print("Debug forwards:")
                print(forwards_df)
            else:
                dump_parquet_files(file_type='forwards', df=forwards_df, directory=f"{output_dir}/{horizon}", name=f"forwards.{horizon}", cols=y_cols_to_dump, debug=self.debug)

    def generate_forwards(self, start_date: Optional[date] = None, end_date: Optional[date] = None):
        """Generate and save forward returns for all configured horizons.
        
        This is the main entry point for forward return generation. It processes each
        horizon sequentially, calculating forward returns for the specified date range
        and saving them to parquet files for use in model training and backtesting.
        
        Args:
            start_date: First date to generate forward returns
            end_date: Last date to generate forward returns
        
        The method performs the following for each horizon:
        1. Determines appropriate lag periods based on configuration
        2. Calculates required buffer days for forward-looking data
        3. Loads historical bar data with sufficient look-ahead period
        4. Optionally incorporates live data for recent periods
        5. Calculates both standard and funding-adjusted forward returns
        6. Saves results to parquet files organized by horizon
        
        Output Structure:
        - Live/New format: /data/forwards/{horizon}/forwards.{horizon}.{date}.parquet
        - Tardis format: /data/tardis_forwards/forward_{horizon}_{date}.parquet
        
        Notes:
        - In update mode, 15-minute horizon incorporates live bar data
        - Ensures no look-ahead bias by respecting buffer day constraints
        """

        update = start_date is None and end_date is None
        if update:
            end_date = yesterday_date()
        else:
            assert end_date < today_date()
            assert start_date <= end_date

        logger.info(f"Generating forwards to {date_to_str(end_date)}")

        for horizon in sorted(self.horizons):
            try:
                self.generate_horizon_forwards(horizon, start_date, end_date, update)
            except Exception as e:
                logger.error(f"Could not generate forwards at {horizon=} : {e}")