"""Server calculations orchestrator for real-time trading system.

This module provides the ServerCalcs class which coordinates all real-time
calculations across multiple time horizons for the live trading server. It manages:
- Loading and updating live market data (bars)
- Rolling window calculations across different time horizons
- Feature engineering updates
- Model signal calculations
- Alpha generation for portfolio optimization

The ServerCalcs class acts as the main orchestrator, delegating horizon-specific
calculations to ServerHorizonCalcs instances while managing the overall workflow
and data flow between components.

Typical usage:
    >>> config = load_config()
    >>> server_calcs = ServerCalcs(
    ...     config=config,
    ...     eod_today=dt.now(timezone.utc),
    ...     feature_horizons=[15, 60, 120, 1440],
    ...     short_term_horizons=[15, 60],
    ...     model_horizons=[15, 60, 120, 1440],
    ...     feature_flds_to_load={60: ['beta_60', 'rsi_60']}
    ... )
    >>> 
    >>> # Update with new market data
    >>> if server_calcs.update():
    ...     # Get latest alpha signals
    ...     alphas_df, alpha_cols = server_calcs.calculate_alphas(horizons=[60, 1440])
"""

import logging
from collections import defaultdict
from datetime import datetime as dt
from datetime import timedelta as td
from multiprocessing.dummy import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lib.bars.bar_generator import BAR_FLDS_AGG_DICT, FIXED_COLS_TO_PERSIST
from lib.calcs.calc_util import adjust_next_funding_time_jumps_in_live_bars
from lib.calcs import Calcs
from lib.data.data_news import load_news
from lib.util.dataframes import carry_forward, check_df, concat, get_min_max_ts, merge_on_index, cols_to_list_str, safe_int32_cast, make_symbol, set_index, clean_poop
from lib.data.dataloader import DataLoader
from lib.util.directory import dir_manager, DirectoryManager
from lib.alpha.forecasts import Forecasts, calculate_horizon_alphas
from lib.data.live_bars import LiveBars
from lib.util.logging_util import KeyLogger
from .server_horizon_calcs import ServerHorizonCalcs
from lib.util.time_util import today_date
from lib.util.util import log_mem_usage, unique_list, log_and_raise
from ..util import shrink_floats

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


class ServerCalcs:
    """Orchestrates real-time calculations across multiple time horizons for live trading.
    
    This class manages the overall workflow for processing live market data and generating
    alpha signals. It coordinates:
    - Loading live bar data from files
    - Updating rolling calculations across different time horizons
    - Running feature engineering calculations
    - Computing model signals
    - Generating final alpha values for portfolio optimization
    
    The class maintains separate ServerHorizonCalcs instances for each time horizon,
    allowing parallel processing while ensuring data consistency across horizons.
    
    Attributes:
        config (dict): Configuration dictionary with trading parameters
        eod_today (dt): End-of-day timestamp for today
        feature_horizons (List[int]): Time horizons for feature calculation (e.g., [15, 60, 120, 1440])
        short_term_horizons (List[int]): Horizons considered short-term (e.g., [15, 60])
        model_horizons (List[int]): Horizons for model calculation
        dir_manager (DirectoryManager): Manager for file paths
        data_loader (DataLoader): Data loading utility
        opt_horizon (int): Optimization horizon from config
        st_risk_horizon (int): Short-term risk horizon from config
        volume_bucket_mins (int): Minutes for volume bucketing
        similarity_threshold (float): Threshold for news deduplication
        live_bars (LiveBars): Utility for loading live bar data
        calcs (Calcs): Calculator for various metrics
        horizon_calcs (Dict[int, ServerHorizonCalcs]): Horizon-specific calculators
        forecasts (Forecasts): Forecast/alpha generator
        feature_flds_to_load (Dict[int, List[str]]): Fields to load by horizon
    """

    def __init__(
            self,
            config: dict,
            eod_today: dt,
            feature_horizons: List[int],
            short_term_horizons: List[int],
            model_horizons: List[int],
            feature_flds_to_load: Dict[int, List[str]],
            server_calcs_dir_manager: DirectoryManager = dir_manager,
            debug: bool = False,
            universe: Optional[List[str]] = None,
            new_bars: bool = True,
            centering_universe: Optional[List[str]] = None
    ):
        """Initialize ServerCalcs with configuration and horizon specifications.

        Args:
            config: Configuration dictionary containing:
                - OPT_HORIZON: Optimization horizon
                - ST_RISK_HORIZON: Short-term risk horizon
                - VOLUME_BUCKET_MINS: Volume bucketing period
                - NEWS_SIMILARITY_THRESHOLD: News dedup threshold
            eod_today: End-of-day datetime for today
            feature_horizons: List of horizons for feature calculation
            short_term_horizons: List of horizons considered short-term
            model_horizons: List of horizons for model calculation
            feature_flds_to_load: Dictionary mapping horizon to fields to load
            server_calcs_dir_manager: Directory manager for paths
            universe: Optional list of symbols to restrict to
            centering_universe: Optional list of symbols to use for cross-sectional
                centering in alpha calculations. If None, uses all symbols in the
                dataframe. Use this to ensure consistency with simulation, which
                uses only universe-file symbols for centering.
        """
        self.config = config
        self.debug = debug
        self.eod_today = eod_today
        self.feature_horizons = feature_horizons
        self.short_term_horizons = short_term_horizons
        self.model_horizons = model_horizons

        self.dir_manager = server_calcs_dir_manager
        self.data_loader = DataLoader(self.config, self.dir_manager)

        self.opt_horizon = self.config['OPT_HORIZON']
        self.st_risk_horizon = self.config['ST_RISK_HORIZON']
        self.volume_bucket_mins = self.config['VOLUME_BUCKET_MINS']
        self.similarity_threshold = self.config['NEWS_SIMILARITY_THRESHOLD']

        self.live_bars = LiveBars(live_bar_dir=self.dir_manager.LIVE_DATA_DIR, universe=universe, use_new=new_bars)
        self.calcs = Calcs(config=config, prod=True)
        self.horizon_calcs: Dict[int, ServerHorizonCalcs] = {}
        logger.info('Set forecasts for server calcs')

        fit_as_of = self.eod_today - td(days=1) if self.debug else None
        self.forecasts = Forecasts(
            config=config,
            debug=self.debug,
            prod=True,
            is_server_forecast=True,
            forecast_dir_manager=self.dir_manager,
            fit_as_of=fit_as_of
        )

        self.feature_flds_to_load = feature_flds_to_load
        self.centering_universe = centering_universe
        if centering_universe:
            logger.info(f"Using centering_universe with {len(centering_universe)} symbols for alpha centering")

    def make_server_horizon_calc(self, horizon: int, df: pd.DataFrame):
        """Create and store a ServerHorizonCalcs instance for a specific horizon.
        
        Args:
            horizon: Time horizon in minutes
            df: Initial DataFrame with features/bars for this horizon
        """
        self.horizon_calcs[horizon] = ServerHorizonCalcs(
            config=self.config,
            horizon=horizon,
            features_df=df,
            server_horizon_calcs_dir_manager=self.dir_manager
        )

    def get_horizon_calc(self, horizon: int) -> ServerHorizonCalcs:
        """Get the ServerHorizonCalcs instance for a specific horizon.
        
        Args:
            horizon: Time horizon in minutes
            
        Returns:
            ServerHorizonCalcs instance for the requested horizon
            
        Raises:
            ValueError: If no ServerHorizonCalcs exists for the requested horizon
        """
        hc = self.horizon_calcs.get(horizon)
        if hc is None:
            raise log_and_raise(f"No server data for {horizon=}")
        return hc

    def get_last_processed_ts(self) -> dt:
        """Get the last processed timestamp from the 1-minute horizon.
        
        The 1-minute horizon is used as the reference for what data has been
        processed across all horizons.
        
        Returns:
            Last processed timestamp
        """
        return self.get_horizon_calc(horizon=1).last_processed_ts

    def get_horizon_df(self, horizon: int) -> pd.DataFrame:
        """Get the features DataFrame for a specific horizon.
        
        Args:
            horizon: Time horizon in minutes
            
        Returns:
            Features DataFrame for the requested horizon
        """
        hc = self.get_horizon_calc(horizon=horizon)
        df = hc.features_df
        return df

    def get_latest_features(self) -> pd.DataFrame:
        """Get the latest features across all horizons at the last processed timestamp.
        
        Combines features from all horizons into a single DataFrame containing
        only the most recent timestamp. Avoids duplicate columns by tracking
        which columns have already been added.
        
        Returns:
            DataFrame with all features at the last processed timestamp
        """
        start_ts = self.get_last_processed_ts()
        all_features_dfs = []
        seen_cols = []
        for hc in self.horizon_calcs.values():
            features_df = hc.features_df
            latest_features_df = features_df[features_df.index.get_level_values('ts') == start_ts]
            cols_to_add = sorted([c for c in latest_features_df if c not in seen_cols])
            logger.info(f"Adding horizon_calc columns from {hc.horizon}: {cols_to_add}")
            all_features_dfs.append(latest_features_df[cols_to_add].sort_index())
            seen_cols += cols_to_add
            logger.info(f"")
        latest_alphas_df = pd.concat(all_features_dfs, verify_integrity=True, axis=1).sort_index()
        logger.info(f"Columns for model processing: {cols_to_list_str(latest_alphas_df)}")
        return latest_alphas_df

    def update_horizon_rolling_fields(
            self,
            horizon: int,
            new_bars_df: pd.DataFrame,
            previous_bars_ts: dt,
            existing_flds: List[str],
            one_min_bars_df_unstacked: pd.DataFrame,
    ):
        """Update rolling window calculations for a specific horizon.
        
        Calculates rolling aggregations (mean, sum, min, max, etc.) over the
        specified horizon window for various bar fields. For the 1-minute horizon,
        simply updates with new data. For other horizons, performs rolling window
        calculations using a buffer of historical data.
        
        The function handles:
        - Fixed columns that are carried forward without aggregation
        - Bar fields that need rolling aggregations based on BAR_FLDS_AGG_DICT
        - VWAP calculations for the horizon
        - Special handling for news and position age fields
        
        Args:
            horizon: Time horizon in minutes
            new_bars_df: New bar data to process
            previous_bars_ts: Timestamp of previous bar data
            existing_flds: List of existing fields in the data
            one_min_bars_df_unstacked: Unstacked 1-minute bars for buffering
        """
        hc = self.horizon_calcs[horizon]
        if horizon == 1:
            hc.update_df(new_bars_df)
            return

        logger.info(f"Updating rolling fields at {horizon=} from {previous_bars_ts}")
        bar_flds = list(set(existing_flds) & set(BAR_FLDS_AGG_DICT.keys()))
        existing_horizon_bars_df = one_min_bars_df_unstacked.iloc[-horizon:].stack(future_stack=True)
        buffered_horizon_bars_df = concat([existing_horizon_bars_df, new_bars_df], quiet=True).sort_index()
        buffered_horizon_bars_df['twap'] = buffered_horizon_bars_df['close_mid']
        buffered_horizon_bars_df = buffered_horizon_bars_df.unstack()
        fld_dfs = []
        if 'news_title' not in new_bars_df:
            new_bars_df['news_title'] = None

        for fld in FIXED_COLS_TO_PERSIST + ['news_title']:
            fld_dfs.append(new_bars_df[[fld]].sort_index())

        for fld in bar_flds:
            # hacky...
            if fld == 'advp':
                continue
            methods = BAR_FLDS_AGG_DICT[fld]
            fld_df = buffered_horizon_bars_df[[fld]]
            fld_rolling = fld_df.rolling(f"{horizon}min")
            for method in methods:
                logger.info(f"Updating rolling bars at {horizon}:{fld} {method=}")
                fld_agg_df = fld_rolling.agg(method)
                fld_agg_df = fld_agg_df[fld_agg_df.index > previous_bars_ts]
                fld_agg_df = fld_agg_df.stack(future_stack=True).sort_index()
                fld_name = f"{fld}_{horizon}" if len(methods) == 1 else f"{fld}_{method}_{horizon}"
                fld_agg_df[fld_name] = fld_agg_df[fld].astype(np.float32)
                fld_dfs.append(fld_agg_df[[fld_name]])

        horizon_bars_df = pd.concat(fld_dfs, axis=1, verify_integrity=True).sort_index()
        horizon_bars_df = self.calcs.calc_vwap(horizon_bars_df, horizon)
        horizon_bars_df = self.calcs.calc_vwap(horizon_bars_df, horizon, vwap_fld="index_price_vwap", dvolume_fld="index_price_dvolume", volume_fld="volume")
        # set to nan so we carry forward when calculate_features
        if 'position_age' not in horizon_bars_df:
            horizon_bars_df['position_age'] = np.nan

        for fld in [f"update_cnt_{horizon}", f"trade_cnt_{horizon}", 'position_age']:
            horizon_bars_df[fld] = safe_int32_cast(horizon_bars_df[fld])

        check_df(horizon_bars_df)
        hc.update_df(horizon_bars_df)
        log_mem_usage()

    def update_rolling_fields(self, new_bars_df: pd.DataFrame):
        """Update rolling window calculations across all horizons in parallel.
        
        Coordinates the update of rolling calculations for all configured horizons
        using multiprocessing for efficiency. Ensures consistent universe across
        horizons by filtering new data to match existing symbols.
        
        Args:
            new_bars_df: New bar data to process across all horizons
        """
        one_min_bars_df = self.get_horizon_df(horizon=1)
        previous_bars_ts = one_min_bars_df.index.get_level_values('ts').max()

        # keep a consistent universe
        universe = list(one_min_bars_df.index.get_level_values('symbol_venue').unique())
        new_bars_df = new_bars_df[new_bars_df.index.get_level_values('symbol_venue').isin(universe)]

        existing_flds = list(one_min_bars_df.columns)
        one_min_bars_df_unstacked = one_min_bars_df.unstack()

        horizons = self.horizon_calcs.keys()
        pool = Pool(len(horizons))
        for _ in pool.starmap(self.update_horizon_rolling_fields, [(hh, new_bars_df, previous_bars_ts, existing_flds, one_min_bars_df_unstacked) for hh in horizons]):
            pass
        pool.close()
        pool.join()

    def update_bars(self) -> bool:
        """Load and process new live bar data.
        
        Performs the complete bar update workflow:
        1. Loads new live bars from files starting after last processed timestamp
        2. Adjusts funding time jumps to handle exchange quirks
        3. Merges news events if available
        4. Adds delisting dates for risk management
        5. Initializes return fields to NaN (calculated later)
        6. Concatenates with previous bar for return calculation
        7. Carries forward specific fields like ADVP and fittable
        8. Calculates returns based on previous close
        9. Updates rolling calculations across all horizons
        
        Returns:
            True if new bars were loaded and processed, False if no new data
        """

        last_bar_ts = self.get_horizon_calc(horizon=1).get_max_ts()
        live_bars_end_dt = self.eod_today if self.debug else None
        live_bars_df = self.live_bars.load_live_bars(start_dt=last_bar_ts, end_dt=live_bars_end_dt)
        if live_bars_df is None:
            return False
        live_bars_df = adjust_next_funding_time_jumps_in_live_bars(live_bars_df)
        news_df = load_news(similarity_threshold=self.similarity_threshold, start_date=today_date(), end_date=today_date(), news_dir=self.dir_manager.NEWS_DIR)
        if news_df is not None:
            live_bars_df = merge_on_index(live_bars_df, news_df['news_title'], debug=True)
            new_news_events = len(live_bars_df[~live_bars_df['news_title'].isna()])
            logger.info(f"Loaded {new_news_events} new news events...")
        else:
            logger.warning("Could not load live news")

        start_ts, latest_ts = get_min_max_ts(live_bars_df)
        logger.info(f"Loaded {len(live_bars_df)} live bars from {start_ts} until {latest_ts}")

        last_bars_df = self.get_horizon_df(horizon=1)
        for fld in ['logret_funding_adj', 'logret_funding_adj_resid_eqmkt', 'logret_funding_adj_resid_wgtmkt']:
            if fld in last_bars_df.columns:
                live_bars_df[fld] = np.nan

        last_previous_ts = last_bars_df.index.get_level_values('ts').max()
        last_previous_ts_df = last_bars_df[last_bars_df.index.get_level_values('ts') == last_previous_ts]

        live_bars_df = self.data_loader.add_delisting_dates(live_bars_df, start_date=today_date(), end_date=today_date())
        # XXX i think these logrets are generating errors somewhere in the logs
        carry_forward_flds = ['advp']
        for fld in carry_forward_flds + ['logret', 'logret_resid_eqmkt', 'logret_resid_wgtmkt']:
            #XXX don't like that it sets fittable to nan...
            live_bars_df[fld] = np.nan

        live_bars_df = make_symbol(live_bars_df)
        bool_flds = ['fittable', 'priceable', 'tradeable', 'expandable']
        live_bars_df = clean_poop(set_index(pd.merge(
            live_bars_df.reset_index(), last_previous_ts_df[bool_flds].reset_index(),
            on='symbol_venue', how='left', suffixes=('', '_poop')))
        )
        live_bars_df = shrink_floats(live_bars_df)
        bars_df = concat([last_previous_ts_df, live_bars_df], strict=False)
        bars_df = carry_forward(bars_df, carry_forward_flds)

        # need 1 before so we can calc the first return
        bars_df = bars_df[bars_df.index.get_level_values('ts') >= last_previous_ts]
        bars_df = self.calcs.calc_returns(bars_df)
        bars_df = bars_df[bars_df.index.get_level_values('ts') > last_previous_ts]

        self.update_rolling_fields(new_bars_df=bars_df)
        return True

    def calculate_horizon_feature(self, horizon: int, carry_forward_map: Dict[int, List[str]]):
        """Calculate features for a specific horizon.
        
        Delegates to the horizon-specific ServerHorizonCalcs instance.
        
        Args:
            horizon: Time horizon in minutes
            carry_forward_map: Fields to carry forward by horizon
        """
        self.get_horizon_calc(horizon).calculate_features(carry_forward_map=carry_forward_map)

    def calculate_features(self, st_only: bool = False) -> None:
        """Calculate features across all configured horizons in parallel.
        
        Coordinates feature calculation with appropriate carry-forward logic:
        - Always carries forward 'fittable' for 1-minute horizon
        - Carries forward fields specified in feature_flds_to_load
        - Carries forward 'position_age' for risk horizons
        - Carries forward 'beta_1440' for all feature horizons
        
        Short-term mode processes only short-term horizons plus 1440 (daily),
        as some daily features are needed even for short-term models.
        
        Args:
            st_only: If True, only calculate short-term features
        """
        logger.info(f"Calculating features, Short term only: {st_only}")

        carry_forward_map = defaultdict(list)
        carry_forward_map[1] += ['fittable', 'priceable', 'tradeable', 'expandable', 'marketcap', 'mkt_wgt']

        for horizon, flds in self.feature_flds_to_load.items():
            carry_forward_map[horizon] += flds

        # we call calculate_risk for st_risk_horizon and opt_horizon, so carry forward position_age
        carry_forward_map[self.st_risk_horizon] += ['position_age']
        carry_forward_map[self.opt_horizon] += ['position_age']

        # even short term forecasts need some of the 1440 features...
        short_term_horizons = unique_list(self.short_term_horizons + [1440])
        feature_horizons = short_term_horizons if st_only else self.feature_horizons
        horizons = sorted(unique_list(feature_horizons + [1, 1440]))

        # should carry forward beta_1440 for every feature horizons if not already
        for hh in horizons:
            if hh == 1 or hh in self.feature_flds_to_load:
                continue
            carry_forward_map[hh] += ['beta_1440', 'mdvp']

        logger.info(f"Calling server calcs for horizons: {horizons}")

        pool = Pool(len(horizons))
        for _ in pool.starmap(self.calculate_horizon_feature, [(hh, carry_forward_map) for hh in horizons]):
            pass
        pool.close()
        pool.join()

    def calculate_and_lag_horizon_model(self, horizon: int):
        """Calculate model signals and lags for a specific horizon.
        
        Delegates to the horizon-specific ServerHorizonCalcs instance.
        
        Args:
            horizon: Time horizon in minutes
        """
        self.get_horizon_calc(horizon).calculate_and_lag_models()

    def calculate_models(self, st_only: bool = False) -> None:
        """Calculate model signals across configured horizons in parallel.
        
        Processes either short-term horizons only or all model horizons,
        computing alpha model signals and their lagged features.
        
        Args:
            st_only: If True, only calculate short-term models
        """

        if st_only:
            model_horizons = self.short_term_horizons
            if len(model_horizons) == 0:
                logger.info(f"No short term models to calculate")
                return
        else:
            model_horizons = self.model_horizons

        pool = Pool(len(model_horizons))
        for _ in pool.starmap(self.calculate_and_lag_horizon_model, [(hh,) for hh in model_horizons]):
            pass
        pool.close()
        pool.join()

    def calculate_alphas(self, horizons: List[int]) -> Tuple[pd.DataFrame, List[str]]:
        """Calculate final alpha signals for specified horizons.
        
        Generates alpha signals by:
        1. Getting latest features/models at the last processed timestamp
        2. Computing model alphas using the Forecasts class
        3. Calculating combined horizon alphas
        
        This is typically called by the AlphaServer to get signals for
        portfolio optimization.
        
        Args:
            horizons: List of horizons to calculate alphas for
            
        Returns:
            Tuple of (alphas_df, alpha_column_names) where:
            - alphas_df: DataFrame with alpha signals at latest timestamp
            - alpha_column_names: List of generated alpha column names
            
        Raises:
            ValueError: If no alpha data is available at the latest timestamp
        """
        logger.info(f"Using alphas at last loaded bar {self.get_last_processed_ts()}")

        latest_alphas_df = self.get_latest_features()
        alphas_df, alpha_cols = self.forecasts.compute_model_alphas_for_server(models_df=latest_alphas_df, horizons=horizons)
        if len(alphas_df) == 0:
            raise log_and_raise(f"No alphas at last_loaded bar {self.get_last_processed_ts()}")

        new_cols = alpha_cols
        alphas_df, alpha_cols = calculate_horizon_alphas(
            config=self.config, df=alphas_df, horizons=horizons,
            centering_symbols=self.centering_universe
        )
        new_cols += alpha_cols

        new_cols = unique_list(new_cols)
        return alphas_df, new_cols

    def update(self, st_only: bool = False) -> bool:
        """Perform a complete update cycle with new market data.
        
        Main entry point for updating the server with new data. Orchestrates:
        1. Loading and processing new bar data
        2. Calculating features across horizons
        3. Computing model signals
        4. Updating last processed timestamps
        
        Memory usage is logged between major steps for monitoring.
        
        Args:
            st_only: If True, only process short-term horizons
            
        Returns:
            True if update completed successfully, False if no new data
        """
        logger.info(f"Updating server, short term only: {st_only}...")
        if not self.update_bars():
            return False
        log_mem_usage()
        self.calculate_features(st_only=st_only)
        log_mem_usage()
        self.calculate_models(st_only=st_only)
        log_mem_usage()

        for hc in self.horizon_calcs.values():
            hc.update_last_processed_ts()

        return True
