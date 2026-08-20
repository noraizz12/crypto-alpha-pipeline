"""Real-time alpha signal generation server for live trading.

This module implements the AlphaServer class which runs continuously to:

1. **Market Data Ingestion**: Loads real-time bars, features, and models
2. **Alpha Calculation**: Computes multi-horizon alpha signals
3. **Portfolio Optimization**: Generates optimal position targets
4. **Target Publishing**: Writes target files for trading execution

The server runs on a configurable schedule, optimizing portfolios at major
intervals and updating short-term alphas between optimizations. It handles:
- Dynamic universe management based on liquidity and listing status
- Position tracking and aging for risk management
- News event integration for alpha generation
- Slack notifications for monitoring

Typical usage:
    server = AlphaServer(config_file='config.json')
    server.loop() # Runs continuously
"""

import asyncio
import gc
import logging
import os
import time
from collections import defaultdict
from datetime import date, datetime as dt
from datetime import timedelta as td, timezone
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from slack_sdk.webhook.async_client import AsyncWebhookClient

from lib.calcs import Calcs
from lib.calcs.calc_util import make_market_weight_col
from lib.pnl_new.binance_pnl import BinancePnl
from lib.data.data_news import load_news
from lib.data.dataloader import DataLoader
from lib.data.loaders import load_mktcap_and_groupings, load_positions
from lib.data.data_files import get_latest_target_files
from lib.external.binance_utils import get_positions
from lib.alpha.forecasts import Forecasts
from lib.portfolio_optimization import PortfolioOptimizer
from lib.universe import filter_universe
from lib.util.aws import write_df_to_s3_parquet
from lib.util.config import extract_horizon_models, extract_model_alpha_list, extract_models, extract_reopt_times, get_config, extract_all_tree_features_horizon, extract_compute_features
from lib.util.dataframes import concat, get_min_max_ts, make_symbol, make_symbol_venue, merge_on_index, remove_1_min_cols, set_index, safe_del, convert_category_cols_to_bool, \
    clean_poop
from lib.util.directory import DirectoryManager, dir_manager
from lib.util.logging_util import KeyLogger
from lib.util.slack import send_slack_async
from lib.util.files import safe_mkdir
from lib.util.time_util import compute_lookback_days, dt_to_str, str_to_dt, end_of_day, next_closest_time, today_date, date_to_str
from lib.util.util import LOCAL, fmoney, get_config_universe, unique_list, log_and_raise, SYMBOL_VENUE
from .server_calcs import ServerCalcs

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


class AlphaServer:
    """Real-time alpha signal generation and portfolio optimization server.
    
    Continuously generates trading signals based on market data, features,
    and predictive models. Optimizes portfolio allocations considering risk,
    transaction costs, and market constraints. Publishes target positions
    for downstream trading systems.
    
    Key responsibilities:
    - Load and process real-time market data (bars, features, models)
    - Calculate multi-horizon alpha signals (15min to monthly)
    - Run portfolio optimization with risk constraints
    - Track position age for risk management
    - Generate and publish target position files
    - Send monitoring alerts via Slack
    
    The server operates on two timescales:
    1. Major reoptimizations (e.g., every 2 hours) - Full portfolio optimization
    2. Minor updates (e.g., every minute) - Short-term alpha updates only
    
    Attributes:
        config: Trading system configuration dictionary
        universe: List of tradeable symbol_venue pairs
        model_horizons: Time horizons for alpha models (minutes)
        feature_horizons: Time horizons for feature calculation
        short_term_horizons: Horizons for high-frequency updates
        trade_bot: Portfolio optimizer instance
        forecasts: Alpha forecasting models
        server_calcs: Server-specific calculations
        alphas_df: Historical alpha signals DataFrame
        slack: Slack client for notifications
    """
    def __init__(
            self,
            eod_today: Optional[dt] = None,
            config_file: Optional[str] = None,
            slack_client: Optional[AsyncWebhookClient] = None,
            debug: bool = False,
            trade_on_start: bool = False,
            trailing_days_to_load: Optional[int] = None,
            server_dir_manager: DirectoryManager = dir_manager,
            latest_alpha_dir: Optional[str] = None,
            raw_target_dir: Optional[str] = None,
            target_dir: Optional[str] = None,
            no_positions: bool = False,
            new_bars: bool = True,
            scale: Optional[float] = None
    ):
        """Initialize the AlphaServer with configuration and runtime parameters.
        
        Sets up the trading universe, loads configuration, initializes data loaders,
        and prepares optimization components. Establishes time horizons for different
        alpha models and feature calculations.
        
        Args:
            eod_today: End-of-day timestamp for server operation. If None, uses
                current UTC time.
            config_file: Path to configuration JSON file. If None, uses default.
            slack_client: Async Slack webhook client for notifications.
            debug: Enable debug mode (disables some operations, enables logging).
                Defaults to BARS_TYPE_NEW.
            trade_on_start: Whether to generate trades immediately on startup.
            trailing_days_to_load: Override for feature lookback periods.
            server_dir_manager: Directory manager for file paths.
            latest_alpha_dir: Directory for latest alpha signal files.
            raw_target_dir: Directory for raw target position files.
            target_dir: Directory for processed target files.
            no_positions: If True, ignore existing positions on startup.
            new_bars: Whether to use new bar format.
            scale: Optional scale factor (0.0-1.0) to apply to target positions.

        Raises:
            AssertionError: If OPT_HORIZON not in model horizons.
            Exception: If no positions file found when required.
        """
        self.debug = debug
        self.no_positions = no_positions
        self.trade_on_start = trade_on_start
        self.config = get_config(config_file)[1]
        self.new_bars = new_bars
        self.scale = scale

        self.eod_today = eod_today
        if self.eod_today is None:
            self.reopt_times = extract_reopt_times(self.config)
            self.eod_today = end_of_day(dt.now(timezone.utc))
        else:
            self.reopt_times = extract_reopt_times(self.config, self.eod_today)

        self.today_date: date = (self.eod_today - td(days=1)).date()
        self.yesterday_date: date = self.today_date - td(days=1)
        self.last_optimize_ts = None

        self.dir_manager = server_dir_manager
        self.data_loader = DataLoader(self.config, self.dir_manager)
        self.pos_dir = self.dir_manager.POSITION_DIR
        self.latest_alpha_dir = latest_alpha_dir
        self.raw_target_dir = self.dir_manager.RAW_TARGET_DIR if raw_target_dir is None else raw_target_dir
        self.target_dir = self.dir_manager.TARGET_DIR if target_dir is None else target_dir

        horizon_models = extract_horizon_models(config=self.config, exclude_zero_weight=True)
        self.model_horizons = sorted(unique_list([hm[0] for hm in horizon_models]))

        self.opt_horizon = self.config['OPT_HORIZON']
        assert self.opt_horizon in self.model_horizons

        self.feature_horizons = unique_list(self.model_horizons + [self.opt_horizon])
        self.short_term_horizons = [hh for hh in self.config['SHORT_TERM_MODEL_HORIZONS'] if hh in self.model_horizons]

        trailing_lookback_periods = self.config['DEFAULT_FEATURE_LOOKBACK_PERIODS'] if trailing_days_to_load is None else trailing_days_to_load
        self.trailing_lookback_periods = {}
        for horizon in self.feature_horizons:
            if horizon == 43200:
                self.trailing_lookback_periods[horizon] = 3
            elif horizon == 10080:
                self.trailing_lookback_periods[horizon] = 12
            else:
                self.trailing_lookback_periods[horizon] = trailing_lookback_periods

        self.reoptimize_interval_mins = self.config['REOPTIMIZE_INTERVAL_MINS']
        self.volume_bucket_mins = self.config['VOLUME_BUCKET_MINS']
        self.max_portfolio_size = self.config['MAX_PORTFOLIO_NOTIONAL']
        self.wave_interval_seconds = self.config['WAVE_INTERVAL_MINS'] * 60
        self.similarity_threshold = self.config['NEWS_SIMILARITY_THRESHOLD']
        self.dynamic_universe = self.config['DYNAMIC_UNIVERSE']

        uni_df = self.data_loader.load_universe_df(
            universe_date=self.yesterday_date,
            filter_dead=True
        )
        if uni_df is None:
            raise log_and_raise(f"Could not load previous days universe file at {self.yesterday_date}")
        uni_df = uni_df[['fittable', 'tradeable', 'expandable', 'priceable', 'advp', 'marketcap', 'symbol_venue']]

        #used to also have fittable here...
        self.uni_df = uni_df[uni_df['tradeable']]

        self.tradeable_universe = list(self.uni_df[self.uni_df['tradeable']]['symbol_venue'].unique())
        logger.info(f"Traderable universe: {sorted(self.tradeable_universe)}")

        if not self.dynamic_universe:
            uni_cnt = len(self.uni_df)
            uni = get_config_universe(self.config, symbol_type=SYMBOL_VENUE)
            self.uni_df = self.uni_df[self.uni_df['symbol_venue'].isin(uni)]
            logger.info(f"No dynamic universe so count from {uni_cnt} -> {len(self.uni_df)} ")

        #clean all this up!
        pos_df = load_positions(mode='latest', pos_dir=self.pos_dir)
        # pos_df will be None in integration tests so set to empty
        pos_symbols = set(pos_df[pos_df['qty'] != 0].index.get_level_values('symbol_venue').unique()) if pos_df is not None else set()
        logger.info(f"Position symbols: {sorted(pos_symbols)}")

        good_symbols = set(self.uni_df['symbol_venue'].unique())
        crap_universe = pos_symbols - good_symbols
        logger.info(f"Position symbols not in good universe: {crap_universe}")
        self.universe = list(good_symbols | pos_symbols)
        logger.info(f"Final universe: {self.universe}")

        self.position_age_df = None
        self.slack = slack_client
        self._message(f"Starting Server {self.eod_today}, Debug: {self.debug} Trade On Start: {self.trade_on_start}, trading {len(self.universe)} universe coins...")

        assert self.volume_bucket_mins < 1440
        self.feature_flds_to_load = self.make_feature_load_map()

        self.trade_bot = PortfolioOptimizer(self.config, horizons=self.model_horizons, scale=self.scale)

        fit_as_of = self.eod_today - td(days=1) if self.debug else None
        logger.info('Set forecasts for server')
        self.forecasts = Forecasts(
            config=self.config,
            debug=self.debug,
            prod=True,
            is_server_forecast=True,
            forecast_dir_manager=self.dir_manager,
            fit_as_of=fit_as_of,
        )
        self.calcs = Calcs(config=self.config, prod=True)

        self.server_calcs = None
        self.setup()

        self.alphas_df: pd.DataFrame = pd.DataFrame()

        # only need it when initialized
        self.model_alpha_cols = extract_model_alpha_list(self.config)


    def make_feature_load_map(self) -> Dict[int, List[str]]:
        fmap = defaultdict(list)
        for horizon in self.model_horizons:
            features_list = set(extract_compute_features(self.config, horizon=horizon, prod=False))
            f1, f2 = extract_all_tree_features_horizon(config=self.config, horizon=horizon)
            horizon_features = set(f1 + f2)
            features_to_load = list(horizon_features & features_list)
            extra_features = ['beta_1440', 'beta_t_1440', 'mdvp']
            if horizon in (43200, 10080, 4320):
                extra_features.append(f"rsi_{horizon}")
            if self.config.get('VOL_BOUND_COEFF', 0) > 0:
                extra_features.append('logret_1440_trstd_cz')
            fmap[horizon] = features_to_load + extra_features
        return fmap

    def _message(self, msg: str, error: bool = False, msg_key: str = "server message") -> None:
        """Send a message to logs and optionally to Slack.
        
        Args:
            msg: Message text to send.
            error: Whether this is an error message.
            msg_key: Key for structured logging.
        """
        if error:
            logger.error("[SLACK] " + msg, key=msg_key)
        else:
            logger.info("[SLACK] " + msg)

        if not self.debug and self.slack is not None:
            if error:
                msg = "[ERROR] " + msg
            if self.debug:
                msg = "[DEBUG] " + msg

            msg = '```' + msg + '```'
            asyncio.run(send_slack_async(self.slack, msg))

    def _message_df(self, df: pd.DataFrame, msg: str = "") -> None:
        """Send a DataFrame as a formatted message to Slack.
        
        Splits large DataFrames into multiple messages if needed.
        
        Args:
            df: DataFrame to send.
            msg: Optional message prefix.
        """
        row_cnt = len(df)
        if row_cnt > 100:
            self._message(msg + "\n" + df.iloc[:100].to_markdown(index=False))
            self._message(msg + "\n" + df.iloc[100:].to_markdown(index=False))
        else:
            self._message(msg + "\n" + df.to_markdown(index=False))

    def update_universe(self, bars_df: pd.DataFrame):
        """Update the trading universe based on current market conditions.
        
        Filters the universe to include only symbols that meet liquidity
        requirements and are not being delisted. Always includes symbols
        with open positions.
        
        Args:
            bars_df: DataFrame with recent bar data for universe filtering.
        
        Raises:
            Exception: If no positions file found in production mode.
        """
        positions_df = load_positions(mode='latest', pos_dir=self.pos_dir)
        # if not self.debug and positions_df is None and self.pos_dir == POSITION_DIR:
        #     raise log_and_raise("No initial positions file found!")

        old_uni = set(self.universe.copy())
        self.universe = filter_universe(init_universe=self.universe, bars_df=bars_df, positions_df=positions_df, date=self.yesterday_date)
        new_uni = set(self.universe)
        removed = old_uni - new_uni
        added = new_uni - old_uni
        logger.info(f"Updating universe removed: {removed} added: {added}")

    def get_position_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position age information to the DataFrame.
        
        Calculates how long each position has been held, which is used
        for risk management (older positions may have different risk
        characteristics).
        
        Args:
            df: DataFrame to add position age to.
            
        Returns:
            DataFrame with 'position_age' column added (in days).
        """
        if 'position_age' in df:
            return df

        # Position age is calculated once in setup() and cached in self.position_age_df
        if self.position_age_df is not None:
            df = df.join(self.position_age_df, on='symbol_venue', how='left')
            df['position_age'] = df['position_age'].fillna(0)
        else:
            df['position_age'] = 0

        df['position_age'] = df['position_age'].astype('Int32')
        return df


    def setup(self) -> None:
        """Initialize server with historical data and calculations.
        
        Loads historical bars, features, and models for all configured
        horizons. Sets up ServerCalcs instances for each horizon with
        the appropriate lookback periods. This method must be called
        before the server can generate alphas.
        
        The setup process:
        1. Loads 1-minute bars for the maximum lookback period
        2. Updates the trading universe based on recent data
        3. For each horizon, loads bars, features, news, and models
        4. Initializes ServerCalcs for incremental calculations
        
        Raises:
            Exception: If required data files cannot be loaded.
        """
        # Calculate position age once at startup
        if self.pos_dir == self.dir_manager.POSITION_DIR:
            logger.info("Calculating position age from BinancePnl...")
            try:
                # Convert today_date to datetime for BinancePnl compatibility
                start_dt = dt.combine(self.today_date, dt.min.time()).replace(tzinfo=timezone.utc)
                pnl_calculator = BinancePnl(
                    config=self.config,
                    start=start_dt,
                    end=dt.now(timezone.utc),
                    pnl_dir_manager=self.dir_manager
                )
                # BinancePnl calculates position_age in merge_data() automatically
                security_df = pnl_calculator.aggregate_by_security_date()
                if security_df is None or security_df.empty:
                    logger.warning("No PnL data available for position age calculation")
                    self.position_age_df = None
                else:
                    max_ts = security_df.index.get_level_values('ts').max()
                    self.position_age_df = security_df.loc[
                        security_df.index.get_level_values('ts') == max_ts, ['position_age']
                    ].reset_index(level='ts', drop=True)
                    logger.info(f"Position age calculated for {len(self.position_age_df)} symbols")
            except Exception as e:
                logger.warning(f"Failed to calculate position age: {e}")
                self.position_age_df = None
        else:
            logger.info("Skipping position age calculation (non-prod position dir)")
            self.position_age_df = None

        max_horizon = max(self.feature_horizons)
        lookback_days = compute_lookback_days(horizon=max_horizon, lags=self.trailing_lookback_periods[max_horizon])
        one_min_start_date = self.today_date - td(days=lookback_days)
        bars_df = self.data_loader.load_bars(
            horizon=1,
            start_date=one_min_start_date,
            end_date=self.yesterday_date,
            symbol_venues=self.universe,
        )
        bars_df = remove_1_min_cols(bars_df)
        logger.info(f"Loaded yesterday bars on {sorted(bars_df.index.get_level_values('symbol_venue').unique())}")

        #remove because we are going to take from the universe file in the following lines
        for col in ['fittable', 'advp', 'priceable']:
            safe_del(bars_df, col)

        bars_df = pd.merge(
            bars_df.reset_index(), self.uni_df,
            left_on='symbol_venue', right_on='symbol_venue', how='inner'
        )
        bars_df = set_index(bars_df, fast=True)
        self.update_universe(bars_df)
        bars_df = bars_df[bars_df.index.get_level_values('symbol_venue').isin(self.universe)]
        bars_df['index_mid_price_diff'] = bars_df['index_price'] - bars_df['close_mid']
        bars_df = bars_df.drop(columns=['index_price_vwap', 'index_mid_price_diff_mean', 'index_mid_price_diff_std', 'last_funding_rate_mean', 'last_funding_rate_std'], errors='ignore')
        bars_df = self.data_loader.add_delisting_dates(bars_df, start_date=self.yesterday_date, end_date=self.yesterday_date)

        mkt_df = load_mktcap_and_groupings(universe_date=self.today_date, mktcap_dir=self.dir_manager.MARKETCAP_DIR, categories_only=True, config=self.config)
        bars_df = set_index(clean_poop(pd.merge(bars_df.reset_index(), mkt_df, left_on='symbol_venue', right_on='symbol_venue', how='left', suffixes=('_poop', ''))))
        for col in bars_df.columns:
            if col.startswith('category_'):
                bars_df[col] = bars_df[col].fillna(0.0)
        logger.info("Done loading categories")

        bars_df = make_market_weight_col(bars_df)

        #note that we center on the tradeable universe b/c that's the same universe in sim that we generate models from
        #XXX but the server tradeable also has dust that we have positions in, should remove these from the centering with an adv check
        self.server_calcs = ServerCalcs(
            config=self.config,
            debug=self.debug,
            eod_today=self.eod_today,
            server_calcs_dir_manager=self.dir_manager,
            universe=self.universe,
            feature_horizons=self.feature_horizons,
            short_term_horizons=self.short_term_horizons,
            model_horizons=self.model_horizons,
            feature_flds_to_load=self.feature_flds_to_load,
            new_bars=self.new_bars,
            centering_universe=self.tradeable_universe
        )
        self.server_calcs.make_server_horizon_calc(horizon=1, df=bars_df)

        for horizon in self.feature_horizons:
            lookback_days = max(compute_lookback_days(self.trailing_lookback_periods[horizon] * horizon), 1)
            start_date = self.today_date - td(days=lookback_days)
            logger.info(f"Setup for horizon {horizon} loading data {lookback_days} days prior to today {self.today_date}")

            bars_horizon_df = self.data_loader.load_bars(
                horizon=horizon,
                start_date=start_date - td(days=1),
                end_date=self.yesterday_date,
                symbol_venues=self.universe,
            )
            if bars_horizon_df is None:
                raise log_and_raise(f"No bars loaded for {horizon}")

            bars_horizon_df = pd.merge(bars_horizon_df.reset_index(), mkt_df, left_on='symbol_venue', right_on='symbol_venue', how='left')
            for col in bars_horizon_df.columns:
                if col.startswith('category_'):
                    bars_horizon_df[col] = bars_horizon_df[col].fillna(0.0)
            bars_horizon_df = set_index(bars_horizon_df)

            cols = self.feature_flds_to_load.get(horizon, [])
            if len(cols) > 0:
                features_df = self.data_loader.load_features(
                    start_date=self.yesterday_date,
                    end_date=self.yesterday_date,
                    horizons=unique_list([horizon, 1440]),
                    cols=cols,
                    symbol_venues=self.universe,
                )
                if features_df is None:
                    raise log_and_raise(f"Could not load feature files for {horizon}, exiting...")
                features_df = merge_on_index(bars_horizon_df, features_df) if features_df is not None else bars_df
            else:
                logger.info(f"No features to load for {horizon}")
                features_df = bars_horizon_df

            news_df = load_news(similarity_threshold=self.similarity_threshold, start_date=start_date, end_date=today_date(), news_dir=self.dir_manager.NEWS_DIR)
            if news_df is None:
                logger.error(f"fail to load news data between {start_date} and {today_date()}", key="fail to load news data for server")
                features_df['news_title'] = None
            else:
                features_df = merge_on_index(features_df, news_df[['news_title']])

            if horizon in self.model_horizons:
                models_to_run = extract_models(self.config, horizons=[horizon])
                models_df = self.data_loader.load_models(
                    start_date=start_date,
                    end_date=self.yesterday_date,
                    horizon=horizon,
                    models=models_to_run,
                    fast=True,
                )
                if models_df is not None:
                    features_df = merge_on_index(features_df, models_df)
                else:
                    logger.warning(f"Missing models at {horizon=}")

            features_df = self.get_position_age(features_df)
            min_ts, max_ts = get_min_max_ts(features_df)
            logger.info(f"Loaded trailing data at {horizon} from {min_ts} to {max_ts}")

            self.server_calcs.make_server_horizon_calc(horizon, features_df)


    def set_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load current positions and add to DataFrame.
        
        Merges position quantities from the latest position file and
        calculates position values in USD. Validates that positions
        are within portfolio size limits.
        
        Args:
            df: DataFrame to add position information to.
            
        Returns:
            DataFrame with 'qty' and 'position' columns added.
            
        Raises:
            Exception: If no positions file found or total is zero.
        """

        if self.no_positions:
            logger.warning(f"Dangerous: running server starting with zero positions!")
            df['qty'] = np.float32(0.0)
            df['position'] = np.float32(0.0)
            return df

        latest_ts = df.index.get_level_values('ts').max()

        # should maybe use get_positions() to get directly from binance?
        positions_df = load_positions(start_date=self.today_date - td(days=1), end_date=max(self.today_date, latest_ts.date()), mode='latest', ts_override=latest_ts, pos_dir=self.pos_dir)

        if positions_df is None and not self.debug:
            raise log_and_raise("No positions file found! setting to zero")

        if positions_df is None:
            for fld in ['qty', 'cost_basis', 'value']:
                df[fld] =  np.float32(0.0)
        else:
            df = merge_on_index(df, positions_df[['qty']].dropna())

        df['qty'] = df['qty'].fillna(0)
        df['position'] = (df['qty'] * df['close_mid']).astype(np.float32)
        total_notional = df['position'].abs().sum()

        if total_notional == 0 and not self.debug:
            print(df[['position', 'qty']])
            print(positions_df)
            raise log_and_raise("No current positions loaded!")

        if total_notional > self.max_portfolio_size:
            logger.warning(f"Positions of {fmoney(total_notional)} greater than max portfolio size {fmoney(self.max_portfolio_size)}")

        logger.info(f"Loaded positions with total notional {total_notional}")
        return df.copy()

    def set_max_positions_from_binance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add maximum position limits from Binance API.
        
        Queries Binance for position limits per symbol to ensure we don't
        exceed exchange-imposed constraints.
        
        Args:
            df: DataFrame to add position limits to.
            
        Returns:
            DataFrame with 'max_position_value' column added.
        """
        if LOCAL:
            logger.info("Not calling binance when not in prod...")
            df['max_position_value'] = np.nan
            return df

        binance_pos_df = get_positions()
        if binance_pos_df is None:
            logger.info("Failed to load positions info from binance")
            df['max_position_value'] = np.nan
            return df

        binance_pos_df = make_symbol_venue(binance_pos_df)
        binance_pos_df = binance_pos_df.rename(columns={'maxNotionalValue': 'max_position_value'})
        df = pd.merge(df.reset_index(), binance_pos_df[['symbol_venue', 'max_position_value']], on='symbol_venue', how='left')
        df = set_index(df)
        logger.info("Loaded max position value from binance")
        logger.info(df[['symbol_venue', 'max_position_value']].to_markdown())
        return df

    def get_latest_alphas(self) -> pd.DataFrame:
        """Get the most recent alpha signals.
        
        Returns:
            DataFrame with latest alpha signals for all symbols.
        """
        ts_idx = self.alphas_df.index.get_level_values('ts')
        df = self.alphas_df[ts_idx == ts_idx.max()].copy()
        return df

    def dump_targets(self, targets_df: pd.DataFrame, optimize: bool):
        """Save target positions to files and send notifications.
        
        Writes target files for the trader to consume. For optimized
        targets, also sends a Slack message with key trades. Uploads
        to S3 in production mode.
        
        Args:
            targets_df: DataFrame with target positions and metadata.
            optimize: Whether these are optimized targets (vs ST-only).
        """
        targets_df = make_symbol(targets_df)
        st_alpha_cols = [f"alpha_{hh}" for hh in self.short_term_horizons]
        lt_alpha_cols = [f"alpha_{hh}" for hh in self.model_horizons if hh not in self.short_term_horizons]

        model_alpha_cols = [col for col in self.model_alpha_cols if col in targets_df.columns]
        targets_df['model_alpha_col'] = targets_df.apply(lambda row: ','.join([f"{col}:{row[col]}" for col in model_alpha_cols]), axis=1)

        bound_cols = ['lbound', 'ubound']
        target_cols = ['position', 'target_position', 'close_mid', 'risk_1440', 'risk_15', 'logret_1440_lz', 'model_alpha_col'] + st_alpha_cols + bound_cols
        if optimize:
            target_cols += ['util', 'alpha_opt'] + lt_alpha_cols

        for col in target_cols:
            if col not in targets_df.columns:
                logger.warning(f"Columns {col} not found in targets df!")
                targets_df[col] = np.nan

        if self.debug:
            pd.options.display.max_rows = 200
            print(targets_df[target_cols])
            return

        _, max_ts = get_min_max_ts(targets_df)
        max_ts = max_ts + td(minutes=1)
        latest_ts = dt_to_str(max_ts)
        alpha_dir = f"{self.raw_target_dir}/{date_to_str(max_ts.date())}"
        safe_mkdir(alpha_dir)
        alpha_filename = f"{alpha_dir}/alpha.{latest_ts}.parquet"
        logger.info(f"Dumping {alpha_filename}...")
        targets_df.to_parquet(alpha_filename)

        target_long_not = targets_df[targets_df['target_position'] > 0]['target_position'].sum()
        target_short_not = targets_df[targets_df['target_position'] < 0]['target_position'].sum()
        pos_long_not = targets_df[targets_df['position'] > 0]['position'].sum()
        pos_short_not = targets_df[targets_df['position'] < 0]['position'].sum()
        tot_desired_dollars = targets_df['desired_trade_dollars_abs'].sum()

        target_dir = f"{self.target_dir}/{date_to_str(max_ts.date())}"
        safe_mkdir(target_dir)
        target_filename = f"{target_dir}/targets"

        if optimize:
            self._message(f"Target: ${target_long_not:.0f} / ${target_short_not:.0f} trade dollars: ${tot_desired_dollars:.0f} Positions: ${pos_long_not:.0f} / ${pos_short_not:.0f}")
            if tot_desired_dollars > 0:
                MSG_COLS = ['symbol', 'target_position', 'position', 'desired_trade_dollars', 'alpha_opt', 'expanding']
                message_df = targets_df.copy()
                message_df = message_df[message_df['desired_trade_dollars_abs'] > 100]
                message_df = message_df.sort_values(by='desired_trade_dollars_abs', ascending=False)[MSG_COLS]
                self._message_df(message_df, msg="TOP TRADES")
            target_filename += ".opt"
        else:
            target_cols = ['position', 'target_position', 'close_mid', 'risk_15', 'logret_1440_lz', 'logret_60_trstd'] + st_alpha_cols + bound_cols

        target_filename += f".{latest_ts}.csv"
        logger.info(f"Writing {target_filename}")
        targets_df[target_cols].to_csv(target_filename)
        if not LOCAL:
            try:
                target_filename = target_filename.split('/')[-1]
                write_df_to_s3_parquet(targets_df[target_cols], key=f"targets/{target_filename}")
            except Exception as e:
                logger.error(f"Could not write targets to S3! {e}", key="fail to write targets to s3")

    def generate_targets(self, optimize: bool):
        """Generate target positions based on current alphas.
        
        Calculates alphas, loads positions, and either runs full
        portfolio optimization or carries forward previous targets
        with updated short-term alphas.
        
        Args:
            optimize: If True, run full portfolio optimization.
                If False, only update short-term alphas.
        """
        logger.info(f"Generating Targets {optimize=}...")

        horizons = self.short_term_horizons if not optimize else self.model_horizons
        logger.info(f"Calculating alphas for horizons {horizons}...")
        latest_alpha_df, new_cols = self.server_calcs.calculate_alphas(horizons=horizons)
        logger.info(f"Adding new columns: {new_cols}")

        latest_alpha_df = self.set_positions(latest_alpha_df)
        if not self.debug:
            latest_alpha_df = self.set_max_positions_from_binance(latest_alpha_df)

        # Recalculate tradeable/expandable filters based on current intraday liquidity
        # This allows symbols that gain liquidity intraday to become expandable
        # latest_alpha_df = self.calcs.calculate_tradeable_filter(latest_alpha_df)
        latest_alpha_df = self.calcs.calculate_expandable_filter(latest_alpha_df)

        latest_alpha_df = self.calcs.calculate_exclusions_and_bounds(latest_alpha_df)
        latest_alpha_df = latest_alpha_df.copy()

        if optimize:
            latest_alpha_df, _ = self.trade_bot.make_alpha_opt(df=latest_alpha_df)
            full_universe_cnt = len(latest_alpha_df)
            expandable_cnt = len(latest_alpha_df[latest_alpha_df['expandable']])
            self._message(f"Optimizing with expandable universe {expandable_cnt} / {full_universe_cnt}")
            uni = sorted(list(make_symbol(latest_alpha_df)['symbol'].unique()))
            logger.info(f"Optimizing with universe: {uni}")
            latest_alpha_df = self.trade_bot.optimize(alpha_df=latest_alpha_df)
            self.last_optimize_ts = dt.now(timezone.utc)
            util_dir = f"{self.dir_manager.UTIL_DIR}/{dt_to_str()}"
            safe_mkdir(util_dir)
            self.trade_bot.dump_util_metrics(f"{util_dir}/util.{date_to_str()}.csv", self.last_optimize_ts)
        else:
            if len(self.alphas_df) > 0:
                logger.info("Not Optimizing, carrying previous opt forward")
                previous_alpha_df = self.alphas_df[self.alphas_df.index.get_level_values('ts') == self.alphas_df.index.get_level_values('ts').max()][['target_opt']].copy().reset_index()
                previous_alpha_df['ts'] = latest_alpha_df.index.get_level_values('ts').max()
                previous_alpha_df = set_index(previous_alpha_df)
                latest_alpha_df = merge_on_index(latest_alpha_df, previous_alpha_df[['target_opt']])
            else:
                latest_alpha_df['target_opt'] = np.nan

        if self.latest_alpha_dir is not None:
            latest_alpha_df.to_parquet(f'{self.latest_alpha_dir}/latest_alpha.parquet')

        ## TODO : fix util so it can get dumped in non-opt files
        latest_alpha_df = self.trade_bot.calculate_targets(timeslice_df=latest_alpha_df)

        self.dump_targets(latest_alpha_df, optimize=optimize)

        first_alphas = self.alphas_df is None or len(self.alphas_df) == 0
        self.alphas_df = latest_alpha_df if first_alphas else concat([self.alphas_df, latest_alpha_df])

    def update(self, st_only: bool) -> bool:
        """Update server calculations with latest market data.
        
        Args:
            st_only: If True, only update short-term horizons.
                If False, update all horizons.
                
        Returns:
            True if update successful, False otherwise.
        """
        return self.server_calcs.update(st_only)

    def get_latest_target_age(self) -> Optional[float]:
        """Get the age of the most recent target file in seconds.
        
        Used to determine if we should skip initial optimization
        when trade_on_start is False.
        
        Returns:
            Age of latest target file in seconds, or None if no files.
        """
        logger.info("Getting latest target file age")
        target_files = get_latest_target_files(curr_date=today_date(), opt_only=True)
        if len(target_files) == 0:
            logging.warning(f"No target files found for {today_date()}")
            return None

        latest_target_file = sorted(target_files, reverse=True, key=os.path.getmtime)[0]
        # use the target time rather than modified time (like 12:02 target might be actually dumped at 12:22)
        target_file_age_seconds = (dt.now(timezone.utc) - str_to_dt(latest_target_file.split('.')[-2])).total_seconds()
        return target_file_age_seconds

    def loop(self) -> None:
        """Main server loop that continuously generates trading signals.
        
        Runs indefinitely, alternating between:
        1. Major reoptimizations at scheduled times (full portfolio optimization)
        2. Minor updates between reoptimizations (short-term alpha updates only)
        
        The loop handles:
        - Checking if recent targets exist (skip if trade_on_start=False)
        - Waiting for scheduled reoptimization times
        - Updating market data and generating targets
        - Sleeping between updates to control CPU usage
        - Garbage collection to manage memory
        
        This method does not return and should be run in a dedicated process.
        """
        next_reoptimize: Optional[dt] = None
        optimized_yet = False
        target_file_age_seconds = self.get_latest_target_age()
        # when target_file_age_seconds is None or target_file_age_seconds > reoptimize_interval_mins * 60.0
        # it means we don't have targets generated for recent period
        # even though we have trade_on_start as False, we should skip the logic below and directly go to generate targets
        if not self.trade_on_start and target_file_age_seconds is not None and target_file_age_seconds < self.reoptimize_interval_mins * 60.0:
            now = dt.now(timezone.utc)
            next_reoptimize = next_closest_time(now, self.reopt_times)
            logger.info(f"Next optimization at {next_reoptimize}, current: {now} at initialization")
            optimized_yet = True
            self.update(st_only=False)

        while True:
            now = dt.now(timezone.utc)
            if next_reoptimize is None or (now > next_reoptimize):
                if self.update(st_only=False):
                    self.generate_targets(optimize=True)
                    next_reoptimize = next_closest_time(now, self.reopt_times)
                    logger.info(f"Next optimization at {next_reoptimize}, current: {now} inside loop")
                    optimized_yet = True
                else:
                    logger.warning("Could not update, skipping this optimization...")
                continue

            next_reopt_secs = (next_reoptimize - dt.now(timezone.utc)).seconds
            logger.info(f"Minutes until next re-optimization {next_reopt_secs / 60.0}")
            if optimized_yet and next_reopt_secs > 20 * 60:
                start_ts = time.time()

                if self.update(st_only=True):
                    self.generate_targets(optimize=False)
                else:
                    logger.warning("Could not update, skipping this target file...")

                end_ts = time.time()
                elapsed_secs = end_ts - start_ts

                wait_secs = int(min(next_reopt_secs, self.wave_interval_seconds) - elapsed_secs)
                if wait_secs > 0:
                    next_reopt_mins = (next_reopt_secs - elapsed_secs) / 60.0
                    logger.info(f"Sleeping {wait_secs / 60} min., next re-opt: {next_reopt_mins} min.")

                    total_sleep_secs = 0
                    while total_sleep_secs < wait_secs:
                        remaining = wait_secs - total_sleep_secs
                        sleep_interval = min(60, remaining)
                        time.sleep(sleep_interval)
                        total_sleep_secs += sleep_interval

            gc.collect()
            time.sleep(30)
