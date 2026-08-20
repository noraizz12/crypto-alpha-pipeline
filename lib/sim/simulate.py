"""Backtesting simulation engine for statistical arbitrage trading strategies.

This module provides comprehensive backtesting capabilities for the trading system,
simulating real-world trading conditions including:

1. **Portfolio Optimization**: Integrates with portfolio optimizer to generate trades
2. **Execution Simulation**: Models market impact, slippage, and trading constraints
3. **PnL Calculation**: Tracks positions, cash, fees, and funding payments
4. **Risk Management**: Enforces position limits and handles liquidations
5. **Multi-Horizon Support**: Tests strategies across different time horizons

The simulation engine accurately models:
- Order execution with realistic slippage and market impact
- Exchange fees and funding rate payments
- Volume constraints and liquidity limitations
- Position aging and risk adjustments
- Delisting and dead asset handling

Key features:
- Supports multiple alpha models and forecast horizons
- Handles both VWAP and open price execution
- Tracks detailed execution statistics
- Generates comprehensive PnL breakdowns
- Supports grid search for parameter optimization

Example:
    >>> config = load_config()
    >>> sim = Simulate(
    ...     config=config,
    ...     sim_dir='/simulations/test_strategy',
    ...     initialize_positions=False,
    ... )
    >>> results = sim.simulate(
    ...     start_date=date(2023, 1, 1),
    ...     end_date=date(2023, 12, 31),
    ...     horizons=[1440],
    ...     models=['momentum']
    ... )
"""

import glob
import logging
import os
from datetime import date, datetime as dt
from datetime import timedelta as td
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from lib.data.loaders import load_mktcap_and_groupings
from lib.pnl.fill_pnl_breakdown import PNL_BREAKDOWN_FEATURES_BARS, PNL_BREAKDOWN_FEATURES_FEATURES
from lib.pnl import FillBreakdown, FillPnl
from lib.calcs import Calcs, make_market_weight_col
from lib.data import load_positions
from lib.data.dataloader import DataLoader
from lib.alpha.forecasts import calculate_horizon_alphas
from lib.portfolio_optimization import OptimizationError, PortfolioOptimizer
from lib.sim.sim_util import get_return_metrics_str
from lib.universe import Universe
from lib.util.config import extract_horizon_models
from lib.util.dataframes import concat, get_min_max_ts, make_symbol, merge_on_index, set_index, make_date, convert_category_cols_to_bool
from lib.util.directory import dir_manager, DirectoryManager
from lib.util.logging_util import KeyLogger
from lib.util.time_util import date_range, date_to_end_dt, date_to_start_dt, date_to_str, to_datetime, today_date, datetime_series_to_int64, datetime_series_to_epoch_seconds
from lib.util.util import get_config_universe, unique_list, log_and_raise, SYMBOL_VENUE

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


class Simulate:
    """Main simulation engine for backtesting trading strategies.
    
    This class orchestrates the entire backtesting process, simulating how the
    trading system would perform historically. It handles:
    
    - Loading historical market data and features
    - Running portfolio optimization to generate target positions
    - Simulating order execution with realistic slippage models
    - Tracking positions, cash, and PnL throughout the simulation
    - Calculating funding payments for perpetual futures
    - Enforcing risk limits and position constraints
    
    The simulation runs on a minute-by-minute basis, making trading decisions
    at specified intervals (e.g., every 120 minutes) and tracking performance
    metrics continuously.
    
    Attributes:
        config (dict): Configuration dictionary with all parameters
        sim_dir (str): Directory for saving simulation results
        trade_bot (PortfolioOptimizer): Portfolio optimization engine
        calcs (Calcs): Feature calculation engine
        execution_stats_list (List[dict]): Execution statistics by timestamp
        pnl_calculator (FillPnl): PnL calculation engine
        
    Example:
        >>> sim = Simulate(
        ...     config=config,
        ...     sim_dir='/simulations/test_strategy',
        ...     initialize_positions=False
        ... )
        >>> results = sim.simulate(
        ...     start_date=date(2023, 1, 1),
        ...     end_date=date(2023, 12, 31)
        ... )
    """

    def __init__(
            self,
            config: dict,
            sim_dir: str,
            initialize_positions: bool = False,
            verbose: bool = False,
            sim_comp: bool = False,
            skip_missing_alphas: bool = False,
            sim_dir_manager: DirectoryManager = dir_manager,
            config_grid: Optional[str] = None,
            alpha_condition: Optional[Literal['rev', 'mom']] = None,
            alpha_dir: Optional[str] = None,
            cleanup_checkpoints: bool = False
    ):
        """Initialize the simulation engine.

        Args:
            config: Configuration dictionary containing all trading parameters
            sim_dir: Directory path for saving simulation results
            initialize_positions: Whether to load existing positions from file
            verbose: Enable verbose logging
            sim_comp: Whether this is a comparison simulation (affects data loading)
            skip_missing_alphas: Continue simulation even if alpha files are missing
            sim_dir_manager: Directory manager for data paths
            config_grid: Grid search configuration name (for parameter optimization)
            alpha_condition: Filter alphas by condition ('rev' for reversal, 'mom' for momentum)
            cleanup_checkpoints: Remove intermediate checkpoint files after simulation
        """
        self.config = config
        self.sim_comp = sim_comp
        self.sim_dir = sim_dir
        self.verbose = verbose
        self.initialize_positions = initialize_positions
        self.skip_missing_alphas = skip_missing_alphas
        self.config_grid = config_grid
        self.alpha_condition = alpha_condition
        self.alpha_dir = alpha_dir
        self.cleanup_checkpoints = cleanup_checkpoints

        self.dir_manager = sim_dir_manager
        self.data_loader = DataLoader(self.config, self.dir_manager)
        self.universe = Universe(config=self.config, debug=self.verbose, universe_dir_manager=self.dir_manager)

        self.forecast_configs = self.config['FCASTS']
        self.reopt_interval = self.config['REOPTIMIZE_INTERVAL_MINS']
        self.max_portfolio_notional = self.config['MAX_PORTFOLIO_NOTIONAL']
        self.opt_horizon = self.config['OPT_HORIZON']
        self.opt_offset_minutes = self.config['OPT_OFFSET_MINS']
        self.max_volume_fraction_participation = self.config['MAX_VOLUME_FRACTION_PARTICIPATION']
        self.min_trade_dollars = self.config['MIN_TRADE_DOLLARS']
        self.default_slippage = self.config['DEFAULT_SLIPPAGE']
        self.exchange_fees = self.config['EXCHANGE_FEES']
        self.risk_horizon = self.config['RISK_HORIZON']
        self.risk_fld = self.config['RISK_FLD'].replace("HORIZON", str(self.risk_horizon))
        self.sim_execution_price = self.config['SIM_EXECUTION_PRICE']
        self.skyrocketing_sigma_level = self.config['SKYROCKETING_SIGMA_LEVEL']
        self.skyrocketing_slippage = self.config['SKYROCKETING_SLIPPAGE']
        self.dynamic_universe = self.config['DYNAMIC_UNIVERSE']
        self.horizon_models = extract_horizon_models(config=self.config, exclude_short_term=True, exclude_zero_weight=True)
        self.horizons = sorted(unique_list([hm[0] for hm in self.horizon_models]))
        # default the vwap horizon to the reopt interval.  used when horizon == 0
        self.vwap_horizon = self.reopt_interval
        if self.sim_execution_price in ('vwap', 'twap'):
            if self.config['VWAP_HORIZON'] != 0:
                self.vwap_horizon = self.config['VWAP_HORIZON']
        self.vwap_name = f"{self.sim_execution_price}_{self.vwap_horizon}"
        self.SUSPICIOUS_PORTFOLIO_SIZE = self.max_portfolio_notional * 1.5

        self.perp_universe = get_config_universe(self.config)

        self.initial_symbol_venue_universe = None
        if not self.dynamic_universe:
            self.initial_symbol_venue_universe = get_config_universe(self.config, symbol_type=SYMBOL_VENUE)

        self.trade_bot = PortfolioOptimizer(config=config, horizons=self.horizons)
        self.calcs = Calcs(config=config)

        self.execution_stats_list = []
        self.fundings_dfs = []
        self.fills_dfs = []
        self.pnl_calculator = None
        logger.info(f"Initializing Simulation {self.initialize_positions} on universe size {len(self.perp_universe)}, writing to {self.sim_dir}")
        if self.alpha_dir is not None:
            logger.info(f"Using alphas from {alpha_dir}")

    def _record_pnl(self, ts_df: pd.DataFrame) -> str:
        """Record PnL and trading statistics for a single timestamp.
        
        Calculates and logs key metrics including:
        - Total PnL for the timestamp
        - Long/short position notionals
        - Trading volumes (buys and sells)
        - Unexecuted trade amounts
        - Slippage costs
        
        Also performs a sanity check on portfolio size to detect errors.
        
        Args:
            ts_df: DataFrame with simulation results for a single timestamp
            
        Returns:
            Pipe-delimited string with timestamp and key metrics
            
        Raises:
            Exception: If portfolio size exceeds suspicious threshold
        """
        pnl = ts_df['pnl'].sum()
        traded_dollars_long = ts_df[ts_df['executed_dollars'] > 0]['executed_dollars'].sum()
        traded_dollars_short = ts_df[ts_df['executed_dollars'] < 0]['executed_dollars'].sum()
        pos_long = ts_df[ts_df['position'] > 0]['position'].sum()
        pos_short = ts_df[ts_df['position'] < 0]['position'].sum()
        total_notional = pos_long - pos_short
        unexecuted = ts_df['unexecuted_dollars'].abs().sum()
        fill_slip = ts_df['fill_slippage'].sum()
        ts = ts_df['ts'].max()

        if total_notional > self.SUSPICIOUS_PORTFOLIO_SIZE and not self.initialize_positions:
            logger.info(f"{ts} PNL ${pnl:.0f} Notional: ${pos_long:.0f} / ${pos_short:.0f} Traded: ${traded_dollars_long:.0f} / ${traded_dollars_short:.0f} Unexecuted: ${unexecuted:.0f} Slip: ${fill_slip:.0f}")
            previous_cols = ['position_previous', 'qty_previous', 'close_mid_previous', 'desired_trade_dollars_previous']
            # Avoid notional load from init position already greater than SUSPICIOUS_PORTFOLIO_SIZE when we don't have previous data columns yet
            if not all(col in ts_df.columns for col in previous_cols):
                previous_cols = []
            print(ts_df[['symbol', 'position', 'close_mid', 'qty', 'limit_qty_pos', 'limit_qty_neg', 'target_opt'] + previous_cols].sort_values(by='position'))
            raise log_and_raise(f"Portfolio grew to {total_notional:.0f}, something is wrong...")

        logger.info(f"{ts} PNL ${pnl:.0f} Notional: ${pos_long:.0f} / ${pos_short:.0f} Traded: ${traded_dollars_long:.0f} / ${traded_dollars_short:.0f} Unexecuted: ${unexecuted:.0f} Slip: ${fill_slip:.0f}")
        ret = f"{ts}|{pnl:.0f}|{pos_long:.0f}|{pos_short:.0f}|{traded_dollars_long:.0f}|{traded_dollars_short:.0f}|{unexecuted:.0f}|{fill_slip:.0f}"

        self.execution_stats_list.append({
            "ts": ts,
            "unexecuted": unexecuted,
            "fill_slip": fill_slip,
            "expected_utility": self.trade_bot.util_metrics.get('expected_utility', 0),
            "expected_risk": self.trade_bot.util_metrics.get('expected_risk', 0),
            "expected_factor_risk": self.trade_bot.util_metrics.get('expected_factor_risk', 0),
            "expected_resid_risk": self.trade_bot.util_metrics.get('expected_resid_risk', 0),
            "expected_return": self.trade_bot.util_metrics.get('expected_return', 0),
            "expected_return_bps": self.trade_bot.util_metrics.get('expected_return_bps', 0),
            "expected_slippage": self.trade_bot.util_metrics.get('expected_slippage', 0),
            "expected_fees": self.trade_bot.util_metrics.get('expected_fees', 0),
        })
        return ret

    def _calculate_funding_for_sim(self, df: pd.DataFrame, funding_df: pd.DataFrame, origin: dt, alpha_times: pd.DatetimeIndex) -> pd.DataFrame:
        """Calculate funding payments for perpetual futures positions.
        
        Funding rates are periodic payments between long and short positions in
        perpetual futures markets. This method:
        
        1. Aggregates funding rates over the reoptimization interval
        2. Calculates the average funding timestamp for position evaluation
        3. Estimates the percentage of the position held at funding time
        
        The calculation approximates funding income by using the average position
        during the funding period rather than tracking exact position changes.
        
        Args:
            df: Main simulation DataFrame
            funding_df: DataFrame with funding rate data
            origin: Time origin for resampling alignment
            alpha_times: Timestamps where alpha signals are generated
            
        Returns:
            DataFrame with added funding rate columns:
            - realized_funding_rate: Sum of funding rates in the period
            - percent_executed_at_funding: Position fraction at funding time
        """

        funding_ts_idx = funding_df.index.get_level_values("ts") == funding_df['next_funding_time']
        funding_ts_df = funding_df.loc[funding_ts_idx, ['last_funding_rate', 'next_funding_time']]
        # Funding Rate Calculation:
        # 1. Sum of funding rates: Aggregates all funding payments that occurred during the last reoptimization period
        #    Example: If reoptimization happens at 6am and 12pm, with funding every 4 hours:
        #    - At 12pm evaluation, we sum funding rates from 8am and 12pm (two funding events)
        # 2. Mean funding timestamp: Represents the effective time point for position evaluation
        #    Using the same example:
        #    - Two funding events (8am and 12pm) average to 10am
        #    - We use the portfolio position at 10am to calculate the funding income
        # Note: This is an approximation method to handle multiple funding events within a reoptimization interval
        # rather than an exact calculation of funding income
        # Group by symbol_venue and resample to handle datetime unstacking issue
        funding_results = []
        for symbol_venue, group in funding_ts_df.groupby(level='symbol_venue', observed=True):
            # Resample funding rates - need to reset index to have DatetimeIndex for resample
            rate_group = group[['last_funding_rate']].reset_index(level='symbol_venue', drop=True)
            rate_resampled = rate_group.resample(f"{self.reopt_interval}min",
                                                 origin=origin,
                                                 label='right',
                                                 closed='right').sum()
            rate_resampled.rename(columns={'last_funding_rate': 'realized_funding_rate'}, inplace=True)

            # Resample funding times - convert to int64 first to avoid datetime issues
            time_group = group[['next_funding_time']].copy().reset_index(level='symbol_venue', drop=True)
            time_group['next_funding_time_int'] = datetime_series_to_int64(time_group['next_funding_time'], context=f"symbol={symbol_venue}")
            time_resampled = time_group[['next_funding_time_int']].resample(f"{self.reopt_interval}min",
                                                                            origin=origin,
                                                                            label='right',
                                                                            closed='right').mean()
            # Convert back to datetime
            time_resampled['next_funding_time'] = to_datetime(time_resampled['next_funding_time_int'], unit='ns').astype('datetime64[ns, UTC]')

            time_resampled.drop(columns=['next_funding_time_int'], inplace=True)

            # Merge rate and time data
            symbol_funding = merge_on_index(rate_resampled, time_resampled)
            symbol_funding['symbol_venue'] = symbol_venue
            funding_results.append(symbol_funding)

        # Combine all symbols
        if funding_results:
            funding_df = pd.concat(funding_results)
            funding_df = funding_df.set_index('symbol_venue', append=True)
        else:
            # Empty case
            funding_df = pd.DataFrame(columns=['realized_funding_rate', 'next_funding_time'])
        # Calculate percent_executed_at_funding if we have data
        if len(funding_df) > 0:
            funding_df['percent_executed_at_funding'] = (self.reopt_interval - (funding_df.index.get_level_values("ts") - funding_df['next_funding_time']).dt.total_seconds() / 60) / self.reopt_interval
            funding_df = funding_df.loc[funding_df.index.get_level_values('ts').isin(alpha_times), ['realized_funding_rate', 'percent_executed_at_funding']]
        else:
            # Create empty dataframe with expected columns and index
            funding_df = pd.DataFrame(columns=['realized_funding_rate', 'percent_executed_at_funding'])
            funding_df.index = pd.MultiIndex.from_tuples([], names=['ts', 'symbol_venue'])

        df = merge_on_index(df, funding_df)
        df['percent_executed_at_funding'] = df['percent_executed_at_funding'].fillna(0)
        df['realized_funding_rate'] = df['realized_funding_rate'].fillna(0)

        return df

    def _sim_data_for_date(
            self,
            sim_date: date,
            horizons: Optional[List[int]] = None,
            models: Optional[List[str]] = None,
            weight_override: bool = False,
            last_pos_df: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        """Load and prepare all data needed for simulating a single date.
        
        This method orchestrates the complex data loading process:
        
        1. Filters universe based on previous day's activity
        2. Loads alpha signals for specified models/horizons
        3. Calculates combined alpha signals
        4. Loads features needed for optimization
        5. Loads bar data for execution simulation
        6. Prepares funding rate data
        7. Adds delisting information
        
        The data is resampled to the reoptimization frequency and aligned
        to ensure consistency across all data sources.
        
        Args:
            sim_date: Date to simulate
            horizons: List of alpha horizons to include (default: all)
            models: List of model names to include (default: all)
            weight_override: Override model weights with equal weighting
            last_pos_df: Previous positions for universe filtering
            
        Returns:
            DataFrame with all data needed for simulation, or None if data missing
        """
        horizon_models = self.horizon_models.copy()
        if models is not None:
            horizon_models = [hm for hm in horizon_models if hm[1] in models]

        if horizons is not None:
            horizon_models = [hm for hm in horizon_models if hm[0] in horizons]

        # Use previous day's universe to match live trading behavior
        # (universe data for today wouldn't be available at trading time)
        universe_df = self.data_loader.load_universe_df(universe_date=sim_date - td(days=1))
        universe_df = universe_df[universe_df['tradeable']]
        if not self.dynamic_universe:
            symbols = self.initial_symbol_venue_universe
        else:
            symbols = list(universe_df['symbol_venue'].unique())
        logger.info(f"Using {self.dynamic_universe=} universe for {sim_date} with {len(symbols)} symbols")

        if last_pos_df is not None:
            position_symbols = list(last_pos_df[last_pos_df['qty'] != 0].index.get_level_values('symbol_venue').unique())
            logger.info(f"Last positions for {sim_date} have {len(position_symbols)} symbols")
            symbols = list(set(position_symbols) | set(symbols))

        logger.info(f'Simulating on {sim_date} with {len(symbols)} symbols {symbols}')
        alphas_df = self.data_loader.load_alphas(
            horizon_models=horizon_models,
            start_date=sim_date,
            end_date=sim_date,
            symbol_venues=symbols,
            verbose=self.verbose,
            prod=self.sim_comp,
            ok_to_return_nothing=self.skip_missing_alphas,
            alpha_dir=self.alpha_dir
        )
        if alphas_df is None:
            logger.warning(f"No alphas for {sim_date}")
            return None
        logger.info(f"Loaded alphas for {sim_date} with {alphas_df.index.get_level_values('symbol_venue').nunique()} symbols")

        alphas_df, _ = calculate_horizon_alphas(
            config=self.config,
            df=alphas_df,
            horizons=horizons if horizons is not None else self.horizons,
            models=models,
            weight_override=weight_override,
            skip_missing_alphas=self.skip_missing_alphas,
            alpha_condition=self.alpha_condition,
        )
        if alphas_df is None:
            logger.warning(f"Could not calculate alphas for {sim_date}")
            return None

        if not self.sim_comp:
            alpha_flds = []
            for hh in unique_list([h[0] for h in horizon_models]):
                alpha_flds += [f"alpha_{hh}", f"alpha_{hh}_rev", f"alpha_{hh}_mom"]
            alphas_df = alphas_df[alpha_flds]

        origin = date_to_start_dt(sim_date) + td(minutes=self.opt_offset_minutes)
        alphas_df = alphas_df.unstack() \
            .resample(f"{self.reopt_interval}min", origin=origin, label='right', closed='right') \
            .last() \
            .stack(future_stack=True)
        end_dt = date_to_end_dt(sim_date)
        ts_vals = alphas_df.index.get_level_values('ts')
        overflow = ts_vals > end_dt
        if overflow.any():
            sv_vals = alphas_df.index.get_level_values('symbol_venue')
            clamped_ts = ts_vals.where(~overflow, end_dt)
            alphas_df.index = pd.MultiIndex.from_arrays([clamped_ts, sv_vals], names=['ts', 'symbol_venue'])
        alphas_df = alphas_df[alphas_df.index.get_level_values('ts') <= end_dt]

        feature_horizons = list({1440, self.reopt_interval, self.opt_horizon, self.risk_horizon})
        risk_features = list(set(self.config['FACTOR_SIGMAS'].keys()) - {"dollar_exposure"})
        vol_bound_features = ['logret_1440_trstd_cz'] if self.config.get('VOL_BOUND_COEFF', 0) > 0 else []
        feature_columns = unique_list([
                                          'dvolume_1440_trmean_cz',
                                          self.risk_fld, 'logret_1440_cz', 'logret_1440_lz',
                                          'beta_1440', 'relative_spread_1440_trmean', f'logret_{self.reopt_interval}_trstd',
                                      ] + risk_features + PNL_BREAKDOWN_FEATURES_FEATURES + vol_bound_features,
                                      ) if not self.sim_comp else None

        features_df = self.data_loader.load_features(
            start_date=sim_date,
            end_date=sim_date,
            horizons=feature_horizons,
            cols=feature_columns,
            symbol_venues=symbols
        )
        if features_df is None:
            logger.warning(f"Could not calculate features for {sim_date}")
            return None

        features_df = features_df.reset_index()

        # Drop columns that will be sourced from universe_df instead
        universe_cols = ['expandable', 'tradeable', 'advp', 'mdvp', 'dvolume_1440_trmean', 'marketcap']
        for col in universe_cols:
            if col in features_df.columns:
                features_df = features_df.drop(columns=[col])
        features_df = pd.merge(features_df, universe_df[universe_cols + ['symbol_venue']], on='symbol_venue', how='left')
        features_df = set_index(features_df)
        features_df['expandable'] = features_df['expandable'].fillna(False)
        features_df['tradeable'] = features_df['tradeable'].fillna(False)

        alpha_times = alphas_df.index.get_level_values('ts').unique()
        features_df = features_df[features_df.index.get_level_values('ts').isin(alpha_times)]
        logger.info(f"Loaded features for {sim_date} with {features_df.index.get_level_values('symbol_venue').nunique()} symbols")
        df = merge_on_index(features_df, alphas_df, how='outer')
        logger.info(f"Combined features and alphas for {sim_date} with {df.index.get_level_values('symbol_venue').nunique()} symbols")

        if not self.sim_comp:
            bar_columns = [
                'last_funding_rate',
                'next_funding_time',
                'close_mid',
                f'dvolume_{self.reopt_interval}',
                f'logret_{self.reopt_interval}',
                f'vwap_{self.reopt_interval}',
                f'high_mid_{self.reopt_interval}',
            ]
            if self.reopt_interval == 1440:
                bar_columns = unique_list(bar_columns + PNL_BREAKDOWN_FEATURES_BARS)
        else:
            bar_columns = None

        bars_df = self.data_loader.load_bars(
            horizon=self.reopt_interval,
            start_date=sim_date,
            end_date=sim_date,
            cols=bar_columns,
            symbol_venues=symbols
        )
        funding_df = bars_df[['last_funding_rate', 'next_funding_time']]

        if self.reopt_interval != 1440:
            # add logret_resid_eqmkt_1440 column here for debugging, should remove it afterwards

            bars_1440_df = self.data_loader.load_bars(
                horizon=1440,
                start_date=sim_date,
                end_date=sim_date,
                cols=unique_list(['logret_1440', 'logret_resid_eqmkt_1440'] + PNL_BREAKDOWN_FEATURES_BARS),
                symbol_venues=symbols
            )
            bars_df = merge_on_index(bars_df, bars_1440_df, how='inner')



        if self.sim_execution_price == 'vwap' and self.vwap_horizon != self.reopt_interval:
            shift = self.reopt_interval - self.vwap_horizon
            logger.info(f"Adding vwap at {self.vwap_horizon} with {shift=} (goes into the future!) careful!")
            end_date = min(sim_date + td(days=2), today_date())

            bars_vwap_df = self.data_loader.load_bars(
                horizon=self.vwap_horizon,
                start_date=sim_date,
                end_date=end_date,
                cols=[self.vwap_name],
                symbol_venues=symbols,
            )
            bars_vwap_df = bars_vwap_df[[self.vwap_name]].unstack().shift(shift).stack(future_stack=True)
            bars_df = merge_on_index(bars_df, bars_vwap_df[[self.vwap_name]], how='inner')

        bars_df = bars_df[bars_df.index.get_level_values('ts').isin(alpha_times)]
        logger.info(f"Loaded bars for {sim_date} with {bars_df.index.get_level_values('symbol_venue').nunique()} symbols")
        df = merge_on_index(df, bars_df, how='outer')
        df = make_market_weight_col(df)
        logger.info(f"Combined features, alphas and bars for {sim_date} with {df.index.get_level_values('symbol_venue').nunique()} symbols")

        df = self.calcs.calculate_volume_forecast(df, horizon=self.reopt_interval)

        # not quite right should reflect the vwap horizon above
        df[f'volume_{self.reopt_interval}'] = df[f'dvolume_{self.reopt_interval}'] / df[self.vwap_name]
        df = self._calculate_funding_for_sim(df=df, funding_df=funding_df, origin=origin, alpha_times=alpha_times)

        df = self.data_loader.add_delisting_dates(df, start_date=sim_date, end_date=sim_date)

        mkt_df = load_mktcap_and_groupings(universe_date=sim_date, mktcap_dir=self.dir_manager.MARKETCAP_DIR, categories_only=True)
        df = pd.merge(df.reset_index(), mkt_df, how='left', left_on='symbol_venue', right_on='symbol_venue')
        df = convert_category_cols_to_bool(df)
        df = set_index(df)

        min_ts, max_ts = get_min_max_ts(df)
        logger.info(f"Loaded data from {min_ts} - {max_ts}")

        # forget why , but this is needed
        df = df.reset_index()
        df['expandable'] = df['expandable'].fillna(False)
        logger.info(f"Final data for {sim_date} has shape {df.shape} with {df.symbol_venue.nunique()} symbols")

        return df

    def _generate_targets(self, ts_df: pd.DataFrame, ts: pd.Timestamp, horizons: Optional[List[int]] = None) -> pd.DataFrame:
        """Generate target positions using portfolio optimization.
        
        Runs the full optimization pipeline:
        
        1. Calculate risk metrics for current positions
        2. Prepare alpha signals for optimization
        3. Apply trading filters (liquidity, expandability)
        4. Run convex optimization to find optimal positions
        5. Calculate target positions and desired trades
        
        If optimization fails at any step, the method falls back to
        keeping current positions and marking trades as unexecuted.
        
        Args:
            ts_df: DataFrame with current market state and positions
            ts: Current timestamp
            horizons: Alpha horizons to use (default: all configured)
            
        Returns:
            DataFrame with added columns:
            - target_opt: Optimal target position
            - desired_trade_dollars: Dollar amount to trade
            - desired_trade_qty: Number of units to trade
            - unexecuted_dollars: Failed trade amounts
        """
        logger.info(f"Optimizing at {ts}")
        opt_fail = False
        ts_df = self.calcs.calculate_risk(ts_df, risk_frequency=self.opt_horizon, sample_frequency=self.risk_horizon)
        try:
            ts_df, _ = self.trade_bot.make_alpha_opt(ts_df, horizons=horizons)
        except OptimizationError:
            opt_fail = True

        if not opt_fail:
            # ts_df = self.calcs.calculate_tradeable_filter(ts_df)
            ts_df = self.calcs.calculate_expandable_filter(ts_df)
            ts_df = self.calcs.calculate_exclusions_and_bounds(ts_df)
            try:
                ts_df = self.trade_bot.optimize(ts_df)
            except OptimizationError:
                opt_fail = True

        if not opt_fail:
            ts_df = self.trade_bot.calculate_targets(ts_df)
            ts_df = self.trade_bot.generate_desired_trades(df=ts_df)
            ts_df['unexecuted_dollars'] = np.float32(0.0)
        else:
            ts_df['unexecuted_dollars'] = ts_df['desired_trade_dollars_previous'] - ts_df['executed_dollars']
            ts_df['desired_trade_dollars'] = ts_df['unexecuted_dollars']
            ts_df['desired_trade_qty'] = ts_df['desired_trade_dollars'] / ts_df['close_mid']

        return ts_df

    def _calculate_position_age(self, ts_df: pd.DataFrame, current_ts: pd.Timestamp) -> pd.DataFrame:
        """Calculate the age of positions for risk management.
        
        Position age is used to identify stale positions that may need
        special handling. The calculation follows these rules:
        
        - Position flips or closures: Age resets to current time
        - No change: Keep existing age
        - Position decrease: Keep existing age (FIFO assumption)
        - Position increase: Weighted average of old and new
        
        This helps identify positions that have been held too long
        and may need increased risk penalties.
        
        Args:
            ts_df: DataFrame with position data
            current_ts: Current simulation timestamp
            
        Returns:
            DataFrame with added columns:
            - position_ts: Timestamp when position was established
            - position_age: Age of position in days
        """
        ts_df['existing_qty'] = ts_df['qty'] - ts_df['executed_qty']
        existing_ts = datetime_series_to_epoch_seconds(ts_df['position_ts_previous'])
        executed_ts = datetime_series_to_epoch_seconds(ts_df['executed_position_ts'])
        position_flip = (ts_df['existing_qty'] * ts_df['qty']) < 0
        position_closed = ts_df['qty'] == 0
        no_change = ts_df['executed_qty'] == 0
        both_zero = (ts_df['existing_qty'] == 0) & (ts_df['qty'] == 0)
        position_increase = (abs(ts_df['qty']) > abs(ts_df['existing_qty'])) & ~position_flip & ~position_closed & ~no_change
        position_decrease = (abs(ts_df['qty']) < abs(ts_df['existing_qty'])) & ~position_flip & ~position_closed & ~no_change

        ts_df['position_ts'] = pd.Series(pd.NaT, index=ts_df.index, dtype='datetime64[ns, UTC]')
        # For position flips and closed positions: use executed_position_ts
        if ((position_flip | position_closed) & ~both_zero).any():
            ts_df.loc[(position_flip | position_closed) & ~both_zero, 'position_ts'] = to_datetime(ts_df.loc[(position_flip | position_closed) & ~both_zero, 'executed_position_ts'])
        # For no change: use existing_qty_ts
        if (no_change & ~both_zero).any():
            ts_df.loc[no_change & ~both_zero, 'position_ts'] = to_datetime(ts_df.loc[no_change & ~both_zero, 'position_ts_previous'])
        # For both existing_qty and qty being zero, set qty_age to NaT (No valid age)
        ts_df.loc[both_zero, 'position_ts'] = pd.Series(pd.NaT, index=ts_df.loc[both_zero].index, dtype='datetime64[ns, UTC]')
        # For position decrease: use position_ts_previous (FIFO assumption)
        if position_decrease.any():
            ts_df.loc[position_decrease, 'position_ts'] = to_datetime(ts_df.loc[position_decrease, 'position_ts_previous'])
        # For position increase: calculate weighted average
        if position_increase.any():
            valid_increase = position_increase & (ts_df['qty'] != 0)

            if valid_increase.any():
                weight_existing = abs(ts_df.loc[valid_increase, 'existing_qty']) / abs(ts_df.loc[valid_increase, 'qty'])
                weight_executed = abs(ts_df.loc[valid_increase, 'executed_qty']) / abs(ts_df.loc[valid_increase, 'qty'])
                weighted_ts = weight_existing * existing_ts[valid_increase] + weight_executed * executed_ts[valid_increase]
                ts_df.loc[valid_increase, 'position_ts'] = to_datetime(weighted_ts, unit='s')
        ts_df['position_ts'] = ts_df['position_ts'].dt.floor('h')
        ts_df['position_age'] = (current_ts - ts_df['position_ts']).dt.days
        ts_df['position_age'] = ts_df['position_age'].fillna(0).astype('Int32')
        return ts_df

    def _execute(self, ts_df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
        """Simulate order execution with realistic market conditions.
        
        This method models the complexities of real trading:
        
        1. Handles missing prices (delisted assets)
        2. Enforces volume constraints (participation limits)
        3. Models execution prices with slippage
        4. Handles extreme price moves (skyrocketing shorts)
        5. Updates positions and calculates fees
        6. Tracks execution statistics
        
        Special cases handled:
        - Dead assets: Liquidate at last known price
        - Skyrocketing shorts: Force liquidation at loss
        - Volume limits: Cap trades at max participation rate
        - Small positions: Round to zero to avoid dust
        
        Args:
            ts_df: DataFrame with desired trades and current state
            ts: Current timestamp
            
        Returns:
            DataFrame with execution results:
            - executed_qty: Actual quantity traded
            - trade_price: Execution price including slippage
            - fees: Trading fees paid
            - position: Updated position
            - cash: Updated cash balance
            - pnl: Current PnL
        """
        logger.info(f"Executing at {ts}")

        bad_price_loc = (~(ts_df['close_mid'] > 0) | ts_df['close_mid'].isna())
        liquidate_loc = None
        bad_prc_cnt = len(ts_df[bad_price_loc])
        if bad_prc_cnt > 0:
            logger.info(f"No prices for {bad_prc_cnt} secs, trying carry forward.")
            do_i_care_loc = ts_df['position'] != 0
            print_df = ts_df[bad_price_loc & do_i_care_loc]
            liquidation_cnt = len(print_df)
            if liquidation_cnt > 0:
                liquidation_symbols = [str(sym) for sym in print_df['symbol_venue'].unique()]
                logger.warning(f"Need to liquidate {liquidation_cnt} positions: {liquidation_symbols}")
                print(print_df[['symbol_venue', 'ts', 'close_mid', 'close_mid_previous', 'position', 'qty', 'desired_trade_qty_previous']])
                liquidate_loc = bad_price_loc & do_i_care_loc

        # put skyrocketing price check before we overwrite close mid with previous close mid for bad_price_loc
        skyrocketing_price_loc = None
        if self.skyrocketing_sigma_level is not None:
            ts_df[f'high_logret_{self.reopt_interval}'] = np.log(ts_df[f'high_mid_{self.reopt_interval}'] / ts_df['close_mid_previous'])
            skyrocketing_price_loc = ts_df[f'high_logret_{self.reopt_interval}'] > self.skyrocketing_sigma_level * ts_df[f'logret_{self.reopt_interval}_trstd_previous']

        ts_df.loc[bad_price_loc, 'close_mid'] = ts_df['close_mid_previous']
        new_bad_price_loc = (~(ts_df['close_mid'] > 0) | ts_df['close_mid'].isna()) & ((ts_df['qty'] != 0) | ~ts_df['desired_trade_qty_previous'].isna())
        if len(ts_df[new_bad_price_loc]) > 0:
            print(ts_df[new_bad_price_loc][['symbol_venue', 'ts', 'close_mid', 'close_mid_previous', 'position', 'qty', 'desired_trade_qty_previous']])
            raise log_and_raise("Bad execution prices even with carry forward!")

        trade_loc = (ts_df['desired_trade_dollars_previous'].abs() >= self.min_trade_dollars) | \
                    ((ts_df['position'].abs() <= self.min_trade_dollars) & (ts_df['expanding_previous'] < 0))
        trade_loc = trade_loc & (ts_df[self.vwap_name] > 0)

        ts_df['executed_qty'] = np.float32(0)
        ts_df.loc[trade_loc, 'executed_qty'] = ts_df['desired_trade_qty_previous'].fillna(0).astype(np.float32)
        ts_df.loc[trade_loc, 'limit_qty_pos'] = (ts_df[f'volume_{self.reopt_interval}'] * self.max_volume_fraction_participation).astype(np.float32)
        ts_df.loc[trade_loc, 'limit_qty_neg'] = (-ts_df[f'volume_{self.reopt_interval}'] * self.max_volume_fraction_participation).astype(np.float32)
        ts_df.loc[trade_loc & (ts_df['executed_qty'] > 0), 'executed_qty'] = ts_df[['limit_qty_pos', 'executed_qty']].min(axis=1).astype(np.float32)
        ts_df.loc[trade_loc & (ts_df['executed_qty'] < 0), 'executed_qty'] = ts_df[['limit_qty_neg', 'executed_qty']].max(axis=1).astype(np.float32)

        ts_df['trade_price'] = np.float32(0)
        ts_df['trade_price_open'] = np.float32(0)
        ts_df['trade_price_vwap'] = np.float32(0)

        buys_loc = trade_loc & (ts_df['executed_qty'] > 0)
        sells_loc = trade_loc & (ts_df['executed_qty'] < 0)
        ts_df.loc[buys_loc, 'trade_price_vwap'] = (ts_df[self.vwap_name] * (1 + self.default_slippage)).astype(np.float32)
        ts_df.loc[sells_loc, 'trade_price_vwap'] = (ts_df[self.vwap_name] * (1 - self.default_slippage)).astype(np.float32)
        ts_df.loc[buys_loc, 'trade_price_open'] = (ts_df['close_mid_previous'] * (1 + self.default_slippage)).astype(np.float32)
        ts_df.loc[sells_loc, 'trade_price_open'] = (ts_df['close_mid_previous'] * (1 - self.default_slippage)).astype(np.float32)
        if self.sim_execution_price == 'vwap':
            ts_df['trade_price'] = ts_df['trade_price_vwap']
        elif self.sim_execution_price == 'open':
            ts_df['trade_price'] = ts_df['trade_price_open']
        else:
            raise log_and_raise(f"Unknown execution price type {self.sim_execution_price}")

        if skyrocketing_price_loc is not None:
            skyrocket_liquidate_loc = skyrocketing_price_loc & (ts_df['qty'] < 0)
            ts_df['liquidation_price'] = (ts_df['close_mid_previous'] * np.exp(ts_df[f'logret_{self.reopt_interval}_trstd_previous'] * self.skyrocketing_sigma_level) * (1 + self.skyrocketing_slippage)).astype(np.float32)
            liquidate_df = ts_df[skyrocket_liquidate_loc]
            if len(liquidate_df) > 0:
                liquidating_securities = ','.join(liquidate_df['symbol'].astype(str) + ': $' + liquidate_df['position'].astype(str) + ': $' + liquidate_df['liquidation_price'].astype(str))
                logger.info(f"Liquidating skyrocketing short securities {ts}: {liquidating_securities}")
                ts_df.loc[skyrocket_liquidate_loc, 'executed_qty'] = -ts_df['qty'].astype(np.float32)
                ts_df.loc[skyrocket_liquidate_loc, 'trade_price'] = ts_df['liquidation_price']

        if liquidate_loc is not None:
            cnt = len(ts_df[liquidate_loc])
            logger.info(f"Liquidating {cnt} dead securities")
            ts_df.loc[liquidate_loc, 'executed_qty'] = -ts_df['qty']
            ts_df.loc[liquidate_loc, 'trade_price'] = ts_df['close_mid']

        ts_df['executed_dollars'] = ts_df['executed_qty'] * ts_df['trade_price']

        self._update_fill_from_sim_timeslice(ts_df, self.exchange_fees)
        ts_df = self._update_funding_df_from_sim_timeslice(ts_df, self.vwap_name)

        ts_df['qty'] += ts_df['executed_qty']
        ts_df['position'] = ts_df['qty'] * ts_df['close_mid']
        ts_df = self._calculate_position_age(ts_df, ts)

        bad_position_loc = (ts_df['qty'] != 0) & (ts_df['position'] == 0)
        bad_positions_df = ts_df[bad_position_loc]
        if len(bad_positions_df) > 0:
            print(bad_positions_df[['symbol', 'symbol_venue', 'qty', 'close_mid', self.vwap_name, 'position', 'executed_qty']])
            raise log_and_raise("Bad positions found!")

        # zero out small positions
        small_pos_loc = ts_df['position'].abs() < .01
        ts_df.loc[small_pos_loc, 'position'] = 0
        ts_df.loc[small_pos_loc, 'qty'] = 0

        ts_df['fees'] = ts_df['executed_dollars'].abs() * self.exchange_fees
        ts_df['fill_slippage'] += (ts_df['trade_price'] - ts_df['close_mid_previous']) * ts_df['executed_qty']
        if 'funding_income' not in ts_df.columns:
            raise log_and_raise("funding_income column missing - funding calculation failed")
        ts_df['cash'] = ts_df['cash'] - ts_df['executed_dollars'] - ts_df['fees'] + ts_df['funding_income']
        ts_df['pnl'] = ts_df['position'] + ts_df['cash']
        return ts_df

    def _update_funding_df_from_sim_timeslice(self, df: pd.DataFrame, px_at_funding: str) -> pd.DataFrame:
        """Calculate and store funding income from simulation timeslice.

        Calculates funding income/payments based on positions and funding rates.
        Negative funding rates mean longs pay shorts (positive income for shorts).

        Args:
            df: DataFrame with simulation data containing:
                - realized_funding_rate: Funding rate at settlement
                - qty: Position quantity before execution
                - executed_qty: Quantity executed in this period
                - percent_executed_at_funding: Fraction executed before funding
                - symbol: Trading symbol
                - symbol_venue: Symbol with venue
            px_at_funding: Column name containing price at funding time

        Returns:
            Original DataFrame with added 'funding_income' column
        """
        df['funding_income'] = (-df['realized_funding_rate'] * (df['qty'] + df['executed_qty'] * df['percent_executed_at_funding']) * df[px_at_funding]).fillna(0)
        valid_funding_idx = (df['funding_income'] != 0) & (~df['funding_income'].isna())
        to_add_df = df.loc[valid_funding_idx, ['ts', 'symbol', 'funding_income', 'symbol_venue']]
        if len(to_add_df) > 0:
            logger.info(f"Adding funding income: {len(to_add_df)}")
            self.fundings_dfs.append(to_add_df)
        else:
            logger.warning(f"No valid funding rates for dataframe on {df.index[0]}")

        return df

    def _update_fill_from_sim_timeslice(self, df: pd.DataFrame, exchange_fees: float):
        """Update fills from a simulation timeslice.

        Processes simulation execution data and calculates commissions based
        on exchange fees. Appends processed fills to internal list for later
        aggregation.

        Args:
            df: DataFrame containing simulation execution data with columns:
                - executed_qty: Quantity executed
                - executed_dollars: Dollar value of execution
                - trade_price: Execution price
                - symbol_venue: Trading symbol with venue
            exchange_fees: Exchange fee rate to calculate commission
        """
        df = df.loc[df['executed_qty'] != 0]
        if len(df) == 0:
            return

        df['commission'] = df['executed_dollars'].abs() * exchange_fees
        df = make_symbol(df)
        df = make_date(df)
        try:
            fills_df = df[['ts', 'symbol_venue', 'date', 'symbol', 'executed_qty', 'trade_price', 'commission']]
            self.fills_dfs.append(fills_df)
        except KeyError as ke:
            raise log_and_raise(f"Missing required column {ke} in simulation fills", df=df)


    def _dump_pnl_file(
            self,
            pnl_fill_breakdown: FillBreakdown,
            models: Optional[List[str]] = None,
            horizons: Optional[List[int]] = None,
            pnl_calculator_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Save PnL analysis files to simulation directory.
        
        Generates multiple output files for analysis:
        - PnL calculator results with execution statistics
        - Fill breakdown showing sources of PnL
        - Summary statistics (Sharpe, returns, etc.)
        
        File names include model/horizon/grid information for
        easy identification in parameter sweeps.
        
        Args:
            pnl_fill_breakdown: Object tracking PnL attribution
            models: List of models used (for filename)
            horizons: List of horizons used (for filename)
            pnl_calculator_df: DataFrame with PnL time series
        """
        logger.info("Dumping pnl files")
        pnl_name = "pnl"
        if models is not None and len(models) == 1:
            pnl_name += f".model_{models[0]}"
        if horizons is not None and len(horizons) == 1:
            pnl_name += f".horizon_{horizons[0]}"
        if self.config_grid is not None:
            pnl_name += f".{self.config_grid}"
        if self.alpha_condition is not None:
            pnl_name += f".condition_{self.alpha_condition}"

        if pnl_calculator_df is not None:
            self._dump_pnl_summary(pnl_calculator_df, pnl_name, pnl_fill_breakdown)

    def _dump_pnl_summary(self, pnl_calculator_df: pd.DataFrame, pnl_name: str, pnl_fill_breakdown: FillBreakdown) -> None:
        """Write detailed PnL summary and breakdown files.
        
        Saves three types of analysis:
        1. Raw calculator output (CSV) with all metrics
        2. Summary statistics (text) with key performance metrics
        3. PnL breakdown (CSV) showing attribution by source
        
        Args:
            pnl_calculator_df: Time series of PnL and metrics
            pnl_name: Base name for output files
            pnl_fill_breakdown: Fill-level PnL attribution
        """
        logger.info("Dumping pnl summary")
        pnl_calculator_df.to_csv(f"{self.sim_dir}/" + pnl_name + ".calculator.csv")
        daily_scaler = 1440 / self.reopt_interval
        pnl_fill_breakdown.load_fills(self.pnl_calculator.calculator.fills_pnl_df[:])

        pnl_summary = get_return_metrics_str(pnl_calculator_df, pnl_name, daily_scaler, pnl_fill_breakdown)
        pnl_summary_file = f"{self.sim_dir}/summary.txt"
        with open(pnl_summary_file, "a") as outfile:
            outfile.write(pnl_summary)
        breakdown_file = f"{self.sim_dir}/" + pnl_name + ".breakdown.csv"
        self.pnl_calculator.calculator.dump_breakdown_file(breakdown_file)

    def _dump_sim_file(self, daily_df_dump_dict: Dict[str, Tuple[pd.DataFrame, str]]) -> None:
        """Save daily simulation results to parquet files.

        Each day's simulation data is saved as a separate parquet file
        for efficient storage and fast loading for analysis.

        Args:
            daily_df_dump_dict: Map of date to (DataFrame, filepath) tuples
        """
        for date in daily_df_dump_dict:
            day_df, dump_file_path = daily_df_dump_dict[date]
            day_df.to_parquet(dump_file_path)

    def _cleanup_checkpoint_files(self) -> None:
        """Remove intermediate checkpoint files after simulation completes.

        During simulation, checkpoint files (YYYYMMDD_HHMM.parquet) are written
        for each timeslice. These are redundant with daily aggregate files
        (sim.*.parquet) and can be safely removed to save disk space.
        """
        pattern = (
            f"{self.sim_dir}/"
            "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]"
            "_[0-9][0-9][0-9][0-9].parquet"
        )
        checkpoint_files = glob.glob(pattern)
        if checkpoint_files:
            logger.info(f"Cleaning up {len(checkpoint_files)} checkpoint files...")
            for filepath in checkpoint_files:
                os.remove(filepath)
            logger.info("Checkpoint cleanup complete")

    def run_pnl_calc(self, pos_df: pd.DataFrame, daily_df_dump_dict: Dict[str, Tuple[pd.DataFrame, str]]) -> pd.DataFrame:
        """Run comprehensive PnL calculation and save results.
        
        Uses the FillPnl calculator to:
        1. Track position changes and cash flows
        2. Calculate realized and unrealized PnL
        3. Attribute PnL to various sources
        4. Generate aggregate statistics
        
        Args:
            pos_df: Initial positions DataFrame
            daily_df_dump_dict: Daily simulation results to save
            
        Returns:
            DataFrame with timestamp-level PnL metrics and statistics
        """
        logger.info("Calculating Pnl")
        # beware there are side effects of some of these!
        self.pnl_calculator.update_position_from_sim_timeslice(initialize_positions=self.initialize_positions, existing_positions_df=pos_df)

        fills_df = concat(self.fills_dfs, fast=True)
        if fills_df is None:
            raise log_and_raise("No fills generated during simulation")
        fills_df = fills_df.reset_index(drop=True)

        fundings_df = concat(self.fundings_dfs, fast=True)
        if fundings_df is None:
            raise log_and_raise("No fundings generated during simulation")

        ts_symbol_pnl_df = self.pnl_calculator.run_pnl_calculation(fills_df=fills_df, fundings_df=fundings_df)

        ts_symbol_pnl_df.to_parquet(f"{self.sim_dir}/pnl_details.parquet")
        self._dump_sim_file(daily_df_dump_dict)
        _ = self.pnl_calculator.run_pnl_aggregation()
        ts_pnl_df = self.pnl_calculator.run_pnl_aggregation_by_ts()

        execution_stats_df = pd.DataFrame(self.execution_stats_list)
        pnl_calculator_df = pd.merge(ts_pnl_df, execution_stats_df, on='ts')

        return pnl_calculator_df

    def simulate(
            self,
            start_date: date,
            end_date: date,
            horizons: Optional[List[int]] = None,
            models: Optional[List[str]] = None,
            weight_override: bool = False,
    ) -> pd.DataFrame:
        """Run the complete backtesting simulation.
        
        This is the main entry point that orchestrates the entire simulation:
        
        1. Initialize PnL tracking and load starting positions
        2. For each date in the range:
           - Load market data, features, and alphas
           - For each reoptimization timestamp:
             - Execute pending trades from previous period
             - Run portfolio optimization
             - Generate new target positions
             - Track PnL and statistics
        3. Calculate final PnL metrics and save all results
        
        The simulation accurately models:
        - Transaction costs (fees and slippage)
        - Volume constraints and market impact
        - Funding payments for perpetual futures
        - Position limits and risk constraints
        - Delisting and dead assets
        
        Args:
            start_date: First date to simulate
            end_date: Last date to simulate (inclusive)
            horizons: Alpha horizons to use (default: all configured)
            models: Alpha models to use (default: all configured)
            weight_override: Use equal weights instead of configured weights
            
        Returns:
            DataFrame with complete simulation results including:
            - Position time series for all instruments
            - Execution details and slippage
            - PnL attribution and statistics
            
        Raises:
            Exception: If required data is missing or simulation fails
        """
        logger.info(f"Simulating from {start_date} to {end_date}, {weight_override=}")
        logger.info(f"Using Horizons: {horizons}, Models: {models}")

        self.pnl_calculator = FillPnl(
            config=self.config,
            start=date_to_start_dt(start_date),
            end=date_to_start_dt(end_date),
            pnl_dir_manager=self.dir_manager
        )

        pnl_fill_breakdown = FillBreakdown(start=date_to_start_dt(start_date), end=date_to_start_dt(end_date))
        daily_df_dump_dict = {}
        position_dfs = []
        pnls = []
        last_timeslice_df = None
        pos_df = None
        self.execution_stats_list = []
        if self.initialize_positions:
            pos_df = load_positions(mode='bod', start_date=start_date, pos_dir=self.dir_manager.POSITION_DIR)
            if pos_df is None:
                logger.warning(f"No initial positions found for {start_date}, maybe not trading?")
            else:
                # Use current value as cost basis to start PnL at zero when initializing positions
                # (don't include accumulated historical PnL in cost_basis)
                pos_df['cash'] = -pos_df['value']
                logger.info("Initializing positions with cash=-value to start PnL at zero")
                pos_df['position'] = pos_df['value']
        last_pos_df = pos_df

        for date in date_range(start_date, end_date):
            logger.info(f"Simulating on date: {date}")
            df = self._sim_data_for_date(sim_date=date, models=models, horizons=horizons, weight_override=weight_override, last_pos_df=last_pos_df)
            if df is None:
                raise log_and_raise(f"No sim data for {date}")

            day_dfs = []
            for ts in sorted(df['ts'].unique()):
                ts_df = make_symbol(df[df['ts'] == ts])
                logger.info(f"Simulating on time: {ts} with df length {len(ts_df)}, unique symbols {ts_df['symbol'].nunique()}")

                for col in ['limit_qty_pos', 'limit_qty_neg', 'executed_qty', 'fill_slippage']:
                    ts_df[col] = np.float32(0)
                for col in ['expanding']:
                    ts_df[col] = 0

                ts_df['position_ts'] = pd.Series(pd.NaT, index=ts_df.index, dtype='datetime64[ns, UTC]')

                if last_timeslice_df is None:
                    if pos_df is not None:
                        logger.info("Setting initial positions...")
                        ts_df = pd.merge(ts_df, pos_df[['symbol', 'cash', 'qty', 'position']], left_on=['symbol'], right_on=['symbol'], how='left')
                        for fld in ['cash', 'qty', 'position']:
                            ts_df[fld] = ts_df[fld].fillna(0)
                    else:
                        for fld in ['cash', 'qty', 'position']:
                            ts_df[fld] = np.float32(0.0)

                    ts_df.loc[ts_df['qty'] != 0, 'position_ts'] = ts

                    for col in ['position_age', 'position_age_previous', 'day_of_week', 'expanding_previous']:
                        ts_df[col] = 0
                    for col in ['qty_previous', 'position_previous', 'pnl', 'fees', 'funding_income', 'executed_dollars', 'unexecuted_dollars', 'desired_trade_dollars_previous', 'target_opt', 'pnl_previous', 'cash_previous', 'desired_trade_qty_previous', 'fill_slippage_previous',
                                'existing_qty', 'expanding_previous']:
                        ts_df[col] = np.float32(0.0)
                    for col in ['trade_price', f'risk_{self.opt_horizon}', 'close_mid_previous', f'risk_{self.opt_horizon}_previous', 'dvolume_1440_trmean_previous', f'high_logret_{self.reopt_interval}_trstd_previous', 'trade_price_vwap', 'trade_price_open',
                                'logret_120_trstd_previous']:
                        ts_df[col] = np.nan
                    for col in ['position_ts_previous', 'executed_position_ts']:
                        ts_df[col] = pd.NA
                else:
                    for col in ['desired_trade_qty', 'desired_trade_dollars', 'qty', 'pnl', 'cash', 'position']:
                        ts_df[col] = np.float32(0.0)
                    ts_df[f'risk_{self.opt_horizon}'] = np.nan
                    ts_df['executed_position_ts'] = ts

                    ts_df = pd.merge(
                        ts_df,
                        last_timeslice_df[[
                            'symbol_venue',
                            'qty',
                            'position',
                            'pnl',
                            'cash',
                            'desired_trade_dollars',
                            'desired_trade_qty',
                            'expanding',
                            'day_of_week',
                            'close_mid',
                            'target_opt',
                            f'risk_{self.opt_horizon}',
                            'dvolume_1440_trmean',
                            f'logret_{self.reopt_interval}_trstd',
                            'fill_slippage',
                            'position_ts',
                            'position_age',
                        ]],
                        on=['symbol_venue'], how='outer', suffixes=('', '_previous'))

                    # print(ts_df[['symbol_venue', 'qty', 'risk_1440', 'risk_1440_previous', 'dvolume_1440_trmean', 'dvolume_1440_trmean_previous']])

                    for col in ['cash', 'position', 'qty', 'fill_slippage']:
                        ts_df[col] = ts_df[f'{col}_previous'].fillna(0)

                    for col in ['position_age', 'expanding', 'day_of_week']:
                        ts_df[f"{col}_previous"] = ts_df[col].fillna(0).astype('Int32')

                    # TODO: track down why we need to do this!
                    ts_df.loc[ts_df['ts'].isna(), 'ts'] = ts
                    ts_df = self._execute(ts_df, ts)
                    filename = f"{self.sim_dir}/{ts.strftime('%Y%m%d_%H%M')}.parquet"
                    ts_df.to_parquet(filename)

                ts_df = self._generate_targets(set_index(ts_df), ts, horizons=horizons).reset_index()

                for col in ['expanding_previous', 'day_of_week']:
                    ts_df[col] = ts_df[col].fillna(0).astype('Int32')

                pnls.append(self._record_pnl(ts_df))
                last_pos_df = set_index(ts_df)
                day_dfs.append(last_pos_df)
                last_timeslice_df = ts_df

            day_df = concat(day_dfs)
            day_filename = f"{self.sim_dir}/sim.{date_to_str(date)}"
            if models is not None and len(models) == 1:
                day_filename += f".model_{models[0]}"
            if horizons is not None and len(horizons) == 1:
                day_filename += f".horizon_{horizons[0]}"
            if self.config_grid is not None:
                day_filename += f".{self.config_grid}"
            if self.alpha_condition is not None:
                day_filename += f".condition_{self.alpha_condition}"
            daily_df_dump_dict[date_to_str(date)] = (day_df, day_filename + ".parquet")
            position_dfs.append(day_df)

        pnl_calculator_df = self.run_pnl_calc(pos_df, daily_df_dump_dict)
        positions_df = concat(position_dfs, quiet=False)
        self._dump_pnl_file(
            pnl_fill_breakdown=pnl_fill_breakdown,
            models=models,
            horizons=horizons,
            pnl_calculator_df=pnl_calculator_df,
        )
        if self.cleanup_checkpoints:
            self._cleanup_checkpoint_files()
        return positions_df
