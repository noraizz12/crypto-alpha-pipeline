"""Historical trading performance analysis and reporting module.

This module provides comprehensive historical performance reporting including
P&L breakdowns, drawdown analysis, monthly/yearly summaries, and
risk-adjusted performance metrics. Supports multiple time horizons
and aggregation levels.

Classes:
    HistTradingReports: Main class for historical trading performance analysis
"""

import logging
from datetime import date, datetime as dt, timezone, timedelta as td
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dash_table import FormatTemplate

from lib.calcs.calc_returns import calc_factor_return
from lib.pnl_new.pnl_util import calculate_performance_statistics, calc_return_metrics, calculate_performance_by_month
from lib.calcs.calcs import Calcs
from lib.data.dataloader import DataLoader
from lib.data.loaders import load_raw_targets_alpha
from lib.pnl import FillBreakdown
from lib.pnl.fill_pnl_breakdown import PNL_BREAKDOWN_FEATURES
from lib.pnl_new.binance_pnl import BinancePnl
from lib.util.config import get_factors, extract_horizon_models, extract_model_alpha_list
from lib.util.dataframes import concat, make_quintile, make_symbol, make_date, remove_infs

from lib.util.time_util import date_to_start_dt, date_to_end_dt
from lib.util.util import fmoney, unique_list

logger = logging.getLogger(__name__)

FMT_MONEY = FormatTemplate.money(2)


class HistTradingReports:
    """Reports for historical trading performance analysis.

    Provides comprehensive historical performance reporting including
    P&L breakdowns, drawdown analysis, monthly/yearly summaries, and
    risk-adjusted performance metrics. Supports multiple time horizons
    and aggregation levels.

    Attributes:
        config: Configuration dictionary
        start_date: Start date for historical analysis
        end_date: End date for historical analysis
        debug: Debug mode flag
        data_loader: DataLoader instance
        calcs: Calcs instance
        pnl_calculator: BinancePnl calculator instance
        factors: List of factor names
        portfolio_df: Portfolio positions DataFrame
        performance_metrics: Performance metrics dictionary
        win_ratios: Win ratio statistics
        hist_fill_breakdown_pnl: Historical fill breakdown P&L data
        hist_fill_breakdown_pnl_keys: Keys for fill breakdown data
        hist_top_drawdowns: Top drawdown periods
        update_ts: Last update timestamp

    """

    def __init__(self, config: dict, start_date: date, end_date: date, debug: bool = False):
        """Initialize with configuration and date range.

        Args:
            config: Configuration dictionary
            start_date: Start date for historical analysis
            end_date: End date for historical analysis
            debug: Debug mode flag
        """
        self.config = config
        self.start_date = start_date
        self.end_date = end_date
        self.debug = debug

        # Get risk factors from config - alpha columns added after data loading
        self.risk_factors = get_factors(self.config)
        self.alpha_cols = extract_model_alpha_list(self.config)
        self.available_alpha_cols: List[str] = []  # Populated after alpha data loads
        self._loaded_alpha_horizons: set = set()  # Track which horizons have been loaded
        # Extract horizons from config (no data loading needed)
        horizon_models = extract_horizon_models(self.config)
        self.alpha_horizons = sorted(set(h for h, _ in horizon_models), reverse=True)
        self.factors = self.risk_factors  # Alpha cols added when alphas are loaded

        # Initialize core components
        self.data_loader = DataLoader(config=self.config)
        self.calcs = Calcs(self.config, prod=False)
        self.pnl_calculator = BinancePnl(
            config=self.config,
            start=date_to_start_dt(self.start_date),
            end=date_to_end_dt(self.end_date)
        )

        self.security_df: Optional[pd.DataFrame] = None
        self.portfolio_df: Optional[pd.DataFrame] = None
        self.security_ts_df = None
        self.fills_df = None
        self.alphas_df: Optional[pd.DataFrame] = None

        self.fill_breakdown_features = None

        self.performance_metrics: Dict[str, Any] = {}
        self.win_ratios: Dict[str, Any] = {}
        self.hist_fill_breakdown_pnl: Dict[str, pd.DataFrame] = {}
        self.hist_fill_breakdown_pnl_keys: List[str] = []
        self.hist_top_drawdowns: Optional[pd.DataFrame] = None
        self.tradelist_corr_df: pd.DataFrame = pd.DataFrame()
        self.alpha_opt_stats_df: Optional[pd.DataFrame] = None
        self.update_ts: Optional[dt] = None

        # Load data on initialization
        self.load_data()

    def load_data(self) -> None:
        """Load and process all data for historical reports.

        This method orchestrates the entire data loading pipeline:
        1. Initialize PNL calculator with date range
        2. Load fill breakdown data
        3. Calculate win ratios
        4. Prepare portfolio data
        5. Generate PNL breakdowns
        6. Calculate performance metrics
        7. Calculate drawdowns
        8. Load balance data
        """
        logger.info(f"Loading historical data from {self.start_date} to {self.end_date}...")

        # Get daily portfolio aggregation with balance data
        self.security_ts_df = self.pnl_calculator.security_ts_pnl_df
        self.security_df = self.pnl_calculator.aggregate_by_security_date()

        self.portfolio_df = self.pnl_calculator.aggregate_daily_portfolio()
        self.fills_df = self.pnl_calculator.get_fills_df()

        # Initialize fill breakdown and load features
        self._initialize_fill_breakdown()

        # Note: Alpha data is loaded lazily when needed (in update_alpha_by_horizon_figures)

        # Calculate win ratios
        self.win_ratios = {
            'mtd': self.fill_breakdown.win_ratio(start_date=self.end_date.replace(day=1)),
            'ytd': self.fill_breakdown.win_ratio(start_date=self.end_date.replace(day=1, month=1)),
            'lifetime': self.fill_breakdown.win_ratio(),
        }

        # PNL breakdown features are loaded lazily when breakdown graphs are requested
        self.hist_fill_breakdown_pnl = {}
        self.hist_fill_breakdown_pnl_keys = list(PNL_BREAKDOWN_FEATURES)

        # Calculate performance metrics
        self.performance_metrics = self._calc_all_performance_metrics(self.portfolio_df)

        self._calculate_drawdowns()

        # Load alpha_opt statistics for timeseries chart
        self._load_alpha_opt_stats()

        # Update timestamp
        self.update_ts = dt.now(timezone.utc)

        logger.info("Historical data loaded successfully")

    def _initialize_fill_breakdown(self):
        """Initialize fill breakdown with minimal data loading.

        Only loads one day of data to get column names. Full feature data
        is loaded lazily when needed for factor analysis or PNL breakdowns.
        """
        fill_breakdown = FillBreakdown(start=date_to_start_dt(self.start_date))
        # Only load one day to get column names (fast)
        fill_breakdown.load_features_df(cols=None, update_name_list=True)
        fill_breakdown.load_fills(self.fills_df)
        self.fill_breakdown = fill_breakdown
        self.fill_breakdown_features = {}
        self._features_loaded = False  # Track if full features have been loaded
        self._loaded_factors: set = set()  # Track which factors have been loaded
        self._loaded_alpha_horizons: set = set()  # Reset because alpha data in features_df is cleared when FillBreakdown is reinitialized

    def _load_features_for_factor(self, factor: str) -> None:
        """Load feature data for a specific factor.

        Args:
            factor: The factor name to load (e.g., 'dvolume_1440_trmean_cz')
        """
        if factor in self._loaded_factors:
            return

        if self.fill_breakdown is None:
            return

        # Check if factor exists in available features
        all_available = self.fill_breakdown.bars_name_list + self.fill_breakdown.features_name_list
        if factor not in all_available and factor != 'dollar_exposure':
            logger.warning(f"Factor '{factor}' not available in features")
            self._loaded_factors.add(factor)
            return

        logger.info(f"Loading feature data for factor: {factor}")

        try:
            # Load just this factor's data - this overwrites features_df
            self.fill_breakdown.load_features_df(cols=[factor], update_name_list=False)
            self._loaded_factors.add(factor)
            # Reset alpha horizons since features_df was overwritten and alpha data is lost
            self._loaded_alpha_horizons = set()
            logger.info(f"Loaded feature data for factor: {factor}")
        except Exception as e:
            logger.error(f"Error loading feature for factor {factor}: {e}")
            self._loaded_factors.add(factor)

    def _load_pnl_breakdown_feature(self, feature: str) -> None:
        """Load feature data for PNL breakdown and calculate breakdown.

        Args:
            feature: The feature name for breakdown (e.g., 'dvolume_1440_lz')
        """
        if feature in self.hist_fill_breakdown_pnl:
            return

        if self.fill_breakdown is None:
            return

        logger.info(f"Loading PNL breakdown for feature: {feature}")

        try:
            # Load the feature data if not already loaded
            self._load_features_for_factor(feature)

            # Calculate breakdown for this feature
            if self.fill_breakdown.features_df is not None and feature in self.fill_breakdown.features_df.columns:
                feature_data = self.fill_breakdown.features_df.groupby(
                    ['date', 'symbol_venue'], observed=False
                ).agg({feature: 'last'}).reset_index()
                self.fill_breakdown_features[feature] = feature_data

                # Calculate cumulative PNL breakdown
                df = self._pnl_by_col(self.fills_df, self.fill_breakdown_features, feature, by_date=True)
                if df is not None:
                    df = df.unstack().fillna(0).cumsum().stack(future_stack=True).reset_index()
                    self.hist_fill_breakdown_pnl[feature] = df
                    logger.info(f"Calculated PNL breakdown for feature: {feature}")
        except Exception as e:
            logger.error(f"Error loading PNL breakdown for {feature}: {e}")

    def _load_alpha_data_for_horizon(self, horizon: int) -> None:
        """Load alpha signals for a specific horizon and merge into features_df.

        Loads individual model alphas for the specified horizon only (e.g., alpha_hl_720)
        from the alpha directory and merges them into fill_breakdown.features_df.

        Args:
            horizon: The horizon to load alphas for (e.g., 720, 1440)
        """
        if horizon in self._loaded_alpha_horizons:
            return

        # Get horizon-model combinations for this specific horizon only
        horizon_models = extract_horizon_models(self.config, horizons=[horizon])
        if not horizon_models:
            logger.warning(f"No models found for horizon {horizon}")
            self._loaded_alpha_horizons.add(horizon)
            return

        logger.info(f"Loading alpha data for horizon {horizon}: {len(horizon_models)} models...")

        try:
            alphas_df = self.data_loader.load_alphas(
                horizon_models=horizon_models,
                start_date=self.start_date,
                end_date=self.end_date,
                prod=True,
                ok_to_return_nothing=True
            )

            if alphas_df is None or alphas_df.empty:
                logger.warning(f"No alpha data loaded for horizon {horizon}")
                self._loaded_alpha_horizons.add(horizon)
                return

            logger.info(f"Loaded alpha data for horizon {horizon}: shape {alphas_df.shape}")

            # Merge alpha columns into features_df (create if needed)
            if self.fill_breakdown is not None:
                # Get alpha columns for this horizon
                horizon_alpha_cols = [f'alpha_{model}_{horizon}' for _, model in horizon_models]
                new_alpha_cols = [col for col in horizon_alpha_cols if col in alphas_df.columns]

                if new_alpha_cols:
                    # Ensure consistent timezone handling before join
                    alpha_subset_df = alphas_df[new_alpha_cols].copy()

                    if self.fill_breakdown.features_df is not None:
                        features_df = self.fill_breakdown.features_df

                        # Normalize timezones - convert both to UTC if needed
                        features_ts = features_df.index.get_level_values('ts')
                        alpha_ts = alpha_subset_df.index.get_level_values('ts')

                        if features_ts.tz is None and alpha_ts.tz is not None:
                            # Features is tz-naive, alpha is tz-aware - localize features to UTC
                            features_df = features_df.reset_index()
                            features_df['ts'] = features_df['ts'].dt.tz_localize('UTC')
                            features_df = features_df.set_index(['ts', 'symbol_venue'])
                        elif features_ts.tz is not None and alpha_ts.tz is None:
                            # Alpha is tz-naive - localize to UTC
                            alpha_subset_df = alpha_subset_df.reset_index()
                            alpha_subset_df['ts'] = alpha_subset_df['ts'].dt.tz_localize('UTC')
                            alpha_subset_df = alpha_subset_df.set_index(['ts', 'symbol_venue'])

                        # Merge on index (ts, symbol_venue)
                        self.fill_breakdown.features_df = features_df.join(
                            alpha_subset_df,
                            how='left'
                        )
                    else:
                        # features_df is None - create from alpha data with date column
                        logger.info(f"Creating features_df from alpha data for horizon {horizon}")
                        alpha_features_df = alpha_subset_df.reset_index()
                        alpha_features_df = make_date(alpha_features_df)
                        self.fill_breakdown.features_df = alpha_features_df.set_index(['ts', 'symbol_venue'])

                    # Update available alpha columns and factors list
                    self.available_alpha_cols = list(set(self.available_alpha_cols + new_alpha_cols))
                    self.factors = self.risk_factors + self.available_alpha_cols

                    logger.info(f"Merged {len(new_alpha_cols)} alpha columns for horizon {horizon}")
                else:
                    logger.warning(f"No alpha columns found for horizon {horizon}")

            self._loaded_alpha_horizons.add(horizon)

        except Exception as e:
            import traceback
            logger.error(f"Error loading alpha data for horizon {horizon}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._loaded_alpha_horizons.add(horizon)  # Mark as loaded to prevent repeated failures

    def _get_cum_pnl_breakdowns(self, total_fills_pnl_df: pd.DataFrame, fill_breakdown_features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Get cumulative P&L breakdowns by various factors.

        Args:
            total_fills_pnl_df: Total fills PNL DataFrame
            fill_breakdown_features: Dictionary of feature DataFrames

        Returns:
            Dictionary mapping feature names to cumulative PNL breakdown DataFrames
        """
        ret = {}
        for col in PNL_BREAKDOWN_FEATURES:
            df = self._pnl_by_col(total_fills_pnl_df, fill_breakdown_features, col, by_date=True)
            if df is not None:
                df = df.unstack().fillna(0).cumsum().stack(future_stack=True).reset_index()
                ret[col] = df
        return ret

    def _pnl_by_col(self, total_fills_pnl_df: pd.DataFrame, fill_breakdown_features: Dict[str, pd.DataFrame], col: str, by_date: bool = False) -> Optional[pd.DataFrame]:
        """Calculate P&L grouped by specified column.

        Args:
            total_fills_pnl_df: Total fills PNL DataFrame
            fill_breakdown_features: Dictionary of feature DataFrames
            col: Column name to group by
            by_date: Whether to group by date as well

        Returns:
            Grouped PNL DataFrame or None if error
        """
        try:
            df = pd.merge(
                total_fills_pnl_df,
                fill_breakdown_features[col],
                on=['symbol_venue', 'date'],
                how='left'
            )
        except Exception as error:
            logger.error(error)
            return None
        df = make_quintile(df, col)
        qcol = f'{col}_quintile'
        groupby = ['date', qcol] if by_date else qcol
        df = df.groupby(groupby, observed=False).agg({'realized_pnl': 'sum'})
        if not by_date:
            df = df.reset_index().sort_values(by=qcol)
        return df

    def _calc_all_performance_metrics(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for all time periods.

        Args:
            portfolio_df: Daily PNL with balance data

        Returns:
            Dictionary containing metrics, monthly performance, and daily volumes
        """
        # Create a copy of daily PNL data for calculations
        # Use no_day_boundary_shift=True for portfolio data (timestamps at 00:00:00)
        daily_pnl = portfolio_df[['commission', 'funding_income', 'fill_dollars_abs', 'net_pnl', 'gross_notional', 'balance', 'net_pnl_cum']].reset_index()
        daily_pnl = make_date(daily_pnl, no_day_boundary_shift=True)
        daily_pnl['date_cmp'] = daily_pnl['date'].dt.date

        daily_pnl_mtd = daily_pnl.loc[daily_pnl['date_cmp'] >= self.end_date.replace(day=1)]
        daily_pnl_ytd = daily_pnl.loc[daily_pnl['date_cmp'] >= self.end_date.replace(day=1, month=1)]
        performance_dict_data = {
            'lifetime': daily_pnl,
            'mtd': daily_pnl_mtd,
            'ytd': daily_pnl_ytd
        }

        performance_dict_result = {}
        for performance_case, perf_df in performance_dict_data.items():
            del perf_df['date_cmp']
            if perf_df.empty:
                logger.warning(f"No data available for {performance_case} performance calculation")
                performance_dict_result[performance_case] = {}
                continue
            perf_df = calc_return_metrics(perf_df)
            stats = calculate_performance_statistics(perf_df, pnl_col='net_pnl')
            for k in list(stats.keys()):
                stats[f"{k}_{performance_case}"] = stats[k]
                del stats[k]
            performance_dict_result[performance_case] = stats

        # Calculate monthly performance
        monthly_perf_df = calculate_performance_by_month(portfolio_df.reset_index())

        # Calculate daily trading volume and turnover
        daily_trading_volume_df = portfolio_df[['fill_dollars_abs', 'gross_notional']].reset_index()
        daily_trading_volume_df = make_date(daily_trading_volume_df, no_day_boundary_shift=True)
        daily_trading_volume_df['turnover'] = remove_infs(daily_trading_volume_df['fill_dollars_abs'] / daily_trading_volume_df['gross_notional'])

        return {
            'metrics': {
                **performance_dict_result['lifetime'],
                **performance_dict_result['mtd'],
                **performance_dict_result['ytd']
            },
            'monthly_performance': monthly_perf_df,
            'daily_trading_volume': daily_trading_volume_df,
        }

    def _calculate_drawdowns(self) -> None:
        """Calculate top drawdown periods from daily PNL data."""
        if self.portfolio_df is None:
            return

        # Calculate cumulative PNL
        # Exclude last day — its daily bin is missing the midnight closing mark
        daily_pnl = self.portfolio_df['net_pnl'].reset_index()
        daily_pnl = make_date(daily_pnl, no_day_boundary_shift=True)
        daily_pnl = daily_pnl.iloc[:-1]
        daily_pnl['cum_pnl'] = daily_pnl['net_pnl'].cumsum()
        daily_pnl['peak'] = daily_pnl['cum_pnl'].cummax()
        daily_pnl['drawdown'] = daily_pnl['cum_pnl'] - daily_pnl['peak']

        # Find drawdown periods
        drawdowns = []
        in_drawdown = False
        dd_start = None
        dd_start_value = None
        dd_trough = None

        for idx, row in daily_pnl.iterrows():
            if row['drawdown'] < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                dd_start = row['date']
                dd_start_value = row['peak']
                dd_trough = row['cum_pnl']
            elif in_drawdown:
                dd_trough = min(dd_trough, row['cum_pnl'])
                if row['drawdown'] >= 0:
                    # End of drawdown — use peak-to-trough loss
                    in_drawdown = False
                    dd_end = daily_pnl.iloc[idx - 1]['date']
                    dd_loss = dd_trough - dd_start_value
                    dd_days = (dd_end - dd_start).days

                    # Calculate return-based drawdown
                    avg_notional = self.portfolio_df.loc[dd_start:dd_end, 'gross_notional'].mean()
                    dd_pct_loss = dd_loss / avg_notional
                    if np.isinf(dd_pct_loss):
                        dd_pct_loss = np.nan

                    drawdowns.append({
                        'start_date': dd_start,
                        'end_date': dd_end,
                        'drawdown_days': dd_days,
                        'dollar_loss': dd_loss,
                        'percent_loss': dd_pct_loss,
                    })

        # Handle ongoing drawdown at the end of data
        if in_drawdown:
            dd_end = daily_pnl.iloc[-1]['date']
            dd_loss = dd_trough - dd_start_value
            dd_days = (dd_end - dd_start).days

            # Calculate return-based drawdown
            avg_notional = self.portfolio_df.loc[dd_start:dd_end, 'gross_notional'].mean()
            dd_pct_loss = dd_loss / avg_notional
            if np.isinf(dd_pct_loss):
                dd_pct_loss = np.nan

            drawdowns.append({
                'start_date': dd_start,
                'end_date': dd_end,
                'drawdown_days': dd_days,
                'dollar_loss': dd_loss,
                'percent_loss': dd_pct_loss,
            })
            logger.debug(f"Added ongoing drawdown: {dd_start.strftime('%Y-%m-%d')} to {dd_end.strftime('%Y-%m-%d')}, loss: ${dd_loss:,.2f}")

        # Convert to DataFrame and sort by dollar loss
        if drawdowns:
            self.hist_top_drawdowns = pd.DataFrame(drawdowns)
            self.hist_top_drawdowns = self.hist_top_drawdowns.sort_values('dollar_loss').head(10)

            # Format dates as strings (YYYY-MM-DD) for display
            self.hist_top_drawdowns['start_date'] = self.hist_top_drawdowns['start_date'].dt.strftime('%Y-%m-%d')
            self.hist_top_drawdowns['end_date'] = self.hist_top_drawdowns['end_date'].dt.strftime('%Y-%m-%d')

            logger.debug(f"Found {len(drawdowns)} drawdown periods, showing top {len(self.hist_top_drawdowns)}")
        else:
            self.hist_top_drawdowns = pd.DataFrame()
            logger.debug("No drawdown periods found (all positive PnL)")

    def _load_alpha_opt_stats(self) -> None:
        """Load and cache alpha_opt statistics for timeseries chart.

        Loads alpha_opt data and calculates daily mean(abs(alpha_opt)) and std(alpha_opt).
        Results are cached in self.alpha_opt_stats_df for use by alpha_opt_timeseries_figure().
        """
        logger.info(f"Loading alpha_opt data from {self.start_date} to {self.end_date}...")

        alpha_df = load_raw_targets_alpha(
            start_dt=date_to_start_dt(self.start_date),
            end_dt=date_to_end_dt(self.end_date),
            cols=['alpha_opt'],
            skip_log=True
        )

        if alpha_df is None or alpha_df.empty:
            logger.warning("No alpha_opt data available for timeseries chart")
            self.alpha_opt_stats_df = pd.DataFrame()
            return

        if 'alpha_opt' not in alpha_df.columns:
            logger.warning("alpha_opt column not found in data")
            self.alpha_opt_stats_df = pd.DataFrame()
            return

        # Reset index to get ts as a column
        alpha_df = alpha_df.reset_index()

        if 'ts' not in alpha_df.columns:
            logger.warning("ts column not found in alpha data")
            self.alpha_opt_stats_df = pd.DataFrame()
            return

        # Group by date (aggregate intraday data to daily)
        alpha_df['date'] = pd.to_datetime(alpha_df['ts']).dt.date

        # Drop NaN values before aggregation
        alpha_df = alpha_df.dropna(subset=['alpha_opt'])

        if alpha_df.empty:
            logger.warning("No non-NaN alpha_opt data available")
            self.alpha_opt_stats_df = pd.DataFrame()
            return

        # Calculate daily statistics: mean(abs(alpha_opt)) and std(alpha_opt)
        daily_stats_df = alpha_df.groupby('date').agg(
            mean_abs_alpha=('alpha_opt', lambda x: np.abs(x).mean()),
            std_alpha=('alpha_opt', 'std')
        ).reset_index()

        # Fill NaN std values (single-value days) with 0
        daily_stats_df['std_alpha'] = daily_stats_df['std_alpha'].fillna(0)

        # Convert to bps for display (alpha_opt is typically in decimal form)
        daily_stats_df['mean_abs_alpha_bps'] = daily_stats_df['mean_abs_alpha'] * 10000
        daily_stats_df['std_alpha_bps'] = daily_stats_df['std_alpha'] * 10000

        self.alpha_opt_stats_df = daily_stats_df
        logger.info(f"Loaded alpha_opt stats for {len(daily_stats_df)} days")

    def get_ts_display(self, n_state: str) -> str:
        """Get timestamp display text.

        Args:
            n_state: State identifier (for logging)

        Returns:
            Formatted timestamp string
        """
        logger.info(f"finish get_ts_display for update {n_state} at {dt.now(timezone.utc)}")
        update_ts_str = self.update_ts.strftime("%Y%m%d %H:%M") if self.update_ts else 'Unknown'
        return (
            f'Data as of: {update_ts_str}, '
            f'dashboard refreshed at {dt.now(timezone.utc).strftime("%Y%m%d %H:%M")}'
        )

    def update_fill_breakdown_dropdown(self, n_state: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """Get options for fill breakdown dropdown.

        Args:
            n_state: State identifier (for logging)

        Returns:
            Tuple of (dropdown options, default value)
        """
        options = [
            {'label': breakdown, 'value': breakdown}
            for breakdown in self.hist_fill_breakdown_pnl_keys
        ]
        value = self.hist_fill_breakdown_pnl_keys[-1] if self.hist_fill_breakdown_pnl_keys else None
        logger.info(
            f"finish update_fill_breakdown_dropdown for update {n_state} "
            f"at {dt.now(timezone.utc)}"
        )
        return options, value

    def update_return_breakdown_figure(self, _: Any) -> Tuple[go.Figure, go.Figure]:
        """Create return breakdown figures by day of week and hour.

        Args:
            _: Unused parameter (for callback compatibility)

        Returns:
            Tuple of (day of week figure, hour of day figure)
        """
        if self.security_ts_df is None:
            return go.Figure(), go.Figure()

        # Aggregate security-level data to portfolio level by timestamp
        hist_pnl_df = self.security_ts_df.reset_index().groupby('ts').agg({
            'net_pnl': 'sum',
            'notional': 'sum'
        }).reset_index()

        if hist_pnl_df.empty:
            return go.Figure(), go.Figure()

        # Sort by timestamp to ensure diff is calculated in chronological order
        hist_pnl_df = hist_pnl_df.sort_values('ts')

        # Calculate period-over-period return at portfolio level
        hist_pnl_df['pnl_diff'] = hist_pnl_df['net_pnl'].diff().fillna(hist_pnl_df['net_pnl'])
        safe_notional = hist_pnl_df['notional'].replace(0, np.nan)
        hist_pnl_df['period_return'] = (hist_pnl_df['pnl_diff'] / safe_notional).fillna(0)
        # Note: clip_col_by_iqr removed - it was clipping valid returns to near-zero
        # because most periods have 0 return, making IQR ≈ 0
        hist_pnl_df['hour_of_day'] = hist_pnl_df['ts'].dt.hour
        hist_pnl_df['day_of_week'] = hist_pnl_df['ts'].dt.day_name()

        # Day of week figure
        fig_dow = go.Figure()
        dow_avg = hist_pnl_df.groupby('day_of_week')['period_return'].mean() * 10000
        dow_avg = dow_avg.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        fig_dow.add_trace(go.Bar(x=dow_avg.index, y=dow_avg.values))
        fig_dow.update_layout(
            title='Average Return by Day of Week',
            xaxis_title='Day of Week',
            yaxis_title='Average Return Bps',
            barmode='group',
        )

        # Hour of day figure
        fig_hod = go.Figure()
        all_hours = pd.Series(index=range(24), data=0.0)
        hod_avg = hist_pnl_df.groupby('hour_of_day')['period_return'].mean() * 10000
        all_hours.update(hod_avg)
        fig_hod.add_trace(go.Bar(x=all_hours.index, y=all_hours.values))
        fig_hod.update_layout(
            title='Average Return by Hour of Day',
            xaxis_title='Hour of Day',
            yaxis_title='Average Return Bps',
            barmode='group',
            xaxis={
                "tickmode": 'linear',
                "tick0": 0,
                "dtick": 1,
                "tickvals": list(range(24)),
                "ticktext": [f"{h}" for h in range(24)],
            },
        )

        return fig_dow, fig_hod

    def get_overall_perf_display(self, n_state: str) -> List[Dict[str, Any]]:
        """Get performance metrics table data.

        Args:
            n_state: State identifier (for logging)

        Returns:
            List of dictionaries for performance table
        """
        # Use the centrally calculated performance metrics
        if 'metrics' not in self.performance_metrics:
            return []

        metrics = self.performance_metrics['metrics']
        win_ratios = self.win_ratios
        res = {
            'metrics': [
                'Start date',
                'Cumulative pnl',
                'Cumulative unlevered return',
                'Cumulative levered return',
                'Annualized unlevered return (Pnl Base)',
                'Annualized levered return (Pnl Base)',
                'Annualized unlevered return (Return Base)',
                'Annualized levered return (Return Base)',
                'Annualized risk',
                'Annualized sharpe (Pnl Base)',
                'Annualized sharpe (Return Base)',
                'Cumulative funding income',
                'Avg funding income',
                'Avg funding income bps',
                'Win ratio',
                'Total trades',
                'Profitable trades',
                "Avg gain per trade",
                "Avg loss per trade",
                'Avg trading volume',
                'Avg turnover',
                'Avg trading fees',
                'Avg trading fees bps',
            ],
        }

        for case in ['mtd', 'ytd', 'lifetime']:
            # Skip cases with no data (e.g., MTD on 1st of month before any trading)
            if f'start_dt_{case}' not in metrics:
                res[case] = ['N/A'] * len(res['metrics'])
                continue
            res[case] = [
                f"{metrics[f'start_dt_{case}']}",
                fmoney(metrics[f'cum_pnl_{case}']),
                f"{metrics[f'cum_unlev_ret_{case}'] * 100:.2f}%",
                f"{metrics[f'cum_lev_ret_{case}'] * 100:.2f}%",
                f"{metrics[f'annualized_unlev_ret_{case}'] * 100:.2f}%",
                f"{metrics[f'annualized_lev_ret_{case}'] * 100:.2f}%",
                f"{metrics[f'annualized_unlev_ret_from_ret_{case}'] * 100:.2f}%",
                f"{metrics[f'annualized_lev_ret_from_ret_{case}'] * 100:.2f}%",
                f"{metrics[f'annualized_risk_{case}'] * 100:.2f}%",
                f"{metrics[f'annualized_sharpe_{case}']:.1f}",
                f"{metrics[f'annualized_sharpe_from_ret_{case}']:.1f}",
                fmoney(metrics[f'cum_fundings_income_{case}']),
                fmoney(metrics[f'avg_fundings_income_{case}']),
                f"{metrics[f'avg_fundings_income_bps_{case}']:.2f}",
                f"{win_ratios[case][0] * 100:.2f}%",
                f"{win_ratios[case][3]}",
                f"{win_ratios[case][1]} / {win_ratios[case][2]}",
                fmoney(win_ratios[case][4]),
                fmoney(win_ratios[case][5]),
                fmoney(metrics[f'volume_{case}']),
                f"{metrics[f'turnover_{case}']:.2f}",
                fmoney(metrics[f'fees_{case}']),
                f"{metrics[f'fees_bps_{case}']:.2f}",
            ]

        data = [dict(zip(res, t)) for t in zip(*res.values())]
        logger.info(
            f"finish get_overall_perf_display for update {n_state} at {dt.now(timezone.utc)}"
        )
        return data

    def get_monthly_summary_table(self, _n_state: str) -> List[Dict[str, Any]]:
        """Get monthly performance summary table data.

        Args:
            _n_state: State identifier (unused, for callback compatibility)

        Returns:
            List of dictionaries for monthly summary table
        """
        if not self.performance_metrics or 'monthly_performance' not in self.performance_metrics:
            return []
        return self.performance_metrics['monthly_performance'].to_dict('records')

    def get_drawdown_table(self, _n_state: str) -> List[Dict[str, Any]]:
        """Get top drawdowns table data.

        Args:
            _n_state: State identifier (unused, for callback compatibility)

        Returns:
            List of dictionaries for drawdown table
        """
        if self.hist_top_drawdowns is None or self.hist_top_drawdowns.empty:
            return []
        return self.hist_top_drawdowns.to_dict('records')

    def overall_hist_figure(self, _n_state: str) -> go.Figure:
        """Create overall historical PNL and utility figures.

        Args:
            _n_state: State identifier (unused, for callback compatibility)

        Returns:
            Tuple of (PNL figure, utility figure)
        """
        if self.portfolio_df is None:
            return go.Figure()

        daily_pnl_df = self.portfolio_df[['net_pnl']].copy()
        new_index = daily_pnl_df.index[0] - td(days=1)
        init_date_row = pd.DataFrame([[0]], columns=daily_pnl_df.columns, index=[new_index])
        daily_pnl_df = concat([init_date_row, daily_pnl_df])
        daily_pnl_df = daily_pnl_df.cumsum().stack().rename('pnl').reset_index()
        daily_pnl_df = daily_pnl_df.rename(columns={'level_0': 'date', 'level_1': 'pnl_type'})
        # Set template explicitly to avoid plotly template corruption bug
        pnl_figure = px.line(
            daily_pnl_df, x='date', y='pnl', title="Historical Daily Pnl",
            color='pnl_type', template='plotly'
        )
        return pnl_figure

    def hist_pnl_breakdown_figure(self, value: str, _n_state: str) -> go.Figure:
        """Create PNL breakdown figure for specified feature.

        Args:
            value: Feature name to breakdown by
            _n_state: State identifier (unused, for callback compatibility)

        Returns:
            PNL breakdown figure
        """
        if not value:
            return go.Figure()

        # Lazy load the PNL breakdown for this feature
        self._load_pnl_breakdown_feature(value)

        if value not in self.hist_fill_breakdown_pnl:
            return go.Figure()
        df = self.hist_fill_breakdown_pnl[value]
        full_col = f"{value}_quintile"
        df = df.copy()
        df[full_col] = df[full_col].astype(str)
        return px.line(df, x='date', y='realized_pnl', color=full_col, title=f"Realized Pnl By {value}", template='plotly')

    def today_positions_figure(self, _n_state: str) -> go.Figure:
        """Create positions over time figure.

        Args:
            _n_state: State identifier (unused, for callback compatibility)

        Returns:
            Positions figure showing longs, shorts, and bias
        """
        if self.portfolio_df is None:
            return go.Figure()

        df = self.portfolio_df[['long_notional', 'short_notional']]
        df['bias'] = df['long_notional'] + df['short_notional']
        df = df.stack().rename('position').reset_index()
        df = df.rename(columns={'level_1': 'side'})
        return px.line(df, x='ts', y='position', color='side', title="Longs v. Shorts & Bias")

    def cum_pnl_hist_figure(self, _n_state: str) -> Tuple[go.Figure, go.Figure]:
        """Create cumulative PNL figures by symbol and trades.

        Args:
            _n_state: State identifier (unused, for callback compatibility)

        Returns:
            Tuple of (PNL by symbol figure, PNL histogram figure)
        """
        df = self.security_df.groupby(['date', 'symbol_venue']).agg({'realized_pnl': 'last'})
        df = make_symbol(df)
        df = df.groupby('symbol').agg({'realized_pnl': 'sum'}).reset_index()
        df = df.sort_values(by='realized_pnl')
        df = df.loc[df['realized_pnl'] != 0]
        fig_symbol = px.bar(df, x='symbol', y='realized_pnl', title="Cum. Realized Pnl By Symbol")

        if not self.pnl_calculator:
            return go.Figure(), go.Figure()

        contracting_fills_df = self.fills_df.loc[~self.fills_df['expanding']]
        fig_trades = px.histogram(
            contracting_fills_df,
            x='realized_pnl',
            nbins=1000,
            labels={'realized_pnl': 'Realized PnL'},
        )
        fig_trades.update_layout(
            title="Histogram of Historical Realized PnL",
            xaxis_title="Realized PnL",
            yaxis_title="Frequency"
        )
        return fig_symbol, fig_trades

    def daily_update_figures(self, n_state: str) -> Tuple[go.Figure, go.Figure, go.Figure]:
        """Create daily update figures for volume, turnover, and balance.

        Args:
            n_state: State identifier (for logging)

        Returns:
            Tuple of (volume figure, turnover figure, balance figure)
        """
        if 'daily_trading_volume' not in self.performance_metrics:
            return go.Figure(), go.Figure(), go.Figure()

        volume_df = self.performance_metrics['daily_trading_volume']
        volume_fig = px.line(volume_df, x='date', y='fill_dollars_abs', title="Daily Trading Volume")
        turnover_fig = px.line(volume_df, x='date', y='turnover', title="Daily Turnover")
        balance_df = self.portfolio_df[['balance']].reset_index()
        balance_df = make_date(balance_df, no_day_boundary_shift=True)
        balances_fig = px.line(balance_df, x='date', y='balance', title="Binance Capital Balance")
        logger.info(f"finish daily_turnover_figure for update {n_state} at {dt.now(timezone.utc)}")
        return volume_fig, turnover_fig, balances_fig

    def update_factor_return_figure(self, factor: str) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
        """Create factor return analysis figures.

        Args:
            factor: Factor name to analyze

        Returns:
            Tuple of (factor return, portfolio exposure, portfolio return, portfolio PNL figures)
        """
        factor_figs = {}
        if self.security_df is None:
            return go.Figure(), go.Figure(), go.Figure(), go.Figure()

        # Lazy load data for this factor
        if factor.startswith('alpha_'):
            # Alpha factor - extract horizon and load that horizon's alphas
            parts = factor.split('_')
            if len(parts) >= 3:
                try:
                    horizon = int(parts[-1])
                    self._load_alpha_data_for_horizon(horizon)
                except ValueError:
                    pass
        elif factor != 'dollar_exposure':
            # Non-alpha factor - lazy load this feature
            self._load_features_for_factor(factor)

        df = self.security_df.copy()
        df['ret'] = df['net_pnl_return']
        df['position'] = df['notional']

        # Merge factor data from features_df if factor is not already in df
        # dollar_exposure is a special case handled in calc_factor_return
        needs_factor_data = factor != 'dollar_exposure' and factor not in df.columns
        if needs_factor_data:
            if self.fill_breakdown is not None and self.fill_breakdown.features_df is not None:
                features_df = self.fill_breakdown.features_df
                if factor in features_df.columns:
                    # Both dataframes share the same index ['ts', 'symbol_venue']
                    df = df.join(features_df[[factor]], how='left')

            # Return empty figures if factor is still not available after merge attempt
            if factor not in df.columns:
                logger.warning(f"Factor '{factor}' not found in security_df or features_df")
                return go.Figure(), go.Figure(), go.Figure(), go.Figure()

        factor_ret_df, port_factor_exposure_df, port_factor_ret_df, port_factor_pnl_df = calc_factor_return(df, factor)
        factor_fig_data = {
            'Factor Return': factor_ret_df,
            'Factor Portfolio Exposure': port_factor_exposure_df,
            'Factor Portfolio Return': port_factor_ret_df,
            'Factor Portfolio Pnl': port_factor_pnl_df,
        }
        for fig_case, data_df in factor_fig_data.items():
            data_df = make_date(data_df)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data_df['date'],
                y=data_df[factor],
                mode='lines',
                name=factor.capitalize(),
            ))
            fig.update_layout(
                title=f"{factor.capitalize()} {fig_case} over Time",
                xaxis_title="Timestamp",
                yaxis_title=factor.capitalize(),
            )
            factor_figs[fig_case] = fig

        return (
            factor_figs['Factor Return'],
            factor_figs['Factor Portfolio Exposure'],
            factor_figs['Factor Portfolio Return'],
            factor_figs['Factor Portfolio Pnl']
        )

    def alpha_opt_timeseries_figure(self, _n_state: str) -> go.Figure:
        """Create alpha_opt statistics timeseries figure.

        Shows mean(abs(alpha_opt)) and std(alpha_opt) over time to visualize
        the evolution of total alpha strength and standard deviation.

        Args:
            _n_state: State identifier (unused, for callback compatibility)

        Returns:
            Figure with two traces: mean absolute alpha and std alpha over time
        """
        if self.alpha_opt_stats_df is None or self.alpha_opt_stats_df.empty:
            logger.warning("No alpha_opt stats data available")
            return go.Figure()

        daily_stats_df = self.alpha_opt_stats_df

        # Create figure with two y-axes
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=daily_stats_df['date'],
            y=daily_stats_df['mean_abs_alpha_bps'],
            mode='lines',
            name='Mean |Alpha| (bps)',
            line=dict(color='blue'),
        ))

        fig.add_trace(go.Scatter(
            x=daily_stats_df['date'],
            y=daily_stats_df['std_alpha_bps'],
            mode='lines',
            name='Std Alpha (bps)',
            line=dict(color='orange'),
            yaxis='y2',
        ))

        fig.update_layout(
            title='Alpha Opt Statistics Over Time',
            xaxis_title='Date',
            yaxis_title='Mean |Alpha| (bps)',
            yaxis2=dict(
                title='Std Alpha (bps)',
                anchor='x',
                overlaying='y',
                side='right',
            ),
        )

        logger.info(f"Created alpha_opt timeseries chart with {len(daily_stats_df)} days of data")
        return fig

    def get_alphas_for_horizon(self, horizon: int) -> List[str]:
        """Get list of available alpha columns for a specific horizon.

        Args:
            horizon: The horizon to filter alphas by (e.g., 1440, 720)

        Returns:
            List of alpha column names for the horizon (e.g., ['alpha_hl_1440', 'alpha_c2vwap_1440'])
        """
        horizon_suffix = f'_{horizon}'
        return [col for col in self.available_alpha_cols if col.endswith(horizon_suffix)]

    def update_alpha_by_horizon_figures(self, horizon: int) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
        """Create alpha analysis figures for all alphas at a given horizon.

        Shows all alpha models at the specified horizon as separate lines on the same charts.

        Args:
            horizon: The horizon to analyze (e.g., 1440, 720)

        Returns:
            Tuple of (factor return, portfolio exposure, portfolio return, portfolio PNL figures)
        """
        if self.security_df is None or horizon is None:
            return go.Figure(), go.Figure(), go.Figure(), go.Figure()

        # Lazy load alpha data for this specific horizon only
        self._load_alpha_data_for_horizon(horizon)

        alpha_cols = self.get_alphas_for_horizon(horizon)
        if not alpha_cols:
            logger.warning(f"No alpha columns found for horizon {horizon}")
            return go.Figure(), go.Figure(), go.Figure(), go.Figure()

        df = self.security_df.copy()
        df['ret'] = df['net_pnl_return']
        df['position'] = df['notional']

        # Merge all alpha columns from features_df
        if self.fill_breakdown is not None and self.fill_breakdown.features_df is not None:
            features_df = self.fill_breakdown.features_df
            cols_to_merge = [col for col in alpha_cols if col in features_df.columns]
            if cols_to_merge:
                df = df.join(features_df[cols_to_merge], how='left')

        # Initialize figures
        fig_names = ['Factor Return', 'Factor Portfolio Exposure', 'Factor Portfolio Return', 'Factor Portfolio Pnl']
        figs = {name: go.Figure() for name in fig_names}

        # Calculate and plot each alpha
        for alpha_col in alpha_cols:
            if alpha_col not in df.columns:
                continue

            # Extract model name from alpha column (e.g., 'hl' from 'alpha_hl_1440')
            parts = alpha_col.split('_')
            model_name = parts[1] if len(parts) >= 3 else alpha_col

            try:
                factor_ret_df, port_factor_exposure_df, port_factor_ret_df, port_factor_pnl_df = calc_factor_return(
                    df, alpha_col
                )

                data_map = {
                    'Factor Return': (factor_ret_df, alpha_col),
                    'Factor Portfolio Exposure': (port_factor_exposure_df, alpha_col),
                    'Factor Portfolio Return': (port_factor_ret_df, alpha_col),
                    'Factor Portfolio Pnl': (port_factor_pnl_df, alpha_col),
                }

                for fig_name, (data_df, col) in data_map.items():
                    data_df = make_date(data_df)
                    figs[fig_name].add_trace(go.Scatter(
                        x=data_df['date'],
                        y=data_df[col],
                        mode='lines',
                        name=model_name,
                    ))
            except Exception as e:
                logger.warning(f"Error calculating factor return for {alpha_col}: {e}")
                continue

        # Update layout for all figures
        for fig_name, fig in figs.items():
            fig.update_layout(
                title=f"Alpha {fig_name} - Horizon {horizon}",
                xaxis_title="Date",
                yaxis_title=fig_name,
                legend_title="Model",
                hovermode='x unified',
            )

        return (
            figs['Factor Return'],
            figs['Factor Portfolio Exposure'],
            figs['Factor Portfolio Return'],
            figs['Factor Portfolio Pnl']
        )
