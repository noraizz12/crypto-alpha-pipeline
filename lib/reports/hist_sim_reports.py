"""Historical simulation performance analysis and reporting module.

This module provides comprehensive long-term simulation performance reporting including
P&L analysis, drawdown tracking, monthly/yearly summaries, and risk-adjusted
performance metrics. Designed to analyze multi-month simulation runs.

Adapted from simulation_report.py for long-term historical analysis.
Reuses calc_return_metrics from lib/sim/sim_util.py for shared code.
"""

import json
import logging
import os
from datetime import datetime as dt, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.data import load_sim_data
from lib.sim.sim_util import calc_return_metrics
from lib.util import get_sim_dirs
from lib.util.dataframes import make_date, remove_infs
from lib.util.directory import SIM_DIR

logger = logging.getLogger(__name__)


class HistSimReports:
    """Reports for historical simulation performance analysis.

    Adapted from SimulationReport for long-term historical analysis.
    Reuses calc_return_metrics from lib/sim/sim_util.py.
    """

    def __init__(self, sim_name: str, sim_dir: Optional[str] = None):
        """Initialize with simulation name.

        Args:
            sim_name: Name of simulation directory
            sim_dir: Optional custom simulation directory path
        """
        self.sim_name = sim_name
        self.sim_base_dir = sim_dir if sim_dir is not None else SIM_DIR
        self.sim_dir = os.path.join(self.sim_base_dir, sim_name)

        # Initialize data attributes (same pattern as simulation_report.py)
        self.sim_df: Optional[pd.DataFrame] = None
        self.aggregate_df: Optional[pd.DataFrame] = None
        self.daily_df: Optional[pd.DataFrame] = None
        self.monthly_df: Optional[pd.DataFrame] = None
        self.yearly_df: Optional[pd.DataFrame] = None
        self.return_metrics: Dict[str, float] = {}
        self.hist_top_drawdowns: Optional[pd.DataFrame] = None
        self.rolling_sharpe_df: Optional[pd.DataFrame] = None
        self.update_ts: Optional[dt] = None

        # Load data on initialization
        self.load_data()

    @classmethod
    def get_available_simulations(cls, sim_dir: Optional[str] = None) -> List[str]:
        """Get list of available simulation directories.

        Scans top-level sims/ directory plus long/ and simcomp/ subdirectories.
        """
        base_dir = sim_dir if sim_dir is not None else SIM_DIR

        # Get top-level sims
        sim_names = get_sim_dirs(base_dir)

        # Also scan long/ and simcomp/ subdirectories
        subdirs = ['long', 'simcomp']
        for subdir in subdirs:
            subdir_path = os.path.join(base_dir, subdir)
            if os.path.isdir(subdir_path):
                sub_sims = get_sim_dirs(subdir_path)
                # Prefix with subdir name for unique identification
                sim_names.extend([f"{subdir}/{s}" for s in sub_sims])

        # Sort by modification time (newest first)
        def get_mtime(name: str) -> float:
            full_path = os.path.join(base_dir, name)
            try:
                return os.path.getmtime(full_path)
            except OSError:
                return 0

        sim_names = sorted(sim_names, key=get_mtime, reverse=True)
        return sim_names

    def load_data(self) -> None:
        """Load and process all simulation data.

        Similar to simulation_report.py load_simulation_data.
        """
        logger.info("Loading simulation data from %s...", self.sim_dir)

        # Clear existing data (same pattern as simulation_report.py)
        self.sim_df = None
        self.aggregate_df = None
        self.daily_df = None

        # Load raw simulation data using shared loader
        self.sim_df = load_sim_data(self.sim_name, sim_dir=self.sim_base_dir)
        if self.sim_df is None or self.sim_df.empty:
            logger.error("No simulation data found in %s", self.sim_dir)
            return

        logger.info("Loaded %d simulation records", len(self.sim_df))

        # Aggregate to timestamp level (same as simulation_report.py)
        self._aggregate_to_timestamp()

        # Aggregate to daily (same as simulation_report.py)
        self._aggregate_to_daily()

        # Calculate metrics using shared calc_return_metrics
        self._calculate_metrics()

        # Aggregate to monthly and yearly
        self._aggregate_to_monthly()
        self._aggregate_to_yearly()

        # Calculate drawdowns
        self._calculate_drawdowns()

        # Calculate rolling Sharpe
        self._calculate_rolling_sharpe()

        # Update timestamp
        self.update_ts = dt.now(timezone.utc)
        logger.info("Simulation data loaded successfully")

    def _aggregate_to_timestamp(self) -> None:
        """Aggregate simulation data by timestamp.

        Same aggregation logic as simulation_report.py.
        """
        if self.sim_df is None:
            return

        df = self.sim_df.copy()
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        if 'ts' not in df.columns:
            logger.error("No 'ts' column in simulation data")
            return

        # Same aggregation as simulation_report.py
        self.aggregate_df = df.groupby('ts').agg({
            'pnl': 'sum',
            'position': 'sum',
            'qty': lambda x: (x != 0).sum(),
            'executed_dollars': lambda x: x.abs().sum(),
            'fees': 'sum',
            'funding_income': 'sum',
        }).reset_index()

        # Add calculated columns (same names as simulation_report.py)
        # pnl is already cumulative from load_sim_data, no cumsum needed
        self.aggregate_df['cumulative_pnl'] = self.aggregate_df['pnl']
        self.aggregate_df['position_count'] = self.aggregate_df['qty']
        self.aggregate_df['net_exposure'] = self.aggregate_df['position']
        self.aggregate_df['dollars_traded_daily'] = self.aggregate_df['executed_dollars']
        self.aggregate_df['cumulative_fees'] = self.aggregate_df['fees'].cumsum()
        self.aggregate_df['cumulative_funding'] = self.aggregate_df['funding_income'].cumsum()
        self.aggregate_df['total_pnl_daily'] = self.aggregate_df['pnl'].diff().fillna(
            self.aggregate_df['pnl'].iloc[0]
        )
        self.aggregate_df['fees_usd_daily'] = self.aggregate_df['fees']
        self.aggregate_df['funding_income_daily'] = self.aggregate_df['funding_income']

        logger.info("Aggregated to %d timestamp records", len(self.aggregate_df))

    def _aggregate_to_daily(self) -> None:
        """Aggregate timestamp data to daily level.

        Same pattern as simulation_report.py.
        """
        if self.aggregate_df is None or self.aggregate_df.empty:
            return

        df = self.aggregate_df.copy()
        df = make_date(df)

        # Calculate gross exposure from raw sim data
        # First sum abs positions per timestamp, then average per day
        sim_df = self.sim_df.copy()
        if isinstance(sim_df.index, pd.MultiIndex):
            sim_df = sim_df.reset_index()
        sim_df = make_date(sim_df)
        # Get gross exposure per timestamp (sum of absolute positions across symbols)
        ts_gross_df = sim_df.groupby(['date', 'ts'])['position'].apply(
            lambda x: x.abs().sum()
        ).reset_index(name='gross_notional')
        # Average gross exposure across timestamps within each day
        gross_exposure_df = ts_gross_df.groupby('date')['gross_notional'].mean().reset_index()

        # Daily aggregation (same as simulation_report.py)
        self.daily_df = df.groupby('date').agg({
            'cumulative_pnl': 'last',
            'position_count': 'mean',
            'net_exposure': 'mean',
            'dollars_traded_daily': 'sum',
            'fees_usd_daily': 'sum',
            'funding_income_daily': 'sum',
            'cumulative_fees': 'last',
            'cumulative_funding': 'last',
        }).reset_index()

        # Merge gross notional
        self.daily_df = self.daily_df.merge(gross_exposure_df, on='date', how='left')

        # Calculate daily P&L
        self.daily_df['daily_pnl'] = self.daily_df['cumulative_pnl'].diff().fillna(
            self.daily_df['cumulative_pnl'].iloc[0]
        )

        # Calculate daily returns (same as simulation_report.py)
        self.daily_df['prev_gross_notional'] = self.daily_df['gross_notional'].shift(1)
        self.daily_df['daily_return'] = remove_infs(
            self.daily_df['daily_pnl'] / self.daily_df['prev_gross_notional']
        )

        # Calculate turnover
        self.daily_df['turnover'] = remove_infs(
            self.daily_df['dollars_traded_daily'] / self.daily_df['gross_notional']
        )

        logger.info("Aggregated to %d daily records", len(self.daily_df))

    def _calculate_metrics(self) -> None:
        """Calculate metrics using calc_return_metrics from lib/sim/sim_util.py.

        Uses pnl.calculator.csv file which has the correct format for calc_return_metrics.
        Same approach as sim_reports.py.
        """
        if self.daily_df is None or self.daily_df.empty:
            return

        # Try to load pnl.calculator.csv for accurate metrics (same as sim_reports.py)
        calculator_csv = os.path.join(self.sim_dir, 'pnl.calculator.csv')
        if os.path.exists(calculator_csv):
            try:
                pnl_df = pd.read_csv(calculator_csv, index_col=0)
                # Ensure required columns exist
                required_cols = ['pnl', 'long', 'short', 'traded_long', 'traded_short', 'fees_usd']
                if all(col in pnl_df.columns for col in required_cols):
                    # Add funding_income if missing
                    if 'funding_income' not in pnl_df.columns:
                        pnl_df['funding_income'] = 0

                    # Calculate daily_scaler from config (same as sim_reports.py)
                    # Default 360 mins = 4 periods per day
                    config_path = os.path.join(self.sim_dir, 'config.json')
                    reoptimize_mins = 360
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                                reoptimize_mins = config.get('REOPTIMIZE_INTERVAL_MINS', 360)
                        except (json.JSONDecodeError, IOError) as exc:
                            logger.warning(
                                "Failed to read config %s: %s. Using default REOPTIMIZE_INTERVAL_MINS=%d",
                                config_path, exc, reoptimize_mins
                            )
                    daily_scaler = 1440 / reoptimize_mins

                    self.return_metrics = calc_return_metrics(pnl_df, daily_scaler=daily_scaler)
                    logger.info("Calculated metrics from pnl.calculator.csv: Sharpe=%.2f",
                                self.return_metrics.get('annualized_sharpe', 0))
                else:
                    logger.warning("pnl.calculator.csv missing required columns, using fallback")
                    self._calculate_metrics_from_daily()
            except (pd.errors.ParserError, IOError) as e:
                logger.warning("Failed to read pnl.calculator.csv: %s, using fallback", e)
                self._calculate_metrics_from_daily()
        else:
            logger.info("No pnl.calculator.csv found, using daily data for metrics")
            self._calculate_metrics_from_daily()

        if self.return_metrics:
            df = self.daily_df
            self.return_metrics['start_date'] = df['date'].iloc[0]
            self.return_metrics['end_date'] = df['date'].iloc[-1]
            self.return_metrics['total_days'] = len(df)

    def _calculate_metrics_from_daily(self) -> None:
        """Calculate metrics from daily aggregated data as fallback."""
        df = self.daily_df.copy()

        # Filter out days with small exposure
        min_exposure = 1000
        df_valid = df[df['prev_gross_notional'] > min_exposure].copy()

        if len(df_valid) < 2:
            logger.warning("Not enough valid data for metrics calculation")
            self.return_metrics = {}
            return

        self._calculate_metrics_manual(df_valid)

    def _calculate_metrics_manual(self, df: pd.DataFrame) -> None:
        """Manual metrics calculation as fallback."""
        daily_returns = df['daily_return'].dropna()
        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()

        trading_days = 365
        annualized_return = mean_daily_return * trading_days
        annualized_risk = std_daily_return * np.sqrt(trading_days)
        annualized_sharpe = annualized_return / annualized_risk if annualized_risk > 0 else 0

        self.return_metrics = {
            'cum_pnl': df['cumulative_pnl'].iloc[-1],
            'cum_fees': df['cumulative_fees'].iloc[-1],
            'cum_funding': df['cumulative_funding'].iloc[-1],
            'avg_notional': df['gross_notional'].mean(),
            'avg_trading_volume': df['dollars_traded_daily'].mean(),
            'annualized_ret': annualized_return,
            'annualized_risk': annualized_risk,
            'annualized_sharpe': annualized_sharpe,
            'daily_turnover': df['turnover'].mean(),
        }

    def _aggregate_to_monthly(self) -> None:
        """Aggregate daily data to monthly level."""
        if self.daily_df is None or self.daily_df.empty:
            return

        df = self.daily_df.copy()
        df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M')

        self.monthly_df = df.groupby('year_month').agg({
            'daily_pnl': 'sum',
            'cumulative_pnl': 'last',
            'position_count': 'mean',
            'gross_notional': 'mean',
            'dollars_traded_daily': 'sum',
            'fees_usd_daily': 'sum',
            'funding_income_daily': 'sum',
            'daily_return': ['mean', 'std'],
        }).reset_index()

        self.monthly_df.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) else col
            for col in self.monthly_df.columns
        ]

        self.monthly_df['monthly_sharpe'] = remove_infs(
            self.monthly_df['daily_return_mean'] / self.monthly_df['daily_return_std']
        ) * np.sqrt(30)

        self.monthly_df['year_month'] = self.monthly_df['year_month'].astype(str)
        logger.info("Aggregated to %d monthly records", len(self.monthly_df))

    def _aggregate_to_yearly(self) -> None:
        """Aggregate daily data to yearly level."""
        if self.daily_df is None or self.daily_df.empty:
            return

        df = self.daily_df.copy()
        df['year'] = pd.to_datetime(df['date']).dt.year

        self.yearly_df = df.groupby('year').agg({
            'daily_pnl': 'sum',
            'cumulative_pnl': 'last',
            'gross_notional': 'mean',
            'dollars_traded_daily': 'sum',
            'fees_usd_daily': 'sum',
            'funding_income_daily': 'sum',
            'daily_return': ['mean', 'std', 'count'],
        }).reset_index()

        self.yearly_df.columns = [
            '_'.join(str(c) for c in col).strip('_') if isinstance(col, tuple) else col
            for col in self.yearly_df.columns
        ]

        self.yearly_df['yearly_sharpe'] = remove_infs(
            self.yearly_df['daily_return_mean'] / self.yearly_df['daily_return_std']
        ) * np.sqrt(365)

        logger.info("Aggregated to %d yearly records", len(self.yearly_df))

    def _calculate_drawdowns(self) -> None:
        """Calculate top drawdown periods using vectorized operations.

        Same logic as simulation_report.py drawdown calculation but optimized
        for performance on long-term simulations.
        """
        if self.daily_df is None or self.daily_df.empty:
            return

        df = self.daily_df.copy().reset_index(drop=True)
        df['peak'] = df['cumulative_pnl'].cummax()
        df['drawdown'] = df['cumulative_pnl'] - df['peak']

        # Vectorized drawdown period detection
        in_dd = df['drawdown'] < 0
        # Detect transitions: start (False->True) and end (True->False)
        dd_starts = in_dd & ~in_dd.shift(1, fill_value=False)
        dd_ends = ~in_dd & in_dd.shift(1, fill_value=False)

        start_indices = df.index[dd_starts].tolist()
        end_indices = df.index[dd_ends].tolist()

        # Handle ongoing drawdown at end
        if in_dd.iloc[-1] and (not end_indices or
                               (start_indices and start_indices[-1] > end_indices[-1])):
            end_indices.append(len(df) - 1)

        drawdowns = []
        for start_idx, end_idx in zip(start_indices, end_indices):
            # end_idx is recovery day for ended drawdowns (trough at end_idx-1),
            # or last index for ongoing drawdowns (trough at end_idx)
            trough_idx = end_idx - 1 if end_idx < len(df) - 1 or not in_dd.iloc[-1] else end_idx

            dd_start = df.loc[start_idx, 'date']
            dd_end = df.loc[trough_idx, 'date']
            dd_start_value = df.loc[start_idx, 'peak']
            dd_min_value = df.loc[trough_idx, 'cumulative_pnl']
            dd_loss = dd_start_value - dd_min_value
            dd_days = (pd.to_datetime(dd_end) - pd.to_datetime(dd_start)).days

            drawdowns.append({
                'start_date': dd_start,
                'end_date': dd_end,
                'duration_days': dd_days,
                'peak_value': dd_start_value,
                'trough_value': dd_min_value,
                'dollar_loss': dd_loss,
                'percent_loss': dd_loss / dd_start_value * 100 if dd_start_value > 0 else 0,
            })

        if drawdowns:
            self.hist_top_drawdowns = pd.DataFrame(drawdowns)
            self.hist_top_drawdowns = self.hist_top_drawdowns.nlargest(5, 'dollar_loss')
            self.hist_top_drawdowns['start_date'] = pd.to_datetime(
                self.hist_top_drawdowns['start_date']
            ).dt.strftime('%Y-%m-%d')
            self.hist_top_drawdowns['end_date'] = pd.to_datetime(
                self.hist_top_drawdowns['end_date']
            ).dt.strftime('%Y-%m-%d')
        else:
            self.hist_top_drawdowns = pd.DataFrame()

    def _calculate_rolling_sharpe(self, window: int = 30) -> None:
        """Calculate rolling Sharpe ratio."""
        if self.daily_df is None or self.daily_df.empty:
            return

        df = self.daily_df.copy()
        returns = df['daily_return'].dropna()

        if len(returns) < window:
            return

        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        rolling_sharpe = remove_infs(rolling_mean / rolling_std) * np.sqrt(365)

        self.rolling_sharpe_df = pd.DataFrame({
            'date': df['date'].iloc[-len(rolling_sharpe):],
            f'sharpe_{window}d': rolling_sharpe.values,
        })

    # --- Figure Methods (same patterns as simulation_report.py) ---

    def create_pnl_figure(self) -> go.Figure:
        """Create P&L figure with cumulative and daily subplots.

        Same pattern as simulation_report.py pnl_fig.
        """
        if self.daily_df is None or self.daily_df.empty:
            return go.Figure()

        df = self.daily_df.copy()

        # Same subplot structure as simulation_report.py
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=('Cumulative P&L', 'Daily P&L'),
            vertical_spacing=0.1
        )

        # Cumulative P&L (same as simulation_report.py)
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['cumulative_pnl'],
            mode='lines',
            name='Cumulative P&L',
            line={'width': 2}
        ), row=1, col=1)

        # Daily P&L bars with colors (same as simulation_report.py)
        colors = df['daily_pnl'].apply(lambda x: 'green' if x > 0 else 'red')
        fig.add_trace(go.Bar(
            x=df['date'],
            y=df['daily_pnl'],
            name='Daily P&L',
            marker_color=colors
        ), row=2, col=1)

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=1)
        fig.update_yaxes(title_text="Daily P&L ($)", row=2, col=1)
        fig.update_layout(height=600, showlegend=False, title=f"P&L - {self.sim_name}")

        return fig

    def create_monthly_pnl_figure(self) -> go.Figure:
        """Create monthly P&L bar chart."""
        if self.monthly_df is None or self.monthly_df.empty:
            return go.Figure()

        df = self.monthly_df.copy()
        colors = df['daily_pnl_sum'].apply(lambda x: 'green' if x > 0 else 'red')

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['year_month'],
            y=df['daily_pnl_sum'],
            marker_color=colors,
            name='Monthly P&L'
        ))
        fig.update_layout(
            title="Monthly P&L",
            xaxis_title="Month",
            yaxis_title="P&L ($)",
        )
        return fig

    def create_drawdown_figure(self) -> go.Figure:
        """Create drawdown chart.

        Same pattern as simulation_report.py drawdown-chart.
        """
        if self.daily_df is None or self.daily_df.empty:
            return go.Figure()

        df = self.daily_df.copy()
        df['peak'] = df['cumulative_pnl'].cummax()
        df['drawdown_pct'] = remove_infs(
            (df['cumulative_pnl'] - df['peak']) / df['peak']
        ) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['drawdown_pct'],
            mode='lines',
            name='Drawdown %',
            line={'color': 'red', 'width': 2},
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ))
        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
        )
        return fig

    def create_rolling_sharpe_figure(self, window: int = 30) -> go.Figure:
        """Create rolling Sharpe ratio chart."""
        if self.rolling_sharpe_df is None:
            self._calculate_rolling_sharpe(window)

        if self.rolling_sharpe_df is None or self.rolling_sharpe_df.empty:
            return go.Figure()

        col_name = f'sharpe_{window}d'
        if col_name not in self.rolling_sharpe_df.columns:
            return go.Figure()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.rolling_sharpe_df['date'],
            y=self.rolling_sharpe_df[col_name],
            mode='lines',
            name=f'{window}d Rolling Sharpe',
            line={'width': 2}
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            title=f"{window}-Day Rolling Sharpe Ratio",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
        )
        return fig

    def create_portfolio_metrics_figure(self) -> go.Figure:
        """Create portfolio metrics figure.

        Same pattern as simulation_report.py portfolio_fig.
        """
        if self.daily_df is None or self.daily_df.empty:
            return go.Figure()

        df = self.daily_df.copy()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Gross Notional',
                'Position Count',
                'Daily Volume',
                'Daily Turnover'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Gross notional
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['gross_notional'],
            mode='lines', name='Gross Notional'
        ), row=1, col=1)

        # Position count
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['position_count'],
            mode='lines', name='Positions'
        ), row=1, col=2)

        # Daily volume
        fig.add_trace(go.Bar(
            x=df['date'], y=df['dollars_traded_daily'],
            name='Volume'
        ), row=2, col=1)

        # Turnover
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['turnover'],
            mode='lines', name='Turnover'
        ), row=2, col=2)

        fig.update_layout(height=600, showlegend=False, title="Portfolio Metrics")
        return fig

    # --- Table Data Methods ---

    def get_summary_data(self) -> List[List[str]]:
        """Get summary statistics table data.

        Same format as simulation_report.py summary_data.
        """
        if not self.return_metrics:
            return []

        m = self.return_metrics
        total_pnl = m.get('cum_pnl', 0)
        total_fees = m.get('cum_fees', 0)
        total_funding = m.get('cum_funding', 0)
        avg_notional = m.get('avg_notional', 0)
        avg_volume = m.get('avg_trading_volume', 0)
        ann_return = m.get('annualized_ret', 0)
        ann_risk = m.get('annualized_risk', 0)
        ann_sharpe = m.get('annualized_sharpe', 0)
        daily_turnover = m.get('daily_turnover', 0)

        # Same format as simulation_report.py
        return [
            ["Total P&L", f"${total_pnl:,.2f}", "Avg Daily Volume", f"${avg_volume:,.0f}"],
            ["Total Fees", f"${total_fees:,.2f}", "Avg Gross Notional", f"${avg_notional:,.0f}"],
            ["Total Funding", f"${total_funding:,.2f}", "Daily Turnover", f"{daily_turnover:.2f}"],
            ["Net P&L", f"${total_pnl + total_fees + total_funding:,.2f}", "", ""],
            ["Annualized Return", f"{ann_return*100:.2f}%",
             "Annualized Risk", f"{ann_risk*100:.2f}%"],
            ["Annualized Sharpe", f"{ann_sharpe:.2f}", "Total Days", f"{m.get('total_days', 0)}"],
        ]

    def get_monthly_table_data(self) -> List[Dict[str, Any]]:
        """Get monthly performance table data."""
        if self.monthly_df is None or self.monthly_df.empty:
            return []

        df = self.monthly_df.copy()
        df = df.sort_values('year_month', ascending=False)

        return df[[
            'year_month', 'daily_pnl_sum', 'cumulative_pnl_last',
            'gross_notional_mean', 'fees_usd_daily_sum',
            'funding_income_daily_sum', 'monthly_sharpe'
        ]].rename(columns={
            'year_month': 'Month',
            'daily_pnl_sum': 'Monthly P&L',
            'cumulative_pnl_last': 'Cumulative P&L',
            'gross_notional_mean': 'Avg Notional',
            'fees_usd_daily_sum': 'Fees',
            'funding_income_daily_sum': 'Funding',
            'monthly_sharpe': 'Sharpe',
        }).to_dict('records')

    def get_yearly_table_data(self) -> List[Dict[str, Any]]:
        """Get yearly performance table data."""
        if self.yearly_df is None or self.yearly_df.empty:
            return []

        df = self.yearly_df.copy()
        df = df.sort_values('year', ascending=False)

        return df[[
            'year', 'daily_pnl_sum', 'cumulative_pnl_last',
            'gross_notional_mean', 'fees_usd_daily_sum',
            'funding_income_daily_sum', 'daily_return_count', 'yearly_sharpe'
        ]].rename(columns={
            'year': 'Year',
            'daily_pnl_sum': 'Yearly P&L',
            'cumulative_pnl_last': 'Cumulative P&L',
            'gross_notional_mean': 'Avg Notional',
            'fees_usd_daily_sum': 'Fees',
            'funding_income_daily_sum': 'Funding',
            'daily_return_count': 'Days',
            'yearly_sharpe': 'Sharpe',
        }).to_dict('records')

    def get_drawdown_table_data(self) -> List[Dict[str, Any]]:
        """Get top drawdowns table data.

        Same format as simulation_report.py drawdown-table.
        """
        if self.hist_top_drawdowns is None or self.hist_top_drawdowns.empty:
            return []
        return self.hist_top_drawdowns.to_dict('records')

    def get_daily_pnl_table_data(self) -> List[Dict[str, Any]]:
        """Get daily P&L and returns table data.

        Same as simulation_report.py daily-pnl-returns-table.
        """
        if self.daily_df is None or self.daily_df.empty:
            return []

        df = self.daily_df.copy()
        df['daily_return_pct'] = df['daily_return'] * 100
        df['cumulative_return_pct'] = (1 + df['daily_return']).cumprod() - 1
        df['cumulative_return_pct'] = df['cumulative_return_pct'] * 100

        return df[[
            'date', 'daily_pnl', 'gross_notional',
            'daily_return_pct', 'cumulative_return_pct'
        ]].sort_values('date', ascending=False).to_dict('records')
