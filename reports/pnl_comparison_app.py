#!/usr/bin/env python3
"""
Simple PnL Comparison Dashboard.

Compares simulation PnL (from simcomp directory) with actual trading PnL.
"""

import argparse
import glob
import logging
import os
from datetime import datetime as dt, timedelta as td
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, callback_context, no_update

from lib.data.loaders import load_raw_targets_alpha
from lib.pnl_new.binance_pnl import BinancePnl
from lib.pnl_new.pnl_util import calculate_performance_statistics
from lib.reports.base_dash_app import BaseDashApp
from lib.util.config import get_config
from lib.util.directory import SIM_DIR
from lib.util.util import LOCAL

logger = logging.getLogger(__name__)


def _fmt_k(val: float) -> str:
    """Format a dollar value compactly for hover text."""
    if abs(val) >= 1000:
        return f"${val / 1000:+,.0f}K"
    return f"${val:+,.0f}"


class PnlComparisonApp(BaseDashApp):
    """Simple PnL Comparison Dashboard."""

    def __init__(self, port=8055, debug=False):
        """Initialize the PnL Comparison Dashboard.

        Args:
            port: Port number to run the server on (default: 8055)
            debug: Enable debug mode (default: False)
        """
        super().__init__("PnL Comparison", port, interval_secs=300)

        # Initialize config
        _, self.config = get_config()

        self.reopt_interval = self.config['REOPTIMIZE_INTERVAL_MINS']

        # Get list of available simcomp simulations, split by lookback
        self.sim_names_1d, self.sim_names_7d = self.get_simcomp_sims()
        self.sim_names = self.sim_names_1d + self.sim_names_7d
        logger.info(f"PnL Comparison Dashboard initialized with {len(self.sim_names_1d)} 1d and {len(self.sim_names_7d)} 7d simulations")

        self.sim_cache = {}
        self.actual_cache = {}

        # Setup the application
        self.setup_layout()
        self.register_callbacks()

    @staticmethod
    def get_simcomp_sims() -> Tuple[List[str], List[str]]:
        """Get available simcomp simulations, split into 1d and 7d lists."""
        simcomp_dir = f"{SIM_DIR}/simcomp"
        if not os.path.exists(simcomp_dir):
            logger.warning(f"Simcomp directory not found: {simcomp_dir}")
            return [], []

        sims_1d, sims_7d = [], []
        for name in os.listdir(simcomp_dir):
            path = os.path.join(simcomp_dir, name)
            if not os.path.isdir(path) or not os.path.exists(f"{path}/pnl.calculator.csv"):
                continue
            if '_1d_' in name:
                sims_1d.append(name)
            elif '_7d_' in name:
                sims_7d.append(name)

        sims_1d.sort(reverse=True)
        sims_7d.sort(reverse=True)
        return sims_1d, sims_7d

    def load_sim_pnl(self, sim_name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load simulation PnL data from detailed parquet files.

        Args:
            sim_name: Name of the simulation

        Returns:
            DataFrame with columns: ts, pnl_diff, trading_volume
        """
        if sim_name in self.sim_cache:
            security_df, portfolio_df = self.sim_cache[sim_name]
        else:
            sim_path = f"{SIM_DIR}/simcomp/{sim_name}"

            # Load all sim.YYYYMMDD.parquet files
            parquet_files = sorted(glob.glob(f"{sim_path}/sim.*.parquet"))
            if not parquet_files:
                return None, None

            logger.info(f"[PNL_COMP] Loading {len(parquet_files)} simulation parquet files from {sim_path}")

            # Load and concatenate all parquet files
            dfs = []
            for pf in parquet_files:
                df = pd.read_parquet(pf)
                dfs.append(df)

            # Combine all dataframes
            security_df = pd.concat(dfs)

            # Aggregate by timestamp (sum PnL across all symbols)
            portfolio_df = security_df.reset_index().groupby('ts').agg({
                'pnl': 'sum',
                'executed_dollars': 'sum'
            }).reset_index()

            # Ensure ts is datetime with UTC
            portfolio_df['ts'] = pd.to_datetime(portfolio_df['ts'], utc=True)
            # Sort by timestamp
            portfolio_df = portfolio_df.sort_values('ts')

            # Calculate PnL diff (incremental PnL between timestamps)
            portfolio_df['pnl_diff'] = portfolio_df['pnl'].diff().fillna(portfolio_df['pnl'].iloc[0])

            # Trading volume is the executed dollars amount
            portfolio_df['trading_volume'] = portfolio_df['executed_dollars'].abs()

            logger.info(f"[PNL_COMP] Loaded sim data: {len(portfolio_df)} rows, {portfolio_df['ts'].min()} to {portfolio_df['ts'].max()}")

            self.sim_cache[sim_name] = security_df, portfolio_df

        return security_df, portfolio_df

    def load_actual_pnl(self, start_dt: dt, end_dt: dt, sim_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """Load actual trading PnL data."""
        if sim_name in self.actual_cache:
            security_df, portfolio_df, baseline_net_pnl_cum = self.actual_cache[sim_name]
        else:
            # Start one reopt_interval earlier so first bucket aligns with sim after right-label shift
            actual_start = start_dt - td(minutes=self.reopt_interval)
            logger.info(f"[PNL_COMP] Loading actual PnL from {actual_start} to {end_dt}")
            pnl_calculator = BinancePnl(config=self.config, start=actual_start, end=end_dt)

            security_df = pnl_calculator.aggregate_by_security_date(
                interval_minutes=self.reopt_interval
            )
            # Shift left-labeled to right-labeled to align with sim bucketing
            security_df = security_df.reset_index()
            security_df['ts'] = security_df['ts'] + td(minutes=self.reopt_interval)
            security_df = security_df.set_index(['ts', 'symbol_venue'])
            portfolio_df = pnl_calculator.aggregate_portfolio(security_df)

            # Baseline from first aggregated timestamp so actual line starts at zero
            first_ts = portfolio_df.index.get_level_values('ts').min()
            baseline_net_pnl_cum = portfolio_df.loc[first_ts, 'net_pnl_cum'].sum()
            logger.info(f"[PNL_COMP] Baseline net_pnl_cum at {first_ts}: {baseline_net_pnl_cum:,.0f}")

            # Load and join alpha columns from raw_targets
            horizon_cols = ['alpha_120', 'alpha_360', 'alpha_720', 'alpha_1440',
                            'alpha_4320', 'alpha_10080', 'alpha_43200']
            alpha_cols = ['ts', 'symbol_venue', 'alpha_opt'] + horizon_cols
            alpha_df = load_raw_targets_alpha(
                start_dt=start_dt, end_dt=end_dt, cols=alpha_cols, skip_log=True
            )
            if alpha_df is not None and not alpha_df.empty:
                # Rename columns to _actual suffix to avoid collision with sim columns
                rename_map = {col: f'{col}_actual' for col in ['alpha_opt'] + horizon_cols}
                alpha_df = alpha_df.rename(columns=rename_map)

                # Floor alpha timestamps to reopt interval to match security_df timestamps
                # Raw targets have irregular timestamps that don't align with the reopt interval
                alpha_df = alpha_df.reset_index()
                alpha_df['ts'] = alpha_df['ts'].dt.floor(f'{self.reopt_interval}min')
                # After flooring, we may have duplicates per (ts, symbol_venue) - take mean
                alpha_df = alpha_df.groupby(['ts', 'symbol_venue']).mean()

                # Join on index (ts, symbol_venue)
                security_df = security_df.join(alpha_df, how='left')
                alpha_count = security_df['alpha_opt_actual'].notna().sum()
                logger.info(f"[PNL_COMP] Joined alpha_opt_actual: {alpha_count}/{len(security_df)} rows")
            else:
                logger.warning("[PNL_COMP] No raw_targets alpha data found")

            self.actual_cache[sim_name] = security_df, portfolio_df, baseline_net_pnl_cum
            logger.info(f"[PNL_COMP] Loaded {len(portfolio_df)} rows of actual PnL data")
            logger.info(f"[PNL_COMP] Actual data range: {portfolio_df.index.get_level_values('ts').min()} to {portfolio_df.index.get_level_values('ts').max()}")

        return security_df, portfolio_df, baseline_net_pnl_cum

    def _compute_divergence_data(
        self,
        sim_security_df: pd.DataFrame,
        actual_security_df: pd.DataFrame,
    ) -> Tuple[List, List[float], List[str]]:
        """Compute per-timestamp divergence with top symbol drivers for hover text.

        Args:
            sim_security_df: MultiIndex (ts, symbol_venue), column 'pnl' (cumulative)
            actual_security_df: MultiIndex (ts, symbol_venue), column 'net_pnl_cum'

        Returns:
            Tuple of (timestamps, divergence_values, hover_texts)
        """
        sim_timestamps = sim_security_df.index.get_level_values('ts').unique().sort_values()
        actual_timestamps = actual_security_df.index.get_level_values('ts').unique().sort_values()

        if len(actual_timestamps) == 0 or len(sim_timestamps) == 0:
            return [], [], []

        # Per-symbol actual baseline at first actual timestamp
        first_actual_ts = actual_timestamps[0]
        baseline_per_symbol = actual_security_df.xs(
            first_actual_ts, level='ts'
        )['net_pnl_cum']

        divergence_timestamps = []
        divergence_values: List[float] = []
        hover_texts: List[str] = []

        for sim_ts in sim_timestamps:
            # Find closest actual timestamp (sim ~xx:03, actual ~xx:00)
            diffs_seconds = np.abs(
                (actual_timestamps - sim_ts).total_seconds()
            )
            closest_idx = diffs_seconds.argmin()
            actual_ts = actual_timestamps[closest_idx]

            # Skip if too far apart (> 10 minutes)
            if diffs_seconds[closest_idx] > 600:
                continue

            # Per-symbol PnL at this timestamp
            sim_pnl = sim_security_df.xs(sim_ts, level='ts')['pnl']
            actual_pnl_cum = actual_security_df.xs(actual_ts, level='ts')['net_pnl_cum']

            # Zero actual using per-symbol baselines
            actual_pnl = actual_pnl_cum - baseline_per_symbol.reindex(
                actual_pnl_cum.index, fill_value=0.0
            )

            # Align on all symbols and compute divergence
            all_symbols = sim_pnl.index.union(actual_pnl.index)
            sim_aligned = sim_pnl.reindex(all_symbols, fill_value=0.0)
            actual_aligned = actual_pnl.reindex(all_symbols, fill_value=0.0)
            per_symbol_div = sim_aligned - actual_aligned

            total_div = per_symbol_div.sum()

            # Top 5 by absolute divergence
            top5_idx = per_symbol_div.abs().nlargest(5).index

            # Build hover text
            lines = [f"Divergence: ${total_div:,.0f}", "\u2500" * 20]
            for symbol in top5_idx:
                div = per_symbol_div[symbol]
                s_val = sim_aligned[symbol]
                a_val = actual_aligned[symbol]
                lines.append(
                    f"{symbol}: {_fmt_k(div)} (S:{_fmt_k(s_val)} A:{_fmt_k(a_val)})"
                )

            divergence_timestamps.append(sim_ts)
            divergence_values.append(total_div)
            hover_texts.append("<br>".join(lines))

        return divergence_timestamps, divergence_values, hover_texts

    def create_pnl_comparison_figure(self, sim_name: str) -> Tuple[go.Figure, str]:
        """Create PnL comparison figure.
        Args:
            sim_name: Name of simulation to compare

        Returns:
            Tuple of (figure, status_message)
        """
        if not sim_name:
            return go.Figure(), "No simulation selected"

        try:
            sim_security_df, sim_portfolio_df = self.load_sim_pnl(sim_name)
            if len(sim_portfolio_df) == 0:
                return go.Figure(), "No simulation data available"

            # Get simulation date range and interval
            sim_start = sim_portfolio_df ['ts'].min()
            sim_end = sim_portfolio_df ['ts'].max()

            # Load actual data for the same period
            actual_security_df, actual_portfolio_df, baseline_net_pnl_cum = self.load_actual_pnl(sim_start, sim_end, sim_name)

            actual_pnl_cum_zeroed = actual_portfolio_df['net_pnl_cum'] - baseline_net_pnl_cum

            # Create figure
            fig = go.Figure()

            # Add simulation trace
            fig.add_trace(go.Scatter(
                x=sim_portfolio_df['ts'],
                y=sim_portfolio_df['pnl'],
                mode='lines',
                name='Simulation',
                line={'color': 'blue', 'width': 2}
            ))

            # Add actual trace
            fig.add_trace(go.Scatter(
                x=actual_portfolio_df.index.get_level_values('ts'),
                y=actual_pnl_cum_zeroed,
                mode='lines',
                name='Actual Trading',
                line={'color': 'green', 'width': 2}
            ))

            # Compute and add divergence trace
            div_timestamps, div_values, div_hovers = self._compute_divergence_data(
                sim_security_df, actual_security_df
            )
            if div_timestamps:
                fig.add_trace(go.Scatter(
                    x=div_timestamps,
                    y=div_values,
                    mode='lines+markers',
                    name='Divergence (Sim-Actual)',
                    line={'color': 'red', 'width': 1.5, 'dash': 'dash'},
                    marker={'size': 6},
                    text=div_hovers,
                    hovertemplate='%{text}<extra></extra>'
                ))

            status = (
                f"Showing {len(sim_portfolio_df)} sim points, "
                f"{len(actual_portfolio_df)} actual points, "
                f"{len(div_timestamps)} divergence points"
            )

            # Update layout
            fig.update_layout(
                title=f"PnL Comparison: {sim_name}",
                xaxis_title='Date',
                yaxis_title='Cumulative PnL ($)',
                hovermode='x unified',
                height=600
            )

            logger.info(f"[PNL_COMP] Created figure with {len(fig.data)} traces")

            return fig, status

        except Exception as e:
            logger.error(f"[PNL_COMP] Error creating figure: {e}", exc_info=True)
            return go.Figure(), f"Error: {str(e)}"

    def create_daily_delta_bar_chart(self, sim_name: str) -> go.Figure:
        """Create bar chart showing daily PnL delta (sim - actual).

        Args:
            sim_name: Name of simulation to compare

        Returns:
            Plotly figure with bar chart
        """
        if not sim_name or sim_name not in self.sim_cache or sim_name not in self.actual_cache:
            return go.Figure()

        try:
            # Use portfolio_df which already has pnl summed across symbols
            _, sim_portfolio_df = self.sim_cache[sim_name]
            _, actual_portfolio_df, _ = self.actual_cache[sim_name]

            # Prepare sim data: get end-of-day cumulative pnl
            # Subtract 1 min so midnight (00:00) snapshots roll back to previous day
            sim_df = sim_portfolio_df.copy()
            sim_df['date'] = (pd.to_datetime(sim_df['ts']) - pd.Timedelta(minutes=1)).dt.date
            sim_daily_df = sim_df.groupby('date').agg({'pnl': 'last'}).reset_index()
            sim_daily_df = sim_daily_df.sort_values('date')
            # Calculate daily pnl as diff of cumulative
            sim_daily_df['pnl_daily'] = sim_daily_df['pnl'].diff()
            sim_daily_df.loc[sim_daily_df.index[0], 'pnl_daily'] = sim_daily_df['pnl'].iloc[0]

            # Prepare actual data: get end-of-day cumulative pnl
            actual_df = actual_portfolio_df.reset_index()
            actual_df['date'] = (pd.to_datetime(actual_df['ts']) - pd.Timedelta(minutes=1)).dt.date
            actual_daily_df = actual_df.groupby('date').agg({'net_pnl_cum': 'last'}).reset_index()
            actual_daily_df = actual_daily_df.sort_values('date')
            # Calculate daily pnl as diff of cumulative
            actual_daily_df['pnl_daily'] = actual_daily_df['net_pnl_cum'].diff()
            actual_daily_df.loc[actual_daily_df.index[0], 'pnl_daily'] = actual_daily_df['net_pnl_cum'].iloc[0]

            # Merge sim and actual
            merged_df = pd.merge(
                sim_daily_df[['date', 'pnl_daily']],
                actual_daily_df[['date', 'pnl_daily']],
                on='date',
                how='outer',
                suffixes=('_sim', '_actual')
            )
            merged_df['pnl_daily_sim'] = merged_df['pnl_daily_sim'].fillna(0)
            merged_df['pnl_daily_actual'] = merged_df['pnl_daily_actual'].fillna(0)
            merged_df['delta'] = merged_df['pnl_daily_actual'] - merged_df['pnl_daily_sim']
            merged_df = merged_df.sort_values('date')

            # Color bars based on positive/negative
            colors = ['green' if d >= 0 else 'red' for d in merged_df['delta']]

            fig = go.Figure(data=go.Bar(
                x=[str(d) for d in merged_df['date']],
                y=merged_df['delta'],
                name='Daily Delta',
                marker_color=colors,
                hovertemplate=(
                    'Date: %{x}<br>'
                    'Delta: $%{y:.2f}<br>'
                    '<extra></extra>'
                )
            ))

            # Add cumulative delta line
            merged_df['delta_cum'] = merged_df['delta'].cumsum()
            fig.add_trace(go.Scatter(
                x=[str(d) for d in merged_df['date']],
                y=merged_df['delta_cum'],
                mode='lines+markers',
                name='Cumulative Delta',
                line={'color': 'blue', 'width': 2},
                yaxis='y2'
            ))

            fig.update_layout(
                title='Daily PnL Delta (Actual - Sim)',
                xaxis_title='Date',
                yaxis_title='Daily Delta ($)',
                yaxis2={
                    'title': 'Cumulative Delta ($)',
                    'overlaying': 'y',
                    'side': 'right'
                },
                height=400,
                showlegend=True,
                legend={'x': 0.01, 'y': 0.99}
            )

            return fig

        except Exception as e:
            logger.error(f"[PNL_COMP] Error creating daily delta bar chart: {e}", exc_info=True)
            return go.Figure()

    def create_symbol_day_delta_heatmap(self, sim_name: str) -> go.Figure:
        """Create bar charts showing PnL delta (actual - sim) by symbol per day.

        Each day has its own column with symbols sorted by delta value (largest at top).

        Args:
            sim_name: Name of simulation to compare

        Returns:
            Plotly figure with horizontal bar charts per day, sorted by delta
        """
        if not sim_name or sim_name not in self.sim_cache or sim_name not in self.actual_cache:
            return go.Figure()

        try:
            sim_security_df, _ = self.sim_cache[sim_name]
            actual_security_df, _, _ = self.actual_cache[sim_name]

            # Prepare sim data: aggregate pnl by date and symbol
            sim_df = sim_security_df.reset_index()
            sim_df['date'] = (pd.to_datetime(sim_df['ts']) - pd.Timedelta(minutes=1)).dt.date

            # Group by date and symbol, take last pnl value (cumulative) per day
            sim_daily_df = sim_df.groupby(['date', 'symbol_venue']).agg({
                'pnl': 'last'
            }).reset_index()

            # Calculate daily pnl change (diff within each symbol)
            sim_daily_df = sim_daily_df.sort_values(['symbol_venue', 'date'])
            sim_daily_df['pnl_daily'] = sim_daily_df.groupby('symbol_venue')['pnl'].diff()
            # First day for each symbol uses the pnl value directly
            first_day_mask = sim_daily_df.groupby('symbol_venue').cumcount() == 0
            sim_daily_df.loc[first_day_mask, 'pnl_daily'] = sim_daily_df.loc[first_day_mask, 'pnl']

            # Prepare actual data: aggregate net_pnl by date and symbol
            actual_df = actual_security_df.reset_index()
            actual_df['date'] = (pd.to_datetime(actual_df['ts']) - pd.Timedelta(minutes=1)).dt.date

            actual_daily_df = actual_df.groupby(['date', 'symbol_venue']).agg({
                'net_pnl': 'sum'  # Sum of pnl diffs gives daily pnl
            }).reset_index()
            actual_daily_df = actual_daily_df.rename(columns={'net_pnl': 'pnl_daily'})

            # Merge sim and actual on date + symbol
            merged_df = pd.merge(
                sim_daily_df[['date', 'symbol_venue', 'pnl_daily']],
                actual_daily_df[['date', 'symbol_venue', 'pnl_daily']],
                on=['date', 'symbol_venue'],
                how='outer',
                suffixes=('_sim', '_actual')
            )
            merged_df['pnl_daily_sim'] = merged_df['pnl_daily_sim'].fillna(0)
            merged_df['pnl_daily_actual'] = merged_df['pnl_daily_actual'].fillna(0)
            merged_df['delta'] = merged_df['pnl_daily_actual'] - merged_df['pnl_daily_sim']

            # Get unique dates sorted
            dates = sorted(merged_df['date'].unique())
            n_days = len(dates)

            if n_days == 0:
                return go.Figure()

            # Create subplots - one column per day
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=1, cols=n_days,
                subplot_titles=[str(d) for d in dates],
                horizontal_spacing=0.02
            )

            # Number of symbols to show per day
            top_n = 20

            for i, date in enumerate(dates, 1):
                day_df = merged_df[merged_df['date'] == date].copy()

                # Take top N by absolute value
                day_df['abs_delta'] = day_df['delta'].abs()
                top_symbols = day_df.nlargest(top_n, 'abs_delta')['symbol_venue'].tolist()
                day_df = day_df[day_df['symbol_venue'].isin(top_symbols)]

                # Sort by delta ascending (largest positive at top when using horizontal bars)
                day_df = day_df.sort_values('delta', ascending=True)

                # Color based on positive/negative
                colors = ['rgba(55, 128, 191, 0.8)' if v >= 0 else 'rgba(219, 64, 82, 0.8)'
                          for v in day_df['delta']]

                fig.add_trace(
                    go.Bar(
                        y=day_df['symbol_venue'],
                        x=day_df['delta'],
                        orientation='h',
                        marker_color=colors,
                        hovertemplate=(
                            'Symbol: %{y}<br>'
                            'PnL Delta: $%{x:,.0f}<br>'
                            '<extra></extra>'
                        ),
                        showlegend=False
                    ),
                    row=1, col=i
                )

                # Only show y-axis labels on first column
                if i > 1:
                    fig.update_yaxes(showticklabels=False, row=1, col=i)

            fig.update_layout(
                title='PnL Delta by Symbol and Day (Actual - Sim) - Sorted by Delta',
                height=max(500, top_n * 22 + 100),
                showlegend=False
            )

            # Update axes
            fig.update_yaxes(tickfont={'size': 9}, row=1, col=1)
            fig.update_xaxes(tickfont={'size': 8})

            return fig

        except Exception as e:
            logger.error(f"[PNL_COMP] Error creating delta heatmap: {e}", exc_info=True)
            return go.Figure()

    def create_trade_volume_delta_heatmap(self, sim_name: str) -> go.Figure:
        """Create bar charts showing trade volume delta (actual - sim) by symbol per day.

        Each day has its own column with symbols sorted by delta value (largest at top).
        Compares total dollars bought and sold between simulation and actual trading.

        Args:
            sim_name: Name of simulation to compare

        Returns:
            Plotly figure with horizontal bar charts per day, sorted by delta
        """
        if not sim_name or sim_name not in self.sim_cache or sim_name not in self.actual_cache:
            return go.Figure()

        try:
            sim_security_df, _ = self.sim_cache[sim_name]
            actual_security_df, _, _ = self.actual_cache[sim_name]

            # Prepare sim data: aggregate executed_dollars by date and symbol
            sim_df = sim_security_df.reset_index()
            sim_df['date'] = (pd.to_datetime(sim_df['ts']) - pd.Timedelta(minutes=1)).dt.date

            # Calculate buy/sell from executed_dollars (positive = buy, negative = sell)
            sim_df['dollars_buy'] = sim_df['executed_dollars'].clip(lower=0)
            sim_df['dollars_sell'] = sim_df['executed_dollars'].clip(upper=0).abs()

            # Group by date and symbol, sum buy/sell volumes
            sim_daily_df = sim_df.groupby(['date', 'symbol_venue']).agg({
                'dollars_buy': 'sum',
                'dollars_sell': 'sum'
            }).reset_index()

            # Prepare actual data: aggregate fill_dollars_buy/sell by date and symbol
            actual_df = actual_security_df.reset_index()
            actual_df['date'] = (pd.to_datetime(actual_df['ts']) - pd.Timedelta(minutes=1)).dt.date

            actual_daily_df = actual_df.groupby(['date', 'symbol_venue']).agg({
                'fill_dollars_buy': 'sum',
                'fill_dollars_sell': 'sum'
            }).reset_index()

            # Merge sim and actual on date + symbol
            merged_df = pd.merge(
                sim_daily_df,
                actual_daily_df,
                on=['date', 'symbol_venue'],
                how='outer',
                suffixes=('_sim', '_actual')
            )

            # Fill NaN with 0 for missing data
            for col in ['dollars_buy', 'dollars_sell', 'fill_dollars_buy', 'fill_dollars_sell']:
                merged_df[col] = merged_df[col].fillna(0)

            # Calculate deltas (actual - sim)
            merged_df['delta_buy'] = merged_df['fill_dollars_buy'] - merged_df['dollars_buy']
            merged_df['delta_sell'] = merged_df['fill_dollars_sell'] - merged_df['dollars_sell']
            # Net volume delta: positive means actual traded more net dollars
            merged_df['delta_net'] = (
                (merged_df['fill_dollars_buy'] - merged_df['fill_dollars_sell']) -
                (merged_df['dollars_buy'] - merged_df['dollars_sell'])
            )

            # Get unique dates sorted
            dates = sorted(merged_df['date'].unique())
            n_days = len(dates)

            if n_days == 0:
                return go.Figure()

            # Create subplots - one column per day
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=1, cols=n_days,
                subplot_titles=[str(d) for d in dates],
                horizontal_spacing=0.02
            )

            # Number of symbols to show per day
            top_n = 20

            for i, date in enumerate(dates, 1):
                day_df = merged_df[merged_df['date'] == date].copy()

                # Sort by delta_net descending (largest positive at top)
                day_df = day_df.sort_values('delta_net', ascending=True)

                # Take top N by absolute value but keep the sorted order
                day_df['abs_delta'] = day_df['delta_net'].abs()
                top_symbols = day_df.nlargest(top_n, 'abs_delta')['symbol_venue'].tolist()
                day_df = day_df[day_df['symbol_venue'].isin(top_symbols)]
                day_df = day_df.sort_values('delta_net', ascending=True)

                # Color based on positive/negative
                colors = ['rgba(55, 128, 191, 0.8)' if v >= 0 else 'rgba(219, 64, 82, 0.8)'
                          for v in day_df['delta_net']]

                fig.add_trace(
                    go.Bar(
                        y=day_df['symbol_venue'],
                        x=day_df['delta_net'],
                        orientation='h',
                        marker_color=colors,
                        hovertemplate=(
                            'Symbol: %{y}<br>'
                            'Net Delta: $%{x:,.0f}<br>'
                            '<extra></extra>'
                        ),
                        showlegend=False
                    ),
                    row=1, col=i
                )

                # Only show y-axis labels on first column
                if i > 1:
                    fig.update_yaxes(showticklabels=False, row=1, col=i)

            fig.update_layout(
                title='Trade Volume Delta by Symbol and Day (Actual - Sim) - Sorted by Delta',
                height=max(500, top_n * 22 + 100),
                showlegend=False
            )

            # Update axes
            fig.update_yaxes(tickfont={'size': 9}, row=1, col=1)
            fig.update_xaxes(tickfont={'size': 8})

            return fig

        except Exception as e:
            logger.error(f"[PNL_COMP] Error creating trade volume heatmap: {e}", exc_info=True)
            return go.Figure()

    def create_alpha_delta_heatmap(self, sim_name: str) -> go.Figure:
        """Create bar charts showing alpha_opt delta (actual - sim) by symbol per day.

        Each day has its own column with symbols sorted by delta value (largest at top).
        Compares average alpha_opt between simulation and actual trading.

        Args:
            sim_name: Name of simulation to compare

        Returns:
            Plotly figure with horizontal bar charts per day, sorted by delta
        """
        if not sim_name or sim_name not in self.sim_cache or sim_name not in self.actual_cache:
            return go.Figure()

        try:
            sim_security_df, _ = self.sim_cache[sim_name]
            actual_security_df, _, _ = self.actual_cache[sim_name]

            # Check if alpha columns exist
            if 'alpha_opt' not in sim_security_df.columns:
                logger.warning("[PNL_COMP] alpha_opt not found in sim_security_df")
                return go.Figure()

            if 'alpha_opt_actual' not in actual_security_df.columns:
                logger.warning("[PNL_COMP] alpha_opt_actual not found in actual_security_df")
                return go.Figure()

            # Prepare sim data: aggregate alpha_opt by date and symbol
            sim_df = sim_security_df.reset_index()
            sim_df['date'] = (pd.to_datetime(sim_df['ts']) - pd.Timedelta(minutes=1)).dt.date

            sim_daily_df = sim_df.groupby(['date', 'symbol_venue']).agg({
                'alpha_opt': 'mean'
            }).reset_index()

            # Prepare actual data: aggregate alpha_opt_actual by date and symbol
            actual_df = actual_security_df.reset_index()
            actual_df['date'] = (pd.to_datetime(actual_df['ts']) - pd.Timedelta(minutes=1)).dt.date

            actual_daily_df = actual_df.groupby(['date', 'symbol_venue']).agg({
                'alpha_opt_actual': 'mean'
            }).reset_index()

            # Merge sim and actual on date + symbol
            merged_df = pd.merge(
                sim_daily_df[['date', 'symbol_venue', 'alpha_opt']],
                actual_daily_df[['date', 'symbol_venue', 'alpha_opt_actual']],
                on=['date', 'symbol_venue'],
                how='inner'  # Only keep rows where we have both
            )

            # Calculate delta (actual - sim)
            merged_df['delta'] = merged_df['alpha_opt_actual'] - merged_df['alpha_opt']

            # Get unique dates sorted
            dates = sorted(merged_df['date'].unique())
            n_days = len(dates)

            if n_days == 0:
                return go.Figure()

            # Create subplots - one column per day
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=1, cols=n_days,
                subplot_titles=[str(d) for d in dates],
                horizontal_spacing=0.02
            )

            # Number of symbols to show per day
            top_n = 20

            for i, date in enumerate(dates, 1):
                day_df = merged_df[merged_df['date'] == date].copy()

                # Take top N by absolute value
                day_df['abs_delta'] = day_df['delta'].abs()
                top_symbols = day_df.nlargest(top_n, 'abs_delta')['symbol_venue'].tolist()
                day_df = day_df[day_df['symbol_venue'].isin(top_symbols)]

                # Sort by delta ascending (largest positive at top when using horizontal bars)
                day_df = day_df.sort_values('delta', ascending=True)

                # Color based on positive/negative
                colors = ['rgba(55, 128, 191, 0.8)' if v >= 0 else 'rgba(219, 64, 82, 0.8)'
                          for v in day_df['delta']]

                fig.add_trace(
                    go.Bar(
                        y=day_df['symbol_venue'],
                        x=day_df['delta'],
                        orientation='h',
                        marker_color=colors,
                        hovertemplate=(
                            'Symbol: %{y}<br>'
                            'Alpha Delta: %{x:.4f}<br>'
                            '<extra></extra>'
                        ),
                        showlegend=False
                    ),
                    row=1, col=i
                )

                # Only show y-axis labels on first column
                if i > 1:
                    fig.update_yaxes(showticklabels=False, row=1, col=i)

            fig.update_layout(
                title='Alpha Opt Delta by Symbol and Day (Actual - Sim)',
                height=max(500, top_n * 22 + 100),
                showlegend=False
            )

            # Update axes
            fig.update_yaxes(tickfont={'size': 9}, row=1, col=1)
            fig.update_xaxes(tickfont={'size': 8})

            return fig

        except Exception as e:
            logger.error(f"[PNL_COMP] Error creating alpha delta heatmap: {e}", exc_info=True)
            return go.Figure()

    def create_alpha_horizon_delta_heatmap(self, sim_name: str, horizon: str) -> go.Figure:
        """Create bar charts showing alpha delta for a specific horizon by symbol per day.

        Args:
            sim_name: Name of simulation to compare
            horizon: Horizon column name (e.g., 'alpha_120', 'alpha_1440')

        Returns:
            Plotly figure with horizontal bar charts per day, sorted by delta
        """
        if not sim_name or sim_name not in self.sim_cache or sim_name not in self.actual_cache:
            return go.Figure()

        try:
            sim_security_df, _ = self.sim_cache[sim_name]
            actual_security_df, _, _ = self.actual_cache[sim_name]

            actual_col = f'{horizon}_actual'

            # Check if columns exist
            if horizon not in sim_security_df.columns:
                logger.warning(f"[PNL_COMP] {horizon} not found in sim_security_df")
                return go.Figure()

            if actual_col not in actual_security_df.columns:
                logger.warning(f"[PNL_COMP] {actual_col} not found in actual_security_df")
                return go.Figure()

            # Prepare sim data
            sim_df = sim_security_df.reset_index()
            sim_df['date'] = (pd.to_datetime(sim_df['ts']) - pd.Timedelta(minutes=1)).dt.date

            sim_daily_df = sim_df.groupby(['date', 'symbol_venue']).agg({
                horizon: 'mean'
            }).reset_index()

            # Prepare actual data
            actual_df = actual_security_df.reset_index()
            actual_df['date'] = (pd.to_datetime(actual_df['ts']) - pd.Timedelta(minutes=1)).dt.date

            actual_daily_df = actual_df.groupby(['date', 'symbol_venue']).agg({
                actual_col: 'mean'
            }).reset_index()
            actual_daily_df = actual_daily_df.rename(columns={actual_col: f'{horizon}_actual_agg'})

            # Merge sim and actual
            merged_df = pd.merge(
                sim_daily_df[['date', 'symbol_venue', horizon]],
                actual_daily_df[['date', 'symbol_venue', f'{horizon}_actual_agg']],
                on=['date', 'symbol_venue'],
                how='inner'
            )

            # Calculate delta
            merged_df['delta'] = merged_df[f'{horizon}_actual_agg'] - merged_df[horizon]

            dates = sorted(merged_df['date'].unique())
            n_days = len(dates)

            if n_days == 0:
                return go.Figure()

            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=1, cols=n_days,
                subplot_titles=[str(d) for d in dates],
                horizontal_spacing=0.02
            )

            top_n = 20

            for i, date in enumerate(dates, 1):
                day_df = merged_df[merged_df['date'] == date].copy()

                day_df['abs_delta'] = day_df['delta'].abs()
                top_symbols = day_df.nlargest(top_n, 'abs_delta')['symbol_venue'].tolist()
                day_df = day_df[day_df['symbol_venue'].isin(top_symbols)]
                day_df = day_df.sort_values('delta', ascending=True)

                colors = ['rgba(55, 128, 191, 0.8)' if v >= 0 else 'rgba(219, 64, 82, 0.8)'
                          for v in day_df['delta']]

                fig.add_trace(
                    go.Bar(
                        y=day_df['symbol_venue'],
                        x=day_df['delta'],
                        orientation='h',
                        marker_color=colors,
                        hovertemplate=(
                            'Symbol: %{y}<br>'
                            'Delta: %{x:.4f}<br>'
                            '<extra></extra>'
                        ),
                        showlegend=False
                    ),
                    row=1, col=i
                )

                if i > 1:
                    fig.update_yaxes(showticklabels=False, row=1, col=i)

            # Human-readable horizon labels
            horizon_labels = {
                'alpha_120': '2h',
                'alpha_360': '6h',
                'alpha_720': '12h',
                'alpha_1440': '1d',
                'alpha_4320': '3d',
                'alpha_10080': '1w',
                'alpha_43200': '1mo'
            }
            label = horizon_labels.get(horizon, horizon)

            fig.update_layout(
                title=f'{label} Alpha Delta by Symbol and Day (Actual - Sim)',
                height=max(500, top_n * 22 + 100),
                showlegend=False
            )

            fig.update_yaxes(tickfont={'size': 9}, row=1, col=1)
            fig.update_xaxes(tickfont={'size': 8})

            return fig

        except Exception as e:
            logger.error(f"[PNL_COMP] Error creating {horizon} delta heatmap: {e}", exc_info=True)
            return go.Figure()

    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.H1("PnL Comparison Dashboard", style={'textAlign': 'center', 'marginBottom': '20px'}),
            html.Hr(),

            # Controls
            html.Div([
                html.Div([
                    html.Label('1-Day Sims:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='sim-selector-1d',
                        options=[{'label': name, 'value': name} for name in self.sim_names_1d],
                        value=self.sim_names_1d[0] if self.sim_names_1d else None,
                        style={'width': '350px', 'display': 'inline-block'}
                    ),
                ], style={'display': 'inline-block', 'marginRight': '20px'}),
                html.Div([
                    html.Label('7-Day Sims:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='sim-selector-7d',
                        options=[{'label': name, 'value': name} for name in self.sim_names_7d],
                        value=self.sim_names_7d[0] if (self.sim_names_7d and not self.sim_names_1d) else None,
                        style={'width': '350px', 'display': 'inline-block'}
                    ),
                ], style={'display': 'inline-block', 'marginRight': '20px'}),
                html.Button('Reload Sims', id='refresh-button', n_clicks=0),
            ], style={'marginBottom': '20px'}),

            # Status message
            html.Div(id='status-message', style={'marginBottom': '10px', 'fontStyle': 'italic'}),

            # Chart with loading indicator
            dcc.Loading(
                id='loading-chart',
                type='default',  # Options: 'default', 'circle', 'dot', 'cube'
                children=[dcc.Graph(id='pnl-chart')]
            ),

            # Daily Delta Bar Chart
            html.Hr(),
            html.H3("Daily PnL Delta", style={'marginTop': '20px'}),
            dcc.Loading(
                id='loading-daily-bar',
                type='default',
                children=[dcc.Graph(id='daily-delta-bar')]
            ),

            # Symbol-Day Delta Heatmap
            html.Hr(),
            html.H3("PnL Delta by Symbol and Day", style={'marginTop': '20px'}),
            dcc.Loading(
                id='loading-heatmap',
                type='default',
                children=[dcc.Graph(id='delta-heatmap')]
            ),

            # Trade Volume Delta Heatmap
            html.Hr(),
            html.H3("Trade Volume Delta by Symbol and Day", style={'marginTop': '20px'}),
            html.P(
                "Compares dollars bought/sold between simulation and actual trading. "
                "Positive values (blue) = actual traded more; Negative (red) = sim traded more.",
                style={'fontSize': '12px', 'color': 'gray', 'marginBottom': '10px'}
            ),
            dcc.Loading(
                id='loading-volume-heatmap',
                type='default',
                children=[dcc.Graph(id='volume-delta-heatmap')]
            ),

            # Alpha Delta Heatmap
            html.Hr(),
            html.H3("Alpha Opt Delta by Symbol and Day", style={'marginTop': '20px'}),
            html.P(
                "Compares average alpha_opt between simulation and actual trading. "
                "Positive values (blue) = actual alpha higher; Negative (red) = sim alpha higher.",
                style={'fontSize': '12px', 'color': 'gray', 'marginBottom': '10px'}
            ),
            dcc.Loading(
                id='loading-alpha-heatmap',
                type='default',
                children=[dcc.Graph(id='alpha-delta-heatmap')]
            ),

            # Alpha Horizon Delta Heatmaps
            html.Hr(),
            html.H3("Alpha Delta by Horizon", style={'marginTop': '20px'}),
            html.P(
                "Per-horizon breakdown of alpha differences. "
                "Shows which horizons contribute most to the alpha_opt delta.",
                style={'fontSize': '12px', 'color': 'gray', 'marginBottom': '10px'}
            ),

            # 2h horizon
            html.H4("2h (alpha_120)", style={'marginTop': '15px', 'marginBottom': '5px'}),
            dcc.Loading(id='loading-alpha-120', type='default',
                        children=[dcc.Graph(id='alpha-120-heatmap')]),

            # 6h horizon
            html.H4("6h (alpha_360)", style={'marginTop': '15px', 'marginBottom': '5px'}),
            dcc.Loading(id='loading-alpha-360', type='default',
                        children=[dcc.Graph(id='alpha-360-heatmap')]),

            # 12h horizon
            html.H4("12h (alpha_720)", style={'marginTop': '15px', 'marginBottom': '5px'}),
            dcc.Loading(id='loading-alpha-720', type='default',
                        children=[dcc.Graph(id='alpha-720-heatmap')]),

            # 1d horizon
            html.H4("1d (alpha_1440)", style={'marginTop': '15px', 'marginBottom': '5px'}),
            dcc.Loading(id='loading-alpha-1440', type='default',
                        children=[dcc.Graph(id='alpha-1440-heatmap')]),

            # 3d horizon
            html.H4("3d (alpha_4320)", style={'marginTop': '15px', 'marginBottom': '5px'}),
            dcc.Loading(id='loading-alpha-4320', type='default',
                        children=[dcc.Graph(id='alpha-4320-heatmap')]),

            # 1w horizon
            html.H4("1w (alpha_10080)", style={'marginTop': '15px', 'marginBottom': '5px'}),
            dcc.Loading(id='loading-alpha-10080', type='default',
                        children=[dcc.Graph(id='alpha-10080-heatmap')]),

            # 1mo horizon
            html.H4("1mo (alpha_43200)", style={'marginTop': '15px', 'marginBottom': '5px'}),
            dcc.Loading(id='loading-alpha-43200', type='default',
                        children=[dcc.Graph(id='alpha-43200-heatmap')]),

            # Auto-refresh interval (5 minutes)
            dcc.Interval(
                id='interval-component',
                interval=self.interval_secs * 1000,
                n_intervals=0
            ),
        ], style={'padding': '20px'})

    def register_callbacks(self):
        """Register dashboard callbacks."""

        @self.app.callback(
            [Output('pnl-chart', 'figure'),
             Output('daily-delta-bar', 'figure'),
             Output('delta-heatmap', 'figure'),
             Output('volume-delta-heatmap', 'figure'),
             Output('alpha-delta-heatmap', 'figure'),
             Output('alpha-120-heatmap', 'figure'),
             Output('alpha-360-heatmap', 'figure'),
             Output('alpha-720-heatmap', 'figure'),
             Output('alpha-1440-heatmap', 'figure'),
             Output('alpha-4320-heatmap', 'figure'),
             Output('alpha-10080-heatmap', 'figure'),
             Output('alpha-43200-heatmap', 'figure'),
             Output('status-message', 'children'),
             Output('sim-selector-1d', 'options'),
             Output('sim-selector-7d', 'options'),
             Output('sim-selector-1d', 'value'),
             Output('sim-selector-7d', 'value')],
            [Input('sim-selector-1d', 'value'),
             Input('sim-selector-7d', 'value'),
             Input('interval-component', 'n_intervals'),
             Input('refresh-button', 'n_clicks')]
        )
        def update_chart(
            sim_name_1d: Optional[str],
            sim_name_7d: Optional[str],
            n_intervals: int,
            n_clicks: Optional[int]
        ):
            """Update all charts and refresh available sims."""
            ctx = callback_context
            triggered_by = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

            # When clearing a dropdown triggers this callback, skip recomputing charts
            if triggered_by in ('sim-selector-1d', 'sim-selector-7d'):
                triggered_value = sim_name_1d if triggered_by == 'sim-selector-1d' else sim_name_7d
                if not triggered_value:
                    return (no_update,) * 17

            # Selecting one dropdown clears the other
            val_1d, val_7d = sim_name_1d, sim_name_7d
            if triggered_by == 'sim-selector-1d' and sim_name_1d:
                val_7d = None
            elif triggered_by == 'sim-selector-7d' and sim_name_7d:
                val_1d = None

            sim_name = val_1d or val_7d

            # Refresh sim lists on button click
            if triggered_by == 'refresh-button':
                old_sim_names = self.sim_names
                self.sim_names_1d, self.sim_names_7d = self.get_simcomp_sims()
                self.sim_names = self.sim_names_1d + self.sim_names_7d
            options_1d = [{'label': name, 'value': name} for name in self.sim_names_1d]
            options_7d = [{'label': name, 'value': name} for name in self.sim_names_7d]

            empty = go.Figure()
            if not sim_name:
                status_msg = f"Refreshed - {len(self.sim_names)} sims available" if triggered_by == 'refresh-button' else "No simulation selected"
                return (empty,) * 12 + (status_msg, options_1d, options_7d, val_1d, val_7d)

            logger.info(f"[CALLBACK] Updating chart for {sim_name} (refresh #{n_intervals})")

            fig, status = self.create_pnl_comparison_figure(sim_name)
            daily_bar_fig = self.create_daily_delta_bar_chart(sim_name)
            heatmap_fig = self.create_symbol_day_delta_heatmap(sim_name)
            volume_heatmap_fig = self.create_trade_volume_delta_heatmap(sim_name)
            alpha_heatmap_fig = self.create_alpha_delta_heatmap(sim_name)

            alpha_120_fig = self.create_alpha_horizon_delta_heatmap(sim_name, 'alpha_120')
            alpha_360_fig = self.create_alpha_horizon_delta_heatmap(sim_name, 'alpha_360')
            alpha_720_fig = self.create_alpha_horizon_delta_heatmap(sim_name, 'alpha_720')
            alpha_1440_fig = self.create_alpha_horizon_delta_heatmap(sim_name, 'alpha_1440')
            alpha_4320_fig = self.create_alpha_horizon_delta_heatmap(sim_name, 'alpha_4320')
            alpha_10080_fig = self.create_alpha_horizon_delta_heatmap(sim_name, 'alpha_10080')
            alpha_43200_fig = self.create_alpha_horizon_delta_heatmap(sim_name, 'alpha_43200')

            if triggered_by == 'refresh-button':
                new_sims = set(self.sim_names) - set(old_sim_names)
                if new_sims:
                    status = f"Refreshed - New sim(s): {', '.join(sorted(new_sims, reverse=True))}. {status}"
                else:
                    status = f"Refreshed - {len(self.sim_names)} sims available (no new sims). {status}"

            return (fig, daily_bar_fig, heatmap_fig, volume_heatmap_fig, alpha_heatmap_fig,
                    alpha_120_fig, alpha_360_fig, alpha_720_fig, alpha_1440_fig,
                    alpha_4320_fig, alpha_10080_fig, alpha_43200_fig,
                    status, options_1d, options_7d, val_1d, val_7d)


def main():
    """Run the dashboard."""
    parser = argparse.ArgumentParser(description='PnL Comparison Dashboard')
    parser.add_argument('-p', '--port', help='Port to run on', type=int, default=None)
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    # Determine port
    if args.port:
        port = args.port
    else:
        port = 8055 if not LOCAL else 8065  # Use different port in local development

    app = PnlComparisonApp(port=port, debug=args.debug)
    app.run()


if __name__ == "__main__":
    main()
