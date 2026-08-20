"""Slippage analysis and execution quality reporting module.

This module provides comprehensive slippage analysis functionality for trade
execution quality measurement. It analyzes the difference between intended
and actual execution prices, measuring both fill slippage (executed trades)
and opportunity cost (unexecuted trades). Provides breakdowns by symbol,
time period, and aggression level.

Classes:
    SlippageReports: Main class for trade execution quality and slippage analysis
"""

import logging
from datetime import datetime as dt, timezone, timedelta as td
from typing import List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html
from pandas import to_datetime

from lib.reports.markouts import FillMarkouts
from lib.trader.trading import Order
from lib.util.config import get_config
from lib.util.time_util import today, today_date
from lib.util.util import fmoney

logger = logging.getLogger(__name__)


class SlippageReports:
    """Reports for trade execution quality and slippage analysis.

    Analyzes the difference between intended and actual execution prices,
    measuring both fill slippage (executed trades) and opportunity cost
    (unexecuted trades). Provides breakdowns by symbol, time period,
    and aggression level.

    Attributes:
        config: Main system configuration
        fill_markouts: FillMarkouts instance for slippage analysis
        today_start_dt: Today's start datetime
        today_date: Today's date
        markouts_lookback_days: Number of days to look back for markouts
        update_ts: Timestamp of last data update
        order_alpha_df: DataFrame with order-level alpha data
        order_slippage_df: DataFrame with slippage metrics
        fill_slippage_df: DataFrame with fill-level slippage
        opp_slippage_df: DataFrame with opportunity cost metrics

    Example:
        >>> sr = SlippageReports(fill_markouts)
        >>> sr.update()
        >>> tables = sr.get_slippage_tables()
    """
    def __init__(self, fill_markouts: FillMarkouts) -> None:
        """Initialize with FillMarkouts instance"""
        # Initialize config
        _, self.config = get_config()

        # Store fill markouts instance
        self.fill_markouts = fill_markouts

        # Date management attributes
        self.today_start_dt = today()
        self.today_date = today_date()

        # Constants
        self.markouts_lookback_days = 90  # MARKOUTS_LOOKBACK_DAYS

        # Update timestamp
        self.update_ts = dt.now(timezone.utc)

        # Initialize slippage-specific data
        self.shortfall_results = {}
        self.slippage_df = None
        self.current_vwap_df = None
        self.avg_vwap_df = None
        self.vwap_shortfall_trailing_avg = 0
        self.vwap_shortfall_perc_trailing_avg = 0
        self.daily_vwapshortfall_stats = {
            'start': self.today_date,
            'shortfall': None,
            'shortfall_bps': None,
        }

        self.update_data()

    def get_ts_display(self, n_state: str) -> str:
        logger.info(f"finish get_ts_display for update {n_state} at {dt.now(timezone.utc)}")
        return f'Data as of: {self.update_ts.strftime("%Y%m%d %H:%M")}, dashboard refreshed at {dt.now(timezone.utc).strftime("%Y%m%d %H:%M")}'

    def update_data(self) -> None:
        """Initialize the markouts data"""
        # Update timestamp
        self.update_ts = dt.now(timezone.utc)
        self.update_shortfall_results()
        self.process_vwap_data()

    def update_shortfall_results(self) -> None:
        # Check if fill_markouts has required data before proceeding
        if self.fill_markouts.bars_df is None:
            logger.warning("FillMarkouts does not have bars_df initialized. Skipping shortfall calculation.")
            return

        if self.fill_markouts.fills_df is None:
            logger.warning("FillMarkouts does not have fills_df initialized. Skipping shortfall calculation.")
            return

        if self.fill_markouts.raw_orders_df is None:
            logger.warning("FillMarkouts does not have raw_orders_df initialized. Skipping shortfall calculation.")
            return

        aggression_levels = [None] + list(range(Order.MIN_AGGRESSION, Order.MAX_AGGRESSION + 1))
        for aggression_level in aggression_levels:
            for compare_px_type in ['vwap', 'start_px']:
                try:
                    vwap_df = self.fill_markouts.calculate_vwap_shortfall(aggression_level=aggression_level, compare_px_type=compare_px_type)
                    self.shortfall_results[(aggression_level, compare_px_type)] = vwap_df
                except Exception as e:
                    logger.error(f"Error calculating VWAP shortfall for aggression {aggression_level}, type {compare_px_type}: {e}")
                    # Initialize empty DataFrame to prevent subsequent errors
                    self.shortfall_results[(aggression_level, compare_px_type)] = pd.DataFrame()

    def process_vwap_data(self) -> None:
        """Process VWAP data after calculating shortfall results"""
        vwap_df = self.shortfall_results.get((None, 'vwap'))
        if vwap_df is None or vwap_df.empty:
            logger.warning("No VWAP data available to process")
            return
        # Process average VWAP data
        self.avg_vwap_df = vwap_df.groupby('date', observed=False).agg(
            {'fill_slip': 'sum', 'fill_dollars_abs': 'sum'})
        # Filter out rows with zero fill_dollars_abs to avoid division by zero
        self.avg_vwap_df = self.avg_vwap_df[self.avg_vwap_df['fill_dollars_abs'] > 0]
        if self.avg_vwap_df.empty:
            logger.warning("No valid VWAP data after filtering zero fill_dollars_abs")
            return
        self.avg_vwap_df['vwap_shortfall_bps'] = self.avg_vwap_df['fill_slip'] / self.avg_vwap_df['fill_dollars_abs'] * 10000
        self.avg_vwap_df.index = to_datetime(self.avg_vwap_df.index)

        # Calculate trailing averages with division by zero protection
        self.vwap_shortfall_trailing_avg = self.avg_vwap_df['fill_slip'].rolling(window='30D').mean().iloc[-1]
        trailing_dollars = self.avg_vwap_df['fill_dollars_abs'].rolling(window='30D').mean().iloc[-1]
        if trailing_dollars > 0:
            self.vwap_shortfall_perc_trailing_avg = self.vwap_shortfall_trailing_avg / trailing_dollars
        else:
            self.vwap_shortfall_perc_trailing_avg = 0
        # Process current VWAP data
        max_ts = vwap_df['ts'].max()
        self.current_vwap_df = vwap_df[vwap_df['ts'] == max_ts].reset_index()
        if 'target_ts' in self.current_vwap_df.columns:
            self.daily_vwapshortfall_stats['start'] = self.current_vwap_df.target_ts.unique()[0]
        # Calculate daily cumulative VWAP stats
        daily_cum_vwap = vwap_df.groupby(['symbol_venue', 'date'], observed=False).agg({'fill_slip': 'sum', 'fill_dollars_abs': 'sum'})
        if daily_cum_vwap.empty:
            logger.warning("No daily cumulative VWAP data available")
            return
        max_date = daily_cum_vwap.index.get_level_values('date').max()
        daily_cum_vwap = daily_cum_vwap.loc[daily_cum_vwap.index.get_level_values('date') == max_date]

        if not daily_cum_vwap.empty:
            # Filter out rows with zero fill_dollars_abs to avoid division by zero
            daily_cum_vwap = daily_cum_vwap[daily_cum_vwap['fill_dollars_abs'] > 0]
            if daily_cum_vwap.empty:
                return
            daily_cum_vwap['daily_vwap_shortfall_bps'] = (daily_cum_vwap['fill_slip'] / daily_cum_vwap['fill_dollars_abs']) * 10000
            self.daily_vwapshortfall_stats['shortfall'] = daily_cum_vwap['fill_slip'].sum()
            total_dollars = daily_cum_vwap['fill_dollars_abs'].sum()
            if total_dollars > 0:
                self.daily_vwapshortfall_stats['shortfall_bps'] = (self.daily_vwapshortfall_stats['shortfall'] / total_dollars) * 10000

            daily_cum_vwap = daily_cum_vwap.reset_index()[['symbol_venue', 'fill_slip', 'daily_vwap_shortfall_bps']].rename(columns={'fill_slip': 'daily_vwap_shortfall'})
            self.current_vwap_df = pd.merge(self.current_vwap_df, daily_cum_vwap, on='symbol_venue', how='left')
            self.current_vwap_df = self.current_vwap_df.loc[~self.current_vwap_df['daily_vwap_shortfall_bps'].isna()]

    def update_slippage_graph(self) -> go.Figure:
        vwap_df = self.shortfall_results.get((None, 'vwap'))
        if vwap_df is None or vwap_df.empty:
            slippage_fig = go.Figure()
            slippage_fig.update_layout(title="Total Slippage over Time - No Data Available")
            return slippage_fig

        aggregate_vwap_df = vwap_df.groupby('date').agg({'total_slip': 'mean', 'order_dollars_abs': 'mean'})
        # Filter out rows with zero order_dollars_abs to avoid division by zero
        aggregate_vwap_df = aggregate_vwap_df[aggregate_vwap_df['order_dollars_abs'] > 0]
        if aggregate_vwap_df.empty:
            slippage_fig = go.Figure()
            slippage_fig.update_layout(title="Total Slippage over Time - No Data Available")
            return slippage_fig
        aggregate_vwap_df['total_slip_bps'] = aggregate_vwap_df['total_slip'] / aggregate_vwap_df['order_dollars_abs'] * 10000
        slippage_fig = px.line(aggregate_vwap_df, y='total_slip_bps', title="Total Slippage over Time")
        return slippage_fig

    def update_slippage_table(self) -> List[dict]:
        data_records = []
        aggression_levels = [None] + list(range(Order.MIN_AGGRESSION, Order.MAX_AGGRESSION + 1))
        for aggression_level in aggression_levels:
            row_data = {'aggression_level': str(aggression_level) if aggression_level is not None else 'All'}
            for compare_px_type in ['vwap', 'start_px']:
                vwap_df = self.shortfall_results.get((aggression_level, compare_px_type))
                if vwap_df is not None and not vwap_df.empty:
                    total_order_dollars = vwap_df['order_dollars_abs'].sum().sum()
                    if total_order_dollars > 0:
                        for case in ['opp_slip', 'fill_slip', 'total_slip']:
                            row_data[f'{case}_bps_{compare_px_type}'] = vwap_df[case].sum().sum() / total_order_dollars * 10000
                    else:
                        for case in ['opp_slip', 'fill_slip', 'total_slip']:
                            row_data[f'{case}_bps_{compare_px_type}'] = 0
                    row_data[f'total_traded_dollars_{compare_px_type}'] = vwap_df['fill_dollars_abs'].sum().sum()
            data_records.append(row_data)
        total_slippage_df = pd.DataFrame(data_records)
        order_mapping = {str(level) if level is not None else 'All': i for i, level in enumerate(aggression_levels)}
        total_slippage_df = total_slippage_df.sort_values(by='aggression_level', key=lambda x: x.map(order_mapping))
        return total_slippage_df.reset_index().to_dict('records')

    def update_slippage_df(self, aggression_level: Optional[int] = None, compare_px_type: str = 'vwap') -> None:
        vwap_df = self.shortfall_results.get((aggression_level, compare_px_type))
        if vwap_df is None or vwap_df.empty:
            self.slippage_df = pd.DataFrame(columns=['symbol_venue', 'total_slip', 'opp_slip', 'fill_slip', 'order_dollars_abs',
                                                      'total_slip_bps', 'opp_slip_bps', 'fill_slip_bps', 'fill_to_opp_ratio'])
            return

        self.slippage_df = vwap_df[['symbol_venue', 'total_slip', 'opp_slip', 'fill_slip', 'order_dollars_abs']]
        self.slippage_df = self.slippage_df.groupby('symbol_venue', observed=False).sum().reset_index()
        self.slippage_df = self.slippage_df.loc[self.slippage_df.order_dollars_abs != 0]
        for case in ['opp_slip', 'fill_slip', 'total_slip']:
            self.slippage_df[f'{case}_bps'] = self.slippage_df[case] / self.slippage_df['order_dollars_abs'] * 10000
        self.slippage_df['fill_to_opp_ratio'] = self.slippage_df['fill_slip'] / self.slippage_df['opp_slip']

    def update_slippage_data(self, n_state: str) -> Tuple[List[str], List[dict], List[dict], List[dict], List[dict], go.Figure]:
        slippage_summary_dict = {'slip': [], 'slip_bps': []}
        slippage_tables = {}
        for case in ['vwap', 'start_px']:
            self.update_slippage_df(compare_px_type=case)
            slippage_tables[case] = self.slippage_df.to_dict('records')
            order_dollars_abs = self.slippage_df['order_dollars_abs'].sum() if len(self.slippage_df) > 0 else 0
            for flip_case in ['fill_slip', 'opp_slip', 'total_slip']:
                slip_sum = self.slippage_df[flip_case].sum() if flip_case in self.slippage_df.columns else 0
                slippage_summary_dict['slip'].append(slip_sum)
                if order_dollars_abs > 0:
                    slippage_summary_dict['slip_bps'].append(slip_sum / order_dollars_abs * 10000)
                else:
                    slippage_summary_dict['slip_bps'].append(0)

        slippage_summary_df = pd.DataFrame(
            slippage_summary_dict,
            index=['Fill Slip VWAP', 'OPP Slip VWAP', 'TOTAL Slip VWAP', 'Fill Slip START PX', 'OPP Slip START PX', 'TOTAL Slip START PX'],
        )
        slippage_summary_table = slippage_summary_df.reset_index().to_dict('records')
        slippage_headline = [
            "Slippage by Symbol All aggression Aggregated",
            *[html.Br() for _ in range(2)],
            f"{self.markouts_lookback_days}-day Trailing From {self.today_start_dt - td(days=self.markouts_lookback_days)} to {self.today_start_dt}",
        ]
        total_slippage_table = self.update_slippage_table()
        total_slippage_graph = self.update_slippage_graph()
        logger.info(f"finish update_slippage_data for update {n_state} at {dt.now(timezone.utc)}")
        return slippage_headline, slippage_summary_table, slippage_tables['vwap'], slippage_tables['start_px'], total_slippage_table, total_slippage_graph

    def update_shortfall_data(self, n_state: str) -> Tuple[List[str], go.Figure]:
        logger.info(f"finish update_shortfall_period for update {n_state} at {dt.now(timezone.utc)}")

        # Handle case when no VWAP data is available
        if self.avg_vwap_df is None or self.avg_vwap_df.empty:
            shortfall_headline = [
                f'VWAP Shortfall from {self.today_start_dt - td(days=self.markouts_lookback_days)} to {self.today_start_dt}',
                *[html.Br() for _ in range(2)],
                "No VWAP data available",
            ]
            vwap_fig = go.Figure()
            vwap_fig.update_layout(title="VWAP Shortfall vs Total Traded Dollars in bps - No Data Available")
            return shortfall_headline, vwap_fig

        shortfall_headline = [
            f'VWAP Shortfall from {self.today_start_dt - td(days=self.markouts_lookback_days)} to {self.today_start_dt}',
            *[html.Br() for _ in range(2)],
            f"{self.markouts_lookback_days} day trailing average {fmoney(self.vwap_shortfall_trailing_avg)}, or {self.vwap_shortfall_perc_trailing_avg * 10000:.2f}bps",
        ]
        vwap_fig = px.line(self.avg_vwap_df, y='vwap_shortfall_bps', title="VWAP Shortfall vs Total Traded Dollars in bps")
        return shortfall_headline, vwap_fig

    def update_vwap_data(self, _) -> Tuple[List[str], List[dict]]:
        # Handle case when no VWAP data is available
        if self.current_vwap_df is None or self.current_vwap_df.empty:
            vwap_headline = [
                f"VWAP Shortfall since {self.daily_vwapshortfall_stats['start']}",
                *[html.Br() for _ in range(2)],
                "No VWAP data available",
            ]
            return vwap_headline, []

        shortfall_val = self.daily_vwapshortfall_stats.get('shortfall')
        shortfall_bps = self.daily_vwapshortfall_stats.get('shortfall_bps')

        if shortfall_val is None or shortfall_bps is None:
            shortfall_text = "No shortfall data available"
        else:
            shortfall_text = f"Today's Total Shortfall {fmoney(shortfall_val)}, {shortfall_bps:.2f} Bps"

        vwap_headline = [
            f"VWAP Shortfall since {self.daily_vwapshortfall_stats['start']}",
            *[html.Br() for _ in range(2)],
            shortfall_text,
        ]
        return vwap_headline, self.current_vwap_df.to_dict('records')
