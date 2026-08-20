"""PnL Report generation utilities.

This module provides utilities for generating PnL reports including
weekly summaries, daily breakdowns, and security-level analysis.
Designed for investor reporting and internal analysis.
"""
import logging
from datetime import datetime as dt
from datetime import timedelta as td
from datetime import timezone
from typing import Optional

import pandas as pd

from lib.pnl_new.binance_pnl import BinancePnl
from lib.util import fmoney
from lib.util.time_util import (
    get_last_complete_week_range, get_week_range,
    get_last_complete_friday_week_range, get_friday_week_range
)

logger = logging.getLogger(__name__)


def _fmt_pct(val: float) -> str:
    """Format percentage with sign prefix for display."""
    sign = '+' if val >= 0 else ''
    return f"{sign}{val:.2f}%"


class WeeklyPnlReport:
    """Weekly PnL report generator for investor letters.

    Generates formatted PnL reports with realized/unrealized breakdown,
    funding income, fees, and top winners/losers by security.
    """

    def __init__(self, config: dict, start_dt: dt, end_dt: dt,
                 single_day: bool = False, custom_range: bool = False,
                 friday_week: bool = False, capital: Optional[float] = None):
        """Initialize the report with date range.

        Args:
            config: Configuration dictionary
            start_dt: Start datetime
            end_dt: End datetime
            single_day: True if this is a single day report
            custom_range: True if this is a custom date range
            friday_week: True if using Friday-Friday week boundaries
            capital: Total investor capital for return calculation (optional)
        """
        self.config = config
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.single_day = single_day
        self.custom_range = custom_range
        self.friday_week = friday_week
        self.capital = capital
        self.pnl: Optional[BinancePnl] = None
        self.portfolio_df: Optional[pd.DataFrame] = None
        self._load_data()

    @classmethod
    def for_week(cls, config: dict, reference_date: Optional[dt] = None,
                 friday_week: bool = False, capital: Optional[float] = None) -> 'WeeklyPnlReport':
        """Create report for a specific week.

        Args:
            config: Configuration dictionary
            reference_date: Any date within the desired week. If None, uses last complete week.
            friday_week: If True, use Friday-Friday week boundaries instead of Monday-Monday.
            capital: Total investor capital for return calculation (optional).

        Returns:
            WeeklyPnlReport instance for the specified week
        """
        if friday_week:
            if reference_date is None:
                start_dt, end_dt = get_last_complete_friday_week_range()
            else:
                if reference_date.tzinfo is None:
                    reference_date = reference_date.replace(tzinfo=timezone.utc)
                start_dt, end_dt = get_friday_week_range(reference_date)
        else:
            if reference_date is None:
                start_dt, end_dt = get_last_complete_week_range()
            else:
                if reference_date.tzinfo is None:
                    reference_date = reference_date.replace(tzinfo=timezone.utc)
                start_dt, end_dt = get_week_range(reference_date)

        return cls(config=config, start_dt=start_dt, end_dt=end_dt,
                   friday_week=friday_week, capital=capital)

    @classmethod
    def for_date(cls, config: dict, report_date: dt,
                 capital: Optional[float] = None) -> 'WeeklyPnlReport':
        """Create report for a single day.

        Args:
            config: Configuration dictionary
            report_date: The date to report on
            capital: Total investor capital for return calculation (optional).

        Returns:
            WeeklyPnlReport instance for the specified day
        """
        if report_date.tzinfo is None:
            report_date = report_date.replace(tzinfo=timezone.utc)

        start_dt = report_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = start_dt + td(days=1)

        return cls(config=config, start_dt=start_dt, end_dt=end_dt,
                   single_day=True, capital=capital)

    @classmethod
    def for_range(cls, config: dict, start_dt: dt, end_dt: dt,
                  capital: Optional[float] = None) -> 'WeeklyPnlReport':
        """Create report for a custom date range.

        Args:
            config: Configuration dictionary
            start_dt: Start date (inclusive)
            end_dt: End date (inclusive)
            capital: Total investor capital for return calculation (optional).

        Returns:
            WeeklyPnlReport instance for the specified range

        Raises:
            ValueError: If start_dt >= end_dt
        """
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)

        start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)

        if start_dt >= end_dt:
            raise ValueError(f"start_dt must be before end_dt: {start_dt} >= {end_dt}")

        # End is exclusive, add 1 day
        end_dt = end_dt + td(days=1)

        return cls(config=config, start_dt=start_dt, end_dt=end_dt,
                   custom_range=True, capital=capital)

    def _load_data(self) -> None:
        """Load PnL data from BinancePnl."""
        logger.info(f"Loading PnL data for {self.start_dt} to {self.end_dt}")
        self.pnl = BinancePnl(config=self.config, start=self.start_dt, end=self.end_dt)
        self.portfolio_df = self.pnl.aggregate_daily_portfolio()
        if self.portfolio_df is None or self.portfolio_df.empty:
            logger.warning("No portfolio PnL data available for period")
            return
        logger.info(f"Loaded {len(self.portfolio_df)} days of portfolio data")

    def get_summary(self) -> dict:
        """Get summary metrics for the period.

        Returns:
            Dictionary with realized_pnl, unrealized_pnl, funding_income,
            commission, net_pnl, and percentage returns
        """
        if self.portfolio_df is None or self.portfolio_df.empty:
            return {
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'funding_income': 0.0,
                'commission': 0.0,
                'net_pnl': 0.0,
                'net_return_pct': 0.0,
                'realized_return_pct': 0.0,
                'unrealized_return_pct': 0.0,
                'capital_return_pct': 0.0,
            }

        realized_pnl = self.portfolio_df['realized_pnl'].sum()
        unrealized_pnl = self.portfolio_df['unrealized_pnl'].sum()
        funding_income = self.portfolio_df['funding_income'].sum()
        commission = self.portfolio_df['commission'].sum()
        net_pnl = realized_pnl + unrealized_pnl - commission + funding_income

        # Calculate percentage returns using daily returns (pnl / gross_notional)
        # net_pnl already includes realized + unrealized - commission + funding_income
        # realized/unrealized percentages show component breakdown (won't sum to net due to fees/funding)
        # Cumulative return = product of (1 + daily_return) - 1
        safe_notional = self.portfolio_df['gross_notional'].replace(0, float('nan'))
        daily_net_ret = (self.portfolio_df['net_pnl'] / safe_notional).fillna(0)
        daily_realized_ret = (self.portfolio_df['realized_pnl'] / safe_notional).fillna(0)
        daily_unrealized_ret = (self.portfolio_df['unrealized_pnl'] / safe_notional).fillna(0)

        net_return_pct = ((1 + daily_net_ret).prod() - 1) * 100
        realized_return_pct = ((1 + daily_realized_ret).prod() - 1) * 100
        unrealized_return_pct = ((1 + daily_unrealized_ret).prod() - 1) * 100

        # Calculate return on total investor capital if specified
        capital_return_pct = (net_pnl / self.capital) * 100 if self.capital else 0.0

        return {
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'funding_income': funding_income,
            'commission': commission,
            'net_pnl': net_pnl,
            'net_return_pct': net_return_pct,
            'realized_return_pct': realized_return_pct,
            'unrealized_return_pct': unrealized_return_pct,
            'capital_return_pct': capital_return_pct,
        }

    def get_top_winners(self, n: int = 5) -> pd.DataFrame:
        """Get top N winning securities."""
        if self.pnl is None:
            return pd.DataFrame()
        return self.pnl.get_top_winners(n)

    def get_top_losers(self, n: int = 5) -> pd.DataFrame:
        """Get top N losing securities."""
        if self.pnl is None:
            return pd.DataFrame()
        return self.pnl.get_top_losers(n)

    def format_report(self) -> str:
        """Format the full report as a string.

        Returns:
            Formatted report string
        """
        if self.portfolio_df is None or self.portfolio_df.empty:
            return "No PnL data available for the specified period"

        summary = self.get_summary()
        lines = []

        # Header
        if self.single_day:
            title = "DAILY PNL REPORT"
            period = f"Date: {self.start_dt.strftime('%Y-%m-%d')} (UTC)"
        elif self.custom_range:
            title = "PNL REPORT"
            end_display = self.end_dt - td(days=1)
            period = f"Period: {self.start_dt.strftime('%Y-%m-%d')} to {end_display.strftime('%Y-%m-%d')} (UTC)"
        elif self.friday_week:
            title = "WEEKLY PNL REPORT"
            end_display = self.end_dt - td(days=1)
            period = f"Period: {self.start_dt.strftime('%Y-%m-%d')} to {end_display.strftime('%Y-%m-%d')} (Fri-Fri UTC)"
        else:
            title = "WEEKLY PNL REPORT"
            end_display = self.end_dt - td(days=1)
            period = f"Period: {self.start_dt.strftime('%Y-%m-%d')} to {end_display.strftime('%Y-%m-%d')} (Mon-Sun UTC)"

        lines.extend([
            "",
            "=" * 60,
            title,
            "=" * 60,
            period,
            "=" * 60,
            f"Realized PnL:    {fmoney(summary['realized_pnl'], thousands_sep=True):>15}  ({_fmt_pct(summary['realized_return_pct']):>8})",
            f"Unrealized PnL:  {fmoney(summary['unrealized_pnl'], thousands_sep=True):>15}  ({_fmt_pct(summary['unrealized_return_pct']):>8})",
            f"Funding Income:  {fmoney(summary['funding_income'], thousands_sep=True):>15}",
            f"Fees:            {fmoney(summary['commission'], thousands_sep=True):>15}",
            "=" * 60,
            f"Net PnL:         {fmoney(summary['net_pnl'], thousands_sep=True):>15}  ({_fmt_pct(summary['net_return_pct']):>8})",
        ])

        # Add capital-based return if capital was specified
        if self.capital is not None:
            lines.append(f"Return on Capital ({fmoney(self.capital, thousands_sep=True)}): {_fmt_pct(summary['capital_return_pct']):>8}")

        lines.extend([
            "=" * 60,
            ""
        ])

        # Daily breakdown (not for single day)
        if not self.single_day:
            lines.append("Daily breakdown:")
            cols = ['realized_pnl', 'unrealized_pnl', 'funding_income', 'commission', 'net_pnl']
            lines.append(self.portfolio_df[cols].to_string())
            lines.append("")

        # Top winners/losers
        winners_df = self.get_top_winners()
        losers_df = self.get_top_losers()

        if not winners_df.empty:
            lines.extend([
                "",
                "=" * 50,
                "TOP 5 WINNERS",
                "=" * 50,
            ])
            for symbol, row in winners_df.iterrows():
                lines.append(f"{symbol:15} {fmoney(row['net_pnl'], thousands_sep=True):>12}")

        if not losers_df.empty:
            lines.extend([
                "",
                "=" * 50,
                "TOP 5 LOSERS",
                "=" * 50,
            ])
            for symbol, row in losers_df.iterrows():
                lines.append(f"{symbol:15} {fmoney(row['net_pnl'], thousands_sep=True):>12}")
            lines.append("=" * 50)

        return "\n".join(lines)

    def print_report(self) -> None:
        """Print the formatted report to stdout."""
        print(self.format_report())
