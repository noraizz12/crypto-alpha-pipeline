from datetime import datetime as dt, timedelta as td, date
from typing import Optional, List, Tuple, Dict
import logging

import pandas as pd

from lib.data import DataLoader
from lib.util import TRADING_START_DT, today, yesterday, merge_on_index, to_datetime
from lib.util.dataframes import make_date, make_quintile

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PNL_BREAKDOWN_FEATURES_BARS = ['dvolume_1440', 'spread_avg_1440']
PNL_BREAKDOWN_FEATURES_FEATURES = ['dvolume_1440_lz', 'day_of_week', 'logret_1440_cz', 'logret_1440_lz', 'logret_1440_lz_cz', 'relative_updates_1440_trmean', 'trade_sz_1440']
PNL_BREAKDOWN_FEATURES = ['close_mid'] + PNL_BREAKDOWN_FEATURES_BARS + PNL_BREAKDOWN_FEATURES_FEATURES


class FillBreakdown:
    """Analyzes trading fill breakdowns by various feature dimensions.

    This class provides functionality to break down trading fills by different market
    features (e.g., volume, spread, returns) to understand which market conditions
    contribute to positive or negative PnL.

    Attributes:
        start (datetime): Start date for analysis
        end (datetime): End date for analysis
        fills_df (pd.DataFrame): DataFrame containing fill data
        features_df (pd.DataFrame): DataFrame containing market features
        features_dfs (List[pd.DataFrame]): List of feature DataFrames
        features_name_list (List[str]): List of available feature column names
        bars_name_list (List[str]): List of available bar column names
        data_loader (DataLoader): Instance for loading market data
    """
    def __init__(self, start: Optional[dt] = None, end: Optional[dt] = None):
        """Initialize FillBreakdown analyzer.

        Args:
            start: Start datetime for analysis (default: TRADING_START_DT)
            end: End datetime for analysis (default: today)

        Raises:
            AssertionError: If start date is after end date
        """
        logger.info(f"Creating Fill Breakdown from {start} to {end}")
        self.start = start or TRADING_START_DT
        self.end = end or today()
        assert self.start <= self.end
        self.fills_df = None
        self.features_df = None
        self.features_dfs = []
        self.features_name_list = []
        self.bars_name_list = []
        self.data_loader = DataLoader()

    def load_fills(self, fills_df: Optional[pd.DataFrame]):
        """Load fills data for analysis.

        Args:
            fills_df: DataFrame containing fill data with columns including
                     'symbol_venue', 'date', 'realized_pnl', 'expanding'
        """
        fills_df = make_date(fills_df)
        self.fills_df = fills_df

    def _load_bars_data(self, start: date, end: date, bars_cols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Load bar data for the specified date range.

        Attempts to load 1440-minute (daily) bars. If data is not available for the
        requested dates, tries to load from the previous day and shifts dates forward.

        Args:
            start: Start date for bars
            end: End date for bars
            bars_cols: Optional list of specific columns to load

        Returns:
            DataFrame with bar data or None if no data available
        """
        if bars_cols is None or len(bars_cols) > 0:
            bars_df = self.data_loader.load_bars(horizon=1440, start_date=start, end_date=end, cols=bars_cols)
            if bars_df is None:
                bars_df = self.data_loader.load_bars(horizon=1440, start_date=start - td(days=1), end_date=end - td(days=1), cols=bars_cols)

            if bars_df is not None:
                bars_df = make_date(bars_df)
                bars_df['date'] = bars_df['date'] + td(days=1)
                return bars_df
        return None

    def _load_features_data(self, start: date, end: date, features_col: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Load feature data for the specified date range.

        Attempts to load features with 1440-minute horizon. If data is not available
        for the requested dates, tries to load from the previous day and shifts dates forward.

        Args:
            start: Start date for features
            end: End date for features
            features_col: Optional list of specific columns to load

        Returns:
            DataFrame with feature data or None if no data available
        """
        features_df = self.data_loader.load_features(horizons=[1440], start_date=start, end_date=end, cols=features_col)
        if features_df is None:
            features_df = self.data_loader.load_features(horizons=[1440], start_date=start - td(days=1), end_date=end - td(days=1), cols=features_col)

        if features_df is not None:
            features_df = make_date(features_df)
            features_df['date'] = features_df['date'] + td(days=1)
            return features_df
        return None

    def _get_start_end(self) -> Tuple[date, date]:
        """Get adjusted start and end dates for data loading.

        If both start and end are today, adjusts start to yesterday to ensure
        data availability.

        Returns:
            Tuple of (start_date, end_date)
        """
        start = yesterday() if (self.start == self.end == today()) else self.start
        return start.date(), self.end.date()

    def load_features_df(self, cols: Optional[List[str]] = None, update_name_list: bool = False):
        """Load and merge bars and features data.

        Args:
            cols: Optional list of specific columns to load. If provided, filters
                  columns into bars_cols and features_col based on name lists.
            update_name_list: If True, only loads one day of data to update the
                             available column name lists
        """
        logger.info(f"Loading features for pnl stats {update_name_list=}")
        start, end = self._get_start_end()

        # only need one day data to get the list of all bars/features names
        bars_cols = features_col = None
        if update_name_list:
            end = start
        elif cols is not None:
            bars_cols = [c for c in cols if c in self.bars_name_list]
            features_col = [c for c in cols if c in self.features_name_list]

        bars_df = self._load_bars_data(start, end, bars_cols)
        features_df = self._load_features_data(start, end, features_col)

        if update_name_list:
            if bars_df is not None:
                self.bars_name_list = list(bars_df.columns.unique())
            if features_df is not None:
                self.features_name_list = list(features_df.columns.unique())
        else:
            if bars_df is not None and features_df is not None:
                self.features_df = merge_on_index(bars_df, features_df)
            elif bars_df is not None:
                self.features_df = bars_df
            elif features_df is not None:
                self.features_df = features_df

    def win_ratio(self, start_date: Optional[date] = None) -> Tuple[float, int, int, float, float]:
        """Calculate win ratio and related statistics for contracting trades.

        Only considers contracting trades (trades that reduce position size) for
        win ratio calculation.

        Args:
            start_date: Optional date to filter trades from this date onwards

        Returns:
            Tuple containing:
                - win_ratio: Proportion of profitable contracting trades (0-1)
                - num_profit_trades: Number of profitable contracting trades
                - num_contracting_trades: Total number of contracting trades
                - num_trades: Total number of trades (including expanding)
                - gain_per_fill: Average profit per profitable trade
                - loss_per_fill: Average loss per losing trade
        """
        num_trades, num_contracting_trades, num_profit_trades, win_ratio, gain_per_fill, loss_per_fill = 0, 0, 0, 0, 0, 0

        if self.fills_df is not None and len(self.fills_df) > 0:
            num_trades = len(self.fills_df)
            contracting_fills_df = self.fills_df[(~self.fills_df['expanding'])]
            if start_date is not None:
                contracting_fills_df = contracting_fills_df.loc[contracting_fills_df['date'] >= to_datetime(start_date)]
                num_trades = len(self.fills_df.loc[self.fills_df['date'] >= to_datetime(start_date)])
            num_profit_trades = len(contracting_fills_df.loc[contracting_fills_df['realized_pnl'] > 0])
            num_contracting_trades = len(contracting_fills_df)

        if num_contracting_trades > 0:
            win_ratio = len(contracting_fills_df.loc[contracting_fills_df['realized_pnl'] > 0]) / num_contracting_trades
            gain_per_fill = contracting_fills_df.loc[contracting_fills_df['realized_pnl'] > 0, 'realized_pnl'].mean()
            loss_per_fill = contracting_fills_df.loc[contracting_fills_df['realized_pnl'] < 0, 'realized_pnl'].mean()
        return win_ratio, num_profit_trades, num_contracting_trades, num_trades, gain_per_fill, loss_per_fill

    def get_pnl_breakdowns(self, col: Optional[str] = None, merge_on_ts: bool = False, use_cum_pnl: bool = False) -> Dict[str, pd.DataFrame]:
        """Generate PnL breakdowns by specified features.

        Args:
            col: Specific column to analyze. If None, uses PNL_BREAKDOWN_FEATURES
            merge_on_ts: If True, merges on timestamp level rather than date level
            use_cum_pnl: If True, returns cumulative PnL by feature quintile

        Returns:
            Dictionary mapping feature names to their PnL breakdown DataFrames
        """
        ret = {}
        breakdown_cols = [col] if col is not None else PNL_BREAKDOWN_FEATURES
        for bcol in breakdown_cols:
            df = self.pnl_by_col(bcol, by_date=use_cum_pnl, merge_on_ts=merge_on_ts)
            if df is not None:
                ret[bcol] = df if not use_cum_pnl else df.unstack().fillna(0).cumsum().stack(future_stack=True).reset_index()
        return ret

    def pnl_by_col(self, col: str, by_date: bool = False, merge_on_ts: bool = False) -> Optional[pd.DataFrame]:
        """Calculate PnL breakdown by quintiles of a specific column.

        Divides the specified feature into quintiles and aggregates realized PnL
        for each quintile to understand which feature values contribute to profits.

        Args:
            col: Column name to analyze
            by_date: If True, groups by date and quintile; otherwise just quintile
            merge_on_ts: If True, merges on timestamp rather than date

        Returns:
            DataFrame with PnL aggregated by feature quintiles, or None if error
        """
        logger.info(f"Pnl by {col}")
        if self.features_df is None or self.fills_df is None:
            return None
        try:
            if not merge_on_ts:
                time_col = 'date'
                col_df = self.features_df.groupby([time_col, 'symbol_venue'], observed=False).agg({col: 'last'}).reset_index()
            else:
                time_col = 'ts'
                col_df = self.features_df
            df = pd.merge(self.fills_df, col_df, on=['symbol_venue', time_col], how='left')

            df = make_quintile(df, col)
            qcol = f'{col}_quintile'
            groupby = [time_col, qcol] if by_date else qcol
            df = df.groupby(groupby, observed=False).agg({'realized_pnl': 'sum'})
            if not by_date:
                df = df.reset_index().sort_values(by=qcol)
            return df
        except ValueError as ve:
            logger.error(ve)
            return None

    def runall(self):
        """Run complete fill breakdown analysis.

        Loads features, calculates win ratio, and generates PnL breakdowns
        for all default features. Prints results to console.
        """
        self.load_features_df()
        print(f"Win Ratio: {self.win_ratio()}")
        breakdowns = self.get_pnl_breakdowns()
        for col, breakdown in breakdowns.items():
            print(f"\nPnl by {col}")
            print(breakdown)
