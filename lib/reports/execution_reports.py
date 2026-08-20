"""Execution monitoring reports: order/fill statistics, latency, rejections.

Loads and parses trading data files (orders, fills, raw_oms)
to compute execution quality metrics across rolling time windows.

Classes:
    ExecutionReports: Main class for execution monitoring data and metrics
"""

import ast
import logging
import os
from datetime import datetime as dt, timezone, timedelta as td
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

from lib.util.directory import dir_manager
from lib.util.time_util import date_to_str, today, today_date

logger = logging.getLogger(__name__)

ROLLING_WINDOWS = ['today', '24h', '1h', '15min']

WINDOW_TIMEDELTAS = {
    '24h': td(hours=24),
    '1h': td(hours=1),
    '15min': td(minutes=15),
    '5min': td(minutes=5),
    '12h': td(hours=12),
}

API_REJECTIONS_URL = (
    "http://statarb-prod-exe.live.sh:9503/api/rejections"
)

_EMPTY_LATENCY = {
    'mean_ms': 0, 'median_ms': 0, 'p95_ms': 0, 'p99_ms': 0, 'count': 0,
}


def _split_line(line: str) -> list:
    """Split pipe-delimited line, converting 'None' to np.nan."""
    parts = line.strip().split('|')
    return [np.nan if p == 'None' else p for p in parts]


def _parse_ts(series: pd.Series) -> pd.Series:
    """Parse timestamp column to datetime.

    Uses format='mixed' because timestamps have inconsistent fractional
    seconds (e.g. '2026-02-22 02:45:41+00:00' vs '...02:45:41.123+00:00').
    """
    return pd.to_datetime(series, format='mixed', utc=True)


def _get_cutoff(window_key: str) -> dt:
    """Get cutoff timestamp for a window key."""
    if window_key == 'today':
        return today().to_pydatetime()
    return dt.now(timezone.utc) - WINDOW_TIMEDELTAS[window_key]


def _filter_by_ts(
    source_df: pd.DataFrame, ts_col: str, cutoff: dt,
) -> pd.DataFrame:
    """Filter dataframe to rows where ts_col >= cutoff."""
    if source_df.empty:
        return pd.DataFrame()
    return source_df[source_df[ts_col] >= cutoff]


def _join_fills_to_orders(
    orders_df: pd.DataFrame, fills_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join fills to orders via orderId/oid (Binance order ID).

    Fill files store om['i'] (Binance order ID) as orderId.
    Order files store om['i'] as oid, om['c'] as koid.
    Join on oid since both originate from om['i'].
    """
    if orders_df.empty or fills_df.empty:
        return pd.DataFrame()

    order_keys_df = orders_df[['oid', 'side', 'aggression']].copy()
    order_keys_df['oid'] = order_keys_df['oid'].astype(str)
    order_keys_df = order_keys_df.drop_duplicates('oid')

    fills_copy_df = fills_df.copy()
    fills_copy_df['orderId'] = fills_copy_df['orderId'].astype(str)

    merged_df = fills_copy_df.merge(
        order_keys_df, left_on='orderId', right_on='oid',
        how='inner', suffixes=('_fill', '_order'),
    )
    merged_df['side'] = merged_df['side_order']
    return merged_df


class ExecutionReports:
    """Execution monitoring data layer.

    Loads trading data files and computes execution quality metrics
    including fill rates, order statistics, latency, and rejections.
    """

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.update_ts: dt = dt.now(timezone.utc)

        # DataFrames — always set by load_data(), never None after init
        self.orders_df: pd.DataFrame = pd.DataFrame()
        self.fills_df: pd.DataFrame = pd.DataFrame()
        self.cancels_df: pd.DataFrame = pd.DataFrame()
        self.rejections_df: pd.DataFrame = pd.DataFrame()

        # Computed metrics
        self.summary: Dict[str, dict] = {}
        self.fill_pct_by_symbol_df: pd.DataFrame = pd.DataFrame()
        self.latency_stats: Dict[str, dict] = {}
        self.open_orders_df: pd.DataFrame = pd.DataFrame()

        self.load_data()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_orders(
        self, date_str: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and parse orders file for a given date.

        Note: lib.data.load_orders() exists but sets acked_ts from exch_ts
        (bug), breaking latency calculation. It also parses alpha columns
        we don't need. Custom loader is intentional.

        Args:
            date_str: Date string in YYYYMMDD format

        Returns:
            Tuple of (orders_df, cancels_df)
        """
        path = os.path.join(
            dir_manager.ORDERS_DIR, date_str,
            f"orders.{date_str}.csv",
        )
        if not os.path.isfile(path):
            logger.warning("Orders file not found: %s", path)
            return pd.DataFrame(), pd.DataFrame()

        rows: List[dict] = []
        cancel_rows: List[dict] = []
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                parts = _split_line(line)
                if not parts:
                    continue
                rec = parts[0]
                if rec == 'ORDER' and len(parts) >= 18:
                    rows.append(self._parse_order_row(parts))
                elif rec == 'CANCEL' and len(parts) >= 6:
                    cancel_rows.append(self._parse_cancel_row(parts))

        orders_df = pd.DataFrame(rows)
        cancels_df = pd.DataFrame(cancel_rows)

        if not orders_df.empty:
            orders_df = self._type_orders(orders_df)
        if not cancels_df.empty:
            cancels_df = self._type_cancels(cancels_df)

        return orders_df, cancels_df

    @staticmethod
    def _parse_order_row(parts: list) -> dict:
        """Parse a single ORDER row from pipe-delimited parts."""
        return {
            'koid': parts[1], 'oid': parts[2], 'symbol': parts[3],
            'side': parts[4], 'order_type': parts[5], 'tif': parts[6],
            'qty': parts[7], 'px': parts[8], 'qty_at_time': parts[9],
            'aggression': parts[10], 'deficit': parts[11],
            'created_ts': parts[12], 'exch_ts': parts[13],
            'acked_ts': parts[14], 'bid': parts[15], 'ask': parts[16],
            'opt_px': parts[17],
        }

    @staticmethod
    def _parse_cancel_row(parts: list) -> dict:
        """Parse a single CANCEL row from pipe-delimited parts."""
        row = {
            'recv_ts': parts[1], 'symbol': parts[2],
            'oid': parts[3], 'qty': parts[4], 'px': parts[5],
        }
        # Cancel format evolved — remaining_qty and koid added later
        if len(parts) >= 7:
            row['remaining_qty'] = parts[6]
        if len(parts) >= 8:
            row['koid'] = parts[7]
        return row

    @staticmethod
    def _type_orders(orders_df: pd.DataFrame) -> pd.DataFrame:
        """Apply correct dtypes to orders dataframe."""
        for col in ['qty', 'px', 'bid', 'ask', 'opt_px',
                     'deficit', 'qty_at_time']:
            orders_df[col] = pd.to_numeric(
                orders_df[col],
            ).astype(np.float32)
        orders_df['aggression'] = pd.to_numeric(
            orders_df['aggression'],
        ).astype('Int64')
        for col in ['created_ts', 'exch_ts', 'acked_ts']:
            orders_df[col] = _parse_ts(orders_df[col])
        orders_df['notional'] = (
            orders_df['qty'] * orders_df['px']
        ).abs()
        return orders_df

    @staticmethod
    def _type_cancels(cancels_df: pd.DataFrame) -> pd.DataFrame:
        """Apply correct dtypes to cancels dataframe."""
        for col in ['qty', 'px']:
            cancels_df[col] = pd.to_numeric(
                cancels_df[col],
            ).astype(np.float32)
        if 'remaining_qty' in cancels_df.columns:
            cancels_df['remaining_qty'] = pd.to_numeric(
                cancels_df['remaining_qty'],
            ).astype(np.float32)
        cancels_df['recv_ts'] = _parse_ts(cancels_df['recv_ts'])
        return cancels_df

    def load_fills(self, date_str: str) -> pd.DataFrame:
        """Load internal fills file for a given date.

        Uses trading/fills/ which updates more frequently than
        binance_fills, keeping the dashboard closer to real-time.

        Note: lib.data.load_oms_fills() exists but uses different
        column names (fill_px, fill_qty) and adds signed qty.
        Custom loader keeps columns consistent with our metrics code.

        Format: FILL|exch_ts|recv_ts|symbol|venue|fill_type|side|
                px|qty|commission|asset|oid|opt_px|notional

        Args:
            date_str: Date string in YYYYMMDD format

        Returns:
            DataFrame of fills with standard column names
        """
        path = os.path.join(
            dir_manager.FILLS_DIR,
            f"fills.{date_str}.csv",
        )
        if not os.path.isfile(path):
            logger.warning("Fills file not found: %s", path)
            return pd.DataFrame()

        rows: List[dict] = []
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                parts = _split_line(line)
                if len(parts) < 14 or parts[0] != 'FILL':
                    continue
                rows.append({
                    'ts_raw': parts[1],
                    'symbol': parts[3],
                    'side': parts[6],
                    'price': parts[7],
                    'qty': parts[8],
                    'orderId': parts[11],
                    'notional_raw': parts[13],
                })

        if not rows:
            return pd.DataFrame()

        fills_df = pd.DataFrame(rows)
        fills_df['price'] = pd.to_numeric(
            fills_df['price'],
        ).astype(np.float32)
        fills_df['qty'] = pd.to_numeric(
            fills_df['qty'],
        ).astype(np.float32)
        fills_df['ts'] = _parse_ts(fills_df['ts_raw'])
        fills_df['notional'] = pd.to_numeric(
            fills_df['notional_raw'],
        ).astype(np.float32).abs()
        fills_df.drop(columns=['ts_raw', 'notional_raw'], inplace=True)
        return fills_df

    def load_rejections(self, date_str: str) -> pd.DataFrame:
        """Parse raw OMS file for CANCELED events (rejections).

        Args:
            date_str: Date string in YYYYMMDD format

        Returns:
            DataFrame of rejection events
        """
        path = os.path.join(
            dir_manager.RAW_OMS_DIR, f"oms.{date_str}.txt",
        )
        if not os.path.isfile(path):
            logger.warning("Raw OMS file not found: %s", path)
            return pd.DataFrame()

        rows = self._parse_oms_cancels(path)
        if not rows:
            return pd.DataFrame()

        rejections_df = pd.DataFrame(rows)
        rejections_df['ts'] = pd.to_datetime(
            rejections_df['live_ts_ms'], unit='ms', utc=True,
        )
        rejections_df['exch_ts'] = pd.to_datetime(
            rejections_df['exch_ts_ms'], unit='ms', utc=True,
        )
        return rejections_df

    @staticmethod
    def _parse_oms_cancels(path: str) -> List[dict]:
        """Extract CANCELED events from raw OMS file."""
        rows: List[dict] = []
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    logger.warning(
                        "Failed to parse OMS line: %.100s", line,
                    )
                    continue
                order_msg = msg.get('o', {})
                if order_msg.get('x', '') != 'CANCELED':
                    continue
                rows.append({
                    'symbol': order_msg['s'],
                    'oid': str(order_msg['i']),
                    'koid': order_msg['c'],
                    'side': order_msg['S'],
                    'order_type': order_msg['o'],
                    'tif': order_msg['f'],
                    'reason': order_msg.get('V', 'UNKNOWN'),
                    'qty': float(order_msg['q']),
                    'px': float(order_msg['p']),
                    'live_ts_ms': msg.get('live_ts'),
                    'exch_ts_ms': order_msg.get('T'),
                })
        return rows

    def _load_multi_day(self, days: int = 2) -> None:
        """Load orders, fills, cancels, rejections for multiple days."""
        all_orders, all_fills = [], []
        all_cancels, all_rejections = [], []

        for offset in range(days):
            day = today_date() - td(days=offset)
            ds = date_to_str(day)

            orders_df, cancels_df = self.load_orders(ds)
            if not orders_df.empty:
                all_orders.append(orders_df)
            if not cancels_df.empty:
                all_cancels.append(cancels_df)

            fills_df = self.load_fills(ds)
            if not fills_df.empty:
                all_fills.append(fills_df)

            rej_df = self.load_rejections(ds)
            if not rej_df.empty:
                all_rejections.append(rej_df)

        self.orders_df = (
            pd.concat(all_orders, ignore_index=True)
            if all_orders else pd.DataFrame()
        )
        self.fills_df = (
            pd.concat(all_fills, ignore_index=True)
            if all_fills else pd.DataFrame()
        )
        self.cancels_df = (
            pd.concat(all_cancels, ignore_index=True)
            if all_cancels else pd.DataFrame()
        )
        self.rejections_df = (
            pd.concat(all_rejections, ignore_index=True)
            if all_rejections else pd.DataFrame()
        )

    def load_api_rejections(self) -> pd.DataFrame:
        """Fetch API-level rejections (-5022) from prod-exe."""
        _side_map = {'BUY': 'B', 'SELL': 'S'}
        try:
            resp = requests.get(API_REJECTIONS_URL, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning("API rejections unreachable: %s", exc)
            return pd.DataFrame()

        rejections = data['rejections']
        if not rejections:
            return pd.DataFrame()

        rows: List[dict] = []
        for rec in rejections:
            rows.append({
                'symbol': rec['symbol'],
                'oid': '',
                'side': _side_map[rec['side']],
                'reason': str(rec['error_code']),
                'order_type': rec['order_type'],
                'tif': rec['time_in_force'],
                'qty': float(rec['quantity']),
                'px': float(rec['price']),
                'ts': pd.to_datetime(rec['timestamp'], utc=True),
                'source': 'api',
            })

        rej_df = pd.DataFrame(rows)
        rej_df['notional'] = (rej_df['qty'] * rej_df['px']).abs()
        return rej_df

    # ------------------------------------------------------------------
    # Metric computations
    # ------------------------------------------------------------------

    def _compute_side_metrics(
        self, result: dict, orders_w_df: pd.DataFrame,
        matched_fills_df: pd.DataFrame, side_label: str,
        side_code: str,
    ) -> None:
        """Compute order/fill metrics for one side (buy or sell)."""
        if not orders_w_df.empty:
            side_ord = orders_w_df[orders_w_df['side'] == side_code]
            ord_dollars = float(side_ord['notional'].sum())
            ord_count = len(side_ord)
        else:
            ord_dollars = 0.0
            ord_count = 0

        result[f'order_{side_label}_dollars'] = ord_dollars
        result[f'order_{side_label}_count'] = ord_count

        if not matched_fills_df.empty:
            side_fill = matched_fills_df[
                matched_fills_df['side'] == side_code
            ]
            fill_dollars = float(side_fill['notional'].sum())
            fill_count = len(side_fill)
        else:
            fill_dollars = 0.0
            fill_count = 0

        result[f'fill_{side_label}_dollars'] = fill_dollars
        result[f'fill_{side_label}_count'] = fill_count

        result[f'fill_pct_{side_label}'] = (
            fill_dollars / ord_dollars if ord_dollars > 0 else 0.0
        )
        result[f'fills_per_order_{side_label}'] = (
            fill_count / ord_count if ord_count > 0 else 0.0
        )

    def compute_rolling_metrics(self, window_key: str) -> dict:
        """Compute fill/order summary metrics for a rolling window.

        Fills are joined to orders so only fills belonging to orders
        within the window are counted, preventing fill % > 100%.

        Args:
            window_key: One of 'today', '24h', '1h', '15min'

        Returns:
            Dict with buy/sell fill $, count, order $, count, fill %
        """
        cutoff = _get_cutoff(window_key)
        result: dict = {'window': window_key}

        orders_w_df = _filter_by_ts(
            self.orders_df, 'created_ts', cutoff,
        )
        fills_w_df = _filter_by_ts(self.fills_df, 'ts', cutoff)

        matched_df = _join_fills_to_orders(orders_w_df, fills_w_df)

        for label, code in [('buy', 'B'), ('sell', 'S')]:
            self._compute_side_metrics(
                result, orders_w_df, matched_df, label, code,
            )

        # Aggression breakdown from orders
        if not orders_w_df.empty:
            agg_grp = orders_w_df.groupby(
                'aggression', observed=False,
            ).agg(
                order_count=('koid', 'size'),
                order_dollars=('notional', 'sum'),
            ).reset_index()
            result['aggression_breakdown'] = agg_grp.to_dict('records')
        else:
            result['aggression_breakdown'] = []

        return result

    def compute_fill_pct_by_symbol(
        self, window_key: str = '24h',
    ) -> pd.DataFrame:
        """Compute fill % breakdown by symbol and side.

        Fills are joined to orders so only fills belonging to orders
        within the window are counted.

        Args:
            window_key: Rolling window key

        Returns:
            DataFrame with fill % by symbol
        """
        cutoff = _get_cutoff(window_key)

        if self.orders_df.empty or self.fills_df.empty:
            return pd.DataFrame()

        orders_w_df = _filter_by_ts(
            self.orders_df, 'created_ts', cutoff,
        )
        fills_w_df = _filter_by_ts(self.fills_df, 'ts', cutoff)

        if orders_w_df.empty:
            return pd.DataFrame()

        order_agg_df = orders_w_df.groupby(
            ['symbol', 'side'], observed=False,
        ).agg(
            order_dollars=('notional', 'sum'),
            order_count=('koid', 'size'),
        ).reset_index()

        matched_df = _join_fills_to_orders(orders_w_df, fills_w_df)
        if not matched_df.empty:
            fill_agg_df = matched_df.groupby(
                ['symbol', 'side'], observed=False,
            ).agg(
                fill_dollars=('notional', 'sum'),
                fill_count=('orderId', 'size'),
            ).reset_index()
        else:
            fill_agg_df = pd.DataFrame(
                columns=[
                    'symbol', 'side', 'fill_dollars', 'fill_count',
                ],
            )

        merged_df = order_agg_df.merge(
            fill_agg_df, on=['symbol', 'side'], how='left',
        )
        # fillna(0): no matching fills = zero fills, not missing data
        merged_df['fill_dollars'] = merged_df['fill_dollars'].fillna(0)
        merged_df['fill_count'] = merged_df['fill_count'].fillna(0)
        merged_df['fill_pct'] = np.where(
            merged_df['order_dollars'] > 0,
            merged_df['fill_dollars'] / merged_df['order_dollars'],
            0.0,
        )
        merged_df['fills_per_order'] = np.where(
            merged_df['order_count'] > 0,
            merged_df['fill_count'] / merged_df['order_count'],
            0.0,
        )
        return merged_df.sort_values('fill_pct', ascending=True)

    def compute_latency_stats(self, window_key: str) -> dict:
        """Compute order latency stats (acked_ts - created_ts).

        Args:
            window_key: Rolling window key

        Returns:
            Dict with mean, median, p95, p99 latency in ms
        """
        empty = {'window': window_key, **_EMPTY_LATENCY}
        cutoff = _get_cutoff(window_key)

        if self.orders_df.empty:
            return empty

        orders_w_df = _filter_by_ts(
            self.orders_df, 'created_ts', cutoff,
        )
        if orders_w_df.empty:
            return empty

        lat = (
            orders_w_df['acked_ts'] - orders_w_df['created_ts']
        ).dt.total_seconds() * 1000
        lat = lat.dropna()
        lat = lat[(lat >= 0) & (lat < 60000)]

        if lat.empty:
            return empty

        return {
            'window': window_key,
            'mean_ms': round(float(lat.mean()), 1),
            'median_ms': round(float(lat.median()), 1),
            'p95_ms': round(float(lat.quantile(0.95)), 1),
            'p99_ms': round(float(lat.quantile(0.99)), 1),
            'count': int(len(lat)),
        }

    def compute_open_orders(self) -> pd.DataFrame:
        """Estimate currently open orders.

        Approximates open orders by finding recent orders that have
        no matching fill or cancel within a short lookback window.

        Returns:
            DataFrame of approximately open orders
        """
        if self.orders_df.empty:
            return pd.DataFrame()

        cutoff = dt.now(timezone.utc) - td(minutes=30)
        recent_df = _filter_by_ts(
            self.orders_df, 'created_ts', cutoff,
        )
        if recent_df.empty:
            return pd.DataFrame()

        filled_oids = self._get_filled_oids()
        cancelled_oids, cancelled_koids = self._get_cancelled_ids()

        open_mask = ~(
            recent_df['oid'].astype(str).isin(filled_oids)
            | recent_df['oid'].astype(str).isin(cancelled_oids)
            | recent_df['koid'].astype(str).isin(cancelled_koids)
        )
        open_df = recent_df[open_mask]
        if open_df.empty:
            return pd.DataFrame()

        cols = [
            'symbol', 'side', 'qty', 'px',
            'aggression', 'created_ts', 'notional',
        ]
        return open_df[cols].sort_values(
            'created_ts', ascending=False,
        )

    def _get_filled_oids(self) -> set:
        """Collect filled order IDs from fills data."""
        if self.fills_df.empty:
            return set()
        return set(self.fills_df['orderId'].astype(str).unique())

    def _get_cancelled_ids(self) -> Tuple[set, set]:
        """Collect cancelled order IDs and koids."""
        if self.cancels_df.empty:
            return set(), set()
        oids = set(self.cancels_df['oid'].astype(str).unique())
        koids = set(
            self.cancels_df['koid'].dropna().astype(str).unique(),
        )
        return oids, koids

    def compute_rejection_summary(
        self, window_key: str = '12h',
    ) -> pd.DataFrame:
        """Summarize rejections by symbol and reason.

        Args:
            window_key: Rolling window key

        Returns:
            DataFrame with rejection counts by symbol/reason
        """
        if self.rejections_df.empty:
            return pd.DataFrame()

        cutoff = _get_cutoff(window_key)
        rej_w_df = self.rejections_df[
            self.rejections_df['ts'] >= cutoff
        ]
        if rej_w_df.empty:
            return pd.DataFrame()

        rej_w_df = rej_w_df.copy()
        rej_w_df['notional'] = (rej_w_df['qty'] * rej_w_df['px']).abs()
        group_cols = [
            'symbol', 'side', 'reason', 'order_type', 'tif', 'source',
        ]
        summary_df = rej_w_df.groupby(
            group_cols, observed=False,
        ).agg(
            count=('oid', 'size'),
            total_qty=('qty', 'sum'),
            total_notional=('notional', 'sum'),
            last_ts=('ts', 'max'),
        ).reset_index().sort_values('count', ascending=False)

        return summary_df

    def compute_aggression_table(
        self, window_key: str = '24h',
    ) -> pd.DataFrame:
        """Compute order/fill breakdown by aggression level.

        Fills are joined to orders so fill % is accurate.

        Args:
            window_key: Rolling window key

        Returns:
            DataFrame with order/fill stats per aggression level
        """
        cutoff = _get_cutoff(window_key)

        if self.orders_df.empty:
            return pd.DataFrame()

        orders_w_df = _filter_by_ts(
            self.orders_df, 'created_ts', cutoff,
        )
        if orders_w_df.empty:
            return pd.DataFrame()

        agg_df = orders_w_df.groupby(
            'aggression', observed=False,
        ).agg(
            order_count=('koid', 'size'),
            order_dollars=('notional', 'sum'),
        ).reset_index()

        # Join fills to orders for accurate fill attribution
        fills_w_df = _filter_by_ts(self.fills_df, 'ts', cutoff)
        matched_df = _join_fills_to_orders(orders_w_df, fills_w_df)

        if not matched_df.empty:
            fill_agg_df = matched_df.groupby(
                'aggression', observed=False,
            ).agg(
                fill_count=('orderId', 'size'),
                fill_dollars=('notional', 'sum'),
            ).reset_index()
            agg_df = agg_df.merge(
                fill_agg_df, on='aggression', how='left',
            )
            # fillna(0): no fills at this aggression = zero, not missing
            agg_df['fill_count'] = agg_df['fill_count'].fillna(0)
            agg_df['fill_dollars'] = agg_df['fill_dollars'].fillna(0)
        else:
            agg_df['fill_count'] = 0
            agg_df['fill_dollars'] = 0.0

        agg_df['fill_pct'] = np.where(
            agg_df['order_dollars'] > 0,
            agg_df['fill_dollars'] / agg_df['order_dollars'],
            0.0,
        )
        return agg_df.sort_values('aggression')

    # ------------------------------------------------------------------
    # Main refresh
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Main refresh: load files and compute all metrics."""
        self.update_ts = dt.now(timezone.utc)
        logger.info("Loading execution data...")

        self._load_multi_day(days=2)

        if not self.rejections_df.empty:
            self.rejections_df['source'] = 'oms'

        api_rej_df = self.load_api_rejections()
        if not api_rej_df.empty:
            self.rejections_df = pd.concat(
                [self.rejections_df, api_rej_df], ignore_index=True,
            )

        self.summary = {}
        for window in ROLLING_WINDOWS:
            self.summary[window] = self.compute_rolling_metrics(window)

        self.fill_pct_by_symbol_df = self.compute_fill_pct_by_symbol(
            '24h',
        )

        self.latency_stats = {}
        for window in ROLLING_WINDOWS:
            self.latency_stats[window] = self.compute_latency_stats(
                window,
            )

        self.open_orders_df = self.compute_open_orders()
        logger.info("Execution data loaded at %s", self.update_ts)

    def get_ts_display(self) -> str:
        """Return formatted timestamp string for display."""
        now_str = dt.now(timezone.utc).strftime('%Y%m%d %H:%M')
        data_str = self.update_ts.strftime('%Y%m%d %H:%M')
        return (
            f"Data as of: {data_str}, "
            f"dashboard refreshed at {now_str}"
        )
