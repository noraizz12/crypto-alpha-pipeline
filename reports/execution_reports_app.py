"""Execution Monitoring Dashboard - Standalone Application.

Displays real-time execution quality metrics: fill rates, order statistics,
latency, rejections, and open orders across rolling time windows.
"""

import argparse
import logging
from typing import List

from dash import html, dcc, Input, Output
from dash.dash_table import DataTable, Format, FormatTemplate

from lib.reports.base_dash_app import BaseDashApp
from lib.reports.execution_reports import ExecutionReports
from lib.util.util import LOCAL

FMT_MONEY = FormatTemplate.money(2)
FMT_PCT = FormatTemplate.percentage(2)
FMT_FLOAT1 = Format.Format(precision=1, scheme=Format.Scheme.fixed)
FMT_FLOAT2 = Format.Format(precision=2, scheme=Format.Scheme.fixed)

logger = logging.getLogger(__name__)

DISPLAY_WINDOWS = ['today', '24h', '1h', '15min']

# ---------------------------------------------------------------
# Column definitions for DataTables
# ---------------------------------------------------------------

FILL_ORDER_COLS = [
    {'id': 'side', 'name': 'Side'},
    {'id': 'window', 'name': 'Window'},
    {'id': 'fill_dollars', 'name': 'Fill $',
     'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'fill_count', 'name': 'Fill #', 'type': 'numeric'},
    {'id': 'order_dollars', 'name': 'Order $',
     'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'order_count', 'name': 'Order #', 'type': 'numeric'},
    {'id': 'fill_pct', 'name': 'Fill %',
     'type': 'numeric', 'format': FMT_PCT},
    {'id': 'fills_per_order', 'name': 'Fills/Order',
     'type': 'numeric', 'format': FMT_FLOAT2},
]

FILL_BY_SYMBOL_COLS = [
    {'id': 'symbol', 'name': 'Symbol'},
    {'id': 'side', 'name': 'Side'},
    {'id': 'order_dollars', 'name': 'Order $',
     'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'order_count', 'name': 'Order #', 'type': 'numeric'},
    {'id': 'fill_dollars', 'name': 'Fill $',
     'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'fill_count', 'name': 'Fill #', 'type': 'numeric'},
    {'id': 'fill_pct', 'name': 'Fill %',
     'type': 'numeric', 'format': FMT_PCT},
    {'id': 'fills_per_order', 'name': 'Fills/Order',
     'type': 'numeric', 'format': FMT_FLOAT2},
]

AGGRESSION_COLS = [
    {'id': 'aggression', 'name': 'Aggression'},
    {'id': 'order_count', 'name': 'Order #', 'type': 'numeric'},
    {'id': 'order_dollars', 'name': 'Order $',
     'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'fill_count', 'name': 'Fill #', 'type': 'numeric'},
    {'id': 'fill_dollars', 'name': 'Fill $',
     'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'fill_pct', 'name': 'Fill %',
     'type': 'numeric', 'format': FMT_PCT},
]

LATENCY_COLS = [
    {'id': 'window', 'name': 'Window'},
    {'id': 'count', 'name': 'Count', 'type': 'numeric'},
    {'id': 'mean_ms', 'name': 'Mean (ms)',
     'type': 'numeric', 'format': FMT_FLOAT1},
    {'id': 'median_ms', 'name': 'Median (ms)',
     'type': 'numeric', 'format': FMT_FLOAT1},
    {'id': 'p95_ms', 'name': 'P95 (ms)',
     'type': 'numeric', 'format': FMT_FLOAT1},
    {'id': 'p99_ms', 'name': 'P99 (ms)',
     'type': 'numeric', 'format': FMT_FLOAT1},
]

REJECTION_COLS = [
    {'id': 'symbol', 'name': 'Symbol'},
    {'id': 'side', 'name': 'Side'},
    {'id': 'reason', 'name': 'Reason'},
    {'id': 'order_type', 'name': 'Order Type'},
    {'id': 'tif', 'name': 'TIF'},
    {'id': 'source', 'name': 'Source'},
    {'id': 'count', 'name': 'Count', 'type': 'numeric'},
    {'id': 'total_qty', 'name': 'Total Qty',
     'type': 'numeric', 'format': FMT_FLOAT1},
    {'id': 'total_notional', 'name': 'Total $',
     'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'last_ts', 'name': 'Last Time'},
]

OPEN_ORDER_COLS = [
    {'id': 'symbol', 'name': 'Symbol'},
    {'id': 'side', 'name': 'Side'},
    {'id': 'qty', 'name': 'Qty', 'type': 'numeric'},
    {'id': 'px', 'name': 'Price',
     'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'aggression', 'name': 'Aggression', 'type': 'numeric'},
    {'id': 'notional', 'name': 'Notional',
     'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'created_ts', 'name': 'Created'},
]

TABLE_STYLE = {
    'sort_action': 'native',
    'style_table': {'overflowX': 'auto', 'width': '100%'},
    'style_header': {
        'backgroundColor': '#f8f9fa',
        'fontWeight': 'bold',
        'border': '1px solid #dee2e6',
    },
    'style_cell': {
        'textAlign': 'right',
        'padding': '8px',
        'border': '1px solid #dee2e6',
        'fontSize': '13px',
    },
    'style_cell_conditional': [
        {'if': {'column_id': col}, 'textAlign': 'left'}
        for col in [
            'symbol', 'side', 'window', 'reason',
            'order_type', 'tif', 'source',
        ]
    ],
}


def _card(title: str, value: str, subtitle: str = '') -> html.Div:
    """Create a summary card component."""
    return html.Div([
        html.H4(title, style={
            'color': '#666', 'margin': '0 0 5px 0', 'fontSize': '13px',
        }),
        html.H2(value, style={'margin': '0', 'fontSize': '22px'}),
        html.P(subtitle, style={
            'color': '#999', 'margin': '5px 0 0 0', 'fontSize': '11px',
        }),
    ], style={
        'backgroundColor': 'white',
        'padding': '15px',
        'borderRadius': '5px',
        'border': '1px solid #dee2e6',
        'flex': '1',
        'margin': '0 8px',
        'textAlign': 'center',
    })


def _section_header(text: str) -> html.H3:
    """Create a section header."""
    return html.H3(text, style={
        'borderBottom': '2px solid #007BFF',
        'paddingBottom': '8px',
        'marginTop': '30px',
        'marginBottom': '15px',
        'fontSize': '16px',
    })


def _fmt_money(val: float) -> str:
    """Format a number as money string."""
    if abs(val) >= 1_000_000:
        return f"${val / 1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"${val / 1_000:.1f}K"
    return f"${val:.0f}"


class ExecutionReportsApp(BaseDashApp):
    """Execution Monitoring Dashboard Application."""

    def __init__(self, port: int = 8058, interval_secs: int = 60,
                 debug: bool = False):
        """Initialize Execution Monitoring Dashboard.

        Args:
            port: Port to run the application on
            interval_secs: Refresh interval in seconds
            debug: Whether to run in debug mode
        """
        super().__init__(
            "Execution Monitor", port, interval_secs, debug,
        )

        self.reports = ExecutionReports(debug=debug)

        self.setup_layout()
        self.register_callbacks()

        logger.info("Execution Monitoring Dashboard initialized")

    def setup_layout(self) -> None:
        """Setup the layout for the Execution Monitoring dashboard."""
        self.app.layout = html.Div([
            self.create_loading_overlay(),
            self.create_header(),

            html.Div([
                # Refresh controls
                dcc.Interval(
                    id='interval-component',
                    interval=self.interval_secs * 1000,
                    n_intervals=0,
                ),
                html.Div([
                    dcc.Loading(
                        id='loading-refresh',
                        children=[
                            html.Div(
                                id='status-message',
                                style={'color': 'blue'},
                            ),
                            html.Button(
                                'Refresh Data',
                                id='refresh-button', n_clicks=0,
                            ),
                            html.Div(
                                id='refresh-state',
                                style={'display': 'none'},
                            ),
                        ],
                        type='default',
                    ),
                ], style={'marginBottom': '15px'}),

                html.Div(id='loaded-ts-text'),

                # Summary cards
                _section_header('Summary'),
                html.Div(id='summary-cards', style={
                    'display': 'flex', 'marginBottom': '20px',
                }),

                # Fill / Order table
                _section_header('Fills & Orders by Side / Window'),
                DataTable(
                    id='fill-order-table',
                    columns=FILL_ORDER_COLS, **TABLE_STYLE,
                ),

                # Fill % by symbol
                _section_header('Fill % by Symbol (15min)'),
                DataTable(
                    id='fill-by-symbol-15min-table',
                    columns=FILL_BY_SYMBOL_COLS, **TABLE_STYLE,
                ),
                _section_header('Fill % by Symbol (24h)'),
                DataTable(
                    id='fill-by-symbol-table',
                    columns=FILL_BY_SYMBOL_COLS, **TABLE_STYLE,
                ),

                # Aggression breakdown
                _section_header('Aggression Breakdown (15min)'),
                DataTable(
                    id='aggression-15min-table',
                    columns=AGGRESSION_COLS, **TABLE_STYLE,
                ),
                _section_header('Aggression Breakdown (24h)'),
                DataTable(
                    id='aggression-table',
                    columns=AGGRESSION_COLS, **TABLE_STYLE,
                ),

                # Latency stats
                _section_header('Order Latency (acked - created)'),
                DataTable(
                    id='latency-table',
                    columns=LATENCY_COLS, **TABLE_STYLE,
                ),

                # Rejections
                _section_header('Rejections (15min)'),
                DataTable(
                    id='rejections-15min-table',
                    columns=REJECTION_COLS, **TABLE_STYLE,
                ),
                _section_header('Rejections (12h)'),
                DataTable(
                    id='rejections-table',
                    columns=REJECTION_COLS, **TABLE_STYLE,
                ),

                # Open orders
                _section_header('Open Orders (approx)'),
                DataTable(
                    id='open-orders-table',
                    columns=OPEN_ORDER_COLS, **TABLE_STYLE,
                ),

            ], style={'padding': '0 20px', 'paddingBottom': '40px'}),

            dcc.Interval(
                id='init-timer', interval=1000,
                n_intervals=0, max_intervals=2,
            ),
        ])

    def register_callbacks(self) -> None:
        """Register all Dash callbacks."""

        @self.app.callback(
            Output('loading-overlay', 'style'),
            Input('init-timer', 'n_intervals'),
        )
        def hide_loading(n: int) -> dict:
            if n >= 1:
                return {'display': 'none'}
            return {
                'position': 'fixed', 'top': 0, 'left': 0,
                'width': '100%', 'height': '100%',
                'backgroundColor': 'rgba(255, 255, 255, 0.95)',
                'zIndex': 9999, 'display': 'block',
            }

        @self.app.callback(
            [Output('refresh-state', 'children'),
             Output('refresh-button', 'disabled'),
             Output('status-message', 'children'),
             Output('status-message', 'style')],
            [Input('interval-component', 'n_intervals'),
             Input('refresh-button', 'n_clicks')],
            prevent_initial_call=True,
        )
        def manual_refresh(n_intervals: int, n_clicks: int):
            logger.info(
                "Refresh triggered n_intervals=%s, n_clicks=%s",
                n_intervals, n_clicks,
            )
            return self.handle_refresh(
                self.reports.load_data, 'execution',
            )

        @self.app.callback(
            Output('loaded-ts-text', 'children'),
            Input('refresh-state', 'children'),
        )
        def update_ts_display(_):
            return self.reports.get_ts_display()

        @self.app.callback(
            Output('summary-cards', 'children'),
            Input('refresh-state', 'children'),
        )
        def update_summary_cards(_) -> List:
            return self._build_summary_cards()

        @self.app.callback(
            Output('fill-order-table', 'data'),
            Input('refresh-state', 'children'),
        )
        def update_fill_order_table(_) -> List[dict]:
            return self._build_fill_order_rows()

        @self.app.callback(
            Output('fill-by-symbol-15min-table', 'data'),
            Input('refresh-state', 'children'),
        )
        def update_fill_by_symbol_15min(_) -> List[dict]:
            df = self.reports.compute_fill_pct_by_symbol('15min')
            if df.empty:
                return []
            return df.to_dict('records')

        @self.app.callback(
            Output('fill-by-symbol-table', 'data'),
            Input('refresh-state', 'children'),
        )
        def update_fill_by_symbol(_) -> List[dict]:
            if self.reports.fill_pct_by_symbol_df.empty:
                return []
            return self.reports.fill_pct_by_symbol_df.to_dict(
                'records',
            )

        @self.app.callback(
            Output('aggression-15min-table', 'data'),
            Input('refresh-state', 'children'),
        )
        def update_aggression_15min(_) -> List[dict]:
            agg_df = self.reports.compute_aggression_table('15min')
            if agg_df.empty:
                return []
            return agg_df.to_dict('records')

        @self.app.callback(
            Output('aggression-table', 'data'),
            Input('refresh-state', 'children'),
        )
        def update_aggression(_) -> List[dict]:
            agg_df = self.reports.compute_aggression_table('24h')
            if agg_df.empty:
                return []
            return agg_df.to_dict('records')

        @self.app.callback(
            Output('latency-table', 'data'),
            Input('refresh-state', 'children'),
        )
        def update_latency(_) -> List[dict]:
            return list(self.reports.latency_stats.values())

        @self.app.callback(
            Output('rejections-15min-table', 'data'),
            Input('refresh-state', 'children'),
        )
        def update_rejections_15min(_) -> List[dict]:
            rej_df = self.reports.compute_rejection_summary('15min')
            if rej_df.empty:
                return []
            rej_df['last_ts'] = rej_df['last_ts'].astype(str)
            return rej_df.to_dict('records')

        @self.app.callback(
            Output('rejections-table', 'data'),
            Input('refresh-state', 'children'),
        )
        def update_rejections(_) -> List[dict]:
            rej_df = self.reports.compute_rejection_summary('12h')
            if rej_df.empty:
                return []
            rej_df['last_ts'] = rej_df['last_ts'].astype(str)
            return rej_df.to_dict('records')

        @self.app.callback(
            Output('open-orders-table', 'data'),
            Input('refresh-state', 'children'),
        )
        def update_open_orders(_) -> List[dict]:
            if self.reports.open_orders_df.empty:
                return []
            result_df = self.reports.open_orders_df.copy()
            result_df['created_ts'] = (
                result_df['created_ts'].astype(str)
            )
            return result_df.to_dict('records')

    # ------------------------------------------------------------------
    # Helper methods for building table data
    # ------------------------------------------------------------------

    def _build_summary_cards(self) -> List:
        """Build summary card components from current metrics."""
        cards = []
        for window in DISPLAY_WINDOWS:
            metrics = self.reports.summary[window]
            fill_buy = metrics['fill_buy_dollars']
            fill_sell = metrics['fill_sell_dollars']
            fill_buy_c = metrics['fill_buy_count']
            fill_sell_c = metrics['fill_sell_count']
            ord_buy = metrics['order_buy_dollars']
            ord_sell = metrics['order_sell_dollars']
            fill_total = fill_buy + fill_sell
            fill_count = fill_buy_c + fill_sell_c
            ord_total = ord_buy + ord_sell
            fill_pct = (
                fill_total / ord_total * 100
                if ord_total > 0 else 0
            )
            ord_count = (
                metrics['order_buy_count']
                + metrics['order_sell_count']
            )

            if window == 'today':
                # Match Slack format: "Today's Fills $X, Count: Y"
                cards.append(_card(
                    title="Today's Fills",
                    value=f"${fill_total:,.0f}, Count: {fill_count}",
                    subtitle=(
                        f"Orders: {_fmt_money(ord_total)} ({ord_count}) | "
                        f"Fill%: {fill_pct:.1f}%"
                    ),
                ))
            else:
                cards.append(_card(
                    title=f'{window} Window',
                    value=f"{_fmt_money(fill_total)} filled",
                    subtitle=(
                        f"Orders: {_fmt_money(ord_total)} ({ord_count}) | "
                        f"Fill%: {fill_pct:.1f}% | "
                        f"Fills: {fill_count}"
                    ),
                ))
        return cards

    def _build_fill_order_rows(self) -> List[dict]:
        """Build fill/order breakdown rows."""
        rows = []
        for window in DISPLAY_WINDOWS:
            metrics = self.reports.summary[window]
            for side in ['buy', 'sell']:
                rows.append({
                    'side': side.upper(),
                    'window': window,
                    'fill_dollars': metrics[f'fill_{side}_dollars'],
                    'fill_count': metrics[f'fill_{side}_count'],
                    'order_dollars': metrics[f'order_{side}_dollars'],
                    'order_count': metrics[f'order_{side}_count'],
                    'fill_pct': metrics[f'fill_pct_{side}'],
                    'fills_per_order': metrics[
                        f'fills_per_order_{side}'
                    ],
                })
        return rows


def main():
    """Main entry point for Execution Monitoring Dashboard."""
    parser = argparse.ArgumentParser(
        description='Execution Monitoring Dashboard',
    )
    parser.add_argument(
        '-p', '--port', help='Port to run on',
        type=int, default=None,
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Run in debug mode',
    )
    parser.add_argument(
        '-i', '--interval', help='Refresh interval in seconds',
        type=int, default=60,
    )
    args = parser.parse_args()

    port = args.port if args.port else (8058 if not LOCAL else 8068)

    app = ExecutionReportsApp(
        port=port,
        interval_secs=args.interval,
        debug=args.debug,
    )

    logger.info(
        "Starting Execution Monitoring Dashboard on port %s", port,
    )
    app.run(debug=args.debug)


if __name__ == '__main__':
    main()
