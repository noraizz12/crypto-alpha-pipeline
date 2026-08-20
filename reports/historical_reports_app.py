"""Historical Reports Dashboard - Standalone Application."""

import argparse
import logging
from datetime import date
from typing import Optional

import plotly.graph_objects as go
from dash import html, dcc, Input, Output
from dash.dash_table import Format, FormatTemplate, DataTable

from lib.reports.base_dash_app import BaseDashApp
from lib.reports.hist_trading_reports import HistTradingReports
from lib.util.config import get_config
from lib.util.time_util import yesterday_date, date_str_to_date
from lib.util.util import LOCAL, PNL_START_DATE

FMT_MONEY = FormatTemplate.money(2)
FMT_PERCENT = FormatTemplate.percentage(2)

DRAWDOWN_COLS = [
    {'id': 'start_date', 'name': 'Start Date'},
    {'id': 'end_date', 'name': 'End Date'},
    {'id': 'drawdown_days', 'name': 'Drawdown Days', 'type': 'numeric'},
    {'id': 'dollar_loss', 'name': 'Dollar Loss', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'percent_loss', 'name': 'Return Loss', 'type': 'numeric', 'format': FMT_PERCENT},
]

HIST_SUMMARY_COLS = [
    {'id': 'metrics', 'name': 'Metrics', 'type': 'text'},
    {'id': 'mtd', 'name': 'Month to Date', 'type': 'text'},
    {'id': 'ytd', 'name': 'Year to Date', 'type': 'text'},
    {'id': 'lifetime', 'name': 'Lifetime', 'type': 'text'},
]

MONTHLY_SUMMARY_COLS = [
    {'id': 'year_month', 'name': 'Month', 'type': 'text'},
    {'id': 'gross_pnl', 'name': 'Gross Pnl', 'type': 'numeric', 'format': FMT_MONEY},
    # {'id': 'notional_abs_daily_mean', 'name': 'Avg Size', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'gross_notional_mean', 'name': 'Avg Size', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'capital_delta', 'name': 'Capital Delta', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'cum_unlev_return', 'name': 'Cum Return', 'type': 'numeric', 'format': FMT_PERCENT},
    {'id': 'market_return', 'name': 'Market Return', 'type': 'numeric', 'format': FMT_PERCENT},
    {'id': 'annualized_unlev_return', 'name': 'Annualized Return', 'type': 'numeric', 'format': FMT_PERCENT},
    {'id': 'annualized_unlev_return_std', 'name': 'Annualized Risk', 'type': 'numeric', 'format': FMT_PERCENT},
    {'id': 'sharpe', 'name': 'Sharpe', 'type': 'numeric', 'format': Format.Format(precision=1, scheme=Format.Scheme.fixed)},
    # {'id': 'fees_usd_daily_sum', 'name': 'Fees', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'commission_sum', 'name': 'Fees', 'type': 'numeric', 'format': FMT_MONEY},
    # {'id': 'funding_income_daily_sum', 'name': 'Fundings', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'funding_income_sum', 'name': 'Fundings', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'net_pnl_sum', 'name': 'Net Pnl', 'type': 'numeric', 'format': FMT_MONEY},
    # {'id': 'total_pnl_daily_sum', 'name': 'Net Pnl', 'type': 'numeric', 'format': FMT_MONEY},
]

logger = logging.getLogger(__name__)


class HistoricalReportsApp(BaseDashApp):
    """Historical Reports Dashboard Application."""

    def __init__(
            self,
            start_date: date,
            end_date: date,
            port: int = 8051,
            interval_secs: int = 120,
            sim_config_file: Optional[str] = None,
            debug: bool = False
    ):
        """
        Initialize Historical Reports Dashboard.

        Args:
            start_date: Start date for historical analysis
            end_date: End date for historical analysis
            port: Port to run the application on
            interval_secs: Refresh interval in seconds (default 120 for historical data)
            sim_config_file: Optional simulation config file
            debug: Whether to run in debug mode
        """
        super().__init__("Historical Reports", port, interval_secs, debug)

        self.start_date = start_date
        self.end_date = end_date

        # Initialize configuration
        _, self.config = get_config(config_file=sim_config_file)

        # Initialize business logic layer
        self.hist_trading_reports = HistTradingReports(
            config=self.config,
            start_date=self.start_date,
            end_date=self.end_date,
            debug=debug
        )

        # Get factors from hist_trading_reports (includes risk factors + alpha columns)
        self.factors = self.hist_trading_reports.factors

        self.setup_layout()
        self.register_callbacks()

        logger.info("Historical Reports Dashboard initialized successfully")

    def setup_layout(self) -> None:
        """Setup the layout for Historical Reports."""
        self.app.layout = html.Div([
            # Loading overlay (hidden after initial load)
            self.create_loading_overlay(),

            # Header with navigation links
            self.create_header(),

            # Main content
            html.Div([
                # Refresh controls
                html.Div([
                    dcc.Interval(
                        id='interval-component',
                        interval=self.interval_secs * 1000,
                        n_intervals=0,
                    ),
                ], style={'marginBottom': '20px'}),

                html.Div([
                    dcc.Loading(
                        id="loading-refresh",
                        children=[
                            html.Div(id='status-message', style={'color': 'blue'}),
                            html.Button("Refresh Data", id='refresh-button', n_clicks=0),
                            html.Div(id='refresh-state', style={'display': 'none'}),
                        ],
                        type="default",
                    ),
                ], style={'marginBottom': '20px'}),

                html.Div(id='loaded-ts-text'),

                # Overall performance metrics
                html.Div([
                    html.H3(children='Historical Pnl', style={'textAlign': 'center'}),
                    DataTable(
                        id='overall-perf-table',
                        columns=HIST_SUMMARY_COLS,
                        sort_action="native",
                    ),
                    dcc.Graph(id='overall-pnl-graph'),
                ], style={'marginBottom': '20px'}),

                # Monthly performance table
                html.Div([
                    html.H3(children='Month Pnl', style={'textAlign': 'center'}),
                    DataTable(
                        id='monthly-perf-table',
                        columns=MONTHLY_SUMMARY_COLS,
                        sort_action="native",
                    ),
                ], style={'marginBottom': '20px'}),

                # Top Drawdowns
                html.Div([
                    html.H3(children='Top Drawdowns', style={'textAlign': 'center'}),
                    DataTable(
                        id='drawdown-table',
                        columns=DRAWDOWN_COLS,
                        sort_action="native",
                    ),
                ], style={'marginBottom': '20px'}),

                # Balance graph
                dcc.Graph(id='balance-graph'),

                # Portfolio & Trading metrics
                html.H3(children='Portfolio & Trading', style={'textAlign': 'center'}),
                dcc.Graph(id='positions-graph'),
                dcc.Graph(id='trading-volume-graph'),
                dcc.Graph(id='turnover-graph'),

                html.Div([
                    html.H3("Return Distribution by Time Period", style={'textAlign': 'center'}),
                    dcc.Graph(id='dow-return-figure'),
                    dcc.Graph(id='hour-return-figure'),
                ], style={'marginBottom': '20px'}),

                # Historical PNL breakdown
                html.H3(children='Historical Pnl Breakdown', style={'textAlign': 'center'}),
                dcc.Graph(id='cum-realized-pnl-symbol'),
                dcc.Graph(id='cum-realized-pnl-trades'),

                # Alpha Opt Statistics Over Time
                html.H3(children='Alpha Opt Statistics', style={'textAlign': 'center'}),
                dcc.Graph(id='alpha-opt-timeseries-graph'),

                # PNL Breakdown by feature
                html.H3(children='Pnl Breakdown by feature', style={'textAlign': 'center'}),
                dcc.Dropdown(id='pnl-breakdown-selection'),
                dcc.Graph(id='pnl-breakdown-graph'),

                # Alpha by Horizon Graphs (all alphas at a horizon on same chart)
                html.H3(children='Alpha Analysis by Horizon', style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id='alpha-horizon-dropdown',
                    options=[{'label': f'Horizon {h} ({h//60}h)' if h >= 60 else f'Horizon {h} ({h}m)',
                              'value': h} for h in self.hist_trading_reports.alpha_horizons],
                    value=self.hist_trading_reports.alpha_horizons[0] if self.hist_trading_reports.alpha_horizons else None,
                    style={"margin-top": "10px"},
                ),
                dcc.Graph(id='alpha-horizon-return-graph', style={'marginBottom': '20px'}),
                dcc.Graph(id='alpha-horizon-exposure-graph', style={'marginBottom': '20px'}),
                dcc.Graph(id='alpha-horizon-port-return-graph', style={'marginBottom': '20px'}),
                dcc.Graph(id='alpha-horizon-pnl-graph', style={'marginBottom': '20px'}),

                # Factor Return/PNL Graphs (individual factors)
                html.H3(children='Factor Return/Pnl Graphs', style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id='factor-dropdown-selection',
                    options=self.factors,
                    value=self.factors[0] if self.factors else None,
                    style={"margin-top": "10px"},
                ),
                dcc.Graph(id='factor-return-graph', style={'marginBottom': '20px'}),
                dcc.Graph(id='port-factor-exposure-graph', style={'marginBottom': '20px'}),
                dcc.Graph(id='port-factor-return-graph', style={'marginBottom': '20px'}),
                dcc.Graph(id='port-factor-pnl-graph', style={'marginBottom': '20px'}),

            ], style={"padding": "0 20px"}),

            # Hidden timer for removing loading overlay
            dcc.Interval(id='init-timer', interval=1000, n_intervals=0, max_intervals=2),
        ])

    def register_callbacks(self) -> None:
        """Register all callbacks for Historical Reports."""

        # Hide loading overlay after initial load
        @self.app.callback(
            Output('loading-overlay', 'style'),
            Input('init-timer', 'n_intervals'),
        )
        def hide_loading_overlay(n: int) -> dict:
            if n >= 1:
                return {'display': 'none'}
            return {
                'position': 'fixed', 'top': 0, 'left': 0,
                'width': '100%', 'height': '100%',
                'backgroundColor': 'rgba(255, 255, 255, 0.95)',
                'zIndex': 9999, 'display': 'block'
            }

        # Refresh callback
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
            logger.info(f"Refresh triggered {n_intervals=}, {n_clicks=}")
            return self.handle_refresh(self.hist_trading_reports.load_data, "historical")

        # Timestamp display
        @self.app.callback(
            Output('loaded-ts-text', 'children'),
            Input('refresh-state', 'children'),
        )
        def get_ts_display(n_state: str) -> str:
            return self.hist_trading_reports.get_ts_display(n_state)

        # Overall performance table
        @self.app.callback(
            Output('overall-perf-table', 'data'),
            Input('refresh-state', 'children'),
        )
        def update_overall_perf(n_state: str):
            return self.hist_trading_reports.get_overall_perf_display(n_state)

        # Overall historical figures
        @self.app.callback(
            Output('overall-pnl-graph', 'figure'),
            Input('refresh-state', 'children')
        )
        def update_overall_hist_figures(n_state: str):
            return self.hist_trading_reports.overall_hist_figure(n_state)

        # PNL breakdown dropdown and graph
        @self.app.callback(
            [Output('pnl-breakdown-selection', 'options'),
             Output('pnl-breakdown-selection', 'value')],
            Input('refresh-state', 'children'),
        )
        def update_fill_breakdown_dropdown(n_state: str):
            return self.hist_trading_reports.update_fill_breakdown_dropdown(n_state)

        @self.app.callback(
            Output('pnl-breakdown-graph', 'figure'),
            [Input('pnl-breakdown-selection', 'value'),
             Input('refresh-state', 'children')],
        )
        def update_hist_pnl_breakdown_figure(value: str, n_state: str) -> go.Figure:
            if value is None:
                return go.Figure()
            return self.hist_trading_reports.hist_pnl_breakdown_figure(value, n_state)

        # Positions graph
        @self.app.callback(
            Output('positions-graph', 'figure'),
            Input('refresh-state', 'children'),
        )
        def update_today_positions_figure(n_state: str) -> go.Figure:
            return self.hist_trading_reports.today_positions_figure(n_state)

        # Cumulative PnL figures
        @self.app.callback(
            [Output('cum-realized-pnl-symbol', 'figure'),
             Output('cum-realized-pnl-trades', 'figure')],
            Input('refresh-state', 'children'),
        )
        def update_cum_pnl_hist_figures(n_state: str):
            return self.hist_trading_reports.cum_pnl_hist_figure(n_state)

        # Daily update figures
        @self.app.callback(
            [Output('trading-volume-graph', 'figure'),
             Output('turnover-graph', 'figure'),
             Output('balance-graph', 'figure')],
            Input('refresh-state', 'children'),
        )
        def update_daily_update_figures(n_state: str):
            return self.hist_trading_reports.daily_update_figures(n_state)

        # Return breakdown figures (day of week and hour)
        @self.app.callback(
            [Output('dow-return-figure', 'figure'),
             Output('hour-return-figure', 'figure')],
            Input('refresh-state', 'children'),
        )
        def update_sim_return_comparison_figures(n_state: str):
            return self.hist_trading_reports.update_return_breakdown_figure(n_state)

        # Alpha by horizon figures (all alphas at a horizon on same charts)
        @self.app.callback(
            [Output('alpha-horizon-return-graph', 'figure'),
             Output('alpha-horizon-exposure-graph', 'figure'),
             Output('alpha-horizon-port-return-graph', 'figure'),
             Output('alpha-horizon-pnl-graph', 'figure')],
            Input('alpha-horizon-dropdown', 'value'),
        )
        def update_alpha_horizon_figures(horizon: int):
            if horizon is None:
                return go.Figure(), go.Figure(), go.Figure(), go.Figure()
            return self.hist_trading_reports.update_alpha_by_horizon_figures(horizon)

        # Factor return figures (individual factors)
        @self.app.callback(
            [Output('factor-return-graph', 'figure'),
             Output('port-factor-exposure-graph', 'figure'),
             Output('port-factor-return-graph', 'figure'),
             Output('port-factor-pnl-graph', 'figure')],
            Input('factor-dropdown-selection', 'value'),
        )
        def update_factor_return_figures(factor: str):
            if factor is None:
                return go.Figure(), go.Figure(), go.Figure(), go.Figure()
            return self.hist_trading_reports.update_factor_return_figure(factor)

        # Monthly performance table
        @self.app.callback(
            Output('monthly-perf-table', 'data'),
            Input('refresh-state', 'children'),
        )
        def update_monthly_perf_table(n_state: str):
            return self.hist_trading_reports.get_monthly_summary_table(n_state)

        # Drawdown table
        @self.app.callback(
            Output('drawdown-table', 'data'),
            Input('refresh-state', 'children'),
        )
        def update_drawdown_table(n_state: str):
            return self.hist_trading_reports.get_drawdown_table(n_state)

        # Alpha Opt Statistics timeseries
        @self.app.callback(
            Output('alpha-opt-timeseries-graph', 'figure'),
            Input('refresh-state', 'children'),
        )
        def update_alpha_opt_timeseries(n_state: str) -> go.Figure:
            return self.hist_trading_reports.alpha_opt_timeseries_figure(n_state)


def main():
    """Main entry point for Historical Reports Dashboard."""
    parser = argparse.ArgumentParser(description='Historical Reports Dashboard')
    parser.add_argument('-p', '--port', help='Port to run on', type=int, default=None)
    parser.add_argument('-g', '--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('-i', '--interval', help='Refresh interval in seconds', type=int, default=120)
    parser.add_argument('-s', '--sim-config', help='Simulation config file', type=str, default=None)
    parser.add_argument('-f', '--from', help='from date YYYYMMDD', required=False, type=int)
    parser.add_argument('-t', '--to', help='to date YYYYMMDD', required=False, type=int)
    args = parser.parse_args()

    if args.port:
        port = args.port
    else:
        port = 8051 if not LOCAL else 8061  # Use different port in local development

    start_date = getattr(args, 'from')
    end_date = getattr(args, 'to')
    start_date = date_str_to_date(start_date) if start_date is not None else PNL_START_DATE
    end_date = date_str_to_date(end_date) if end_date is not None else yesterday_date()

    app = HistoricalReportsApp(
        port=port,
        interval_secs=args.interval,
        sim_config_file=args.sim_config,
        debug=args.debug,
        start_date=start_date,
        end_date=end_date
    )

    logger.info(f"Starting Historical Reports Dashboard on port {port}")
    app.run(debug=args.debug)

if __name__ == "__main__":
    main()
