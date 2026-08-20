"""Trading Reports Dashboard - Standalone Application."""

import argparse
import logging
from typing import Optional

import plotly.graph_objects as go
from dash import html, dcc, Input, Output
from dash.dash_table import DataTable
from dash.dash_table import Format, FormatTemplate

from lib.reports.base_dash_app import BaseDashApp, COMMON_TABLE_PROPS
from lib.reports.trading_reports import TradingReports
from lib.util.config import get_config
from lib.util.util import LOCAL

# Trading reporting constants
FMT_MONEY = FormatTemplate.money(2)
FMT_PERCENT = FormatTemplate.percentage(2)

# PnL table column definitions
PNL_COLS = [
    {'id': 'symbol', 'name': 'Symbol'},
    {'id': 'net_pnl', 'name': 'Daily Tot Pnl', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'net_pnl_return', 'name': 'Daily Tot Return', 'type': 'numeric', 'format': FMT_PERCENT},
    {'id': 'notional', 'name': 'Position', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'portfolio_pct', 'name': 'Portfolio %', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'close_mid', 'name': 'Close Mid', 'type': 'numeric', 'format': FormatTemplate.money(4)},
    {'id': 'position_age', 'name': 'Position Age (days)', 'type': 'numeric', 'format': Format.Format(precision=0, scheme=Format.Scheme.fixed)},
    {'id': 'realized_pnl', 'name': 'Daily Real Pnl', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'unrealized_pnl', 'name': 'Daily Unreal Pnl', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'unrealized_pnl_tot_cum', 'name': 'Cum Unreal Pnl', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'unrealized_pnl_tot_cum_return', 'name': 'Cum Unreal Return', 'type': 'numeric', 'format': FMT_PERCENT},
    {'id': 'funding_income', 'name': 'Daily Funding Income', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'alpha_opt', 'name': 'Alpha Opt Bps', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'risk_1440', 'name': 'Risk %', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'last_funding_rate', 'name': 'Funding Rates Bps', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
]

PNL_CONT_COLS = [
    {'id': 'symbol', 'name': 'Symbol'},
    {'id': 'trading', 'name': 'Trading', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'fraction_done', 'name': 'Target Fraction Done', 'type': 'numeric', 'format': FMT_PERCENT},
    {'id': 'target_position', 'name': 'Target', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'notional', 'name': 'Position', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'lbound', 'name': 'Pos. Lower Bound', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'ubound', 'name': 'Pos. Upper Bound', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'alpha_st', 'name': 'S.T. Alpha Opt Bps', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
]
TODAY_SUMMARY_COLS = [
    {'id': 'metrics', 'name': 'Metrics', 'type': 'text'},
    {'id': 'today', 'name': 'Today Results', 'type': 'text'},
]

FILL_COLS = [
    {'id': 'ts', 'name': 'Timestamp'},
    {'id': 'fill_qty', 'name': 'Fill Quantity', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'fill_px', 'name': 'Fill PX', 'type': 'numeric', 'format': FormatTemplate.money(8)},
]

ALPHA_BREAKDOWN_COLS = [
    {'id': 'horizon', 'name': 'Horizon'},
    {'id': 'model', 'name': 'Model'},
    {'id': 'value_bps', 'name': 'Value (bps)', 'type': 'numeric', 'format': Format.Format(precision=4, scheme=Format.Scheme.fixed)},
]

logger = logging.getLogger(__name__)


class TradingReportsApp(BaseDashApp):
    """Trading Reports Dashboard Application."""
    
    def __init__(
            self,
            port: int = 8050,
            interval_secs: int = 60,
            sim_config_file: Optional[str] = None,
            debug: bool = False
    ):
        """
        Initialize Trading Reports Dashboard.
        
        Args:
            port: Port to run the application on
            interval_secs: Refresh interval in seconds
            sim_config_file: Optional simulation config file
            debug: Whether to run in debug mode
        """
        super().__init__("Trading Reports", port, interval_secs, debug)

        # Initialize configuration and core components
        _, self.config = get_config(config_file=sim_config_file)
        self.debug = debug
        self.trading_reports = TradingReports(config=self.config, debug=self.debug)
        
        # Setup the application
        self.setup_layout()
        self.register_callbacks()
        logger.info("Trading Reports Dashboard initialized successfully")
    
    def setup_layout(self):
        """Setup the layout for Trading Reports."""
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
                
                # Intraday PnL section
                html.H3(children='Intraday Pnl', style={'textAlign': 'center'}),
                html.Div([
                    html.Div(
                        DataTable(
                            id='perf-table',
                            columns=TODAY_SUMMARY_COLS,
                        ),
                        style={"width": "40%", "display": "inline-block", "verticalAlign": "top"},
                    ),
                    html.Div(
                        DataTable(
                            id='perf-table-right',
                            columns=TODAY_SUMMARY_COLS,
                        ),
                        style={"width": "40%", "display": "inline-block", "verticalAlign": "top", 
                               "marginLeft": "40px", 'marginBottom': '20px'},
                    ),
                ], style={'marginBottom': '20px', "display": "flex"}),
                
                # PnL Graph
                html.Div([
                    dcc.Graph(id='pnl-graph'),
                ], style={'marginBottom': '20px'}),
                
                # PnL Tables
                html.Div([
                    html.H3('Pnl info', id='pnl-table-headline', 
                           style={'textAlign': 'center', 'fontSize': '20px'}),
                    html.Div(
                        DataTable(
                            id='pnl-table',
                            columns=PNL_COLS,
                            **COMMON_TABLE_PROPS,
                        ),
                        style={'marginBottom': '60px'},
                    ),
                    html.H3('Trading', id='trading-table-headline', 
                           style={'textAlign': 'center', 'fontSize': '20px'}),
                    html.Div(
                        DataTable(
                            id='pnl-cont-table',
                            columns=PNL_CONT_COLS,
                            **COMMON_TABLE_PROPS,
                        ),
                    ),
                ], style={'marginBottom': '20px'}),
                
                # Portfolio & Trading
                html.Div([
                    html.H3(children='Portfolio & Trading', style={'textAlign': 'center'}),
                    dcc.Graph(id='buysell-graph'),
                ], style={'marginBottom': '20px'}),
                
                # PnL Breakdown
                html.Div([
                    html.H3(children='Pnl Breakdown', style={'textAlign': 'center'}),
                    dcc.Graph(id='total-unrealized-pnl-symbol-graph'),
                    dcc.Graph(id='unrealized-pnl-symbol-graph'),
                    dcc.Graph(id='realized-pnl-symbol-graph'),
                    dcc.Graph(id='realized-pnl-trades-graph'),
                ], style={'marginBottom': '20px'}),
                
                # Single Symbol Data
                html.Div([
                    html.H3(children='Today Single Symbol Data', style={'textAlign': 'center'}),
                    dcc.Dropdown(id='symbol-dropdown-selection', style={'marginBottom': '20px'}),
                    html.Pre('Alpha Opt Breakdown',
                            style={'marginBottom': '10px', 'marginTop': '20px', 'fontSize': '16px'}),
                    DataTable(
                        columns=ALPHA_BREAKDOWN_COLS,
                        data=[],
                        sort_action="native",
                        id='alpha-breakdown-table',
                        page_size=10,
                        page_action='native',
                        style_data_conditional=[
                            {'if': {'filter_query': '{value_bps} > 0'}, 'color': 'green'},
                            {'if': {'filter_query': '{value_bps} < 0'}, 'color': 'red'},
                        ],
                    ),
                    html.Pre('Buy Fill Executions',
                            style={'marginBottom': '20px', 'marginTop': '20px', 'fontSize': '16px'}),
                    DataTable(
                        columns=FILL_COLS,
                        sort_action="native",
                        id='symbol-buy-fill-table',
                        page_size=10,
                        page_action='native',
                    ),
                    html.Pre('Sell Fill Executions',
                            style={'marginBottom': '20px', 'marginTop': '20px', 'fontSize': '16px'}),
                    DataTable(
                        columns=FILL_COLS,
                        sort_action="native",
                        id='symbol-sell-fill-table',
                        page_size=10,
                        page_action='native',
                    ),
                    dcc.Graph(id='symbol-pnl-graph'),
                ], style={'marginBottom': '20px'}),
                
                # Alpha Breakdown
                html.Div([
                    html.H3(children='Today Alpha Breakdown', style={'textAlign': 'center'}),
                    dcc.Dropdown(
                        id='alpha-dropdown-selection',
                        value='alpha_opt',
                        options=['alpha_opt'] + self.trading_reports.corr_checks_cols,
                        style={'marginBottom': '20px'},
                    ),
                    dcc.Graph(id='alpha-symbol-graph'),
                    DataTable(
                        sort_action="native",
                        id='alpha-symbol-table',
                    ),
                ], style={'marginBottom': '20px'}),
                
            ], style={"padding": "0 20px"}),
            
            # Hidden timer for removing loading overlay
            dcc.Interval(id='init-timer', interval=1000, n_intervals=0, max_intervals=2),
        ])
    
    def register_callbacks(self):
        """Register all callbacks for Trading Reports."""
        
        # Hide loading overlay after initial load
        @self.app.callback(
            Output('loading-overlay', 'style'),
            Input('init-timer', 'n_intervals'),
        )
        def hide_loading_overlay(n):
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
            return self.handle_refresh(self.trading_reports.load_data, "trading")
        
        # Timestamp display
        @self.app.callback(
            Output('loaded-ts-text', 'children'),
            Input('refresh-state', 'children'),
        )
        def get_ts_display(n_state: str):
            return self.trading_reports.get_ts_display(n_state)
        
        # Performance tables and headlines
        @self.app.callback(
            [Output('perf-table', 'data'),
             Output('perf-table-right', 'data'),
             Output('pnl-table-headline', 'children'),
             Output('trading-table-headline', 'children')],
            Input('refresh-state', 'children'),
        )
        def update_perf_display(n_state: str):
            return self.trading_reports.get_performance_stats(n_state)
        
        # PnL graph
        @self.app.callback(
            Output('pnl-graph', 'figure'),
            Input('refresh-state', 'children'),
        )
        def update_intraday_pnl_figure(n_state: str):
            return self.trading_reports.intraday_pnl_figure(n_state)
        
        # Symbol dropdown
        @self.app.callback(
            [Output('symbol-dropdown-selection', 'options'),
             Output('symbol-dropdown-selection', 'value')],
            Input('refresh-state', 'children'),
        )
        def update_symbol_dropdown(n_state: str):
            return self.trading_reports.update_symbol_dropdown(n_state)
        
        # Single symbol data
        @self.app.callback(
            [Output('symbol-buy-fill-table', 'data'),
             Output('symbol-sell-fill-table', 'data'),
             Output('symbol-pnl-graph', 'figure'),
             Output('alpha-breakdown-table', 'data')],
            [Input('symbol-dropdown-selection', 'value'),
             Input('refresh-state', 'children')],
        )
        def update_today_single_symbol_result(value: str, n_state: str):
            if value is None:
                return [], [], go.Figure(), []
            buy_fills, sell_fills, fig = self.trading_reports.get_today_single_symbol_result(value, n_state)
            alpha_breakdown = self.trading_reports.get_alpha_breakdown_for_symbol(value)
            return buy_fills, sell_fills, fig, alpha_breakdown
        
        # Realized PnL by trades
        @self.app.callback(
            Output('realized-pnl-trades-graph', 'figure'),
            Input('refresh-state', 'children'),
        )
        def update_today_realized_by_trades_figure(n_state: str):
            return self.trading_reports.today_realized_by_trades_figure(n_state)
        
        # PnL by symbol figures
        @self.app.callback(
            [Output('realized-pnl-symbol-graph', 'figure'),
             Output('unrealized-pnl-symbol-graph', 'figure'),
             Output('total-unrealized-pnl-symbol-graph', 'figure')],
            Input('refresh-state', 'children'),
        )
        def update_today_pnl_by_symbol_figure(n_state: str):
            return self.trading_reports.today_pnl_by_symbol_figure(n_state)
        
        # Alpha breakdown
        @self.app.callback(
            [Output('alpha-symbol-graph', 'figure'),
             Output('alpha-symbol-table', 'data'),
             Output('alpha-symbol-table', 'columns')],
            [Input('alpha-dropdown-selection', 'value'),
             Input('refresh-state', 'children')],
        )
        def update_today_alpha_by_symbol_figure(value: str, n_state: str):
            if value is None:
                return go.Figure(), [], []
            return self.trading_reports.today_alpha_by_symbol_figure(value, n_state)
        
        # Buy/Sell figure
        @self.app.callback(
            Output('buysell-graph', 'figure'),
            Input('refresh-state', 'children'),
        )
        def update_buy_sell_figure(n_state: str):
            return self.trading_reports.buy_sell_figure(n_state)
        
        #PnL tables data
        @self.app.callback(
            [Output('pnl-table', 'data'),
             Output('pnl-cont-table', 'data')],
            Input('refresh-state', 'children'),
        )
        def update_pnl_tables(_):
            if self.trading_reports.security_daily_pnl_df is not None:
                return (
                    self.trading_reports.security_daily_pnl_df.to_dict('records'),
                    self.trading_reports.today_trading_df.to_dict('records')
                )
            return [], []


def main():
    """Main entry point for Trading Reports Dashboard."""
    parser = argparse.ArgumentParser(description='Trading Reports Dashboard')
    parser.add_argument('-p', '--port', help='Port to run on', type=int, default=None)
    parser.add_argument('-d', '--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('-i', '--interval', help='Refresh interval in seconds', type=int, default=300)
    parser.add_argument('-s', '--sim-config', help='Simulation config file', type=str, default=None)
    args = parser.parse_args()
    
    # Determine port
    if args.port:
        port = args.port
    else:
        port = 8050 if not LOCAL else 8060  # Use different port in local development
    
    # Create and run the application
    app = TradingReportsApp(
        port=port,
        interval_secs=args.interval,
        sim_config_file=args.sim_config,
        debug=args.debug
    )
    
    logger.info(f"Starting Trading Reports Dashboard on port {port}")
    app.run(debug=args.debug)


if __name__ == "__main__":
    main()