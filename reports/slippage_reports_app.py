"""Slippage Reports Dashboard - Standalone Application."""

import argparse
import logging
from datetime import timedelta as td
from typing import Optional

from dash import html, dcc, Input, Output
from dash.dash_table import DataTable
from dash.dash_table import Format, FormatTemplate

from lib.pnl_new.binance_pnl import BinancePnl
from lib.reports.markouts import FillMarkouts
from lib.util.config import get_config
from lib.util.time_util import today
from lib.util.util import LOCAL
from lib.reports.base_dash_app import BaseDashApp
from lib.reports.slippage_reports import SlippageReports

FMT_MONEY = FormatTemplate.money(2)

SLIPPAGE_HEADLINE_COLS = [
    {'id': 'index', 'name': 'Case'},
    {'id': 'slip', 'name': 'Slippage Value', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'slip_bps', 'name': 'Slippage Bps', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
]

SLIPPAGE_BREAKDOWN_COLS = [
    {'id': 'aggression_level', 'name': 'Aggression Level'},
    {'id': 'total_traded_dollars_vwap', 'name': 'Total Traded Dollars', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'total_slip_bps_vwap', 'name': 'Total Slip Bps by VWAP', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'fill_slip_bps_vwap', 'name': 'Fill Slip Bps by VWAP', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'opp_slip_bps_vwap', 'name': 'Opportunity Slip Bps by VWAP', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'total_slip_bps_start_px', 'name': 'Total Slip Bps by START PX', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'fill_slip_bps_start_px', 'name': 'Fill Slip Bps by START PX', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'opp_slip_bps_start_px', 'name': 'Opportunity Slip Bps by START PX', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
]

SLIPPAGE_COLS = [
    {'id': 'symbol_venue', 'name': 'Symbol'},
    {'id': 'total_slip', 'name': 'Total Slippage Dollars', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'total_slip_bps', 'name': 'Total Slippage Bps', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'fill_to_opp_ratio', 'name': 'Fill Slippage vs Opportunity Slippage', 'type': 'numeric', 'format': FormatTemplate.percentage(2)},
    {'id': 'fill_slip', 'name': 'Fill Slippage Dollars', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'fill_slip_bps', 'name': 'Fill Slippage Bps', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
    {'id': 'opp_slip', 'name': 'Opportunity Slippage Dollars', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'opp_slip_bps', 'name': 'Opportunity Slippage Bps', 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
]

VWAP_COLS = [
    {'id': 'symbol_venue', 'name': 'Symbol'},
    {'id': 'close_mid', 'name': 'Current Px', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'avg_fill_px', 'name': 'Avg. Cost', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'target_period_vwap', 'name': 'VWAP', 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'daily_vwap_shortfall', 'name': "Today's Aggregated VWAP Shortfall", 'type': 'numeric', 'format': FMT_MONEY},
    {'id': 'daily_vwap_shortfall_bps', 'name': "Today's Aggregated VWAP Shortfall bps", 'type': 'numeric', 'format': Format.Format(precision=2, scheme=Format.Scheme.fixed)},
]

logger = logging.getLogger(__name__)


class SlippageReportsApp(BaseDashApp):
    """Slippage Reports Dashboard Application."""
    
    def __init__(self, port: int = 8052, interval_secs: int = 60, 
                 sim_config_file: Optional[str] = None, debug: bool = False):
        """
        Initialize Slippage Reports Dashboard.
        
        Args:
            port: Port to run the application on
            interval_secs: Refresh interval in seconds
            sim_config_file: Optional simulation config file
            debug: Whether to run in debug mode
        """
        super().__init__("Slippage Reports", port, interval_secs, debug)
        
        # Initialize config and required components
        _, self.config = get_config(sim_config_file) if sim_config_file else get_config()

        # Calculate time ranges for markouts
        hist_start_dt = today() - td(days=90 if not debug else 2)  # Default lookback period
        today_start_dt = today()
        markouts_lookback_days = 90 if not debug else 2

        # Initialize FillMarkouts for slippage analysis
        self.fill_markouts = FillMarkouts(self.config)
        markouts_start_dt = max(today_start_dt - td(days=markouts_lookback_days), hist_start_dt)
        self.fill_markouts.set_start_end(markouts_start_dt, today_start_dt)

        # Prepare markouts data - let FillMarkouts load bars from file (needs more columns than BinancePnl provides)
        # Only preload fills from BinancePnl since it already has the needed columns
        self.pnl_calculator = BinancePnl(
            config=self.config,
            start=hist_start_dt,
        )

        preload_fills_df = self.pnl_calculator.fills_df
        if preload_fills_df is not None:
            preload_fills_idx = preload_fills_df['ts'] >= markouts_start_dt
            preload_fills_df = preload_fills_df.loc[preload_fills_idx]

        # Let FillMarkouts load bars from file - it needs columns like vwap, volume, dvolume, etc.
        # that BinancePnl doesn't load
        self.fill_markouts.prepare_markouts_data(
            data_source="file",
            preload_fills_df=preload_fills_df,
        )
        model_alpha_lookback_days = 30 if not debug else 2
        self.fill_markouts.load_model_alpha_df(start_dt=today_start_dt - td(days=model_alpha_lookback_days))
        self.fill_markouts.run_model_alpha_markouts()

        # Initialize Slippage Reports
        self.slippage_reports = SlippageReports(self.fill_markouts)
        
        # Setup the application
        self.setup_layout()
        self.register_callbacks()
        
        logger.info("Slippage Reports Dashboard initialized successfully")
    
    def setup_layout(self):
        """Setup the layout for Slippage Reports."""
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
                        interval=self.interval_secs * 1000 * 6,  # Slower refresh for slippage
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
                
                # Slippage by Symbol Section
                html.Pre(
                    id='slippage-headline', children='Slippage by Symbol',
                    style={'textAlign': 'center', 'fontSize': '16px'},
                ),
                html.Div([
                    DataTable(
                        columns=SLIPPAGE_HEADLINE_COLS,
                        sort_action="native",
                        id='slippage-headline-table',
                    ),
                ], style={'marginBottom': '20px'}),
                
                # Total Slippage by Aggression Section
                html.H2("Total Slippage by Aggression and Compare Px Type", 
                       style={'textAlign': 'center', 'marginBottom': '20px', 
                              'marginTop': '20px', 'fontSize': '16px'}),
                html.Div([
                    DataTable(
                        columns=SLIPPAGE_BREAKDOWN_COLS,
                        sort_action="native",
                        id='total-slippage-table',
                    ),
                ], style={'marginBottom': '20px'}),
                
                # Total Slippage over Time Graph
                html.Div([
                    html.Pre(children='Total Slippage over Time', 
                            style={'textAlign': 'center', 'fontSize': '16px'}),
                    dcc.Graph(id='total-slippage-figure'),
                ], style={'marginBottom': '20px'}),
                
                # Slippage using VWAP by Symbol
                html.H2("Slippage using VWAP by Symbol", 
                       style={'textAlign': 'center', 'marginBottom': '20px', 
                              'marginTop': '20px', 'fontSize': '16px'}),
                html.Div([
                    DataTable(
                        columns=SLIPPAGE_COLS,
                        sort_action="native",
                        id='slippage-vwap-table',
                    ),
                ], style={'marginBottom': '20px'}),
                
                # Slippage using START PX by Symbol
                html.H2("Slippage using START PX by Symbol", 
                       style={'textAlign': 'center', 'marginBottom': '20px', 
                              'marginTop': '20px', 'fontSize': '16px'}),
                html.Div([
                    DataTable(
                        columns=SLIPPAGE_COLS,
                        sort_action="native",
                        id='slippage-start-px-table',
                    ),
                ], style={'marginBottom': '20px'}),
                
                # VWAP Shortfall Analysis
                html.Div([
                    html.Pre(id='shortfall-headline', children='VWAP Shortfall', 
                            style={'textAlign': 'center', 'fontSize': '16px'}),
                    dcc.Graph(id='shortfall-figure'),
                ], style={'marginBottom': '20px'}),
                
                # Today VWAP Shortfall Table
                html.Div([
                    html.Pre(id='vwap-headline', children='Today VWAP Shortfall', 
                            style={'textAlign': 'center', 'fontSize': '16px'}),
                    DataTable(
                        columns=VWAP_COLS,
                        sort_action="native",
                        id='vwap-table',
                    ),
                ], style={'marginBottom': '20px'}),
                
            ], style={"padding": "0 20px"}),
            
            # Hidden timer for removing loading overlay
            dcc.Interval(id='init-timer', interval=1000, n_intervals=0, max_intervals=2),
        ])
    
    def register_callbacks(self):
        """Register all callbacks for Slippage Reports."""
        
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
            return self.handle_refresh(self.slippage_reports.update_data, "slippage")
        
        # Timestamp display
        @self.app.callback(
            Output('loaded-ts-text', 'children'),
            Input('refresh-state', 'children'),
        )
        def get_ts_display(n_state: str):
            return self.slippage_reports.get_ts_display(n_state)
        
        # Main slippage data callback
        @self.app.callback(
            [Output('slippage-headline', 'children'),
             Output('slippage-headline-table', 'data'),
             Output('slippage-vwap-table', 'data'),
             Output('slippage-start-px-table', 'data'),
             Output('total-slippage-table', 'data'),
             Output('total-slippage-figure', 'figure')],
            Input('refresh-state', 'children'),
        )
        def update_slippage_data(n_state: str):
            return self.slippage_reports.update_slippage_data(n_state)
        
        # Shortfall data callback
        @self.app.callback(
            [Output('shortfall-headline', 'children'),
             Output('shortfall-figure', 'figure')],
            Input('refresh-state', 'children'),
        )
        def update_shortfall_data(n_state: str):
            return self.slippage_reports.update_shortfall_data(n_state)
        
        # VWAP data callback
        @self.app.callback(
            [Output('vwap-headline', 'children'),
             Output('vwap-table', 'data')],
            Input('refresh-state', 'children'),
        )
        def update_vwap_data(n_state: str):
            return self.slippage_reports.update_vwap_data(n_state)


def main():
    """Main entry point for Slippage Reports Dashboard."""
    parser = argparse.ArgumentParser(description='Slippage Reports Dashboard')
    parser.add_argument('-p', '--port', help='Port to run on', type=int, default=None)
    parser.add_argument('-d', '--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('-i', '--interval', help='Refresh interval in seconds', type=int, default=60)
    parser.add_argument('-s', '--sim-config', help='Simulation config file', type=str, default=None)
    args = parser.parse_args()
    
    # Determine port
    if args.port:
        port = args.port
    else:
        port = 8054 if not LOCAL else 8064
    
    # Create and run the application
    app = SlippageReportsApp(
        port=port,
        interval_secs=args.interval,
        sim_config_file=args.sim_config,
        debug=args.debug
    )
    
    logger.info(f"Starting Slippage Reports Dashboard on port {port}")
    app.run(debug=args.debug)


if __name__ == "__main__":
    main()
