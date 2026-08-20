#!/usr/bin/env python3
"""
Prod Fits Reports Dashboard Application.

This standalone Dash application provides visualization and monitoring of
production model fitting statistics and T-statistics over time.

The dashboard displays T-statistics for different model horizons and tracks
model fitting performance over a configurable lookback period.
"""

import logging
import argparse
from datetime import datetime as dt, timezone
from dash import html, dcc, Input, Output, State

from lib.reports.base_dash_app import BaseDashApp
from lib.reports.prod_fits_reports import ProdFitsReports
from lib.util.util import LOCAL

logger = logging.getLogger(__name__)


class FitsDashApp(BaseDashApp):
    """Prod Fits Reports Dashboard Application.
    
    Displays production model fitting statistics and T-statistics visualization
    for monitoring model performance over time.
    """

    def __init__(self, port: int, debug: bool =False):
        """Initialize the Fits Reports Dashboard.
        
        Args:
            port: Port number to run the server on (default: 8053)
            debug: Enable debug mode (default: False)
        """
        super().__init__("Prod Fits Reports", port, interval_secs=300)

        self.debug = debug
        # Initialize production fits reports directly
        self.prod_fits_reports = ProdFitsReports()
        
        # Setup the application
        self.setup_layout()
        self.register_callbacks()
        
        logger.info(f"Prod Fits Reports Dashboard initialized on port {port}")

    def setup_layout(self):
        """Setup the dashboard layout with all UI components."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Production Model Fits Dashboard", 
                       style={'textAlign': 'center', 'marginBottom': '20px'}),
                html.Hr(),
            ]),
            
            # Loading overlay
            self.create_loading_overlay(),
            
            # Status display
            html.Div([
                html.Div(id='fits-timestamp-display', 
                        style={'textAlign': 'center', 'marginBottom': '10px'}),
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='fits-interval-component',
                interval=self.interval_secs * 1000,  # Convert to milliseconds
                n_intervals=0
            ),
            
            # Hidden div to store refresh state
            html.Div(id='fits-refresh-state', style={'display': 'none'}),
            
            # Main content area for T-statistics plots
            html.Div([
                html.H3(f"T-Statistics Time Series (Last {self.prod_fits_reports.lookback_days} days)",
                       style={'textAlign': 'center', 'marginTop': '30px', 'marginBottom': '20px'}),
                html.Div(id='fits-t-figures-container'),
            ], style={'padding': '20px'}),
            
        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa'})

    def register_callbacks(self):
        """Register all Dash callbacks for interactivity."""
        
        # Refresh callback
        @self.app.callback(
            Output('fits-refresh-state', 'children'),
            [Input('fits-interval-component', 'n_intervals')]
        )
        def trigger_refresh(n_intervals):
            """Trigger data refresh on interval.
            
            Args:
                n_intervals: Number of intervals passed
                
            Returns:
                str: Refresh state identifier
            """
            if n_intervals > 0:
                return self._handle_refresh(self.prod_fits_reports.update_data, "fits")
            return f"fits-{n_intervals}"
        
        # Update timestamp display
        @self.app.callback(
            Output('fits-timestamp-display', 'children'),
            [Input('fits-refresh-state', 'children')],
            prevent_initial_call=False
        )
        def update_timestamp(n_state):
            """Update the timestamp display"""
            if n_state:
                return self.prod_fits_reports.get_ts_display(n_state)
            return "Initializing..."
        
        # Update T-statistics figures
        @self.app.callback(
            Output('fits-t-figures-container', 'children'),
            [Input('fits-refresh-state', 'children')],
            prevent_initial_call=False
        )
        def update_fits_figures(n_state):
            """Update the T-statistics figures for all horizons"""
            if n_state:
                return self.prod_fits_reports.update_all_fits_t_figures(n_state)
            return html.Div("Loading T-statistics figures...")
        
        # Loading overlay callback
        @self.app.callback(
            Output('loading-overlay', 'style'),
            [Input('fits-refresh-state', 'children')],
            [State('loading-overlay', 'style')],
            prevent_initial_call=False
        )
        def show_loading(refresh_state, current_style):
            """Show loading overlay during refresh"""
            # Always hide the overlay - it should only show on initial page load
            return {**current_style, 'display': 'none'}

    def _handle_refresh(self, updater_func, report_type):
        """Handle refresh with locking mechanism"""
        acquired = self.refresh_lock.acquire(blocking=False)
        if not acquired:
            logger.info("Refresh already in progress for %s, skipping", report_type)
            return f"{report_type}-skipped"
        
        try:
            logger.info("Starting %s refresh", report_type)
            updater_func()
            logger.info("Completed %s refresh", report_type)
            return f"{report_type}-updated-{dt.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        except Exception as exc:
            logger.error("Error during %s refresh: %s", report_type, exc)
            return f"{report_type}-error"
        finally:
            self.refresh_lock.release()


def main():
    """Main entry point for the Prod Fits Reports Dashboard."""

    parser = argparse.ArgumentParser(description='Prod Fits Reports Dashboard')
    parser.add_argument('-p', '--port', help='Port to run on', type=int, default=None)
    parser.add_argument('-d', '--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()

    # Determine port
    if args.port:
        port = args.port
    else:
        port = 8052 if not LOCAL else 8062  # Use different port in local development

    # Create and run the app
    app = FitsDashApp(port=port, debug=args.debug)
    app.run(debug=args.debug)


if __name__ == '__main__':
    main()