"""Historical Simulation Reports Dashboard - Long-term sim performance analysis.

Adapted from simulation_report.py for long-term historical analysis.
Same layout and callback patterns as simulation_report.py.
"""

import argparse
import logging

from dash import html, dcc, Input, Output, callback_context
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Scheme

from lib.reports.base_dash_app import BaseDashApp
from lib.reports.hist_sim_reports import HistSimReports

logger = logging.getLogger(__name__)

# Default port (same range as other reports)
REPORT_PORT = 8057


class HistSimReportsApp(BaseDashApp):
    """Historical Simulation Reports Dashboard Application.

    Inherits from BaseDashApp to get kill file monitoring and other common functionality.
    """

    def __init__(
            self,
            sim_name: str = None,
            port: int = REPORT_PORT,
            interval_secs: int = 300,
            debug: bool = False
    ):
        """Initialize Historical Simulation Reports Dashboard.

        Args:
            sim_name: Name of simulation to load initially (optional)
            port: Port to run the application on
            interval_secs: Refresh interval in seconds (default 300)
            debug: Whether to run in debug mode
        """
        # Initialize base class (sets up self.app, logging, kill file monitoring)
        super().__init__("Historical Simulation Reports", port, interval_secs, debug)

        # Get list of available simulations (same as simulation_report.py)
        self.available_simulations = HistSimReports.get_available_simulations()

        # Initialize with provided sim or None (let user select)
        self.sim_name = sim_name
        self.hist_sim_reports = None

        # Setup the dash app layout and callbacks (same as simulation_report.py)
        self.setup_layout()
        self.register_callbacks()

        logger.info("Historical Simulation Reports Dashboard initialized")

    def setup_layout(self):
        """Setup the Dash app layout.

        Same structure as simulation_report.py.
        """
        # Create dropdown options
        dropdown_options = [
            {'label': sim, 'value': sim} for sim in self.available_simulations
        ]

        self.app.layout = html.Div([
            # Store component to track data loading state (same as simulation_report.py)
            dcc.Store(id='data-loaded-state'),

            html.H1("Historical Simulation Report", style={'textAlign': 'center'}),

            # Simulation selector (same as simulation_report.py)
            html.Div([
                html.Label("Select Simulation:",
                          style={'marginRight': '10px', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='simulation-dropdown',
                    options=dropdown_options,
                    value=self.sim_name,
                    style={'width': '400px', 'display': 'inline-block', 'verticalAlign': 'middle'},
                    clearable=False
                ),
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),

            # Current simulation name
            html.H3(id="simulation-name", style={'textAlign': 'center', 'marginBottom': '20px'}),

            # Reload button (same as simulation_report.py)
            html.Div([
                html.Button("Reload Data", id="reload-btn", n_clicks=0,
                           style={'marginBottom': '10px'}),
                html.Div(id="load-status", style={"marginTop": "10px", "color": "green"}),
            ], style={'marginBottom': '30px', 'textAlign': 'center'}),

            # Summary statistics (same format as simulation_report.py)
            html.Div([
                html.H3("Summary Statistics",
                        style={'textAlign': 'center', 'marginBottom': '20px'}),
                html.Div(id='summary-stats', style={'marginBottom': '30px'}),
            ], style={
                'backgroundColor': '#ffffff',
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '30px'
            }),

            # P&L over time (same as simulation_report.py)
            html.Div([
                html.H3("P&L Performance", style={'textAlign': 'center'}),
                dcc.Graph(id='pnl-timeseries'),
            ], style={'marginBottom': '30px'}),

            # Monthly P&L
            html.Div([
                html.H3("Monthly P&L", style={'textAlign': 'center'}),
                dcc.Graph(id='monthly-pnl-chart'),
                DataTable(
                    id='monthly-table',
                    columns=[
                        {'name': 'Month', 'id': 'Month', 'type': 'text'},
                        {'name': 'Monthly P&L', 'id': 'Monthly P&L', 'type': 'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                        {'name': 'Cumulative P&L', 'id': 'Cumulative P&L', 'type': 'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                        {'name': 'Avg Notional', 'id': 'Avg Notional', 'type': 'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=0, group=True)},
                        {'name': 'Fees', 'id': 'Fees', 'type': 'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                        {'name': 'Funding', 'id': 'Funding', 'type': 'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                        {'name': 'Sharpe', 'id': 'Sharpe', 'type': 'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=2)},
                    ],
                    sort_action='native',
                    page_size=12,
                    style_cell={'textAlign': 'center'},
                    style_data_conditional=[
                        {'if': {'column_id': 'Monthly P&L', 'filter_query': '{Monthly P&L} > 0'},
                         'color': 'green'},
                        {'if': {'column_id': 'Monthly P&L', 'filter_query': '{Monthly P&L} < 0'},
                         'color': 'red'},
                    ]
                ),
            ], style={'marginBottom': '30px'}),

            # Yearly Statistics
            html.Div([
                html.H3("Yearly Statistics", style={'textAlign': 'center'}),
                DataTable(
                    id='yearly-table',
                    columns=[
                        {'name': 'Year', 'id': 'Year', 'type': 'numeric'},
                        {'name': 'Yearly P&L', 'id': 'Yearly P&L', 'type': 'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                        {'name': 'Cumulative P&L', 'id': 'Cumulative P&L', 'type': 'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                        {'name': 'Avg Notional', 'id': 'Avg Notional', 'type': 'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=0, group=True)},
                        {'name': 'Fees', 'id': 'Fees', 'type': 'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                        {'name': 'Funding', 'id': 'Funding', 'type': 'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                        {'name': 'Days', 'id': 'Days', 'type': 'numeric'},
                        {'name': 'Sharpe', 'id': 'Sharpe', 'type': 'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=2)},
                    ],
                    sort_action='native',
                    style_cell={'textAlign': 'center'},
                    style_data_conditional=[
                        {'if': {'column_id': 'Yearly P&L', 'filter_query': '{Yearly P&L} > 0'},
                         'color': 'green'},
                        {'if': {'column_id': 'Yearly P&L', 'filter_query': '{Yearly P&L} < 0'},
                         'color': 'red'},
                    ]
                ),
            ], style={'marginBottom': '30px'}),

            # Portfolio metrics (same as simulation_report.py)
            html.Div([
                html.H3("Portfolio Metrics", style={'textAlign': 'center'}),
                dcc.Graph(id='portfolio-metrics'),
            ], style={'marginBottom': '30px'}),

            # Rolling Sharpe
            html.Div([
                html.H3("Rolling Sharpe Ratio", style={'textAlign': 'center'}),
                dcc.Graph(id='rolling-sharpe-chart'),
            ], style={'marginBottom': '30px'}),

            # Drawdown Analysis (same as simulation_report.py)
            html.Div([
                html.H3("Drawdown Analysis", style={'textAlign': 'center'}),
                html.Div([
                    dcc.Graph(id='drawdown-chart'),
                ], style={'marginBottom': '20px'}),
                html.Div([
                    html.H4("Top 5 Largest Drawdowns", style={'textAlign': 'center'}),
                    DataTable(
                        id='drawdown-table',
                        columns=[
                            {'name': 'Start Date', 'id': 'start_date', 'type': 'datetime'},
                            {'name': 'End Date', 'id': 'end_date', 'type': 'datetime'},
                            {'name': 'Duration (Days)', 'id': 'duration_days', 'type': 'numeric'},
                            {'name': 'Peak Value ($)', 'id': 'peak_value', 'type': 'numeric',
                             'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                            {'name': 'Trough Value ($)', 'id': 'trough_value', 'type': 'numeric',
                             'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                            {'name': 'Dollar Loss', 'id': 'dollar_loss', 'type': 'numeric',
                             'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                            {'name': 'Percent Loss', 'id': 'percent_loss', 'type': 'numeric',
                             'format': Format(scheme=Scheme.fixed, precision=2)},
                        ],
                        sort_action='native',
                        page_size=5,
                        style_cell={'textAlign': 'center'},
                        style_data_conditional=[
                            {'if': {'column_id': 'dollar_loss'}, 'color': 'red'},
                            {'if': {'column_id': 'percent_loss'}, 'color': 'red'},
                        ]
                    ),
                ]),
            ], style={'marginBottom': '30px'}),

            # Daily PnL and Returns table (same as simulation_report.py)
            html.Div([
                html.H3("Daily P&L and Returns", style={'textAlign': 'center'}),
                DataTable(
                    id='daily-pnl-returns-table',
                    columns=[
                        {'name': 'Date', 'id': 'date', 'type': 'datetime'},
                        {'name': 'Daily P&L', 'id': 'daily_pnl', 'type': 'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                        {'name': 'Gross Notional', 'id': 'gross_notional', 'type': 'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=0, group=True)},
                        {'name': 'Daily Return (%)', 'id': 'daily_return_pct', 'type': 'numeric',
                         'format': Format(scheme=Scheme.fixed, precision=4)},
                        {'name': 'Cumulative Return (%)', 'id': 'cumulative_return_pct',
                         'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=2)},
                    ],
                    sort_action='native',
                    filter_action='native',
                    page_action='native',
                    page_size=20,
                    style_cell={'textAlign': 'center'},
                    style_data_conditional=[
                        {'if': {'column_id': 'daily_pnl', 'filter_query': '{daily_pnl} > 0'},
                         'color': 'green'},
                        {'if': {'column_id': 'daily_pnl', 'filter_query': '{daily_pnl} < 0'},
                         'color': 'red'},
                        {'if': {'column_id': 'daily_return_pct',
                                'filter_query': '{daily_return_pct} > 0'}, 'color': 'green'},
                        {'if': {'column_id': 'daily_return_pct',
                                'filter_query': '{daily_return_pct} < 0'}, 'color': 'red'},
                    ]
                ),
            ]),
        ])

    def register_callbacks(self):
        """Register Dash callbacks.

        Same pattern as simulation_report.py.
        """

        @self.app.callback(
            Output("simulation-name", "children"),
            Output("load-status", "children"),
            Output("data-loaded-state", "data"),
            Input("simulation-dropdown", "value"),
            Input("reload-btn", "n_clicks"),
            prevent_initial_call=False,
        )
        def update_simulation(selected_sim, n_clicks):  # pylint: disable=unused-argument
            """Update simulation when dropdown changes or reload button is clicked.

            Same pattern as simulation_report.py.
            """
            logger.info("=== Update simulation callback triggered ===")
            logger.info("  Selected sim: %s", selected_sim)
            logger.info("  Current sim: %s", self.sim_name)

            ctx = callback_context

            if not selected_sim:
                return "No Simulation Selected", "", None

            # Check if we need to load new simulation
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

            status = ""
            if selected_sim != self.sim_name or triggered_id == 'reload-btn':
                logger.info("Loading simulation: %s", selected_sim)
                self.sim_name = selected_sim
                try:
                    self.hist_sim_reports = HistSimReports(selected_sim)
                    status = f"Loaded: {selected_sim}"
                except (OSError, ValueError) as e:
                    logger.error("Failed to load simulation: %s", e)
                    return f"Error: {selected_sim}", f"Failed: {e}", None

            sim_display = f"Simulation: {self.sim_name}"
            data_state = {"loaded": True, "sim_name": self.sim_name}

            return sim_display, status, data_state

        @self.app.callback(
            Output('summary-stats', 'children'),
            Input('data-loaded-state', 'data'),
        )
        def update_summary_stats(data_state):
            """Update summary statistics table.

            Same html.Table format as simulation_report.py.
            """
            if not data_state or not self.hist_sim_reports:
                return html.Div("No data loaded")

            summary_data = self.hist_sim_reports.get_summary_data()
            if not summary_data:
                return html.Div("No summary data available")

            # Same table format as simulation_report.py
            return html.Table([
                html.Tbody([
                    html.Tr([
                        html.Td(cell, style={
                            'padding': '8px',
                            'textAlign': 'left' if i % 2 == 0 else 'right',
                            'fontWeight': 'bold' if i % 2 == 0 else 'normal',
                            'borderBottom': '1px solid #ddd',
                            'width': '25%'
                        }) for i, cell in enumerate(row)
                    ]) for row in summary_data
                ])
            ], style={
                'width': '80%',
                'margin': '0 auto',
                'borderCollapse': 'collapse',
                'backgroundColor': '#f9f9f9',
                'border': '1px solid #ddd',
                'borderRadius': '5px'
            })

        @self.app.callback(
            Output('pnl-timeseries', 'figure'),
            Input('data-loaded-state', 'data'),
        )
        def update_pnl_figure(data_state):
            if not data_state or not self.hist_sim_reports:
                return {}
            return self.hist_sim_reports.create_pnl_figure()

        @self.app.callback(
            Output('monthly-pnl-chart', 'figure'),
            Input('data-loaded-state', 'data'),
        )
        def update_monthly_pnl_figure(data_state):
            if not data_state or not self.hist_sim_reports:
                return {}
            return self.hist_sim_reports.create_monthly_pnl_figure()

        @self.app.callback(
            Output('monthly-table', 'data'),
            Input('data-loaded-state', 'data'),
        )
        def update_monthly_table(data_state):
            if not data_state or not self.hist_sim_reports:
                return []
            return self.hist_sim_reports.get_monthly_table_data()

        @self.app.callback(
            Output('yearly-table', 'data'),
            Input('data-loaded-state', 'data'),
        )
        def update_yearly_table(data_state):
            if not data_state or not self.hist_sim_reports:
                return []
            return self.hist_sim_reports.get_yearly_table_data()

        @self.app.callback(
            Output('portfolio-metrics', 'figure'),
            Input('data-loaded-state', 'data'),
        )
        def update_portfolio_metrics(data_state):
            if not data_state or not self.hist_sim_reports:
                return {}
            return self.hist_sim_reports.create_portfolio_metrics_figure()

        @self.app.callback(
            Output('rolling-sharpe-chart', 'figure'),
            Input('data-loaded-state', 'data'),
        )
        def update_rolling_sharpe(data_state):
            if not data_state or not self.hist_sim_reports:
                return {}
            return self.hist_sim_reports.create_rolling_sharpe_figure()

        @self.app.callback(
            Output('drawdown-chart', 'figure'),
            Input('data-loaded-state', 'data'),
        )
        def update_drawdown_chart(data_state):
            if not data_state or not self.hist_sim_reports:
                return {}
            return self.hist_sim_reports.create_drawdown_figure()

        @self.app.callback(
            Output('drawdown-table', 'data'),
            Input('data-loaded-state', 'data'),
        )
        def update_drawdown_table(data_state):
            if not data_state or not self.hist_sim_reports:
                return []
            return self.hist_sim_reports.get_drawdown_table_data()

        @self.app.callback(
            Output('daily-pnl-returns-table', 'data'),
            Input('data-loaded-state', 'data'),
        )
        def update_daily_pnl_table(data_state):
            if not data_state or not self.hist_sim_reports:
                return []
            return self.hist_sim_reports.get_daily_pnl_table_data()


def main():
    """Main entry point for Historical Simulation Reports Dashboard."""
    parser = argparse.ArgumentParser(
        description='Historical Simulation Reports Dashboard - Long-term sim performance analysis'
    )
    parser.add_argument('sim_name', nargs='?', help='Name of simulation to load initially')
    parser.add_argument('-p', '--port', help='Port to run on', type=int, default=REPORT_PORT)
    parser.add_argument('-d', '--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('-i', '--interval', help='Refresh interval in seconds', type=int, default=300)
    args = parser.parse_args()

    app = HistSimReportsApp(
        sim_name=args.sim_name,
        port=args.port,
        interval_secs=args.interval,
        debug=args.debug
    )

    logger.info("Starting Historical Simulation Reports Dashboard on port %s", args.port)
    app.run(debug=args.debug)


if __name__ == "__main__":
    main()
