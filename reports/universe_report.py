import os
import glob
import logging
import argparse
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Output, Input
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Scheme

from lib.data.dataloader import DataLoader
from lib.reports.base_dash_app import BaseDashApp
from lib.util.directory import CONFIG_DIR, dir_manager
from lib.util.config import get_config
from lib.util.util import LOCAL
from lib.util.time_util import today_date

logger = logging.getLogger(__name__)


class UniverseReport(BaseDashApp):
    """Universe Report Dashboard Application.

    Inherits from BaseDashApp to get kill file monitoring and other common functionality.
    """

    def __init__(self, port: int = 8056, interval_secs: int = 300, debug: bool = False):
        # Initialize base class (sets up self.app, logging, kill file monitoring)
        super().__init__("Universe Report", port, interval_secs, debug)

        self.data_loader = DataLoader()
        self.universe_dir = os.path.join(dir_manager.DATA_DIR, 'universe')
        self.universe_df = None

        # Initialize with default config
        self.update_config()

        # Load universe data on startup
        self.load_universe_data()

        # Setup the dash app layout and callbacks
        self.setup_layout()
        self.register_callbacks()
    
    def update_config(self, config_file: Optional[str] = None):
        """Update configuration"""
        self.config = get_config(config_file)[1]
    
    def load_universe_data(self):
        """Load all universe data from the universe directory"""
        try:
            logger.info(f"Loading universe data from {self.universe_dir}")
            
            # Check if the directory exists
            if not os.path.exists(self.universe_dir):
                logger.error(f"Universe directory does not exist: {self.universe_dir}")
                return
            
            # Find all universe parquet files
            universe_files = glob.glob(os.path.join(self.universe_dir, "universe.*.parquet"))
            
            if not universe_files:
                logger.error("No universe files found")
                self.universe_df = None
                return
            
            # Load all universe files
            all_dfs = []
            for file_path in sorted(universe_files):
                try:
                    df = pd.read_parquet(file_path)
                    all_dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue
            
            if all_dfs:
                self.universe_df = pd.concat(all_dfs, ignore_index=True)
                
                # Convert date to datetime if it's string
                if self.universe_df['date'].dtype == 'object':
                    self.universe_df['date'] = pd.to_datetime(self.universe_df['date'])
                
                # Sort by date
                self.universe_df.sort_values('date', inplace=True)
                
                # Add priceable column if not present (for compatibility)
                if 'priceable' not in self.universe_df.columns:
                    # Assume priceable if not dead and has advp > 0
                    self.universe_df['priceable'] = (~self.universe_df['dead']) & (self.universe_df['advp'] > 0)
                
                logger.info(f"Loaded {len(self.universe_df)} universe records from {len(all_dfs)} files")
                logger.info(f"Date range: {self.universe_df['date'].min()} to {self.universe_df['date'].max()}")
                logger.info(f"Columns: {list(self.universe_df.columns)}")
            else:
                logger.error("No universe data loaded")
                self.universe_df = None

        except Exception as e:
            logger.error(f"Error loading universe data: {e}")
            self.universe_df = None

    def get_daily_symbol_changes(self) -> tuple[list[str], list[str]]:
        """Get symbols added and dropped between the two most recent dates.

        Compares tradeable symbols between the latest date and the previous date
        in the universe data.

        Returns:
            Tuple of (symbols_added, symbols_dropped) lists
        """
        if self.universe_df is None or len(self.universe_df) == 0:
            logger.warning("No universe data available for symbol changes")
            return [], []

        # Get unique dates sorted
        dates = sorted(self.universe_df['date'].unique())

        if len(dates) < 2:
            logger.warning("Not enough dates to compute symbol changes")
            return [], []

        latest_date = dates[-1]
        previous_date = dates[-2]

        # Get tradeable symbols for latest and previous dates
        latest_df = self.universe_df[
            (self.universe_df['date'] == latest_date) & (self.universe_df['tradeable'])
        ]
        previous_df = self.universe_df[
            (self.universe_df['date'] == previous_date) & (self.universe_df['tradeable'])
        ]

        latest_symbols = set(latest_df['symbol'].unique())
        previous_symbols = set(previous_df['symbol'].unique())

        # Calculate added and dropped
        symbols_added = sorted(latest_symbols - previous_symbols)
        symbols_dropped = sorted(previous_symbols - latest_symbols)

        logger.info(
            f"Symbol changes {previous_date.strftime('%Y-%m-%d')} -> "
            f"{latest_date.strftime('%Y-%m-%d')}: "
            f"+{len(symbols_added)} added, -{len(symbols_dropped)} dropped"
        )

        return symbols_added, symbols_dropped

    def get_upcoming_delistings(self) -> list[dict]:
        """Get list of upcoming delistings from exchange metadata and manual delistings."""
        today = today_date()
        delisting_dict = self.data_loader.get_delisting_dict(today, today)
        delistings = [
            {'symbol_venue': symbol, 'delisting_date': str(delist_date)}
            for symbol, delist_date in delisting_dict.items()
            if delist_date >= today
        ]
        delistings.sort(key=lambda x: x['delisting_date'])
        return delistings

    def setup_layout(self):
        """Setup the Dash app layout"""
        self.app.layout = html.Div([
            html.H1("Universe Report", style={'textAlign': 'center'}),
            
            # Configuration selector
            html.Div([
                html.Label("Configuration File:", style={'marginBottom': '10px', 'fontSize': '16px'}),
                dcc.Dropdown(
                    id='config-selector',
                    options=[{'label': f, 'value': f} for f in os.listdir(CONFIG_DIR) if f.endswith('.json')],
                    value=None,
                    placeholder="Select a config file (optional)",
                    style={'marginBottom': '20px'}
                ),
                html.Button("Reload Data", id="reload-btn", n_clicks=0, style={'marginBottom': '10px'}),
                html.Div(id="load-status", style={"marginTop": "10px", "color": "green"}),
            ], style={'marginBottom': '30px'}),
            
            
            # Summary statistics
            html.Div([
                html.H3("Summary Statistics", style={'textAlign': 'center'}),
                html.Div(id='summary-stats', style={'textAlign': 'center', 'marginBottom': '20px'}),
            ]),

            # Daily Symbol Changes
            html.Div([
                html.H3("Daily Symbol Changes (Latest)", style={'textAlign': 'center'}),
                html.Div([
                    html.Div([
                        html.H4("Symbols Added", style={'color': 'green', 'textAlign': 'center'}),
                        html.Div(id='symbols-added', style={'textAlign': 'center', 'padding': '10px'}),
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    html.Div([
                        html.H4("Symbols Dropped", style={'color': 'red', 'textAlign': 'center'}),
                        html.Div(id='symbols-dropped', style={'textAlign': 'center', 'padding': '10px'}),
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                ]),
            ], style={'marginBottom': '30px', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),

            # Upcoming Delistings
            html.Div([
                html.H3("Upcoming Delistings", style={'textAlign': 'center'}),
                DataTable(
                    id='delistings-table',
                    columns=[
                        {'name': 'Symbol', 'id': 'symbol_venue', 'type': 'text'},
                        {'name': 'Delisting Date', 'id': 'delisting_date', 'type': 'text'},
                    ],
                    sort_action='native',
                    page_size=10,
                    page_action='native',
                    style_cell={'textAlign': 'center'},
                ),
            ], style={'marginBottom': '30px', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),

            # Universe count over time
            html.Div([
                html.H3("Universe Count Over Time", style={'textAlign': 'center'}),
                dcc.Graph(id='universe-count-timeseries'),
            ], style={'marginBottom': '30px'}),
            
            # Filter counts over time (stacked area chart)
            html.Div([
                html.H3("Filter Breakdown Over Time", style={'textAlign': 'center'}),
                dcc.Graph(id='filter-breakdown-timeseries'),
            ], style={'marginBottom': '30px'}),
            
            # Percentage of universe meeting filters
            html.Div([
                html.H3("Percentage of Universe Meeting Filters", style={'textAlign': 'center'}),
                dcc.Graph(id='filter-percentage-timeseries'),
            ], style={'marginBottom': '30px'}),
            
            # ADVP distribution over time
            html.Div([
                html.H3("ADVP Distribution Over Time", style={'textAlign': 'center'}),
                dcc.Graph(id='advp-distribution'),
            ], style={'marginBottom': '30px'}),
            
            # Total ADVP over time
            html.Div([
                html.H3("Total ADVP Over Time", style={'textAlign': 'center'}),
                dcc.Graph(id='total-advp-timeseries'),
            ], style={'marginBottom': '30px'}),
            
            # Total Market Cap over time
            html.Div([
                html.H3("Total Market Cap Over Time", style={'textAlign': 'center'}),
                dcc.Graph(id='total-marketcap-timeseries'),
            ], style={'marginBottom': '30px'}),
            
            # Current universe table
            html.Div([
                html.H3("Current Universe Status", style={'textAlign': 'center'}),
                html.Div([
                    html.Label("Filter by status:", style={'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='status-filter',
                        options=[
                            {'label': 'All', 'value': 'all'},
                            {'label': 'Priceable', 'value': 'priceable'},
                            {'label': 'Tradeable', 'value': 'tradeable'},
                            {'label': 'Fittable', 'value': 'fittable'},
                            {'label': 'Expandable', 'value': 'expandable'},
                            {'label': 'Dead', 'value': 'dead'},
                        ],
                        value='all',
                        style={'marginBottom': '15px', 'width': '200px'}
                    ),
                ]),
                DataTable(
                    id='current-universe-table',
                    columns=[
                        {'name': 'Symbol', 'id': 'symbol', 'type': 'text'},
                        {'name': 'Market Cap', 'id': 'marketcap', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=0, group=True)},
                        {'name': 'ADVP', 'id': 'advp', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=0, group=True)},
                        {'name': 'MDVP', 'id': 'mdvp', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=0, group=True)},
                        {'name': 'DVol TrMean', 'id': 'dvolume_1440_trmean', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=0, group=True)},
                        {'name': 'Position Value', 'id': 'position_value', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=0, group=True)},
                        {'name': 'Priceable', 'id': 'priceable', 'type': 'text'},
                        {'name': 'Fittable', 'id': 'fittable', 'type': 'text'},
                        {'name': 'Tradeable', 'id': 'tradeable', 'type': 'text'},
                        {'name': 'Expandable', 'id': 'expandable', 'type': 'text'},
                        {'name': 'Dead', 'id': 'dead', 'type': 'text'},
                    ],
                    sort_action='native',
                    filter_action='native',
                    page_action='none',
                    style_cell={'textAlign': 'center'},
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'priceable', 'filter_query': '{priceable} = "✓"'},
                            'backgroundColor': 'lightgreen',
                        },
                        {
                            'if': {'column_id': 'tradeable', 'filter_query': '{tradeable} = "✓"'},
                            'backgroundColor': 'lightgreen',
                        },
                        {
                            'if': {'column_id': 'fittable', 'filter_query': '{fittable} = "✓"'},
                            'backgroundColor': 'lightgreen',
                        },
                        {
                            'if': {'column_id': 'expandable', 'filter_query': '{expandable} = "✓"'},
                            'backgroundColor': 'lightgreen',
                        },
                        {
                            'if': {'column_id': 'dead', 'filter_query': '{dead} = "✓"'},
                            'backgroundColor': 'lightcoral',
                        },
                    ]
                ),
            ]),
        ])
    
    def register_callbacks(self):
        """Register Dash callbacks"""
        
        @self.app.callback(
            Output("load-status", "children"),
            Input("reload-btn", "n_clicks"),
            Input("config-selector", "value"),
            prevent_initial_call=False,
        )
        def reload_data(n_clicks, config_file):  # pylint: disable=unused-argument
            """Reload universe data when button is clicked or config changes"""
            if config_file:
                self.update_config(config_file)
            
            self.load_universe_data()
            
            if self.universe_df is not None and len(self.universe_df) > 0:
                min_date = self.universe_df['date'].min()
                max_date = self.universe_df['date'].max()

                return f"Loaded {len(self.universe_df)} universe records successfully! Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            return "No universe data loaded. Please check the logs."

        @self.app.callback(
            Output('symbols-added', 'children'),
            Output('symbols-dropped', 'children'),
            Input("reload-btn", "n_clicks"),
            prevent_initial_call=False,
        )
        def update_symbol_changes(n_clicks):  # pylint: disable=unused-argument
            """Update daily symbol changes display"""
            if self.universe_df is None:
                return "No data", "No data"

            symbols_added, symbols_dropped = self.get_daily_symbol_changes()

            # Format added symbols
            if symbols_added:
                added_content = html.Div([
                    html.Span(f"({len(symbols_added)}) ", style={'fontWeight': 'bold'}),
                    html.Span(", ".join(symbols_added))
                ])
            else:
                added_content = "None"

            # Format dropped symbols
            if symbols_dropped:
                dropped_content = html.Div([
                    html.Span(f"({len(symbols_dropped)}) ", style={'fontWeight': 'bold'}),
                    html.Span(", ".join(symbols_dropped))
                ])
            else:
                dropped_content = "None"

            return added_content, dropped_content

        @self.app.callback(
            Output('delistings-table', 'data'),
            Input("reload-btn", "n_clicks"),
            prevent_initial_call=False,
        )
        def update_delistings_table(n_clicks):  # pylint: disable=unused-argument
            return self.get_upcoming_delistings()

        @self.app.callback(
            Output('summary-stats', 'children'),
            Output('universe-count-timeseries', 'figure'),
            Output('filter-breakdown-timeseries', 'figure'),
            Output('filter-percentage-timeseries', 'figure'),
            Output('advp-distribution', 'figure'),
            Output('total-advp-timeseries', 'figure'),
            Output('total-marketcap-timeseries', 'figure'),
            Output('current-universe-table', 'data'),
            Input('status-filter', 'value'),
            Input("reload-btn", "n_clicks"),
        )
        def update_displays(status_filter, n_clicks):  # pylint: disable=unused-argument
            """Update all displays using entire history"""
            if self.universe_df is None:
                return "", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), []
            
            # Use all data (no date filtering)
            filtered_df = self.universe_df.copy()
            
            if len(filtered_df) == 0:
                return "No data loaded", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), []
            
            # Calculate daily statistics
            daily_stats = filtered_df.groupby('date').agg({
                'symbol': 'count',
                'priceable': 'sum',
                'tradeable': 'sum',
                'fittable': 'sum',
                'expandable': 'sum',
                'dead': 'sum',
                'advp': ['mean', 'median', 'sum'],
                'marketcap': ['mean', 'median', 'sum']
            }).reset_index()
            
            # Flatten column names
            daily_stats.columns = ['date', 'total_count', 'priceable', 'tradeable', 
                                   'fittable', 'expandable', 'dead', 'advp_mean', 'advp_median', 'advp_sum',
                                   'marketcap_mean', 'marketcap_median', 'marketcap_sum']
            
            # Summary statistics
            latest_date = daily_stats['date'].max()
            latest_stats = daily_stats[daily_stats['date'] == latest_date].iloc[0]
            
            summary_text = html.Div([
                html.P(f"Latest Date: {latest_date.strftime('%Y-%m-%d')}"),
                html.P(f"Total Symbols: {latest_stats['total_count']:.0f}"),
                html.P(f"Priceable: {latest_stats['priceable']:.0f} ({latest_stats['priceable']/latest_stats['total_count']*100:.1f}%)"),
                html.P(f"Tradeable: {latest_stats['tradeable']:.0f} ({latest_stats['tradeable']/latest_stats['total_count']*100:.1f}%)"),
                html.P(f"Fittable: {latest_stats['fittable']:.0f} ({latest_stats['fittable']/latest_stats['total_count']*100:.1f}%)"),
                html.P(f"Expandable: {latest_stats['expandable']:.0f} ({latest_stats['expandable']/latest_stats['total_count']*100:.1f}%)"),
                html.P(f"Dead: {latest_stats['dead']:.0f} ({latest_stats['dead']/latest_stats['total_count']*100:.1f}%)"),
                html.P(f"Mean ADVP: ${latest_stats['advp_mean']:,.0f}"),
                html.P(f"Median ADVP: ${latest_stats['advp_median']:,.0f}"),
                html.P(f"Total ADVP: ${latest_stats['advp_sum']:,.0f}"),
                html.P(f"Total Market Cap: ${latest_stats['marketcap_sum']:,.0f}"),
            ])
            
            # Universe count over time
            count_fig = go.Figure()
            count_fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['total_count'],
                mode='lines',
                name='Total Universe',
                line={'width': 2}
            ))
            count_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Symbols",
                hovermode='x unified'
            )
            
            # Filter breakdown (stacked area chart)
            breakdown_fig = go.Figure()
            
            # Calculate exclusive counts for stacking
            daily_stats['only_priceable'] = daily_stats['priceable'] - daily_stats['tradeable']
            daily_stats['only_tradeable'] = daily_stats['tradeable'] - daily_stats['fittable']
            daily_stats['only_fittable'] = daily_stats['fittable'] - daily_stats['expandable']
            
            breakdown_fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['expandable'],
                mode='lines',
                name='Expandable',
                stackgroup='one',
                fillcolor='darkgreen'
            ))
            breakdown_fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['only_fittable'],
                mode='lines',
                name='Fittable (not expandable)',
                stackgroup='one',
                fillcolor='green'
            ))
            breakdown_fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['only_tradeable'],
                mode='lines',
                name='Tradeable (not fittable)',
                stackgroup='one',
                fillcolor='yellow'
            ))
            breakdown_fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['only_priceable'],
                mode='lines',
                name='Priceable (not tradeable)',
                stackgroup='one',
                fillcolor='orange'
            ))
            breakdown_fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['dead'],
                mode='lines',
                name='Dead',
                stackgroup='one',
                fillcolor='red'
            ))
            
            breakdown_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Symbols",
                hovermode='x unified'
            )
            
            # Percentage of universe meeting filters
            percentage_fig = go.Figure()
            
            percentage_fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['priceable'] / daily_stats['total_count'] * 100,
                mode='lines',
                name='Priceable %',
                line={'width': 2}
            ))
            percentage_fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['tradeable'] / daily_stats['total_count'] * 100,
                mode='lines',
                name='Tradeable %',
                line={'width': 2}
            ))
            percentage_fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['fittable'] / daily_stats['total_count'] * 100,
                mode='lines',
                name='Fittable %',
                line={'width': 2}
            ))
            percentage_fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['expandable'] / daily_stats['total_count'] * 100,
                mode='lines',
                name='Expandable %',
                line={'width': 2}
            ))
            percentage_fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['dead'] / daily_stats['total_count'] * 100,
                mode='lines',
                name='Dead %',
                line={'width': 2, 'dash': 'dash', 'color': 'red'}
            ))
            
            percentage_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Percentage of Universe (%)",
                hovermode='x unified',
                yaxis={'range': [0, 100]}
            )
            
            # ADVP distribution
            advp_fig = go.Figure()
            
            # Add percentile lines
            advp_fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['advp_median'],
                mode='lines',
                name='Median ADVP',
                line={'width': 2}
            ))
            advp_fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['advp_mean'],
                mode='lines',
                name='Mean ADVP',
                line={'width': 2, 'dash': 'dash'}
            ))
            
            # Add reference lines for thresholds
            advp_fig.add_hline(y=15e6, line_dash="dash", line_color="green", 
                              annotation_text="Tradeable/Fittable Threshold ($15M)")
            advp_fig.add_hline(y=55e6, line_dash="dash", line_color="darkgreen", 
                              annotation_text="Expandable Threshold ($55M)")
            
            advp_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="ADVP (USD)",
                hovermode='x unified',
                yaxis_type='log'
            )
            
            # Total ADVP over time
            total_advp_fig = go.Figure()
            total_advp_fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['advp_sum'],
                mode='lines',
                name='Total ADVP',
                line={'width': 2, 'color': 'blue'}
            ))
            
            # Add a secondary trace for tradeable universe ADVP
            tradeable_advp = filtered_df[filtered_df['tradeable']].groupby('date')['advp'].sum().reset_index()
            total_advp_fig.add_trace(go.Scatter(
                x=tradeable_advp['date'],
                y=tradeable_advp['advp'],
                mode='lines',
                name='Tradeable Universe ADVP',
                line={'width': 2, 'color': 'green', 'dash': 'dash'}
            ))
            
            total_advp_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Total ADVP (USD)",
                hovermode='x unified',
                yaxis={'tickformat': ',.0f'}
            )
            
            # Total Market Cap over time
            total_marketcap_fig = go.Figure()
            total_marketcap_fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['marketcap_sum'],
                mode='lines',
                name='Total Market Cap',
                line={'width': 2, 'color': 'purple'}
            ))
            
            # Add a secondary trace for tradeable universe market cap
            tradeable_marketcap = filtered_df[filtered_df['tradeable']].groupby('date')['marketcap'].sum().reset_index()
            total_marketcap_fig.add_trace(go.Scatter(
                x=tradeable_marketcap['date'],
                y=tradeable_marketcap['marketcap'],
                mode='lines',
                name='Tradeable Universe Market Cap',
                line={'width': 2, 'color': 'darkgreen', 'dash': 'dash'}
            ))
            
            total_marketcap_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Total Market Cap (USD)",
                hovermode='x unified',
                yaxis={'tickformat': ',.0f'}
            )
            
            # Current universe table
            latest_df = filtered_df[filtered_df['date'] == filtered_df['date'].max()].copy()
            
            # Apply status filter
            if status_filter == 'priceable':
                latest_df = latest_df[latest_df['priceable']]
            elif status_filter == 'tradeable':
                latest_df = latest_df[latest_df['tradeable']]
            elif status_filter == 'fittable':
                latest_df = latest_df[latest_df['fittable']]
            elif status_filter == 'expandable':
                latest_df = latest_df[latest_df['expandable']]
            elif status_filter == 'dead':
                latest_df = latest_df[latest_df['dead']]
            
            # Sort non-dead first, then by ADVP descending
            latest_df = latest_df.sort_values(['dead', 'advp'], ascending=[True, False])
            
            # Convert boolean columns to checkmarks
            for col in ['priceable', 'tradeable', 'fittable', 'expandable', 'dead']:
                latest_df[col] = latest_df[col].apply(lambda x: '✓' if x else '')
            
            table_data = latest_df[['symbol', 'marketcap', 'advp', 'mdvp', 'dvolume_1440_trmean',
                                    'position_value', 'priceable', 'fittable', 'tradeable',
                                    'expandable', 'dead']].to_dict('records')
            
            return summary_text, count_fig, breakdown_fig, percentage_fig, advp_fig, total_advp_fig, total_marketcap_fig, table_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Universe report showing filter counts over time')
    parser.add_argument('-p', '--port', help='port', type=int, default=None)
    parser.add_argument('-d', '--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('-i', '--interval', help='Refresh interval in seconds', type=int, default=300)
    args = parser.parse_args()

    # Determine port
    if args.port:
        port = args.port
    else:
        port = 8056 if not LOCAL else 8066

    report = UniverseReport(port=port, interval_secs=args.interval, debug=args.debug)
    report.run(debug=args.debug)
