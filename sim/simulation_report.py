import argparse
import logging.config
import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, Output, Input, callback_context
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Scheme
from plotly.subplots import make_subplots

from lib.util.directory import ROOT_DIR
from lib.util.logging_util import get_logging_config

logging.config.dictConfig(get_logging_config("simulation_report"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define simulation directory path
SIM_DIR = os.path.join(ROOT_DIR, 'sims')

# Default port for all reports
REPORT_PORT = 8064


class SimulationReport:
    def __init__(self, sim_name: Optional[str] = None, port: int = REPORT_PORT):
        self.port = port
        self.app = Dash(__name__)
        
        # Get list of available simulations
        self.available_simulations = self.get_available_simulations()
        
        # Don't automatically load any simulation - let user select
        self.sim_name = None
        
        # Initialize data attributes
        self.sim_dir = None
        self.pnl_details_path = None
        self.pnl_details_df = None
        self.aggregate_df = None
        self.sim_df = None
        
        # Setup the dash app layout and callbacks
        self.setup_layout()
        self.setup_callbacks()
    
    def get_available_simulations(self) -> list:
        """Get list of available simulation directories"""
        try:
            if not os.path.exists(SIM_DIR):
                logger.error(f"Simulation directory does not exist: {SIM_DIR}")
                return []
            
            # Get all directories in SIM_DIR that contain pnl_details.parquet
            simulations = []
            for item in os.listdir(SIM_DIR):
                item_path = os.path.join(SIM_DIR, item)
                if os.path.isdir(item_path):
                    pnl_file = os.path.join(item_path, 'pnl_details.parquet')
                    if os.path.exists(pnl_file):
                        # Get the modification time of the directory
                        mod_time = os.path.getmtime(item_path)
                        simulations.append((item, mod_time))
            
            # Sort by modification time in descending order (most recent first)
            simulations.sort(key=lambda x: x[1], reverse=True)
            
            # Extract just the simulation names
            simulation_names = [sim[0] for sim in simulations]
            
            logger.info(f"Found {len(simulation_names)} simulations with pnl_details.parquet")
            return simulation_names
        except Exception as e:
            logger.error(f"Error getting available simulations: {e}")
            return []
    
    def update_simulation_paths(self):
        """Update paths for current simulation"""
        if self.sim_name:
            self.sim_dir = os.path.join(SIM_DIR, self.sim_name)
            self.pnl_details_path = os.path.join(self.sim_dir, 'pnl_details.parquet')
    
    def load_simulation_data(self):
        """Load simulation data from the simulation directory"""
        # Clear all existing data before loading new data
        logger.info("Clearing existing data before loading new simulation")
        self.sim_df = None
        self.pnl_details_df = None
        self.aggregate_df = None
        
        if not self.sim_dir:
            logger.error("No simulation directory set - sim_dir is None")
            return
        
        # Update pnl_details_path when sim_dir changes
        self.pnl_details_path = os.path.join(self.sim_dir, 'pnl_details.parquet')
            
        try:
            logger.info(f"Loading simulation data from {self.sim_dir}")
            
            # Load all sim.{date}.parquet files
            sim_files = []
            for file in os.listdir(self.sim_dir):
                if file.startswith('sim.') and file.endswith('.parquet'):
                    sim_files.append(os.path.join(self.sim_dir, file))
            
            if not sim_files:
                logger.warning(f"No sim.*.parquet files found in {self.sim_dir}")
                # Fall back to loading pnl_details.parquet if available
                if os.path.exists(self.pnl_details_path):
                    logger.info(f"Falling back to loading {self.pnl_details_path}")
                    self.pnl_details_df = pd.read_parquet(self.pnl_details_path)
                else:
                    logger.error(f"No simulation data found in {self.sim_dir}")
                    return
            else:
                # Load and concatenate all sim files
                logger.info(f"Found {len(sim_files)} sim files to load")
                sim_dfs = []
                for file in sorted(sim_files):
                    df = pd.read_parquet(file)
                    sim_dfs.append(df)
                
                # Concatenate all dataframes
                self.sim_df = pd.concat(sim_dfs, ignore_index=False)
                logger.info(f"Loaded {len(self.sim_df)} records from sim files")
                logger.info(f"Sim data columns: {list(self.sim_df.columns)[:10]}...")
                logger.info(f"Sim data index: {self.sim_df.index.names}")
                
                # Debug: Check P&L values
                if 'pnl' in self.sim_df.columns:
                    logger.info(f"P&L range: min={self.sim_df['pnl'].min():.2f}, max={self.sim_df['pnl'].max():.2f}")
                    # Get last timestamp
                    if 'ts' in self.sim_df.index.names:
                        last_ts = self.sim_df.index.get_level_values('ts').max()
                        last_data = self.sim_df.xs(last_ts, level='ts')
                        total_pnl_last = last_data['pnl'].sum()
                        logger.info(f"Total P&L at last timestamp {last_ts}: ${total_pnl_last:,.2f}")
                
                # Also load pnl_details if it exists for backward compatibility
                if os.path.exists(self.pnl_details_path):
                    self.pnl_details_df = pd.read_parquet(self.pnl_details_path)
            
            # Reset index to convert MultiIndex to regular columns
            if hasattr(self, 'pnl_details_df') and self.pnl_details_df is not None:
                if isinstance(self.pnl_details_df.index, pd.MultiIndex):
                    self.pnl_details_df = self.pnl_details_df.reset_index()
            
            # Create aggregate dataframe by timestamp
            # Use sim_df if available (more detailed data), otherwise use pnl_details_df
            if hasattr(self, 'sim_df') and self.sim_df is not None:
                # Reset index if needed
                if isinstance(self.sim_df.index, pd.MultiIndex):
                    self.sim_df = self.sim_df.reset_index()
                
                # Calculate aggregates from sim data
                # IMPORTANT: 'pnl' in sim files is cumulative P&L per symbol across the entire simulation
                # At each timestamp, we need to sum across symbols to get total portfolio P&L
                logger.info("Starting aggregation of sim data...")
                self.aggregate_df = self.sim_df.groupby('ts').agg({
                    # P&L and position metrics
                    'pnl': 'sum',  # Sum of cumulative P&L across all symbols
                    'cash': 'sum',  # Total cash
                    'position': 'sum',  # Total position value (notional)
                    'qty': lambda x: (x != 0).sum(),  # Count of non-zero positions
                    
                    # Trading metrics - these need to be carefully handled
                    'executed_dollars': lambda x: x.abs().sum(),  # Total dollars traded at this timestamp
                    'fees': 'sum',  # Total fees at this timestamp
                    'funding_income': 'sum',  # Total funding income at this timestamp
                    
                    # Risk metrics
                    'risk_1440': 'mean',  # Average risk
                }).reset_index()
                logger.info(f"Aggregated to {len(self.aggregate_df)} timestamp records")
                
                # Since pnl is already cumulative in the sim files, we don't need to cumsum again
                self.aggregate_df['cumulative_pnl'] = self.aggregate_df['pnl']
                
                # Log P&L info
                logger.info(f"Aggregate P&L stats:")
                logger.info(f"  First cumulative P&L: ${self.aggregate_df['cumulative_pnl'].iloc[0]:,.2f}")
                logger.info(f"  Last cumulative P&L: ${self.aggregate_df['cumulative_pnl'].iloc[-1]:,.2f}")
                logger.info(f"  Min cumulative P&L: ${self.aggregate_df['cumulative_pnl'].min():,.2f}")
                logger.info(f"  Max cumulative P&L: ${self.aggregate_df['cumulative_pnl'].max():,.2f}")
                
                # For fees and funding, we need to calculate cumulative if they're incremental
                # Check if fees are already cumulative by seeing if they only increase
                fees_diff = self.aggregate_df['fees'].diff()
                if (fees_diff < 0).any():
                    # Fees can go negative (income), so just cumsum
                    self.aggregate_df['cumulative_fees'] = self.aggregate_df['fees'].cumsum()
                else:
                    # Fees might already be cumulative
                    self.aggregate_df['cumulative_fees'] = self.aggregate_df['fees']
                
                self.aggregate_df['cumulative_funding'] = self.aggregate_df['funding_income'].cumsum()
                
                # Calculate daily P&L as the difference between consecutive cumulative values
                self.aggregate_df['total_pnl_daily'] = self.aggregate_df['pnl'].diff().fillna(self.aggregate_df['pnl'].iloc[0])
                self.aggregate_df['total_pnl_cumulative'] = self.aggregate_df['cumulative_pnl']
                
                # Log daily P&L calculation
                logger.info(f"Daily P&L calculation:")
                logger.info(f"  Sum of daily P&Ls: ${self.aggregate_df['total_pnl_daily'].sum():,.2f}")
                logger.info(f"  Should match last cumulative P&L: ${self.aggregate_df['cumulative_pnl'].iloc[-1]:,.2f}")
                self.aggregate_df['fees_usd_daily'] = self.aggregate_df['fees']
                self.aggregate_df['fees_usd_cumulative'] = self.aggregate_df['cumulative_fees']
                self.aggregate_df['funding_income_daily'] = self.aggregate_df['funding_income']
                self.aggregate_df['funding_income_cumulative'] = self.aggregate_df['cumulative_funding']
                
                # For executed dollars, we need to calculate daily values
                self.aggregate_df['dollars_traded_daily'] = self.aggregate_df['executed_dollars']
                
                self.aggregate_df['notional'] = self.aggregate_df['position']
                self.aggregate_df['position_count'] = self.aggregate_df['qty']
                
                # For volume split, estimate 50/50 if not available
                self.aggregate_df['dollars_buy_daily'] = self.aggregate_df['executed_dollars'] / 2
                self.aggregate_df['dollars_sell_daily'] = self.aggregate_df['executed_dollars'] / 2
                self.aggregate_df['fill_cnt_daily'] = self.aggregate_df['qty']  # Approximate
                
            elif hasattr(self, 'pnl_details_df') and self.pnl_details_df is not None:
                # Original aggregation logic for pnl_details_df
                self.aggregate_df = self.pnl_details_df.groupby('ts').agg({
                # P&L aggregations - these are per-symbol daily increments, so sum across symbols
                'realized_daily': 'sum',
                'unrealized_daily': 'sum',
                'total_pnl_daily': 'sum',  # Sum daily increments across all symbols
                'total_pnl_cumulative': 'sum',  # Sum cumulative P&L across all symbols
                'unrealized_pnl': 'sum',
                
                # Position aggregations
                'notional': lambda x: x[x != 0].sum(),  # Sum of non-zero positions
                'qty': lambda x: x[x != 0].count(),  # Count of non-zero positions
                
                # Trading activity
                'fill_cnt_daily': 'sum',
                'dollars_traded_daily': 'sum',
                'dollars_buy_daily': 'sum',
                'dollars_sell_daily': 'sum',
                
                # Costs
                'fees_usd_daily': 'sum',
                'fees_usd_cumulative': 'sum',
                'funding_income_daily': 'sum',
                'funding_income_cumulative': 'sum',
            }).reset_index()
            
            # Calculate additional metrics
            self.aggregate_df['net_exposure'] = self.aggregate_df['notional']
            self.aggregate_df['position_count'] = self.aggregate_df['qty']
            
            # Calculate average position size using absolute values from the raw data
            if hasattr(self, 'sim_df') and self.sim_df is not None:
                # Calculate sum of absolute positions per timestamp from sim data
                notional_abs_sum = self.sim_df[self.sim_df['position'] != 0].groupby('ts')['position'].apply(lambda x: x.abs().sum()).reset_index(name='notional_abs_sum')
                self.aggregate_df = self.aggregate_df.merge(notional_abs_sum, on='ts', how='left')
                self.aggregate_df['avg_position_size'] = self.aggregate_df['notional_abs_sum'] / self.aggregate_df['position_count'].replace(0, np.nan)
            elif hasattr(self, 'pnl_details_df') and self.pnl_details_df is not None:
                # Calculate sum of absolute notionals per timestamp
                notional_abs_sum = self.pnl_details_df[self.pnl_details_df['notional'] != 0].groupby('ts')['notional'].apply(lambda x: x.abs().sum()).reset_index(name='notional_abs_sum')
                self.aggregate_df = self.aggregate_df.merge(notional_abs_sum, on='ts', how='left')
                self.aggregate_df['avg_position_size'] = self.aggregate_df['notional_abs_sum'] / self.aggregate_df['position_count'].replace(0, np.nan)
            else:
                # Fallback: use absolute value of net exposure as approximation
                self.aggregate_df['avg_position_size'] = self.aggregate_df['notional'].abs() / self.aggregate_df['position_count'].replace(0, np.nan)
            
            # Use the already calculated cumulative values
            self.aggregate_df['cumulative_pnl'] = self.aggregate_df['total_pnl_cumulative']
            self.aggregate_df['cumulative_fees'] = self.aggregate_df['fees_usd_cumulative']
            self.aggregate_df['cumulative_funding'] = self.aggregate_df['funding_income_cumulative']
            
            # Don't calculate turnover at timestamp level - it's not meaningful
            # Turnover should be calculated at daily level only
            # Set to 0 for now, will be calculated properly at daily level
            self.aggregate_df['turnover'] = 0
            
            if hasattr(self, 'pnl_details_df') and self.pnl_details_df is not None:
                logger.info(f"Loaded {len(self.pnl_details_df)} PnL detail records")
            if self.aggregate_df is not None:
                logger.info(f"Aggregated to {len(self.aggregate_df)} timestamp records")
                logger.info(f"Date range: {self.aggregate_df['ts'].min()} to {self.aggregate_df['ts'].max()}")
            
        except Exception as e:
            logger.error(f"Error loading simulation data: {e}")
            self.pnl_details_df = None
            self.aggregate_df = None
    
    def setup_layout(self):
        """Setup the Dash app layout"""
        # Create dropdown options
        dropdown_options = [{'label': sim, 'value': sim} for sim in self.available_simulations]
        
        self.app.layout = html.Div([
            # Store component to track data loading state
            dcc.Store(id='data-loaded-state'),
            
            html.H1("Simulation Report", style={'textAlign': 'center'}),
            
            # Simulation selector
            html.Div([
                html.Label("Select Simulation:", style={'marginRight': '10px', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='simulation-dropdown',
                    options=dropdown_options,
                    value=self.sim_name,
                    style={'width': '300px', 'display': 'inline-block', 'verticalAlign': 'middle'},
                    clearable=False
                ),
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            # Current simulation name
            html.H3(id="simulation-name", style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            # Reload button
            html.Div([
                html.Button("Reload Data", id="reload-btn", n_clicks=0, style={'marginBottom': '10px'}),
                html.Div(id="load-status", style={"marginTop": "10px", "color": "green"}),
            ], style={'marginBottom': '30px', 'textAlign': 'center'}),
            
            # Summary statistics
            html.Div([
                html.H3("Summary Statistics", style={'textAlign': 'center', 'marginBottom': '20px'}),
                html.Div(id='summary-stats', style={'marginBottom': '30px'}),
            ], style={
                'backgroundColor': '#ffffff',
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '30px'
            }),
            
            # P&L over time
            html.Div([
                html.H3("P&L Performance", style={'textAlign': 'center'}),
                dcc.Graph(id='pnl-timeseries'),
            ], style={'marginBottom': '30px'}),
            
            # Portfolio metrics
            html.Div([
                html.H3("Portfolio Metrics", style={'textAlign': 'center'}),
                dcc.Graph(id='portfolio-metrics'),
            ], style={'marginBottom': '30px'}),
            
            # Trading activity and costs
            html.Div([
                html.H3("Trading Activity & Costs", style={'textAlign': 'center'}),
                dcc.Graph(id='trading-activity'),
            ], style={'marginBottom': '30px'}),
            
            # Hourly patterns
            html.Div([
                html.H3("Intraday Patterns", style={'textAlign': 'center'}),
                dcc.Graph(id='hourly-patterns'),
            ], style={'marginBottom': '30px'}),
            
            # Daily returns distribution
            html.Div([
                html.H3("Daily Returns Distribution", style={'textAlign': 'center'}),
                dcc.Graph(id='returns-distribution'),
            ], style={'marginBottom': '30px'}),
            
            # Monthly statistics table
            html.Div([
                html.H3("Monthly Statistics", style={'textAlign': 'center'}),
                DataTable(
                    id='monthly-stats-table',
                    columns=[
                        {'name': 'Month', 'id': 'month', 'type': 'text'},
                        {'name': 'Monthly P&L', 'id': 'monthly_pnl', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                        {'name': 'Cumulative P&L', 'id': 'cumulative_pnl', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                        {'name': 'Avg Positions', 'id': 'avg_positions', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=1, group=True)},
                        {'name': 'Avg Net Exposure', 'id': 'avg_net_exposure', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=0, group=True)},
                        {'name': 'Volume Traded', 'id': 'volume_traded', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=0, group=True)},
                        {'name': 'Fees', 'id': 'fees', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                        {'name': 'Funding', 'id': 'funding', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                        {'name': 'Sharpe', 'id': 'sharpe', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=2, group=False)},
                    ],
                    sort_action='native',
                    filter_action='native',
                    page_action='native',
                    page_size=20,
                    style_cell={'textAlign': 'center'},
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'monthly_pnl', 'filter_query': '{monthly_pnl} > 0'},
                            'color': 'green',
                        },
                        {
                            'if': {'column_id': 'monthly_pnl', 'filter_query': '{monthly_pnl} < 0'},
                            'color': 'red',
                        },
                        {
                            'if': {'column_id': 'cumulative_pnl', 'filter_query': '{cumulative_pnl} > 0'},
                            'color': 'green',
                        },
                        {
                            'if': {'column_id': 'cumulative_pnl', 'filter_query': '{cumulative_pnl} < 0'},
                            'color': 'red',
                        },
                    ]
                ),
            ]),
            
            # Daily PnL and Returns table
            html.Div([
                html.H3("Daily P&L, Notional Size, and Returns", style={'textAlign': 'center'}),
                DataTable(
                    id='daily-pnl-returns-table',
                    columns=[
                        {'name': 'Date', 'id': 'date', 'type': 'datetime'},
                        {'name': 'Daily P&L', 'id': 'daily_pnl', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                        {'name': 'Gross Notional (EOD)', 'id': 'gross_notional', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=0, group=True)},
                        {'name': 'Daily Return (%)', 'id': 'daily_return_pct', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=4, group=False)},
                        {'name': 'Cumulative Return (%)', 'id': 'cumulative_return_pct', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=2, group=False)},
                    ],
                    sort_action='native',
                    filter_action='native',
                    page_action='native',
                    page_size=20,
                    style_cell={'textAlign': 'center'},
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'daily_pnl', 'filter_query': '{daily_pnl} > 0'},
                            'color': 'green',
                        },
                        {
                            'if': {'column_id': 'daily_pnl', 'filter_query': '{daily_pnl} < 0'},
                            'color': 'red',
                        },
                        {
                            'if': {'column_id': 'daily_return_pct', 'filter_query': '{daily_return_pct} > 0'},
                            'color': 'green',
                        },
                        {
                            'if': {'column_id': 'daily_return_pct', 'filter_query': '{daily_return_pct} < 0'},
                            'color': 'red',
                        },
                    ]
                ),
            ]),
            
            # Single-Security Risk Analysis
            html.Div([
                html.H3("Single-Security Risk Analysis", style={'textAlign': 'center'}),
                
                # Cumulative P&L by Security Bar Chart
                html.Div([
                    html.H4("Cumulative P&L by Security", style={'textAlign': 'center'}),
                    dcc.Graph(id='security-pnl-bar-chart'),
                ], style={'marginBottom': '30px'}),
                
                # Time series of daily single security P&L
                html.Div([
                    html.H4("Daily Single-Security P&L History", style={'textAlign': 'center'}),
                    dcc.Graph(id='single-security-pnl-timeseries'),
                ], style={'marginBottom': '30px'}),
                
                # Table of largest single-day security P&L moves
                html.Div([
                    html.H4("Top 10 Largest Single-Day Security P&L Moves", style={'textAlign': 'center'}),
                    DataTable(
                        id='top-security-pnl-table',
                        columns=[
                            {'name': 'Symbol', 'id': 'symbol', 'type': 'text'},
                            {'name': 'Date', 'id': 'date', 'type': 'datetime'},
                            {'name': 'P&L', 'id': 'pnl', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                        ],
                        sort_action='native',
                        page_size=10,
                        style_cell={'textAlign': 'center'},
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'pnl', 'filter_query': '{pnl} > 0'},
                                'color': 'green',
                            },
                            {
                                'if': {'column_id': 'pnl', 'filter_query': '{pnl} < 0'},
                                'color': 'red',
                            },
                        ]
                    ),
                ]),
            ], style={'marginBottom': '30px'}),
            
            # Drawdown Analysis
            html.Div([
                html.H3("Drawdown Analysis", style={'textAlign': 'center'}),
                
                # Drawdown chart
                html.Div([
                    dcc.Graph(id='drawdown-chart'),
                ], style={'marginBottom': '20px'}),
                
                # Table of largest drawdowns
                html.Div([
                    html.H4("Top 5 Largest Drawdowns", style={'textAlign': 'center'}),
                    DataTable(
                        id='drawdown-table',
                        columns=[
                            {'name': 'Start Date', 'id': 'start_date', 'type': 'datetime'},
                            {'name': 'End Date', 'id': 'end_date', 'type': 'datetime'},
                            {'name': 'Duration (Days)', 'id': 'duration_days', 'type': 'numeric'},
                            {'name': 'Peak Value ($)', 'id': 'peak_value', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                            {'name': 'Trough Value ($)', 'id': 'trough_value', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                            {'name': 'Dollar Loss', 'id': 'dollar_loss', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                            {'name': 'Percent Loss', 'id': 'percent_loss', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=2, group=False)},
                        ],
                        sort_action='native',
                        page_size=5,
                        style_cell={'textAlign': 'center'},
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'dollar_loss'},
                                'color': 'red',
                            },
                            {
                                'if': {'column_id': 'percent_loss'},
                                'color': 'red',
                            },
                        ]
                    ),
                ]),
            ], style={'marginBottom': '30px'}),
        ])
    
    def setup_callbacks(self):
        """Setup Dash callbacks"""
        
        @self.app.callback(
            Output("simulation-name", "children"),
            Output("load-status", "children"),
            Output("data-loaded-state", "data"),
            Input("simulation-dropdown", "value"),
            Input("reload-btn", "n_clicks"),
            prevent_initial_call=False,
        )
        def update_simulation(selected_sim, n_clicks):  # pylint: disable=unused-argument
            """Update simulation when dropdown changes or reload button is clicked"""
            logger.info(f"=== Update simulation callback triggered ===")
            logger.info(f"  Selected sim: {selected_sim}")
            logger.info(f"  Current sim: {self.sim_name}")
            
            # Check which input triggered the callback
            ctx = callback_context
            if not ctx.triggered:
                trigger_id = None
                logger.info("  Trigger: None (initial call)")
            else:
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
                logger.info(f"  Trigger: {trigger_id}")
            
            # Update simulation if dropdown changed or if we have a selection but no sim_name
            if selected_sim and (trigger_id == 'simulation-dropdown' or self.sim_name is None):
                if selected_sim != self.sim_name:
                    logger.info(f"  Updating simulation from {self.sim_name} to {selected_sim}")
                    self.sim_name = selected_sim
                    self.update_simulation_paths()
                    
                    # Load data for the new simulation
                    logger.info(f"  Loading data for simulation: {self.sim_name}")
                    self.load_simulation_data()
            elif trigger_id == 'reload-btn' and self.sim_name:
                # Reload current simulation
                logger.info(f"  Reloading data for simulation: {self.sim_name}")
                self.load_simulation_data()
            else:
                logger.info("  No simulation selected or no change, skipping data load")
            
            # Update simulation name display
            sim_name_display = f"Current Simulation: {self.sim_name}" if self.sim_name else "No Simulation Selected"
            
            # Update status message and data state
            if self.aggregate_df is not None and len(self.aggregate_df) > 0:
                min_date = self.aggregate_df['ts'].min()
                max_date = self.aggregate_df['ts'].max()
                status_msg = f"Loaded {len(self.aggregate_df)} aggregate records successfully! Date range: {min_date} to {max_date}"
                data_state = {"loaded": True, "sim_name": self.sim_name, "timestamp": pd.Timestamp.now().isoformat()}
            else:
                status_msg = "No simulation data loaded. Please check the logs."
                data_state = {"loaded": False, "sim_name": None, "timestamp": pd.Timestamp.now().isoformat()}
            
            return sim_name_display, status_msg, data_state
        
        @self.app.callback(
            Output('summary-stats', 'children'),
            Output('pnl-timeseries', 'figure'),
            Output('portfolio-metrics', 'figure'),
            Output('trading-activity', 'figure'),
            Output('hourly-patterns', 'figure'),
            Output('returns-distribution', 'figure'),
            Output('monthly-stats-table', 'data'),
            Output('daily-pnl-returns-table', 'data'),
            Output('security-pnl-bar-chart', 'figure'),
            Output('single-security-pnl-timeseries', 'figure'),
            Output('top-security-pnl-table', 'data'),
            Output('drawdown-chart', 'figure'),
            Output('drawdown-table', 'data'),
            Input("data-loaded-state", "data"),  # Trigger when data state changes
        )
        def update_displays(data_state):  # pylint: disable=unused-argument
            """Update all displays"""
            logger.info("=== Update displays callback triggered ===")
            logger.info(f"  Data state: {data_state}")
            
            # Check if data is loaded
            if not data_state or not data_state.get("loaded", False):
                logger.info("  Data not loaded yet, returning empty displays")
                return "", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), [], [], go.Figure(), go.Figure(), [], go.Figure(), []
            
            if self.aggregate_df is None:
                logger.warning("  No aggregate_df available, returning empty displays")
                return "", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), [], [], go.Figure(), go.Figure(), [], go.Figure(), []
            
            logger.info(f"  Aggregate_df shape: {self.aggregate_df.shape}")
            logger.info(f"  Aggregate_df columns: {list(self.aggregate_df.columns)}")
            
            df = self.aggregate_df.copy()
            
            # Calculate summary statistics
            total_pnl = df['cumulative_pnl'].iloc[-1]  # Use the correctly calculated cumulative_pnl
            total_fees = df['cumulative_fees'].iloc[-1]  # Use the correctly calculated cumulative_fees
            total_funding = df['cumulative_funding'].iloc[-1]  # Use the correctly calculated cumulative_funding
            
            logger.info(f"  Summary statistics:")
            logger.info(f"    Total P&L: ${total_pnl:,.2f}")
            logger.info(f"    Total Fees: ${total_fees:,.2f}")
            logger.info(f"    Total Funding: ${total_funding:,.2f}")
            
            # Add date column for daily aggregations
            df['date'] = pd.to_datetime(df['ts']).dt.date
            
            # For total traded, group by date first and take last value per day, then sum
            total_traded = df.groupby('date')['dollars_traded_daily'].last().sum()
            avg_positions = df['position_count'].mean()
            max_exposure = df['net_exposure'].abs().max()
            
            # Calculate average daily metrics
            # Group by date to get daily totals
            if hasattr(self, 'sim_df') and self.sim_df is not None:
                # With sim data, we need to sum executed_dollars within each day
                # as executed_dollars is per trade/timestamp, not cumulative
                daily_volume = df.groupby('date')['dollars_traded_daily'].sum()
            else:
                # With pnl_details data, dollars_traded_daily is cumulative, so take last
                daily_volume = df.groupby('date')['dollars_traded_daily'].last()
            
            # For turnover, recalculate properly: daily volume / end-of-day gross notional
            daily_notional = df.groupby('date')['notional_abs_sum'].last()  # End-of-day gross notional
            daily_turnover = daily_volume / (daily_notional + 1e-10)
            avg_daily_volume = daily_volume.mean()
            avg_daily_turnover = daily_turnover.mean()
            
            # Calculate daily statistics
            # First get the daily P&L from the raw data
            logger.info("  Calculating daily P&L...")
            if hasattr(self, 'sim_df') and self.sim_df is not None:
                # With sim data, we can calculate daily P&L more accurately
                sim_data = self.sim_df.copy()
                
                # The sim data already has a 'date' column
                if 'date' in sim_data.columns:
                    sim_data['date'] = pd.to_datetime(sim_data['date']).dt.date
                else:
                    # Fallback: extract date from index
                    if isinstance(sim_data.index, pd.MultiIndex):
                        # MultiIndex case - get the ts level
                        if 'ts' in sim_data.index.names:
                            ts_index = pd.to_datetime(sim_data.index.get_level_values('ts'))
                            sim_data['date'] = ts_index.date if isinstance(ts_index, pd.DatetimeIndex) else ts_index.dt.date
                        else:
                            # Try to find the timestamp level by position (usually first)
                            ts_index = pd.to_datetime(sim_data.index.get_level_values(0))
                            sim_data['date'] = ts_index.date if isinstance(ts_index, pd.DatetimeIndex) else ts_index.dt.date
                    else:
                        # Single index case - assume it's the timestamp
                        ts_index = pd.to_datetime(sim_data.index)
                        sim_data['date'] = ts_index.date if isinstance(ts_index, pd.DatetimeIndex) else ts_index.dt.date
                
                # Get the P&L change for each day
                # First, get the total portfolio P&L at each timestamp
                portfolio_pnl_by_ts = sim_data.groupby('ts')['pnl'].sum().reset_index()
                portfolio_pnl_by_ts['date'] = pd.to_datetime(portfolio_pnl_by_ts['ts']).dt.date
                
                # Now get first and last portfolio P&L for each day
                daily_pnl_by_date = portfolio_pnl_by_ts.groupby('date')['pnl'].agg(['first', 'last'])
                daily_pnl = daily_pnl_by_date['last'] - daily_pnl_by_date['first']
                
                logger.info(f"    Using sim_df path - calculated daily P&L")
                logger.info(f"    Sum of daily P&Ls: ${daily_pnl.sum():,.2f}")
                logger.info(f"    Number of days: {len(daily_pnl)}")
                
            elif hasattr(self, 'pnl_details_df') and self.pnl_details_df is not None:
                pnl_details = self.pnl_details_df.copy()
                pnl_details['date'] = pd.to_datetime(pnl_details['date'])
                daily_pnl_by_symbol = pnl_details.groupby(['date', 'symbol_venue'])['total_pnl_daily'].last()
                daily_pnl = daily_pnl_by_symbol.groupby('date').sum()
            else:
                # Fallback to using aggregated data
                daily_pnl = df.groupby('date')['total_pnl_daily'].sum()
            
            # Then aggregate other metrics from the timestamp-level data
            # Note: df['date'] already set above
            if hasattr(self, 'sim_df') and self.sim_df is not None:
                # For sim data, we need different aggregation logic
                daily_stats = df.groupby('date').agg({
                    'cumulative_pnl': 'last',  # End of day cumulative P&L
                    'position_count': 'mean',
                    'net_exposure': 'mean',
                    'dollars_traded_daily': 'sum',  # Sum for the day since it's per timestamp
                    'fees_usd_daily': 'sum',  # Sum for the day
                    'funding_income_daily': 'sum',  # Sum for the day
                }).reset_index()
            else:
                # For pnl_details data, values are already daily cumulative
                daily_stats = df.groupby('date').agg({
                    'cumulative_pnl': 'last',  # End of day cumulative P&L
                    'position_count': 'mean',
                    'net_exposure': 'mean',
                    'dollars_traded_daily': 'last',  # Last value of the day since it's already cumulative
                    'fees_usd_daily': 'last',  # Last value of the day since it's already cumulative
                    'funding_income_daily': 'last',  # Last value of the day since it's already cumulative
                }).reset_index()
            
            # Add the correctly calculated daily P&L
            # Convert daily_pnl index to date if needed
            if hasattr(daily_pnl.index[0], 'date'):
                daily_pnl.index = daily_pnl.index.date
            daily_stats['daily_pnl'] = daily_stats['date'].map(daily_pnl.to_dict()).fillna(0)
            
            # Reorder columns to match expected order
            daily_stats = daily_stats[['date', 'daily_pnl', 'cumulative_pnl', 'position_count', 
                                       'net_exposure', 'dollars_traded_daily', 'fees_usd_daily', 'funding_income_daily']]
            daily_stats.columns = ['date', 'daily_pnl', 'cumulative_pnl', 'positions', 
                                   'net_exposure', 'volume_traded', 'fees', 'funding']
            
            # Initialize returns variables for later use
            daily_returns = pd.Series()
            daily_stats_with_exposure = pd.DataFrame()
            
            # Log the daily P&L used for summary statistics
            logger.info(f"  Daily P&L for summary stats:")
            logger.info(f"    Using total_pnl from line 605: ${total_pnl:,.2f}")
            logger.info(f"    This should match summary.txt lifetime_pnl: $6,750,616.1")
            
            # Calculate Sharpe ratio properly
            # First calculate gross exposure (sum of absolute values of all positions) by timestamp
            # Add gross exposure to the aggregate dataframe
            if hasattr(self, 'sim_df') and self.sim_df is not None:
                gross_exposure_by_ts = self.sim_df.groupby('ts')['position'].apply(
                    lambda x: x.abs().sum()
                ).reset_index(name='gross_exposure')
            elif hasattr(self, 'pnl_details_df') and self.pnl_details_df is not None:
                gross_exposure_by_ts = self.pnl_details_df.groupby('ts')['notional'].apply(
                    lambda x: x.abs().sum()
                ).reset_index(name='gross_exposure')
            else:
                # If no detailed data, create empty dataframe
                gross_exposure_by_ts = pd.DataFrame(columns=['ts', 'gross_exposure'])
            
            # Merge with aggregate df to get gross exposure by timestamp
            df_with_exposure = df.merge(gross_exposure_by_ts, on='ts', how='left')
            
            # Get end-of-day gross exposure for each date
            df_with_exposure['date'] = pd.to_datetime(df_with_exposure['ts']).dt.date
            daily_gross_exposure = df_with_exposure.groupby('date').agg({
                'gross_exposure': 'last'  # End of day gross exposure
            }).reset_index()
            
            # Merge with daily stats
            daily_stats_with_exposure = daily_stats.merge(daily_gross_exposure, on='date', how='left')
            
            # Shift gross exposure to get previous day's exposure (t-1)
            daily_stats_with_exposure['prev_gross_exposure'] = daily_stats_with_exposure['gross_exposure'].shift(1)
            
            # Remove first day since we don't have t-1 exposure
            daily_stats_with_exposure = daily_stats_with_exposure.dropna(subset=['prev_gross_exposure'])
            
            # Filter out days with zero or very small exposure to avoid division issues
            min_exposure_threshold = 1000
            days_before_filter = len(daily_stats_with_exposure)
            daily_stats_with_exposure = daily_stats_with_exposure[daily_stats_with_exposure['prev_gross_exposure'] > min_exposure_threshold]
            days_after_filter = len(daily_stats_with_exposure)
            
            if days_before_filter > days_after_filter:
                logger.info(f"Filtered out {days_before_filter - days_after_filter} days with exposure < ${min_exposure_threshold}")
            
            # Calculate daily returns as daily P&L / t-1 gross exposure
            if len(daily_stats_with_exposure) > 0 and daily_stats_with_exposure['prev_gross_exposure'].mean() > 0:
                daily_returns = 1 + (daily_stats_with_exposure['daily_pnl'] / daily_stats_with_exposure['prev_gross_exposure'])

                mean_daily_return = (daily_returns.prod() - 1) / len(daily_returns)
                daily_std = daily_returns.std()
                
                # Use 365 trading days for annualization (crypto trades every day)
                trading_days_per_year = 365
                
                # Annualize the return and risk
                annualized_return = mean_daily_return * trading_days_per_year
                annualized_risk = daily_std * np.sqrt(trading_days_per_year)
                
                # Calculate Sharpe as annualized return / annualized risk
                annualized_sharpe = annualized_return / annualized_risk if annualized_risk > 0 else 0
                
                # Daily Sharpe for reference
                daily_sharpe = mean_daily_return / daily_std if daily_std > 0 else 0
                
                avg_gross_exposure = daily_stats_with_exposure['prev_gross_exposure'].mean()
                
                # Log the calculations for debugging
                logger.info(f"Sharpe Calculation Debug:")
                logger.info(f"  Number of days: {len(daily_returns)}")
                logger.info(f"  Mean daily return: {mean_daily_return:.6f} ({mean_daily_return*100:.4f}%)")
                logger.info(f"  Daily std dev: {daily_std:.6f} ({daily_std*100:.4f}%)")
                logger.info(f"  Annualized return: {annualized_return:.4f} ({annualized_return*100:.2f}%)")
                logger.info(f"  Annualized risk: {annualized_risk:.4f} ({annualized_risk*100:.2f}%)")
                logger.info(f"  Annualized Sharpe: {annualized_sharpe:.4f}")
                logger.info(f"  Daily Sharpe: {daily_sharpe:.4f}")
                logger.info(f"  Average gross exposure: ${avg_gross_exposure:,.0f}")
                logger.info(f"  Total P&L: ${daily_stats_with_exposure['daily_pnl'].sum():,.0f}")
                logger.info(f"  First few daily returns: {daily_returns.head(5).values}")
            else:
                daily_returns = pd.Series([0])
                mean_daily_return = 0
                daily_std = 0
                annualized_return = 0
                annualized_risk = 0
                annualized_sharpe = 0
                daily_sharpe = 0
                avg_gross_exposure = 0
                logger.warning("No valid data for Sharpe calculation")
            
            # Create summary statistics table
            summary_data = [
                # P&L Metrics
                ["Total P&L", f"${total_pnl:,.2f}", "Avg Daily Volume", f"${avg_daily_volume:,.0f}"],
                ["Total Fees", f"${total_fees:,.2f}", "Average Positions", f"{avg_positions:.1f}"],
                ["Total Funding", f"${total_funding:,.2f}", "Max Net Exposure", f"${max_exposure:,.0f}"],
                ["Net P&L", f"${total_pnl + total_fees + total_funding:,.2f}", "Avg Gross Exposure", f"${avg_gross_exposure:,.0f}"],
                # Performance Metrics
                ["Mean Daily Return", f"{mean_daily_return*100:.2f}%", "Annualized Return", f"{annualized_return*100:.2f}%"],
                ["Daily Std Dev", f"{daily_std*100:.2f}%", "Annualized Risk", f"{annualized_risk*100:.2f}%"],
                ["Avg Daily Turnover", f"{avg_daily_turnover:.2f}", "Annualized Sharpe", f"{annualized_sharpe:.2f}"],
            ]
            
            summary_text = html.Table([
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
            
            # P&L time series
            pnl_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    subplot_titles=('Cumulative P&L', 'Daily P&L'),
                                    vertical_spacing=0.1)
            
            # Cumulative P&L
            pnl_fig.add_trace(go.Scatter(
                x=df['ts'],
                y=df['cumulative_pnl'],
                mode='lines',
                name='Cumulative P&L',
                line={'width': 2}
            ), row=1, col=1)
            
            # Daily P&L as bars
            pnl_fig.add_trace(go.Bar(
                x=daily_stats['date'],
                y=daily_stats['daily_pnl'],
                name='Daily P&L',
                marker_color=daily_stats['daily_pnl'].apply(lambda x: 'green' if x > 0 else 'red')
            ), row=2, col=1)
            
            pnl_fig.update_xaxes(title_text="Date", row=2, col=1)
            pnl_fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=1)
            pnl_fig.update_yaxes(title_text="Daily P&L ($)", row=2, col=1)
            pnl_fig.update_layout(height=600, showlegend=False)
            
            # Portfolio metrics
            portfolio_fig = make_subplots(rows=2, cols=2,
                                          subplot_titles=('Net Exposure', 'Position Count', 
                                                         'Average Position Size', 'Turnover'),
                                          vertical_spacing=0.15,
                                          horizontal_spacing=0.1)
            
            # Net exposure
            portfolio_fig.add_trace(go.Scatter(
                x=df['ts'],
                y=df['net_exposure'],
                mode='lines',
                name='Net Exposure',
                line={'width': 1}
            ), row=1, col=1)
            
            # Position count
            portfolio_fig.add_trace(go.Scatter(
                x=df['ts'],
                y=df['position_count'],
                mode='lines',
                name='Position Count',
                line={'width': 1}
            ), row=1, col=2)
            
            # Average position size
            portfolio_fig.add_trace(go.Scatter(
                x=df['ts'],
                y=df['avg_position_size'],
                mode='lines',
                name='Avg Position Size',
                line={'width': 1}
            ), row=2, col=1)
            
            # Turnover - calculate and plot daily turnover
            # Group by date and calculate daily turnover
            df_daily = df.groupby('date').agg({
                'dollars_traded_daily': 'sum' if hasattr(self, 'sim_df') and self.sim_df is not None else 'last',
                'notional_abs_sum': 'mean'  # Average gross notional for the day
            })
            df_daily['turnover'] = df_daily['dollars_traded_daily'] / (df_daily['notional_abs_sum'] + 1e-10)
            
            # Plot as a bar chart for daily values
            portfolio_fig.add_trace(go.Bar(
                x=df_daily.index,
                y=df_daily['turnover'],
                name='Daily Turnover',
                marker_color='purple'
            ), row=2, col=2)
            
            portfolio_fig.update_yaxes(title_text="USD", row=1, col=1)
            portfolio_fig.update_yaxes(title_text="Count", row=1, col=2)
            portfolio_fig.update_yaxes(title_text="USD", row=2, col=1)
            portfolio_fig.update_yaxes(title_text="Ratio", row=2, col=2)
            portfolio_fig.update_layout(height=600, showlegend=False)
            
            # Trading activity and costs
            activity_fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                         subplot_titles=('Trading Volume', 'Cumulative Trading Fees', 'Funding Income'),
                                         vertical_spacing=0.1)
            
            # Trading volume (total absolute value)
            activity_fig.add_trace(go.Scatter(
                x=df['ts'],
                y=df['dollars_traded_daily'],
                mode='lines',
                name='Total Volume Traded',
                line={'width': 2, 'color': 'blue'},
                fill='tozeroy',
                fillcolor='rgba(0, 100, 255, 0.2)'
            ), row=1, col=1)
            
            # Fees
            activity_fig.add_trace(go.Scatter(
                x=df['ts'],
                y=-df['cumulative_fees'],  # Negative because fees are costs
                mode='lines',
                name='Cumulative Trading Fees',
                line={'width': 1, 'color': 'orange'},
                fill='tozeroy'
            ), row=2, col=1)
            
            # Funding
            activity_fig.add_trace(go.Scatter(
                x=df['ts'],
                y=df['funding_income_daily'],
                mode='lines',
                name='Funding Income',
                line={'width': 1},
                fill='tozeroy',
                fillcolor='rgba(0,100,80,0.2)' if df['funding_income_daily'].sum() > 0 else 'rgba(255,0,0,0.2)'
            ), row=3, col=1)
            
            activity_fig.update_xaxes(title_text="Date", row=3, col=1)
            activity_fig.update_yaxes(title_text="Volume ($)", row=1, col=1)
            activity_fig.update_yaxes(title_text="Fees ($)", row=2, col=1)
            activity_fig.update_yaxes(title_text="Funding ($)", row=3, col=1)
            activity_fig.update_layout(height=800, showlegend=True)
            
            # Hourly patterns
            df['hour'] = pd.to_datetime(df['ts']).dt.hour
            hourly_stats = df.groupby('hour').agg({
                'total_pnl_daily': 'mean',  # Average of the daily P&L increments for each hour
                'dollars_traded_daily': 'mean',
                'position_count': 'mean',
                'turnover': 'mean'
            }).reset_index()
            
            hourly_fig = make_subplots(rows=1, cols=2,
                                        subplot_titles=('Average P&L by Hour', 'Average Volume by Hour'),
                                        horizontal_spacing=0.1)
            
            # P&L by hour
            hourly_fig.add_trace(go.Bar(
                x=hourly_stats['hour'],
                y=hourly_stats['total_pnl_daily'],
                name='Avg P&L',
                marker_color=hourly_stats['total_pnl_daily'].apply(lambda x: 'green' if x > 0 else 'red')
            ), row=1, col=1)
            
            # Volume by hour
            hourly_fig.add_trace(go.Bar(
                x=hourly_stats['hour'],
                y=hourly_stats['dollars_traded_daily'],
                name='Avg Volume',
                marker_color='blue'
            ), row=1, col=2)
            
            
            hourly_fig.update_xaxes(title_text="Hour (UTC)", row=1, col=1)
            hourly_fig.update_xaxes(title_text="Hour (UTC)", row=1, col=2)
            hourly_fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
            hourly_fig.update_yaxes(title_text="Volume ($)", row=1, col=2)
            hourly_fig.update_layout(height=400, showlegend=False)
            
            # Daily returns distribution
            returns_dist_fig = make_subplots(rows=2, cols=2,
                                             subplot_titles=('Daily Returns Histogram', 'Daily Returns Time Series',
                                                            'Q-Q Plot', ''),
                                             vertical_spacing=0.15,
                                             horizontal_spacing=0.15)
            
            # Calculate daily returns if we have the data
            if 'daily_returns' in locals() and len(daily_returns) > 0:
                # Histogram of daily returns
                returns_dist_fig.add_trace(go.Histogram(
                    x=daily_returns * 100,  # Convert to percentage
                    nbinsx=50,
                    name='Daily Returns',
                    marker_color='blue',
                    opacity=0.7
                ), row=1, col=1)
                
                # Add normal distribution overlay
                returns_mean = daily_returns.mean()
                returns_std = daily_returns.std()
                x_range = np.linspace(daily_returns.min() * 100, daily_returns.max() * 100, 100)
                normal_y = (1 / (returns_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range/100 - returns_mean) / returns_std) ** 2)
                # Scale the normal distribution to match histogram
                hist_values, hist_bins = np.histogram(daily_returns * 100, bins=50)
                bin_width = hist_bins[1] - hist_bins[0]
                normal_y_scaled = normal_y * len(daily_returns) * bin_width / 100
                
                returns_dist_fig.add_trace(go.Scatter(
                    x=x_range,
                    y=normal_y_scaled,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='red', width=2)
                ), row=1, col=1)
                
                # Time series of daily returns
                returns_dates = daily_stats_with_exposure['date'][:len(daily_returns)]
                returns_dist_fig.add_trace(go.Scatter(
                    x=returns_dates,
                    y=daily_returns * 100,  # Convert to percentage
                    mode='lines+markers',
                    name='Daily Returns',
                    line=dict(width=1),
                    marker=dict(size=4)
                ), row=1, col=2)
                
                # Add mean and +/- 2 std bands
                returns_dist_fig.add_hline(y=returns_mean * 100, line_dash="dash", line_color="green", 
                                          annotation_text=f"Mean: {returns_mean*100:.2f}%", row=1, col=2)
                returns_dist_fig.add_hline(y=(returns_mean + 2*returns_std) * 100, line_dash="dot", line_color="red", 
                                          annotation_text=f"+2σ: {(returns_mean + 2*returns_std)*100:.2f}%", row=1, col=2)
                returns_dist_fig.add_hline(y=(returns_mean - 2*returns_std) * 100, line_dash="dot", line_color="red", 
                                          annotation_text=f"-2σ: {(returns_mean - 2*returns_std)*100:.2f}%", row=1, col=2)
                
                # Q-Q plot to check normality
                sorted_returns = np.sort(daily_returns * 100)
                theoretical_quantiles = np.percentile(np.random.normal(returns_mean * 100, returns_std * 100, 10000), 
                                                     np.linspace(0, 100, len(sorted_returns)))
                
                returns_dist_fig.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_returns,
                    mode='markers',
                    name='Q-Q Plot',
                    marker=dict(size=5)
                ), row=2, col=1)
                
                # Add diagonal reference line
                min_val = min(theoretical_quantiles.min(), sorted_returns.min())
                max_val = max(theoretical_quantiles.max(), sorted_returns.max())
                returns_dist_fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Normal Line',
                    line=dict(color='red', dash='dash')
                ), row=2, col=1)
                
                
            else:
                # If no returns data, show empty plot with message
                returns_dist_fig.add_annotation(
                    text="No daily returns data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
            
            returns_dist_fig.update_xaxes(title_text="Daily Return (%)", row=1, col=1)
            returns_dist_fig.update_yaxes(title_text="Frequency", row=1, col=1)
            returns_dist_fig.update_xaxes(title_text="Date", row=1, col=2)
            returns_dist_fig.update_yaxes(title_text="Daily Return (%)", row=1, col=2)
            returns_dist_fig.update_xaxes(title_text="Theoretical Quantiles (%)", row=2, col=1)
            returns_dist_fig.update_yaxes(title_text="Sample Quantiles (%)", row=2, col=1)
            returns_dist_fig.update_layout(height=800, showlegend=False)
            
            # Calculate monthly statistics
            # Add month column to daily stats
            daily_stats['month'] = pd.to_datetime(daily_stats['date']).dt.to_period('M')
            
            # Calculate monthly Sharpe ratio
            # First, merge daily stats with exposure data to get daily returns for each month
            monthly_sharpe_list = []
            if len(daily_stats_with_exposure) > 0:
                daily_stats_with_exposure['month'] = pd.to_datetime(daily_stats_with_exposure['date']).dt.to_period('M')
                
                for month in daily_stats_with_exposure['month'].unique():
                    month_data = daily_stats_with_exposure[daily_stats_with_exposure['month'] == month]
                    if len(month_data) > 1 and month_data['prev_gross_exposure'].sum() > 0:
                        # Calculate daily returns for this month
                        month_returns = month_data['daily_pnl'] / month_data['prev_gross_exposure']
                        # Remove extreme outliers
                        month_returns_clean = month_returns[(month_returns > -0.5) & (month_returns < 0.5)]
                        if len(month_returns_clean) > 1:
                            # Calculate annualized Sharpe for the month
                            mean_return = month_returns_clean.mean()
                            std_return = month_returns_clean.std()
                            if std_return > 0:
                                # Annualize using sqrt(365) for daily data
                                annualized_sharpe = (mean_return / std_return) * np.sqrt(365)
                            else:
                                annualized_sharpe = 0
                        else:
                            annualized_sharpe = 0
                    else:
                        annualized_sharpe = 0
                    monthly_sharpe_list.append({'month': month, 'sharpe': annualized_sharpe})
                
                monthly_sharpe_df = pd.DataFrame(monthly_sharpe_list)
            else:
                # No exposure data, create empty sharpe dataframe
                monthly_sharpe_df = pd.DataFrame(columns=['month', 'sharpe'])
            
            # Group by month and aggregate
            monthly_stats = daily_stats.groupby('month').agg({
                'daily_pnl': 'sum',  # Sum of daily P&Ls for the month
                'positions': 'mean',  # Average positions during the month
                'net_exposure': 'mean',  # Average net exposure during the month
                'volume_traded': 'sum',  # Total volume traded in the month
                'fees': 'sum',  # Total fees for the month
                'funding': 'sum',  # Total funding for the month
            }).reset_index()
            
            # Merge with Sharpe ratios
            if len(monthly_sharpe_df) > 0:
                monthly_stats = monthly_stats.merge(monthly_sharpe_df, on='month', how='left')
                monthly_stats['sharpe'] = monthly_stats['sharpe'].fillna(0)
            else:
                monthly_stats['sharpe'] = 0
            
            # Sort by month ascending to calculate cumulative P&L correctly
            monthly_stats = monthly_stats.sort_values('month', ascending=True)
            
            # Calculate cumulative P&L as cumsum of monthly P&Ls
            monthly_stats['cumulative_pnl'] = monthly_stats['daily_pnl'].cumsum()
            
            # Convert month period to string for display
            monthly_stats['month'] = monthly_stats['month'].astype(str)
            
            # Rename columns for the table
            monthly_stats.columns = ['month', 'monthly_pnl', 'avg_positions', 
                                     'avg_net_exposure', 'volume_traded', 'fees', 'funding', 'sharpe', 'cumulative_pnl']
            
            # Reorder columns to match table definition
            monthly_stats = monthly_stats[['month', 'monthly_pnl', 'cumulative_pnl', 'avg_positions', 
                                          'avg_net_exposure', 'volume_traded', 'fees', 'funding', 'sharpe']]
            
            # Sort by month descending (most recent first) for display
            monthly_stats = monthly_stats.sort_values('month', ascending=False)
            
            # Monthly stats table
            table_data = monthly_stats.to_dict('records')
            
            # Create daily PnL returns table data
            daily_pnl_returns_data = []
            if len(daily_stats_with_exposure) > 0:
                # Create a dataframe with the required columns
                pnl_returns_df = daily_stats_with_exposure.copy()
                pnl_returns_df['daily_return_pct'] = (daily_returns * 100).values[:len(pnl_returns_df)]
                
                # Calculate cumulative returns (compound returns)
                pnl_returns_df['cumulative_return_pct'] = ((1 + daily_returns).cumprod() - 1) * 100
                
                # Select and rename columns for the table
                pnl_returns_df = pnl_returns_df[['date', 'daily_pnl', 'gross_exposure', 'daily_return_pct', 'cumulative_return_pct']]
                pnl_returns_df.columns = ['date', 'daily_pnl', 'gross_notional', 'daily_return_pct', 'cumulative_return_pct']
                
                # Sort by date descending and convert to records
                daily_pnl_returns_data = pnl_returns_df.sort_values('date', ascending=False).to_dict('records')
            
            # Calculate single-security risk analysis and cumulative P&L by security
            single_security_fig = go.Figure()
            security_pnl_bar_fig = go.Figure()
            top_security_pnl_data = []
            
            if hasattr(self, 'sim_df') and self.sim_df is not None:
                # Calculate daily P&L by security
                sim_data = self.sim_df.copy()
                
                # Ensure we have date column
                if 'date' not in sim_data.columns:
                    # Extract date from timestamp
                    if 'ts' in sim_data.columns:
                        sim_data['date'] = pd.to_datetime(sim_data['ts']).dt.date
                    elif isinstance(sim_data.index, pd.MultiIndex):
                        # MultiIndex case - get the ts level
                        if 'ts' in sim_data.index.names:
                            ts_index = pd.to_datetime(sim_data.index.get_level_values('ts'))
                            sim_data['date'] = ts_index.date if isinstance(ts_index, pd.DatetimeIndex) else ts_index.dt.date
                        else:
                            # Try to find the timestamp level by position (usually first)
                            ts_index = pd.to_datetime(sim_data.index.get_level_values(0))
                            sim_data['date'] = ts_index.date if isinstance(ts_index, pd.DatetimeIndex) else ts_index.dt.date
                    else:
                        # Single index case - assume it's the timestamp
                        ts_index = pd.to_datetime(sim_data.index)
                        sim_data['date'] = ts_index.date if isinstance(ts_index, pd.DatetimeIndex) else ts_index.dt.date
                else:
                    # Ensure date column is datetime.date objects
                    sim_data['date'] = pd.to_datetime(sim_data['date']).dt.date
                
                # Get symbol from index if it's there
                if 'symbol_venue' in sim_data.index.names:
                    sim_data = sim_data.reset_index()
                elif 'symbol' in sim_data.columns:
                    sim_data['symbol_venue'] = sim_data['symbol']
                
                # Calculate daily P&L for each symbol
                # Group by date and symbol, get first and last P&L
                daily_symbol_pnl = sim_data.groupby(['date', 'symbol_venue'])['pnl'].agg(['first', 'last'])
                daily_symbol_pnl['daily_pnl'] = daily_symbol_pnl['last'] - daily_symbol_pnl['first']
                daily_symbol_pnl = daily_symbol_pnl.reset_index()
                
                # Create time series plot showing min and max daily security P&L
                daily_extremes = daily_symbol_pnl.groupby('date')['daily_pnl'].agg(['min', 'max', 'mean', 'std'])
                
                single_security_fig = go.Figure()
                
                # Add shaded area for range
                single_security_fig.add_trace(go.Scatter(
                    x=daily_extremes.index,
                    y=daily_extremes['max'],
                    mode='lines',
                    name='Max Daily Security P&L',
                    line=dict(color='lightgreen', width=1),
                    showlegend=True
                ))
                
                single_security_fig.add_trace(go.Scatter(
                    x=daily_extremes.index,
                    y=daily_extremes['min'],
                    mode='lines',
                    name='Min Daily Security P&L',
                    line=dict(color='lightcoral', width=1),
                    fill='tonexty',
                    fillcolor='rgba(200, 200, 200, 0.3)',
                    showlegend=True
                ))
                
                # Add mean line
                single_security_fig.add_trace(go.Scatter(
                    x=daily_extremes.index,
                    y=daily_extremes['mean'],
                    mode='lines',
                    name='Mean Daily Security P&L',
                    line=dict(color='blue', width=2)
                ))
                
                # Add zero line
                single_security_fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                
                single_security_fig.update_layout(
                    title="Daily Single-Security P&L Range",
                    xaxis_title="Date",
                    yaxis_title="P&L ($)",
                    height=400,
                    showlegend=True
                )
                
                # Get top 10 largest absolute P&L moves
                daily_symbol_pnl['abs_pnl'] = daily_symbol_pnl['daily_pnl'].abs()
                top_moves = daily_symbol_pnl.nlargest(10, 'abs_pnl')[['symbol_venue', 'date', 'daily_pnl']]
                top_moves.columns = ['symbol', 'date', 'pnl']
                top_security_pnl_data = top_moves.to_dict('records')
                
                # Create cumulative P&L by security bar chart
                # Get the last (most recent) P&L value for each symbol
                final_pnl_by_symbol = sim_data.groupby('symbol_venue')['pnl'].last().sort_values()
                
                # Filter out symbols with negligible P&L
                min_pnl_threshold = 100  # $100 minimum to show
                significant_symbols = final_pnl_by_symbol[final_pnl_by_symbol.abs() >= min_pnl_threshold]
                
                # If we have too many symbols, show top and bottom performers
                max_symbols_to_show = 50
                if len(significant_symbols) > max_symbols_to_show:
                    # Get top gainers and losers
                    n_each_side = max_symbols_to_show // 2
                    top_gainers = significant_symbols.nlargest(n_each_side)
                    top_losers = significant_symbols.nsmallest(n_each_side)
                    symbols_to_show = pd.concat([top_losers, top_gainers]).sort_values()
                else:
                    symbols_to_show = significant_symbols
                
                # Create bar chart
                security_pnl_bar_fig = go.Figure()
                
                # Color bars based on positive/negative P&L
                colors = ['green' if x > 0 else 'red' for x in symbols_to_show.values]
                
                security_pnl_bar_fig.add_trace(go.Bar(
                    x=symbols_to_show.index,
                    y=symbols_to_show.values,
                    marker_color=colors,
                    text=[f'${x:,.0f}' for x in symbols_to_show.values],
                    textposition='outside',
                    name='Cumulative P&L'
                ))
                
                security_pnl_bar_fig.update_layout(
                    title=f"Cumulative P&L by Security (Showing {len(symbols_to_show)} of {len(final_pnl_by_symbol)} symbols)",
                    xaxis_title="Symbol",
                    yaxis_title="Cumulative P&L ($)",
                    height=600,
                    showlegend=False,
                    xaxis={'tickangle': -45}
                )
                
                # Add a horizontal line at y=0
                security_pnl_bar_fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                
            else:
                # No sim_df data, create empty figure with message
                security_pnl_bar_fig.add_annotation(
                    text="No per-security P&L data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
                
            # Calculate drawdown analysis
            drawdown_fig = go.Figure()
            drawdown_data = []
            
            # Calculate drawdowns from cumulative P&L
            cumulative_pnl = df['cumulative_pnl'].values
            timestamps = pd.to_datetime(df['ts'])
            
            # Calculate running maximum
            running_max = pd.Series(cumulative_pnl).expanding().max()
            
            # Calculate drawdown as percentage from peak
            drawdown_pct = (cumulative_pnl - running_max) / (running_max + 1e-10) * 100
            drawdown_dollars = cumulative_pnl - running_max
            
            # Create drawdown chart
            drawdown_fig.add_trace(go.Scatter(
                x=timestamps,
                y=drawdown_pct,
                mode='lines',
                name='Drawdown %',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)'
            ))
            
            drawdown_fig.update_layout(
                title="Portfolio Drawdown Over Time",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=400,
                showlegend=False
            )
            
            # Find individual drawdown periods
            drawdowns = []
            in_drawdown = False
            
            for i in range(1, len(cumulative_pnl)):
                if not in_drawdown and drawdown_pct[i] < 0:
                    # Start of drawdown
                    in_drawdown = True
                    start_idx = i - 1  # Peak is the previous point
                    peak_value = cumulative_pnl[start_idx]
                    start_date = timestamps[start_idx]
                    
                elif in_drawdown and (drawdown_pct[i] >= 0 or i == len(cumulative_pnl) - 1):
                    # End of drawdown or end of data
                    if i == len(cumulative_pnl) - 1 and drawdown_pct[i] < 0:
                        # Still in drawdown at end
                        trough_idx = i
                    else:
                        # Recovered from drawdown
                        trough_idx = i - 1
                    
                    # Find the actual trough (minimum point) in this drawdown period
                    trough_search_start = start_idx
                    trough_search_end = trough_idx + 1
                    actual_trough_idx = trough_search_start + np.argmin(cumulative_pnl[trough_search_start:trough_search_end])
                    
                    trough_value = cumulative_pnl[actual_trough_idx]
                    end_date = timestamps[actual_trough_idx]
                    
                    dollar_loss = peak_value - trough_value
                    percent_loss = (dollar_loss / peak_value) * 100 if peak_value > 0 else 0
                    
                    if dollar_loss > 0:  # Only record actual losses
                        drawdowns.append({
                            'start_date': start_date,
                            'end_date': end_date,
                            'duration_days': (end_date - start_date).days,
                            'peak_value': peak_value,
                            'trough_value': trough_value,
                            'dollar_loss': dollar_loss,
                            'percent_loss': percent_loss
                        })
                    
                    in_drawdown = False
            
            # Sort by dollar loss and get top 5
            if drawdowns:
                drawdowns_df = pd.DataFrame(drawdowns)
                top_drawdowns = drawdowns_df.nlargest(5, 'dollar_loss')
                drawdown_data = top_drawdowns.to_dict('records')
            
            return summary_text, pnl_fig, portfolio_fig, activity_fig, hourly_fig, returns_dist_fig, table_data, daily_pnl_returns_data, security_pnl_bar_fig, single_security_fig, top_security_pnl_data, drawdown_fig, drawdown_data
    
    def run(self):
        """Run the Dash app"""
        logger.info(f"Starting Simulation Report for {self.sim_name} on port {self.port}")
        self.app.run(debug=False, port=self.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation report showing P&L and portfolio metrics over time')
    parser.add_argument('sim_name', nargs='?', help='Name of the simulation directory (optional)')
    parser.add_argument('-p', '--port', help='port', type=int, default=None)
    args = parser.parse_args()
    
    port = args.port if args.port else REPORT_PORT
    
    report = SimulationReport(sim_name=args.sim_name, port=port)
    report.run()