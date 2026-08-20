import os
import logging.config
import argparse
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Output, Input, State, callback_context
import dash
from dash.dash_table import DataTable
from dash.dash_table.Format import Format, Scheme

from lib.util.config import get_config
from lib.util.directory import CONFIG_DIR, ROOT_DIR
from lib.util.util import LOCAL
from lib.util.logging_util import get_logging_config
from lib.data import DataLoader
from lib.features import get_available_features_for_horizons

# Import the base SimulationReport class
from simulation_report import SimulationReport, SIM_DIR, REPORT_PORT

logging.config.dictConfig(get_logging_config("simulation_report_with_factors"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SimulationReportWithFactors(SimulationReport):
    """Extended simulation report that includes factor analysis"""
    
    def __init__(self, sim_name: Optional[str] = None, port: int = REPORT_PORT, 
                 factor_name: Optional[str] = None, horizon: int = 1440):
        """
        Initialize the report with factor analysis capabilities
        
        Args:
            sim_name: Name of the simulation directory
            port: Port to run the dash app on
            factor_name: Name of the feature to use as a factor (e.g., 'beta_1440', 'logret_1440_trmean')
            horizon: Horizon for loading features (default 1440)
        """
        # Factor analysis parameters - must be set before calling parent __init__
        self.factor_name = factor_name
        self.horizon = horizon
        self.factor_data = None
        self.factor_exposures = None
        self.factor_pnl = None
        
        # Initialize DataLoader
        self.data_loader = DataLoader()
        
        # Initialize parent class (this will call setup_layout)
        super().__init__(sim_name, port)
        
        # Load factor data if simulation is loaded
        if self.sim_name and self.factor_name:
            self.load_factor_data()
    
    def load_factor_data(self):
        """Load feature data to use as factors"""
        try:
            if not hasattr(self, 'sim_df') or self.sim_df is None or self.sim_df.empty:
                logger.warning("No simulation data loaded, cannot load factor data")
                return
            
            logger.info(f"Loading factor data for feature: {self.factor_name}")
            
            # Get date range and symbols from simulation data
            # The parent class resets index, so we should have columns
            ts_values = pd.to_datetime(self.sim_df['ts'])
            symbols = self.sim_df['symbol_venue'].unique().tolist()
            
            start_date = ts_values.min().date()
            end_date = ts_values.max().date()
            
            # Load features for the horizon and date range
            logger.info(f"Loading features for horizon {self.horizon} from {start_date} to {end_date}")
            logger.info(f"Loading factor '{self.factor_name}' for {len(symbols)} symbols - this may take a few minutes...")
            
            # Only load the specific factor we need, plus any required columns
            cols_to_load = [self.factor_name]
            feature_df = self.data_loader.load_features(
                horizons=[self.horizon],  # Pass as list
                start_date=start_date,
                end_date=end_date,
                symbol_venues=symbols,
                cols=cols_to_load  # Only load the factor we need
            )
            
            if feature_df is None or feature_df.empty:
                logger.warning("No feature data loaded")
                return
            
            # Check if the requested factor exists
            if self.factor_name not in feature_df.columns:
                available_factors = [col for col in feature_df.columns if not col.startswith('_')]
                logger.error(f"Factor '{self.factor_name}' not found in features. Available: {available_factors[:10]}...")
                return
            
            # Extract the factor data - keep it with MultiIndex
            self.factor_data = feature_df[[self.factor_name]].copy()
            logger.info(f"Loaded factor data with {len(self.factor_data)} records")
            logger.info(f"Factor data columns: {list(self.factor_data.columns)}")
            logger.info(f"Factor data index names: {self.factor_data.index.names}")
            
            # Important: Do NOT reset index here - keep the MultiIndex for merging
            
            # Calculate factor exposures
            self.calculate_factor_exposures()
            
        except Exception as e:
            logger.error(f"Error loading factor data: {e}")
            import traceback
            traceback.print_exc()
    
    def calculate_factor_exposures(self):
        """Calculate factor exposures and P&L attribution"""
        try:
            if self.factor_data is None or self.sim_df is None:
                logger.warning("Missing data for factor exposure calculation")
                return
            
            logger.info("Calculating factor exposures...")
            
            # Merge on MultiIndex directly
            logger.info(f"Merging sim_df (shape: {self.sim_df.shape}) with factor_data (shape: {self.factor_data.shape})")
            logger.info(f"sim_df index type: {type(self.sim_df.index)}, names: {self.sim_df.index.names}")
            logger.info(f"factor_data index type: {type(self.factor_data.index)}, names: {self.factor_data.index.names}")
            
            # Reset both indices and merge on columns
            sim_data = self.sim_df.reset_index()
            factor_data = self.factor_data.reset_index()
            
            # Ensure timestamps have the same timezone for merging
            # Convert both to UTC if needed
            if hasattr(sim_data['ts'].dtype, 'tz'):
                if sim_data['ts'].dt.tz is None:
                    sim_data['ts'] = pd.to_datetime(sim_data['ts']).dt.tz_localize('UTC')
            else:
                sim_data['ts'] = pd.to_datetime(sim_data['ts']).dt.tz_localize('UTC')
                
            if hasattr(factor_data['ts'].dtype, 'tz'):
                if factor_data['ts'].dt.tz is None:
                    factor_data['ts'] = pd.to_datetime(factor_data['ts']).dt.tz_localize('UTC')
            else:
                factor_data['ts'] = pd.to_datetime(factor_data['ts']).dt.tz_localize('UTC')
            
            # Merge on ts and symbol_venue columns
            merged_df = sim_data.merge(
                factor_data,
                on=['ts', 'symbol_venue'],
                how='left',
                suffixes=('_sim', '_factor')
            )
            
            logger.info(f"Merged data shape: {merged_df.shape}")
            
            # Handle column naming after merge
            # If the factor exists in both dataframes, use the one from features (_factor suffix)
            factor_col_name = self.factor_name
            if f"{self.factor_name}_factor" in merged_df.columns:
                factor_col_name = f"{self.factor_name}_factor"
            elif self.factor_name not in merged_df.columns and f"{self.factor_name}_sim" in merged_df.columns:
                # Factor only exists in sim data, not in features
                logger.warning(f"Factor '{self.factor_name}' not found in feature data, using simulation data")
                factor_col_name = f"{self.factor_name}_sim"
            
            logger.info(f"Using factor column: '{factor_col_name}'")
            
            # Calculate factor exposure: factor loading * position value
            merged_df['factor_exposure'] = merged_df[factor_col_name] * merged_df['position']
            
            # Handle missing factor values
            missing_factor = merged_df[factor_col_name].isna().sum()
            if missing_factor > 0:
                logger.warning(f"Missing factor values for {missing_factor} records ({100*missing_factor/len(merged_df):.1f}%)")
                merged_df['factor_exposure'] = merged_df['factor_exposure'].fillna(0)
            
            # Calculate factor-weighted positions
            # For factor P&L, we need to track how exposure changes lead to P&L
            merged_df = merged_df.sort_values(['symbol_venue', 'ts'])
            
            # Calculate P&L contribution from factor exposure
            # This is simplified - ideally we'd decompose returns into factor and residual components
            merged_df['prev_position'] = merged_df.groupby('symbol_venue')['position'].shift(1)
            merged_df['position_change'] = merged_df['position'] - merged_df['prev_position'].fillna(0)
            
            # Calculate returns (price changes)
            # Use close_mid as the price field
            merged_df['prev_close'] = merged_df.groupby('symbol_venue')['close_mid'].shift(1)
            merged_df['price_return'] = (merged_df['close_mid'] - merged_df['prev_close']) / merged_df['prev_close']
            merged_df['price_return'] = merged_df['price_return'].fillna(0)
            
            # Factor P&L = factor exposure * price return
            # This assumes the factor exposure drives returns
            merged_df['factor_pnl'] = merged_df['factor_exposure'].shift(1) * merged_df['price_return']
            merged_df['factor_pnl'] = merged_df['factor_pnl'].fillna(0)
            
            # Store the results
            self.factor_exposures = merged_df[['ts', 'symbol_venue', factor_col_name, 
                                               'position', 'factor_exposure', 'factor_pnl']].copy()
            # Rename the factor column to the standard name for consistency
            self.factor_exposures.rename(columns={factor_col_name: self.factor_name}, inplace=True)
            
            # Aggregate by timestamp
            self.factor_pnl = self.factor_exposures.groupby('ts').agg({
                'factor_exposure': 'sum',  # Total factor exposure
                'factor_pnl': 'sum',       # Total factor P&L
                'position': lambda x: x.abs().sum()  # Gross position for scaling
            }).reset_index()
            
            # Calculate cumulative factor P&L
            self.factor_pnl['cumulative_factor_pnl'] = self.factor_pnl['factor_pnl'].cumsum()
            
            # Calculate factor exposure as % of gross position
            self.factor_pnl['factor_exposure_pct'] = (
                self.factor_pnl['factor_exposure'].abs() / 
                (self.factor_pnl['position'] + 1e-10) * 100
            )
            
            logger.info(f"Calculated factor exposures for {len(self.factor_pnl)} timestamps")
            
        except Exception as e:
            logger.error(f"Error calculating factor exposures: {e}")
            import traceback
            traceback.print_exc()
    
    def get_available_factors(self) -> List[str]:
        """Get list of available factors from feature files"""
        try:
            return get_available_features_for_horizons([self.horizon])
        except Exception as e:
            logger.error(f"Error getting available factors: {e}")
            return []
    
    def setup_layout(self):
        """Setup the Dash app layout with factor analysis components"""
        # Get base layout from parent
        super().setup_layout()
        
        # Get available factors
        available_factors = self.get_available_factors()
        factor_options = [{'label': f, 'value': f} for f in available_factors]
        
        # Add factor analysis section to the layout
        factor_section = html.Div([
            html.H3("Factor Analysis", style={'textAlign': 'center', 'marginTop': '30px'}),
            
            # Loading indicator for factor analysis
            dcc.Loading(
                id="loading-factor-analysis",
                type="default",
                children=[
            
            # Factor selector
            html.Div([
                html.Label("Select Factor:", style={'marginRight': '10px', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='factor-dropdown',
                    options=factor_options,
                    value=self.factor_name,
                    style={'width': '400px', 'display': 'inline-block', 'verticalAlign': 'middle'},
                    placeholder="Select a factor..."
                ),
                html.Label("Horizon:", style={'marginLeft': '20px', 'marginRight': '10px', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='horizon-dropdown',
                    options=[
                        {'label': '15 min', 'value': 15},
                        {'label': '60 min', 'value': 60},
                        {'label': '120 min', 'value': 120},
                        {'label': '360 min', 'value': 360},
                        {'label': '720 min', 'value': 720},
                        {'label': '1440 min (1 day)', 'value': 1440},
                        {'label': '4320 min (3 days)', 'value': 4320},
                        {'label': '10080 min (1 week)', 'value': 10080},
                    ],
                    value=self.horizon,
                    style={'width': '200px', 'display': 'inline-block', 'verticalAlign': 'middle'},
                    clearable=False
                ),
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            # Factor load status
            html.Div(id="factor-load-status", style={"marginTop": "10px", "marginBottom": "20px", "textAlign": "center"}),
            
            # Factor exposure over time
            html.Div([
                html.H4("Factor Exposure Over Time", style={'textAlign': 'center'}),
                dcc.Graph(id='factor-exposure-timeseries'),
            ], style={'marginBottom': '30px'}),
            
            # Factor P&L analysis
            html.Div([
                html.H4("Factor P&L Analysis", style={'textAlign': 'center'}),
                dcc.Graph(id='factor-pnl-analysis'),
            ], style={'marginBottom': '30px'}),
            
            # Factor statistics
            html.Div([
                html.H4("Factor Statistics", style={'textAlign': 'center'}),
                html.Div(id='factor-statistics', style={'marginBottom': '30px'}),
            ]),
            
            # Top factor exposures table
            html.Div([
                html.H4("Current Top Factor Exposures", style={'textAlign': 'center'}),
                DataTable(
                    id='top-factor-exposures-table',
                    columns=[
                        {'name': 'Symbol', 'id': 'symbol', 'type': 'text'},
                        {'name': 'Factor Value', 'id': 'factor_value', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=4)},
                        {'name': 'Position ($)', 'id': 'position', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=0, group=True)},
                        {'name': 'Factor Exposure ($)', 'id': 'factor_exposure', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=0, group=True)},
                        {'name': 'Factor P&L ($)', 'id': 'factor_pnl', 'type': 'numeric', 'format': Format(scheme=Scheme.fixed, precision=2, group=True)},
                    ],
                    sort_action='native',
                    page_size=20,
                    style_cell={'textAlign': 'center'},
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'factor_pnl', 'filter_query': '{factor_pnl} > 0'},
                            'color': 'green',
                        },
                        {
                            'if': {'column_id': 'factor_pnl', 'filter_query': '{factor_pnl} < 0'},
                            'color': 'red',
                        },
                    ]
                ),
            ]),
                ]  # End of Loading children
            ),  # End of Loading component
        ], style={
            'backgroundColor': '#f5f5f5',
            'padding': '20px',
            'borderRadius': '10px',
            'marginTop': '30px',
            'marginBottom': '30px'
        })
        
        # Insert the factor section at the end of the page
        # Find the main div and its children
        main_div = self.app.layout
        children = list(main_div.children)
        
        # Add factor section at the end (append to children)
        children.append(factor_section)
        main_div.children = children
    
    def load_simulation_data(self):
        """Override to also clear factor data when loading new simulation"""
        # Clear factor data
        self.factor_data = None
        self.factor_exposures = None
        self.factor_pnl = None
        
        # Call parent's load_simulation_data
        super().load_simulation_data()
        
        # Reload factor data if a factor is selected and we have sim data
        if self.factor_name and self.sim_name and self.sim_df is not None:
            self.load_factor_data()
    
    def setup_callbacks(self):
        """Setup Dash callbacks including factor analysis callbacks"""
        # Call parent callbacks - this will set up all the base callbacks
        super().setup_callbacks()
        
        # Add factor analysis callbacks
        @self.app.callback(
            Output('factor-load-status', 'children'),
            Output('factor-exposure-timeseries', 'figure'),
            Output('factor-pnl-analysis', 'figure'),
            Output('factor-statistics', 'children'),
            Output('top-factor-exposures-table', 'data'),
            Input('factor-dropdown', 'value'),
            Input('horizon-dropdown', 'value'),
            prevent_initial_call=False
        )
        def update_factor_analysis(selected_factor, selected_horizon):
            """Update factor analysis when factor selection changes"""
            
            # Check if factor or horizon changed
            if selected_factor != self.factor_name or selected_horizon != self.horizon:
                self.factor_name = selected_factor
                self.horizon = selected_horizon
                if selected_factor:
                    self.load_factor_data()
            
            # Default empty returns
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="Select a factor to view analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            
            if not self.factor_name:
                return "No factor selected", empty_fig, empty_fig, "", []
            
            if self.factor_pnl is None:
                loading_msg = f"⏳ Loading factor '{self.factor_name}' data... This may take 1-2 minutes for a full year of data."
                return loading_msg, empty_fig, empty_fig, "", []
            
            # Status message
            status_msg = f"✓ Factor '{self.factor_name}' loaded successfully (horizon: {self.horizon} minutes)"
            
            # Factor exposure time series
            exposure_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                         subplot_titles=('Factor Exposure ($)', 'Factor Exposure (% of Gross)'),
                                         vertical_spacing=0.1)
            
            # Absolute exposure
            exposure_fig.add_trace(go.Scatter(
                x=self.factor_pnl['ts'],
                y=self.factor_pnl['factor_exposure'],
                mode='lines',
                name='Factor Exposure',
                line={'width': 2, 'color': 'blue'}
            ), row=1, col=1)
            
            # Add zero line
            exposure_fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
            
            # Exposure as percentage
            exposure_fig.add_trace(go.Scatter(
                x=self.factor_pnl['ts'],
                y=self.factor_pnl['factor_exposure_pct'],
                mode='lines',
                name='Factor Exposure %',
                line={'width': 2, 'color': 'green'}
            ), row=2, col=1)
            
            exposure_fig.update_xaxes(title_text="Date", row=2, col=1)
            exposure_fig.update_yaxes(title_text="Exposure ($)", row=1, col=1)
            exposure_fig.update_yaxes(title_text="Exposure (%)", row=2, col=1)
            exposure_fig.update_layout(height=600, showlegend=False)
            
            # Factor P&L analysis
            pnl_fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                    subplot_titles=('Cumulative Factor P&L', 'Daily Factor P&L', 'Factor P&L vs Total P&L'),
                                    vertical_spacing=0.1)
            
            # Cumulative factor P&L
            pnl_fig.add_trace(go.Scatter(
                x=self.factor_pnl['ts'],
                y=self.factor_pnl['cumulative_factor_pnl'],
                mode='lines',
                name='Cumulative Factor P&L',
                line={'width': 2, 'color': 'purple'}
            ), row=1, col=1)
            
            # Daily factor P&L
            daily_factor_pnl = self.factor_pnl.copy()
            daily_factor_pnl['date'] = pd.to_datetime(daily_factor_pnl['ts']).dt.date
            daily_factor_stats = daily_factor_pnl.groupby('date')['factor_pnl'].sum().reset_index()
            
            pnl_fig.add_trace(go.Bar(
                x=daily_factor_stats['date'],
                y=daily_factor_stats['factor_pnl'],
                name='Daily Factor P&L',
                marker_color=daily_factor_stats['factor_pnl'].apply(lambda x: 'green' if x > 0 else 'red')
            ), row=2, col=1)
            
            # Compare factor P&L to total P&L
            if self.aggregate_df is not None:
                # Merge factor P&L with total P&L
                total_pnl = self.aggregate_df[['ts', 'cumulative_pnl']].copy()
                comparison = self.factor_pnl.merge(total_pnl, on='ts', how='left')
                
                # Calculate percentage of P&L from factor
                comparison['factor_pnl_pct'] = (
                    comparison['cumulative_factor_pnl'] / 
                    (comparison['cumulative_pnl'] + 1e-10) * 100
                )
                
                pnl_fig.add_trace(go.Scatter(
                    x=comparison['ts'],
                    y=comparison['factor_pnl_pct'],
                    mode='lines',
                    name='Factor P&L %',
                    line={'width': 2, 'color': 'orange'}
                ), row=3, col=1)
            
            pnl_fig.update_xaxes(title_text="Date", row=3, col=1)
            pnl_fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
            pnl_fig.update_yaxes(title_text="P&L ($)", row=2, col=1)
            pnl_fig.update_yaxes(title_text="Factor P&L (%)", row=3, col=1)
            pnl_fig.update_layout(height=800, showlegend=False)
            
            # Calculate factor statistics
            total_factor_pnl = self.factor_pnl['cumulative_factor_pnl'].iloc[-1]
            avg_exposure = self.factor_pnl['factor_exposure'].abs().mean()
            max_exposure = self.factor_pnl['factor_exposure'].abs().max()
            avg_exposure_pct = self.factor_pnl['factor_exposure_pct'].mean()
            
            # Calculate correlation between factor exposure and returns
            if self.aggregate_df is not None and len(self.aggregate_df) > 1:
                # Merge with returns data
                returns_data = self.aggregate_df[['ts', 'total_pnl_daily']].copy()
                factor_returns = self.factor_pnl.merge(returns_data, on='ts', how='inner')
                
                # Calculate correlation
                if len(factor_returns) > 10:
                    factor_correlation = factor_returns['factor_exposure'].corr(factor_returns['total_pnl_daily'])
                else:
                    factor_correlation = 0
            else:
                factor_correlation = 0
            
            # Create statistics table
            stats_data = [
                ["Total Factor P&L", f"${total_factor_pnl:,.2f}", "Average Exposure", f"${avg_exposure:,.0f}"],
                ["Max Exposure", f"${max_exposure:,.0f}", "Avg Exposure %", f"{avg_exposure_pct:.1f}%"],
                ["Factor Correlation", f"{factor_correlation:.3f}", "", ""],
            ]
            
            stats_table = html.Table([
                html.Tbody([
                    html.Tr([
                        html.Td(cell, style={
                            'padding': '8px',
                            'textAlign': 'left' if i % 2 == 0 else 'right',
                            'fontWeight': 'bold' if i % 2 == 0 else 'normal',
                            'borderBottom': '1px solid #ddd',
                            'width': '25%'
                        }) for i, cell in enumerate(row)
                    ]) for row in stats_data
                ])
            ], style={
                'width': '60%',
                'margin': '0 auto',
                'borderCollapse': 'collapse',
                'backgroundColor': '#f9f9f9',
                'border': '1px solid #ddd',
                'borderRadius': '5px'
            })
            
            # Get current top factor exposures
            if self.factor_exposures is not None:
                # Get the most recent timestamp
                latest_ts = self.factor_exposures['ts'].max()
                current_exposures = self.factor_exposures[
                    self.factor_exposures['ts'] == latest_ts
                ].copy()
                
                # Calculate absolute exposure for sorting
                current_exposures['abs_exposure'] = current_exposures['factor_exposure'].abs()
                
                # Get top 20 by absolute exposure
                top_exposures = current_exposures.nlargest(20, 'abs_exposure')
                
                # Prepare table data
                table_data = []
                for _, row in top_exposures.iterrows():
                    table_data.append({
                        'symbol': row['symbol_venue'],
                        'factor_value': row[self.factor_name],
                        'position': row['position'],
                        'factor_exposure': row['factor_exposure'],
                        'factor_pnl': row['factor_pnl']
                    })
            else:
                table_data = []
            
            return status_msg, exposure_fig, pnl_fig, stats_table, table_data


def main():
    parser = argparse.ArgumentParser(description='Simulation report with factor analysis')
    parser.add_argument('sim_name', nargs='?', help='Name of the simulation directory (optional)')
    parser.add_argument('-p', '--port', help='port', type=int, default=None)
    parser.add_argument('-f', '--factor', help='Initial factor to analyze', type=str, default=None)
    parser.add_argument('--horizon', help='Horizon for loading features', type=int, default=1440)
    args = parser.parse_args()
    
    port = args.port if args.port else REPORT_PORT
    
    report = SimulationReportWithFactors(
        sim_name=args.sim_name, 
        port=port,
        factor_name=args.factor,
        horizon=args.horizon
    )
    report.run()


if __name__ == "__main__":
    main()