import argparse
import glob
import logging
import os
from datetime import date
from typing import Optional, List, Tuple

import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Output, Input, State

from lib.data.dataloader import DataLoader
from lib.reports.base_dash_app import BaseDashApp
from lib.util.config import get_config
from lib.util.directory import CONFIG_DIR, dir_manager, DirectoryManager
from lib.util.time_util import date_str_to_date

DEFAULT_PORT = 8060

logger = logging.getLogger(__name__)


def load_config(config_file: Optional[str] = None) -> dict:
    """Load configuration from file.

    Args:
        config_file: Optional config filename (without path)

    Returns:
        Configuration dictionary
    """
    if config_file:
        full_path = f'{CONFIG_DIR}/{config_file}'
    else:
        full_path = None

    _, config = get_config(config_file=full_path)
    return config


class AlphaReport(BaseDashApp):
    """Alpha Report Dashboard Application.

    Inherits from BaseDashApp to get kill file monitoring and other common functionality.
    """

    def __init__(self, port: int = DEFAULT_PORT, interval_secs: int = 300, debug: bool = False):
        # Initialize base class (sets up self.app, logging, kill file monitoring)
        super().__init__("Alpha Report", port, interval_secs, debug)

        self.data_loader = DataLoader()
        self.data_dir = dir_manager.DATA_DIR
        self.alpha_dir = os.path.join(dir_manager.DATA_DIR, 'alpha')  # Parent alpha directory
        self.alpha_df = None

        # Initialize with default config
        self.update_config()

        # Get available alpha directories
        self.alpha_directories = self.get_available_alpha_directories()

        # Get available horizons and models for default directory
        self.horizons, self.models = self.get_available_horizons_models()

        # Setup the dash app layout and callbacks
        self.setup_layout()
        self.register_callbacks()
    
    def update_config(self, config_file: Optional[str] = None):
        """Update configuration"""
        self.config = load_config(config_file)
    
    def get_available_alpha_directories(self) -> List[str]:
        """Get list of available alpha directories under DATA_DIR"""
        alpha_dirs = []
        
        # Look for directories starting with 'alpha' in DATA_DIR
        try:
            for item in os.listdir(self.data_dir):
                if item.startswith('alpha') and os.path.isdir(os.path.join(self.data_dir, item)):
                    alpha_dirs.append(item)
            alpha_dirs.sort()
            logger.info(f"Found alpha directories: {alpha_dirs}")
        except Exception as e:
            logger.error(f"Error finding alpha directories: {e}")
            alpha_dirs = ['alpha']  # Default fallback
        
        return alpha_dirs
    
    def get_available_horizons_models(self, alpha_dir: Optional[str] = None) -> Tuple[List[int], List[str]]:
        """Get available horizons and models from the alpha directory structure"""
        if alpha_dir is None:
            alpha_dir = self.alpha_dir
            
        horizons = []
        models = set()
        has_prod = False
        has_dev = False
        
        # Check both dev and prod directory structures
        for subdir in ['dev', 'prod']:
            dir_path = os.path.join(alpha_dir, subdir)
            if os.path.exists(dir_path):
                if subdir == 'prod':
                    has_prod = True
                else:
                    has_dev = True
                    
                # Get horizons
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isdir(item_path) and item.isdigit():
                        horizons.append(int(item))
                        
                        # Get models for this horizon
                        for model in os.listdir(item_path):
                            model_path = os.path.join(item_path, model)
                            if os.path.isdir(model_path):
                                models.add(model)
        
        horizons = sorted(list(set(horizons)))
        models = sorted(list(models))
        
        logger.info(f"Found horizons in {alpha_dir}: {horizons}")
        logger.info(f"Found models in {alpha_dir}: {models}")
        logger.info(f"Has prod: {has_prod}, Has dev: {has_dev}")

        return horizons, models

    def get_models_for_env_horizon(self, env: str, horizon: int, alpha_dir: Optional[str] = None) -> List[str]:
        """Get list of models available for a specific environment and horizon."""
        if alpha_dir is None:
            alpha_dir = self.alpha_dir

        env_dir = os.path.join(alpha_dir, env, str(horizon))
        if not os.path.exists(env_dir):
            return []

        models = []
        for item in os.listdir(env_dir):
            item_path = os.path.join(env_dir, item)
            if os.path.isdir(item_path):
                models.append(item)

        return sorted(models)

    def load_alpha_data(self, start_date: date, end_date: date, horizon: int, model: str, prod: bool = False, alpha_dir: Optional[str] = None):
        """Load alpha data for specified parameters"""
        try:
            if alpha_dir is None:
                alpha_dir = self.alpha_dir
                
            logger.info(f"Loading alpha data for {model} at horizon {horizon} from {start_date} to {end_date}, prod={prod}, dir={alpha_dir}")
            
            # # Create a custom DirectoryManager if using non-default alpha directory
            # if alpha_dir != self.alpha_dir:
            #     custom_dir_manager = DirectoryManager(data_dir=self.data_dir)
            #     custom_dir_manager.ALPHA_DIR_DEV = alpha_dir
            #     data_loader = DataLoader(data_loader_dir_manager=custom_dir_manager)
            # else:
            data_loader = self.data_loader
            
            # Use load_alphas for new directory structure
            self.alpha_df = data_loader.load_alphas(
                horizon_models=[(horizon, model)],
                start_date=start_date,
                end_date=end_date,
                prod=prod
            )
            
            if self.alpha_df is not None:
                logger.info(f"Loaded {len(self.alpha_df)} alpha records at 1-minute resolution")
                logger.info(f"Alpha columns: {list(self.alpha_df.columns)}")
                
                # Filter to horizon frequency by selecting timestamps at horizon intervals
                logger.info(f"Filtering alpha data to {horizon}-minute frequency")
                
                # Get timestamps and filter to those divisible by horizon
                ts_index = self.alpha_df.index.get_level_values('ts')
                # Select timestamps where minutes since midnight is divisible by horizon
                mask = (ts_index.hour * 60 + ts_index.minute) % horizon == 0
                self.alpha_df = self.alpha_df[mask]
                
                logger.info(f"Filtered to {len(self.alpha_df)} records at {horizon}-minute resolution")
            else:
                logger.warning("No alpha data loaded")
                
        except Exception as e:
            logger.error(f"Error loading alpha data: {e}")
            self.alpha_df = None
    
    def load_forward_returns(self, start_date: date, end_date: date, horizon: int, 
                           universe: Optional[List[str]] = None, alpha_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load forward returns for specified parameters"""
        try:
            # Create a custom DirectoryManager if using non-default alpha directory
            if alpha_dir and alpha_dir != self.alpha_dir:
                custom_dir_manager = DirectoryManager(data_dir=self.data_dir)
                custom_dir_manager.ALPHA_DIR_DEV = alpha_dir
                data_loader = DataLoader(data_loader_dir_manager=custom_dir_manager)
            else:
                data_loader = self.data_loader
            
            # Use the DataLoader's method
            forwards_df = data_loader.load_forward_returns(
                start_date=start_date,
                end_date=end_date,
                horizon=horizon,
                universe=universe
            )
            
            if forwards_df is None or len(forwards_df) == 0:
                logger.warning("No forward returns data loaded")
                return None
            
            logger.info(f"Loaded {len(forwards_df)} forward return records at {horizon}-minute resolution")
            return forwards_df
            
        except Exception as e:
            logger.error(f"Error loading forward returns: {e}")
            return None
    
    def calculate_rolling_correlation(self, alpha_df: pd.DataFrame, forwards_df: pd.DataFrame, 
                                    alpha_col: str, horizon: int, forward_col: Optional[str] = None) -> Optional[pd.Series]:
        """Calculate rolling correlation between alphas and forward returns"""
        try:
            # Default to raw forward returns if not specified
            if forward_col is None:
                forward_col = f'y_raw1_{horizon}'
            
            # Merge alphas with forwards
            alpha_merge = alpha_df[[alpha_col]].reset_index()
            forwards_merge = forwards_df[[forward_col]].reset_index()
            
            merged_df = pd.merge(alpha_merge, forwards_merge, on=['ts', 'symbol_venue'], how='inner')
            
            if len(merged_df) == 0:
                logger.warning("No matching data between alphas and forwards")
                return None
            
            logger.info(f"Merged {len(merged_df)} records between alphas and forwards")
            
            # Set index back
            merged_df = merged_df.set_index(['ts', 'symbol_venue']).sort_index()
            
            # Calculate rolling correlation across all symbols
            rolling_window = 5  # 5 days for daily data
            
            logger.info(f"Using rolling window of {rolling_window} days")
            
            # Group by timestamp and calculate correlation across all symbols at each time
            correlations = []
            
            # Get unique timestamps
            unique_timestamps = sorted(merged_df.index.get_level_values('ts').unique())
            logger.info(f"Found {len(unique_timestamps)} unique timestamps")
            
            # Need at least rolling_window timestamps
            if len(unique_timestamps) < rolling_window:
                logger.warning(f"Insufficient timestamps ({len(unique_timestamps)}) for {rolling_window}-day rolling correlation")
                return None
            
            # Calculate correlation for each rolling window
            for i in range(rolling_window - 1, len(unique_timestamps)):
                window_start = unique_timestamps[i - rolling_window + 1]
                window_end = unique_timestamps[i]
                
                # Get data for this window
                window_data = merged_df.loc[(merged_df.index.get_level_values('ts') >= window_start) & 
                                          (merged_df.index.get_level_values('ts') <= window_end)]
                
                if len(window_data) > 0:
                    # Calculate correlation across all symbols in this window
                    corr = window_data[alpha_col].corr(window_data[forward_col])
                    if not pd.isna(corr):
                        correlations.append({
                            'date': window_end,
                            'correlation': corr
                        })
            
            if not correlations:
                logger.warning("No valid correlations calculated")
                return None
            
            # Convert to series
            daily_corr = pd.DataFrame(correlations).set_index('date')['correlation']
            daily_corr.index = pd.to_datetime(daily_corr.index)
            
            return daily_corr
            
        except Exception as e:
            logger.error(f"Error calculating rolling correlation: {e}")
            return None
    
    def calculate_per_security_correlations(self, alpha_df: pd.DataFrame, forwards_df: pd.DataFrame,
                                          alpha_col: str, forward_col: str) -> Optional[pd.DataFrame]:
        """Calculate correlation for each security"""
        try:
            # Merge alphas with forwards
            alpha_merge = alpha_df[[alpha_col]].reset_index()
            forwards_merge = forwards_df[[forward_col]].reset_index()
            
            merged_df = pd.merge(alpha_merge, forwards_merge, on=['ts', 'symbol_venue'], how='inner')
            
            if len(merged_df) == 0:
                logger.warning("No matching data for per-security correlations")
                return None
            
            # Calculate correlation for each symbol
            correlations = []
            for symbol in merged_df['symbol_venue'].unique():
                symbol_data = merged_df[merged_df['symbol_venue'] == symbol]
                if len(symbol_data) >= 10:  # Require at least 10 data points
                    corr = symbol_data[alpha_col].corr(symbol_data[forward_col])
                    if not pd.isna(corr):
                        correlations.append({
                            'symbol': symbol,
                            'correlation': corr,
                            'count': len(symbol_data)
                        })
            
            if not correlations:
                return None
            
            return pd.DataFrame(correlations).sort_values('correlation', ascending=False)
            
        except Exception as e:
            logger.error(f"Error calculating per-security correlations: {e}")
            return None
    
    def create_correlation_figure(self, daily_corr: Optional[pd.Series], total_corr: Optional[float],
                                 title_suffix: str = "") -> go.Figure:
        """Create a correlation figure with consistent styling"""
        corr_fig = go.Figure()
        
        if daily_corr is not None:
            # Add correlation trace
            corr_fig.add_trace(go.Scatter(
                x=daily_corr.index,
                y=daily_corr.values,
                mode='lines',
                name='Rolling Correlation',
                line={'width': 2, 'color': 'purple'}
            ))
            
            # Add reference lines
            corr_fig.add_hline(y=0, line_dash="dash", line_color="gray")
            corr_fig.add_hline(y=0.5, line_dash="dot", line_color="lightgreen", 
                             annotation_text="0.5")
            corr_fig.add_hline(y=-0.5, line_dash="dot", line_color="lightcoral", 
                             annotation_text="-0.5")
            
            # Add average correlation
            avg_corr = daily_corr.mean()
            corr_fig.add_hline(y=avg_corr, line_dash="dash", line_color="blue", 
                             annotation_text=f"Avg: {avg_corr:.3f}")
            
            # Add title with total correlation
            title_text = f"Rolling Correlation: Alpha vs {title_suffix}"
            if total_corr is not None:
                title_text += f"<br><sub>Total Correlation: {total_corr:.4f}</sub>"
            
            corr_fig.update_layout(
                title={
                    'text': title_text,
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Date",
                yaxis_title="Correlation",
                yaxis_range=[-1, 1],
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
        else:
            corr_fig.add_annotation(
                text="Insufficient data for rolling correlation calculation",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
        
        return corr_fig
    
    def setup_layout(self):
        """Setup the Dash app layout"""
        # Get date range from available files
        min_date, max_date = self.get_date_range()
        
        self.app.layout = html.Div([
            html.H1("Alpha Report", style={'textAlign': 'center'}),


            # Loading status message at the top
            html.Div(id="loading-message", style={
                'textAlign': 'center', 
                'fontSize': '18px', 
                'color': 'blue',
                'marginBottom': '20px',
                'minHeight': '30px'
            }),
            
            # Configuration and parameters
            html.Div([
                # Alpha directory selector
                html.Div([
                    html.Label("Alpha Directory:", style={'marginBottom': '10px', 'fontSize': '16px'}),
                    dcc.Dropdown(
                        id='alpha-dir-selector',
                        options=[{'label': d, 'value': d} for d in self.alpha_directories],
                        value='alpha',  # Default to standard alpha directory
                        style={'marginBottom': '20px', 'width': '300px'}
                    ),
                ], style={'marginBottom': '20px'}),
                        
                        # Date range selector
                        html.Div([
                            html.Label("Date Range:", style={'marginBottom': '10px', 'fontSize': '16px'}),
                            dcc.DatePickerRange(
                                id='date-range-picker',
                                start_date=min_date,
                                end_date=max_date,
                                display_format='YYYY-MM-DD',
                                style={'marginBottom': '20px'}
                            ),
                        ], style={'marginBottom': '20px'}),
                        
                        # Horizon selector
                        html.Div([
                            html.Label("Horizon:", style={'marginBottom': '10px', 'fontSize': '16px'}),
                            dcc.Dropdown(
                                id='horizon-selector',
                                options=[{'label': str(h), 'value': h} for h in self.horizons],
                                value=self.horizons[0] if self.horizons else None,
                                style={'marginBottom': '20px', 'width': '200px'}
                            ),
                        ], style={'display': 'inline-block', 'marginRight': '20px'}),
                        
                        # Model selector
                        html.Div([
                            html.Label("Model:", style={'marginBottom': '10px', 'fontSize': '16px'}),
                            dcc.Dropdown(
                                id='model-selector',
                                options=[{'label': m, 'value': m} for m in self.models],
                                value=self.models[0] if self.models else None,
                                style={'marginBottom': '20px', 'width': '200px'}
                            ),
                        ], style={'display': 'inline-block', 'marginRight': '20px'}),
                        
                        # Prod/Dev selector
                        html.Div([
                            html.Label("Environment:", style={'marginBottom': '10px', 'fontSize': '16px'}),
                            dcc.Dropdown(
                                id='env-selector',
                                options=[
                                    {'label': 'Production', 'value': 'prod'},
                                    {'label': 'Development', 'value': 'dev'}
                                ],
                                value='prod',
                                style={'marginBottom': '20px', 'width': '200px'}
                            ),
                        ], style={'display': 'inline-block', 'marginRight': '20px'}),
                        
                        # Recenter checkbox
                        html.Div([
                            dcc.Checklist(
                                id='recenter-checkbox',
                                options=[{'label': ' Recenter (subtract cross-sectional median)', 'value': 'recenter'}],
                                value=[],
                                style={'marginTop': '25px', 'fontSize': '16px'}
                            ),
                        ], style={'display': 'inline-block', 'marginRight': '20px'}),
                        
                        # Dynamic universe checkbox
                        html.Div([
                            dcc.Checklist(
                                id='dynamic-universe-checkbox',
                                options=[{'label': ' Dynamic Universe', 'value': 'dynamic'}],
                                value=['dynamic'],  # Default to True
                                style={'marginTop': '25px', 'fontSize': '16px'}
                            ),
                        ], style={'display': 'inline-block', 'marginRight': '20px'}),
                        
                        # Load button
                        html.Div([
                            html.Button("Load Alpha Data", id="load-btn", n_clicks=0, 
                                       style={'marginTop': '25px', 'height': '40px', 'fontSize': '16px'}),
                        ], style={'display': 'inline-block'}),
                        
                
                html.Div(id="load-status", style={"marginTop": "10px", "color": "green"}),
            ], style={'marginBottom': '30px'}),
            
            # Loading wrapper for all charts
            dcc.Loading(
                id="loading",
                type="default",
                children=[
                    # Summary statistics
                    html.Div([
                        html.H3("Alpha Statistics", style={'textAlign': 'center'}),
                        html.Div(id='summary-stats', style={'textAlign': 'center', 'marginBottom': '20px'}),
                    ]),
                    
                    # Alpha mean over time
                    html.Div([
                        html.H3("Alpha Mean Over Time", style={'textAlign': 'center'}),
                        dcc.Graph(id='alpha-mean-timeseries'),
                    ], style={'marginBottom': '30px'}),
                    
                    # Alpha standard deviation over time
                    html.Div([
                        html.H3("Alpha Standard Deviation Over Time", style={'textAlign': 'center'}),
                        dcc.Graph(id='alpha-std-timeseries'),
                    ], style={'marginBottom': '30px'}),
                    
                    # Alpha distribution
                    html.Div([
                        html.H3("Alpha Distribution", style={'textAlign': 'center'}),
                        dcc.Graph(id='alpha-distribution'),
                    ], style={'marginBottom': '30px'}),
                    
                    # Momentum vs Reversion breakdown
                    html.Div([
                        html.H3("Momentum vs Reversion Alpha Over Time", style={'textAlign': 'center'}),
                        dcc.Graph(id='mom-rev-timeseries'),
                    ], style={'marginBottom': '30px'}),
                    
                    # Lag-based Alpha Mean
                    html.Div([
                        html.H3("Lag-based Alpha Mean Over Time", style={'textAlign': 'center'}),
                        dcc.Graph(id='lag-mean-timeseries'),
                    ], style={'marginBottom': '30px'}),
                    
                    # Lag-based Alpha Standard Deviation
                    html.Div([
                        html.H3("Lag-based Alpha Standard Deviation Over Time", style={'textAlign': 'center'}),
                        dcc.Graph(id='lag-std-timeseries'),
                    ], style={'marginBottom': '30px'}),
                    
                    # Lag-based Momentum vs Reversion Condition Counts
                    html.Div([
                        html.H3("Lag-based Momentum vs Reversion Condition Counts Over Time", style={'textAlign': 'center'}),
                        dcc.Graph(id='lag-condition-counts'),
                    ], style={'marginBottom': '30px'}),
                    
                    # Positive vs Negative Alpha Counts
                    html.Div([
                        html.H3("Positive vs Negative Alpha Counts Over Time", style={'textAlign': 'center'}),
                        dcc.Graph(id='pos-neg-alpha-counts'),
                    ], style={'marginBottom': '30px'}),
                    
                    # Alpha Min/Max Outliers Over Time
                    html.Div([
                        html.H3("Alpha Min/Max Outliers Over Time", style={'textAlign': 'center'}),
                        dcc.Graph(id='alpha-outliers-timeseries'),
                    ], style={'marginBottom': '30px'}),
                    
                    # Rolling Correlation with Forward Returns
                    html.Div([
                        html.H3("Rolling Correlation: Alpha vs Forward Returns", style={'textAlign': 'center'}),
                        dcc.Graph(id='alpha-forward-correlation'),
                    ], style={'marginBottom': '30px'}),
                    
                    # Rolling Correlation with Funding-Adjusted Market-Residualized Forward Returns
                    html.Div([
                        html.H3("Rolling Correlation: Alpha vs Funding-Adjusted Market-Residualized Forward Returns", style={'textAlign': 'center'}),
                        dcc.Graph(id='alpha-forward-correlation-adjusted'),
                    ], style={'marginBottom': '30px'}),
                    
                    # Per-Security Correlation Bar Chart
                    html.Div([
                        html.H3("Per-Security Alpha Correlations", style={'textAlign': 'center'}),
                        dcc.Graph(id='per-security-correlations'),
                    ], style={'marginBottom': '30px'}),
                ],
            ),
        ])
    
    def get_date_range(self, alpha_dir: Optional[str] = None) -> Tuple[date, date]:
        """Get the available date range from alpha files"""
        if alpha_dir is None:
            alpha_dir = self.alpha_dir
            
        try:
            # Look for any alpha files to determine date range
            # Check all subdirectory patterns (direct, dev, and prod) to get the full date range
            files = []

            # First try direct pattern
            pattern = f"{alpha_dir}/*/alphas.*.parquet"
            files.extend(glob.glob(pattern))

            # Also check dev subdirectory
            pattern = f"{alpha_dir}/dev/*/*/alphas.*.parquet"
            files.extend(glob.glob(pattern))

            # Also check prod subdirectory (may have newer files)
            pattern = f"{alpha_dir}/prod/*/*/alphas.*.parquet"
            files.extend(glob.glob(pattern))
            
            if not files:
                # Default to last 30 days
                end_date = date.today()
                start_date = end_date - pd.Timedelta(days=30)
                return start_date, end_date
            
            # Extract dates from filenames
            dates = []
            for f in files:
                parts = os.path.basename(f).split('.')
                if len(parts) >= 5:
                    date_str = parts[4]
                    try:
                        file_date = date_str_to_date(date_str)
                        dates.append(file_date)
                    except:
                        continue
            
            if dates:
                # Use actual date range from files, capped to last 30 days
                end_date = min(date.today(), max(dates))
                start_date = max(end_date - pd.Timedelta(days=30), min(dates))
                logger.info(f"Date range from files: {min(dates)} to {max(dates)}, using {start_date} to {end_date}")
                return start_date, end_date
            else:
                # Default to last 90 days
                end_date = date.today()
                start_date = end_date - pd.Timedelta(days=90)
                return start_date, end_date
                
        except Exception as e:
            logger.error(f"Error getting date range: {e}")
            # Default to last 90 days
            end_date = date.today()
            start_date = end_date - pd.Timedelta(days=90)
            return start_date, end_date
    
    def register_callbacks(self):
        """Register Dash callbacks"""
        
        # Callback to update horizons, models, and date range when directory changes
        @self.app.callback(
            Output('horizon-selector', 'options'),
            Output('horizon-selector', 'value'),
            Output('model-selector', 'options'),
            Output('model-selector', 'value'),
            Output('date-range-picker', 'start_date'),
            Output('date-range-picker', 'end_date'),
            Input('alpha-dir-selector', 'value')
        )
        def update_directory_dependent_options(selected_dir):
            """Update horizons, models, and date range when alpha directory changes"""
            if not selected_dir:
                return [], None, [], None, None, None
                
            # Get full path
            alpha_dir = os.path.join(self.data_dir, selected_dir)
            
            # Get available horizons and models
            horizons, models = self.get_available_horizons_models(alpha_dir)
            
            # Get date range
            min_date, max_date = self.get_date_range(alpha_dir)
            
            # Create options
            horizon_options = [{'label': str(h), 'value': h} for h in horizons]
            model_options = [{'label': m, 'value': m} for m in models]
            
            # Set default values
            horizon_value = horizons[0] if horizons else None
            model_value = models[0] if models else None
            
            return horizon_options, horizon_value, model_options, model_value, min_date, max_date

        # Callback to update model options when horizon or env changes
        @self.app.callback(
            Output('model-selector', 'options', allow_duplicate=True),
            Output('model-selector', 'value', allow_duplicate=True),
            Input('horizon-selector', 'value'),
            Input('env-selector', 'value'),
            State('alpha-dir-selector', 'value'),
            prevent_initial_call=True
        )
        def update_model_options(selected_horizon, selected_env, selected_dir):
            """Update available models based on selected horizon and environment."""
            if not selected_horizon or not selected_env:
                return [], None

            alpha_dir = os.path.join(self.data_dir, selected_dir) if selected_dir else self.alpha_dir
            models = self.get_models_for_env_horizon(selected_env, selected_horizon, alpha_dir)
            options = [{'label': m, 'value': m} for m in models]
            default_value = models[0] if models else None

            return options, default_value

        # Callback to show loading message when button is clicked
        @self.app.callback(
            Output("loading-message", "children"),
            Input("load-btn", "n_clicks"),
            State("model-selector", "value"),
            State("horizon-selector", "value"),
            prevent_initial_call=True
        )
        def show_loading_message(n_clicks, model, horizon):
            """Show loading message immediately when button is clicked"""
            if n_clicks and n_clicks > 0 and model and horizon:
                return f"Loading {model} alpha data at {horizon}-minute horizon..."
            return ""
        
        @self.app.callback(
            Output("load-status", "children"),
            Output('summary-stats', 'children'),
            Output('alpha-mean-timeseries', 'figure'),
            Output('alpha-std-timeseries', 'figure'),
            Output('alpha-distribution', 'figure'),
            Output('mom-rev-timeseries', 'figure'),
            Output('lag-mean-timeseries', 'figure'),
            Output('lag-std-timeseries', 'figure'),
            Output('lag-condition-counts', 'figure'),
            Output('pos-neg-alpha-counts', 'figure'),
            Output('alpha-outliers-timeseries', 'figure'),
            Output('alpha-forward-correlation', 'figure'),
            Output('alpha-forward-correlation-adjusted', 'figure'),
            Output('per-security-correlations', 'figure'),
            Input("load-btn", "n_clicks"),
            State("alpha-dir-selector", "value"),
            State("date-range-picker", "start_date"),
            State("date-range-picker", "end_date"),
            State("horizon-selector", "value"),
            State("model-selector", "value"),
            State("env-selector", "value"),
            State("recenter-checkbox", "value"),
            State("dynamic-universe-checkbox", "value"),
            prevent_initial_call=True,
        )
        def update_displays(n_clicks, alpha_dir_name, start_date_str, end_date_str, horizon, model, env, recenter_values, dynamic_universe_values):
            """Update all displays when load button is clicked"""

            # Only proceed if button was actually clicked
            if n_clicks == 0:
                return "Click 'Load Alpha Data' to begin", "", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()
            
            # Convert date strings to date objects
            if start_date_str and end_date_str and horizon and model:
                start_date = date_str_to_date(start_date_str.replace('-', ''))
                end_date = date_str_to_date(end_date_str.replace('-', ''))
                prod = (env == 'prod')
                
                # Get full alpha directory path
                alpha_dir = os.path.join(self.data_dir, alpha_dir_name) if alpha_dir_name else self.alpha_dir
                
                # Load alpha data
                self.load_alpha_data(start_date, end_date, horizon, model, prod, alpha_dir)
                
                if self.alpha_df is None or len(self.alpha_df) == 0:
                    return "No alpha data loaded", "", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()
                
                # Get the alpha column name
                alpha_col = f'alpha_{model}_{horizon}'
                
                if alpha_col not in self.alpha_df.columns:
                    logger.warning(f"Column {alpha_col} not found. Available columns: {list(self.alpha_df.columns)}")
                    return f"Alpha column {alpha_col} not found in data", "", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()
                
                # Apply universe filtering if dynamic universe is disabled
                if not (dynamic_universe_values and 'dynamic' in dynamic_universe_values):
                    logger.info("Filtering to static universe from config")
                    # Get the symbol universe from config
                    symbol_universe = self.config.get('SYMBOL_UNIVERSE', [])
                    if symbol_universe:
                        # Create symbol_venue list from symbols (assuming binance-futures venue)
                        symbol_venue_universe = [f"{symbol}USDT_binance-futures" for symbol in symbol_universe]
                        
                        # Filter the alpha dataframe
                        initial_count = len(self.alpha_df)
                        self.alpha_df = self.alpha_df[self.alpha_df.index.get_level_values('symbol_venue').isin(symbol_venue_universe)]
                        filtered_count = len(self.alpha_df)
                        
                        logger.info(f"Filtered from {initial_count} to {filtered_count} records using {len(symbol_universe)} symbols from config")
                        
                        if len(self.alpha_df) == 0:
                            return "No data after universe filtering", "", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()
                    else:
                        logger.warning("SYMBOL_UNIVERSE not found in config")
                
                # Apply recentering if requested
                if recenter_values and 'recenter' in recenter_values:
                    logger.info("Applying cross-sectional recentering")
                    # Group by timestamp and subtract the median at each timestamp
                    self.alpha_df[alpha_col] = self.alpha_df.groupby(level='ts')[alpha_col].transform(
                        lambda x: x - x.median()
                    )
                    
                    # Also recenter momentum and reversion columns if they exist
                    mom_col = f'alpha_{model}_{horizon}_mom'
                    rev_col = f'alpha_{model}_{horizon}_rev'
                    
                    if mom_col in self.alpha_df.columns:
                        self.alpha_df[mom_col] = self.alpha_df.groupby(level='ts')[mom_col].transform(
                            lambda x: x - x.median()
                        )
                    
                    if rev_col in self.alpha_df.columns:
                        self.alpha_df[rev_col] = self.alpha_df.groupby(level='ts')[rev_col].transform(
                            lambda x: x - x.median()
                        )
                    
                    logger.info("Recentering complete")
                
                # Calculate daily statistics
                daily_stats = self.alpha_df.groupby(self.alpha_df.index.get_level_values('ts').date)[alpha_col].agg(['mean', 'std', 'count'])
                daily_stats.index = pd.to_datetime(daily_stats.index)
                
                # Summary statistics
                overall_mean = self.alpha_df[alpha_col].mean()
                overall_std = self.alpha_df[alpha_col].std()
                total_count = len(self.alpha_df)
                
                # Add recentering and universe status to summary
                recenter_status = " (Recentered)" if recenter_values and 'recenter' in recenter_values else ""
                universe_status = " (Dynamic)" if dynamic_universe_values and 'dynamic' in dynamic_universe_values else " (Static)"
                
                summary_text = html.Div([
                    html.P(f"Model: {model}, Horizon: {horizon}{recenter_status}{universe_status}"),
                    html.P(f"Date Range: {start_date} to {end_date}"),
                    html.P(f"Total Data Points: {total_count:,}"),
                    html.P(f"Overall Mean Alpha: {overall_mean:.6f}"),
                    html.P(f"Overall Std Dev: {overall_std:.6f}"),
                ])
                
                # Mean alpha over time
                mean_fig = go.Figure()
                mean_fig.add_trace(go.Scatter(
                    x=daily_stats.index,
                    y=daily_stats['mean'],
                    mode='lines',
                    name='Daily Mean Alpha',
                    line={'width': 2}
                ))
                mean_fig.add_hline(y=overall_mean, line_dash="dash", line_color="red", 
                                  annotation_text=f"Overall Mean: {overall_mean:.6f}")
                mean_fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Mean Alpha",
                    hovermode='x unified'
                )
                
                # Standard deviation over time
                std_fig = go.Figure()
                std_fig.add_trace(go.Scatter(
                    x=daily_stats.index,
                    y=daily_stats['std'],
                    mode='lines',
                    name='Daily Std Dev',
                    line={'width': 2, 'color': 'orange'}
                ))
                std_fig.add_hline(y=overall_std, line_dash="dash", line_color="red", 
                                 annotation_text=f"Overall Std: {overall_std:.6f}")
                std_fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Standard Deviation",
                    hovermode='x unified'
                )
                
                # Alpha distribution histogram
                # Clip to 1st-99th percentile to remove outliers that compress the histogram
                dist_fig = go.Figure()
                alpha_values = self.alpha_df[alpha_col].dropna()

                if len(alpha_values) == 0:
                    logger.warning(f"No valid alpha values for distribution histogram (column: {alpha_col})")
                    dist_fig.add_annotation(
                        text="No valid alpha values for histogram",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=14)
                    )
                else:
                    p1, p99 = alpha_values.quantile(0.01), alpha_values.quantile(0.99)
                    alpha_clipped = alpha_values[(alpha_values >= p1) & (alpha_values <= p99)]

                    if len(alpha_clipped) == 0:
                        logger.warning(f"No values within 1st-99th percentile range (column: {alpha_col})")
                        dist_fig.add_annotation(
                            text="All alpha values are outliers",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5,
                            showarrow=False,
                            font=dict(size=14)
                        )
                    else:
                        dist_fig.add_trace(go.Histogram(
                            x=alpha_clipped,
                            nbinsx=100,
                            name='Alpha Distribution',
                            histnorm='probability'
                        ))
                        dist_fig.update_layout(
                            xaxis_title="Alpha Value",
                            yaxis_title="Probability",
                            showlegend=False,
                            title=f"Alpha Distribution (1st-99th percentile, n={len(alpha_clipped):,})"
                        )
                
                # Momentum vs Reversion breakdown
                mom_rev_fig = go.Figure()
                
                # Get the momentum and reversion column names
                mom_col = f'alpha_{model}_{horizon}_mom'
                rev_col = f'alpha_{model}_{horizon}_rev'
                
                # Check if momentum and reversion columns exist
                if mom_col in self.alpha_df.columns and rev_col in self.alpha_df.columns:
                    # Calculate daily mean for momentum and reversion
                    daily_mom = self.alpha_df.groupby(self.alpha_df.index.get_level_values('ts').date)[mom_col].mean()
                    daily_rev = self.alpha_df.groupby(self.alpha_df.index.get_level_values('ts').date)[rev_col].mean()
                    daily_mom.index = pd.to_datetime(daily_mom.index)
                    daily_rev.index = pd.to_datetime(daily_rev.index)
                    
                    # Add momentum trace
                    mom_rev_fig.add_trace(go.Scatter(
                        x=daily_mom.index,
                        y=daily_mom.values,
                        mode='lines',
                        name='Momentum Alpha',
                        line={'width': 2, 'color': 'green'}
                    ))
                    
                    # Add reversion trace
                    mom_rev_fig.add_trace(go.Scatter(
                        x=daily_rev.index,
                        y=daily_rev.values,
                        mode='lines',
                        name='Reversion Alpha',
                        line={'width': 2, 'color': 'red'}
                    ))
                    
                    # Add total alpha trace (should equal momentum + reversion)
                    mom_rev_fig.add_trace(go.Scatter(
                        x=daily_stats.index,
                        y=daily_stats['mean'],
                        mode='lines',
                        name='Total Alpha',
                        line={'width': 2, 'color': 'blue', 'dash': 'dash'}
                    ))
                    
                    mom_rev_fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Mean Alpha",
                        hovermode='x unified',
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                else:
                    # If columns don't exist, show a message
                    mom_rev_fig.add_annotation(
                        text="Momentum and Reversion breakdown not available for this model/horizon",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=14)
                    )
                
                # Lag-based Alpha visualization
                lag_mean_fig = go.Figure()
                lag_std_fig = go.Figure()
                
                # Look for lag columns (L0, L1, etc.)
                lag_cols = []
                for col in self.alpha_df.columns:
                    if col.startswith(f'{model}_{horizon}_L') and col[len(f'{model}_{horizon}_L'):].isdigit():
                        lag_cols.append(col)
                
                if lag_cols:
                    # Sort lag columns by lag number
                    lag_cols.sort(key=lambda x: int(x.split('_L')[-1]))
                    
                    # Calculate daily statistics for each lag
                    for lag_col in lag_cols:
                        lag_num = lag_col.split('_L')[-1]
                        
                        # Daily mean
                        daily_lag_mean = self.alpha_df.groupby(self.alpha_df.index.get_level_values('ts').date)[lag_col].mean()
                        daily_lag_mean.index = pd.to_datetime(daily_lag_mean.index)
                        
                        # Daily std
                        daily_lag_std = self.alpha_df.groupby(self.alpha_df.index.get_level_values('ts').date)[lag_col].std()
                        daily_lag_std.index = pd.to_datetime(daily_lag_std.index)
                        
                        # Add mean trace
                        lag_mean_fig.add_trace(go.Scatter(
                            x=daily_lag_mean.index,
                            y=daily_lag_mean.values,
                            mode='lines',
                            name=f'Lag {lag_num}',
                            line={'width': 2}
                        ))
                        
                        # Add std trace
                        lag_std_fig.add_trace(go.Scatter(
                            x=daily_lag_std.index,
                            y=daily_lag_std.values,
                            mode='lines',
                            name=f'Lag {lag_num}',
                            line={'width': 2}
                        ))
                    
                    # Update mean figure layout
                    lag_mean_fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Mean Alpha",
                        hovermode='x unified',
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    # Update std figure layout
                    lag_std_fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Standard Deviation",
                        hovermode='x unified',
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                else:
                    # If no lag columns exist, show a message
                    for fig in [lag_mean_fig, lag_std_fig]:
                        fig.add_annotation(
                            text="Lag-based alpha breakdown not available for this model/horizon",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5,
                            showarrow=False,
                            font=dict(size=14)
                        )
                
                # Lag-based Condition Counts visualization
                lag_condition_fig = go.Figure()
                
                # Look for condition columns for each lag
                if lag_cols:
                    for lag_col in lag_cols:
                        lag_num = lag_col.split('_L')[-1]
                        condition_col = f'{model}_{horizon}_L{lag_num}_condition'
                        
                        if condition_col in self.alpha_df.columns:
                            # Group by date and condition, count occurrences
                            condition_counts = self.alpha_df.groupby([
                                self.alpha_df.index.get_level_values('ts').date,
                                self.alpha_df[condition_col]
                            ]).size().unstack(fill_value=0)
                            
                            condition_counts.index = pd.to_datetime(condition_counts.index)
                            
                            # Add traces for momentum (1) and reversion (-1)
                            if 1 in condition_counts.columns:
                                lag_condition_fig.add_trace(go.Scatter(
                                    x=condition_counts.index,
                                    y=condition_counts[1],
                                    mode='lines',
                                    name=f'L{lag_num} Momentum',
                                    line={'width': 2}
                                ))
                            
                            if -1 in condition_counts.columns:
                                lag_condition_fig.add_trace(go.Scatter(
                                    x=condition_counts.index,
                                    y=condition_counts[-1],
                                    mode='lines',
                                    name=f'L{lag_num} Reversion',
                                    line={'width': 2, 'dash': 'dash'}
                                ))
                    
                    lag_condition_fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Count",
                        hovermode='x unified',
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                else:
                    # If no lag columns exist, show a message
                    lag_condition_fig.add_annotation(
                        text="Lag-based condition counts not available for this model/horizon",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=14)
                    )
                
                # Positive vs Negative Alpha Counts visualization
                pos_neg_fig = go.Figure()
                
                # Count positive and negative alphas by date
                daily_positive = self.alpha_df[self.alpha_df[alpha_col] > 0].groupby(
                    self.alpha_df[self.alpha_df[alpha_col] > 0].index.get_level_values('ts').date
                ).size()
                
                daily_negative = self.alpha_df[self.alpha_df[alpha_col] <= 0].groupby(
                    self.alpha_df[self.alpha_df[alpha_col] <= 0].index.get_level_values('ts').date
                ).size()
                
                # Convert index to datetime
                if len(daily_positive) > 0:
                    daily_positive.index = pd.to_datetime(daily_positive.index)
                if len(daily_negative) > 0:
                    daily_negative.index = pd.to_datetime(daily_negative.index)
                
                # Add positive alpha count trace
                if len(daily_positive) > 0:
                    pos_neg_fig.add_trace(go.Scatter(
                        x=daily_positive.index,
                        y=daily_positive.values,
                        mode='lines',
                        name='Positive Alphas',
                        line={'width': 2, 'color': 'green'}
                    ))
                
                # Add negative alpha count trace
                if len(daily_negative) > 0:
                    pos_neg_fig.add_trace(go.Scatter(
                        x=daily_negative.index,
                        y=daily_negative.values,
                        mode='lines',
                        name='Negative Alphas',
                        line={'width': 2, 'color': 'red'}
                    ))
                
                # Update layout
                pos_neg_fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Count",
                    hovermode='x unified',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                # Alpha Min/Max Outliers Over Time
                outliers_fig = go.Figure()
                
                # Calculate daily min and max
                daily_min = self.alpha_df.groupby(self.alpha_df.index.get_level_values('ts').date)[alpha_col].min()
                daily_max = self.alpha_df.groupby(self.alpha_df.index.get_level_values('ts').date)[alpha_col].max()
                daily_min.index = pd.to_datetime(daily_min.index)
                daily_max.index = pd.to_datetime(daily_max.index)
                
                # Add max trace
                outliers_fig.add_trace(go.Scatter(
                    x=daily_max.index,
                    y=daily_max.values,
                    mode='lines',
                    name='Daily Maximum',
                    line={'width': 2, 'color': 'darkgreen'}
                ))
                
                # Add min trace
                outliers_fig.add_trace(go.Scatter(
                    x=daily_min.index,
                    y=daily_min.values,
                    mode='lines',
                    name='Daily Minimum',
                    line={'width': 2, 'color': 'darkred'}
                ))
                
                # Add zero line
                outliers_fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                                      annotation_text="Zero")
                
                # Calculate percentiles for context
                p99 = self.alpha_df[alpha_col].quantile(0.99)
                p1 = self.alpha_df[alpha_col].quantile(0.01)
                
                # Add percentile lines
                outliers_fig.add_hline(y=p99, line_dash="dot", line_color="lightgreen", 
                                      annotation_text=f"99th percentile: {p99:.6f}")
                outliers_fig.add_hline(y=p1, line_dash="dot", line_color="lightcoral", 
                                      annotation_text=f"1st percentile: {p1:.6f}")
                
                outliers_fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Alpha Value",
                    hovermode='x unified',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                # Rolling Correlation with Forward Returns
                # Initialize both correlation figures
                corr_fig = go.Figure()
                corr_fig_adjusted = go.Figure()
                forwards_df = None  # Initialize for later use in per-security correlations

                # Only calculate correlations if alpha data is available
                if self.alpha_df is not None and len(self.alpha_df) > 0:
                    # Get universe from alpha data
                    universe = list(self.alpha_df.index.get_level_values('symbol_venue').unique())
                    logger.info(f"Loading forwards for {len(universe)} symbols from alpha data")

                    # Load forward returns
                    forwards_df = self.load_forward_returns(start_date, end_date, horizon, universe, alpha_dir)

                    if forwards_df is not None:
                        # Raw forward returns correlation
                        raw_forward_col = f'y_raw1_{horizon}'
                        if raw_forward_col in forwards_df.columns:
                            # Calculate total correlation
                            alpha_merge = self.alpha_df[[alpha_col]].reset_index()
                            forwards_merge = forwards_df[[raw_forward_col]].reset_index()

                            merged_df = pd.merge(alpha_merge, forwards_merge, on=['ts', 'symbol_venue'], how='inner')

                            if len(merged_df) > 0:
                                total_corr = merged_df[alpha_col].corr(merged_df[raw_forward_col])
                                logger.info(f"Total correlation (raw) across all data points: {total_corr:.4f}")
                            else:
                                total_corr = None

                            # Calculate rolling correlation
                            daily_corr = self.calculate_rolling_correlation(self.alpha_df, forwards_df, alpha_col, horizon, raw_forward_col)

                            # Create figure for raw returns
                            corr_fig = self.create_correlation_figure(daily_corr, total_corr, "Forward Returns")
                        else:
                            corr_fig.add_annotation(
                                text=f"Raw forward returns column {raw_forward_col} not found",
                                xref="paper", yref="paper",
                                x=0.5, y=0.5,
                                showarrow=False,
                                font=dict(size=14)
                            )

                        # Funding-adjusted market-residualized forward returns correlation
                        adjusted_forward_col = f'y_funding_adj_resid_wgtmkt1_{horizon}'
                        if adjusted_forward_col in forwards_df.columns:
                            # Calculate total correlation
                            alpha_merge = self.alpha_df[[alpha_col]].reset_index()
                            forwards_merge_adj = forwards_df[[adjusted_forward_col]].reset_index()

                            merged_df_adj = pd.merge(alpha_merge, forwards_merge_adj, on=['ts', 'symbol_venue'], how='inner')

                            if len(merged_df_adj) > 0:
                                total_corr_adj = merged_df_adj[alpha_col].corr(merged_df_adj[adjusted_forward_col])
                                logger.info(f"Total correlation (adjusted) across all data points: {total_corr_adj:.4f}")
                            else:
                                total_corr_adj = None

                            # Calculate rolling correlation
                            daily_corr_adj = self.calculate_rolling_correlation(self.alpha_df, forwards_df, alpha_col, horizon, adjusted_forward_col)

                            # Create figure for adjusted returns
                            corr_fig_adjusted = self.create_correlation_figure(daily_corr_adj, total_corr_adj,
                                                                             "Funding-Adjusted Market-Residualized Forward Returns")
                        else:
                            corr_fig_adjusted.add_annotation(
                                text=f"Adjusted forward returns column {adjusted_forward_col} not found",
                                xref="paper", yref="paper",
                                x=0.5, y=0.5,
                                showarrow=False,
                                font=dict(size=14)
                            )
                    else:
                        # No forward returns data
                        for fig in [corr_fig, corr_fig_adjusted]:
                            fig.add_annotation(
                                text="Forward returns data not available",
                                xref="paper", yref="paper",
                                x=0.5, y=0.5,
                                showarrow=False,
                                font=dict(size=14)
                            )
                else:
                    # No alpha data available for correlation
                    for fig in [corr_fig, corr_fig_adjusted]:
                        fig.add_annotation(
                            text="Alpha data not available for correlation calculation",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5,
                            showarrow=False,
                            font=dict(size=14)
                        )
                
                # Per-Security Correlation Bar Chart
                per_security_fig = go.Figure()

                if forwards_df is not None and self.alpha_df is not None and len(self.alpha_df) > 0:
                    # Calculate per-security correlations for both return types
                    raw_forward_col = f'y_raw1_{horizon}'
                    adjusted_forward_col = f'y_funding_adj_resid_wgtmkt1_{horizon}'
                    
                    raw_corrs = None
                    adj_corrs = None
                    
                    if raw_forward_col in forwards_df.columns:
                        raw_corrs = self.calculate_per_security_correlations(
                            self.alpha_df, forwards_df, alpha_col, raw_forward_col
                        )
                    
                    if adjusted_forward_col in forwards_df.columns:
                        adj_corrs = self.calculate_per_security_correlations(
                            self.alpha_df, forwards_df, alpha_col, adjusted_forward_col
                        )
                    
                    # Merge correlations if both exist
                    if raw_corrs is not None and adj_corrs is not None:
                        # Merge on symbol
                        merged_corrs = pd.merge(
                            raw_corrs[['symbol', 'correlation']],
                            adj_corrs[['symbol', 'correlation']],
                            on='symbol',
                            suffixes=('_raw', '_adj')
                        )
                        
                        # Sort by average correlation
                        merged_corrs['avg_corr'] = (merged_corrs['correlation_raw'] + merged_corrs['correlation_adj']) / 2
                        merged_corrs = merged_corrs.sort_values('avg_corr', ascending=True)  # Ascending for horizontal bar
                        
                        # Limit to top and bottom 30 symbols
                        if len(merged_corrs) > 60:
                            top_30 = merged_corrs.tail(30)
                            bottom_30 = merged_corrs.head(30)
                            merged_corrs = pd.concat([bottom_30, top_30])
                        
                        # Create bar chart
                        per_security_fig.add_trace(go.Bar(
                            y=merged_corrs['symbol'],
                            x=merged_corrs['correlation_raw'],
                            name='Raw Forward Returns',
                            orientation='h',
                            marker=dict(color='lightblue')
                        ))
                        
                        per_security_fig.add_trace(go.Bar(
                            y=merged_corrs['symbol'],
                            x=merged_corrs['correlation_adj'],
                            name='Funding-Adj Market-Resid Returns',
                            orientation='h',
                            marker=dict(color='darkblue')
                        ))
                        
                        # Add zero line
                        per_security_fig.add_vline(x=0, line_dash="dash", line_color="gray")
                        
                        per_security_fig.update_layout(
                            xaxis_title="Correlation",
                            yaxis_title="Symbol",
                            xaxis_range=[-1, 1],
                            barmode='group',
                            height=max(600, len(merged_corrs) * 20),  # Dynamic height
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="right",
                                x=0.99
                            )
                        )
                    elif raw_corrs is not None:
                        # Only raw correlations available
                        raw_corrs = raw_corrs.sort_values('correlation', ascending=True)
                        
                        # Limit to top and bottom 30
                        if len(raw_corrs) > 60:
                            top_30 = raw_corrs.tail(30)
                            bottom_30 = raw_corrs.head(30)
                            raw_corrs = pd.concat([bottom_30, top_30])
                        
                        per_security_fig.add_trace(go.Bar(
                            y=raw_corrs['symbol'],
                            x=raw_corrs['correlation'],
                            name='Raw Forward Returns',
                            orientation='h',
                            marker=dict(color='lightblue')
                        ))
                        
                        per_security_fig.add_vline(x=0, line_dash="dash", line_color="gray")
                        
                        per_security_fig.update_layout(
                            xaxis_title="Correlation",
                            yaxis_title="Symbol",
                            xaxis_range=[-1, 1],
                            height=max(600, len(raw_corrs) * 20)
                        )
                    else:
                        per_security_fig.add_annotation(
                            text="No per-security correlations could be calculated",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5,
                            showarrow=False,
                            font=dict(size=14)
                        )
                else:
                    per_security_fig.add_annotation(
                        text="Forward returns data not available",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=14)
                    )
                
                status_msg = f"Loaded {total_count:,} alpha values for {model} (horizon={horizon}) from {start_date} to {end_date}"
                
                return status_msg, summary_text, mean_fig, std_fig, dist_fig, mom_rev_fig, lag_mean_fig, lag_std_fig, lag_condition_fig, pos_neg_fig, outliers_fig, corr_fig, corr_fig_adjusted, per_security_fig
            
            return "Please select all parameters and click Load", "", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Alpha report showing alpha statistics over time')
    parser.add_argument('-p', '--port', help='port', type=int, default=DEFAULT_PORT)
    parser.add_argument('-d', '--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('-i', '--interval', help='Refresh interval in seconds', type=int, default=300)
    args = parser.parse_args()

    report = AlphaReport(port=args.port, interval_secs=args.interval, debug=args.debug)
    report.run(debug=args.debug)