import os
import glob
import logging.config
import argparse
from typing import Optional, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Output, Input, State
from dash.dash_table import DataTable
from dash.dash_table.Format import Format

from lib.util.config import get_config
from lib.util.directory import CONFIG_DIR, dir_manager
from lib.util.util import LOCAL
from lib.util.logging_util import get_logging_config

logging.config.dictConfig(get_logging_config("fits_comparison_report"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_PORT = 8061

class FitsComparisonReport:
    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self.app = Dash(__name__)
        self.fits_base_dir = os.path.join(dir_manager.DATA_DIR, 'fits')
        self.fits_data = {}  # Will store data for both directories
        self.classifier_models = {}  # Will store classifier models for both directories
        self.metrics_data = {}  # Will store metrics data for both directories
        
        # Initialize with default config
        self.update_config()
        
        # Get available directories
        self.available_dirs = self.get_available_directories()
        
        # Load fits data for default directories if available
        # Prefer dev and dev_backup/dev_shitty if both exist
        if 'dev' in self.available_dirs and 'dev_backup' in self.available_dirs:
            self.load_fits_data('dev', 'dir1')
            self.load_fits_data('dev_backup', 'dir2')
        elif 'dev' in self.available_dirs and 'dev_shitty' in self.available_dirs:
            self.load_fits_data('dev', 'dir1')
            self.load_fits_data('dev_shitty', 'dir2')
        elif len(self.available_dirs) >= 2:
            self.load_fits_data(self.available_dirs[0], 'dir1')
            self.load_fits_data(self.available_dirs[1], 'dir2')
        elif len(self.available_dirs) == 1:
            self.load_fits_data(self.available_dirs[0], 'dir1')
            self.load_fits_data(self.available_dirs[0], 'dir2')
        
        # Setup the dash app layout and callbacks
        self.setup_layout()
        self.setup_callbacks()
    
    def update_config(self, config_file: Optional[str] = None):
        """Update configuration"""
        _, config = get_config(config_file=f'{CONFIG_DIR}/{config_file}' if config_file else None)
        self.config = config
    
    def get_available_directories(self) -> List[str]:
        """Get list of available directories under fits/"""
        try:
            if not os.path.exists(self.fits_base_dir):
                logger.error(f"Fits base directory does not exist: {self.fits_base_dir}")
                return []
            
            # Get all subdirectories
            dirs = [d for d in os.listdir(self.fits_base_dir) 
                   if os.path.isdir(os.path.join(self.fits_base_dir, d))]
            
            logger.info(f"Found {len(dirs)} directories: {dirs}")
            return sorted(dirs)
        except Exception as e:
            logger.error(f"Error getting directories: {e}")
            return []
    
    def load_metrics_data(self, dir_name: str, label: str):
        """Load metrics data (.metrics files) from the specified directory"""
        try:
            fits_dir = os.path.join(self.fits_base_dir, dir_name)
            logger.info(f"Loading metrics data from {fits_dir} as {label}")
            
            # Check if the directory exists
            if not os.path.exists(fits_dir):
                logger.error(f"Fits directory does not exist: {fits_dir}")
                return
            
            # Find all metrics files recursively
            metrics_files = glob.glob(os.path.join(fits_dir, "**", "*.metrics"), recursive=True)
            
            if not metrics_files:
                logger.info(f"No metrics files found in {fits_dir}")
                return
            
            # Initialize storage for this label
            if label not in self.metrics_data:
                self.metrics_data[label] = {}
            
            for file_path in sorted(metrics_files):
                try:
                    # Extract horizon and model from filename
                    # Format: classifier.{horizon}.{model}.{date}.metrics
                    basename = os.path.basename(file_path)
                    parts = basename.split('.')
                    if len(parts) >= 5:
                        horizon = parts[1]
                        model = parts[2]
                        date = parts[3]
                        
                        key = f"{model}_{horizon}"
                        if key not in self.metrics_data[label]:
                            self.metrics_data[label][key] = []
                        
                        # Read metrics file
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                            metrics = {}
                            for line in lines:
                                if ':' in line:
                                    metric_name, value = line.strip().split(':')
                                    try:
                                        metrics[metric_name.strip()] = float(value.strip())
                                    except ValueError:
                                        logger.warning(f"Could not parse metric value: {line.strip()}")
                            
                            if metrics:
                                metrics['date'] = pd.to_datetime(date, format='%Y%m%d')
                                metrics['horizon'] = int(horizon)
                                metrics['model'] = model
                                metrics['source_dir'] = dir_name
                                self.metrics_data[label][key].append(metrics)
                        
                except Exception as e:
                    logger.warning(f"Failed to load metrics from {file_path}: {e}")
                    continue
            
            # Convert list of dicts to DataFrame for each model-horizon combination
            for key, value in self.metrics_data[label].items():
                if value:
                    self.metrics_data[label][key] = pd.DataFrame(value)
                    self.metrics_data[label][key].sort_values('date', inplace=True)
                    logger.info(f"Loaded {len(self.metrics_data[label][key])} metrics records for {label}/{key}")
                    
            logger.info(f"Loaded metrics data for {len(self.metrics_data[label])} model-horizon combinations in {label}")
                
        except Exception as e:
            logger.error(f"Error loading metrics data from {dir_name}: {e}")
    
    def load_fits_data(self, dir_name: str, label: str):
        """Load all fits CSV data from the specified directory"""
        try:
            fits_dir = os.path.join(self.fits_base_dir, dir_name)
            logger.info(f"Loading fits data from {fits_dir} as {label}")
            
            # Check if the directory exists
            if not os.path.exists(fits_dir):
                logger.error(f"Fits directory does not exist: {fits_dir}")
                return
            
            # Find all fits CSV files recursively
            # Try both naming patterns: fits.{dir_name}.*.csv and fits.dev.*.csv
            fits_files = glob.glob(os.path.join(fits_dir, "**", f"fits.{dir_name}.*.csv"), recursive=True)
            
            # If no files found with dir_name pattern, try with 'dev' pattern (for dev_backup case)
            if not fits_files:
                fits_files = glob.glob(os.path.join(fits_dir, "**", "fits.dev.*.csv"), recursive=True)
            
            if not fits_files:
                logger.error(f"No fits files found in {fits_dir}")
                return
            
            # Group files by horizon and model type
            if label not in self.fits_data:
                self.fits_data[label] = {}
            
            for file_path in sorted(fits_files):
                try:
                    # Extract horizon and model from filename
                    # Format: fits.{dir_name}.{horizon}.{model}.{date}.csv
                    basename = os.path.basename(file_path)
                    parts = basename.split('.')
                    if len(parts) >= 5:
                        horizon = parts[2]
                        model = parts[3]
                        date = parts[4]
                        
                        key = f"{model}_{horizon}"
                        if key not in self.fits_data[label]:
                            self.fits_data[label][key] = []
                        
                        # Load CSV data
                        df = pd.read_csv(file_path)
                        df['file_date'] = pd.to_datetime(date, format='%Y%m%d')
                        df['file_horizon'] = int(horizon)
                        df['file_model'] = model
                        df['source_dir'] = dir_name
                        
                        self.fits_data[label][key].append(df)
                        
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue
            
            # Concatenate data for each model-horizon combination
            for key, value in self.fits_data[label].items():
                if value:
                    self.fits_data[label][key] = pd.concat(value, ignore_index=True)
                    self.fits_data[label][key].sort_values('file_date', inplace=True)
                    # Rename file_date to date for consistency
                    self.fits_data[label][key].rename(columns={'file_date': 'date'}, inplace=True)
                    logger.info(f"Loaded {len(self.fits_data[label][key])} records for {label}/{key}")
                    
            logger.info(f"Loaded fits data for {len(self.fits_data[label])} model-horizon combinations in {label}")
                
        except Exception as e:
            logger.error(f"Error loading fits data from {dir_name}: {e}")
    
    def load_classifier_models(self, dir_name: str, label: str):
        """Load classifier models (SVM/Random Forest) from joblib files"""
        try:
            fits_dir = os.path.join(self.fits_base_dir, dir_name)
            logger.info(f"Loading classifier models from {fits_dir} as {label}")
            
            # Find all joblib files recursively
            joblib_files = glob.glob(os.path.join(fits_dir, "**", "classifier.*.joblib"), recursive=True)
            
            if not joblib_files:
                logger.info(f"No joblib model files found in {fits_dir}")
                return
            
            if label not in self.classifier_models:
                self.classifier_models[label] = {}
            
            for file_path in joblib_files:
                try:
                    # Extract model info from filename
                    # Format: classifier.{horizon}.{model}.{date}.joblib
                    basename = os.path.basename(file_path)
                    parts = basename.split('.')
                    if len(parts) >= 4:
                        horizon = parts[1]
                        model = parts[2]
                        date = parts[3]
                        
                        key = f"{model}_{horizon}"
                        
                        # Also get corresponding features file
                        features_file = file_path.replace('.joblib', '.features')
                        
                        if key not in self.classifier_models[label]:
                            self.classifier_models[label][key] = []
                        
                        model_data = {
                            'date': pd.to_datetime(date, format='%Y%m%d'),
                            'model_path': file_path,
                            'features_path': features_file if os.path.exists(features_file) else None,
                            'source_dir': dir_name
                        }
                        
                        self.classifier_models[label][key].append(model_data)
                        
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
                    continue
            
            # Sort models by date
            for key in self.classifier_models[label]:
                self.classifier_models[label][key] = sorted(
                    self.classifier_models[label][key], 
                    key=lambda x: x['date']
                )
                logger.info(f"Found {len(self.classifier_models[label][key])} models for {label}/{key}")
                
            logger.info(f"Found SVM models for {len(self.classifier_models[label])} model types in {label}")
                
        except Exception as e:
            logger.error(f"Error loading classifier models from {dir_name}: {e}")
    
    def get_available_models(self, label: str = None):
        """Get list of available models across both directories or for a specific label"""
        if label and label in self.fits_data:
            return sorted(list(self.fits_data[label].keys()))
        
        # Get union of models from both directories
        models = set()
        for data_dict in self.fits_data.values():
            models.update(data_dict.keys())
        return sorted(list(models))
    
    def get_available_horizons(self):
        """Get list of available horizons across all data"""
        horizons = set()
        for data_dict in self.fits_data.values():
            for key in data_dict.keys():
                # Extract horizon from model_horizon key
                parts = key.split('_')
                if len(parts) >= 2:
                    horizon = parts[-1]  # Last part is horizon
                    horizons.add(int(horizon))
        return sorted(list(horizons))
    
    def get_models_for_horizon(self, horizon):
        """Get list of models available for a specific horizon across all data"""
        models = set()
        horizon_str = str(horizon)
        for data_dict in self.fits_data.values():
            for key in data_dict.keys():
                if key.endswith(f'_{horizon_str}'):
                    # Extract model name (everything before the last underscore)
                    model = '_'.join(key.split('_')[:-1])
                    models.add(model)
        return sorted(list(models))
    
    def setup_layout(self):
        """Setup the Dash app layout"""
        # Defer layout generation to ensure data is loaded
        self.app.layout = self._generate_layout
    
    def _generate_layout(self):
        """Generate layout after data is loaded"""
        available_dirs = self.available_dirs
        available_horizons = self.get_available_horizons()
        
        return html.Div([
            html.H1("Fits Comparison Report", style={'textAlign': 'center'}),
            
            # Directory selectors
            html.Div([
                html.H3("Select Directories to Compare", style={'textAlign': 'center'}),
                html.Div([
                    html.Div([
                        html.Label("Directory 1:", style={'marginBottom': '10px', 'fontSize': '16px', 'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='dir1-selector',
                            options=[{'label': d, 'value': d} for d in available_dirs],
                            value='dev' if 'dev' in available_dirs else (available_dirs[0] if available_dirs else None),
                            placeholder="Select first directory",
                            style={'marginBottom': '20px'}
                        ),
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    html.Div([
                        html.Label("Directory 2:", style={'marginBottom': '10px', 'fontSize': '16px', 'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='dir2-selector',
                            options=[{'label': d, 'value': d} for d in available_dirs],
                            value='dev_backup' if 'dev_backup' in available_dirs else (available_dirs[1] if len(available_dirs) > 1 else (available_dirs[0] if available_dirs else None)),
                            placeholder="Select second directory",
                            style={'marginBottom': '20px'}
                        ),
                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                ], style={'marginBottom': '20px'}),
            ], style={'marginBottom': '30px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
            
            # Reload button
            html.Div([
                html.Button("Reload Data", id="reload-btn", n_clicks=0, 
                          style={'marginBottom': '10px', 'padding': '10px 20px', 'fontSize': '16px'}),
                html.Div(id="load-status", style={"marginTop": "10px", "color": "green"}),
            ], style={'marginBottom': '30px', 'textAlign': 'center'}),
            
            # Model selector
            html.Div([
                html.H3("Select Model to Compare", style={'textAlign': 'center'}),
                html.Div([
                    html.Div([
                        html.Label("Select Horizon:", style={'marginBottom': '10px', 'fontSize': '16px'}),
                        dcc.Dropdown(
                            id='horizon-selector',
                            options=[{'label': str(h), 'value': h} for h in available_horizons],
                            value=available_horizons[0] if available_horizons else None,
                            placeholder="Select a horizon",
                            style={'marginBottom': '20px'}
                        ),
                    ], style={'width': '32%', 'display': 'inline-block'}),
                    html.Div([
                        html.Label("Select Model:", style={'marginBottom': '10px', 'fontSize': '16px'}),
                        dcc.Dropdown(
                            id='model-selector',
                            placeholder="Select a model",
                            style={'marginBottom': '20px'}
                        ),
                    ], style={'width': '32%', 'display': 'inline-block', 'marginLeft': '2%'}),
                    html.Div([
                        html.Label("Select Lag:", style={'marginBottom': '10px', 'fontSize': '16px'}),
                        dcc.Dropdown(
                            id='lag-selector',
                            options=[
                                {'label': 'All Lags', 'value': 'all'},
                                {'label': 'Lag 0', 'value': 0},
                                {'label': 'Lag 1', 'value': 1},
                            ],
                            value='all',
                            placeholder="Select lag",
                            style={'marginBottom': '20px'}
                        ),
                    ], style={'width': '32%', 'float': 'right', 'display': 'inline-block'}),
                ], style={'marginBottom': '20px'}),
            ]),
            
            # Comparison mode selector
            html.Div([
                html.Label("Visualization Mode:", style={'marginBottom': '10px', 'fontSize': '16px', 'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='comparison-mode',
                    options=[
                        {'label': ' Overlay (Same Chart)', 'value': 'overlay'},
                        {'label': ' Side-by-Side', 'value': 'sidebyside'},
                        {'label': ' Difference', 'value': 'difference'}
                    ],
                    value='overlay',
                    inline=True,
                    style={'marginBottom': '20px'}
                ),
            ], style={'marginBottom': '30px', 'textAlign': 'center'}),
            
            # T-statistics comparison
            html.Div([
                html.H4("T-Statistics Comparison", style={'textAlign': 'center'}),
                dcc.Graph(id='tstat-comparison'),
            ], style={'marginBottom': '30px'}),
            
            # Coefficients comparison
            html.Div([
                html.H4("Coefficients Comparison", style={'textAlign': 'center'}),
                dcc.Graph(id='coeff-comparison'),
            ], style={'marginBottom': '30px'}),
            
            # Model quality metrics comparison
            html.Div([
                html.H4("Model Quality Metrics Comparison", style={'textAlign': 'center'}),
                dcc.Graph(id='quality-comparison'),
            ], style={'marginBottom': '30px'}),
            
            # Classifier performance metrics comparison
            html.Div([
                html.H4("Classifier Performance Metrics", style={'textAlign': 'center'}),
                dcc.Graph(id='classifier-metrics-comparison'),
            ], style={'marginBottom': '30px'}),
            
            # Summary comparison table
            html.Div([
                html.H4("Model Statistics Comparison", style={'textAlign': 'center'}),
                html.Div(id='comparison-tables-container'),
            ], style={'marginBottom': '30px'}),
            
            # Difference analysis
            html.Div([
                html.H4("Statistical Difference Analysis", style={'textAlign': 'center'}),
                html.Div(id='difference-analysis'),
            ], style={'marginBottom': '30px'}),
        ])
    
    def setup_callbacks(self):
        """Setup Dash callbacks"""
        
        @self.app.callback(
            Output("load-status", "children"),
            Input("reload-btn", "n_clicks"),
            State("dir1-selector", "value"),
            State("dir2-selector", "value"),
            prevent_initial_call=False,
        )
        def reload_data(n_clicks, dir1, dir2):
            """Reload fits data when button is clicked"""
            if dir1:
                self.load_fits_data(dir1, 'dir1')
                self.load_classifier_models(dir1, 'dir1')
                self.load_metrics_data(dir1, 'dir1')
            if dir2:
                self.load_fits_data(dir2, 'dir2')
                self.load_classifier_models(dir2, 'dir2')
                self.load_metrics_data(dir2, 'dir2')
            
            count1 = len(self.fits_data.get('dir1', {}))
            count2 = len(self.fits_data.get('dir2', {}))
            
            return f"Loaded data: {dir1} ({count1} models) vs {dir2} ({count2} models)"
        
        @self.app.callback(
            Output('model-selector', 'options'),
            Output('model-selector', 'value'),
            Input('horizon-selector', 'value'),
        )
        def update_model_options(selected_horizon):
            """Update available models for selected horizon"""
            if not selected_horizon:
                return [], None
            
            models = self.get_models_for_horizon(selected_horizon)
            options = [{'label': m, 'value': m} for m in models]
            
            # Select first model by default
            default_value = models[0] if models else None
            
            return options, default_value
        
        @self.app.callback(
            Output('tstat-comparison', 'figure'),
            Output('coeff-comparison', 'figure'),
            Output('quality-comparison', 'figure'),
            Output('classifier-metrics-comparison', 'figure'),
            Output('comparison-tables-container', 'children'),
            Output('difference-analysis', 'children'),
            Input('horizon-selector', 'value'),
            Input('model-selector', 'value'),
            Input('lag-selector', 'value'),
            Input('comparison-mode', 'value'),
            Input("dir1-selector", "value"),
            Input("dir2-selector", "value"),
            Input("reload-btn", "n_clicks"),
        )
        def update_comparison_displays(selected_horizon, selected_model, selected_lag, comparison_mode, 
                                      dir1, dir2, n_clicks):
            """Update comparison displays"""
            if not selected_horizon or not selected_model:
                return go.Figure(), go.Figure(), go.Figure(), go.Figure(), html.Div(), html.Div()
            
            # Construct the key from model and horizon
            key = f"{selected_model}_{selected_horizon}"
            
            # Get data for both directories
            df1 = self.fits_data.get('dir1', {}).get(key) if 'dir1' in self.fits_data else None
            df2 = self.fits_data.get('dir2', {}).get(key) if 'dir2' in self.fits_data else None
            
            if df1 is None and df2 is None:
                return go.Figure(), go.Figure(), go.Figure(), html.Div("No data available"), html.Div()
            
            # Filter by lag if specified
            if selected_lag != 'all' and selected_lag is not None:
                if df1 is not None and 'lag' in df1.columns:
                    df1 = df1[df1['lag'] == selected_lag].copy()
                if df2 is not None and 'lag' in df2.columns:
                    df2 = df2[df2['lag'] == selected_lag].copy()
            
            # Get metrics data for both directories
            metrics_df1 = self.metrics_data.get('dir1', {}).get(key) if 'dir1' in self.metrics_data else None
            metrics_df2 = self.metrics_data.get('dir2', {}).get(key) if 'dir2' in self.metrics_data else None
            
            # Create comparison figures based on mode
            if comparison_mode == 'overlay':
                tstat_fig = self._create_overlay_comparison(df1, df2, 'tstat', dir1, dir2, selected_lag)
                coeff_fig = self._create_overlay_comparison(df1, df2, 'coeff', dir1, dir2, selected_lag)
                quality_fig = self._create_quality_overlay(df1, df2, dir1, dir2)
                metrics_fig = self._create_metrics_overlay(metrics_df1, metrics_df2, dir1, dir2)
            elif comparison_mode == 'sidebyside':
                tstat_fig = self._create_sidebyside_comparison(df1, df2, 'tstat', dir1, dir2, selected_lag)
                coeff_fig = self._create_sidebyside_comparison(df1, df2, 'coeff', dir1, dir2, selected_lag)
                quality_fig = self._create_quality_sidebyside(df1, df2, dir1, dir2)
                metrics_fig = self._create_metrics_sidebyside(metrics_df1, metrics_df2, dir1, dir2)
            else:  # difference
                tstat_fig = self._create_difference_plot(df1, df2, 'tstat', dir1, dir2, selected_lag)
                coeff_fig = self._create_difference_plot(df1, df2, 'coeff', dir1, dir2, selected_lag)
                quality_fig = self._create_quality_difference(df1, df2, dir1, dir2)
                metrics_fig = self._create_metrics_difference(metrics_df1, metrics_df2, dir1, dir2)
            
            # Create comparison tables
            tables_container = self._create_comparison_tables(df1, df2, dir1, dir2)
            
            # Create difference analysis
            diff_analysis = self._create_difference_analysis(df1, df2, dir1, dir2)
            
            return tstat_fig, coeff_fig, quality_fig, metrics_fig, tables_container, diff_analysis
    
    def _create_overlay_comparison(self, df1, df2, metric, dir1_name, dir2_name, selected_lag='all'):
        """Create overlay comparison chart"""
        fig = go.Figure()
        
        colors1 = ['blue', 'lightblue', 'darkblue', 'royalblue']
        colors2 = ['red', 'lightcoral', 'darkred', 'indianred']
        
        # Add traces for first directory
        if df1 is not None and len(df1) > 0:
            if 'condition' in df1.columns:
                color_idx = 0
                for condition in df1['condition'].unique():
                    # If lag column exists and we're showing all lags, iterate through them
                    if 'lag' in df1.columns and selected_lag == 'all':
                        for lag in df1['lag'].unique():
                            mask = (df1['condition'] == condition) & (df1['lag'] == lag)
                            subset = df1[mask]
                            
                            if len(subset) > 0:
                                fig.add_trace(go.Scatter(
                                    x=subset['date'],
                                    y=subset[metric],
                                    mode='lines+markers',
                                    name=f"{dir1_name}: {condition} (lag {lag})",
                                    line={'width': 2, 'color': colors1[color_idx % len(colors1)]},
                                    marker={'size': 6}
                                ))
                                color_idx += 1
                    else:
                        # Single lag or no lag column
                        mask = df1['condition'] == condition
                        subset = df1[mask]
                        
                        if len(subset) > 0:
                            lag_text = f" (lag {selected_lag})" if selected_lag != 'all' else ""
                            fig.add_trace(go.Scatter(
                                x=subset['date'],
                                y=subset[metric],
                                mode='lines+markers',
                                name=f"{dir1_name}: {condition}{lag_text}",
                                line={'width': 2, 'color': colors1[color_idx % len(colors1)]},
                                marker={'size': 6}
                            ))
                            color_idx += 1
            else:
                fig.add_trace(go.Scatter(
                    x=df1['date'],
                    y=df1[metric],
                    mode='lines+markers',
                    name=f"{dir1_name}",
                    line={'width': 2, 'color': 'blue'},
                    marker={'size': 6}
                ))
        
        # Add traces for second directory
        if df2 is not None and len(df2) > 0:
            if 'condition' in df2.columns:
                color_idx = 0
                for condition in df2['condition'].unique():
                    # If lag column exists and we're showing all lags, iterate through them
                    if 'lag' in df2.columns and selected_lag == 'all':
                        for lag in df2['lag'].unique():
                            mask = (df2['condition'] == condition) & (df2['lag'] == lag)
                            subset = df2[mask]
                            
                            if len(subset) > 0:
                                fig.add_trace(go.Scatter(
                                    x=subset['date'],
                                    y=subset[metric],
                                    mode='lines+markers',
                                    name=f"{dir2_name}: {condition} (lag {lag})",
                                    line={'width': 2, 'color': colors2[color_idx % len(colors2)], 'dash': 'dash'},
                                    marker={'size': 6, 'symbol': 'diamond'}
                                ))
                                color_idx += 1
                    else:
                        # Single lag or no lag column
                        mask = df2['condition'] == condition
                        subset = df2[mask]
                        
                        if len(subset) > 0:
                            lag_text = f" (lag {selected_lag})" if selected_lag != 'all' else ""
                            fig.add_trace(go.Scatter(
                                x=subset['date'],
                                y=subset[metric],
                                mode='lines+markers',
                                name=f"{dir2_name}: {condition}{lag_text}",
                                line={'width': 2, 'color': colors2[color_idx % len(colors2)], 'dash': 'dash'},
                                marker={'size': 6, 'symbol': 'diamond'}
                            ))
                            color_idx += 1
            else:
                fig.add_trace(go.Scatter(
                    x=df2['date'],
                    y=df2[metric],
                    mode='lines+markers',
                    name=f"{dir2_name}",
                    line={'width': 2, 'color': 'red', 'dash': 'dash'},
                    marker={'size': 6, 'symbol': 'diamond'}
                ))
        
        # Add reference lines for t-statistic
        if metric == 'tstat':
            fig.add_hline(y=1.96, line_dash="dash", line_color="green", 
                         annotation_text="95% significance")
            fig.add_hline(y=-1.96, line_dash="dash", line_color="green")
        
        fig.add_hline(y=0, line_dash="solid", line_color="gray")
        
        metric_label = "T-Statistic" if metric == 'tstat' else "Coefficient"
        fig.update_layout(
            title=f"{metric_label} Comparison: {dir1_name} vs {dir2_name}",
            xaxis_title="Date",
            yaxis_title=metric_label,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def _create_sidebyside_comparison(self, df1, df2, metric, dir1_name, dir2_name, selected_lag='all'):
        """Create side-by-side comparison charts"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'{dir1_name}', f'{dir2_name}'),
            horizontal_spacing=0.12
        )
        
        colors = ['blue', 'lightblue', 'darkblue', 'royalblue']
        
        # Plot for first directory
        if df1 is not None and len(df1) > 0:
            if 'condition' in df1.columns:
                color_idx = 0
                for condition in df1['condition'].unique():
                    if 'lag' in df1.columns and selected_lag == 'all':
                        for lag in df1['lag'].unique():
                            mask = (df1['condition'] == condition) & (df1['lag'] == lag)
                            subset = df1[mask]
                            
                            if len(subset) > 0:
                                fig.add_trace(go.Scatter(
                                    x=subset['date'],
                                    y=subset[metric],
                                    mode='lines+markers',
                                    name=f"{condition} (lag {lag})",
                                    line={'width': 2, 'color': colors[color_idx % len(colors)]},
                                    showlegend=True
                                ), row=1, col=1)
                                color_idx += 1
                    else:
                        mask = df1['condition'] == condition
                        subset = df1[mask]
                        
                        if len(subset) > 0:
                            lag_text = f" (lag {selected_lag})" if selected_lag != 'all' else ""
                            fig.add_trace(go.Scatter(
                                x=subset['date'],
                                y=subset[metric],
                                mode='lines+markers',
                                name=f"{condition}{lag_text}",
                                line={'width': 2, 'color': colors[color_idx % len(colors)]},
                                showlegend=True
                            ), row=1, col=1)
                            color_idx += 1
            else:
                fig.add_trace(go.Scatter(
                    x=df1['date'],
                    y=df1[metric],
                    mode='lines+markers',
                    name=metric,
                    line={'width': 2}
                ), row=1, col=1)
        
        # Plot for second directory
        if df2 is not None and len(df2) > 0:
            if 'condition' in df2.columns:
                color_idx = 0
                for condition in df2['condition'].unique():
                    if 'lag' in df2.columns and selected_lag == 'all':
                        for lag in df2['lag'].unique():
                            mask = (df2['condition'] == condition) & (df2['lag'] == lag)
                            subset = df2[mask]
                            
                            if len(subset) > 0:
                                fig.add_trace(go.Scatter(
                                    x=subset['date'],
                                    y=subset[metric],
                                    mode='lines+markers',
                                    name=f"{condition} (lag {lag})",
                                    line={'width': 2, 'color': colors[color_idx % len(colors)]},
                                    showlegend=False
                                ), row=1, col=2)
                                color_idx += 1
                    else:
                        mask = df2['condition'] == condition
                        subset = df2[mask]
                        
                        if len(subset) > 0:
                            lag_text = f" (lag {selected_lag})" if selected_lag != 'all' else ""
                            fig.add_trace(go.Scatter(
                                x=subset['date'],
                                y=subset[metric],
                                mode='lines+markers',
                                name=f"{condition}{lag_text}",
                                line={'width': 2, 'color': colors[color_idx % len(colors)]},
                                showlegend=False
                            ), row=1, col=2)
                            color_idx += 1
            else:
                fig.add_trace(go.Scatter(
                    x=df2['date'],
                    y=df2[metric],
                    mode='lines+markers',
                    name=metric,
                    line={'width': 2},
                    showlegend=False
                ), row=1, col=2)
        
        # Add reference lines
        if metric == 'tstat':
            for col in [1, 2]:
                fig.add_hline(y=1.96, line_dash="dash", line_color="green", row=1, col=col)
                fig.add_hline(y=-1.96, line_dash="dash", line_color="green", row=1, col=col)
        
        for col in [1, 2]:
            fig.add_hline(y=0, line_dash="solid", line_color="gray", row=1, col=col)
        
        metric_label = "T-Statistic" if metric == 'tstat' else "Coefficient"
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text=metric_label, row=1, col=1)
        
        fig.update_layout(
            title=f"{metric_label} Side-by-Side Comparison",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_difference_plot(self, df1, df2, metric, dir1_name, dir2_name, selected_lag='all'):
        """Create difference plot between two datasets"""
        fig = go.Figure()
        
        if df1 is None or df2 is None or len(df1) == 0 or len(df2) == 0:
            fig.add_annotation(
                text="Cannot compute difference - data missing from one or both directories",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Merge dataframes on date and calculate differences
        if 'condition' in df1.columns and 'lag' in df1.columns:
            # Complex case with conditions and lags
            for condition in set(df1['condition'].unique()) & set(df2['condition'].unique()):
                for lag in set(df1['lag'].unique()) & set(df2['lag'].unique()):
                    mask1 = (df1['condition'] == condition) & (df1['lag'] == lag)
                    mask2 = (df2['condition'] == condition) & (df2['lag'] == lag)
                    
                    subset1 = df1[mask1][['date', metric]].set_index('date')
                    subset2 = df2[mask2][['date', metric]].set_index('date')
                    
                    # Align data and calculate difference
                    aligned = pd.concat([subset1, subset2], axis=1, keys=['dir1', 'dir2'], join='inner')
                    if len(aligned) > 0:
                        diff = aligned['dir2'][metric] - aligned['dir1'][metric]
                        
                        fig.add_trace(go.Scatter(
                            x=diff.index,
                            y=diff.values,
                            mode='lines+markers',
                            name=f"{condition} (lag {lag})",
                            line={'width': 2}
                        ))
        else:
            # Simple case
            df1_indexed = df1[['date', metric]].set_index('date')
            df2_indexed = df2[['date', metric]].set_index('date')
            
            aligned = pd.concat([df1_indexed, df2_indexed], axis=1, keys=['dir1', 'dir2'], join='inner')
            if len(aligned) > 0:
                diff = aligned['dir2'][metric] - aligned['dir1'][metric]
                
                fig.add_trace(go.Scatter(
                    x=diff.index,
                    y=diff.values,
                    mode='lines+markers',
                    name='Difference',
                    line={'width': 2}
                ))
                
                # Add confidence bands
                mean_diff = diff.mean()
                std_diff = diff.std()
                
                fig.add_hline(y=mean_diff, line_dash="dash", line_color="blue", 
                             annotation_text=f"Mean: {mean_diff:.4f}")
                fig.add_hline(y=mean_diff + 2*std_diff, line_dash="dot", line_color="red",
                             annotation_text=f"+2σ: {mean_diff + 2*std_diff:.4f}")
                fig.add_hline(y=mean_diff - 2*std_diff, line_dash="dot", line_color="red",
                             annotation_text=f"-2σ: {mean_diff - 2*std_diff:.4f}")
        
        fig.add_hline(y=0, line_dash="solid", line_color="gray")
        
        metric_label = "T-Statistic" if metric == 'tstat' else "Coefficient"
        fig.update_layout(
            title=f"{metric_label} Difference: {dir2_name} - {dir1_name}",
            xaxis_title="Date",
            yaxis_title=f"{metric_label} Difference",
            hovermode='x unified'
        )
        
        return fig
    
    def _create_quality_overlay(self, df1, df2, dir1_name, dir2_name):
        """Create overlay comparison for quality metrics"""
        # Check if we have condition column to create a more detailed view
        has_conditions = (df1 is not None and 'condition' in df1.columns) or \
                        (df2 is not None and 'condition' in df2.columns)
        
        if has_conditions:
            # Create 3 rows: Total, Mom, Rev
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Number of Observations (Average)', 
                              'Number of Observations (Momentum)',
                              'Number of Observations (Reversal)',
                              'Impurity'),
                vertical_spacing=0.12,
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}]]
            )
        else:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Number of Observations', 'Impurity'),
                vertical_spacing=0.15
            )
        
        # Debug logging
        if df1 is not None and len(df1) > 0:
            logger.info(f"Quality metrics for {dir1_name}:")
            logger.info(f"  Total rows: {len(df1)}")
            logger.info(f"  Unique dates: {df1['date'].nunique()}")
            logger.info(f"  nobs range: {df1['nobs'].min()} to {df1['nobs'].max()}")
            logger.info(f"  nobs mean: {df1['nobs'].mean():.0f}")
            
            # Check what happens with drop_duplicates
            df1_unique_test = df1.drop_duplicates(subset=['date'])
            logger.info(f"  After drop_duplicates: {len(df1_unique_test)} rows")
            logger.info(f"  nobs after dedup: {df1_unique_test['nobs'].tolist()[:5]}")
        
        if df2 is not None and len(df2) > 0:
            logger.info(f"Quality metrics for {dir2_name}:")
            logger.info(f"  Total rows: {len(df2)}")
            logger.info(f"  Unique dates: {df2['date'].nunique()}")
            logger.info(f"  nobs range: {df2['nobs'].min()} to {df2['nobs'].max()}")
            logger.info(f"  nobs mean: {df2['nobs'].mean():.0f}")
            
            df2_unique_test = df2.drop_duplicates(subset=['date'])
            logger.info(f"  After drop_duplicates: {len(df2_unique_test)} rows")
            logger.info(f"  nobs after dedup: {df2_unique_test['nobs'].tolist()[:5]}")
        
        # Number of observations
        if df1 is not None and len(df1) > 0:
            if has_conditions and 'condition' in df1.columns:
                # Aggregate by date for overall average
                df1_agg = df1.groupby('date').agg({'nobs': 'mean', 'impurity': 'mean'}).reset_index()
                logger.info(f"  {dir1_name} aggregated nobs: {df1_agg['nobs'].tolist()[:5]}")
                
                # Also aggregate by date and condition
                df1_mom = df1[df1['condition'] == 'mom'].groupby('date').agg({'nobs': 'mean'}).reset_index()
                df1_rev = df1[df1['condition'] == 'rev'].groupby('date').agg({'nobs': 'mean'}).reset_index()
                
                logger.info(f"  {dir1_name} mom nobs: {df1_mom['nobs'].tolist()[:5] if len(df1_mom) > 0 else 'None'}")
                logger.info(f"  {dir1_name} rev nobs: {df1_rev['nobs'].tolist()[:5] if len(df1_rev) > 0 else 'None'}")
                
                # Plot average
                fig.add_trace(
                    go.Scatter(
                        x=df1_agg['date'],
                        y=df1_agg['nobs'],
                        mode='lines+markers',
                        name=f"{dir1_name}",
                        line={'width': 2, 'color': 'blue'},
                        marker={'size': 4}
                    ),
                    row=1, col=1
                )
                
                # Plot momentum
                if len(df1_mom) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=df1_mom['date'],
                            y=df1_mom['nobs'],
                            mode='lines+markers',
                            name=f"{dir1_name} (mom)",
                            line={'width': 2, 'color': 'blue'},
                            marker={'size': 4}
                        ),
                        row=2, col=1
                    )
                
                # Plot reversal
                if len(df1_rev) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=df1_rev['date'],
                            y=df1_rev['nobs'],
                            mode='lines+markers',
                            name=f"{dir1_name} (rev)",
                            line={'width': 2, 'color': 'blue'},
                            marker={'size': 4}
                        ),
                        row=3, col=1
                    )
                
                # Plot impurity
                fig.add_trace(
                    go.Scatter(
                        x=df1_agg['date'],
                        y=df1_agg['impurity'],
                        mode='lines+markers',
                        name=f"{dir1_name}",
                        line={'width': 2, 'color': 'blue'},
                        marker={'size': 4},
                        showlegend=False
                    ),
                    row=4, col=1
                )
            else:
                # Original logic for no conditions
                df1_agg = df1.groupby('date').agg({'nobs': 'mean', 'impurity': 'mean'}).reset_index()
                logger.info(f"  {dir1_name} aggregated nobs: {df1_agg['nobs'].tolist()[:5]}")
                
                fig.add_trace(
                    go.Scatter(
                        x=df1_agg['date'],
                        y=df1_agg['nobs'],
                        mode='lines+markers',
                        name=f"{dir1_name}",
                        line={'width': 2, 'color': 'blue'}
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df1_agg['date'],
                        y=df1_agg['impurity'],
                        mode='lines+markers',
                        name=f"{dir1_name}",
                        line={'width': 2, 'color': 'blue'},
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        if df2 is not None and len(df2) > 0:
            if has_conditions and 'condition' in df2.columns:
                # Aggregate by date for overall average
                df2_agg = df2.groupby('date').agg({'nobs': 'mean', 'impurity': 'mean'}).reset_index()
                logger.info(f"  {dir2_name} aggregated nobs: {df2_agg['nobs'].tolist()[:5]}")
                
                # Also aggregate by date and condition
                df2_mom = df2[df2['condition'] == 'mom'].groupby('date').agg({'nobs': 'mean'}).reset_index()
                df2_rev = df2[df2['condition'] == 'rev'].groupby('date').agg({'nobs': 'mean'}).reset_index()
                
                logger.info(f"  {dir2_name} mom nobs: {df2_mom['nobs'].tolist()[:5] if len(df2_mom) > 0 else 'None'}")
                logger.info(f"  {dir2_name} rev nobs: {df2_rev['nobs'].tolist()[:5] if len(df2_rev) > 0 else 'None'}")
                
                # Plot average
                fig.add_trace(
                    go.Scatter(
                        x=df2_agg['date'],
                        y=df2_agg['nobs'],
                        mode='lines+markers',
                        name=f"{dir2_name}",
                        line={'width': 2, 'color': 'red', 'dash': 'dash'},
                        marker={'size': 4, 'symbol': 'diamond'}
                    ),
                    row=1, col=1
                )
                
                # Plot momentum
                if len(df2_mom) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=df2_mom['date'],
                            y=df2_mom['nobs'],
                            mode='lines+markers',
                            name=f"{dir2_name} (mom)",
                            line={'width': 2, 'color': 'red', 'dash': 'dash'},
                            marker={'size': 4, 'symbol': 'diamond'}
                        ),
                        row=2, col=1
                    )
                
                # Plot reversal
                if len(df2_rev) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=df2_rev['date'],
                            y=df2_rev['nobs'],
                            mode='lines+markers',
                            name=f"{dir2_name} (rev)",
                            line={'width': 2, 'color': 'red', 'dash': 'dash'},
                            marker={'size': 4, 'symbol': 'diamond'}
                        ),
                        row=3, col=1
                    )
                
                # Plot impurity
                fig.add_trace(
                    go.Scatter(
                        x=df2_agg['date'],
                        y=df2_agg['impurity'],
                        mode='lines+markers',
                        name=f"{dir2_name}",
                        line={'width': 2, 'color': 'red', 'dash': 'dash'},
                        marker={'size': 4, 'symbol': 'diamond'},
                        showlegend=False
                    ),
                    row=4, col=1
                )
            else:
                # Original logic for no conditions
                df2_agg = df2.groupby('date').agg({'nobs': 'mean', 'impurity': 'mean'}).reset_index()
                logger.info(f"  {dir2_name} aggregated nobs: {df2_agg['nobs'].tolist()[:5]}")
                
                fig.add_trace(
                    go.Scatter(
                        x=df2_agg['date'],
                        y=df2_agg['nobs'],
                        mode='lines+markers',
                        name=f"{dir2_name}",
                        line={'width': 2, 'color': 'red', 'dash': 'dash'}
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df2_agg['date'],
                        y=df2_agg['impurity'],
                        mode='lines+markers',
                        name=f"{dir2_name}",
                        line={'width': 2, 'color': 'red', 'dash': 'dash'},
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        if has_conditions:
            fig.update_xaxes(title_text="Date", row=4, col=1)
            fig.update_yaxes(title_text="Count (Avg)", row=1, col=1)
            fig.update_yaxes(title_text="Count (Mom)", row=2, col=1)
            fig.update_yaxes(title_text="Count (Rev)", row=3, col=1)
            fig.update_yaxes(title_text="Impurity", row=4, col=1)
            fig.update_layout(height=800, hovermode='x unified')
        else:
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_yaxes(title_text="Impurity", row=2, col=1)
            fig.update_layout(height=600, hovermode='x unified')
        
        return fig
    
    def _create_quality_sidebyside(self, df1, df2, dir1_name, dir2_name):
        """Create side-by-side quality metrics comparison"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{dir1_name} - Observations', f'{dir2_name} - Observations',
                f'{dir1_name} - Impurity', f'{dir2_name} - Impurity'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        if df1 is not None and len(df1) > 0:
            df1_unique = df1.drop_duplicates(subset=['date'])
            fig.add_trace(
                go.Scatter(
                    x=df1_unique['date'],
                    y=df1_unique['nobs'],
                    mode='lines+markers',
                    name='Observations',
                    line={'width': 2}
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df1_unique['date'],
                    y=df1_unique['impurity'],
                    mode='lines+markers',
                    name='Impurity',
                    line={'width': 2}
                ),
                row=2, col=1
            )
        
        if df2 is not None and len(df2) > 0:
            df2_unique = df2.drop_duplicates(subset=['date'])
            fig.add_trace(
                go.Scatter(
                    x=df2_unique['date'],
                    y=df2_unique['nobs'],
                    mode='lines+markers',
                    name='Observations',
                    line={'width': 2},
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df2_unique['date'],
                    y=df2_unique['impurity'],
                    mode='lines+markers',
                    name='Impurity',
                    line={'width': 2},
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Impurity", row=2, col=1)
        fig.update_yaxes(title_text="Impurity", row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, hovermode='x unified')
        
        return fig
    
    def _create_quality_difference(self, df1, df2, dir1_name, dir2_name):
        """Create difference plot for quality metrics"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Observations Difference', 'Impurity Difference'),
            vertical_spacing=0.15
        )
        
        if df1 is not None and df2 is not None and len(df1) > 0 and len(df2) > 0:
            df1_unique = df1.drop_duplicates(subset=['date']).set_index('date')
            df2_unique = df2.drop_duplicates(subset=['date']).set_index('date')
            
            # Align data
            aligned = pd.concat([df1_unique[['nobs', 'impurity']], 
                                df2_unique[['nobs', 'impurity']]], 
                                axis=1, keys=['dir1', 'dir2'], join='inner')
            
            if len(aligned) > 0:
                nobs_diff = aligned['dir2']['nobs'] - aligned['dir1']['nobs']
                impurity_diff = aligned['dir2']['impurity'] - aligned['dir1']['impurity']
                
                fig.add_trace(
                    go.Scatter(
                        x=nobs_diff.index,
                        y=nobs_diff.values,
                        mode='lines+markers',
                        name='Observations Diff',
                        line={'width': 2}
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=impurity_diff.index,
                        y=impurity_diff.values,
                        mode='lines+markers',
                        name='Impurity Diff',
                        line={'width': 2}
                    ),
                    row=2, col=1
                )
                
                # Add zero lines
                fig.add_hline(y=0, line_dash="solid", line_color="gray", row=1, col=1)
                fig.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1)
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Count Difference", row=1, col=1)
        fig.update_yaxes(title_text="Impurity Difference", row=2, col=1)
        
        fig.update_layout(
            height=600,
            title=f"Quality Metrics Difference: {dir2_name} - {dir1_name}",
            showlegend=False,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_comparison_tables(self, df1, df2, dir1_name, dir2_name):
        """Create comparison tables for latest statistics"""
        tables = []
        
        # Get latest data for both directories
        latest1 = df1[df1['date'] == df1['date'].max()] if df1 is not None and len(df1) > 0 else None
        latest2 = df2[df2['date'] == df2['date'].max()] if df2 is not None and len(df2) > 0 else None
        
        # Also get metrics data if available
        key = None
        if latest1 is not None and 'file_model' in latest1.columns and 'file_horizon' in latest1.columns:
            model = latest1['file_model'].iloc[0]
            horizon = latest1['file_horizon'].iloc[0]
            key = f"{model}_{horizon}"
        elif latest2 is not None and 'file_model' in latest2.columns and 'file_horizon' in latest2.columns:
            model = latest2['file_model'].iloc[0]
            horizon = latest2['file_horizon'].iloc[0]
            key = f"{model}_{horizon}"
        
        metrics1 = None
        metrics2 = None
        if key:
            metrics1 = self.metrics_data.get('dir1', {}).get(key) if 'dir1' in self.metrics_data else None
            metrics2 = self.metrics_data.get('dir2', {}).get(key) if 'dir2' in self.metrics_data else None
            
            if metrics1 is not None and len(metrics1) > 0:
                metrics1 = metrics1[metrics1['date'] == metrics1['date'].max()]
            if metrics2 is not None and len(metrics2) > 0:
                metrics2 = metrics2[metrics2['date'] == metrics2['date'].max()]
        
        metrics = ['tstat', 'coeff', 'stderr', 'nobs']
        
        # Check if we have condition columns
        has_conditions = (latest1 is not None and 'condition' in latest1.columns) or \
                        (latest2 is not None and 'condition' in latest2.columns)
        
        # Create table for dir1
        if latest1 is not None:
            table1_data = []
            for metric in metrics:
                if metric in latest1.columns:
                    values = latest1[metric]
                    row = {
                        'metric': metric, 
                        'overall': values.mean(),
                    }
                    
                    # Add breakdown by condition if available
                    if has_conditions and 'condition' in latest1.columns:
                        mom_values = latest1[latest1['condition'] == 'mom'][metric] if 'mom' in latest1['condition'].values else pd.Series()
                        rev_values = latest1[latest1['condition'] == 'rev'][metric] if 'rev' in latest1['condition'].values else pd.Series()
                        row['mom'] = mom_values.mean() if len(mom_values) > 0 else None
                        row['rev'] = rev_values.mean() if len(rev_values) > 0 else None
                    else:
                        row['std'] = values.std() if len(values) > 1 else 0
                        row['count'] = len(values)
                    
                    table1_data.append(row)
            
            # Add classifier metrics if available
            if metrics1 is not None and len(metrics1) > 0:
                if 'insample' in metrics1.columns:
                    table1_data.append({
                        'metric': 'classifier_insample',
                        'overall': metrics1['insample'].iloc[0] if len(metrics1) > 0 else None
                    })
                if 'outsample' in metrics1.columns:
                    table1_data.append({
                        'metric': 'classifier_outsample',
                        'overall': metrics1['outsample'].iloc[0] if len(metrics1) > 0 else None
                    })
            
            # Add note about what date this is from
            date_str = latest1['date'].iloc[0].strftime('%Y-%m-%d') if pd.notna(latest1['date'].iloc[0]) else 'Unknown'
            
            # Choose columns based on whether we have conditions
            if has_conditions and 'condition' in latest1.columns:
                columns = [
                    {'name': 'Metric', 'id': 'metric', 'type': 'text'},
                    {'name': 'Overall', 'id': 'overall', 'type': 'text'},  # Use text type for mixed formatting
                    {'name': 'Momentum', 'id': 'mom', 'type': 'text'},
                    {'name': 'Reversal', 'id': 'rev', 'type': 'text'},
                ]
                note_text = f"Date: {date_str} | Values are averages across lags"
            else:
                columns = [
                    {'name': 'Metric', 'id': 'metric', 'type': 'text'},
                    {'name': 'Mean', 'id': 'overall', 'type': 'text'},  # Use text for mixed formatting
                    {'name': 'Std Dev', 'id': 'std', 'type': 'text'},
                    {'name': 'N', 'id': 'count', 'type': 'numeric'},
                ]
                note_text = f"Date: {date_str} | Values are averages across conditions/lags"
            
            # Format values appropriately based on metric type
            for row in table1_data:
                if row['metric'] == 'nobs':
                    # Format nobs as integers
                    for key in ['overall', 'mom', 'rev', 'std']:
                        if key in row and row[key] is not None:
                            row[key] = f"{int(round(row[key]))}"
                    if 'count' in row:
                        row['count'] = int(row['count'])
                elif row['metric'] in ['tstat', 'coeff', 'stderr']:
                    # Format statistics with 4 decimal places
                    for key in ['overall', 'mom', 'rev', 'std']:
                        if key in row and row[key] is not None:
                            row[key] = f"{row[key]:.4f}"
                    if 'count' in row:
                        row['count'] = int(row['count'])
                elif row['metric'] in ['classifier_insample', 'classifier_outsample']:
                    # Format classifier metrics with 4 decimal places
                    if 'overall' in row and row['overall'] is not None:
                        row['overall'] = f"{row['overall']:.4f}"
            
            tables.append(html.Div([
                html.H5(f"{dir1_name} - Latest Statistics", style={'textAlign': 'center'}),
                html.P(note_text, style={'textAlign': 'center', 'fontSize': '12px', 'color': 'gray'}),
                DataTable(
                    columns=columns,
                    data=table1_data,
                    style_cell={'textAlign': 'center'},
                )
            ], style={'width': '48%', 'display': 'inline-block'}))
        
        # Create table for dir2
        if latest2 is not None:
            table2_data = []
            for metric in metrics:
                if metric in latest2.columns:
                    values = latest2[metric]
                    row = {
                        'metric': metric, 
                        'overall': values.mean(),
                    }
                    
                    # Add breakdown by condition if available
                    if has_conditions and 'condition' in latest2.columns:
                        mom_values = latest2[latest2['condition'] == 'mom'][metric] if 'mom' in latest2['condition'].values else pd.Series()
                        rev_values = latest2[latest2['condition'] == 'rev'][metric] if 'rev' in latest2['condition'].values else pd.Series()
                        row['mom'] = mom_values.mean() if len(mom_values) > 0 else None
                        row['rev'] = rev_values.mean() if len(rev_values) > 0 else None
                    else:
                        row['std'] = values.std() if len(values) > 1 else 0
                        row['count'] = len(values)
                    
                    table2_data.append(row)
            
            # Add classifier metrics if available
            if metrics2 is not None and len(metrics2) > 0:
                if 'insample' in metrics2.columns:
                    table2_data.append({
                        'metric': 'classifier_insample',
                        'overall': metrics2['insample'].iloc[0] if len(metrics2) > 0 else None
                    })
                if 'outsample' in metrics2.columns:
                    table2_data.append({
                        'metric': 'classifier_outsample',
                        'overall': metrics2['outsample'].iloc[0] if len(metrics2) > 0 else None
                    })
            
            # Add note about what date this is from
            date_str = latest2['date'].iloc[0].strftime('%Y-%m-%d') if pd.notna(latest2['date'].iloc[0]) else 'Unknown'
            
            # Choose columns based on whether we have conditions
            if has_conditions and 'condition' in latest2.columns:
                columns = [
                    {'name': 'Metric', 'id': 'metric', 'type': 'text'},
                    {'name': 'Overall', 'id': 'overall', 'type': 'text'},  # Use text type for mixed formatting
                    {'name': 'Momentum', 'id': 'mom', 'type': 'text'},
                    {'name': 'Reversal', 'id': 'rev', 'type': 'text'},
                ]
                note_text = f"Date: {date_str} | Values are averages across lags"
            else:
                columns = [
                    {'name': 'Metric', 'id': 'metric', 'type': 'text'},
                    {'name': 'Mean', 'id': 'overall', 'type': 'text'},  # Use text for mixed formatting
                    {'name': 'Std Dev', 'id': 'std', 'type': 'text'},
                    {'name': 'N', 'id': 'count', 'type': 'numeric'},
                ]
                note_text = f"Date: {date_str} | Values are averages across conditions/lags"
            
            # Format values appropriately based on metric type
            for row in table2_data:
                if row['metric'] == 'nobs':
                    # Format nobs as integers
                    for key in ['overall', 'mom', 'rev', 'std']:
                        if key in row and row[key] is not None:
                            row[key] = f"{int(round(row[key]))}"
                    if 'count' in row:
                        row['count'] = int(row['count'])
                elif row['metric'] in ['tstat', 'coeff', 'stderr']:
                    # Format statistics with 4 decimal places
                    for key in ['overall', 'mom', 'rev', 'std']:
                        if key in row and row[key] is not None:
                            row[key] = f"{row[key]:.4f}"
                    if 'count' in row:
                        row['count'] = int(row['count'])
                elif row['metric'] in ['classifier_insample', 'classifier_outsample']:
                    # Format classifier metrics with 4 decimal places
                    if 'overall' in row and row['overall'] is not None:
                        row['overall'] = f"{row['overall']:.4f}"
            
            tables.append(html.Div([
                html.H5(f"{dir2_name} - Latest Statistics", style={'textAlign': 'center'}),
                html.P(note_text, style={'textAlign': 'center', 'fontSize': '12px', 'color': 'gray'}),
                DataTable(
                    columns=columns,
                    data=table2_data,
                    style_cell={'textAlign': 'center'},
                )
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}))
        
        return html.Div(tables)
    
    def _create_difference_analysis(self, df1, df2, dir1_name, dir2_name):
        """Create statistical difference analysis"""
        if df1 is None or df2 is None or len(df1) == 0 or len(df2) == 0:
            return html.Div("Insufficient data for difference analysis")
        
        analysis_content = []
        
        # Align data on common dates
        df1_indexed = df1.set_index('date')
        df2_indexed = df2.set_index('date')
        common_dates = sorted(set(df1_indexed.index) & set(df2_indexed.index))
        
        if len(common_dates) == 0:
            return html.Div("No overlapping dates for comparison")
        
        # Calculate statistics for common dates
        metrics_to_analyze = ['tstat', 'coeff', 'nobs', 'impurity']
        
        stats_data = []
        for metric in metrics_to_analyze:
            if metric in df1.columns and metric in df2.columns:
                values1 = df1_indexed.loc[common_dates, metric]
                values2 = df2_indexed.loc[common_dates, metric]
                
                diff = values2 - values1
                
                # Calculate paired t-test
                mean_diff = diff.mean()
                std_diff = diff.std()
                n = len(diff)
                t_stat = mean_diff / (std_diff / np.sqrt(n)) if std_diff > 0 else 0
                
                # Correlation
                corr = values1.corr(values2)
                
                stats_data.append({
                    'metric': metric,
                    'mean_diff': mean_diff,
                    'std_diff': std_diff,
                    't_statistic': t_stat,
                    'correlation': corr,
                    'n_samples': n
                })
        
        # Create statistics table
        analysis_content.append(
            DataTable(
                columns=[
                    {'name': 'Metric', 'id': 'metric', 'type': 'text'},
                    {'name': 'Mean Diff', 'id': 'mean_diff', 'type': 'numeric', 'format': Format(precision=4)},
                    {'name': 'Std Diff', 'id': 'std_diff', 'type': 'numeric', 'format': Format(precision=4)},
                    {'name': 'T-Stat', 'id': 't_statistic', 'type': 'numeric', 'format': Format(precision=3)},
                    {'name': 'Correlation', 'id': 'correlation', 'type': 'numeric', 'format': Format(precision=3)},
                    {'name': 'N', 'id': 'n_samples', 'type': 'numeric'},
                ],
                data=stats_data,
                style_cell={'textAlign': 'center'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{t_statistic} > 1.96 || {t_statistic} < -1.96'},
                        'backgroundColor': 'lightgreen',
                    }
                ]
            )
        )
        
        # Add interpretation
        significant_diffs = [s for s in stats_data if abs(s['t_statistic']) > 1.96]
        if significant_diffs:
            interpretation = html.Div([
                html.Hr(),
                html.P(f"Significant differences (|t| > 1.96) found in: {', '.join([s['metric'] for s in significant_diffs])}",
                      style={'fontWeight': 'bold', 'color': 'green'}),
                html.P(f"Based on {stats_data[0]['n_samples']} overlapping dates between {dir1_name} and {dir2_name}")
            ])
        else:
            interpretation = html.Div([
                html.Hr(),
                html.P("No statistically significant differences found between the two models.",
                      style={'color': 'gray'}),
                html.P(f"Based on {stats_data[0]['n_samples']} overlapping dates between {dir1_name} and {dir2_name}")
            ])
        
        analysis_content.append(interpretation)
        
        return html.Div(analysis_content)
    
    def _create_metrics_overlay(self, metrics_df1, metrics_df2, dir1_name, dir2_name):
        """Create overlay comparison for classifier performance metrics"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('In-Sample Performance', 'Out-of-Sample Performance'),
            vertical_spacing=0.15
        )
        
        # Plot in-sample metrics
        if metrics_df1 is not None and len(metrics_df1) > 0 and 'insample' in metrics_df1.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df1['date'],
                    y=metrics_df1['insample'],
                    mode='lines+markers',
                    name=f"{dir1_name}",
                    line={'width': 2, 'color': 'blue'},
                    marker={'size': 6}
                ),
                row=1, col=1
            )
        
        if metrics_df2 is not None and len(metrics_df2) > 0 and 'insample' in metrics_df2.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df2['date'],
                    y=metrics_df2['insample'],
                    mode='lines+markers',
                    name=f"{dir2_name}",
                    line={'width': 2, 'color': 'red', 'dash': 'dash'},
                    marker={'size': 6, 'symbol': 'diamond'}
                ),
                row=1, col=1
            )
        
        # Plot out-of-sample metrics
        if metrics_df1 is not None and len(metrics_df1) > 0 and 'outsample' in metrics_df1.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df1['date'],
                    y=metrics_df1['outsample'],
                    mode='lines+markers',
                    name=f"{dir1_name}",
                    line={'width': 2, 'color': 'blue'},
                    marker={'size': 6},
                    showlegend=False
                ),
                row=2, col=1
            )
        
        if metrics_df2 is not None and len(metrics_df2) > 0 and 'outsample' in metrics_df2.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df2['date'],
                    y=metrics_df2['outsample'],
                    mode='lines+markers',
                    name=f"{dir2_name}",
                    line={'width': 2, 'color': 'red', 'dash': 'dash'},
                    marker={'size': 6, 'symbol': 'diamond'},
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Add reference line at 0.5 (random chance for binary classification)
        for row in [1, 2]:
            fig.add_hline(y=0.5, line_dash="dot", line_color="gray", 
                         annotation_text="Random (0.5)", row=row, col=1)
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)
        
        # Set y-axis range for better comparison
        fig.update_yaxes(range=[0.45, 0.70], row=1, col=1)
        fig.update_yaxes(range=[0.45, 0.70], row=2, col=1)
        
        fig.update_layout(
            title=f"Classifier Performance: {dir1_name} vs {dir2_name}",
            height=600,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def _create_metrics_sidebyside(self, metrics_df1, metrics_df2, dir1_name, dir2_name):
        """Create side-by-side comparison for classifier performance metrics"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{dir1_name} - In-Sample', f'{dir2_name} - In-Sample',
                f'{dir1_name} - Out-of-Sample', f'{dir2_name} - Out-of-Sample'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # Plot for first directory
        if metrics_df1 is not None and len(metrics_df1) > 0:
            if 'insample' in metrics_df1.columns:
                fig.add_trace(
                    go.Scatter(
                        x=metrics_df1['date'],
                        y=metrics_df1['insample'],
                        mode='lines+markers',
                        name='In-Sample',
                        line={'width': 2, 'color': 'blue'}
                    ),
                    row=1, col=1
                )
            
            if 'outsample' in metrics_df1.columns:
                fig.add_trace(
                    go.Scatter(
                        x=metrics_df1['date'],
                        y=metrics_df1['outsample'],
                        mode='lines+markers',
                        name='Out-of-Sample',
                        line={'width': 2, 'color': 'green'}
                    ),
                    row=2, col=1
                )
        
        # Plot for second directory
        if metrics_df2 is not None and len(metrics_df2) > 0:
            if 'insample' in metrics_df2.columns:
                fig.add_trace(
                    go.Scatter(
                        x=metrics_df2['date'],
                        y=metrics_df2['insample'],
                        mode='lines+markers',
                        name='In-Sample',
                        line={'width': 2, 'color': 'blue'},
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            if 'outsample' in metrics_df2.columns:
                fig.add_trace(
                    go.Scatter(
                        x=metrics_df2['date'],
                        y=metrics_df2['outsample'],
                        mode='lines+markers',
                        name='Out-of-Sample',
                        line={'width': 2, 'color': 'green'},
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # Add reference lines at 0.5
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=row, col=col)
        
        # Update axes
        for col in [1, 2]:
            fig.update_xaxes(title_text="Date", row=2, col=col)
            fig.update_yaxes(title_text="Accuracy", row=1, col=col)
            fig.update_yaxes(title_text="Accuracy", row=2, col=col)
            fig.update_yaxes(range=[0.45, 0.70], row=1, col=col)
            fig.update_yaxes(range=[0.45, 0.70], row=2, col=col)
        
        fig.update_layout(
            title="Classifier Performance Side-by-Side Comparison",
            height=600,
            showlegend=False,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_metrics_difference(self, metrics_df1, metrics_df2, dir1_name, dir2_name):
        """Create difference plot for classifier performance metrics"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('In-Sample Performance Difference', 'Out-of-Sample Performance Difference'),
            vertical_spacing=0.15
        )
        
        if metrics_df1 is None or metrics_df2 is None or len(metrics_df1) == 0 or len(metrics_df2) == 0:
            fig.add_annotation(
                text="Cannot compute difference - metrics data missing from one or both directories",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Merge dataframes on date
        df1_indexed = metrics_df1.set_index('date')
        df2_indexed = metrics_df2.set_index('date')
        
        # Align data
        common_dates = sorted(set(df1_indexed.index) & set(df2_indexed.index))
        
        if len(common_dates) > 0:
            # In-sample difference
            if 'insample' in df1_indexed.columns and 'insample' in df2_indexed.columns:
                insample_diff = df2_indexed.loc[common_dates, 'insample'] - df1_indexed.loc[common_dates, 'insample']
                
                fig.add_trace(
                    go.Scatter(
                        x=insample_diff.index,
                        y=insample_diff.values,
                        mode='lines+markers',
                        name='In-Sample Diff',
                        line={'width': 2, 'color': 'purple'}
                    ),
                    row=1, col=1
                )
                
                # Add statistics
                mean_diff = insample_diff.mean()
                std_diff = insample_diff.std()
                
                fig.add_hline(y=mean_diff, line_dash="dash", line_color="blue", 
                             annotation_text=f"Mean: {mean_diff:.4f}", row=1, col=1)
                fig.add_hline(y=mean_diff + 2*std_diff, line_dash="dot", line_color="red",
                             annotation_text=f"+2σ: {mean_diff + 2*std_diff:.4f}", row=1, col=1)
                fig.add_hline(y=mean_diff - 2*std_diff, line_dash="dot", line_color="red",
                             annotation_text=f"-2σ: {mean_diff - 2*std_diff:.4f}", row=1, col=1)
            
            # Out-of-sample difference
            if 'outsample' in df1_indexed.columns and 'outsample' in df2_indexed.columns:
                outsample_diff = df2_indexed.loc[common_dates, 'outsample'] - df1_indexed.loc[common_dates, 'outsample']
                
                fig.add_trace(
                    go.Scatter(
                        x=outsample_diff.index,
                        y=outsample_diff.values,
                        mode='lines+markers',
                        name='Out-of-Sample Diff',
                        line={'width': 2, 'color': 'orange'}
                    ),
                    row=2, col=1
                )
                
                # Add statistics
                mean_diff = outsample_diff.mean()
                std_diff = outsample_diff.std()
                
                fig.add_hline(y=mean_diff, line_dash="dash", line_color="blue", 
                             annotation_text=f"Mean: {mean_diff:.4f}", row=2, col=1)
                fig.add_hline(y=mean_diff + 2*std_diff, line_dash="dot", line_color="red",
                             annotation_text=f"+2σ: {mean_diff + 2*std_diff:.4f}", row=2, col=1)
                fig.add_hline(y=mean_diff - 2*std_diff, line_dash="dot", line_color="red",
                             annotation_text=f"-2σ: {mean_diff - 2*std_diff:.4f}", row=2, col=1)
        
        # Add zero lines
        fig.add_hline(y=0, line_dash="solid", line_color="gray", row=1, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1)
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy Difference", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy Difference", row=2, col=1)
        
        fig.update_layout(
            title=f"Classifier Performance Difference: {dir2_name} - {dir1_name}",
            height=600,
            showlegend=False,
            hovermode='x unified'
        )
        
        return fig
    
    def run(self):
        """Run the Dash app"""
        logger.info(f"Starting Fits Comparison Report on port {self.port}")
        self.app.run(debug=False, port=self.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fits comparison report for comparing model statistics across different fits directories')
    parser.add_argument('-p', '--port', help='port', type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    
    report = FitsComparisonReport(port=args.port)
    report.run()