import os
import glob
import logging.config
import argparse
import traceback
from collections import deque
from typing import Optional, Literal
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import Dash, html, dcc, Output, Input, State
from dash.dash_table import DataTable
from dash.dash_table.Format import Format
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline

from lib.util.config import get_config
from lib.util.directory import CONFIG_DIR, dir_manager
from lib.util.util import LOCAL
from lib.util.logging_util import get_logging_config

logging.config.dictConfig(get_logging_config("fits_report"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FitsReport:
    def __init__(self, port: int = 8057, env: Literal['dev', 'prod'] = 'dev'):
        self.port = port
        self.env = env
        self.app = Dash(__name__)
        self.fits_dir = os.path.join(dir_manager.DATA_DIR, 'fits', env)
        self.fits_data = {}
        self.classifier_models = {}

        # Initialize with default config
        self.update_config()

        # Load fits data on startup
        self.load_fits_data()
        self.load_classifier_models()
        
        # Setup the dash app layout and callbacks
        self.setup_layout()
        self.setup_callbacks()
    
    def update_config(self, config_file: Optional[str] = None):
        """Update configuration"""
        _, config = get_config(config_file=f'{CONFIG_DIR}/{config_file}' if config_file else None)
        self.config = config
    
    def load_fits_data(self):
        """Load all fits CSV data from the fits directory"""
        try:
            logger.info(f"Loading fits data from {self.fits_dir}")
            
            # Check if the directory exists
            if not os.path.exists(self.fits_dir):
                logger.error(f"Fits directory does not exist: {self.fits_dir}")
                return
            
            # Find all fits CSV files recursively for the specified environment
            # Pattern: fits.{env}.{horizon}.{model}.{date}.csv
            pattern = f"fits.{self.env}.*.csv"
            fits_files = glob.glob(os.path.join(self.fits_dir, "**", pattern), recursive=True)
            
            if not fits_files:
                logger.error("No fits files found")
                return
            
            # Group files by horizon and model type
            # Build in temp dict to avoid race conditions with display callbacks
            new_fits_data = {}

            for file_path in sorted(fits_files):
                try:
                    # Extract horizon and model from filename
                    # Format: fits.{env}.{horizon}.{model}.{date}.csv
                    basename = os.path.basename(file_path)
                    parts = basename.split('.')
                    if len(parts) >= 5:
                        horizon = parts[2]
                        model = parts[3]
                        date = parts[4]

                        key = f"{model}_{horizon}"
                        if key not in new_fits_data:
                            new_fits_data[key] = []

                        # Load CSV data
                        df = pd.read_csv(file_path)
                        df['file_date'] = pd.to_datetime(date, format='%Y%m%d')
                        df['file_horizon'] = int(horizon)
                        df['file_model'] = model

                        new_fits_data[key].append(df)

                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    logger.warning(traceback.format_exc())
                    continue

            # Concatenate data for each model-horizon combination
            for key, value in new_fits_data.items():
                if value:
                    new_fits_data[key] = pd.concat(value, ignore_index=True)
                    new_fits_data[key].sort_values('file_date', inplace=True)
                    new_fits_data[key].rename(columns={'file_date': 'date'}, inplace=True)
                    date_min = new_fits_data[key]['date'].min().strftime('%Y-%m-%d')
                    date_max = new_fits_data[key]['date'].max().strftime('%Y-%m-%d')
                    n_dates = new_fits_data[key]['date'].nunique()
                    logger.info(f"  {key}: {len(new_fits_data[key])} records, "
                                f"{n_dates} dates, range {date_min} to {date_max}")

            # Atomic swap - prevents race conditions with concurrent callbacks
            self.fits_data = new_fits_data

            logger.info(f"Loaded {self.env} fits data: "
                        f"{len(self.fits_data)} model-horizon combinations, "
                        f"fits_dir={self.fits_dir}")
                
        except Exception as e:
            logger.error(f"Error loading fits data: {e}")
    
    def load_classifier_models(self):
        """Load classifier models (SVM/Random Forest) from joblib files"""
        try:
            logger.info(f"Loading classifier models from {self.fits_dir}")

            # Find all joblib files recursively (stored alongside CSV files)
            joblib_files = glob.glob(os.path.join(self.fits_dir, "**", "classifier.*.joblib"), recursive=True)

            if not joblib_files:
                logger.info("No joblib model files found")
                return
            
            # Build in temp dict to avoid race conditions with display callbacks
            new_classifier_models = {}

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

                        if key not in new_classifier_models:
                            new_classifier_models[key] = []

                        model_data = {
                            'date': pd.to_datetime(date, format='%Y%m%d'),
                            'model_path': file_path,
                            'features_path': features_file if os.path.exists(features_file) else None
                        }

                        new_classifier_models[key].append(model_data)

                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
                    continue

            # Sort models by date
            for key in new_classifier_models:
                new_classifier_models[key] = sorted(
                    new_classifier_models[key], key=lambda x: x['date']
                )
                logger.info(f"Found {len(new_classifier_models[key])} models for {key}")

            # Atomic swap - prevents race conditions with concurrent callbacks
            self.classifier_models = new_classifier_models

            logger.info(f"Found SVM models for {len(self.classifier_models)} model types")
                
        except Exception as e:
            logger.error(f"Error loading classifier models: {e}")
    
    def get_available_models(self):
        """Get list of available models"""
        models = list(self.fits_data.keys())
        return sorted(models)
    
    def get_available_classifier_models(self):
        """Get list of available classifier models"""
        models = list(self.classifier_models.keys())
        return sorted(models)
    
    def get_available_horizons(self):
        """Get list of available horizons"""
        horizons = set()
        for key in self.fits_data.keys():
            # Extract horizon from model_horizon key
            parts = key.split('_')
            if len(parts) >= 2:
                horizon = parts[-1]  # Last part is horizon
                horizons.add(int(horizon))
        return sorted(list(horizons))
    
    def get_models_for_horizon(self, horizon):
        """Get list of models available for a specific horizon"""
        models = set()
        horizon_str = str(horizon)
        for key in self.fits_data.keys():
            if key.endswith(f'_{horizon_str}'):
                # Extract model name (everything before the last underscore)
                model = '_'.join(key.split('_')[:-1])
                models.add(model)
        return sorted(list(models))
    
    def get_available_classifier_horizons(self):
        """Get list of available horizons for classifier models"""
        horizons = set()
        for key in self.classifier_models.keys():
            parts = key.split('_')
            if len(parts) >= 2:
                horizon = parts[-1]
                horizons.add(int(horizon))
        return sorted(list(horizons))
    
    def get_classifier_models_for_horizon(self, horizon):
        """Get list of classifier models available for a specific horizon"""
        models = set()
        horizon_str = str(horizon)
        for key in self.classifier_models.keys():
            if key.endswith(f'_{horizon_str}'):
                model = '_'.join(key.split('_')[:-1])
                models.add(model)
        return sorted(list(models))
    
    def get_feature_importance_over_time(self, model_key):
        """Extract feature importance data for all dates of a given model"""
        if model_key not in self.classifier_models:
            return pd.DataFrame()
        
        importance_data = []
        
        for model_info in self.classifier_models[model_key]:
            try:
                # Load the model
                model = joblib.load(model_info['model_path'])
                
                # Extract the actual model from pipeline if needed
                actual_model = model
                if isinstance(model, Pipeline):
                    for _, step in model.steps:
                        if hasattr(step, 'feature_importances_'):
                            actual_model = step
                            break
                
                if not hasattr(actual_model, 'feature_importances_'):
                    continue
                
                # Get feature names
                feature_names = None
                if model_info['features_path'] and os.path.exists(model_info['features_path']):
                    try:
                        with open(model_info['features_path'], 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            feature_names = []
                            for line in lines:
                                line = line.strip()
                                if line.startswith('insample:') or line.startswith('outsample:'):
                                    continue
                                if ':' in line:
                                    feature_name = line.split(':')[0].strip()
                                    feature_names.append(feature_name)
                                else:
                                    feature_names.append(line)
                    except Exception as e:
                        logger.warning(f"Error reading features file: {e}")
                
                if feature_names is None:
                    feature_names = [f"Feature {i}" for i in range(len(actual_model.feature_importances_))]
                
                # Create dataframe for this date
                for feat, imp in zip(feature_names[:len(actual_model.feature_importances_)], 
                                    actual_model.feature_importances_):
                    importance_data.append({
                        'date': model_info['date'],
                        'feature': feat,
                        'importance': imp
                    })
                    
            except Exception as e:
                logger.warning(f"Error loading model for feature importance: {e}")
                continue
        
        return pd.DataFrame(importance_data)
    
    def setup_layout(self):
        """Setup the Dash app layout"""
        # Defer layout generation to ensure data is loaded
        self.app.layout = self._generate_layout
    
    def _generate_layout(self):
        """Generate layout - dropdowns are populated dynamically via callbacks"""
        return html.Div([
            html.H1("Fits Report", style={'textAlign': 'center'}),
            
            # Environment selector and reload button
            html.Div([
                html.Div([
                    html.Label("Environment:", style={'marginBottom': '5px', 'fontSize': '16px',
                                                      'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id='env-selector',
                        options=[
                            {'label': 'Dev', 'value': 'dev'},
                            {'label': 'Prod', 'value': 'prod'},
                        ],
                        value=self.env,
                        inline=True,
                        style={'fontSize': '14px'},
                        inputStyle={'marginRight': '5px'},
                        labelStyle={'marginRight': '20px'}
                    ),
                ], style={'display': 'inline-block', 'marginRight': '30px', 'verticalAlign': 'middle'}),
                html.Div([
                    html.Button("Reload Data", id="reload-btn", n_clicks=0),
                ], style={'display': 'inline-block', 'verticalAlign': 'middle'}),
                dcc.Loading(
                    id="loading-indicator",
                    type="default",
                    children=html.Div(id="load-status",
                                      style={"marginTop": "10px", "color": "green"}),
                ),
            ], style={'marginBottom': '30px'}),

            # Model selector for fits data
            html.Div([
                html.H3("Model Fits Over Time", style={'textAlign': 'center'}),
                html.Div([
                    html.Div([
                        html.Label("Select Horizon:", style={'marginBottom': '10px', 'fontSize': '16px'}),
                        dcc.Dropdown(
                            id='horizon-selector',
                            placeholder="Select a horizon",
                            style={'marginBottom': '20px'}
                        ),
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    html.Div([
                        html.Label("Select Model:", style={'marginBottom': '10px', 'fontSize': '16px'}),
                        dcc.Dropdown(
                            id='model-selector',
                            placeholder="Select a model",
                            style={'marginBottom': '20px'}
                        ),
                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                ], style={'marginBottom': '20px'}),
            ]),
            
            # T-statistics over time
            html.Div([
                html.H4("T-Statistics Over Time", style={'textAlign': 'center'}),
                dcc.Graph(id='tstat-timeseries'),
            ], style={'marginBottom': '30px'}),
            
            # Coefficients over time
            html.Div([
                html.H4("Coefficients Over Time", style={'textAlign': 'center'}),
                dcc.Graph(id='coeff-timeseries'),
            ], style={'marginBottom': '30px'}),
            
            # Model quality metrics
            html.Div([
                html.H4("Model Quality Metrics", style={'textAlign': 'center'}),
                dcc.Graph(id='quality-metrics'),
            ], style={'marginBottom': '30px'}),
            
            # Summary statistics table
            html.Div([
                html.H4("Current Model Statistics", style={'textAlign': 'center'}),
                DataTable(
                    id='model-stats-table',
                    columns=[
                        {'name': 'Metric', 'id': 'metric', 'type': 'text'},
                        {'name': 'Lag 0', 'id': 'lag0', 'type': 'numeric', 'format': Format(precision=4)},
                        {'name': 'Lag 1', 'id': 'lag1', 'type': 'numeric', 'format': Format(precision=4)},
                        {'name': 'Rev', 'id': 'rev', 'type': 'numeric', 'format': Format(precision=4)},
                        {'name': 'Mom', 'id': 'mom', 'type': 'numeric', 'format': Format(precision=4)},
                    ],
                    style_cell={'textAlign': 'center'},
                ),
            ], style={'marginBottom': '30px'}),
            
            # Random Forest Model Analysis
            html.Div([
                html.H3("Random Forest Model Analysis", style={'textAlign': 'center'}),
                html.Div([
                    html.Div([
                        html.Label("Select Horizon:", style={'marginBottom': '10px', 'fontSize': '16px'}),
                        dcc.Dropdown(
                            id='classifier-horizon-selector',
                            placeholder="Select a horizon",
                            style={'marginBottom': '20px'}
                        ),
                    ], style={'width': '32%', 'display': 'inline-block'}),
                    html.Div([
                        html.Label("Select Model:", style={'marginBottom': '10px', 'fontSize': '16px'}),
                        dcc.Dropdown(
                            id='classifier-model-selector',
                            placeholder="Select a classifier model",
                            style={'marginBottom': '20px'}
                        ),
                    ], style={'width': '32%', 'display': 'inline-block', 'marginLeft': '2%'}),
                    html.Div([
                        html.Label("Select Date:", style={'marginBottom': '10px', 'fontSize': '16px'}),
                        dcc.Dropdown(
                            id='classifier-date-selector',
                            placeholder="Select a date",
                            style={'marginBottom': '20px'}
                        ),
                    ], style={'width': '32%', 'float': 'right', 'display': 'inline-block'}),
                ], style={'marginBottom': '20px'}),
            ]),
            
            # Feature importance plot
            html.Div([
                html.H4("Feature Importance", style={'textAlign': 'center'}),
                dcc.Graph(id='feature-importance'),
            ], style={'marginBottom': '30px'}),
            
            # Feature importance over time
            html.Div([
                html.H4("Feature Importance Over Time", style={'textAlign': 'center'}),
                dcc.Graph(id='feature-importance-timeseries'),
            ], style={'marginBottom': '30px'}),
            
            # Model performance metrics
            html.Div([
                html.H4("Random Forest Model Details", style={'textAlign': 'center'}),
                html.Div(id='rf-model-details'),
            ], style={'marginBottom': '30px'}),
        ])
    
    def setup_callbacks(self):
        """Setup Dash callbacks"""

        @self.app.callback(
            Output("load-status", "children"),
            Output("horizon-selector", "options"),
            Output("horizon-selector", "value"),
            Output("classifier-horizon-selector", "options"),
            Output("classifier-horizon-selector", "value"),
            Input("env-selector", "value"),
            Input("reload-btn", "n_clicks"),
        )
        def reload_data(selected_env, n_clicks):  # pylint: disable=unused-argument
            """Reload fits data when environment changes or reload button is clicked"""
            env_changed = selected_env != self.env
            force_reload = (n_clicks or 0) > 0 and dash.callback_context.triggered_id == "reload-btn"

            if env_changed or force_reload or not self.fits_data:
                logger.info(f"Env switch: {self.env} -> {selected_env}, "
                            f"reload={force_reload}")
                self.env = selected_env
                self.fits_dir = os.path.join(dir_manager.DATA_DIR, 'fits', selected_env)
                self.load_fits_data()
                self.load_classifier_models()
            else:
                logger.info(f"Skipping reload, data already loaded for {selected_env}")

            fits_count = len(self.fits_data)
            classifier_count = len(self.classifier_models)
            status = (f"Loaded {fits_count} fits and {classifier_count} classifier "
                      f"model combinations from {selected_env}")

            horizons = self.get_available_horizons()
            horizon_opts = [{'label': str(h), 'value': h} for h in horizons]
            horizon_val = horizons[0] if horizons else None

            cls_horizons = self.get_available_classifier_horizons()
            cls_horizon_opts = [{'label': str(h), 'value': h} for h in cls_horizons]
            cls_horizon_val = cls_horizons[0] if cls_horizons else None

            return status, horizon_opts, horizon_val, cls_horizon_opts, cls_horizon_val

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
            default_value = models[0] if models else None

            return options, default_value

        @self.app.callback(
            Output('classifier-model-selector', 'options'),
            Output('classifier-model-selector', 'value'),
            Input('classifier-horizon-selector', 'value'),
        )
        def update_classifier_model_options(selected_horizon):
            """Update available classifier models for selected horizon"""
            if not selected_horizon:
                return [], None

            models = self.get_classifier_models_for_horizon(selected_horizon)
            options = [{'label': m, 'value': m} for m in models]
            default_value = models[0] if models else None

            return options, default_value

        @self.app.callback(
            Output('classifier-date-selector', 'options'),
            Output('classifier-date-selector', 'value'),
            Input('classifier-horizon-selector', 'value'),
            Input('classifier-model-selector', 'value'),
        )
        def update_classifier_date_options(selected_horizon, selected_model):
            """Update available dates for selected classifier model"""
            if not selected_horizon or not selected_model:
                return [], None

            key = f"{selected_model}_{selected_horizon}"

            if key not in self.classifier_models:
                return [], None

            dates = [{'label': m['date'].strftime('%Y-%m-%d'), 'value': m['date'].strftime('%Y%m%d')}
                     for m in self.classifier_models[key]]
            latest_date = dates[-1]['value'] if dates else None

            return dates, latest_date

        @self.app.callback(
            Output('tstat-timeseries', 'figure'),
            Output('coeff-timeseries', 'figure'),
            Output('quality-metrics', 'figure'),
            Output('model-stats-table', 'data'),
            Input('horizon-selector', 'value'),
            Input('model-selector', 'value'),
        )
        def update_fits_displays(selected_horizon, selected_model):
            """Update fits data displays"""
            if not selected_horizon or not selected_model:
                return go.Figure(), go.Figure(), go.Figure(), []
            
            # Construct the key from model and horizon
            key = f"{selected_model}_{selected_horizon}"
            
            if key not in self.fits_data:
                return go.Figure(), go.Figure(), go.Figure(), []
            
            df = self.fits_data[key]
            
            # Check if df is valid
            if df is None or isinstance(df, list) or len(df) == 0:
                return go.Figure(), go.Figure(), go.Figure(), []
            
            # T-statistics over time
            tstat_fig = go.Figure()
            
            # Group by condition and lag
            if 'condition' in df.columns and 'lag' in df.columns:
                for condition in df['condition'].unique():
                    for lag in df['lag'].unique():
                        mask = (df['condition'] == condition) & (df['lag'] == lag)
                        subset = df[mask]
                        
                        if len(subset) > 0:
                            tstat_fig.add_trace(go.Scatter(
                                x=subset['date'],
                                y=subset['tstat'],
                                mode='lines+markers',
                                name=f"{condition} (lag {lag})",
                                line={'width': 2}
                            ))
            else:
                # If no condition/lag columns, just plot t-statistics over time
                tstat_fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['tstat'],
                    mode='lines+markers',
                    name='T-statistic',
                    line={'width': 2}
                ))
            
            # Add reference lines
            tstat_fig.add_hline(y=1.96, line_dash="dash", line_color="green", 
                              annotation_text="95% significance")
            tstat_fig.add_hline(y=-1.96, line_dash="dash", line_color="green")
            tstat_fig.add_hline(y=0, line_dash="solid", line_color="gray")
            
            tstat_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="T-Statistic",
                hovermode='x unified'
            )
            
            # Coefficients over time
            coeff_fig = go.Figure()
            
            if 'condition' in df.columns and 'lag' in df.columns:
                for condition in df['condition'].unique():
                    for lag in df['lag'].unique():
                        mask = (df['condition'] == condition) & (df['lag'] == lag)
                        subset = df[mask]
                        
                        if len(subset) > 0:
                            coeff_fig.add_trace(go.Scatter(
                                x=subset['date'],
                                y=subset['coeff'],
                                mode='lines+markers',
                                name=f"{condition} (lag {lag})",
                                line={'width': 2}
                            ))
            else:
                # If no condition/lag columns, just plot coefficients over time
                coeff_fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['coeff'],
                    mode='lines+markers',
                    name='Coefficient',
                    line={'width': 2}
                ))
            
            coeff_fig.add_hline(y=0, line_dash="solid", line_color="gray")
            
            coeff_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Coefficient",
                hovermode='x unified'
            )
            
            # Quality metrics (number of observations, impurity)
            quality_fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Number of Observations', 'Impurity'),
                vertical_spacing=0.15
            )
            
            if 'condition' in df.columns:
                # Number of observations
                for condition in df['condition'].unique():
                    mask = df['condition'] == condition
                    subset = df[mask].drop_duplicates(subset=['date'])
                    
                    quality_fig.add_trace(
                        go.Scatter(
                            x=subset['date'],
                            y=subset['nobs'],
                            mode='lines+markers',
                            name=f"{condition}",
                            line={'width': 2}
                        ),
                        row=1, col=1
                    )
                
                # Impurity
                for condition in df['condition'].unique():
                    mask = df['condition'] == condition
                    subset = df[mask].drop_duplicates(subset=['date'])
                    
                    quality_fig.add_trace(
                        go.Scatter(
                            x=subset['date'],
                            y=subset['impurity'],
                            mode='lines+markers',
                            name=f"{condition}",
                            line={'width': 2},
                            showlegend=False
                        ),
                        row=2, col=1
                    )
            else:
                # Without condition, just plot overall metrics
                df_unique = df.drop_duplicates(subset=['date'])
                
                quality_fig.add_trace(
                    go.Scatter(
                        x=df_unique['date'],
                        y=df_unique['nobs'],
                        mode='lines+markers',
                        name='Observations',
                        line={'width': 2}
                    ),
                    row=1, col=1
                )
                
                quality_fig.add_trace(
                    go.Scatter(
                        x=df_unique['date'],
                        y=df_unique['impurity'],
                        mode='lines+markers',
                        name='Impurity',
                        line={'width': 2},
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            quality_fig.update_xaxes(title_text="Date", row=2, col=1)
            quality_fig.update_yaxes(title_text="Count", row=1, col=1)
            quality_fig.update_yaxes(title_text="Impurity", row=2, col=1)
            quality_fig.update_layout(height=600, hovermode='x unified')
            
            # Summary statistics table
            latest_date = df['date'].max()
            latest_df = df[df['date'] == latest_date]
            
            # Create summary table
            table_data = []
            
            metrics = ['tstat', 'coeff', 'stderr', 'nobs']
            
            if 'lag' in df.columns and 'condition' in df.columns:
                for metric in metrics:
                    row = {'metric': metric}
                    
                    # Lag 0
                    lag0_data = latest_df[latest_df['lag'] == 0]
                    if len(lag0_data) > 0:
                        row['lag0'] = lag0_data[metric].mean()
                    else:
                        row['lag0'] = None
                    
                    # Lag 1
                    lag1_data = latest_df[latest_df['lag'] == 1]
                    if len(lag1_data) > 0:
                        row['lag1'] = lag1_data[metric].mean()
                    else:
                        row['lag1'] = None
                    
                    # Rev
                    rev_data = latest_df[latest_df['condition'] == 'rev']
                    if len(rev_data) > 0:
                        row['rev'] = rev_data[metric].mean()
                    else:
                        row['rev'] = None
                    
                    # Mom
                    mom_data = latest_df[latest_df['condition'] == 'mom']
                    if len(mom_data) > 0:
                        row['mom'] = mom_data[metric].mean()
                    else:
                        row['mom'] = None
                    
                    table_data.append(row)
            else:
                # Simple table without lag/condition breakdown
                for metric in metrics:
                    if metric in latest_df.columns:
                        row = {
                            'metric': metric,
                            'lag0': latest_df[metric].mean(),
                            'lag1': None,
                            'rev': None,
                            'mom': None
                        }
                        table_data.append(row)
            
            return tstat_fig, coeff_fig, quality_fig, table_data
        
        @self.app.callback(
            Output('feature-importance', 'figure'),
            Output('feature-importance-timeseries', 'figure'),
            Output('rf-model-details', 'children'),
            Input('classifier-horizon-selector', 'value'),
            Input('classifier-model-selector', 'value'),
            Input('classifier-date-selector', 'value'),
        )
        def update_rf_displays(selected_horizon, selected_model, selected_date):
            """Update Random Forest model displays"""
            if not selected_horizon or not selected_model or not selected_date:
                return go.Figure(), go.Figure(), html.Div("No model selected")
            
            # Construct the key from model and horizon
            key = f"{selected_model}_{selected_horizon}"
            
            if key not in self.classifier_models:
                return go.Figure(), go.Figure(), html.Div("No model selected")
            
            # Find the model for the selected date
            model_info = None
            for m in self.classifier_models[key]:
                if m['date'].strftime('%Y%m%d') == selected_date:
                    model_info = m
                    break
            
            if not model_info:
                return go.Figure(), go.Figure(), html.Div("Model not found for selected date")
            
            try:
                # Load the model
                model = joblib.load(model_info['model_path'])
                
                # Extract the actual model from pipeline if needed
                actual_model = model
                if isinstance(model, Pipeline):
                    # Find the estimator in the pipeline
                    for name, step in model.steps:
                        if hasattr(step, 'feature_importances_'):
                            actual_model = step
                            break
                
                # Log model type for debugging
                logger.info(f"Model type: {type(model).__name__}")
                logger.info(f"Actual model type: {type(actual_model).__name__}")
                if hasattr(actual_model, '__class__'):
                    logger.info(f"Actual model class: {actual_model.__class__}")
                
                # Create subplot with two plots
                
                # Feature importance plot
                if hasattr(actual_model, 'feature_importances_'):
                    # Get feature names if available
                    feature_names = None
                    if model_info['features_path'] and os.path.exists(model_info['features_path']):
                        try:
                            with open(model_info['features_path'], 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                feature_names = []
                                for line in lines:
                                    line = line.strip()
                                    # Skip insample/outsample metrics (legacy format)
                                    if line.startswith('insample:') or line.startswith('outsample:'):
                                        continue
                                    # Extract feature name (before the colon)
                                    if ':' in line:
                                        feature_name = line.split(':')[0].strip()
                                        feature_names.append(feature_name)
                                    else:
                                        feature_names.append(line)
                        except Exception as e:
                            logger.warning(f"Error reading features file: {e}")
                    
                    if feature_names is None:
                        feature_names = [f"Feature {i}" for i in range(len(actual_model.feature_importances_))]
                    
                    # Sort by importance
                    importance_df = pd.DataFrame({
                        'feature': feature_names[:len(actual_model.feature_importances_)],
                        'importance': actual_model.feature_importances_
                    }).sort_values('importance', ascending=True)
                    
                    # Show top 20 features
                    top_features = importance_df.tail(20)
                    
                    # Create two subplots: bar chart and treemap
                    importance_fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Top 20 Feature Importances', 'Feature Importance Treemap'),
                        specs=[[{"type": "bar"}, {"type": "treemap"}]],
                        column_widths=[0.5, 0.5]
                    )
                    
                    # Bar chart
                    importance_fig.add_trace(
                        go.Bar(
                            x=top_features['importance'],
                            y=top_features['feature'],
                            orientation='h',
                            marker_color='lightblue',
                            name='Importance'
                        ),
                        row=1, col=1
                    )
                    
                    # Treemap visualization for all features
                    all_features = importance_df[importance_df['importance'] > 0.001]  # Filter very small values
                    
                    # Create hierarchical data for treemap
                    importance_fig.add_trace(
                        go.Treemap(
                            labels=all_features['feature'],
                            parents=[""] * len(all_features),
                            values=all_features['importance'],
                            textinfo="label+value+percent root",
                            marker={
                                'colorscale': 'Blues',
                                'cmid': all_features['importance'].mean()
                            },
                            hovertemplate='<b>%{label}</b><br>Importance: %{value:.4f}<br>%{percentRoot}<extra></extra>'
                        ),
                        row=1, col=2
                    )
                    
                    importance_fig.update_layout(
                        title_text=f"Feature Analysis - {selected_model} (Horizon: {selected_horizon}, Date: {selected_date})",
                        showlegend=False,
                        height=600,
                        margin={'l': 200}
                    )
                    
                    importance_fig.update_xaxes(title_text="Importance", row=1, col=1)
                    importance_fig.update_yaxes(title_text="Feature", row=1, col=1)
                    
                else:
                    importance_fig = go.Figure()
                    importance_fig.add_annotation(
                        text="Feature importances not available for this model",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                
                # Model details
                details_content = []
                
                # Model type information
                if isinstance(model, Pipeline):
                    details_content.append(html.H5("Pipeline Model"))
                    details_content.append(html.P(f"Steps: {', '.join([name for name, _ in model.steps])}"))
                    if actual_model != model:
                        details_content.append(html.P(f"Estimator: {type(actual_model).__name__}"))
                
                if isinstance(actual_model, (RandomForestRegressor, RandomForestClassifier)):
                    model_type = "Random Forest Regressor" if isinstance(actual_model, RandomForestRegressor) else "Random Forest Classifier"
                    if not isinstance(model, Pipeline):
                        details_content.append(html.H5(model_type))
                    details_content.append(html.P(f"Number of trees: {actual_model.n_estimators}"))
                    details_content.append(html.P(f"Max depth: {actual_model.max_depth}"))
                    details_content.append(html.P(f"Min samples split: {actual_model.min_samples_split}"))
                    details_content.append(html.P(f"Min samples leaf: {actual_model.min_samples_leaf}"))
                    details_content.append(html.P(f"Max features: {actual_model.max_features}"))
                    
                    if hasattr(actual_model, 'oob_score_'):
                        details_content.append(html.P(f"OOB Score: {actual_model.oob_score_:.4f}"))
                    
                    # Add tree complexity analysis
                    if hasattr(actual_model, 'estimators_'):
                        tree_depths = []
                        tree_nodes = []
                        for tree in actual_model.estimators_[:min(10, len(actual_model.estimators_))]:  # Sample first 10 trees
                            tree_depths.append(tree.tree_.max_depth)
                            tree_nodes.append(tree.tree_.node_count)
                        
                        details_content.append(html.Hr())
                        details_content.append(html.H6("Tree Complexity (first 10 trees):"))
                        details_content.append(html.P(f"Average depth: {np.mean(tree_depths):.1f}"))
                        details_content.append(html.P(f"Average nodes: {np.mean(tree_nodes):.1f}"))
                        details_content.append(html.P(f"Max depth observed: {max(tree_depths)}"))
                        details_content.append(html.P(f"Max nodes observed: {max(tree_nodes)}"))
                    
                elif not isinstance(model, Pipeline):
                    details_content.append(html.P(f"Model type: {type(actual_model).__name__}"))
                
                details_content.append(html.Hr())
                details_content.append(html.P(f"Model file: {os.path.basename(model_info['model_path'])}"))
                
                # Add sunburst chart for feature importance hierarchy if we have features
                if hasattr(actual_model, 'feature_importances_') and 'feature_names' in locals() and feature_names:
                    # Group features by type/prefix for hierarchical view
                    feature_groups = {}
                    for feat, imp in zip(feature_names[:len(actual_model.feature_importances_)], 
                                        actual_model.feature_importances_):
                        # Skip non-feature entries like insample/outsample
                        if feat.lower() in ['insample', 'outsample', 'in_sample', 'out_sample']:
                            continue
                        # Extract feature group from name (e.g., "logret_120_trmean" -> "logret")
                        group = feat.split('_')[0] if '_' in feat else 'other'
                        if group not in feature_groups:
                            feature_groups[group] = []
                        feature_groups[group].append((feat, imp))
                    
                    # Create sunburst data
                    labels = ["All Features"]
                    parents = [""]
                    values = [sum(actual_model.feature_importances_)]
                    
                    for group, features in feature_groups.items():
                        if sum(imp for _, imp in features) > 0.01:  # Only show significant groups
                            labels.append(group)
                            parents.append("All Features")
                            values.append(sum(imp for _, imp in features))
                            
                            # Add individual features in this group
                            for feat, imp in sorted(features, key=lambda x: x[1], reverse=True)[:5]:  # Top 5 per group
                                if imp > 0.005:
                                    labels.append(feat)
                                    parents.append(group)
                                    values.append(imp)
                    
                    sunburst_fig = go.Figure(go.Sunburst(
                        labels=labels,
                        parents=parents,
                        values=values,
                        branchvalues="total",
                        hovertemplate='<b>%{label}</b><br>Importance: %{value:.4f}<br>%{percentParent}<extra></extra>',
                        marker={'colorscale': 'Blues'}
                    ))
                    
                    sunburst_fig.update_layout(
                        title="Feature Importance Hierarchy",
                        height=400,
                        margin={'t': 50, 'l': 0, 'r': 0, 'b': 0}
                    )
                    
                    details_content.append(html.Hr())
                    details_content.append(dcc.Graph(figure=sunburst_fig))
                
                # Add tree structure visualization SECTION - OUTSIDE the feature_names conditional
                details_content.append(html.Hr())
                details_content.append(html.H5("Tree Detail", style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}))
                
                # Check for Random Forest models (both Classifier and Regressor)
                if isinstance(actual_model, (RandomForestRegressor, RandomForestClassifier)) and hasattr(actual_model, 'estimators_'):
                    logger.info(f"Creating tree visualizations for {type(actual_model).__name__} with {len(actual_model.estimators_)} trees")
                    
                    # Visualize the first tree
                    tree = actual_model.estimators_[0]
                    
                    # Get feature names if available (reuse from above or load again)
                    tree_feature_names = None
                    if 'feature_names' in locals() and feature_names:
                        tree_feature_names = feature_names[:len(actual_model.feature_importances_)]
                        logger.info(f"Using {len(tree_feature_names)} feature names for tree visualization")
                    else:
                        # Try to load feature names again if not available
                        if model_info['features_path'] and os.path.exists(model_info['features_path']):
                            try:
                                with open(model_info['features_path'], 'r', encoding='utf-8') as f:
                                    lines = f.readlines()
                                    tree_feature_names = []
                                    for line in lines:
                                        line = line.strip()
                                        # Skip insample/outsample metrics (legacy format)
                                        if line.startswith('insample:') or line.startswith('outsample:'):
                                            continue
                                        # Extract feature name (before the colon)
                                        if ':' in line:
                                            feature_name = line.split(':')[0].strip()
                                            tree_feature_names.append(feature_name)
                                        else:
                                            tree_feature_names.append(line)
                                    tree_feature_names = tree_feature_names[:len(actual_model.feature_importances_)]
                                    logger.info(f"Loaded {len(tree_feature_names)} feature names for tree visualization")
                            except Exception as e:
                                logger.warning(f"Error reading features file for tree: {e}")
                                logger.info("No feature names available for tree visualization")
                    
                    # Create tree structure visualization
                    tree_fig = self._create_tree_visualization(tree, tree_feature_names)
                    
                    details_content.append(html.Hr())
                    details_content.append(html.H6("Decision Tree Structure - Network View (First Tree)"))
                    details_content.append(dcc.Graph(figure=tree_fig))
                    
                    # Create sklearn-style tree visualization
                    sklearn_tree_fig = self._create_sklearn_tree_visualization(tree, tree_feature_names)
                    
                    details_content.append(html.Hr())
                    details_content.append(html.H6("Decision Tree Structure - Scikit-learn Style (First Tree)"))
                    details_content.append(dcc.Graph(figure=sklearn_tree_fig))
                    
                    # Create histogram of leaf node observations
                    leaf_histogram = self._create_leaf_node_histogram(actual_model, max_trees=10)
                    
                    details_content.append(html.Hr())
                    details_content.append(html.H6("Leaf Node Sample Distribution (First 10 Trees)"))
                    details_content.append(dcc.Graph(figure=leaf_histogram))
                else:
                    logger.info(f"Model type {type(actual_model).__name__} does not support tree visualization")
                    details_content.append(html.P("Tree visualization not available for this model type", 
                                                style={'textAlign': 'center', 'color': 'gray'}))
                
                # Create feature importance over time plot
                importance_timeseries_fig = self._create_feature_importance_timeseries(key)
                
                return importance_fig, importance_timeseries_fig, html.Div(details_content)
                
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return go.Figure(), go.Figure(), html.Div(f"Error loading model: {str(e)}")
    
    def _create_feature_importance_timeseries(self, selected_model):
        """Create a time series plot of feature importance over time"""
        
        # Get feature importance data over time
        importance_df = self.get_feature_importance_over_time(selected_model)
        
        if importance_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No feature importance data available over time",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Get top N features by average importance
        top_n = 15
        avg_importance = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        top_features = avg_importance.head(top_n).index.tolist()
        
        # Filter to top features
        plot_df = importance_df[importance_df['feature'].isin(top_features)]
        
        # Create line plot
        fig = go.Figure()
        
        # Add a trace for each feature
        for feature in top_features:
            feature_data = plot_df[plot_df['feature'] == feature].sort_values('date')
            
            fig.add_trace(go.Scatter(
                x=feature_data['date'],
                y=feature_data['importance'],
                mode='lines+markers',
                name=feature,
                line={'width': 2},
                marker={'size': 6},
                hovertemplate='<b>%{fullData.name}</b><br>' +
                            'Date: %{x|%Y-%m-%d}<br>' +
                            'Importance: %{y:.4f}<br>' +
                            '<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Top {top_n} Feature Importance Over Time - {selected_model.split('_')[0]} (Horizon: {selected_model.split('_')[-1]})",
            xaxis_title="Date",
            yaxis_title="Feature Importance",
            hovermode='x unified',
            legend={
                'orientation': 'v',
                'yanchor': 'top',
                'y': 1,
                'xanchor': 'left',
                'x': 1.02
            },
            margin={'r': 200},  # Make room for legend
            height=600
        )
        
        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeslider_thickness=0.05
        )
        
        return fig
    
    def _create_tree_visualization(self, tree, feature_names=None, max_depth=3):
        """Create a visualization of the decision tree structure"""
        
        # Extract tree structure
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        value = tree.tree_.value
        n_samples = tree.tree_.n_node_samples
        
        # Create node positions using hierarchical layout
        node_x = {}
        node_y = {}
        node_info = {}
        edges_x = []
        edges_y = []
        
        # BFS to create positions
        queue = deque([(0, 0, 0, 1)])  # node_id, depth, x_position, x_width
        
        while queue:
            node_id, depth, x_pos, x_width = queue.popleft()
            
            if depth > max_depth:
                continue
                
            # Store node position
            node_x[node_id] = x_pos
            node_y[node_id] = -depth  # negative to have root at top
            
            # Store node info
            is_leaf = children_left[node_id] == children_right[node_id]
            
            if not is_leaf:
                # Get feature name
                if feature_names and feature[node_id] < len(feature_names):
                    feat_name = feature_names[feature[node_id]]
                    # Skip metric nodes
                    if feat_name.lower() in ['insample', 'outsample', 'in_sample', 'out_sample']:
                        feat_name = f"Feature {feature[node_id]}"
                else:
                    feat_name = f"Feature {feature[node_id]}"
                
                node_info[node_id] = {
                    'text': f"{feat_name}<br>≤ {threshold[node_id]:.3f}",
                    'hover': f"Split Feature: {feat_name}<br>Threshold: {threshold[node_id]:.3f}<br>Samples: {n_samples[node_id]}<br>Value: {value[node_id][0][0]:.3f}",
                    'color': 'lightblue',
                    'symbol': 'square'
                }
                
                # Add children to queue
                if depth < max_depth:
                    # Left child
                    left_id = children_left[node_id]
                    left_x = x_pos - x_width/4
                    queue.append((left_id, depth + 1, left_x, x_width/2))
                    
                    # Right child
                    right_id = children_right[node_id]
                    right_x = x_pos + x_width/4
                    queue.append((right_id, depth + 1, right_x, x_width/2))
                    
                    # Add edges
                    # Left edge
                    edges_x.extend([x_pos, left_x, None])
                    edges_y.extend([-depth, -(depth + 1), None])
                    
                    # Right edge
                    edges_x.extend([x_pos, right_x, None])
                    edges_y.extend([-depth, -(depth + 1), None])
            else:
                # Leaf node
                node_info[node_id] = {
                    'text': f"Leaf<br>{value[node_id][0][0]:.3f}",
                    'hover': f"Leaf Node<br>Samples: {n_samples[node_id]}<br>Predicted Value: {value[node_id][0][0]:.3f}",
                    'color': 'lightgreen',
                    'symbol': 'circle'
                }
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edges_x, y=edges_y,
            line={'width': 2, 'color': '#888'},
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        
        # Create node traces (one for each color/symbol combination)
        node_traces = []
        
        # Group nodes by color and symbol
        node_groups = {}
        for node_id, info in node_info.items():
            key = (info['color'], info['symbol'])
            if key not in node_groups:
                node_groups[key] = {'x': [], 'y': [], 'text': [], 'hover': []}
            node_groups[key]['x'].append(node_x[node_id])
            node_groups[key]['y'].append(node_y[node_id])
            node_groups[key]['text'].append(info['text'])
            node_groups[key]['hover'].append(info['hover'])
        
        # Create traces for each group
        for (color, symbol), data in node_groups.items():
            node_trace = go.Scatter(
                x=data['x'], y=data['y'],
                mode='markers+text',
                hoverinfo='text',
                hovertext=data['hover'],
                text=data['text'],
                textposition="middle center",
                marker={
                    'color': color,
                    'size': 40,
                    'symbol': symbol,
                    'line': {'width': 2, 'color': 'DarkSlateGrey'}
                },
                showlegend=False
            )
            node_traces.append(node_trace)
        
        # Create annotations for edge labels
        annotations = []
        queue_labels = deque([(0, 0, 0, 1)])  # node_id, depth, x_position, x_width
        
        while queue_labels:
            node_id, depth, x_pos, x_width = queue_labels.popleft()
            
            if depth >= max_depth or children_left[node_id] == children_right[node_id]:
                continue
                
            # Left child
            left_id = children_left[node_id]
            left_x = x_pos - x_width/4
            
            # Right child
            right_id = children_right[node_id]
            right_x = x_pos + x_width/4
            
            # Add annotations for True/False branches
            annotations.append(
                {
                    'x': (x_pos + left_x) / 2,
                    'y': (-depth - 0.5),
                    'text': 'True',
                    'showarrow': False,
                    'font': {'size': 10, 'color': 'green'}
                }
            )
            
            annotations.append(
                {
                    'x': (x_pos + right_x) / 2,
                    'y': (-depth - 0.5),
                    'text': 'False',
                    'showarrow': False,
                    'font': {'size': 10, 'color': 'red'}
                }
            )
            
            # Add children to queue
            if depth + 1 < max_depth:
                queue_labels.append((left_id, depth + 1, left_x, x_width/2))
                queue_labels.append((right_id, depth + 1, right_x, x_width/2))
        
        # Create figure
        fig = go.Figure(data=[edge_trace] + node_traces)
        
        fig.update_layout(
            title=f"Decision Tree Structure (Max Depth: {max_depth})",
            showlegend=False,
            hovermode='closest',
            margin={'b': 20, 'l': 5, 'r': 5, 't': 40},
            annotations=annotations,
            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
            plot_bgcolor='white',
            height=600
        )
        
        return fig
    
    def _create_sklearn_tree_visualization(self, tree, feature_names=None, max_depth=4):
        """Create a scikit-learn style tree visualization using rectangles"""
        
        # Extract tree structure
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        value = tree.tree_.value
        n_samples = tree.tree_.n_node_samples
        impurity = tree.tree_.impurity
        
        # Create figure
        fig = go.Figure()
        
        # Calculate tree layout
        node_positions = {}
        node_widths = {}
        level_heights = {}
        
        # BFS to calculate positions
        queue = deque([(0, 0, 0, 1.0)])  # node_id, depth, x_center, width
        # max_samples = n_samples[0]  # Root node samples for scaling
        
        while queue:
            node_id, depth, x_center, width = queue.popleft()
            
            if depth > max_depth:
                continue
                
            # Store position and width
            node_positions[node_id] = (x_center, depth)
            node_widths[node_id] = width * 0.9  # Leave some gap between nodes
            
            # Track level heights
            if depth not in level_heights:
                level_heights[depth] = 0
            level_heights[depth] = max(level_heights[depth], 1)
            
            # Add children to queue if not a leaf and within depth limit
            if children_left[node_id] != children_right[node_id] and depth < max_depth:
                # Calculate child positions
                left_width = width * 0.5
                right_width = width * 0.5
                
                left_center = x_center - width * 0.25
                right_center = x_center + width * 0.25
                
                queue.append((children_left[node_id], depth + 1, left_center, left_width))
                queue.append((children_right[node_id], depth + 1, right_center, right_width))
        
        # Draw nodes and edges
        for node_id, (x_center, depth) in node_positions.items():
            y_center = -depth * 2  # Negative to have root at top
            width = node_widths[node_id]
            height = 0.8
            
            # Determine node color based on value (for regression)
            node_value = value[node_id][0][0]
            
            # Create node rectangle
            is_leaf = children_left[node_id] == children_right[node_id]
            
            # Node text
            if not is_leaf:
                if feature_names and feature[node_id] < len(feature_names):
                    feat_name = feature_names[feature[node_id]]
                    if feat_name.lower() in ['insample', 'outsample', 'in_sample', 'out_sample']:
                        feat_name = f"Feature {feature[node_id]}"
                else:
                    feat_name = f"Feature {feature[node_id]}"
                
                node_text = (
                    f"{feat_name} ≤ {threshold[node_id]:.3f}<br>"
                    f"mse = {impurity[node_id]:.3f}<br>"
                    f"samples = {n_samples[node_id]}<br>"
                    f"value = {node_value:.3f}"
                )
            else:
                node_text = (
                    f"mse = {impurity[node_id]:.3f}<br>"
                    f"samples = {n_samples[node_id]}<br>"
                    f"value = {node_value:.3f}"
                )
            
            # Draw rectangle
            fig.add_shape(
                type="rect",
                x0=x_center - width/2,
                y0=y_center - height/2,
                x1=x_center + width/2,
                y1=y_center + height/2,
                line={'color': 'black', 'width': 2},
                fillcolor="lightblue" if not is_leaf else "lightgreen",
                opacity=0.8
            )
            
            # Add text annotation
            fig.add_annotation(
                x=x_center,
                y=y_center,
                text=node_text,
                showarrow=False,
                font={'size': 10},
                align="center"
            )
            
            # Draw edges to children
            if not is_leaf and depth < max_depth:
                left_child = children_left[node_id]
                right_child = children_right[node_id]
                
                if left_child in node_positions:
                    left_x, left_y = node_positions[left_child]
                    # Draw line from bottom of parent to top of left child
                    fig.add_shape(
                        type="line",
                        x0=x_center,
                        y0=y_center - height/2,
                        x1=left_x,
                        y1=-left_y * 2 + height/2,
                        line={'color': 'black', 'width': 1}
                    )
                    
                if right_child in node_positions:
                    right_x, right_y = node_positions[right_child]
                    # Draw line from bottom of parent to top of right child
                    fig.add_shape(
                        type="line",
                        x0=x_center,
                        y0=y_center - height/2,
                        x1=right_x,
                        y1=-right_y * 2 + height/2,
                        line={'color': 'black', 'width': 1}
                    )
        
        # Update layout with better margins and range
        fig.update_layout(
            title=f"Decision Tree (Scikit-learn Style, Max Depth: {max_depth})",
            showlegend=False,
            xaxis={
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False,
                'range': [-0.6, 1.6]  # Expanded range to show full tree
            },
            yaxis={
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False,
                'range': [-(max_depth + 1) * 2 - 0.5, 1]
            },
            plot_bgcolor='white',
            height=800,
            margin={'t': 50, 'l': 100, 'r': 100, 'b': 50}  # Increased left/right margins
        )
        
        # Add invisible scatter trace to enable hover (shapes don't support hover)
        hover_x = []
        hover_y = []
        hover_text = []
        
        for node_id, (x_center, depth) in node_positions.items():
            hover_x.append(x_center)
            hover_y.append(-depth * 2)
            
            is_leaf = children_left[node_id] == children_right[node_id]
            if not is_leaf:
                if feature_names and feature[node_id] < len(feature_names):
                    feat_name = feature_names[feature[node_id]]
                    if feat_name.lower() in ['insample', 'outsample', 'in_sample', 'out_sample']:
                        feat_name = f"Feature {feature[node_id]}"
                else:
                    feat_name = f"Feature {feature[node_id]}"
                    
                hover_text.append(
                    f"Node {node_id}<br>"
                    f"Split: {feat_name} ≤ {threshold[node_id]:.3f}<br>"
                    f"MSE: {impurity[node_id]:.3f}<br>"
                    f"Samples: {n_samples[node_id]}<br>"
                    f"Value: {value[node_id][0][0]:.3f}"
                )
            else:
                hover_text.append(
                    f"Leaf Node {node_id}<br>"
                    f"MSE: {impurity[node_id]:.3f}<br>"
                    f"Samples: {n_samples[node_id]}<br>"
                    f"Value: {value[node_id][0][0]:.3f}"
                )
        
        fig.add_trace(go.Scatter(
            x=hover_x,
            y=hover_y,
            mode='markers',
            marker={'size': 1, 'opacity': 0},
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=False
        ))
        
        return fig
    
    def _create_leaf_node_histogram(self, forest_model, max_trees=10):
        """Create histogram showing distribution of observations in leaf nodes"""
        
        leaf_samples = []
        
        # Collect leaf node sample counts from first N trees
        for tree in forest_model.estimators_[:max_trees]:
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            n_samples = tree.tree_.n_node_samples
            
            # Find leaf nodes (nodes where left child == right child)
            n_nodes = tree.tree_.node_count
            for node_id in range(n_nodes):
                if children_left[node_id] == children_right[node_id]:  # Leaf node
                    leaf_samples.append(n_samples[node_id])
        
        # Create histogram
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=leaf_samples,
            nbinsx=30,
            name='Leaf Nodes',
            marker_color='lightblue',
            opacity=0.7,
            hovertemplate='Samples in leaf: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add statistics
        mean_samples = np.mean(leaf_samples)
        median_samples = np.median(leaf_samples)
        
        fig.add_vline(x=mean_samples, line_dash="dash", line_color="red", 
                      annotation_text=f"Mean: {mean_samples:.1f}")
        fig.add_vline(x=median_samples, line_dash="dash", line_color="green", 
                      annotation_text=f"Median: {median_samples:.1f}")
        
        fig.update_layout(
            title=f"Distribution of Samples in Leaf Nodes<br><sub>Total leaf nodes: {len(leaf_samples)} from {min(max_trees, len(forest_model.estimators_))} trees</sub>",
            xaxis_title="Number of Samples in Leaf Node",
            yaxis_title="Count of Leaf Nodes",
            bargap=0.1,
            showlegend=False,
            height=400,
            margin={'t': 80, 'l': 60, 'r': 60, 'b': 60}
        )
        
        # Add text box with statistics
        stats_text = (
            f"Total Leaf Nodes: {len(leaf_samples)}<br>"
            f"Min Samples: {min(leaf_samples)}<br>"
            f"Max Samples: {max(leaf_samples)}<br>"
            f"Mean Samples: {mean_samples:.1f}<br>"
            f"Median Samples: {median_samples:.1f}<br>"
            f"Std Dev: {np.std(leaf_samples):.1f}"
        )
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.95, y=0.95,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font={'size': 10},
            align="left"
        )
        
        return fig
    
    def run(self):
        """Run the Dash app"""
        logger.info(f"Starting Fits Report on port {self.port}")
        self.app.run(debug=False, port=self.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fits report showing model statistics over time')
    parser.add_argument('-p', '--port', help='port', type=int, default=None)
    parser.add_argument('-e', '--env', help='environment (dev or prod)', type=str,
                        default='prod', choices=['dev', 'prod'])
    args = parser.parse_args()

    port = args.port if args.port else (8067 if LOCAL else 8057)

    report = FitsReport(port=port, env=args.env)
    report.run()
