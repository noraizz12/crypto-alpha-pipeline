"""Fits Comparison Report v2 - A simplified Dash application for comparing fits data.

This report allows selecting two fits subdirectories (e.g., 'prod' and 'dev'),
finding overlapping model/horizon combinations, and loading data for comparison.
"""
import os
import glob
import logging.config
import argparse
from typing import Optional, List, Dict, Tuple, Set, Any
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, html, dcc, Output, Input, State
from dash.dash_table import DataTable

from lib.util.directory import dir_manager
from lib.util.logging_util import get_logging_config

logging.config.dictConfig(get_logging_config("fits_comparison_report2"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_PORT = 8062


class FitsDataLoader:
    """Handles loading and organizing fits data from directories.

    Separates data loading logic from display logic for easier debugging.
    """

    def __init__(self):
        self.fits_base_dir = os.path.join(dir_manager.DATA_DIR, 'fits')
        self.available_dirs: List[str] = []
        self._discover_directories()

    def _discover_directories(self) -> None:
        """Discover available fits subdirectories."""
        if not os.path.exists(self.fits_base_dir):
            logger.error(f"Fits base directory does not exist: {self.fits_base_dir}")
            return

        self.available_dirs = sorted([
            d for d in os.listdir(self.fits_base_dir)
            if os.path.isdir(os.path.join(self.fits_base_dir, d))
        ])
        logger.info(f"Found {len(self.available_dirs)} fits directories: {self.available_dirs}")

    def get_model_horizons_for_directory(self, dir_name: str) -> Set[Tuple[str, str]]:
        """Get all (model, horizon) combinations available in a directory.

        Args:
            dir_name: Name of the fits subdirectory (e.g., 'prod', 'dev')

        Returns:
            Set of (model, horizon) tuples
        """
        fits_dir = os.path.join(self.fits_base_dir, dir_name)
        if not os.path.exists(fits_dir):
            logger.warning(f"Directory does not exist: {fits_dir}")
            return set()

        model_horizons = set()

        # Structure is: fits/{dir_name}/{horizon}/{model}/
        for horizon_dir in os.listdir(fits_dir):
            horizon_path = os.path.join(fits_dir, horizon_dir)
            if not os.path.isdir(horizon_path):
                continue

            # Check if this is a horizon directory (numeric)
            if not horizon_dir.isdigit():
                continue

            for model_dir in os.listdir(horizon_path):
                model_path = os.path.join(horizon_path, model_dir)
                if os.path.isdir(model_path):
                    # Verify there are CSV files
                    csv_files = glob.glob(os.path.join(model_path, "fits.*.csv"))
                    if csv_files:
                        model_horizons.add((model_dir, horizon_dir))

        logger.info(f"Found {len(model_horizons)} model/horizon combinations in {dir_name}")
        return model_horizons

    def get_overlapping_model_horizons(
        self, dir1: str, dir2: str
    ) -> List[Tuple[str, str]]:
        """Get model/horizon combinations that exist in both directories.

        Args:
            dir1: First directory name
            dir2: Second directory name

        Returns:
            Sorted list of (model, horizon) tuples present in both directories
        """
        mh1 = self.get_model_horizons_for_directory(dir1)
        mh2 = self.get_model_horizons_for_directory(dir2)

        overlap = mh1 & mh2
        # Sort by horizon (numeric) then model
        sorted_overlap = sorted(overlap, key=lambda x: (int(x[1]), x[0]))

        logger.info(
            f"Found {len(overlap)} overlapping model/horizons between {dir1} and {dir2}"
        )
        return sorted_overlap

    def load_fits_data(
        self, dir_name: str, model: str, horizon: str
    ) -> Optional[pd.DataFrame]:
        """Load all fits CSV data for a specific model/horizon from a directory.

        Args:
            dir_name: Fits subdirectory name (e.g., 'prod', 'dev')
            model: Model name (e.g., 'ba', 'hl', 'c2vwap')
            horizon: Horizon value (e.g., '360', '1440')

        Returns:
            DataFrame with all fits data concatenated, or None if no data found
        """
        fits_dir = os.path.join(self.fits_base_dir, dir_name, horizon, model)

        if not os.path.exists(fits_dir):
            logger.warning(f"Fits directory does not exist: {fits_dir}")
            return None

        # Find all fits CSV files
        csv_pattern = os.path.join(fits_dir, "fits.*.csv")
        csv_files = sorted(glob.glob(csv_pattern))

        if not csv_files:
            logger.warning(f"No CSV files found in {fits_dir}")
            return None

        logger.info(f"Loading {len(csv_files)} CSV files from {fits_dir}")

        all_dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Extract date from filename: fits.{env}.{horizon}.{model}.{date}.csv
                basename = os.path.basename(csv_file)
                parts = basename.split('.')
                if len(parts) >= 5:
                    date_str = parts[4]
                    df['file_date'] = pd.to_datetime(date_str, format='%Y%m%d')
                    df['source_dir'] = dir_name
                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
                continue

        if not all_dfs:
            return None

        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.sort_values('file_date', inplace=True)

        # Use 'date' column if available, otherwise use 'file_date'
        if 'as_of' in combined_df.columns:
            combined_df['date'] = pd.to_datetime(combined_df['as_of'])
        else:
            combined_df['date'] = combined_df['file_date']

        logger.info(
            f"Loaded {len(combined_df)} rows for {dir_name}/{model}/{horizon} "
            f"from {combined_df['date'].min()} to {combined_df['date'].max()}"
        )

        return combined_df

    def load_metrics_data(
        self, dir_name: str, model: str, horizon: str
    ) -> Optional[pd.DataFrame]:
        """Load classifier metrics data for a specific model/horizon.

        Args:
            dir_name: Fits subdirectory name
            model: Model name
            horizon: Horizon value

        Returns:
            DataFrame with metrics data, or None if not found
        """
        fits_dir = os.path.join(self.fits_base_dir, dir_name, horizon, model)

        if not os.path.exists(fits_dir):
            return None

        metrics_pattern = os.path.join(fits_dir, "classifier.*.metrics")
        metrics_files = sorted(glob.glob(metrics_pattern))

        if not metrics_files:
            return None

        metrics_records = []
        for metrics_file in metrics_files:
            try:
                basename = os.path.basename(metrics_file)
                parts = basename.split('.')
                if len(parts) >= 4:
                    date_str = parts[3]

                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    metrics = {'date': pd.to_datetime(date_str, format='%Y%m%d')}
                    for line in lines:
                        if ':' in line:
                            key, value = line.strip().split(':')
                            try:
                                metrics[key.strip()] = float(value.strip())
                            except ValueError:
                                pass

                    if len(metrics) > 1:
                        metrics_records.append(metrics)
            except Exception as e:
                logger.warning(f"Failed to load metrics from {metrics_file}: {e}")
                continue

        if not metrics_records:
            return None

        metrics_df = pd.DataFrame(metrics_records)
        metrics_df.sort_values('date', inplace=True)

        logger.info(f"Loaded {len(metrics_df)} metrics records for {dir_name}/{model}/{horizon}")

        return metrics_df

    def load_features_data(
        self, dir_name: str, model: str, horizon: str
    ) -> Optional[pd.DataFrame]:
        """Load feature importance data from .features files.

        Args:
            dir_name: Fits subdirectory name
            model: Model name
            horizon: Horizon value

        Returns:
            DataFrame with feature importance data over time, or None if not found
        """
        fits_dir = os.path.join(self.fits_base_dir, dir_name, horizon, model)

        if not os.path.exists(fits_dir):
            return None

        features_pattern = os.path.join(fits_dir, "classifier.*.features")
        features_files = sorted(glob.glob(features_pattern))

        if not features_files:
            return None

        features_records = []
        for features_file in features_files:
            try:
                basename = os.path.basename(features_file)
                parts = basename.split('.')
                if len(parts) >= 4:
                    date_str = parts[3]

                    with open(features_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    record = {'date': pd.to_datetime(date_str, format='%Y%m%d')}
                    for line in lines:
                        if ':' in line:
                            feature_name, importance = line.strip().split(':')
                            try:
                                record[feature_name.strip()] = float(importance.strip())
                            except ValueError:
                                pass

                    if len(record) > 1:
                        features_records.append(record)
            except Exception as e:
                logger.warning(f"Failed to load features from {features_file}: {e}")
                continue

        if not features_records:
            return None

        features_df = pd.DataFrame(features_records)
        features_df.sort_values('date', inplace=True)

        logger.info(
            f"Loaded {len(features_df)} feature importance records "
            f"with {len(features_df.columns) - 1} features for {dir_name}/{model}/{horizon}"
        )

        return features_df

    def load_model_params(
        self, dir_name: str, model: str, horizon: str
    ) -> Optional[pd.DataFrame]:
        """Load model parameters from .joblib files.

        Extracts key hyperparameters from the trained models without loading
        the full model into memory.

        Args:
            dir_name: Fits subdirectory name
            model: Model name
            horizon: Horizon value

        Returns:
            DataFrame with model parameters over time, or None if not found
        """
        fits_dir = os.path.join(self.fits_base_dir, dir_name, horizon, model)

        if not os.path.exists(fits_dir):
            return None

        joblib_pattern = os.path.join(fits_dir, "classifier.*.joblib")
        joblib_files = sorted(glob.glob(joblib_pattern))

        if not joblib_files:
            return None

        model_records = []
        for joblib_file in joblib_files:
            try:
                basename = os.path.basename(joblib_file)
                parts = basename.split('.')
                if len(parts) >= 4:
                    date_str = parts[3]

                    # Load the model
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pipeline = joblib.load(joblib_file)

                    record = {'date': pd.to_datetime(date_str, format='%Y%m%d')}

                    # Extract parameters from the pipeline
                    if hasattr(pipeline, 'get_params'):
                        params = pipeline.get_params()

                        # Extract key RandomForest parameters
                        rf_params = [
                            'randomforestclassifier__n_estimators',
                            'randomforestclassifier__max_depth',
                            'randomforestclassifier__min_samples_leaf',
                            'randomforestclassifier__min_samples_split',
                            'randomforestclassifier__max_samples',
                        ]
                        for param in rf_params:
                            if param in params:
                                # Shorten the key name
                                short_name = param.replace('randomforestclassifier__', '')
                                record[short_name] = params[param]

                        # Get OOB score if available
                        if 'randomforestclassifier' in params:
                            rf_model = params['randomforestclassifier']
                            if hasattr(rf_model, 'oob_score_'):
                                record['oob_score'] = rf_model.oob_score_

                    model_records.append(record)
            except Exception as e:
                logger.warning(f"Failed to load model from {joblib_file}: {e}")
                continue

        if not model_records:
            return None

        model_df = pd.DataFrame(model_records)
        model_df.sort_values('date', inplace=True)

        logger.info(
            f"Loaded {len(model_df)} model parameter records for {dir_name}/{model}/{horizon}"
        )

        return model_df

    def load_all_data(
        self, dir_name: str, model: str, horizon: str
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Load all data types for a model/horizon combination.

        Args:
            dir_name: Fits subdirectory name
            model: Model name
            horizon: Horizon value

        Returns:
            Dictionary with keys: 'fits', 'metrics', 'features', 'model_params'
        """
        return {
            'fits': self.load_fits_data(dir_name, model, horizon),
            'metrics': self.load_metrics_data(dir_name, model, horizon),
            'features': self.load_features_data(dir_name, model, horizon),
            'model_params': self.load_model_params(dir_name, model, horizon),
        }


class FitsComparisonReport2:
    """Dash application for comparing fits data between directories."""

    def __init__(self, port: int = DEFAULT_PORT, debug: bool = False):
        self.port = port
        self.debug = debug
        self.app = Dash(__name__)

        # Initialize data loader
        self.data_loader = FitsDataLoader()

        # Store loaded data - each dir has fits, metrics, features, model_params
        self.loaded_data: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {
            'dir1': {'fits': None, 'metrics': None, 'features': None, 'model_params': None},
            'dir2': {'fits': None, 'metrics': None, 'features': None, 'model_params': None},
        }

        # Setup layout and callbacks
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Create the Dash layout."""
        available_dirs = self.data_loader.available_dirs

        # Set default values
        default_dir1 = 'prod' if 'prod' in available_dirs else (
            available_dirs[0] if available_dirs else None
        )
        default_dir2 = 'dev' if 'dev' in available_dirs else (
            available_dirs[1] if len(available_dirs) > 1 else default_dir1
        )

        self.app.layout = html.Div([
            html.H1("Fits Comparison Report v2", style={'textAlign': 'center'}),

            # Directory selection section
            html.Div([
                html.H3("Step 1: Select Directories to Compare"),
                html.Div([
                    html.Div([
                        html.Label("Directory 1:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='dir1-selector',
                            options=[{'label': d, 'value': d} for d in available_dirs],
                            value=default_dir1,
                            placeholder="Select first directory",
                        ),
                    ], style={'width': '45%', 'display': 'inline-block'}),

                    html.Div([
                        html.Label("Directory 2:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='dir2-selector',
                            options=[{'label': d, 'value': d} for d in available_dirs],
                            value=default_dir2,
                            placeholder="Select second directory",
                        ),
                    ], style={'width': '45%', 'display': 'inline-block', 'marginLeft': '5%'}),
                ]),
            ], style={
                'padding': '20px',
                'backgroundColor': '#f0f0f0',
                'borderRadius': '5px',
                'marginBottom': '20px'
            }),

            # Model/Horizon selection section
            html.Div([
                html.H3("Step 2: Select Model/Horizon"),
                html.Label("Available combinations (present in both directories):"),
                dcc.Dropdown(
                    id='model-horizon-selector',
                    placeholder="Select model/horizon combination",
                    style={'marginTop': '10px'}
                ),
            ], style={
                'padding': '20px',
                'backgroundColor': '#f0f0f0',
                'borderRadius': '5px',
                'marginBottom': '20px'
            }),

            # Load data button
            html.Div([
                html.Button(
                    "Load Data",
                    id='load-data-btn',
                    n_clicks=0,
                    style={
                        'padding': '10px 30px',
                        'fontSize': '16px',
                        'backgroundColor': '#4CAF50',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '5px',
                        'cursor': 'pointer'
                    }
                ),
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),

            # Status message
            html.Div(id='status-message', style={
                'padding': '15px',
                'marginBottom': '20px',
                'borderRadius': '5px'
            }),

            # Data summary tables
            html.Div([
                html.H3("Loaded Data Summary"),
                html.Div([
                    html.Div([
                        html.H4(id='dir1-title'),
                        html.Div(id='dir1-summary'),
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                    html.Div([
                        html.H4(id='dir2-title'),
                        html.Div(id='dir2-summary'),
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
                ]),
            ], id='data-summary-section', style={'display': 'none'}),

            # T-Stat comparison charts section
            html.Div([
                html.H3("T-Stat Comparison Charts"),
                html.P("T-statistics over time by condition (mom/rev) and lag"),
                html.Div(id='tstat-charts-container'),
            ], id='charts-section', style={'display': 'none', 'marginTop': '20px'}),

            # Debug output section (visible when --debug flag is used)
            html.Div([
                html.H3("Debug: Raw Data Preview"),
                html.Div([
                    html.H4("Directory 1 Data (first 10 rows):"),
                    html.Pre(id='dir1-debug-output', style={
                        'backgroundColor': '#f8f8f8',
                        'padding': '10px',
                        'borderRadius': '5px',
                        'overflow': 'auto',
                        'maxHeight': '300px'
                    }),
                ]),
                html.Div([
                    html.H4("Directory 2 Data (first 10 rows):"),
                    html.Pre(id='dir2-debug-output', style={
                        'backgroundColor': '#f8f8f8',
                        'padding': '10px',
                        'borderRadius': '5px',
                        'overflow': 'auto',
                        'maxHeight': '300px'
                    }),
                ]),
            ], id='debug-section', style={'display': 'block' if self.debug else 'none'}),

            # Store for data (client-side)
            dcc.Store(id='data-store'),
        ])

    def _setup_callbacks(self):
        """Setup Dash callbacks."""

        @self.app.callback(
            Output('model-horizon-selector', 'options'),
            Output('model-horizon-selector', 'value'),
            Input('dir1-selector', 'value'),
            Input('dir2-selector', 'value'),
        )
        def update_model_horizon_options(dir1: str, dir2: str):
            """Update model/horizon dropdown based on selected directories."""
            if not dir1 or not dir2:
                return [], None

            overlapping = self.data_loader.get_overlapping_model_horizons(dir1, dir2)

            options = [
                {
                    'label': f"{model} @ {horizon}min",
                    'value': f"{model}|{horizon}"
                }
                for model, horizon in overlapping
            ]

            # Default to first option
            default_value = options[0]['value'] if options else None

            return options, default_value

        @self.app.callback(
            Output('status-message', 'children'),
            Output('status-message', 'style'),
            Output('data-summary-section', 'style'),
            Output('charts-section', 'style'),
            Output('dir1-title', 'children'),
            Output('dir2-title', 'children'),
            Output('dir1-summary', 'children'),
            Output('dir2-summary', 'children'),
            Output('tstat-charts-container', 'children'),
            Output('dir1-debug-output', 'children'),
            Output('dir2-debug-output', 'children'),
            Input('load-data-btn', 'n_clicks'),
            State('dir1-selector', 'value'),
            State('dir2-selector', 'value'),
            State('model-horizon-selector', 'value'),
            prevent_initial_call=True,
        )
        def load_and_display_data(
            n_clicks: int, dir1: str, dir2: str, model_horizon: str
        ):
            """Load data and update display."""
            if not model_horizon:
                return (
                    "Please select a model/horizon combination.",
                    {'backgroundColor': '#ffcccc', 'padding': '15px', 'borderRadius': '5px'},
                    {'display': 'none'},
                    {'display': 'none'},
                    "", "", "", "",
                    [],
                    "", ""
                )

            model, horizon = model_horizon.split('|')

            # Load all data from both directories
            self.loaded_data['dir1'] = self.data_loader.load_all_data(dir1, model, horizon)
            self.loaded_data['dir2'] = self.data_loader.load_all_data(dir2, model, horizon)

            df1 = self.loaded_data['dir1']['fits']
            df2 = self.loaded_data['dir2']['fits']

            # Create status message
            status_parts = []
            data1 = self.loaded_data['dir1']
            data2 = self.loaded_data['dir2']

            dir1_counts = []
            if data1['fits'] is not None:
                dir1_counts.append(f"fits:{len(data1['fits'])}")
            if data1['metrics'] is not None:
                dir1_counts.append(f"metrics:{len(data1['metrics'])}")
            if data1['features'] is not None:
                dir1_counts.append(f"features:{len(data1['features'])}")
            if data1['model_params'] is not None:
                dir1_counts.append(f"models:{len(data1['model_params'])}")

            if dir1_counts:
                status_parts.append(f"{dir1}: {', '.join(dir1_counts)}")
            else:
                status_parts.append(f"{dir1}: No data found")

            dir2_counts = []
            if data2['fits'] is not None:
                dir2_counts.append(f"fits:{len(data2['fits'])}")
            if data2['metrics'] is not None:
                dir2_counts.append(f"metrics:{len(data2['metrics'])}")
            if data2['features'] is not None:
                dir2_counts.append(f"features:{len(data2['features'])}")
            if data2['model_params'] is not None:
                dir2_counts.append(f"models:{len(data2['model_params'])}")

            if dir2_counts:
                status_parts.append(f"{dir2}: {', '.join(dir2_counts)}")
            else:
                status_parts.append(f"{dir2}: No data found")

            status_msg = f"Loaded {model} @ {horizon}min: " + " | ".join(status_parts)

            success = df1 is not None or df2 is not None
            status_style = {
                'backgroundColor': '#ccffcc' if success else '#ffcccc',
                'padding': '15px',
                'borderRadius': '5px',
                'marginBottom': '20px'
            }

            # Create summary tables with all data types
            dir1_summary = self._create_data_summary(self.loaded_data['dir1'])
            dir2_summary = self._create_data_summary(self.loaded_data['dir2'])

            # Debug output
            dir1_debug = self._create_debug_output(df1) if df1 is not None else "No data"
            dir2_debug = self._create_debug_output(df2) if df2 is not None else "No data"

            # Create t-stat comparison charts
            tstat_charts = self._create_tstat_charts(df1, df2, dir1, dir2)

            return (
                status_msg,
                status_style,
                {'display': 'block'},
                {'display': 'block'} if tstat_charts else {'display': 'none'},
                f"Directory: {dir1}",
                f"Directory: {dir2}",
                dir1_summary,
                dir2_summary,
                tstat_charts,
                dir1_debug,
                dir2_debug,
            )

    def _create_data_summary(
        self, data: Dict[str, Optional[pd.DataFrame]]
    ) -> html.Div:
        """Create a summary of all loaded data types."""
        sections = []

        # Fits data summary
        df = data.get('fits')
        if df is not None and len(df) > 0:
            fits_data = []
            fits_data.append({
                'Metric': 'Date Range',
                'Value': f"{df['date'].min().strftime('%Y-%m-%d')} to "
                         f"{df['date'].max().strftime('%Y-%m-%d')}"
            })
            fits_data.append({'Metric': 'Total Rows', 'Value': str(len(df))})
            fits_data.append({'Metric': 'Unique Dates', 'Value': str(df['date'].nunique())})

            if 'condition' in df.columns:
                conditions = df['condition'].unique()
                fits_data.append({
                    'Metric': 'Conditions',
                    'Value': ', '.join(sorted(conditions.astype(str)))
                })

            if 'lag' in df.columns:
                lags = sorted(df['lag'].unique())
                fits_data.append({
                    'Metric': 'Lags',
                    'Value': ', '.join(str(lag) for lag in lags)
                })

            if 'tstat' in df.columns:
                fits_data.append({
                    'Metric': 'Mean T-Stat',
                    'Value': f"{df['tstat'].mean():.4f}"
                })

            if 'nobs' in df.columns:
                fits_data.append({
                    'Metric': 'Mean N-Obs',
                    'Value': f"{df['nobs'].mean():,.0f}"
                })

            sections.append(html.Div([
                html.H5("Regression Fits", style={'marginTop': '10px'}),
                DataTable(
                    columns=[
                        {'name': 'Metric', 'id': 'Metric'},
                        {'name': 'Value', 'id': 'Value'},
                    ],
                    data=fits_data,
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'fontWeight': 'bold'},
                )
            ]))

        # Metrics data summary
        metrics_df = data.get('metrics')
        if metrics_df is not None and len(metrics_df) > 0:
            metrics_data = []
            metrics_data.append({
                'Metric': 'Records',
                'Value': str(len(metrics_df))
            })
            if 'insample' in metrics_df.columns:
                metrics_data.append({
                    'Metric': 'Latest In-Sample',
                    'Value': f"{metrics_df['insample'].iloc[-1]:.4f}"
                })
                metrics_data.append({
                    'Metric': 'Mean In-Sample',
                    'Value': f"{metrics_df['insample'].mean():.4f}"
                })
            if 'outsample' in metrics_df.columns:
                metrics_data.append({
                    'Metric': 'Latest Out-Sample',
                    'Value': f"{metrics_df['outsample'].iloc[-1]:.4f}"
                })
                metrics_data.append({
                    'Metric': 'Mean Out-Sample',
                    'Value': f"{metrics_df['outsample'].mean():.4f}"
                })

            sections.append(html.Div([
                html.H5("Classifier Metrics", style={'marginTop': '10px'}),
                DataTable(
                    columns=[
                        {'name': 'Metric', 'id': 'Metric'},
                        {'name': 'Value', 'id': 'Value'},
                    ],
                    data=metrics_data,
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'fontWeight': 'bold'},
                )
            ]))

        # Features data summary
        features_df = data.get('features')
        if features_df is not None and len(features_df) > 0:
            # Get feature columns (exclude 'date')
            feature_cols = [c for c in features_df.columns if c != 'date']

            # Get top 5 features by mean importance
            mean_importance = features_df[feature_cols].mean().sort_values(ascending=False)
            top_features = mean_importance.head(5)

            features_data = []
            features_data.append({
                'Metric': 'Records',
                'Value': str(len(features_df))
            })
            features_data.append({
                'Metric': 'Num Features',
                'Value': str(len(feature_cols))
            })

            for feat_name, importance in top_features.items():
                features_data.append({
                    'Metric': f'Top: {feat_name}',
                    'Value': f"{importance:.4f}"
                })

            sections.append(html.Div([
                html.H5("Feature Importance", style={'marginTop': '10px'}),
                DataTable(
                    columns=[
                        {'name': 'Metric', 'id': 'Metric'},
                        {'name': 'Value', 'id': 'Value'},
                    ],
                    data=features_data,
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'fontWeight': 'bold'},
                )
            ]))

        # Model params summary
        model_df = data.get('model_params')
        if model_df is not None and len(model_df) > 0:
            model_data = []
            model_data.append({
                'Metric': 'Models Loaded',
                'Value': str(len(model_df))
            })

            # Show latest model params
            latest = model_df.iloc[-1]
            param_cols = [c for c in model_df.columns if c != 'date']
            for col in param_cols[:5]:  # Show up to 5 params
                model_data.append({
                    'Metric': col,
                    'Value': str(latest[col])
                })

            sections.append(html.Div([
                html.H5("Model Parameters", style={'marginTop': '10px'}),
                DataTable(
                    columns=[
                        {'name': 'Metric', 'id': 'Metric'},
                        {'name': 'Value', 'id': 'Value'},
                    ],
                    data=model_data,
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'fontWeight': 'bold'},
                )
            ]))

        if not sections:
            return html.Div("No data available")

        return html.Div(sections)

    def _create_tstat_charts(
        self,
        df1: Optional[pd.DataFrame],
        df2: Optional[pd.DataFrame],
        dir1_name: str,
        dir2_name: str
    ) -> List[Any]:
        """Create t-stat comparison charts by condition and lag.

        Creates one chart per (condition, lag) combination, with both directories
        plotted on the same chart for easy comparison.

        Args:
            df1: Fits data from first directory
            df2: Fits data from second directory
            dir1_name: Name of first directory (for legend)
            dir2_name: Name of second directory (for legend)

        Returns:
            List of dcc.Graph components, one per chart
        """
        if df1 is None and df2 is None:
            return []

        # Combine dataframes to find all unique conditions and lags
        conditions = set()
        lags = set()

        for df in [df1, df2]:
            if df is not None and 'condition' in df.columns and 'lag' in df.columns:
                conditions.update(df['condition'].unique())
                lags.update(df['lag'].unique())

        if not conditions or not lags:
            return [html.P("No condition/lag data available for charts")]

        # Sort for consistent ordering
        conditions = sorted(conditions)
        lags = sorted(lags)

        charts = []

        # Create a chart for each condition/lag combination
        for condition in conditions:
            for lag in lags:
                fig = go.Figure()

                # Add trace for dir1
                if df1 is not None:
                    df1_filtered = df1[
                        (df1['condition'] == condition) & (df1['lag'] == lag)
                    ].copy()
                    if len(df1_filtered) > 0:
                        df1_filtered = df1_filtered.sort_values('date')
                        fig.add_trace(go.Scatter(
                            x=df1_filtered['date'],
                            y=df1_filtered['tstat'],
                            mode='lines+markers',
                            name=dir1_name,
                            line={'color': 'blue'},
                            marker={'size': 6}
                        ))

                # Add trace for dir2
                if df2 is not None:
                    df2_filtered = df2[
                        (df2['condition'] == condition) & (df2['lag'] == lag)
                    ].copy()
                    if len(df2_filtered) > 0:
                        df2_filtered = df2_filtered.sort_values('date')
                        fig.add_trace(go.Scatter(
                            x=df2_filtered['date'],
                            y=df2_filtered['tstat'],
                            mode='lines+markers',
                            name=dir2_name,
                            line={'color': 'red'},
                            marker={'size': 6}
                        ))

                # Add horizontal line at y=0 for reference
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

                # Update layout
                fig.update_layout(
                    title=f"T-Stat: {condition} (lag={lag})",
                    xaxis_title="Date",
                    yaxis_title="T-Statistic",
                    legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02},
                    height=350,
                    margin={'l': 50, 'r': 20, 't': 60, 'b': 50},
                    hovermode='x unified'
                )

                charts.append(
                    html.Div([
                        dcc.Graph(figure=fig)
                    ], style={
                        'width': '48%',
                        'display': 'inline-block',
                        'verticalAlign': 'top',
                        'marginBottom': '20px'
                    })
                )

        return charts

    def _create_debug_output(self, df: pd.DataFrame) -> str:
        """Create debug output showing first few rows of data."""
        if df is None:
            return "No data"

        # Show columns and first 10 rows
        output = f"Columns: {list(df.columns)}\n\n"
        output += df.head(10).to_string()
        return output

    def run(self):
        """Run the Dash application."""
        logger.info(f"Starting Fits Comparison Report v2 on port {self.port}")
        if self.debug:
            logger.info("Debug mode enabled - showing raw data preview")
        self.app.run(debug=False, port=self.port)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fits comparison report v2 - compare fits between directories'
    )
    parser.add_argument(
        '-p', '--port',
        help='Port to run the Dash app on',
        type=int,
        default=DEFAULT_PORT
    )
    parser.add_argument(
        '--debug',
        help='Enable debug mode to show raw data preview',
        action='store_true'
    )
    args = parser.parse_args()

    report = FitsComparisonReport2(port=args.port, debug=args.debug)
    report.run()


if __name__ == "__main__":
    main()
