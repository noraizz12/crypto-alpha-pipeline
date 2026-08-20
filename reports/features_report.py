#!/usr/bin/env python3
"""Interactive feature visualization dashboard for statistical arbitrage system.

This module provides a comprehensive web-based dashboard for exploring and analyzing
engineered features across multiple time horizons and trading pairs. It enables:

- Time series visualization of feature evolution
- Cross-sectional analysis at specific timestamps
- Statistical summaries and distributions
- Feature correlation analysis
- Multi-symbol and multi-horizon comparisons

The dashboard uses Plotly Dash for interactive visualizations and integrates with
the system's DataLoader for efficient feature data access.

Performance Note:
    Data is sampled at 60-minute intervals by default to improve loading speed
    and reduce memory usage. This provides sufficient granularity for feature
    analysis while maintaining responsive dashboard performance.

Usage:
    python features_report.py [--port PORT] [--debug]
"""

import argparse
import glob
import logging.config
import os
import warnings
from datetime import datetime as dt, timedelta as td, date
from typing import List, Optional

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import dcc, html, Input, Output, State, dash_table
from plotly.subplots import make_subplots
from scipy import stats

from lib.data.dataloader import DataLoader
from lib.util.config import get_config
from lib.util.logging_util import get_logging_config

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.config.dictConfig(get_logging_config("features_report"))

# Default port for all reports
REPORT_PORT = 8066
logger = logging.getLogger(__name__)

# Constants
DEFAULT_HORIZONS = [15, 60, 120, 360, 720, 1440, 4320, 10080, 43200]
DEFAULT_SYMBOLS = []  # Empty means all symbols
DATE_FORMAT = '%Y-%m-%d'

# Removed feature categories - now showing actual feature names from directory

# Global data loader instance
data_loader = None
config = None
_cached_date_range = None  # Cache for feature date range


def initialize_data_loader(config_file: Optional[str] = None) -> DataLoader:
    """Initialize the global data loader instance.
    
    Args:
        config_file: Path to configuration file. If None, uses default.
    
    Returns:
        Initialized DataLoader instance
    """
    global data_loader, config
    _, config = get_config(config_file)
    data_loader = DataLoader(config=config)
    return data_loader


def get_available_features(horizon: int) -> List[str]:
    """Get list of available features for a given horizon.

    Args:
        horizon: Time horizon in minutes

    Returns:
        List of feature names available for the horizon
    """
    if not data_loader:
        logger.warning("Data loader not initialized, returning empty feature list")
        return []

    # Read actual features from directory
    features_dir = f"{data_loader.dir_manager.FEATURES_DIR}/{horizon}"

    try:
        # List all subdirectories in the horizon directory
        if os.path.exists(features_dir):
            features = [d for d in os.listdir(features_dir)
                       if os.path.isdir(os.path.join(features_dir, d))]
            logger.info(f"Found {len(features)} features for horizon {horizon}")
            return sorted(features)
        else:
            logger.warning(f"Features directory not found: {features_dir}")
            return []
    except Exception as e:
        logger.error(f"Error reading features directory: {e}")
        return []


def get_feature_date_range(horizon: int = 1440) -> tuple[str, str]:
    """Get the date range from available feature files.

    Results are cached to avoid repeated file system scans on each page load.

    Args:
        horizon: Time horizon to check (default 1440)

    Returns:
        Tuple of (start_date, end_date) as strings in YYYY-MM-DD format
    """
    global _cached_date_range

    # Return cached result if available
    if _cached_date_range is not None:
        return _cached_date_range

    # Default to last 30 days if no data found
    default_end = dt.now().date()
    default_start = default_end - td(days=30)

    if not data_loader:
        return default_start.strftime(DATE_FORMAT), default_end.strftime(DATE_FORMAT)

    features_dir = data_loader.dir_manager.FEATURES_DIR

    try:
        # Get all parquet files from any feature directory for the horizon
        pattern = os.path.join(features_dir, str(horizon), '*', '*.parquet')
        files = glob.glob(pattern)

        if not files:
            logger.warning(f"No feature files found for horizon {horizon}")
            return default_start.strftime(DATE_FORMAT), default_end.strftime(DATE_FORMAT)

        # Extract dates from filenames (format: feature_name.YYYYMMDD.parquet)
        dates = []
        for f in files:
            basename = os.path.basename(f)
            parts = basename.split('.')
            # Need at least 3 parts: name.YYYYMMDD.parquet
            if len(parts) >= 3:
                date_str = parts[-2]  # Get the date part before .parquet
                if len(date_str) == 8 and date_str.isdigit():
                    try:
                        file_date = dt.strptime(date_str, '%Y%m%d').date()
                        dates.append(file_date)
                    except ValueError:
                        continue

        if not dates:
            logger.warning("Could not extract dates from feature files")
            return default_start.strftime(DATE_FORMAT), default_end.strftime(DATE_FORMAT)

        min_date = min(dates)
        max_date = max(dates)

        # Cap to last 30 days for performance - loading more data significantly
        # increases dashboard startup time. If data span < 30 days, use min_date.
        start_date = max(min_date, max_date - td(days=30))

        logger.info(f"Feature date range: {start_date} to {max_date}")
        _cached_date_range = (start_date.strftime(DATE_FORMAT), max_date.strftime(DATE_FORMAT))
        return _cached_date_range

    except Exception as e:
        logger.error(f"Error getting feature date range: {e}")
        return default_start.strftime(DATE_FORMAT), default_end.strftime(DATE_FORMAT)


def apply_universe_filter_to_features(df: pd.DataFrame, start_date: date, end_date: date, universe_filter: str) -> pd.DataFrame:
    """Apply universe filter to features dataframe on a day-by-day basis.
    
    Args:
        df: Features dataframe indexed by (ts, symbol_venue)
        start_date: Start date for filtering
        end_date: End date for filtering
        universe_filter: Filter name ('priceable', 'featureable', 'fittable', 'tradeable', 'expandable')
    
    Returns:
        Filtered dataframe with only symbols that pass the filter on each respective day
    """
    if not data_loader or universe_filter == 'all':
        return df
    
    try:
        logger.info(f"Applying {universe_filter} filter to features day-by-day")
        
        # Store filtered data for each day
        filtered_dfs = []
        
        # Process each day
        current_date = start_date
        while current_date <= end_date:
            # Load universe for this specific day
            universe_df = data_loader.load_universe_df(
                universe_date=current_date,
                filter_dead=True,
                convert_categorical=False
            )
            
            if universe_df is not None and not universe_df.empty:
                if universe_filter not in universe_df.columns:
                    logger.error(f"{current_date}: Column '{universe_filter}' not found in universe columns: {universe_df.columns.tolist()}")
                    continue
                    
                # Now we know the column exists
                # Get symbols that pass the filter for this day
                filtered_universe = universe_df[universe_df[universe_filter] == True]
                valid_symbols = set(filtered_universe.index)
                
                # Debug logging
                if len(valid_symbols) == 0:
                    logger.warning(f"{current_date}: No symbols pass {universe_filter} filter")
                    # Show some stats about the universe
                    filter_counts = universe_df[universe_filter].value_counts()
                    logger.info(f"{current_date}: {universe_filter} filter - True: {filter_counts.get(True, 0)}, False: {filter_counts.get(False, 0)}")
                else:
                    logger.info(f"{current_date}: {len(valid_symbols)} symbols pass {universe_filter} filter")
                
                # Get data for this day
                day_start = pd.Timestamp(current_date, tz='UTC')
                day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                
                # Filter the dataframe for this day and valid symbols
                day_mask = (df.index.get_level_values('ts') >= day_start) & (df.index.get_level_values('ts') <= day_end)
                day_data = df[day_mask]
                
                if not day_data.empty:
                    # Get unique symbols in the features data for this day
                    day_symbols = set(day_data.index.get_level_values('symbol_venue').unique())
                    logger.info(f"{current_date}: Features data has {len(day_symbols)} unique symbols")
                    
                    # Check for intersection
                    matching_symbols = day_symbols.intersection(valid_symbols)
                    if len(matching_symbols) == 0:
                        logger.warning(f"{current_date}: No overlap between feature symbols and {universe_filter} universe symbols")
                        # Log a few examples of each
                        feature_sample = list(day_symbols)[:5]
                        universe_sample = list(valid_symbols)[:5]
                        logger.info(f"Sample feature symbols: {feature_sample}")
                        logger.info(f"Sample universe symbols: {universe_sample}")
                        
                        # Check if it's just a venue suffix issue
                        if feature_sample and universe_sample:
                            logger.info(f"Feature symbol format: {feature_sample[0]}")
                            logger.info(f"Universe symbol format: {universe_sample[0]}")
                    
                    # Filter by symbols that pass the universe filter for this specific day
                    symbol_mask = day_data.index.get_level_values('symbol_venue').isin(valid_symbols)
                    filtered_day_data = day_data[symbol_mask]
                    
                    if not filtered_day_data.empty:
                        filtered_dfs.append(filtered_day_data)
                        logger.info(f"{current_date}: Kept {len(filtered_day_data)} of {len(day_data)} rows ({len(filtered_day_data.index.get_level_values('symbol_venue').unique())} symbols)")
                else:
                    logger.warning(f"{current_date}: No feature data found for this day")
            
            current_date += td(days=1)
        
        # Concatenate all filtered days
        if filtered_dfs:
            result = pd.concat(filtered_dfs)
            logger.info(f"After {universe_filter} filtering: {len(result)} rows, {len(result.index.get_level_values('symbol_venue').unique())} unique symbols")
            return result
        else:
            logger.warning(f"No data remained after applying {universe_filter} filter")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error applying universe filter: {e}", exc_info=True)
        return df  # Return original dataframe on error


def load_feature_data(
    features: List[str],
    symbols: Optional[List[str]],
    universe_filter: str,
    horizon: int,
    start_date: date,
    end_date: date,
    sample_freq: Optional[int] = 60
) -> pd.DataFrame:
    """Load feature data for specified parameters.
    
    Args:
        features: List of feature names to load
        symbols: Optional list of symbol_venue strings (None for all)
        universe_filter: Filter to apply ('all', 'priceable', 'featureable', 'fittable', 'tradeable', 'expandable')
        horizon: Time horizon in minutes
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        sample_freq: Sample every Nth minute (default: 60 for hourly samples)
    
    Returns:
        DataFrame with features indexed by (ts, symbol_venue)
    """
    if not data_loader:
        raise RuntimeError("Data loader not initialized")
    
    # Ensure features is a valid list with no None values
    if not features or not isinstance(features, list):
        logger.error(f"Invalid features: {features}")
        return pd.DataFrame()
    
    # Filter out None values
    features = [f for f in features if f is not None]
    if not features:
        logger.error("No valid features after filtering None values")
        return pd.DataFrame()
    
    # Load features using new format
    try:
        logger.info(f"Calling load_features with horizons=[{horizon}], start_date={start_date}, end_date={end_date}, cols={features}, symbol_venues={symbols}, sample_freq={sample_freq}")
        df = data_loader.load_features(
            horizons=[horizon],
            start_date=start_date,
            end_date=end_date,
            cols=features,
            symbol_venues=symbols,  # Load all symbols first if no specific symbols selected
            strict=False,
            fast=True,  # Speed up loading by skipping some validation
            sample_freq=sample_freq  # Sample every Nth minute for faster loading
        )
        logger.info(f"Loaded dataframe shape: {df.shape if df is not None else 'None'}")
    except Exception as e:
        logger.error(f"Error loading features: {e}", exc_info=True)
        return pd.DataFrame()
    
    if df is None or df.empty:
        logger.warning(f"No data found for features {features} from {start_date} to {end_date}")
        return pd.DataFrame()
    
    # Apply universe filter day-by-day if specified
    if universe_filter != 'all' and not symbols:
        logger.info(f"Applying {universe_filter} filter to loaded features")
        df = apply_universe_filter_to_features(df, start_date, end_date, universe_filter)
    
    return df


def create_layout() -> html.Div:
    """Create the dashboard layout.

    Returns:
        Dash HTML layout
    """
    # Get dynamic date range from available feature files
    start_date, end_date = get_feature_date_range()

    return html.Div([
        # Header
        html.Div([
            html.H1("Feature Analysis Dashboard", className="text-center mb-4"),
            html.Hr()
        ]),

        # Control Panel
        html.Div([
            # Row 1: Date range and horizon
            html.Div([
                html.Div([
                    html.Label("Date Range"),
                    dcc.DatePickerRange(
                        id='date-range-picker',
                        start_date=start_date,
                        end_date=end_date,
                        display_format='YYYY-MM-DD',
                        className="form-control"
                    )
                ], className="col-md-6"),
                
                html.Div([
                    html.Label("Time Horizon"),
                    dcc.Dropdown(
                        id='horizon-dropdown',
                        options=[{'label': f'{h} min', 'value': h} for h in DEFAULT_HORIZONS],
                        value=1440,
                        className="form-control"
                    )
                ], className="col-md-6"),
            ], className="row mb-3"),
            
            # Row 2: Symbol selection
            html.Div([
                html.Div([
                    html.Label("Symbols (leave empty for all)"),
                    dcc.Dropdown(
                        id='symbol-dropdown',
                        options=[],  # Will be populated by callback
                        value=[],  # Empty means all symbols
                        multi=True,
                        placeholder="Select symbols (default: all)",
                        className="form-control"
                    )
                ], className="col-md-12"),
            ], className="row mb-3"),
            
            # Row 3: Universe filter
            html.Div([
                html.Div([
                    html.Label("Universe Filter"),
                    dcc.Dropdown(
                        id='universe-filter-dropdown',
                        options=[
                            {'label': 'All Symbols', 'value': 'all'},
                            {'label': 'Priceable (>$25M ADVP)', 'value': 'priceable'},
                            {'label': 'Featureable (>$30M ADVP + 30d history)', 'value': 'featureable'},
                            {'label': 'Fittable (>$30M ADVP)', 'value': 'fittable'},
                            {'label': 'Tradeable (>$30M ADVP)', 'value': 'tradeable'},
                            {'label': 'Expandable (>$35M ADVP)', 'value': 'expandable'},
                        ],
                        value='all',
                        clearable=False,
                        className="form-control"
                    )
                ], className="col-md-12"),
            ], className="row mb-3"),
            
            # Row 4: Feature selection
            html.Div([
                html.Div([
                    html.Label("Features"),
                    dcc.Dropdown(
                        id='feature-dropdown',
                        options=[],  # Will be populated by callback
                        value=[],  # Empty default, will be set by callback
                        multi=True,
                        className="form-control"
                    )
                ], className="col-md-12"),
            ], className="row mb-3"),
            
            # Row 5: Load button and status
            html.Div([
                html.Div([
                    html.Button('Load Data', id='load-button', className='btn btn-primary'),
                    dcc.Loading(
                        id="loading-indicator",
                        type="default",
                        children=html.Span(id='load-status', className='ml-3')
                    )
                ], className="col-md-12"),
            ], className="row mb-3"),
        ], className="container mb-4"),
        
        # Tabs for different views
        dcc.Tabs(id='main-tabs', value='time-series', children=[
            dcc.Tab(label='Time Series', value='time-series'),
            dcc.Tab(label='Cross-Sectional', value='cross-section'),
            dcc.Tab(label='Statistics', value='statistics'),
            dcc.Tab(label='Correlation', value='correlation'),
        ]),
        
        # Tab content with loading indicator
        dcc.Loading(
            id="tab-loading",
            type="circle",
            children=html.Div(id='tab-content', className="container mt-4")
        ),
        
        # Hidden div to store loaded data
        html.Div(id='data-store', style={'display': 'none'})
    ])


# Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Feature Analysis Dashboard"
app.layout = create_layout  # Use function for dynamic layout on each page load


# Callbacks
@app.callback(
    Output('symbol-dropdown', 'options'),
    Input('horizon-dropdown', 'value')
)
def update_symbol_options(_):
    """Update available symbols based on selected horizon."""
    if not data_loader:
        return []
    
    # Get universe symbols from config
    if config and 'SYMBOL_UNIVERSE' in config:
        symbols = config['SYMBOL_UNIVERSE']
        # Add venue suffix
        universe_symbols = [f"{sym}_binance-futures" for sym in symbols]
    else:
        # Default symbols
        universe_symbols = DEFAULT_SYMBOLS
    
    options = [{'label': sym, 'value': sym} for sym in sorted(universe_symbols)]
    return options


@app.callback(
    [Output('feature-dropdown', 'options'),
     Output('feature-dropdown', 'value')],
    [Input('horizon-dropdown', 'value')],
    [State('feature-dropdown', 'value')]
)
def update_feature_options(horizon, current_features):
    """Update available features based on selected horizon."""
    if not horizon:
        return [], []
    
    features = get_available_features(horizon)
    
    # Safety check
    if not features:
        logger.warning(f"No features available for horizon {horizon}")
        return [], []
    
    # Create simple options list with actual feature names
    options = [{'label': f, 'value': f} for f in sorted(features)]
    
    # Set default value if none selected
    if not current_features:
        # Try to find RSI or beta as default
        default_features = []
        for f in features:
            if 'rsi' in f.lower() or 'beta' in f.lower():
                default_features.append(f)
                if len(default_features) >= 2:
                    break
        selected = default_features[:1] if default_features else []
        logger.info(f"Setting default features: {selected}")
        return options, selected
    
    # Filter out None values from current features
    current_features = [f for f in current_features if f is not None]
    return options, current_features


@app.callback(
    [Output('data-store', 'children'),
     Output('load-status', 'children')],
    [Input('load-button', 'n_clicks')],
    [State('feature-dropdown', 'value'),
     State('symbol-dropdown', 'value'),
     State('universe-filter-dropdown', 'value'),
     State('horizon-dropdown', 'value'),
     State('date-range-picker', 'start_date'),
     State('date-range-picker', 'end_date')]
)
def load_data(n_clicks, features, symbols, universe_filter, horizon, start_date, end_date):
    """Load feature data based on user selections."""
    if not n_clicks:
        return '', ''
    
    logger.info(f"load_data called with features={features}, type={type(features)}")
    
    # Filter out None values from features list
    if features:
        features = [f for f in features if f is not None]
    
    if not features:
        return '', 'Please select at least one feature'
    
    # If no symbols selected, use None to load all symbols
    if not symbols:
        symbols = None  # None means load all symbols
    else:
        # Ensure symbols have the venue suffix
        symbols = [sym if '_binance-futures' in sym else f"{sym}_binance-futures" for sym in symbols]
    
    try:
        # Parse dates
        start_date = dt.strptime(start_date, DATE_FORMAT).date()
        end_date = dt.strptime(end_date, DATE_FORMAT).date()
        
        # Log loading info
        logger.info(f"Loading features: {features}, symbols: {symbols}, universe_filter: {universe_filter}, horizon: {horizon}, dates: {start_date} to {end_date}")
        
        # Load data
        df = load_feature_data(
            features=features,
            symbols=symbols,
            universe_filter=universe_filter,
            horizon=horizon,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is None or df.empty:
            return '', 'No data found for selected parameters'
        
        # Log data size
        actual_symbols = df.index.get_level_values('symbol_venue').unique()
        logger.info(f"DataFrame has {len(df):,} rows, {len(actual_symbols)} symbols")
        
        # For large datasets, sample or aggregate to improve performance
        MAX_ROWS_FOR_DISPLAY = 500000  # Limit for interactive display
        if len(df) > MAX_ROWS_FOR_DISPLAY:
            logger.info(f"Dataset too large ({len(df):,} rows), sampling for display...")
            # Sample by taking every Nth row to maintain time series continuity
            sample_rate = max(1, len(df) // MAX_ROWS_FOR_DISPLAY)
            df = df.iloc[::sample_rate]
            logger.info(f"Sampled to {len(df):,} rows (every {sample_rate}th row)")
            status = f"Loaded {len(df):,} rows (sampled from {len(df) * sample_rate:,}) for {len(features)} features and {len(actual_symbols)} symbols"
            if universe_filter != 'all':
                status += f" (filtered by {universe_filter})"
        else:
            status = f"Loaded {len(df):,} rows for {len(features)} features and {len(actual_symbols)} symbols"
            if universe_filter != 'all':
                status += f" (filtered by {universe_filter})"
        
        # Store data more efficiently
        # For very large datasets, we could use parquet, but for now stick with JSON
        # but use more efficient parameters
        logger.info("Converting DataFrame to JSON...")
        try:
            # Reset index and use more efficient JSON format
            df_reset = df.reset_index()
            # Convert timestamps to strings to avoid serialization issues
            df_reset['ts'] = df_reset['ts'].astype(str)
            df_json = df_reset.to_json(orient='split', date_format='iso')
            logger.info("JSON conversion complete")
        except Exception as e:
            logger.error(f"Error converting to JSON: {e}")
            # If JSON fails, try with even more sampling
            logger.info("Retrying with more aggressive sampling...")
            df = df.iloc[::10]  # Take every 10th row
            df_reset = df.reset_index()
            df_reset['ts'] = df_reset['ts'].astype(str)
            df_json = df_reset.to_json(orient='split', date_format='iso')
            status = f"Loaded {len(df):,} rows (heavily sampled) for {len(features)} features and {len(actual_symbols)} symbols"
        
        return df_json, status
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return '', f'Error: {str(e)}'


@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('data-store', 'children')]
)
def render_tab_content(active_tab, data_json):
    """Render content based on selected tab and loaded data."""
    if not data_json:
        return html.Div("Please load data first", className="text-center mt-5")
    
    # Parse stored data
    df = pd.read_json(data_json, orient='split')
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index(['ts', 'symbol_venue'])
    
    if active_tab == 'time-series':
        return render_time_series_view(df)
    elif active_tab == 'cross-section':
        return render_cross_section_view(df)
    elif active_tab == 'statistics':
        return render_statistics_view(df)
    elif active_tab == 'correlation':
        return render_correlation_view(df)
    
    return html.Div()


def render_time_series_view(df: pd.DataFrame) -> html.Div:
    """Render time series visualizations.
    
    Args:
        df: Feature data DataFrame
    
    Returns:
        Dash HTML component with time series plots
    """
    
    # Options for visualization
    viz_options = html.Div([
        html.Div([
            html.Div([
                html.Label("Plot Type:"),
                dcc.RadioItems(
                    id='ts-plot-type',
                    options=[
                        {'label': 'Separate Features', 'value': 'separate'},
                        {'label': 'Combined (Normalized)', 'value': 'combined'},
                        {'label': 'Feature Comparison', 'value': 'comparison'}
                    ],
                    value='separate',
                    inline=True
                )
            ], className="col-md-4"),
            
            html.Div([
                html.Label("Rolling Window:"),
                dcc.Input(
                    id='ts-rolling-window',
                    type='number',
                    value=0,
                    min=0,
                    max=100,
                    placeholder="0 for none"
                )
            ], className="col-md-4"),
            
            html.Div([
                html.Label("Options:"),
                dcc.Checklist(
                    id='ts-show-percentiles',
                    options=[
                        {'label': 'Show 25/75 percentiles', 'value': 'percentiles'},
                        {'label': 'Force aggregated view', 'value': 'aggregate'}
                    ],
                    value=[]
                )
            ], className="col-md-4"),
        ], className="row mb-3")
    ])
    
    return html.Div([
        viz_options,
        html.Hr(),
        html.Div(id='ts-summary-stats'),
        html.Div(id='ts-plots-container')
    ])


@app.callback(
    [Output('ts-plots-container', 'children'),
     Output('ts-summary-stats', 'children')],
    [Input('ts-plot-type', 'value'),
     Input('ts-rolling-window', 'value'),
     Input('ts-show-percentiles', 'value'),
     Input('data-store', 'children')]
)
def update_time_series_plots(plot_type, rolling_window, show_percentiles, data_json):
    """Update time series plots based on visualization options."""
    if not data_json:
        return html.Div(), html.Div()
    
    # Parse data
    df = pd.read_json(data_json, orient='split')
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index(['ts', 'symbol_venue'])
    
    features = [col for col in df.columns if col not in ['symbol', 'date']]
    symbols = df.index.get_level_values('symbol_venue').unique()
    
    # Create summary statistics
    summary_stats = create_time_series_summary(df, features, symbols)
    
    # Create plots
    if plot_type == 'separate':
        plots = create_separate_feature_plots(df, features, symbols, rolling_window, show_percentiles)
    elif plot_type == 'combined':
        plots = create_combined_feature_plot(df, features, symbols)
    elif plot_type == 'comparison':
        plots = create_feature_comparison_plot(df, features, symbols)
    else:
        plots = html.Div()
    
    return plots, summary_stats


def create_time_series_summary(df, features, symbols):
    """Create summary statistics for time series view."""
    latest_ts = df.index.get_level_values('ts').max()
    
    # Calculate current statistics
    summary_data = []
    for feature in features:
        latest_data = df.xs(latest_ts, level='ts')[feature].dropna()
        if len(latest_data) > 0:
            summary_data.append({
                'Feature': feature,
                'Mean': latest_data.mean(),
                'Std': latest_data.std(),
                'Min': latest_data.min(),
                'Max': latest_data.max(),
                'Count': len(latest_data),
                'NaN%': (df.xs(latest_ts, level='ts')[feature].isna().sum() / len(symbols) * 100)
            })
    
    if not summary_data:
        return html.Div()
    
    summary_df = pd.DataFrame(summary_data)
    
    # Format numeric columns
    for col in ['Mean', 'Std', 'Min', 'Max', 'NaN%']:
        summary_df[col] = summary_df[col].round(4)
    
    return html.Div([
        html.H5(f"Current Statistics (as of {latest_ts})"),
        dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in summary_df.columns],
            data=summary_df.to_dict('records'),
            style_cell={'textAlign': 'left'},
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            }
        )
    ], className="mb-4")


def create_separate_feature_plots(df, features, symbols, rolling_window, show_percentiles):
    """Create separate subplot for each feature."""
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=features,
        vertical_spacing=0.1
    )
    
    # Check if we should show aggregated view
    force_aggregate = show_percentiles and 'aggregate' in show_percentiles
    show_individual = len(symbols) <= 10 and not force_aggregate
    
    # Plot each feature
    for i, feature in enumerate(features):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        if show_individual:
            # Show individual symbol lines for small number of symbols
            for symbol in symbols:
                symbol_data = df.xs(symbol, level='symbol_venue')[feature].dropna()
                
                if len(symbol_data) == 0:
                    continue
                
                # Apply rolling window if specified
                if rolling_window and rolling_window > 0:
                    symbol_data = symbol_data.rolling(rolling_window).mean()
                
                # Main line
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data.index,
                        y=symbol_data.values,
                        name=f"{symbol}",
                        legendgroup=symbol,
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )
        else:
            # Show aggregated statistics for many symbols
            # Calculate mean and std across symbols
            feature_mean = df[feature].groupby('ts').mean()
            feature_std = df[feature].groupby('ts').std()
            feature_median = df[feature].groupby('ts').median()
            
            # Apply rolling window if specified
            if rolling_window and rolling_window > 0:
                feature_mean = feature_mean.rolling(rolling_window).mean()
                feature_std = feature_std.rolling(rolling_window).mean()
                feature_median = feature_median.rolling(rolling_window).mean()
            
            # Plot mean line
            fig.add_trace(
                go.Scatter(
                    x=feature_mean.index,
                    y=feature_mean.values,
                    name='Mean',
                    line=dict(color='blue', width=2),
                    legendgroup='mean',
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
            
            # Plot median line
            fig.add_trace(
                go.Scatter(
                    x=feature_median.index,
                    y=feature_median.values,
                    name='Median',
                    line=dict(color='green', width=2, dash='dash'),
                    legendgroup='median',
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
            
            # Add ±1 std bands
            fig.add_trace(
                go.Scatter(
                    x=feature_mean.index,
                    y=(feature_mean + feature_std).values,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=feature_mean.index,
                    y=(feature_mean - feature_std).values,
                    fill='tonexty',
                    fillcolor='rgba(0, 100, 200, 0.2)',
                    line=dict(width=0),
                    name='±1 std',
                    legendgroup='std',
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
        
        # Add percentile bands if requested (works for both modes)
        if show_percentiles and 'percentiles' in show_percentiles:
            data_25 = df[feature].groupby('ts').quantile(0.25)
            data_75 = df[feature].groupby('ts').quantile(0.75)
            
            if rolling_window and rolling_window > 0:
                data_25 = data_25.rolling(rolling_window).mean()
                data_75 = data_75.rolling(rolling_window).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=data_25.index,
                    y=data_25.values,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data_75.index,
                    y=data_75.values,
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.2)',
                    line=dict(width=0),
                    showlegend=False,
                    name='25-75 percentile'
                ),
                row=row, col=col
            )
    
    title = f"Feature Time Series ({len(symbols)} symbols)"
    if not show_individual:
        title += " - Aggregated View"
    
    fig.update_layout(
        height=400 * n_rows,
        title=title,
        showlegend=True
    )
    
    return html.Div([
        dcc.Graph(figure=fig, style={'height': f'{400 * n_rows}px'})
    ])


def create_combined_feature_plot(df, features, symbols):
    """Create combined plot with normalized features."""
    fig = go.Figure()
    
    # Normalize each feature to [0, 1] range
    for feature in features:
        feature_data = df[feature]
        normalized = (feature_data - feature_data.min()) / (feature_data.max() - feature_data.min())
        
        for symbol in symbols:
            symbol_data = normalized.xs(symbol, level='symbol_venue')
            fig.add_trace(
                go.Scatter(
                    x=symbol_data.index,
                    y=symbol_data.values,
                    name=f"{feature} ({symbol})",
                    mode='lines'
                )
            )
    
    fig.update_layout(
        height=600,
        title="Normalized Feature Comparison",
        yaxis_title="Normalized Value [0, 1]",
        showlegend=True
    )
    
    return html.Div([
        dcc.Graph(figure=fig, style={'height': '600px'})
    ])


def create_feature_comparison_plot(df, features, _):
    """Create feature comparison with correlation info."""
    # Calculate feature correlations
    corr_matrix = df[features].corr()
    
    # Create scatter matrix
    fig = px.scatter_matrix(
        df[features].reset_index(),
        dimensions=features,
        title="Feature Scatter Matrix"
    )
    
    fig.update_layout(height=800)
    
    # Add correlation heatmap
    corr_fig = px.imshow(
        corr_matrix,
        labels=dict(x="Features", y="Features", color="Correlation"),
        x=features,
        y=features,
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1,
        title="Feature Correlation Heatmap"
    )
    
    return html.Div([
        html.Div([
            dcc.Graph(figure=fig, style={'height': '800px'})
        ], className="mb-4"),
        html.Div([
            dcc.Graph(figure=corr_fig, style={'height': '600px'})
        ])
    ])


def render_cross_section_view(df: pd.DataFrame) -> html.Div:
    """Render cross-sectional analysis view.
    
    Args:
        df: Feature data DataFrame
    
    Returns:
        Dash HTML component with cross-sectional plots
    """
    # Get unique timestamps
    timestamps = df.index.get_level_values('ts').unique()
    latest_ts = timestamps[-1]
    
    return html.Div([
        html.H3("Cross-Sectional Analysis"),
        html.Hr(),
        
        # Timestamp selector
        html.Div([
            html.Label("Select Timestamp:"),
            dcc.Dropdown(
                id='cs-timestamp-dropdown',
                options=[{'label': str(ts), 'value': str(ts)} for ts in timestamps[-20:]],
                value=str(latest_ts)
            )
        ], className="mb-3"),
        
        # Plots container
        html.Div(id='cs-plots-container')
    ])


@app.callback(
    Output('cs-plots-container', 'children'),
    [Input('cs-timestamp-dropdown', 'value'),
     Input('data-store', 'children')]
)
def update_cross_section_plots(timestamp, data_json):
    """Update cross-sectional plots based on selected timestamp."""
    if not timestamp or not data_json:
        return html.Div()
    
    # Parse data
    df = pd.read_json(data_json, orient='split')
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index(['ts', 'symbol_venue'])
    
    # Filter for selected timestamp
    ts = pd.to_datetime(timestamp)
    cs_data = df.xs(ts, level='ts')
    
    features = [col for col in df.columns if col not in ['symbol', 'date']]
    
    # Create distribution plots
    plots = []
    for feature in features:
        # Histogram
        fig_hist = px.histogram(
            cs_data, x=feature,
            title=f"{feature} Distribution at {ts}",
            nbins=30
        )
        
        # Box plot
        fig_box = px.box(
            cs_data.reset_index(), y=feature,
            title=f"{feature} Box Plot at {ts}",
            points="all"
        )
        
        plots.append(html.Div([
            html.Div([
                dcc.Graph(figure=fig_hist)
            ], className="col-md-6"),
            html.Div([
                dcc.Graph(figure=fig_box)
            ], className="col-md-6")
        ], className="row mb-3"))
    
    return html.Div(plots)


def render_statistics_view(df: pd.DataFrame) -> html.Div:
    """Render statistical summary view.
    
    Args:
        df: Feature data DataFrame
    
    Returns:
        Dash HTML component with statistical summaries
    """
    features = [col for col in df.columns if col not in ['symbol', 'date']]
    
    # Calculate statistics by symbol
    stats_by_symbol = []
    for symbol in df.index.get_level_values('symbol_venue').unique():
        symbol_data = df.xs(symbol, level='symbol_venue')
        
        for feature in features:
            feature_series = symbol_data[feature].dropna()
            if len(feature_series) > 0:
                feature_stats = {
                    'Symbol': symbol,
                    'Feature': feature,
                    'Count': feature_series.count(),
                    'Mean': feature_series.mean(),
                    'Std': feature_series.std(),
                    'Min': feature_series.min(),
                    '25%': feature_series.quantile(0.25),
                    '50%': feature_series.quantile(0.50),
                    '75%': feature_series.quantile(0.75),
                    'Max': feature_series.max(),
                    'Skew': feature_series.skew(),
                    'Kurt': feature_series.kurtosis(),
                    'NaN%': (symbol_data[feature].isna().sum() / len(symbol_data) * 100)
                }
                
                # Add normality test
                if len(feature_series) >= 20:
                    _, p_value = stats.normaltest(feature_series)
                    feature_stats['Normality_p'] = p_value
                else:
                    feature_stats['Normality_p'] = np.nan
                    
                stats_by_symbol.append(feature_stats)
    
    stats_df = pd.DataFrame(stats_by_symbol)
    
    # Format numeric columns
    numeric_cols = ['Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Skew', 'Kurt', 'NaN%', 'Normality_p']
    for col in numeric_cols:
        if col in stats_df.columns:
            stats_df[col] = stats_df[col].round(4)
    
    # Create download link
    csv_string = stats_df.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8," + csv_string
    
    return html.Div([
        html.Div([
            html.H3("Feature Statistics by Symbol", className="d-inline-block"),
            html.A(
                'Download CSV',
                id='download-stats',
                download="feature_statistics.csv",
                href=csv_string,
                className="btn btn-secondary float-right"
            )
        ], className="clearfix"),
        html.Hr(),
        
        # Summary cards
        html.Div([
            html.Div([
                html.Div([
                    html.H5("Total Features"),
                    html.H3(len(features))
                ], className="card-body text-center")
            ], className="col-md-3 card"),
            
            html.Div([
                html.Div([
                    html.H5("Total Symbols"),
                    html.H3(len(df.index.get_level_values('symbol_venue').unique()))
                ], className="card-body text-center")
            ], className="col-md-3 card"),
            
            html.Div([
                html.Div([
                    html.H5("Date Range"),
                    html.P(f"{df.index.get_level_values('ts').min().date()} to {df.index.get_level_values('ts').max().date()}")
                ], className="card-body text-center")
            ], className="col-md-3 card"),
            
            html.Div([
                html.Div([
                    html.H5("Total Data Points"),
                    html.H3(f"{len(df):,}")
                ], className="card-body text-center")
            ], className="col-md-3 card"),
        ], className="row mb-4"),
        
        # Statistics table
        dash_table.DataTable(
            id='stats-table',
            columns=[{"name": i, "id": i} for i in stats_df.columns],
            data=stats_df.to_dict('records'),
            filter_action="native",
            sort_action="native",
            page_action="native",
            page_size=20,
            style_cell={'textAlign': 'left'},
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                    'if': {
                        'filter_query': '{Normality_p} < 0.05',
                        'column_id': 'Normality_p'
                    },
                    'backgroundColor': '#ff9999',
                    'color': 'black',
                }
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            export_format='csv'
        ),
        
        # Feature distribution plots
        html.Hr(),
        html.H4("Feature Distributions"),
        html.Div(id='feature-dist-plots')
    ])


@app.callback(
    Output('feature-dist-plots', 'children'),
    Input('data-store', 'children')
)
def update_feature_distributions(data_json):
    """Generate feature distribution plots."""
    if not data_json:
        return html.Div()
    
    # Parse data
    df = pd.read_json(data_json, orient='split')
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index(['ts', 'symbol_venue'])
    
    features = [col for col in df.columns if col not in ['symbol', 'date']]
    
    # Create distribution plots for first 6 features
    plots = []
    for feature in features[:6]:
        # Create histogram with KDE
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=df[feature].dropna(),
            name='Histogram',
            nbinsx=50,
            histnorm='probability density'
        ))
        
        # Add KDE
        kde_data = df[feature].dropna()
        if len(kde_data) > 10:
            kde_x = np.linspace(kde_data.min(), kde_data.max(), 100)
            kde = stats.gaussian_kde(kde_data)
            kde_y = kde(kde_x)
            
            fig.add_trace(go.Scatter(
                x=kde_x,
                y=kde_y,
                mode='lines',
                name='KDE',
                line=dict(color='red', width=2)
            ))
        
        fig.update_layout(
            title=f"{feature} Distribution",
            xaxis_title=feature,
            yaxis_title="Density",
            height=300
        )
        
        plots.append(html.Div([
            dcc.Graph(figure=fig)
        ], className="col-md-6"))
    
    return html.Div(plots, className="row")


def render_correlation_view(df: pd.DataFrame) -> html.Div:
    """Render feature correlation analysis.
    
    Args:
        df: Feature data DataFrame
    
    Returns:
        Dash HTML component with correlation matrices
    """
    features = [col for col in df.columns if col not in ['symbol', 'date']]
    
    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        labels=dict(x="Features", y="Features", color="Correlation"),
        x=features,
        y=features,
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1,
        title="Feature Correlation Matrix"
    )
    
    fig.update_layout(height=800)
    
    return html.Div([
        html.H3("Feature Correlations"),
        html.Hr(),
        dcc.Graph(figure=fig, style={'height': '800px'})
    ])


def main():
    """Main entry point for the feature report dashboard."""
    parser = argparse.ArgumentParser(description='Feature Analysis Dashboard')
    parser.add_argument('--port', type=int, default=REPORT_PORT, help='Port to run the dashboard on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    
    args = parser.parse_args()
    
    # Initialize data loader
    initialize_data_loader(args.config)
    
    # Run the app
    app.run(debug=args.debug, host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    main()