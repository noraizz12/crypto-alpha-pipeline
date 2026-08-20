"""Production model fitting reports and visualization.

This module provides reporting functionality for monitoring production model
training status, coefficient values, and statistical significance.
"""

import logging
from datetime import datetime as dt
from datetime import timedelta as td, timezone
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from dash import dcc, html

from lib.data.dataloader import DataLoader
from lib.util.config import extract_horizon_models, get_config
from lib.util.time_util import today
from lib.util.directory import dir_manager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ProdFitsReports:
    """Reports for production model fitting and training status.

    Monitors the status of model training, coefficient values, and
    statistical significance. Provides visualization of model parameters
    and training schedules across different horizons.

    Attributes:
        data_loader: DataLoader instance for data access
        config: Main system configuration
        horizon_models: List of (horizon, model) tuples
        horizons: Sorted list of unique horizons
        update_ts: Timestamp of last data update

    Example:
        >>> pfr = ProdFitsReports()
        >>> status = pfr.check_model_status()
        >>> charts = pfr.get_plots()
    """
    def __init__(self):
        """Initialize with shared resources"""
        # Initialize config and data loader
        _, self.config = get_config()
        self.data_loader = DataLoader(config=self.config)

        # Extract horizon models info
        self.horizon_models = extract_horizon_models(self.config, exclude_zero_weight=True)
        self.horizons = sorted({h for h, _ in self.horizon_models})

        # Configuration
        # Increased to 150 days to capture sufficient 43200 horizon data points
        # (43200 fits files are generated less frequently, ~monthly)
        self.lookback_days = 150

        # Update timestamp
        self.update_ts = dt.now(timezone.utc)

        self.update_data()

    def update_data(self):
        """Update fits data"""
        try:
            # Calculate start and end dates
            start_date = today() - td(days=self.lookback_days)
            end_date = today()

            # Pass horizon_models directly to load_fits to only load needed combinations
            self.fits_df = self.data_loader.load_fits(
                horizon_models=self.horizon_models,
                prod=True,
                start_date=start_date,
                end_date=end_date,
                fits_dir=dir_manager.FITS_DIR_PROD,
            )
            logger.info(f"Loaded fits_df shape: {self.fits_df.shape}")
            logger.info(f"Date range: {self.fits_df['as_of'].min()} to {self.fits_df['as_of'].max()}")
            logger.info(f"Unique models: {self.fits_df['name'].unique()}")
            logger.info(f"Unique horizons: {self.fits_df['horizon'].unique()}")
            logger.info(f"Unique conditions: {self.fits_df['condition'].unique()}")
        except FileNotFoundError:
            logger.warning(f"Fits directory not found: {dir_manager.FITS_DIR_PROD}. Initializing empty DataFrame.")
            # Initialize empty DataFrame with expected columns
            self.fits_df = pd.DataFrame(columns=['as_of', 'horizon', 'name', 'condition', 'lag', 'tstat'])
        except Exception as e:
            logger.error(f"Error loading fits data: {e}. Initializing empty DataFrame.")
            self.fits_df = pd.DataFrame(columns=['as_of', 'horizon', 'name', 'condition', 'lag', 'tstat'])

        # Update timestamp
        self.update_ts = dt.now(timezone.utc)

    def get_ts_display(self, n_state: str) -> str:
        logger.info(f"finish get_ts_display for update {n_state} at {dt.now(timezone.utc)}")
        return f'Data as of: {self.update_ts.strftime("%Y%m%d %H:%M")}, dashboard refreshed at {dt.now(timezone.utc).strftime("%Y%m%d %H:%M")}'

    def update_all_fits_t_figures(self, _):
        horizon_figures = self.update_fits_t_figure()
        # Create a list of dcc.Graph components, one for each horizon
        graph_components = []
        for horizon, figure in horizon_figures.items():
            graph_components.append(
                html.Div([
                    html.H4(f"Horizon {horizon}", style={'textAlign': 'center'}),
                    dcc.Graph(figure=figure),
                ], style={'marginBottom': '30px'}),
            )

        return graph_components

    def update_fits_t_figure(self) -> Dict[int, go.Figure]:
        horizon_figures = {}
        # Condition color schemes
        condition_colors = {
            'mom': ['rgba(0, 0, 255, 1)', 'rgba(0, 128, 255, 1)', 'rgba(75, 0, 130, 1)'],
            'rev': ['rgba(255, 0, 0, 1)', 'rgba(255, 99, 71, 1)', 'rgba(128, 0, 0, 1)'],
        }
        legend_items = set()

        # Calculate 90-day cutoff date
        cutoff_date = today() - td(days=self.lookback_days)

        # Filter fits_df to only include data within the last 90 days
        filtered_df = self.fits_df[self.fits_df['as_of'] >= cutoff_date].copy()

        # Generate a separate figure for each horizon
        for horizon in self.horizons:
            # Get only models that have data for this specific horizon (filtered to 90 days)
            horizon_df = filtered_df[filtered_df.horizon == int(horizon)]
            models = sorted(horizon_df['name'].unique()) if not horizon_df.empty else []

            if not models:
                # Skip horizons with no models
                continue

            n_models = len(models)
            cols = min(3, n_models)  # Max 3 columns
            rows = (n_models + cols - 1) // cols  # Ceiling division for number of rows
            fig = sp.make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[f"Model: {model}" for model in models],
                shared_yaxes=False,
                shared_xaxes=False,
                horizontal_spacing=0.12,
                vertical_spacing=0.2,
            )

            # For each model (subplot)
            for m_idx, model in enumerate(models):
                row = (m_idx // cols) + 1
                col = (m_idx % cols) + 1

                for condition, colors in condition_colors.items():
                    df = filtered_df.loc[
                        (filtered_df.horizon == int(horizon)) &
                        (filtered_df.name == model) &
                        (filtered_df.condition == condition)
                    ]
                    if df.empty:
                        continue
                    df = df.sort_values(['as_of', 'lag']).drop_duplicates(['as_of', 'lag'], keep='first')
                    df = df.pivot(index='as_of', columns='lag', values='tstat')
                    for lag_idx, lag in enumerate(df.columns):
                        color = colors[lag_idx % len(colors)]
                        legend_key = f"{condition}_lag{lag}"
                        first_appearance = legend_key not in legend_items
                        if first_appearance:
                            legend_items.add(legend_key)
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df[lag],
                                mode='lines',
                                name=f"{condition}_lag{lag}",
                                line={"color": color},
                                legendgroup=legend_key,
                                showlegend=first_appearance,
                            ),
                            row=row, col=col,
                        )
            # Update layout for this horizon's figure
            fig.update_layout(
                title=f"T-stat Time Series for Horizon {horizon}",
                height=800,
                width=max(1500, 400 * cols),
                legend={
                    "groupclick": "toggleitem",
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.1,
                    "xanchor": "center",
                    "x": 0.5,
                },
            )

            for r in range(1, rows + 1):
                fig.update_yaxes(title_text="T-stat", row=r, col=1)
            for c in range(1, cols + 1):
                fig.update_xaxes(title_text="As of", row=rows, col=c)
            horizon_figures[int(horizon)] = fig

        logger.info(f"finish update_fits_t_figure for update at {dt.now(timezone.utc)}")
        return horizon_figures
