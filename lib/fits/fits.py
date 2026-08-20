"""Model fitting and regression module for the statistical arbitrage trading system.

This module handles the fitting of predictive models across multiple time horizons,
including classifiers for regime detection and linear
regressions for alpha signal generation. It implements a sophisticated two-stage
approach: first classifying market regimes (momentum vs mean-reversion), then
fitting separate models for each regime.

Key functionality:
- Fits alpha models with lagged dependent variables across time horizons
- Implements classifiers (SVM, Random Forest) to detect momentum/mean-reversion regimes
- Supports both cross-sectional and security-specific model fitting
- Handles feature selection with L1 regularization and cross-validation
- Manages rolling window refitting for adaptive model parameters
- Persists fitted models and classifiers for production deployment

The fitting process uses regularized regressions with HAC standard errors
for robust inference, and supports various return types including raw,
market-residualized, and funding-adjusted returns.

Note: Random seed is fixed at 42 for reproducibility across all random operations.
"""

import glob
import logging
from datetime import date, datetime as dt
from datetime import timedelta as td
from multiprocessing.dummy import Pool
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from lib.alpha.models import Models
from lib.calcs.calc_util import make_cx_features, make_market_weight_col
from lib.calcs.calcs import Calcs
from lib.data.dataloader import DataLoader
from lib.universe import Universe
from lib.util.config import extract_horizons, extract_max_lags, extract_models, extract_all_tree_features_horizon, extract_tree_features_horizon_model
from lib.util.dataframes import get_min_max_ts, merge_on_index
from lib.util.directory import DirectoryManager, dir_manager
from lib.util.files import safe_mkdir
from lib.util.logging_util import KeyLogger
from lib.util.time_util import compute_lookback_days, date_str_to_date, date_to_end_dt, date_to_start_dt, date_to_str, today_date
from lib.util.util import log_mem_usage, unique_list, extract_horizon_from_feature, log_and_raise, SYMBOL_VENUE
from .fit_util import make_classification_bar_features, MIN_FITTING_OBS
from .model_horizon import ModelHorizonFit

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)

pd.options.mode.chained_assignment = None

# Set random seed for reproducibility
np.random.seed(42)


def _remove_missing_features(classification_features: List[str], models_df: pd.DataFrame) -> List[str]:
    new_classification_features = []
    seen_features = set()
    for feature in classification_features:
        if feature.startswith('cx.'):
            feature = feature.split('.')[1]
        if feature in seen_features:
            # Skip duplicate features (can occur when cx.X and X both exist in config)
            continue
        seen_features.add(feature)
        if feature in models_df.columns:
            new_classification_features.append(feature)
        else:
            # just warn since we don't have a full history on some features.
            logger.warning(f"Missing classification feature: {feature=}")
    return new_classification_features

class Fits:
    """Manages the fitting of alpha models across multiple horizons and models.
    
    This is the main orchestrator class for the model fitting pipeline. It handles
    data loading, rolling window generation, parallel model fitting, and persistence
    of results. The class supports both development and production modes with
    different fitting schedules and output directories.
    
    The fitting process involves:
    1. Loading historical bar, feature, and forward return data
    2. Creating rolling fitting windows based on configuration
    3. Running ModelHorizonFit for each model/horizon combination
    4. Persisting fitted coefficients and SVM classifiers
    5. Managing refit schedules for production deployment
    
    Attributes:
        config: System configuration dictionary
        prod: Production mode flag - affects output directory and refit logic
        debug: Debug mode for additional logging
        force: Force refitting even if recent fits exist
        regenerate: Regenerate all fits from scratch
        dir_manager: Directory manager for data paths
        data_loader: DataLoader instance for retrieving data
        fits_dir: Output directory for fitted models
        models: Models instance for alpha calculations
        calcs: Calcs instance for return calculations
        classification_history_days: Lookback for classification data
        svm_regularization: L1 regularization parameter for SVM
        pool_size: Number of parallel processes for fitting
        horizons: List of time horizons to fit
        models_to_run: List of model names to fit
        model_regularization: Per-model regularization parameters
    """

    def __init__(
            self,
            config: dict,
            prod: bool = False,
            horizons: Optional[List[int]] = None,
            models: Optional[List[str]] = None,
            pool_size: int = 4,
            debug: bool = False,
            force: bool = False,
            regenerate: bool = False,
            classification_history_days: Optional[int] = None,
            fits_dir_manager: DirectoryManager = dir_manager,
            base_fits_dir: Optional[str] = None,
    ):
        self.config = config
        self.prod = prod
        self.debug = debug
        self.force = force
        self.regenerate = regenerate
        self.dir_manager = fits_dir_manager
        self.data_loader = DataLoader(self.config, self.dir_manager)
        self.universe = Universe(config=self.config, debug=self.debug, universe_dir_manager=self.dir_manager)
        if base_fits_dir is not None:
            self.fits_dir = base_fits_dir
        elif prod:
            self.fits_dir = self.dir_manager.FITS_DIR_PROD
        else:
            self.fits_dir = self.dir_manager.FITS_DIR_DEV

        self.classifier_dir = self.fits_dir
        self.models = Models(self.config)
        self.calcs = Calcs(self.config)
        self.classification_history_days = self.config['CLASSIFICATION_HISTORY_DAYS'] if classification_history_days is None else classification_history_days
        self.svm_regularization = self.config['SVM_REGULARIZATION']
        self.weighted_fit = self.config['WEIGHTED_REGRESSION']
        
        self.pool_size = pool_size
        self.horizons = extract_horizons(self.config) if horizons is None else horizons
        self.models_to_run = extract_models(self.config, self.horizons) if models is None else models

        self.model_regularization: Dict[str, float] = {}
        self.model_classifiers: Dict[str, Tuple[dt, Any]] = {}

        self.CLASSIFICATION_BAR_FEATURES = make_classification_bar_features(self.horizons)
        self.classification_history_window_mult = 3

        logger.info(f"Running fits on horizons: {self.horizons}, models: {self.models_to_run} pool size: {self.pool_size}")


    def _build_forward_data(
            self,
            new_symbols: List[str],
            common_symbols: List[str],
            horizon: int,
            classification_start_date: date,
            end_date: date,
            previous_end_date: date
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        existing_forwards_df = new_forwards_df = None

        if len(new_symbols) > 0:
            logger.info(f"Loading forwards on new symbols...")
            new_forwards_df = self.data_loader.load_forward_returns(
                start_date=classification_start_date,
                end_date=end_date,
                horizon=horizon,
                universe=new_symbols
            )
            if new_forwards_df is None:
                logger.warning(f"No new forward returns loaded between {classification_start_date}, and {end_date} for {horizon=}")
                new_forwards_df = pd.DataFrame()

        if len(common_symbols) > 0:
            logger.info(f"Loading forwards on existing symbols...")
            existing_forwards_df = self.data_loader.load_forward_returns(
                start_date=previous_end_date,
                end_date=end_date,
                horizon=horizon,
                universe=common_symbols
            )
            if existing_forwards_df is None:
                raise log_and_raise(f"No existing forward returns loaded between {previous_end_date}, and {end_date} for {horizon=}")

            _, max_ts = get_min_max_ts(existing_forwards_df)
            if max_ts.date() < end_date:
                end_date = max_ts.date()
                logger.info(f"New forwards only going to {date_to_str(end_date)} at {horizon=}")

        return new_forwards_df, existing_forwards_df

    def _build_features_data(
            self,
            new_symbols: List[str],
            common_symbols: List[str],
            horizon: int,
            feature_classification_features: List[str],
            horizon_features: List[str],
            classification_start_date: date,
            end_date: date,
            previous_end_date: Optional[date]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if previous_end_date is not None:
            assert previous_end_date <= end_date

        existing_features_df = new_features_df = None

        extracted_horizons = [horizon]
        for ff in horizon_features:
            hh = extract_horizon_from_feature(ff)
            if hh is not None:
                extracted_horizons.append(hh)
        horizons = unique_list(extracted_horizons)

        if len(new_symbols) > 0:
            logger.info(f"Loading new features for {new_symbols}")
            new_features_df = self.data_loader.load_features(
                start_date=classification_start_date,
                end_date=end_date,
                horizons=horizons,
                cols=feature_classification_features,
                symbol_venues=new_symbols,
                strict=False
            )
            if new_features_df is None:
                logger.warning(f"No new features returns loaded between {classification_start_date}, and {end_date} for {horizon=}")
                new_features_df = pd.DataFrame()

        if len(common_symbols) > 0 and len(feature_classification_features) > 0:
            existing_features_df = self.data_loader.load_features(
                start_date=previous_end_date,
                end_date=end_date,
                horizons=horizons,
                cols=feature_classification_features,
                symbol_venues=common_symbols,
                strict=False
            )
            if existing_features_df is None:
                raise log_and_raise(f"No existing features loaded between {previous_end_date} and {end_date} for {horizon=}: {feature_classification_features}")

            _, max_ts = get_min_max_ts(existing_features_df)
            if max_ts.date() < end_date:
                end_date = max_ts.date()
                logger.info(f"New features only going to {date_to_str(end_date)}")

        return new_features_df, existing_features_df

    def _build_models_data(
            self,
            new_symbols: List[str],
            common_symbols: List[str],
            horizon: int,
            classification_start_date: date,
            end_date: date,
            previous_end_date: date,
            models: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        existing_models_df = new_models_df = None
        if len(new_symbols) > 0:
            new_models_df = self.data_loader.load_models(
                start_date=classification_start_date,
                end_date=end_date,
                horizon=horizon,
                models=models,
                universe=new_symbols
            )
            if new_models_df is None:
                logger.warning(f"No new models returns loaded between {classification_start_date}, and {end_date} for {horizon=}")
                new_models_df = pd.DataFrame()

        if len(common_symbols) > 0:
            existing_models_df = self.data_loader.load_models(
                start_date=previous_end_date,
                end_date=end_date,
                horizon=horizon,
                models=models,
                universe=common_symbols
            )
            if existing_models_df is None:
                raise log_and_raise(f"No existing models returns loaded between {previous_end_date}, and {end_date} for {horizon=}")

            _, max_ts = get_min_max_ts(existing_models_df)
            if max_ts.date() < end_date:
                end_date = max_ts.date()
                logger.info(f"New models only going to {date_to_str(end_date)}")

        return new_models_df, existing_models_df


    def _build_bars_data(
            self,
            new_symbols: List[str],
            common_symbols: List[str],
            horizon: int,
            classification_start_date: date,
            bar_classification_features: List[str],
            end_date: date,
            previous_end_date: date
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        existing_bars_df = new_bars_df = None

        bar_fit_flds = [f'logret_{horizon}', f'logret_resid_eqmkt_{horizon}', f"logret_resid_wgtmkt_{horizon}", f'dvolume_{horizon}', 'advp']
        bar_cols = unique_list(bar_classification_features + bar_fit_flds)

        if len(new_symbols) > 0:
            new_symbols = sorted(new_symbols)
            logger.info(f"Loading bars on new symbols: {new_symbols}")
            new_bars_df = self.data_loader.load_bars(
                horizon=horizon,
                start_date=classification_start_date,
                end_date=end_date,
                symbol_venues=new_symbols,
                cols=bar_cols,
                strict=False,
                ok_to_return_nothing=True
            )
            if new_bars_df is None:
                logger.warning(f"No new bars returns loaded between {classification_start_date}, and {end_date} for {horizon=}")
            else:
                _, max_ts = get_min_max_ts(new_bars_df)
                if max_ts.date() < end_date:
                    logger.warning(f"New symbol bars only going to {date_to_str(max_ts.date())} for request to {date_to_str(end_date)}")

        if len(common_symbols) > 0:
            common_symbols = sorted(common_symbols)
            logger.info(f"Loading bars on existing symbols: {common_symbols}")

            existing_bars_df = self.data_loader.load_bars(
                horizon=horizon,
                start_date=previous_end_date,
                end_date=end_date,
                symbol_venues=common_symbols,
                cols=bar_cols,
                ok_to_return_nothing=True
            )
            if existing_bars_df is None:
                logger.warning(f"No existing bars returns loaded between {previous_end_date}, and {end_date} for {horizon=}")
            else:
                _, max_ts = get_min_max_ts(existing_bars_df)
                if max_ts.date() < end_date:
                    logger.warning(f"Existing symbol bars only going to  {date_to_str(max_ts.date())} for request to {date_to_str(end_date)}")

        return new_bars_df, existing_bars_df

    def _build_fit_data(
            self,
            horizon: int,
            start_date: date,
            end_date: date,
            fitting_lookback_days: int,
            use_classifier: bool,
            model_names: List[str],
        cached_fit_data_df: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.DataFrame]:
        """Load and merge all data required for model fitting.
        
        Assembles a comprehensive dataset by loading and merging:
        - Forward returns (dependent variables)
        - Features (independent variables)
        - Model predictions (primary independent variables)
        - Bar data (for filtering and additional features)
        
        The method handles both the fitting window and a longer classification
        window if SVM classifiers are being used. It also computes cross-sectional
        features if specified in the configuration.
        
        Args:
            horizon: Time horizon in minutes
            start_date: Start date for data loading
            end_date: End date for data loading
            fitting_lookback_days: Number of days for fitting window
            
        Returns:
            DataFrame with all merged data ready for model fitting
            
        Raises:
            Exception: If any required data is missing or cannot be loaded
        """
        logger.info(f"Building fit data between {start_date} - {end_date} for {model_names}")

        if cached_fit_data_df is not None:
            previous_universe = set(cached_fit_data_df.index.get_level_values('symbol_venue').unique())
            previous_start_ts, previous_end_ts = get_min_max_ts(cached_fit_data_df)
            logger.info(f"Cached data runs from {previous_start_ts} - {previous_end_ts}")
            previous_end_date = previous_end_ts.date()
            assert previous_end_date <= end_date
        else:
            logger.info(f"Not using cached data...")
            previous_universe = set()
            previous_start_ts = previous_end_ts = previous_end_date = None

        universe = self.universe.load_universe_symbols(
            universe_source='file',
            universe_date=start_date - td(days=1),
            symbol_type=SYMBOL_VENUE,
            filter='fittable'
        )
        universe_set = set(universe)
        new_symbols = list(universe_set - previous_universe)
        dropped_symbols = list(previous_universe - universe_set)
        common_symbols = list(universe_set & previous_universe)
        if len(dropped_symbols) > 0:
            logger.info(f"Dropping symbols: {dropped_symbols}")

        if len(new_symbols) > 0:
            logger.info(f"Adding symbols: {new_symbols}")

        logger.info(f"Classification symbol count: {len(common_symbols)}")

        if use_classifier:
            # lookback_days = max(fitting_lookback_days, self.classification_history_days) + 1
            lookback_days = fitting_lookback_days * self.classification_history_window_mult

            classification_start_date = start_date - td(days=lookback_days)
            horizon_features, _ = extract_all_tree_features_horizon(self.config, horizon)
            logger.info(f"Using classification features from {classification_start_date} at {horizon}: {horizon_features}")
            cx_features = [hf.split('.')[1] for hf in horizon_features if hf.startswith('cx.')]
            horizon_features = cx_features + [hf for hf in horizon_features if not hf.startswith('cx.')]
            bar_classification_features = make_classification_bar_features([horizon])
            feature_classification_features = [feature for feature in horizon_features if feature not in bar_classification_features]
        else:
            feature_classification_features = []
            bar_classification_features = ['advp']
            horizon_features = []
            classification_start_date = start_date - td(days=1)

        additions_forwards_df, existing_forwards_df = self._build_forward_data(
            new_symbols=new_symbols,
            common_symbols=common_symbols,
            horizon=horizon,
            classification_start_date=classification_start_date,
            end_date=end_date,
            previous_end_date=previous_end_date
        )
        additions_features_df, existing_features_df = self._build_features_data(
            new_symbols=new_symbols,
            common_symbols=common_symbols,
            horizon=horizon,
            classification_start_date=classification_start_date,
            feature_classification_features=feature_classification_features,
            horizon_features=horizon_features,
            end_date=end_date,
            previous_end_date=previous_end_date
        )
        additions_models_df, existing_models_df = self._build_models_data(
            new_symbols=new_symbols,
            common_symbols=common_symbols,
            horizon=horizon,
            classification_start_date=classification_start_date,
            end_date=end_date,
            previous_end_date=previous_end_date,
            models=model_names
        )
        additions_horizon_bars_df, existing_horizon_bars_df = self._build_bars_data(
            new_symbols=new_symbols,
            common_symbols=common_symbols,
            horizon=horizon,
            bar_classification_features=bar_classification_features,
            classification_start_date=classification_start_date,
            end_date=end_date,
            previous_end_date=previous_end_date
        )

        if cached_fit_data_df is not None:
            orig_cnt = len(cached_fit_data_df)

            sym_idx = cached_fit_data_df.index.get_level_values('symbol_venue').isin(common_symbols)

            old_min_ts = cached_fit_data_df.index.get_level_values('ts').min()

            cached_fit_data_df = cached_fit_data_df[sym_idx]
            sym_cnt = len(cached_fit_data_df)

            dt_idx = cached_fit_data_df.index.get_level_values('ts') > date_to_start_dt(classification_start_date)
            cached_fit_data_df = cached_fit_data_df[dt_idx]
            new_min_ts = cached_fit_data_df.index.get_level_values('ts').min()

            logger.info(f"Cached count {orig_cnt} -> (remove symbols) {sym_cnt} -> (remove dates) {len(cached_fit_data_df)}  {old_min_ts} -> {new_min_ts}")

            old_existing_df = cached_fit_data_df

            new_existing_df = merge_on_index(existing_horizon_bars_df, existing_models_df)
            if existing_features_df is not None and len(existing_features_df) > 0:
                new_existing_df = merge_on_index(new_existing_df, existing_features_df)
            new_existing_df = merge_on_index(new_existing_df, existing_forwards_df)
        else:
            old_existing_df = pd.DataFrame()
            new_existing_df = pd.DataFrame()

        if len(new_symbols) > 0 and len(additions_models_df) > 0:
            additions_df = merge_on_index(additions_horizon_bars_df, additions_models_df)
            if additions_features_df is not None and len(additions_features_df) > 0:
                additions_df = merge_on_index(additions_df, additions_features_df)
            additions_df = merge_on_index(additions_df, additions_forwards_df)
        else:
            additions_df = pd.DataFrame()

        if previous_end_ts is not None and len(new_symbols) > 0 and len(additions_df) > 0:
            new_additions_idx = additions_df.index.get_level_values('ts') > previous_end_ts
            new_additions_df = additions_df[new_additions_idx]
            old_additions_df = additions_df[~new_additions_idx]
        else:
            new_additions_df = additions_df
            old_additions_df = pd.DataFrame()

        logger.info(f"Fit data rows old_existing: {len(old_existing_df)} new_existing: {len(new_existing_df)} old_additions: {len(old_additions_df)} new_additions: {len(new_additions_df)}")

        fitting_df = pd.concat([old_existing_df, new_existing_df, old_additions_df, new_additions_df], verify_integrity=True)
        if fitting_df is None:
            logger.warning("No fitting data generated")
            return None

        if use_classifier:
            horizon_features, _ = extract_all_tree_features_horizon(self.config, horizon)
            cx_features = [hf.split('.')[1] for hf in horizon_features if hf.startswith('cx.')]
            fitting_df = make_cx_features(fitting_df, cx_features)

        min_ts, max_ts = get_min_max_ts(fitting_df)
        missing_days = (min_ts.date() - classification_start_date).days
        logger.info(f"Built fitting data between {min_ts} to {max_ts} missing data for {missing_days} days")
        if missing_days > (self.classification_history_days / 2):
            logger.warning(f"Missing {missing_days} days for classification which is less than half {self.classification_history_days}, should not run classification")
            return None

        if self.weighted_fit:
            fitting_df = make_market_weight_col(fitting_df)

        return fitting_df

    def run_model_horizon_fit(
            self,
            forecast: dict,
            fitting_window_df: pd.DataFrame,
            horizon: int,
            features: List[str],
            weighted_fit: bool,
            lags: int,
            return_type: str,
            classification_df: Optional[pd.DataFrame] = None,
            svm_regularization: Optional[float] = None,
            classifier: Optional[Pipeline] = None
    ) -> Optional[ModelHorizonFit]:
        try:
            return ModelHorizonFit(
                prod=self.prod,
                debug=self.debug,
                config=self.config,
                forecast=forecast,
                fitting_df=fitting_window_df,
                classification_df=classification_df,
                classification_features=features,
                weighted_fit=weighted_fit,
                horizon=horizon,
                lags=lags,
                return_type=return_type,
                classifier_dir=self.classifier_dir,
                svm_regularization=svm_regularization,
                classifier=classifier
            ).run_fits()
        except AssertionError as e:
            logger.error(f"Assertion caught, investigate... {e}", key="see assertion error in ModelHorizonFit fits run")
            raise
        except Exception as e:
            logger.exception(e)
            raise

    def generate_classification_window(
            self,
            models_df: pd.DataFrame,
            fitting_lookback_days: int,
            fit_start_ts: dt,
            features: List[str]
    ) -> pd.DataFrame:
        classification_window_df = None

        if self.classification_history_days > 0 and len(features) > 0:
            lookback = max(fitting_lookback_days, self.classification_history_days)
            classification_start_date = fit_start_ts - td(days=lookback)

            after_start_idx = models_df.index.get_level_values('ts') > classification_start_date
            classification_window_df = models_df[after_start_idx & (models_df.index.get_level_values('ts') < fit_start_ts)]

            if len(classification_window_df) == 0:
                min_ts, _ = get_min_max_ts(models_df)
                raise log_and_raise(f"No fittable data before {fit_start_ts=}, data starts at {min_ts}")

            min_ts, max_ts = get_min_max_ts(classification_window_df)

            logger.info(f"Generating classification window between {min_ts} -> {max_ts} with cnt {len(classification_window_df)}")
        else:
            logger.info(f"not running classifier {self.classification_history_days=} feature cnt: {len(features)}")

        return classification_window_df

    def generate_fitting_window(
            self,
            models_df: pd.DataFrame,
            fit_start_ts: dt,
            fit_end_ts: dt,
    ) -> Optional[pd.DataFrame]:
        logger.info(f"Generating fitting window between {fit_start_ts} -> {fit_end_ts}")

        models_df = models_df[models_df.index.get_level_values('ts') < fit_end_ts]
        fitting_window_df = models_df[(models_df.index.get_level_values('ts') >= fit_start_ts)]
        fitting_cnt = len(fitting_window_df)
        logger.info(f"Fitting window has {fitting_cnt} points")

        if fitting_cnt == 0:
            min_ts, max_ts = get_min_max_ts(models_df)
            logger.error(f"No points in models_df after {fit_start_ts}, goes from {min_ts} - {max_ts}")
            return None

        if fitting_cnt < MIN_FITTING_OBS:
            min_fitting_ts, max_fitting_ts = get_min_max_ts(fitting_window_df)
            logger.error(f"Not enough data ({fitting_cnt}) to fit from {fit_start_ts} to {fit_end_ts}, df goes from {min_fitting_ts} to {max_fitting_ts}")
            return None

        return fitting_window_df

    def gather_fitting_dates(self, refitting_days: int, start_date: dt, fitting_end_date: dt, fitting_lookback_days: Optional[int] = None) -> List[Tuple[dt, dt]]:
        assert fitting_end_date >= start_date

        fitting_start_date = start_date - td(days=fitting_lookback_days)
        fitting_dates = []
        logger.info(f"Generating fit dates refitting days: {refitting_days}, lookback days: {fitting_lookback_days} with data between {fitting_start_date} and {fitting_end_date}")
        assert fitting_start_date < fitting_end_date

        while fitting_end_date >= start_date:
            if fitting_lookback_days is not None:
                lookback_start_date = fitting_end_date - td(days=fitting_lookback_days)
                if lookback_start_date < fitting_start_date:
                    if len(fitting_dates) == 0:
                        logger.error(f"{lookback_start_date=} before {fitting_start_date=}, need to expand window!", key="fitting dates length as zero")
                    break
            else:
                lookback_start_date = fitting_start_date

            logger.info(f"Adding fit window {lookback_start_date} - {fitting_end_date}")
            fitting_dates.append((lookback_start_date, fitting_end_date))
            if fitting_end_date == start_date:
                break
            fitting_end_date = max(fitting_end_date - td(days=refitting_days), start_date)

        if len(fitting_dates) == 0:
            raise log_and_raise("No fitting dates generated")

        if self.prod:
            fit_start, fit_end = fitting_dates[:1][0]
            logger.info(f"For prod update just fitting on {date_to_str(fit_start)} - {date_to_str(fit_end)}")
            fitting_dates = fitting_dates[:1]

        # proceed forward through time...
        fitting_dates = sorted(fitting_dates, key=lambda dd: dd[1])
        return fitting_dates

    def dump_fits(self, fit_df: pd.DataFrame):
        """Persist fitted model coefficients to CSV files.
        
        Saves fitted coefficients and metadata to CSV files organized by
        horizon and model name. The file structure differs between production
        and development modes, and between different data source types.
        
        Args:
            fit_df: DataFrame containing fitted coefficients with columns:
                - horizon: Time horizon in minutes
                - name: Model name
                - as_of: Fit date
                - condition: Regime (mom/rev)
                - lag: Lag period
                - coeff: Fitted coefficient
                - tstat: T-statistic
                - stderr: Standard error
                - nobs: Number of observations
                
        File naming conventions:
        - Production (legacy): fits.{horizon}.{date}.prod.csv
        - Development: fits.{model}.{horizon}.{date}.csv
        - New format: {fits_dir}/{horizon}/{model}/fits.{model}.{horizon}.{date}.csv
        
        In debug mode, outputs to fits.debug.csv instead.
        """
        if fit_df is None:
            logger.warning("No fits to dump")
            return

        min_ts = fit_df['as_of'].min()
        max_ts = fit_df['as_of'].max()
        logger.info(f"Dumping prod: {self.prod} fits from {min_ts} - {max_ts}")

        fit_df = fit_df.sort_values(by=['horizon', 'name', 'as_of', 'condition'])
        if self.debug:
            fit_df.to_csv("fits.debug.csv")
            print(fit_df)
        else:
            # Always use new format for production
            for horizon in fit_df['horizon'].unique():
                horizon_fit_df = fit_df[fit_df['horizon'] == horizon]
                for name in horizon_fit_df['name'].unique():
                    model_fit_df = horizon_fit_df[horizon_fit_df['name'] == name]
                    start_date = date_to_str(model_fit_df['as_of'].min())
                    fit_filename, fit_filepath = self.data_loader.make_fit_file_name(fits_dir=self.fits_dir, model=name, horizon=horizon, prod=self.prod, date_str=start_date)
                    safe_mkdir(fit_filepath)
                    logger.info(f"Dumping {fit_filename}")
                    model_fit_df.to_csv(fit_filename, index=False)


    def get_latest_fit_file_date(self, horizon: int, model: str) -> Optional[date]:
        latest_fit_file = self.get_latest_fit_file(horizon=horizon, model=model)
        if latest_fit_file is None:
            return None
        toks = latest_fit_file.split(".")
        # Always use new format
        maybe_date = toks[-2]
        try:
            file_date = date_str_to_date(maybe_date)
        except ValueError:
            logger.warning(f"No date found in filename {latest_fit_file}")
            return None
        return file_date

    def get_latest_fit_file(self, horizon: int, model: str) -> Optional[str]:
        # Always use new format
        fit_file_glob, _ = self.data_loader.make_fit_file_name(fits_dir=self.fits_dir, horizon=horizon, model=model, prod=self.prod, date_str="*")
        fit_files = sorted(glob.glob(fit_file_glob), reverse=True)
        if len(fit_files) > 0:
            return fit_files[0]
        return None


    def _get_cached_classifier(self, name: str, ts: dt) -> Optional[Pipeline]:
        if name in self.model_classifiers:
            classifier_date, cached_classifier = self.model_classifiers[name]
            if classifier_date < (ts.date() - td(days=7)):
                return cached_classifier
        return None


    def generate_rolling_fits(self, start_date: Optional[date] = None, end_date: Optional[date] = None):
        """Generate model fits using rolling windows across all configured horizons.
        
        This is the main entry point for the fitting pipeline. It processes each
        horizon sequentially (from longest to shortest), creating rolling fitting
        windows and running parallel model fits. The method handles both development
        mode (full historical fitting) and production mode (incremental updates).
        
        Args:
            start_date: Start date for fitting (required in dev mode)
            end_date: End date for fitting (defaults to today in prod mode)
            
        The method performs the following for each horizon:
        1. Checks if refitting is needed based on schedule
        2. Loads all required data (bars, features, models, forwards)
        3. Creates rolling fitting windows based on configuration
        4. Runs ModelHorizonFit for each model in parallel
        5. Aggregates and persists results
        
        In production mode, only fits the most recent window. In development
        mode, fits all historical windows for backtesting.
        """
        if self.prod:
            if end_date is None:
                end_date = today_date()
        else:
            assert start_date is not None

        end_date = min(end_date, today_date())

        fit_dfs = []
        fit_configs = self.config['FCASTS']
        horizon_to_max_lags = extract_max_lags(self.config)
        for horizon in sorted(self.horizons, reverse=True):
            logger.info(f"Generating fits from {start_date} to {end_date} for {horizon=}")

            #reset for each horizon
            self.model_classifiers = {}
            fit_config = fit_configs[str(horizon)]
            models = fit_config['models']
            if self.models_to_run is not None:
                models = [f for f in models if f['name'] in self.models_to_run]

            if len(models) == 0:
                logger.info(f"Not models to run at {horizon}")
                continue

            refitting_days = fit_config['refitting_days']
            fitting_lookback_days = fit_config['fitting_lookback_days']
            classification_features, _ = extract_all_tree_features_horizon(self.config, horizon)
            return_type = fit_config['return_type']
            lags = horizon_to_max_lags[horizon]
            horizon_days = compute_lookback_days(horizon, lags)
            horizon_end_date = end_date - td(days=horizon_days)
            horizon_start_date = horizon_end_date
            if start_date is not None:
                horizon_start_date = min(horizon_start_date, start_date)

            model_names = [m['name'] for m in models]
            use_classifier = False
            for model in models:
                if model['fit_type'] not in ('vanilla', 'security'):
                    use_classifier = True
                    break

            logger.info(f"Rolling fits {horizon} : {model_names} at lags {lags} with {horizon_days=}")

            if self.prod and not self.debug and not self.force:
                test_model = models[0]['name']
                latest_fit_file_date = self.get_latest_fit_file_date(horizon=horizon, model=test_model)
                logger.info(f"Latest fit file for {horizon=} is {latest_fit_file_date} for model {test_model} {refitting_days=}")
                if latest_fit_file_date is not None:
                    next_fit_file_date = latest_fit_file_date + td(days=refitting_days)
                    # if next_fit_file_date > horizon_end_date + td(days=1):
                    if next_fit_file_date > end_date:
                        logger.info(f"Not generating fits for {horizon=} until {next_fit_file_date} which is after {horizon_end_date}")
                        continue

            for model in models:
                if model['fit_type'] in ('svm_adapt', 'svm_cv'):
                    self.model_regularization[model['name']] = self.svm_regularization

            logger.info(f"Fit dates for {horizon=} {start_date} to {horizon_end_date}")

            fitting_dates = self.gather_fitting_dates(
                start_date=horizon_start_date,
                fitting_end_date=horizon_end_date,
                fitting_lookback_days=fitting_lookback_days,
                refitting_days=refitting_days,
            )

            cached_fit_data_df = None
            for fit_start_date, fit_end_date in fitting_dates:
                logger.info(f"Running fits for period {fit_start_date} to {fit_end_date} for {horizon=}")
                log_mem_usage()

                models_df = self._build_fit_data(
                    start_date=fit_start_date,
                    end_date=fit_end_date,
                    horizon=horizon,
                    fitting_lookback_days=fitting_lookback_days,
                    cached_fit_data_df=cached_fit_data_df,
                    model_names=model_names,
                    use_classifier=use_classifier
                )
                if models_df is None:
                    logger.warning(f"no data for fit data for {horizon=} from {fit_start_date} to {fit_end_date}")
                    continue

                cached_fit_data_df = models_df

                # Create _abs columns for classifier features (needed before _remove_missing_features)
                for col in models_df.columns:
                    if col.endswith('_L0'):
                        models_df[f"{col}_abs"] = models_df[col].abs()

                fitting_window_df = self.generate_fitting_window(
                    models_df=models_df,
                    fit_start_ts=date_to_start_dt(fit_start_date),
                    fit_end_ts=date_to_end_dt(fit_end_date),
                )
                if fitting_window_df is None:
                    logger.error(f"Can't run fits for {horizon=} from {fit_start_date} to {fit_end_date}")
                    continue

                fit_cnt = len(fitting_window_df)
                min_ts, max_ts = get_min_max_ts(fitting_window_df)
                logger.info(f"Running fits on timestamp {min_ts} - {max_ts} with {fit_cnt} observations")
                if fit_cnt < MIN_FITTING_OBS:
                    logger.warning(f"Too few observations to fit {fit_cnt}! from {fit_start_date} - {fit_end_date}, continuing...")
                    continue

                if use_classifier:
                    classification_features = _remove_missing_features(classification_features, models_df)
                    classification_df = self.generate_classification_window(
                        models_df=models_df,
                        fitting_lookback_days=fitting_lookback_days,
                        fit_start_ts=date_to_start_dt(fit_start_date),
                        features=classification_features
                    )

                    old_df = fitting_window_df
                    fitting_window_new_df = fitting_window_df.dropna(subset=classification_features)
                    after_fit_cnt = len(fitting_window_new_df)
                    if after_fit_cnt < MIN_FITTING_OBS:
                        print(old_df[classification_features].iloc[0])
                        logger.error(f"Fitting data length: {fit_cnt} -> {after_fit_cnt}, once removing NAs for classification features!", key="fitting data length too small in Fits")
                        continue
                    fitting_window_df = fitting_window_new_df
                else:
                    classification_df = models_df

                if len(fitting_window_df) == 0:
                    raise log_and_raise(f"Zero length fitting window for {fit_start_date} to {fit_end_date}")

                if len(classification_df) == 0:
                    raise log_and_raise(f"Zero length classification window for {fit_start_date} to {fit_end_date}")

                subsampled_classification_df = classification_df.sample(frac=0.1, random_state=42)

                pool = Pool(self.pool_size)
                if self.pool_size == 1:
                    for fc in models:
                        model_name = fc['name']
                        model_lag_name = f"{model_name}_{horizon}_L0"
                        if model_lag_name not in subsampled_classification_df.columns or subsampled_classification_df[model_lag_name].fillna(0).sum() == 0:
                            logger.warning(f"No non-zero model indicator for {model_lag_name}, not fitting")
                            continue

                        svm_regularization = self.model_regularization[model_name] if fc['fit_type'] in ('svm_cv', 'svm_adapt') else None
                        cached_classifier = self._get_cached_classifier(model_name, min_ts)
                        model_classifier_features = extract_tree_features_horizon_model(self.config, horizon, model_name)
                        model_classifier_features = _remove_missing_features(model_classifier_features, subsampled_classification_df)

                        mhf = self.run_model_horizon_fit(
                            forecast=fc,
                            fitting_window_df=fitting_window_df,
                            horizon=horizon,
                            features=model_classifier_features,
                            weighted_fit=self.weighted_fit,
                            lags=lags,
                            return_type=return_type,
                            classification_df=subsampled_classification_df,
                            svm_regularization=svm_regularization,
                            classifier=cached_classifier
                        )

                        if mhf is not None and mhf.result_df is not None:
                            fit_df = mhf.result_df
                            if not (self.prod or self.debug):
                                self.dump_fits(fit_df)
                            fit_dfs.append(fit_df)

                            if mhf.fit_type.startswith('svm'):
                                self.model_regularization[mhf.name] = mhf.svm_regularization
                            if mhf.use_classifier:
                                self.model_classifiers[mhf.name] = (mhf.as_of, mhf.classifier)

                else:
                    all_params = []
                    for fc in models:
                        model_name = fc['name']

                        model_lag_name = f"{model_name}_{horizon}_L0"
                        if model_lag_name not in subsampled_classification_df.columns or subsampled_classification_df[model_lag_name].fillna(0).sum() == 0:
                            logger.warning(f"No non-zero model indicator for {model_lag_name}, not fitting")
                            continue

                        svm_regularization = self.model_regularization[model_name] if fc['fit_type'] in ('svm_cv', 'svm_adapt') else None
                        cached_classifier = self._get_cached_classifier(model_name, min_ts)
                        model_classifier_features = extract_tree_features_horizon_model(self.config, horizon, model_name)
                        model_classifier_features = _remove_missing_features(model_classifier_features, subsampled_classification_df)

                        all_params.append((
                            fc,
                            fitting_window_df.copy(),
                            horizon,
                            model_classifier_features,
                            self.weighted_fit,
                            lags,
                            return_type,
                            subsampled_classification_df.copy(),
                            svm_regularization,
                            cached_classifier
                        ))

                    model_results = {}
                    for mhf in pool.starmap(self.run_model_horizon_fit, all_params):
                        if mhf is not None and mhf.result_df is not None:
                            model_name = mhf.name
                            if model_name not in model_results:
                                model_results[model_name] = mhf.result_df
                            else:
                                model_results[model_name] = pd.concat([model_results[model_name], mhf.result_df])

                            if mhf.fit_type.startswith('svm'):
                                self.model_regularization[mhf.name] = mhf.svm_regularization
                            if mhf.use_classifier:
                                self.model_classifiers[mhf.name] = (mhf.as_of, mhf.classifier)

                    # Add all concatenated results to fit_dfs
                    for model_name, result_df in model_results.items():
                        if not (self.prod or self.debug):
                            self.dump_fits(result_df)
                        fit_dfs.append(result_df)

                    pool.close()
                    pool.join()

        if len(fit_dfs) == 0:
            logger.warning("No fit dataframes produced!")
            return

        fits_df = pd.concat(fit_dfs)

        if self.prod:
            self.dump_fits(fits_df)
