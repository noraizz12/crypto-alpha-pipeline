"""Model calculation module for generating lagged model signals.

This module provides functionality for computing model values (e.g., hl, c2vwap) and
creating lagged versions for use in forecasting. It handles the generation of model
signals across multiple time horizons and manages the storage of calculated values.

Key components:
- generate_model_lags: Creates lagged versions of model signals for autoregressive features
- ModelCalcs: Main class for orchestrating model calculations across dates and horizons
"""

import logging
from datetime import date
from datetime import timedelta as td
from multiprocessing.dummy import Pool
from typing import List, Optional, Tuple

import pandas as pd

from lib.data import dump_parquet_files
from lib.data.dataloader import DataLoader
from lib.alpha.models import Models
from lib.data.loaders import load_mktcap_and_groupings
from lib.universe import Universe
from lib.util.config import extract_horizons, extract_models
from lib.util.dataframes import check_df, check_missing_ts, get_min_max_ts, log_col_summary, merge_on_index, set_index, convert_category_cols_to_bool, clean_poop
from lib.util.directory import DirectoryManager, dir_manager
from lib.util.time_util import date_range

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_model_lags(df: pd.DataFrame, model_name: str, lags: int, horizon: int, prod: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """Generate lagged versions of a model signal for autoregressive features.

    Creates lagged columns for a given model signal by shifting the data by multiples
    of the horizon. For example, if model_name='hl_1440' and lags=3, this creates
    columns hl_1440_L0 (current), hl_1440_L1 (1 period lag), hl_1440_L2, and hl_1440_L3.

    Args:
        df: DataFrame with MultiIndex (ts, symbol_venue) containing the base model signal
        model_name: Name of the model signal (e.g., 'hl_1440', 'c2vwap_60')
        lags: Number of lagged periods to generate (minimum 1 for classifier features)
        horizon: Time horizon in minutes for the lag period
        prod: If True, optimizes for production by only computing lags for recent data

    Returns:
        Tuple containing:
            - DataFrame with added lag columns
            - List of column names that were created (L0 through L{lags})

    Note:
        - L0 represents the current value (no lag)
        - L1 represents the value from 'horizon' minutes ago
        - Always creates at least L0 and L1 (even if lags=0) for classifier features
        - In production mode, only processes recent data to improve performance
        - Missing timestamps are checked and logged as warnings

    Example:
        >>> df = pd.DataFrame(...)  # Contains 'hl_1440' column
        >>> df, lag_cols = generate_model_lags(df, 'hl_1440', lags=2, horizon=1440)
        >>> # df now contains: hl_1440_L0, hl_1440_L1, hl_1440_L2
    """
    logger.info(f"Generating model lags/indeps {model_name=} {lags=} at {horizon=}")
    min_ts, max_ts = get_min_max_ts(df)

    recent_df = df
    if prod:
        lagged_start_dt = max_ts - td(minutes=lags * horizon)
        recent_df = recent_df[recent_df.index.get_level_values('ts') >= lagged_start_dt]
        min_ts, max_ts = get_min_max_ts(recent_df)
        logger.info(f"Prod, so calculating model lags past {lagged_start_dt}, actual {min_ts} for {model_name}:L{lags}")

    check_df(recent_df)
    indeps = [f'{model_name}_L0']
    missing_ts = check_missing_ts(recent_df)

    logger.info(f"Checking model lags/indeps {model_name=} {lags=} at {horizon=} from {min_ts=}, {max_ts=}, and {missing_ts=}")
    # always want to compute L1 as a feature (for classifier at least)
    for ii in range(1, max(lags + 1, 2)):
        signal_var = f'{model_name}_L{ii}'
        signal_var_lagged = f'{model_name}_L{ii - 1}'
        logger.info(f"Creating {signal_var}")
        try:
            recent_df[signal_var] = recent_df[[signal_var_lagged]].unstack().shift(horizon).stack(future_stack=True)
        except Exception as e:
            recent_df[[signal_var_lagged]].to_parquet(f'signal_var_{horizon}_{signal_var_lagged}.parquet')
            raise e
        indeps.append(signal_var)
        if not prod:
            log_col_summary(recent_df, signal_var)

    if prod:
        df.loc[df.index.get_level_values('ts') == max_ts, indeps] = recent_df.loc[recent_df.index.get_level_values('ts') == max_ts, indeps]
    else:
        df[indeps] = recent_df[indeps]

    return df, indeps


class ModelCalcs:
    """Orchestrates the calculation of model signals across dates and horizons.

    This class manages the complete pipeline for generating model values (e.g., hl, c2vwap)
    across multiple time horizons and dates. It loads required features and bar data,
    computes model signals using the Models class, and saves the results to parquet files.

    The class supports parallel processing of multiple models and handles both live and
    Tardis data formats.

    Attributes:
        config: System configuration dictionary
        debug: Debug mode flag for additional logging
        dir_manager: Directory manager for path resolution
        data_loader: DataLoader instance for loading market data
        output_dir: Directory for saving model output files
        pool_size: Number of processes for parallel execution
        horizons: List of time horizons to process
        models_to_run: List of model names to calculate
        models: Models instance for signal calculations
    """

    def __init__(self,
        config: dict,
        models_to_run: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None,
        debug: bool = False,
        pool_size: int = 4,
        models_dir_manager: DirectoryManager = dir_manager,
        output_dir: Optional[str] = None,
    ):
        """Initialize ModelCalcs instance.

        Args:
            config: System configuration dictionary containing model specifications
            models_to_run: Optional list of specific models to calculate. If None, uses all configured
            horizons: Optional list of time horizons to process. If None, uses all configured
            debug: If True, enables debug logging and file outputs
            pool_size: Number of parallel processes for model calculation (default: 4)
            models_dir_manager: Directory manager for path resolution
            output_dir: Optional custom output directory. If None, uses default.
        """
        self.config = config
        self.debug = debug

        self.dir_manager = models_dir_manager
        self.data_loader = DataLoader(self.config, self.dir_manager)
        self.universe = Universe(config=self.config, universe_dir_manager=self.dir_manager)
        self.output_dir = output_dir if output_dir is not None else self.dir_manager.MODELS_DIR
        self.pool_size = pool_size

        self.horizons = horizons if horizons is not None else extract_horizons(self.config)
        self.models_to_run = models_to_run if models_to_run is not None else extract_models(self.config, horizons=self.horizons)
        self.models = Models(config=self.config)

        logger.info(f"Initiating ModelCalcs with {self.horizons} : {self.models_to_run}")


    def process_model(self, forecast: dict, horizon: int, models_df: pd.DataFrame, start_date: date) -> pd.DataFrame:
        """Process a single model for a given horizon and save results.

        Calculates the model signal (if not already present), saves it to a parquet file,
        and returns the updated DataFrame. This method is designed to be called in parallel
        for different models.

        Args:
            forecast: Dictionary containing model configuration with keys 'name' and 'lags'
            horizon: Time horizon in minutes for the model
            models_df: DataFrame containing features and bar data
            start_date: Start date for the output file naming

        Returns:
            DataFrame with the calculated model signal added

        Note:
            - Checks if model signal already exists to avoid recalculation
            - Saves only the L0 (current) lag to minimize file size
            - Handles both new format and legacy output formats
        """
        name = forecast['name']
        lags = int(forecast['lags'])
        model = f"{name}_{horizon}"
        lagged_model = f"{model}_L0"

        logger.info(f"Computing model {name=} {horizon=} {lags=}")
        if lagged_model not in models_df.columns:
            models_df = self.models.calculate(models_df, name, horizon)
        else:
            logger.info(f"{lagged_model} already in models_df")

        # Always use new format
        models_df = set_index(models_df)
        models_df = models_df.sort_index()
        dump_parquet_files(file_type='models', df=models_df, directory=f'{self.output_dir}/{horizon}/{name}', name=f'models.{horizon}.{name}', cols=[lagged_model], start_date=start_date, debug=self.debug)
        return models_df

    def process_models(self, start_date: date, end_date: date):
        """Process all configured models across the specified date range.

        Main entry point that orchestrates model calculation for all configured models
        and horizons over the given date range. Processes one day at a time to manage
        memory usage.

        Args:
            start_date: First date to process (inclusive)
            end_date: Last date to process (inclusive)

        Note:
            - Processes dates sequentially to avoid memory issues
            - For each date and horizon:
                1. Loads required features and bar data
                2. Merges data into a single DataFrame
                3. Calculates all models for that horizon (potentially in parallel)
                4. Saves results to parquet files
            - Skips dates/horizons where feature data is unavailable
            - Always includes 1440-minute features as some are required across horizons
        """
        fit_configs = self.config['FCASTS']
        for model_date in date_range(start_date, end_date):
            logger.info(f"Generating models at {date}")

            uni_date = model_date - td(days=1)
            fittable_symbol_venues = self.universe.load_universe_symbols(universe_source='file', universe_date=uni_date, filter='tradeable')
            categorical_df = load_mktcap_and_groupings(universe_date=uni_date, mktcap_dir=self.dir_manager.MARKETCAP_DIR, categories_only=True)

            for horizon in self.horizons:
                fcasts = fit_configs[str(horizon)]["models"]
                models_to_run = []
                for model_to_run in self.models_to_run:
                    for fcast in fcasts:
                        if model_to_run == fcast['name']:
                            models_to_run.append(fcast)

                if len(models_to_run) == 0:
                    logger.info(f"Horizon {horizon} has no models to run...")
                    continue

                model_names = [m['name'] for m in models_to_run]
                use_cat = 'grev' in model_names
                bar_feature_cols, feature_feature_cols = self.models.extract_models_features(models_to_run)

                bar_feature_cols.append('advp')
                horizon_bars_df = self.data_loader.load_bars(horizon=horizon, start_date=model_date, end_date=model_date, symbol_venues=fittable_symbol_venues, cols=bar_feature_cols)

                logger.info(f"Models {model_names} at {horizon} using features {feature_feature_cols}")
                #XXX should be able to figure out horizons needed instead of hardcoding
                if len(feature_feature_cols) > 0:
                    features_df = self.data_loader.load_features(start_date=model_date, end_date=model_date, horizons=[horizon, 1440], symbol_venues=fittable_symbol_venues, cols=feature_feature_cols)
                    if features_df is None:
                        logger.info(f"Not generating models for {model_date=} {horizon=} since no features loaded")
                        continue

                    models_df = merge_on_index(horizon_bars_df, features_df)
                else:
                    models_df = horizon_bars_df

                models_df = models_df.reset_index()

                if use_cat:
                    models_df = clean_poop(pd.merge(models_df, categorical_df, how='left', left_on='symbol_venue', right_on='symbol_venue', suffixes=('', '_poop')))
                    models_df = convert_category_cols_to_bool(models_df)

                assert len(models_df) % 1440 == 0

                if self.pool_size == 1:
                    for fc in models_to_run:
                        self.process_model(forecast=fc, horizon=horizon, models_df=models_df, start_date=start_date)
                else:
                    pool = Pool(processes=self.pool_size)
                    for _ in pool.starmap(
                            self.process_model,
                            [(fc, horizon, models_df.copy(), start_date) for fc in models_to_run]):
                        pass
                    pool.close()
                    pool.join()
