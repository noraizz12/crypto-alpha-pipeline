import logging
import math
from datetime import datetime as dt, date
from datetime import timedelta as td
from multiprocessing.dummy import Pool
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import re

from lib.calcs import Calcs
from lib.util.config import extract_horizon_models, extract_max_lags, extract_models, extract_tree_features, extract_tree_features_horizon_model
from lib.data import dump_parquet_files
from lib.data.dataloader import DataLoader
from lib.util.dataframes import check_df, get_min_max_ts, log_col_summary, merge_on_index, remove_infs, cols_to_list_str
from lib.fits.fit_util import make_classification_bar_features, extract_feature_importances
from lib.alpha.model_calcs import generate_model_lags
from lib.util.time_util import date_to_start_dt, date_to_str, today_date
from lib.calcs.calc_util import make_cx_features
from lib.util.directory import DirectoryManager, dir_manager
from lib.util.util import fpct, log_mem_usage, unique_list, log_and_raise
from lib.util.logging_util import KeyLogger
from lib.calcs.calc_util import calc_data_scaling_factor

pd.options.mode.chained_assignment = None

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


class NoFitsError(Exception):
    pass

def calculate_horizon_alphas(
        config: dict,
        df: pd.DataFrame,
        horizons: List[int],
        models: Optional[List[str]] = None,
        weight_override: bool = False,
        skip_missing_alphas: bool = False,
        alpha_condition: Optional[Literal['rev', 'mom']] = None,
        centering_symbols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    logger.info(f"Calculating Total Alphas for {horizons} / models: {models}")
    min_ts, max_ts = get_min_max_ts(df)

    max_alpha_global = config['MAX_ALPHA']
    min_alpha_global = -max_alpha_global
    center_alpha = config['CENTER_ALPHA_OPT']
    SIGMA_BOUND = config['SIGMA_BOUND_INDEP']

    new_cols = []
    for horizon in horizons:
        cols = [f'alpha_{horizon}', f'alpha_{horizon}_rev', f'alpha_{horizon}_mom']
        new_cols += cols
    new_cols = [cc for cc in new_cols if cc not in df.columns]
    new_df = pd.DataFrame(index=df.index, columns=new_cols, dtype=np.float32).fillna(0.0)
    df = pd.concat([df, new_df], axis=1)

    if center_alpha:
        df = center_adjust_alphas(df, max_alpha=max_alpha_global, method='mean', centering_symbols=centering_symbols)

    for horizon in horizons:
        logger.info(f"Calculating Total Alphas {horizon=}")
        for fcast in config['FCASTS'][str(horizon)]['models']:
            if weight_override:
                weight = 1.0
            else:
                weight = float(fcast['weight'])
                if weight == 0:
                    continue

            name = fcast['name']
            if models is not None and name not in models:
                continue

            alpha_str = f'alpha_{name}_{horizon}'
            alpha_str_rev = f"{alpha_str}_rev"
            alpha_str_mom = f"{alpha_str}_mom"

            if skip_missing_alphas and alpha_str not in df.columns:
                logger.warning(f"{alpha_str} not in columns {df.columns}")
                continue

            asize = df[alpha_str].abs().mean()
            if np.isnan(asize):
                raise log_and_raise(f"Got NaN alpha for {alpha_str}", df=df[df[alpha_str].isna()][[alpha_str]])
            if asize == 0:
                logger.warning(f"{name}:{horizon} alpha is all zeros from {min_ts} - {max_ts}")

            if center_alpha:
                logger.info(f"Using centered alphas")
                alpha_str_rev = f"{alpha_str_rev}_recenter"
                alpha_str_mom = f"{alpha_str_mom}_recenter"
                alpha_str = f"{alpha_str}_recenter"

            amin = df[alpha_str].min()
            amax = df[alpha_str].max()
            amean = df[alpha_str].mean()
            astd = df[alpha_str].std()

            min_alpha = max(amean - astd * SIGMA_BOUND, min_alpha_global)
            max_alpha = min(amean + astd * SIGMA_BOUND, max_alpha_global)
            logger.info(f"Clipping alpha between {min_alpha} and {max_alpha} for {name}:{horizon} from {min_ts} - {max_ts}")

            new_cols.append(alpha_str)

            logger.info(f"Adding model {name}={horizon} at weight={weight} min={amin} max={amax} absize={asize} strength={astd}")
            df[f'alpha_{horizon}'] += df[alpha_str].clip(min_alpha, max_alpha) * weight
            df[f'alpha_{horizon}_rev'] += df[alpha_str_rev].clip(min_alpha, max_alpha) * weight
            df[f'alpha_{horizon}_mom'] += df[alpha_str_mom].clip(min_alpha, max_alpha) * weight

        for case in ['', '_mom', '_rev']:
            horizon_mean = float(df[f'alpha_{horizon}{case}'].abs().mean())
            horizon_min = float(df[f'alpha_{horizon}{case}'].min())
            horizon_max = float(df[f'alpha_{horizon}{case}'].max())
            horizon_std = float(df[f'alpha_{horizon}{case}'].std())
            logger.info(f"Alpha {case} at {horizon=} {horizon_mean=} {horizon_std=} {horizon_min=} {horizon_max=}")

        if alpha_condition == 'mom':
            df[f'alpha_{horizon}'] = df[f'alpha_{horizon}_mom']
        elif alpha_condition == 'rev':
            df[f'alpha_{horizon}'] = df[f'alpha_{horizon}_rev']

    return df, new_cols


def center_adjust_alphas(
        alpha_df: pd.DataFrame,
        max_alpha: Optional[float] = None,
        method: str = 'median',
        verbose: bool = False,
        centering_symbols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Adjust alpha momentum and reversion components by subtracting cross-sectional center (mean or median).

    For each alpha_{model}_{horizon} that has _rev and _mom components:
    1. Calculates cross-sectional center (mean or median) for _rev and _mom columns
    2. Creates _recentered versions by subtracting the center
    3. Optionally clips the recentered values to [-max_alpha, max_alpha]
    4. Combines them into alpha_{model}_{horizon}_recenter

    Parameters:
    -----------
    alpha_df : pd.DataFrame
        DataFrame with MultiIndex (ts, symbol_venue) containing alpha values
    max_alpha : Optional[float]
        If provided, clips the recentered _rev and _mom values to [-max_alpha, max_alpha]
    method : str, default 'median'
        Centering method - either 'mean' or 'median'
    centering_symbols : Optional[List[str]]
        If provided, compute cross-sectional center using only these symbols.
        The centering adjustment is still applied to ALL symbols in the dataframe.
        This ensures consistency between live trading and simulation when they
        have different trading universes.

    Returns:
    --------
    pd.DataFrame
        DataFrame with added _recentered and _recenter columns

    Examples:
    --------
    >>> # Creates alpha_hl_60_rev_recentered, alpha_hl_60_mom_recentered, alpha_hl_60_recenter
    >>> adjusted_df = center_adjust_alphas(alpha_df)

    >>> # With mean centering
    >>> adjusted_df = center_adjust_alphas(alpha_df, method='mean')

    >>> # With clipping to [-0.05, 0.05]
    >>> adjusted_df = center_adjust_alphas(alpha_df, max_alpha=0.05)

    >>> # Center using only universe-file symbols
    >>> adjusted_df = center_adjust_alphas(alpha_df, centering_symbols=['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures'])
    """

    logger.info(f"Center adjusting rev/mom alphas with {method=}, centering_symbols={len(centering_symbols) if centering_symbols else 'all'}")

    # Validate method
    if method not in ['mean', 'median']:
        raise ValueError(f"method must be 'mean' or 'median', got '{method}'")

    # Find all unique model_horizon combinations that have both _rev and _mom
    model_horizon_pattern = re.compile(r'^alpha_(\w+_\d+)_(rev|mom)$')
    model_horizons = set()

    for col in alpha_df.columns:
        match = model_horizon_pattern.match(col)
        if match:
            model_horizons.add(match.group(1))

    if not model_horizons:
        logger.warning("No alpha columns found with _rev and _mom components")
        return alpha_df

    # If centering_symbols provided, create a mask for those symbols
    # We compute center using only centering_symbols, but apply adjustment to ALL symbols
    if centering_symbols is not None:
        centering_mask = alpha_df.index.get_level_values('symbol_venue').isin(centering_symbols)
        centering_df = alpha_df[centering_mask]
        logger.info(f"Using {centering_mask.sum()} rows from {len(centering_symbols)} centering symbols")
    else:
        centering_df = alpha_df

    # Process each model_horizon that has both components
    for model_horizon in sorted(model_horizons):
        logger.info(f"Center adjusting {model_horizon}")
        rev_col = f'alpha_{model_horizon}_rev'
        mom_col = f'alpha_{model_horizon}_mom'

        # Check both columns exist
        if rev_col not in alpha_df.columns or mom_col not in alpha_df.columns:
            logger.warning(f"Skipping {model_horizon}: missing rev or mom column")
            continue

        # Calculate cross-sectional center using centering_df (filtered if centering_symbols provided)
        # Then broadcast to all symbols via vectorized reindex on timestamp
        ts_centers = centering_df.groupby(level='ts')[[rev_col, mom_col]].agg(method)
        ts_values = alpha_df.index.get_level_values('ts')
        centers = ts_centers.reindex(ts_values)
        centers.index = alpha_df.index
        
        # Create recentered versions
        rev_recentered = f'{rev_col}_recenter'
        mom_recentered = f'{mom_col}_recenter'
        
        alpha_df[rev_recentered] = alpha_df[rev_col] - centers[rev_col]
        alpha_df[mom_recentered] = alpha_df[mom_col] - centers[mom_col]

        if verbose:
            rev_mean = alpha_df[rev_col].mean()
            new_rev_mean = alpha_df[rev_recentered].mean()
            logger.info(f"Centering {model_horizon} rev alpha {rev_mean} -> {new_rev_mean}")
            mom_mean = alpha_df[mom_col].mean()
            new_mom_mean = alpha_df[mom_recentered].mean()
            logger.info(f"Centering {model_horizon} mom alpha {mom_mean} -> {new_mom_mean}")

        # Apply clipping if max_alpha is specified
        if max_alpha is not None:
            alpha_df[rev_recentered] = alpha_df[rev_recentered].clip(-max_alpha, max_alpha)
            alpha_df[mom_recentered] = alpha_df[mom_recentered].clip(-max_alpha, max_alpha)
        
        # Create combined recentered alpha
        recenter_col = f'alpha_{model_horizon}_recenter'
        alpha_df[recenter_col] = alpha_df[rev_recentered] + alpha_df[mom_recentered]

        # Center adjust the combined recenter column using centering_symbols if provided
        if centering_symbols is not None:
            # Compute center from centering_df only, then broadcast to all symbols via vectorized reindex
            ts_combined_centers = alpha_df.loc[centering_mask, [recenter_col]].groupby(level='ts').agg(method)
            ts_values = alpha_df.index.get_level_values('ts')
            combined_centers = ts_combined_centers.reindex(ts_values)
            combined_centers.index = alpha_df.index
        else:
            combined_centers = alpha_df.groupby(level='ts')[[recenter_col]].transform(method)
        alpha_df[recenter_col] = alpha_df[recenter_col] - combined_centers[recenter_col]
        
        # Log statistics
        # for col_name, col_data in [(rev_recentered, alpha_df[rev_recentered]),
        #                            (mom_recentered, alpha_df[mom_recentered]),
        #                            (recenter_col, alpha_df[recenter_col])]:
        #     data = col_data.dropna()
        #     if len(data) > 0:
        #         logger.info(f"{col_name}: mean={data.mean():.6f}, std={data.std():.6f}, "
        #                    f"min={data.min():.6f}, max={data.max():.6f}")
    
    return alpha_df


class Forecasts:
    def __init__(
            self,
            config: dict,
            prod: bool,
            fit_as_of: Optional[dt] = None,
            debug: bool = False,
            is_server_forecast: bool = False,
            horizons: Optional[List[int]] = None,
            models: Optional[List[str]] = None,
            output_dir: Optional[str] = None,
            fits_dir: Optional[str] = None,
            forecast_dir_manager: DirectoryManager = dir_manager,
):
        logger.info(f"Forecasts {prod=} {debug=} {horizons=}, {models=}")
        self.config = config
        self.prod = prod
        self.debug = debug
        self.is_server_forecast = is_server_forecast
        self.fit_as_of = fit_as_of

        self.dir_manager = forecast_dir_manager
        self.data_loader = DataLoader(self.config, self.dir_manager)

        if prod:
            self.output_dir = self.dir_manager.ALPHA_DIR_PROD
            self.fits_dir = self.dir_manager.FITS_DIR_PROD
        else:
            self.output_dir = self.dir_manager.ALPHA_DIR_DEV
            self.fits_dir = self.dir_manager.FITS_DIR_DEV

        if output_dir is not None:
            self.output_dir = output_dir

        if fits_dir is not None:
            self.fits_dir = fits_dir

        self.classifier_dir = self.fits_dir

        self.min_tstat = self.config['MIN_TSTAT']
        self.zero_out_wrong_direction_coeffs = self.config['ENFORCE_REV_MOM']
        self.max_alpha = self.config['MAX_ALPHA']
        self.new_model_weight = not self.config['NEW_SCALE_ALPHA']

        self.horizon_models = extract_horizon_models(config=self.config, horizons=horizons, models=models, exclude_zero_weight=is_server_forecast)
        self.horizons: List[int] = sorted(unique_list([hm[0] for hm in self.horizon_models]))
        self.models_to_run = sorted(unique_list([hm[1] for hm in self.horizon_models]))

        logger.info(f"Running on {self.models_to_run}")

        self.max_lags: Dict[int, int] = extract_max_lags(self.config)
        self.fits_df: pd.DataFrame = self._load_fits()

        self.calcs = Calcs(self.config)
        self.classifier_dict = self.data_loader.load_classifiers(
            horizons=self.horizons,
            classifier_dir=self.classifier_dir,
            end_date=self.fit_as_of.date() if self.fit_as_of is not None else None,
        )

    def _load_fits(self) -> pd.DataFrame:
        fit_df = self.data_loader.load_fits(
            horizon_models=self.horizon_models,
            prod=self.prod,
            fits_dir=self.fits_dir,
            end_date=self.fit_as_of
        )
        if self.fit_as_of is not None:
            logger.info(f"only considering fits prior to {self.fit_as_of}")
            fit_df['as_of'] = self.fit_as_of
            fit_df = fit_df[fit_df['as_of'] <= self.fit_as_of]
            if len(fit_df) == 0:
                raise log_and_raise(f"No fits with asof <= {self.fit_as_of}")
        logger.info(fit_df)
        return fit_df

    def _set_class(self, models_df: pd.DataFrame, indep_horizon: str, asof: date, horizon: int, name: str) -> pd.DataFrame:
        models_df['class'] = 0
        model_horizon_features = extract_tree_features_horizon_model(self.config, horizon, name)

        # if features of specific model horizon is empty, there won't be classifier model trained from fits
        if len(model_horizon_features) == 0:
            return models_df

        model_classifiers = self.classifier_dict.get(indep_horizon)
        if model_classifiers is None:
            logger.error(f"No classifier found for model {indep_horizon=}")
            return models_df

        model_classifier = classifier_filename = None
        classifier_dates = sorted(model_classifiers.keys())
        for classifier_date in classifier_dates:
            if classifier_date <= asof:
                classifier_filename, model_classifier = model_classifiers[classifier_date]

        if model_classifier is not None:
            # Get feature names directly from classifier to preserve exact order and any duplicates
            # Using extract_feature_importances().keys() can lose duplicates due to dict key uniqueness
            if hasattr(model_classifier, 'feature_names_in_'):
                classifier_features = list(model_classifier.feature_names_in_)
            else:
                # Fallback for older classifiers without feature_names_in_
                classifier_features = list(extract_feature_importances(model_classifier, only_nonzero=False).keys())

            logger.info(f"Classifier {indep_horizon} using {classifier_filename.split('/')[-1]} expects {len(classifier_features)} columns: {sorted(set(classifier_features))}")
            fail = False
            for col in classifier_features:
                if col not in models_df.columns:
                    logger.error(f"Classifier {indep_horizon} expects column {col} but it's not in the dataframe")
                    fail = True

            if fail:
                log_and_raise(f"Dataframe has columns: {sorted(models_df.columns)}")

            try:
                class_df = remove_infs(models_df[classifier_features]).fillna(0)
            except KeyError as ke:
                logger.error(f"Cant get classifier features for {indep_horizon}:{asof=}, {classifier_filename} {ke} available: {cols_to_list_str(models_df)}", key="no classifier features found for model")
                raise

            missing_features = sorted(list(set(classifier_features) - set(class_df.columns)))
            if len(missing_features) > 0:
                logger.error(f"Dataframe missing classifier features {missing_features}")
            else:
                logger.info(f"All {len(classifier_features)} features found for classifier")

            try:
                models_df['class'] = model_classifier.predict(class_df)
            except Exception as e:
                log_and_raise(f"Can't predict class for {indep_horizon}: {e}", df=class_df)
        else:
            classifier_dates = [date_to_str(dd) for dd in classifier_dates]
            logger.error(f"Could not find svm for {indep_horizon=} prior to {asof=}, dates: {classifier_dates}", key="fail to find svm from model_svms")

        return models_df

    @staticmethod
    def log_and_check_alphas(alpha_s: pd.Series, log_str: str) -> bool:
        no_nan_alpha_s = alpha_s.fillna(0)
        alpha_min = no_nan_alpha_s.min()
        alpha_max = no_nan_alpha_s.max()
        alpha_std = no_nan_alpha_s.std()
        alpha_absmean = no_nan_alpha_s.abs().mean()
        logger.info(f"{log_str} Stats min={alpha_min} max={alpha_max} std={alpha_std:.6f} absmean={alpha_absmean:.6f}")
        if np.isnan(alpha_absmean):
            raise log_and_raise(f"{log_str} Nans in forecast")

        if alpha_absmean == 0:
            logger.warning(f"{log_str} All alphas are zero or nan!")
            # raise RuntimeError()
            # logger.warning(alpha_s.to_markdown())
            return False

        return True

    def compute_horizon_alpha(self, horizon: int, models_df: pd.DataFrame, prod: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        logger.info(f"Computing alpha at horizon {horizon} cnt: {len(models_df)}")
        new_cols = []
        fit_configs = self.config['FCASTS']

        for forecast in fit_configs[str(horizon)]['models']:
            name = forecast['name']
            lags = int(forecast['lags'])
            weight = float(forecast['weight'])
            model_horizon = f"{name}_{horizon}"
            if weight == 0:
                logger.info(f"Not computing alpha on zero weight model {model_horizon}")
                continue

            logger.info(f"Computing alpha {model_horizon}:{lags} {weight=}")
            models_df, alpha_cols = self.apply_coeffs(
                forecast=forecast,
                horizon=horizon,
                fit_df=self.fits_df,
                models_df=models_df,
                prod=prod
            )
            logger.info(f"Added columns {alpha_cols}")

            new_cols += alpha_cols
        return models_df, new_cols

    def compute_model_alphas_for_server(self, models_df: pd.DataFrame, horizons: Optional[List[int]] = None, pool_size: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        check_df(models_df)

        if horizons is None:
            horizons = self.horizons
        else:
            logger.info(f"Just processing alphas on {horizons=}")
            assert len(horizons) > 0

        if pool_size is None:
            pool_size = len(horizons)

        new_cols = []
        max_ts = models_df.index.get_level_values('ts').max()
        max_idx = models_df.index.get_level_values('ts') == max_ts

        latest_models_df = models_df[max_idx]

        for horizon in horizons:
            total_horizon_alpha_col = f"alpha_{horizon}"
            models_df[total_horizon_alpha_col] = np.float32(0)

        if pool_size == 1:
            for horizon in horizons:
                latest_models_df, cols = self.compute_horizon_alpha(horizon=horizon, models_df=latest_models_df, prod=True)
                new_cols += cols
            models_df.loc[max_idx, new_cols] = latest_models_df[new_cols]
        else:
            pool = Pool(processes=pool_size)
            for latest_horizon_models_df, new_horizon_cols in pool.starmap(
                    self.compute_horizon_alpha,
                    [(hh, latest_models_df.copy(), True) for hh in horizons]):
                models_df.loc[max_idx, new_horizon_cols] = latest_horizon_models_df[new_horizon_cols]
                new_cols += new_horizon_cols
            pool.close()
            pool.join()

        return models_df, new_cols

    def _generate_model_alpha(self, fcast: dict, horizon: int, models_df: pd.DataFrame, fit_df: pd.DataFrame, start_date: date) -> Optional[pd.DataFrame]:
        name = fcast['name']
        lags = int(fcast['lags'])
        weight = int(fcast['weight'])
        name_horizon = f"{name}_{horizon}"
        logger.info(f"Generating alphas for {name_horizon}")

        models_df, _ = generate_model_lags(models_df, model_name=name_horizon, lags=lags, horizon=horizon)
        models_df = models_df[models_df.index.get_level_values('ts') >= date_to_start_dt(start_date)]

        try:
            alphas_df, alpha_cols = self.apply_coeffs(forecast=fcast, fit_df=fit_df, models_df=models_df, horizon=horizon)
        except NoFitsError:
            logger.info(f"Continuing without {name}:{horizon} fits.")
            return None
        except (pd.errors.InvalidIndexError, ValueError) as ve:
            if weight == 0:
                print(ve)
                logger.warning(f"Zero weight model {name} {horizon} with no fits, moving on...")
                return None
            raise

        alphas_df = alphas_df[alpha_cols]
        log_col_summary(alphas_df, f"alpha_{name_horizon}")
        if not self.debug:
            fit_df.to_csv(f"{self.output_dir}/fits.csv", index=False)

        prod_str = 'prod' if self.prod else 'dev'
        dump_parquet_files(file_type='alphas', df=alphas_df, directory=f'{self.output_dir}/{horizon}/{name}', name=f'alphas.{prod_str}.{horizon}.{name}', start_date=start_date, debug=self.debug)
        return alphas_df

    def generate_rolling_alphas(self, fit_file: str, start_date: date, end_date: date, verbose: bool = False, chunk_days: int = 90, pool_size: Optional[int] = 1):
        window_start_date = start_date
        window_end_date = min([window_start_date + td(days=chunk_days), today_date(), end_date])

        # some features only here in 1440
        horizons = list(set(self.horizons + [1440]))
        features = unique_list(extract_tree_features(self.config, horizons=self.horizons) + ['dvolume_1440_trmean', 'tradeable'])
        classification_bar_features = make_classification_bar_features(horizons)
        cx_features = [feature for feature in features if feature.startswith('cx.')]
        features = [feature for feature in features if feature not in classification_bar_features and feature not in cx_features]
        pool_size = len(horizons) if pool_size is None else pool_size
        logger.info(f"Generating rolling alphas between {start_date} and {end_date} with {pool_size=} {chunk_days=}")

        while window_start_date <= end_date:
            logger.info(f"Generating alphas between {window_start_date} and {window_end_date}")
            self.classifier_dict = self.data_loader.load_classifiers(
                start_date=window_start_date,
                end_date=window_end_date,
                horizons=horizons,
                models=self.models_to_run,
                classifier_dir=self.classifier_dir,
            )
            features_df = self.data_loader.load_features(
                start_date=window_start_date,
                end_date=window_end_date,
                horizons=horizons,
                cols=features,
            )

            if features_df is None:
                raise log_and_raise(f"No feature files could be loaded for this period {window_start_date} to {window_end_date}")

            logging.info(f"Loaded non-bar features: {cols_to_list_str(features_df)}")
            # features_df = features_df[features_df['tradeable'].fillna(value=False)]

            for horizon in reversed(self.horizons):
                models_to_run = extract_models(self.config, [horizon])
                models_to_run = list(set(self.models_to_run) & set(models_to_run))

                max_lags = self.max_lags[horizon]
                lag_days = max_lags * math.ceil(horizon / 1440)
                lagged_model_start_date = window_start_date - td(days=lag_days)
                logger.info(f"Generating alpha files at {horizon=} {max_lags=} from lagged start {lagged_model_start_date}")

                models_df = self.data_loader.load_models(
                    start_date=lagged_model_start_date,
                    end_date=window_end_date,
                    horizon=horizon,
                    models=models_to_run,
                    verbose=verbose,
                )
                if models_df is None:
                    logger.info(f"No models to run at horizon {horizon}")
                    continue

                # needed for classifier.  not sure if this is the right place...
                #its not
                for col in models_df.columns:
                    if col.endswith('_L0'):
                        models_df[f"{col}_abs"] = models_df[col].abs()

                models_df = merge_on_index(models_df, features_df)

                bars_df = self.data_loader.load_bars(
                    horizon=horizon,
                    start_date=lagged_model_start_date,
                    end_date=window_end_date,
                    cols=[f'logret_{horizon}', f'dvolume_{horizon}', f'last_funding_rate_mean_{horizon}'],
                )
                models_df = merge_on_index(models_df, bars_df)

                horizon_cx_features = extract_tree_features(self.config, horizons=[horizon])
                horizon_cx_features = [feature for feature in horizon_cx_features if feature.startswith("cx.")]
                models_df = make_cx_features(models_df, horizon_cx_features)

                if self.prod:
                    assert fit_file is None
                    fit_df = self.data_loader.load_fits(
                        prod=True,
                        horizons=self.horizons,
                        end_date=window_end_date,
                        fits_dir=self.fits_dir,
                    )
                elif fit_file is not None:
                    fit_df = self.data_loader.load_fits(fit_file=fit_file, fits_dir=self.fits_dir)
                else:
                    fit_df = self.data_loader.load_fits(
                        horizons=[horizon],
                        models=models_to_run,
                        start_date=window_start_date,
                        end_date=window_end_date,
                        fits_dir=self.fits_dir,
                    )

                models = self.config['FCASTS'][str(horizon)]['models']
                if self.models_to_run is not None:
                    models = [m for m in models if m['name'] in self.models_to_run]

                for model in models.copy():
                    model_name = f"{model['name']}_{horizon}"
                    if f"{model_name}_L0" not in models_df.columns:
                        logger.warning(f"No model data found for {model_name} in {window_start_date} - {window_end_date}, not generating alpha")
                        models.remove(model)

                if pool_size == 1:
                    for fc in models:
                        self._generate_model_alpha(fcast=fc, horizon=horizon, models_df=models_df, fit_df=fit_df, start_date=window_start_date)
                else:
                    pool = Pool(processes=pool_size)
                    for _ in pool.starmap(
                            self._generate_model_alpha,
                            [(fc, horizon, models_df.copy(), fit_df.copy(), window_start_date) for fc in models]):
                        pass
                    pool.close()
                    pool.join()

            window_start_date += td(days=chunk_days)
            window_end_date = min([window_start_date + td(days=chunk_days), today_date(), end_date])

    def apply_coeffs(
            self,
            forecast: dict,
            fit_df: pd.DataFrame,
            models_df: pd.DataFrame,
            horizon: int,
            prod: bool = False
    ) -> Tuple[pd.DataFrame, List[str]]:
        check_df(models_df)

        name = indep = forecast['name']
        lags = int(forecast['lags'])
        fit_type = forecast['fit_type']

        min_fit_date = fit_df['as_of'].min()
        models_df = models_df[models_df.index.get_level_values('ts') > min_fit_date]
        if len(models_df) == 0:
            raise log_and_raise(f"No models data after fit date {min_fit_date}, not setting coefficients (--latest?)...")
        min_ts, max_ts = get_min_max_ts(models_df)
        logger.info(f"Setting coeffs on models from {min_ts} to {max_ts}")
        last_asof = max_ts

        alpha_col = f'alpha_{name}_{horizon}'
        alpha_col_rev = f"{alpha_col}_rev"
        alpha_col_mom = f"{alpha_col}_mom"
        indep_horizon = f"{indep}_{horizon}"
        alpha_cols = [alpha_col, alpha_col_rev, alpha_col_mom]
        for col in alpha_cols:
            models_df[col] = np.float32(0)

        # zero out weak or wrong direction alpha
        fit_df.loc[fit_df['tstat'].abs() < self.min_tstat, 'coeff_smooth'] = 0
        if self.zero_out_wrong_direction_coeffs:
            fit_df.loc[(fit_df['condition'] == 'rev') & (fit_df['coeff_smooth'] > 0), 'coeff_smooth'] = 0
            fit_df.loc[(fit_df['condition'] == 'mom') & (fit_df['coeff_smooth'] < 0), 'coeff_smooth'] = 0

        model_fit_df = fit_df[(fit_df['name'] == name) & (fit_df['horizon'] == horizon)]
        if not prod:
            model_fit_df = model_fit_df[model_fit_df['as_of'] <= max_ts]

        if len(model_fit_df) == 0:
            logger.warning(f"No fit coeffs for {name=} {horizon=} {max_ts=}!")
            raise NoFitsError(f"No fit coeffs for {name=} {horizon=} {max_ts=}!")

        logger.info(f"Processing coeffs on {alpha_col} for {lags=}")

        # initialize the coeffs and conditions
        for ii in range(0, lags + 1):
            alpha_lag_col = f'{indep_horizon}_L{ii}'
            condition_col = f'{alpha_lag_col}_condition'
            coeff_col = f'{alpha_lag_col}_coeff'
            err_col = f'{alpha_lag_col}_err'
            models_df[condition_col] = np.int8(0)
            models_df[coeff_col] = np.float32(0.0)
            models_df[err_col] = np.float32(0.0)

        fit_dates = sorted(model_fit_df['as_of'].unique())
        logger.info(f"Using fitting dates of {fit_dates}")
        for asof in reversed(fit_dates):
            if last_asof < min_ts:
                logger.info(f"{last_asof=} < {min_ts=}, so not processing more fix dates")
                break

            logger.info(f"processing {alpha_col} with fits between {asof=} and {last_asof=}")
            # set alphas for timestamps strictly greater than the end of the relevant fit
            if pd.isna(asof):
                raise log_and_raise(f"NaT in fit_df for {alpha_col}")

            #fit as_of ts means we should not apply it until t+1
            time_loc = (models_df.index.get_level_values('ts') > asof) & (models_df.index.get_level_values('ts') <= last_asof)
            model_ts_df = models_df[time_loc]

            total_row_cnt = len(model_ts_df)
            if total_row_cnt == 0:
                logger.warning(f"No alphas {alpha_col} between {asof} and {last_asof} for {indep_horizon}")
                last_asof = asof
                continue

            coeff_min_ts, coeff_max_ts = get_min_max_ts(model_ts_df)
            logger.info(f"Applying coeffs on models between {coeff_min_ts} and {coeff_max_ts}")

            model_asof = model_ts_df.index.get_level_values('ts').min().date()

            use_classifier = fit_type not in ('vanilla', 'security')
            if use_classifier:
                model_ts_df = self._set_class(models_df=model_ts_df, indep_horizon=indep_horizon, asof=model_asof, horizon=horizon, name=name)

                mom_cond_loc = model_ts_df['class'] == 1
                rev_cond_loc = model_ts_df['class'] == -1

                mom_len = len(model_ts_df[mom_cond_loc])
                rev_len = len(model_ts_df[rev_cond_loc])
                unk_len = len(model_ts_df[model_ts_df['class'] == 0])
                logger.info(f"Split obs in mom:{mom_len}, rev:{rev_len}, unk:{unk_len}")

            for ii in range(0, lags + 1):
                alpha_lag_col = f'{indep_horizon}_L{ii}'
                logger.info(f"Processing coeffs on {alpha_lag_col}")

                condition_col = f'{alpha_lag_col}_condition'
                coeff_col = f'{alpha_lag_col}_coeff'
                err_col = f'{alpha_lag_col}_err'
                weight_col = f'{alpha_lag_col}_weight'

                if alpha_lag_col not in alpha_cols:
                    alpha_cols += [alpha_lag_col, coeff_col, condition_col, err_col, weight_col]

                lag_fit_df = model_fit_df[(model_fit_df['as_of'] == asof) & (model_fit_df['lag'] == ii)]

                ok_row_cnt = 0
                lag_log_str = f"{alpha_col}:{ii}"

                for _, row in lag_fit_df.sort_values(by='condition').iterrows():
                    coeff = float(row['coeff_smooth'])
                    condition = row['condition']
                    std_err = float(row['stderr'])
                    cond_log_str = f"{lag_log_str}:{condition}: "

                    logger.info(f"{cond_log_str} Setting alpha from {asof} to {last_asof} coeff:{coeff}")
                    assert condition in ("rev", "mom")

                    if use_classifier:
                        cond_loc = mom_cond_loc if condition == "mom" else rev_cond_loc
                    else:
                        model_ts_df['class'] = 0
                        cond_loc = model_ts_df['class'] == 0

                    row_cnt = len(model_ts_df[cond_loc])
                    if row_cnt == 0:
                        logger.warning(f"{cond_log_str} No data points in matching condition {condition}...")
                        continue

                    logger.info(f"{cond_log_str} Setting {row_cnt} points, {fpct(row_cnt / total_row_cnt)}")
                    ok_row_cnt += row_cnt

                    model_ts_df.loc[cond_loc, condition_col] = np.int8(-1) if condition == "rev" else np.int8(1)
                    model_ts_df.loc[cond_loc, err_col] = np.float32(std_err)
                    model_ts_df.loc[cond_loc, coeff_col] = np.float32(coeff)
                    failure = self.log_and_check_alphas(model_ts_df.loc[cond_loc, alpha_lag_col], log_str=cond_log_str)

                # use model_ts_alpha_df added to mom or rev since model_ts_df[alpha_col] has been changed as cumulative value

                if self.new_model_weight:
                    inv_err_col = f"{err_col}_inv"
                    model_ts_df[inv_err_col] = remove_infs(1.0 / model_ts_df[err_col]).fillna(0)
                    error_scaling_factor = calc_data_scaling_factor(model_ts_df[inv_err_col])
                    model_ts_df[weight_col] = (0.5 * (np.tanh(error_scaling_factor * (model_ts_df[inv_err_col] - model_ts_df[inv_err_col].mean())) + 1)).fillna(0)
                else:
                    model_ts_df[weight_col] = (1.0 / model_ts_df[err_col]).clip(lower=0, upper=1).fillna(0)

                model_ts_alpha_df = (model_ts_df[alpha_lag_col] * model_ts_df[coeff_col] * model_ts_df[weight_col]).fillna(0.0)
                model_ts_df[alpha_col] += model_ts_alpha_df

                if use_classifier:
                    model_ts_df.loc[mom_cond_loc, f"{alpha_col}_mom"] += model_ts_alpha_df
                    model_ts_df.loc[rev_cond_loc, f"{alpha_col}_rev"] += model_ts_alpha_df

                logger.info(f"Set {fpct(ok_row_cnt / total_row_cnt)} on {indep_horizon} lag={ii}")
                log_mem_usage()
                failure = self.log_and_check_alphas(model_ts_df[alpha_col], log_str=lag_log_str)

            models_df.loc[time_loc, alpha_cols] = model_ts_df[alpha_cols].astype(np.float32)
            last_asof = asof

        return models_df, alpha_cols
