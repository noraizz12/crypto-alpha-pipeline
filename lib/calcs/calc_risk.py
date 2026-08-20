"""Risk calculation utilities for the statistical arbitrage trading system.

This module provides functions for calculating risk metrics including correlation
matrices, volatility measures, and other risk-related calculations.
"""

import logging
from datetime import datetime as dt, timedelta as td
from typing import Optional, Tuple, Literal, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from lib.util.dataframes import get_min_max_ts, get_last_timeslice, log_col_summary, unstack_feature, set_index, remove_infs, concat
from lib.util.logging_util import KeyLogger
from lib.util.util import log_and_raise
from .calc_returns import calculate_weighted_mkt_return
from .calc_util import get_recent_series_df, get_sparse_rows_for_prod

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)

# Constants for beta calculation
MIN_BETA_POINTS = 30
BETA_RET_BOUND = .1
MAX_BETA = 10


def calc_ret_corr(df: pd.DataFrame, resample_frequency: int, ret_col: str = 'logret',
                  start_dt: Optional[dt] = None, simplify_symbol_name: bool = False) -> pd.DataFrame:
    """Calculate correlation matrix of returns across symbols.
    
    Args:
        df: Input dataframe with return data
        resample_frequency: Frequency in minutes for resampling returns
        ret_col: Name of return column to use
        start_dt: Optional start date for correlation calculation
        simplify_symbol_name: Whether to remove 'USDT' suffix from symbol names
    
    Returns:
        Correlation matrix as DataFrame with symbols as index/columns
    """
    if start_dt is not None:
        df = df.loc[df.index.get_level_values('ts') > start_dt]
    ret_df = df[ret_col].unstack()
    if simplify_symbol_name:
        ret_df.columns = [c.split('USDT')[0] for c in ret_df.columns]
    resampled_ret_df = ret_df.resample(f"{resample_frequency}min", label='right', origin='end_day').sum()
    corr_df = resampled_ret_df.corr()
    min_ts, max_ts = get_min_max_ts(df)
    logger.info(f"Calculate {resample_frequency}min {ret_col} correlation {corr_df.mean().mean()} between {min_ts} and {max_ts}")
    return corr_df


def calculate_beta_single_symbol(period_df: pd.DataFrame, symbol_venue: str, ret_col: str, mkt_col: str, period_end: dt) -> Tuple[float, float]:
    """Calculate beta coefficient for a single symbol.
    
    Performs linear regression of symbol returns against market returns to
    estimate market beta. Includes outlier handling and statistical significance.
    
    Args:
        period_df: DataFrame with returns for the lookback period
        symbol_venue: Symbol to calculate beta for
        ret_col: Symbol return column name
        mkt_col: Market return column name
        period_end: End timestamp of calculation period
    
    Returns:
        Tuple of (beta coefficient, t-statistic)
    
    Notes:
        - Requires minimum 30 data points
        - Clips returns at +/-10% to handle outliers
        - Beta is capped at +/-10
    """
    logger.info(f"Calculating beta for {symbol_venue} {period_end}")
    try:
        period_symbol_df = period_df.xs(symbol_venue, level='symbol_venue').dropna()
    except KeyError:
        period_symbol_df = period_df[period_df.index.get_level_values('symbol_venue') == symbol_venue].dropna()

    data_points = len(period_symbol_df)

    # default values for beta and tstat, successful beta should override them
    beta = 1.0
    tstat = 0.0

    if data_points < MIN_BETA_POINTS:
        logger.info(f"Cant calculate beta for {symbol_venue} at {period_end}, {ret_col} and {mkt_col}, not enough data {data_points}!")
        return beta, tstat

    x = period_symbol_df[ret_col].values
    y = period_symbol_df[mkt_col].values

    x_clipped = np.clip(x, -BETA_RET_BOUND, BETA_RET_BOUND)

    if len(np.unique(x_clipped)) < MIN_BETA_POINTS:
        logger.info(f"Cant calculate beta for {symbol_venue} at {period_end}, {ret_col} and {mkt_col}, not enough independent data {len(np.unique(x_clipped))} after clip!")
        return beta, tstat

    # Reshape for sklearn (needs 2D array)
    X = x_clipped.reshape(-1, 1)

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    # Get beta coefficient
    beta = model.coef_[0]

    y_pred = model.predict(X)
    residuals = y - y_pred
    df = data_points - 2
    mse = np.sum(residuals ** 2) / df
    x_mean = np.mean(x_clipped)
    x_diff_sq_sum = np.sum((x_clipped - x_mean) ** 2)
    se_beta = np.sqrt(mse / x_diff_sq_sum)

    tstat = beta / se_beta

    # Clip beta if needed
    if beta > MAX_BETA:
        logger.info(f"Bad beta {symbol_venue} {beta}")
        beta = MAX_BETA
    elif beta < -MAX_BETA:
        logger.info(f"Bad beta {symbol_venue} {beta}")
        beta = -MAX_BETA

    return beta, tstat


def calc_semi_std(x, side: Literal[1, -1]):
    """Calculate semi-standard deviation (upside or downside).
    
    Computes standard deviation considering only values above (upside) or
    below (downside) the mean.
    
    Args:
        x: Array of values
        side: 1 for upside deviation, -1 for downside deviation
    
    Returns:
        Semi-standard deviation value
    """
    mean = x.mean()
    if side == 1:
        deviations = np.maximum(x - mean, 0)
    else:
        deviations = np.maximum(mean - x, 0)
    return np.sqrt(np.sum(deviations ** 2) / len(x))


def calculate_trailing_semi_variance(
        df: pd.DataFrame,
        feature: str,
        frequency: int,
        start_dt: Optional[dt] = None,
        lookback_periods: Optional[int] = None,
        resample_frequency: Optional[int] = None,
        ffill_na: bool = False,
        feature_lookback_periods: Optional[int] = None,
        lagged_lookback_minutes: Optional[int] = None,
        reopt_times: Optional[List[dt]] = None,
        max_lags: Optional[int] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate trailing upside and downside semi-variance.
    
    Computes separate volatility measures for positive and negative deviations
    from the mean, useful for asymmetric risk assessment.
    
    Args:
        df: Input dataframe with feature data
        feature: Feature name to calculate semi-variance for
        frequency: Time frequency in minutes
        start_dt: Start datetime for calculation
        lookback_periods: Number of periods for rolling window
        resample_frequency: Frequency for resampling
        ffill_na: Whether to forward fill missing values
        feature_lookback_periods: Default lookback periods if not specified
        lagged_lookback_minutes: Lagged lookback minutes for production
        reopt_times: Reoptimization times for production
        max_lags: Maximum lags for production
    
    Returns:
        Tuple of (DataFrame with upside/downside std and ratio columns, list of new column names)
    
    Notes:
        - Creates {feature}_trstd_u (upside) and {feature}_trstd_d (downside)
        - Ratio indicates skewness of return distribution
    """
    if lookback_periods is None:
        lookback_periods = feature_lookback_periods or 90  # default

    if resample_frequency is None:
        resample_frequency = frequency

    lookback_minutes = lookback_periods * frequency

    if feature not in df.columns:
        logger.error(f"{feature} not in dataframe, not calculating trailing semi variance")
        return df, []

    new_cols = [f'{feature}_trstd_u', f'{feature}_trstd_d', f'{feature}_trstd_udratio']
    for col in new_cols:
        if col not in df.columns:
            df[col] = np.nan

    logger.info(f"Calculating trailing semi variance on {feature} with {lookback_minutes=}...")

    series_df, window = get_recent_series_df(
        df=df,
        start_dt=start_dt,
        lookback_periods=lookback_periods,
        frequency=frequency,
        feature_freq=feature,
        lagged_lookback_minutes=lagged_lookback_minutes
    )

    if ffill_na:
        series_df = series_df.ffill()
    else:
        series_df = series_df.fillna(0.0)

    min_ts, max_ts = get_min_max_ts(series_df)

    logger.info(f"Calculating trailing semi variance {feature} from {min_ts} to {max_ts} with lookback minutes {lookback_minutes}, rolling window of {window} periods...")
    rolling = series_df.resample(f"{resample_frequency}min", label='right', origin='end_day').last().rolling(window=window)

    trailing_stdup_df = pd.DataFrame(rolling.apply(lambda x: calc_semi_std(x, 1)).astype(np.float32))
    trailing_stddown_df = pd.DataFrame(rolling.apply(lambda x: calc_semi_std(x, -1)).astype(np.float32))
    trailing_stdudratio_df = (trailing_stdup_df / trailing_stddown_df).rename(columns={f"{feature}": f'{feature}_trstd_udratio'})
    trailing_stdup_df = trailing_stdup_df.rename(columns={f"{feature}": f'{feature}_trstd_u'})
    trailing_stddown_df = trailing_stddown_df.rename(columns={f"{feature}": f'{feature}_trstd_d'})
    concat_dfs = [trailing_stdup_df, trailing_stddown_df, trailing_stdudratio_df]

    both_df = pd.concat(concat_dfs, axis=1, verify_integrity=True)
    new_df = both_df.resample("1min", label='right', closed='right').last().ffill(limit=frequency).stack(future_stack=True).sort_index()
    new_df = new_df.loc[new_df.index.get_level_values('ts') <= max_ts]

    if reopt_times is not None:
        new_df = get_sparse_rows_for_prod(new_df, reopt_times=reopt_times, max_lags=max_lags, frequency=frequency)

    if start_dt is not None:
        new_df = new_df[new_df.index.get_level_values('ts') > start_dt]

    logger.info(f"Merging {new_cols}")
    new_df = new_df.reindex(df[df.index.get_level_values('ts') >= new_df.index.get_level_values('ts').min()].index)
    df.loc[new_df.index, new_cols] = new_df

    for col in new_cols:
        log_col_summary(get_last_timeslice(df), col)

    return df, new_cols


def calculate_risk_factor(
        df: pd.DataFrame,
        feature: str,
        frequency: int,
        start_dt: Optional[dt] = None,
        prod: bool = False,
        lagged_lookback_minutes: Optional[int] = None,
        feature_sigma_bound: float = 5,
        filter_col: str = 'fittable') -> Tuple[pd.DataFrame, List[str]]:
    """Calculate cross-sectional z-scores for risk metrics.
    
    Computes cross-sectional mean and standard deviation at each timestamp,
    then calculates z-scores for use in risk-based position sizing.
    
    Args:
        df: Input dataframe with feature data
        feature: Feature name to calculate cross-sectional z-score for
        frequency: Time frequency in minutes
        start_dt: Optional start datetime
        prod: Whether running in production mode
        lagged_lookback_minutes: Lagged lookback for production mode
        feature_sigma_bound: Bound for z-score clipping
        filter_col: Column name to use for filtering (default: 'fittable')
    
    Returns:
        Tuple of (DataFrame with cross-sectional z-score column, list of new column names)
    
    Notes:
        - Creates {feature}_cz column with cross-sectional z-scores
        - Z-scores are clipped to [-feature_sigma_bound, feature_sigma_bound]
        - In production mode, adjusts start time by lagged lookback
    """

    zscore_name = f"{feature}_cz"
    if prod and start_dt is not None and lagged_lookback_minutes is not None:
        start_dt_old = start_dt
        start_dt -= td(minutes=lagged_lookback_minutes)
        logger.info(f"Calculating {zscore_name} at {frequency} moving start from {start_dt_old} -> {start_dt}")
    else:
        logger.info(f"Calculating {zscore_name} at {frequency} from {start_dt}")

    resampled_fitted_df = unstack_feature(features_df=df[df[filter_col].fillna(False)], feature=feature, start=start_dt)

    # DIAGNOSTIC: Log cross-sectional statistics
    n_symbols = resampled_fitted_df.shape[1]
    n_timestamps = resampled_fitted_df.shape[0]
    nan_per_ts = resampled_fitted_df.isna().sum(axis=1)
    valid_symbols_per_ts = n_symbols - nan_per_ts
    logger.info(f"DIAG {feature}_cz: Unstacked {n_timestamps} timestamps x {n_symbols} fittable symbols, "
                f"avg {valid_symbols_per_ts.mean():.1f} valid per ts")

    # Check for identical values across symbols
    unique_per_ts = resampled_fitted_df.nunique(axis=1)
    low_unique_ts = (unique_per_ts <= 2).sum()
    logger.info(f"DIAG {feature}_cz: {low_unique_ts}/{n_timestamps} timestamps have <=2 unique values across symbols")

    means_df = resampled_fitted_df.mean(axis=1).rename(f'{feature}_crmean')
    stds_df = resampled_fitted_df.std(axis=1).rename(f'{feature}_crstd')

    # DIAGNOSTIC: Log what's causing zero stds
    zero_std_ts = stds_df[stds_df == 0].dropna()
    if len(zero_std_ts) > 0:
        sample_ts = zero_std_ts.index[:3]
        for ts in sample_ts:
            row_values = resampled_fitted_df.loc[ts].dropna()
            logger.info(f"DIAG {feature}_cz: Sample zero-std timestamp {ts}: {len(row_values)} values, "
                        f"unique={row_values.nunique()}, values={row_values.value_counts().head(3).to_dict()}")

    both_df = pd.concat([means_df, stds_df], axis=1).reset_index()
    resampled_df = unstack_feature(features_df=df, feature=feature, start=start_dt)
    stacked_df = resampled_df.stack(future_stack=True).reset_index()
    new_df = pd.merge(stacked_df, both_df, on='ts', how='left')

    # if len(new_df[new_df[f'{feature}_crstd'] == 0]):
    #     logger.warning(f"Std deviation of 0 for {feature}")
    # new_df[zscore_name] = remove_infs((new_df[feature] - new_df[f'{feature}_crmean']) / new_df[f'{feature}_crstd']).fillna(0.0)

    # Check for zero or NaN standard deviations
    zero_std_mask = (new_df[f'{feature}_crstd'] == 0) | new_df[f'{feature}_crstd'].isna()
    if zero_std_mask.any():
        logger.warning(f"Cross sectional Std deviation of 0 or NaN for {feature} on {zero_std_mask.sum()} / {len(new_df)} points")

    # Calculate z-scores, handling division by zero/NaN cases
    # When std is 0 or NaN, set z-score to 0
    new_df[zscore_name] = 0.0  # Initialize with 0
    valid_std_mask = ~zero_std_mask
    if valid_std_mask.any():
        new_df.loc[valid_std_mask, zscore_name] = (
            (new_df.loc[valid_std_mask, feature] - new_df.loc[valid_std_mask, f'{feature}_crmean']) /
            new_df.loc[valid_std_mask, f'{feature}_crstd']
        )

    #not sure i really like this, but when this doesn't happen we end up with stuff in the index that isn't in the original dataframe
    new_df = new_df[~new_df[zscore_name].isna()]

    # Remove any remaining inf/nan values and clip
    new_df[zscore_name] = remove_infs(new_df[zscore_name]).fillna(0.0)
    new_df[zscore_name] = new_df[zscore_name].clip(lower=-feature_sigma_bound, upper=feature_sigma_bound)

    pinned_zscores_df = new_df[new_df[zscore_name].abs() == feature_sigma_bound]
    if len(pinned_zscores_df) > 0:
        logger.warning(f"Cross sectional zscore hitting bound on {len(pinned_zscores_df)} points for {feature}")

    new_df = set_index(new_df)[[zscore_name]]

    # Properly merge the z-score values back into the original dataframe
    # Use the index from new_df to ensure proper alignment
    new_idx = new_df.index.intersection(df.index)
    df.loc[new_idx, zscore_name] = new_df[zscore_name]
    new_cols = [zscore_name]

    return df, new_cols

def apply_old_position_risk_multiplier(df: pd.DataFrame, risk_col: str, old_position_days: int, old_position_risk_mult: float) -> pd.DataFrame:
    """Apply increased risk penalty for old positions.
    
    Multiplies risk metric for positions older than threshold to encourage
    closing stale positions.
    
    Args:
        df: Input dataframe with position age data
        risk_col: Name of risk column to adjust
        old_position_days: Days threshold for considering position old
        old_position_risk_mult: Multiplier to apply to old positions
    
    Returns:
        Tuple of (DataFrame with adjusted risk values, empty list)
    
    Notes:
        - Positions > old_position_days get old_position_risk_mult applied
        - Helps prevent holding losing positions indefinitely
        - Returns empty list for consistency with other functions
    """
    if 'position_age' not in df.columns:
        logger.warning("Cannot find position age in dataframe to apply old position multiplier")
        return df

    old_position_loc = df['position_age'] > old_position_days
    if old_position_loc.any():
        df.loc[old_position_loc, risk_col] = df[risk_col] * old_position_risk_mult
        cnt = len(df[old_position_loc])
        logger.info(f"Positions over {old_position_days} days old: {cnt}")

    return df


def calculate_risk(df: pd.DataFrame, risk_frequency: int, sample_frequency: int,
                   risk_fld: str, old_position_days: int, old_position_risk_mult: float) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate risk metrics from trailing volatility.
    
    Computes annualized risk measure based on return standard deviation,
    properly scaled for the measurement frequency.
    
    Args:
        df: Input dataframe with return std columns
        risk_frequency: Risk horizon in minutes
        sample_frequency: Data sampling frequency in minutes
        risk_fld: Template for risk field name (contains HORIZON placeholder)
        old_position_days: Days threshold for old positions
        old_position_risk_mult: Risk multiplier for old positions
    
    Returns:
        Tuple of (DataFrame with risk column, list of new column names)
    
    Notes:
        - Risk = volatility * sqrt(horizon/year_minutes)
        - Applies old position risk multiplier
        - Validates that risk data has variation
    """
    risk_col = f'risk_{risk_frequency}'
    risk_fld_resolved = risk_fld.replace("HORIZON", str(sample_frequency))
    logger.info(f"Calculating risk field: {risk_fld_resolved} -> {risk_col}")
    df[risk_col] = df[risk_fld_resolved] * np.sqrt(risk_frequency / sample_frequency)

    df = apply_old_position_risk_multiplier(df, risk_col, old_position_days, old_position_risk_mult)

    if df[risk_col].std() == 0 or df[risk_col].fillna(0).sum() == 0:
        raise log_and_raise("Bad Risk Data!, no variation or all nan", df=df)
    log_col_summary(df, risk_col)

    return df, [risk_col]


def calculate_pca(
        df: pd.DataFrame,
        frequency: int,
        start_dt: Optional[dt] = None,
        lookback_periods: Optional[int] = None,
        beta_lookback_periods: Optional[int] = None,
        prod: bool = False,
        eigenvalue_threshold: float = 0.01
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate PCA-based residualized returns using proper factor model methodology.

    Performs Principal Component Analysis on cross-sectional returns to extract
    systematic market factors, then calculates residual (idiosyncratic) returns
    by subtracting the systematic component from each symbol's return.

    Mathematical Framework:
    -----------------------
    Given returns matrix R (T x N) where T = timestamps, N = symbols:

    1. Fit PCA on historical returns to get eigenvectors V and eigenvalues
    2. Keep k eigenvectors where eigenvalue_i / sum(eigenvalues) >= threshold
    3. Project current returns onto factors: F = R @ V[:, :k]  (factor scores)
    4. Reconstruct systematic returns: R_sys = F @ V[:, :k].T
    5. Calculate residuals: R_resid = R - R_sys

    Args:
        df: Input dataframe with return data, multi-indexed by (ts, symbol_venue)
        frequency: Time frequency in minutes (e.g., 1440 for daily)
        start_dt: Start datetime for calculation (required for non-prod)
        lookback_periods: Number of periods for rolling PCA window
        beta_lookback_periods: Default lookback if lookback_periods not specified
        prod: Whether running in production mode (single timestamp vs full history)
        eigenvalue_threshold: Minimum explained variance ratio to retain component
            (default 0.01 = 1%)

    Returns:
        Tuple of (DataFrame with PCA residual columns, list of new column names)

    Output Columns:
        - pca_resid_{frequency}: Residualized return (actual - systematic)
        - pca_factor_ret_{frequency}: Aggregate systematic factor return (non-prod only)
        - pca_n_components_{frequency}: Number of components retained (non-prod only)
        - pca_explained_var_{frequency}: Total variance explained (non-prod only)

    Notes:
        - Residuals represent idiosyncratic returns not explained by common factors
        - In production mode, only calculates for the most recent timestamp
        - Mean pairwise correlation of residuals should be near zero
    """
    if lookback_periods is None:
        lookback_periods = beta_lookback_periods

    ret_col = f'logret_{frequency}'
    lookback_minutes = lookback_periods * frequency

    logger.info(
        f"Calculating PCA_new at {frequency}min with lookback={lookback_periods} periods "
        f"({lookback_minutes} mins), eigenvalue_threshold={eigenvalue_threshold}, "
        f"start_dt={start_dt}, prod={prod}"
    )

    # Output column names
    resid_col = f'pca_resid_{frequency}'
    new_cols = [resid_col]
    if not prod:
        new_cols.extend([
            f'pca_factor_ret_{frequency}',
            f'pca_n_components_{frequency}',
            f'pca_explained_var_{frequency}'
        ])

    # Get returns data as matrix: rows=timestamps, cols=symbols
    returns_df = remove_infs(unstack_feature(features_df=df, feature=ret_col)).fillna(0)

    if len(returns_df) == 0:
        logger.error("No returns data available for PCA calculation")
        return df, []

    # Extract symbol names from MultiIndex columns (feature_name, symbol_venue)
    symbols = [col[1] for col in returns_df.columns]
    returns_df.columns = symbols
    n_symbols = len(symbols)

    # Determine timestamps to process
    last_ts = df.index.get_level_values('ts').max()
    if prod:
        timesteps = [last_ts]
    else:
        if start_dt is None:
            # Default: start after lookback period
            start_dt = returns_df.index.min() + td(minutes=lookback_minutes)
        timesteps = pd.date_range(start=start_dt, end=last_ts, freq=f"{frequency}min")

    logger.info(
        f"PCA_new: Processing {len(timesteps)} timesteps from "
        f"{timesteps[0] if len(timesteps) > 0 else 'N/A'} to "
        f"{timesteps[-1] if len(timesteps) > 0 else 'N/A'}"
    )

    result_dfs = []

    for ts in timesteps:
        # Get historical returns for PCA fitting
        hist_start = ts - td(minutes=lookback_minutes)
        hist_returns_df = returns_df[(returns_df.index >= hist_start) & (returns_df.index < ts)]

        if len(hist_returns_df) < max(30, n_symbols):
            logger.warning(
                f"PCA_new: Insufficient history at {ts}: {len(hist_returns_df)} rows, "
                f"need at least {max(30, n_symbols)}"
            )
            continue

        # Get returns for current timestamp
        current_returns_row = returns_df[returns_df.index == ts]
        if len(current_returns_row) == 0:
            logger.warning(f"PCA_new: No data for timestamp {ts}, skipping")
            continue

        current_returns = current_returns_row.values.flatten()

        try:
            # Step 1: Fit PCA on historical returns
            pca = PCA(n_components=None, random_state=42)
            pca.fit(hist_returns_df.values)

            explained_var_ratio = pca.explained_variance_ratio_

            # Step 2: Determine components to keep based on eigenvalue threshold
            n_keep = int(np.sum(explained_var_ratio >= eigenvalue_threshold))
            if n_keep == 0:
                logger.warning(
                    f"PCA_new: No components above threshold {eigenvalue_threshold} at {ts}, "
                    f"using top component"
                )
                n_keep = 1

            # Step 3: Get eigenvectors (loadings) for retained components
            # pca.components_ shape: (n_components, n_features) = (n_components, n_symbols)
            # We want V as (n_symbols, n_keep)
            eigenvectors = pca.components_[:n_keep, :].T  # (n_symbols, n_keep)

            # Step 4: Project current returns onto factors to get factor scores
            # factor_scores = R @ V -> (1,) @ (n_symbols, n_keep) needs reshape
            # Actually: current_returns is (n_symbols,), eigenvectors is (n_symbols, n_keep)
            factor_scores = current_returns @ eigenvectors  # (n_keep,)

            # Step 5: Reconstruct systematic returns
            # systematic = factor_scores @ V.T -> (n_keep,) @ (n_keep, n_symbols) = (n_symbols,)
            systematic_returns = factor_scores @ eigenvectors.T  # (n_symbols,)

            # Step 6: Calculate residuals
            residuals = current_returns - systematic_returns

            # Build output dataframe for this timestamp
            ts_result_df = pd.DataFrame(index=symbols)
            ts_result_df[resid_col] = residuals.astype(np.float32)

            total_explained_var = float(np.sum(explained_var_ratio[:n_keep]))

            if not prod:
                # Aggregate factor return: weighted by explained variance
                weights = explained_var_ratio[:n_keep]
                weights_normalized = weights / weights.sum()
                aggregate_factor_ret = float(np.dot(factor_scores, weights_normalized))

                ts_result_df[f'pca_factor_ret_{frequency}'] = np.float32(aggregate_factor_ret)
                ts_result_df[f'pca_n_components_{frequency}'] = np.float32(n_keep)
                ts_result_df[f'pca_explained_var_{frequency}'] = np.float32(total_explained_var)

            if prod:
                logger.info(
                    f"PCA_new at {ts}: retained {n_keep} components, "
                    f"explained_var={total_explained_var:.3f}, "
                    f"mean_abs_resid={np.mean(np.abs(residuals)):.6f}"
                )

            ts_result_df['ts'] = ts
            result_dfs.append(ts_result_df)

        except Exception as e:
            logger.error(f"PCA_new: Failed at {ts}: {e}", exc_info=True)
            # Create zero-filled result for this timestamp
            ts_result_df = pd.DataFrame(index=symbols)
            ts_result_df[resid_col] = np.float32(0)
            if not prod:
                ts_result_df[f'pca_factor_ret_{frequency}'] = np.float32(0)
                ts_result_df[f'pca_n_components_{frequency}'] = np.float32(0)
                ts_result_df[f'pca_explained_var_{frequency}'] = np.float32(0)
            ts_result_df['ts'] = ts
            result_dfs.append(ts_result_df)

    if len(result_dfs) == 0:
        logger.warning("PCA_new: No results generated")
        return df, []

    # Combine results and merge back to original dataframe
    if prod:
        pca_df = set_index(result_dfs[0].reset_index().rename(columns={'index': 'symbol_venue'}))
    else:
        combined_df = concat(result_dfs)
        combined_df = combined_df.reset_index().rename(columns={'index': 'symbol_venue'})
        pca_df = set_index(combined_df)

        # Resample to 1-min and forward fill to match original dataframe frequency
        pca_df = pca_df.unstack() \
            .resample("1min", label='right', closed='right') \
            .last() \
            .ffill() \
            .stack(future_stack=True)

    # Merge results back to original dataframe
    common_idx = pca_df.index.intersection(df.index)
    for col in new_cols:
        if col in pca_df.columns:
            df.loc[common_idx, col] = pca_df.loc[common_idx, col].astype(np.float32)

    # Log summary statistics
    if resid_col in df.columns:
        resid_data = df[resid_col].dropna()
        if len(resid_data) > 0:
            logger.info(
                f"PCA_new results: {len(resid_data)} residuals, "
                f"mean={resid_data.mean():.6f}, std={resid_data.std():.6f}"
            )

    return df, new_cols


def calculate_beta(df: pd.DataFrame, frequency: int, start_dt: Optional[dt] = None,
                   lookback_periods: Optional[int] = None, beta_lookback_periods: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate rolling market beta for all symbols.

    Computes beta coefficients by regressing funding-adjusted returns against
    volume-weighted market returns over a rolling window.

    Args:
        df: Input dataframe with return data
        frequency: Time frequency in minutes
        start_dt: Start datetime for beta calculation
        lookback_periods: Number of periods for rolling window
        beta_lookback_periods: Default lookback if not specified

    Returns:
        Tuple of (DataFrame with added beta and beta t-statistic columns, list of new column names)

    Notes:
        - Uses funding-adjusted returns for accurate beta estimation
        - Forward fills beta values between calculation points
        - Requires pre-calculated weighted market returns
    """
    if lookback_periods is None:
        lookback_periods = beta_lookback_periods

    # First calculate weighted market return if needed
    df, wgt_col = calculate_weighted_mkt_return(df, filter_col='fittable', ret_col='logret_funding_adj', frequency=frequency)

    end = df.index.get_level_values('ts').max()
    beta_name = f'beta_{frequency}'
    beta_t_name = f'beta_t_{frequency}'
    ret_col = f"logret_funding_adj_{frequency}"
    mkt_col = f'logret_funding_adj_wgtmkt_{frequency}'
    lookback_minutes = lookback_periods * frequency
    logger.info(f"Calculating {beta_name} with data from {start_dt} to {end} with lookback {lookback_periods}")

    # Note: Assumes weighted market return is already calculated
    symbol_venues = df.index.get_level_values('symbol_venue').unique()
    timesteps = pd.date_range(start=start_dt, end=end, freq=f"{frequency}min")
    for period_end in timesteps:
        period_start = period_end - td(minutes=lookback_minutes)
        period_df = df[(df.index.get_level_values('ts') >= period_start) & (df.index.get_level_values('ts') < period_end)][[ret_col, mkt_col]]
        for symbol_venue in symbol_venues:
            beta, tstat = calculate_beta_single_symbol(period_df=period_df, symbol_venue=symbol_venue, ret_col=ret_col, mkt_col=mkt_col, period_end=period_end)
            beta_loc = (df.index.get_level_values('symbol_venue') == symbol_venue) & (df.index.get_level_values('ts') == period_end)
            df.loc[beta_loc, beta_name] = beta
            df.loc[beta_loc, beta_t_name] = tstat

    df[beta_name] = df[[beta_name]].unstack().ffill().stack(future_stack=True)
    df[beta_t_name] = df[[beta_t_name]].unstack().ffill().stack(future_stack=True)

    new_cols = [beta_name, beta_t_name, wgt_col]
    return df, new_cols
