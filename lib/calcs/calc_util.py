"""Calculation utilities for the statistical arbitrage trading system.

This module provides shared utility functions used by calcs.py and calc_returns.py
to avoid circular dependencies.
"""

import logging
import math
from datetime import datetime as dt, timedelta as td
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from lib.util.dataframes import remove_infs, unstack_feature, set_index, get_min_max_ts, ensure_utc_ts, get_last_timeslice, log_col_summary, get_df_idx, round_col_to_zero, safe_del, DF_INDEX
from lib.util.time_util import to_datetime
from lib.util.util import unique_list
from lib.util.util import log_and_raise

logger = logging.getLogger(__name__)


def make_market_weight_col(df: pd.DataFrame) -> pd.DataFrame:
    """Add market capitalization weight column based on average daily volume.

    Uses log transformation of ADVP (average daily volume in dollars) to create
    weights that reduce the influence of extreme values while maintaining
    relative importance.

    Args:
        df: DataFrame with 'advp' column

    Returns:
        DataFrame with 'mkt_wgt' column added
    """
    # Use log(advp / 1e6) to preserve original scaling, clip to minimum of 1 to avoid negative weights
    df['mkt_wgt'] = np.log(np.maximum(df['advp'] / 1e6, 1.0))
    return df


def adjust_next_funding_time_jumps_in_live_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Fix funding time jumps that occur at hour boundaries in live data.
    
    Corrects an issue where next_funding_time can jump incorrectly at the
    start of a new hour (minute 0) by using the previous minute's value.
    
    Args:
        df: Input dataframe with 'next_funding_time' column
    
    Returns:
        DataFrame with corrected funding times
    """
    current_hour_idx = df.index.get_level_values('ts').minute == 0
    # Get previous row's next_funding_time (which would be at minute 59)
    prev_funding = df['next_funding_time'].groupby('symbol_venue', observed=False).shift()
    # For rows at minute 0, correct if previous (59th minute) funding time doesn't match current
    mismatch_funding_idx = current_hour_idx & (df['next_funding_time'] != prev_funding)
    # Apply correction using previous funding time
    df.loc[mismatch_funding_idx, 'next_funding_time'] = prev_funding[mismatch_funding_idx]
    return df


def calculate_trailing_z(df: pd.DataFrame, feature: str, feature_sigma_bound: float = 5) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate trailing z-score (standardized value).
    
    Standardizes feature by subtracting trailing mean and dividing by trailing
    standard deviation. Clips extreme values for stability.
    
    Args:
        df: Input dataframe with feature and its trailing statistics
        feature: Base feature name (expects {feature}_trmean and {feature}_trstd)
        feature_sigma_bound: Maximum absolute z-score value (default: 5)
    
    Returns:
        Tuple of (DataFrame with added {feature}_lz column, list of new column names)
    
    Notes:
        - Z-scores are clipped to [-feature_sigma_bound, feature_sigma_bound]
        - Missing values filled with 0
    """
    new_col = f'{feature}_lz'
    df[f'{feature}_demean'] = df[feature] - df[f'{feature}_trmean']
    df = round_col_to_zero(df, f'{feature}_demean')

    # DIAGNOSTIC: Log zero std and inf statistics before final calculation
    zero_std_mask = (df[f'{feature}_trstd'] == 0) | (df[f'{feature}_trstd'].abs() < 1e-10)
    zero_std_count = zero_std_mask.sum()
    logger.info(f"DIAG {feature}_lz: {zero_std_count}/{len(df)} points have zero/near-zero trstd")

    raw_z = df[f'{feature}_demean'] / df[f'{feature}_trstd']
    inf_count = np.isinf(raw_z).sum()
    nan_count = raw_z.isna().sum()
    logger.info(f"DIAG {feature}_lz: Division produced {inf_count} inf, {nan_count} NaN values")

    df[new_col] = (remove_infs(raw_z)).fillna(0).clip(lower=-feature_sigma_bound, upper=feature_sigma_bound).astype(np.float32)

    # DIAGNOSTIC: Log final _lz distribution
    zero_lz_count = (df[new_col] == 0).sum()
    logger.info(f"DIAG {feature}_lz: Final result has {zero_lz_count}/{len(df)} zero values ({100*zero_lz_count/len(df):.1f}%)")

    df = safe_del(df, f'{feature}_demean')
    return df, [new_col]


def calculate_quintiles(df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate cross-sectional quintile rankings.
    
    Assigns quintile labels (1-5) based on cross-sectional ranking at each
    timestamp, useful for relative value strategies.
    
    Args:
        df: Input dataframe with feature column
        col: Column name to calculate quintiles for
    
    Returns:
        Tuple of (DataFrame with quintile column added as {col}_q, list of new column names)
    """
    new_col = f"{col}_q"
    df[new_col] = pd.qcut(df[col], q=5, labels=[f"Q1_{col}", f"Q2_{col}", f"Q3_{col}", f"Q4_{col}", f"Q5_{col}"])
    return df, [new_col]


def calculate_abs_factor(df: pd.DataFrame, feature: str, start_idx: pd.Series = None) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate absolute value of standardized returns.
    
    Creates volatility-like features from absolute z-scores, capturing
    magnitude of moves regardless of direction.
    
    Args:
        df: Input dataframe with z-score columns
        feature: Feature prefix for abs calculation
        start_idx: Optional boolean index mask for filtering rows
    
    Returns:
        Tuple of (DataFrame with absolute z-score column, list of new column names)
    """
    abs_fld = f"{feature}_abs"
    if start_idx is not None:
        df[abs_fld] = df.loc[start_idx, feature].abs()
    else:
        df[abs_fld] = df[feature].abs()
    return df, [abs_fld]


def calculate_time_factors(df: pd.DataFrame, start_idx: pd.Series = None) -> Tuple[pd.DataFrame, List[str]]:
    """Add time-based features for seasonality modeling.
    
    Extracts hour of day and day of week features to capture intraday
    and weekly patterns in trading behavior.
    
    Args:
        df: Input dataframe with timestamp index
        start_idx: Optional boolean index mask for filtering rows
    
    Returns:
        Tuple of (DataFrame with hour_of_day and day_of_week columns, list of new column names)
    """
    new_cols = ['hour_of_day', 'day_of_week']

    if start_idx is None:
        start_idx = pd.Series(True, index=df.index)

    df['ts_start'] = df.index.get_level_values('ts')
    df_small = df[start_idx]
    df.loc[start_idx, 'ts_start'] = df_small.index.get_level_values('ts') - pd.Timedelta(minutes=1)
    df.loc[start_idx, 'hour_of_day'] = df_small['ts_start'].dt.hour

    # 0 = Monday
    df.loc[start_idx, 'day_of_week'] = df_small['ts_start'].dt.dayofweek
    del df['ts_start']
    for col in new_cols:
        df[col] = df[col].fillna(0).astype('Int32')
    return df, new_cols


def calculate_delta(df: pd.DataFrame, feature: str, frequency: int, start_dt: Optional[dt] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate absolute and percentage changes for a feature.
    
    Computes the difference between current and previous values (lagged by frequency).
    Creates two new columns:
    - {feature}_{frequency}_d: Absolute change
    - {feature}_{frequency}_dp: Percentage change
    
    Args:
        df: Input dataframe with feature data
        feature: Base feature name to calculate deltas for
        frequency: Time frequency in minutes for lag calculation
        start_dt: Optional start datetime for calculations
        
    Returns:
        Tuple of (DataFrame with delta columns added, list of new column names)
        
    Example:
        >>> calculate_delta(df, feature='dvolume', frequency=60)
        # Creates dvolume_60_d and dvolume_60_dp columns
    """
    delta_col = f"{feature}_d"
    delta_pct_col = f"{feature}_dp"

    logger.info(f"Calculating deltas on {feature}: {delta_col}")

    idx = get_df_idx(df, start_dt)
    new_df = unstack_feature(df, feature=feature).diff(periods=frequency).stack(future_stack=True)

    df.loc[idx, delta_col] = new_df[feature]
    df.loc[idx, delta_pct_col] = remove_infs(df[delta_col] / (df[feature] - df[delta_col])) - 1.0

    return df, [delta_col, delta_pct_col]


def calculate_weighted_feature(df: pd.DataFrame, feature: str, weight: str) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate weighted version of a feature.
    
    Creates market-wide weighted features using specified weights, typically
    for constructing custom market indices.
    
    Args:
        df: Input dataframe with feature and weight columns
        feature: Feature column to weight
        weight: Weight column name
    
    Returns:
        Tuple of (DataFrame with weighted feature column, list of new column names)
    """
    logger.info(f"Calculating feature {feature} weighted by {weight}")

    assert 'ts' not in df.columns
    weighted_feature = f"{feature}_w_{weight}"
    new_cols = [weighted_feature]

    df[weighted_feature] = df[feature] * df[weight]

    df = df.reset_index()
    weight_sums_df = df[[weight, 'ts']].groupby('ts', observed=False).sum().reset_index()
    weight_sums_df[f"{weight}_sum"] = weight_sums_df[weight]
    del weight_sums_df[weight]
    df = df.reset_index()
    df = pd.merge(df, weight_sums_df, on='ts', how='left')
    df[weighted_feature] = df[weighted_feature] / df[f"{weight}_sum"]
    df = set_index(df)
    return df, new_cols


def get_recent_series_df(df: pd.DataFrame, start_dt: Optional[dt], lookback_periods: int,
                         frequency: int, feature_freq: str, lagged_lookback_minutes: Optional[int] = None) -> Tuple[pd.DataFrame, int]:
    """Get recent data series for rolling calculations.
    
    Extracts and prepares time series data for a specific feature, handling
    lookback requirements and production optimizations.
    
    Args:
        df: Input dataframe with 'ts' in index (must be timezone-aware)
        start_dt: Start datetime for calculations
        lookback_periods: Number of periods needed for lookback
        frequency: Time frequency in minutes
        feature_freq: Feature column name with frequency suffix
        lagged_lookback_minutes: Lagged lookback minutes value (for prod mode)
    
    Returns:
        Tuple of (unstacked series dataframe, adjusted window size)
    
    Notes:
        - In production mode, limits data to reduce memory usage
        - Adjusts window size if insufficient history
    """
    # Ensure df has timezone-aware timestamps in index
    df = ensure_utc_ts(df)

    lookback_minutes = lookback_periods * frequency
    window = lookback_periods

    if start_dt is None:
        recent_df = df
    else:
        # Ensure start_dt is timezone-aware to avoid comparison issues
        start_dt = to_datetime(start_dt)
        min_ts, _ = get_min_max_ts(df)
        start_with_lookback = start_dt - td(minutes=lookback_minutes)
        logger.info(f"{feature_freq} looking back to {start_with_lookback} from start: {start_dt}, {lookback_minutes=}")
        if start_with_lookback < min_ts:
            min_start = min_ts + td(minutes=lookback_minutes)
            raise log_and_raise(f"Not enough lookback for calculation {feature_freq}!, need to start past {min_start} for {lookback_minutes=}, goes back to {min_ts} > {start_with_lookback}")
        recent_df = df[df.index.get_level_values('ts') > start_with_lookback]

    series_df = unstack_feature(features_df=recent_df, feature=feature_freq)
    len_post_resample = math.floor(len(series_df) / frequency)
    if len_post_resample < window:
        logger.warning(f"Not enough history {lookback_periods=}, using lookback window {len_post_resample}")
        window = len_post_resample
    elif lagged_lookback_minutes is not None:
        lookback_window = lookback_minutes + lagged_lookback_minutes
        series_df = series_df.tail(lookback_window)
        min_ts = series_df.index.min()
        logger.info(f"Using last {lookback_window} minutes of history from {min_ts}")

    return series_df, window


def get_sparse_rows_for_prod(df: pd.DataFrame, reopt_times: List[dt], max_lags: int, frequency: int) -> pd.DataFrame:
    """Filter dataframe to only required timestamps for production.
    
    In production, only calculate features at reoptimization times and
    recent timestamps needed for lagged features.
    
    Args:
        df: Input dataframe to filter
        reopt_times: List of reoptimization times as strings
        max_lags: max lags
        frequency: Time frequency in minutes
    
    Returns:
        Filtered dataframe with only necessary timestamps
    """
    times = [pd.Timestamp(tt) for tt in reopt_times]
    _, max_ts = get_min_max_ts(df)
    for lag in range(max_lags + 1):
        times.append(pd.Timestamp(max_ts - td(minutes=frequency * lag)))
    times = sorted(unique_list(times))
    times_str = [tt.strftime('%Y%m%d_%H:%M') for tt in times]
    len_before = len(df)
    df = df[df.index.get_level_values('ts').isin(times)]
    len_after = len(df)
    logger.info(f"Sparse calculating for {frequency}:{max_lags} just at these times {times_str} taking dataframe rows {len_before} -> {len_after}")
    return df


def calculate_trailing_mean(
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
        max_lags: Optional[int] = None,
        feature_sigma_bound: float = 5
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate trailing mean and standard deviation for a feature.
    
    Computes rolling statistics over a lookback window, essential for
    z-score normalization and outlier detection.
    
    Args:
        df: Input dataframe with feature data
        feature: Feature name to calculate statistics for
        frequency: Time frequency in minutes
        start_dt: Start datetime for calculation
        lookback_periods: Number of periods for rolling window
        resample_frequency: Frequency for resampling
        ffill_na: Whether to forward fill missing values
        feature_lookback_periods: Default lookback periods if not specified
        lagged_lookback_minutes: Lagged lookback minutes for production
        reopt_times: Reoptimization times for production
        max_lags: Maximum lags for production
        feature_sigma_bound: Bound for z-score clipping
    
    Returns:
        Tuple of (DataFrame with added trailing mean/std/lz columns, list of new column names)
    
    Notes:
        - Creates {feature}_trmean, {feature}_trstd, and {feature}_lz columns
        - In production mode, only calculates at specific timestamps
        - Handles both bar-level and resampled calculations
    """

    if lookback_periods is None:
        lookback_periods = feature_lookback_periods

    if resample_frequency is None:
        resample_frequency = frequency

    lookback_minutes = lookback_periods * frequency

    if feature not in df.columns:
        logger.error(f"{feature} not in dataframe, not calculating trailing mean")
        return df, []

    new_cols = [f'{feature}_trmean', f'{feature}_trstd', f'{feature}_lz']
    for col in new_cols:
        if col not in df.columns:
            df[col] = np.nan

    logger.info(f"Calculating trailing mean on {feature} with {lookback_minutes=} {ffill_na=} {resample_frequency=} {start_dt=}...")

    series_df, window = get_recent_series_df(
        df=df,
        start_dt=start_dt,
        lookback_periods=lookback_periods,
        frequency=frequency,
        feature_freq=feature,
        lagged_lookback_minutes=lagged_lookback_minutes
    )

    # DIAGNOSTIC: Log NaN statistics before fillna
    nan_count = series_df.isna().sum().sum()
    total_count = series_df.size
    nan_pct = 100 * nan_count / total_count if total_count > 0 else 0
    symbols_with_all_nan = (series_df.isna().all()).sum()
    logger.info(f"DIAG {feature}: Before fillna - {nan_count}/{total_count} NaN values ({nan_pct:.1f}%), "
                f"{symbols_with_all_nan} symbols with ALL NaN")

    if ffill_na:
        series_df = series_df.ffill()
    else:
        series_df = series_df.fillna(0.0)

    min_ts, max_ts = get_min_max_ts(series_df)
    logger.info(f"Calculating trailing mean/std/lz on {feature} from {min_ts} to {max_ts}")

    logger.info(f"Calculating trailing {feature} with lookback minutes {lookback_minutes}, rolling window of {window} periods...")
    rolling = series_df.resample(f"{resample_frequency}min", label='right', origin='end_day').last().rolling(window=window)

    trailing_mean_df = pd.DataFrame(rolling.mean().astype(np.float32)).rename(columns={f"{feature}": f'{feature}_trmean'})
    trailing_std_df = pd.DataFrame(rolling.std().astype(np.float32)).rename(columns={f"{feature}": f'{feature}_trstd'})

    # DIAGNOSTIC: Log zero/near-zero std statistics
    std_values = trailing_std_df.values.flatten()
    zero_std_count = np.sum(np.isclose(std_values, 0, atol=1e-10) | np.isnan(std_values))
    logger.info(f"DIAG {feature}: After rolling std - {zero_std_count}/{len(std_values)} values have zero/NaN std")

    concat_dfs = [trailing_mean_df, trailing_std_df, series_df]

    both_df = pd.concat(concat_dfs, axis=1, verify_integrity=True)
    new_df = both_df.resample("1min", label='right', closed='right').last().ffill(limit=frequency).stack(future_stack=True).sort_index()
    new_df = new_df.loc[new_df.index.get_level_values('ts') <= max_ts]

    if reopt_times is not None:
        logger.info(f"Just setting trailing mean at sparse rows for prod")
        new_df = get_sparse_rows_for_prod(new_df, reopt_times=reopt_times, max_lags=max_lags, frequency=frequency)

    if start_dt is not None:
        logger.info(f"Only keeping data past {start_dt}")
        new_df = new_df[new_df.index.get_level_values('ts') > start_dt]
        if len(new_df) == 0:
            raise log_and_raise(f"No data past {start_dt} in dataframe!")

    new_df, z_cols = calculate_trailing_z(new_df, feature, feature_sigma_bound)
    logger.info(f"Merging {feature} with {new_cols}")
    new_df = new_df.reindex(df[df.index.get_level_values('ts') >= new_df.index.get_level_values('ts').min()].index)
    df.loc[new_df.index, new_cols] = new_df

    new_cols += z_cols
    for col in new_cols:
        log_col_summary(get_last_timeslice(df), col)

    return df, new_cols


def calculate_minmax(
        df: pd.DataFrame,
        feature: str,
        frequency: int,
        start_dt: Optional[dt] = None,
        lookback_periods: Optional[int] = None,
        feature_lookback_periods: Optional[int] = None,
        lagged_lookback_minutes: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate rolling minimum and maximum values.
    
    Tracks extreme values over a lookback window, useful for identifying
    support/resistance levels and mean reversion opportunities.
    
    Args:
        df: Input dataframe with feature data
        feature: Feature name to calculate min/max for
        frequency: Time frequency in minutes
        start_dt: Start datetime for calculation
        lookback_periods: Number of periods for rolling window
        feature_lookback_periods: Default lookback if not specified
        lagged_lookback_minutes: Lagged lookback for production mode
    
    Returns:
        Tuple of (DataFrame with min/max columns, list of new column names)
    
    Notes:
        - Creates {feature}_{frequency}_min and {feature}_{frequency}_max columns
        - Also creates cross-sectional z-scores for min/max values
    """
    if lookback_periods is None:
        lookback_periods = feature_lookback_periods or 90  # default

    logger.info(f"Calculating min/max {feature} with lookback periods {lookback_periods} from {start_dt}")

    new_cols = []
    for flavor in ['min', 'max']:
        for col in [f'{feature}_{flavor}', f'{feature}_{flavor}_cz']:
            new_cols.append(col)

    recent_df, window = get_recent_series_df(
        df=df,
        start_dt=start_dt,
        lookback_periods=lookback_periods,
        frequency=frequency,
        feature_freq=feature,
        lagged_lookback_minutes=lagged_lookback_minutes
    )

    series_max_df = recent_df.rolling(window=window).max().astype(np.float32)
    series_max_df = series_max_df.rename(columns={
        f'{feature}': f'{feature}_max',
    })
    series_min_df = recent_df.rolling(window=window).min().astype(np.float32)
    series_min_df = series_min_df.rename(columns={
        f'{feature}': f'{feature}_min',
    })
    minmax_series_df = pd.concat([series_max_df, series_min_df], axis=1, verify_integrity=True)
    new_df = minmax_series_df.resample("1min", label='right', closed='right').last().ffill().stack(future_stack=True)

    for flavor in ['min', 'max']:
        col = f'{feature}_{flavor}'
        idx = df.index[-len(new_df):]
        df.loc[idx, col] = new_df[col]
        # Calculate cross-sectional z-score using existing mean/std columns
        df.loc[idx, f'{feature}_{flavor}_cz'] = (df[f'{feature}_{flavor}'] - df[f'{feature}_trmean']) / df[f'{feature}_trstd']

    return df, new_cols


def calc_data_scaling_factor(vals_s: pd.Series, min_scale: Optional[float] = None, max_scale: Optional[float] = None) -> float:
    """Determine scaling factor for tanh transformation based on data distribution.

    Calculates an appropriate scaling factor to use with tanh transformation
    to map data into the effective range of tanh (-2 to 2). Uses both
    standard deviation and range-based approaches.

    Args:
        vals_s: Series or array of values to be scaled.
        min_scale: Optional minimum scaling factor constraint.
        max_scale: Optional maximum scaling factor constraint.

    Returns:
        Scaling factor to use with tanh transformation.

    Notes:
        - Primary method: 1/(2*std) to make tanh sensitive in middle range
        - Alternative method: 4/(max-min) to fit data in tanh effective range
        - Takes minimum of both methods to avoid over-scaling
        - Applies min/max constraints if provided
        - Returns 1.0 if std is 0 or range is 0
    """
    # Calculate statistics of the error data
    err_std = float(np.std(vals_s))
    err_min = float(np.min(vals_s))
    err_max = float(np.max(vals_s))

    # Choose scaling factor based on the data distribution
    # A common approach is to use 1/(2*std) as the scaling factor
    # This makes the tanh function sensitive in the "middle" range of values
    scaling_factor = 1.0 / (2.0 * err_std) if err_std > 0 else 1.0

    # For better control of the range, you can use the min and max values
    # This ensures most values fall within the effective range of tanh (-2 to 2)
    alt_scaling_factor = 4.0 / (err_max - err_min) if (err_max - err_min) > 0 else 1.0
    logger.info(f"Calculate scaling factors based on {len(vals_s)} datapoints, {err_std=}, {err_min=}, {err_max=}, {scaling_factor=}, {alt_scaling_factor}")

    # Choose the smaller of the two to ensure we don't over-scale
    scaling_factor = min(scaling_factor, alt_scaling_factor)

    # Apply optional min/max limits if provided
    if min_scale is not None:
        scaling_factor = max(scaling_factor, min_scale)
    if max_scale is not None:
        scaling_factor = min(scaling_factor, max_scale)

    return scaling_factor


def make_cx_features(df: pd.DataFrame, cx_features: List[str], lookback: int = 30) -> pd.DataFrame:
    """Create cross-sectional (universe-wide) features from individual features.
    
    Cross-sectional features capture market-wide behavior by calculating the mean
    of a feature across all tradeable symbols at each timestamp, then z-scoring
    this mean value over a rolling window.
    
    Args:
        df: DataFrame with MultiIndex (ts, symbol_venue) containing feature data.
            Must have a 'fittable' column to filter tradeable symbols.
        cx_features: List of feature names to convert to cross-sectional features.
            Can be prefixed with 'cx.' which will be stripped for processing.
        lookback: Number of periods for rolling window calculations. Defaults to 30.
        
    Returns:
        DataFrame with added cross-sectional features. New columns are named
        'cx.{feature}' and contain z-scored universe means.
        
    Notes:
        - Only includes symbols where fittable=True in universe mean calculation
        - Z-scores are calculated as (value - rolling_mean) / rolling_std
        - NaN z-scores are filled with 0
        - Intermediate columns (_trmean, _trstd, _z) are not retained
    """
    cx_features = [feature.split('.')[1] if feature.startswith('cx.') else feature for feature in cx_features]
    if len(cx_features) == 0:
        logger.warning("No CX features found in config")
        return df

    logger.info(f"Making CX features {cx_features}")
    df = df.reset_index()
    for feature in cx_features:
        cx_feature = f'cx.{feature}'

        safe_del(df, cx_feature)

        # Calculate mean feature across all securities at each timestamp
        if 'fittable' in df.columns:
            logger.info("Calculating cx feature on fittable securities...")
            fdf = df[df['fittable'].fillna(False)]
        else:
            logger.info("Calculating cx feature on all securities")
            fdf = df
        df_cx = fdf.groupby('ts').agg({feature: 'mean'}).rename(columns={feature: cx_feature}).reset_index()

        # Calculate trailing mean and standard deviation for z-score calculation
        df_cx[f'{cx_feature}_trmean'] = df_cx[cx_feature].rolling(window=lookback).mean()
        df_cx[f'{cx_feature}_trstd'] = df_cx[cx_feature].rolling(window=lookback).std()

        # Calculate z-score
        df_cx[f'{cx_feature}_z'] = (df_cx[cx_feature] - df_cx[f'{cx_feature}_trmean']) / df_cx[f'{cx_feature}_trstd']

        # Fill NaN values with 0 for the z-score
        df_cx[f'{cx_feature}'] = remove_infs(df_cx[f'{cx_feature}_z']).fillna(0)

        # Merge back to original dataframe
        df = pd.merge(df, df_cx[['ts', cx_feature]], left_on='ts', right_on='ts', how='left')

    df = df.set_index(DF_INDEX)
    return df
