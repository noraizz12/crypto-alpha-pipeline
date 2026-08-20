"""Extreme/spikiness feature calculations for statistical arbitrage.

Implements three academic metrics for characterizing return distribution tail behavior:

1. Relative Jump Ratio (BNS 2004): fraction of variance from jumps vs continuous movement
2. Normalized Exceedance Rate: frequency of returns exceeding k*sigma
3. Crow-Siddiqui Robust Kurtosis: robust tail heaviness measure

These features help the alpha model size positions appropriately for assets with
different tail risk profiles, without imposing hard constraints.
"""

import logging
from datetime import datetime as dt
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from lib.calcs.calc_util import get_recent_series_df, get_sparse_rows_for_prod
from lib.util.dataframes import get_min_max_ts, remove_infs, log_col_summary, get_last_timeslice

logger = logging.getLogger(__name__)

MU1 = np.sqrt(2.0 / np.pi)  # ~0.7979, BPV scaling constant


def _compute_rj(x: np.ndarray) -> np.float32:
    """Relative Jump Ratio (Barndorff-Nielsen & Shephard 2004).

    RJ = max(0, (RV - BPV) / RV)
    where RV = sum(r_i^2), BPV = mu_1^{-2} * sum(|r_i| * |r_{i-1}|).

    Returns fraction of realized variance attributable to jumps. Range [0, 1].
    """
    rv = np.sum(x ** 2)
    if rv < 1e-20:
        return np.nan
    abs_x = np.abs(x)
    bpv = (1.0 / (MU1 ** 2)) * np.sum(abs_x[1:] * abs_x[:-1])
    return max(0.0, (rv - bpv) / rv)


def _compute_exceedance(x: np.ndarray, k: float) -> np.float32:
    """Normalized exceedance rate.

    count(|r_i| > k * sigma_window) / N
    Higher = more frequent tail events.
    """
    sigma = np.std(x, ddof=1)
    if sigma < 1e-20:
        return np.nan
    return np.sum(np.abs(x) > k * sigma) / len(x)


def _compute_csk(x: np.ndarray) -> np.float32:
    """Crow-Siddiqui Robust Kurtosis.

    K_CS = (Q_97.5 - Q_2.5) / (Q_75 - Q_25)
    Normal distribution baseline = 2.91.
    """
    q025 = np.quantile(x, 0.025)
    q975 = np.quantile(x, 0.975)
    q25 = np.quantile(x, 0.25)
    q75 = np.quantile(x, 0.75)
    iqr = q75 - q25
    if abs(iqr) < 1e-20:
        return np.nan
    return (q975 - q025) / iqr


def calculate_extreme_metrics(
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
        exceedance_k: float = 3.0
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate extreme/spikiness metrics on rolling windows.

    Computes three tail-risk features for each symbol:
    - {feature}_rj: Relative Jump Ratio [0, 1]
    - {feature}_exceedance: Normalized exceedance rate
    - {feature}_csk: Crow-Siddiqui robust kurtosis

    Follows the same pattern as calculate_trailing_semi_variance in calc_risk.py.

    Args:
        df: Input dataframe with feature data (multi-index: ts, symbol_venue)
        feature: Feature name (e.g. 'logret_720') to compute metrics on
        frequency: Time frequency in minutes
        start_dt: Start datetime for calculation
        lookback_periods: Number of periods for rolling window
        resample_frequency: Frequency for resampling
        ffill_na: Whether to forward fill missing values
        feature_lookback_periods: Default lookback periods if not specified
        lagged_lookback_minutes: Lagged lookback minutes for production
        reopt_times: Reoptimization times for production
        max_lags: Maximum lags for production
        exceedance_k: Sigma threshold for exceedance rate (default 3.0)

    Returns:
        Tuple of (DataFrame with new columns, list of new column names)
    """
    if lookback_periods is None:
        lookback_periods = feature_lookback_periods or 90

    if resample_frequency is None:
        resample_frequency = frequency

    lookback_minutes = lookback_periods * frequency

    if feature not in df.columns:
        logger.error("%s not in dataframe, not calculating extreme metrics", feature)
        return df, []

    new_cols = [f'{feature}_rj', f'{feature}_exceedance', f'{feature}_csk']
    for col in new_cols:
        if col not in df.columns:
            df[col] = np.nan

    logger.info("Calculating extreme metrics on %s with lookback_minutes=%d...",
                feature, lookback_minutes)

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

    logger.info("Calculating extreme metrics %s from %s to %s with rolling window of %d periods...",
                feature, min_ts, max_ts, window)

    resampled = series_df.resample(
        f"{resample_frequency}min", label='right', origin='end_day'
    ).last()

    rolling = resampled.rolling(window=window)

    # Relative Jump Ratio - vectorized approach
    r_squared = resampled ** 2
    rv_df = r_squared.rolling(window=window).sum()
    abs_r = resampled.abs()
    bpv_df = (abs_r * abs_r.shift(1)).rolling(
        window=window, min_periods=window - 1
    ).sum() / (MU1 ** 2)
    rj_df = ((rv_df - bpv_df) / rv_df).clip(lower=0.0)
    rj_df = remove_infs(rj_df).astype(np.float32)

    # Exceedance rate - rolling apply
    exceedance_df = pd.DataFrame(
        rolling.apply(lambda x: _compute_exceedance(x, exceedance_k), raw=True)
    ).astype(np.float32)

    # Crow-Siddiqui robust kurtosis - rolling apply
    csk_df = pd.DataFrame(
        rolling.apply(_compute_csk, raw=True)
    ).astype(np.float32)

    # Rename first-level column (feature name) to target names
    rj_df = rj_df.rename(columns={feature: f'{feature}_rj'})
    exceedance_df = exceedance_df.rename(columns={feature: f'{feature}_exceedance'})
    csk_df = csk_df.rename(columns={feature: f'{feature}_csk'})

    both_df = pd.concat([rj_df, exceedance_df, csk_df], axis=1, verify_integrity=True)

    # Resample to 1-min, ffill within frequency, stack back to multi-index
    new_df = both_df.resample(
        "1min", label='right', closed='right'
    ).last().ffill(limit=frequency).stack(future_stack=True).sort_index()
    new_df = new_df.loc[new_df.index.get_level_values('ts') <= max_ts]

    if reopt_times is not None:
        new_df = get_sparse_rows_for_prod(
            new_df, reopt_times=reopt_times, max_lags=max_lags, frequency=frequency
        )

    if start_dt is not None:
        new_df = new_df[new_df.index.get_level_values('ts') > start_dt]

    logger.info("Merging %s", new_cols)
    new_df = new_df.reindex(
        df[df.index.get_level_values('ts') >= new_df.index.get_level_values('ts').min()].index
    )
    df.loc[new_df.index, new_cols] = new_df

    for col in new_cols:
        log_col_summary(get_last_timeslice(df), col)

    return df, new_cols
