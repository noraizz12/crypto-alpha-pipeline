"""Volume calculation utilities for the statistical arbitrage trading system.

This module provides volume-related calculation functions used by the main
calculation engine.
"""

import logging
from datetime import datetime as dt, timedelta as td
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

from lib.util.dataframes import remove_infs, merge_on_index, unstack_feature, make_date, DF_INDEX, get_df_idx
from lib.util.util import log_and_raise
from lib.util.time_util import to_datetime

logger = logging.getLogger(__name__)


def calculate_advp(df: pd.DataFrame, lookback_days: int, allow_short_history: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate average daily volume in USD (ADVP).
    
    Computes rolling average of daily dollar volume, accounting for missing
    data by dividing total volume by actual trading days.
    
    Args:
        df: Input dataframe with 'dvolume' column
        lookback_days: Number of days for rolling average
    
    Returns:
        Tuple of (DataFrame with added 'advp' column, list of new column names)
    
    Notes:
        - Handles missing data by counting actual trading days
        - Critical metric for liquidity filtering
    """
    lookback_mins = lookback_days * 1440
    advp_col = 'advp'
    logger.info(f"Calculating {advp_col} lookback: {lookback_days} days...")
    unstacked_df = df[['dvolume']].unstack()
    unstacked_filled_df = unstacked_df.fillna(0)
    unstacked_len = len(unstacked_filled_df)
    if unstacked_len < lookback_mins and not allow_short_history:
        raise log_and_raise(f"dataframe has {unstacked_len} periods which is less than the necessary {lookback_mins} for {advp_col}")

    # calculate advp by rolling sum of volume divided by rolling sum of non-na days
    horizon_dvolume_df = unstacked_filled_df.rolling(window=lookback_mins).sum()
    horizon_days_df = unstacked_df.notna().rolling(window=lookback_mins).agg('sum').divide(1440).fillna(0).astype(int)
    horizon_df = remove_infs(horizon_dvolume_df / horizon_days_df).stack(future_stack=True).rename(columns={'dvolume': advp_col})
    if advp_col in df.columns:
        logger.warning("Overwriting advp!")
        del df[advp_col]
    df = merge_on_index(df, horizon_df)
    return df, [advp_col]


def calculate_mdvp(df: pd.DataFrame, lookback_days: int,
                   allow_short_history: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate median daily volume in USD (MDVP).

    Computes rolling median of daily dollar volume using dvolume_1440.
    More robust to volume spikes than the mean-based ADVP, making it
    a better liquidity filter for symbols with occasional extreme volume days.

    Args:
        df: Input dataframe with dvolume_1440 column and ['ts', 'symbol_venue'] index
        lookback_days: Number of days for rolling median window
        allow_short_history: If True, skip length check for short dataframes

    Returns:
        Tuple of (DataFrame with added 'mdvp' column, list of new column names)
    """
    mdvp_col = 'mdvp'
    dvolume_col = 'dvolume_1440'
    logger.info("Calculating %s lookback: %d days from column %s...",
                mdvp_col, lookback_days, dvolume_col)

    unstacked_df = df[[dvolume_col]].unstack()
    unstacked_len = len(unstacked_df)

    # dvolume_1440 is already daily-aggregated; resample to one row per day
    # using .last() then compute rolling median, then upsample back
    min_required = lookback_days * 1440
    if unstacked_len < min_required and not allow_short_history:
        raise log_and_raise(
            f"dataframe has {unstacked_len} periods which is less than "
            f"the necessary {min_required} for {mdvp_col}"
        )

    daily_df = unstacked_df.fillna(0).resample('1D').last()
    median_df = daily_df.rolling(window=lookback_days, min_periods=1).median()
    median_df = median_df.resample('1min').ffill()
    median_df = median_df.reindex(unstacked_df.index, method='ffill')

    horizon_df = median_df.stack(future_stack=True).rename(
        columns={dvolume_col: mdvp_col}
    )

    if mdvp_col in df.columns:
        logger.warning("Overwriting mdvp!")
        del df[mdvp_col]
    df = merge_on_index(df, horizon_df)
    return df, [mdvp_col]


def calculate_volume_forecast(df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, List[str]]:
    """Forecast expected volume over a horizon.
    
    Estimates future volume based on time-of-day patterns and recent activity.
    
    Args:
        df: Input dataframe with volume bucket data
        horizon: Forecast horizon in minutes
    
    Returns:
        Tuple of (DataFrame with volume forecast columns, list of new column names)
    """
    # well -- it's not the best forecasts...
    df[f'dvolume_{horizon}_forecast'] = df[f'dvolume_{horizon}']
    # df[f'dvolume_{volume_bucket_mins}_forecast'] = df['dvolume_1440_mult'] * df[f'median_time_bucket_dvolume_{volume_bucket_mins}']
    # features_df['dvolume_1440_mult'] = (features_df['dvolume_1440'] / features_df['dvolume_1440_trmean']).astype(np.float32)
    df['dvolume_1_forecast'] = df[f'dvolume_{horizon}_forecast'] / horizon
    
    new_cols = [f'dvolume_{horizon}_forecast', 'dvolume_1_forecast']
    return df, new_cols


def calculate_volume_buckets(df: pd.DataFrame, frequency: int, start_dt: Optional[dt] = None,
                           lookback_days: int = 45) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate median volume by time of day buckets.
    
    Computes rolling median dollar volume for each time period (e.g., 9:30am)
    to capture intraday volume patterns. Used for volume normalization.
    
    Args:
        df: Input dataframe with dollar volume data
        frequency: Time frequency in minutes
        start_dt: Start datetime for calculation
        lookback_days: Days of history for median calculation (default: 45)
    
    Returns:
        Tuple of (DataFrame with added median time bucket volume column, list of new column names)
    
    Notes:
        - Creates time-of-day buckets at frequency granularity
        - Forward fills missing values
        - Handles midnight rollover correctly
    """
    if start_dt is not None:
        start_with_lookback = start_dt - td(days=lookback_days + 1)
        recent_df = df[get_df_idx(df, start_with_lookback)]
    else:
        recent_df = df

    min_date = recent_df['date'].min()
    max_date = recent_df['date'].max()
    hist_days = (max_date - min_date).days
    if hist_days < lookback_days:
        raise log_and_raise(f"Data goes from {min_date} to {max_date} with start {start_dt}, need at least {lookback_days} days, {hist_days}")

    logger.info(f"Calculating volume buckets for time slices {frequency} with adv lookback {lookback_days}")
    resampled_df = unstack_feature(features_df=recent_df, feature=f'dvolume_{frequency}')
    resampled_df = resampled_df.stack(future_stack=True).reset_index()

    resampled_df = make_date(resampled_df)
    resampled_df['time'] = resampled_df['ts'].dt.time
    mdvolume_name = f"median_time_bucket_dvolume_{frequency}"
    vols_dfs = []
    for symbol in sorted(resampled_df['symbol_venue'].unique()):
        df_symbol = resampled_df[resampled_df['symbol_venue'] == symbol]
        df_symbol = df_symbol[[f'dvolume_{frequency}', 'date', 'time']].set_index(['date', 'time'])
        df_symbol = df_symbol.unstack().fillna(0)
        df_symbol = df_symbol.rolling(lookback_days).median()
        df_symbol = df_symbol.shift(1).stack(future_stack=True)
        df_symbol['symbol_venue'] = symbol
        vols_dfs.append(df_symbol)

    ## XXX TODO: this has duplicate indices.  FIXX!!!
    vols_df = pd.concat(vols_dfs)
    if len(vols_df) == 0:
        raise log_and_raise("Unable to calculate volume buckets.  Likely not enough lookback...")

    vols_df = vols_df.rename(columns={f'dvolume_{frequency}': mdvolume_name})
    vols_df[mdvolume_name] = vols_df[mdvolume_name].astype(np.float32)
    vols_df = vols_df.reset_index()
    vols_df['ts'] = to_datetime(vols_df['date'].astype(str) + ' ' + vols_df['time'].astype(str))
    midnight_idx = vols_df['time'].astype(str) == '00:00:00'
    vols_df.loc[midnight_idx, 'ts'] = to_datetime((vols_df['date'] + td(days=1)).astype(str) + ' ' + vols_df['time'].astype(str))
    vols_df = vols_df[['symbol_venue', 'ts', mdvolume_name]] \
        .set_index(DF_INDEX) \
        .unstack() \
        .resample("1min", label='right', closed='right') \
        .last() \
        .ffill() \
        .stack(future_stack=True)

    new_cols = [mdvolume_name]

    df = merge_on_index(df, vols_df)
    return df, new_cols


def calculate_mktcap_turnover(
        df: pd.DataFrame,
        frequency: int,
        feature_sigma_bound: float = 5) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate market cap turnover ratio.

    Computes the ratio of trailing mean dollar volume to market cap,
    representing how quickly the market cap turns over through trading.

    Args:
        df: Input dataframe with dvolume_{frequency}_trmean and market_cap columns
        frequency: Time frequency in minutes
        feature_sigma_bound: Bound for clipping extreme values

    Returns:
        Tuple of (DataFrame with mktcap_turnover column, list of new column names)

    Notes:
        - Higher values indicate more liquid/actively traded assets
        - Useful for identifying trading activity relative to size
    """
    dvolume_col = f'dvolume_{frequency}_trmean'
    new_col = f'mktcap_turnover_{frequency}'
    logger.info(f"Calculating {new_col} ...")

    if dvolume_col not in df.columns:
        raise ValueError(f"Missing required column: {dvolume_col}")
    if 'marketcap' not in df.columns:
        raise ValueError("Missing required column: marketcap")

    df[new_col] = remove_infs(df[dvolume_col] / df['marketcap']).fillna(0).astype(np.float32)

    new_cols = [new_col]
    return df, new_cols
