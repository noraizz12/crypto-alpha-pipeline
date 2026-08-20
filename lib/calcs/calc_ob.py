"""Order book calculation utilities for the statistical arbitrage trading system.

This module provides order book and microstructure-related calculation functions.
"""

import logging
from datetime import datetime as dt
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from .calc_util import calculate_delta, calculate_trailing_mean
from lib.util.dataframes import remove_infs

logger = logging.getLogger(__name__)


def calculate_spread_factor(
        df: pd.DataFrame,
        frequency: int,
        lookback_periods: int,
        start_dt: Optional[dt] = None,
        ffill_na: bool = True,
        feature_lookback_periods: Optional[int] = None,
        lagged_lookback_minutes: Optional[int] = None,
        reopt_times: Optional[List[dt]] = None,
        max_lags: Optional[int] = None,
        feature_sigma_bound: float = 5) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate relative bid-ask spread.
    
    Normalizes spread by mid price to get a comparable liquidity metric
    across different price levels.
    
    Args:
        df: Input dataframe with spread_avg column
        frequency: Time frequency in minutes
        lookback_periods: Number of periods for trailing mean calculation
        start_dt: Optional start datetime
        ffill_na: Whether to forward fill missing values
        feature_lookback_periods: Default lookback periods if not specified
        lagged_lookback_minutes: Lagged lookback for production mode
        reopt_times: Reoptimization times for production mode
        max_lags: Maximum lags for production mode
        feature_sigma_bound: Bound for z-score clipping
    
    Returns:
        Tuple of (DataFrame with relative_spread column and trailing stats, list of new column names)
    """
    spread_col = f'relative_spread_{frequency}'
    new_cols = [spread_col]
    logger.info(f"Calculating spread factor with lookback periods {lookback_periods}")
    df[spread_col] = df[f'spread_avg_{frequency}'] / df['close_mid']

    # Calculate trailing mean and get the new columns it creates
    df, tm_cols = calculate_trailing_mean(
        df=df,
        feature=spread_col,
        frequency=frequency,
        lookback_periods=lookback_periods,
        start_dt=start_dt,
        ffill_na=ffill_na,
        feature_lookback_periods=feature_lookback_periods,
        lagged_lookback_minutes=lagged_lookback_minutes,
        reopt_times=reopt_times,
        max_lags=max_lags,
        feature_sigma_bound=feature_sigma_bound
    )

    # Add the trailing mean columns to our list
    new_cols.extend(tm_cols)

    return df, new_cols


def calculate_bid_ask_imbalance(df: pd.DataFrame, frequency: int, start_dt: Optional[dt] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate bid-ask size imbalance and its changes.
    
    Computes the log ratio of bid size to ask size as a measure of order book
    imbalance. Also calculates the delta (change) in this imbalance.
    
    Creates columns:
    - ba_imbal_{frequency}: Log(bid_size / ask_size)
    - ba_imbal_{frequency}_d: Absolute change in imbalance
    - ba_imbal_{frequency}_dp: Percentage change in imbalance
    
    Args:
        df: Input dataframe with bid_sz_avg and ask_sz_avg columns
        frequency: Time frequency in minutes
        start_dt: Optional start datetime for calculations
        
    Returns:
        Tuple of (DataFrame with bid-ask imbalance columns added, list of new column names)
        
    Note:
        - Positive values indicate more bid pressure (buyers)
        - Negative values indicate more ask pressure (sellers)
        - Values are fillna(0) to handle cases where sizes are 0
    """
    imbal_col_base = 'ba_imbal'
    imbal_col = f"{imbal_col_base}_{frequency}"
    df[imbal_col] = np.log(df[f'bid_sz_avg_{frequency}'] / df[f'ask_sz_avg_{frequency}']).fillna(0)

    new_cols = [imbal_col]

    # Calculate delta and track the new columns it creates
    df, delta_cols = calculate_delta(df, imbal_col, frequency, start_dt)
    new_cols.extend(delta_cols)

    return df, new_cols


def calculate_trade_size(
        df: pd.DataFrame,
        frequency: int,
        start_dt: Optional[dt] = None,
        lookback_periods: Optional[int] = None,
        feature_lookback_periods: Optional[int] = None,
        lagged_lookback_minutes: Optional[int] = None,
        reopt_times: Optional[List[dt]] = None,
        max_lags: Optional[int] = None,
        feature_sigma_bound: float = 5) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate average trade size.
    
    Computes volume per trade as an indicator of institutional vs retail
    trading activity.
    
    Args:
        df: Input dataframe with volume and trade_cnt columns
        frequency: Time frequency in minutes
        start_dt: Start datetime for calculation
        lookback_periods: Number of periods for trailing mean calculation
        feature_lookback_periods: Default lookback periods if not specified
        lagged_lookback_minutes: Lagged lookback for production mode
        reopt_times: Reoptimization times for production mode
        max_lags: Maximum lags for production mode
        feature_sigma_bound: Bound for z-score clipping
    
    Returns:
        Tuple of (DataFrame with trade_sz columns, list of new column names)
    """
    if lookback_periods is None:
        lookback_periods = feature_lookback_periods or 90  # default

    new_col = f'trade_sz_{frequency}'
    logger.info(f"Calculating {new_col} ...")

    df[new_col] = remove_infs(df[f'volume_{frequency}'] / df[f'trade_cnt_{frequency}']).fillna(0).astype(np.float32)

    df, tm_cols = calculate_trailing_mean(
        df=df,
        feature=new_col,
        frequency=frequency,
        lookback_periods=lookback_periods,
        start_dt=start_dt,
        ffill_na=True,
        feature_lookback_periods=feature_lookback_periods,
        lagged_lookback_minutes=lagged_lookback_minutes,
        reopt_times=reopt_times,
        max_lags=max_lags,
        feature_sigma_bound=feature_sigma_bound
    )

    new_cols = [new_col] + tm_cols
    return df, new_cols


def calculate_update_size(df: pd.DataFrame, frequency: int, start_dt: Optional[dt] = None,
                          lookback_periods: Optional[int] = None, feature_lookback_periods: Optional[int] = None,
                          lagged_lookback_minutes: Optional[int] = None, reopt_times: Optional[List[dt]] = None,
                          max_lags: Optional[int] = None, feature_sigma_bound: float = 5) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate relative order book update frequency.
    
    Normalizes update count by volume to measure market activity intensity
    independent of trading volume.
    
    Args:
        df: Input dataframe with update_cnt and volume columns
        frequency: Time frequency in minutes
        start_dt: Start datetime for calculation
        lookback_periods: Number of periods for trailing mean calculation
        feature_lookback_periods: Default lookback periods if not specified
        lagged_lookback_minutes: Lagged lookback for production mode
        reopt_times: Reoptimization times for production mode
        max_lags: Maximum lags for production mode
        feature_sigma_bound: Bound for z-score clipping
    
    Returns:
        Tuple of (DataFrame with relative_updates columns, list of new column names)
    """
    if lookback_periods is None:
        lookback_periods = feature_lookback_periods or 90  # default

    new_col = f'relative_updates_{frequency}'
    logger.info(f"Calculating {new_col} ...")

    df[new_col] = remove_infs(df[f'update_cnt_{frequency}'] / df[f'volume_{frequency}']).fillna(0).astype(np.float32)

    df, tm_cols = calculate_trailing_mean(
        df=df,
        feature=new_col,
        frequency=frequency,
        lookback_periods=lookback_periods,
        start_dt=start_dt,
        ffill_na=True,
        feature_lookback_periods=feature_lookback_periods,
        lagged_lookback_minutes=lagged_lookback_minutes,
        reopt_times=reopt_times,
        max_lags=max_lags,
        feature_sigma_bound=feature_sigma_bound
    )

    new_cols = [new_col] + tm_cols
    return df, new_cols
