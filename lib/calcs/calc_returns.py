"""Return and price calculation utilities for the statistical arbitrage trading system.

This module provides functions for calculating various return metrics including
log returns, VWAP, residualized returns, and funding-adjusted returns. It's designed
to be used by the Calcs class for feature engineering.

Functions moved here from calcs.py to avoid circular dependencies and improve
code organization.
"""

import logging
from datetime import datetime as dt
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

from lib.util.dataframes import merge_on_index, remove_infs, set_index, safe_del, get_min_max_ts
from lib.util.logging_util import KeyLogger
from .calc_util import make_market_weight_col, get_recent_series_df
from ..util import log_and_raise

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


def calc_logret(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Calculate log returns from close prices.
    
    Computes log(close_t / close_{t-1}) for each symbol. Handles missing data
    by forward-filling prices and validates bar continuity.
    
    Args:
        df: Input dataframe with 'close_mid' prices
    
    Returns:
        Tuple of (DataFrame with added 'logret' column, list of new column names)
    
    Notes:
        - Checks that open_mid equals previous close_mid for bar validation
        - Only calculates returns for bars with update_cnt > 0
    """

    try:
        df = merge_on_index(
            df, df[["close_mid"]].unstack().shift(1).ffill().stack(future_stack=True),
            suffixes=("", "_prev")
        )
    except Exception as e:
        raise log_and_raise(f"Error shifting close_mid for return calculation: {e}", df=df)
        
    if 'open_mid' in df.columns:
        bad_bars_idx = df['open_mid'] != df['close_mid_prev']
        bad_bar_cnt = len(df[bad_bars_idx])
        logger.warning(f"Found {bad_bar_cnt} bad bars (open mid != prev mid) out of {len(df)}")

    df.loc[df["close_mid"].isna(), "close_mid"] = df["close_mid_prev"]
    df['logret'] = np.nan
    df.loc[df['update_cnt'] > 0, "logret"] = np.log(df["close_mid"] / df["close_mid_prev"])
    del df['close_mid_prev']

    return df, ['logret']


def calc_vwap(df: pd.DataFrame, horizon: Optional[int] = None, vwap_fld: str = "vwap",
              dvolume_fld: str = "dvolume", volume_fld: str = "volume") -> Tuple[pd.DataFrame, list]:
    """Calculate volume-weighted average price (VWAP).
    
    Args:
        df: Input dataframe with volume and dollar volume columns
        horizon: Optional time horizon for rolling VWAP calculation
        vwap_fld: Base name for VWAP column
        dvolume_fld: Base name for dollar volume column
        volume_fld: Base name for volume column
    
    Returns:
        Tuple of (DataFrame with added VWAP column, list of new column names)
    
    Notes:
        VWAP = Dollar Volume / Volume
    """
    if horizon is not None:
        vwap_fld = f"{vwap_fld}_{horizon}"
        dvolume_fld = f"{dvolume_fld}_{horizon}"
        volume_fld = f"{volume_fld}_{horizon}"

    df[vwap_fld] = remove_infs(df[dvolume_fld] / df[volume_fld]).astype(np.float32)
    return df, [vwap_fld]


def calc_funding_adjusted_logret(df: pd.DataFrame, keep_realized_funding: bool = False) -> Tuple[pd.DataFrame, list]:
    """Calculate funding rate adjusted log returns.
    
    Adjusts log returns by subtracting realized funding rates at funding times
    (typically every 8 hours). This gives the true P&L from holding futures.
    
    Args:
        df: Input dataframe with 'logret' and funding rate columns
        keep_realized_funding: Whether to keep the realized funding column
    
    Returns:
        Tuple of (DataFrame with added 'logret_funding_adj' column, list of new column names)
    
    Notes:
        - Funding is only realized when ts equals next_funding_time
        - Funding adjustment = log(1 + funding_rate)
    """

    df['realized_funding_log_rate'] = np.log(df['last_funding_rate'] + 1)
    non_funding_ts_idx = df.index.get_level_values("ts").values != df['next_funding_time'].values
    df.loc[non_funding_ts_idx, ['realized_funding_log_rate']] = 0

    df['logret_funding_adj'] = df['logret'] - df['realized_funding_log_rate']

    new_cols = ['logret_funding_adj']
    if keep_realized_funding:
        new_cols.append('realized_funding_log_rate')
    else:
        df = df.drop(columns=['realized_funding_log_rate'])

    return df, new_cols


def calculate_weighted_mkt_return(df: pd.DataFrame, filter_col: Optional[str] = None, ret_col: str = 'logret', frequency: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    """Calculate volume-weighted market return for residualization.
    
    Computes the market return weighted by log of average daily volume (market cap proxy)
    for each timestamp. Only includes fittable instruments in the calculation.
    
    Args:
        df: Input dataframe with returns and volume data
        ret_col: Name of return column to weight (default: 'logret')
        frequency: Optional frequency suffix for column names
    
    Returns:
        Tuple of (updated dataframe with market return column, market column name)
    
    Notes:
        - Uses log(ADVP) as market weight proxy
        - Only fittable instruments contribute to market return
        - Market return is calculated as weighted average across all symbols
    """
    mkt_col = f'{ret_col}_wgtmkt'
    if frequency is not None:
        ret_col += f'_{frequency}'
        mkt_col += f'_{frequency}'

    logger.info(f"Calculating weighted market {ret_col} into {mkt_col}")

    df = make_market_weight_col(df)
    df = df.reset_index()
    df['wgt_logret'] = df[ret_col] * df['mkt_wgt']

    if filter_col is not None:
        logger.info(f"Filtering {filter_col} for weighted market return...")
        new_df = df[df[filter_col]]
    else:
        new_df = df

    new_df = new_df[['ts', 'wgt_logret', 'mkt_wgt']].groupby('ts', observed=False).agg({'wgt_logret': 'sum', 'mkt_wgt': 'sum'}).reset_index()

    safe_del(df, mkt_col)
    new_df[mkt_col] = new_df['wgt_logret'] / new_df['mkt_wgt']
    new_df = new_df.reset_index()
    del df['wgt_logret']
    del df['mkt_wgt']
    df = pd.merge(df, new_df[['ts', mkt_col]], left_on='ts', right_on='ts', how='left')
    df = set_index(df)
    return df, mkt_col


def calculate_resid_return(df: pd.DataFrame, filter_col: str, ret_col: str = 'logret', market_ret_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, list]:
    """Calculate equal-weighted market-residualized returns.
    
    Computes returns minus the equal-weighted average return of all fittable
    instruments, removing market-wide movements.
    
    Args:
        df: Input dataframe with return column
        ret_col: Name of return column to residualize
        market_ret_df: Optional pre-calculated market returns
    
    Returns:
        Tuple of (DataFrame with added residual return column, list of new column names)
    """
    logger.info(f"Calculating residual returns for {ret_col} on filter {filter_col} {'calculated' if market_ret_df is None else 'cached'}")
    df = df.reset_index()
    if market_ret_df is None:
        new_df = df[df[filter_col]][['ts', ret_col]].groupby('ts', observed=False).mean().reset_index()
        df = pd.merge(df, new_df, on='ts', how='left', suffixes=('', '_mkt'))
    else:
        df = pd.merge(df, market_ret_df[['ts', f'{ret_col}_mkt']], on='ts', how='left')
    df[f'{ret_col}_resid_eqmkt'] = df[ret_col] - df[f'{ret_col}_mkt']
    df = set_index(df)
    return df, [f'{ret_col}_resid_eqmkt']


def calculate_resampled_rsi(recent_df: pd.DataFrame, resample_frequency: int, window: int,
                            feature_freq: str, rsi_col: str) -> pd.DataFrame:
    """Helper function to calculate RSI on resampled data.
    
    Resamples the input data to the specified frequency and calculates the
    Relative Strength Index using the standard RSI formula with exponential
    moving averages.
    
    Args:
        recent_df: DataFrame with return data to calculate RSI from
        resample_frequency: Minutes to resample to before RSI calculation
        window: Number of periods for RSI calculation
        feature_freq: Name of the feature column to use
        rsi_col: Name for the output RSI column
        
    Returns:
        DataFrame with RSI values at resampled frequency
        
    Note:
        RSI = 100 - (100 / (1 + RS))
        where RS = average gain / average loss
        When average loss is 0, RS is set to 100 (RSI near 100)
    """
    resampled_df = recent_df.resample(f"{resample_frequency}min", label='right', origin='end_day').last()
    up = resampled_df.clip(lower=0)
    down = -resampled_df.clip(upper=0)
    avg_up = up.rolling(window=window).mean()
    avg_down = down.rolling(window=window).mean()
    # fillna with 100 when avg down is zero, so rsi will be close to 100
    rs = (avg_up / avg_down).fillna(100)
    rsi = (100 - (100 / (1 + rs))).rename(columns={f"{feature_freq}": f'{rsi_col}'})
    return rsi


def calculate_rsi(
        df: pd.DataFrame,
        frequency: int,
        feature: str,
        start_dt: Optional[dt] = None,
        lookback_periods: Optional[int] = None,
        resample_frequency: Optional[int] = None,
        feature_lookback_periods: Optional[int] = None,
        lagged_lookback_minutes: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Relative Strength Index (RSI) technical indicator.
    
    RSI is a momentum oscillator that measures the speed and magnitude of price
    changes. Values range from 0 to 100, with traditional overbought/oversold
    levels at 70/30.
    
    The implementation handles multi-day resampling carefully to avoid lookahead
    bias by calculating RSI values day-by-day when resample_frequency > 1440.
    
    Args:
        df: Input dataframe with price/return data
        frequency: Time frequency in minutes
        feature: Base feature to calculate RSI on (default: 'logret')
        start_dt: Optional start datetime for calculations
        lookback_periods: Number of periods for RSI calculation window
        resample_frequency: Frequency in minutes for resampling before RSI calc
        feature_lookback_periods: Default lookback periods if not specified
        lagged_lookback_minutes: lookback minutes (for prod mode)
        
    Returns:
        Tuple of (DataFrame with RSI column added, list of new column names)
        
    Note:
        - RSI > 70 traditionally indicates overbought conditions
        - RSI < 30 traditionally indicates oversold conditions
        - Uses rolling mean for average gain/loss calculation
        - Forward fills missing values up to frequency limit
    """
    if lookback_periods is None:
        lookback_periods = feature_lookback_periods or 90  # default value
    if resample_frequency is None:
        resample_frequency = frequency

    rsi_col = f"rsi_{frequency}"
    logger.info(f"Calculating rsi on {rsi_col} with {resample_frequency=}, {lookback_periods=}")

    recent_df, window = get_recent_series_df(
        df=df,
        start_dt=start_dt,
        lookback_periods=lookback_periods,
        frequency=frequency,
        feature_freq=feature,
        lagged_lookback_minutes=lagged_lookback_minutes
    )
    min_ts, max_ts = get_min_max_ts(recent_df)
    logger.info(f"Calculating rsi on {feature} from {min_ts} to {max_ts}")

    rsi_current = calculate_resampled_rsi(
        recent_df=recent_df,
        resample_frequency=resample_frequency,
        window=window,
        feature_freq=feature,
        rsi_col=rsi_col
    )

    # if resample_frequency > 1440, it means we need to resample and forward fill across days, so introduce this method to calculate day by day
    # rsi_prev is to calculate feature value from previous_date:00:00:00 to previous_date:23:59:00
    # rsi_current is to calculate feature value for current_date:00:00:00
    # then we concat these two together and ffill to make sure there is no ffill more than 1 day
    if resample_frequency > 1440:
        rsi_prev = calculate_resampled_rsi(
            recent_df=recent_df.loc[recent_df.index <= start_dt],
            resample_frequency=resample_frequency,
            window=window,
            feature_freq=feature,
            rsi_col=rsi_col
        )
        rsi = pd.concat([rsi_prev, rsi_current]).sort_index()
    else:
        rsi = rsi_current

    new_df = rsi.resample("1min", label='right', closed='right').last().ffill(limit=frequency).stack(future_stack=True).sort_index()
    new_df = new_df.loc[new_df.index.get_level_values('ts') <= max_ts]
    if start_dt is not None:
        new_df = new_df[new_df.index.get_level_values('ts') > start_dt]
    new_df = new_df.reindex(df[df.index.get_level_values('ts') >= new_df.index.get_level_values('ts').min()].index)
    df.loc[new_df.index, rsi_col] = new_df[rsi_col].astype(np.float32)

    return df, [rsi_col]


def calc_factor_return(df: pd.DataFrame, factor: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate factor portfolio returns for analysis.
    
    Constructs long-short portfolios based on factor quintiles to measure
    factor performance and validate alpha signals.
    
    Args:
        df: Input dataframe with factor and return data
        factor: Factor column name to analyze
    
    Returns:
        Tuple of (long returns, short returns, long cumulative, short cumulative)
    
    Notes:
        - Long portfolio: top quintile (Q5)
        - Short portfolio: bottom quintile (Q1)
        - Equal weighted within quintiles
    """

    if 'ret' not in df.columns:
        # calculate individual return and shift 1 period forward
        df['ret'] = np.log(df['close_mid'] / df['close_mid_previous'])

    ret_df = df['ret'].unstack()
    ret_df = ret_df.shift(1)

    # build factor loading matrix
    if factor == 'dollar_exposure':
        factor_df = pd.DataFrame(1, index=ret_df.index, columns=ret_df.columns)
    else:
        factor_df = df[factor].unstack()

    # calculate factor return as regress individual return on factor loading
    numerator = (factor_df * ret_df).sum(axis=1)
    denominator = (factor_df ** 2).sum(axis=1)
    factor_ret_df = numerator / denominator

    port_pos_df = df['position'].unstack()
    port_weight_df = port_pos_df.div(port_pos_df.abs().sum(axis=1), axis=0)

    # calculate portfolio factor return using weight and portfolio factor pnl using position
    port_factor_ret_df = (port_weight_df * factor_df).sum(axis=1) * factor_ret_df
    port_factor_pnl_df = (port_pos_df * factor_df).sum(axis=1) * factor_ret_df
    port_factor_exposure_df = (port_pos_df * factor_df).sum(axis=1)

    return (
        factor_ret_df.cumsum().rename(factor).reset_index(),
        port_factor_exposure_df.rename(factor).reset_index(),
        port_factor_ret_df.cumsum().rename(factor).reset_index(),
        port_factor_pnl_df.cumsum().rename(factor).reset_index()
    )


def calculate_weighted_resid_return(df: pd.DataFrame, filter_col: str, ret_col: str = 'logret', market_ret_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate volume-weighted market-residualized returns.
    
    Computes returns minus the volume-weighted market return, where weights
    are based on log(ADVP) as a market cap proxy.
    
    Args:
        df: Input dataframe with return column
        ret_col: Name of return column to residualize  
        market_ret_df: Optional pre-calculated weighted market returns
    
    Returns:
        Tuple of (DataFrame with added weighted residual return column, list of new column names)
    """
    logger.info(f"Calculating weighted residual {ret_col}")
    new_cols = []

    if market_ret_df is None:
        df = make_market_weight_col(df)
        df, mkt_col = calculate_weighted_mkt_return(df, filter_col=filter_col, ret_col=ret_col)
        new_cols.append(mkt_col)
    else:
        df = df.reset_index()
        df = pd.merge(df, market_ret_df[['ts', f'{ret_col}_wgtmkt']], on='ts', how='left')
        df = set_index(df)

    resid_col = f'{ret_col}_resid_wgtmkt'
    df[resid_col] = df[ret_col] - df[f'{ret_col}_wgtmkt']
    new_cols.append(resid_col)

    return df, new_cols
