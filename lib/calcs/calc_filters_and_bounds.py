"""Filter and bounds calculation utilities for the statistical arbitrage trading system.

This module provides liquidity and tradability filter calculations, as well as
position bounds and exclusion logic for portfolio optimization.
"""

import logging
from datetime import datetime as dt
from datetime import timedelta as td
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from lib.util.util import fpct, fmoney, log_and_raise
from lib.util.dataframes import make_date, make_daily_carry_forward_values, set_index, make_symbol

logger = logging.getLogger(__name__)

DEFAULT_ERROR_THRESHHOLD = 0.10


def _run_filter(df: pd.DataFrame, col: str, idx: pd.Series, start_dt: Optional[dt] = None,
                error_threshold: Optional[float] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Apply boolean filter and validate coverage.
    
    Helper function to apply filters while ensuring sufficient data coverage
    to avoid issues from sparse data.
    
    Args:
        df: Input dataframe
        col: Column name for the filter
        idx: Boolean index to apply
        start_dt: Start datetime for validation
        error_threshold: Maximum allowed missing data fraction
    
    Returns:
        Tuple of (DataFrame with filter applied, list containing the new column name)
    
    Raises:
        Exception if missing data exceeds threshold
    """

    if error_threshold is None:
        error_threshold = DEFAULT_ERROR_THRESHHOLD

    df[col] = False
    ts_idx = None
    if start_dt is not None:
        max_ts = df.index.get_level_values('ts').max()
        if max_ts <= start_dt:
            log_and_raise(f"Data only goes to {max_ts} for start of {start_dt}")
        logger.info(f"Getting data past {start_dt} where max value is {max_ts}")
        ts_idx = df.index.get_level_values('ts') > start_dt
        idx &= ts_idx
    df.loc[idx, col] = True

    if ts_idx is not None:
        total_count = len(df[ts_idx].index.get_level_values('symbol_venue').unique())
        pass_count = len(df[df[col] & ts_idx].index.get_level_values('symbol_venue').unique())
    else:
        total_count = len(df.index.get_level_values('symbol_venue').unique())
        pass_count = len(df[df[col]].index.get_level_values('symbol_venue').unique())

    if total_count == 0:
        raise log_and_raise("No data passed to filter")

    pass_pct = pass_count / total_count
    logger.info(f"{col} filter pass_pct: {fpct(pass_pct)}, pass_cnt: {pass_count}")
    if pass_pct < error_threshold:
        raise log_and_raise(f"Less than {error_threshold} of securities are {col}! {pass_count=}, {total_count=}, {pass_pct=},  {start_dt}", df=df)

    return df, [col]


def calculate_fittable_filter(
        df: pd.DataFrame,
        min_advp: float,
        advp_col: str = 'advp',
        start_dt: Optional[dt] = None,
        error_threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate and apply fittable filter based on liquidity and data quality.

    Note that this is essentially "featureable" and used for determining feature universe
    
    Determines which instruments have sufficient liquidity and data quality
    for model fitting. Checks ADVP, price validity, and update count.
    
    Args:
        df: Input dataframe with market data
        min_advp: Minimum average daily volume in USD
        advp_col: Column name for average daily volume
        frequency: Optional frequency for update count field
        start_dt: Optional start datetime for filter validation
        
    Returns:
        Tuple of (DataFrame with 'fittable' column added, list containing 'fittable')
        
    Notes:
        - Requires ADVP > min_advp_fittable (typically $15M)
        - Ensures positive prices and non-zero update counts
        - Uses first values per symbol to avoid look-ahead bias
    """
    avg = df[advp_col].mean()
    logger.info(f"Calculating fittable filter on {advp_col} > {min_advp}, {avg=}")

    drop_cols = [f'first_{advp_col}']
    if 'date' not in df.columns:
        drop_cols += ['date']
        df = make_date(df)
    df = make_daily_carry_forward_values(df, cols=[advp_col])

    idx = (df[f'first_{advp_col}'] > min_advp)
    df = df.drop(columns=drop_cols, errors='ignore')
    return _run_filter(df, 'fittable', idx, start_dt, error_threshold=error_threshold)


def calculate_tradeable_filter(df: pd.DataFrame, min_advp_tradeable: float, start_dt: Optional[dt] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate and apply tradeable filter based on liquidity.
    
    Uses trailing mean dollar volume to identify instruments with sufficient
    liquidity for active trading.
    
    Args:
        df: Input dataframe with dvolume_1440_trmean column
        min_advp_tradeable: Minimum average daily volume in USD
        start_dt: Optional start datetime for filter validation
        
    Returns:
        Tuple of (DataFrame with 'tradeable' column added, list containing 'tradeable')
        
    Notes:
        - Filter is based on trailing mean daily dollar volume
        - Calls run_filter to apply the filter and validate coverage
    """

    #simple volume here, and the stricter volume checks on expandable
    idx = df['dvolume_1440_trmean'] > min_advp_tradeable
    if 'dead' in df.columns:
        idx &= ~df['dead']
    else:
        logger.info("Not considering dead in determining tradeable")

    if 'position_value' in df.columns:
        idx |= df['position_value'].fillna(0) != 0
    else:
        logger.info("Not considering position_value for tradeable")

    return _run_filter(df, 'tradeable', idx, start_dt)


def calculate_price_filter(df: pd.DataFrame, min_dvolume: float, frequency: int, start_dt: Optional[dt] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate and apply tradeable filter based on liquidity.

    Uses trailing mean dollar volume to identify instruments with sufficient
    liquidity for active trading.

    ASSUMES 1 MIN BARS

    Args:
        df: Input dataframe with dvolume_1440_trmean column
        min_dvolume: Minimum average daily volume in USD
        start_dt: Optional start datetime for filter validation

    Returns:
        Tuple of (DataFrame with 'tradeable' column added, list containing 'tradeable')

    Notes:
        - Filter is based on trailing mean daily dollar volume
        - Calls run_filter to apply the filter and validate coverage
    """

    update_cnt_fld = 'update_cnt'
    if frequency is not None:
        update_cnt_fld += f"_{frequency}"

    logger.info(f"Calculating priceable filter {frequency} with {min_dvolume}")

    tmp_cols = ['dvolume_sum', 'first_close_mid', f'{update_cnt_fld}_sum']
    if 'date' not in df.columns:
        tmp_cols += ['date']
        df = make_date(df)
    df = make_daily_carry_forward_values(df, cols=['close_mid'])

    df = make_symbol(df)
    agged_df = df.groupby(['date', 'symbol']).agg({'dvolume': 'sum', update_cnt_fld: 'sum'}).reset_index()
    df = set_index(pd.merge(df.reset_index(), agged_df, on=['symbol', 'date'], how='left', suffixes=('', '_sum')))

    idx = (df['dvolume_sum'] > min_dvolume) & (df['first_close_mid'] > 0) & (df[f'{update_cnt_fld}_sum'] > 0)
    if 'dead' in df.columns:
        idx &= ~df['dead']

    df = df.drop(columns=tmp_cols, errors='ignore')
    return _run_filter(df, 'priceable', idx, start_dt)


def calculate_expandable_filter(df: pd.DataFrame, min_advp_expandable: float,
                                max_market_cap_expandable_frac: float = 0.05,
                                filter_delisting: bool = True, delisting_buffer_days: int = 6,
                                start_dt: Optional[dt] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate and apply expandable filter based on high liquidity threshold.

    Identifies instruments with sufficient liquidity for position expansion
    without significant market impact. Also filters out instruments approaching
    delisting.

    Args:
        df: Input dataframe with dvolume_1440_trmean and delisting_date columns
        min_advp_expandable: Minimum average daily volume for expansion (typically $55M)
        max_market_cap_expandable_frac: Max fraction of market cap used for expandable check
        filter_delisting: Whether to filter out delisting instruments
        delisting_buffer_days: Days before delisting to start filtering
        start_dt: Optional start datetime for filter validation

    Returns:
        Tuple of (DataFrame with 'expandable' column added, list containing 'expandable')

    Notes:
        - Requires higher liquidity threshold than tradeable filter
        - Filters out instruments within delisting_buffer_days of delisting
        - Uses 10% error threshold for stricter validation
    """
    orig_cnt = len(df.index.get_level_values('symbol_venue').unique())
    uni_idx = (df['dvolume_1440_trmean'] > min_advp_expandable) & (df['mdvp'] > min_advp_expandable) & (df['marketcap'].isna() | (df['marketcap'] * max_market_cap_expandable_frac > min_advp_expandable))
    volume_filter_cnt = len(df[uni_idx].index.get_level_values('symbol_venue').unique())
    logger.info(f"Removing expandable {orig_cnt} -> {volume_filter_cnt} for low dvolume_1440_trmean/mdvp/marketcap below {min_advp_expandable}")
    idx = uni_idx
    if 'dead' in df.columns:
        idx &= ~df['dead']
    else:
        logger.info(f"Not considering dead for expandable...")

    if filter_delisting and 'delisting_date' in df.columns:
        logger.info(f"Making upcoming delistings unexpandable...")
        # Adjust delisting_date by buffer days earlier
        # If a token has delisting_buffer_days to its delisting, label expandable as False
        df["delisting_date_adjusted"] = df['delisting_date'] - td(days=delisting_buffer_days)
        non_delisting_idx = ~(df.index.get_level_values('ts') > df['delisting_date_adjusted'])
        del df['delisting_date_adjusted']

        delisting_symbols = set(df.loc[~non_delisting_idx].index.get_level_values('symbol_venue'))
        if len(delisting_symbols) > 0:
            logger.info(f"Removing {delisting_symbols} for delisting")

        idx &= non_delisting_idx
    else:
        logger.info(f"Not considering delisting_date for expandable...")

    total_filter_cnt = len(df[idx].index.get_level_values('symbol_venue').unique())
    logger.info(f"Removing expandable {orig_cnt - total_filter_cnt} for low dvolume_1440_trmean, marketcap and delisting")

    return _run_filter(df, 'expandable', idx, start_dt, error_threshold=0.10)


def adjust_bounds_with_max_notional_value(df: pd.DataFrame, tradeable_uni_idx: pd.Series) -> pd.DataFrame:
    """Adjust position bounds based on portfolio constraints.
    
    Scales position limits to ensure total portfolio stays within max notional
    value while respecting individual position limits.
    
    Args:
        df: Input dataframe with close prices and existing bounds
        tradeable_uni_idx: Boolean mask of tradeable instruments
    
    Returns:
        DataFrame with adjusted bound columns
        
    Notes:
        - Only applies adjustments if 'max_position_value' column exists
        - Ensures bounds don't exceed max position value limits
        - Logs any bounds that were limited by max position value
    """
    if 'max_position_value' not in df.columns:
        logger.info("No bounds adjusted by max_position_value since col not in the df, which only seen in sim")
        return df

    df['neg_max_position_value'] = -df['max_position_value']

    bounds_limited_idx = tradeable_uni_idx & ((df['ubound'] > df['max_position_value']) | (df['lbound'] < df['neg_max_position_value']))
    if bounds_limited_idx.any():
        bounds_limited_df = df.loc[bounds_limited_idx, ['ubound', 'lbound', 'max_position_value', 'neg_max_position_value', 'position']]
        for idx, row in bounds_limited_df.iterrows():
            logger.warning(f"Bounds adjusted from binance: {idx} ubound={row['ubound']}, lbound={row['lbound']}, max_position_value={row['max_position_value']}, position={row['position']}")
    else:
        logger.info("No bounds adjusted by max_position_value")

    #max_position_value from binance!!
    df.loc[tradeable_uni_idx, 'ubound'] = df[['ubound', 'max_position_value']].min(axis=1)
    df.loc[tradeable_uni_idx, 'lbound'] = df[['lbound', 'neg_max_position_value']].max(axis=1)
    df = df.drop('neg_max_position_value', axis=1)
    return df


def _apply_vol_bound_scaling(df: pd.DataFrame, vol_bound_coeff: float,
                             vol_bound_floor: float) -> pd.DataFrame:
    """Apply volatility-based scaling to max_adv_bound.

    Reduces position bounds for names with abnormally high realized
    volatility relative to the cross-section, using the pre-computed
    logret_1440_trstd_cz feature.

    Args:
        df: DataFrame with max_adv_bound and logret_1440_trstd_cz columns
        vol_bound_coeff: Penalty coefficient (higher = more aggressive)
        vol_bound_floor: Minimum scaling factor (e.g. 0.4 = max 60% reduction)

    Returns:
        DataFrame with max_adv_bound adjusted in-place
    """
    trstd_cz_col = 'logret_1440_trstd_cz'
    if trstd_cz_col not in df.columns:
        logger.warning(f"{trstd_cz_col} not in df, skipping vol bound scaling")
        return df

    trstd_cz = df[trstd_cz_col].fillna(0)
    vol_scale = np.where(
        trstd_cz > 1.0,
        (1.0 / (1.0 + vol_bound_coeff * (trstd_cz - 1.0))).clip(vol_bound_floor, 1.0),
        1.0
    )
    n_penalized = (vol_scale < 1.0).sum()
    pre_mean = df['max_adv_bound'].mean()
    df['max_adv_bound'] = (df['max_adv_bound'] * vol_scale).astype(np.float32)
    post_mean = df['max_adv_bound'].mean()

    if n_penalized > 0:
        avg_scale = vol_scale[vol_scale < 1.0].mean()
        penalized_symbols = df.index.get_level_values('symbol_venue')[vol_scale < 1.0].unique()
        logger.info(f"Vol bound scaling: {n_penalized}/{len(df)} rows penalized, "
                    f"avg scale={avg_scale:.3f} (coeff={vol_bound_coeff}, floor={vol_bound_floor}), "
                    f"mean max_adv_bound ${pre_mean:,.0f} -> ${post_mean:,.0f}, "
                    f"penalized symbols: {list(penalized_symbols[:10])}"
                    f"{'...' if len(penalized_symbols) > 10 else ''}")
    else:
        logger.info(f"Vol bound scaling: no rows penalized "
                    f"(coeff={vol_bound_coeff}, floor={vol_bound_floor}, "
                    f"max trstd_cz={trstd_cz.max():.2f})")

    return df


def calculate_exclusions_and_bounds(
        df: pd.DataFrame,
        max_position_pct: float,
        max_portfolio_notional: float,
        max_volume_fraction: float,
        exclude_nonalpha_trades: bool,
        max_move_filter: float,
        delisting_buffer_days: int,
        position_bounds: List[Tuple[float, float]],
        max_dvol_sigma: float = 2.0,
        min_funding_rate: float = -0.005,
        bound_col: Optional[str] = 'mdvp_or_mktcap',
        vol_bound_coeff: float = 0.0,
        vol_bound_floor: float = 0.4,
) -> pd.DataFrame:
    """Calculate trading exclusions and position bounds.

    Determines which instruments can be traded and their position limits based
    on liquidity, portfolio constraints, and risk limits.

    Args:
        df: Input dataframe with market data and filters
        max_position_pct: Maximum position as percentage of portfolio
        max_portfolio_notional: Maximum portfolio size in USD
        max_volume_fraction: Maximum participation rate
        exclude_nonalpha_trades: Whether to exclude non-alpha trades
        max_move_filter: Maximum allowed price move (z-score)
        delisting_buffer_days: Days before delisting to start filtering
        position_bounds: List of (percentile, bound_pct) tuples
        max_dvol_sigma: Maximum dollar volume z-score for expansion
        min_funding_rate: Minimum funding rate before restricting shorts
        bound_col: Column to use for position bound percentiles
        vol_bound_coeff: Volatility penalty coefficient for max_adv_bound.
            Names with logret_1440_trstd_cz > 1.0 get bounds scaled by
            1/(1 + coeff*(cz-1)). Set to 0 to disable.
        vol_bound_floor: Minimum vol scaling factor (e.g. 0.4 = max 60% reduction)

    Returns:
        DataFrame with exclusion flags and position bounds

    Notes:
        - Sets bounds based on volume participation limits
        - Excludes instruments failing liquidity filters
        - Applies portfolio-wide position limits
        - Optionally scales bounds down for high-volatility names
    """
    has_vol_col = 'logret_1440_trstd_cz' in df.columns
    logger.info(f"Setting bounds... vol_bound_coeff={vol_bound_coeff}, "
                f"vol_bound_floor={vol_bound_floor}, "
                f"logret_1440_trstd_cz in df={has_vol_col}")
    df['position'] = df['position'].fillna(0)
    df['tradeable'] = df['tradeable'].fillna(False)
    df['expandable'] = df['expandable'].fillna(False)

    tradeable_uni_idx = df['tradeable']
    expandable_uni_idx = df['expandable'] & tradeable_uni_idx

    tradeable_cnt = len(df[tradeable_uni_idx])
    expandable_cnt = len(df[expandable_uni_idx])
    logger.info(f"Tradeable/Expandable filter taking universe {len(df)} -> {tradeable_cnt} -> {expandable_cnt}")
    if expandable_cnt == 0 or tradeable_cnt == 0:
        raise log_and_raise("No tradeable or expandable securities")

    max_position =max_position_pct * max_portfolio_notional
    max_position_adv_fraction = max_volume_fraction

    # don't let untradeables move from where they are
    df['ubound'] = df['position'].fillna(0)
    df['lbound'] = df['position'].fillna(0)

    for col in ['ubound', 'lbound']:
        assert not df[col].isna().any()

    cutoffs = [0] + [pb[0] for pb in position_bounds]
    bounds = [pb[1] for pb in position_bounds]

    logger.info(f"Cutoffs: {cutoffs}")
    logger.info(f"Percentages: {bounds}")

    if bound_col == 'mdvp_or_mktcap':
        df['mdvp_or_mktcap'] = df[['mdvp', 'marketcap']].min(axis=1)

    # Calculate percentile breakpoints
    percentiles = df[bound_col].quantile(cutoffs)
    logger.info(f"Using bound {bound_col} with percentiles: {percentiles}")
    # Create bins and assign values
    df['max_adv_bound'] = pd.cut(
        df[bound_col],
        bins=percentiles,
        labels=bounds,
        include_lowest=True,
        ordered=False
    ).astype(np.float32) * max_portfolio_notional

    # Apply volatility-based scaling to max_adv_bound
    if vol_bound_coeff > 0:
        df = _apply_vol_bound_scaling(df, vol_bound_coeff, vol_bound_floor)

    # for anything tradeable set outer bounds
    df.loc[tradeable_uni_idx, 'ubound'] = (df['dvolume_1440_trmean'].fillna(0) * max_position_adv_fraction).clip(0, max_position).clip(upper=df['max_adv_bound']).astype(np.float32)
    df.loc[tradeable_uni_idx, 'lbound'] = (df['dvolume_1440_trmean'].fillna(0) * -max_position_adv_fraction).clip(-max_position, 0).clip(lower=-df['max_adv_bound']).astype(np.float32)

    for col in ['ubound', 'lbound']:
        assert not df[col].isna().any()

    # limit the bounds with max_position_value from binance
    df = adjust_bounds_with_max_notional_value(df, tradeable_uni_idx)

    big_volume_idx = df['dvolume_1440_lz'] < max_dvol_sigma
    # don't let unexpandable get bigger
    df.loc[tradeable_uni_idx & ~expandable_uni_idx & big_volume_idx, 'ubound'] = df[['position', 'ubound']].min(axis=1).clip(lower=0)
    df.loc[tradeable_uni_idx & ~expandable_uni_idx & big_volume_idx, 'lbound'] = df[['position', 'lbound']].max(axis=1).clip(upper=0)

    funding_rate_idx = df['last_funding_rate'] < min_funding_rate
    funding_rate_bounded_symbols = df[funding_rate_idx].index.get_level_values('symbol_venue').unique()
    logger.info(f"Restricting shorts on {funding_rate_bounded_symbols} securities...")
    df.loc[funding_rate_idx, 'lbound'] = 0.0

    # don't expand against alpha
    if exclude_nonalpha_trades:
        logger.info("Preventing expansion against OPT alpha")
        df.loc[df['alpha_opt'] < 0, 'ubound'] = df['position'].clip(lower=0)
        df.loc[df['alpha_opt'] > 0, 'lbound'] = df['position'].clip(upper=0)

    if max_move_filter > 0:
        rockets_df = df[df['logret_1440_lz'] > max_move_filter]
        if len(rockets_df) > 0:
            logger.info("rockets:\n" + rockets_df[['logret_1440_lz']].to_markdown())
            df.loc[df['logret_1440_lz'] > max_move_filter, 'lbound'] = df['lbound'] / 2.0

    df["delisting_date_adjusted"] = df['delisting_date'] - td(days=delisting_buffer_days)
    non_delisting_idx = ~(df.index.get_level_values('ts') > df['delisting_date_adjusted'])
    del df['delisting_date_adjusted']
    df.loc[~non_delisting_idx, 'lbound'] = 0.0
    df.loc[~non_delisting_idx, 'ubound'] = 0.0

    max_portfolio_bound = df['ubound'].sum() + df['lbound'].abs().sum()
    logger.info(f"Max portfolio bounds (s.t. max port size!) {fmoney(max_portfolio_bound)}")

    for col in ['ubound', 'lbound']:
        assert not df[col].isna().any()

    return df
