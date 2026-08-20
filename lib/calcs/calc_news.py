"""News-based alpha signal calculations.

This module provides functions for calculating news-based trading signals
from preprocessed news data. The approach uses empirically-derived keyword
weights and meta-feature adjustments based on historical return analysis.

Key findings from empirical analysis (adhoc/news_alpha_analysis_v2.py):
- Keywords only work predictively for articles (not twitter)
- is_low_news_flow adds +25 bps standalone value
- is_best_news_condition adds +25.5 bps standalone value
- is_twitter has strong negative signal (-136 bps)
- Signal decays after 6 hours; reverses after 24 hours

Usage:
    from lib.calcs.calc_news import calculate_news_alpha

    df, new_cols = calculate_news_alpha(
        bars_df=bars_df,
        news_df=news_df,
        horizon=1440
    )
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lib.util.dataframes import merge_on_index

logger = logging.getLogger(__name__)


# =============================================================================
# EMPIRICALLY-DERIVED WEIGHTS (from adhoc analysis on 1440-min horizon)
# =============================================================================

# Keyword weights based on statistically significant return spreads (bps)
# Positive = bullish signal, Negative = bearish signal
KEYWORD_WEIGHTS: Dict[str, float] = {
    # Bullish keywords (statistically significant at 1440-min horizon)
    'kw_breaking_news': 0.0033,   # +32.7 bps spread
    'kw_breakout': 0.0020,        # +19.6 bps spread
    'kw_rally': 0.0016,           # +16.0 bps spread
    'kw_today': 0.0014,           # +14.0 bps spread
    'kw_upgrade': 0.0013,         # +13.1 bps spread (significant at 360-min)
    'kw_ath': 0.0007,             # +7.0 bps spread
    'kw_acquisition': 0.0005,     # +5.0 bps spread
    'kw_etf': 0.0000,             # Not significant

    # Bearish keywords
    'kw_hack': -0.0010,           # Bearish signal
    'kw_dump': -0.0005,           # Bearish signal
    'kw_pump': -0.0002,           # Slightly bearish (pump and dump)
    'kw_lawsuit': 0.0000,         # Mixed signal across horizons
    'kw_sec': 0.0000,             # Mixed signal (positive at low_flow)
    'kw_vitalik': 0.0000,         # Not significant
    'kw_ipo': 0.0000,             # Not significant
    'kw_cz': 0.0000,              # Not significant
    'kw_airdrop': 0.0000,         # Not significant
}

# Meta feature weights (based on standalone value analysis)
META_WEIGHTS: Dict[str, float] = {
    'is_low_news_flow': 0.0025,       # +24.8 bps standalone
    'is_best_news_condition': 0.0025,  # +25.5 bps standalone
    'is_twitter': -0.0136,             # -136.2 bps standalone (STRONG negative)
    'is_article': 0.0000,              # Used for conditioning, not additive
    'is_clustered_news': 0.0000,       # Not significant standalone
}

# Decay parameters
DEFAULT_DECAY_HALFLIFE_HOURS = 6.0  # Signal halves every 6 hours
MAX_SIGNAL_AGE_HOURS = 24.0         # Signal zeroed after 24 hours


# =============================================================================
# SIGNAL CALCULATION FUNCTIONS
# =============================================================================

def calculate_keyword_signal(
    news_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    article_only: bool = True
) -> pd.Series:
    """Calculate weighted keyword signal from news features.

    Args:
        news_df: DataFrame with keyword columns (kw_*)
        weights: Optional custom weights dict. Uses KEYWORD_WEIGHTS if None.
        article_only: If True, only apply keyword signal for articles.

    Returns:
        Series with keyword signal values (indexed same as news_df)
    """
    if weights is None:
        weights = KEYWORD_WEIGHTS

    signal = pd.Series(0.0, index=news_df.index, dtype=np.float32)

    for keyword, weight in weights.items():
        if keyword in news_df.columns and weight != 0:
            keyword_values = news_df[keyword].fillna(0).astype(np.float32)
            signal += keyword_values * weight

    # Apply article conditioning: keywords only work for articles
    if article_only and 'is_article' in news_df.columns:
        article_mask = news_df['is_article'] == 1
        signal = signal * article_mask.astype(np.float32)
        logger.debug(f"Applied article filter: {article_mask.sum():,} of {len(news_df):,} are articles")

    return signal


def calculate_meta_signal(
    news_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None
) -> pd.Series:
    """Calculate signal from meta features.

    Args:
        news_df: DataFrame with meta feature columns
        weights: Optional custom weights dict. Uses META_WEIGHTS if None.

    Returns:
        Series with meta signal values
    """
    if weights is None:
        weights = META_WEIGHTS

    signal = pd.Series(0.0, index=news_df.index, dtype=np.float32)

    for feature, weight in weights.items():
        if feature in news_df.columns and weight != 0:
            feature_values = news_df[feature].fillna(0).astype(np.float32)
            signal += feature_values * weight

    return signal


def apply_time_decay(
    signal: pd.Series,
    hours_since_news: pd.Series,
    halflife_hours: float = DEFAULT_DECAY_HALFLIFE_HOURS,
    max_age_hours: float = MAX_SIGNAL_AGE_HOURS
) -> pd.Series:
    """Apply exponential time decay to news signal.

    Signal decays as: signal * exp(-t * ln(2) / halflife)

    Args:
        signal: Raw signal values
        hours_since_news: Hours since the news event
        halflife_hours: Half-life for decay (default: 6 hours)
        max_age_hours: Maximum age before signal is zeroed (default: 24 hours)

    Returns:
        Decayed signal values
    """
    hours = hours_since_news.fillna(999.0).clip(lower=0)

    # Calculate decay factor: exp(-t * ln(2) / halflife)
    decay_factor = np.exp(-hours * np.log(2) / halflife_hours)

    # Zero out signals older than max_age
    decay_factor = np.where(hours > max_age_hours, 0.0, decay_factor)

    decayed_signal = signal * decay_factor.astype(np.float32)

    return decayed_signal


def aggregate_news_to_symbol_minute(
    news_df: pd.DataFrame,
    keyword_cols: Optional[List[str]] = None,
    meta_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Aggregate multiple news items per symbol-minute.

    Args:
        news_df: Raw news DataFrame with ts, symbol_venue columns
        keyword_cols: List of keyword columns to aggregate (max)
        meta_cols: List of meta columns to aggregate

    Returns:
        Aggregated DataFrame with one row per (ts_minute, symbol_venue)
    """
    if keyword_cols is None:
        keyword_cols = [col for col in news_df.columns if col.startswith('kw_')]

    if meta_cols is None:
        meta_cols = ['is_twitter', 'is_article', 'is_low_news_flow',
                     'is_clustered_news', 'is_best_news_condition']

    # Ensure ts_minute exists
    if 'ts_minute' not in news_df.columns:
        news_df = news_df.copy()
        news_df['ts_minute'] = pd.to_datetime(news_df['ts']).dt.floor('min')

    # Build aggregation dict
    agg_dict = {}

    # Keywords: use max (any keyword present = 1)
    for col in keyword_cols:
        if col in news_df.columns:
            agg_dict[col] = 'max'

    # Binary meta features: use max
    for col in meta_cols:
        if col in news_df.columns:
            agg_dict[col] = 'max'

    # Continuous features
    if 'hours_since_last_news' in news_df.columns:
        agg_dict['hours_since_last_news'] = 'min'  # Use most recent

    # Count news items
    agg_dict['ts'] = 'count'

    if not agg_dict:
        logger.warning("No columns to aggregate in news_df")
        return news_df

    grouped = news_df.groupby(['ts_minute', 'symbol_venue']).agg(agg_dict)
    grouped = grouped.rename(columns={'ts': 'news_count'})
    grouped = grouped.reset_index()

    logger.info(f"Aggregated {len(news_df):,} news items to {len(grouped):,} symbol-minute observations")

    return grouped


def calculate_news_alpha(
    bars_df: pd.DataFrame,
    news_df: pd.DataFrame,
    horizon: int = 1440,
    keyword_weights: Optional[Dict[str, float]] = None,
    meta_weights: Optional[Dict[str, float]] = None,
    decay_halflife_hours: float = DEFAULT_DECAY_HALFLIFE_HOURS,
    article_only: bool = True,
    normalize: bool = True,
    sigma_bound: float = 5.0
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate news-based alpha signal and merge with bars data.

    This is the main entry point for news alpha calculation. It:
    1. Aggregates news to symbol-minute level
    2. Calculates keyword signal (conditioned on is_article)
    3. Adds meta feature signal
    4. Applies time decay
    5. Merges with bars DataFrame

    Args:
        bars_df: DataFrame with bar data, indexed by (ts, symbol_venue)
        news_df: DataFrame with preprocessed news data
        horizon: Time horizon in minutes (for column naming)
        keyword_weights: Custom keyword weights (uses defaults if None)
        meta_weights: Custom meta weights (uses defaults if None)
        decay_halflife_hours: Half-life for signal decay
        article_only: If True, keyword signal only applies to articles
        normalize: If True, z-score normalize the final signal
        sigma_bound: Clip normalized signal to [-bound, +bound]

    Returns:
        Tuple of (DataFrame with news alpha column added, list of new column names)

    Example:
        >>> bars_df, new_cols = calculate_news_alpha(bars_df, news_df, horizon=1440)
        >>> print(new_cols)
        ['alpha_news_1440']
    """
    alpha_col = f'alpha_news_{horizon}'
    logger.info(f"Calculating {alpha_col} with decay_halflife={decay_halflife_hours}h, article_only={article_only}")

    # Validate inputs
    if news_df.empty:
        logger.warning("Empty news_df, returning zero alpha")
        bars_df[alpha_col] = np.float32(0.0)
        return bars_df, [alpha_col]

    if 'ts' not in news_df.columns or 'symbol_venue' not in news_df.columns:
        raise ValueError("news_df must have 'ts' and 'symbol_venue' columns")

    # Step 1: Aggregate news to symbol-minute level
    news_agg = aggregate_news_to_symbol_minute(news_df)

    # Step 2: Calculate keyword signal
    keyword_signal = calculate_keyword_signal(
        news_agg,
        weights=keyword_weights,
        article_only=article_only
    )

    # Step 3: Calculate meta signal
    meta_signal = calculate_meta_signal(news_agg, weights=meta_weights)

    # Step 4: Combine signals
    raw_signal = keyword_signal + meta_signal

    # Step 5: Apply time decay
    if 'hours_since_last_news' in news_agg.columns:
        news_agg[alpha_col] = apply_time_decay(
            raw_signal,
            news_agg['hours_since_last_news'],
            halflife_hours=decay_halflife_hours
        )
    else:
        news_agg[alpha_col] = raw_signal
        logger.warning("No hours_since_last_news column, skipping decay")

    # Step 6: Prepare for merge with bars
    news_agg['ts_minute'] = pd.to_datetime(news_agg['ts_minute'])

    # Handle timezone: bars typically have UTC, news may not
    # Check bars_df timezone from index
    bars_index = bars_df.index.get_level_values('ts')
    bars_tz = bars_index.tz

    if bars_tz is not None:
        # Bars have timezone, localize news to match
        if news_agg['ts_minute'].dt.tz is None:
            news_agg['ts_minute'] = news_agg['ts_minute'].dt.tz_localize('UTC')
        else:
            news_agg['ts_minute'] = news_agg['ts_minute'].dt.tz_convert('UTC')
    else:
        # Bars don't have timezone, strip from news if present
        if news_agg['ts_minute'].dt.tz is not None:
            news_agg['ts_minute'] = news_agg['ts_minute'].dt.tz_localize(None)

    # Create merge DataFrame with just the alpha column
    alpha_df = news_agg[['ts_minute', 'symbol_venue', alpha_col]].copy()
    alpha_df = alpha_df.rename(columns={'ts_minute': 'ts'})
    alpha_df = alpha_df.set_index(['ts', 'symbol_venue'])

    # Step 7: Merge with bars
    bars_df = merge_on_index(bars_df, alpha_df, how='left')

    # Fill NaN (no news) with 0
    bars_df[alpha_col] = bars_df[alpha_col].fillna(0.0)

    # Step 8: Normalize if requested
    if normalize:
        # Calculate stats excluding zeros (no-news periods)
        non_zero_mask = bars_df[alpha_col] != 0
        if non_zero_mask.sum() > 100:
            signal_mean = bars_df.loc[non_zero_mask, alpha_col].mean()
            signal_std = bars_df.loc[non_zero_mask, alpha_col].std()

            if signal_std > 1e-10:
                bars_df[alpha_col] = (bars_df[alpha_col] - signal_mean) / signal_std
                bars_df[alpha_col] = bars_df[alpha_col].clip(-sigma_bound, sigma_bound)
                logger.info(f"Normalized {alpha_col}: mean={signal_mean:.6f}, std={signal_std:.6f}")
        else:
            logger.warning(f"Not enough non-zero values to normalize ({non_zero_mask.sum()})")

    bars_df[alpha_col] = bars_df[alpha_col].astype(np.float32)

    n_nonzero = (bars_df[alpha_col] != 0).sum()
    logger.info(f"Created {alpha_col}: {n_nonzero:,} non-zero values out of {len(bars_df):,}")

    return bars_df, [alpha_col]


def calculate_news_event_indicator(
    bars_df: pd.DataFrame,
    news_df: pd.DataFrame,
    horizon: int = 1440
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate binary news event indicator.

    Creates a simple 0/1 indicator for whether news occurred at each
    symbol-minute. Useful as a filter or for event studies.

    Args:
        bars_df: DataFrame with bar data
        news_df: DataFrame with news data
        horizon: Time horizon for column naming

    Returns:
        Tuple of (DataFrame with news_event column, list of new column names)
    """
    event_col = f'news_event_{horizon}'

    if news_df.empty:
        bars_df[event_col] = np.int8(0)
        return bars_df, [event_col]

    # Get unique (ts_minute, symbol_venue) pairs with news
    news_df = news_df.copy()
    news_df['ts_minute'] = pd.to_datetime(news_df['ts']).dt.floor('min')
    if news_df['ts_minute'].dt.tz is not None:
        news_df['ts_minute'] = news_df['ts_minute'].dt.tz_localize(None)

    news_events = news_df[['ts_minute', 'symbol_venue']].drop_duplicates()
    news_events[event_col] = np.int8(1)
    news_events = news_events.rename(columns={'ts_minute': 'ts'})
    news_events = news_events.set_index(['ts', 'symbol_venue'])

    # Merge with bars
    bars_df = merge_on_index(bars_df, news_events, how='left')
    bars_df[event_col] = bars_df[event_col].fillna(0).astype(np.int8)

    n_events = (bars_df[event_col] == 1).sum()
    logger.info(f"Created {event_col}: {n_events:,} events out of {len(bars_df):,} rows")

    return bars_df, [event_col]
