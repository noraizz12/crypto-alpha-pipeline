"""News preprocessing and signal calculation utilities.

This module provides functions for:
1. Loading and normalizing news from multiple sources
2. Keyword extraction from news text (bullish/bearish indicators)
3. News type classification (Twitter vs Article)
4. Meta features about news coverage patterns (flow, clustering)
5. Source-specific keyword weight computation
6. Volume-weighted signal adjustment
7. News signal calculation and aggregation

These preprocessing steps are applied during news file processing to add
feature columns to the daily parquet files.
"""

import ast
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lib.util.logging_util import KeyLogger

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


# Keywords discovered through feature analysis
BULLISH_KEYWORDS: Dict[str, str] = {
    'breaking_news': r'\bbreaking\b',
    'rally': r'\brally\b|\brallies\b|\brallying\b',
    'ath': r'\bath\b|\ball[- ]time high',
    'today': r'\btoday\b',
    'acquisition': r'\bacquisition\b|\bacquire[sd]?\b',
    'breakout': r'\bbreakout\b|\bbreak out\b',
    'etf': r'\betf\b',
    'upgrade': r'\bupgrade[sd]?\b',
    'pump': r'\bpump\b|\bpumping\b',
}

BEARISH_KEYWORDS: Dict[str, str] = {
    'vitalik': r'\bvitalik\b',
    'ipo': r'\bipo\b',
    'cz': r'\bcz\b',
    'airdrop': r'\bairdrop\b',
    'hack': r'\bhack\b|\bhacked\b|\bhacking\b',
    'sec': r'\bsec\b',
    'lawsuit': r'\blawsuit\b|\bsue[sd]?\b',
    'dump': r'\bdump\b|\bdumping\b',
}

ALL_KEYWORDS: Dict[str, str] = {**BULLISH_KEYWORDS, **BEARISH_KEYWORDS}

# List of keyword column names for reference
KEYWORD_COLUMNS = [f'kw_{name}' for name in ALL_KEYWORDS]
TYPE_COLUMNS = ['is_twitter', 'is_article']
META_COLUMNS = [
    'duplicate_count',
    'symbol_news_count',
    'news_flow_rank',
    'is_low_news_flow',
    'hours_since_last_news',
    'is_clustered_news',
    'is_best_news_condition',
]

# All preprocessing columns
ALL_PREPROCESS_COLUMNS = KEYWORD_COLUMNS + TYPE_COLUMNS + META_COLUMNS


def extract_keyword_features(
    news_df: pd.DataFrame,
    text_col: str = 'text'
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract keyword indicator features from news text.

    Creates binary columns for each keyword indicating presence in the text.
    Also creates news type indicators (is_twitter, is_article).

    Args:
        news_df: DataFrame with news data containing text column
        text_col: Name of column containing text to analyze

    Returns:
        Tuple of (DataFrame with keyword columns added, list of new column names)
    """
    news_df = news_df.copy()
    new_cols = []

    # Create combined text column if not exists
    if text_col not in news_df.columns:
        title_text = news_df['title'].fillna('') if 'title' in news_df.columns else ''
        body_text = news_df['body'].fillna('') if 'body' in news_df.columns else ''
        news_df[text_col] = title_text + ' ' + body_text

    text_lower = news_df[text_col].str.lower()

    # Extract keyword features
    for name, pattern in ALL_KEYWORDS.items():
        col_name = f'kw_{name}'
        news_df[col_name] = text_lower.str.contains(
            pattern, regex=True, na=False
        ).astype(np.int8)
        new_cols.append(col_name)

    # News type features
    if 'type' in news_df.columns:
        news_df['is_twitter'] = (news_df['type'] == 'Twitter').astype(np.int8)
        news_df['is_article'] = (news_df['type'] == 'Article').astype(np.int8)
    else:
        news_df['is_twitter'] = np.int8(0)
        news_df['is_article'] = np.int8(0)
    new_cols.extend(['is_twitter', 'is_article'])

    logger.info(f"Extracted {len(new_cols)} keyword features from {len(news_df):,} news items")

    return news_df, new_cols


def calc_news_meta_features(
    news_df: pd.DataFrame,
    prev_day_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate meta features about news coverage patterns.

    Meta features capture information about news itself rather than content:
    1. news_flow: How much news this symbol typically gets (low flow = better signal)
    2. hours_since_last: Time since last news on this symbol
    3. is_clustered: Whether this is part of clustered news (multiple items within 6h)

    Analysis showed:
    - Low news flow symbols: +90.1bps spread (vs +43.5bps for high flow)
    - Low flow + recent news (<6h): +113.7bps spread (best combination)

    Args:
        news_df: DataFrame with news data, must have 'ts' and 'symbol' or 'symbol_venue'
        prev_day_df: Optional DataFrame with previous day's news for cross-day continuity

    Returns:
        Tuple of (DataFrame with meta features added, list of new column names)
    """
    news_df = news_df.copy()
    new_cols = []

    # Ensure symbol column exists
    if 'symbol' not in news_df.columns and 'symbol_venue' in news_df.columns:
        news_df['symbol'] = news_df['symbol_venue'].str.replace('_binance-futures', '')

    if 'symbol' not in news_df.columns:
        logger.warning("No symbol column found, cannot compute meta features")
        return news_df, new_cols

    if 'ts' not in news_df.columns:
        logger.warning("No ts column found, cannot compute meta features")
        return news_df, new_cols

    # Sort by symbol and time for sequential calculations
    news_df = news_df.sort_values(['symbol', 'ts'])

    # 1. News flow: count of news per symbol in the dataset
    symbol_news_count = news_df.groupby('symbol').size().reset_index(name='symbol_news_count')
    news_df = news_df.merge(symbol_news_count, on='symbol', how='left')

    # Convert to tercile rank (1=low flow, 2=medium, 3=high)
    try:
        news_df['news_flow_rank'] = pd.qcut(
            news_df['symbol_news_count'].rank(method='dense'),
            q=3,
            labels=[1, 2, 3]
        ).astype(np.int8)
    except ValueError:
        # Not enough unique values for 3 quantiles
        news_df['news_flow_rank'] = np.int8(2)

    # Binary indicator for low news flow (best for signal)
    median_count = news_df['symbol_news_count'].median()
    news_df['is_low_news_flow'] = (
        news_df['symbol_news_count'] < median_count
    ).astype(np.int8)
    new_cols.extend(['symbol_news_count', 'news_flow_rank', 'is_low_news_flow'])

    # 2. Hours since last news on this symbol
    # For cross-day continuity, combine with previous day's data
    if prev_day_df is not None and not prev_day_df.empty:
        # Ensure prev_day_df has symbol column
        if 'symbol' not in prev_day_df.columns and 'symbol_venue' in prev_day_df.columns:
            prev_day_df = prev_day_df.copy()
            prev_day_df['symbol'] = prev_day_df['symbol_venue'].str.replace('_binance-futures', '')

        if 'symbol' in prev_day_df.columns and 'ts' in prev_day_df.columns:
            # Get the last news timestamp for each symbol from previous day
            prev_last_news = prev_day_df.groupby('symbol')['ts'].max().reset_index()
            prev_last_news.columns = ['symbol', 'prev_day_last_ts']

            # Merge with current day data
            news_df = news_df.merge(prev_last_news, on='symbol', how='left')

            # Calculate shift within current day
            news_df['prev_news_ts'] = news_df.groupby('symbol')['ts'].shift(1)

            # For first news of day, use previous day's last news timestamp
            first_of_day_mask = news_df['prev_news_ts'].isna()
            news_df.loc[first_of_day_mask, 'prev_news_ts'] = news_df.loc[
                first_of_day_mask, 'prev_day_last_ts'
            ]

            # Clean up
            if 'prev_day_last_ts' in news_df.columns:
                del news_df['prev_day_last_ts']
        else:
            news_df['prev_news_ts'] = news_df.groupby('symbol')['ts'].shift(1)
    else:
        news_df['prev_news_ts'] = news_df.groupby('symbol')['ts'].shift(1)

    news_df['hours_since_last_news'] = (
        (news_df['ts'] - news_df['prev_news_ts']).dt.total_seconds() / 3600
    ).astype(np.float32)

    # Fill NaN (first news for symbol with no prior history) with a large value
    news_df['hours_since_last_news'] = news_df['hours_since_last_news'].fillna(999.0)

    # Clip negative values to 0 (can occur due to timezone issues or timestamp rounding)
    news_df['hours_since_last_news'] = news_df['hours_since_last_news'].clip(lower=0.0)

    # Assert no negative values exist
    assert (news_df['hours_since_last_news'] >= 0).all(), \
        "hours_since_last_news contains negative values after clipping"
    new_cols.append('hours_since_last_news')

    # 3. Is clustered: recent news within 6 hours
    news_df['is_clustered_news'] = (
        news_df['hours_since_last_news'] < 6.0
    ).astype(np.int8)
    new_cols.append('is_clustered_news')

    # 4. Combined best condition: low flow + clustered
    news_df['is_best_news_condition'] = (
        (news_df['is_low_news_flow'] == 1) & (news_df['is_clustered_news'] == 1)
    ).astype(np.int8)
    new_cols.append('is_best_news_condition')

    # Clean up temporary column
    if 'prev_news_ts' in news_df.columns:
        del news_df['prev_news_ts']

    logger.info(
        f"Computed meta features: {news_df['is_low_news_flow'].sum():,} low-flow news, "
        f"{news_df['is_clustered_news'].sum():,} clustered, "
        f"{news_df['is_best_news_condition'].sum():,} best condition"
    )

    return news_df, new_cols


def preprocess_news(
    news_df: pd.DataFrame,
    extract_keywords: bool = True,
    calc_meta: bool = True,
    prev_day_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Apply all preprocessing steps to news data.

    Convenience function that applies keyword extraction and meta feature
    calculation in sequence.

    Args:
        news_df: DataFrame with news data
        extract_keywords: Whether to extract keyword features
        calc_meta: Whether to calculate meta features
        prev_day_df: Optional DataFrame with previous day's news for cross-day continuity

    Returns:
        DataFrame with all preprocessing features added
    """
    if extract_keywords:
        news_df, _ = extract_keyword_features(news_df)

    if calc_meta:
        news_df, _ = calc_news_meta_features(news_df, prev_day_df=prev_day_df)

    return news_df


# =============================================================================
# SIGNAL CALCULATION FUNCTIONS
# =============================================================================

# Default keyword weights from analysis (used when source-specific weights unavailable)
DEFAULT_KEYWORD_WEIGHTS: Dict[str, float] = {
    'acquisition': 0.4, 'hack': 0.3, 'etf': 0.3, 'sec': 0.2,
    'breaking_news': 0.2, 'today': 0.15,
    'lawsuit': -0.5, 'ipo': -0.5, 'cz': -0.3, 'pump': -0.2,
    'ath': -0.15, 'upgrade': -0.1, 'airdrop': 0.2,
    'breakout': 0.0, 'rally': 0.0, 'dump': 0.0, 'vitalik': -0.2,
}


def compute_source_weights(
    news_df: pd.DataFrame,
    return_col: str = 'y_resid_wgtmkt1_1440'
) -> Dict[str, Dict[str, float]]:
    """Compute optimal keyword weights per source using correlation.

    Different news sources have different keyword effects. For example,
    'today' is bullish for CryptoNews but bearish for TreeOfAlpha.

    Args:
        news_df: DataFrame with news data, keyword features, and forward returns
        return_col: Name of column containing forward returns

    Returns:
        Dict mapping source name to dict of keyword weights
    """
    if 'api_source' not in news_df.columns:
        logger.warning("No api_source column found, using combined weights")
        return {}

    if return_col not in news_df.columns:
        logger.warning(f"No {return_col} column found, cannot compute weights")
        return {}

    source_weights: Dict[str, Dict[str, float]] = {}

    for source in news_df['api_source'].unique():
        source_df = news_df[news_df['api_source'] == source]
        weights: Dict[str, float] = {}

        for kw in ALL_KEYWORDS:
            col = f'kw_{kw}'
            if col not in source_df.columns:
                continue

            corr = source_df[col].corr(source_df[return_col])
            if pd.isna(corr):
                weights[kw] = 0.0
            else:
                # Scale correlation to reasonable weight, cap at +/- 0.5
                weights[kw] = float(np.clip(corr * 100, -0.5, 0.5))

        source_weights[source] = weights
        logger.debug(f"Computed {len(weights)} weights for source {source}")

    return source_weights


def apply_volume_weighting(
    signal: np.ndarray,
    dvolume_cz: pd.Series,
    weight_cap: Tuple[float, float] = (0.5, 2.0)
) -> np.ndarray:
    """Apply inverse volume weighting to bullish signals.

    Bullish news at low cross-sectional volume is upweighted because
    the opportunity hasn't been recognized yet. Analysis showed:
    - Low vol + bullish: +135bps
    - High vol + bullish: +62bps

    Args:
        signal: Array of signal values
        dvolume_cz: Series of cross-sectional volume z-scores
        weight_cap: Tuple of (min_weight, max_weight) to cap weights

    Returns:
        Volume-weighted signal array
    """
    signal = signal.copy()
    valid_vol = dvolume_cz.notna()

    if valid_vol.sum() == 0:
        return signal

    vol_std = dvolume_cz[valid_vol].std()
    if vol_std == 0:
        return signal

    # Low volume (negative cz) -> weight > 1
    # High volume (positive cz) -> weight < 1
    vol_weight = 1 - (dvolume_cz / (2 * vol_std))
    vol_weight = vol_weight.clip(weight_cap[0], weight_cap[1])

    # Apply only to bullish signals
    bullish_mask = (signal > 0) & valid_vol
    signal[bullish_mask] = signal[bullish_mask] * vol_weight[bullish_mask].values

    return signal


def apply_meta_feature_weighting(
    signal: np.ndarray,
    is_low_news_flow: pd.Series,
    is_clustered_news: pd.Series,
    low_flow_boost: float = 1.5,
    clustered_boost: float = 1.3
) -> np.ndarray:
    """Apply meta feature weighting to signal.

    Upweights signals on low news flow symbols and clustered news,
    as these conditions showed stronger signal effectiveness.

    Args:
        signal: Array of signal values
        is_low_news_flow: Binary indicator for low news flow symbols
        is_clustered_news: Binary indicator for clustered news
        low_flow_boost: Multiplier for low news flow (default 1.5)
        clustered_boost: Multiplier for clustered news (default 1.3)

    Returns:
        Weighted signal array
    """
    signal = signal.copy()

    # Apply low flow boost
    low_flow_mask = is_low_news_flow == 1
    signal[low_flow_mask] *= low_flow_boost

    # Apply clustered news boost
    clustered_mask = is_clustered_news == 1
    signal[clustered_mask] *= clustered_boost

    return signal


def calc_news_signal(
    news_df: pd.DataFrame,
    source_weights: Optional[Dict[str, Dict[str, float]]] = None,
    use_volume_weighting: bool = True,
    use_meta_weighting: bool = False,
    return_col: str = 'y_resid_wgtmkt1_1440',
    volume_col: str = 'dvolume_cz'
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate news-based trading signal.

    Computes a signal using:
    1. Source-specific keyword weights (or computes them if not provided)
    2. News type adjustment (Twitter bearish, Article bullish)
    3. Optional inverse volume weighting on bullish signals
    4. Optional meta feature weighting (low news flow, clustered news)

    Args:
        news_df: DataFrame with news data and keyword features
        source_weights: Pre-computed source weights, or None to compute
        use_volume_weighting: Whether to apply volume weighting
        use_meta_weighting: Whether to apply meta feature weighting
        return_col: Column name for forward returns (used if computing weights)
        volume_col: Column name for cross-sectional volume z-score

    Returns:
        Tuple of (DataFrame with signal column added, list of new column names)
    """
    news_df = news_df.copy()

    # Ensure keyword features exist
    keyword_cols = [f'kw_{kw}' for kw in ALL_KEYWORDS]
    missing_cols = [c for c in keyword_cols if c not in news_df.columns]
    if missing_cols:
        news_df, _ = extract_keyword_features(news_df)

    # Compute source weights if not provided
    if source_weights is None and return_col in news_df.columns:
        source_weights = compute_source_weights(news_df, return_col)

    signal = np.zeros(len(news_df), dtype=np.float32)

    # Apply source-specific weights
    if source_weights and 'api_source' in news_df.columns:
        for source in news_df['api_source'].unique():
            mask = news_df['api_source'] == source
            weights = source_weights.get(source, {})

            for kw in ALL_KEYWORDS:
                col = f'kw_{kw}'
                if col in news_df.columns:
                    weight = weights.get(kw, 0.0)
                    signal[mask.values] += (
                        news_df.loc[mask, col].values * weight
                    ).astype(np.float32)

        logger.info(f"Applied source-specific weights for {len(source_weights)} sources")
    else:
        # Fallback: use default combined weights from analysis
        for kw, weight in DEFAULT_KEYWORD_WEIGHTS.items():
            col = f'kw_{kw}'
            if col in news_df.columns:
                signal += news_df[col].values * weight

        logger.info("Applied default combined weights (no source-specific)")

    # News type adjustment
    if 'is_twitter' in news_df.columns:
        signal -= news_df['is_twitter'].values * 0.15
    if 'is_article' in news_df.columns:
        signal += news_df['is_article'].values * 0.1

    # Volume weighting
    if use_volume_weighting and volume_col in news_df.columns:
        signal = apply_volume_weighting(signal, news_df[volume_col])
        logger.info("Applied inverse volume weighting to bullish signals")

    # Meta feature weighting (low news flow + clustered news)
    if use_meta_weighting:
        # Compute meta features if not already present
        if 'is_low_news_flow' not in news_df.columns:
            news_df, _ = calc_news_meta_features(news_df)

        signal = apply_meta_feature_weighting(
            signal,
            news_df['is_low_news_flow'],
            news_df['is_clustered_news']
        )
        logger.info("Applied meta feature weighting (low flow + clustered)")

    news_df['news_signal'] = signal.astype(np.float32)

    return news_df, ['news_signal']


def aggregate_news_signal(
    news_df: pd.DataFrame,
    signal_col: str = 'news_signal',
    agg_method: str = 'sum'
) -> pd.DataFrame:
    """Aggregate news signals to symbol-time level.

    Multiple news items can occur for the same symbol at the same time.
    This function aggregates them into a single signal per symbol-time.

    Args:
        news_df: DataFrame with news signals, indexed by ts and symbol
        signal_col: Name of signal column to aggregate
        agg_method: Aggregation method ('sum', 'mean', 'max')

    Returns:
        DataFrame with aggregated signals at symbol-time level
    """
    if 'ts_minute' not in news_df.columns:
        news_df = news_df.copy()
        news_df['ts_minute'] = news_df['ts'].dt.floor('min')

    # Group by symbol and time
    group_cols = ['ts_minute', 'symbol']
    if 'symbol_venue' in news_df.columns and 'symbol' not in news_df.columns:
        news_df['symbol'] = news_df['symbol_venue'].str.replace('_binance-futures', '')

    if agg_method == 'sum':
        agg_df = news_df.groupby(group_cols)[signal_col].sum().reset_index()
    elif agg_method == 'mean':
        agg_df = news_df.groupby(group_cols)[signal_col].mean().reset_index()
    elif agg_method == 'max':
        agg_df = news_df.groupby(group_cols)[signal_col].max().reset_index()
    else:
        raise ValueError(f"Unknown agg_method: {agg_method}")

    # Also count number of news items
    count_df = news_df.groupby(group_cols).size().reset_index(name='news_count')
    agg_df = agg_df.merge(count_df, on=group_cols)

    logger.info(
        f"Aggregated {len(news_df):,} news items to {len(agg_df):,} "
        f"symbol-time observations using {agg_method}"
    )

    return agg_df


# =============================================================================
# NEWS FILE LOADING AND PROCESSING
# =============================================================================

def load_source_file(filepath: str, api_source: str) -> pd.DataFrame:
    """Load a single source file and normalize to common schema.

    Args:
        filepath: Path to the source CSV file (JSON lines)
        api_source: Source identifier (treeofalpha, cryptonews, cryptopanic)

    Returns:
        DataFrame with normalized columns
    """
    # Import here to avoid circular imports
    from lib.news.news_util import (
        FIELD_MAPPING,
        parse_timestamp,
        extract_tickers_from_suggestions,
    )

    if not os.path.exists(filepath):
        logger.debug(f"File not found: {filepath}")
        return pd.DataFrame()

    records = []
    mapping = FIELD_MAPPING.get(api_source, {})

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Try JSON first, fall back to ast.literal_eval for single quotes
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    data = ast.literal_eval(line)

                record = {'api_source': api_source}

                # Map fields according to source mapping
                for common_field, source_field in mapping.items():
                    if source_field and source_field in data:
                        record[common_field] = data[source_field]
                    else:
                        record[common_field] = None

                # Extract tickers
                if 'tickers' in data and data['tickers']:
                    record['tickers'] = data['tickers']
                elif record.get('suggestions'):
                    record['tickers'] = extract_tickers_from_suggestions(record['suggestions'])
                else:
                    record['tickers'] = []

                # Parse timestamp
                record['ts'] = parse_timestamp(record.get('time_raw'), api_source)

                records.append(record)

            except (json.JSONDecodeError, SyntaxError, ValueError) as e:
                logger.warning(f"Failed to parse line {line_num} in {filepath}: {e}")
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} records from {filepath}")
    return df


def process_news_date(
    date_str: str,
    universe: Optional[List[str]],
    similarity_threshold: float,
    news_dir: str,
    prev_day_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """Process all news sources for a single date.

    Args:
        date_str: Date string in YYYYMMDD format
        universe: List of symbols to filter for (None = no filter)
        similarity_threshold: Deduplication threshold
        news_dir: Base directory for news data
        prev_day_df: Optional DataFrame with previous day's processed news

    Returns:
        Processed DataFrame or None if no data
    """
    # Import here to avoid circular imports
    from lib.data.data_news import deduplicate_news_df
    from lib.news.news_util import (
        NEWS_SOURCES,
        OUTPUT_COLUMNS_BASE,
        serialize_list_to_json,
        serialize_dict_to_json,
    )
    from lib.util.dataframes import safe_del
    from lib.util.util import remove_non_ascii_and_extra_whitespace, TARDIS_EXCHANGE

    date_dir = f"{news_dir}/{date_str}"

    if not os.path.isdir(date_dir):
        logger.debug(f"No directory found for {date_str}")
        return None

    dfs = []
    for source in NEWS_SOURCES:
        filepath = f"{date_dir}/{source}.{date_str}.csv"
        source_df = load_source_file(filepath, source)
        if not source_df.empty:
            dfs.append(source_df)

    if not dfs:
        logger.warning(f"No news data found for {date_str}")
        return None

    # Combine all sources
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(df)} records from {len(dfs)} sources for {date_str}")

    # Filter by timestamp - only keep records from this date
    if 'ts' in df.columns:
        df = df[df['ts'].notna()]
        if df.empty:
            logger.warning(f"No valid timestamps for {date_str}")
            return None

    # Clean text fields
    df['title'] = df['title'].apply(
        lambda x: remove_non_ascii_and_extra_whitespace(x) if pd.notna(x) else None
    )
    df['body'] = df['body'].apply(
        lambda x: remove_non_ascii_and_extra_whitespace(x) if pd.notna(x) else None
    )

    # Explode tickers: each article with multiple tickers becomes multiple rows
    # This ensures each ticker mentioned gets its own entry for that article
    df['tickers'] = df['tickers'].apply(lambda x: x if isinstance(x, list) and x else [None])
    original_count = len(df)
    df = df.explode('tickers', ignore_index=True)
    df = df.rename(columns={'tickers': 'ticker'})

    # Remove rows with no ticker
    df = df[df['ticker'].notna()]
    if df.empty:
        logger.info(f"No news with tickers for {date_str}")
        return None

    logger.info(f"Exploded {original_count} articles to {len(df)} ticker-specific rows")

    # Filter to universe if specified
    if universe:
        # Create set of base tickers from universe (strip USDT)
        universe_tickers = {s.replace('USDT', '') for s in universe}
        df = df[df['ticker'].isin(universe_tickers)]
        if df.empty:
            logger.info(f"No news matching universe for {date_str}")
            return None
        logger.info(f"Filtered to {len(df)} records matching universe")

    # Create symbol_venue for compatibility with existing code
    df['symbol_venue'] = df['ticker'].apply(
        lambda x: f"{x}USDT_{TARDIS_EXCHANGE}" if pd.notna(x) else None
    )

    # Round timestamp to minute
    df['ts'] = pd.to_datetime(df['ts']).dt.ceil('min')

    # Count duplicates before removing them (duplicate_count indicates article significance)
    dup_cols = ['title', 'body', 'symbol_venue']
    dup_counts = df.groupby(dup_cols).size().reset_index(name='duplicate_count')
    df = df.merge(dup_counts, on=dup_cols, how='left')
    logger.info(f"Duplicate counts: max={df['duplicate_count'].max()}, "
                f"mean={df['duplicate_count'].mean():.2f}, "
                f"articles with >1 duplicate={len(df[df['duplicate_count'] > 1])}")

    # Drop duplicates based on title, body, symbol_venue (keep duplicate_count from first)
    df = df.drop_duplicates(subset=dup_cols, keep='first')

    # Apply semantic deduplication
    df = deduplicate_news_df(df, similarity_threshold=similarity_threshold)

    # Apply news preprocessing (keyword extraction and meta features)
    df = preprocess_news(df, extract_keywords=True, calc_meta=True, prev_day_df=prev_day_df)

    # Clean up intermediate columns
    for col in ['time_raw', 'suggestions', 'twitter_info', 'ticker', 'text', 'symbol']:
        safe_del(df, col)

    # Convert complex types to JSON strings for parquet storage
    df['votes'] = df['votes'].apply(serialize_dict_to_json)
    df['topics'] = df['topics'].apply(serialize_list_to_json)

    # Select and order output columns (note: tickers column removed since we exploded it)
    output_columns = [c for c in OUTPUT_COLUMNS_BASE if c != 'tickers'] + ALL_PREPROCESS_COLUMNS
    output_cols = [c for c in output_columns if c in df.columns]
    df = df[output_cols]

    logger.info(f"Processed {len(df)} records for {date_str}")
    return df


def process_news_date_range(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    universe: Optional[List[str]] = None,
    similarity_threshold: float = 0.9,
    news_dir: Optional[str] = None,
    regen: bool = False,
    debug: bool = False
) -> int:
    """Process news files for a date range.

    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        universe: List of symbols to filter for
        similarity_threshold: Deduplication threshold
        news_dir: Base directory for news data (uses default if None)
        regen: If True, regenerate existing files
        debug: If True, print output instead of saving

    Returns:
        Number of files processed
    """
    from lib.util.directory import dir_manager
    from lib.util.time_util import date_to_str

    if news_dir is None:
        news_dir = dir_manager.NEWS_DIR_NEW

    # Find all date directories
    if not os.path.exists(news_dir):
        logger.error(f"News directory not found: {news_dir}")
        return 0

    date_dirs = []
    for entry in os.listdir(news_dir):
        entry_path = os.path.join(news_dir, entry)
        if os.path.isdir(entry_path) and len(entry) == 8 and entry.isdigit():
            date_dirs.append(entry)

    date_dirs = sorted(date_dirs)

    if not date_dirs:
        logger.warning("No date directories found")
        return 0

    # Filter by date range
    if start_date:
        start_str = start_date.strftime('%Y%m%d')
        date_dirs = [d for d in date_dirs if d >= start_str]

    if end_date:
        end_str = end_date.strftime('%Y%m%d')
        date_dirs = [d for d in date_dirs if d <= end_str]

    # Skip today's date (incomplete data)
    today_str = date_to_str()
    date_dirs = [d for d in date_dirs if d != today_str]

    processed = 0
    prev_day_df = None

    for idx, date_str in enumerate(date_dirs):
        output_file = f"{news_dir}/news.{date_str}.parquet"

        # Try to load previous day's processed file for cross-day continuity
        if idx > 0:
            prev_date_str = date_dirs[idx - 1]
            prev_file = f"{news_dir}/news.{prev_date_str}.parquet"
            if os.path.exists(prev_file):
                try:
                    prev_day_df = pd.read_parquet(prev_file)
                    logger.debug(f"Loaded {len(prev_day_df)} records from previous day {prev_date_str}")
                except Exception as e:
                    logger.warning(f"Could not load previous day file {prev_file}: {e}")
                    prev_day_df = None
            else:
                prev_day_df = None

        # Skip if already processed (unless regen)
        if not regen and os.path.exists(output_file):
            logger.debug(f"Skipping {date_str} (already processed)")
            continue

        logger.info(f"Processing {date_str}")
        df = process_news_date(
            date_str,
            universe,
            similarity_threshold,
            news_dir,
            prev_day_df=prev_day_df,
        )

        if df is None or df.empty:
            continue

        if debug:
            print(f"\n=== {date_str} ===")
            print(df.to_string())
        else:
            df.to_parquet(output_file)
            logger.info(f"Saved {output_file}")

        processed += 1

    return processed
