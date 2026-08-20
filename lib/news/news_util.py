"""
Shared utilities for news source modules.

This module contains common functionality used across all news sources,
including universe loading, timestamp parsing, and field mapping.
"""

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

from dateutil import parser as date_parser

from lib.universe import Universe
from lib.util.util import SYMBOL_BASE


logger = logging.getLogger(__name__)

# =============================================================================
# NEWS SOURCE CONFIGURATION
# =============================================================================

# Supported news sources
NEWS_SOURCES: List[str] = ['treeofalpha', 'cryptonews', 'cryptopanic', 'cryptoslate']

# Field mapping from each source to common schema
# Format: {source: {common_field: source_field}}
FIELD_MAPPING: Dict[str, Dict[str, Optional[str]]] = {
    'treeofalpha': {
        '_id': '_id',
        'title': 'title',
        'body': None,  # TreeOfAlpha doesn't have body
        'url': 'url',
        'source': 'sourceName',
        'time_raw': 'time',
        'live_ts': 'live_ts',
        'suggestions': 'suggestions',
        'sentiment': None,
        'type': 'source',
        'topics': None,
        'votes': None,
        'panic_score': None,
        'twitter_info': 'info',
    },
    'cryptonews': {
        '_id': '_id',
        'title': 'title',
        'body': 'body',
        'url': 'url',
        'source': 'source',
        'time_raw': 'time',
        'live_ts': 'live_ts',
        'tickers': 'tickers',
        'suggestions': 'suggestions',
        'sentiment': 'sentiment',
        'type': 'type',
        'topics': 'topics',
        'votes': None,
        'panic_score': None,
        'twitter_info': None,
    },
    'cryptopanic': {
        '_id': '_id',
        'title': 'title',
        'body': 'body',
        'url': 'url',
        'source': 'source',
        'time_raw': 'time',
        'live_ts': 'live_ts',
        'tickers': 'tickers',
        'suggestions': 'suggestions',
        'sentiment': 'sentiment',
        'type': 'type',
        'topics': None,
        'votes': 'votes',
        'panic_score': 'panic_score',
        'twitter_info': None,
    },
    'cryptoslate': {
        '_id': '_id',
        'title': 'title',
        'body': 'body',
        'url': 'url',
        'source': 'source',
        'time_raw': 'time',
        'live_ts': 'live_ts',
        'tickers': 'tickers',
        'suggestions': 'suggestions',
        'sentiment': 'sentiment',
        'type': 'type',
        'topics': 'topics',
        'votes': None,
        'panic_score': None,
        'twitter_info': None,
    },
}

# Base output columns (before preprocessing columns are added)
OUTPUT_COLUMNS_BASE: List[str] = [
    'ts',
    'symbol_venue',
    'title',
    'body',
    'url',
    'source',
    'api_source',
    'tickers',
    'sentiment',
    'type',
    'topics',
    'votes',
    'panic_score',
    '_id',
    'live_ts',
]


# =============================================================================
# TIMESTAMP PARSING
# =============================================================================

def parse_timestamp(time_value, api_source: str) -> Optional[datetime]:
    """Parse timestamp from various formats to datetime.

    Args:
        time_value: Raw timestamp value (ms int, RFC 2822 string, or ISO 8601 string)
        api_source: Source name to determine format

    Returns:
        datetime object or None if parsing fails
    """
    if time_value is None:
        return None

    try:
        if api_source == 'treeofalpha':
            # TreeOfAlpha uses milliseconds timestamp
            if isinstance(time_value, (int, float)):
                return datetime.fromtimestamp(
                    time_value / 1000, tz=timezone.utc
                ).replace(tzinfo=None)
            return None

        if api_source == 'cryptonews':
            # CryptoNews uses RFC 2822 format: "Tue, 13 Jan 2026 23:58:52 -0500"
            if isinstance(time_value, str):
                return date_parser.parse(time_value).replace(tzinfo=None)
            return None

        if api_source == 'cryptopanic':
            # CryptoPanic uses ISO 8601 format: "2026-01-13T23:05:04Z"
            if isinstance(time_value, str):
                return date_parser.parse(time_value).replace(tzinfo=None)
            return None

        if api_source == 'cryptoslate':
            # CryptoSlate uses ISO 8601 format: "2026-01-13T23:05:04"
            if isinstance(time_value, str):
                return date_parser.parse(time_value).replace(tzinfo=None)
            return None

    except (ValueError, TypeError, OverflowError) as e:
        logger.debug(f"Failed to parse timestamp {time_value} from {api_source}: {e}")
        return None

    return None


def parse_date_arg(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d")


# =============================================================================
# TICKER EXTRACTION
# =============================================================================

def extract_tickers_from_suggestions(suggestions: list) -> List[str]:
    """Extract ticker symbols from suggestions array.

    Args:
        suggestions: List of suggestion dicts with 'coin' field

    Returns:
        List of ticker symbols
    """
    if not suggestions or not isinstance(suggestions, list):
        return []

    tickers = []
    for suggestion in suggestions:
        if isinstance(suggestion, dict):
            coin = suggestion.get('coin')
            if coin and isinstance(coin, str):
                tickers.append(coin.upper())

    return list(set(tickers))


def get_primary_ticker(tickers) -> Optional[str]:
    """Get primary ticker from a list of tickers.

    Args:
        tickers: List of ticker symbols

    Returns:
        First ticker or None
    """
    if isinstance(tickers, list) and tickers:
        return tickers[0]
    return None


# =============================================================================
# JSON SERIALIZATION FOR PARQUET
# =============================================================================

def serialize_list_to_json(value) -> Optional[str]:
    """Convert list to JSON string for parquet storage."""
    if isinstance(value, list):
        return json.dumps(value)
    return None


def serialize_dict_to_json(value) -> Optional[str]:
    """Convert dict to JSON string for parquet storage."""
    if isinstance(value, dict):
        return json.dumps(value)
    return None


# =============================================================================
# UNIVERSE LOADING
# =============================================================================

# Major tickers to use as fallback when universe file not available
MAJOR_TICKERS = [
    "BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX", "DOT", "LINK",
    "MATIC", "UNI", "LTC", "ATOM", "FIL", "APT", "ARB", "OP", "SUI", "SEI",
]


def load_tickers_for_date(
    universe: Universe,
    target_date: date,
    use_prior_day: bool = True
) -> Set[str]:
    """Load universe tickers for a specific date.

    For historical downloads, we use the prior day's universe to determine
    which symbols were relevant at that time. This ensures we're filtering
    based on what was in the universe when the news was published.

    Args:
        universe: Universe instance to load from
        target_date: The date of the news data being processed
        use_prior_day: If True, load universe from prior day (default for historical)

    Returns:
        Set of ticker symbols (without USDT suffix)
    """
    universe_date = target_date - timedelta(days=1) if use_prior_day else target_date

    try:
        symbols = universe.load_universe_symbols(
            universe_source='file',
            universe_date=universe_date,
            filter='fittable',
            symbol_type=SYMBOL_BASE
        )

        if symbols:
            tickers = {s.replace('USDT', '') for s in symbols}
            logger.info(f"Loaded {len(tickers)} tickers for universe date {universe_date}")
            return tickers

    except Exception as e:
        logger.warning(f"Could not load universe for {universe_date}: {e}")

    # Fallback to major tickers if no universe file exists
    logger.warning(f"Using fallback major tickers for {target_date}")
    return set(MAJOR_TICKERS)


def get_query_tickers(tickers: Set[str], major_only: bool = True) -> str:
    """Get comma-separated ticker string for API queries.

    Some APIs have URL length limits, so we filter to major tickers
    that are in the current universe.

    Args:
        tickers: Set of tickers from universe
        major_only: If True, only include major tickers (for API limits)

    Returns:
        Comma-separated ticker string
    """
    if major_only:
        query_tickers = [t for t in MAJOR_TICKERS if t in tickers]
        if not query_tickers:
            query_tickers = MAJOR_TICKERS[:10]
    else:
        query_tickers = list(tickers)

    return ",".join(query_tickers)
