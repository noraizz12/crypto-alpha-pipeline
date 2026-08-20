"""
News module - Cryptocurrency news ingestion from multiple sources.

This module provides real-time and historical news collection from various
cryptocurrency news sources including TreeOfAlpha, CryptoNews, and CryptoPanic.

Usage:
    # Run all sources in real-time
    python -m lib.news.news_server

    # Run individual source for historical download
    python -m lib.news.source_cryptonews --historical --start-date 2024-01-01 --end-date 2024-01-31
    python -m lib.news.source_treeofalpha --historical --start-date 2024-01-01 --end-date 2024-01-31
    python -m lib.news.source_cryptopanic --historical --start-date 2024-01-01 --end-date 2024-01-31
"""

__all__ = [
    'TreeOfAlphaSource',
    'CryptoNewsSource',
    'CryptoPanicSource',
    'CryptoSlateSource',
]


def __getattr__(name: str):
    """Lazy import to avoid RuntimeWarning when running modules directly."""
    if name == 'TreeOfAlphaSource':
        from .source_treeofalpha import TreeOfAlphaSource
        return TreeOfAlphaSource
    if name == 'CryptoNewsSource':
        from .source_cryptonews import CryptoNewsSource
        return CryptoNewsSource
    if name == 'CryptoPanicSource':
        from .source_cryptopanic import CryptoPanicSource
        return CryptoPanicSource
    if name == 'CryptoSlateSource':
        from .source_cryptoslate import CryptoSlateSource
        return CryptoSlateSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
