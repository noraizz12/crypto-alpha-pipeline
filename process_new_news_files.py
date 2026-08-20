#!/usr/bin/env python3
"""
Process news files from multiple sources into unified daily parquet files.

This script reads news data from treeofalpha, cryptonews, and cryptopanic sources,
normalizes the fields to a common schema, deduplicates, and outputs a single
parquet file per day.

Usage:
    # Process all unprocessed dates
    python process_new_news_files.py

    # Process specific date range
    python process_new_news_files.py --start-date 2026-01-01 --end-date 2026-01-13

    # Regenerate existing files
    python process_new_news_files.py --regen

    # Debug mode (print to stdout)
    python process_new_news_files.py --debug
"""

import argparse
import logging.config

from lib.news.news_util import parse_date_arg
from lib.news.preprocess import process_news_date_range
from lib.universe import Universe
from lib.util.config import get_config
from lib.util.logging_util import get_logging_config


logging.config.dictConfig(get_logging_config("process_new_news"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process news files from multiple sources into unified daily parquet files"
    )
    parser.add_argument(
        '-c', '--config',
        help='Config file path',
        required=False
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Debug mode: print to stdout instead of saving'
    )
    parser.add_argument(
        '-r', '--regen',
        action='store_true',
        help='Regenerate existing parquet files'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '-t', '--similarity-threshold',
        type=float,
        default=None,
        help='Similarity threshold for deduplication (0-1)'
    )
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='Do not filter by universe (include all tickers)'
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    _, config = get_config(args.config)

    similarity_threshold = args.similarity_threshold
    if similarity_threshold is None:
        similarity_threshold = config.get('NEWS_SIMILARITY_THRESHOLD', 0.9)

    # Load universe for filtering
    universe = None
    if not args.no_filter:
        try:
            universe = Universe(config=config).load_universe_symbols(
                universe_source='file',
                symbol_type='pair',
                filter='fittable'
            )
            logger.info(f"Loaded universe with {len(universe)} symbols")
        except Exception as e:
            logger.warning(f"Could not load universe: {e}. Processing without filter.")

    start_date = parse_date_arg(args.start_date) if args.start_date else None
    end_date = parse_date_arg(args.end_date) if args.end_date else None

    processed = process_news_date_range(
        start_date=start_date,
        end_date=end_date,
        universe=universe,
        similarity_threshold=similarity_threshold,
        regen=args.regen,
        debug=args.debug
    )

    logger.info(f"Processed {processed} date(s)")


if __name__ == '__main__':
    main()
