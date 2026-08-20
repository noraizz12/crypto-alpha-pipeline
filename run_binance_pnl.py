#!/usr/bin/env python
"""
Run Binance PnL analysis for a specified date range.

This script loads Binance fills and position snapshots, merges them,
and displays the results.

Usage:
    python run_binance_pnl.py                              # Run from PNL_START_DATE to today
    python run_binance_pnl.py --from=20241009              # Run from date to today
    python run_binance_pnl.py --from=20241009 --to=20241010  # Run for date range
"""
import argparse
import logging
import sys

from lib.pnl_new.binance_pnl import BinancePnl
from lib.util.time_util import today_date, date_str_to_dt, date_to_start_dt, date_to_end_dt
from lib.util.util import PNL_START_DATE
from lib.util.config import get_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Binance PnL analysis for a date range',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run from PNL_START_DATE to today
  %(prog)s --from=20241009              # Run from date to today
  %(prog)s --from=20241009 --to=20241010  # Run for date range
        """
    )
    parser.add_argument(
        '--from',
        dest='from',
        default=None,
        help='Start date in YYYYMMDD format (default: PNL_START_DATE)'
    )
    parser.add_argument(
        '--to',
        dest='to',
        default=None,
        help='End date in YYYYMMDD format (default: today)'
    )
    return vars(parser.parse_args())


def main():
    """Main entry point."""
    args = parse_args()

    _, config = get_config()

    # Determine date range
    if args['from'] is None:
        # No arguments - run from PNL_START_DATE to today
        start_date = PNL_START_DATE
        end_date = today_date()
        logger.info("No dates specified, running from PNL_START_DATE (%s) to today: %s", start_date, end_date)
    else:
        # Parse start date
        start_date = date_str_to_dt(args['from'])
        if args['to'] is None:
            # Only start specified - run until today
            end_date = today_date()
            logger.info("Start date specified, running from %s to today: %s", start_date, end_date)
        else:
            # Both start and end specified
            end_date = date_str_to_dt(args['to'])
            logger.info("Date range specified: %s to %s", start_date, end_date)

    # Convert dates to datetimes
    start_dt = date_to_start_dt(start_date)
    end_dt = date_to_end_dt(end_date)

    logger.info("="*80)
    logger.info("Binance PnL Analysis")
    logger.info("="*80)
    logger.info("Start: %s", start_dt)
    logger.info("End:   %s", end_dt)
    logger.info("="*80)

    # Create BinancePnl instance
    try:
        binance_pnl = BinancePnl(config=config, start=start_dt, end=end_dt)
    except Exception as e:
        logger.error("Failed to create BinancePnl: %s", e, exc_info=True)
        return 1

    # Aggregate to daily resolution
    try:
        daily_df = binance_pnl.aggregate_by_security_date()
    except Exception as e:
        logger.error("Failed to aggregate daily: %s", e, exc_info=True)
        return 1

    if daily_df is None:
        logger.warning("No daily aggregated data available")
        return 0

    print("="*80)
    print("Daily Data by Symbol")
    print("="*80)
    print(daily_df)
    print()

    # Aggregate daily data to portfolio level
    try:
        portfolio_df = binance_pnl.aggregate_daily_portfolio()
    except Exception as e:
        logger.error("Failed to aggregate portfolio: %s", e, exc_info=True)
        return 1

    if portfolio_df is None:
        logger.warning("No portfolio aggregated data available")
        return 0

    # Display portfolio data
    print("="*80)
    print("Portfolio-Level Daily Data")
    print("="*80)
    print(portfolio_df)
    print()

    portfolio_df.to_csv("binance_pnl_portfolio.csv")

    return 0


if __name__ == '__main__':
    sys.exit(main())
