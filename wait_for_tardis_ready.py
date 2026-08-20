"""Wait for Tardis data to be available for yesterday's date.

This script polls the Tardis API until data for yesterday is ready,
then exits successfully. Used by update.sh to start the pipeline
as soon as data is available (typically around 3 AM UTC).
"""
import argparse
import logging.config
import sys
import time
from datetime import date, timedelta
from typing import Optional

import requests

from lib.util.logging_util import get_logging_config

logging.config.dictConfig(get_logging_config("wait_for_tardis_ready"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Tardis API endpoint
TARDIS_API_URL = "https://api.tardis.dev/v1/exchanges/binance-futures"

# Poll interval in seconds (5 minutes)
POLL_INTERVAL = 300

# Maximum wait time in seconds (6 hours) - fail if data not ready after this
MAX_WAIT_TIME = 6 * 60 * 60


def get_exported_until() -> Optional[date]:
    """Fetch exportedUntil date from Tardis API.

    Returns:
        The exportedUntil date, or None if API call fails
    """
    try:
        response = requests.get(TARDIS_API_URL, timeout=30)
        response.raise_for_status()
        data = response.json()

        datasets = data.get('datasets')
        if datasets is None:
            logger.error("No 'datasets' field in API response")
            return None

        exported_until = datasets.get('exportedUntil')
        if exported_until is None:
            logger.error("No 'exportedUntil' field in datasets")
            return None

        # Parse the date (format: 2025-12-10T00:00:00.000Z)
        exported_date_str = exported_until.split('T')[0]
        exported_date = date.fromisoformat(exported_date_str)
        return exported_date

    except requests.RequestException:
        logger.exception("API request failed")
        return None
    except (ValueError, KeyError):
        logger.exception("Failed to parse API response")
        return None


def is_data_ready(target_date: Optional[date] = None) -> bool:
    """Check if Tardis data for target date is available.

    Args:
        target_date: The date we need data for (default: yesterday)

    Returns:
        True if data is ready, False otherwise
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    exported_date = get_exported_until()

    if exported_date is None:
        logger.warning("Could not get exportedUntil from API, assuming not ready")
        return False

    # Data is ready if exportedUntil > target_date
    # e.g., exportedUntil = Dec 10 means Dec 9 data is ready
    data_ready = exported_date > target_date

    logger.info(f"Target date: {target_date}, exportedUntil: {exported_date}, ready: {data_ready}")
    return data_ready


def wait_for_data(target_date: Optional[date] = None, poll_interval: int = POLL_INTERVAL,
                  max_wait_time: int = MAX_WAIT_TIME) -> bool:
    """Poll until data is ready or max wait time exceeded.

    Args:
        target_date: The date we need data for (default: yesterday)
        poll_interval: Seconds between API checks
        max_wait_time: Maximum seconds to wait before giving up

    Returns:
        True if data became ready, False if timeout
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    logger.info(f"Waiting for Tardis data for {target_date}...")

    start_time = time.time()
    check_count = 0

    while True:
        check_count += 1
        elapsed = time.time() - start_time

        if is_data_ready(target_date):
            logger.info(f"Tardis data is ready! (after {check_count} checks, {elapsed:.0f} seconds)")
            logger.info("Waiting 2 minutes buffer before starting pipeline...")
            time.sleep(120)  # 2 minute buffer
            return True

        if elapsed >= max_wait_time:
            logger.error(f"Timeout waiting for Tardis data after {elapsed:.0f} seconds")
            return False

        remaining = max_wait_time - elapsed
        logger.info(f"Data not ready, waiting {poll_interval} seconds... "
                    f"(check #{check_count}, {remaining:.0f}s remaining)")
        time.sleep(poll_interval)


def main() -> int:
    """Main entry point.

    Returns:
        0 if data is ready, 1 if timeout or error
    """
    parser = argparse.ArgumentParser(description='Wait for Tardis data to be ready')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='Debug mode (poll every 10 seconds)')
    parser.add_argument('--test-date', dest='test_date', type=str, default=None,
                        help='Test with specific date (YYYY-MM-DD) instead of yesterday')
    parser.add_argument('--max-wait', dest='max_wait', type=int, default=MAX_WAIT_TIME,
                        help=f'Maximum wait time in seconds (default: {MAX_WAIT_TIME})')
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    # Determine poll interval
    poll_interval = 10 if args.debug else POLL_INTERVAL

    # Determine target date
    target_date = None
    if args.test_date:
        try:
            target_date = date.fromisoformat(args.test_date)
            logger.info(f"Using test date: {target_date}")
        except ValueError:
            logger.error(f"Invalid date format: {args.test_date}. Use YYYY-MM-DD")
            return 1

    # Wait for data
    success = wait_for_data(
        target_date=target_date,
        poll_interval=poll_interval,
        max_wait_time=args.max_wait
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
