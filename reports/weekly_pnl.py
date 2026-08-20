"""Weekly PnL Report Script.

Generates weekly/daily PnL reports for investor letters.
Default week boundaries are Monday 00:00 UTC to Monday 00:00 UTC.
Use --friday for Friday 00:00 UTC to Friday 00:00 UTC boundaries (for investor emails).

Usage:
    python weekly_pnl.py                              # Last complete Mon-Mon week
    python weekly_pnl.py --friday                     # Last complete Fri-Fri week
    python weekly_pnl.py --friday --capital 2.5e6    # Fri-Fri with capital return
    python weekly_pnl.py --week-of 20260102           # Week containing date (Mon-Mon)
    python weekly_pnl.py --week-of 20260102 --friday  # Week containing date (Fri-Fri)
    python weekly_pnl.py --date 20260109              # Single day report
    python weekly_pnl.py --start 20260101 --end 20260109  # Custom range
"""
import argparse
import logging
import sys

from lib.reports.pnl_report_util import WeeklyPnlReport
from lib.util.config import get_config
from lib.util.time_util import date_str_to_dt

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate PnL report (weekly/daily/custom range)')
    parser.add_argument('--week-of', type=str, help='Date (YYYYMMDD) within the desired week')
    parser.add_argument('--date', type=str, help='Single date (YYYYMMDD) for daily report')
    parser.add_argument('--start', type=str, help='Start date (YYYYMMDD) for custom range')
    parser.add_argument('--end', type=str, help='End date (YYYYMMDD) for custom range')
    parser.add_argument('--friday', action='store_true',
                        help='Use Friday-Friday week boundaries (default is Monday-Monday)')
    parser.add_argument('--capital', type=float, default=None,
                        help='Total investor capital for return calculation (e.g., 2.5e6)')
    args = parser.parse_args()

    _, config = get_config()

    # Determine report type and create appropriate report
    if args.start and args.end:
        start_dt = date_str_to_dt(args.start)
        end_dt = date_str_to_dt(args.end)
        if start_dt >= end_dt:
            logger.error("--start must be before --end")
            sys.exit(1)
        report = WeeklyPnlReport.for_range(config, start_dt, end_dt, capital=args.capital)

    elif args.start or args.end:
        logger.error("--start and --end must be used together")
        sys.exit(1)

    elif args.date:
        report_date = date_str_to_dt(args.date)
        report = WeeklyPnlReport.for_date(config, report_date, capital=args.capital)

    elif args.week_of:
        reference_date = date_str_to_dt(args.week_of)
        report = WeeklyPnlReport.for_week(config, reference_date,
                                          friday_week=args.friday, capital=args.capital)

    else:
        report = WeeklyPnlReport.for_week(config, friday_week=args.friday, capital=args.capital)

    report.print_report()
    logger.info("Report completed")


if __name__ == "__main__":
    main()
