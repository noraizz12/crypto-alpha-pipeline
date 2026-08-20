import argparse
import logging.config
from datetime import timedelta as td

from lib.pnl import FillBreakdown
from lib.util.time_util import date_str_to_dt, today
from lib.util.logging_util import get_logging_config

logging.config.dictConfig(get_logging_config("realized_breakdown"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Realized PnL Report')
    parser.add_argument('-f', '--from', help='from date YYYYMMDD', required=False, type=int)
    parser.add_argument('-t', '--to', help='to date YYYYMMDD', required=False, type=int)
    args = vars(parser.parse_args())

    if args['to'] is None:
        end_date = today()
    else:
        end_date = date_str_to_dt(args['to'])

    if args['from'] is None:
        start_date = end_date - td(days=2)
    else:
        start_date = date_str_to_dt(args['from'])

    if start_date > end_date:
        logger.error("start date after end date, exiting...")
        exit(1)

    FillBreakdown(start=start_date, end=end_date).runall()
