import argparse
from datetime import timedelta as td
import logging.config

from lib.util.config import get_config
from lib.reports.markouts import FillMarkouts
from lib.util.time_util import yesterday, date_str_to_dt
from lib.util.logging_util import get_logging_config

logging.config.dictConfig(get_logging_config("markouts"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Markout trades')
    parser.add_argument('-f', '--from', help='from date YYYYMMDD', required=False, type=int, default=None)
    parser.add_argument('-t', '--to', help='to date YYYYMMDD', required=False, type=int, default=None)
    parser.add_argument('-c', '--config', help='config file', required=False)
    parser.add_argument('-s', '--save', dest='save', action='store_true', required=False)
    parser.set_defaults(update=False, debug=False, save=False)
    args = vars(parser.parse_args())

    _, config = get_config(args.get('config'))
    end = args.get('to')
    if end is not None:
        end_dt = date_str_to_dt(end)
    else:
        end_dt = yesterday()

    start = args.get('from')
    if start is not None:
        start_dt = date_str_to_dt(start)
    else:
        start_dt = end_dt - td(days=7)

    fill_markouts = FillMarkouts(config)
    fill_markouts.calculate_markouts(start=start_dt, end=end_dt, save=args.get('save'))
