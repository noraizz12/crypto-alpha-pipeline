import argparse
import logging.config
from datetime import timedelta as td

from lib.reports.markouts import MomentumMarkouts
from lib.util.logging_util import get_logging_config
from lib.util.time_util import yesterday, date_str_to_dt

logging.config.dictConfig(get_logging_config("momentum"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Markout trades')
    parser.add_argument('-f', '--from', help='from date YYYYMMDD', required=False, type=int, default=None)
    parser.add_argument('-t', '--to', help='to date YYYYMMDD', required=False, type=int, default=None)
    parser.add_argument('-c', '--cooldown', dest='cooldown', action='store_true', required=False)
    parser.add_argument('-l', '--lookout', help='lookout mins', required=False, type=int)
    parser.add_argument('-r', '--filter', help='filter level', required=False, type=int)
    parser.set_defaults(cooldown=False, lookout=360, filter=6)
    args = vars(parser.parse_args())

    end = args.get('to')
    if end is not None:
        end_dt = date_str_to_dt(end)
    else:
        end_dt = yesterday()

    start = args.get('from')
    if start is not None:
        start_dt = date_str_to_dt(start)
    else:
        start_dt = end_dt - td(days=30)

    MomentumMarkouts(
        start_dt=start_dt,
        end_dt=end_dt,
        cooldown=args.get('cooldown'),
        lookout_mins=args.get('lookout'),
        logret_1440_lz_filter=args.get('filter'),
    ).run()
