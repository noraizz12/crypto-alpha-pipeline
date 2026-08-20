import argparse
import logging.config

from lib.util.config import get_config
from lib.util.slack import JobMonitor
from lib.util.time_util import date_range, date_str_to_date, yesterday_date
from lib.universe import Universe
from lib.util.logging_util import get_logging_config

logging.config.dictConfig(get_logging_config("generate_universe"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate universe for system')
    parser.add_argument('-f', '--from', help='from date YYYYMMDD', required=False, type=int)
    parser.add_argument('-t', '--to', help='to date YYYYMMDD', required=False, type=int)
    parser.add_argument('-c', '--config', help='config file', required=False)
    parser.add_argument('-l', '--limit', type=int, default=None, help='top N symbols to query')
    parser.add_argument('-o', '--output-dir', help='directory to dump data', required=False, default=None)
    parser.add_argument('-d', '--debug', dest='debug', action='store_true', required=False)
    parser.add_argument('-p', '--prod', dest='prod', action='store_true', required=False)
    parser.add_argument('-u', '--update', help='update latest universe', dest='update', action='store_true', required=False)
    parser.add_argument('-nc', '--notify', help='slack channel to notify', required=False, type=str, default=None)
    parser.add_argument('-q', '--quiet', help='suppress slack notifications', action='store_true', required=False)
    parser.set_defaults(update=False, debug=False, all_perps=False, quiet=False)
    args = vars(parser.parse_args())

    limit = args['limit']
    debug = args['debug']
    update = args['update']
    prod = args['prod']
    notify_channel = args['notify']
    quiet = args['quiet']
    _, config = get_config(args.get('config'))
    output_dir = args['output_dir']

    if update:
        end_date = yesterday_date()
        start_date = end_date
    else:
        start_date = date_str_to_date(args['from'])
        end_date = date_str_to_date(args['to'])

    logger.info(f"Start generating universe from {start_date} to {end_date}  {debug=} {limit=}")
    with JobMonitor("Generate universe", notify_channel, debug, quiet):
        uni = Universe(config, debug=debug, top_n=limit, prod=prod)
        for universe_date in date_range(start_date, end_date):
            uni.generate_universe_file(universe_date=universe_date)
    logger.info(f"Done generating universe from {start_date} to {end_date}")
