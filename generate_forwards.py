import argparse
import logging.config

from lib.util.time_util import date_str_to_date
from lib.fits.forwards import Forwards
from lib.util.slack import JobMonitor
from lib.util.logging_util import get_logging_config
from lib.util.config import get_config, extract_horizons

logging.config.dictConfig(get_logging_config("generate_forwards"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate forward data')
    parser.add_argument('-f', '--from', help='from date YYYYMMDD', required=False, type=int, default="20210614")
    parser.add_argument('-t', '--to', help='to date YYYYMMDD', required=False, type=int, default=None)
    parser.add_argument('-u', '--update', help='use existing bar file and update', dest='update', action='store_true', required=False)
    parser.add_argument('-z', '--horizons', help='horizon to run', required=False, type=str, default=None)
    parser.add_argument('-vg', '--overwrite', help='overwrite', dest='overwrite', action='store_true', required=False)
    parser.add_argument('-c', '--config', help='config file', required=False)
    parser.add_argument('-d', '--debug', dest='debug', action='store_true', required=False)
    parser.add_argument('-o', '--output-dir', help='output directory', required=False, type=str, default=None)
    parser.add_argument('-nc', '--notify', help='slack channel to notify', required=False, type=str, default=None)
    parser.add_argument('-q', '--quiet', help='suppress slack notifications', action='store_true', required=False)
    parser.set_defaults(update=False, debug=False, overwrite=False, quiet=False)
    args = vars(parser.parse_args())

    config_file, config = get_config(args.get('config'))
    overwrite = args['overwrite']
    update = args['update']
    debug = args['debug']
    output_dir = args.get('output_dir')
    notify_channel = args['notify']
    quiet = args['quiet']
    horizons = args.get('horizons')
    if horizons is not None:
        horizons = [int(h) for h in horizons.split(',')]
    else:
        horizons = extract_horizons(config)

    with JobMonitor("Generate forwards", notify_channel, debug, quiet):
        if update:
            start_date = end_date = None
        else:
            start_date = date_str_to_date(args['from'])
            end_date = date_str_to_date(args['to'])

        Forwards(
            config=config,
            debug=debug,
            update=update,
            horizons=horizons,
            output_dir=output_dir
        ).generate_forwards(start_date, end_date)
