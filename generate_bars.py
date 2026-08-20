import argparse
import logging.config
import sys
from datetime import timedelta as td

from lib.bars.bar_generator import BarGenerator
from lib.data import find_latest_barfile_date
from lib.universe import Universe
from lib.util.config import get_config, extract_horizons
from lib.util.logging_util import get_logging_config
from lib.util.slack import JobMonitor
from lib.util.time_util import yesterday_date, today_date, date_str_to_date
from lib.util.util import TARDIS_EXCHANGE

logging.config.dictConfig(get_logging_config("generate_bars"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BARS_BACKFILL_START_DATE = date_str_to_date("20230101")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Data For input into simulation')
    parser.add_argument('-f', '--from', help='from date YYYYMMDD', required=False, type=int, default="20210614")
    parser.add_argument('-t', '--to', help='to date YYYYMMDD', required=False, type=int, default=None)
    parser.add_argument('-u', '--update', help='use existing bar file and update', dest='update', action='store_true', required=False)
    parser.add_argument('-r', '--force', help='create bars even if exist', action='store_true', required=False)
    parser.add_argument('-v', '--venue', help='venue to restrict to', required=False, default=TARDIS_EXCHANGE)
    parser.add_argument('-z', '--horizons', help='horizons', required=False, type=str)
    parser.add_argument('-c', '--config', help='config file', required=False, default=None)
    parser.add_argument('-p', '--pool-size', help='pool size', required=False, type=int, default=6)
    parser.add_argument('-d', '--debug', dest='debug', action='store_true', required=False)
    parser.add_argument('-bf', '--backfill', dest='backfill', action='store_true', required=False)
    parser.add_argument('-i', '--input-dir', help='input directory', required=False, default=None)
    parser.add_argument('-o', '--output-dir', help='output directory', required=False, default=None)
    parser.add_argument('-nc', '--notify', help='slack channel to notify', required=False, type=str, default=None)
    parser.add_argument('-q', '--quiet', help='suppress slack notifications', action='store_true', required=False)
    parser.set_defaults(update=False, debug=False, force=False, backfill=False, quiet=False)
    args = vars(parser.parse_args())

    config_file, config = get_config(args.get('config'))
    start_date = end_date = None
    venue = args['venue']
    debug = args['debug']
    force = args['force']
    update = args['update']
    backfill = args['backfill']
    pool_size = args['pool_size']
    input_dir = args['input_dir']
    output_dir = args['output_dir']
    notify_channel = args['notify']
    quiet = args['quiet']
    CHUNK_DAYS = 30
    horizons = args.get('horizons')
    if horizons is not None:
        horizons = [int(h) for h in horizons.split(',')]
    else:
        horizons = extract_horizons(config)
        horizons.append(1)

    with JobMonitor("Generate bars", notify_channel, debug, quiet):
        if backfill:
            universe = Universe(config=config)
            new_symbols = universe.get_todays_new_symbols(check_bars=True)
            if len(new_symbols) > 0:
                logger.info(f"Backfilling bars for {new_symbols} from {BARS_BACKFILL_START_DATE} to {yesterday_date()}")
                # generate bars only for the new symbols
                BarGenerator(
                    config=config,
                    venue=venue,
                    start_date=BARS_BACKFILL_START_DATE,
                    end_date=yesterday_date(),
                    chunk_days=CHUNK_DAYS,
                    horizons=horizons,
                    debug=debug,
                    pool_size=pool_size,
                    backfill=True,
                    output_dir=output_dir,
                    symbols=new_symbols,
                ).run()
            logger.info(f"Done backfilling bars from {BARS_BACKFILL_START_DATE} to {yesterday_date()} for {new_symbols}")
            sys.exit(0)

        if update:
            end_date = yesterday_date()
            if force:
                start_date = end_date
            else:
                start_date = find_latest_barfile_date() + td(days=1)
            logger.info(f"Updating bar files from {start_date} to {end_date}")
            if start_date == today_date():
                logger.warning(f"Bar files already generated to {start_date}...")
                sys.exit(1)
        else:
            start_date = date_str_to_date(args['from'])
            if args['to'] is not None:
                end_date = date_str_to_date(args['to'])
            else:
                end_date = today_date() - td(days=1)
            if start_date > end_date:
                logger.error("start date after end date, exiting...")
                sys.exit(1)

        BarGenerator(
            config=config,
            venue=venue,
            start_date=start_date,
            end_date=end_date,
            chunk_days=CHUNK_DAYS,
            horizons=horizons,
            debug=debug,
            pool_size=pool_size,
            output_dir=output_dir,
        ).run()
        logger.info(f"Done generating bars from {start_date} to {end_date}")
