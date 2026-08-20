import argparse
import os
import logging.config
import shutil
import sys
from datetime import timedelta as td

from lib.util.config import get_config, override_config_value
from lib.data import find_latest_alpha_file_date
from lib.alpha.forecasts import Forecasts
from lib.util.slack import JobMonitor
from lib.util.time_util import date_str_to_date, date_to_str, yesterday_date
from lib.util.directory import dir_manager
from lib.util.logging_util import get_logging_config

logging.config.dictConfig(get_logging_config("generate_alphas"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Data For input into simulation')
    parser.add_argument('-f', '--from', help='from date YYYYMMDD', required=False, type=int)
    parser.add_argument('-t', '--to', help='to date YYYYMMDD', required=False, type=int)
    parser.add_argument('-u', '--update', help='update latest alpahs', dest='update', action='store_true', required=False)
    parser.add_argument('-m', '--models', help='models', required=False, type=str)
    parser.add_argument('-c', '--config', help='config file', required=False)
    parser.add_argument('-z', '--horizons', help='horizons', required=False, type=str)
    parser.add_argument('-k', '--chunk-days', help='days to chunk at a time', required=False, type=int, default=None)
    parser.add_argument('-p', '--pool-size', help='pool size', required=False, type=int, default=None)
    parser.add_argument('-i', '--fit-file', help='fit file to use', required=False, type=str, default=None)
    parser.add_argument('-s', '--output-dir', help='output dir to dump alphas', required=False, type=str, default=None)
    parser.add_argument('-fd', '--fits-dir', help='fits dir', required=False, type=str, default=None)
    parser.add_argument('-d', '--debug', dest='debug', action='store_true', required=False)
    parser.add_argument('-g', '--regen', dest='regenerate', action='store_true')
    parser.add_argument('-r', '--prod', dest='prod', action='store_true', required=False)
    parser.add_argument('-nc', '--notify', help='slack channel to notify', required=False, type=str, default=None)
    parser.add_argument('-q', '--quiet', help='suppress slack notifications', action='store_true', required=False)
    parser.add_argument('--config-override', help='Override config values (KEY=VAL)', action='append', default=[])
    parser.set_defaults(update=False, debug=False, prod=False, regenerate=False, quiet=False)
    args = vars(parser.parse_args())

    config_file, config = get_config(args.get('config'))
    
    # Apply config overrides
    for override in args.get('config_override', []):
        if '=' not in override:
            logger.error(f"Invalid config override format: {override}. Expected KEY=VALUE")
            sys.exit(1)
        
        key, value = override.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        try:
            config = override_config_value(config, key, value)
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to apply config override '{override}': {e}")
            sys.exit(1)
    update = args['update']
    prod = args['prod']
    debug = args['debug']

    if prod:
        chunk_days = args['chunk_days'] if args['chunk_days'] is not None else 30
        pool_size = args['pool_size'] if args['pool_size'] is not None else 1
    else:
        chunk_days = args['chunk_days'] if args['chunk_days'] is not None else 90
        pool_size = args['pool_size']
    fit_file = args.get('fit_file')
    fits_dir = args['fits_dir']
    regenerate = args['regenerate']
    notify_channel = args['notify']
    quiet = args['quiet']

    if args['output_dir'] is not None:
        output_dir = args['output_dir']
    elif prod:
        output_dir = dir_manager.ALPHA_DIR_PROD
    else:
        output_dir = dir_manager.ALPHA_DIR_DEV

    horizons = args.get('horizons')
    if horizons is not None:
        horizons = [int(h) for h in horizons.split(',')]
        logger.info(f"Running for just horizons {horizons}")

    models = args.get('models')
    if models is not None:
        models = list(models.split(','))
        logger.info(f"Running for just models {models}")

    with JobMonitor("Generate alphas", notify_channel, debug, quiet):
        if update:
            if models is not None:
                logger.error("Cannot set models when doing update")
                sys.exit(1)
            if horizons is not None:
                logger.error("Cannot set horizons when doing update")
                sys.exit(1)

            end_date = yesterday_date()
            if args['from'] is None:
                latest_alpha_file_date = find_latest_alpha_file_date()
                start_date = latest_alpha_file_date + td(days=1) if latest_alpha_file_date is not None else end_date
            else:
                start_date = date_str_to_date(args['from'])
        else:
            if output_dir is None:
                logger.error("Need to set alpha-dir for non-prod alpha generation")
                sys.exit(1)
            start_date = date_str_to_date(args['from'])
            if args['to'] is not None:
                end_date = date_str_to_date(args['to'])
            else:
                end_date = yesterday_date()

        if start_date > end_date:
            logger.error("start date after end date, exiting...")
            sys.exit(1)

        logger.info(f"Generating alphas from {start_date} to {end_date} with {chunk_days=} and {pool_size=} using fits from {fit_file=}")

        if regenerate:
            # under new structure, we only need to move folders for the specific horizon/model when regenerate
            # Always use new directory structure for regeneration
            for horizon in horizons:
                for model in models:
                    alphas_dir = f"{output_dir}/{horizon}/{model}"
                    shutil.move(alphas_dir, f"{alphas_dir}.{date_to_str()}.bak")
                    os.mkdir(alphas_dir)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        Forecasts(
            config=config,
            prod=prod,
            debug=debug,
            horizons=horizons,
            models=models,
            output_dir=output_dir,
            fits_dir=fits_dir
        ).generate_rolling_alphas(
            start_date=start_date,
            end_date=end_date,
            chunk_days=chunk_days,
            pool_size=pool_size,
            fit_file=fit_file
        )
        logger.info(f"Done generating alphas from {start_date} to {end_date}")
