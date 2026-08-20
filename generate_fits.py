import argparse
import logging.config
import os
import shutil
import sys
import time

from lib.util.config import extract_horizons, get_config
from lib.fits.fits import Fits
from lib.util.slack import JobMonitor
from lib.util.time_util import date_str_to_date, date_to_str, today_date
from lib.util.directory import dir_manager
from lib.util.logging_util import get_logging_config

logging.config.dictConfig(get_logging_config("generate_fits"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Data For input into simulation')
    parser.add_argument('-f', '--from', help='from date YYYYMMDD', required=False, type=int)
    parser.add_argument('-t', '--to', help='to date YYYYMMDD', required=False, type=int)
    parser.add_argument('-u', '--update', help='update latest fits', dest='update', action='store_true', required=False)
    parser.add_argument('-pr', '--prod', help='generate prod fits', dest='prod', action='store_true', required=False)
    parser.add_argument('-r', '--force', help='produce prod fit regardless of existing file', dest='force', action='store_true', required=False)
    parser.add_argument('-o', '--output-dir', help='directory to dump data for fit', required=False, type=str, default=None)
    parser.add_argument('-c', '--config', help='config file', required=False)
    parser.add_argument('-z', '--horizons', help='horizons', required=False, type=str)
    parser.add_argument('-m', '--models', help='models', required=False, type=str)
    parser.add_argument('-p', '--pool-size', help='pool size', required=False, type=int, default=3)
    parser.add_argument('-ld', '--classification-days', help='classification days', required=False, type=int, default=None)
    parser.add_argument('-sf', '--security-fit', dest='security_fit', action='store_true')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true')
    parser.add_argument('-g', '--regen', dest='regenerate', action='store_true')
    parser.add_argument('-nc', '--notify', help='slack channel to notify', required=False, type=str, default=None)
    parser.add_argument('-q', '--quiet', help='suppress slack notifications', action='store_true', required=False)
    parser.set_defaults(update=False, prod=False, erase=False, debug=False, all_perps=False, force=False, regenerate=False, security_fit=False, quiet=False)
    args = vars(parser.parse_args())

    update = args['update']
    prod = args['prod']
    pool_size = args['pool_size']
    security_fit = args['security_fit']
    debug = args['debug']
    force = args['force']
    regenerate = args['regenerate']
    classification_days = args['classification_days']
    notify_channel = args['notify']
    quiet = args['quiet']
    if args['output_dir'] is not None:
        base_fits_dir = args['output_dir']
    elif prod:
        base_fits_dir = dir_manager.FITS_DIR_PROD
    else:
        base_fits_dir = dir_manager.FITS_DIR_DEV

    config_file, config = get_config(args.get('config'))

    with JobMonitor("Generate fits", notify_channel, debug, quiet):
        if update:
            start_date = end_date = today_date()
        else:
            if not regenerate and not debug:
                logger.warning("Sure you dont want to regenerate? Regenerate will keep a copy of current fit, otherwise data will generated in the current fit folder")
                time.sleep(20)
            start_date = date_str_to_date(args['from'])
            end_date = date_str_to_date(args['to'])

            if start_date > end_date:
                logger.error("start date after end date, exiting...")
                sys.exit(1)

        horizons = args.get('horizons')
        if horizons is not None:
            horizons = [int(h) for h in horizons.split(',')]
        else:
            horizons = extract_horizons(config)

        models = args.get('models')
        if models is not None:
            models = list(models.split(','))

        if regenerate and not debug:
            # under new structure, we only need to move folders for the specific horizon/model when regenerate
            for horizon in horizons:
                for model in models:
                    fits_dir = f"{base_fits_dir}/{horizon}/{model}"
                    shutil.move(fits_dir, f"{fits_dir}.{date_to_str()}.bak")
                    os.mkdir(fits_dir)

        if not debug and (prod or regenerate):
            shutil.copy(config_file, f"{base_fits_dir}/config.{date_to_str()}.json")

        fits = Fits(
            config=config,
            horizons=horizons,
            models=models,
            pool_size=pool_size,
            classification_history_days=classification_days,
            debug=debug,
            prod=prod,
            force=force,
            regenerate=regenerate,
            base_fits_dir=base_fits_dir
        ).generate_rolling_fits(start_date=start_date, end_date=end_date)
