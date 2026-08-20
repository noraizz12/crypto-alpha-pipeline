import argparse
import logging.config
import sys
from datetime import timedelta as td

import pandas as pd

from lib.data import find_latest_model_file_date
from lib.alpha.model_calcs import ModelCalcs
from lib.util.config import extract_horizons, extract_models, get_config
from lib.util.directory import dir_manager
from lib.util.logging_util import get_logging_config
from lib.util.slack import JobMonitor
from lib.util.time_util import date_str_to_date, today_date, yesterday_date

pd.options.display.max_rows = 500

logging.config.dictConfig(get_logging_config("generate_models"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Data For input into simulation')
    parser.add_argument('-f', '--from', help='from date YYYYMMDD', required=False, type=int)
    parser.add_argument('-t', '--to', help='to date YYYYMMDD', required=False, type=int)
    parser.add_argument('-u', '--update', help='use existing bar file and update', dest='update', action='store_true', required=False)
    parser.add_argument('-s', '--output-dir', help='directory to dump data', required=False, default=None)
    parser.add_argument('-z', '--horizons', help='horizons', required=False, type=str)
    parser.add_argument('-m', '--models', help='models', required=False, type=str)
    parser.add_argument('-c', '--config', help='config file', required=False)
    parser.add_argument('-p', '--pool-size', help='pool size', required=False, type=int, default=6)
    parser.add_argument('-d', '--debug', dest='debug', action='store_true', required=False)
    parser.add_argument('-nc', '--notify', help='slack channel to notify', required=False, type=str, default=None)
    parser.add_argument('-q', '--quiet', help='suppress slack notifications', action='store_true', required=False)
    parser.set_defaults(update=False, debug=False, quiet=False)
    args = vars(parser.parse_args())

    config_file, config = get_config(args.get('config'))
    pool_size = args['pool_size']
    update = args['update']
    debug = args['debug']
    output_dir = args['output_dir']
    output_dir = args['output_dir'] if args['output_dir'] is not None else dir_manager.MODELS_DIR

    horizons = args.get('horizons')
    horizons = [int(h) for h in horizons.split(",")] if horizons is not None else extract_horizons(config)

    models = args.get('models')
    models = list(models.split(",")) if models is not None else extract_models(config, horizons=horizons)
    notify_channel = args['notify']
    quiet = args['quiet']

    with JobMonitor("Generate models", notify_channel, debug, quiet):
        if update:
            end_date = yesterday_date()
            for horizon in horizons:
                for model in models:
                    latest_model_file_date = find_latest_model_file_date(models_dir=output_dir, horizon=horizon, model=model)
                    start_date = latest_model_file_date + td(days=1) if latest_model_file_date is not None else end_date - td(days=5)
                    if start_date == today_date():
                        logger.warning(f"{horizon} {model} model files already generated to {start_date}, moving to next horizon model...")
                        continue
                    elif start_date > end_date:
                        logger.error(f"start date after end date for {horizon} {model} model, moving to next horizon model...")
                        continue
                    logger.info(f"Generating {horizon} {model} model from {start_date} to {end_date} with {pool_size=}")
                    model_calcs = ModelCalcs(
                        config=config,
                        horizons=[horizon],
                        models_to_run=[model],
                        debug=debug,
                        pool_size=pool_size,
                        output_dir=output_dir
                    ).process_models(start_date=start_date, end_date=end_date)

        else:
            start_date = date_str_to_date(args['from'])
            end_date = date_str_to_date(args["to"]) if args["to"] is not None else today_date() - td(days=1)

            if start_date is None or start_date > end_date:
                logger.error(f"wrong start date {start_date}, end date {end_date}, exiting...")
                sys.exit(1)

            ModelCalcs(
                config=config,
                horizons=horizons,
                models_to_run=models,
                debug=debug,
                pool_size=pool_size,
                output_dir=output_dir
            ).process_models(start_date=start_date, end_date=end_date)
        logger.info(f"Done generating models from {start_date} to {end_date}")
