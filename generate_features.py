import argparse
import logging.config
from datetime import timedelta as td
import sys

from lib.util.config import extract_horizons, get_config
from lib.data import find_latest_feature_file_date
from lib.alpha.features import generate_rolling_features
from lib.util.slack import JobMonitor
from lib.util.time_util import date_str_to_date, today_date, yesterday_date
from lib.util.directory import dir_manager
from lib.util.logging_util import get_logging_config

logging.config.dictConfig(get_logging_config("generate_features"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Data For input into simulation')
    parser.add_argument('-f', '--from', help='from date YYYYMMDD', required=False, type=int)
    parser.add_argument('-t', '--to', help='to date YYYYMMDD', required=False, type=int)
    parser.add_argument('-u', '--update', help='use existing bar file and update', dest='update', action='store_true', required=False)
    parser.add_argument('-o', '--output-dir', help='directory to dump data', required=False, type=str, default=None)
    parser.add_argument('-c', '--config', help='config file', required=False, default=None)
    parser.add_argument('-z', '--horizons', help='horizons', required=False, type=str)
    parser.add_argument('-fs', '--features', help='selected features to generate', required=False, type=str)
    parser.add_argument('-k', '--chunk-days', help='days to chunk at a time', required=False, type=int, default=1)
    parser.add_argument('-p', '--pool-size', help='pool size', required=False, type=int, default=None)
    parser.add_argument('-d', '--debug', dest='debug', action='store_true', required=False)
    parser.add_argument('-nc', '--notify', help='slack channel to notify', required=False, type=str, default=None)
    parser.add_argument('-q', '--quiet', help='suppress slack notifications', action='store_true', required=False)
    parser.set_defaults(update=False, debug=False, quiet=False)
    args = vars(parser.parse_args())

    config_file, config = get_config(args.get('config'))
    update = args['update']
    debug = args['debug']
    output_dir = args['output_dir'] if args['output_dir'] is not None else dir_manager.FEATURES_DIR

    pool_size = args['pool_size']
    chunk_days = args['chunk_days']
    notify_channel = args['notify']
    quiet = args['quiet']

    horizons = args.get('horizons')
    horizons = [int(h) for h in horizons.split(",")] if horizons is not None else extract_horizons(config)
    # if features is None, we will generate all features, otherwise only selected features
    features = args.get('features')
    features = list(features.split(",")) if features is not None else None

    with JobMonitor("Generate features", notify_channel, debug, quiet):
        start_date = None
        if update:
            end_date = yesterday_date()
            for horizon in sorted(horizons):
                latest_feature_file_date = find_latest_feature_file_date(features_dir=output_dir, horizon=horizon)
                start_date = latest_feature_file_date + td(days=1) if latest_feature_file_date is not None else end_date - td(days=5)
                logger.info(f"Updating {horizon} feature files from {start_date} to {end_date}")
                if start_date == today_date():
                    logger.warning(f"{horizon} feature files already generated to {start_date}, moving to next horizon...")
                    continue
                elif start_date > end_date:
                    logger.error(f"start date after end date for {horizon} feature, moving to next horizon...")
                    continue
                logger.info(f"Generating {horizon} features from {start_date} to {end_date} with {chunk_days=} and {pool_size=}")
                generate_rolling_features(
                    config=config,
                    pool_size=pool_size,
                    chunk_days=chunk_days,
                    debug=debug,
                    start_date=start_date,
                    end_date=end_date,
                    frequencies=[horizon],
                    output_dir=output_dir,
                    features=features
                )
        else:
            start_date = date_str_to_date(args['from'])
            end_date = date_str_to_date(args["to"]) if args["to"] is not None else yesterday_date()

            if start_date is None or start_date > end_date:
                logger.error(f"wrong start date {start_date}, end date {end_date}, exiting...")
                sys.exit(1)

            logger.info(f"Generating features from {start_date} to {end_date} with {chunk_days=} and {pool_size=}")
            generate_rolling_features(
                config=config,
                pool_size=pool_size,
                chunk_days=chunk_days,
                debug=debug,
                start_date=start_date,
                end_date=end_date,
                frequencies=horizons,
                output_dir=output_dir,
                features=features
            )
        logger.info(f"Done generating features from {start_date} to {end_date}")
