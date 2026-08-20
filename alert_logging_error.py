import argparse
import os
import logging.config
from lib.util.directory import LOG_DIR
from lib.util.logging_util import process_log, get_logging_config

logging.config.dictConfig(get_logging_config("alert_errors"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Alert logging errors')
    parser.add_argument('-f', '--file', help='filename to alert errors', required=False, type=str, default=os.path.join(LOG_DIR, "trader.err"))
    args = vars(parser.parse_args())
    log_file = args['file']

    try:
        process_log(log_file)
    except Exception as e:
        logger.exception(f"Error in process_log function: {str(e)}")
