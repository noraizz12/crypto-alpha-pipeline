import argparse
import logging.config
import time

from slack_sdk.webhook.client import WebhookClient

from lib.external.download_binance import DownloadBinance
from lib.util.logging_util import get_logging_config, KeyLogger
from lib.util.slack import SLACK_PNL_WEBHOOK
from lib.util.time_util import seconds_until_next_utc_midnight
from lib.util.util import LOCAL

logging.config.dictConfig(get_logging_config("binance"))
original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor Binance data')
    parser.add_argument('-l', '--loop', help='loop every l minutes', required=False, type=int, default=0)
    parser.add_argument('-d', '--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    args = vars(parser.parse_args())

    loop = args['loop']
    debug = args['debug']

    slack_client = WebhookClient(SLACK_PNL_WEBHOOK) if not LOCAL else None
    binance = DownloadBinance(debug=debug)

    while True:
        binance.run()

        # Calculate time until next midnight and next loop interval
        seconds_to_midnight = seconds_until_next_utc_midnight()
        sleep_time = 60 * loop

        # Sleep until the earlier of: next loop interval or next midnight
        if seconds_to_midnight < sleep_time:
            logger.info(f"Sleeping for {seconds_to_midnight:.0f} seconds until UTC midnight")
            time.sleep(seconds_to_midnight)
        else:
            logger.info(f"Sleeping for {sleep_time} seconds (regular interval)")
            time.sleep(sleep_time)
