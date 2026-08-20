import argparse
import logging.config
import os
import sys
import threading
from datetime import timedelta as td

from slack_sdk.webhook.async_client import AsyncWebhookClient

from lib.server import AlphaServer
from lib.util.time_util import date_str_to_dt
from lib.util.slack import SLACK_WEBHOOK
from lib.util.directory import dir_manager
from lib.util.util import LOCAL
from lib.util.logging_util import get_logging_config, LOG_FORMAT
from lib.util.opsgenie import raise_alert, HIGH

logging.config.dictConfig(get_logging_config("server"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

KILL_FILE = f"{dir_manager.TRADING_DIR}/server.kill"


def check_kill_file():
    if os.path.isfile(KILL_FILE):
        try:
            os.remove(KILL_FILE)
            logger.info(f"{KILL_FILE=} found and removed, shutting down server...")
            os._exit(0)
        except Exception as e:
            logger.error(f"{KILL_FILE=} found but failed to remove and shut down due to {e}")

    timer = threading.Timer(50, check_kill_file)
    timer.daemon = True
    timer.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the target portfolio generator')
    parser.add_argument('-s', '--date', help='date YYYYMMDD', required=False, type=int)
    parser.add_argument('-c', '--config', help='config file', required=False)
    parser.add_argument('-d', '--debug', dest='debug', action='store_true')
    parser.add_argument('-t', '--trade-on-start', dest='trade_on_start', action='store_true')
    parser.add_argument('-p', '--no-existing-positions', dest='no_positions', action='store_true')
    parser.add_argument('--scale', type=float, default=None,
                        help='Scale factor for positions (0.0-1.0)')
    parser.set_defaults(debug=False, trade_on_start=False, no_positions=False)
    args = vars(parser.parse_args())

    run_date = args.get('date')
    config_file = args.get('config')
    debug = args['debug']
    trade_on_start = args['trade_on_start']
    no_positions = args['no_positions']
    scale = args['scale']
    if scale is not None and not 0.0 <= scale <= 1.0:
        parser.error("--scale must be between 0.0 and 1.0")
    if run_date is not None:
        debug = True
        run_date = date_str_to_dt(run_date)
        # the server expects midnight after the day
        run_date += td(days=1)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console)

    slack_client = AsyncWebhookClient(SLACK_WEBHOOK) if not LOCAL else None
    if not LOCAL and not debug:
        check_kill_file()

    server = AlphaServer(
        eod_today=run_date,
        config_file=config_file,
        debug=debug,
        slack_client=slack_client,
        trade_on_start=trade_on_start,
        no_positions=no_positions,
        new_bars=True,
        scale=scale
    )
    if debug:
        if not server.update(st_only=False):
            logger.error("Server update not successful!")
            sys.exit(1)
        server.generate_targets(optimize=True)
    else:
        try:
            server.loop()
        except Exception as e:
            raise_alert(key=f"server crash: {type(e).__name__}", priority=HIGH, description=f"{e}")
            raise
