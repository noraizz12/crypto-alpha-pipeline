import argparse
import logging.config
import time

from slack_sdk.webhook.client import WebhookClient

from lib.pnl_new.pnl_monitor import PnlMonitorNew
from lib.util import today_date
from lib.util.logging_util import get_logging_config, KeyLogger
from lib.util.slack import SLACK_PNL_WEBHOOK
from lib.util.time_util import wait_until_minute
from lib.util.util import LOCAL
from lib.util.config import get_config

logging.config.dictConfig(get_logging_config("binance"))
original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor Pnl')
    parser.add_argument('-l', '--loop', help='loop every l minutes', required=False, type=int, default=8)
    parser.add_argument('-a', '--alert', help='alert no new fill after a minutes', required=False, type=int, default=60)
    parser.add_argument('-d', '--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    args = vars(parser.parse_args())

    loop = args['loop']
    alert_mins = args['alert']
    debug = args['debug']

    _, config = get_config()
    assert loop <= 15

    slack_client = WebhookClient(SLACK_PNL_WEBHOOK) if not LOCAL else None
    pnl_mon = PnlMonitorNew(config=config, debug=debug, alert_mins=alert_mins)
    last_date = today_date()
    while True:
        if not debug:
            logger.info(f"Waiting until {loop}min boundary")
            wait_until_minute(loop)

        msgs = pnl_mon.run()
        if loop == 0 or LOCAL:
            print(msgs)
            break
        else:
            slack_client.send(text=msgs)

        # wait until close to the next time
        sleep_time = 60 * loop * .7
        logger.info(f"Sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)

