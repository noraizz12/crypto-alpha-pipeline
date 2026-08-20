import argparse
import asyncio
import logging.config
import os
import sys
import threading

from slack_sdk.webhook.client import WebhookClient

from lib.trader import Trader
from lib.util.directory import dir_manager
from lib.util.slack import SLACK_WEBHOOK
from lib.util.util import LOCAL
from lib.util.logging_util import get_logging_config, LOG_FORMAT
from lib.util.config import get_config
from lib.util.opsgenie import raise_alert, HIGH

KILL_FILE = f"{dir_manager.TRADING_DIR}/trader.kill"


def check_kill_file(logger_instance):
    if os.path.isfile(KILL_FILE):
        try:
            os.remove(KILL_FILE)
            logger_instance.info(f"{KILL_FILE=} found and removed, shutting down trader...")
            os._exit(0)
        except Exception as e:
            logger_instance.error(f"{KILL_FILE=} found but failed to remove and shut down due to {e}")

    timer = threading.Timer(50, check_kill_file, args=[logger_instance])
    timer.daemon = True
    timer.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--mode', help='mode', required=False, type=str)
    parser.add_argument('-a', '--aggression', help='aggression', required=False, type=int, default=0)
    parser.add_argument('-d', '--debug', dest='debug', action='store_true')
    parser.add_argument('-p', '--paper', dest='paper_trade', action='store_true')
    parser.add_argument('-no', '--old-oms', dest='old_oms', action='store_true')
    parser.add_argument('-md', '--market-data-only', dest='market_data_only', action='store_true')
    parser.add_argument('-n', '--no-trade', dest='no_trade', action='store_true')
    parser.add_argument('-nl', '--no-limits', dest='no_limits', action='store_true')
    parser.add_argument('-c', '--config', help='config file', required=False)
    parser.add_argument('-o', '--order', help='order string', required=False, type=str)
    parser.add_argument('--eod', dest='eod', action='store_true', help='enable EOD reversal overlay')
    parser.add_argument('--local-oms', dest='local_oms', action='store_true', help='bypass ZMQ, send orders directly to Binance')
    parser.set_defaults(debug=False, paper_trade=False, market_data_only=False, no_trade=False, no_limits=False, eod=False, local_oms=False, mode="")
    args = vars(parser.parse_args())

    config_file, config = get_config()
    debug = args['debug']
    paper_trade = args['paper_trade']
    old_oms = args['old_oms']
    local_oms = args['local_oms']
    market_data_only = args['market_data_only']
    no_trade = args['no_trade']
    no_limits = args['no_limits']
    eod = args['eod']
    mode = args['mode'].lower().strip()

    auto_mode = mode == ""

    if auto_mode:
        logging.config.dictConfig(get_logging_config("trader"))
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    slack_client = WebhookClient(SLACK_WEBHOOK) if not LOCAL else None

    # Start kill file checker for graceful shutdown (only in prod auto mode)
    if auto_mode and not LOCAL and not debug:
        check_kill_file(logger)

    trader = Trader(
        config=config,
        paper_trade=paper_trade,
        debug=debug,
        market_data_only=market_data_only,
        no_trade=no_trade,
        no_limits=no_limits,
        trade_on_startup=auto_mode,
        quote_only_mode=not auto_mode,
        slack_client=slack_client,
        old_oms=old_oms,
        eod=eod,
        local_oms=local_oms
    )

    if auto_mode:
        print("Running normal mode...")
        try:
            asyncio.run(trader.run_autotrader())
        except Exception as e:
            raise_alert(key=f"trader crash: {type(e).__name__}", priority=HIGH, description=f"{e}")
            raise
    elif mode == "cancel":
        trader.cancel_all()
    elif mode == "flatten":
        aggression = args['aggression']
        print(f"Flattening all positions with {aggression=}!")
        trader.flatten_all(aggression=aggression)
    elif mode == "manual":
        order = args['order']
        print(f"manual order mode for: {order}")
        order = trader.manual_order_from_str(order)
        print(f"Order sent: {order}")
    else:
        logger.info(f'Unknown mode: "{mode}"')

    asyncio.run(trader.shutdown())
