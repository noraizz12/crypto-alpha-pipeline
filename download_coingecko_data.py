import argparse
import logging.config

from lib.external.coingecko import CoinGecko
from lib.util.slack import JobMonitor
from lib.util.directory import dir_manager
from lib.util.logging_util import get_logging_config

logging.config.dictConfig(get_logging_config("download_coingecko"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch top cryptocurrencies by market cap from CoinGecko')
    parser.add_argument('-l', '--limit', type=int, default=500, help='top N symbols to query')
    parser.add_argument('-s', '--output-dir', help='directory to dump data', required=False, default=None)
    parser.add_argument('-d', '--debug', dest='debug', action='store_true', required=False)
    parser.add_argument('-nc', '--notify', help='slack channel to notify', required=False, type=str, default=None)
    parser.add_argument('-q', '--quiet', help='suppress slack notifications', action='store_true', required=False)
    parser.set_defaults(debug=False, quiet=False)
    args = vars(parser.parse_args())
    limit = args['limit']
    debug = args['debug']
    notify_channel = args['notify']
    quiet = args['quiet']
    output_dir = args['output_dir'] if args['output_dir'] is not None else dir_manager.MARKETCAP_DIR
    pages_needed = (limit + 99) // 100
    with JobMonitor("Download coingecko", notify_channel, debug, quiet):
        CoinGecko(limit=limit, pages_needed=pages_needed, output_dir=output_dir, debug=debug).download_coingecko_data()
