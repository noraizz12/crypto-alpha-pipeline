import argparse
import logging.config

from lib.external.defi_llama import download_defillama_data
from lib.util.slack import JobMonitor
from lib.util.directory import dir_manager
from lib.util.logging_util import get_logging_config

logging.config.dictConfig(get_logging_config("download_defillama"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# DeFiLlama API endpoints

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch TVL and yield data from DeFiLlama')
    parser.add_argument('-l', '--limit', type=int, default=100, help='top N protocols to query')
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
    output_dir = args['output_dir'] if args['output_dir'] is not None else dir_manager.DEFI_DIR
    with JobMonitor("Download DeFiLlama", notify_channel, debug, quiet):
        download_defillama_data(limit, output_dir, debug)
