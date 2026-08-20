import argparse
import glob
import logging.config
import os

from lib.util.config import get_config
from lib.data.data_news import load_raw_news
from lib.util.time_util import date_str_to_date, today_date
from lib.util.directory import dir_manager
from lib.universe import Universe
from lib.util.logging_util import get_logging_config

logging.config.dictConfig(get_logging_config("process_news"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process news files')
    parser.add_argument('-c', '--config', help='config file', required=False)
    parser.add_argument('-d', '--debug', dest='debug', action='store_true', required=False)
    parser.add_argument('-l', '--regen', dest='regen', action='store_true', required=False)
    parser.add_argument('-t', '--similarity-threshold', dest='similarity_threshold', type=float, required=False, default=None)
    parser.set_defaults(debug=False, regen=False)
    args = vars(parser.parse_args())

    config_file, config = get_config(args.get('config'))
    regen = args['regen']
    debug = args['debug']
    similarity_threshold = args['similarity_threshold'] if args['similarity_threshold'] is not None else config['NEWS_SIMILARITY_THRESHOLD']

    universe = Universe(config=config).load_universe_symbols(universe_source='file', symbol_type='pair', filter='fittable')

    news_files = glob.glob(f"{dir_manager.NEWS_DIR}/*.csv")
    for news_file in sorted(news_files):
        # News is processed during the daily update (at day start)
        # Skip processing today's news since it just starts collecting and is incomplete
        # If we process now, we won't process today's news again tomorrow, then the data will be left incomplete
        file_date = date_str_to_date(news_file.split('.')[1])
        if file_date == today_date():
            continue
        processed_file = news_file[:-4] + ".parquet"
        if not regen and os.path.exists(processed_file):
            continue

        logger.info(f"Processing {news_file}")
        try:
            df = load_raw_news(similarity_threshold=similarity_threshold, news_file=news_file, universe=universe)
        except ValueError as ve:
            logger.warning(ve)
            continue
        if debug:
            print(df)
        else:
            df.to_parquet(processed_file)
