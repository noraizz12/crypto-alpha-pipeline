import argparse
import logging.config
from datetime import datetime as dt
from datetime import timedelta as td

from lib.external.binance_utils import get_hist_trades
from lib.util.files import write_df_to_csv
from lib.util.directory import dir_manager
from lib.util.logging_util import get_logging_config
from lib.util.time_util import yesterday_date, date_to_str, date_str_to_date, date_range, date_to_start_dt

logging.config.dictConfig(get_logging_config("binance_fills"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def backfill_binance_fills(backfill_dt: dt):
    trades_df = get_hist_trades(start_dt=backfill_dt, end_dt=backfill_dt + td(days=1))

    if trades_df is not None:
        logger.info(f'backfill {backfill_dt} fills with {trades_df.shape} records')
        write_df_to_csv(trades_df, dir_manager.BINANCE_FILLS_DIR + f"/bfills.{date_to_str(backfill_dt)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor Binance data')
    parser.add_argument('-f', '--from', help='from date YYYYMMDD', required=False, type=int)
    parser.add_argument('-t', '--to', help='to date YYYYMMDD', required=False, type=int)
    args = vars(parser.parse_args())

    if args['from'] is not None:
        start_date = date_str_to_date(args['from'])
    else:
        start_date = yesterday_date()

    if args['to'] is not None:
        end_date = date_str_to_date(args['to'])
    else:
        end_date = yesterday_date()

    for date in date_range(start_date, end_date):
        backfill_binance_fills(date_to_start_dt(date))
