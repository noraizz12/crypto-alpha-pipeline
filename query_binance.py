import argparse
import logging.config

import pandas as pd

from lib.external.binance_utils import get_positions, get_balances, get_funding, get_hist_trades, get_financing_rates, get_exchange_info, get_account_info
from lib.util.files import write_df_to_csv
from lib.util.time_util import date_str_to_date, today_date
from lib.util.time_util import date_to_str, date_range, date_to_start_dt, date_to_end_dt
from lib.util.directory import dir_manager
from lib.util.logging_util import get_logging_config

logging.config.dictConfig(get_logging_config("query_binance"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitor Binance data')
    parser.add_argument('-m', '--mode', help='loop every l minutes', required=True, type=str)
    parser.add_argument('-f', '--from', help='from date YYYYMMDD', required=False, type=int)
    parser.add_argument('-t', '--to', help='to date YYYYMMDD', required=False, type=int)
    args = vars(parser.parse_args())

    mode = args['mode']

    pd.options.display.max_rows = 500

    if mode == "funding":
        df = get_funding()
        print(df['income'].sum())
        print(df)
    elif mode == "positions":
        positions_df = get_positions()
        print(positions_df)
        print(positions_df.iloc[0])
    elif mode == "balances":
        print(get_balances())
    elif mode == "exchange":
        print(get_exchange_info())
    elif mode == "financing":
        print(get_financing_rates())
    elif mode == "account":
        print(get_account_info())
    elif mode == "trades":
        start = args.get('from')
        end = args.get('to')
        start_date = date_str_to_date(start) if start is not None else today_date()
        if end is not None:
            end_date = date_str_to_date(args['to'])
        else:
            end_date = today_date()

        for dd in date_range(start_date, end_date):
            start_dt = date_to_start_dt(dd)
            if dd == today_date():
                end_dt = None
            else:
                end_dt = date_to_end_dt(dd)

            trades_df = get_hist_trades(start_dt, end_dt)
            trade_cnt = len(trades_df)
            logger.info(f"Loaded {trade_cnt} trades for {dd}")
            write_df_to_csv(trades_df, f"{dir_manager.BINANCE_FILLS_DIR}/bfills.{date_to_str(dd)}")
    else:
        print(f"Unknown mode {mode}")
