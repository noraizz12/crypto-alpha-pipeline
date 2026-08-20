import logging
import os
from datetime import datetime as dt
from datetime import timedelta as td
from datetime import timezone

import pandas as pd

from lib.data import load_single_live_bar_df
from lib.external import get_delisting_schedule
from lib.external.binance_utils import get_balances, get_funding, get_hist_trades, get_positions
from lib.util import date_to_str, millis_to_dt, TARDIS_EXCHANGE, raise_alert, HIGH
from lib.util.config import get_config
from lib.util.directory import dir_manager
from lib.universe import Universe
from lib.util.files import write_df_to_csv
from lib.util.logging_util import KeyLogger
from lib.util.time_util import today
from lib.util.util import SYMBOL_BASE, STABLECOINS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = KeyLogger(logger)


class DownloadBinance:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.config = get_config()[1]
        self.universe = Universe(config=self.config, debug=self.debug)


    def update_delisting_info(self):
        # query delisting schedule from binance
        # note this is spot delisting schedule
        delisting_schedule = get_delisting_schedule()
        if delisting_schedule is None:
            return

        # find the existing delisting records
        existing_delisting_records = set()
        file_path = f"{dir_manager.TRADING_DIR}/delisting.txt"
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    symbol = line.split(',')[0].strip()
                    existing_delisting_records.add(symbol)
        logger.info(f"Seeing existing delisting records {existing_delisting_records}")

        universe = self.universe.load_universe_symbols(universe_source='file', symbol_type=SYMBOL_BASE)

        # update new schedule records
        new_delisting_records = []
        new_delisting_msg = "Seeing delisting info"
        for schedule in delisting_schedule:
            delisting_ts = date_to_str(millis_to_dt(schedule['delistTime']) - td(days=1))
            new_delisting_msg += f" @{delisting_ts} for "
            for symbol in schedule['symbols']:
                symbol_perp = symbol + '_' + TARDIS_EXCHANGE
                if symbol in universe and symbol_perp not in existing_delisting_records:
                    new_delisting_msg += f"{symbol}, "
                    new_delisting_records.append((symbol_perp, delisting_ts))
        logger.info(f"Seeing new delisting records {new_delisting_records}")

        with open(file_path, 'a') as f:
            for symbol, date in new_delisting_records:
                if symbol not in existing_delisting_records:
                    f.write(f"{symbol}, {date}\n")

        if len(new_delisting_records) > 0:
            raise_alert(key='Binance delisting alert', priority=HIGH, description=new_delisting_msg)

    def run(self):
        now = dt.now(timezone.utc)
        today_str = date_to_str(now.date())
        logger.info(f"Updating Binance Data {today_str}")

        positions_df = get_positions()
        if positions_df is not None:
            positions_df['ts'] = now
            pos_file_path = dir_manager.BINANCE_POSITION_DIR + f"/binance_pos.{today_str}"

            positions_df = positions_df[positions_df['qty'] != 0]

            # Detect position closes: tracked symbols missing from API response
            if os.path.exists(f"{pos_file_path}.csv"):
                tracked_symbols = set(pd.read_csv(f"{pos_file_path}.csv", usecols=['symbol'])['symbol'].unique())
                closed_symbols = tracked_symbols - set(positions_df['symbol'])
                if closed_symbols:
                    logger.info(f"Recording position closes for: {sorted(closed_symbols)}")
                    zero_row = {col: 0 for col in positions_df.columns}
                    zero_row.update({'ts': now, 'positionSide': 'BOTH'})
                    close_rows_df = pd.DataFrame([{**zero_row, 'symbol': s} for s in closed_symbols])
                    positions_df = pd.concat([positions_df, close_rows_df], ignore_index=True)

            positions_df = positions_df.sort_values(by='value')
            if not self.debug:
                write_df_to_csv(positions_df, pos_file_path, append=True)
        else:
            logger.warning("Could not get positions from binance!")

        balances_df = get_balances()
        if balances_df is not None:
            balances_df = balances_df[balances_df['totalWalletBalance'] != 0]
            balances_df['symbol'] = balances_df['asset'] + "USDT"
            prices_df = load_single_live_bar_df(latest=True)
            balances_df = pd.merge(balances_df, prices_df[['symbol', 'close_mid']], left_on='symbol', right_on='symbol', how='left')
            # Stablecoins have price = 1.0
            balances_df.loc[balances_df['asset'].isin(STABLECOINS), 'close_mid'] = 1.0
            balances_df['balance'] = balances_df['totalWalletBalance'] * balances_df['close_mid']
            balances_df['ts'] = now
            if not self.debug:
                write_df_to_csv(balances_df, dir_manager.BALANCES_DIR + f"/bal.{today_str}", append=True)
        else:
            logger.warning("Could not get balances from binance!")

        funding_df = get_funding(today())
        if funding_df is not None:
            funding_df['ts'] = now
            if not self.debug:
                write_df_to_csv(funding_df, dir_manager.FUNDING_INCOME_DIR + f"/funding.{today_str}", append=False)
        else:
            logger.warning("Could not get funding")

        trades_df = get_hist_trades()
        if trades_df is not None:
            if not self.debug:
                write_df_to_csv(trades_df, dir_manager.BINANCE_FILLS_DIR + f"/bfills.{today_str}")
        else:
            logger.warning("Could not get trades from binance")

