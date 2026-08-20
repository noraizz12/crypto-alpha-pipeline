import json
import logging
import queue
import threading
import time
from collections import defaultdict
from datetime import datetime as dt
from datetime import timedelta as td, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from slack_sdk.webhook.client import WebhookClient

from lib.util.aws import write_df_to_s3_parquet, write_text_array_to_s3_gzip
from lib.external.binance_utils import get_financing_rates, get_open_interest
from lib.util.dataframes import remove_infs, shrink_floats
from lib.util.files import safe_mkdir
from lib.util.time_util import DATE_FORMAT, next_n_minute, to_datetime
from lib.util.slack import SLACK_WEBHOOK
from lib.util.directory import DATA_DIR, dir_manager
from lib.util.util import LOCAL, TARDIS_EXCHANGE, exception_msg
from lib.util.logging_util import KeyLogger

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)

BAR_MINS = 1
MD_CHECK_MINS = 10
OI_CHECK_MINS = 30
OI_DELAYED_START_MINS = 5
QUOTE_LATENCY_ALERT_MS = 60
QUOTE_LATENCY_ALERT_CNT = 5

TRADE_COL_MAP = {
    's': 'symbol',
    'p': 'price',
    'q': 'amount',
    'T': 'exch_ts',
    'recv_ts': 'recv_ts',
    'current_mid': 'current_mid',
    'f': 'first_trade_id',
    'l': 'last_trade_id',
}
QUOTE_COLUMNS = [
    "symbol",
    "recv_ts",
    "exch_ts",
]

for ii in range(10):
    QUOTE_COLUMNS += [f"bidpx{ii}", f"bidsz{ii}"]

for ii in range(10):
    QUOTE_COLUMNS += [f"askpx{ii}", f"asksz{ii}"]


def _next_day_ts(ts) -> int:
    return int((dt.fromtimestamp(ts, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + td(days=1)).timestamp())


def _next_bar_minute(ts) -> int:
    return next_n_minute(ts, BAR_MINS)


def _next_md_check_minute(ts) -> int:
    return next_n_minute(ts, MD_CHECK_MINS)


def _next_oi_check_minute(ts) -> int:
    return next_n_minute(ts, OI_CHECK_MINS)


def format_latency_slack_msg(md_type: str, df: pd.DataFrame) -> Dict[str, Any]:
    df_string = df.to_string()
    msg = {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"```1-min {md_type} data latency check in ms:\n{df_string}\n```"
                }
            }
        ]
    }
    return msg


class OpenInterestUpdater(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, name="OpenInterestUpdaterThread")
        self.symbols = []
        self.dead = False
        self.please_kill_me = False
        self.current_oi_df = None
        self.next_check = time.time() + OI_DELAYED_START_MINS * 60

    def update_symbols(self, new_symbols: List[str]):
        if len(new_symbols) > len(self.symbols):
            logger.info(f"Update new symbols for OpenInterestUpdater {len(new_symbols)} compared to {len(self.symbols)}")
        self.symbols = new_symbols

    def kill_me(self):
        self.please_kill_me = True

    def run(self):
        """Main loop for the thread"""
        while True:
            time.sleep(30)
            ts = time.time()
            if ts > self.next_check:
                logger.info("Fetching Open Interest data...")
                self.next_check = _next_oi_check_minute(ts)
                if self.symbols:
                    self.current_oi_df = get_open_interest(self.symbols)
                if self.please_kill_me:
                    logger.info("Exiting Open Interest Listener Loop...")
                    break
        self.dead = True


class DataAggregator(threading.Thread):
    def __init__(self, quote_only_mode: bool = False, debug: bool = False):
        threading.Thread.__init__(self, name="DataAggregatorThread")
        self.quote_only_mode = quote_only_mode
        self.debug = debug

        self.dead = False
        self.please_kill_me = False

        self.current_book = {}
        self.current_quotes = []
        self.current_trades = []
        self.update_cnt = defaultdict(int)
        self.last_quotes = None
        self.last_trades = None
        self.data_bucket = None
        self.quote_mid_df = None

        self.dollar_tracker = defaultdict(float)
        self.qty_tracker = defaultdict(float)

        self.last_ts = time.time()
        self.next_minute = _next_bar_minute(self.last_ts)
        self.next_md_check_minute = _next_md_check_minute(self.last_ts)
        self.next_day = _next_day_ts(self.last_ts)
        self.num_consecutive_high_quote_latency = 0
        self.slack_client = WebhookClient(SLACK_WEBHOOK)

        self.oi_updater = OpenInterestUpdater()
        self.oi_updater.start()


    def _mk_data_dir(self):
        yyyymmdd = dt.now(timezone.utc).date().strftime(DATE_FORMAT)
        safe_mkdir(f"{dir_manager.LIVE_DATA_DIR}/{yyyymmdd}")
        self.data_bucket = f"live/{yyyymmdd}"

    def _get_bar_ts(self):
        return self.next_minute - (60 * BAR_MINS)

    def reset_vwap(self):
        self.dollar_tracker = defaultdict(float)
        self.qty_tracker = defaultdict(float)

    def vwap(self, symbol: str) -> Optional[float]:
        try:
            dollars = self.dollar_tracker[symbol]
            qty = self.qty_tracker[symbol]
        except KeyError:
            logger.error(f"Could not get vwap on {symbol}", key='fail to get vwap')
            return None
        if qty == 0:
            return None
        return dollars / qty

    def add_quote_data(self, msg: dict) -> bool:
        qobj = {'recv_ts': np.float64(dt.now(timezone.utc).timestamp() * 1000)}
        try:
            qobj['exch_ts'] = np.float64(msg['E'])
            sym = msg['s']
            for idx, bid in enumerate(msg['b']):
                qobj[f"bidpx{idx}"] = np.float32(bid[0])
                qobj[f"bidsz{idx}"] = np.float32(bid[1])
            for idx, ask in enumerate(msg['a']):
                qobj[f"askpx{idx}"] = np.float32(ask[0])
                qobj[f"asksz{idx}"] = np.float32(ask[1])

            self.current_quotes.append([sym] + list(qobj.values()))
            self.current_book[sym] = qobj
            self.update_cnt[sym] += 1
        except Exception as e:
            logger.error(f"Bad quote message {msg} with error {e}", key="see bad quote message")
            return False
        return True

    def add_trade_data(self, msg: dict):
        if self.quote_only_mode:
            logger.error("Getting trade data in quote-only mode!!!", key="see trade data in quote-only mode")

        ts = dt.now(timezone.utc).timestamp() * 1000

        try:
            book = self.current_book[msg['s']]
            msg['current_mid'] = (book['askpx0'] + book['bidpx0']) / 2.0
        except Exception as e:
            msg['current_mid'] = 0.0
            logger.warning(f"Could not find current mid for trade on {e}")

        try:
            msg['recv_ts'] = ts
            self.current_trades.append(msg)
        except Exception as e:
            logger.error(f"Bad trade message {msg} with error {e}", key="see bad trade message")

    def collect_quote_messages(self) -> Optional[pd.DataFrame]:
        quote_cnt = len(self.last_quotes)
        logger.info(f"Collecting {quote_cnt} quote messages...")

        if quote_cnt == 0:
            no_quote_msg = "quote count is 0, looks like they stopped flowing!!"
            if not LOCAL:
                self.slack_client.send(text=no_quote_msg)
            logger.warning(no_quote_msg)
            self.quote_mid_df = None
            return None

        #TODO: symbol as category...
        quote_df = pd.DataFrame(self.last_quotes, columns=QUOTE_COLUMNS)
        quote_df['latency'] = ((quote_df['recv_ts'] - quote_df['exch_ts']) * 1e3).astype(np.float32)
        # quote_df['exch_ts'] = to_datetime(quote_df['exch_ts'].astype(np.float64), unit='ms')
        # quote_df['recv_ts'] = to_datetime(quote_df['recv_ts'].astype(np.float64), unit='ms')

        for ii in range(10):
            quote_df[f"bidpx{ii}"] = quote_df[f"bidpx{ii}"].astype(np.float32)
            quote_df[f"askpx{ii}"] = quote_df[f"askpx{ii}"].astype(np.float32)
            quote_df[f"bidsz{ii}"] = quote_df[f"bidsz{ii}"].astype(np.float32)
            quote_df[f"asksz{ii}"] = quote_df[f"asksz{ii}"].astype(np.float32)

        quote_df['mid'] = (quote_df['bidpx0'] + quote_df['askpx0']) / 2
        self.quote_mid_df = quote_df[['symbol', 'exch_ts', 'mid']]
        quote_df['bid_sz'] = quote_df['bidsz0'] + quote_df['bidsz1'] + quote_df['bidsz2'] + quote_df['bidsz3'] + quote_df['bidsz4']
        quote_df['ask_sz'] = quote_df['asksz0'] + quote_df['asksz1'] + quote_df['asksz2'] + quote_df['asksz3'] + quote_df['asksz4']
        quote_df['close_wgt_mid'] = (quote_df['bidpx0'] * quote_df['bid_sz'] + quote_df['askpx0'] * quote_df['ask_sz']) / (quote_df['bid_sz'] + quote_df['ask_sz'])
        quote_df['spread'] = quote_df['askpx0'] - quote_df['bidpx0']

        quote_df = quote_df.groupby('symbol') \
            .agg(mid_std=('mid', 'std'),
                 open_mid=('mid', 'first'),
                 close_mid=('mid', 'last'),
                 close_wgt_mid=('close_wgt_mid', 'last'),
                 low_mid=('mid', 'min'),
                 high_mid=('mid', 'max'),
                 update_cnt=('mid', 'count'),
                 spread_avg=('spread', 'mean'),
                 bid_sz_avg=('bid_sz', 'mean'),
                 ask_sz_avg=('ask_sz', 'mean'),
                 book_latency=('latency', 'mean'),
                 ).reset_index()
        quote_df['update_cnt'] = quote_df['update_cnt'].astype('Int32')
        return quote_df

    def collect_trade_messages(self) -> pd.DataFrame:
        trade_cnt = len(self.last_trades)
        logger.info(f"Collecting {trade_cnt} trade messages...")

        if trade_cnt == 0:
            logger.warning("No trade messages!")
            cols = list(TRADE_COL_MAP.values()) + ['venue', 'end', 'vwap', 'dvolume', 'volume', 'high_trade', 'low_trade', 'close_trade', 'trade_cnt']
            return pd.DataFrame(columns=cols)

        trades_df = pd.DataFrame.from_records(self.last_trades)
        try:
            trades_df = trades_df[TRADE_COL_MAP.keys()]
        except KeyError as e:
            logger.error(f"Bad trade data, missing keys with error {e}, trades df as {trades_df}", key="see bad trade data with missing cols")
            raise

        trades_df = trades_df.rename(columns=TRADE_COL_MAP)
        trades_df['price'] = trades_df['price'].astype(np.float32)
        trades_df['amount'] = trades_df['amount'].astype(np.float32)
        trades_df['notional'] = trades_df['price'] * trades_df['amount']
        if self.quote_mid_df is not None:
            # concat quote mid with trade and sort by exch_ts
            # forward fill the mid in trades by symbol
            # if still missing, just use its own current_mid in trades_df
            trades_df = pd.concat([self.quote_mid_df, trades_df]).sort_values(by='exch_ts')
            trades_df['mid'] = trades_df.groupby('symbol')['mid'].ffill()
            trades_df['mid'] = trades_df['mid'].fillna(trades_df['current_mid'])
            trades_df = trades_df.loc[~trades_df['recv_ts'].isna()]
            trades_df = trades_df.reset_index(drop=True)
        else:
            trades_df['mid'] = trades_df['current_mid']
        trades_df['bid_trade_dollars'] = np.float32(0)
        trades_df['ask_trade_dollars'] = np.float32(0)
        valid_mid_idx = trades_df['mid'] > 0
        ask_idx = trades_df['price'] > trades_df['mid']
        trades_df.loc[valid_mid_idx & ask_idx, 'ask_trade_dollars'] = trades_df['notional']
        trades_df.loc[valid_mid_idx & ~ask_idx, 'bid_trade_dollars'] = trades_df['notional']
        trades_df['latency'] = ((trades_df['recv_ts'] - trades_df['exch_ts']) * 1e3).astype(np.float32)
        trades_df['exch_ts'] = to_datetime(trades_df['exch_ts'], unit='ms')
        trades_df['recv_ts'] = to_datetime(trades_df['recv_ts'], unit='ms')
        trades_df['trade_cnt'] = trades_df['last_trade_id'] - trades_df['first_trade_id'] + 1
        trades_df = trades_df.groupby('symbol') \
            .agg(high_trade=('price', 'max'),
                 low_trade=('price', 'min'),
                 close_trade=('price', 'last'),
                 dvolume=('notional', 'sum'),
                 trade_cnt=('trade_cnt', 'sum'),
                 volume=('amount', 'sum'),
                 trade_latency=('latency', 'mean'),
                 bid_trade_dollars=('bid_trade_dollars', 'sum'),
                 ask_trade_dollars=('ask_trade_dollars', 'sum'),
                 ).reset_index()
        trades_df['trade_cnt'] = trades_df['trade_cnt'].astype('Int32')
        trades_df['vwap'] = remove_infs(trades_df['dvolume'].fillna(0) / trades_df['volume'])

        for col in ['volume', 'dvolume', 'bid_trade_dollars', 'ask_trade_dollars']:
            trades_df[col] = trades_df[col].fillna(0)

        for _, row in trades_df.iterrows():
            self.dollar_tracker[row['symbol']] += row['dvolume']
            self.qty_tracker[row['symbol']] += row['volume']

        return trades_df

    def kill_me(self):
        if not self.quote_only_mode:
            self.please_kill_me = True
            if hasattr(self, 'oi_updater'):
                self.oi_updater.kill_me()

    def run(self):
        if not self.quote_only_mode:
            self._mk_data_dir()

        financing_rates_df = None
        while True:
            time.sleep(0.01)
            ts = time.time()
            # query funding related 1 second before next minute
            if financing_rates_df is None and ts > self.next_minute - 1:
                financing_rates_df = get_financing_rates()
            if ts > self.next_minute:
                logger.info("Aggregating Trades/Quotes...")
                self.last_quotes = self.current_quotes
                self.last_trades = self.current_trades
                self.current_quotes = []
                self.current_trades = []
                self.next_minute = _next_bar_minute(ts)
                if ts > self.next_day:
                    self._mk_data_dir()
                    self.next_day = _next_day_ts(ts)

                quotes_df = self.collect_quote_messages()
                if quotes_df is None:
                    break

                trades_df = self.collect_trade_messages()
                current_symbols = sorted(self.current_book.keys())
                self.oi_updater.update_symbols(current_symbols)

                if ts > self.next_md_check_minute:
                    self.next_md_check_minute = _next_md_check_minute(ts)
                    try:
                        book_latency_description = (quotes_df[['book_latency']] / 1e3).describe()
                        trade_latency_description = (trades_df[['trade_latency']] / 1e3).describe()
                        quotes_msg = format_latency_slack_msg('quotes', book_latency_description)
                        trades_msg = format_latency_slack_msg('trades', trade_latency_description)
                        if not LOCAL:
                            self.slack_client.send(blocks=quotes_msg['blocks'])
                            self.slack_client.send(blocks=trades_msg['blocks'])
                        else:
                            logger.info(book_latency_description)
                            logger.info(trade_latency_description)
                        if quotes_df['book_latency'].mean() > QUOTE_LATENCY_ALERT_MS * 1000:
                            self.num_consecutive_high_quote_latency += 1
                            if self.num_consecutive_high_quote_latency >= QUOTE_LATENCY_ALERT_CNT:
                                logger.error(f"1-min quote latency description {book_latency_description.to_string()}, ", key=f"see consecutive 1-min avg quote latency higher than {QUOTE_LATENCY_ALERT_MS} ms")
                        else:
                            self.num_consecutive_high_quote_latency = 0
                    except Exception as e:
                        logger.error(exception_msg("Error publishing Latency check", e), key="fail to publish latency check")

                bar_df = pd.merge(quotes_df, trades_df, left_on=['symbol'], right_on=['symbol'], how='left')

                if financing_rates_df is not None:
                    financing_rates_df = financing_rates_df.rename(columns={
                        'lastFundingRate': 'last_funding_rate',
                        'indexPrice': 'index_price',
                        'estimatedSettlePrice': 'estimated_settle_price',
                        'nextFundingTime': 'next_funding_time',
                    })
                    bar_df = pd.merge(bar_df, financing_rates_df[['symbol', 'last_funding_rate', 'index_price', 'estimated_settle_price', 'next_funding_time']], on=['symbol'], how='left')
                    bar_df['index_price_dvolume'] = bar_df['index_price'] * bar_df['volume']
                    bar_df['index_mid_price_diff'] = bar_df['index_price'] - bar_df['close_mid']
                    bar_df['next_funding_time'] = pd.to_datetime(bar_df['next_funding_time'], unit='ms').dt.tz_localize('UTC')
                else:
                    logger.warning("Setting binance funding rates to nan")
                    bar_df['last_funding_rate'] = np.nan
                    bar_df['index_price'] = np.nan
                    bar_df['estimated_settle_price'] = np.nan
                    bar_df['next_funding_time'] = pd.NaT
                    bar_df['index_price_dvolume'] = np.nan
                    bar_df['index_mid_price_diff'] = np.nan
                # reset to None so we query for next min bar
                financing_rates_df = None

                if self.oi_updater and self.oi_updater.current_oi_df is not None:
                    bar_df = pd.merge(bar_df, self.oi_updater.current_oi_df[['symbol', 'openInterest']], on='symbol', how='left')
                    bar_df = bar_df.rename(columns={'openInterest': 'open_interest'})
                else:
                    logger.info("Setting binance open interest to nan")
                    bar_df['open_interest'] = np.nan

                bar_ts = self._get_bar_ts()
                bar_df['end'] = to_datetime(bar_ts * 1e9)
                bar_df['trade_cnt'] = bar_df['trade_cnt'].fillna(0)
                bar_df['venue'] = TARDIS_EXCHANGE
                bar_df = shrink_floats(bar_df)

                bar_file_name = self.data_bucket + f"/{bar_ts}.parquet"

                if self.debug:
                    logger.info("Not writing bar data....")
                    continue

                logger.info(f"Writing {bar_file_name}")
                bar_df.to_parquet(f"{DATA_DIR}/{bar_file_name}")
                if not LOCAL:
                    trade_file_name = self.data_bucket + f"/raw.trade.{bar_ts}.txt"
                    quote_file_name = self.data_bucket + f"/raw.quote.{bar_ts}.txt"
                    try:
                        write_df_to_s3_parquet(bar_df, key=bar_file_name)
                        write_text_array_to_s3_gzip([json.dumps(str(jj)) for jj in self.last_trades], key=trade_file_name)
                        write_text_array_to_s3_gzip([','.join([str(it) for it in ln]) for ln in self.last_quotes], key=quote_file_name)
                    except Exception as e:
                        logger.error(f"Fail to write {bar_file_name=}, {trade_file_name=} and {quote_file_name=} since {e}", key='fail to write market data into s3')

                if self.please_kill_me:
                    logger.info("Exiting Data Aggregation Loop...")
                    break
        if not self.please_kill_me:
            logger.warning("Exited Data Aggregator loop, problem with market data???")
        self.dead = True
