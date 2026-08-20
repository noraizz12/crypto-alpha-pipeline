import asyncio
import logging
import math
import os
import time
from collections import defaultdict
from datetime import datetime as dt
from datetime import timezone
from typing import Any, Dict, List, Optional, Tuple

import cvxpy as cp
import cvxpy.error
import numpy as np
import pandas as pd
from slack_sdk.webhook.client import WebhookClient

from ..data import load_positions, get_latest_target_files
from lib.data.dataloader import DataLoader
from lib.external.binance_utils import get_balances, get_exchange_info, get_positions
from lib.portfolio_optimization import PortfolioOptimizer
from lib.data.loaders import load_metadata, load_single_live_bar_df
from lib.universe import Universe
from lib.util.config import extract_horizon_models
from lib.util.dataframes import highlight_df_rows, make_symbol, remove_infs, safe_get_from_series, shrink_floats
from lib.util.directory import REPORT_DIR, dir_manager
from lib.util.files import safe_mkdir
from lib.util.logging_util import KeyLogger
from lib.util.time_util import date_to_str, dt_to_millis, dt_to_str, to_datetime, today_date, yesterday_date
from lib.util.util import LOCAL, PROD, STATARB_OMS_IP, STATARB_OMS_PORT, STATARB_CPP_OMS_IP, STATARB_CPP_OMS_INGEST_PORT, SYMBOL_PAIR
from lib.util.util import fmoney, fpct, get_nested_dict_value, safe_del_from_dict, unique_list, log_and_raise
from .md_aggregator import DataAggregator
from .oms_listener import OMSListener, CppOMSListener, CppOMSRestReplyListener
from .positions import PositionRecorder
from .trade_quote_poller import TradeQuotePoller
from .trader_eod import EODReversalManager
from .trading import Cancel, Fill, FillType, Order, Side, get_agg_side_fills_info
from .zmq_util import add_publisher, add_pusher

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)

logger = KeyLogger(original_logger)

LEVERAGE = 1.0
IMBALANCE_THRESHOLD = 0.025
FILL_DEFICIT_THRESHOLD_PCT = 0.05
IMBALANCE_AMT_TILT = .5
TARGET_STALE_SECONDS = 60 * 60
ORDER_ACK_LATENCY_SECONDS = 60

MIN_AGGRESSION = -9
BASE_AGGRESSION = MIN_AGGRESSION

NOTRADE_FILE = dir_manager.TRADING_DIR + "/notrade.txt"
LIQUIDATE_FILE = dir_manager.TRADING_DIR + "/liquidate.txt"

pd.set_option("display.precision", 4)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')


class Trader:
    def __init__(
            self,
            config: dict,
            debug: bool = False,
            market_data_only: bool = False,
            no_trade: bool = False,
            no_limits: bool = False,
            trade_on_startup: bool = False,
            quote_only_mode: bool = False,
            paper_trade: bool = True,
            limit_dollars_traded: bool = True,
            slack_client: Optional[WebhookClient] = None,
            old_oms: bool = False,
            eod: bool = False,
            local_oms: bool = False
    ):
        assert not (local_oms and old_oms), "Cannot use --local-oms with --old-oms"
        self.config = config
        self.old_oms = old_oms
        self.local_oms = local_oms
        self.quote_only_mode = quote_only_mode
        self.trade_on_startup = trade_on_startup
        self.paper_trade = paper_trade
        self.market_data_only = market_data_only
        self.no_trade = no_trade
        self.no_limits = no_limits
        self.debug = debug or paper_trade
        self.please_kill_me = False
        self.no_ack_in_current_wave = True
        self.dir_manager = dir_manager
        self.data_loader = DataLoader(config=self.config, data_loader_dir_manager=self.dir_manager)

        #remove later
        self.max_aggression = 6
        if no_limits:
            self.max_aggression = 6
            limit_dollars_traded = False

        logger.info(f"Creating Trader, {PROD=} {LOCAL=}, debug:{self.debug}, {paper_trade=} No Limits: {self.no_limits}")

        self.today = today_date()
        self.fraction_of_period_elapsed = 0.0
        self.last_target_file_loaded_ts = None
        self.last_target_file_ts = None

        self.slack = slack_client
        horizon_models = extract_horizon_models(config=self.config, exclude_zero_weight=True)
        self.model_horizons = sorted(unique_list([hm[0] for hm in horizon_models]))

        self.wave_interval_seconds = self.config['WAVE_INTERVAL_MINS'] * 60
        assert self.wave_interval_seconds > 0
        self.min_trade_amt = self.config['MIN_TRADE_DOLLARS']
        self.max_move_filter = self.config['MAX_MOVE_FILTER']
        assert 1 < self.max_move_filter <= 10.0
        self.order_acceleration = self.config['ORDER_ACCELERATION']
        assert 0 < self.order_acceleration <= 2.0
        self.reoptimize_interval_secs = self.config['REOPTIMIZE_INTERVAL_MINS'] * 60.0
        assert self.reoptimize_interval_secs > 0
        self.step_frac = self.wave_interval_seconds / self.reoptimize_interval_secs
        assert self.step_frac <= 1.0
        self.st_factor_sigmas = self.config['ST_FACTOR_SIGMAS']
        self.st_kappa = self.config['ST_KAPPA']
        assert self.st_kappa > 0
        self.max_order_dollars = self.config['MAX_ORDER_DOLLARS']
        assert self.max_order_dollars > 0
        self.max_total_order_dollars = self.config['MAX_TOTAL_ORDER_DOLLARS']
        assert self.max_total_order_dollars > 0
        self.short_term_alphas = self.config['SHORT_TERM_MODEL_HORIZONS']
        self.reopt_minutes = self.config['REOPTIMIZE_INTERVAL_MINS']
        assert self.reopt_minutes > 0
        self.max_port_size = self.config['MAX_PORTFOLIO_NOTIONAL']
        self.max_turnover = self.config['MAX_TURNOVER']
        assert self.max_turnover > 0

        #just warns right now...
        self.max_dollars_per_opt = self.config['MAX_DOLLARS_PER_OPT_TO_TRADE_MULT'] * self.max_port_size * self.max_turnover * self.reopt_minutes / 1440 if limit_dollars_traded else None

        #just warns...
        self.max_trading_bias = self.config['MAX_TRADING_BIAS']

        self.symbols = Universe(config=self.config).load_universe_symbols(universe_source='file', filter='tradeable', symbol_type=SYMBOL_PAIR)

        self.orders_file = None
        self.fills_file = None
        self.raw_oms_file = None

        self.tick_size = self.make_precision_map()

        self.current_target_df = None
        self.outstanding_orders: Dict[str, Order] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.oid_to_koid: Dict[str, str] = {}
        self.metadata_date = yesterday_date()
        self.metadata_df = None
        self.seen_msg_key = set()
        self.current_wave_fills: Dict[str, List[Fill]] = defaultdict(list)
        self.dollars_traded_this_opt = 0.0

        self.pos_recorder = PositionRecorder(prod=True)

        self.current_aggression = {}
        self.fill_slip_dollars: float = 0.0

        self.eod_manager = EODReversalManager(config) if eod else None

        self.order_problem_cnt = defaultdict(int)
        self.error_msg_cnt = defaultdict(int)
        self.aggregator = None
        self.quote_poller = None
        self.oms_listener = None
        self.oms_response_listener = None

        if not self.paper_trade:
            self.max_port_size = self.get_max_portfolio_size()
            if self.max_port_size is None or self.max_port_size < 2000:
                logger.error(f"Could not get initial portfolio size {self.max_port_size}, exiting", key='max_port_size too small')
                raise RuntimeError()

        self._open_files()
        self.notrade_list = self.update_no_trade_list(file_path=NOTRADE_FILE)
        self.liquidate_list = self.update_liquidate_list(file_path=LIQUIDATE_FILE)

        self.oms_pub_socket = None
        self._local_oms_sender = None
        if PROD and (not self.debug or self.paper_trade):
            if self.local_oms:
                self._create_local_oms()
            else:
                self._create_oms_publisher()


    def get_remaining_waves(self) -> int:
        remaining_fraction = max(0.0, 1.0 - self.fraction_of_period_elapsed)
        remaining_waves = int(math.floor(self.reoptimize_interval_secs * remaining_fraction / self.wave_interval_seconds))
        return remaining_waves

    # Expected liquidate file format:
    # BNXUSDT, 12
    # EOSUSDT, 6
    @staticmethod
    def update_liquidate_list(file_path: str) -> List[Dict[str, Any]]:
        liquidate_list = []
        with open(file_path) as ff:
            for line in ff.readlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [part.strip() for part in line.split(',')]
                record = {'symbol': parts[0], 'liquidate_hour': int(parts[1]), 'liquidate_start_dt': dt.now(timezone.utc), 'liquidate_init_pos': 0}
                liquidate_list.append(record)
                logger.info(f"Adding {record} to liquidate list")
        return liquidate_list

    @staticmethod
    def update_no_trade_list(file_path: str) -> List[str]:
        no_trade_list = []
        with open(file_path) as ff:
            for symbol in ff.readlines():
                symbol = symbol.strip()
                logger.info(f"Adding {symbol} to no trade list")
                no_trade_list.append(symbol)
        return no_trade_list

    def connect_to_market_data(self, symbols: Optional[List[str]] = None):
        logger.info("Connecting to market data...")
        self.aggregator = DataAggregator(quote_only_mode=self.debug, debug=self.debug)
        self.quote_poller = TradeQuotePoller(config=self.config, aggregator=self.aggregator, quote_only_mode=self.quote_only_mode, symbols=symbols)
        self.quote_poller.start()
        self.quote_poller.wait_for_book(wait_seconds=30)

    def update_symbols(self):
        # concerned that this will not catch slow ticking securities...
        new_symbols = []
        for sym in self.symbols:
            if sym in self.aggregator.current_book:
                new_symbols.append(sym)
        logger.info(f"Symbols going from {len(self.symbols)} -> {len(new_symbols)}")
        self.symbols = new_symbols

    def _message(self, msg: str, error: bool = False, msg_key: str = "trader message", msg_to_channel: bool = False, error_trigger_threshold: int = 1):
        if error:
            self.error_msg_cnt[msg_key] += 1
            # alert if error msg count greater than its threshold
            if self.error_msg_cnt[msg_key] >= error_trigger_threshold:
                logger.error(msg, key=msg_key)
                # refresh to 0 after alert the error
                self.error_msg_cnt[msg_key] = 0
            else:
                logger.warning(msg)
        else:
            logger.info(msg)

        if self.debug:
            # print(msg)
            return

        if self.slack is not None:
            if error:
                msg = "[ERROR] " + msg
            if self.debug:
                msg = "[DEBUG] " + msg
            if msg_to_channel and msg_key not in self.seen_msg_key:
                msg = '<!channel>```' + msg + '```'
                self.seen_msg_key.add(msg_key)
            else:
                msg = '```' + msg + '```'
            self.slack.send(text=msg)

    def _order_file_name(self) -> str:
        paper = "paper." if self.paper_trade else ""
        today_str = date_to_str(self.today)
        order_dir =  f"{dir_manager.ORDERS_DIR}/{today_str}"
        safe_mkdir(order_dir)
        return f"{order_dir}/orders.{paper}{today_str}.csv"

    def _fill_file_name(self) -> str:
        paper = "paper." if self.paper_trade else ""
        return f"{dir_manager.FILLS_DIR}/fills.{paper}{date_to_str(self.today)}.csv"

    def _raw_oms_file_name(self) -> str:
        return f"{dir_manager.RAW_OMS_DIR}/oms.{date_to_str(self.today)}.txt"

    def _open_files(self):
        if self.orders_file is not None:
            self.orders_file.close()
        self.orders_file = open(self._order_file_name(), "a")

        if self.fills_file is not None:
            self.fills_file.close()
        self.fills_file = open(self._fill_file_name(), "a")

        if not self.paper_trade:
            if self.raw_oms_file is not None:
                self.raw_oms_file.close()
            self.raw_oms_file = open(self._raw_oms_file_name(), "a")

    def _create_oms_publisher(self):
        if self.old_oms:
            logger.info(f"Starting OMS sockets at {STATARB_OMS_IP}:{STATARB_OMS_PORT + 1}")
            self.oms_pub_socket = add_publisher(connect_url=f"tcp://{STATARB_OMS_IP}:{STATARB_OMS_PORT + 1}", async_ctx=False)
        else:
            endpoint = f"{STATARB_CPP_OMS_IP}:{STATARB_CPP_OMS_INGEST_PORT}"
            logger.info(f"Starting C++ OMS sockets at {endpoint}")
            self.oms_pub_socket = add_pusher(connect_url=f"tcp://{endpoint}")

    def _create_local_oms(self):
        from .local_oms import LocalOMSSender  # pylint: disable=import-outside-toplevel
        logger.info("Starting Local OMS (direct Binance REST)")
        self._local_oms_sender = LocalOMSSender()

    def _create_oms_listener(self):
        if self.local_oms:
            from .local_oms import LocalOMSListener  # pylint: disable=import-outside-toplevel
            logger.info("Starting Local OMS Listener (direct Binance WebSocket)")
            self.oms_listener = LocalOMSListener()
            self.oms_listener.start()
            return
        self.oms_listener = OMSListener(self.config) if self.old_oms else CppOMSListener(self.config)
        if not self.old_oms:
            self.oms_response_listener = CppOMSRestReplyListener(self.config)
            self.oms_response_listener.start()
        self.oms_listener.start()

    async def shutdown(self):
        logger.info("Shutting down trader")

        self.please_kill_me = True
        if self.quote_poller is not None:
            await self.quote_poller.kill_me()

            shutdown_wait = time.time() + 10
            while time.time() < shutdown_wait:
                if self.quote_poller.dead:
                    break
                await asyncio.sleep(1)

            if not self.quote_poller.dead:
                logger.error("The quote poller never died!", key="quote_poller never died")

            if self.quote_poller.is_alive():
                self.quote_poller.join()


        if self.local_oms:
            if self._local_oms_sender is not None:
                logger.info("Closing Local OMS sender..")
                self._local_oms_sender.close()
        elif self.oms_pub_socket is not None:
            logger.info("Closing OMS Pub socket..")
            self.oms_pub_socket.close()

    def make_precision_map(self):
        logger.info("Making trade precision map")
        ret = {}
        exchange_info = get_exchange_info()
        exchange_symbols = exchange_info['symbols']
        if len(exchange_symbols) == 0:
            raise log_and_raise(f"No exchange info from binance!")

        no_info_symbols = set(self.symbols.copy())

        for symbol_data in exchange_symbols:
            symbol = symbol_data['symbol']
            if symbol in self.symbols:
                logger.info(f"getting exchange info on {symbol}")
                pf = [f for f in symbol_data['filters'] if f['filterType'] == 'PRICE_FILTER']
                ls = [f for f in symbol_data['filters'] if f['filterType'] == 'LOT_SIZE']
                tick_size = float(pf[0]['tickSize'])
                max_qty = int(ls[0]['maxQty'])
                ret[symbol] = {
                    'px': int(symbol_data['pricePrecision']),
                    'qty': int(symbol_data['quantityPrecision']),
                    'px_tick_size': tick_size,
                    'max_qty': max_qty
                }
                logger.info(f"Setting tick size / qty on {symbol} to {tick_size} / {max_qty}")
                no_info_symbols.remove(symbol)

        if len(no_info_symbols) > 0:
            logger.error(f"Not exchange data on symbols: {no_info_symbols}")

        return ret

    def get_max_portfolio_size(self) -> float:
        if self.debug:
            return self.max_port_size

        # not sure about this -- maybe don't size up and down based on this
        try:
            balances_df = get_balances()
            if balances_df is None:
                raise ValueError()
        except Exception as e:
            self._message(f"ERROR: Could not get balance from binance due to {e}, using {self.max_port_size}", error=True, msg_key='could not get balance from binance', error_trigger_threshold=3)
            if self.max_port_size is not None:
                return self.max_port_size
            raise

        STABLECOINS = ['USDTUSDT', 'USDCUSDT']
        balances_df = balances_df[balances_df['totalWalletBalance'] != 0]
        balances_df['symbol'] = balances_df['asset'] + "USDT"
        prices_df = load_single_live_bar_df(latest=True)
        balances_df = pd.merge(balances_df, prices_df[['symbol', 'close_mid']], left_on='symbol', right_on='symbol', how='left')
        balances_df.loc[balances_df['symbol'].isin(STABLECOINS), 'close_mid'] = 1.0
        balances_df['balance'] = balances_df['totalWalletBalance'] * balances_df['close_mid']

        balance = balances_df['balance'].sum()
        unrealized_pnl = balances_df['umUnrealizedPNL'].sum()
        tot = balance + unrealized_pnl
        self.max_port_size = min(math.floor(tot * LEVERAGE), self.config['MAX_PORTFOLIO_NOTIONAL'])
        self._message(f"Using max portfolio size total capital: {tot} {LEVERAGE=} {self.max_port_size}")
        return self.max_port_size

    def increase_aggression(self, symbol: str, incr: int = 1) -> int:
        aggression = self.get_aggression(symbol)
        return self.set_aggression(symbol, aggression + incr)

    def decrease_aggression(self, symbol: str, incr: int = 1) -> int:
        aggression = self.get_aggression(symbol)
        return self.set_aggression(symbol, aggression - incr)

    def get_aggression(self, symbol: str) -> int:
        return self.current_aggression.get(symbol, BASE_AGGRESSION)

    def set_aggression(self, symbol: str, aggression: int) -> int:
        aggression = max(aggression, MIN_AGGRESSION)
        aggression = min(aggression, self.max_aggression)
        self.current_aggression[symbol] = aggression
        return aggression

    def get_vwap(self, symbol: str) -> float:
        if self.aggregator is None:
            return self.get_current_mid(symbol)

        return self.aggregator.vwap(symbol)

    def get_current_mid(self, symbol: str) -> Optional[float]:
        if self.aggregator is None:
            return self.current_target_df[self.current_target_df['symbol'] == symbol]['opt_px'].iloc[0]

        book = None
        try:
            cnt = self.aggregator.update_cnt[symbol]
            if cnt == 0:
                self._message(f"Not getting quote updates on {symbol}!", error=True, msg_key="no quote updates")
                raise KeyError
            else:
                book = self.aggregator.current_book[symbol]
        except KeyError:
            logger.warning(f"Could not get current quote for {symbol}")

        if book is not None:
            try:
                val = (float(book['bidpx0']) + float(book['askpx0'])) * .5
                if val > 0:
                    return val
            except Exception as e:
                logger.warning(f"Bad quote book on {symbol} since {e}")

        self._message(f"ERROR: Could not get current mid for {symbol} with {book=}", error=True, msg_key=f'could not get current mid for {symbol}', error_trigger_threshold=3)
        return None

    def update_liquidate_init_pos(self, positions_df: pd.DataFrame):
        symbol_to_pos_dict = positions_df.set_index('symbol')['value'].to_dict()
        for liquidate_case in self.liquidate_list:
            liquidate_init_pos = symbol_to_pos_dict.get(liquidate_case['symbol'], 0)
            liquidate_case['liquidate_init_pos'] = liquidate_init_pos

    def refresh_positions(self, from_local_file: bool, init: bool = False):
        if from_local_file or self.debug:
            positions_df = load_positions(mode='latest')
        else:
            logger.info("Updating positions from server...")
            positions_df = get_positions()

        if positions_df is None or len(positions_df) == 0:
            logger.warning("Could not get positions...")
            return

        positions_df = positions_df[positions_df['qty'] != 0.0]
        # update liquidate init position when we first load position
        if init and len(self.liquidate_list) > 0:
            self.update_liquidate_init_pos(positions_df)

        total_notional = 0
        nonupdated_position_symbols = set(self.pos_recorder.get_symbols()) - set(positions_df['symbol'].unique())
        for _, row in positions_df.dropna(subset=['qty']).sort_values(by='symbol').iterrows():
            symbol = row['symbol']
            their_qty = row['qty']
            notional = row['value']
            # unrealized_profit = row['unRealizedProfit']
            # mark_price = row['markPrice']

            total_notional += abs(notional)

            sec_pos = self.pos_recorder.get_security(symbol)
            sec_pos.refresh_qty(their_qty)
            cost_basis = row['cost_basis']
            if pd.isna(cost_basis):
                logger.warning(f"Nan cost basis on {symbol} from position file with {notional=} {their_qty=}")
                # last mid not set yet perhaps
                cost_basis = their_qty * self.get_current_mid(symbol)
            sec_pos.refresh_cost_basis(cost_basis)
            # sec_pos.unrealized_profit = unrealized_profit

        for symbol in nonupdated_position_symbols:
            sec_pos = self.pos_recorder.get_security(symbol)
            sec_pos.refresh_qty(0.0)

        if total_notional > self.max_port_size:
            over_percentage = (total_notional - self.max_port_size) / self.max_port_size * 100
            is_error = over_percentage >= 30
            message = f"Positions of ${total_notional:.0f} greater than max portfolio size ${self.max_port_size:.0f} by {over_percentage:.1f}%!"
            # self._message(message, error=is_error, msg_key="notional greater than max")

    def _write_report(self, df: pd.DataFrame, name: str):
        if self.debug:
            print(df)
            return

        df['dump_ts'] = dt.now(timezone.utc)
        df.to_csv(f"{REPORT_DIR}/{name}.csv", index=False)
        df.to_csv(f"{REPORT_DIR}/{name}.{dt_to_str()}.csv", index=False)

    def report_positions(self):
        # XXX do i need to copy???
        df = self.current_target_df.copy()
        df['vwap'] = df['vwap'].fillna(0)
        df['ret_since_opt'] = df['ret_since_opt'] * 100.0
        df = df[(df['position'] != 0) | (df['target_position'] != 0)]
        df['trading'] = df['target_position'] - df['position']
        if 'target_position_wave' not in df.columns:
            df['target_position_wave'] = df['target_position'] * self.fraction_of_period_elapsed
        df['trading_wave'] = df['target_position_wave'] - df['position']
        df['target_ts'] = df['ts']
        df['update_ts'] = dt.now(timezone.utc)

        positions_df = df[['update_ts', 'symbol', 'target_ts', 'position', 'target_position', 'trading', 'cost_basis', 'unrealized_profit', 'unrealized_return']]
        self._write_report(positions_df.sort_values(by='unrealized_profit'), name="pnl/pnl")

        st_alpha_cols = [f'alpha_{hh}' for hh in self.short_term_alphas]
        alpha_df = df[['update_ts', 'target_ts', 'symbol', 'trading_wave', 'alpha_opt'] + st_alpha_cols]
        self._write_report(alpha_df.sort_values(by='trading_wave'), name="wave/wave")

        if df['vwap'].sum() != 0:
            prices_df = df
            not_tracking_idx = (prices_df['avg_cost'] > prices_df['vwap']) & (prices_df['position'] > 0)
            highlight_df_rows(prices_df, rows=not_tracking_idx)
            prices_df = prices_df[['update_ts', 'target_ts', 'symbol', 'position', 'current_px', 'avg_cost', 'vwap', 'vwap_shortfall', 'ret_since_opt', 'unrealized_profit']]
            self._write_report(prices_df.sort_values('vwap_shortfall'), name="vwap_shortfall/shortfall")
            vwap_shortfall = df['vwap_shortfall'].sum()
            self._message(f"vwap shortfall: {fmoney(vwap_shortfall)}")
        if self.metadata_df is not None and len(self.metadata_df) > 0:
            delisting_df = pd.merge(self.metadata_df, df[['update_ts', 'symbol', 'position']], on='symbol', how='inner')
            if len(delisting_df) > 0:
                self._message(delisting_df.sort_values(by='position').to_markdown(index=False), msg_key=f"delisting_alert_{self.today}", msg_to_channel=True)

    def update_df_positions_and_prices(self):
        self.current_target_df['qty'] = 0.0
        self.current_target_df['cost_basis'] = 0.0
        self.current_target_df['unrealized_profit'] = np.nan
        self.current_target_df['vwap'] = np.nan
        self.current_target_df['avg_cost'] = np.nan
        self.current_target_df['execution_qty'] = np.nan

        for sec_pos in self.pos_recorder.get_pos():
            symbol = sec_pos.symbol
            px = self.get_current_mid(symbol)
            if px is None or np.isnan(px):
                if abs(sec_pos.qty) > 0:
                    # XXX this can be very dangerous.
                    self._message(f"ERROR: Could not get price for {symbol} with {px=}, {sec_pos.qty=}", error=True, msg_key=f'could not get price for {symbol}', error_trigger_threshold=3)
                px = 0.0

            vwap = self.get_vwap(symbol)
            sec_pos.update_px(px)
            sec_pos.vwap = vwap
            # logger.info(f"Setting {sec_pnl.symbol} to qty: {sec_pnl.qty}, px: {sec_pnl.mark}")
            sym_idx = self.current_target_df['symbol'] == symbol
            self.current_target_df.loc[sym_idx, 'qty'] = sec_pos.qty
            self.current_target_df.loc[sym_idx, 'cost_basis'] = sec_pos.cost_basis
            self.current_target_df.loc[sym_idx, 'unrealized_profit'] = sec_pos.unrealized_profit
            self.current_target_df.loc[sym_idx, 'current_px'] = px
            self.current_target_df.loc[sym_idx, 'vwap'] = vwap
            self.current_target_df.loc[sym_idx, 'avg_cost'] = sec_pos.avg_px()
            self.current_target_df.loc[sym_idx, 'execution_qty'] = sec_pos.execution_qty

        self.current_target_df['position'] = self.current_target_df['qty'] * self.current_target_df['current_px']
        self.current_target_df.loc[self.current_target_df['qty'] == 0.0, 'position'] = 0.0

        total_notional = self.current_target_df['position'].abs().sum()
        self.current_target_df['unrealized_return'] = 100.0 * self.current_target_df['unrealized_profit'] / total_notional
        self.current_target_df['ret_since_opt'] = (self.current_target_df['current_px'] - self.current_target_df['opt_px']) / self.current_target_df['opt_px']
        self.current_target_df['vwap_shortfall'] = (self.current_target_df['avg_cost'] - self.current_target_df['vwap']) * self.current_target_df['execution_qty']

        long = self.current_target_df[self.current_target_df['position'] > 0]['position'].sum()
        short = self.current_target_df[self.current_target_df['position'] < 0]['position'].sum()
        total = long - short
        if total == 0:
            logger.error("Setting dataframe positions to zero!!!!", key="positions set to zero")
            print(self.current_target_df[['position', 'current_px', 'qty', 'symbol']])
        else:
            logger.info(f"Updated dataframe positions to ${long:.0f}/${short:.0f}")

    def get_latest_targets(self, opt_only: bool) -> Optional[Tuple[pd.DataFrame, float, bool]]:
        logger.info(f"Getting latest target files {opt_only=}")
        target_files = get_latest_target_files(curr_date=self.today, opt_only=opt_only)

        if len(target_files) == 0:
            logging.warning(f"No target files found for {self.today}")
            return None

        if not self.trade_on_startup and self.last_target_file_ts is not None:
            logger.info(f"Getting targets modified after {self.last_target_file_ts}")
            target_files = [f for f in target_files if dt.fromtimestamp(os.path.getmtime(f), tz=timezone.utc) > self.last_target_file_ts]
            if len(target_files) == 0:
                if (dt.now(timezone.utc) - self.last_target_file_ts).seconds > self.reoptimize_interval_secs:
                    logger.warning(f"Should have target file by now given our reoptimize interval!! last target: {self.last_target_file_ts}")
                logger.info(f"No targets found after {self.last_target_file_ts}")
                return None
        else:
            logger.info("Trading on startup...")
            self.trade_on_startup = False

        latest_target_file = sorted(target_files, reverse=True, key=lambda x: os.path.getmtime(x))[0]
        logger.info(f"Loading {latest_target_file}...")
        df = pd.read_csv(latest_target_file)
        df = shrink_floats(df)
        df = df.drop(columns=['lbound', 'ubound'], errors='ignore')
        df = df.rename(columns={'position': 'opt_position', 'close_mid': 'opt_px'})

        # why is this needed???? everything breaks without it so fix
        df['current_px'] = df['opt_px']
        df['ts'] = to_datetime(df['ts'])
        df = make_symbol(df)
        df = df.set_index('symbol_venue')

        reopt = 'opt' in latest_target_file

        portfolio_size = df['target_position'].abs().sum()
        if reopt and (pd.isna(portfolio_size) or portfolio_size == 0):
            self._message("Zero portfolio generated! not trading it!", error=True, msg_key="zero portfolio generated")
            return None

        target_file_age_seconds = (dt.now(timezone.utc) - dt.fromtimestamp(os.path.getmtime(latest_target_file), tz=timezone.utc)).total_seconds()

        return df, target_file_age_seconds, reopt

    def get_limit_price(self, symbol: str, side: Side, aggression: int) -> Optional[Dict[str, float]]:
        try:
            sym = self.aggregator.current_book[symbol]
        except KeyError:
            logger.warning(f"Could not get current quote for {symbol}")
            return None

        limit = None
        limit_size = None
        if side == Side.BUY:
            if aggression == 0:
                limit = sym['bidpx0']
                limit_size = sym['bidsz0']
            elif aggression == 1:
                limit = (float(sym['bidpx0']) + float(sym['askpx0'])) * .5
                limit_size = (float(sym['bidsz0']) + float(sym['asksz0'])) * .5
            elif aggression < -9:
                # in no mans land!
                tick = abs(float(sym['bidpx0']) - float(sym['bidpx1']))
                limit = float(sym['bidpx9']) - (tick * (abs(aggression) - 9))
                limit_size = float(sym['bidsz9'])  # approximate
            elif aggression < 0:
                limit = sym[f'bidpx{-aggression}']
                limit_size = sym[f'bidsz{-aggression}']
            elif aggression > 1:
                limit = sym[f'askpx{aggression - 2}']
                limit_size = sym[f'asksz{aggression - 2}']
            else:
                raise RuntimeError(f"Impossible {aggression=}")
        elif side == Side.SELL:
            if aggression == 0:
                limit = sym['askpx0']
                limit_size = sym['asksz0']
            elif aggression == 1:
                limit = (float(sym['bidpx0']) + float(sym['askpx0'])) * .5
                limit_size = (float(sym['bidsz0']) + float(sym['asksz0'])) * .5
            elif aggression < -9:
                # in no mans land!
                tick = abs(float(sym['askpx0']) - float(sym['askpx1']))
                limit = float(sym['askpx9']) + (tick * (abs(aggression) - 9))
                limit_size = float(sym['asksz9'])
            elif aggression < 0:
                limit = sym[f'askpx{-aggression}']
                limit_size = sym[f'asksz{-aggression}']
            elif aggression > 1:
                limit = sym[f'bidpx{aggression - 2}']
                limit_size = sym[f'bidsz{aggression - 2}']
            else:
                raise RuntimeError(f"Impossible {aggression=}")

        ret = {
            'bid': float(sym['bidpx0']),
            'ask': float(sym['askpx0']),
            'limit': float(limit),
            'limit_size': float(limit_size)
        }
        return ret

    def _get_orders_from_symbol(self, symbol: str) -> List[Tuple[dt, Order]]:
        orders = []
        for oid, order in self.outstanding_orders.items():
            if order.symbol == symbol:
                orders.append((order.acked_ts, order))
        # sort by order acked_ts so earlier orders have better pos in queue
        orders.sort(key=lambda x: x[0])
        return orders

    def cancel_oid(self, oid: str) -> bool:
        if oid not in self.outstanding_orders:
            logger.warning(f"Trying to cancel unknown {oid=}")
            return False

        order = self.outstanding_orders[oid]
        if self.old_oms:
            ret = self.send(f"CANCEL {oid}")
        else:
            msg = order.cpp_oms_cancel_order_msg(oid)
            ret = self.send(msg)

        if self.paper_trade:
            safe_del_from_dict(self.outstanding_orders, oid)
            safe_del_from_dict(self.pending_orders, order.koid)
            safe_del_from_dict(self.oid_to_koid, oid)
            self.orders_file.write(f"CANCEL|{order.symbol}|{oid}|{order.remaining_qty}|{order.px}\n")
        return ret

    def cancel_all(self, symbols: List[str] = None):
        if symbols is None:
            symbols = self.symbols
            self.pending_orders = {}

        for symbol in symbols:
            msg = f"CANCEL-ALL {symbol}" if self.old_oms else Order.cpp_oms_cancel_all_open_orders_msg(symbol)
            self.send(msg)
            if self.paper_trade:
                orders = self._get_orders_from_symbol(symbol)
                for acked_ts, order in orders:
                    self.orders_file.write(f"CANCEL|{symbol}|{order.oid}|{order.remaining_qty}|{order.px}\n")
                    safe_del_from_dict(self.outstanding_orders, order.oid)
                    safe_del_from_dict(self.pending_orders, order.koid)
                    safe_del_from_dict(self.oid_to_koid, order.oid)

    def create_flatten_order(self, symbol: str, aggression: int) -> Optional[Order]:
        qty = self.pos_recorder.get_qty(symbol)
        logger.info(f"Position qty in {symbol} is {qty}, flattening...")
        if qty == 0:
            return None

        order = self.create_order(
            symbol=symbol,
            trade_qty=-qty,
            current_qty=qty,
            aggression=aggression,
            liquidate_only=True
        )
        self._message(str(order))
        self.send_order(order)
        return order

    def send_order(self, order: Order):
        if order is None:
            logger.warning("Null order!")
            return

        if self.debug:
            logger.info(f"not sending {order}")
            return

        if self.paper_trade:
            # for paper_trade order, koid = oid in make_paper_order()
            order.make_paper_order()
            self.outstanding_orders[order.oid] = order
            self.oid_to_koid[order.oid] = order.koid
            self.orders_file.write(order.order_file_ln())
            return

        if order.symbol not in self.tick_size:
            logger.warning(f"Could not look up tick size symbol {order.symbol} not sending order....")
            return

        if self.old_oms:
            msg = order.oms_create_order_msg(precision=self.tick_size[order.symbol])
        else:
            msg = order.cpp_oms_create_order_msg(precision=self.tick_size[order.symbol])

        if msg is None:
            logger.warning(f"Not sending order {order}")
            return

        if self.send(msg):
            # reset to True when we really send a order msg
            self.no_ack_in_current_wave = True
            self.pending_orders[order.koid] = order

    def flatten_all(self, aggression: int = 0):
        for symbol in self.symbols:
            order = self.create_flatten_order(symbol, aggression)
            self.send_order(order)
        if self.paper_trade:
            self.outstanding_orders = {}
            self.oid_to_koid = {}

    def calc_deficit_pct(self, symbol: str, amt_done: float, orig_trade_amt: float) -> float:
        assert orig_trade_amt != 0

        frac_done = amt_done / orig_trade_amt
        frac_expected_done = self.fraction_of_period_elapsed - self.step_frac
        deficit = frac_done - frac_expected_done
        logger.info(f"DeficitCalc: {symbol=} {orig_trade_amt=:.0f} {amt_done=:.0f} {frac_done=:.2f} {frac_expected_done=:.2f} deficit: {fpct(deficit)}")
        # positive is over expanded, negative is under expanded
        return deficit

    def create_order(
            self,
            symbol: str,
            aggression: int,
            current_qty: float,

            trade_amt: Optional[float] = None,
            trade_qty: Optional[float] = None,

            deficit_pct: Optional[float] = None,
            current_px: Optional[float] = None,
            liquidate_only: bool = False,
            opt_px: Optional[float] = None,
            order_alpha_dict: Optional[Dict[str, float]] = None
    ) -> Optional[Order]:

        assert (trade_qty is None) or (trade_amt is None)

        if order_alpha_dict is None:
            order_alpha_dict = {}

        aggression = Order.bound_aggression(aggression)
        if trade_amt is not None:
            side = Side.from_float(trade_amt)
        else:
            side = Side.from_float(trade_qty)

        if self.aggregator is not None:
            limit_price_info = self.get_limit_price(symbol=symbol, side=side, aggression=aggression)
            if limit_price_info is None:
                logger.error(f"No book information on {symbol} for order, not sending...", key="no book information")
                return None

            limit = limit_price_info['limit']
            if limit is None:
                if current_px is None:
                    logger.error(f"Can't get current or bar price for {symbol}!", key="no current or bar price")
                    return None
                limit = current_px
            assert limit > 0
            limit_size = limit_price_info['limit_size']
        else:
            limit_price_info = None
            limit = current_px
            limit_size = None

        qty = trade_qty if trade_qty is not None else trade_amt / limit
        if abs(qty) > limit_size:
            logger.warning(f"Order qty {qty} is larger than limit size {limit_size} for {symbol}, not clipping...")
            sz = max(limit_size * 2.0, abs(qty) / .33)
            qty = np.sign(qty) * sz

        # order.px = self.round_to_tick_size(px, signed_qty, tick_size), so we need to do the same for the max_order_dollars check below
        tick_adjusted_limit = Order.round_to_tick_size(px=limit, signed_qty=qty, tick_size=self.tick_size.get(symbol, {}).get('px_tick_size'))
        if abs(qty * tick_adjusted_limit) > self.max_order_dollars:
            logger.warning(f"Order on {symbol} {qty * limit} > {self.max_order_dollars}, clipping....")
            qty = np.sign(qty) * (self.max_order_dollars * .9) / limit

        binance_max_order_qty = get_nested_dict_value(d=self.tick_size, outer_key=symbol, inner_key='max_qty', default_val=None)
        if binance_max_order_qty is not None and abs(qty) > binance_max_order_qty:
            logger.warning(f"Order qty {qty} is larger than max qty from binance {binance_max_order_qty} for {symbol}, clipping to {binance_max_order_qty * .9}...")
            qty = np.sign(qty) * (binance_max_order_qty * .9)

        if aggression > 0:
            qtytmp = qty
            if qty > 0 > current_qty:
                qtytmp = min(-current_qty, qty)
            elif qty < 0 < current_qty:
                qtytmp = max(-current_qty, qty)

            if qty != qtytmp:
                logger.info(f"Reducing qty to hit 0 {qty} -> {qtytmp} on {symbol} {aggression=}")
                qty = qtytmp

        order = Order(
            symbol=symbol,
            pending=True,
            signed_qty=qty,
            px=limit,
            aggression=aggression,
            qty_at_time_of_order=current_qty,
            px_at_time_of_order=current_px,
            deficit=deficit_pct,
            opt_px=opt_px,
            tick_size=self.tick_size.get(symbol, {}).get('px_tick_size'),
            order_alpha_dict=order_alpha_dict
        )

        logger.info(f"Creating order {side=} {symbol=} @ {limit=}, {aggression=} with abs qty: {abs(qty)}, order book available size: {limit_size}")

        if liquidate_only and not order.liquidation():
            logger.error(f"Liquidation Orders Only! Current order is not liquidating, {str(order)}", key="liquidation only")
            raise RuntimeError()

        if limit_price_info is not None:
            order.set_bid_ask(
                bid=limit_price_info['bid'],
                ask=limit_price_info['ask']
            )
        return order

    def manual_order_from_str(self, order_str: str) -> Order:
        # "B ETHUSDT 0.01 @ 2500 [LIMIT GTX]"
        order = Order.from_str(ostr=order_str)
        self.connect_to_market_data(symbols=[order.symbol])
        quotes = self.aggregator.current_book[order.symbol]
        order.set_bid_ask(
            bid=float(quotes['bidpx0']),
            ask=float(quotes['askpx0'])
        )
        logger.info(f"Current market: {order.bid} / {order.ask}")
        if not order.check_market_price():
            raise RuntimeError("Not Sending marketable order!")

        self.send_order(order)
        return order

    def send(self, msg: str) -> bool:
        if self.debug:
            logger.info("DEBUG MODE -- NOT Sending: " + msg)
            return True

        if not PROD:
            logger.warning("PROD not set and not DEBUG -- NOT Sending: " + msg)
            return True

        if self.local_oms:
            logger.info(f"PROD LOCALOMS -- Actually Sending: {msg}")
            return self._local_oms_sender.send(msg)

        if self.oms_pub_socket is None:
            logger.error("No OMS channel open!", key="null OMS socket")
            return False

        if self.oms_pub_socket.closed:
            logger.error("Closed OMS channel open!", key="no oms open")
            return False

        logger.info(f"PROD {'OLDOMS' if self.old_oms else 'CPPOMS'}-- Actually Sending: {msg}")
        if self.old_oms:
            self.oms_pub_socket.send_pyobj(msg)
        else:
            self.oms_pub_socket.send_string(msg)

        return True

    def send_lst(self, msgs: List[str]) -> bool:
        logger.info(f"Sending {len(msgs)} orders to OMS")
        ret = True
        for msg in msgs:
            if not self.send(msg):
                ret = False
        return ret

    def liquidate_big_move(self, symbol: str, aggression: int = 4):
        self.cancel_all([symbol])
        self.send_order(self.create_flatten_order(symbol, aggression=aggression))

    def fill_order(self, fill: Fill) -> None:
        order = None
        try:
            order = self.outstanding_orders[fill.oid]
        except KeyError:
            self._message(f"Filling unknown order {fill.oid} on {fill.symbol} at {fill.px}", error=True, msg_key="unknown order fill")

        commission_asset_price = self.get_current_mid(fill.commission_asset + "USDT") if fill.commission_asset != "USDT" else 1.0
        fill.calc_commission_usd({fill.commission_asset: commission_asset_price})

        self.pos_recorder.add_fill(fill)

        if order is not None:
            order.fill(fill)
            if order.remaining_qty == 0:
                safe_del_from_dict(self.outstanding_orders, order.oid)
                safe_del_from_dict(self.oid_to_koid, order.oid)

                old_aggression = self.get_aggression(fill.symbol)
                new_aggression = self.decrease_aggression(fill.symbol)
                logger.info(f"Order completely filled {fill.symbol}, aggression {old_aggression} -> {new_aggression}")

            fill_slip = fill.fill_slip()
            if fill_slip is not None:
                self.fill_slip_dollars += fill_slip

        self.order_problem_cnt[fill.symbol] = 0

        self.dollars_traded_this_opt += abs(fill.notional())
        self.current_wave_fills[fill.symbol].append(fill)
        self.fills_file.write(str(fill) + "\n")
        self.fills_file.flush()

    def log_raw_oms_message(self, ts: dt, msg: Dict[str, Any]) -> None:
        msg['live_ts'] = dt_to_millis(ts)
        self.raw_oms_file.write(str(msg) + "\n")

    def process_oms_messages(self):
        # https://binance-docs.github.io/apidocs/futures/en/#event-order-update
        msgs = self.oms_listener.get_msgs()
        for ts, msg in msgs:
            if msg.get('typ') == 'NEW_ORDER_OID_MAPPING':
                self.oid_to_koid[msg['oid']] = msg['koid']
                logger.info(f"New Order Mapping Found: {msg['symbol']} koid {msg['koid']} -> oid {msg['oid']}")
                order = self.pending_orders.get(msg['koid'])
                if order is not None and order.created_ts is not None and (dt.now(timezone.utc) - order.created_ts).total_seconds() > ORDER_ACK_LATENCY_SECONDS:
                    logger.error(f"OMS latency found for {msg['symbol']} koid {msg['koid']} -> oid {msg['oid']} sent at {order.created_ts}", key=f"OMS latency over {ORDER_ACK_LATENCY_SECONDS} secs")
                continue
            elif msg.get('typ') == 'EXCHANGE_REJECTION_MSG':
                self._message(f"Msg from exchange for order rejection {msg}")
                continue
            self.log_raw_oms_message(ts, msg)
            try:
                om = msg['o']
                execution_type = om['x']
                order_status = om['X']
            except KeyError as ke:
                self._message(f"ERROR: Could not parse order message {msg} since {ke}", error=True, msg_key='not parse order msg', error_trigger_threshold=2)
                continue

            if execution_type == 'NEW' and order_status == 'NEW':
                # logger.info(str(om))
                acked_order = Order.from_binance_json(msg, ts)
                if acked_order is None:
                    return
                logger.info(f"Order Accepted: {acked_order}")
                self.no_ack_in_current_wave = False

                if acked_order.oid in self.oid_to_koid:
                    koid = self.oid_to_koid[acked_order.oid]
                    if koid in self.pending_orders:
                        pending_order = self.pending_orders[koid]
                        acked_order.update_from_pending_order(pending_order)
                        safe_del_from_dict(self.pending_orders, koid)
                    else:
                        logger.warning(f"Could not find pending order for {acked_order.symbol} order oid={acked_order.oid} koid={acked_order.koid} in pending_orders: {self.pending_orders.keys()}")
                else:
                    logger.warning(f"Could not find pending order for {acked_order.symbol} order oid={acked_order.oid} koid={acked_order.koid} in oid_to_koid mapping")

                self.outstanding_orders[acked_order.oid] = acked_order
                self.orders_file.write(acked_order.order_file_ln())

            elif execution_type == 'TRADE' and order_status in ('FILLED', 'PARTIALLY_FILLED'):
                fill = Fill.from_binance_json(msg, ts)
                if fill is not None:
                    self.fill_order(fill)
            elif execution_type == 'CANCELED' and order_status in ('CANCELED', 'EXPIRED'):
                # logger.info(str(om))
                cancel = Cancel.from_binance_json(msg, ts)
                logger.info(f"acked cancelled order on {cancel.short_description()}")

                order = None
                try:
                    order = self.outstanding_orders[cancel.oid]
                except KeyError as ke:
                    logger.warning(f"Got cancel message on unknown oid: {cancel.oid} on {cancel.symbol}")

                if order is None:
                    if cancel.oid in self.oid_to_koid:
                        koid = self.oid_to_koid[cancel.oid]
                        safe_del_from_dict(self.pending_orders, koid)
                        safe_del_from_dict(self.oid_to_koid, cancel.oid)
                    else:
                        logger.warning(f"Got cancel message on unknown oid: {cancel.oid} on {cancel.symbol}, but cannot find from outstanding orders or oid mapping")
                else:
                    cancel.update_from_order(order)
                    safe_del_from_dict(self.outstanding_orders, order.oid)
                    safe_del_from_dict(self.oid_to_koid, order.oid)
                self.orders_file.write(cancel.order_file_ln())
            else:
                logger.warning(f"Unknown OMS message: {om}")

    def paper_fill_orders(self):
        logger.info("Paper Filling Orders...")
        for oid, order in list(self.outstanding_orders.items()):
            try:
                book = self.aggregator.current_book[order.symbol]
            except KeyError:
                logger.warning(f"No book for {order.symbol}, can't fill!")
                continue

            last_mid = (book['askpx0'] + book['bidpx0']) * .5
            logger.info(f"Open order {order.side} {order.remaining_qty} {order.symbol}@{order.px} current mid: {last_mid}")

            if (order.side == Side.BUY and book['askpx0'] <= order.px) or (order.side == Side.SELL and book['bidpx0'] >= order.px):
                self.fill_order(Fill(
                    symbol=order.symbol,
                    side=order.side,
                    px=order.px,
                    qty=order.remaining_qty,
                    fill_type=FillType.FAKE,
                    oid=order.oid
                ))

    def log_theoretical_order_adjustment(self, order: Optional[Order], existing_orders: List[Tuple[dt, Order]]) -> Optional[Order]:
        if order is None:
            cancel_existing_msg = "Since new order is None, cancel existing"
            for acked_ts, existing_order in existing_orders:
                cancel_existing_msg += f" {existing_order.koid} {existing_order.oid} {existing_order.side} {existing_order.symbol} {existing_order.remaining_qty} @ {existing_order.px},"
            logger.info(cancel_existing_msg)
        elif len(existing_orders) > 0:
            acked_ts, existing_order = existing_orders[0]
            # if order px or side differs, simply cancel all
            if not math.isclose(order.px, existing_order.px) or order.side != existing_order.side:
                cancel_existing_msg = f"Since new order px {order.px} or side {order.side} differs, cancel existing"
                for acked_ts, existing_order in existing_orders:
                    cancel_existing_msg += f" {existing_order.koid} {existing_order.oid} {existing_order.side} {existing_order.symbol} {existing_order.remaining_qty} @ {existing_order.px},"
                logger.info(cancel_existing_msg)
            else:
                total_qty_existing_order = 0
                pending_cancel_oids = []
                cancel_existing_msg = f"Since new order qty {order.orig_qty} not enough, cancel existing"
                existing_order_msg = ""
                for acked_ts, existing_order in existing_orders:
                    total_qty_existing_order += existing_order.remaining_qty
                    if order.orig_qty < total_qty_existing_order:
                        pending_cancel_oids.append(existing_order.oid)
                        existing_order_msg += f" {existing_order.koid} {existing_order.oid} {existing_order.side} {existing_order.symbol} {existing_order.remaining_qty} @ {existing_order.px}, "
                if len(pending_cancel_oids) == 0:
                    logger.info(f'Have existing order{existing_order_msg}, so adjust same px new order {order.koid} {order.short_description()} qty to {order.orig_qty - total_qty_existing_order}')
                    # order.orig_qty -= total_qty_existing_order
                    # order.remaining_qty -= total_qty_existing_order
                else:
                    logger.info(cancel_existing_msg + existing_order_msg)
                    # for oid in pending_cancel_oids:
                    # self.cancel_oid(oid)
        return order

    def calc_order_precision_round_amount(self, symbol: str, current_px: float) -> float:
        return max(self.min_trade_amt, max(0, 10 ** (-self.tick_size.get(symbol, {}).get('qty', 0)) * current_px / 2))

    def update_liquidate_target_positions(self, target_df: pd.DataFrame) -> pd.DataFrame:
        # override target_position to the position proportional to the elapsed time
        for liquidate_case in self.liquidate_list:
            symbol = liquidate_case['symbol']
            liquidate_start_dt = liquidate_case['liquidate_start_dt']
            liquidate_hour = liquidate_case['liquidate_hour']
            liquidate_idx = target_df['symbol'] == symbol
            if not liquidate_idx.any():
                continue
            elapsed_hours = (dt.now(timezone.utc) - liquidate_start_dt).total_seconds() / 3600
            remaining_proportion = max(0, 1 - (elapsed_hours / liquidate_hour))
            target_df.loc[liquidate_idx, 'target_position'] = liquidate_case['liquidate_init_pos'] * remaining_proportion
            logger.info(f"Liquidate {symbol} to {target_df.loc[liquidate_idx, 'target_position']} from {target_df.loc[liquidate_idx, 'position']}, start as {liquidate_start_dt}, within {liquidate_hour} hours")
            liquidate_target_df = target_df.loc[liquidate_idx, ['symbol', 'position', 'target_position', 'opt_position', 'qty', 'current_px', 'unrealized_profit']].sort_values(by='position')
            self.send_large_dataframe_to_slack(liquidate_target_df)
        return target_df

    def check_trade_amount(self, df: pd.DataFrame):
        orig_trade_amt = df['orig_trade_amt'].abs().sum()
        target_notional = df['target_position'].abs().sum()
        current_notional = df['position'].abs().sum()
        target_bias_abs = abs(df['target_position'].sum())

        if self.max_dollars_per_opt is not None and orig_trade_amt > self.max_dollars_per_opt:
            error_msg = f"trade list {orig_trade_amt} > max dollars per opt {self.max_dollars_per_opt}, {target_notional=} {current_notional=}"
            # logger.error(error_msg, key='trade list amount greater than max dollars per opt (still sending)')
            logger.warning(error_msg)
            #XXX TODO do something other than just alert?

        if target_bias_abs / target_notional > self.max_trading_bias:
            error_msg = f"Seeing trade list target bias vs target notional {target_bias_abs}/{target_notional} out of max trading bias {self.max_trading_bias}"
            logger.error(error_msg, key='trade list bias ratio out of bound')


    def optimize(self, target_df: pd.DataFrame, wave_notional: float) -> pd.DataFrame:
        n = len(target_df)
        w = cp.Variable(n)
        kappa = cp.Parameter(nonneg=True)
        kappa.value = self.st_kappa
        corr_mat = np.eye(n)
        std_diag = np.diag(target_df['risk_st'].values)
        sigma = std_diag @ corr_mat @ std_diag
        delta = w - target_df['target_position'].values
        resid_risk = cp.quad_form(delta, sigma)
        factor_loadings, factor_cov = PortfolioOptimizer.generate_factor_matrix(target_df, self.st_factor_sigmas)
        factor_risk = cp.quad_form(factor_loadings.T @ w, factor_cov)
        total_risk = factor_risk + resid_risk

        current_factor_risk = cp.quad_form(factor_loadings.T @ target_df['trade_amt'], factor_cov)
        current_resid_risk = cp.quad_form(target_df['trade_amt'], sigma)
        logger.info(f"{current_resid_risk.value=} {current_factor_risk.value=}")

        logger.info("Current Factor Exposures")
        PortfolioOptimizer.check_factor_exposures(factor_sigmas=self.st_factor_sigmas, factor_loadings=factor_loadings, targets_s=target_df['position'])

        traded_dollars = cp.sum(cp.abs(target_df['position'].values - w.T))

        mu = target_df['alpha_st'].values
        lbound = (target_df['lbound']).values
        ubound = (target_df['ubound']).values

        assert len(target_df[target_df['lbound'] > target_df['ubound']][['lbound', 'ubound']]) == 0

        ret = mu.T @ w
        obj_func = ret - (kappa * total_risk)
        prob = cp.Problem(
            cp.Maximize(obj_func),
            [
                w <= ubound,
                w >= lbound,
                traded_dollars <= wave_notional
            ])

        solved = False
        try:
            prob.solve(verbose=True, solver='SCS')
            solved = True
        except cvxpy.error.SolverError as se:
            self._message(f"Trade Optimizer failed: {str(se)}", msg_key="trade optimizer solver error")

        if not solved or prob.status not in ('optimal', 'solved'):
            self._message(f"Trade Optimizer failed, status: {prob.status}", msg_key="trade optimizer not solved")
            target_df['st_target_position'] = target_df['position']
        else:
            logger.info("Optimization successful")
            target_df['st_target_position'] = w.value

        target_df.loc[target_df['st_target_position'].abs() < self.min_trade_amt, 'st_target_position'] = 0.0
        target_df['st_trade_amt'] = target_df['st_target_position'] - target_df['position']

        small_st_trade_idx = target_df['st_trade_amt'].abs() < self.min_trade_amt
        too_small_total = target_df[small_st_trade_idx]['st_trade_amt'].abs().sum()
        logger.info(f"Removing {fmoney(too_small_total)} in small optimized trades...")
        target_df.loc[small_st_trade_idx, 'st_trade_amt'] = 0.0

        trades_df = target_df[target_df['st_trade_amt'] != 0]
        if len(trades_df) > 0:
            trades_msg_df = trades_df[['symbol', 'trade_amt', 'st_trade_amt', 'position', 'alpha_st', 'risk_st', 'fraction_remaining', 'expanding', 'aggression']].sort_values(by='trade_amt')
            self.send_large_dataframe_to_slack(trades_msg_df)
        else:
            logger.warning("No Optimized trades!")
        expanding_dollars = trades_df[trades_df['expanding'] == 1]['st_trade_amt'].sum()
        contracting_dollars = trades_df[trades_df['expanding'] == -1]['st_trade_amt'].sum()
        traded_dollars = trades_df['st_trade_amt'].abs().sum()
        st_bias = target_df['st_target_position'].sum()

        logger.info("ST Opt Factor Exposures")
        PortfolioOptimizer.check_factor_exposures(factor_sigmas=self.st_factor_sigmas, factor_loadings=factor_loadings, targets_s=target_df['st_target_position'])
        # st_delta = target_df['st_target_position'] - target_df['position']
        try:
            opt_factor_risk = cp.quad_form(factor_loadings.T @ w, factor_cov).value
            opt_resid_risk = cp.quad_form(delta, sigma).value
            opt_tot_risk = opt_resid_risk + opt_factor_risk
            logger.info(f"{opt_resid_risk=} {opt_factor_risk=} {opt_tot_risk=}")
        except Exception as e:
            logger.warning(f"Could not calculate risk for trade optimmization {e} ... not critical")
            logger.warning(f"{factor_loadings}")
            logger.warning(f"f{factor_cov}")
            logger.warning(f"{delta}")

        logger.info(f"{traded_dollars=}\n{st_bias=}")
        logger.info(f"{expanding_dollars=} {contracting_dollars=}")

        return target_df

    def optimize_and_send_orders(self):
        target_df = self.current_target_df
        target_df = self.update_liquidate_target_positions(target_df)
        target_df['orig_trade_amt'] = target_df['target_position'] - target_df['opt_position']
        self.check_trade_amount(df=target_df)

        target_df['opt_qty'] = target_df['opt_position'] / target_df['opt_px']
        target_df['orig_trade_qty'] = target_df['orig_trade_amt'] / target_df['opt_px']
        target_df['target_qty'] = target_df['opt_qty'] + target_df['orig_trade_qty']
        target_df['dollars_done'] = target_df['position'] - target_df['opt_position']
        target_df['qty_done'] = target_df['qty'] - target_df['opt_qty']
        # in_play_idx = target_df['orig_trade_amt'] != 0.0

        buy_idx = target_df['orig_trade_amt'] > 0
        sell_idx = target_df['orig_trade_amt'] < 0
        mixed_idx = (target_df['target_position'] * target_df['opt_position']) < 0
        long_idx = target_df['position'] >= 0
        short_idx = target_df['position'] <= 0

        target_df['trade_qty'] = target_df['target_qty'] - target_df['qty']
        target_df['trade_amt'] = target_df['trade_qty'] * target_df['current_px']
        target_df['fixed_target_position'] = target_df['position'] + target_df['trade_amt']
        target_df['fixed_target_position2'] = target_df['target_qty'] * target_df['current_px']

        target_df['expanding'] = 0
        expanding_idx = (buy_idx & long_idx) | (sell_idx & short_idx)
        contracting_idx = (buy_idx & (target_df['position'] < 0)) | (sell_idx & (target_df['position'] > 0))
        target_df.loc[expanding_idx, 'expanding'] = 1
        target_df.loc[contracting_idx, 'expanding'] = -1

        target_df['expanding_trade_amt'] = 0.0
        target_df['contracting_trade_amt'] = 0.0
        target_df.loc[expanding_idx, 'expanding_trade_amt'] = target_df['fixed_target_position'] - target_df['position']
        target_df.loc[contracting_idx, 'contracting_trade_amt'] = target_df['fixed_target_position'] - target_df['position']

        #for trades crossing through zero, need to partition orders
        target_df.loc[mixed_idx, 'expanding_trade_amt'] = target_df['fixed_target_position']
        target_df.loc[mixed_idx, 'contracting_trade_amt'] = -target_df['position']
        target_df.loc[mixed_idx, 'trade_amt'] = target_df['contracting_trade_amt']
        target_df.loc[mixed_idx, 'trade_qty'] = target_df['trade_amt'] / target_df['current_px']

        target_df['trade_amt_abs'] = target_df['trade_amt'].abs()
        target_df['lbound'] = target_df['position']
        target_df['ubound'] = target_df['position']
        valid_px_idx = ~target_df['current_px'].isna()
        target_df.loc[buy_idx & valid_px_idx, 'ubound'] = target_df[['fixed_target_position2', 'position']].max(axis=1)
        target_df.loc[~buy_idx & valid_px_idx, 'lbound'] = target_df[['fixed_target_position2', 'position']].min(axis=1)

        target_df['net_expanding_dollars'] = target_df['trade_amt_abs'] * target_df['expanding']
        expanding_dollars = target_df['expanding_trade_amt'].abs().sum()
        contracting_dollars = target_df['contracting_trade_amt'].abs().sum()
        expanding_ratio = expanding_dollars / contracting_dollars

        target_df['fraction_done'] = remove_infs(target_df['qty_done'] / target_df['orig_trade_qty']).fillna(1.0)
        target_df['fraction_remaining'] = remove_infs(1.0 - (target_df['dollars_done'] / target_df['orig_trade_amt'])).fillna(0.0).clip(lower=0.0).astype(np.float32)

        # short term risk on the *remaining* portion.  also only consider risk when we are trading against alpha, else the vol is a positive and we want to be passive
        target_df['aggression'] = target_df['alpha_st'] * np.sign(target_df['trade_amt'])
        target_df['risk_st'] = np.float32(0.0)
        target_df.loc[target_df['aggression'] > 0, 'risk_st'] = target_df['risk_15'] * target_df['fraction_remaining']

        # print(target_df[in_play_idx][['trade_amt', 'orig_trade_amt', 'opt_position', 'target_position', 'position', 'target_position', 'fixed_target_position', 'dollars_done']])

        current_size = target_df['position'].abs().sum()
        target_size = target_df['target_position'].abs().sum()
        current_long = target_df[long_idx]['position'].sum()
        current_short = target_df[short_idx]['position'].sum()
        target_long = target_df[target_df['target_position'] > 0]['target_position'].sum()
        target_short = target_df[target_df['target_position'] < 0]['target_position'].sum()
        orig_trade_notional = target_df['orig_trade_amt'].abs().sum()
        remaining_trade_notional = target_df['trade_amt_abs'].sum()

        # self._message(target_df[['trade_amt', 'orig_trade_amt']].sort_values(by='orig_trade_amt').to_markdown())

        fraction_of_trade_dollars_done = (orig_trade_notional - remaining_trade_notional) / orig_trade_notional
        accelerated_order_pct = self.get_accelerated_order_pct()
        # remaining_time_percent = max(0.0, 1.0 - self.fraction_of_period_elapsed)
        remaining_waves = self.get_remaining_waves()
        remaining_waves = max(1, remaining_waves)
        current_bias = target_df['position'].sum()
        target_bias = target_df['target_position'].sum()
        # fraction_remaining = min(1.0, self.order_percent - fraction_of_trade_dollars_done)
        wave_notional = self.order_acceleration * (remaining_trade_notional / remaining_waves)
        assert wave_notional >= 0

        # self._message(target_df[(target_df['orig_trade_amt'] != 0)]['alpha_st'].describe().to_markdown())

        logger.info(f"{expanding_dollars=} {contracting_dollars=} {expanding_ratio=}")
        logger.info(f"{orig_trade_notional=}\n{remaining_trade_notional=}\n{fraction_of_trade_dollars_done=}")
        logger.info(f"{current_size=} {target_size=}")
        logger.info(f"{current_long=} {current_short=}")
        logger.info(f"{target_long=} {target_short=}")
        logger.info(f"{current_bias=} {target_bias=}")
        logger.info(f"{wave_notional=}\n{remaining_waves=}\n{accelerated_order_pct=}\n{self.fraction_of_period_elapsed=}")
        target_df['target_position_wave'] = target_df['target_position'] * self.fraction_of_period_elapsed

        try:
            target_df = self.optimize(target_df, wave_notional)
        except Exception as e:
            target_df['st_trade_amt'] = np.float32(0.0)

        orders = []
        unsent_amt = 0.0
        max_single_order_dollars = 0.0
        max_single_order_dollars_str = None
        max_total_order_dollars = 0.0
        long_trade_dollars = short_trade_dollars = 0.0

        processed_symbols = set()

        for idx, row in target_df.iterrows():
            trade_amt_abs = row['trade_amt_abs']
            symbol = row['symbol']

            # if self.max_dollars_per_opt is not None and self.dollars_traded_this_opt > self.max_dollars_per_opt:
            #     logger.warning(f"Traded {fmoney(self.dollars_traded_this_opt)} > {fmoney(self.max_dollars_per_opt)}, skipping order of {fmoney(trade_amt_abs)} on {symbol}")
            #     continue

            processed_symbols.add(symbol)
            existing_orders = self._get_orders_from_symbol(symbol)

            # XXXX CLEAN THIS UP
            model_alpha_str = safe_get_from_series(row, 'model_alpha_col')
            # model_alpha_str formatted as "alpha_{model1}_{horizon1}:alpha_value1,alpha_{model2}_{horizon2}:alpha_value2,alpha_{model3}_{horizon3}:alpha_value3"
            # so we parse it into order_alpha_dict with alpha_{model1}_{horizon1} as key and alpha_value1 as value
            order_alpha_dict = {}
            if model_alpha_str is not None:
                order_alpha_dict = {item.split(':')[0]: float(item.split(':')[1]) for item in model_alpha_str.split(',')}
            for horizon in self.model_horizons:
                order_alpha_dict[f'alpha_{horizon}'] = safe_get_from_series(row, f'alpha_{horizon}')
            # XXX

            if symbol in self.notrade_list:
                unsent_amt += trade_amt_abs
                logger.info(f"Not sending order for {symbol} since on no-trade list")
                continue

            trade_amt = row['trade_amt']
            trade_amt = max(min(trade_amt, self.max_order_dollars), -self.max_order_dollars)
            current_px = row['current_px']
            if np.isnan(current_px):
                if not np.isnan(trade_amt):
                    logger.warning(f"Bad price on {symbol}, not sending order for {fmoney(trade_amt)}")
                continue

            # when we apply qty precision, we will round it and could make the order less than min trade amount, so need this extra round amount to protect
            order_precision_round_amt = self.calc_order_precision_round_amount(symbol, current_px)
            current_qty = self.pos_recorder.get_qty(symbol)
            current_position = current_qty * current_px
            opt_px = row['opt_px']
            target_position = row['target_position']

            logret_1440_lz = row['logret_1440_lz']
            if (logret_1440_lz > self.max_move_filter) and (trade_amt < 0):
                trade_amt = max(trade_amt, -current_position)
                logger.info(f"Not going short on rocket on {symbol}, max trade amount {trade_amt}", key=f"rocket {symbol} {logret_1440_lz=}")

            if trade_amt_abs == 0:
                continue

            if trade_amt_abs < self.min_trade_amt + order_precision_round_amt:
                if target_position != 0.0:
                    logger.info(f"Not sending small order of {trade_amt_abs=} on {symbol}")
                    unsent_amt += trade_amt_abs
                    continue

                # aggressively close out dust
                order = self.create_order(
                    symbol=symbol,
                    current_px=current_px,
                    trade_amt=trade_amt,
                    current_qty=current_qty,
                    aggression=self.max_aggression,
                    opt_px=opt_px,
                    order_alpha_dict=order_alpha_dict
                )
            else:
                st_trade_amt = row['st_trade_amt']

                # our optimized orders...
                if st_trade_amt != 0:
                    aggression = self.get_aggression(symbol) + 3
                    order = self.create_order(
                        symbol=symbol,
                        current_px=current_px,
                        trade_amt=st_trade_amt,
                        current_qty=current_qty,
                        aggression=aggression,
                        opt_px=opt_px,
                        order_alpha_dict=order_alpha_dict
                    )
                else:
                    # alpha_st = row['alpha_st']
                    # trade_in_alpha_direction = (trade_amt * alpha_st) >= 0
                    # WRONG_DIRECTION_FRACTION = .05
                    # if not trade_in_alpha_direction:
                    #     wd_trade_amt = trade_amt * WRONG_DIRECTION_FRACTION
                    #     if abs(wd_trade_amt) < self.min_trade_amt:
                    #         wd_trade_amt = np.sign(trade_amt) * self.min_trade_amt
                    #     logger.info(f"Wrong ST Alpha on {symbol} {trade_amt} -> {wd_trade_amt}")
                    #     trade_amt = wd_trade_amt

                    fraction_done = row['fraction_done']
                    if fraction_done > self.fraction_of_period_elapsed:
                        logger.info(f"Not sending overfilled {symbol} {fraction_done=} > {self.fraction_of_period_elapsed=}")
                        continue

                    orig_trade_amt = row['orig_trade_qty'] * row['current_px']
                    scaled_trade_amt = (self.fraction_of_period_elapsed - fraction_done) * orig_trade_amt

                    if abs(scaled_trade_amt) > abs(trade_amt):
                        logger.warning(f"scaled amount more than trade amount on {symbol} {scaled_trade_amt=} {trade_amt=} {fraction_done=} {self.fraction_of_period_elapsed=}")
                        scaled_trade_amt = trade_amt

                    # assert abs(scaled_trade_amt) <= abs(trade_amt)
                    falling_behind_amt = self.fraction_of_period_elapsed - fraction_done
                    falling_behind = falling_behind_amt > .25
                    if falling_behind:
                        incr = 2
                        if falling_behind_amt > .50:
                            incr = 4
                        logger.info(f"Trading {symbol} since we are falling behind {falling_behind_amt} and increasing aggression {incr}")
                        self.increase_aggression(symbol=symbol, incr=incr)

                    # if not trade_in_alpha_direction and not falling_behind:
                    #     orig_amt = scaled_trade_amt
                    #     scaled_trade_amt = scaled_trade_amt * .05
                    #     logger.info(f"Reducing order size on {symbol} {orig_amt} -> {scaled_trade_amt} on {symbol}, {alpha_st=} in opposite direction")

                    # EOD reversal overlay: modulate speed and aggression
                    eod_aggression_boost = 0
                    if self.eod_manager is not None and self.eod_manager.is_active():
                        eod_speed, eod_aggression_boost = self.eod_manager.get_execution_modifier(symbol, scaled_trade_amt)
                        scaled_trade_amt *= eod_speed

                    scaled_trade_amt_abs = abs(scaled_trade_amt)

                    if scaled_trade_amt_abs > self.min_trade_amt + order_precision_round_amt:
                        order = self.create_order(
                            symbol=symbol,
                            current_px=current_px,
                            trade_amt=scaled_trade_amt,
                            current_qty=current_qty,
                            aggression=self.get_aggression(symbol) + eod_aggression_boost,
                            opt_px=opt_px,
                            order_alpha_dict=order_alpha_dict
                        )
                    else:
                        unsent_amt += scaled_trade_amt_abs
                        logger.info(f"Not sending order of {scaled_trade_amt} on {symbol}, too small...")
                        continue

            order = self.log_theoretical_order_adjustment(order, existing_orders)
            if order is not None:
                new_order_amount = abs(order.order_amt())
                max_total_order_dollars += new_order_amount
                if new_order_amount > max_single_order_dollars:
                    max_single_order_dollars = new_order_amount
                    max_single_order_dollars_str = str(order)
                orders.append(order)

            if trade_amt > 0:
                long_trade_dollars += trade_amt
            else:
                short_trade_dollars += trade_amt

        if max_single_order_dollars > (self.max_order_dollars + 1):
            logger.error(f"Seeing single order {max_single_order_dollars_str} trade amount {max_single_order_dollars} greater than {self.max_order_dollars}", key=f'max_order_dollars over {self.max_order_dollars}')
        elif max_total_order_dollars > self.max_total_order_dollars:
            logger.error(f"Seeing orders total trade amount {max_total_order_dollars} greater than {self.max_total_order_dollars}", key=f'max_total_order_dollars over {self.max_total_order_dollars}')
        else:
            # if not exceed limits, good to send all orders
            if len(orders) > 0:
                for order in orders:
                    self.send_order(order)

                logger.info(f"Sent {len(orders)} orders...")
                orders_df = self.format_orders_to_table(orders)
                buy_orders_df = orders_df.loc[orders_df["Order Amt"] > 0].sort_values("Order Amt", ascending=False)
                sell_orders_df = orders_df.loc[orders_df["Order Amt"] < 0].sort_values("Order Amt", ascending=True)
                buy_orders_df["Order Amt"] = buy_orders_df["Order Amt"].apply(lambda x: f'${x:,.2f}')
                sell_orders_df["Order Amt"] = sell_orders_df["Order Amt"].apply(lambda x: f'${x:,.2f}')
                self.send_large_dataframe_to_slack(buy_orders_df)
                self.send_large_dataframe_to_slack(sell_orders_df)

        self.current_target_df = target_df

        # cancel all other orders if symbol not processed
        leftover_symbols = [symbol for symbol in self.symbols if symbol not in processed_symbols]
        logger.info(f'Cancel all other orders if symbol not processed {leftover_symbols}')
        self.cancel_all(leftover_symbols)

        total_traded = orig_trade_notional - remaining_trade_notional
        pct_done = total_traded / orig_trade_notional if orig_trade_notional != 0 else 0
        target_age = dt.now(timezone.utc) - self.last_target_file_ts
        self._message(f"Trading ${long_trade_dollars:.0f} / ${short_trade_dollars:.0f} at {fpct(self.fraction_of_period_elapsed)}, Amount Done: ${total_traded:.0f}/${orig_trade_notional:.0f} ({fpct(pct_done)}), Unsent Amt: ${unsent_amt:.0f}")
        self._message(f"Target Age: {target_age} Target long/short: {fmoney(target_long)} / {fmoney(target_short)}")
        if pct_done > 1.02:
            logger.error(f"Seeing over trading, total_traded {total_traded}, orig_trade_notional {orig_trade_notional}, remaining_trade_notional {remaining_trade_notional}", key='over trading')

    def send_large_dataframe_to_slack(self, df, max_rows_per_message=20):
        total_rows = len(df)
        # Split into chunks and send multiple messages
        for start_idx in range(0, total_rows, max_rows_per_message):
            end_idx = min(start_idx + max_rows_per_message, total_rows)
            chunk_df = df.iloc[start_idx:end_idx]
            self._message(chunk_df.to_markdown(index=False))

    @staticmethod
    def format_orders_to_table(orders: List[Order]) -> pd.DataFrame:
        data = []
        for order in orders:
            order_dict = {
                "Symbol": order.symbol,
                "Order Amt": order.order_amt(),
                "Agg": order.aggression,
                "Px": order.px,
                "Bid": order.bid,
                "Ask": order.ask,
                "Orig Qty": order.orig_qty,
                "Curr Qty": order.qty_at_time_of_order,
                "Def": order.deficit,
            }
            data.append(order_dict)
        df = pd.DataFrame(data)
        format_rules = {
            "Orig Qty": "{:.4f}",
            "Px": "{:.6f}",
            "Curr Qty": "{:.0f}",
            "Bid": "{:.4f}",
            "Ask": "{:.4f}",
            "Agg": "{:.4f}",
            "Def": "{:.4f}",
        }
        for column, format_str in format_rules.items():
            df[column] = df[column].map(lambda x: format_str.format(x) if pd.notnull(x) else "")
        return df

    def get_accelerated_order_pct(self) -> float:
        return min(1.0, self.fraction_of_period_elapsed * self.order_acceleration)

    def report_open_orders(self):
        open_buys = open_sells = 0.0
        for order in self.outstanding_orders.values():
            if order is not None:
                notional = order.order_amt()
                if notional > 0:
                    open_buys += notional
                else:
                    open_sells += notional

        pnl_msg = f"Open Orders: {fmoney(open_buys)}/{fmoney(open_sells)} Fill Slip: ${self.fill_slip_dollars:.0f}"
        self._message(pnl_msg)

    def update_alphas(self, df: pd.DataFrame):
        if 'alpha_st' not in self.current_target_df:
            self.current_target_df['alpha_st'] = np.float32(0)
        avg_st_alpha_before = self.current_target_df['alpha_st'].abs().mean()
        st_alpha_cols = [f"alpha_{hh}" for hh in self.short_term_alphas]
        self.current_target_df.update(df[st_alpha_cols])
        # cut 15 min alpha in half since 7 mins to make output the file

        self.current_target_df['alpha_st'] = np.float32(0)
        for alpha_col in st_alpha_cols:
            self.current_target_df['alpha_st'] += self.current_target_df[alpha_col]

        avg_st_alpha_after = self.current_target_df['alpha_st'].abs().mean()
        logger.info(f"Updated ST alphas {avg_st_alpha_before} -> {avg_st_alpha_after}")

    def report_current_wave_fills(self):
        fills_messages = []
        for symbol, fills in self.current_wave_fills.items():
            for side in [Side.BUY, Side.SELL]:
                fill_info = get_agg_side_fills_info(symbol, fills, side)
                if fill_info:
                    fills_messages.append(fill_info)
        if fills_messages:
            fills_messages_df = pd.DataFrame(
                fills_messages,
                columns=['symbol', 'side', 'qty', 'avg_cost', 'num_trades', 'notional', 'slippage', 'slippage_bps']
            ).sort_values('notional', ascending=False)
            total_fills_notional = fills_messages_df['notional'].sum()
            self._message(f"5-minute Fills: {fmoney(total_fills_notional)}")
            self.send_large_dataframe_to_slack(fills_messages_df)

    def update_delisting_metadata(self):
        if self.metadata_date == self.today or dt.now(timezone.utc).hour <= 8:
            return
        logger.info("Updating delisting metadata")
        self.metadata_date = self.today
        self.metadata_df = load_metadata(start_date=self.today, end_date=self.today, metadata_dir=self.dir_manager.BINANCE_META_DIR)
        if self.metadata_df is not None:
            self.metadata_df = self.metadata_df.loc[self.metadata_df['availableTo'] > self.today, ['symbol', 'availableTo']].reset_index(drop=True)

    def report_unfilled_orders(self):
        slipd = {}
        slip_tot = unfilled_buy_tot = unfilled_sell_tot = 0
        now = dt.now(timezone.utc)
        lost_orders = []
        for oid, order in self.outstanding_orders.items():
            current_px = self.get_current_mid(order.symbol)
            sign = Side.sign(order.side)
            if current_px is not None and order.px_at_time_of_order is not None:
                slippage = (current_px - order.px_at_time_of_order) * order.remaining_qty * sign
            else:
                slippage = 0
            slip_tot += slippage
            order_amt = order.order_amt()
            if order_amt > 0:
                unfilled_buy_tot += order_amt
            else:
                unfilled_sell_tot += order_amt

            aggression = order.aggression
            if order.fill_cnt == 0:
                aggression = self.increase_aggression(order.symbol)

            slipd[abs(slippage)] = f"{order.symbol} ${order_amt:.0f} created: {dt_to_str(order.created_ts)} Id: {oid} Aggression: {order.aggression} -> {aggression} Slippage: ${slippage:.0f}"
            if order.created_ts is not None:
                age_mins = (now - order.created_ts).seconds / 60.0
                if age_mins > 60:
                    lost_orders.append(oid)

        if len(slipd) > 0:
            msgs = [f"UNFILLED ORDERS, Total Slippage: ${slip_tot:.0f} Unfilled Total ${unfilled_buy_tot:.0f}/${unfilled_sell_tot:.0f}", ]
            msgs += dict(sorted(slipd.items())).values()
            self._message("\n".join(msgs))

        for oid in lost_orders:
            logger.warning(f"Removing stale order: {oid}")
            safe_del_from_dict(self.outstanding_orders, oid)
            safe_del_from_dict(self.oid_to_koid, oid)

        # msgs = []
        # for koid, order in self.pending_orders.items():
        #     logger.warning(f"Unaccepted Order: {order}")
        #     msgs.append(f"Investigate, exchange unaccepted order {koid}: {order.short_description()}")
        #     self.order_problem_cnt[order.symbol] += 1
        # if len(msgs) > 0:
        #     self._message("\n".join(msgs))

    async def run_autotrader(self):
        self._message(f"Launching autotrader {self.debug=}, {self.paper_trade=}, {self.trade_on_startup=}, {self.market_data_only=}, {self.no_trade=}, {self.no_limits=}, wave-interval-mins={(self.wave_interval_seconds / 60.0):.0f} max_dollars_per_opt={fmoney(self.max_dollars_per_opt)}")

        self.connect_to_market_data()

        self.refresh_positions(from_local_file=self.paper_trade or self.debug, init=True)
        if not self.paper_trade and not self.debug and not self.market_data_only:
            self._create_oms_listener()

        self.update_symbols()

        start_date = dt.now(timezone.utc).date()
        start_up = True

        ii = 1
        while True:
            ii += 1
            self.today = today_date()
            if dt.now(timezone.utc).date() != start_date:
                self._message(f"Trader moving to {dt.now(timezone.utc).date()}")
                if not self.market_data_only:
                    self._open_files()
                start_date = dt.now(timezone.utc).date()
                if self.eod_manager is not None:
                    self.eod_manager.reset()
                # reset pnl

            self.update_delisting_metadata()

            if not self.debug and (self.aggregator.dead or self.quote_poller.too_many_errors()):
                logger.warning("Quote aggregator dead! Trying to restart!")
                await self.quote_poller.kill_me()
                self.quote_poller.join()
                self.connect_to_market_data()

            if self.market_data_only:
                logger.info(f"Trader runs market data only {self.market_data_only}, listening market data...")
                await asyncio.sleep(10)
                continue

            self.fraction_of_period_elapsed = min(self.fraction_of_period_elapsed + self.step_frac, 1.0)
            assert self.fraction_of_period_elapsed > 0

            new_targets_df_and_age = self.get_latest_targets(opt_only=start_up)
            if new_targets_df_and_age is not None:
                now = dt.now(timezone.utc)
                new_targets_df, target_file_age_seconds, reopt = new_targets_df_and_age
                max_target_ts = new_targets_df['ts'].max()
                if reopt:
                    self.dollars_traded_this_opt = 0.0
                    target_file_asof_age_seconds = (now - max_target_ts).total_seconds()
                    if target_file_asof_age_seconds > (self.reoptimize_interval_secs + TARGET_STALE_SECONDS):
                        logger.error(f"Latest opt target file updated at {max_target_ts}, {target_file_asof_age_seconds / 3600} hours ago", key="opt target file too stale")
                    fraction_passed = min(target_file_asof_age_seconds / self.reoptimize_interval_secs, 1)
                    self.fraction_of_period_elapsed = max(fraction_passed, self.step_frac)
                    logger.info(f"Generating new targets at {fpct(self.fraction_of_period_elapsed)}")

                    self.last_target_file_loaded_ts = now
                    self.last_target_file_ts = max_target_ts
                    self.current_target_df = new_targets_df.copy()
                    self.current_aggression = {}

                    if not self.debug:
                        self.aggregator.reset_vwap()
                        self.pos_recorder.reset_avg_cost()

                    start_up = False
                else:
                    logger.info("Got non-opt alphas, updating...")
                    print(new_targets_df)

                self.update_alphas(new_targets_df)
            else:
                if self.current_target_df is None:
                    logger.info("Waiting for first target...")
                    self.pos_recorder.dump_positions()
                    time.sleep(60)
                    continue
                else:
                    logger.info(f"Re-sending targets at {fpct(self.fraction_of_period_elapsed)}")

            # TODO: only cancel if need be...
            if not self.debug:
                self.report_unfilled_orders()
                self.cancel_all()
                # XXX: sleep to avoid cancel place timing issue
                time.sleep(2)

            if not self.paper_trade:
                self.refresh_positions(from_local_file=False)
            self.update_df_positions_and_prices()

            # EOD reversal overlay: capture at 23:45, compute signal at 00:00
            if self.eod_manager is not None:
                now_utc = dt.now(timezone.utc)
                if self.eod_manager.should_capture(now_utc):
                    n_cap = self.eod_manager.capture_reference_prices(
                        symbols=self.symbols,
                        get_mid_fn=self.get_current_mid,
                        target_df=self.current_target_df,
                    )
                    self._message(f"EOD Reversal: captured {n_cap} reference prices")

                if self.eod_manager.should_compute_signal(now_utc):
                    self.eod_manager.compute_signal(
                        symbols=self.symbols,
                        get_mid_fn=self.get_current_mid,
                    )
                    self._message(self.eod_manager.get_signal_summary())

            if self.no_trade:
                logger.info("Skip optimizing or sending orders in no trade mode")
            else:
                self.optimize_and_send_orders()

            if self.debug:
                if self.quote_poller is not None:
                    self.quote_poller.kill_me()
                break

            self.pos_recorder.dump_positions()
            self.report_open_orders()
            self.current_wave_fills.clear()
            if ii % 2 == 0:
                self.report_positions()

            next_wave = time.time() + self.wave_interval_seconds
            logger.info(f"Sending next wave at {dt.fromtimestamp(next_wave, tz=timezone.utc)}")
            while time.time() < next_wave:
                if self.paper_trade:
                    self.paper_fill_orders()
                else:
                    self.process_oms_messages()
                    self.raw_oms_file.flush()

                self.orders_file.flush()
                self.fills_file.flush()

                if self.aggregator.dead:
                    self._message("Data aggregator died multiple times, keep try restarting...", error=True, msg_key='data aggregator died multiple times', error_trigger_threshold=3)
                    await self.quote_poller.kill_me()
                    self.quote_poller.join()
                    self.connect_to_market_data()

                await asyncio.sleep(10)
                self.update_df_positions_and_prices()
            self.report_current_wave_fills()
            if self.no_ack_in_current_wave:
                logger.warning("No order acked in current wave, please check statarb-oms, oms_listener, and other related")
