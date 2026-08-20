import asyncio
import json
import logging
import threading
import time
from datetime import datetime as dt
from datetime import timedelta as td, timezone
from asyncio.exceptions import TimeoutError
from typing import Optional, List, Dict, Any

import websockets

from ..data import load_positions
from lib.data.dataloader import DataLoader
from lib.universe import Universe
from .md_aggregator import DataAggregator
from lib.util.time_util import today,  yesterday_date
from lib.util.util import chunk_list, exception_msg, log_and_raise
from lib.util.logging_util import KeyLogger

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)

BINANCE_WEBSOCKET = "wss://fstream.binance.com/ws"
SUB_CNT_LIMIT = 100


class TradeQuotePoller(threading.Thread):
    def __init__(self, config: dict, aggregator: DataAggregator, quote_only_mode: bool = False, symbols: Optional[List[str]] = None):
        threading.Thread.__init__(self, name="TradeQuoteThread")
        logger.info(f"Making TradeQuotePoller {quote_only_mode=}")
        self.config = config
        self.quote_only_mode = quote_only_mode
        self.error_cnt = 0
        self.request_id = 0
        self.ws = None
        self.please_kill_me = False
        self.dead = False
        self.data_loader = DataLoader(config=self.config)
        self.universe = Universe(config=self.config, debug=False)
        # base_symbols represents current symbol universe

        if symbols is None:
            symbols = self.universe.load_universe_symbols(universe_source='file', filter='tradeable', symbol_type='pair')
            if symbols is None:
                raise log_and_raise("Not collecting quotes since universe could not be loaded")
        else:
            logger.info(f"Manually specifying {symbols}")

        pos_df = load_positions(mode='latest')
        if pos_df is not None:
            pos_symbols = set(pos_df[pos_df['qty'] != 0]['symbol'].unique())
            pos_symbols = pos_symbols - set(symbols)
            logger.info(f"Adding {pos_symbols} position symbols to universe")
            symbols = list(set(symbols) | pos_symbols)

        self.symbols = symbols
        self.loop = None
        self.aggregator = aggregator
        self.aggregator.quote_only_mode = quote_only_mode
        if not quote_only_mode:
            logger.info("Starting data aggregator thread")
            self.aggregator.start()

    async def subscribe_to_symbols(self, symbols: List[str]):
        logger.info(f"Subscribing to binance for {symbols}")
        symbol_chunks = chunk_list(symbols, 100)
        for symbols in symbol_chunks:
            req = self._generate_sub_req(symbols)
            logger.info(f"Binance subscribe: {req}")
            await self.ws.send(json.dumps(req))

    async def subscribe_updated_universe(self):
        #we resubscribe at utc midnight, so the universe file needs to come from the previous day
        symbols = self.universe.load_universe_symbols(universe_source='file', universe_date=yesterday_date(), symbol_type='pair')
        extra_symbols = sorted(ss for ss in symbols if ss not in self.symbols)
        logger.info(f"Subscribing to binance for updated universe {extra_symbols}")
        await self.subscribe_to_symbols(extra_symbols)
        self.symbols = self.symbols + extra_symbols

    async def subscribe(self):
        logger.info("Subscribing to binance....")
        self.ws = await websockets.connect(BINANCE_WEBSOCKET, close_timeout=2)
        await self.subscribe_to_symbols(self.symbols)

    def wait_for_book(self, wait_seconds: int = 20):
        logger.info(f"Waiting for book {wait_seconds=}")
        end = time.time() + wait_seconds
        while time.time() < end:
            time.sleep(.01)
            if len(self.aggregator.current_book) == len(self.symbols):
                logger.info(f"Quote Book Full with {len(self.symbols)} instruments")
                return
        missing_symbols = set(self.symbols) - set(self.aggregator.current_book.keys())
        logger.warning(f"book not ready after {wait_seconds} secs.  missing: {missing_symbols}")

    def get_book(self, symbol: str) -> dict:
        return self.aggregator.current_book[symbol]

    def too_many_errors(self) -> bool:
        return self.error_cnt > 20

    def _generate_sub_req(self, symbols: List[str]) -> Dict[str, Any]:
        self.request_id += 1
        ret = {
            "method": "SUBSCRIBE",
            "id": self.request_id,
        }
        params = []
        for sym in symbols:
            sym = sym.replace('-', '').lower()
            params.append(f"{sym}@depth10@100ms")
            if not self.quote_only_mode:
                params.append(f"{sym}@aggTrade")
        ret["params"] = params
        return ret

    async def collect_quotes(self):
        logger.info("Starting to collect quotes...")
        await self.subscribe()

        refresh_subscribe_date = today() + td(days=1)
        while True:
            await asyncio.sleep(0)

            if self.please_kill_me:
                logger.info("Ending quote collections...")
                start = time.time()
                try:
                    await self.ws.close()
                except Exception as e:
                    logger.error(exception_msg("Error closing TradeQuotePoller websocket", e), key="fail to close TradeQuotePoller websocket")
                wait_time = time.time() - start
                logger.info(f"Waited {wait_time} seconds to close quote poller")
                break

            # refresh subscribe date will be tomorrow, so trigger subscribe update if current datetime move to the next day (5 mins as buffer for universe generate)
            if dt.now(timezone.utc) - td(seconds=60 * 5) > refresh_subscribe_date:
                logger.info(f"Trade Quote Poller moving to {refresh_subscribe_date}")
                refresh_subscribe_date = today() + td(days=1)
                await self.subscribe_updated_universe()

            try:
                response = await asyncio.wait_for(self.ws.recv(), timeout=2)
            except TimeoutError as te:
                logger.warning(f"Market data timed out: {te} continuing...")
                continue
            except Exception as e:
                logger.warning(f"fail to receive TradeQuotePoller market data for {e}")
                self.error_cnt += 1
                await asyncio.sleep(1)
                continue

            try:
                response = json.loads(response)
                msg_type = response['e']
            except Exception as e:
                if 'result' not in response:
                    logger.error(f"Bad message {response}", key="see bad message from TradeQuotePoller")
                continue

            if msg_type == 'depthUpdate':
                self.aggregator.add_quote_data(response)
            elif msg_type == 'aggTrade':
                self.aggregator.add_trade_data(response)
            else:
                logger.error(f"Unknown message type {msg_type=} with {response}", key="see unknown message type from TradeQuotePoller")
                continue

            #reset errors on successful loop
            self.error_cnt = 0

        self.dead = True

    async def kill_me(self):
        self.aggregator.kill_me()
        self.please_kill_me = True

    def run(self):
        asyncio.run(self.collect_quotes())
