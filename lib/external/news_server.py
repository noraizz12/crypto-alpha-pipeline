import asyncio
import json
import logging.config
import ssl
from datetime import datetime as dt, timezone

import certifi
import websockets
from slack_sdk.webhook.client import WebhookClient

from lib.util.config import get_config
from lib.util.time_util import date_to_str, dt_to_millis
from lib.util.slack import SLACK_WEBHOOK
from lib.util.directory import dir_manager
from lib.util.util import LOCAL, truncate_to_words, SYMBOL_BASE
from lib.util.logging_util import get_logging_config, KeyLogger
from lib.universe import Universe

logging.config.dictConfig(get_logging_config("news"))
original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


NEWS_WEBSOCKET = "wss://news.treeofalpha.com/ws"
NUM_NEWS_BODY_WORDS = 20


class News:
    def __init__(self, config: dict):
        self.slack_client = WebhookClient(SLACK_WEBHOOK) if not LOCAL else None

        self.config = config
        self.universe = Universe(self.config)
        self.symbols = self.universe.load_universe_symbols(universe_source='file', filter='fittable', symbol_type=SYMBOL_BASE)
        logger.info(f"Tracking news on symbols: {self.symbols}")

        self.news_file = None
        self._open_file()

    def _open_file(self):
        self.news_file = open(f"{dir_manager.NEWS_DIR}/news.{date_to_str()}.csv", "a")

    async def receive_messages(self, websocket):
        while True:
            try:
                message = await websocket.recv()
                if "pong" in message:
                    logger.info(f'News server receiving heartbeats {dt.now(timezone.utc)}')
                    continue

                message = json.loads(message)
                message['live_ts'] = dt_to_millis()
                logger.info(f"Received message: {message['title'].lower()}")
                self.news_file.write(str(message)+"\n")
                self.news_file.flush()

                suggestions = message.get('suggestions')
                if suggestions is not None and len(suggestions) > 0:
                    coin = suggestions[0].get('coin')
                    if coin is not None and coin in self.symbols:
                        coin = coin.upper()
                        #i cant fucking stand it any more
                        if self.slack_client is not None and coin != "DOGE":
                            message_body = message.get('body')
                            message_details = message['title'] if message_body is None else truncate_to_words(message_body, NUM_NEWS_BODY_WORDS)
                            slack_msg = f"{coin}: {message_details}"
                            self.slack_client.send(text=slack_msg)
            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection to the WebSocket has been closed.")
                raise
            except Exception as e:
                logger.info(f"Error decoding message {e}")

    async def send_heartbeats(self, websocket):

        start_date = dt.now(timezone.utc).date()
        while True:
            if dt.now(timezone.utc).date() != start_date:
                self._open_file()
                start_date = dt.now(timezone.utc).date()
            try:
                await websocket.send('ping')
            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed during heartbeat")
                raise
            await asyncio.sleep(10)

    async def listen_to_websocket(self):
        self.slack_client.send(text="Starting News Server")
        retry_count = 0
        while True:
            try:
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                async with websockets.connect(NEWS_WEBSOCKET, ssl=ssl_context) as websocket:
                    retry_count = 0
                    # Run receiving and heartbeat sending coroutines concurrently
                    receive_task = asyncio.ensure_future(self.receive_messages(websocket))
                    heartbeat_task = asyncio.ensure_future(self.send_heartbeats(websocket))
                    # This will run both tasks concurrently and will run forever unless the WebSocket connection is closed.
                    await asyncio.gather(receive_task, heartbeat_task)
            except Exception as e:
                logger.info(f"failed in listen_to_websocket: {e}, retry connecting...")
                retry_count += 1
                # Wait for 5, 10, 15, 20, ... seconds before reconnecting
                await asyncio.sleep(min(retry_count * 5, 60))
                if retry_count > 5:
                    logger.error("Keep failing in news server connection", key="consecutive failed connection to news websocket")



if __name__ == '__main__':
    config_file, config = get_config()
    news = News(config)
    asyncio.run(news.listen_to_websocket())
