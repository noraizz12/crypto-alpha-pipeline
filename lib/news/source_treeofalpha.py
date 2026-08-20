"""
TreeOfAlpha News Source - WebSocket-based real-time cryptocurrency news.

This source connects to the TreeOfAlpha WebSocket feed for real-time
cryptocurrency news. Also supports historical data download via REST API.

Usage:
    # Run standalone (real-time only)
    python -m lib.news.source_treeofalpha [--debug]

    # Download historical data
    python -m lib.news.source_treeofalpha --historical --start-date 2024-01-01 --end-date 2024-01-31
"""

import argparse
import asyncio
import json
import logging.config
import os
import ssl
from datetime import date, datetime
from typing import Optional

import aiohttp
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
from lib.news.news_util import load_tickers_for_date


logging.config.dictConfig(get_logging_config("news_treeofalpha"))
original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


TREEOFALPHA_WEBSOCKET = "wss://news.treeofalpha.com/ws"
# Partial history (no auth required, up to 3000 items)
TREEOFALPHA_API_NEWS = "https://news.treeofalpha.com/api/news?limit=3000"
# Full history (requires API key)
TREEOFALPHA_API_ALL_NEWS = "https://news.treeofalpha.com/api/allNews"
NUM_NEWS_BODY_WORDS = 20
HEARTBEAT_INTERVAL_SECONDS = 10

# Historical data limitations
# TreeOfAlpha free tier only returns ~3000 most recent articles (~7 days)
HISTORICAL_LOOKBACK_DAYS = 7


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d")


class TreeOfAlphaSource:
    """WebSocket-based news source from TreeOfAlpha."""

    SOURCE_NAME = "treeofalpha"
    SUPPORTS_HISTORICAL = True

    def __init__(self, config: dict, debug: bool = False):
        self.debug = debug
        self.config = config
        self.slack_client = WebhookClient(SLACK_WEBHOOK) if not LOCAL and not debug else None

        # Load universe symbols for filtering relevant news
        self.universe = Universe(self.config)
        self.symbols = self.universe.load_universe_symbols(
            universe_source='file',
            filter='fittable',
            symbol_type=SYMBOL_BASE
        )
        logger.info(f"Tracking news on symbols: {self.symbols}")

        # File handle for writing news (opened lazily)
        self.news_file = None
        self.current_date: Optional[str] = None

    def _open_file(self, date_str: Optional[str] = None) -> None:
        """Open a new news file for the specified or current date."""
        if self.news_file:
            self.news_file.close()
        self.current_date = date_str or date_to_str()
        # Create date subdirectory if needed
        date_dir = f"{dir_manager.NEWS_DIR_NEW}/{self.current_date}"
        os.makedirs(date_dir, exist_ok=True)
        filepath = f"{date_dir}/{self.SOURCE_NAME}.{self.current_date}.csv"
        # pylint: disable=consider-using-with
        self.news_file = open(filepath, "a", encoding="utf-8")
        logger.info(f"Opened news file: {filepath}")

    def _check_date_rollover(self, current_date_str: str) -> None:
        """Check if we need to roll over to a new day's file."""
        if self.debug:
            return
        if current_date_str != self.current_date:
            logger.info(f"Date rollover: {self.current_date} -> {current_date_str}")
            self._open_file(current_date_str)

    def _process_message(self, message: dict) -> Optional[dict]:
        """Process a raw WebSocket message and return normalized format."""
        message['live_ts'] = dt_to_millis()
        return message

    def _write_message(self, message: dict) -> None:
        """Write a message to file or stdout."""
        if self.debug:
            print(json.dumps(message, indent=2))
        else:
            self.news_file.write(str(message) + "\n")
            self.news_file.flush()

    def _notify_slack(self, message: dict) -> None:
        """Send Slack notification for relevant news."""
        if self.slack_client is None:
            return

        suggestions = message.get('suggestions')
        if suggestions is None or len(suggestions) == 0:
            return

        coin = suggestions[0].get('coin')
        if coin is None or coin not in self.symbols:
            return

        coin = coin.upper()
        # Skip DOGE spam
        if coin == "DOGE":
            return

        message_body = message.get('body')
        message_details = (
            message['title'] if message_body is None
            else truncate_to_words(message_body, NUM_NEWS_BODY_WORDS)
        )
        slack_msg = f"[TreeOfAlpha] {coin}: {message_details}"

        try:
            self.slack_client.send(text=slack_msg)
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {e}")

    async def download_historical(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Download historical news for a date range.

        Note: TreeOfAlpha free tier only returns ~3000 most recent articles
        (approximately 7 days). Older dates will return no data.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Total number of articles downloaded
        """
        # Warn about historical limitations
        from datetime import timedelta
        earliest_available = datetime.now() - timedelta(days=HISTORICAL_LOOKBACK_DAYS)
        if start_date < earliest_available:
            logger.warning(
                f"TreeOfAlpha free tier only has ~{HISTORICAL_LOOKBACK_DAYS} days of history. "
                f"Requested start {start_date.date()} is before earliest available "
                f"~{earliest_available.date()}. Results may be incomplete or empty."
            )

        logger.info(
            f"Starting historical download from {start_date.date()} to {end_date.date()}"
        )

        total_articles = 0
        articles_by_date: dict = {}

        async with aiohttp.ClientSession() as session:
            try:
                logger.info("Fetching news from TreeOfAlpha API...")
                async with session.get(TREEOFALPHA_API_NEWS, timeout=120) as response:
                    if response.status != 200:
                        logger.error(f"API returned status {response.status}")
                        return 0

                    all_news = await response.json()
                    logger.info(f"Fetched {len(all_news)} articles from API")

            except asyncio.TimeoutError:
                logger.error("API request timed out")
                return 0
            except aiohttp.ClientError as e:
                logger.error(f"API request failed: {e}")
                return 0

        # Filter and group articles by date
        for article in all_news:
            article_time = article.get('time')
            if article_time is None:
                continue

            # Get date string for this article (use UTC to match API timestamps)
            article_date = datetime.utcfromtimestamp(article_time / 1000)
            date_str = article_date.strftime("%Y%m%d")

            # Filter by date range (compare date strings to avoid timezone issues)
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            if date_str < start_str or date_str > end_str:
                continue

            if date_str not in articles_by_date:
                articles_by_date[date_str] = []

            # Add live_ts for consistency
            article['live_ts'] = dt_to_millis()
            articles_by_date[date_str].append(article)

        # Write articles to files grouped by date
        for date_str in sorted(articles_by_date.keys()):
            articles = articles_by_date[date_str]

            # Load universe from prior day for historical accuracy
            article_date = datetime.strptime(date_str, "%Y%m%d").date()
            self.tickers = load_tickers_for_date(
                self.universe, article_date, use_prior_day=True
            )

            if not self.debug:
                self._open_file(date_str)

            for article in articles:
                self._write_message(article)
                total_articles += 1

            logger.info(f"  {date_str}: {len(articles)} articles")

        logger.info(f"Historical download complete: {total_articles} total articles")
        return total_articles

    async def _receive_messages(self, websocket) -> None:
        """Receive and process messages from the WebSocket."""
        while True:
            try:
                raw_message = await websocket.recv()

                # Handle heartbeat responses
                if "pong" in raw_message:
                    logger.debug(f"Received heartbeat pong")
                    continue

                message = json.loads(raw_message)
                processed = self._process_message(message)

                if processed:
                    logger.info(f"Received message: {message.get('title', '').lower()[:60]}")
                    self._write_message(processed)
                    self._notify_slack(message)

            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection to the WebSocket has been closed.")
                raise
            except json.JSONDecodeError as e:
                logger.warning(f"Error decoding message: {e}")
            except Exception as e:
                logger.warning(f"Error processing message: {e}")

    async def _send_heartbeats(self, websocket) -> None:
        """Send periodic heartbeats to keep the connection alive."""
        start_date = date_to_str()
        while True:
            # Check for date rollover
            current_date = date_to_str()
            if current_date != start_date:
                self._check_date_rollover(current_date)
                start_date = current_date

            try:
                await websocket.send('ping')
                logger.debug("Sent heartbeat ping")
            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed during heartbeat")
                raise

            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

    async def run_realtime(self) -> None:
        """Main loop for real-time news collection."""
        if not self.debug:
            self._open_file()

        if self.slack_client:
            self.slack_client.send(text="Starting TreeOfAlpha News Source")

        logger.info("Starting TreeOfAlpha WebSocket connection")
        retry_count = 0

        while True:
            try:
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                async with websockets.connect(
                    TREEOFALPHA_WEBSOCKET,
                    ssl=ssl_context
                ) as websocket:
                    retry_count = 0
                    logger.info("Connected to TreeOfAlpha WebSocket")

                    # Run receiving and heartbeat tasks concurrently
                    receive_task = asyncio.ensure_future(self._receive_messages(websocket))
                    heartbeat_task = asyncio.ensure_future(self._send_heartbeats(websocket))

                    await asyncio.gather(receive_task, heartbeat_task)

            except Exception as e:
                logger.warning(f"WebSocket connection failed: {e}, reconnecting...")
                retry_count += 1
                wait_time = min(retry_count * 5, 60)
                await asyncio.sleep(wait_time)

                if retry_count > 5:
                    logger.error(
                        "Multiple consecutive connection failures",
                        key="treeofalpha_connection_failures"
                    )

    def close(self) -> None:
        """Clean up resources."""
        if self.news_file:
            self.news_file.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TreeOfAlpha News Source - Real-time cryptocurrency news"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode: print to stdout instead of file, no Slack'
    )
    parser.add_argument(
        '--historical',
        action='store_true',
        help='Download historical data instead of real-time streaming'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for historical download (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for historical download (YYYY-MM-DD)'
    )
    return parser.parse_args()


async def main() -> None:
    """Main entry point for standalone execution."""
    args = parse_args()
    _, config = get_config()

    source = TreeOfAlphaSource(config, debug=args.debug)

    try:
        if args.historical:
            if not args.start_date or not args.end_date:
                logger.error("--start-date and --end-date required for historical mode")
                return

            start = parse_date(args.start_date)
            end = parse_date(args.end_date)

            if start > end:
                logger.error("Start date must be before end date")
                return

            await source.download_historical(start, end)
        else:
            await source.run_realtime()
    except KeyboardInterrupt:
        logger.info("Shutting down TreeOfAlpha source")
    finally:
        source.close()


if __name__ == '__main__':
    asyncio.run(main())
