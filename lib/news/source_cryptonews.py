"""
CryptoNews API Source - REST API-based cryptocurrency news with sentiment.

This source polls the CryptoNews API for cryptocurrency news articles including
sentiment analysis. Supports both real-time polling and historical data download.

Usage:
    # Run standalone real-time polling
    python -m lib.news.source_cryptonews [--debug]

    # Download historical data
    python -m lib.news.source_cryptonews --historical --start-date 2024-01-01 --end-date 2024-01-31
"""

import argparse
import asyncio
import json
import logging.config
import os
from datetime import date, datetime, timedelta
from typing import Optional

import aiohttp
from slack_sdk.webhook.client import WebhookClient

from lib.util.config import get_config
from lib.util.time_util import date_to_str, dt_to_millis
from lib.util.slack import SLACK_WEBHOOK
from lib.util.directory import dir_manager
from lib.util.util import LOCAL, SYMBOL_BASE
from lib.util.logging_util import get_logging_config, KeyLogger
from lib.universe import Universe
from lib.news.news_util import load_tickers_for_date, get_query_tickers, MAJOR_TICKERS


logging.config.dictConfig(get_logging_config("news_cryptonews"))
original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


CRYPTONEWS_API_BASE = "https://cryptonews-api.com/api/v1"
# Get your API key at https://cryptonews-api.com/
CRYPTONEWS_API_KEY = os.environ.get("CRYPTONEWS_API_KEY", "")
POLL_INTERVAL_SECONDS = 60
MAX_ITEMS_PER_REQUEST = 100  # Paid plan limit
HISTORICAL_REQUEST_DELAY = 1.0  # Delay between historical requests

# Historical data limitations - CryptoNews has data back to ~2019
EARLIEST_HISTORICAL_DATE = datetime(2019, 1, 1)


def date_to_api_format(date_obj: datetime) -> str:
    """Convert datetime to API format MMDDYYYY."""
    return date_obj.strftime("%m%d%Y")


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d")


class CryptoNewsSource:
    """REST API-based news source from CryptoNews API."""

    SOURCE_NAME = "cryptonews"
    SUPPORTS_HISTORICAL = True

    def __init__(self, config: dict, debug: bool = False):
        self.debug = debug
        self.config = config
        self.api_key = CRYPTONEWS_API_KEY

        self.slack_client = WebhookClient(SLACK_WEBHOOK) if not LOCAL and not debug else None

        # Load universe symbols for filtering relevant news
        self.universe = Universe(self.config)
        self.symbols = self.universe.load_universe_symbols(
            universe_source='file',
            filter='fittable',
            symbol_type=SYMBOL_BASE
        )
        # Create ticker set for matching (without USDT suffix)
        self.tickers = {s.replace('USDT', '') for s in self.symbols}
        logger.info(f"Tracking news on {len(self.tickers)} tickers")

        # Track seen article IDs to avoid duplicates (for real-time mode)
        self.seen_ids: set = set()

        # File handle for writing news
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

    def _check_date_rollover(self) -> None:
        """Check if we need to roll over to a new day's file."""
        if self.debug:
            return
        current = date_to_str()
        if current != self.current_date:
            logger.info(f"Date rollover: {self.current_date} -> {current}")
            self._open_file()

    def _get_query_tickers(self) -> str:
        """Get comma-separated ticker string for API query."""
        return get_query_tickers(self.tickers, major_only=True)

    def _build_api_url(
        self,
        date_range: Optional[str] = None,
        page: int = 1
    ) -> str:
        """Build the API URL with parameters."""
        tickers_str = self._get_query_tickers()
        params = [
            f"token={self.api_key}",
            f"items={MAX_ITEMS_PER_REQUEST}",
            f"tickers={tickers_str}",
        ]
        if date_range:
            params.append(f"date={date_range}")
        if page > 1:
            params.append(f"page={page}")
        return f"{CRYPTONEWS_API_BASE}?{'&'.join(params)}"

    async def _fetch_news(
        self,
        session: aiohttp.ClientSession,
        date_range: Optional[str] = None,
        page: int = 1
    ) -> list:
        """Fetch news from the CryptoNews API."""
        if not self.api_key:
            logger.debug("No API key configured, skipping fetch")
            return []

        url = self._build_api_url(date_range=date_range, page=page)

        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', [])
                if response.status == 403:
                    try:
                        error_data = await response.json()
                        error_msg = error_data.get('message', 'Unknown error')
                        logger.warning(f"API returned 403: {error_msg}")
                    except (json.JSONDecodeError, aiohttp.ContentTypeError):
                        logger.warning("API returned 403 - check API key validity")
                else:
                    logger.warning(f"API returned status {response.status}")
                return []
        except asyncio.TimeoutError:
            logger.warning("API request timed out")
            return []
        except aiohttp.ClientError as e:
            logger.warning(f"API request failed: {e}")
            return []

    def _extract_tickers_from_article(self, article: dict) -> list:
        """Extract ticker symbols mentioned in an article."""
        tickers_found = []
        article_tickers = article.get('tickers', [])
        if isinstance(article_tickers, str):
            article_tickers = article_tickers.split(',')

        for ticker in article_tickers:
            ticker = ticker.strip().upper()
            if ticker in self.tickers:
                tickers_found.append(ticker)
        return tickers_found

    def _process_article(self, article: dict, skip_dedup: bool = False) -> Optional[dict]:
        """Process a single article and return normalized format."""
        article_id = article.get('news_url') or article.get('title', '')
        if not article_id:
            return None

        if not skip_dedup:
            if article_id in self.seen_ids:
                return None
            self.seen_ids.add(article_id)

            if len(self.seen_ids) > 10000:
                self.seen_ids = set(list(self.seen_ids)[-5000:])

        tickers = self._extract_tickers_from_article(article)

        news_record = {
            'title': article.get('title', ''),
            'body': article.get('text', ''),
            'source': article.get('source_name', 'cryptonews-api'),
            'url': article.get('news_url', ''),
            'time': article.get('date', ''),
            'sentiment': article.get('sentiment', 'Neutral'),
            'type': article.get('type', 'Article'),
            'tickers': tickers,
            'suggestions': [{'coin': t, 'found': [t]} for t in tickers],
            'topics': article.get('topics', []),
            '_id': article_id,
            'live_ts': dt_to_millis(),
            'api_source': 'cryptonews-api.com'
        }

        return news_record

    def _write_article(self, news_record: dict) -> None:
        """Write a news record to file or stdout."""
        if self.debug:
            print(json.dumps(news_record, indent=2))
        else:
            self.news_file.write(json.dumps(news_record) + "\n")
            self.news_file.flush()

    def _notify_slack(self, news_record: dict) -> None:
        """Send Slack notification for relevant news."""
        if self.slack_client is None:
            return

        tickers = news_record.get('tickers', [])
        if not tickers:
            return

        ticker = tickers[0]
        sentiment = news_record.get('sentiment', 'Neutral')
        title = news_record.get('title', '')[:100]

        sentiment_emoji = {
            'Positive': ':chart_with_upwards_trend:',
            'Negative': ':chart_with_downwards_trend:',
            'Neutral': ':bar_chart:'
        }.get(sentiment, ':newspaper:')

        msg = f"[CryptoNews] {sentiment_emoji} [{ticker}] {sentiment}: {title}"
        try:
            self.slack_client.send(text=msg)
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {e}")

    async def download_historical(
        self,
        start_date: datetime,
        end_date: datetime,
        max_pages: int = 100
    ) -> int:
        """Download historical news for a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            max_pages: Maximum pages to fetch per day

        Returns:
            Total number of articles downloaded
        """
        # Validate date range
        if start_date < EARLIEST_HISTORICAL_DATE:
            logger.warning(
                f"CryptoNews historical data starts from {EARLIEST_HISTORICAL_DATE.date()}. "
                f"Adjusting start date from {start_date.date()} to {EARLIEST_HISTORICAL_DATE.date()}"
            )
            start_date = EARLIEST_HISTORICAL_DATE

        if start_date > end_date:
            logger.warning(
                f"Start date {start_date.date()} is after end date {end_date.date()}. "
                "No data to download."
            )
            return 0

        total_articles = 0
        current_date = start_date

        logger.info(
            f"Starting historical download from {start_date.date()} to {end_date.date()}"
        )

        async with aiohttp.ClientSession() as session:
            while current_date <= end_date:
                date_str = current_date.strftime("%Y%m%d")
                api_date = date_to_api_format(current_date)
                api_date_range = f"{api_date}-{api_date}"

                # Load universe from prior day for historical accuracy
                self.tickers = load_tickers_for_date(
                    self.universe, current_date.date(), use_prior_day=True
                )

                if not self.debug:
                    self._open_file(date_str)

                logger.info(f"Fetching news for {current_date.date()}")
                day_articles = 0
                page = 1
                consecutive_empty = 0

                while page <= max_pages:
                    articles = await self._fetch_news(
                        session,
                        date_range=api_date_range,
                        page=page
                    )

                    if not articles:
                        consecutive_empty += 1
                        if consecutive_empty >= 2:
                            break
                        page += 1
                        await asyncio.sleep(HISTORICAL_REQUEST_DELAY)
                        continue

                    consecutive_empty = 0

                    for article in articles:
                        news_record = self._process_article(article, skip_dedup=True)
                        if news_record:
                            self._write_article(news_record)
                            day_articles += 1

                    logger.debug(f"  Page {page}: {len(articles)} articles")
                    page += 1
                    await asyncio.sleep(HISTORICAL_REQUEST_DELAY)

                logger.info(f"  {current_date.date()}: {day_articles} articles")
                total_articles += day_articles
                current_date += timedelta(days=1)

        logger.info(f"Historical download complete: {total_articles} total articles")
        return total_articles

    async def run_realtime(self) -> None:
        """Main loop for real-time news polling."""
        if not self.debug:
            self._open_file()

        if self.slack_client:
            self.slack_client.send(text="Starting CryptoNews Source")

        logger.info("Starting CryptoNews API polling loop")

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    self._check_date_rollover()

                    articles = await self._fetch_news(session)
                    new_count = 0

                    for article in articles:
                        news_record = self._process_article(article)
                        if news_record:
                            new_count += 1
                            self._write_article(news_record)
                            logger.info(
                                f"New article: {news_record['title'][:60]}... "
                                f"[{news_record.get('sentiment', 'N/A')}]"
                            )

                            if news_record.get('tickers'):
                                self._notify_slack(news_record)

                    if new_count > 0:
                        logger.info(f"Fetched {new_count} new articles")
                    else:
                        logger.debug("No new articles")

                except Exception as e:
                    logger.error(f"Error in polling loop: {e}", exc_info=True)

                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    def close(self) -> None:
        """Clean up resources."""
        if self.news_file:
            self.news_file.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CryptoNews API Source - Cryptocurrency news with sentiment"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode: print to stdout instead of file, no Slack'
    )
    parser.add_argument(
        '--historical',
        action='store_true',
        help='Download historical data instead of real-time polling'
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
    parser.add_argument(
        '--max-pages',
        type=int,
        default=100,
        help='Maximum pages to fetch per day for historical (default: 100)'
    )
    return parser.parse_args()


async def main() -> None:
    """Main entry point for standalone execution."""
    args = parse_args()
    _, config = get_config()

    source = CryptoNewsSource(config, debug=args.debug)

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

            await source.download_historical(start, end, max_pages=args.max_pages)
        else:
            await source.run_realtime()
    except KeyboardInterrupt:
        logger.info("Shutting down CryptoNews source")
    finally:
        source.close()


if __name__ == '__main__':
    asyncio.run(main())
