"""
CryptoPanic API Source - Aggregated cryptocurrency news with community sentiment.

This source polls the CryptoPanic API for aggregated cryptocurrency news from 50+
sources with community voting (bullish/bearish sentiment). Supports both real-time
polling and historical data download.

Usage:
    # Run standalone real-time polling
    python -m lib.news.source_cryptopanic [--debug]

    # Download historical data
    python -m lib.news.source_cryptopanic --historical --start-date 2024-01-01 --end-date 2024-01-31
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
from lib.news.news_util import load_tickers_for_date


logging.config.dictConfig(get_logging_config("news_cryptopanic"))
original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


CRYPTOPANIC_API_BASE = "https://cryptopanic.com/api/growth/v2/posts/"
# Get your free API key at https://cryptopanic.com/developers/api/
CRYPTOPANIC_API_KEY = os.environ.get("CRYPTOPANIC_API_KEY", "")
POLL_INTERVAL_SECONDS = 60
HISTORICAL_REQUEST_DELAY = 1.0

# Historical data limitations - CryptoPanic has data back to ~2017
EARLIEST_HISTORICAL_DATE = datetime(2017, 1, 1)

# Common name to ticker mappings
NAME_TO_TICKER = {
    'BITCOIN': 'BTC',
    'ETHEREUM': 'ETH',
    'SOLANA': 'SOL',
    'RIPPLE': 'XRP',
    'CARDANO': 'ADA',
    'DOGECOIN': 'DOGE',
    'POLKADOT': 'DOT',
    'CHAINLINK': 'LINK',
    'AVALANCHE': 'AVAX',
    'LITECOIN': 'LTC',
}


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d")


def parse_published_at(published_at: str) -> datetime:
    """Parse ISO 8601 timestamp from API."""
    # Format: 2018-03-01T21:00:44.575689Z
    try:
        return datetime.fromisoformat(published_at.replace('Z', '+00:00'))
    except ValueError:
        # Fallback for different formats
        return datetime.strptime(published_at[:19], "%Y-%m-%dT%H:%M:%S")


class CryptoPanicSource:
    """REST API-based news source from CryptoPanic aggregator."""

    SOURCE_NAME = "cryptopanic"
    SUPPORTS_HISTORICAL = True

    def __init__(self, config: dict, debug: bool = False):
        self.debug = debug
        self.config = config
        self.api_key = CRYPTOPANIC_API_KEY

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
        # CryptoPanic uses standard coin symbols
        major_tickers = [
            "BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "AVAX", "DOT", "LINK",
            "MATIC", "UNI", "LTC", "ATOM", "FIL", "APT", "ARB", "OP", "SUI", "SEI",
        ]
        query_tickers = [t for t in major_tickers if t in self.tickers]
        if not query_tickers:
            query_tickers = major_tickers[:10]
        return ",".join(query_tickers)

    def _build_api_url(self, page_url: Optional[str] = None) -> str:
        """Build the API URL with parameters."""
        if page_url:
            # Use pagination URL directly
            return page_url

        tickers_str = self._get_query_tickers()
        params = [
            f"auth_token={self.api_key}",
            "public=true",
            f"currencies={tickers_str}",
            "regions=en",
        ]
        return f"{CRYPTOPANIC_API_BASE}?{'&'.join(params)}"

    async def _fetch_news(
        self,
        session: aiohttp.ClientSession,
        page_url: Optional[str] = None
    ) -> tuple:
        """Fetch news from the CryptoPanic API.

        Returns:
            Tuple of (results list, next page URL or None)
        """
        if not self.api_key:
            logger.warning("No API key configured, skipping fetch")
            return [], None

        url = self._build_api_url(page_url=page_url)

        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get('results', [])
                    next_url = data.get('next')
                    return results, next_url
                if response.status == 401:
                    logger.error("API returned 401 - check API key validity")
                elif response.status == 429:
                    logger.warning("API rate limit exceeded")
                else:
                    logger.warning(f"API returned status {response.status}")
                return [], None
        except asyncio.TimeoutError:
            logger.warning("API request timed out")
            return [], None
        except aiohttp.ClientError as e:
            logger.warning(f"API request failed: {e}")
            return [], None

    def _extract_tickers_from_post(self, post: dict) -> list:
        """Extract ticker symbols from a post's instruments field or title/description."""
        tickers_found = []

        # First try instruments field (if available in API tier)
        instruments = post.get('instruments', [])
        for instrument in instruments:
            code = instrument.get('code', '').upper()
            if code in self.tickers:
                tickers_found.append(code)

        # If no instruments, try to extract from title and description
        if not tickers_found:
            text = f"{post.get('title', '')} {post.get('description', '')}".upper()

            # Check for ticker symbols
            for ticker in self.tickers:
                if f" {ticker} " in f" {text} " or f"({ticker})" in text:
                    tickers_found.append(ticker)

            # Check for common names (Bitcoin -> BTC, etc.)
            for name, ticker in NAME_TO_TICKER.items():
                if name in text and ticker in self.tickers and ticker not in tickers_found:
                    tickers_found.append(ticker)

        return list(set(tickers_found))  # Remove duplicates

    def _compute_sentiment(self, votes: dict) -> str:
        """Compute sentiment from vote counts."""
        if not votes:
            return "Neutral"

        positive = votes.get('positive', 0) or 0
        negative = votes.get('negative', 0) or 0

        if positive > negative * 2:
            return "Positive"
        if negative > positive * 2:
            return "Negative"
        return "Neutral"

    def _process_post(self, post: dict, skip_dedup: bool = False) -> Optional[dict]:
        """Process a single post and return normalized format."""
        post_id = post.get('id')
        if not post_id:
            return None

        post_id_str = str(post_id)

        if not skip_dedup:
            if post_id_str in self.seen_ids:
                return None
            self.seen_ids.add(post_id_str)

            # Keep seen_ids from growing unbounded
            if len(self.seen_ids) > 10000:
                self.seen_ids = set(list(self.seen_ids)[-5000:])

        tickers = self._extract_tickers_from_post(post)
        votes = post.get('votes', {})
        source_info = post.get('source', {})
        content = post.get('content', {})

        # Construct URL from slug if original_url not available
        slug = post.get('slug', '')
        url = post.get('original_url') or post.get('url', '')
        if not url and slug:
            url = f"https://cryptopanic.com/news/{post_id_str}/{slug}"

        news_record = {
            'title': post.get('title', ''),
            'body': content.get('clean', '') or post.get('description', ''),
            'source': source_info.get('title', source_info.get('domain', 'cryptopanic')),
            'url': url,
            'time': post.get('published_at', ''),
            'sentiment': self._compute_sentiment(votes),
            'type': post.get('kind', 'news'),
            'tickers': tickers,
            'suggestions': [{'coin': t, 'found': [t]} for t in tickers],
            'votes': {
                'positive': votes.get('positive', 0),
                'negative': votes.get('negative', 0),
                'important': votes.get('important', 0),
                'liked': votes.get('liked', 0),
                'comments': votes.get('comments', 0),
            },
            'panic_score': post.get('panic_score'),
            '_id': post_id_str,
            'live_ts': dt_to_millis(),
            'api_source': 'cryptopanic.com'
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
        votes = news_record.get('votes', {})
        vote_str = f"+{votes.get('positive', 0)}/-{votes.get('negative', 0)}"

        sentiment_emoji = {
            'Positive': ':chart_with_upwards_trend:',
            'Negative': ':chart_with_downwards_trend:',
            'Neutral': ':bar_chart:'
        }.get(sentiment, ':newspaper:')

        msg = f"[CryptoPanic] {sentiment_emoji} [{ticker}] {vote_str}: {title}"
        try:
            self.slack_client.send(text=msg)
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {e}")

    async def download_historical(
        self,
        start_date: datetime,
        end_date: datetime,
        max_pages: int = 10000
    ) -> int:
        """Download historical news for a date range.

        Note: CryptoPanic API returns news in reverse chronological order.
        We fetch pages until we've passed the start_date.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            max_pages: Maximum pages to fetch

        Returns:
            Total number of articles downloaded
        """
        # Validate date range
        if start_date < EARLIEST_HISTORICAL_DATE:
            logger.warning(
                f"CryptoPanic historical data starts from {EARLIEST_HISTORICAL_DATE.date()}. "
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

        # Convert to timestamps for comparison
        start_ts = start_date.timestamp()
        end_ts = (end_date + timedelta(days=1)).timestamp()

        logger.info(
            f"Starting historical download from {start_date.date()} to {end_date.date()}"
        )

        async with aiohttp.ClientSession() as session:
            page = 1
            next_url = None
            reached_start = False

            skipped_future = 0
            collected = 0
            # Store raw posts by date - we'll process them later with correct universe
            posts_by_date: dict = {}

            while page <= max_pages and not reached_start:
                results, next_url = await self._fetch_news(session, page_url=next_url)

                if not results:
                    logger.info(f"No more results at page {page}")
                    break

                page_collected = 0
                page_skipped_future = 0
                oldest_post_dt = None

                for post in results:
                    published_at = post.get('published_at')
                    if not published_at:
                        continue

                    post_dt = parse_published_at(published_at)
                    post_ts = post_dt.timestamp()
                    oldest_post_dt = post_dt

                    # Check if we've gone past the start date
                    if post_ts < start_ts:
                        reached_start = True
                        break

                    # Skip if after end date
                    if post_ts >= end_ts:
                        page_skipped_future += 1
                        continue

                    # Get date string for grouping
                    date_str = post_dt.strftime("%Y%m%d")

                    if date_str not in posts_by_date:
                        posts_by_date[date_str] = []

                    # Store raw post - we'll process with correct universe later
                    posts_by_date[date_str].append(post)
                    page_collected += 1

                skipped_future += page_skipped_future
                collected += page_collected

                # Log progress every 100 pages or when we start collecting
                if page % 100 == 0 or (page_collected > 0 and collected == page_collected):
                    oldest_str = oldest_post_dt.strftime("%Y-%m-%d") if oldest_post_dt else "N/A"
                    logger.info(
                        f"Page {page}: collected={collected}, skipped_future={skipped_future}, "
                        f"oldest_post={oldest_str}"
                    )

                page += 1

                if not next_url:
                    logger.info("No more pages available from API")
                    break

                await asyncio.sleep(HISTORICAL_REQUEST_DELAY)

        # Process and write articles grouped by date, using correct universe for each date
        for date_str in sorted(posts_by_date.keys()):
            posts = posts_by_date[date_str]

            # Load universe from prior day for historical accuracy
            article_date = datetime.strptime(date_str, "%Y%m%d").date()
            self.tickers = load_tickers_for_date(
                self.universe, article_date, use_prior_day=True
            )

            if not self.debug:
                self._open_file(date_str)

            date_articles = 0
            for post in posts:
                news_record = self._process_post(post, skip_dedup=True)
                if news_record:
                    self._write_article(news_record)
                    total_articles += 1
                    date_articles += 1

            logger.info(f"  {date_str}: {date_articles} articles")

        logger.info(f"Historical download complete: {total_articles} total articles")
        return total_articles

    async def run_realtime(self) -> None:
        """Main loop for real-time news polling."""
        if not self.debug:
            self._open_file()

        if self.slack_client:
            self.slack_client.send(text="Starting CryptoPanic Source")

        logger.info("Starting CryptoPanic API polling loop")

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    self._check_date_rollover()

                    results, _ = await self._fetch_news(session)
                    new_count = 0

                    for post in results:
                        news_record = self._process_post(post)
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
        description="CryptoPanic API Source - Aggregated cryptocurrency news"
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
        default=10000,
        help='Maximum pages to fetch for historical (default: 10000)'
    )
    return parser.parse_args()


async def main() -> None:
    """Main entry point for standalone execution."""
    args = parse_args()
    _, config = get_config()

    source = CryptoPanicSource(config, debug=args.debug)

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
        logger.info("Shutting down CryptoPanic source")
    finally:
        source.close()


if __name__ == '__main__':
    asyncio.run(main())
