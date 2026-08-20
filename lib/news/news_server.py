"""
News Server - Orchestrates multiple cryptocurrency news sources.

This server runs all configured news sources concurrently in real-time mode.
Each source runs in its own async task and writes to its own output files.

Usage:
    # Run all sources in real-time
    python -m lib.news.news_server [--debug]

    # Run specific sources only
    python -m lib.news.news_server --sources treeofalpha,cryptonews

Sources:
    - treeofalpha: WebSocket-based real-time news from TreeOfAlpha
    - cryptonews: REST API polling from CryptoNews API (includes sentiment)

Output:
    - TreeOfAlpha: {NEWS_DIR}/news.{YYYYMMDD}.csv
    - CryptoNews: {NEWS_DIR}/cryptonews.{YYYYMMDD}.csv
"""

import argparse
import asyncio
import logging.config
import signal
from typing import Optional

from slack_sdk.webhook.client import WebhookClient

from lib.util.config import get_config
from lib.util.slack import SLACK_WEBHOOK
from lib.util.util import LOCAL
from lib.util.logging_util import get_logging_config, KeyLogger
from lib.news.source_treeofalpha import TreeOfAlphaSource
from lib.news.source_cryptonews import CryptoNewsSource
# from lib.news.source_cryptopanic import CryptoPanicSource
from lib.news.source_cryptoslate import CryptoSlateSource


logging.config.dictConfig(get_logging_config("news_server"))
original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


# Registry of available news sources
NEWS_SOURCES = {
    'treeofalpha': TreeOfAlphaSource,
    'cryptonews': CryptoNewsSource,
    # 'cryptopanic': CryptoPanicSource,
    'cryptoslate': CryptoSlateSource,
}


class NewsServer:
    """Orchestrates multiple news sources running concurrently."""

    def __init__(
        self,
        config: dict,
        sources: Optional[list] = None,
        debug: bool = False
    ):
        self.config = config
        self.debug = debug
        self.slack_client = WebhookClient(SLACK_WEBHOOK) if not LOCAL and not debug else None

        # Determine which sources to run
        if sources:
            self.source_names = [s.strip().lower() for s in sources]
        else:
            self.source_names = list(NEWS_SOURCES.keys())

        # Validate source names
        invalid_sources = [s for s in self.source_names if s not in NEWS_SOURCES]
        if invalid_sources:
            raise ValueError(f"Unknown sources: {invalid_sources}")

        # Initialize sources
        self.sources = {}
        for name in self.source_names:
            source_class = NEWS_SOURCES[name]
            self.sources[name] = source_class(config, debug=debug)
            logger.info(f"Initialized source: {name}")

        # Track running tasks
        self.tasks: dict = {}
        self.shutdown_event = asyncio.Event()

    async def _run_source(self, name: str) -> None:
        """Run a single source with error handling and restart logic."""
        source = self.sources[name]
        restart_count = 0
        max_restarts = 10

        while not self.shutdown_event.is_set():
            try:
                logger.info(f"Starting source: {name}")
                await source.run_realtime()
            except asyncio.CancelledError:
                logger.info(f"Source {name} cancelled")
                break
            except Exception as e:
                restart_count += 1
                logger.error(
                    f"Source {name} failed (attempt {restart_count}): {e}",
                    exc_info=True
                )

                if restart_count >= max_restarts:
                    logger.error(
                        f"Source {name} exceeded max restarts ({max_restarts})",
                        key=f"news_source_{name}_max_restarts"
                    )
                    break

                # Exponential backoff
                wait_time = min(30, 5 * restart_count)
                logger.info(f"Restarting source {name} in {wait_time}s...")
                await asyncio.sleep(wait_time)

    async def run(self) -> None:
        """Run all configured sources concurrently."""
        if self.slack_client:
            source_list = ", ".join(self.source_names)
            self.slack_client.send(text=f"Starting News Server with sources: {source_list}")

        logger.info(f"Starting News Server with {len(self.sources)} sources")

        # Create tasks for each source
        for name in self.source_names:
            task = asyncio.create_task(self._run_source(name))
            self.tasks[name] = task

        # Wait for all tasks (or until shutdown)
        try:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        except asyncio.CancelledError:
            logger.info("News Server received cancellation")

    async def shutdown(self) -> None:
        """Gracefully shutdown all sources."""
        logger.info("Initiating graceful shutdown...")
        self.shutdown_event.set()

        # Cancel all tasks
        for name, task in self.tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled task: {name}")

        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)

        # Close all sources
        for name, source in self.sources.items():
            try:
                source.close()
                logger.info(f"Closed source: {name}")
            except Exception as e:
                logger.warning(f"Error closing source {name}: {e}")

        if self.slack_client:
            self.slack_client.send(text="News Server shutdown complete")

        logger.info("News Server shutdown complete")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="News Server - Run multiple cryptocurrency news sources"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode: print to stdout instead of file, no Slack'
    )
    parser.add_argument(
        '--sources',
        type=str,
        help=f'Comma-separated list of sources to run (available: {", ".join(NEWS_SOURCES.keys())})'
    )
    return parser.parse_args()


async def main() -> None:
    """Main entry point."""
    args = parse_args()
    _, config = get_config()

    # Parse sources if specified
    sources = args.sources.split(',') if args.sources else None

    server = NewsServer(config, sources=sources, debug=args.debug)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(server.shutdown())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await server.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
