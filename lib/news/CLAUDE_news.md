# CLAUDE_news.md

## Purpose and Key Responsibilities

The `lib/news` module provides cryptocurrency news ingestion from multiple sources for use in alpha generation. It supports both real-time streaming and historical data download.

**Key responsibilities:**
- Real-time news collection from WebSocket and REST API sources
- Historical news download for backtesting
- News normalization to a common format
- Ticker extraction and relevance filtering
- Slack notifications for relevant news
- File-based persistence with daily rollover

## Architecture

### News Server (Orchestrator)
`news_server.py` - Runs all configured news sources concurrently in real-time mode.

```bash
# Run all sources
python -m lib.news.news_server [--debug]

# Run specific sources
python -m lib.news.news_server --sources treeofalpha,cryptonews
```

### Source Plugins
Each news source is implemented as a separate module with a common interface:

| Source | File | Type | Historical |
|--------|------|------|------------|
| TreeOfAlpha | `source_treeofalpha.py` | WebSocket | Yes (partial, ~3000 items) |
| CryptoNews | `source_cryptonews.py` | REST API | Yes |
| CryptoPanic | `source_cryptopanic.py` | REST API | Yes (limited by tier) |

### Source Interface
Each source class must implement:
- `SOURCE_NAME: str` - Unique identifier
- `SUPPORTS_HISTORICAL: bool` - Whether historical download is supported
- `run_realtime() -> None` - Main loop for real-time collection
- `close() -> None` - Clean up resources
- `download_historical(start_date, end_date)` - (if SUPPORTS_HISTORICAL)

## Key Files

### news_server.py
**Orchestrator** that runs multiple news sources concurrently.

Key components:
- `NEWS_SOURCES` - Registry mapping source names to classes
- `NewsServer` - Main orchestrator class
  - `run()` - Start all sources as async tasks
  - `shutdown()` - Graceful shutdown with task cancellation
  - `_run_source()` - Per-source runner with restart logic

Features:
- Automatic restart on source failure (up to 10 times)
- Exponential backoff on failures
- Signal handlers for graceful shutdown (SIGTERM, SIGINT)
- Slack notifications on start/stop

### source_treeofalpha.py
**WebSocket-based** real-time news from TreeOfAlpha with historical download support.

```bash
# Real-time streaming
python -m lib.news.source_treeofalpha [--debug]

# Historical download (partial history, ~3000 most recent items)
python -m lib.news.source_treeofalpha --historical --start-date 2026-01-01 --end-date 2026-01-14
```

Key components:
- `TreeOfAlphaSource` - WebSocket client + REST API for historical
- Heartbeat mechanism to keep connection alive
- Date rollover for daily file rotation

**Note:** Historical download uses the public API endpoint which returns ~3000 most recent items (no auth required). Full history requires API key authentication.

Output: `{NEWS_DIR_NEW}/{YYYYMMDD}/treeofalpha.{YYYYMMDD}.csv`

### source_cryptonews.py
**REST API-based** news with sentiment analysis from CryptoNews API.

```bash
# Real-time polling
python -m lib.news.source_cryptonews [--debug]

# Historical download
python -m lib.news.source_cryptonews --historical --start-date 2024-01-01 --end-date 2024-01-31
```

Key components:
- `CryptoNewsSource` - REST API client
- 60-second polling interval for real-time
- Deduplication using article URL/title
- Sentiment field (Positive/Negative/Neutral)

Output: `{NEWS_DIR_NEW}/{YYYYMMDD}/cryptonews.{YYYYMMDD}.csv`

**API Limitations (Trial Plan):**
- 3 items per request (increase to 100 with paid plan)
- Requires `tickers` parameter
- Historical date format: `MMDDYYYY-MMDDYYYY`

### source_cryptopanic.py
**REST API-based** aggregated news from CryptoPanic with community voting.

```bash
# Real-time polling
python -m lib.news.source_cryptopanic [--debug]

# Historical download
python -m lib.news.source_cryptopanic --historical --start-date 2026-01-01 --end-date 2026-01-14
```

Key components:
- `CryptoPanicSource` - REST API client (v2 Developer API)
- 60-second polling interval for real-time
- Community voting data (positive/negative/important)
- Ticker extraction from title/description with name mapping (Bitcoin -> BTC)

Output: `{NEWS_DIR_NEW}/{YYYYMMDD}/cryptopanic.{YYYYMMDD}.csv`

**API Tier (Growth):**
- Base URL: `https://cryptopanic.com/api/growth/v2/posts/`
- Full pagination support for historical downloads
- Filters: `rising`, `hot`, `bullish`, `bearish`, `important`
- Note: Some fields like `instruments` may still require Enterprise tier

## Output Format

### TreeOfAlpha Output
Raw JSON messages from WebSocket with added `live_ts` timestamp:
```json
{
  "title": "Bitcoin breaks $100k",
  "body": "Full article text...",
  "suggestions": [{"coin": "BTC", "found": ["BTC"]}],
  "live_ts": 1704067200000
}
```

### CryptoNews Output
Normalized format with sentiment:
```json
{
  "title": "Bitcoin ETF Approved",
  "body": "Article text...",
  "source": "coindesk",
  "url": "https://...",
  "time": "2024-01-10T15:30:00Z",
  "sentiment": "Positive",
  "type": "Article",
  "tickers": ["BTC"],
  "suggestions": [{"coin": "BTC", "found": ["BTC"]}],
  "topics": ["ETF", "Regulation"],
  "_id": "unique_article_id",
  "live_ts": 1704067200000,
  "api_source": "cryptonews-api.com"
}
```

## Configuration

News sources use the main config for:
- Universe filtering (which tickers to track)
- Slack webhook for notifications
- Data directories

No news-specific config parameters currently.

## Adding a New Source

1. Create `source_{name}.py` with class implementing:
   ```python
   class NewSource:
       SOURCE_NAME = "newsource"
       SUPPORTS_HISTORICAL = True/False

       def __init__(self, config: dict, debug: bool = False):
           ...

       async def run_realtime(self) -> None:
           ...

       async def download_historical(self, start_date, end_date) -> int:
           # Only if SUPPORTS_HISTORICAL
           ...

       def close(self) -> None:
           ...
   ```

2. Add to `NEWS_SOURCES` registry in `news_server.py`:
   ```python
   NEWS_SOURCES = {
       'treeofalpha': TreeOfAlphaSource,
       'cryptonews': CryptoNewsSource,
       'newsource': NewSource,  # Add here
   }
   ```

3. Export from `__init__.py`

## Common Patterns

### Debug Mode
All sources support `--debug` flag:
- Prints to stdout instead of file
- Disables Slack notifications
- Useful for testing API connectivity

### Universe Filtering
Sources filter news by universe tickers:
```python
self.universe = Universe(self.config)
self.symbols = self.universe.load_universe_symbols(
    universe_source='file',
    filter='fittable',
    symbol_type=SYMBOL_BASE
)
```

### File Management
Daily file rotation with date string in filename:
```python
filepath = f"{dir_manager.NEWS_DIR}/news.{date_str}.csv"
```

### Error Handling
- Automatic reconnection on WebSocket failures
- Exponential backoff on repeated failures
- Slack alerts after multiple consecutive failures

## Future Sources (To-Do)

The following Tier 1 news sources have been identified for potential integration:

### LunarCrush
- **URL:** https://lunarcrush.com/developers/api
- **Type:** REST API
- **Free Tier:** Yes (limited requests)
- **Features:**
  - Social sentiment metrics (Galaxy Score, AltRank)
  - News aggregation with social context
  - Influencer tracking
  - Real-time social volume data
- **API Example:** `https://lunarcrush.com/api3/coins/BTC`
- **Notes:** Strong social sentiment signals; combines news with Twitter/Reddit activity

### Messari
- **URL:** https://messari.io/api
- **Type:** REST API
- **Free Tier:** Yes (limited)
- **Features:**
  - Research-quality news and analysis
  - Asset profiles and fundamentals
  - On-chain metrics
  - Governance/protocol updates
- **API Example:** `https://data.messari.io/api/v1/news`
- **Notes:** Higher quality, research-focused content; good for fundamental news

### The Block (Data Dashboard)
- **URL:** https://www.theblock.co/api
- **Type:** REST API
- **Free Tier:** Limited
- **Features:**
  - Institutional-grade news
  - On-chain data and charts
  - Research reports
- **Notes:** Premium source; may require paid subscription for full access

### CoinGecko News
- **URL:** https://www.coingecko.com/en/api
- **Type:** REST API
- **Free Tier:** Yes
- **Features:**
  - News feed per coin
  - Status updates from projects
  - Exchange/market news
- **API Example:** `https://api.coingecko.com/api/v3/news`
- **Notes:** Good supplementary source; already used for price data

### Priority Order for Implementation
1. **LunarCrush** - Unique social sentiment signals
2. **Messari** - Research-quality content
3. **CoinGecko** - Easy integration if already using their API
4. **The Block** - If budget allows for premium access
