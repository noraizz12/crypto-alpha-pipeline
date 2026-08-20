"""
CryptoSlate Scraper Source - Web-scraped cryptocurrency news per coin.

This source scrapes CryptoSlate's per-coin news pages for cryptocurrency news
articles. No API key required. Supports both real-time polling and historical
scraping with pagination.

Usage:
    # Run standalone real-time polling
    python -m lib.news.source_cryptoslate [--debug]

    # Download historical data (scrapes paginated news archives)
    python -m lib.news.source_cryptoslate --historical --start-date 2025-01-01 --end-date 2025-01-31
"""

import argparse
import asyncio
import json
import logging.config
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple

import aiohttp
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from slack_sdk.webhook.client import WebhookClient

from lib.util.config import get_config
from lib.util.time_util import date_to_str, dt_to_millis
from lib.util.slack import SLACK_WEBHOOK
from lib.util.directory import dir_manager
from lib.util.util import LOCAL, SYMBOL_BASE
from lib.util.logging_util import get_logging_config, KeyLogger
from lib.universe import Universe
from lib.news.news_util import load_tickers_for_date, MAJOR_TICKERS


logging.config.dictConfig(get_logging_config("news_cryptoslate"))
original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)


CRYPTOSLATE_BASE_URL = "https://cryptoslate.com"
POLL_INTERVAL_SECONDS = 300  # 5 minutes between full universe scans
REQUEST_DELAY_SECONDS = 2.0  # Delay between individual page requests
HISTORICAL_REQUEST_DELAY = 2.0
MAX_PAGES_PER_COIN = 50  # Safety limit for historical pagination

# User-Agent to avoid bot detection
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Mapping from our ticker symbols to CryptoSlate URL slugs
# Mapping from ticker symbols to CryptoSlate URL slugs.
# Scraped from cryptoslate.com/coins/ index pages.
# Coverage: ~85% of our expandable universe (389/460 tickers).
TICKER_TO_SLUG: Dict[str, str] = {
    "1000SATS": "sats",
    "AAVE": "aave",
    "ACE": "fusionist",
    "ACH": "alchemy-pay",
    "ACT": "acet",
    "ACX": "across-protocol",
    "ADA": "cardano",
    "AERGO": "aergo",
    "AERO": "aerodrome-finance",
    "AEVO": "aevo",
    "AGIX": "singularitynet",
    "AGLD": "adventure-gold",
    "AI": "ai-meta-club",
    "AI16Z": "ai16z",
    "AIA": "deagentai",
    "AIXBT": "aixbt-by-virtuals",
    "AKT": "akash-network",
    "ALGO": "algorand",
    "ALICE": "myneighboralice",
    "ALLO": "allora",
    "ALPHA": "alpha-finance-lab",
    "ALT": "alitas",
    "ANIME": "anime",
    "ANKR": "ankr-network",
    "ANT": "aragon",
    "APE": "apecoin",
    "APR": "apriori",
    "APT": "aptos",
    "AR": "arweave",
    "ARB": "arbitrum",
    "ARC": "arcadeum",
    "ARK": "ark",
    "ARKM": "arkham",
    "ARPA": "arpa-chain",
    "ASTER": "aster",
    "ASTR": "astar",
    "ATH": "aethir",
    "AUCTION": "bounce-token",
    "AUDIO": "audius",
    "AVAX": "avalanche",
    "AVNT": "avantis",
    "AXL": "axelar",
    "AXS": "axie-infinity",
    "B3": "b3-base",
    "BABY": "baby",
    "BAKE": "bakerytoken",
    "BAL": "balancer",
    "BAN": "banano",
    "BANANA": "apeswap-finance",
    "BANANAS31": "banana-for-scale",
    "BAND": "band-protocol",
    "BARD": "lombard",
    "BAT": "basic-attention-token",
    "BB": "bouncebit",
    "BCH": "bitcoin-cash",
    "BEAT": "audiera",
    "BEL": "bella-protocol",
    "BERA": "berachain",
    "BIGTIME": "big-time",
    "BIO": "bio-protocol",
    "BIRB": "moonbirds",
    "BLESS": "bless",
    "BLZ": "bluzelle",
    "BNB": "bnb",
    "BNT": "bancor",
    "BOME": "book-of-meme",
    "BOND": "barnbridge",
    "BONK": "bonk",
    "BRETT": "brett",
    "BREV": "brevis",
    "BTC": "bitcoin",
    "BTR": "bitrue-coin",
    "BLUR": "blur",
    "C98": "coin98",
    "CAKE": "pancakeswap",
    "CC": "canton-network",
    "CELO": "celo",
    "CELR": "celer-network",
    "CETUS": "cetus-protocol",
    "CFX": "conflux",
    "CGPT": "chaingpt",
    "CHILLGUY": "just-a-chill-guy",
    "CHR": "chromia",
    "CHZ": "chiliz",
    "CKB": "nervos-network",
    "CLO": "callisto-network",
    "COMP": "compound",
    "COTI": "coti",
    "COW": "cow-protocol",
    "CRV": "curve-dao-token",
    "CTK": "shentu",
    "CVC": "civic",
    "CVX": "convex-finance",
    "CYBER": "cyberconnect",
    "CYS": "cysic",
    "DAR": "mines-of-dalarnia",
    "DASH": "dash",
    "DEEP": "deepbook-protocol",
    "DEGEN": "degen",
    "DENT": "dent",
    "DEXE": "dexe",
    "DF": "dforce",
    "DOGE": "dogecoin",
    "DOLO": "dolomite",
    "DOT": "polkadot",
    "DRIFT": "drift",
    "DUSK": "dusk-network",
    "DYDX": "dydx",
    "DYM": "dymension",
    "EDU": "open-campus",
    "EGLD": "multiversx",
    "EIGEN": "eigenlayer",
    "ENA": "ethena",
    "ENJ": "enjin-coin",
    "ENS": "ethereum-name-service",
    "ERA": "era-token",
    "ESPORTS": "yooldo",
    "ETC": "ethereum-classic",
    "ETH": "ethereum",
    "ETHFI": "ether-fi",
    "ETHW": "ethereumpow",
    "EUL": "euler",
    "FARTCOIN": "fartcoin",
    "FET": "fetch-ai",
    "FF": "falcon-finance",
    "FIL": "filecoin",
    "FLM": "flamingo",
    "FLOKI": "floki",
    "FLOW": "flow-dapper-labs",
    "FLUID": "fluid",
    "FLUX": "zelcash-flux",
    "FOGO": "fogo",
    "FORM": "binaryx-new",
    "FRAX": "frax-share",
    "FRONT": "frontier",
    "FTM": "fantom",
    "FUN": "funtoken",
    "GALA": "gala",
    "GAL": "galxe",
    "GAS": "gas",
    "GIGGLE": "giggle-fund",
    "GLM": "golem",
    "GMT": "stepn",
    "GMX": "gmx",
    "GOAT": "sonic-the-goat",
    "GPS": "goplus-security",
    "GRASS": "grass",
    "GRIFFAIN": "griffain",
    "GRT": "the-graph",
    "GTC": "gitcoin",
    "HBAR": "hedera",
    "HEMI": "hemi",
    "HIFI": "hifi-finance",
    "HIGH": "highstreet",
    "HIVE": "hive-blockchain",
    "HMSTR": "hamster-kombat",
    "HNT": "helium",
    "HOME": "defi-app",
    "HOOK": "hooked-protocol",
    "HOT": "hydro-protocol",
    "HUMA": "huma-finance",
    "HYPER": "hyperlane",
    "HYPE": "hyperliquid",
    "ICNT": "impossible-cloud-network",
    "ICP": "internet-computer",
    "ID": "everest",
    "ILV": "illuvium",
    "IMX": "immutable-x",
    "INIT": "initia",
    "INJ": "injective",
    "IO": "io-net",
    "IOST": "iostoken",
    "IOTA": "iota",
    "IOTX": "iotex",
    "IP": "story",
    "IRYS": "irys",
    "JASMY": "jasmy",
    "JELLYJELLY": "jelly-my-jelly",
    "JOE": "joe-coin",
    "JTO": "jito",
    "JUP": "jupiter",
    "KAIA": "kaia",
    "KAS": "kaspa",
    "KAVA": "kava",
    "KDA": "kadena",
    "KEY": "selfkey",
    "KGEN": "kgen",
    "KITE": "kite",
    "KMNO": "kamino-finance",
    "KNC": "kyber-network-crystal-v2",
    "KSM": "kusama",
    "LA": "latoken",
    "LAYER": "unilayer",
    "LDO": "lido-dao",
    "LEVER": "leverfi",
    "LIGHT": "lightning",
    "LINA": "linear",
    "LINEA": "linea",
    "LINK": "chainlink",
    "LISTA": "lista-dao",
    "LIT": "lighter",
    "LOOM": "loom-network",
    "LPT": "livepeer",
    "LQTY": "liquity",
    "LRC": "loopring",
    "LSK": "lisk",
    "LTC": "litecoin",
    "LUNA": "luna-by-virtuals",
    "MAGIC": "magic",
    "MANA": "decentraland",
    "MANTA": "manta-network",
    "MASK": "mask-network",
    "MATIC": "polygon",
    "ME": "magic-eden",
    "MELANIA": "melania-meme",
    "MEME": "memetic",
    "MERL": "merlin-chain",
    "MET": "metronome",
    "METIS": "metisdao",
    "MEW": "cat-in-a-dogs-world",
    "MINA": "mina",
    "MIRA": "mira",
    "MKR": "maker",
    "MMT": "momentum",
    "MOCA": "moca-network",
    "MON": "mon-protocol",
    "MOODENG": "moo-deng-moodengsol-com",
    "MORPHO": "morpho",
    "MOVE": "bluemove",
    "MTL": "metal",
    "MUBARAK": "mubarak",
    "MYX": "myx-finance",
    "NEAR": "near-protocol",
    "NEIRO": "neiro-on-sol",
    "NEO": "neo",
    "NEWT": "newton-protocol",
    "NIGHT": "midnight",
    "NIL": "nillion",
    "NMR": "numeraire",
    "NOM": "nomina",
    "NOT": "notcoin",
    "OCEAN": "ocean-protocol",
    "OGN": "origin-protocol",
    "OMG": "omg-network",
    "OMNI": "omni-network",
    "ONDO": "ondo-finance",
    "ONE": "harmony",
    "ONG": "ontology-gas",
    "ONT": "ontology",
    "OP": "optimism",
    "OPEN": "openledger",
    "ORBS": "orbs",
    "ORCA": "orca",
    "ORDI": "ordi",
    "ORDER": "orderly-network",
    "PARTI": "particle-network",
    "PENDLE": "pendle",
    "PENGU": "pudgy-penguins",
    "PEOPLE": "constitutiondao",
    "PEPE": "pepe",
    "PERP": "perpetual-protocol",
    "PHA": "phala-network",
    "PHB": "phoenix-global",
    "PIEVERSE": "pieverse",
    "PIPPIN": "pippin",
    "PIXEL": "pixels",
    "PLUME": "plume",
    "PNUT": "peanut-the-squirrel",
    "POL": "polygon",
    "POLYX": "polymesh",
    "PONKE": "ponke",
    "PORTAL": "portal",
    "POWER": "power-protocol",
    "POWR": "powerledger",
    "PROM": "prom",
    "PROMPT": "wayfinder",
    "PROVE": "succinct",
    "PTB": "portal-to-bitcoin",
    "PUMP": "pumpbtc",
    "PUNDIX": "pundix-new",
    "PYTH": "pyth-network",
    "QNT": "quant-network",
    "QTUM": "qtum",
    "RAVE": "ravedao",
    "RDNT": "radiant-capital",
    "RECALL": "recall",
    "RED": "redstone",
    "REEF": "reef",
    "REI": "rei-network",
    "REN": "republic-protocol",
    "RENDER": "render-token",
    "REZ": "renzo",
    "RIF": "rifampicin",
    "RIVER": "river",
    "RLC": "iexec-rlc",
    "RNDR": "render-token",
    "ROSE": "oasis-network",
    "RPL": "rocket-pool",
    "RSR": "reserve-rights",
    "RUNE": "thorchain",
    "RVN": "ravencoin",
    "SAFE": "safe-deal",
    "SAGA": "saga",
    "SAND": "the-sandbox",
    "SAPIEN": "sapien-io",
    "SCRT": "secret",
    "SEI": "sei",
    "SENT": "sentient",
    "SFP": "safepal",
    "SHIB": "shiba-inu",
    "SIGN": "sign",
    "SIREN": "siren",
    "SKR": "solana-mobile-seeker",
    "SKY": "sky",
    "SNX": "synthetix",
    "SOL": "solana",
    "SOMI": "somnia",
    "SOPH": "sophon",
    "SPELL": "spell-token",
    "SPK": "spark",
    "SQD": "subsquid",
    "SRM": "serum",
    "SSV": "ssv-network",
    "STABLE": "stable",
    "STEEM": "steem",
    "STG": "stargate-finance",
    "STMX": "stormx",
    "STORJ": "storj",
    "STRAX": "stratis",
    "STRK": "starknet",
    "STX": "stacks",
    "SUI": "sui",
    "SUN": "sun",
    "SUPER": "superverse",
    "SUSHI": "sushiswap",
    "SXP": "solar",
    "SXT": "space-and-time",
    "SYRUP": "maple-finance",
    "TAG": "tagger",
    "TAO": "bittensor",
    "THETA": "theta-network",
    "TIA": "celestia",
    "TNSR": "tensor",
    "TON": "toncoin",
    "TOSHI": "toshi",
    "TRB": "tellor",
    "TREE": "treehouse",
    "TRU": "truefi-token",
    "TRUMP": "official-trump",
    "TRUTH": "swarm-network",
    "TRX": "tron",
    "TST": "test",
    "TURBO": "turbo",
    "TWT": "trust-wallet-token",
    "UAI": "unifai-network",
    "UB": "unibase",
    "UNI": "uniswap",
    "USELESS": "useless-coin",
    "USTC": "terraclassicusd",
    "USUAL": "usual",
    "UXLINK": "uxlink",
    "VANA": "vana",
    "VET": "vechain",
    "VINE": "vine-coin",
    "VIRTUAL": "virtuals-protocol",
    "VTHO": "vethor-token",
    "VVV": "venice-token",
    "W": "wormhole",
    "WAL": "walrus",
    "WAVES": "waves",
    "WCT": "walletconnect-token",
    "WIF": "dogwifhat",
    "WLD": "worldcoin",
    "WLFI": "world-liberty-financial",
    "WOO": "wootrade",
    "XAI": "new-xai-gork",
    "XAN": "anoma",
    "XCN": "onyxcoin",
    "XLM": "stellar",
    "XMR": "monero",
    "XPIN": "xpin-network",
    "XPL": "plasma",
    "XRP": "xrp",
    "XTZ": "tezos",
    "XVG": "verge",
    "XVS": "venus",
    "YFI": "yearn-finance",
    "YGG": "yield-guild-games",
    "ZAMA": "zama",
    "ZEC": "zcash",
    "ZEN": "horizen",
    "ZETA": "zetachain",
    "ZIL": "zilliqa",
    "ZK": "zksync",
    "ZKC": "boundless",
    "ZKJ": "polyhedra-network",
    "ZRO": "layerzero",
    "ZRX": "0x",
}


def _parse_relative_date(text: str) -> Optional[datetime]:
    """Parse relative date strings like '2 days ago', '5 hours ago'.

    Args:
        text: Relative time string from CryptoSlate

    Returns:
        Approximate datetime or None
    """
    text = text.strip().lower()
    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)

    patterns = [
        (r"(\d+)\s*second", "seconds"),
        (r"(\d+)\s*minute", "minutes"),
        (r"(\d+)\s*hour", "hours"),
        (r"(\d+)\s*day", "days"),
        (r"(\d+)\s*week", "weeks"),
        (r"(\d+)\s*month", "months"),
        (r"(\d+)\s*year", "years"),
    ]

    for pattern, unit in patterns:
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            if unit == "months":
                return now - timedelta(days=value * 30)
            if unit == "years":
                return now - timedelta(days=value * 365)
            if unit == "weeks":
                return now - timedelta(weeks=value)
            return now - timedelta(**{unit: value})

    # Try parsing absolute dates
    try:
        return date_parser.parse(text).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


class CryptoSlateSource:
    """Web scraper-based news source from CryptoSlate per-coin pages."""

    SOURCE_NAME = "cryptoslate"
    SUPPORTS_HISTORICAL = True

    def __init__(self, config: dict, debug: bool = False):
        self.debug = debug
        self.config = config

        self.slack_client = (
            WebhookClient(SLACK_WEBHOOK) if not LOCAL and not debug else None
        )

        # Load universe symbols for filtering relevant news
        self.universe = Universe(self.config)
        self.symbols = self.universe.load_universe_symbols(
            universe_source='file',
            filter='fittable',
            symbol_type=SYMBOL_BASE
        )
        self.tickers = {s.replace('USDT', '') for s in self.symbols}
        logger.info(f"Tracking news on {len(self.tickers)} tickers")

        # Build slug -> ticker reverse mapping for the active universe
        self._slug_to_ticker: Dict[str, str] = {}
        self._ticker_to_slug: Dict[str, str] = {}
        self._build_slug_mappings()

        # Track seen article URLs to avoid duplicates (for real-time mode)
        self.seen_urls: Set[str] = set()

        # File handle for writing news
        self.news_file = None
        self.current_date: Optional[str] = None

    def _build_slug_mappings(self) -> None:
        """Build mappings between tickers and CryptoSlate URL slugs."""
        self._slug_to_ticker.clear()
        self._ticker_to_slug.clear()
        for ticker in self.tickers:
            slug = TICKER_TO_SLUG.get(ticker)
            if slug:
                self._ticker_to_slug[ticker] = slug
                self._slug_to_ticker[slug] = ticker
            else:
                # Fallback: try lowercase ticker as slug
                fallback_slug = ticker.lower()
                self._ticker_to_slug[ticker] = fallback_slug
                self._slug_to_ticker[fallback_slug] = ticker

        logger.info(
            f"Built slug mappings for {len(self._ticker_to_slug)} tickers "
            f"({len(TICKER_TO_SLUG)} explicit, rest fallback)"
        )

    def _open_file(self, date_str: Optional[str] = None) -> None:
        """Open a new news file for the specified or current date."""
        if self.news_file:
            self.news_file.close()
        self.current_date = date_str or date_to_str()
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

    async def _fetch_page(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> Optional[str]:
        """Fetch a page and return HTML content.

        Args:
            session: aiohttp session
            url: URL to fetch

        Returns:
            HTML string or None on failure
        """
        try:
            async with session.get(
                url, headers=REQUEST_HEADERS, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    return await response.text()
                if response.status == 404:
                    logger.debug(f"Page not found: {url}")
                    return None
                logger.warning(f"HTTP {response.status} for {url}")
                return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching {url}")
            return None
        except aiohttp.ClientError as exc:
            logger.warning(f"Request failed for {url}: {exc}")
            return None

    def _parse_news_page(
        self, html: str, ticker: str, slug: str
    ) -> List[dict]:
        """Parse news articles from a CryptoSlate news page.

        Args:
            html: Raw HTML content
            ticker: Ticker symbol (e.g. BTC)
            slug: CryptoSlate slug (e.g. bitcoin)

        Returns:
            List of parsed article dicts
        """
        soup = BeautifulSoup(html, "html.parser")
        articles = []

        # Strategy 1: dedicated news page /news/{slug}/ — <article> tags
        article_tags = soup.find_all("article")

        # Strategy 2: coin page /coins/{slug}/ — news widget list items
        if not article_tags:
            news_widget = soup.select_one(".coin-news, .news-widget")
            if news_widget:
                article_tags = news_widget.find_all("a", href=True)

        # Strategy 3: generic list items with links
        if not article_tags:
            article_tags = soup.select(".post-list a, .list-post a, .news-list a")

        for tag in article_tags:
            try:
                parsed = self._parse_single_article(tag, ticker, slug)
                if parsed:
                    articles.append(parsed)
            except Exception as exc:
                logger.debug(f"Error parsing article for {ticker}: {exc}")

        return articles

    def _parse_single_article(
        self, tag, ticker: str, slug: str
    ) -> Optional[dict]:
        """Parse a single article element from CryptoSlate.

        CryptoSlate article structure:
            <article class="cs-category-*">
              <a href="...">
                <div class="cs-category-*__content">
                  <div class="eyebrow"><span>Category</span></div>
                  <h2 or h3>Title</h2 or h3>
                  <p>Excerpt</p>
                  <div class="post-meta">
                    <span class="read">3 weeks ago</span>
                  </div>
                </div>
              </a>
            </article>

        Args:
            tag: BeautifulSoup element
            ticker: Ticker symbol
            slug: CryptoSlate slug

        Returns:
            Parsed article dict or None
        """
        # Extract URL from the <a> inside <article>, or the tag itself
        if tag.name == "a":
            url = tag.get("href", "")
        else:
            link = tag.find("a", href=True)
            url = link["href"] if link else ""

        if not url:
            return None

        # Make URL absolute
        if url.startswith("/"):
            url = CRYPTOSLATE_BASE_URL + url

        # Skip non-article links (pagination, category pages, etc.)
        if "cryptoslate.com" in url and "/news/" in url and url.endswith("/news/"):
            return None

        # Extract title — try headings first
        title = ""
        for heading_tag in ["h2", "h3", "h1", "h4"]:
            heading = tag.find(heading_tag)
            if heading:
                title = heading.get_text(strip=True)
                break
        if not title:
            # Fall back to the <a> title attribute
            link = tag if tag.name == "a" else tag.find("a")
            if link:
                title = link.get("title", "")
        if not title or len(title) < 10:
            return None

        # Extract date from post-meta > span.read (e.g. "3 weeks ago")
        article_time = None

        # Primary: look for span.read inside div.post-meta
        meta_el = tag.select_one(".post-meta .read")
        if meta_el:
            article_time = _parse_relative_date(meta_el.get_text(strip=True))

        # Fallback: <time> tag with datetime attribute
        if article_time is None:
            time_tag = tag.find("time")
            if time_tag:
                dt_attr = time_tag.get("datetime")
                if dt_attr:
                    try:
                        article_time = date_parser.parse(dt_attr).replace(
                            tzinfo=None
                        )
                    except (ValueError, TypeError):
                        pass
                if article_time is None:
                    article_time = _parse_relative_date(
                        time_tag.get_text(strip=True)
                    )

        # Fallback: scan all spans for relative date text
        if article_time is None:
            for span in tag.find_all("span"):
                text = span.get_text(strip=True)
                if re.search(
                    r"\d+\s+(second|minute|hour|day|week|month|year)", text
                ):
                    article_time = _parse_relative_date(text)
                    if article_time:
                        break

        # Extract category from div.eyebrow > span
        category = ""
        eyebrow = tag.select_one(".eyebrow span")
        if eyebrow:
            category = eyebrow.get_text(strip=True)
        else:
            cat_el = tag.select_one(
                ".category, .tag, .label, .post-category"
            )
            if cat_el:
                category = cat_el.get_text(strip=True)

        # Extract excerpt from <p> (skip very short or date-like text)
        excerpt = ""
        for p_tag in tag.find_all("p"):
            text = p_tag.get_text(strip=True)
            if len(text) >= 20 and not re.match(
                r"^\d+\s+(second|minute|hour|day)", text
            ):
                excerpt = text
                break

        time_str = article_time.isoformat() if article_time else ""

        return {
            "title": title,
            "body": excerpt,
            "source": "cryptoslate",
            "url": url,
            "time": time_str,
            "sentiment": "",
            "type": category or "Article",
            "tickers": [ticker],
            "suggestions": [{"coin": ticker, "found": [ticker]}],
            "topics": [category] if category else [],
            "_id": url,
            "live_ts": dt_to_millis(),
            "api_source": "cryptoslate.com",
        }

    def _is_new_article(self, article: dict) -> bool:
        """Check if article has not been seen before."""
        url = article.get("_id", "")
        if url in self.seen_urls:
            return False
        self.seen_urls.add(url)

        # Prevent unbounded memory growth
        if len(self.seen_urls) > 50000:
            self.seen_urls = set(list(self.seen_urls)[-25000:])

        return True

    def _write_article(self, news_record: dict) -> None:
        """Write a news record to file or stdout."""
        if self.debug:
            tickers = news_record.get("tickers", [])
            title = news_record.get("title", "")[:80]
            time_str = news_record.get("time", "")
            print(f"  [{','.join(tickers)}] {time_str} | {title}")
        else:
            self.news_file.write(json.dumps(news_record) + "\n")
            self.news_file.flush()

    def _notify_slack(self, news_record: dict) -> None:
        """Send Slack notification for relevant news."""
        if self.slack_client is None:
            return

        tickers = news_record.get("tickers", [])
        if not tickers:
            return

        ticker = tickers[0]
        title = news_record.get("title", "")[:100]
        category = news_record.get("type", "")

        msg = f"[CryptoSlate] :newspaper: [{ticker}] {category}: {title}"
        try:
            self.slack_client.send(text=msg)
        except Exception as exc:
            logger.warning(f"Failed to send Slack notification: {exc}")

    async def _scrape_coin_news(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        slug: str,
        page: int = 1,
    ) -> List[dict]:
        """Scrape news for a single coin.

        Tries the dedicated /news/{slug}/ page first (has pagination and more
        articles). Falls back to /coins/{slug}/ page (has a small news widget
        with ~6 articles) if the news page doesn't exist.

        Args:
            session: aiohttp session
            ticker: Ticker symbol
            slug: CryptoSlate URL slug
            page: Page number (1-indexed)

        Returns:
            List of parsed article dicts
        """
        # Only the dedicated news page supports pagination
        if page == 1:
            url = f"{CRYPTOSLATE_BASE_URL}/news/{slug}/"
        else:
            url = f"{CRYPTOSLATE_BASE_URL}/news/{slug}/page/{page}/"

        html = await self._fetch_page(session, url)

        # Fall back to coin page for page 1 if news page doesn't exist
        if html is None and page == 1:
            url = f"{CRYPTOSLATE_BASE_URL}/coins/{slug}/"
            html = await self._fetch_page(session, url)

        if html is None:
            return []

        return self._parse_news_page(html, ticker, slug)

    async def _scrape_universe(
        self, session: aiohttp.ClientSession
    ) -> int:
        """Scrape news for all tickers in the universe.

        Args:
            session: aiohttp session

        Returns:
            Total new articles found
        """
        total_new = 0

        for ticker, slug in self._ticker_to_slug.items():
            articles = await self._scrape_coin_news(session, ticker, slug)

            for article in articles:
                if self._is_new_article(article):
                    total_new += 1
                    self._write_article(article)

                    if article.get("tickers"):
                        self._notify_slack(article)

            if articles:
                logger.debug(f"{ticker}: {len(articles)} articles on page")

            await asyncio.sleep(REQUEST_DELAY_SECONDS)

        return total_new

    async def download_historical(
        self,
        start_date: datetime,
        end_date: datetime,
        max_pages: int = MAX_PAGES_PER_COIN,
    ) -> int:
        """Download historical news by scraping paginated news archives.

        Scrapes multiple pages per coin until articles fall outside the date
        range. Articles are assigned to daily files based on their parsed
        publication date.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            max_pages: Maximum pages to scrape per coin

        Returns:
            Total number of articles downloaded
        """
        total_articles = 0
        seen_urls: Set[str] = set()

        logger.info(
            f"Starting historical download from {start_date.date()} to {end_date.date()} "
            f"for {len(self._ticker_to_slug)} coins"
        )

        # Track open file handles by date
        open_files: Dict[str, object] = {}

        def get_file_for_date(dt_obj: datetime):
            ds = dt_obj.strftime("%Y%m%d")
            if ds not in open_files:
                date_dir = f"{dir_manager.NEWS_DIR_NEW}/{ds}"
                os.makedirs(date_dir, exist_ok=True)
                filepath = f"{date_dir}/{self.SOURCE_NAME}.{ds}.csv"
                # pylint: disable=consider-using-with
                open_files[ds] = open(filepath, "a", encoding="utf-8")
                logger.info(f"Opened historical file: {filepath}")
            return open_files[ds]

        try:
            async with aiohttp.ClientSession() as session:
                for ticker, slug in self._ticker_to_slug.items():
                    coin_articles = 0
                    reached_start = False

                    for page in range(1, max_pages + 1):
                        articles = await self._scrape_coin_news(
                            session, ticker, slug, page=page
                        )

                        if not articles:
                            break

                        page_has_valid = False
                        for article in articles:
                            url = article.get("_id", "")
                            if url in seen_urls:
                                continue
                            seen_urls.add(url)

                            time_str = article.get("time", "")
                            if not time_str:
                                continue

                            try:
                                article_dt = datetime.fromisoformat(time_str)
                            except ValueError:
                                continue

                            # Skip articles after end_date
                            if article_dt > end_date + timedelta(days=1):
                                continue

                            # Stop if we've gone past start_date
                            if article_dt < start_date:
                                reached_start = True
                                continue

                            page_has_valid = True

                            if self.debug:
                                tickers = article.get("tickers", [])
                                title = article.get("title", "")[:80]
                                print(
                                    f"  [{','.join(tickers)}] "
                                    f"{article_dt.date()} | {title}"
                                )
                            else:
                                fh = get_file_for_date(article_dt)
                                fh.write(json.dumps(article) + "\n")
                                fh.flush()

                            coin_articles += 1

                        if reached_start and not page_has_valid:
                            break

                        await asyncio.sleep(HISTORICAL_REQUEST_DELAY)

                    if coin_articles > 0:
                        logger.info(f"  {ticker}: {coin_articles} articles")
                    total_articles += coin_articles

        finally:
            for fh in open_files.values():
                fh.close()

        logger.info(f"Historical download complete: {total_articles} total articles")
        return total_articles

    async def run_realtime(self) -> None:
        """Main loop for real-time news scraping."""
        if not self.debug:
            self._open_file()

        if self.slack_client:
            self.slack_client.send(text="Starting CryptoSlate Source")

        logger.info(
            f"Starting CryptoSlate scraping loop "
            f"({len(self._ticker_to_slug)} coins, "
            f"{POLL_INTERVAL_SECONDS}s interval)"
        )

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    self._check_date_rollover()

                    new_count = await self._scrape_universe(session)

                    if new_count > 0:
                        logger.info(f"Found {new_count} new articles")
                    else:
                        logger.debug("No new articles")

                except Exception as exc:
                    logger.error(
                        f"Error in scraping loop: {exc}", exc_info=True
                    )

                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    def close(self) -> None:
        """Clean up resources."""
        if self.news_file:
            self.news_file.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CryptoSlate Scraper Source - Per-coin cryptocurrency news"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: print to stdout instead of file, no Slack",
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Download historical data instead of real-time polling",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for historical download (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for historical download (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=MAX_PAGES_PER_COIN,
        help=f"Maximum pages to scrape per coin for historical (default: {MAX_PAGES_PER_COIN})",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of tickers to scrape (default: full universe)",
    )
    return parser.parse_args()


async def main() -> None:
    """Main entry point for standalone execution."""
    args = parse_args()
    _, config = get_config()

    source = CryptoSlateSource(config, debug=args.debug)

    # Override universe if specific tickers requested
    if args.tickers:
        override_tickers = {t.strip().upper() for t in args.tickers.split(",")}
        source.tickers = override_tickers
        source._build_slug_mappings()
        logger.info(f"Overriding universe with {len(override_tickers)} tickers")

    try:
        if args.historical:
            if not args.start_date or not args.end_date:
                logger.error(
                    "--start-date and --end-date required for historical mode"
                )
                return

            start = datetime.strptime(args.start_date, "%Y-%m-%d")
            end = datetime.strptime(args.end_date, "%Y-%m-%d")

            if start > end:
                logger.error("Start date must be before end date")
                return

            await source.download_historical(
                start, end, max_pages=args.max_pages
            )
        else:
            await source.run_realtime()
    except KeyboardInterrupt:
        logger.info("Shutting down CryptoSlate source")
    finally:
        source.close()


if __name__ == "__main__":
    asyncio.run(main())
