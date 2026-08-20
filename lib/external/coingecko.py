"""CoinGecko API interface for cryptocurrency market data.

This module provides functionality to download market capitalization and category
data from CoinGecko API for cryptocurrencies traded on Binance perpetual futures.
The data is used for:
- Market cap-based position sizing and risk management
- Category-based grouping for correlation analysis
- Universe filtering based on market cap rankings

The module respects CoinGecko's rate limits with built-in delays between requests.
"""
import time
import logging
from datetime import datetime as dt, timezone
from typing import List, Dict, Any

import pandas as pd
from pycoingecko import CoinGeckoAPI

from lib.util.time_util import date_to_str
from lib.util.opsgenie import raise_alert, HIGH
from .binance_utils import get_exchange_info

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


API_KEY = "CG-kZc5n9vdxQDKSFQ5EXf41iCn"

# Number of top categories to fetch for categorization
TOP_N_CATEGORIES = 30
MARKET_DATA_FIELDS = [
    'total_value_locked',
    'mcap_to_tvl_ratio',
    'fdv_to_tvl_ratio',
    'fully_diluted_valuation',
    'market_cap_fdv_ratio',
    'total_supply',
    'max_supply',
    'max_supply_infinite',
    'circulating_supply',]
COMMUNITY_DATA_FIELDS = [
    'facebook_likes',
    'reddit_average_posts_48h',
    'reddit_average_comments_48h',
    'reddit_subscribers',
    'reddit_accounts_active_48h',
    'telegram_channel_user_count',]
DEVELOPER_DATA_FIELDS = [
    'forks',
    'stars',
    'subscribers',
    'total_issues',
    'closed_issues',
    'pull_requests_merged',
    'pull_request_contributors',
    'commit_count_4_weeks',]
OTHER_METADATA_FIELDS = [
    'sentiment_votes_up_percentage',
    'sentiment_votes_down_percentage',
    'watchlist_portfolio_users',]


class CoinGecko:
    """CoinGecko API client for fetching cryptocurrency market data.
    
    Handles downloading market cap data and category classifications for
    cryptocurrencies. Filters results to only include coins traded on
    Binance perpetual futures.
    
    Attributes:
        debug: If True, prints data instead of saving to files
        limit: Maximum number of coins per category to fetch
        pages_needed: Number of pages to fetch for market cap data
        top_n_categories: Number of top categories to process
        output_dir: Directory to save downloaded data
        cg: CoinGeckoAPI client instance
    """
    def __init__(self, limit: int, pages_needed: int, output_dir: str, debug: bool = False, top_n_categories: int = TOP_N_CATEGORIES):
        """Initialize CoinGecko API client.
        
        Args:
            limit: Maximum coins per category to fetch
            pages_needed: Number of pages for market cap data (100 coins per page)
            output_dir: Directory path for saving output files
            debug: If True, print data instead of saving files
            top_n_categories: Number of top market cap categories to process
        """
        self.debug = debug
        self.limit = limit
        self.pages_needed = pages_needed
        self.top_n_categories = top_n_categories
        self.output_dir = output_dir
        self.cg = CoinGeckoAPI(api_key=API_KEY)

    def _get_binance_perp_universe(self) -> List[str]:
        """Get list of perpetual futures symbols traded on Binance.
        
        Queries Binance exchange info and filters for active USDT perpetual
        contracts with COIN underlying type.
        
        Returns:
            List[str]: Symbol names (e.g., ['BTCUSDT', 'ETHUSDT'])
        """
        data = get_exchange_info()
        df = pd.DataFrame(data['symbols'])
        df = df.loc[(df['contractType'] == 'PERPETUAL') & (df['status'] == 'TRADING') & (df['quoteAsset'] == 'USDT') & (df['underlyingType'] == 'COIN')]
        universe = df['symbol'].to_list()
        return universe


    def download_coingecko_category_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich market cap data with category classifications.
        
        Fetches top categories by market cap and creates boolean columns indicating
        whether each coin belongs to each category. Also includes GMCI index categories.
        
        Args:
            df: DataFrame with coin data (must have 'id' column)
            
        Returns:
            pd.DataFrame: Input df with added category columns:
                - category_{name}: Boolean columns for each category
                - category_count: Total number of categories per coin
                
        Notes:
            - Adds 1.5 second delay between API calls to respect rate limits
            - Includes special handling for GMCI (Galaxy Market Cap Index) categories
            - Categories are normalized (spaces replaced with underscores, lowercase)
        """
        logger.info(f"Fetching {self.top_n_categories} coin categories from CoinGecko...")
        try:
            categories = self.cg.get_coins_categories()
            top_categories = categories[:self.top_n_categories]
            extra_categories = self.cg.get_coins_categories_list()
            extra_categories = [{'id': cat['category_id'], 'name': cat['name']} for cat in extra_categories if 'GMCI' in cat['name']]
            top_categories = top_categories + extra_categories
            logger.info(f"Retrieved {len(categories)} categories, using top {len(top_categories)}")

            df_coin_ids = set(df['id'])
            category_to_coins = {cat['name']: set() for cat in top_categories}
            categorized_coins = set()
            errors = 0
            for category in top_categories:
                time.sleep(1.5)
                category_id = category['id']
                category_name = category['name']
                logger.info(f"Fetching coins for category: {category_name}")
                try:
                    category_coins = self.cg.get_coins_markets(
                        vs_currency='usd',
                        category=category_id,
                        order='market_cap_desc',
                        per_page=self.limit,
                    )
                except Exception as e:
                    logger.error(f"Error fetching coins for category {category_name}: {e}")
                    errors += 1
                    continue
                for coin in category_coins:
                    coin_id = coin['id']
                    if coin_id in df_coin_ids:
                        category_to_coins[category_name].add(coin_id)
                        categorized_coins.add(coin_id)

            logger.info(f"Found category data for {len(categorized_coins)} out of {len(df_coin_ids)} coins")
            if errors:
                raise_alert(key='CoinGecko category fetch errors', priority=HIGH, description=f"{errors}/{len(top_categories)} categories failed")
            # Add categories to our dataframe
            for category_name, coin_ids in category_to_coins.items():
                col_name = f"category_{category_name.replace(' ', '_').lower()}"
                df[col_name] = df['id'].isin(coin_ids)
            df['category_count'] = df[[col for col in df.columns if col.startswith('category_')]].sum(axis=1)
        except Exception as e:
            logger.error(f"Error in category processing: {e}")
            raise
        return df


    def download_coingecko_marketcap_data(self) -> pd.DataFrame:
        """Download market capitalization data for top cryptocurrencies.
        
        Fetches market cap rankings and values for coins, processing multiple
        pages to get comprehensive coverage of the market.
        
        Returns:
            pd.DataFrame: Market cap data with columns:
                - id: CoinGecko coin ID
                - symbol: Coin ticker symbol
                - name: Full coin name
                - market_cap: Market capitalization in USD
                - market_cap_rank: Global ranking by market cap
                
        Notes:
            - Fetches 100 coins per page
            - Removes duplicate symbols (keeps first occurrence)
            - Adds 1.5 second delay between page requests
        """
        all_coins = []
        errors = 0
        for page in range(1, self.pages_needed + 1):
            time.sleep(1.5)
            logger.info(f"Fetching page {page} of {self.pages_needed}...")
            try:
                coins_page = self.cg.get_coins_markets(
                    vs_currency='usd',
                    order='market_cap_desc',
                    per_page=100,
                    page=page,
                )
            except Exception as e:
                logger.error(f"Error fetching page {page}: {e}")
                errors += 1
                continue
            all_coins.extend(coins_page)
        if not all_coins:
            raise_alert(key='CoinGecko marketcap fetch errors', priority=HIGH, description=f"all {self.pages_needed} pages failed")
            raise RuntimeError(f"CoinGecko: all {self.pages_needed} marketcap pages failed")
        if errors:
            raise_alert(key='CoinGecko marketcap fetch errors', priority=HIGH, description=f"{errors}/{self.pages_needed} pages failed")

        df = pd.DataFrame(all_coins)
        df = df[['id', 'symbol', 'name', 'market_cap', 'market_cap_rank']]
        df = df.drop_duplicates('symbol', keep='first')
        df = df.reset_index(drop=True)
        return df
    
    def _extract_coin_details(self, coin_data: Dict[str, Any], coin_id: str) -> Dict[str, Any]:
        """Extract detailed fields from a single coin's API response.

        Args:
            coin_data: Response from get_coin_by_id API call

        Returns:
            Dict with extracted fields from market_data, community_data,
            developer_data, and other metadata
        """
        result = {}

        # Extract market data fields
        market_data = coin_data['market_data']
        for field in MARKET_DATA_FIELDS:
            value = market_data.get(field)
            # Handle nested dicts with 'usd' key
            if isinstance(value, dict) and 'usd' in value:
                value = value['usd']
            result[field] = value

        # Extract community data fields
        community_data = coin_data['community_data']
        for field in COMMUNITY_DATA_FIELDS:
            result[field] = community_data.get(field)

        # Extract developer data fields
        developer_data = coin_data['developer_data']
        for field in DEVELOPER_DATA_FIELDS:
            result[field] = developer_data.get(field)

        # Extract other metadata fields
        for field in OTHER_METADATA_FIELDS:
            result[field] = coin_data.get(field)

        result['id'] = coin_id
        return result

    def download_coingecko_detailed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich dataframe with detailed coin data from get_coin_by_id.

        Fetches additional data not available in the bulk markets endpoint:
        - Market data: TVL, FDV, supply metrics
        - Community data: Reddit, Telegram, Facebook stats
        - Developer data: GitHub metrics
        - Sentiment and watchlist data

        Args:
            df: DataFrame with 'id' column containing CoinGecko coin IDs

        Returns:
            pd.DataFrame: Input df with additional columns for detailed metrics

        Notes:
            - Calls get_coin_by_id for each coin (slower but more data)
            - Adds 1.5 second delay between API calls for rate limiting
            - Logs progress every 10 coins
        """
        logger.info(f"Fetching detailed data for {len(df)} coins...")
        detailed_data = []

        for count, (_, row) in enumerate(df.iterrows()):
            time.sleep(1.5)
            coin_id = row['id']
            if count % 10 == 0:
                logger.info(f"Fetching detailed data: {count + 1}/{len(df)} ({coin_id})")
            try:
                coin_data = self.cg.get_coin_by_id(
                    coin_id,
                    localization=False,
                    tickers=False,
                    market_data=True,
                    community_data=True,
                    developer_data=True,
                    sparkline=False
                )
            except Exception as e:
                logger.error(f"Error fetching detailed data for {coin_id}: {e}")
                continue

            try:
                detailed_data.append(self._extract_coin_details(coin_data, coin_id))
            except Exception as e:
                logger.error(f"Error extracting detailed data for {coin_id}: {e}")
                continue

        failed = len(df) - len(detailed_data)
        if failed:
            raise_alert(key='CoinGecko detailed fetch errors', priority=HIGH, description=f"{failed}/{len(df)} coins failed")
        if detailed_data:
            details_df = pd.DataFrame(detailed_data)
            df = pd.merge(df, details_df, on='id', how='left')
            logger.info(f"Added {len(details_df.columns) - 1} detailed data columns")
        else:
            logger.error("No CoinGecko detailed data found!")
            raise RuntimeError(f"CoinGecko: all {len(df)} detailed data requests failed")

        return df

    def download_coingecko_data(self):
        """Main method to download and process all CoinGecko data.

        Orchestrates the full data download process:
        1. Downloads market cap data for top coins
        2. Filters to only Binance perpetual futures symbols
        3. Enriches with detailed coin data (TVL, supply, community, developer, sentiment)
        4. Enriches with category classifications
        5. Saves to parquet file with timestamp

        The output file contains comprehensive market data for use in:
        - Position sizing based on market cap
        - Risk management and correlation analysis
        - Universe selection and filtering
        - Alternative data signals (community, developer activity)

        Output format: marketcap.{YYYYMMDD_HHMM}.parquet

        Notes:
            - Converts symbols to uppercase with USDT suffix
            - Filters to Binance perp universe BEFORE detailed fetch (reduces API calls)
            - Adds UTC timestamp to all records
            - In debug mode, prints sample data instead of saving
        """
        df = self.download_coingecko_marketcap_data()
        df['symbol'] = df['symbol'].str.upper() + 'USDT'

        # Filter to Binance perp universe BEFORE detailed fetch to minimize API calls
        binance_perp_universe = self._get_binance_perp_universe()
        df = df.loc[df['symbol'].isin(binance_perp_universe)].copy()
        logger.info(f"Filtered to {len(df)} Binance perp symbols")

        # Fetch detailed data (community, developer, sentiment, supply metrics)
        df = self.download_coingecko_detailed_data(df)

        # Fetch category data
        df = self.download_coingecko_category_data(df)

        now = dt.now(timezone.utc)
        df['timestamp'] = now
        date_str = date_to_str(now)

        if self.debug:
            print(f"Get market cap data at {now}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(df[['symbol', 'market_cap', 'market_cap_rank', 'circulating_supply',
                      'sentiment_votes_up_percentage', 'stars']].head(10).to_markdown())
        else:
            filename = f"{self.output_dir}/marketcap.{date_str}.parquet"
            logger.info(f'Dumping {filename}')
            df.to_parquet(filename)
