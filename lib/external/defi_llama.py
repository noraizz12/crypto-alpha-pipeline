"""DeFiLlama API interface for DeFi protocol analytics.

This module provides functionality to download Total Value Locked (TVL) and yield
data from DeFiLlama API. The data is used for:
- Tracking DeFi protocol health and growth metrics
- Analyzing yield opportunities across different protocols
- Understanding capital flows in the DeFi ecosystem
- Risk assessment based on TVL trends

The module respects API rate limits with built-in delays between requests.
"""
import time
import logging
from datetime import datetime as dt, timezone
from typing import List

import pandas as pd
import requests

from lib.util.files import safe_mkdir
from lib.util.time_util import to_datetime, date_to_str
from lib.util.opsgenie import raise_alert, clear_alert, HIGH

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# API endpoint URLs
DEFILLAMA_BASE_URL = "https://api.llama.fi"
DEFILLAMA_YIELDS_BASE_URL = "https://yields.llama.fi"
PROTOCOLS_ENDPOINT = "/protocols"
PROTOCOL_TVL_ENDPOINT = "/protocol/{}"
POOLS_ENDPOINT = "/pools"
POOL_CHART_ENDPOINT = "/chart/{}"


def get_defillama_protocols() -> List:
    """Fetch list of all DeFi protocols from DeFiLlama.

    Returns:
        List: Protocol data including names, TVL, chains, and categories

    Notes:
        - No rate limiting needed for this endpoint
    """
    logger.info("Fetching protocols from DeFiLlama...")
    response = requests.get(f"{DEFILLAMA_BASE_URL}{PROTOCOLS_ENDPOINT}", timeout=30)
    response.raise_for_status()
    protocols = response.json()
    logger.info(f"Retrieved {len(protocols)} protocols")
    return protocols


def download_historical_tvl(protocols: List, limit: int) -> pd.DataFrame:
    """Download historical TVL data for the top protocols by TVL.

    Fetches time series TVL data for the largest DeFi protocols, providing
    insights into protocol growth and capital flows over time.

    Args:
        protocols: List of protocol data from get_defillama_protocols()
        limit: Maximum number of protocols to process (sorted by TVL)

    Returns:
        pd.DataFrame: Historical TVL data with columns:
            - protocol_name: Human-readable protocol name
            - protocol_slug: URL-friendly protocol identifier
            - chain: Blockchain the protocol operates on
            - category: Protocol category (DEX, Lending, etc.)
            - date: Date of the TVL snapshot
            - timestamp: Unix timestamp
            - tvl: Total value locked in USD

    Notes:
        - Adds 1 second delay between protocol requests
        - Converts timestamps to dates for easier analysis
    """

    def safe_tvl_key(x):
        """Safe key function for sorting protocols by TVL."""
        tvl = x.get('tvl')
        return tvl if tvl is not None else 0

    sorted_protocols = sorted(protocols, key=safe_tvl_key, reverse=True)
    top_protocols = sorted_protocols[:limit]
    logger.info(f"Downloading historical TVL for top {len(top_protocols)} protocols")

    all_tvl_data = []
    errors = 0
    for i, protocol in enumerate(top_protocols):
        time.sleep(1)
        protocol_slug = protocol.get('slug')
        logger.info(f"Fetching historical TVL for {protocol_slug} ({i}/{len(top_protocols)})")
        try:
            response = requests.get(f"{DEFILLAMA_BASE_URL}{PROTOCOL_TVL_ENDPOINT.format(protocol_slug)}", timeout=30)
            response.raise_for_status()
            protocol_data = response.json()
        except Exception as e:
            logger.error(f"Error fetching historical TVL for {protocol_slug}: {e}")
            errors += 1
            continue
        if 'tvl' in protocol_data:
            tvl_history = protocol_data['tvl']
            for entry in tvl_history:
                all_tvl_data.append({
                    'protocol_name': protocol.get('name'),
                    'protocol_slug': protocol_slug,
                    'chain': protocol.get('chain'),
                    'category': protocol.get('category'),
                    'date': entry.get('date'),
                    'timestamp': entry.get('date'),
                    'tvl': entry.get('totalLiquidityUSD'),
                })

    if not all_tvl_data:
        raise_alert(key='DeFiLlama TVL fetch errors', priority=HIGH, description=f"all {len(top_protocols)} protocols failed")
        raise RuntimeError(f"DeFiLlama: all {len(top_protocols)} TVL requests failed")
    if errors:
        logger.warning(f"DeFiLlama TVL: {errors}/{len(top_protocols)} protocols failed")
    clear_alert(key='DeFiLlama TVL fetch errors')

    df = pd.DataFrame(all_tvl_data)
    df['date'] = to_datetime(df['date'], unit='s').dt.date
    logger.info(f'Historical TVL data shape: {df.shape}')
    return df


def download_yield_pools(limit: int) -> pd.DataFrame:
    """Download current yield pools data from DeFiLlama.

    Fetches data about yield-generating pools across DeFi protocols,
    sorted by total value locked (TVL).

    Args:
        limit: Maximum number of pools to return (top by TVL)

    Returns:
        pd.DataFrame: Pool data sorted by TVL

    Notes:
        - Returns current snapshot of yield opportunities
        - Useful for identifying high-yield pools with significant TVL
    """
    logger.info("Fetching yield pools from DeFiLlama...")
    response = requests.get(f"{DEFILLAMA_YIELDS_BASE_URL}{POOLS_ENDPOINT}", timeout=30)
    response.raise_for_status()
    pools = response.json()['data']
    df_pools = pd.DataFrame(pools)
    logger.info(f"Retrieved {len(df_pools)} yield pools, select top {limit}")
    df_pools = df_pools.sort_values('tvlUsd', ascending=False).head(limit)
    return df_pools


def download_historical_yield_data(current_pools_df: pd.DataFrame) -> pd.DataFrame:
    """Download historical yield data for top pools by TVL.

    Fetches time series yield data for pools, showing how yields and TVL
    have changed over time. Useful for analyzing yield sustainability and
    pool performance.

    Args:
        current_pools_df: DataFrame from download_yield_pools() containing pool info

    Returns:
        pd.DataFrame: Historical yield data with columns:
            - pool_id: Unique pool identifier
            - pool_name: Pool symbol/name
            - project: Protocol hosting the pool
            - chain: Blockchain network
            - date: Date of the snapshot
            - timestamp: Unix timestamp
            - tvlUsd: TVL in USD
            - apy: Annual percentage yield
            - apyBase: Base yield (from trading fees, etc.)
            - apyReward: Additional reward yield (from incentives)

    Notes:
        - Adds 1 second delay between pool requests
        - Processes each pool individually to get full history
    """
    logger.info("Downloading historical yield data...")
    all_historical_data = []
    errors = 0
    for i, (_, pool) in enumerate(current_pools_df.iterrows()):
        time.sleep(2)
        pool_id = pool['pool']
        logger.info(f"Fetching historical yield for pool {pool_id} ({i}/{len(current_pools_df)})")
        try:
            response = requests.get(f"{DEFILLAMA_YIELDS_BASE_URL}{POOL_CHART_ENDPOINT.format(pool_id)}", timeout=30)
            response.raise_for_status()
            pool_data = response.json()['data']
        except Exception as e:
            logger.error(f"Error fetching historical yield for pool {pool_id}: {e}")
            errors += 1
            continue
        if not pool_data:
            logger.warning(f"No historical data for pool {pool_id}")
            continue
        for entry in pool_data:
            entry.update({
                'pool_id': pool_id,
                'pool_name': pool.get('symbol', ''),
                'project': pool.get('project', ''),
                'chain': pool.get('chain', ''),
            })
            all_historical_data.append(entry)

    if not all_historical_data:
        raise_alert(key='DeFiLlama yield fetch errors', priority=HIGH, description=f"all {len(current_pools_df)} pools failed")
        raise RuntimeError(f"DeFiLlama: all {len(current_pools_df)} yield requests failed")
    if errors:
        logger.warning(f"DeFiLlama yield: {errors}/{len(current_pools_df)} pools failed")
    clear_alert(key='DeFiLlama yield fetch errors')

    df = pd.DataFrame(all_historical_data)
    df['date'] = to_datetime(df['timestamp']).dt.date
    logger.info(f'Historical yield data shape: {df.shape}')
    return df


def dump_defillama_data(df: pd.DataFrame, case_name: str, date_str: str, output_dir: str, debug: bool = False):
    """Save DeFiLlama data to parquet file or print for debugging.

    Args:
        df: DataFrame to save
        case_name: Type of data ('tvl' or 'yield')
        date_str: Timestamp string for filename
        output_dir: Directory path for output file
        debug: If True, print sample instead of saving

    Output format: {output_dir}/{case_name}_{date_str}.parquet
    """
    filename = f"{output_dir}/{case_name}_{date_str}.parquet"
    if debug:
        logger.info(f'{case_name} data for sample: {df.head().to_markdown()}')
    else:
        safe_mkdir(output_dir)
        df.to_parquet(filename)
        logger.info(f'Dumping {filename}')


def download_defillama_data(limit: int, output_dir: str, debug: bool = False):
    """Main function to download and process DeFiLlama data.
    
    Orchestrates the complete data download process:
    1. Fetches protocol list and historical TVL data
    2. Downloads current yield pools and their history
    3. Saves both datasets with timestamps
    
    Args:
        limit: Maximum number of protocols/pools to process
        output_dir: Directory for saving output files
        debug: If True, print data instead of saving
        
    Output files:
        - tvl_{timestamp}.parquet: Historical TVL data
        - yield_{timestamp}.parquet: Historical yield data
        
    Use cases:
        - Analyzing DeFi protocol growth trends
        - Identifying yield opportunities and risks
        - Understanding capital flows in DeFi
        - Backtesting yield strategies
    """
    protocols = get_defillama_protocols()
    tvl_df = download_historical_tvl(protocols, limit)

    current_pools = download_yield_pools(limit)
    yield_df = download_historical_yield_data(current_pools)

    now = dt.now(timezone.utc)
    date_str = date_to_str(now)
    dump_defillama_data(tvl_df, 'tvl', date_str, output_dir, debug)
    dump_defillama_data(yield_df, 'yield', date_str, output_dir, debug)
