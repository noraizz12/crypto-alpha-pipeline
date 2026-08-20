import hashlib
import hmac
import json
import logging
import time
from datetime import datetime as dt, timezone
from io import StringIO
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

from lib.util.aws import load_aws_secrets
from lib.util.dataframes import concat
from lib.util.opsgenie import HIGH, raise_alert
from lib.util.time_util import MILLIS_IN_DAY, beginning_of_day_millis, dt_to_millis, dt_to_str, millis_to_dt, millis_to_dt_str, to_datetime, today
from lib.util.util import log_and_raise

PAPI_URL = "papi.binance.com"
FAPI_URL = "fapi.binance.com"
SAPI_URL = "api.binance.com"


TIMEOUT_SECS = 5
INIT_JUMP_MILLS = 1000 * 60 * 120
MIN_JUMP_MILLS = 1000 * 60 * 5

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


try:
    BINANCE_SECRET_DICT = load_aws_secrets(statarb_secretid='statarb/binance-ro')
    BINANCE_KEY = BINANCE_SECRET_DICT['API']
    BINANCE_SECRET = BINANCE_SECRET_DICT['SECRET']
except Exception as e:
    raise log_and_raise(f"Could not look up binance keys! {e}")


def _hmac_signature(query_string: str) -> str:
    m = hmac.new(BINANCE_SECRET.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256)
    return m.hexdigest()


def call_binance(endpoint: str, url: str, query_string: str = "") -> Optional[str]:
    ts = int(1000 * dt.now(timezone.utc).timestamp())
    query_string = f"timestamp={ts}{query_string}"
    signature = _hmac_signature(query_string)
    url = f"https://{url}{endpoint}?{query_string}&signature={signature}"
    logger.info(f"Calling {url}")
    try:
        ret = requests.get(
            timeout=TIMEOUT_SECS,
            url=url,
            headers={
                'X-MBX-APIKEY': BINANCE_KEY
            })
    except Exception as e:
        logger.warning(f"Exception calling binance: {str(e)}")
        return None
    if ret.status_code != 200:
        logger.warning(f"Bad response from binance on endpoint {endpoint} {ret.status_code} {ret.text}")
        return None

    if len(ret.text) == 0 or ret.text == "[]":
        try:
            ret_json = ret.json()
        except Exception as e:
            logger.warning(f"Bad response from binance on endpoint {endpoint} {ret.status_code} {ret.text} {e}")
            ret_json = "NA"
            
        logger.warning(f"Binance response but no text from {endpoint}: {ret_json}")
        return None

    return ret.text


def get_positions() -> Optional[pd.DataFrame]:
    """Get current futures positions from Binance Portfolio Margin API.

    Returns:
        DataFrame with position data containing:
        - symbol: Trading pair (e.g., 'BTCUSDT')
        - positionAmt: Signed quantity in base asset; positive=long, negative=short (renamed to 'qty')
        - entryPrice: Volume-weighted average entry price for the position
        - markPrice: Current mark price used for unrealized PnL calculation
        - unRealizedProfit: Mark-to-market PnL; (markPrice - entryPrice) × positionAmt
        - liquidationPrice: Price at which position will be liquidated
        - leverage: Effective leverage (1-125x depending on symbol)
        - maxNotionalValue: Maximum position size allowed in USDT
        - marginType: 'cross' (shared margin) or 'isolated' (position-specific margin)
        - isolatedMargin: Margin allocated to this position if using isolated mode
        - isAutoAddMargin: Whether system auto-adds margin to prevent liquidation
        - positionSide: 'BOTH' (one-way mode), 'LONG', or 'SHORT' (hedge mode)
        - notional: Current position value in USDT; markPrice × positionAmt (renamed to 'value')
        - isolatedWallet: Wallet balance allocated to isolated position
        - updateTime: Unix timestamp (ms) of last position update
        - cost_basis: Total capital invested; qty × entryPrice (calculated field added here)

    API Documentation:
    https://developers.binance.com/docs/derivatives/portfolio-margin/account/Position-Information
    """
    logger.info("Getting binance positions...")
    endpoint = "/papi/v1/um/positionRisk"
    resp = call_binance(endpoint=endpoint, url=PAPI_URL)
    if resp is None:
        return None

    try:
        df = pd.read_json(StringIO(resp))
    except ValueError as ve:
        logger.warning(f"could not read position response.. {ve} {resp}")
        return None

    if len(df) == 0:
        logger.warning("No positions from binance!")
        return None

    df = df.rename(columns={'positionAmt': 'qty', 'notional': 'value'})
    df['cost_basis'] = df['qty'] * df['entryPrice']
    return df


def get_balances() -> Optional[pd.DataFrame]:
    """Get account balances from Binance Portfolio Margin API.

    Returns:
        DataFrame with balance data containing:
        - asset: Asset name (e.g., 'USDT', 'BTC', 'BNB')
        - totalWalletBalance: Total balance including all positions and unrealized PnL
        - crossMarginAsset: Balance available for cross-margin positions
        - crossMarginBorrowed: Amount borrowed for cross-margin trading
        - crossMarginFree: Cross-margin balance available to withdraw
        - crossMarginInterest: Accrued interest on cross-margin borrows
        - crossMarginLocked: Cross-margin balance locked in open orders
        - umWalletBalance: USD-M futures wallet balance
        - umUnrealizedPNL: Unrealized PnL from USD-M futures positions
        - cmWalletBalance: COIN-M futures wallet balance
        - cmUnrealizedPNL: Unrealized PnL from COIN-M futures positions
        - updateTime: Unix timestamp (ms) of last balance update
        - negativeBalance: Debt amount if account has negative balance

    API Documentation:
    https://developers.binance.com/docs/derivatives/portfolio-margin/account/Query-Portfolio-Margin-Account-Balance
    """
    logger.info("Getting binance balances...")
    endpoint = "/papi/v1/balance"
    resp = call_binance(endpoint=endpoint, url=PAPI_URL)
    if resp is None:
        return None
    try:
        df = pd.read_json(StringIO(resp))
    except Exception as e:
        logger.warning(f"could not read balance response: {e} {resp}")
        return None
    return df


def get_funding(start_dt: Optional[dt] = None) -> Optional[pd.DataFrame]:
    """Get funding fee income from Binance Portfolio Margin API.

    Returns:
        DataFrame with funding income data containing:
        - symbol: Trading pair that generated the funding payment
        - incomeType: Always 'FUNDING_FEE' for this query
        - income: Funding amount; positive=received, negative=paid
        - asset: Currency of payment (typically 'USDT')
        - info: Additional information string
        - time: Unix timestamp (ms) when funding was settled (converted to datetime)
        - tranId: Unique transaction ID for this funding payment
        - tradeId: Removed by this function (not applicable to funding)

    Note: Funding fees are settled every 8 hours (00:00, 08:00, 16:00 UTC) on perpetual contracts.
    The fee is based on position size and the funding rate at settlement time.

    API Documentation:
    https://developers.binance.com/docs/derivatives/portfolio-margin/account/Query-Transaction-History
    """
    endpoint = "/papi/v1/um/income"
    if start_dt is None:
        start_dt = today()
    start_time = beginning_of_day_millis(start_dt)
    end_time = start_time + MILLIS_IN_DAY - 1
    logger.info(f"Getting binance Income between {millis_to_dt_str(start_time)} to {millis_to_dt_str(end_time)}")
    qs = f"&incomeType=FUNDING_FEE&limit=1000&startTime={start_time}&endTime={end_time}"
    resp = call_binance(endpoint=endpoint, url=PAPI_URL, query_string=qs)
    if resp is None:
        return None
    try:
        df = pd.read_json(StringIO(resp))
    except Exception as e:
        logger.warning(f"could not read balance response.. {e} {resp}")
        return None

    del df['tradeId']
    df['time'] = to_datetime(df['time'] / 1000, unit='s')
    return df


def get_financing_rates() -> Optional[pd.DataFrame]:
    """Get current funding rates (premium index) for USDS-Margined Futures.

    Returns:
        DataFrame with funding rate data containing:
        - symbol: Trading pair (e.g., 'BTCUSDT')
        - markPrice: Current mark price used for liquidation and PnL
        - indexPrice: Underlying index price from spot exchanges
        - estimatedSettlePrice: Predicted price at next settlement
        - lastFundingRate: Most recent funding rate applied (8-hour rate)
        - nextFundingTime: Unix timestamp (ms) of next funding settlement
        - interestRate: Base interest rate component
        - time: Unix timestamp (ms) of this data snapshot

    Note: Funding rate is applied every 8 hours. Positive rate means longs pay shorts;
    negative rate means shorts pay longs. Rate is applied to position notional value.

    API Documentation:
    https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/Mark-Price-and-Funding-Rate
    """
    logger.info("Getting binance premium Index...")
    endpoint = "/fapi/v1/premiumIndex"
    resp = call_binance(endpoint=endpoint, url=FAPI_URL)
    if resp is None:
        return None

    try:
        df = pd.read_json(StringIO(resp))
    except Exception as e:
        logger.warning(f"could not read financing rate response.. {e} {resp}")
        return None
    return df


def get_open_interest(symbols: List[str]) -> Optional[pd.DataFrame]:
    """Get current open interest for USDS-Margined Futures symbols.

    Returns:
        DataFrame with open interest data containing:
        - symbol: Trading pair (e.g., 'BTCUSDT')
        - openInterest: Total number of outstanding contracts (sum of all long positions)
        - time: Unix timestamp (ms) of this data snapshot

    Note: Open interest represents the total number of active contracts held by market participants.
    It does not count both sides of a trade; one long and one short = one contract.
    Rising OI with rising price suggests new money entering (bullish); falling OI with rising
    price suggests short covering (potentially bearish reversal).

    API Documentation:
    https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/Open-Interest
    """
    logger.info("Getting binance Open Interest...")
    endpoint = "/fapi/v1/openInterest"
    res = []
    for symbol in symbols:
        qs = f"&symbol={symbol}"
        resp = call_binance(endpoint=endpoint, url=FAPI_URL, query_string=qs)
        if resp is None:
            continue
        res.append(resp)
    try:
        res_str = '\n'.join(res)
        df = pd.read_json(StringIO(res_str), lines=True)
    except Exception as e:
        logger.warning(f"could not read open interest response.. {e} {res_str}")
        return None
    return df


def get_exchange_info() -> Optional[dict]:
    """Get exchange trading rules and symbol information for USDS-Margined Futures.

    Returns:
        Dictionary with exchange information containing:
        - timezone: Exchange timezone (always 'UTC')
        - serverTime: Current server time in Unix timestamp (ms)
        - rateLimits: Array of rate limit rules (request limits per interval)
        - exchangeFilters: Exchange-wide filters and restrictions
        - symbols: Array of trading pair information, each containing:
            - symbol: Trading pair name (e.g., 'BTCUSDT')
            - pair: Underlying pair
            - contractType: 'PERPETUAL' or expiration date for quarterly futures
            - deliveryDate: Settlement date for quarterly futures
            - onboardDate: Listing date in Unix timestamp (ms)
            - status: 'TRADING', 'PRE_TRADING', 'SETTLING', etc.
            - baseAsset: Base currency (e.g., 'BTC')
            - quoteAsset: Quote currency (e.g., 'USDT')
            - marginAsset: Asset used for margin (typically same as quoteAsset)
            - pricePrecision: Decimal places for price
            - quantityPrecision: Decimal places for quantity
            - baseAssetPrecision: Decimal places for base asset
            - quotePrecision: Decimal places for quote asset
            - underlyingType: 'COIN' or 'INDEX'
            - filters: Trading rules including PRICE_FILTER, LOT_SIZE, MARKET_LOT_SIZE,
                      MAX_NUM_ORDERS, PERCENT_PRICE, MIN_NOTIONAL
            - orderTypes: Supported order types (LIMIT, MARKET, STOP, etc.)
            - timeInForce: Supported time-in-force values (GTC, IOC, FOK, GTX)

    API Documentation:
    https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/Exchange-Information
    """
    logger.info("Getting binance exchange info...")
    endpoint = "/fapi/v1/exchangeInfo"
    resp = call_binance(endpoint=endpoint, url=FAPI_URL)
    if resp is None:
        return None
    resp = json.loads(resp)
    return resp


def get_account_info() -> Optional[dict]:
    """Get spot account information including balances.

    Returns:
        Dictionary with spot account information containing:
        - makerCommission: Maker fee rate in basis points (e.g., 10 = 0.1%)
        - takerCommission: Taker fee rate in basis points
        - buyerCommission: Buyer commission rate in basis points
        - sellerCommission: Seller commission rate in basis points
        - canTrade: Whether account is allowed to trade
        - canWithdraw: Whether account is allowed to withdraw
        - canDeposit: Whether account is allowed to deposit
        - updateTime: Unix timestamp (ms) of last account update
        - accountType: Account classification (e.g., 'SPOT', 'MARGIN')
        - balances: Array of asset balances, each containing:
            - asset: Currency name (e.g., 'BTC', 'USDT')
            - free: Available balance for trading/withdrawal
            - locked: Balance locked in open orders
        - permissions: Array of enabled permissions (e.g., ['SPOT', 'MARGIN', 'FUTURES'])

    API Documentation:
    https://developers.binance.com/docs/binance-spot-api-docs/rest-api/public-api-endpoints#account-information-user_data
    """
    logger.info("Getting account info...")
    endpoint = "/api/v3/account"
    resp = call_binance(endpoint=endpoint, url="api.binance.com")
    if resp is None:
        return None
    resp = json.loads(resp)
    return resp


def get_delisting_schedule() -> Optional[dict]:
    """Get schedule of upcoming spot token delistings.

    Returns:
        Array of delisting information, each entry containing:
        - delistTime: Unix timestamp (ms) when trading will be disabled
        - symbols: Array of trading pairs being delisted (e.g., ['BTCUSDT', 'ETHUSDT'])
        - coin: Base asset being delisted (e.g., 'BTC')

    Note: Binance periodically delists low-liquidity tokens. After delisting, trading stops
    but users can still withdraw assets. Check this endpoint regularly to avoid holding
    delisted tokens.

    API Documentation:
    https://developers.binance.com/docs/wallet/asset/delist-schedule
    """
    logger.info("Getting delisting schedule...")
    endpoint = "/sapi/v1/spot/delist-schedule"
    resp = call_binance(endpoint=endpoint, url="api.binance.com")
    if resp is None:
        return None
    resp = json.loads(resp)
    return resp


def get_hist_trades(start_dt: Optional[dt] = None, end_dt: Optional[dt] = None) -> Optional[pd.DataFrame]:
    """Get historical trade fills from Binance Portfolio Margin API.

    Returns:
        DataFrame with trade fill data containing:
        - symbol: Trading pair for the trade (e.g., 'BTCUSDT')
        - id: Unique trade ID from Binance
        - orderId: Order ID that generated this fill
        - side: 'BUY' or 'SELL'
        - price: Execution price per unit of base asset
        - qty: Quantity of base asset filled (always positive, check 'side' for direction)
        - quoteQty: Total trade value in USDT; calculated as price × qty
        - realizedPnl: Realized profit/loss from closing/reducing position; 0 for opening trades
        - marginAsset: Asset used for margin (typically 'USDT')
        - commission: Trading fee charged for this fill
        - commissionAsset: Asset in which fee was charged (typically 'USDT')
        - time: Unix timestamp (ms) when trade was executed
        - positionSide: 'BOTH' (one-way mode), 'LONG', or 'SHORT' (hedge mode)
        - buyer: Whether this account was the buyer (true/false)
        - maker: Whether this fill was a maker order (true) or taker order (false)
        - ts: Datetime timestamp added by this function (converted from 'time')

    Note on quoteQty: This is the total notional value traded in USDT (quote asset).
    It represents the actual dollar value of the trade: price × qty.
    Example: price=7819.01, qty=0.002 → quoteQty=15.63802 USDT

    Note on realizedPnl: Only non-zero when reducing/closing a position. Opening trades
    or increasing positions show realizedPnl=0. This is the immediate PnL impact of
    the trade, not including unrealized PnL from remaining position.

    API Documentation:
    https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api/Account-Trade-List
    """
    endpoint = "/papi/v1/um/userTrades"
    jump_millis = INIT_JUMP_MILLS
    if start_dt is None:
        start_dt = today()
    if end_dt is None:
        end_dt = dt.now(timezone.utc)
    else:
        end_dt = min(dt.now(timezone.utc), end_dt)

    logger.info(f"Getting binance trades between {dt_to_str(start_dt)} to {dt_to_str(end_dt)}")

    start_millis = dt_to_millis(start_dt)
    end_millis = dt_to_millis(end_dt)

    dfs = []
    while start_millis < end_millis:
        raw_next_millis = start_millis + jump_millis
        next_millis = min(raw_next_millis, end_millis)
        logger.info(f"interval: {millis_to_dt(start_millis)} - {millis_to_dt(next_millis)}")
        qs = f"&limit=1000&startTime={start_millis}&endTime={next_millis}&recvWindow=30000"
        resp = call_binance(endpoint=endpoint, url=PAPI_URL, query_string=qs)
        time.sleep(1)

        if resp is None:
            start_millis = raw_next_millis
            logger.warning(f"No response from binance! move to {raw_next_millis}")
            continue

        df = pd.read_json(StringIO(resp))
        if len(df) >= 1000:
            if jump_millis > MIN_JUMP_MILLS:
                jump_millis = int(jump_millis / 2)
                logger.warning(f"Reduce jump_millis to {jump_millis} since hit trade query limit")
                continue
            else:
                raise_alert(key='trade records query hit limit', priority=HIGH, description=f'see history trades query length {len(df)} hit limit, need to check if lost any records')

        # if we don't hit limit, get jump_millis back to initial value and we could move start_millis
        start_millis = raw_next_millis
        jump_millis = INIT_JUMP_MILLS
        if len(df) > 0:
            for col in ['qty', 'quoteQty']:
                df[col] = df[col].astype(np.float32)
            dfs.append(df)
            logger.info(f'append trade records {df.shape[0]}')
        else:
            logger.warning(f"No trades returned from binance... {resp}")

    df = concat(dfs, fast=True)
    if df is None:
        return None

    df['ts'] = to_datetime(df['time'], unit='ms')
    return df
