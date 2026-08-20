"""Trading system components for order management and execution tracking.

This module provides core trading abstractions including orders, fills, and cancellations
for cryptocurrency futures trading. It handles order lifecycle management, fill tracking,
commission calculations, and integration with Binance futures exchange.

Key Components:
    - Side: Buy/Sell enumeration with conversion utilities
    - OrderType: Various order types (LIMIT, MARKET, POST_ONLY, etc.)
    - TimeInForce: Order duration specifications (GTC, GTX, IOC)
    - Fill: Execution record with commission tracking
    - Order: Complete order management with aggression, deficit tracking
    - Cancel: Order cancellation tracking

The module supports:
    - Binance futures exchange integration
    - Commission calculation in multiple assets
    - Slippage tracking against optimal prices
    - Order aggression levels for execution priority
    - Position expansion/reduction logic
    - Multi-horizon alpha tracking for orders

Example:
    >>> # Create a limit order
    >>> order = Order(
    ...     symbol="BTCUSDT",
    ...     px=42000.0,
    ...     signed_qty=0.1,  # positive for buy
    ...     pending=True,
    ...     aggression=2,
    ...     qty_at_time_of_order=0.0
    ... )
    >>> 
    >>> # Parse a fill from Binance
    >>> fill = Fill.from_binance_json(binance_msg, recv_ts=dt.now(timezone.utc))
    >>> fill.calc_commission_usd({"BNB": 300.0})
"""

import hashlib
import logging
import math
import time
import json
from datetime import datetime as dt, timezone
from enum import StrEnum
from typing import Any, Dict, List, Optional, Self

from lib.util.util import fpct, log_and_raise
from lib.util.logging_util import KeyLogger
from lib.util.time_util import to_datetime

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)

DEFAULT_COMMISSION_ASSET = "BNB"
USDT = "USDT"


class Side(StrEnum):
    """Order side enumeration for buy/sell directions.
    
    Provides consistent representation of trade direction with conversion
    utilities from various formats (string, float, sign).
    
    Attributes:
        BUY: Buy side order (positive quantity)
        SELL: Sell side order (negative quantity)
    """
    BUY = "B"
    SELL = "S"

    @classmethod
    def from_string(cls, side_str: str) -> Self:
        """Convert string representation to Side enum.
        
        Args:
            side_str: String representation ("BUY", "B", "SELL", "S")
            
        Returns:
            Side enum value
            
        Raises:
            RuntimeError: If side_str is not recognized
        """
        if side_str.upper() in ("BUY", "B"):
            return cls.BUY
        if side_str.upper() in ("SELL", "S"):
            return cls.SELL

        raise RuntimeError(f"Unknown side {side_str}")

    @classmethod
    def from_float(cls, amt: float | int) -> Self:
        """Determine side from signed quantity.
        
        Args:
            amt: Signed quantity (positive=buy, negative=sell)
            
        Returns:
            Side enum value
            
        Raises:
            RuntimeError: If amt is zero
        """
        if amt > 0:
            return cls.BUY
        if amt < 0:
            return cls.SELL

        raise RuntimeError(f"No side to {amt}")

    @classmethod
    def sign(cls, side: Self) -> int:
        """Get mathematical sign for a side.
        
        Args:
            side: Side enum value
            
        Returns:
            1 for BUY, -1 for SELL
            
        Raises:
            RuntimeError: If side is not BUY or SELL
        """
        if side == side.BUY:
            return 1
        if side == side.SELL:
            return -1
        raise RuntimeError(f"Unknown Side {side}")


class OrderType(StrEnum):
    """Order type enumeration for different execution methods.
    
    Defines how orders interact with the order book and position management.
    
    Attributes:
        LIMIT: Standard limit order
        REDUCE: Limit order that only reduces position
        POST_ONLY: Maker-only order that adds liquidity
        POST_ONLY_REDUCE: Maker-only order that reduces position
        MARKET: Immediate execution at best available price
    """
    LIMIT = 'LIMIT'
    REDUCE = 'REDUCE'
    POST_ONLY = 'POST_ONLY'
    POST_ONLY_REDUCE = 'POST_ONLY_REDUCE'
    MARKET = 'MARKET'

    @classmethod
    def from_string(cls, ot_str: str) -> Self:
        """Convert string to OrderType enum.
        
        Args:
            ot_str: String representation of order type
            
        Returns:
            OrderType enum value
            
        Raises:
            RuntimeError: If order type string is not recognized
        """
        if ot_str.upper() == "LIMIT":
            return cls.LIMIT
        if ot_str.upper() == "POST_ONLY":
            return cls.POST_ONLY
        if ot_str.upper() == "REDUCE":
            return cls.REDUCE
        if ot_str.upper() == "POST_ONLY_REDUCE":
            return cls.POST_ONLY_REDUCE
        if ot_str.upper() == "MARKET":
            return cls.MARKET

        raise RuntimeError(f"Unknown OrderType {ot_str}")


class TimeInForce(StrEnum):
    """Time-in-force specifications for order duration.
    
    Controls how long an order remains active in the order book.
    
    Attributes:
        GOOD_TIL_CROSS: Post-only order, cancelled if would cross
        IMMEDIATE_OR_CANCEL: Fill immediately or cancel remainder
        GOOD_TIL_CANCEL: Remain until filled or manually cancelled
        FILL_OR_KILL: Fill immediately or cancel all (all or nothing)
    """
    GOOD_TIL_CROSS = "GTX"
    IMMEDIATE_OR_CANCEL = "IOC"
    GOOD_TIL_CANCEL = "GTC"
    FILL_OR_KILL = "FOK"

    @classmethod
    def from_string(cls, tif_str: str) -> Self:
        """Convert string to TimeInForce enum.
        
        Args:
            tif_str: String representation ("GTX", "GTC", "IOC")
            
        Returns:
            TimeInForce enum value
            
        Raises:
            RuntimeError: If time-in-force string not recognized
        """
        if tif_str.upper() == "GTX":
            return cls.GOOD_TIL_CROSS
        if tif_str.upper() == "GTC":
            return cls.GOOD_TIL_CANCEL
        if tif_str.upper() == "IOC":
            return cls.IMMEDIATE_OR_CANCEL
        if tif_str.upper() == "FOK":
            return cls.FILL_OR_KILL

        raise RuntimeError(f"Unknown TimeInForce {tif_str}")


class PositionSide(StrEnum):
    """Position side enumeration one-way or hedge mode.

    Controls position mode: default BOTH for One-way Mode; LONG or SHORT for Hedge Mode.

    Attributes:
        BOTH:
        LONG:
        SHORT
    """
    BOTH = "BOTH"
    LONG = "LONG"
    SHORT = "SHORT"

    @classmethod
    def from_string(cls, position_mode_str: str) -> Self:
        """Convert string to PositionSide enum.

        Args:
            position_mode_str: String representation ("BOTH", "LONG", "SHORT")

        Returns:
            PositionSide enum value

        Raises:
            RuntimeError: If positon-mode-str string not recognized
        """
        if position_mode_str.upper() == "BOTH":
            return cls.BOTH
        if position_mode_str.upper() == "LONG":
            return cls.LONG
        if position_mode_str.upper() == "SHORT":
            return cls.SHORT

        raise RuntimeError(f"Unknown PositionMode {position_mode_str}")


class FillType(StrEnum):
    """Fill type classification.
    
    Attributes:
        NORMAL: Real exchange fill
        FAKE: Simulated/paper trading fill
    """
    NORMAL = "N"
    FAKE = "F"


class OMSPayloadType(StrEnum):
    """Different payload types for OMS.
    """
    UM_CREATE_ORDER = "UM_CREATE_ORDER"
    UM_CANCEL_ORDER = "UM_CANCEL_ORDER"
    UM_CANCEL_ALL_OPEN_ORDERS = "UM_CANCEL_ALL_OPEN_ORDERS"
    UM_GET_ACCOUNT_TRADE_LIST = "UM_GET_ACCOUNT_TRADE_LIST"
    UM_QUERY_ORDER = "UM_QUERY_ORDER"
    UM_QUERY_ALL_ORDERS = "UM_QUERY_ALL_ORDERS"
    UM_QUERY_ALL_CURRENT_OPEN_ORDERS = "UM_QUERY_ALL_CURRENT_OPEN_ORDERS"
    UM_GET_INCOME_HISTORY = "UM_GET_INCOME_HISTORY"
    UM_GET_POSITION_RISK = "UM_GET_POSITION_RISK"


class Venue(StrEnum):
    """Trading venue enumeration.
    
    Attributes:
        BINANCE: Binance futures exchange
    """
    BINANCE = "BINANCE"


class Fill:
    """Trade execution record with commission and slippage tracking.
    
    Represents a completed trade execution including price, quantity,
    commission details, and optional slippage calculation against
    optimal price.
    
    Attributes:
        symbol: Trading symbol (e.g., "BTCUSDT")
        side: Buy or sell side
        px: Execution price
        qty: Executed quantity (absolute value)
        fill_type: Normal or fake/simulated fill
        exch_ts: Exchange timestamp of execution
        recv_ts: Local timestamp when fill was received
        commission: Commission amount in commission_asset
        commission_asset: Asset used for commission payment
        commission_usd: Commission converted to USD value
        oid: Order ID associated with this fill
        venue: Exchange venue (e.g., Binance)
        opt_px: Optimal price for slippage calculation
    """
    
    def __init__(
            self,
            symbol: str,
            side: Side,
            px: float,
            qty: float,
            exch_ts: Optional[dt] = None,
            recv_ts: Optional[dt] = None,
            fill_type: FillType = FillType.NORMAL,
            commission: float = 0.0,
            commission_asset: str = DEFAULT_COMMISSION_ASSET,
            venue: Optional[Venue] = Venue.BINANCE,
            oid: Optional[str] = None,
            opt_px: Optional[float] = None
    ):
        self.symbol: str = symbol
        self.side: Side = side
        self.px: float = px
        self.qty: float = qty
        self.fill_type: FillType = fill_type
        if fill_type == FillType.FAKE:
            exch_ts = recv_ts = dt.now(timezone.utc)
        else:
            assert exch_ts is not None
            assert recv_ts is not None
        self.exch_ts: dt = exch_ts
        self.recv_ts: dt = recv_ts
        self.commission: float = commission
        self.commission_asset: str = commission_asset
        self.oid: str = oid
        self.koid: Optional[str] = None
        self.venue: Venue = venue
        self.commission_usd: float = 0
        self.opt_px: float = opt_px

    @classmethod
    def from_binance_json(cls, msg: dict, recv_ts: Optional[dt] = None) -> Optional[Self]:
        """Parse Binance WebSocket fill message into Fill object.
        
        Args:
            msg: Binance WebSocket message dictionary
            recv_ts: Local timestamp when message was received
            
        Returns:
            Fill object or None if parsing fails
            
        Note:
            Expects Binance futures user data stream ORDER_TRADE_UPDATE format
        """
        # logger.info("RAW: " + str(msg))
        try:
            ts = dt.fromtimestamp(msg['T'] / 1000.0, tz=timezone.utc)
            om = msg['o']
            oid = om['i']
            koid = om['c'] if 'c' in om else None

            side = Side.from_string(om['S'])
            qty = float(om['l'])

            # orig_px = float(om['p'])
            avg_px = float(om['ap'])
            # last_f_px = float(om['L'])

            commission = float(om['n'])
            commission_asset = om['N']
        except Exception as e:
            logger.error(f"Could not parse new order message from binance {msg=} with error {e}", key="fail to parse fill message from binance")
            return None

        fill = cls(
            symbol=om['s'],
            side=side,
            px=avg_px,
            qty=qty,
            fill_type=FillType.NORMAL,
            exch_ts=ts,
            recv_ts=recv_ts,
            commission=commission,
            commission_asset=commission_asset,
            venue=Venue.BINANCE,
            oid=oid
        )
        fill.koid = koid
        return fill

    def notional(self, include_fees: bool = True) -> float:
        """Calculate notional value of the fill.
        
        Returns signed notional: negative for buys (cash outflow),
        positive for sells (cash inflow).
        
        Args:
            include_fees: Whether to subtract commission from notional
            
        Returns:
            Signed notional value in USD
        """
        notional = self.px * self.qty * (Side.sign(self.side) * -1)
        if include_fees:
            notional -= self.commission_usd

        if (self.side == Side.SELL and notional <= 0) or (self.side == Side.BUY and notional >= 0):
            logger.warning(f"Suspicious fill on {self.symbol} side: {self.side} px: {self.px} qty: {self.qty} comm: {self.commission_usd}")

        return notional

    def signed_qty(self) -> float:
        """Get quantity with sign based on side.
        
        Returns:
            Positive quantity for buys, negative for sells
        """
        return self.qty * Side.sign(self.side)

    def fill_slip(self) -> Optional[float]:
        """Calculate slippage in USD against optimal price.
        
        Slippage = signed_qty * (actual_price - optimal_price)
        
        Returns:
            Slippage in USD, or None if no optimal price set
        """
        if self.opt_px is None:
            return None
        return self.signed_qty() * (self.px - self.opt_px)

    def calc_commission_usd(self, commission_px_dict: Dict[str, Optional[float]]):
        """Convert commission to USD using provided asset prices.
        
        Args:
            commission_px_dict: Map of asset symbols to USD prices
            
        Raises:
            Exception: If commission asset price not provided
        """
        # if commission_asset not in commission_px_dict or commission_px_dict[commission_asset] is None, log and raise the error
        if commission_px_dict.get(self.commission_asset, None) is None:
            raise log_and_raise(f"failed to get {self.commission_asset} commission px from {commission_px_dict.keys()} dict")
        self.commission_usd = self.commission * commission_px_dict[self.commission_asset]

    def slack_str(self) -> str:
        """Format fill information for Slack notification.
        
        Returns:
            Human-readable fill summary with slippage
        """
        slip = self.fill_slip()
        if slip is not None:
            bps = 1000.0 * slip / abs(self.notional())
            slip_str = f"${slip:.2f} / {bps:.1f} bps"
        else:
            slip_str = "NA"
        return f"FILLED {self.side} {self.qty} {self.symbol} @ ${self.px} amt: ${-self.notional():.2f} slip: {slip_str}"

    def __str__(self) -> str:
        """String representation for logging/persistence.
        
        Returns:
            Pipe-delimited fill record
        """
        return f"FILL|{self.exch_ts}|{self.recv_ts}|{self.symbol}|{self.venue}|{self.fill_type}|{self.side}|{self.px}|{self.qty}|{self.commission}|{self.commission_asset}|{self.oid}|{self.opt_px}|{self.notional()}"


def get_agg_side_fills_info(symbol: str, fills: List[Fill], side: Side) -> List[Any]:
    """Aggregate fill statistics for a given symbol and side.
    
    Args:
        symbol: Trading symbol to filter
        fills: List of Fill objects to aggregate
        side: Side to filter (BUY or SELL)
        
    Returns:
        List containing:
        - symbol: Trading symbol
        - side: Buy or sell side
        - qty: Total quantity filled
        - avg_cost: Average fill price
        - cnt: Number of fills
        - notional: Total notional value
        - slip: Total slippage in USD
        - slip_bps: Slippage in basis points
        
        Empty list if no fills match criteria
    """
    cnt = qty = notional = slip = 0
    for fill in fills:
        if fill.side == side:
            cnt += 1
            qty += fill.qty
            notional += abs(fill.notional())
            fill_slip = fill.fill_slip()
            slip += fill_slip if fill_slip is not None else 0
    if qty == 0:
        return []
    slip_bps = slip / notional * 10000 if notional else 0
    avg_cost = notional / qty
    return [symbol, side, qty, avg_cost, cnt, notional, slip, slip_bps]


class Order:
    """Complete order representation with lifecycle management.
    
    Manages order state from creation through acknowledgment and execution.
    Supports aggression-based order type selection, position tracking,
    and multi-horizon alpha signal recording.
    
    The aggression parameter controls order placement relative to the
    order book:
    - Positive aggression: More aggressive, crosses spread sooner
    - Negative aggression: Passive, posts liquidity
    - Aggression determines OrderType and TimeInForce automatically
    
    Class Attributes:
        MAX_MANUAL_NOTIONAL_LIMIT: Maximum notional for manual orders ($10k)
        MAX_AGGRESSION: Maximum aggression level (6)
        MIN_AGGRESSION: Minimum aggression level (-9)
        MODEL_HORIZONS: Supported model time horizons in minutes
        MODELS: Supported alpha model types
        
    Attributes:
        symbol: Trading symbol
        px: Order price (may be rounded to tick size)
        side: Derived from signed_qty sign
        orig_qty: Original order quantity (absolute)
        remaining_qty: Unfilled quantity remaining
        pending: True if not yet acknowledged by exchange
        koid: Internal order ID (live order id)
        oid: Exchange order ID
        aggression: Order aggression level
        deficit: Position deficit this order addresses
        qty_at_time_of_order: Position when order was created
        px_at_time_of_order: Market price when created
        opt_px: Optimal price for slippage calculation
        order_type: LIMIT, REDUCE, POST_ONLY, etc.
        time_in_force: GTC, GTX, IOC
        reduce_only: If true, only reduces position
        exch_ts: Exchange acknowledgment time
        created_ts: Local order creation time
        acked_ts: Local acknowledgment receipt time
        bid/ask: Market prices at order time
        order_alpha_dict: Alpha signals at order time
    """
    
    MAX_MANUAL_NOTIONAL_LIMIT = 10000
    MAX_AGGRESSION = 6
    MIN_AGGRESSION = -9
    MODEL_HORIZONS = [15, 60, 120, 360, 720, 1440, 4320, 10080, 43200]
    MODELS = ["hl", "c2vwap", "slz", "vadj", "ba", "badj", "oi", "rsi"]

    def __init__(
            self,
            symbol: str,
            px: float,
            signed_qty: float,
            pending: bool,
            koid: Optional[str] = None,
            aggression: Optional[int] = None,
            deficit: Optional[float] = None,
            qty_at_time_of_order: Optional[float] = None,
            px_at_time_of_order: Optional[float] = None,
            opt_px: Optional[float] = None,
            order_type: Optional[OrderType] = None,
            time_in_force: Optional[TimeInForce] = None,
            reduce_only: bool = False,
            exch_ts: Optional[dt] = None,
            created_ts: Optional[dt] = None,
            acked_ts: Optional[dt] = None,
            oid: Optional[str] = None,
            tick_size: Optional[float] = None,
            order_alpha_dict: Dict[str, float] = None
    ):
        assert signed_qty != 0
        assert px > 0

        self.symbol: str = symbol
        self.pending: bool = pending

        if self.pending:
            created_ts = dt.now(timezone.utc)
            acked_ts = None
            exch_ts = None
            oid = None
        else:
            assert exch_ts is not None
            assert oid is not None
            acked_ts = dt.now(timezone.utc) if acked_ts is None else acked_ts

        self.created_ts: Optional[dt] = created_ts
        self.acked_ts: Optional[dt] = acked_ts
        self.exch_ts: Optional[dt] = exch_ts
        self.oid: Optional[str] = oid

        self.px: float = self.round_to_tick_size(px, signed_qty, tick_size)
        self.side = Side.BUY if signed_qty > 0 else Side.SELL
        self.qty_at_time_of_order: float = qty_at_time_of_order
        self.px_at_time_of_order: float = px_at_time_of_order
        self.opt_px: float = opt_px
        self.orig_qty: float = abs(signed_qty)
        self.remaining_qty: float = abs(signed_qty)
        self.koid: str = koid if koid is not None else self.make_orderid()
        self.fill_cnt = 0

        self.aggression: Optional[int] = None

        if aggression is not None:
            logger.warning(f"Setting order type on {symbol} {self.side} {signed_qty}/{qty_at_time_of_order} {aggression=}")

            assert self.qty_at_time_of_order is not None

            not_overshooting = abs(signed_qty) <= abs(qty_at_time_of_order)
            self.aggression = self.bound_aggression(aggression)
            if self.aggression >= 0:
                time_in_force = TimeInForce.GOOD_TIL_CANCEL
                if not self.expanding() and not_overshooting:
                    order_type = OrderType.REDUCE
                else:
                    order_type = OrderType.LIMIT
            else:
                time_in_force = TimeInForce.GOOD_TIL_CROSS
                if not self.expanding() and not_overshooting:
                    order_type = OrderType.POST_ONLY_REDUCE
                else:
                    order_type = OrderType.POST_ONLY
        else:
            logger.info(f"Aggression is None {symbol} (order instantiation from binance json hopefully)")
            order_type = OrderType.LIMIT if order_type is None else order_type
            time_in_force = TimeInForce.GOOD_TIL_CROSS if time_in_force is None else time_in_force

        self.order_type: OrderType = order_type
        self.time_in_force: TimeInForce = time_in_force
        self.reduce_only: bool = reduce_only
        self.deficit: float = deficit

        self.bid: Optional[float] = None
        self.ask: Optional[float] = None
        self.order_alpha_dict = order_alpha_dict if order_alpha_dict is not None else {}

    @classmethod
    def from_str(cls, ostr: str) -> Self:
        """Parse order from string representation.
        
        Formats supported:
        - "B ETHUSDT 0.01 @ 2500" (defaults to LIMIT GTX)
        - "B ETHUSDT 0.01 @ 2500 LIMIT GTX"
        
        Args:
            ostr: Order string
            
        Returns:
            Order object
            
        Raises:
            RuntimeError: If parsing fails or notional exceeds limit
        """
        # "B ETHUSDT 0.01 @ 2500 [LIMIT GTX]"
        toks = ostr.split(' ')
        if len(toks) == 5:
            side_str, symbol, qty, _, px = toks
            order_type = OrderType.LIMIT
            time_in_force = TimeInForce.GOOD_TIL_CROSS
        elif len(toks) == 7:
            side_str, symbol, qty, _, px, order_type, time_in_force = toks
            order_type = OrderType.from_string(order_type)
            time_in_force = TimeInForce.from_string(time_in_force)
        else:
            logger.error(f"Could not parse order string! {toks=}", key="fail to parse order string")
            raise RuntimeError()

        side = Side.from_string(side_str)
        try:
            px = float(px)
            qty = float(qty)
        except Exception as e:
            logger.error(f"Could not parse order string! {px=} {qty=}", key="fail to parse order string for px and qty")
            raise

        signed_qty = qty * Side.sign(side)

        order = cls(
            symbol=symbol,
            pending=True,
            signed_qty=signed_qty,
            px=px,
            order_type=order_type,
            time_in_force=time_in_force
        )

        abs_order_amt = abs(order.order_amt())
        if abs_order_amt > Order.MAX_MANUAL_NOTIONAL_LIMIT:
            logger.error(f"Order Amount ${abs_order_amt:.0f} greater than {Order.MAX_MANUAL_NOTIONAL_LIMIT}", key="see order amount greater than notional limit")
            raise RuntimeError()

        return order

    @classmethod
    def from_binance_json(cls, msg: dict, recv_ts: Optional[dt] = None) -> Optional[Self]:
        # logger.info("RAW: " + str(msg))
        try:
            ts = dt.fromtimestamp(msg['T'] / 1000.0, tz=timezone.utc)
            om = msg['o']
            symbol = om['s']
            oid = om['i']
            koid = om['c'] if 'c' in om else None
            side = Side.from_string(om['S'])
            qty = float(om['q'])
            px = float(om['p'])
            signed_qty = Side.sign(side) * qty
            order_type = OrderType.from_string(om['o'])
            time_in_force = TimeInForce.from_string(om['f'])
            reduce_only = om['R']
        except Exception as e:
            logger.error(f"Could not parse new order message from binance {msg=} with error {e}", key="fail to parse order message from binance")
            return None

        order = cls(
            symbol=symbol,
            pending=False,
            signed_qty=signed_qty,
            px=px,
            order_type=order_type,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            exch_ts=ts,
            acked_ts=recv_ts,
            oid=oid
        )
        order.koid = koid
        assert order.side == side

        return order

    @classmethod
    def from_file_line(cls, row: str) -> Optional[Self]:
        # TODO: not all of these are loaded
        try:
            _, koid, oid, symbol, side, order_type, time_in_force, orig_qty, px, qty_at_time_of_order, aggression, deficit, created_ts, exch_ts, acked_ts, bid, ask, opt_px = row.split("|")
        except Exception as e:
            logger.error(f"Could not read order line row:{row} with error {e}", key="fail to read order line")
            return None

        try:
            order_type = OrderType.from_string(order_type)
            qty_at_time_of_order = float(qty_at_time_of_order)
            px_at_time_of_order = (float(bid) + float(ask)) * .5
            px = float(px)
            opt_px = float(opt_px)
            aggression = int(aggression)
            deficit = float(deficit)
            time_in_force = TimeInForce.from_string(time_in_force)
            signed_qty = Side.sign(Side.from_string(side)) * float(orig_qty)
            created_ts = to_datetime(created_ts) if created_ts != "None" else dt.now(timezone.utc)
            exch_ts = to_datetime(exch_ts) if exch_ts != "None" else exch_ts
            acked_ts = to_datetime(acked_ts) if acked_ts != "None" else exch_ts
        except ValueError as ve:
            logger.error(f"Could not convert type for order line with error {ve}", key="fail to convert type for order line")
            return None

        try:
            order = cls(
                symbol=symbol,
                oid=oid,
                koid=koid,
                signed_qty=signed_qty,
                qty_at_time_of_order=qty_at_time_of_order,
                px_at_time_of_order=px_at_time_of_order,
                order_type=order_type,
                time_in_force=time_in_force,
                px=px,
                opt_px=opt_px,
                pending=False,
                aggression=aggression,
                deficit=deficit,
                exch_ts=exch_ts,
                acked_ts=acked_ts,
                created_ts=created_ts
            )
        except Exception as e:
            logger.error(f"Could not create order from order line {row} with error {e}", key="fail to create order from order line")
            return None

        order.set_bid_ask(bid=float(bid), ask=float(ask))
        return order

    @classmethod
    def bound_aggression(cls, aggression: int) -> int:
        aggression = min(cls.MAX_AGGRESSION, aggression)
        aggression = max(cls.MIN_AGGRESSION, aggression)
        return aggression

    @classmethod
    def round_to_tick_size(cls, px: float, signed_qty: float, tick_size: Optional[float] = None) -> float:
        if tick_size is not None:
            # ceil on the buys and floor on the asks
            if signed_qty > 0:
                return math.ceil(px / tick_size) * tick_size
            else:
                return math.floor(px / tick_size) * tick_size
        else:
            return px

    def acked(self) -> bool:
        return self.exch_ts is not None

    def liquidation(self) -> bool:
        return Side.sign(self.side) * self.qty_at_time_of_order < 0

    def set_bid_ask(self, bid: float, ask: float):
        assert bid < ask
        self.bid = bid
        self.ask = ask

    def make_paper_order(self):
        self.pending = False
        assert self.created_ts is not None
        self.exch_ts = self.created_ts
        self.acked_ts = self.created_ts
        self.oid = self.koid

    def update_from_pending_order(self, pending_order: Self):
        MAX_PCT_DIFF = 0.01
        assert self.symbol == pending_order.symbol
        px_delta = abs(pending_order.px - self.px) / pending_order.px
        qty_delta = abs(pending_order.orig_qty - self.orig_qty) / pending_order.orig_qty

        if px_delta > MAX_PCT_DIFF or qty_delta > MAX_PCT_DIFF:
            logger.warning("Do these order match??")

        logger.info(f"Order {self.oid} {self.symbol} : Pending -> Accepted px: {pending_order.px} -> {self.px}, qty: {pending_order.orig_qty} -> {self.orig_qty}")

        self.created_ts = pending_order.created_ts
        self.bid = pending_order.bid
        self.ask = pending_order.ask
        self.deficit = pending_order.deficit
        self.qty_at_time_of_order = pending_order.qty_at_time_of_order
        self.px_at_time_of_order = pending_order.px_at_time_of_order
        self.aggression = pending_order.aggression
        self.koid = pending_order.koid
        self.opt_px = pending_order.opt_px
        self.order_alpha_dict = pending_order.order_alpha_dict

    def fill(self, fill: Fill):
        if fill.side != self.side:
            logger.warning(f"Fill came back as {fill.side} even though order was {self.side}")
            remaining_qty = self.remaining_qty + abs(fill.qty)
        else:
            remaining_qty = self.remaining_qty - abs(fill.qty)

        if remaining_qty < 1e-10:
            if remaining_qty < -1e-8:
                logger.warning(f"Stuffed fill on {self.symbol} fill: {fill.qty} order: {self.remaining_qty}")
            remaining_qty = 0

        logger.info(f"Outstanding order quantity {self.side} {self.symbol} {fill.qty} from {self.remaining_qty} -> {remaining_qty}")
        self.remaining_qty = remaining_qty

        fill.opt_px = self.opt_px
        self.fill_cnt += 1

    def order_amt(self) -> float:
        return self.remaining_qty * self.px * Side.sign(self.side)

    def expanding(self) -> bool:
        """Check if order expands position or reduces it.
        
        Returns:
            True if order increases position size, False if reducing
        """
        return (Side.sign(self.side) * self.qty_at_time_of_order) >= 0

    def check_market_price(self) -> bool:
        """Validate order price against current market.
        
        Ensures limit orders are not immediately marketable,
        which would result in taker fees.
        
        Returns:
            False if order would cross the spread, True otherwise
        """
        if self.side == Side.BUY and self.px >= self.ask:
            logger.warning(f"Marketable limit price {self.px} {self.bid=} / {self.ask=}")
            return False
        elif self.side == Side.SELL and self.px <= self.bid:
            logger.warning(f"Marketable limit price {self.px} {self.bid=} / {self.ask=}")
            return False
        return True

    def make_orderid(self) -> str:
        """Generate unique internal order ID.
        
        Creates hash-based ID from symbol, price, and timestamp.
        
        Returns:
            11-digit order ID string
        """
        ostr = self.symbol + str(self.px) + dt.now(timezone.utc).strftime("%y%m%d%H%M")
        self.koid = str(int(hashlib.sha1(ostr.encode("utf-8")).hexdigest(), 16) % (10 ** 11))
        return self.koid

    # legacy create order message
    def oms_create_order_msg(self, precision: Dict[str, float]) -> Optional[str]:
        mult = 1.0
        decimal_places = int(abs(math.log10(mult)))
        p_prec = precision['px']
        q_prec = int(precision['qty'] - decimal_places)
        qty_str = f"{self.orig_qty:.{q_prec}f}"
        if float(qty_str) == 0:
            logger.warning(f"zero quantity order for {self.symbol}! {self.order_type} -> {qty_str}")
            return None

        msg_row = f"PLACE {self.koid} {self.side} {self.symbol} {qty_str} @ {self.px:0.0{p_prec}f} {self.order_type} {self.time_in_force}"
        return msg_row

    # new oms create order message
    def cpp_oms_create_order_msg(self, precision: Dict[str, float]) -> Optional[str]:
        mult = 1.0
        decimal_places = int(abs(math.log10(mult)))
        p_prec = precision['px']
        q_prec = int(precision['qty'] - decimal_places)
        qty_str = f"{self.orig_qty:.{q_prec}f}"
        if float(qty_str) == 0:
            logger.warning(f"zero quantity order for {self.symbol}! {self.order_type} -> {qty_str}")
            return None
        px_str = f"{self.px:0.0{p_prec}f}" if self.px is not None else None

        if self.koid is not None:
            self.koid = self.make_orderid()
            logger.warning(f"Order does not have valid koid, creating new one: {self.koid}")

        client_oid = self.koid

        mapped_side = "BUY" if self.side == Side.BUY else "SELL"
        mapped_order_type = "MARKET" if self.order_type == "MARKET" else "LIMIT"

        # do not set tif for a market order, use "GTX" to implement post only orders
        mapped_time_in_force = "GTX" if "POST_ONLY" in self.order_type else self.time_in_force
        mapped_time_in_force = None if self.order_type == "MARKET" else mapped_time_in_force

        # currently use defaults
        reduce_only = True if "REDUCE" in self.order_type else None
        recv_window = 5000 # currently not parameterized use 5s default
        position_side = PositionSide.BOTH # default is one-way mode

        timestamp_ms = int(time.time() * 1000)

        payload = {
            "symbol": self.symbol,
            "side": mapped_side,
            "type": mapped_order_type,
            # note that this timestamp will be overwritten again by the OMS system with the current time
            # when the order is sent to Binance endpoint in order to avoid falling outside the recvWindow
            "timestamp": timestamp_ms,
            "timeInForce": mapped_time_in_force,
            "quantity": qty_str,
            "price": px_str,
            "positionSide": position_side,
            "reduceOnly": reduce_only,
            "newClientOrderId": client_oid,
            "recvWindow": recv_window,
        }

        # fields with None must not be sent
        payload = {k: v for k, v in payload.items() if v is not None}

        # wrap in envelope defining the payload type
        env = {
            "payload_type": OMSPayloadType.UM_CREATE_ORDER,
            "timestamp": timestamp_ms,
            "payload": payload,
        }
        return json.dumps(env)

    def cpp_oms_cancel_order_msg(self, oid=None) -> str:
        timestamp_ms = int(time.time() * 1000)
        recv_window = 5000  # currently not parameterized use 5s default

        payload = {
            "symbol": self.symbol,
            "timestamp": timestamp_ms,
            "recvWindow": recv_window,
        }

        if oid is not None:
            payload["orderId"] = oid

        if self.koid:
            payload["origClientOrderId"] = self.koid

        env = {
            "payload_type": OMSPayloadType.UM_CANCEL_ORDER,
            "timestamp": timestamp_ms,
            "payload": payload,
        }
        return json.dumps(env)

    @classmethod
    def cpp_oms_cancel_all_open_orders_msg(cls, symbol) -> str:
        timestamp_ms = int(time.time() * 1000)
        recv_window = 5000  # currently not parameterized use 5s default

        payload = {
            "symbol": symbol,
            "timestamp": timestamp_ms,
            "recvWindow": recv_window,
        }
        env = {
            "payload_type": OMSPayloadType.UM_CANCEL_ALL_OPEN_ORDERS,
            "timestamp": timestamp_ms,
            "payload": payload,
        }
        return json.dumps(env)

    def order_file_ln(self) -> str:
        order_info = f"ORDER|{self.koid}|{self.oid}|{self.symbol}|{self.side}|{self.order_type}|{self.time_in_force}|{self.orig_qty}|{self.px}|{self.qty_at_time_of_order}|{self.aggression}|{self.deficit}|{self.created_ts}|{self.exch_ts}|{self.acked_ts}|{self.bid}|{self.ask}|{self.opt_px}"
        order_info += "|"

        alpha_horizons = [
            f"alpha_{horizon}:{self.order_alpha_dict.get(f'alpha_{horizon}')}"
            for horizon in Order.MODEL_HORIZONS
        ]
        alpha_models = [
            f"alpha_{model}_{horizon}:{self.order_alpha_dict.get(f'alpha_{model}_{horizon}')}"
            for horizon in Order.MODEL_HORIZONS
            for model in Order.MODELS
        ]
        order_info += ",".join(alpha_horizons + alpha_models)
        order_info += "\n"

        return order_info

    def short_description(self) -> str:
        return f"{self.side} {self.orig_qty} {self.symbol}@{self.px}"

    def __str__(self) -> str:
        order_amt = self.order_amt()
        msg = f"ORDER {self.side} {self.symbol} {self.orig_qty:.4f} @ {self.px:.6f} {self.order_type} ${order_amt:.0f} CURR_QTY: {self.qty_at_time_of_order}"
        if self.bid is not None and self.ask is not None:
            msg += f" [B:{self.bid:.4f}/A:{self.ask:.4f}]"
        if self.aggression is not None:
            msg += f" AGG:{self.aggression}"
        if self.deficit is not None:
            msg += f" DEF:{fpct(self.deficit)}"
        return msg


class Cancel:
    """Order cancellation record.
    
    Tracks order cancellations received from the exchange with
    details about the cancelled order.
    
    Attributes:
        symbol: Trading symbol
        oid: Exchange order ID
        side: Order side that was cancelled
        qty: Original order quantity
        px: Order price
        recv_ts: Timestamp when cancellation was received
        remaining_qty: Quantity remaining when cancelled
        koid: Internal order ID
    """
    
    def __init__(self, symbol: str, oid: str, side: Side, qty: float, px: float, recv_ts: dt):
        self.symbol: str = symbol
        self.oid: str = oid
        self.side: Side = side
        self.qty: float = qty
        self.px: float = px
        self.recv_ts: dt = recv_ts
        self.remaining_qty: Optional[float] = None
        self.koid: Optional[str] = None

    @classmethod
    def from_binance_json(cls, msg: dict, recv_ts: Optional[dt] = None) -> Optional[Self]:
        """Parse Binance WebSocket cancel message.
        
        Args:
            msg: Binance ORDER_TRADE_UPDATE message
            recv_ts: Local receipt timestamp
            
        Returns:
            Cancel object or None if parsing fails
        """
        try:
            om = msg['o']
            oid = om['i']
            koid = om['c'] if 'c' in om else None
            symbol = om['s']
            side = Side.from_string(om['S'])
            qty = float(om['q'])
            px = float(om['p'])
        except Exception as e:
            logger.error(f"Could not parse cancel message {msg} with error {e}", key="fail to parse cancel message")
            return None

        cancel = cls(
            symbol=symbol,
            oid=oid,
            side=side,
            qty=qty,
            px=px,
            recv_ts=recv_ts
        )
        cancel.koid = koid
        return cancel

    def update_from_order(self, order: Order) -> None:
        """Update cancel with order details.
        
        Args:
            order: Order object that was cancelled
        """
        self.remaining_qty = order.remaining_qty
        self.koid = order.koid

    def short_description(self) -> str:
        """Get brief cancel description.
        
        Returns:
            String like "B 0.1 BTCUSDT@42000"
        """
        return f"{self.side} {self.qty} {self.symbol}@{self.px}"

    def order_file_ln(self) -> str:
        """Format cancel for order file logging.
        
        Returns:
            Pipe-delimited cancel record with newline
        """
        return f"CANCEL|{self.recv_ts}|{self.symbol}|{self.oid}|{self.qty}|{self.px}|{self.remaining_qty}|{self.koid}\n"
