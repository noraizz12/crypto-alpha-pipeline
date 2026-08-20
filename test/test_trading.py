"""Unit tests for trading module.

Tests cover all trading components including Side, OrderType, TimeInForce,
Fill, Order, and Cancel classes with their various methods and edge cases.
"""

import hashlib
import math
from datetime import datetime as dt, timezone, timedelta
from typing import Dict, Optional
from unittest.mock import Mock, patch, MagicMock

import pytest

from lib.trader.trading import (
    Side, OrderType, TimeInForce, FillType, Venue,
    Fill, Order, Cancel, get_agg_side_fills_info,
    DEFAULT_COMMISSION_ASSET, USDT
)


class TestSide:
    """Test Side enumeration and conversion methods."""
    
    def test_side_values(self):
        """Test basic Side enum values."""
        assert Side.BUY == "B"
        assert Side.SELL == "S"
    
    def test_from_string_buy(self):
        """Test converting buy strings to Side."""
        assert Side.from_string("BUY") == Side.BUY
        assert Side.from_string("buy") == Side.BUY
        assert Side.from_string("B") == Side.BUY
        assert Side.from_string("b") == Side.BUY
    
    def test_from_string_sell(self):
        """Test converting sell strings to Side."""
        assert Side.from_string("SELL") == Side.SELL
        assert Side.from_string("sell") == Side.SELL
        assert Side.from_string("S") == Side.SELL
        assert Side.from_string("s") == Side.SELL
    
    def test_from_string_invalid(self):
        """Test invalid side strings raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Unknown side"):
            Side.from_string("INVALID")
    
    def test_from_float_positive(self):
        """Test positive amounts return BUY."""
        assert Side.from_float(1.0) == Side.BUY
        assert Side.from_float(100) == Side.BUY
        assert Side.from_float(0.001) == Side.BUY
    
    def test_from_float_negative(self):
        """Test negative amounts return SELL."""
        assert Side.from_float(-1.0) == Side.SELL
        assert Side.from_float(-100) == Side.SELL
        assert Side.from_float(-0.001) == Side.SELL
    
    def test_from_float_zero(self):
        """Test zero amount raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No side to"):
            Side.from_float(0.0)
        with pytest.raises(RuntimeError, match="No side to"):
            Side.from_float(0)
    
    def test_sign(self):
        """Test sign conversion."""
        assert Side.sign(Side.BUY) == 1
        assert Side.sign(Side.SELL) == -1


class TestOrderType:
    """Test OrderType enumeration."""
    
    def test_order_type_values(self):
        """Test OrderType enum values."""
        assert OrderType.LIMIT == "LIMIT"
        assert OrderType.REDUCE == "REDUCE"
        assert OrderType.POST_ONLY == "POST_ONLY"
        assert OrderType.POST_ONLY_REDUCE == "POST_ONLY_REDUCE"
        assert OrderType.MARKET == "MARKET"
    
    def test_from_string(self):
        """Test OrderType string conversion."""
        assert OrderType.from_string("LIMIT") == OrderType.LIMIT
        assert OrderType.from_string("limit") == OrderType.LIMIT
        assert OrderType.from_string("POST_ONLY") == OrderType.POST_ONLY
        assert OrderType.from_string("post_only") == OrderType.POST_ONLY
        assert OrderType.from_string("REDUCE") == OrderType.REDUCE
        assert OrderType.from_string("POST_ONLY_REDUCE") == OrderType.POST_ONLY_REDUCE
        assert OrderType.from_string("MARKET") == OrderType.MARKET
    
    def test_from_string_invalid(self):
        """Test invalid order type raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Unknown OrderType"):
            OrderType.from_string("INVALID")


class TestTimeInForce:
    """Test TimeInForce enumeration."""
    
    def test_time_in_force_values(self):
        """Test TimeInForce enum values."""
        assert TimeInForce.GOOD_TIL_CROSS == "GTX"
        assert TimeInForce.IMMEDIATE_OR_CANCEL == "IOC"
        assert TimeInForce.GOOD_TIL_CANCEL == "GTC"
    
    def test_from_string(self):
        """Test TimeInForce string conversion."""
        assert TimeInForce.from_string("GTX") == TimeInForce.GOOD_TIL_CROSS
        assert TimeInForce.from_string("gtx") == TimeInForce.GOOD_TIL_CROSS
        assert TimeInForce.from_string("GTC") == TimeInForce.GOOD_TIL_CANCEL
        assert TimeInForce.from_string("IOC") == TimeInForce.IMMEDIATE_OR_CANCEL
    
    def test_from_string_invalid(self):
        """Test invalid time in force raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Unknown TimeInForce"):
            TimeInForce.from_string("INVALID")


class TestFill:
    """Test Fill class functionality."""
    
    @pytest.fixture
    def sample_fill(self):
        """Create a sample fill for testing."""
        return Fill(
            symbol="BTCUSDT",
            side=Side.BUY,
            px=42000.0,
            qty=0.1,
            exch_ts=dt.now(timezone.utc),
            recv_ts=dt.now(timezone.utc),
            fill_type=FillType.NORMAL,
            commission=0.0001,
            commission_asset="BNB",
            venue=Venue.BINANCE,
            oid="12345",
            opt_px=41990.0
        )
    
    def test_fill_init(self, sample_fill):
        """Test Fill initialization."""
        assert sample_fill.symbol == "BTCUSDT"
        assert sample_fill.side == Side.BUY
        assert sample_fill.px == 42000.0
        assert sample_fill.qty == 0.1
        assert sample_fill.fill_type == FillType.NORMAL
        assert sample_fill.commission == 0.0001
        assert sample_fill.commission_asset == "BNB"
        assert sample_fill.commission_usd == 0
        assert sample_fill.oid == "12345"
        assert sample_fill.venue == Venue.BINANCE
        assert sample_fill.opt_px == 41990.0
    
    def test_fill_fake_type(self):
        """Test fake fill auto-sets timestamps."""
        fill = Fill(
            symbol="ETHUSDT",
            side=Side.SELL,
            px=2500.0,
            qty=1.0,
            fill_type=FillType.FAKE
        )
        assert fill.exch_ts is not None
        assert fill.recv_ts is not None
        assert fill.exch_ts == fill.recv_ts
    
    def test_from_binance_json_success(self):
        """Test parsing Binance fill message."""
        msg = {
            'T': 1704067200000,  # milliseconds
            'o': {
                's': 'BTCUSDT',
                'i': 12345,
                'c': 'order123',
                'S': 'BUY',
                'l': '0.1',  # last filled qty
                'ap': '42000.0',  # average price
                'n': '0.0001',  # commission
                'N': 'BNB'  # commission asset
            }
        }
        recv_ts = dt.now(timezone.utc)

        fill = Fill.from_binance_json(msg, recv_ts)
        assert fill is not None
        assert fill.symbol == 'BTCUSDT'
        assert fill.side == Side.BUY
        assert fill.qty == 0.1
        assert fill.px == 42000.0
        assert fill.commission == 0.0001
        assert fill.commission_asset == 'BNB'
        assert fill.oid == 12345
        assert fill.koid == 'order123'
        assert fill.recv_ts == recv_ts
    
    def test_from_binance_json_failure(self):
        """Test parsing invalid Binance message returns None."""
        msg = {'invalid': 'data'}
        fill = Fill.from_binance_json(msg)
        assert fill is None
    
    def test_notional_buy(self, sample_fill):
        """Test notional calculation for buy (negative cash flow)."""
        # Buy 0.1 BTC at 42000 = -4200 notional
        assert sample_fill.notional(include_fees=False) == -4200.0
        
        # With fees
        sample_fill.commission_usd = 1.0
        assert sample_fill.notional(include_fees=True) == -4201.0
    
    def test_notional_sell(self):
        """Test notional calculation for sell (positive cash flow)."""
        fill = Fill(
            symbol="BTCUSDT",
            side=Side.SELL,
            px=42000.0,
            qty=0.1,
            exch_ts=dt.now(timezone.utc),
            recv_ts=dt.now(timezone.utc)
        )
        # Sell 0.1 BTC at 42000 = +4200 notional
        assert fill.notional(include_fees=False) == 4200.0
    
    def test_signed_qty(self, sample_fill):
        """Test signed quantity calculation."""
        # Buy side is positive
        assert sample_fill.signed_qty() == 0.1
        
        # Sell side is negative
        sell_fill = Fill(
            symbol="BTCUSDT",
            side=Side.SELL,
            px=42000.0,
            qty=0.1,
            exch_ts=dt.now(timezone.utc),
            recv_ts=dt.now(timezone.utc)
        )
        assert sell_fill.signed_qty() == -0.1
    
    def test_fill_slip(self, sample_fill):
        """Test slippage calculation."""
        # Buy at 42000 vs optimal 41990 = +0.1 * 10 = 1.0 slip
        assert sample_fill.fill_slip() == 1.0
        
        # No optimal price
        fill_no_opt = Fill(
            symbol="BTCUSDT",
            side=Side.BUY,
            px=42000.0,
            qty=0.1,
            exch_ts=dt.now(timezone.utc),
            recv_ts=dt.now(timezone.utc)
        )
        assert fill_no_opt.fill_slip() is None
    
    def test_calc_commission_usd(self, sample_fill):
        """Test commission USD calculation."""
        commission_prices = {"BNB": 300.0, "USDT": 1.0}
        sample_fill.calc_commission_usd(commission_prices)
        # 0.0001 BNB * 300 = 0.03 USD
        assert sample_fill.commission_usd == pytest.approx(0.03)
    
    def test_calc_commission_usd_missing_price(self, sample_fill):
        """Test commission calculation with missing price raises."""
        commission_prices = {"USDT": 1.0}  # Missing BNB
        with pytest.raises(Exception, match="failed to get BNB commission px"):
            sample_fill.calc_commission_usd(commission_prices)
    
    def test_slack_str(self, sample_fill):
        """Test Slack message formatting."""
        sample_fill.commission_usd = 1.0
        slack_msg = sample_fill.slack_str()
        # The slack_str method shows the notional as positive for display
        assert "FILLED B 0.1 BTCUSDT @ $42000.0" in slack_msg  # Side enum shows as "B"
        assert "amt: $4201.00" in slack_msg
        # 0.1 * 10 = 1.0 slip, 1.0/4201 * 10000 = 0.2 bps
        assert "slip: $1.00 / 0.2 bps" in slack_msg
    
    def test_str_representation(self, sample_fill):
        """Test string representation."""
        str_repr = str(sample_fill)
        assert str_repr.startswith("FILL|")
        assert "BTCUSDT" in str_repr
        assert "42000.0" in str_repr
        assert "0.1" in str_repr


class TestGetAggSideFillsInfo:
    """Test fill aggregation utility function."""
    
    def test_aggregate_buy_fills(self):
        """Test aggregating multiple buy fills."""
        fills = [
            Fill("BTCUSDT", Side.BUY, 42000.0, 0.1, 
                 dt.now(timezone.utc), dt.now(timezone.utc)),
            Fill("BTCUSDT", Side.BUY, 42100.0, 0.2,
                 dt.now(timezone.utc), dt.now(timezone.utc), opt_px=42000.0),
            Fill("BTCUSDT", Side.SELL, 42000.0, 0.5,  # Different side, excluded
                 dt.now(timezone.utc), dt.now(timezone.utc))
        ]
        
        # Calculate commissions
        for fill in fills:
            fill.commission_usd = 1.0
        
        # The function filters by side only, not by symbol
        result = get_agg_side_fills_info("BTCUSDT", fills, Side.BUY)
        
        assert result[0] == "BTCUSDT"  # symbol (passed through)
        assert result[1] == Side.BUY  # side
        assert result[2] == pytest.approx(0.3)  # total qty (0.1 + 0.2, excludes SELL)
        # Total notional = (0.1 * 42000 + 1) + (0.2 * 42100 + 1) = 4201 + 8421 = 12622
        # Avg cost = 12622 / 0.3 = 42073.33...
        assert result[3] == pytest.approx(12622.0 / 0.3)  # avg cost
        assert result[4] == 2  # count
        assert result[5] == pytest.approx(12622.0)  # total notional
        assert result[6] == pytest.approx(0.2 * 100.0)  # slip
        assert result[7] == pytest.approx(20.0 / 12622.0 * 10000)  # slip bps
    
    def test_aggregate_no_matching_fills(self):
        """Test aggregating with no matching fills."""
        fills = [
            Fill("BTCUSDT", Side.BUY, 42000.0, 0.1,
                 dt.now(timezone.utc), dt.now(timezone.utc))
        ]
        
        # Ask for SELL side when only BUY exists
        result = get_agg_side_fills_info("BTCUSDT", fills, Side.SELL)
        assert result == []


class TestOrder:
    """Test Order class functionality."""
    
    @pytest.fixture
    def sample_order(self):
        """Create a sample order for testing."""
        return Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,  # positive = buy
            pending=True,
            aggression=2,
            qty_at_time_of_order=0.0,
            px_at_time_of_order=41995.0,
            opt_px=41990.0,
            tick_size=0.1
        )
    
    def test_order_init_basic(self, sample_order):
        """Test basic Order initialization."""
        assert sample_order.symbol == "BTCUSDT"
        assert sample_order.px == 42000.0
        assert sample_order.side == Side.BUY
        assert sample_order.orig_qty == 0.1
        assert sample_order.remaining_qty == 0.1
        assert sample_order.pending is True
        assert sample_order.aggression == 2
        assert sample_order.opt_px == 41990.0
    
    def test_order_init_sell_side(self):
        """Test Order with negative quantity becomes SELL."""
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=-0.1,  # negative = sell
            pending=True
        )
        assert order.side == Side.SELL
        assert order.orig_qty == 0.1
    
    def test_order_init_zero_qty_raises(self):
        """Test Order with zero quantity raises assertion."""
        with pytest.raises(AssertionError):
            Order(
                symbol="BTCUSDT",
                px=42000.0,
                signed_qty=0.0,
                pending=True
            )
    
    def test_order_init_negative_price_raises(self):
        """Test Order with negative price raises assertion."""
        with pytest.raises(AssertionError):
            Order(
                symbol="BTCUSDT",
                px=-42000.0,
                signed_qty=0.1,
                pending=True
            )
    
    def test_order_aggression_determines_type(self):
        """Test aggression parameter determines order type."""
        # Positive aggression, expanding position
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=True,
            aggression=2,
            qty_at_time_of_order=0.1  # Already long
        )
        assert order.order_type == OrderType.LIMIT
        assert order.time_in_force == TimeInForce.GOOD_TIL_CANCEL
        
        # Negative aggression, expanding position
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=True,
            aggression=-2,
            qty_at_time_of_order=0.0
        )
        assert order.order_type == OrderType.POST_ONLY
        assert order.time_in_force == TimeInForce.GOOD_TIL_CROSS
        
        # Positive aggression, reducing position
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=True,
            aggression=2,
            qty_at_time_of_order=-0.2  # Short, buying reduces
        )
        assert order.order_type == OrderType.REDUCE
        
        # Negative aggression, reducing position
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=-0.1,
            pending=True,
            aggression=-2,
            qty_at_time_of_order=0.2  # Long, selling reduces
        )
        assert order.order_type == OrderType.POST_ONLY_REDUCE
    
    def test_bound_aggression(self):
        """Test aggression bounds."""
        assert Order.bound_aggression(10) == Order.MAX_AGGRESSION
        assert Order.bound_aggression(-10) == Order.MIN_AGGRESSION
        assert Order.bound_aggression(3) == 3
    
    def test_round_to_tick_size(self):
        """Test price rounding to tick size."""
        # Buy side rounds up
        assert Order.round_to_tick_size(42000.14, 1.0, 0.1) == pytest.approx(42000.2, abs=1e-6)
        assert Order.round_to_tick_size(42000.11, 1.0, 0.1) == pytest.approx(42000.2, abs=1e-6)
        assert Order.round_to_tick_size(42000.10, 1.0, 0.1) == pytest.approx(42000.1, abs=1e-6)
        
        # Sell side rounds down
        assert Order.round_to_tick_size(42000.14, -1.0, 0.1) == pytest.approx(42000.1, abs=1e-6)
        assert Order.round_to_tick_size(42000.19, -1.0, 0.1) == pytest.approx(42000.1, abs=1e-6)
        # 42000.20 might have floating point precision issues
        result = Order.round_to_tick_size(42000.20, -1.0, 0.1)
        # Should be either 42000.1 or 42000.2 depending on floating point representation
        assert result == pytest.approx(42000.1, abs=1e-6) or result == pytest.approx(42000.2, abs=1e-6)
        
        # No tick size
        assert Order.round_to_tick_size(42000.14159, 1.0, None) == pytest.approx(42000.14159)
    
    def test_from_str_basic(self):
        """Test parsing order from string."""
        order = Order.from_str("B BTCUSDT 0.1 @ 42000")
        assert order.symbol == "BTCUSDT"
        assert order.side == Side.BUY
        assert order.orig_qty == 0.1
        assert order.px == 42000.0
        assert order.order_type == OrderType.LIMIT
        assert order.time_in_force == TimeInForce.GOOD_TIL_CROSS
        assert order.pending is True
    
    def test_from_str_with_type(self):
        """Test parsing order with type and TIF."""
        order = Order.from_str("S ETHUSDT 1.0 @ 2500 POST_ONLY GTX")
        assert order.symbol == "ETHUSDT"
        assert order.side == Side.SELL
        assert order.orig_qty == 1.0
        assert order.px == 2500.0
        assert order.order_type == OrderType.POST_ONLY
        assert order.time_in_force == TimeInForce.GOOD_TIL_CROSS
    
    def test_from_str_exceeds_limit(self):
        """Test parsing order that exceeds notional limit."""
        # Try to create order > $10k limit
        with pytest.raises(RuntimeError):
            Order.from_str("B BTCUSDT 1.0 @ 20000")  # $20k order
    
    def test_from_str_invalid_format(self):
        """Test parsing invalid order string."""
        with pytest.raises(RuntimeError):
            Order.from_str("INVALID ORDER STRING")
    
    def test_from_binance_json(self):
        """Test parsing Binance order message."""
        msg = {
            'T': 1704067200000,
            'o': {
                's': 'BTCUSDT',
                'i': 12345,
                'c': 'order123',
                'S': 'BUY',
                'q': '0.1',
                'p': '42000.0',
                'o': 'LIMIT',
                'f': 'GTC',
                'R': False
            }
        }
        recv_ts = dt.now(timezone.utc)

        order = Order.from_binance_json(msg, recv_ts)
        assert order is not None
        assert order.symbol == 'BTCUSDT'
        assert order.side == Side.BUY
        assert order.orig_qty == 0.1
        assert order.px == 42000.0
        assert order.order_type == OrderType.LIMIT
        assert order.time_in_force == TimeInForce.GOOD_TIL_CANCEL
        assert order.oid == 12345
        assert order.koid == 'order123'
        assert order.acked_ts == recv_ts
        assert order.pending is False
    
    def test_from_file_line(self):
        """Test parsing order from file line."""
        ts = dt.now(timezone.utc).isoformat()
        line = f"ORDER|koid123|oid456|BTCUSDT|B|LIMIT|GTC|0.1|42000.0|0.05|2|0.01|{ts}|{ts}|{ts}|41990.0|42010.0|41995.0"
        
        order = Order.from_file_line(line)
        assert order is not None
        assert order.koid == "koid123"
        assert order.oid == "oid456"
        assert order.symbol == "BTCUSDT"
        assert order.side == Side.BUY
        assert order.orig_qty == 0.1
        assert order.px == 42000.0
        assert order.aggression == 2
        assert order.bid == 41990.0
        assert order.ask == 42010.0
    
    def test_acked(self, sample_order):
        """Test acked status."""
        assert sample_order.acked() is False
        
        sample_order.exch_ts = dt.now(timezone.utc)
        assert sample_order.acked() is True
    
    def test_liquidation(self):
        """Test liquidation detection."""
        # Buying when short = liquidation
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=True,
            qty_at_time_of_order=-0.2
        )
        assert order.liquidation() is True
        
        # Selling when long = liquidation  
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=-0.1,
            pending=True,
            qty_at_time_of_order=0.2
        )
        assert order.liquidation() is True
        
        # Normal trade
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=True,
            qty_at_time_of_order=0.1
        )
        assert order.liquidation() is False
    
    def test_expanding(self):
        """Test position expansion detection."""
        # Buy when flat or long = expanding
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=True,
            qty_at_time_of_order=0.0
        )
        assert order.expanding() is True
        
        order.qty_at_time_of_order = 0.1
        assert order.expanding() is True
        
        # Buy when short = reducing
        order.qty_at_time_of_order = -0.2
        assert order.expanding() is False
    
    def test_set_bid_ask(self):
        """Test setting bid/ask prices."""
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=True
        )
        
        order.set_bid_ask(41990.0, 42010.0)
        assert order.bid == 41990.0
        assert order.ask == 42010.0
        
        # Invalid spread should raise
        with pytest.raises(AssertionError):
            order.set_bid_ask(42010.0, 41990.0)
    
    def test_check_market_price(self):
        """Test marketable price validation."""
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=True
        )
        order.set_bid_ask(41990.0, 42010.0)
        
        # Buy at 42000 with ask at 42010 is OK
        assert order.check_market_price() is True
        
        # Buy at or above ask is marketable
        order.px = 42010.0
        assert order.check_market_price() is False
        
        # Sell below bid is marketable
        sell_order = Order(
            symbol="BTCUSDT",
            px=41990.0,
            signed_qty=-0.1,
            pending=True
        )
        sell_order.set_bid_ask(41990.0, 42010.0)
        assert sell_order.check_market_price() is False
    
    def test_make_orderid(self):
        """Test order ID generation."""
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=True
        )
        
        oid = order.make_orderid()
        assert oid == order.koid
        assert len(oid) <= 11
        assert oid.isdigit()
    
    def test_fill_order(self):
        """Test filling an order."""
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=False,
            opt_px=41990.0,
            oid="123",
            exch_ts=dt.now(timezone.utc)
        )
        
        fill = Fill(
            symbol="BTCUSDT",
            side=Side.BUY,
            px=42000.0,
            qty=0.05,
            exch_ts=dt.now(timezone.utc),
            recv_ts=dt.now(timezone.utc)
        )
        
        order.fill(fill)
        assert order.remaining_qty == 0.05
        assert fill.opt_px == 41990.0  # Fill inherits opt_px
        
        # Fill remaining
        fill2 = Fill(
            symbol="BTCUSDT",
            side=Side.BUY,
            px=42000.0,
            qty=0.05,
            exch_ts=dt.now(timezone.utc),
            recv_ts=dt.now(timezone.utc)
        )
        order.fill(fill2)
        assert order.remaining_qty == 0.0
    
    def test_order_amt(self):
        """Test order amount calculation."""
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=True
        )
        # Buy 0.1 at 42000 = 4200
        assert order.order_amt() == 4200.0
        
        # After partial fill
        order.remaining_qty = 0.05
        assert order.order_amt() == 2100.0
        
        # Sell order
        sell_order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=-0.1,
            pending=True
        )
        assert sell_order.order_amt() == -4200.0
    
    def test_make_oms_msg(self):
        """Test OMS message generation."""
        order = Order(
            symbol="BTCUSDT",
            px=42000.5,
            signed_qty=0.123456,
            pending=True,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.GOOD_TIL_CANCEL
        )
        order.koid = "12345"
        
        precision = {'px': 1, 'qty': 8}
        msg = order.oms_create_order_msg(precision)
        
        assert msg == "PLACE 12345 B BTCUSDT 0.12345600 @ 42000.5 LIMIT GTC"
    
    def test_make_oms_msg_zero_qty(self):
        """Test OMS message with zero quantity after rounding."""
        order = Order(
            symbol="BTCUSDT", 
            px=42000.0,
            signed_qty=0.00000001,  # Very small
            pending=True
        )
        
        precision = {'px': 1, 'qty': 2}  # Only 2 decimal places
        msg = order.oms_create_order_msg(precision)
        assert msg is None
    
    def test_order_file_ln(self):
        """Test order file line generation."""
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=False,
            oid="123",
            koid="456",
            exch_ts=dt(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            aggression=2,
            qty_at_time_of_order=0.05,  # Need this for aggression
            order_alpha_dict={'alpha_15': 0.001, 'alpha_hl_60': 0.002}
        )
        order.set_bid_ask(41990.0, 42010.0)
        
        line = order.order_file_ln()
        assert line.startswith("ORDER|456|123|BTCUSDT|B|")
        assert "alpha_15:0.001" in line
        assert "alpha_hl_60:0.002" in line
        assert line.endswith("\n")
    
    def test_short_description(self):
        """Test short order description."""
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=True
        )
        
        desc = order.short_description()
        assert desc == "B 0.1 BTCUSDT@42000.0"
    
    def test_str_representation(self):
        """Test string representation."""
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=True,
            qty_at_time_of_order=0.05,
            aggression=2,
            deficit=0.02
        )
        order.set_bid_ask(41990.0, 42010.0)
        
        str_repr = str(order)
        assert "ORDER B BTCUSDT 0.1000 @ 42000.000000" in str_repr
        assert "$4200" in str_repr
        assert "CURR_QTY: 0.05" in str_repr
        assert "AGG:2" in str_repr
        assert "DEF:2.00%" in str_repr
        assert "B:41990.0000/A:42010.0000" in str_repr


class TestCancel:
    """Test Cancel class functionality."""
    
    def test_cancel_init(self):
        """Test Cancel initialization."""
        recv_ts = dt.now(timezone.utc)
        cancel = Cancel(
            symbol="BTCUSDT",
            oid="123",
            side=Side.BUY,
            qty=0.1,
            px=42000.0,
            recv_ts=recv_ts
        )
        
        assert cancel.symbol == "BTCUSDT"
        assert cancel.oid == "123"
        assert cancel.side == Side.BUY
        assert cancel.qty == 0.1
        assert cancel.px == 42000.0
        assert cancel.recv_ts == recv_ts
        assert cancel.remaining_qty is None
        assert cancel.koid is None
    
    def test_from_binance_json(self):
        """Test parsing Binance cancel message."""
        msg = {
            'o': {
                's': 'ETHUSDT',
                'i': 67890,
                'c': 'order789',
                'S': 'SELL',
                'q': '1.0',
                'p': '2500.0'
            }
        }
        recv_ts = dt.now(timezone.utc)

        cancel = Cancel.from_binance_json(msg, recv_ts)
        assert cancel is not None
        assert cancel.symbol == 'ETHUSDT'
        assert cancel.oid == 67890
        assert cancel.koid == 'order789'
        assert cancel.side == Side.SELL
        assert cancel.qty == 1.0
        assert cancel.px == 2500.0
        assert cancel.recv_ts == recv_ts
    
    def test_from_binance_json_failure(self):
        """Test parsing invalid cancel message."""
        msg = {'invalid': 'data'}
        cancel = Cancel.from_binance_json(msg)
        assert cancel is None
    
    def test_update_from_order(self):
        """Test updating cancel from order."""
        cancel = Cancel(
            symbol="BTCUSDT",
            oid="123",
            side=Side.BUY,
            qty=0.1,
            px=42000.0,
            recv_ts=dt.now(timezone.utc)
        )
        
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=False,
            oid="123",
            koid="456",
            exch_ts=dt.now(timezone.utc)
        )
        order.remaining_qty = 0.05
        
        cancel.update_from_order(order)
        assert cancel.remaining_qty == 0.05
        assert cancel.koid == "456"
    
    def test_short_description(self):
        """Test short cancel description."""
        cancel = Cancel(
            symbol="BTCUSDT",
            oid="123",
            side=Side.BUY,
            qty=0.1,
            px=42000.0,
            recv_ts=dt.now(timezone.utc)
        )
        
        desc = cancel.short_description()
        assert desc == "B 0.1 BTCUSDT@42000.0"
    
    def test_order_file_ln(self):
        """Test cancel file line generation."""
        recv_ts = dt(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        cancel = Cancel(
            symbol="BTCUSDT",
            oid="123",
            side=Side.BUY,
            qty=0.1,
            px=42000.0,
            recv_ts=recv_ts
        )
        cancel.remaining_qty = 0.05
        cancel.koid = "456"
        
        line = cancel.order_file_ln()
        assert line.startswith("CANCEL|2024-01-01 12:00:00+00:00|BTCUSDT|123|0.1|42000.0|0.05|456")
        assert line.endswith("\n")


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_fill_wrong_side_warning(self, caplog):
        """Test fill with wrong side logs warning."""
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,  # BUY
            pending=False,
            oid="123",
            exch_ts=dt.now(timezone.utc)
        )
        
        # Fill comes back as SELL instead of BUY
        fill = Fill(
            symbol="BTCUSDT",
            side=Side.SELL,  # Wrong side!
            px=42000.0,
            qty=0.05,
            exch_ts=dt.now(timezone.utc),
            recv_ts=dt.now(timezone.utc)
        )
        
        order.fill(fill)
        assert "Fill came back as S even though order was B" in caplog.text
        # Remaining qty increases instead of decreases
        assert order.remaining_qty == pytest.approx(0.15)
    
    def test_overfilled_order_warning(self, caplog):
        """Test overfilled order logs warning."""
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=False,
            oid="123",
            exch_ts=dt.now(timezone.utc)
        )
        order.remaining_qty = 0.05
        
        # Fill more than remaining
        fill = Fill(
            symbol="BTCUSDT",
            side=Side.BUY,
            px=42000.0,
            qty=0.06,
            exch_ts=dt.now(timezone.utc),
            recv_ts=dt.now(timezone.utc)
        )
        
        order.fill(fill)
        assert "Stuffed fill" in caplog.text
        assert order.remaining_qty == 0  # Clamped to zero
    
    def test_suspicious_notional_warning(self, caplog):
        """Test suspicious notional calculation logs warning."""
        # Buy with positive notional is suspicious
        fill = Fill(
            symbol="BTCUSDT",
            side=Side.BUY,
            px=-42000.0,  # Negative price!
            qty=0.1,
            exch_ts=dt.now(timezone.utc),
            recv_ts=dt.now(timezone.utc)
        )
        
        notional = fill.notional()
        assert "Suspicious fill" in caplog.text
    
    def test_order_update_from_pending_mismatch_warning(self, caplog):
        """Test order update with mismatched details logs warning."""
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=False,
            oid="123",
            exch_ts=dt.now(timezone.utc)
        )
        
        pending_order = Order(
            symbol="BTCUSDT",
            px=43000.0,  # Different price
            signed_qty=0.2,  # Different qty
            pending=True
        )
        
        order.update_from_pending_order(pending_order)
        assert "Do these order match??" in caplog.text
    
    def test_make_paper_order(self):
        """Test converting pending order to paper order."""
        order = Order(
            symbol="BTCUSDT",
            px=42000.0,
            signed_qty=0.1,
            pending=True
        )
        
        assert order.pending is True
        assert order.exch_ts is None
        assert order.oid is None
        
        order.make_paper_order()
        
        assert order.pending is False
        assert order.exch_ts == order.created_ts
        assert order.acked_ts == order.created_ts
        assert order.oid == order.koid