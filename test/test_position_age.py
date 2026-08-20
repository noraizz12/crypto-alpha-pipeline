#!/usr/bin/env python3
"""Unit tests for position age calculation in BinancePnl."""
# pylint: disable=no-value-for-parameter,unexpected-keyword-arg,no-member,protected-access
# pylint: disable=import-outside-toplevel,unnecessary-dunder-call

from datetime import datetime as dt, timezone
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from lib.pnl_new.binance_pnl import BinancePnl


def create_mock_binance_pnl() -> Mock:
    """Create a mock BinancePnl instance with _calculate_position_age method."""
    mock_pnl = Mock(spec=BinancePnl)
    calc_age = BinancePnl._calculate_position_age.__get__(mock_pnl, BinancePnl)
    mock_pnl._calculate_position_age = calc_age
    return mock_pnl


def test_calculate_position_age_empty_positions() -> None:
    """Test that empty position history returns zero age for all symbols."""
    print("=" * 80)
    print("Testing _calculate_position_age: empty positions")
    print("=" * 80)

    with patch('lib.pnl_new.binance_pnl.load_binance_positions') as mock_load:
        mock_load.return_value = pd.DataFrame()

        merged_df = pd.DataFrame({
            'qty': [100.0, -50.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 10, 1, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 10, 1, tzinfo=timezone.utc), 'ETHUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))

        mock_pnl = create_mock_binance_pnl()
        mock_pnl.end_date = dt(2025, 10, 1).date()
        mock_pnl.dir_manager = Mock()
        mock_pnl.dir_manager.BINANCE_POSITION_DIR = '/mock/path'

        result = mock_pnl._calculate_position_age(merged_df)
        assert (result == 0).all()

    print("✓ Empty positions test passed - returns zeros")


def test_calculate_position_age_none_positions() -> None:
    """Test that None position history returns zero age for all symbols."""
    print("=" * 80)
    print("Testing _calculate_position_age: None positions")
    print("=" * 80)

    with patch('lib.pnl_new.binance_pnl.load_binance_positions') as mock_load:
        mock_load.return_value = None

        merged_df = pd.DataFrame({
            'qty': [100.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 10, 1, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))

        mock_pnl = create_mock_binance_pnl()
        mock_pnl.end_date = dt(2025, 10, 1).date()
        mock_pnl.dir_manager = Mock()
        mock_pnl.dir_manager.BINANCE_POSITION_DIR = '/mock/path'

        result = mock_pnl._calculate_position_age(merged_df)
        assert (result == 0).all()

    print("✓ None positions test passed - returns zeros")


def test_calculate_position_age_no_crossings() -> None:
    """Test position age when there are no zero crossings - returns age from first record."""
    print("=" * 80)
    print("Testing _calculate_position_age: no crossings")
    print("=" * 80)

    with patch('lib.pnl_new.binance_pnl.load_binance_positions') as mock_load:
        # Position history: always positive, no crossings
        hist_positions_df = pd.DataFrame({
            'qty': [100.0, 150.0, 200.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 25, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 27, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))
        mock_load.return_value = hist_positions_df

        merged_df = pd.DataFrame({
            'qty': [200.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))

        mock_pnl = create_mock_binance_pnl()
        mock_pnl.end_date = dt(2025, 9, 30).date()
        mock_pnl.dir_manager = Mock()
        mock_pnl.dir_manager.BINANCE_POSITION_DIR = '/mock/path'

        result = mock_pnl._calculate_position_age(merged_df)

        # Position started on Sept 25, current is Sept 30, so age = 5 days
        assert result.iloc[0] == 5.0

    print("✓ No crossings test passed - age from first record")


def test_calculate_position_age_with_sign_change() -> None:
    """Test position age resets after sign change (long to short)."""
    print("=" * 80)
    print("Testing _calculate_position_age: sign change crossing")
    print("=" * 80)

    with patch('lib.pnl_new.binance_pnl.load_binance_positions') as mock_load:
        # Position crossed from negative to positive on Sept 28
        hist_positions_df = pd.DataFrame({
            'qty': [-100.0, -50.0, 100.0, 150.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 25, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 27, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 28, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))
        mock_load.return_value = hist_positions_df

        merged_df = pd.DataFrame({
            'qty': [150.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))

        mock_pnl = create_mock_binance_pnl()
        mock_pnl.end_date = dt(2025, 9, 30).date()
        mock_pnl.dir_manager = Mock()
        mock_pnl.dir_manager.BINANCE_POSITION_DIR = '/mock/path'

        result = mock_pnl._calculate_position_age(merged_df)

        # Position restarted on Sept 28, current is Sept 30, so age = 2 days
        assert result.iloc[0] == 2.0

    print("✓ Sign change crossing test passed - age resets correctly")


def test_calculate_position_age_from_zero() -> None:
    """Test position age when going from near-zero to significant position."""
    print("=" * 80)
    print("Testing _calculate_position_age: from zero to position")
    print("=" * 80)

    with patch('lib.pnl_new.binance_pnl.load_binance_positions') as mock_load:
        # Position was near-zero, then became significant on Sept 28
        hist_positions_df = pd.DataFrame({
            'qty': [0.00001, 0.00005, 100.0, 150.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 25, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 27, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 28, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))
        mock_load.return_value = hist_positions_df

        merged_df = pd.DataFrame({
            'qty': [150.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))

        mock_pnl = create_mock_binance_pnl()
        mock_pnl.end_date = dt(2025, 9, 30).date()
        mock_pnl.dir_manager = Mock()
        mock_pnl.dir_manager.BINANCE_POSITION_DIR = '/mock/path'

        result = mock_pnl._calculate_position_age(merged_df)

        # Position started on Sept 28 (from near-zero), current is Sept 30, age = 2 days
        assert result.iloc[0] == 2.0

    print("✓ From zero to position test passed")


def test_calculate_position_age_multiple_crossings() -> None:
    """Test that with multiple crossings, uses the last one matching current direction."""
    print("=" * 80)
    print("Testing _calculate_position_age: multiple crossings")
    print("=" * 80)

    with patch('lib.pnl_new.binance_pnl.load_binance_positions') as mock_load:
        # Multiple crossings: positive -> negative -> positive
        hist_positions_df = pd.DataFrame({
            'qty': [100.0, -50.0, 50.0, -30.0, 80.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 25, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 26, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 27, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 28, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))
        mock_load.return_value = hist_positions_df

        merged_df = pd.DataFrame({
            'qty': [80.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))

        mock_pnl = create_mock_binance_pnl()
        mock_pnl.end_date = dt(2025, 9, 30).date()
        mock_pnl.dir_manager = Mock()
        mock_pnl.dir_manager.BINANCE_POSITION_DIR = '/mock/path'

        result = mock_pnl._calculate_position_age(merged_df)

        # Last positive crossing on Sept 30, so age = 0 days
        assert result.iloc[0] == 0.0

    print("✓ Multiple crossings test passed - uses last matching direction")


def test_calculate_position_age_short_position() -> None:
    """Test position age for short positions."""
    print("=" * 80)
    print("Testing _calculate_position_age: short position")
    print("=" * 80)

    with patch('lib.pnl_new.binance_pnl.load_binance_positions') as mock_load:
        # Position went from positive to negative on Sept 28
        hist_positions_df = pd.DataFrame({
            'qty': [100.0, 50.0, -100.0, -150.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 25, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 27, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 28, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))
        mock_load.return_value = hist_positions_df

        merged_df = pd.DataFrame({
            'qty': [-150.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))

        mock_pnl = create_mock_binance_pnl()
        mock_pnl.end_date = dt(2025, 9, 30).date()
        mock_pnl.dir_manager = Mock()
        mock_pnl.dir_manager.BINANCE_POSITION_DIR = '/mock/path'

        result = mock_pnl._calculate_position_age(merged_df)

        # Short position started on Sept 28, current is Sept 30, age = 2 days
        assert result.iloc[0] == 2.0

    print("✓ Short position test passed")


def test_calculate_position_age_near_zero_tolerance() -> None:
    """Test that near-zero quantities (floating point) are treated as zero."""
    print("=" * 80)
    print("Testing _calculate_position_age: near-zero tolerance")
    print("=" * 80)

    with patch('lib.pnl_new.binance_pnl.load_binance_positions') as mock_load:
        # Position with floating point near-zero (like 2.966377e-16)
        hist_positions_df = pd.DataFrame({
            'qty': [100.0, 2.966377e-16, 100.0, 150.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 25, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 27, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 28, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))
        mock_load.return_value = hist_positions_df

        merged_df = pd.DataFrame({
            'qty': [150.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))

        mock_pnl = create_mock_binance_pnl()
        mock_pnl.end_date = dt(2025, 9, 30).date()
        mock_pnl.dir_manager = Mock()
        mock_pnl.dir_manager.BINANCE_POSITION_DIR = '/mock/path'

        result = mock_pnl._calculate_position_age(merged_df)

        # Near-zero on Sept 27 treated as zero, position restarted Sept 28, age = 2 days
        assert result.iloc[0] == 2.0

    print("✓ Near-zero tolerance test passed")


def test_calculate_position_age_zero_current_position() -> None:
    """Test that zero current position returns zero age."""
    print("=" * 80)
    print("Testing _calculate_position_age: zero current position")
    print("=" * 80)

    with patch('lib.pnl_new.binance_pnl.load_binance_positions') as mock_load:
        hist_positions_df = pd.DataFrame({
            'qty': [100.0, 150.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 25, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))
        mock_load.return_value = hist_positions_df

        # Current position is zero
        merged_df = pd.DataFrame({
            'qty': [0.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))

        mock_pnl = create_mock_binance_pnl()
        mock_pnl.end_date = dt(2025, 9, 30).date()
        mock_pnl.dir_manager = Mock()
        mock_pnl.dir_manager.BINANCE_POSITION_DIR = '/mock/path'

        result = mock_pnl._calculate_position_age(merged_df)

        # Zero position should have zero age
        assert result.iloc[0] == 0.0

    print("✓ Zero current position test passed")


def test_calculate_position_age_missing_symbol_in_history() -> None:
    """Test that symbol not in history returns zero age."""
    print("=" * 80)
    print("Testing _calculate_position_age: missing symbol in history")
    print("=" * 80)

    with patch('lib.pnl_new.binance_pnl.load_binance_positions') as mock_load:
        # History only has BTC, not ETH
        hist_positions_df = pd.DataFrame({
            'qty': [100.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 25, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))
        mock_load.return_value = hist_positions_df

        # Current position is ETH (not in history)
        merged_df = pd.DataFrame({
            'qty': [50.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'ETHUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))

        mock_pnl = create_mock_binance_pnl()
        mock_pnl.end_date = dt(2025, 9, 30).date()
        mock_pnl.dir_manager = Mock()
        mock_pnl.dir_manager.BINANCE_POSITION_DIR = '/mock/path'

        result = mock_pnl._calculate_position_age(merged_df)

        # ETH not in history, should have zero age
        assert result.iloc[0] == 0.0

    print("✓ Missing symbol in history test passed")


def test_calculate_position_age_multiple_symbols() -> None:
    """Test position age calculation for multiple symbols simultaneously."""
    print("=" * 80)
    print("Testing _calculate_position_age: multiple symbols")
    print("=" * 80)

    with patch('lib.pnl_new.binance_pnl.load_binance_positions') as mock_load:
        # BTC: no crossings (5 days), ETH: crossed on Sept 28 (2 days)
        hist_positions_df = pd.DataFrame({
            'qty': [100.0, 150.0, -50.0, 100.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 25, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 25, tzinfo=timezone.utc), 'ETHUSDT_binance-futures'),
            (dt(2025, 9, 28, tzinfo=timezone.utc), 'ETHUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))
        mock_load.return_value = hist_positions_df

        merged_df = pd.DataFrame({
            'qty': [150.0, 100.0],
        }, index=pd.MultiIndex.from_tuples([
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'BTCUSDT_binance-futures'),
            (dt(2025, 9, 30, tzinfo=timezone.utc), 'ETHUSDT_binance-futures'),
        ], names=['ts', 'symbol_venue']))

        mock_pnl = create_mock_binance_pnl()
        mock_pnl.end_date = dt(2025, 9, 30).date()
        mock_pnl.dir_manager = Mock()
        mock_pnl.dir_manager.BINANCE_POSITION_DIR = '/mock/path'

        result = mock_pnl._calculate_position_age(merged_df)

        # BTC: 5 days (no crossing), ETH: 2 days (crossed on Sept 28)
        btc_age = result.xs('BTCUSDT_binance-futures', level='symbol_venue').iloc[0]
        eth_age = result.xs('ETHUSDT_binance-futures', level='symbol_venue').iloc[0]

        assert btc_age == 5.0
        assert eth_age == 2.0

    print("✓ Multiple symbols test passed - each calculated independently")
