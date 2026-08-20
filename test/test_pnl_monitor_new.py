#!/usr/bin/env python3
"""Unit tests for PnlMonitorNew class."""
# pylint: disable=no-value-for-parameter,unexpected-keyword-arg,no-member

from datetime import datetime as dt, timezone
from unittest.mock import Mock, patch
import pandas as pd

import pytest
from lib.pnl_new.pnl_monitor import PnlMonitorNew, _get_intraday_pnl_str


def create_mock_pnl_calculator():
    """Create a mock Pnl calculator with test data."""
    mock_calc = Mock()

    # Mock fills dataframe
    mock_calc.fills_df = pd.DataFrame([
        {
            'ts': dt(2025, 10, 4, 10, 0, tzinfo=timezone.utc),
            'date': dt(2025, 10, 4).date(),
            'symbol_venue': 'BTCUSDT_binance-futures',
            'side': 'B',
            'fill_px': 50000.0,
            'fill_qty': 10.0,
            'commission': 2.5,
        }
    ])

    return mock_calc


def test_pnl_monitor_new_init():
    """Test PnlMonitorNew initialization."""
    print("=" * 80)
    print("Testing PnlMonitorNew Initialization")
    print("=" * 80)

    # Create monitor with minimal config
    monitor = PnlMonitorNew(config={}, alert_mins=60, debug=True)

    # Verify basic attributes
    assert monitor.alert_mins == 60
    assert monitor.debug is True
    assert monitor.config == {}

    print("✓ PnlMonitorNew initialization test passed")


def test_calc_pnl_no_update():
    """Test calc_pnl basic operation."""
    print("=" * 80)
    print("Testing calc_pnl (basic operation)")
    print("=" * 80)

    with patch('lib.pnl_new.pnl_monitor.BinancePnl') as MockBinancePnl:
        # Mock the BinancePnl calculator
        mock_pnl_instance = Mock()
        MockBinancePnl.return_value = mock_pnl_instance

        # Mock aggregate_daily_portfolio return value
        portfolio_df = pd.DataFrame([{
            'net_pnl': 1000.0,
            'realized_pnl': 800.0,
            'unrealized_pnl': 200.0,
            'commission': 100.0,
            'funding_income': 50.0,
            'long_notional': 500000.0,
            'short_notional': -450000.0,
            'gross_notional': 950000.0,
            'fill_dollars_abs': 100000.0,
            'fill_count': 10,
            'unrealized_pnl_tot_cum': 200.0,
            'balance': 1000000.0
        }])
        mock_pnl_instance.aggregate_daily_portfolio.return_value = portfolio_df

        # Create monitor
        monitor = PnlMonitorNew(config={}, alert_mins=60, debug=True)

        # Call calc_pnl
        result = monitor.calc_pnl()

        # Verify BinancePnl was called
        assert MockBinancePnl.called

        # Verify result mapping
        assert result['total_pnl_daily'] == 1000.0
        assert result['realized_daily'] == 800.0
        assert result['unrealized_daily'] == 200.0
        assert result['fees_usd_daily'] == 100.0
        assert result['long'] == 500000.0
        assert result['short'] == -450000.0
        assert result['dollars_traded_daily'] == 100000.0
        assert result['fill_cnt_daily'] == 10
        assert result['unrealized_pnl_tot_cum'] == 200.0

    print("✓ calc_pnl test passed")


def test_get_intraday_pnl_str():
    """Test _get_intraday_pnl_str static method."""
    print("=" * 80)
    print("Testing _get_intraday_pnl_str")
    print("=" * 80)

    latest_pnl_dict = {
        'total_pnl_daily': 1000.0,
        'realized_daily': 800.0,
        'unrealized_daily': 200.0,
        'fees_usd_daily': 50.0,
        'long': 500000.0,
        'short': -450000.0,
        'notional': 950000.0,  # Add required field
        'balance': 1000000.0,  # Add required field
        'net_balance': 950000.0,  # Add required field
        'dollars_traded_daily': 100000.0,
        'fill_cnt_daily': 10,
        'cum_unrealized_pnl': 150.0,
        'unrealized_pnl_tot_cum': 150.0,  # Add required field
        'funding_income': 25.0,  # Add required field
        'mtd_net_pnl': 5000.0  # Add required field
    }

    total_balance = 1000000.0
    funding = 25.0

    # Call module-level function
    msg = _get_intraday_pnl_str(latest_pnl_dict)

    # Verify output contains key information
    assert 'Daily Pnl: $1,000' in msg
    assert 'Realized: $800' in msg
    assert 'Unrealized: $200' in msg
    assert 'Fees: $50' in msg
    assert 'Funding income: $25' in msg
    assert 'Count: 10' in msg

    print("✓ _get_intraday_pnl_str test passed")


if __name__ == '__main__':
    test_pnl_monitor_new_init()
    test_calc_pnl_no_update()
    test_get_intraday_pnl_str()

    print("\n" + "=" * 80)
    print("All PnlMonitorNew tests passed!")
    print("=" * 80)
