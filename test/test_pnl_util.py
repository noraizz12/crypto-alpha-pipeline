#!/usr/bin/env python3
"""Unit tests for pnl_util functions."""

import sys
from datetime import datetime as dt, timezone
import pandas as pd
import numpy as np

from lib.pnl_new.pnl_util import (
    calc_pnl_returns,
    aggregate_to_daily,
    calculate_performance_statistics,
    calculate_performance_by_month,
    compute_commissions
)


def test_calc_pnl_returns() -> int:
    """Test calc_pnl_returns function."""
    print("\n" + "=" * 80)
    print("Testing calc_pnl_returns")
    print("=" * 80)

    # Create sample PnL data
    data = {
        'ts': pd.date_range('2025-10-01', periods=5, freq='1h', tz=timezone.utc),
        'pnl_net': [100.0, -50.0, 200.0, 0.0, 150.0],
        'pnl_gross': [110.0, -40.0, 210.0, 10.0, 160.0],
        'notional': [10000.0, 5000.0, 20000.0, 0.0, 15000.0]
    }
    pnl_df = pd.DataFrame(data).set_index('ts')

    # Test with default pnl_col='pnl_net'
    result_df = calc_pnl_returns(pnl_df.copy())

    print("\n1. Input data:")
    print(pnl_df.to_string())

    print("\n2. Result with pnl_net returns:")
    print(result_df[['pnl_net', 'notional', 'pnl_net_ret']].to_string())

    # Verify calculations
    assert 'pnl_net_ret' in result_df.columns, "pnl_net_ret column should be created"
    assert result_df['pnl_net_ret'].iloc[0] == 100.0 / 10000.0, "First return should be 0.01"
    assert result_df['pnl_net_ret'].iloc[1] == -50.0 / 5000.0, "Second return should be -0.01"
    assert pd.isna(result_df['pnl_net_ret'].iloc[3]), "Return should be NaN when notional is 0 (meaningless metric)"

    # Test with pnl_col='pnl_gross'
    result_df = calc_pnl_returns(pnl_df.copy(), pnl_col='pnl_gross')

    print("\n3. Result with pnl_gross returns:")
    print(result_df[['pnl_gross', 'notional', 'pnl_gross_ret']].to_string())

    assert 'pnl_gross_ret' in result_df.columns, "pnl_gross_ret column should be created"
    assert result_df['pnl_gross_ret'].iloc[0] == 110.0 / 10000.0, "First gross return should be 0.011"

    print("\n✓ All calc_pnl_returns tests passed!")
    return 0


def test_aggregate_to_daily() -> int:
    """Test aggregate_to_daily function."""
    print("\n" + "=" * 80)
    print("Testing aggregate_to_daily")
    print("=" * 80)

    # Create intraday PnL data (hourly for 2 days)
    timestamps = pd.date_range('2025-10-01', periods=48, freq='1h', tz=timezone.utc)

    data = {
        'ts': timestamps,
        'qty': [10.0 + i * 0.1 for i in range(48)],
        'mark_price': [50000.0 + i * 10 for i in range(48)],
        'position': [500000.0 + i * 1000 for i in range(48)],
        'cost_basis': [49000.0] * 48,
        'cost_cum': [490000.0 + i * 100 for i in range(48)],
        'notional': [500000.0 + i * 1000 for i in range(48)],
        'pnl_gross_cum': [1000.0 + i * 50 for i in range(48)],
        'pnl_net_cum': [900.0 + i * 45 for i in range(48)],
        'realized_pnl_cum': [100.0 + i * 10 for i in range(48)],
        'unrealized_pnl_cum': [800.0 + i * 35 for i in range(48)],
        'commission_cum': [100.0 + i * 5 for i in range(48)],
        'funding_income_cum': [50.0 + i * 2 for i in range(48)],
        'abs_dollars_cum': [10000.0 + i * 500 for i in range(48)],
        'fill_qty': [0.1] * 48,
        'fill_dollars': [5000.0] * 48,
        'fill_count': [1.0] * 48,
        'commission': [5.0] * 48,
        'funding_income': [2.0] * 48,
        'realized_pnl': [10.0] * 48,
        'pnl_gross': [50.0] * 48,
        'pnl_net': [45.0] * 48,
        'unrealized_pnl': [35.0] * 48,
        'pnl_gross_ret': [0.0001] * 48,
        'pnl_net_ret': [0.00009] * 48,
    }
    pnl_df = pd.DataFrame(data).set_index('ts')

    print("\n1. Input intraday data (first 5 rows):")
    print(pnl_df.head().to_string())

    # Aggregate to daily
    daily_df = aggregate_to_daily(pnl_df)

    print("\n2. Daily aggregated data:")
    print(daily_df.to_string())

    # Verify results
    assert len(daily_df) == 2, "Should have 2 days of data"
    assert daily_df.index[0].date() == dt(2025, 10, 1).date(), "First day should be 2025-10-01"
    assert daily_df.index[1].date() == dt(2025, 10, 2).date(), "Second day should be 2025-10-02"

    # Verify last values (end-of-day snapshots)
    assert daily_df['qty'].iloc[0] == pnl_df['qty'].iloc[23], "Day 1 qty should be last value"
    assert daily_df['qty'].iloc[1] == pnl_df['qty'].iloc[47], "Day 2 qty should be last value"

    # Verify summed values (flow variables)
    assert daily_df['fill_qty'].iloc[0] == 0.1 * 24, "Day 1 fill_qty should be sum of 24 hours"
    assert daily_df['commission'].iloc[0] == 5.0 * 24, "Day 1 commission should be sum of 24 hours"
    assert daily_df['funding_income'].iloc[0] == 2.0 * 24, "Day 1 funding_income should be sum of 24 hours"

    # Verify pnl_gross is summed (it's a flow variable after our changes)
    assert daily_df['pnl_gross'].iloc[0] == 50.0 * 24, "Day 1 pnl_gross should be sum of 24 hours"
    assert daily_df['pnl_net'].iloc[0] == 45.0 * 24, "Day 1 pnl_net should be sum of 24 hours"

    # Verify cumulative funding income is last value
    assert daily_df['funding_income_cum'].iloc[0] == pnl_df['funding_income_cum'].iloc[23], "Day 1 funding_income_cum should be last value"

    # Verify returns are recalculated (not summed)
    expected_day1_pnl_gross_ret = daily_df['pnl_gross'].iloc[0] / daily_df['notional'].iloc[0]
    assert abs(daily_df['pnl_gross_ret'].iloc[0] - expected_day1_pnl_gross_ret) < 1e-6, "Day 1 return should be recalculated"

    print("\n✓ All aggregate_to_daily tests passed!")
    return 0


def test_aggregate_to_daily_assertion() -> int:
    """Test that aggregate_to_daily asserts on intervals > 1 day."""
    print("\n" + "=" * 80)
    print("Testing aggregate_to_daily assertion on large intervals")
    print("=" * 80)

    # Create data with > 1 day interval
    data = {
        'ts': [
            dt(2025, 10, 1, tzinfo=timezone.utc),
            dt(2025, 10, 3, tzinfo=timezone.utc)  # 2 day gap!
        ],
        'pnl_net': [100.0, 200.0],
        'notional': [10000.0, 20000.0]
    }
    pnl_df = pd.DataFrame(data).set_index('ts')

    # Should raise AssertionError
    try:
        aggregate_to_daily(pnl_df)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        print(f"\n✓ Correctly raised AssertionError: {e}")
        assert "exceeds 1 day maximum" in str(e)

    return 0


def test_calculate_performance_statistics() -> int:
    """Test calculate_performance_statistics function."""
    print("\n" + "=" * 80)
    print("Testing calculate_performance_statistics")
    print("=" * 80)

    # Create daily PnL data with known statistics
    n_days = 100
    dates = pd.date_range('2025-01-01', periods=n_days, freq='1D', tz=timezone.utc)

    # Create realistic returns with some volatility
    np.random.seed(42)
    daily_returns = np.random.normal(0.001, 0.02, n_days)  # 0.1% mean, 2% std

    notional = 1000000.0  # $1M notional
    balance = 500000.0  # $500k balance (leverage = 2x)
    daily_pnl = daily_returns * notional
    daily_traded = np.random.uniform(50000, 150000, n_days)  # $50k-$150k per day
    daily_funding = np.random.uniform(10, 50, n_days)  # $10-$50 funding per day
    commission = 100.0  # $100 commission per day

    data = {
        'ts': dates,
        'date': dates.date,
        'net_pnl': daily_pnl,
        'pnl_net': daily_pnl,
        'pnl_net_ret': daily_returns,
        'net_pnl_unlev_ret': daily_returns,
        'net_pnl_lev_ret': daily_returns * 2,  # 2x leverage
        'notional': [notional] * n_days,
        'gross_notional': [notional] * n_days,
        'balance': [balance] * n_days,
        'commission': [commission] * n_days,
        'fees_bps_daily': [(commission / notional) * 10000] * n_days,  # fees in bps
        'funding_income': daily_funding,
        'funding_income_bps_daily': (daily_funding / notional) * 10000,  # funding in bps
        'funding_income_cum': np.cumsum(daily_funding),
        'abs_dollars_cum': np.cumsum(daily_traded),
        'fill_dollars_abs': daily_traded,
        'turnover': (daily_traded / notional),  # turnover as ratio
        'fill_count': [10.0] * n_days
    }
    daily_pnl_df = pd.DataFrame(data).set_index('ts')

    print("\n1. Input daily data (first 10 rows):")
    print(daily_pnl_df.head(10).to_string())

    # Calculate statistics
    stats = calculate_performance_statistics(daily_pnl_df, pnl_col='pnl_net')

    print("\n2. Performance Statistics:")
    for key, value in stats.items():
        if 'pct' in key or 'return' in key:
            print(f"   {key}: {value:.4%}")
        else:
            print(f"   {key}: {value:,.2f}")

    # Verify calculations
    assert stats['total_days'] == n_days, f"Should have {n_days} days"
    assert abs(stats['mean_daily_return'] - daily_returns.mean()) < 1e-10, "Mean return should match"

    # pandas .std() uses ddof=1 by default (sample std)
    expected_std = daily_pnl_df['pnl_net_ret'].std()
    assert abs(stats['std_daily_return'] - expected_std) < 1e-10, "Std return should match pandas std"

    # Verify annualization (365 days)
    expected_ann_return = daily_pnl_df['pnl_net_ret'].mean() * 365
    expected_ann_std = daily_pnl_df['pnl_net_ret'].std() * np.sqrt(365)
    assert abs(stats['annualized_return'] - expected_ann_return) < 1e-10, "Annualized return should match"
    assert abs(stats['annualized_std'] - expected_ann_std) < 1e-10, "Annualized std should match"

    # Verify Sharpe ratio
    expected_sharpe = expected_ann_return / expected_ann_std
    if np.isinf(expected_sharpe):
        expected_sharpe = np.nan
    assert abs(stats['sharpe_ratio'] - expected_sharpe) < 1e-10, "Sharpe ratio should match"

    # Verify other statistics
    assert stats['avg_notional'] == notional, "Average notional should match"
    assert stats['avg_daily_commission'] == 100.0, "Average commission should be 100"
    assert stats['pct_positive_days'] == (daily_returns > 0).sum() / n_days, "Win rate should match"
    assert stats['avg_daily_fill_count'] == 10.0, "Average fill count should be 10"

    # Verify funding income statistics
    assert 'total_funding_income' in stats, "total_funding_income should be in stats"
    assert 'avg_daily_funding_income' in stats, "avg_daily_funding_income should be in stats"
    assert 'daily_funding_income_pct_notional' in stats, "daily_funding_income_pct_notional should be in stats"

    expected_total_funding = daily_pnl_df['funding_income_cum'].iloc[-1]
    expected_avg_funding = daily_pnl_df['funding_income'].mean()
    expected_funding_pct = (expected_avg_funding / notional) * 100

    assert abs(stats['total_funding_income'] - expected_total_funding) < 1e-6, "Total funding income should match"
    assert abs(stats['avg_daily_funding_income'] - expected_avg_funding) < 1e-6, "Average daily funding income should match"
    assert abs(stats['daily_funding_income_pct_notional'] - expected_funding_pct) < 1e-6, "Funding income % notional should match"

    print("\n3. Funding Income Statistics:")
    print(f"   Total funding income: ${stats['total_funding_income']:,.2f}")
    print(f"   Average daily funding income: ${stats['avg_daily_funding_income']:,.2f}")
    print(f"   Daily funding income % notional: {stats['daily_funding_income_pct_notional']:.4f}%")

    print("\n✓ All calculate_performance_statistics tests passed!")
    return 0


def test_calculate_performance_statistics_missing_column() -> int:
    """Test that calculate_performance_statistics raises error on missing column."""
    print("\n" + "=" * 80)
    print("Testing calculate_performance_statistics with missing column")
    print("=" * 80)

    # Create data without the required return column
    data = {
        'ts': pd.date_range('2025-10-01', periods=10, freq='1D', tz=timezone.utc),
        'pnl_net': [100.0] * 10,
        'notional': [10000.0] * 10
    }
    daily_pnl_df = pd.DataFrame(data).set_index('ts')

    # Should raise ValueError
    try:
        calculate_performance_statistics(daily_pnl_df, pnl_col='pnl_net')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"\n✓ Correctly raised ValueError: {e}")
        assert "pnl_net_ret" in str(e)

    return 0


def test_compute_commissions_missing_commission_asset() -> int:
    """Test that compute_commissions raises ValueError when commission_asset is missing."""
    print("\n" + "=" * 80)
    print("Testing compute_commissions with missing commission_asset column")
    print("=" * 80)

    # Create fills without commission_asset column
    fills_df = pd.DataFrame({
        'ts': pd.date_range('2025-10-01', periods=3, freq='1h', tz=timezone.utc),
        'symbol_venue': ['BTCUSDT_binance-futures'] * 3,
        'commission': [1.5, 2.0, 1.8],
    })

    bars_df = pd.DataFrame({
        'ts': pd.date_range('2025-10-01', periods=3, freq='1h', tz=timezone.utc),
        'symbol_venue': ['BTCUSDT_binance-futures'] * 3,
        'close_mid': [50000.0, 50100.0, 50200.0],
    }).set_index(['ts', 'symbol_venue'])

    # Should raise ValueError for missing commission_asset
    try:
        compute_commissions(fills_df, bars_df)
        assert False, "Should have raised ValueError for missing commission_asset"
    except ValueError as e:
        print(f"\n✓ Correctly raised ValueError: {e}")
        assert "commission_asset" in str(e)

    print("\n✓ Test passed: ValueError raised for missing commission_asset")
    return 0


def test_compute_commissions_none_bars() -> int:
    """Test that compute_commissions raises ValueError when bars_df is None."""
    print("\n" + "=" * 80)
    print("Testing compute_commissions with None bars_df")
    print("=" * 80)

    # Create fills with commission_asset and commission_raw (as created by data loader)
    fills_df = pd.DataFrame({
        'ts': pd.date_range('2025-10-01', periods=3, freq='1h', tz=timezone.utc),
        'symbol_venue': ['BTCUSDT_binance-futures'] * 3,
        'commission_asset': ['BNB'] * 3,
        'commission_raw': [1.5, 2.0, 1.8],
    })

    # Should raise ValueError for None bars_df (data loading failure)
    try:
        compute_commissions(fills_df, None)
        assert False, "Should have raised ValueError for None bars_df"
    except ValueError as e:
        print(f"\n✓ Correctly raised ValueError: {e}")
        assert "bars_df is None" in str(e)

    print("\n✓ Test passed: ValueError raised when bars_df is None")
    return 0


def test_compute_commissions_empty_bars() -> int:
    """Test that compute_commissions raises ValueError when bars_df is empty."""
    print("\n" + "=" * 80)
    print("Testing compute_commissions with empty bars_df")
    print("=" * 80)

    fills_df = pd.DataFrame({
        'ts': pd.date_range('2025-10-01', periods=3, freq='1h', tz=timezone.utc),
        'symbol_venue': ['BTCUSDT_binance-futures'] * 3,
        'commission_asset': ['BNB'] * 3,
        'commission_raw': [1.5, 2.0, 1.8],
    })

    empty_bars_df = pd.DataFrame(columns=['ts', 'symbol_venue', 'close_mid']).set_index(['ts', 'symbol_venue'])

    # Should raise ValueError for empty bars_df (data loading failure)
    try:
        compute_commissions(fills_df, empty_bars_df)
        assert False, "Should have raised ValueError for empty bars_df"
    except ValueError as e:
        print(f"\n✓ Correctly raised ValueError: {e}")
        assert "bars_df is empty" in str(e)

    print("\n✓ Test passed: ValueError raised when bars_df is empty")
    return 0


def test_compute_commissions_success() -> int:
    """Test that compute_commissions correctly computes commissions with price data."""
    print("\n" + "=" * 80)
    print("Testing compute_commissions with valid data")
    print("=" * 80)

    # Create fills with BNB commission (using commission_raw as created by data loader)
    fills_df = pd.DataFrame({
        'ts': pd.date_range('2025-10-01 10:00', periods=3, freq='1h', tz=timezone.utc),
        'symbol_venue': ['BTCUSDT_binance-futures'] * 3,
        'commission_asset': ['BNB'] * 3,
        'commission_raw': [0.01, 0.02, 0.015],  # Commission in BNB (raw)
    })

    # Create price data for BNBUSDT_binance-futures
    bars_df = pd.DataFrame({
        'ts': pd.date_range('2025-10-01 10:00', periods=3, freq='1h', tz=timezone.utc),
        'symbol_venue': ['BNBUSDT_binance-futures'] * 3,
        'close_mid': [600.0, 610.0, 605.0],  # BNB prices in USDT
    }).set_index(['ts', 'symbol_venue'])

    print("\n1. Input fills_df:")
    print(fills_df.to_string())

    print("\n2. Input bars_df (BNB prices):")
    print(bars_df.to_string())

    result_df = compute_commissions(fills_df.copy(), bars_df)

    print("\n3. Result with commission in USD:")
    print(result_df[['ts', 'commission']].to_string())

    # Verify commissions were converted
    expected_commissions = [0.01 * 600.0, 0.02 * 610.0, 0.015 * 605.0]
    for i, expected in enumerate(expected_commissions):
        actual = result_df['commission'].iloc[i]
        assert abs(actual - expected) < 1e-6, f"Commission {i} should be {expected}, got {actual}"

    print(f"\n✓ Commission 0: 0.01 BNB × $600 = ${result_df['commission'].iloc[0]:.2f}")
    print(f"✓ Commission 1: 0.02 BNB × $610 = ${result_df['commission'].iloc[1]:.2f}")
    print(f"✓ Commission 2: 0.015 BNB × $605 = ${result_df['commission'].iloc[2]:.2f}")

    print("\n✓ Test passed: Commissions computed correctly")
    return 0


def test_calculate_performance_by_month() -> int:
    """Test calculate_performance_by_month correctly assigns dates to months.

    This test verifies that midnight timestamps are correctly attributed to their
    actual month, not shifted to the previous month. For example, data with
    ts=2026-02-01 00:00:00 should be counted as February, not January.
    """
    print("\n" + "=" * 80)
    print("Testing calculate_performance_by_month date attribution")
    print("=" * 80)

    # Create sample data spanning month boundary
    # ts at midnight should be attributed to that day's month, not previous month
    data = {
        'ts': [
            pd.Timestamp('2026-01-30 00:00:00', tz=timezone.utc),
            pd.Timestamp('2026-01-31 00:00:00', tz=timezone.utc),
            pd.Timestamp('2026-02-01 00:00:00', tz=timezone.utc),  # Should be Feb, not Jan
            pd.Timestamp('2026-02-02 00:00:00', tz=timezone.utc),
        ],
        'net_pnl': [1000.0, 2000.0, -500.0, 1500.0],
        'gross_notional': [100000.0, 100000.0, 100000.0, 100000.0],
        'balance': [50000.0, 52000.0, 51500.0, 53000.0],
        'commission': [10.0, 10.0, 10.0, 10.0],
        'funding_income': [5.0, 5.0, 5.0, 5.0],
        'logret_cum_wgtmkt': [0.00, 0.01, 0.03, 0.02],  # Cumulative market log return
    }
    df = pd.DataFrame(data)

    print("\n1. Input data:")
    print(df[['ts', 'net_pnl']].to_string())

    # Call the function
    result_df = calculate_performance_by_month(df)

    print("\n2. Monthly aggregation result:")
    print(result_df[['year_month', 'net_pnl_sum']].to_string())

    # Verify January has only Jan 30 and Jan 31 data
    jan_row = result_df[result_df['year_month'] == '2026-01']
    assert len(jan_row) == 1, "Should have one row for January"
    jan_pnl = jan_row['net_pnl_sum'].iloc[0]
    expected_jan_pnl = 1000.0 + 2000.0  # Jan 30 + Jan 31
    assert abs(jan_pnl - expected_jan_pnl) < 1e-6, f"January PnL should be {expected_jan_pnl}, got {jan_pnl}"

    # Verify February has Feb 1 and Feb 2 data (NOT shifted to January)
    feb_row = result_df[result_df['year_month'] == '2026-02']
    assert len(feb_row) == 1, "Should have one row for February"
    feb_pnl = feb_row['net_pnl_sum'].iloc[0]
    expected_feb_pnl = -500.0 + 1500.0  # Feb 1 + Feb 2
    assert abs(feb_pnl - expected_feb_pnl) < 1e-6, f"February PnL should be {expected_feb_pnl}, got {feb_pnl}"

    print(f"\n✓ January PnL: ${jan_pnl:,.2f} (Jan 30 + Jan 31)")
    print(f"✓ February PnL: ${feb_pnl:,.2f} (Feb 1 + Feb 2)")

    # Verify market return calculation
    # January: logret_cum_wgtmkt goes from 0.00 to 0.01, so log_return = 0.01
    jan_mkt_ret = jan_row['market_return'].iloc[0]
    expected_jan_mkt_ret = np.exp(0.01 - 0.00) - 1  # ~0.01005
    assert abs(jan_mkt_ret - expected_jan_mkt_ret) < 1e-6, f"January market return should be {expected_jan_mkt_ret}, got {jan_mkt_ret}"

    # February: logret_cum_wgtmkt goes from 0.03 to 0.02 (first=0.03, last=0.02), so log_return = -0.01
    feb_mkt_ret = feb_row['market_return'].iloc[0]
    expected_feb_mkt_ret = np.exp(0.02 - 0.03) - 1  # ~-0.00995
    assert abs(feb_mkt_ret - expected_feb_mkt_ret) < 1e-6, f"February market return should be {expected_feb_mkt_ret}, got {feb_mkt_ret}"

    print(f"✓ January market return: {jan_mkt_ret:.4%}")
    print(f"✓ February market return: {feb_mkt_ret:.4%}")
    print("\n✓ Test passed: Midnight timestamps correctly attributed to their month!")
    return 0


def test_calculate_performance_by_month_empty_df() -> int:
    """Test calculate_performance_by_month handles empty dataframe gracefully."""
    print("\n" + "=" * 80)
    print("Testing calculate_performance_by_month with empty dataframe")
    print("=" * 80)

    # Create empty dataframe with required columns and proper dtypes
    df = pd.DataFrame({
        'ts': pd.Series([], dtype='datetime64[ns, UTC]'),
        'net_pnl': pd.Series([], dtype='float64'),
        'gross_notional': pd.Series([], dtype='float64'),
        'balance': pd.Series([], dtype='float64'),
        'commission': pd.Series([], dtype='float64'),
        'funding_income': pd.Series([], dtype='float64'),
        'logret_cum_wgtmkt': pd.Series([], dtype='float64'),
    })

    print("\n1. Input: Empty dataframe with proper dtypes")

    # Call the function - should not raise error
    result_df = calculate_performance_by_month(df)

    print(f"\n2. Result shape: {result_df.shape}")

    # Verify result is empty but has expected columns
    assert len(result_df) == 0, "Result should be empty"
    assert 'year_month' in result_df.columns, "Should have year_month column"

    print("\n✓ Test passed: Empty dataframe handled gracefully!")
    return 0


def test_calculate_performance_statistics_empty_df() -> int:
    """Test calculate_performance_statistics raises appropriate error on empty dataframe.

    This test verifies that passing an empty dataframe results in a ValueError
    for missing return column, which is why the caller should check for empty
    before calling.
    """
    print("\n" + "=" * 80)
    print("Testing calculate_performance_statistics with empty dataframe")
    print("=" * 80)

    # Create empty dataframe with required columns
    df = pd.DataFrame({
        'date': pd.Series([], dtype='datetime64[ns]'),
        'net_pnl': pd.Series([], dtype='float64'),
        'gross_notional': pd.Series([], dtype='float64'),
        'balance': pd.Series([], dtype='float64'),
        'commission': pd.Series([], dtype='float64'),
        'funding_income': pd.Series([], dtype='float64'),
        'fill_dollars_abs': pd.Series([], dtype='float64'),
    })

    print("\n1. Input: Empty dataframe")

    # This should raise ValueError for missing return column - caller must check for empty first
    try:
        calculate_performance_statistics(df, pnl_col='net_pnl')
        print("\n✗ Expected error was not raised!")
        return 1
    except (IndexError, KeyError, ValueError) as e:
        print(f"\n2. Expected error raised: {type(e).__name__}")
        print(f"   Message: {str(e)[:80]}...")

    print("\n✓ Test passed: Empty dataframe correctly raises error (caller must check)!")
    return 0


def run_all_tests() -> int:
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RUNNING ALL PNL_UTIL TESTS")
    print("=" * 80)

    result = 0
    result |= test_calc_pnl_returns()
    result |= test_aggregate_to_daily()
    result |= test_aggregate_to_daily_assertion()
    result |= test_calculate_performance_statistics()
    result |= test_calculate_performance_statistics_missing_column()
    result |= test_calculate_performance_by_month()
    result |= test_calculate_performance_by_month_empty_df()
    result |= test_calculate_performance_statistics_empty_df()
    result |= test_compute_commissions_missing_commission_asset()
    result |= test_compute_commissions_none_bars()
    result |= test_compute_commissions_empty_bars()
    result |= test_compute_commissions_success()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    return result


if __name__ == "__main__":
    sys.exit(run_all_tests())
