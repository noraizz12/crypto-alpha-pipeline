#!/usr/bin/env python3
"""Unit test for SecurityPnl class."""

import sys
from datetime import datetime as dt, timezone, time
import pandas as pd
from lib.pnl_new.security_pnl import SecurityPnl

def test_security_pnl():
    """Test SecurityPnl calculation with simple example."""

    print("=" * 80)
    print("Testing SecurityPnl")
    print("=" * 80)

    # Create simple test data
    symbol_venue = "BTCUSDT_binance-futures"
    start_dt = dt(2025, 10, 1, tzinfo=timezone.utc)
    end_dt = dt(2025, 10, 2, tzinfo=timezone.utc)

    # Test fills: Buy 10 @ 50000, then Buy 5 @ 51000, then Sell 8 @ 52000
    # Note: Sell qty should be negative
    fills_df = pd.DataFrame([
        {
            'ts': dt(2025, 10, 1, 10, 0, tzinfo=timezone.utc),
            'side': 'B',
            'fill_px': 50000.0,
            'fill_qty': 10.0,
            'commission': 2.5,
            'realized_pnl': 0.0
        },
        {
            'ts': dt(2025, 10, 1, 12, 0, tzinfo=timezone.utc),
            'side': 'B',
            'fill_px': 51000.0,
            'fill_qty': 5.0,
            'commission': 1.275,
            'realized_pnl': 0.0
        },
        {
            'ts': dt(2025, 10, 1, 14, 0, tzinfo=timezone.utc),
            'side': 'S',
            'fill_px': 52000.0,
            'fill_qty': -8.0,  # Negative for sell
            'commission': 2.08,
            'realized_pnl': 0.0
        }
    ])

    # Test funding income: receive funding at 12:00
    fundings_df = pd.DataFrame([
        {
            'ts': dt(2025, 10, 1, 12, 0, tzinfo=timezone.utc),
            'symbol_venue': symbol_venue,
            'funding_income': 25.50  # Positive = received
        }
    ])

    # Test bars: prices at different times
    bars_df = pd.DataFrame([
        {'ts': dt(2025, 10, 1, 10, 0, tzinfo=timezone.utc), 'close_mid': 50000.0, 'volume': 100.0, 'dvolume': 5000000.0},
        {'ts': dt(2025, 10, 1, 12, 0, tzinfo=timezone.utc), 'close_mid': 51000.0, 'volume': 100.0, 'dvolume': 5100000.0},
        {'ts': dt(2025, 10, 1, 14, 0, tzinfo=timezone.utc), 'close_mid': 52000.0, 'volume': 100.0, 'dvolume': 5200000.0},
    ])
    bars_df = bars_df.set_index('ts')

    # Initial position: 0 qty, 0 cost
    initial_position = {'qty': 0.0, 'cost_basis': 0.0, 'value': 0.0}

    # No pnl_times (calculate every minute)
    pnl_times = None

    print("\n1. Test Data:")
    print(f"   Symbol: {symbol_venue}")
    print(f"   Initial position: {initial_position}")
    print(f"\n   Fills:")
    print(fills_df.to_string(index=False))
    print(f"\n   Fundings:")
    print(fundings_df.to_string(index=False))
    print(f"\n   Bars:")
    print(bars_df.to_string())

    # Create SecurityPnl and calculate
    print("\n2. Running PnL Calculation...")
    sec_pnl = SecurityPnl(
        symbol_venue=symbol_venue,
        start_dt=start_dt,
        end_dt=end_dt,
        fills_df=fills_df,
        bars_df=bars_df,
        fundings_df=fundings_df,
        initial_position=initial_position,
        pnl_times=pnl_times
    )

    pnl_df = sec_pnl.calculate_pnl()

    print("\n3. PnL Results:")
    print(pnl_df.to_string(index=False))

    # Verify calculations
    print("\n4. Verification:")

    # Row 1: Buy 10 @ 50000
    row1 = pnl_df.iloc[0]
    print(f"\n   After buy 10 @ 50000:")
    print(f"   - Qty: {row1['qty']} (expected: 10)")
    print(f"   - Cost cum: {row1['cost_cum']} (expected: 500000)")
    print(f"   - Cost basis: {row1['cost_basis']} (expected: 50000)")
    print(f"   - Position: {row1['position']} (expected: 500000)")
    print(f"   - PnL gross: {row1['pnl_gross']} (expected: 0)")
    print(f"   - Commission cum: {row1['commission_cum']} (expected: 2.5)")
    print(f"   - PnL net: {row1['pnl_net']} (expected: -2.5)")

    # Row 2: Buy 5 @ 51000 (total 15 @ avg 50333.33) + Funding 25.50
    row2 = pnl_df.iloc[1]
    print(f"\n   After buy 5 @ 51000 + funding 25.50:")
    print(f"   - Qty: {row2['qty']} (expected: 15)")
    print(f"   - Cost cum: {row2['cost_cum']} (expected: 755000)")
    print(f"   - Cost basis: {row2['cost_basis']:.2f} (expected: 50333.33)")
    print(f"   - Position: {row2['position']} (expected: 765000)")
    print(f"   - PnL gross cum: {row2['pnl_gross_cum']} (expected: 10000)")
    print(f"   - Commission cum: {row2['commission_cum']} (expected: 3.775)")
    print(f"   - Funding income cum: {row2['funding_income_cum']} (expected: 25.50)")
    print(f"   - PnL net cum: {row2['pnl_net_cum']:.2f} (expected: 10021.725 = 10000 - 3.775 + 25.50)")

    # Row 3: Sell 8 @ 52000 (remaining 7)
    row3 = pnl_df.iloc[2]
    print(f"\n   After sell 8 @ 52000:")
    print(f"   - Qty: {row3['qty']} (expected: 7)")
    print(f"   - Cost cum: {row3['cost_cum']} (expected: 339000)")
    print(f"   - Cost basis: {row3['cost_basis']:.2f} (expected: 48428.57)")
    print(f"   - Position: {row3['position']} (expected: 364000)")
    print(f"   - PnL gross cum: {row3['pnl_gross_cum']} (expected: 25000)")
    print(f"   - Commission cum: {row3['commission_cum']} (expected: 5.855)")
    print(f"   - Funding income cum: {row3['funding_income_cum']} (expected: 25.50)")
    print(f"   - PnL net cum: {row3['pnl_net_cum']:.2f} (expected: 25019.645 = 25000 - 5.855 + 25.50)")

    # Verify funding income is included in net PnL
    assert 'funding_income' in pnl_df.columns, "funding_income column should exist"
    assert 'funding_income_cum' in pnl_df.columns, "funding_income_cum column should exist"
    assert row2['funding_income'] == 25.50, "Row 2 should have funding income"
    assert row2['funding_income_cum'] == 25.50, "Row 2 cumulative funding should be 25.50"
    assert row3['funding_income_cum'] == 25.50, "Row 3 cumulative funding should still be 25.50"

    # Verify net PnL formula: gross - commissions + funding
    expected_net_cum_row2 = row2['pnl_gross_cum'] - row2['commission_cum'] + row2['funding_income_cum']
    assert abs(row2['pnl_net_cum'] - expected_net_cum_row2) < 0.01, "Net PnL should include funding income"

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    sys.exit(test_security_pnl())
