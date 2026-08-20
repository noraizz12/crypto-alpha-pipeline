#!/usr/bin/env python3
"""Unit test for Pnl class."""

import sys
from datetime import datetime as dt, timezone
import pandas as pd
from lib.pnl_new.pnl import Pnl
from lib.pnl_new.pnl_util import calc_pnl_returns

def test_pnl():
    """Test Pnl calculation with multiple securities."""

    print("=" * 80)
    print("Testing Pnl")
    print("=" * 80)

    # Test parameters
    start_dt = dt(2025, 10, 1, tzinfo=timezone.utc)
    end_dt = dt(2025, 10, 2, tzinfo=timezone.utc)

    # Test fills: Two securities with trades
    # BTC: Buy 10 @ 50000, then Sell 5 @ 52000
    # ETH: Buy 20 @ 3000, then Buy 10 @ 3100
    fills_df = pd.DataFrame([
        {
            'ts': dt(2025, 10, 1, 10, 0, tzinfo=timezone.utc),
            'date': pd.Timestamp('2025-10-01').normalize(),
            'symbol_venue': 'BTCUSDT_binance-futures',
            'side': 'B',
            'fill_px': 50000.0,
            'fill_qty': 10.0,
            'commission': 2.5,
            'realized_pnl': 0.0
        },
        {
            'ts': dt(2025, 10, 1, 12, 0, tzinfo=timezone.utc),
            'date': pd.Timestamp('2025-10-01').normalize(),
            'symbol_venue': 'BTCUSDT_binance-futures',
            'side': 'S',
            'fill_px': 52000.0,
            'fill_qty': -5.0,
            'commission': 1.3,
            'realized_pnl': 0.0
        },
        {
            'ts': dt(2025, 10, 1, 10, 30, tzinfo=timezone.utc),
            'date': pd.Timestamp('2025-10-01').normalize(),
            'symbol_venue': 'ETHUSDT_binance-futures',
            'side': 'B',
            'fill_px': 3000.0,
            'fill_qty': 20.0,
            'commission': 0.3,
            'realized_pnl': 0.0
        },
        {
            'ts': dt(2025, 10, 1, 13, 0, tzinfo=timezone.utc),
            'date': pd.Timestamp('2025-10-01').normalize(),
            'symbol_venue': 'ETHUSDT_binance-futures',
            'side': 'B',
            'fill_px': 3100.0,
            'fill_qty': 10.0,
            'commission': 0.155,
            'realized_pnl': 0.0
        }
    ])

    # Test funding income
    fundings_df = pd.DataFrame([
        {
            'ts': dt(2025, 10, 1, 12, 0, tzinfo=timezone.utc),
            'symbol_venue': 'BTCUSDT_binance-futures',
            'funding_income': 15.25
        },
        {
            'ts': dt(2025, 10, 1, 13, 0, tzinfo=timezone.utc),
            'symbol_venue': 'ETHUSDT_binance-futures',
            'funding_income': 8.75
        }
    ])

    # Test bars: prices at different times for both securities
    bars_df = pd.DataFrame([
        {'ts': dt(2025, 10, 1, 10, 0, tzinfo=timezone.utc), 'symbol_venue': 'BTCUSDT_binance-futures',
         'close_mid': 50000.0, 'volume': 100.0, 'dvolume': 5000000.0},
        {'ts': dt(2025, 10, 1, 12, 0, tzinfo=timezone.utc), 'symbol_venue': 'BTCUSDT_binance-futures',
         'close_mid': 52000.0, 'volume': 100.0, 'dvolume': 5200000.0},
        {'ts': dt(2025, 10, 1, 10, 30, tzinfo=timezone.utc), 'symbol_venue': 'ETHUSDT_binance-futures',
         'close_mid': 3000.0, 'volume': 200.0, 'dvolume': 600000.0},
        {'ts': dt(2025, 10, 1, 13, 0, tzinfo=timezone.utc), 'symbol_venue': 'ETHUSDT_binance-futures',
         'close_mid': 3100.0, 'volume': 200.0, 'dvolume': 620000.0},
    ])
    bars_df = bars_df.set_index(['ts', 'symbol_venue'])

    # Initial positions: BTC has existing position, ETH starts at 0
    initial_positions = {
        'BTCUSDT_binance-futures': {'qty': 2.0, 'cost_basis': 49000.0, 'value': 100000.0}
    }

    print("\n1. Test Data:")
    print(f"   Initial positions: {initial_positions}")
    print(f"\n   Fills:")
    print(fills_df.to_string(index=False))
    print(f"\n   Fundings:")
    print(fundings_df.to_string(index=False))
    print(f"\n   Bars:")
    print(bars_df.to_string())

    # Minimal config for testing
    config = {
        'DYNAMIC_UNIVERSE': False
    }

    # Create Pnl with test data directly
    print("\n2. Running PnL Calculation...")
    pnl = Pnl(
        config=config,
        start=start_dt,
        end=end_dt,
        initial_positions=initial_positions,
        pnl_times=None,
        fills_df=fills_df,
        bars_df=bars_df,
        fundings_df=fundings_df,
        symbol_venues=['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
    )

    pnl_df = pnl.calculate(pool_size=2)

    print("\n3. Security PnL Results:")
    print(pnl_df.to_string())

    # Verify security-level calculations
    print("\n4. Verification - Security Level:")

    # BTC: Initial 2 @ 49000, Buy 10 @ 50000 (total 12 @ avg 49833.33), Sell 5 @ 52000 (remaining 7)
    btc_pnl = pnl_df.xs('BTCUSDT_binance-futures', level='symbol_venue')
    btc_row2 = btc_pnl.iloc[-1]  # After sell
    print(f"\n   BTC after sell 5 @ 52000:")
    print(f"   - Qty: {btc_row2['qty']} (expected: 7)")
    print(f"   - Cost cum: {btc_row2['cost_cum']:.2f}")
    print(f"   - Cost basis: {btc_row2['cost_basis']:.2f}")
    print(f"   - Position: {btc_row2['position']:.2f}")
    print(f"   - PnL gross: {btc_row2['pnl_gross']:.2f}")
    print(f"   - PnL net: {btc_row2['pnl_net']:.2f}")

    # ETH: Buy 20 @ 3000, Buy 10 @ 3100 (total 30 @ avg 3033.33)
    eth_pnl = pnl_df.xs('ETHUSDT_binance-futures', level='symbol_venue')
    eth_row2 = eth_pnl.iloc[-1]  # After second buy
    print(f"\n   ETH after buy 10 @ 3100:")
    print(f"   - Qty: {eth_row2['qty']} (expected: 30)")
    print(f"   - Cost cum: {eth_row2['cost_cum']:.2f} (expected: 91000)")
    print(f"   - Cost basis: {eth_row2['cost_basis']:.2f} (expected: 3033.33)")
    print(f"   - Position: {eth_row2['position']:.2f} (expected: 93000)")
    print(f"   - PnL gross: {eth_row2['pnl_gross']:.2f} (expected: 2000)")

    # Test aggregate PnL
    print("\n5. Portfolio-Level Aggregation:")
    agg_pnl_df = pnl.aggregate_pnl()
    print(agg_pnl_df.to_string())

    # Verify aggregation
    print("\n6. Verification - Portfolio Level:")
    last_agg = agg_pnl_df.iloc[-1]
    print(f"\n   Final portfolio metrics:")
    print(f"   - PnL net cum: {last_agg['pnl_net_cum']:.2f}")
    print(f"   - PnL gross cum: {last_agg['pnl_gross_cum']:.2f}")
    print(f"   - Commission cum: {last_agg['commission_cum']:.3f}")
    print(f"   - Funding income cum: {last_agg['funding_income_cum']:.2f} (expected: 24.00 = 15.25 + 8.75)")
    print(f"   - Realized PnL cum: {last_agg['realized_pnl_cum']:.2f}")
    print(f"   - Unrealized PnL cum: {last_agg['unrealized_pnl_cum']:.2f}")
    print(f"   - Notional: {last_agg['notional']:.2f}")
    print(f"   - Long value: {last_agg['long_value']:.2f}")
    print(f"   - Short value: {last_agg['short_value']:.2f}")

    # Verify funding income is aggregated correctly
    assert 'funding_income_cum' in agg_pnl_df.columns, "funding_income_cum should exist in aggregated PnL"
    # At 12:00, only BTC has funding (15.25)
    # At 13:00, only ETH has funding (8.75), cumulative is 24.0
    # The cumulative at each timestamp shows the total funding up to that time
    row_12 = agg_pnl_df.loc[agg_pnl_df.index == dt(2025, 10, 1, 12, 0, tzinfo=timezone.utc)].iloc[0]
    row_13 = agg_pnl_df.loc[agg_pnl_df.index == dt(2025, 10, 1, 13, 0, tzinfo=timezone.utc)].iloc[0]
    assert abs(row_12['funding_income_cum'] - 15.25) < 0.01, "12:00 should have cumulative BTC funding of 15.25"
    assert abs(row_13['funding_income_cum'] - 24.0) < 0.01, "13:00 should have cumulative funding of 24.0 (15.25 + 8.75)"

    # Test calc_pnl_returns
    print("\n7. Calculate PnL Returns:")
    stats_df = calc_pnl_returns(agg_pnl_df)
    print(stats_df[['pnl_net', 'notional', 'pnl_net_ret']].to_string())

    print("\n   Statistics verification:")
    last_stats = stats_df.iloc[-1]
    expected_return = last_stats['pnl_net'] / last_stats['notional']
    print(f"   - PnL return: {last_stats['pnl_net_ret']:.6f} (expected: {expected_return:.6f})")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    sys.exit(test_pnl())
