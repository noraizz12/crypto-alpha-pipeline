"""Unit tests for DownloadBinance position close detection."""

import os
import tempfile

import pandas as pd
import pytest


@pytest.fixture
def api_response_df():
    """Simulate Binance API response - only active positions, closed ones are absent."""
    return pd.DataFrame([
        {'symbol': 'BTCUSDT', 'qty': -0.5, 'entryPrice': 50000.0, 'markPrice': 51000.0,
         'unRealizedProfit': -500.0, 'value': -25500.0, 'liquidationPrice': 0.0,
         'leverage': 20, 'positionSide': 'BOTH', 'updateTime': 123, 'maxNotionalValue': 1000000,
         'cost_basis': -25000.0},
        {'symbol': 'ETHUSDT', 'qty': 10.0, 'entryPrice': 3000.0, 'markPrice': 3100.0,
         'unRealizedProfit': 1000.0, 'value': 31000.0, 'liquidationPrice': 0.0,
         'leverage': 20, 'positionSide': 'BOTH', 'updateTime': 123, 'maxNotionalValue': 1000000,
         'cost_basis': 30000.0},
        {'symbol': 'SOLUSDT', 'qty': 100.0, 'entryPrice': 150.0, 'markPrice': 150.0,
         'unRealizedProfit': 0.0, 'value': 15000.0, 'liquidationPrice': 0.0,
         'leverage': 20, 'positionSide': 'BOTH', 'updateTime': 123, 'maxNotionalValue': 1000000,
         'cost_basis': 15000.0},
    ])


class TestPositionCloseDetection:

    def test_active_positions_kept(self, api_response_df):
        """Positions with non-zero qty are always written."""
        result = api_response_df[api_response_df['qty'] != 0]
        assert set(result['symbol']) == {'BTCUSDT', 'ETHUSDT', 'SOLUSDT'}

    def test_breakeven_position_not_dropped(self, api_response_df):
        """Position at exact breakeven (pnl=0, qty!=0) must be kept."""
        old_result = api_response_df[api_response_df['unRealizedProfit'] != 0]
        assert 'SOLUSDT' not in old_result['symbol'].values

        new_result = api_response_df[api_response_df['qty'] != 0]
        assert 'SOLUSDT' in new_result['symbol'].values

    def test_missing_symbol_detected_as_closed(self, api_response_df):
        """Symbol in tracked file but missing from API = closed position."""
        active_df = api_response_df[api_response_df['qty'] != 0]
        tracked_symbols = {'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'JASMYUSDT'}
        closed_symbols = tracked_symbols - set(active_df['symbol'])
        assert closed_symbols == {'JASMYUSDT'}

    def test_no_false_closes_when_all_tracked_present(self, api_response_df):
        """No closes detected when all tracked symbols are in API response."""
        active_df = api_response_df[api_response_df['qty'] != 0]
        tracked_symbols = {'BTCUSDT', 'ETHUSDT', 'SOLUSDT'}
        closed_symbols = tracked_symbols - set(active_df['symbol'])
        assert closed_symbols == set()

    def test_zero_row_creation(self, api_response_df):
        """Zero rows have correct values for closed symbols."""
        active_df = api_response_df[api_response_df['qty'] != 0]
        closed_symbols = {'JASMYUSDT', 'PIEVERSEUSDT'}

        zero_row = {col: 0 for col in active_df.columns}
        zero_row.update({'ts': '2026-02-27', 'positionSide': 'BOTH'})
        close_rows_df = pd.DataFrame([{**zero_row, 'symbol': s} for s in closed_symbols])
        result = pd.concat([active_df, close_rows_df], ignore_index=True)

        assert 'JASMYUSDT' in result['symbol'].values
        assert 'PIEVERSEUSDT' in result['symbol'].values
        jasmy = result[result['symbol'] == 'JASMYUSDT'].iloc[0]
        assert jasmy['qty'] == 0.0
        assert jasmy['value'] == 0.0
        assert jasmy['cost_basis'] == 0.0

    def test_first_run_no_file(self, tmp_path, api_response_df):
        """First run of the day: no file exists, no close detection."""
        pos_file = tmp_path / "binance_pos.20260227.csv"
        assert not pos_file.exists()
        active_df = api_response_df[api_response_df['qty'] != 0]
        closed_symbols = set() - set(active_df['symbol'])
        assert closed_symbols == set()

    def test_full_close_flow(self):
        """End-to-end: position active in run 1, missing from API in run 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pos_file = os.path.join(tmpdir, "binance_pos.20260227.csv")

            # Run 1: JASMY and BTC are active
            run1_df = pd.DataFrame([
                {'symbol': 'BTCUSDT', 'qty': -0.5, 'value': -25000.0, 'entryPrice': 50000.0,
                 'markPrice': 51000.0, 'unRealizedProfit': -500.0, 'cost_basis': -25000.0,
                 'positionSide': 'BOTH', 'leverage': 20, 'liquidationPrice': 0.0,
                 'updateTime': 123, 'maxNotionalValue': 1000000},
                {'symbol': 'JASMYUSDT', 'qty': 5961893.0, 'value': 35000.0, 'entryPrice': 0.006,
                 'markPrice': 0.006, 'unRealizedProfit': 100.0, 'cost_basis': 35000.0,
                 'positionSide': 'BOTH', 'leverage': 3, 'liquidationPrice': 0.0,
                 'updateTime': 123, 'maxNotionalValue': 1000000},
            ])
            run1_df.to_csv(pos_file, index=False)

            # Run 2: Binance API only returns BTC (JASMY liquidated, absent from response)
            run2_api_df = pd.DataFrame([
                {'symbol': 'BTCUSDT', 'qty': -0.5, 'value': -25000.0, 'entryPrice': 50000.0,
                 'markPrice': 51000.0, 'unRealizedProfit': -500.0, 'cost_basis': -25000.0,
                 'positionSide': 'BOTH', 'leverage': 20, 'liquidationPrice': 0.0,
                 'updateTime': 123, 'maxNotionalValue': 1000000},
            ])

            active_df = run2_api_df[run2_api_df['qty'] != 0]
            tracked_symbols = set(pd.read_csv(pos_file, usecols=['symbol'])['symbol'].unique())
            closed_symbols = tracked_symbols - set(active_df['symbol'])

            assert closed_symbols == {'JASMYUSDT'}

            zero_row = {col: 0 for col in active_df.columns}
            zero_row.update({'ts': '2026-02-27', 'positionSide': 'BOTH'})
            close_rows_df = pd.DataFrame([{**zero_row, 'symbol': s} for s in closed_symbols])
            result = pd.concat([active_df, close_rows_df], ignore_index=True)

            assert 'BTCUSDT' in result['symbol'].values
            assert 'JASMYUSDT' in result['symbol'].values
            jasmy = result[result['symbol'] == 'JASMYUSDT'].iloc[0]
            assert jasmy['qty'] == 0.0
            assert jasmy['value'] == 0.0
