#!/usr/bin/env python3
"""Unit tests for residualized returns to ensure they sum to zero."""

import pytest
import pandas as pd
import numpy as np
from lib.calcs.calc_returns import calculate_resid_return, calculate_weighted_resid_return
from lib.calcs.calc_util import make_market_weight_col


class TestResidualizedReturns:
    """Test that residualized returns have the correct mathematical properties."""
    
    def test_equal_weighted_residuals_sum_to_zero(self):
        """Test that equal-weighted residualized returns sum to zero at each timestamp."""
        # Create sample data with multiple timestamps
        timestamps = pd.date_range('2023-01-01', periods=5, freq='h', tz='UTC')
        symbols = ['BTCUSDT_binance', 'ETHUSDT_binance', 'BNBUSDT_binance', 'ADAUSDT_binance']
        
        # Create test data
        data = []
        for ts in timestamps:
            for symbol in symbols:
                # Random returns between -5% and 5%
                data.append({
                    'ts': ts,
                    'symbol_venue': symbol,
                    'logret': np.random.uniform(-0.05, 0.05),
                    'fittable': True
                })
        
        df = pd.DataFrame(data).set_index(['ts', 'symbol_venue'])
        
        # Calculate residualized returns
        resid_df, _ = calculate_resid_return(df, filter_col='fittable', ret_col='logret')
        
        # Check that residuals sum to approximately zero at each timestamp
        for ts in timestamps:
            ts_data = resid_df.loc[ts]
            # Only check fittable symbols
            fittable_data = ts_data[ts_data['fittable']]
            residual_sum = fittable_data['logret_resid_eqmkt'].sum()
            
            # Should be very close to zero (within floating point precision)
            assert abs(residual_sum) < 1e-10, f"Residuals at {ts} sum to {residual_sum}, expected ~0"
    
    def test_equal_weighted_residuals_with_non_fittable(self):
        """Test residualized returns when some symbols are not fittable."""
        # Create sample data
        ts = pd.Timestamp('2023-01-01 12:00:00', tz='UTC')
        data = pd.DataFrame([
            {'ts': ts, 'symbol_venue': 'BTCUSDT_binance', 'logret': 0.01, 'fittable': True},
            {'ts': ts, 'symbol_venue': 'ETHUSDT_binance', 'logret': 0.02, 'fittable': True},
            {'ts': ts, 'symbol_venue': 'BNBUSDT_binance', 'logret': -0.01, 'fittable': True},
            {'ts': ts, 'symbol_venue': 'LOWLIQ_binance', 'logret': 0.10, 'fittable': False},  # Not included
        ]).set_index(['ts', 'symbol_venue'])
        
        # Calculate residualized returns
        resid_df, _ = calculate_resid_return(data, filter_col='fittable', ret_col='logret')
        
        # Only fittable symbols should have residuals calculated
        fittable_data = resid_df[resid_df['fittable']]
        residual_sum = fittable_data['logret_resid_eqmkt'].sum()
        
        assert abs(residual_sum) < 1e-10, f"Fittable residuals sum to {residual_sum}, expected ~0"
        
        # Non-fittable should have residual calculated but it won't be included in market
        non_fittable_resid = resid_df.loc[(ts, 'LOWLIQ_binance'), 'logret_resid_eqmkt']
        # It should be logret - market_return where market is calculated from fittable only
        market_ret = fittable_data['logret'].mean()
        expected_non_fittable_resid = 0.10 - market_ret
        assert abs(non_fittable_resid - expected_non_fittable_resid) < 1e-10, \
            f"Non-fittable residual should be {expected_non_fittable_resid}, got {non_fittable_resid}"
    
    def test_volume_weighted_residuals_sum_to_zero(self):
        """Test that volume-weighted residualized returns sum to zero when weighted by volume."""
        # Create sample data with volumes
        timestamps = pd.date_range('2023-01-01', periods=3, freq='h', tz='UTC')
        
        data = []
        for ts in timestamps:
            # Different volumes for each symbol
            data.extend([
                {'ts': ts, 'symbol_venue': 'BTCUSDT_binance', 'logret': 0.02, 'dvolume': 1000000, 'fittable': True, 'advp': 10000000},
                {'ts': ts, 'symbol_venue': 'ETHUSDT_binance', 'logret': -0.01, 'dvolume': 500000, 'fittable': True, 'advp': 5000000},
                {'ts': ts, 'symbol_venue': 'BNBUSDT_binance', 'logret': 0.03, 'dvolume': 200000, 'fittable': True, 'advp': 2000000},
            ])
        
        df = pd.DataFrame(data).set_index(['ts', 'symbol_venue'])
        
        # Calculate volume-weighted residualized returns
        resid_df, _ = calculate_weighted_resid_return(df, filter_col='fittable', ret_col='logret')
        
        # Check that weighted residuals sum to approximately zero at each timestamp
        for ts in timestamps:
            ts_data = resid_df.loc[ts]
            
            # Get the weight column that was created
            if 'market_weight' in ts_data.columns:
                weights = ts_data['market_weight']
                weighted_residual_sum = (ts_data['logret_resid_wgtmkt'] * weights).sum()
                
                assert abs(weighted_residual_sum) < 1e-10, \
                    f"Volume-weighted residuals at {ts} sum to {weighted_residual_sum}, expected ~0"
    
    def test_residuals_are_demeaned(self):
        """Test that residuals are properly demeaned (return - market_return)."""
        # Create controlled data
        ts = pd.Timestamp('2023-01-01 12:00:00', tz='UTC')
        data = pd.DataFrame([
            {'ts': ts, 'symbol_venue': 'BTCUSDT_binance', 'logret': 0.04, 'fittable': True},
            {'ts': ts, 'symbol_venue': 'ETHUSDT_binance', 'logret': 0.02, 'fittable': True},
            {'ts': ts, 'symbol_venue': 'BNBUSDT_binance', 'logret': -0.03, 'fittable': True},
        ]).set_index(['ts', 'symbol_venue'])
        
        # Calculate residualized returns
        resid_df, _ = calculate_resid_return(data, filter_col='fittable', ret_col='logret')
        
        # Market return should be the mean
        expected_market = (0.04 + 0.02 + (-0.03)) / 3  # = 0.01
        
        # Check each residual
        assert abs(resid_df.loc[(ts, 'BTCUSDT_binance'), 'logret_resid_eqmkt'] - (0.04 - expected_market)) < 1e-10
        assert abs(resid_df.loc[(ts, 'ETHUSDT_binance'), 'logret_resid_eqmkt'] - (0.02 - expected_market)) < 1e-10
        assert abs(resid_df.loc[(ts, 'BNBUSDT_binance'), 'logret_resid_eqmkt'] - (-0.03 - expected_market)) < 1e-10
    
    def test_funding_adjusted_residuals_sum_to_zero(self):
        """Test that funding-adjusted residualized returns also sum to zero."""
        # Create sample data with funding-adjusted returns
        timestamps = pd.date_range('2023-01-01', periods=3, freq='h', tz='UTC')
        symbols = ['BTCUSDT_binance', 'ETHUSDT_binance', 'BNBUSDT_binance']
        
        data = []
        for ts in timestamps:
            for symbol in symbols:
                data.append({
                    'ts': ts,
                    'symbol_venue': symbol,
                    'logret_funding_adj': np.random.uniform(-0.03, 0.03),
                    'fittable': True
                })
        
        df = pd.DataFrame(data).set_index(['ts', 'symbol_venue'])
        
        # Calculate residualized returns for funding-adjusted returns
        resid_df, _ = calculate_resid_return(df, filter_col='fittable', ret_col='logret_funding_adj')
        
        # Check that residuals sum to approximately zero at each timestamp
        for ts in timestamps:
            ts_data = resid_df.loc[ts]
            fittable_data = ts_data[ts_data['fittable']]
            residual_sum = fittable_data['logret_funding_adj_resid_eqmkt'].sum()
            
            assert abs(residual_sum) < 1e-10, \
                f"Funding-adjusted residuals at {ts} sum to {residual_sum}, expected ~0"
    
    def test_empty_fittable_symbols(self):
        """Test handling when no symbols are fittable at a timestamp."""
        ts = pd.Timestamp('2023-01-01 12:00:00', tz='UTC')
        data = pd.DataFrame([
            {'ts': ts, 'symbol_venue': 'BTCUSDT_binance', 'logret': 0.01, 'fittable': False},
            {'ts': ts, 'symbol_venue': 'ETHUSDT_binance', 'logret': 0.02, 'fittable': False},
        ]).set_index(['ts', 'symbol_venue'])
        
        # Calculate residualized returns
        resid_df, _ = calculate_resid_return(data, filter_col='fittable', ret_col='logret')
        
        # When no symbols are fittable, market return is NaN, so residuals are NaN
        assert resid_df['logret_resid_eqmkt'].isna().all(), \
            "All residuals should be NaN when no symbols are fittable"
    
    def test_single_fittable_symbol(self):
        """Test handling when only one symbol is fittable at a timestamp."""
        ts = pd.Timestamp('2023-01-01 12:00:00', tz='UTC')
        data = pd.DataFrame([
            {'ts': ts, 'symbol_venue': 'BTCUSDT_binance', 'logret': 0.01, 'fittable': True},
            {'ts': ts, 'symbol_venue': 'ETHUSDT_binance', 'logret': 0.02, 'fittable': False},
        ]).set_index(['ts', 'symbol_venue'])
        
        # Calculate residualized returns
        resid_df, _ = calculate_resid_return(data, filter_col='fittable', ret_col='logret')
        
        # Single fittable symbol should have zero residual (return - return = 0)
        btc_resid = resid_df.loc[(ts, 'BTCUSDT_binance'), 'logret_resid_eqmkt']
        assert abs(btc_resid) < 1e-10, f"Single symbol residual should be ~0, got {btc_resid}"
    
    def test_large_universe_residuals(self):
        """Test residuals sum to zero with a large universe of symbols."""
        # Create a larger universe
        ts = pd.Timestamp('2023-01-01 12:00:00', tz='UTC')
        n_symbols = 100
        
        data = []
        for i in range(n_symbols):
            data.append({
                'ts': ts,
                'symbol_venue': f'SYMBOL{i}_binance',
                'logret': np.random.uniform(-0.05, 0.05),
                'dvolume': np.random.uniform(100000, 10000000),
                'fittable': True,
                'advp': np.random.uniform(1000000, 100000000)
            })
        
        df = pd.DataFrame(data).set_index(['ts', 'symbol_venue'])
        
        # Test equal-weighted residuals
        resid_df, _ = calculate_resid_return(df, filter_col='fittable', ret_col='logret')
        fittable_data = resid_df[resid_df['fittable']]
        eq_residual_sum = fittable_data['logret_resid_eqmkt'].sum()
        assert abs(eq_residual_sum) < 1e-10, \
            f"Large universe equal-weighted residuals sum to {eq_residual_sum}, expected ~0"
        
        # Test volume-weighted residuals
        wgt_resid_df, _ = calculate_weighted_resid_return(df, filter_col='fittable', ret_col='logret')
        if 'market_weight' in wgt_resid_df.columns:
            fittable_wgt = wgt_resid_df[wgt_resid_df['fittable']]
            weights = fittable_wgt['market_weight']
            wgt_residual_sum = (fittable_wgt['logret_resid_wgtmkt'] * weights).sum()
            assert abs(wgt_residual_sum) < 1e-10, \
                f"Large universe volume-weighted residuals sum to {wgt_residual_sum}, expected ~0"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])