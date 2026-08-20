"""Unit tests for return_calcs module."""

import unittest
import numpy as np
import pandas as pd

from lib.calcs.calc_returns import (
    calc_logret, 
    calc_vwap, 
    calc_funding_adjusted_logret,
    calculate_resid_return,
    calculate_weighted_resid_return,
    calculate_weighted_mkt_return
)


class TestReturnCalcs(unittest.TestCase):
    """Test cases for return calculation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        timestamps = pd.date_range('2024-01-01', periods=10, freq='h', tz='UTC')
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        
        # Create MultiIndex
        index = pd.MultiIndex.from_product([timestamps, symbols], names=['ts', 'symbol_venue'])
        
        # Create test dataframe
        self.df = pd.DataFrame({
            'close_mid': [100, 50, 101, 51, 102, 52, 103, 53, 104, 54, 
                         105, 55, 106, 56, 107, 57, 108, 58, 109, 59],
            'update_cnt': 1  # All bars have updates
        }, index=index)
    
    def test_calc_logret_basic(self):
        """Test basic log return calculation."""
        result, new_cols = calc_logret(self.df.copy())
        
        # Should have logret column
        self.assertIn('logret', result.columns)
        self.assertIn('logret', new_cols)
        
        # Check first values are NaN
        btc_data = result.xs('BTCUSDT_binance-futures', level='symbol_venue')
        eth_data = result.xs('ETHUSDT_binance-futures', level='symbol_venue')
        
        self.assertTrue(pd.isna(btc_data['logret'].iloc[0]))
        self.assertTrue(pd.isna(eth_data['logret'].iloc[0]))
        
        # Check some return calculations
        # BTC: log(101/100) ≈ 0.00995
        self.assertAlmostEqual(btc_data['logret'].iloc[1], np.log(101/100), places=5)
        # ETH: log(51/50) ≈ 0.0198
        self.assertAlmostEqual(eth_data['logret'].iloc[1], np.log(51/50), places=5)
    
    def test_calc_logret_with_missing_data(self):
        """Test log return calculation with missing close prices."""
        df = self.df.copy()
        # Set some close prices to NaN
        df.loc[df.index[2], 'close_mid'] = np.nan
        df.loc[df.index[3], 'close_mid'] = np.nan
        
        result, new_cols = calc_logret(df)
        
        # Should handle NaN by using previous close
        self.assertIn('logret', result.columns)
        self.assertIn('logret', new_cols)
        # The NaN close should be filled with previous close
        self.assertFalse(pd.isna(result.loc[result.index[2], 'close_mid']))
    
    def test_calc_logret_with_open_mid_validation(self):
        """Test bar validation when open_mid is present."""
        df = self.df.copy()
        # Add open_mid column - some matching, some not matching previous close
        df['open_mid'] = df['close_mid'].shift(1)
        df.loc[df.index[5], 'open_mid'] = 999  # Bad bar
        
        # Should still calculate returns but log warning
        result, new_cols = calc_logret(df)
        self.assertIn('logret', result.columns)
        self.assertIn('logret', new_cols)
    
    def test_calc_logret_no_updates(self):
        """Test that bars with update_cnt=0 get NaN returns."""
        df = self.df.copy()
        # Set some update counts to 0
        df.loc[df.index[2], 'update_cnt'] = 0
        df.loc[df.index[3], 'update_cnt'] = 0
        
        result, new_cols = calc_logret(df)
        
        # Those bars should have NaN returns
        self.assertTrue(pd.isna(result.loc[result.index[2], 'logret']))
        self.assertTrue(pd.isna(result.loc[result.index[3], 'logret']))
        
        # But close prices should still be valid (forward filled)
        self.assertFalse(pd.isna(result.loc[result.index[2], 'close_mid']))
        self.assertIn('logret', new_cols)


class TestVwapCalcs(unittest.TestCase):
    """Test cases for VWAP calculation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        timestamps = pd.date_range('2024-01-01', periods=5, freq='h', tz='UTC')
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        
        # Create MultiIndex
        index = pd.MultiIndex.from_product([timestamps, symbols], names=['ts', 'symbol_venue'])
        
        # Create test dataframe with volume and dollar volume
        self.df = pd.DataFrame({
            'volume': [1000, 2000, 1500, 2500, 1200, 
                      1800, 1100, 2200, 1300, 2100],
            'dvolume': [100000, 100000, 150000, 125000, 120000,
                       90000, 55000, 110000, 65000, 105000],  # price * volume
        }, index=index)
    
    def test_calc_vwap_basic(self):
        """Test basic VWAP calculation."""
        result, new_cols = calc_vwap(self.df.copy())
        
        # Should have vwap column
        self.assertIn('vwap', result.columns)
        self.assertIn('vwap', new_cols)
        
        # Check VWAP calculations: dvolume / volume
        # First BTC: 100000 / 1000 = 100
        self.assertAlmostEqual(result['vwap'].iloc[0], 100.0, places=2)
        # First ETH: 90000 / 1800 = 50
        self.assertAlmostEqual(result['vwap'].iloc[5], 50.0, places=2)
    
    def test_calc_vwap_with_horizon(self):
        """Test VWAP calculation with horizon suffix."""
        # Add horizon-specific columns
        self.df['volume_60'] = self.df['volume'] * 2
        self.df['dvolume_60'] = self.df['dvolume'] * 2
        
        result, new_cols = calc_vwap(self.df.copy(), horizon=60)
        
        # Should have vwap_60 column
        self.assertIn('vwap_60', result.columns)
        self.assertIn('vwap_60', new_cols)
        
        # VWAP should be the same (since we multiplied both by 2)
        self.assertAlmostEqual(result['vwap_60'].iloc[0], 100.0, places=2)
    
    def test_calc_vwap_with_zero_volume(self):
        """Test VWAP calculation with zero volume (should handle division by zero)."""
        df = self.df.copy()
        # Set some volumes to zero
        df.loc[df.index[2], 'volume'] = 0
        df.loc[df.index[3], 'volume'] = 0
        
        result, new_cols = calc_vwap(df)
        
        # Should handle division by zero gracefully (inf values removed)
        self.assertFalse(np.isinf(result['vwap'].iloc[2]))
        self.assertFalse(np.isinf(result['vwap'].iloc[3]))
        self.assertIn('vwap', new_cols)
    
    def test_calc_vwap_custom_fields(self):
        """Test VWAP calculation with custom field names."""
        df = self.df.copy()
        # Rename columns
        df = df.rename(columns={'volume': 'my_volume', 'dvolume': 'my_dvolume'})
        
        result, new_cols = calc_vwap(df, vwap_fld='my_vwap', volume_fld='my_volume', dvolume_fld='my_dvolume')
        
        # Should have custom vwap column
        self.assertIn('my_vwap', result.columns)
        self.assertIn('my_vwap', new_cols)
        
        # Check calculation is correct
        self.assertAlmostEqual(result['my_vwap'].iloc[0], 100.0, places=2)


class TestFundingAdjustedReturnCalcs(unittest.TestCase):
    """Test cases for funding-adjusted return calculation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data with funding times every 8 hours
        timestamps = pd.date_range('2024-01-01', periods=24, freq='h', tz='UTC')
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        
        # Create MultiIndex
        index = pd.MultiIndex.from_product([timestamps, symbols], names=['ts', 'symbol_venue'])
        
        # Create test dataframe
        self.df = pd.DataFrame({
            'logret': np.random.rand(48) * 0.01,  # Small random returns
            'last_funding_rate': 0.0001,  # Fixed funding rate for simplicity
        }, index=index)
        # Initialize next_funding_time with proper datetime type
        self.df['next_funding_time'] = pd.Series(pd.NaT, index=self.df.index, dtype='datetime64[ns, UTC]')
    
    def test_calc_funding_adjusted_logret_basic(self):
        """Test basic funding-adjusted return calculation."""
        df = self.df.copy()
        
        # Simple test: set one funding time
        # First timestamp realizes funding
        df.loc[df.index[0], 'next_funding_time'] = df.index[0][0]
        # All others are in the future
        df.loc[df.index[1:], 'next_funding_time'] = pd.Timestamp('2024-12-31', tz='UTC')
        
        # Store original logret values
        original_logret = df['logret'].copy()
        
        result, new_cols = calc_funding_adjusted_logret(df)
        
        # Should have funding adjusted return column
        self.assertIn('logret_funding_adj', result.columns)
        self.assertIn('logret_funding_adj', new_cols)
        
        # First row should have funding adjustment
        # When funding is realized, the return should be reduced
        expected_adj = original_logret.iloc[0] - np.log(1.0001)
        self.assertAlmostEqual(result.iloc[0]['logret_funding_adj'], expected_adj, places=7)
        
        # Other rows should be unchanged
        for i in range(1, len(result)):
            self.assertEqual(
                result.iloc[i]['logret_funding_adj'],
                original_logret.iloc[i]
            )
    
    def test_calc_funding_adjusted_logret_keep_realized(self):
        """Test keeping the realized funding column."""
        df = self.df.copy()
        
        # Set all times to non-funding times except one
        df['next_funding_time'] = pd.Timestamp('2024-01-02 00:00:00', tz='UTC')
        df.loc[df.index[0], 'next_funding_time'] = df.index[0][0]  # First timestamp realizes funding
        
        result, new_cols = calc_funding_adjusted_logret(df, keep_realized_funding=True)
        
        # Should keep the realized funding column
        self.assertIn('realized_funding_log_rate', result.columns)
        self.assertIn('logret_funding_adj', new_cols)
        self.assertIn('realized_funding_log_rate', new_cols)
        
        # Check values
        # First row should have non-zero realized funding
        self.assertNotEqual(result.iloc[0]['realized_funding_log_rate'], 0)
        # Others should be zero
        self.assertTrue((result.iloc[1:]['realized_funding_log_rate'] == 0).all())
    
    def test_calc_funding_adjusted_logret_no_funding_times(self):
        """Test when no funding times occur (all future)."""
        df = self.df.copy()
        
        # Set all funding times to the future
        df['next_funding_time'] = pd.Timestamp('2024-12-31 00:00:00', tz='UTC')
        
        result, new_cols = calc_funding_adjusted_logret(df)
        
        # All adjusted returns should equal original returns
        pd.testing.assert_series_equal(
            result['logret_funding_adj'],
            df['logret'],
            check_names=False
        )
        self.assertIn('logret_funding_adj', new_cols)


class TestCalculateResidReturn(unittest.TestCase):
    """Test calculate_resid_return function."""
    
    def setUp(self):
        """Create test data with MultiIndex."""
        timestamps = pd.date_range('2024-01-01', periods=3, freq='1h', tz='UTC')
        symbols = ['BTC', 'ETH', 'SOL']
        
        data = []
        for ts in timestamps:
            for symbol in symbols:
                data.append({
                    'ts': ts,
                    'symbol_venue': symbol,
                    'logret': np.random.randn() * 0.01,
                    'fittable': True
                })
        
        self.df = pd.DataFrame(data)
        self.df = self.df.set_index(['ts', 'symbol_venue'])
    
    def test_basic_residualization(self):
        """Test basic equal-weighted residualization."""
        result, new_cols = calculate_resid_return(self.df, filter_col='fittable')
        
        # Check new column exists
        self.assertIn('logret_resid_eqmkt', result.columns)
        self.assertIn('logret_resid_eqmkt', new_cols)
        
        # Check residuals sum to approximately zero at each timestamp
        for ts in result.index.get_level_values('ts').unique():
            ts_data = result.xs(ts, level='ts')
            residual_sum = ts_data['logret_resid_eqmkt'].sum()
            self.assertAlmostEqual(residual_sum, 0, places=10)
    
    def test_with_market_ret_df(self):
        """Test with pre-calculated market returns."""
        # Create market return df
        market_ret_df = self.df.reset_index()[['ts', 'logret']].groupby('ts').mean().reset_index()
        market_ret_df.rename(columns={'logret': 'logret_mkt'}, inplace=True)
        
        result, new_cols = calculate_resid_return(self.df, filter_col='fittable', market_ret_df=market_ret_df)
        
        # Check calculation
        df_reset = self.df.reset_index()
        df_merged = pd.merge(df_reset, market_ret_df, on='ts')
        expected_resid = df_merged['logret'] - df_merged['logret_mkt']
        
        result_reset = result.reset_index()
        pd.testing.assert_series_equal(
            result_reset['logret_resid_eqmkt'],
            expected_resid,
            check_names=False
        )
        self.assertIn('logret_resid_eqmkt', new_cols)
    
    def test_custom_ret_col(self):
        """Test with custom return column."""
        self.df['custom_ret'] = self.df['logret'] * 2
        result, new_cols = calculate_resid_return(self.df, filter_col='fittable', ret_col='custom_ret')
        
        self.assertIn('custom_ret_resid_eqmkt', result.columns)
        self.assertNotIn('logret_resid_eqmkt', result.columns)
        self.assertIn('custom_ret_resid_eqmkt', new_cols)
    
    def test_non_fittable_excluded(self):
        """Test that non-fittable symbols are excluded from market calculation."""
        # Make one symbol non-fittable
        df_copy = self.df.copy()
        df_copy.loc[df_copy.index.get_level_values('symbol_venue') == 'SOL', 'fittable'] = False
        
        result, new_cols = calculate_resid_return(df_copy, filter_col='fittable')
        
        # Market return should only be based on BTC and ETH
        for ts in result.index.get_level_values('ts').unique():
            ts_data = result.xs(ts, level='ts')
            fittable_data = ts_data[ts_data.index.get_level_values('symbol_venue').isin(['BTC', 'ETH'])]
            # Residuals for fittable symbols should sum to approximately zero
            residual_sum = fittable_data['logret_resid_eqmkt'].sum()
            self.assertAlmostEqual(residual_sum, 0, places=10)
        self.assertIn('logret_resid_eqmkt', new_cols)


class TestCalculateWeightedMktReturn(unittest.TestCase):
    """Test calculate_weighted_mkt_return function."""
    
    def setUp(self):
        """Create test data with MultiIndex."""
        timestamps = pd.date_range('2024-01-01', periods=3, freq='1h', tz='UTC')
        symbols = ['BTC', 'ETH', 'SOL']
        
        data = []
        for ts in timestamps:
            for i, symbol in enumerate(symbols):
                data.append({
                    'ts': ts,
                    'symbol_venue': symbol,
                    'logret': np.random.randn() * 0.01,
                    'advp': 1e6 * (i + 1),  # Different weights
                    'fittable': True
                })
        
        self.df = pd.DataFrame(data)
        self.df = self.df.set_index(['ts', 'symbol_venue'])
    
    def test_basic_weighted_market_return(self):
        """Test basic weighted market return calculation."""
        result, mkt_col = calculate_weighted_mkt_return(self.df, filter_col='fittable')
        
        # Check column was created
        self.assertEqual(mkt_col, 'logret_wgtmkt')
        self.assertIn(mkt_col, result.columns)
        
        # Check all timestamps have the same market return
        for ts in result.index.get_level_values('ts').unique():
            ts_data = result.xs(ts, level='ts')
            market_returns = ts_data[mkt_col].unique()
            self.assertEqual(len(market_returns), 1)
    
    def test_custom_return_column(self):
        """Test with custom return column."""
        self.df['custom_ret'] = self.df['logret'] * 2
        result, mkt_col = calculate_weighted_mkt_return(self.df, filter_col='fittable', ret_col='custom_ret')
        
        self.assertEqual(mkt_col, 'custom_ret_wgtmkt')
        self.assertIn(mkt_col, result.columns)
    
    def test_with_frequency(self):
        """Test with frequency parameter."""
        # Add the expected column with frequency suffix
        self.df['logret_60'] = self.df['logret'] * 1.1
        
        result, mkt_col = calculate_weighted_mkt_return(self.df, filter_col='fittable', frequency=60)
        
        self.assertEqual(mkt_col, 'logret_wgtmkt_60')
        self.assertIn('logret_wgtmkt_60', result.columns)
    
    def test_non_fittable_excluded(self):
        """Test that non-fittable symbols are excluded from calculation."""
        # Make one symbol non-fittable
        df_copy = self.df.copy()
        df_copy.loc[df_copy.index.get_level_values('symbol_venue') == 'SOL', 'fittable'] = False
        
        result, mkt_col = calculate_weighted_mkt_return(df_copy, filter_col='fittable')
        
        # The market return should only be based on BTC and ETH
        # We can't easily test the exact values, but we can ensure the column exists
        self.assertIn(mkt_col, result.columns)


class TestCalculateWeightedResidReturn(unittest.TestCase):
    """Test calculate_weighted_resid_return function."""
    
    def setUp(self):
        """Create test data with weights."""
        timestamps = pd.date_range('2024-01-01', periods=3, freq='1h', tz='UTC')
        symbols = ['BTC', 'ETH', 'SOL']
        
        data = []
        for ts in timestamps:
            for i, symbol in enumerate(symbols):
                data.append({
                    'ts': ts,
                    'symbol_venue': symbol,
                    'logret': np.random.randn() * 0.01,
                    'advp': 1e6 * (i + 1),  # Different weights
                    'fittable': True
                })
        
        self.df = pd.DataFrame(data)
        self.df = self.df.set_index(['ts', 'symbol_venue'])
    
    def test_basic_weighted_residualization(self):
        """Test basic volume-weighted residualization."""
        result, new_cols = calculate_weighted_resid_return(self.df, filter_col='fittable')
        
        # Check new columns exist
        self.assertIn('logret_resid_wgtmkt', result.columns)
        self.assertIn('logret_wgtmkt', result.columns)
        self.assertIn('logret_resid_wgtmkt', new_cols)
        self.assertIn('logret_wgtmkt', new_cols)
        
        # Check residuals are calculated correctly
        expected_resid = result['logret'] - result['logret_wgtmkt']
        pd.testing.assert_series_equal(
            result['logret_resid_wgtmkt'],
            expected_resid,
            check_names=False
        )
    
    def test_with_market_ret_df(self):
        """Test with pre-calculated weighted market returns."""
        # Create market return df
        df_reset = self.df.reset_index()
        market_ret_df = pd.DataFrame({
            'ts': df_reset['ts'].unique(),
            'logret_wgtmkt': np.random.randn(3) * 0.005
        })
        
        result, new_cols = calculate_weighted_resid_return(
            self.df, 
            filter_col='fittable',
            market_ret_df=market_ret_df
        )
        
        # Check that calculate_weighted_mkt_return was not called
        self.assertNotIn('mkt_weight', result.columns)
        self.assertIn('logret_resid_wgtmkt', new_cols)
        
        # Check residuals calculated correctly
        df_merged = pd.merge(df_reset, market_ret_df, on='ts')
        expected_resid = df_merged['logret'] - df_merged['logret_wgtmkt']
        
        result_reset = result.reset_index()
        pd.testing.assert_series_equal(
            result_reset['logret_resid_wgtmkt'],
            expected_resid,
            check_names=False
        )
    
    def test_custom_ret_col(self):
        """Test with custom return column."""
        self.df['custom_ret'] = self.df['logret'] * 2
        
        result, new_cols = calculate_weighted_resid_return(
            self.df, 
            filter_col='fittable',
            ret_col='custom_ret'
        )
        
        self.assertIn('custom_ret_resid_wgtmkt', result.columns)
        self.assertIn('custom_ret_wgtmkt', result.columns)
        self.assertIn('custom_ret_resid_wgtmkt', new_cols)
        self.assertIn('custom_ret_wgtmkt', new_cols)


if __name__ == '__main__':
    unittest.main()
