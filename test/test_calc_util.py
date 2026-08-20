"""Unit tests for calc_util module."""

import unittest
import numpy as np
import pandas as pd

from lib.calcs.calc_util import make_market_weight_col, calculate_quintiles, calculate_abs_factor, calculate_trailing_z, calculate_time_factors


class TestCalcUtil(unittest.TestCase):
    """Test cases for calc_util functions."""
    
    def test_make_market_weight_col(self):
        """Test make_market_weight_col function."""
        # Create test data
        df = pd.DataFrame({
            'symbol': ['BTC', 'ETH', 'SOL'],
            'advp': [1e9, 1e8, 1e7]  # Different ADVPs
        })
        
        # Apply function
        result = make_market_weight_col(df)
        
        # Check column was added
        self.assertIn('mkt_wgt', result.columns)
        
        # Check calculation is correct (advp / 1e6, clipped to min 1)
        expected_weights = np.log(np.maximum(df['advp'] / 1e6, 1.0))
        pd.testing.assert_series_equal(
            result['mkt_wgt'],
            expected_weights,
            check_names=False
        )
        
        # Check relative ordering (higher ADVP = higher weight)
        self.assertGreater(result.loc[0, 'mkt_wgt'], result.loc[1, 'mkt_wgt'])
        self.assertGreater(result.loc[1, 'mkt_wgt'], result.loc[2, 'mkt_wgt'])
    
    def test_make_market_weight_col_with_zeros(self):
        """Test handling of zero ADVP values."""
        df = pd.DataFrame({
            'symbol': ['BTC', 'ETH'],
            'advp': [1e9, 0]  # Zero ADVP
        })

        result = make_market_weight_col(df)

        # Zero ADVP is clipped to 1, so log(1) = 0 (no negative weights)
        self.assertEqual(result.loc[1, 'mkt_wgt'], 0.0)
    
    def test_calculate_quintiles(self):
        """Test calculate_quintiles function."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        result, new_cols = calculate_quintiles(df, 'value')
        
        # Check return types
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsInstance(new_cols, list)
        self.assertEqual(new_cols, ['value_q'])
        
        # Check column was added
        self.assertIn('value_q', result.columns)
        
        # Check quintile labels
        expected_labels = ['Q1_value', 'Q1_value', 'Q2_value', 'Q2_value', 'Q3_value', 
                          'Q3_value', 'Q4_value', 'Q4_value', 'Q5_value', 'Q5_value']
        pd.testing.assert_series_equal(
            result['value_q'].astype(str),
            pd.Series(expected_labels),
            check_names=False
        )
    
    def test_calculate_abs_factor(self):
        """Test calculate_abs_factor function."""
        df = pd.DataFrame({
            'logret_lz': [-2.5, -1.0, 0.0, 1.0, 2.5]
        })
        
        result, new_cols = calculate_abs_factor(df, 'logret_lz')
        
        # Check return types
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsInstance(new_cols, list)
        self.assertEqual(new_cols, ['logret_lz_abs'])
        
        # Check column was added
        self.assertIn('logret_lz_abs', result.columns)
        
        # Check values are absolute
        expected = pd.Series([2.5, 1.0, 0.0, 1.0, 2.5])
        pd.testing.assert_series_equal(
            result['logret_lz_abs'],
            expected,
            check_names=False
        )
    
    def test_calculate_abs_factor_with_index(self):
        """Test calculate_abs_factor with start index filter."""
        df = pd.DataFrame({
            'logret_lz': [-2.5, -1.0, 0.0, 1.0, 2.5]
        })
        
        # Only apply to last 3 rows
        start_idx = pd.Series([False, False, True, True, True])
        result, new_cols = calculate_abs_factor(df, 'logret_lz', start_idx)
        
        # Check only specified rows were affected
        self.assertTrue(pd.isna(result.loc[0, 'logret_lz_abs']))
        self.assertTrue(pd.isna(result.loc[1, 'logret_lz_abs']))
        self.assertEqual(result.loc[2, 'logret_lz_abs'], 0.0)
        self.assertEqual(result.loc[3, 'logret_lz_abs'], 1.0)
        self.assertEqual(result.loc[4, 'logret_lz_abs'], 2.5)
    
    def test_calculate_trailing_z(self):
        """Test calculate_trailing_z function."""
        df = pd.DataFrame({
            'logret': [0.01, 0.02, -0.01, 0.03, -0.02],
            'logret_trmean': [0.01, 0.01, 0.01, 0.01, 0.01],
            'logret_trstd': [0.01, 0.01, 0.01, 0.01, 0.01]
        })
        
        result, new_cols = calculate_trailing_z(df, 'logret', feature_sigma_bound=3)
        
        # Check return types
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsInstance(new_cols, list)
        self.assertEqual(new_cols, ['logret_lz'])
        
        # Check column was added
        self.assertIn('logret_lz', result.columns)
        
        # Check z-score calculation
        expected = pd.Series([0.0, 1.0, -2.0, 2.0, -3.0], dtype=np.float32)
        pd.testing.assert_series_equal(
            result['logret_lz'],
            expected,
            check_names=False
        )
    
    def test_calculate_trailing_z_with_clipping(self):
        """Test calculate_trailing_z with sigma bound clipping."""
        df = pd.DataFrame({
            'logret': [0.1, -0.1],  # Large values that will exceed bounds
            'logret_trmean': [0.0, 0.0],
            'logret_trstd': [0.01, 0.01]  # Will result in z-scores of ±10
        })
        
        result, new_cols = calculate_trailing_z(df, 'logret', feature_sigma_bound=5)
        
        # Check values are clipped
        self.assertEqual(result.loc[0, 'logret_lz'], 5.0)  # Clipped to upper bound
        self.assertEqual(result.loc[1, 'logret_lz'], -5.0)  # Clipped to lower bound
    
    def test_calculate_time_factors(self):
        """Test calculate_time_factors function."""
        # Create test data with MultiIndex
        timestamps = pd.date_range('2024-01-01 14:30:00', periods=5, freq='1h', tz='UTC')
        symbols = ['BTC', 'ETH']
        index = pd.MultiIndex.from_product([timestamps, symbols], names=['ts', 'symbol_venue'])
        
        df = pd.DataFrame({
            'value': range(10)
        }, index=index)
        
        result, new_cols = calculate_time_factors(df)
        
        # Check return types
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIsInstance(new_cols, list)
        self.assertEqual(new_cols, ['hour_of_day', 'day_of_week'])
        
        # Check columns were added
        self.assertIn('hour_of_day', result.columns)
        self.assertIn('day_of_week', result.columns)
        
        # Check hour_of_day values (14:29, 15:29, 16:29, 17:29, 18:29 due to -1 minute)
        expected_hours = [14, 14, 15, 15, 16, 16, 17, 17, 18, 18]  # Sorted by timestamp first, then symbol
        pd.testing.assert_series_equal(
            result['hour_of_day'],
            pd.Series(expected_hours, index=result.index, dtype='Int32'),
            check_names=False
        )
        
        # Check day_of_week (Monday = 0)
        expected_days = [0] * 10  # All Monday
        pd.testing.assert_series_equal(
            result['day_of_week'],
            pd.Series(expected_days, index=result.index, dtype='Int32'),
            check_names=False
        )
    
    def test_calculate_time_factors_with_index(self):
        """Test calculate_time_factors with start index filter."""
        timestamps = pd.date_range('2024-01-01 14:30:00', periods=3, freq='1h', tz='UTC')
        index = pd.MultiIndex.from_product([timestamps, ['BTC']], names=['ts', 'symbol_venue'])
        
        df = pd.DataFrame({'value': [1, 2, 3]}, index=index)
        
        # Only apply to last 2 rows
        start_idx = pd.Series([False, True, True], index=df.index)
        result, new_cols = calculate_time_factors(df, start_idx)
        
        # Check first row has default values (0)
        self.assertEqual(result.iloc[0]['hour_of_day'], 0)
        self.assertEqual(result.iloc[0]['day_of_week'], 0)
        
        # Check other rows have correct values
        self.assertEqual(result.iloc[1]['hour_of_day'], 15)
        self.assertEqual(result.iloc[2]['hour_of_day'], 16)


if __name__ == '__main__':
    unittest.main()