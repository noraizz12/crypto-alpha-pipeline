"""Unit tests for the features module."""

import unittest
from typing import List, Tuple

from lib.alpha.features import classify_features


class TestClassifyFeatures(unittest.TestCase):
    """Test cases for the classify_features function."""
    
    def test_fixed_columns(self):
        """Test classification of fixed columns from FIXED_COLS_TO_PERSIST."""
        features = ['close_trade', 'close_mid', 'advp', 'index_price']
        bar_features, feature_features = classify_features(features)
        
        self.assertEqual(bar_features, features)
        self.assertEqual(feature_features, [])
    
    def test_bar_fields_with_horizon(self):
        """Test classification of bar fields with HORIZON placeholder."""
        features = ['volume_HORIZON', 'dvolume_HORIZON', 'logret_HORIZON']
        bar_features, feature_features = classify_features(features)
        
        self.assertEqual(bar_features, features)
        self.assertEqual(feature_features, [])
    
    def test_bar_fields_with_numeric_horizon(self):
        """Test classification of bar fields with numeric horizons."""
        features = ['volume_60', 'dvolume_1440', 'logret_15']
        bar_features, feature_features = classify_features(features)
        
        self.assertEqual(bar_features, features)
        self.assertEqual(feature_features, [])
    
    def test_calculated_bar_fields(self):
        """Test classification of calculated bar fields like vwap."""
        features = ['vwap_HORIZON', 'vwap_60', 'index_price_vwap_1440']
        bar_features, feature_features = classify_features(features)
        
        self.assertEqual(bar_features, features)
        self.assertEqual(feature_features, [])
    
    def test_aggregated_bar_fields(self):
        """Test classification of bar fields with aggregation methods."""
        features = [
            'index_mid_price_diff_mean_HORIZON',
            'index_mid_price_diff_std_60',
            'last_funding_rate_mean_1440',
            'last_funding_rate_std_HORIZON'
        ]
        bar_features, feature_features = classify_features(features)
        
        self.assertEqual(bar_features, features)
        self.assertEqual(feature_features, [])
    
    def test_feature_file_fields(self):
        """Test classification of fields from feature files."""
        features = [
            'beta_HORIZON',
            'rsi_60',
            'logret_HORIZON_lz',
            'dvolume_1440_trmean',
            'news_event_decayed_HORIZON'
        ]
        bar_features, feature_features = classify_features(features)
        
        self.assertEqual(bar_features, [])
        self.assertEqual(feature_features, features)
    
    def test_mixed_features(self):
        """Test classification of mixed bar and feature file fields."""
        features = [
            'close_trade',  # bar field (fixed)
            'volume_60',  # bar field
            'vwap_HORIZON',  # bar field (calculated)
            'beta_1440',  # feature field
            'logret_HORIZON_lz',  # feature field
            'index_mid_price_diff_mean_HORIZON',  # bar field (aggregated)
            'rsi_60'  # feature field
        ]
        expected_bar = ['close_trade', 'volume_60', 'vwap_HORIZON', 'index_mid_price_diff_mean_HORIZON']
        expected_feature = ['beta_1440', 'logret_HORIZON_lz', 'rsi_60']
        
        bar_features, feature_features = classify_features(features)
        
        self.assertEqual(bar_features, expected_bar)
        self.assertEqual(feature_features, expected_feature)
    
    def test_edge_cases(self):
        """Test edge cases and potential ambiguities."""
        features = [
            'twap',  # Should be bar field (in BAR_FLDS_AGG_DICT)
            'twap_mean_60',  # Should be bar field (twap with mean aggregation)
            'high_trade_HORIZON',  # Should be bar field
            'low_trade_max_1440',  # Should be bar field (low_trade with max aggregation)
            'unknown_field_60',  # Should be feature field (not in bar fields)
            'close_mid'  # Should be bar field (fixed column)
        ]
        
        bar_features, feature_features = classify_features(features)
        
        # Check specific classifications
        self.assertIn('twap', bar_features)
        self.assertIn('twap_mean_60', bar_features)
        self.assertIn('high_trade_HORIZON', bar_features)
        self.assertIn('low_trade_max_1440', bar_features)
        self.assertIn('close_mid', bar_features)
        self.assertIn('unknown_field_60', feature_features)
    
    def test_empty_list(self):
        """Test with empty feature list."""
        features = []
        bar_features, feature_features = classify_features(features)
        
        self.assertEqual(bar_features, [])
        self.assertEqual(feature_features, [])
    
    def test_single_field_bar(self):
        """Test with single bar field."""
        features = ['close_trade']
        bar_features, feature_features = classify_features(features)
        
        self.assertEqual(bar_features, ['close_trade'])
        self.assertEqual(feature_features, [])
    
    def test_single_field_feature(self):
        """Test with single feature field."""
        features = ['beta_1440']
        bar_features, feature_features = classify_features(features)
        
        self.assertEqual(bar_features, [])
        self.assertEqual(feature_features, ['beta_1440'])
    
    def test_all_horizons(self):
        """Test with all possible horizon values."""
        horizons = ['15', '60', '120', '360', '720', '1440', '4320', '10080', '43200']
        features = [f'volume_{h}' for h in horizons]
        features.extend([f'beta_{h}' for h in horizons])  # Feature fields
        
        bar_features, feature_features = classify_features(features)
        
        # All volume_{horizon} should be bar fields
        for h in horizons:
            self.assertIn(f'volume_{h}', bar_features)
            self.assertIn(f'beta_{h}', feature_features)


if __name__ == '__main__':
    unittest.main()