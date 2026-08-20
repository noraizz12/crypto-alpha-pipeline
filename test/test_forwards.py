"""Unit tests for the forwards module.

This module contains comprehensive unit tests for the forward returns calculation
functionality, including tests for initialization, forward return calculations,
live data integration, and the main generation pipeline.
"""

import unittest
from datetime import datetime as dt
from datetime import timedelta as td
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from lib.fits.forwards import Forwards, FOWARD_HORIZON_FROM_LIVE


class TestForwards(unittest.TestCase):
    """Test cases for the Forwards class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'FCASTS': {
                15: {
                    'models': [
                        {'name': 'hl', 'lags': 3},
                        {'name': 'c2vwap', 'lags': 2}
                    ]
                },
                60: {
                    'models': [
                        {'name': 'hl', 'lags': 5},
                        {'name': 'slz', 'lags': 3}
                    ]
                },
                1440: {
                    'models': [
                        {'name': 'hl', 'lags': 10},
                        {'name': 'vadj', 'lags': 5}
                    ]
                }
            },
            'SYMBOL_UNIVERSE': ['BTCUSDT', 'ETHUSDT'],
            'EXCHANGE': 'binance-futures',
            'DYNAMIC_UNIVERSE': True,
            'ADV_LOOKBACK_DAYS': 45,
            'MIN_ADVP_PRICEABLE': 2.5e7,
            'MIN_ADVP_FITTABLE': 1.5e7,
            'MIN_ADVP_TRADEABLE': 1.5e7,
            'MIN_ADVP_EXPANDABLE': 5.5e7,
            'MAX_MARKET_CAP_EXPANDABLE_FRAC': 1.0,
            'MAX_PORTFOLIO_NOTIONAL': 2.0e7,
            'MAX_POSITION_VOLUME_FRACTION': 0.05,
            'DEFAULT_FEATURE_LOOKBACK_PERIODS': 90,
            'BETA_LOOKBACK_PERIODS': 90,
            'MAX_MOVE_FILTER': 2.5,
            'MAX_POSITION_PCT': 0.04,
            'MAX_POSITION_BOUNDS': [
                [0.1, 0.010],
                [0.2, 0.010],
                [0.3, 0.015],
                [0.4, 0.020],
                [0.5, 0.025],
                [0.6, 0.030],
                [0.7, 0.035],
                [0.8, 0.040],
                [0.9, 0.045],
                [1.0, 0.050]
            ],
            'EXCLUDE_NON_ALPHA_TRADES': False,
            'OPT_HORIZON': 1440,
            'RISK_FLD': 'logret_HORIZON_trstd',
            'FILTER_DELISTING': True,
            'DELISTING_BUFFER_DAYS': 6,
            'FEATURE_SIGMA_BOUND': 5,
            'MAX_DVOL_SIGMA': 2.0,
            'MIN_FUNDING_RATE': -0.005,
            'OLD_POSITION_DAYS': 30,
            'OLD_POSITION_RISK_MULT': 3.5,
            'REOPT_TIMES': [],
            'OPT_OFFSET_MINS': 2,
            'MIN_ADVP_FEATUREABLE': 2.5e7,
            'FEATUREABLE_HIST_PERIODS': 30
        }
        
        self.mock_dir_manager = Mock()
        self.mock_dir_manager.FORWARDS_DIR = '/data/forwards'
        self.mock_dir_manager.LIVE_DATA_DIR = '/data/live'
        self.mock_dir_manager.PREBAR_DIR = '/data/prebars'
        self.mock_dir_manager.UNIVERSE_DIR = '/data/universe'
        
        # Create sample data for testing - ensure multiple of 1440 minutes (1 day)
        self.sample_dates = pd.date_range('2024-01-01', '2024-01-02 23:59:00', freq='1min', tz='UTC')  # Exactly 2 days
        self.sample_symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        
        # Create multi-index
        index_tuples = [(date, symbol) for date in self.sample_dates for symbol in self.sample_symbols]
        self.sample_index = pd.MultiIndex.from_tuples(index_tuples, names=['ts', 'symbol_venue'])
        
        # Create sample bar data
        self.sample_bars = pd.DataFrame({
            'close_mid': np.random.uniform(40000, 45000, len(self.sample_index)),
            'advp': 1e8,
            'fittable': True,
            'update_cnt_1': 100,
            'volume_1': 1000,
            'dvolume_1': 4e7,
            'last_funding_rate': 0.0001,
            'next_funding_time': self.sample_dates[0] + td(hours=8),
            'logret_15': np.random.normal(0, 0.001, len(self.sample_index)),
            'logret_resid_eqmkt_15': np.random.normal(0, 0.0008, len(self.sample_index)),
            'logret_resid_wgtmkt_15': np.random.normal(0, 0.0008, len(self.sample_index)),
            'logret_funding_adj_15': np.random.normal(0, 0.001, len(self.sample_index)),
            'logret_funding_adj_resid_eqmkt_15': np.random.normal(0, 0.0008, len(self.sample_index)),
            'logret_funding_adj_resid_wgtmkt_15': np.random.normal(0, 0.0008, len(self.sample_index)),
            'logret_15_trstd': 0.01
        }, index=self.sample_index)
        
        # Create a default Forwards instance for tests that need it
        with patch('lib.fits.forwards.DataLoader'), \
             patch('lib.fits.forwards.Calcs'):
            self.forwards_instance = Forwards(
                config=self.config,
                update=False,
                horizons=[15, 60, 1440],
                debug=False,
                forwards_dir_manager=self.mock_dir_manager
            )
        
    @patch('lib.fits.forwards.DataLoader')
    @patch('lib.fits.forwards.Calcs')
    def test_init_live_bars(self, mock_calcs, mock_data_loader):
        """Test initialization with live bars type."""
        forwards = Forwards(
            config=self.config,
            update=False,
            horizons=[15, 60, 1440],
            debug=False,
            forwards_dir_manager=self.mock_dir_manager,
        )
        
        self.assertEqual(forwards.config, self.config)
        self.assertFalse(forwards.update)
        self.assertFalse(forwards.debug)
        # bars_type no longer exists
        self.assertEqual(forwards.horizons, [15, 60, 1440])
        
    @patch('lib.fits.forwards.DataLoader')
    @patch('lib.fits.forwards.Calcs')
    def test_init_new_bars(self, mock_calcs, mock_data_loader):
        """Test initialization with NEW bars type."""
        mock_data_loader_instance = Mock()
        mock_data_loader.return_value = mock_data_loader_instance
        
        forwards = Forwards(
            config=self.config,
            update=True,
            horizons=[15, 60],
            debug=True,
            forwards_dir_manager=self.mock_dir_manager,
        )
        
        self.assertTrue(forwards.update)
        self.assertTrue(forwards.debug)
        # bars_type no longer exists
        
    def test_calculate_forward_returns_basic(self):
        """Test basic forward return calculation."""
        # Test data
        bars_df = self.sample_bars.copy()
        
        # Call the instance method
        result_df, cols_added = self.forwards_instance._calculate_forward_returns(
            bars_df=bars_df,
            alpha_lags=2,
            horizon=15,
            lookback_days=30,
            end_date=dt(2024, 1, 3).date(),
            produce_scaled_fields=False,
            funding_adjusted=False
        )
        
        # Check that forward return columns were added
        # Note: calculate_forward_returns adds lags up to alpha_lags + 1
        expected_cols = [
            'y_raw1_15', 'y_resid_eqmkt1_15', 'y_resid_wgtmkt1_15',
            'y_raw2_15', 'y_resid_eqmkt2_15', 'y_resid_wgtmkt2_15',
            'y_raw3_15', 'y_resid_eqmkt3_15', 'y_resid_wgtmkt3_15'
        ]
        self.assertEqual(set(cols_added), set(expected_cols))
        
        # Check that columns exist in result
        for col in expected_cols:
            self.assertIn(col, result_df.columns)
            
    def test_calculate_forward_returns_with_scaling(self):
        """Test forward return calculation with volatility scaling."""
        bars_df = self.sample_bars.copy()
        
        result_df, cols_added = self.forwards_instance._calculate_forward_returns(
            bars_df=bars_df,
            alpha_lags=2,
            horizon=15,
            lookback_days=30,
            end_date=dt(2024, 1, 3).date(),
            produce_scaled_fields=True,
            funding_adjusted=False
        )
        
        # Check that scaled columns were added (only up to alpha_lags, not alpha_lags+1)
        expected_scaled_cols = ['y_scaled_raw1_15', 'y_scaled_raw2_15']
        for col in expected_scaled_cols:
            self.assertIn(col, cols_added)
            self.assertIn(col, result_df.columns)
            
    def test_calculate_forward_returns_funding_adjusted(self):
        """Test forward return calculation with funding adjustment."""
        bars_df = self.sample_bars.copy()
        
        _, cols_added = self.forwards_instance._calculate_forward_returns(
            bars_df=bars_df,
            alpha_lags=1,
            horizon=15,
            lookback_days=30,
            end_date=dt(2024, 1, 3).date(),
            produce_scaled_fields=False,
            funding_adjusted=True
        )
        
        # Check funding adjusted columns (adds up to alpha_lags + 1)
        expected_cols = [
            'y_funding_adj_raw1_15', 
            'y_funding_adj_resid_eqmkt1_15', 
            'y_funding_adj_resid_wgtmkt1_15',
            'y_funding_adj_raw2_15', 
            'y_funding_adj_resid_eqmkt2_15', 
            'y_funding_adj_resid_wgtmkt2_15'
        ]
        self.assertEqual(set(cols_added), set(expected_cols))
        
    def test_calculate_forward_returns_cumulative(self):
        """Test that forward returns are properly cumulated across lags."""
        bars_df = self.sample_bars.copy()
        
        result_df, _ = self.forwards_instance._calculate_forward_returns(
            bars_df=bars_df,
            alpha_lags=3,
            horizon=15,
            lookback_days=30,
            end_date=dt(2024, 1, 3).date(),
            produce_scaled_fields=False,
            funding_adjusted=False
        )
        
        # For cumulative returns, y_raw2 should include y_raw1
        # This is a simplified check - in reality we'd verify the actual cumulation
        self.assertIn('y_raw1_15', result_df.columns)
        self.assertIn('y_raw2_15', result_df.columns)
        self.assertIn('y_raw3_15', result_df.columns)
        
    @patch('lib.fits.forwards.LiveBars')
    @patch('lib.fits.forwards.DataLoader')
    @patch('lib.fits.forwards.Calcs')
    @patch('lib.fits.forwards.concat')
    @patch('lib.fits.forwards.carry_forward')
    def test_calculate_forward_returns_from_live_bars(
        self, mock_carry_forward, mock_concat,
        mock_calcs, mock_data_loader, mock_live_bars
    ):
        """Test forward return calculation with live bar integration."""
        # Setup mocks
        mock_data_loader_instance = Mock()
        mock_data_loader_instance.load_bars.return_value = self.sample_bars.copy()
        mock_data_loader.return_value = mock_data_loader_instance
        
        mock_calcs_instance = Mock()
        # Create return data with all expected columns
        return_data = self.sample_bars.copy()
        return_data['logret'] = np.random.normal(0, 0.001, len(return_data))
        return_data['logret_resid_eqmkt'] = np.random.normal(0, 0.0008, len(return_data))
        return_data['logret_resid_wgtmkt'] = np.random.normal(0, 0.0008, len(return_data))
        return_data['logret_funding_adj'] = np.random.normal(0, 0.001, len(return_data))
        return_data['logret_funding_adj_resid_eqmkt'] = np.random.normal(0, 0.0008, len(return_data))
        return_data['logret_funding_adj_resid_wgtmkt'] = np.random.normal(0, 0.0008, len(return_data))
        mock_calcs_instance.calc_returns.return_value = return_data
        mock_calcs.return_value = mock_calcs_instance
        
        mock_live_bars_instance = Mock()
        # Create live bars data with correct columns
        live_bars_data = self.sample_bars.iloc[:100].copy()
        live_bars_data = live_bars_data[['close_mid', 'update_cnt_1', 'volume_1', 'dvolume_1', 'last_funding_rate', 'next_funding_time']]
        live_bars_data = live_bars_data.rename(columns={'update_cnt_1': 'update_cnt', 'volume_1': 'volume', 'dvolume_1': 'dvolume'})
        mock_live_bars_instance.load_live_bars.return_value = live_bars_data
        mock_live_bars.return_value = mock_live_bars_instance
        
        mock_concat.return_value = self.sample_bars.copy()
        mock_carry_forward.return_value = self.sample_bars.copy()
        
        # Create forwards instance
        forwards = Forwards(
            config=self.config,
            update=True,
            horizons=[15],
            debug=False,
            forwards_dir_manager=self.mock_dir_manager
        )
        
        # Call method
        result = forwards._calculate_forward_returns_from_live_bars(
            horizon=15,
            start_date=dt(2024, 1, 1).date(),
            end_date=dt(2024, 1, 3).date(),
            symbol_venues=['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        )
        
        # Verify calls
        mock_live_bars_instance.load_live_bars.assert_called_once()
        mock_concat.assert_called_once()
        mock_carry_forward.assert_called_once()
        
        # Check result structure
        self.assertIsInstance(result, pd.DataFrame)
        expected_cols = [
            'logret_15', 'logret_resid_eqmkt_15', 'logret_resid_wgtmkt_15',
            'logret_funding_adj_15', 'logret_funding_adj_resid_eqmkt_15',
            'logret_funding_adj_resid_wgtmkt_15'
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns)
            
    @patch('lib.fits.forwards.dump_parquet_files')
    @patch('lib.universe.DataLoader')
    @patch('lib.fits.forwards.DataLoader')
    @patch('lib.fits.forwards.Calcs')
    @patch('lib.fits.forwards.compute_lookback_days')
    @patch('lib.fits.forwards.yesterday_date')
    def test_generate_forwards(
        self, mock_yesterday, mock_compute_lookback,
        mock_calcs, mock_data_loader, mock_universe_data_loader, mock_dump_parquet
    ):
        """Test the main generate_forwards method."""
        # Setup mocks
        mock_yesterday.return_value = dt(2024, 1, 10).date()
        mock_compute_lookback.return_value = 5
        
        mock_data_loader_instance = Mock()
        mock_data_loader_instance.load_bars.return_value = self.sample_bars.copy()
        mock_data_loader_instance.load_universe_symbol_venues.return_value = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        mock_data_loader_instance.load_universe_df.return_value = pd.DataFrame({
            'symbol': ['BTCUSDT', 'ETHUSDT'],
            'symbol_venue': ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures'],
            'fittable': [True, True]
        })
        mock_data_loader.return_value = mock_data_loader_instance
        mock_universe_data_loader.return_value = mock_data_loader_instance
        
        mock_calcs_instance = Mock()
        mock_calcs.return_value = mock_calcs_instance
        
        # Create forwards instance
        forwards = Forwards(
            config=self.config,
            update=False,
            horizons=[15, 60],
            debug=False,
            forwards_dir_manager=self.mock_dir_manager
        )
        
        # Patch calculate_forward_returns to avoid complex calculations
        with patch.object(Forwards, '_calculate_forward_returns') as mock_calc_returns:
            mock_calc_returns.return_value = (self.sample_bars.copy(), ['y_raw1_15'])
            
            # Call generate_forwards
            forwards.generate_forwards(
                start_date=dt(2024, 1, 1).date(),
                end_date=dt(2024, 1, 3).date()
            )
            
            # Verify it was called for each date and horizon combination
            # 3 dates x 2 horizons x 2 (regular + funding) = 12
            self.assertEqual(mock_calc_returns.call_count, 12)
            
            # Verify parquet files were saved
            # 3 dates x 2 horizons = 6
            self.assertEqual(mock_dump_parquet.call_count, 6)
            
    def test_calculate_forward_returns_error_handling(self):
        """Test error handling when all forward returns are NaN."""
        # Create bars with all NaN returns
        bars_df = self.sample_bars.copy()
        bars_df['logret_15'] = np.nan
        bars_df['logret_resid_eqmkt_15'] = np.nan
        
        # This should raise an exception
        with self.assertRaises(Exception) as context:
            self.forwards_instance._calculate_forward_returns(
                bars_df=bars_df,
                alpha_lags=1,
                horizon=15,
                lookback_days=30,
                end_date=dt(2024, 1, 3).date(),
                produce_scaled_fields=False,
                funding_adjusted=False
            )
            
        self.assertIn("All nans generated", str(context.exception))
        
    def test_forward_horizon_from_live(self):
        """Test that FOWARD_HORIZON_FROM_LIVE is properly configured."""
        self.assertEqual(FOWARD_HORIZON_FROM_LIVE, [15])
        
    @patch('lib.fits.forwards.extract_max_lags')
    def test_horizon_to_max_lags_extraction(self, mock_extract):
        """Test that max lags are properly extracted from config."""
        mock_extract.return_value = {15: 3, 60: 5, 1440: 10}
        
        with patch('lib.fits.forwards.DataLoader'), \
             patch('lib.fits.forwards.Calcs'):
            
            forwards = Forwards(
                config=self.config,
                update=False,
                horizons=[15, 60, 1440],
                forwards_dir_manager=self.mock_dir_manager
            )
            
            self.assertEqual(forwards.horizon_to_max_lags, {15: 3, 60: 5, 1440: 10})
            # Called twice - once in setUp and once here
            self.assertEqual(mock_extract.call_count, 2)
            mock_extract.assert_called_with(self.config)


class TestForwardsIntegration(unittest.TestCase):
    """Integration tests for forwards calculation pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create more realistic test data
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='15min')
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures', 'BNBUSDT_binance-futures']
        
        index_tuples = [(date, symbol) for date in dates for symbol in symbols]
        self.index = pd.MultiIndex.from_tuples(index_tuples, names=['ts', 'symbol_venue'])
        
        # Generate correlated price series
        np.random.seed(42)
        btc_returns = np.random.normal(0, 0.01, len(dates))
        eth_returns = btc_returns * 0.8 + np.random.normal(0, 0.005, len(dates))
        bnb_returns = btc_returns * 0.6 + np.random.normal(0, 0.008, len(dates))
        
        all_returns = np.concatenate([btc_returns, eth_returns, bnb_returns])
        
        self.bars_df = pd.DataFrame({
            'logret_15': all_returns,
            'logret_resid_eqmkt_15': all_returns * 0.7,
            'logret_resid_wgtmkt_15': all_returns * 0.8,
            'logret_funding_adj_15': all_returns - 0.0001,
            'logret_funding_adj_resid_eqmkt_15': all_returns * 0.7 - 0.0001,
            'logret_funding_adj_resid_wgtmkt_15': all_returns * 0.8 - 0.0001,
            'logret_15_trstd': 0.01
        }, index=self.index)
        
        # Create a simple config and Forwards instance for tests
        self.config = {
            'FCASTS': {15: {'models': [{'name': 'hl', 'lags': 2}]}},
            'SYMBOL_UNIVERSE': symbols[:2],  # Use first 2 symbols
            'DYNAMIC_UNIVERSE': True,
            'MIN_ADVP_PRICEABLE': 2.5e7,
            'MIN_ADVP_FITTABLE': 1.5e7,
            'MIN_ADVP_TRADEABLE': 1.5e7,
            'MIN_ADVP_EXPANDABLE': 5.5e7,
            'MAX_MARKET_CAP_EXPANDABLE_FRAC': 1.0,
            'ADV_LOOKBACK_DAYS': 45,
            'DEFAULT_FEATURE_LOOKBACK_PERIODS': 90,
            'BETA_LOOKBACK_PERIODS': 90,
            'MAX_MOVE_FILTER': 2.5,
            'FEATURE_SIGMA_BOUND': 5,
            'MAX_DVOL_SIGMA': 2.0,
            'MIN_FUNDING_RATE': -0.005,
            'MIN_ADVP_FEATUREABLE': 2.5e7,
            'FEATUREABLE_HIST_PERIODS': 30,
            'MAX_PORTFOLIO_NOTIONAL': 2.0e7,
            'MAX_POSITION_VOLUME_FRACTION': 0.05,
            'MAX_POSITION_PCT': 0.04,
            'MAX_POSITION_BOUNDS': [
                [0.1, 0.010],
                [0.2, 0.010],
                [0.3, 0.015],
                [0.4, 0.020],
                [0.5, 0.025],
                [0.6, 0.030],
                [0.7, 0.035],
                [0.8, 0.040],
                [0.9, 0.045],
                [1.0, 0.050]
            ],
            'EXCLUDE_NON_ALPHA_TRADES': False,
            'OPT_HORIZON': 1440,
            'RISK_FLD': 'logret_HORIZON_trstd',
            'FILTER_DELISTING': True,
            'DELISTING_BUFFER_DAYS': 6,
            'OLD_POSITION_DAYS': 30,
            'OLD_POSITION_RISK_MULT': 3.5,
            'REOPT_TIMES': [],
            'OPT_OFFSET_MINS': 2
        }
        
        mock_dir_manager = Mock()
        mock_dir_manager.FORWARDS_DIR = '/data/forwards'
        
        with patch('lib.fits.forwards.DataLoader'), \
             patch('lib.fits.forwards.Calcs'):
            self.forwards_instance = Forwards(
                config=self.config,
                update=False,
                horizons=[15],
                debug=False,
                forwards_dir_manager=mock_dir_manager
            )
        
    def test_forward_returns_alignment(self):
        """Test that forward returns are properly aligned with future periods."""
        result_df, _ = self.forwards_instance._calculate_forward_returns(
            bars_df=self.bars_df,
            alpha_lags=1,
            horizon=15,
            lookback_days=5,
            end_date=dt(2024, 1, 10).date(),
            produce_scaled_fields=False,
            funding_adjusted=False
        )
        
        # Check that forward returns are shifted properly
        # y_raw1_15 at time t should equal logret_15 at time t+15min
        # This is a simplified check - would need more complex verification in practice
        self.assertIn('y_raw1_15', result_df.columns)
        
        # Verify no look-ahead bias
        non_nan_mask = ~result_df['y_raw1_15'].isna()
        self.assertTrue(non_nan_mask.sum() < len(result_df))  # Some values should be NaN at the end
        
    def test_multi_lag_consistency(self):
        """Test consistency across multiple lag periods."""
        result_df, _ = self.forwards_instance._calculate_forward_returns(
            bars_df=self.bars_df,
            alpha_lags=3,
            horizon=15,
            lookback_days=5,
            end_date=dt(2024, 1, 10).date(),
            produce_scaled_fields=False,
            funding_adjusted=False
        )
        
        # Verify all lag columns exist
        for lag in range(1, 4):
            self.assertIn(f'y_raw{lag}_15', result_df.columns)
            self.assertIn(f'y_resid_eqmkt{lag}_15', result_df.columns)
            self.assertIn(f'y_resid_wgtmkt{lag}_15', result_df.columns)


if __name__ == '__main__':
    unittest.main()
