"""Unit tests for the model_calcs module.

Tests cover model lag generation and the ModelCalcs class for orchestrating
model signal calculations across dates and horizons.
"""

import unittest
import json
from datetime import datetime as dt
from datetime import date
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd

from lib.alpha.model_calcs import generate_model_lags, ModelCalcs


class TestGenerateModelLags(unittest.TestCase):
    """Test cases for the generate_model_lags function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test data with proper MultiIndex using UTC timestamps
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D', tz='UTC')
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        self.index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])

        # Create test dataframe with a model signal
        self.df = pd.DataFrame(index=self.index)
        # The function expects the _L0 column to exist
        self.df['hl_1440_L0'] = np.random.randn(len(self.index))

    def test_generate_model_lags_basic(self):
        """Test basic lag generation with 2 lags."""
        df, lag_cols = generate_model_lags(
            df=self.df.copy(),
            model_name='hl_1440',
            lags=2,
            horizon=1440,
            prod=False
        )

        # Check that lag columns were created
        self.assertIn('hl_1440_L0', df.columns)
        self.assertIn('hl_1440_L1', df.columns)
        self.assertIn('hl_1440_L2', df.columns)

        # Check returned column list
        self.assertEqual(lag_cols, ['hl_1440_L0', 'hl_1440_L1', 'hl_1440_L2'])

        # L0 should be present and unchanged
        self.assertIn('hl_1440_L0', df.columns)

    def test_generate_model_lags_minimum_lags(self):
        """Test that at least L0 and L1 are created even with lags=0."""
        df, lag_cols = generate_model_lags(
            df=self.df.copy(),
            model_name='hl_1440',
            lags=0,
            horizon=1440,
            prod=False
        )

        # Should create L0 and L1 minimum for classifier
        self.assertIn('hl_1440_L0', df.columns)
        self.assertIn('hl_1440_L1', df.columns)
        self.assertNotIn('hl_1440_L2', df.columns)

        self.assertEqual(lag_cols, ['hl_1440_L0', 'hl_1440_L1'])

    def test_generate_model_lags_production_mode(self):
        """Test lag generation in production mode (recent data only)."""
        # In production mode, only recent data should be processed
        df, _ = generate_model_lags(
            df=self.df.copy(),
            model_name='hl_1440',
            lags=1,
            horizon=1440,
            prod=True
        )

        # Should still create the lag columns
        self.assertIn('hl_1440_L0', df.columns)
        self.assertIn('hl_1440_L1', df.columns)

        # In prod mode, only the most recent timestamp gets updated
        max_ts = df.index.get_level_values('ts').max()
        recent_mask = df.index.get_level_values('ts') == max_ts

        # Check that values exist for recent timestamp
        self.assertFalse(df.loc[recent_mask, 'hl_1440_L0'].isna().all())

    def test_generate_model_lags_creates_columns(self):
        """Test that lag columns are created correctly."""
        # Create a model with known values
        self.df['model_x_L0'] = 100.0

        df, _ = generate_model_lags(
            df=self.df.copy(),
            model_name='model_x',
            lags=3,
            horizon=1440,
            prod=False
        )

        # Check all lag columns were created
        self.assertIn('model_x_L0', df.columns)
        self.assertIn('model_x_L1', df.columns)
        self.assertIn('model_x_L2', df.columns)
        self.assertIn('model_x_L3', df.columns)

        # L0 should remain unchanged
        self.assertTrue((df['model_x_L0'] == 100.0).all())

    @patch('lib.alpha.model_calcs.logger')
    def test_generate_model_lags_logging(self, mock_logger):
        """Test that appropriate logging occurs."""
        _, _ = generate_model_lags(
            df=self.df.copy(),
            model_name='hl_1440',
            lags=2,
            horizon=1440,
            prod=False
        )

        # Check that info logging was called
        mock_logger.info.assert_called()

    @patch('lib.alpha.model_calcs.check_missing_ts')
    def test_generate_model_lags_checks_missing_timestamps(self, mock_check_missing):
        """Test that missing timestamps are checked."""
        mock_check_missing.return_value = []

        _, _ = generate_model_lags(
            df=self.df.copy(),
            model_name='hl_1440',
            lags=1,
            horizon=1440,
            prod=False
        )

        # Verify check_missing_ts was called
        mock_check_missing.assert_called_once()


class TestModelCalcsInit(unittest.TestCase):
    """Test cases for ModelCalcs class initialization."""

    def setUp(self):
        """Set up test fixtures."""
        # Load test config with all required keys
        with open('test/fixtures/models/config_models_generation_test.json') as f:
            self.config = json.load(f)

        # Mock DirectoryManager
        self.mock_dir_manager = MagicMock()
        self.mock_dir_manager.MODELS_DIR = '/data/models'

    @patch('lib.alpha.model_calcs.Universe')
    @patch('lib.alpha.model_calcs.Models')
    @patch('lib.alpha.model_calcs.DataLoader')
    def test_init_default_parameters(self, mock_data_loader, mock_models, mock_universe):
        """Test initialization with default parameters."""
        model_calcs = ModelCalcs(
            config=self.config,
            models_dir_manager=self.mock_dir_manager
        )

        self.assertEqual(model_calcs.config, self.config)
        self.assertFalse(model_calcs.debug)
        self.assertEqual(model_calcs.pool_size, 4)
        self.assertEqual(model_calcs.output_dir, '/data/models')

        # Check that DataLoader and Models were instantiated
        mock_data_loader.assert_called_once()
        mock_models.assert_called_once_with(config=self.config)

    @patch('lib.alpha.model_calcs.Universe')
    @patch('lib.alpha.model_calcs.Models')
    @patch('lib.alpha.model_calcs.DataLoader')
    def test_init_custom_parameters(self, mock_data_loader, mock_models, mock_universe):
        """Test initialization with custom parameters."""
        model_calcs = ModelCalcs(
            config=self.config,
            models_to_run=['hl', 'slz'],
            horizons=[1440],
            debug=True,
            pool_size=2,
            output_dir='/custom/output',
            models_dir_manager=self.mock_dir_manager
        )

        self.assertTrue(model_calcs.debug)
        self.assertEqual(model_calcs.pool_size, 2)
        self.assertEqual(model_calcs.output_dir, '/custom/output')
        # bars_type no longer exists
        self.assertEqual(model_calcs.models_to_run, ['hl', 'slz'])
        self.assertEqual(model_calcs.horizons, [1440])

    @patch('lib.alpha.model_calcs.extract_models')
    @patch('lib.alpha.model_calcs.Universe')
    @patch('lib.alpha.model_calcs.extract_horizons')
    @patch('lib.alpha.model_calcs.Models')
    @patch('lib.alpha.model_calcs.DataLoader')
    def test_init_extracts_from_config(self, mock_data_loader, mock_models,
                                       mock_extract_horizons, mock_universe, mock_extract_models):
        """Test that horizons and models are extracted from config when not provided."""
        mock_extract_horizons.return_value = [60, 1440]
        mock_extract_models.return_value = ['hl', 'c2vwap', 'slz']

        model_calcs = ModelCalcs(
            config=self.config,
            models_dir_manager=self.mock_dir_manager
        )

        mock_extract_horizons.assert_called_once_with(self.config)
        mock_extract_models.assert_called_once_with(self.config, horizons=[60, 1440])

        self.assertEqual(model_calcs.horizons, [60, 1440])
        self.assertEqual(model_calcs.models_to_run, ['hl', 'c2vwap', 'slz'])


class TestModelCalcsMethods(unittest.TestCase):
    """Test cases for ModelCalcs class methods."""

    def setUp(self):
        """Set up test fixtures."""
        # Load test config with all required keys
        with open('test/fixtures/models/config_models_generation_test.json') as f:
            self.config = json.load(f)

        # Create test dataframe with minute-level data
        # For a single day, we need 1440 minutes × 2 symbols = 2880 rows
        dates = pd.date_range(start='2024-01-01', end='2024-01-01 23:59:00', freq='min')
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        self.index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])

        # Create random test data
        np.random.seed(42)
        n_rows = len(self.index)
        self.models_df = pd.DataFrame(
            index=self.index,
            data={
                'close': np.random.uniform(100, 200, n_rows),
                'volume': np.random.uniform(1000, 2000, n_rows),
                'fittable': np.random.choice([True, False], n_rows, p=[0.9, 0.1]),
            }
        )

    @patch('lib.alpha.model_calcs.dump_parquet_files')
    @patch('lib.alpha.model_calcs.Models')
    @patch('lib.alpha.model_calcs.DataLoader')
    def test_process_model_creates_signal(self, mock_data_loader, mock_models_class, mock_dump):
        """Test process_model creates model signal when not present."""
        # Mock the Models instance
        mock_models_instance = MagicMock()
        mock_models_class.return_value = mock_models_instance

        # Mock calculate to add the model column
        def add_model_column(df, name, horizon):
            df[f'{name}_{horizon}_L0'] = 0.1
            return df
        mock_models_instance.calculate.side_effect = add_model_column

        model_calcs = ModelCalcs(
            config=self.config,
            models_dir_manager=MagicMock()
        )

        forecast = {'name': 'hl', 'lags': 3}
        # Reset index as process_model expects columns, not index
        models_df = self.models_df.copy().reset_index()
        result_df = model_calcs.process_model(
            forecast=forecast,
            horizon=1440,
            models_df=models_df,
            start_date=date(2024, 1, 1)
        )

        # Check that calculate was called
        mock_models_instance.calculate.assert_called_once()
        call_args = mock_models_instance.calculate.call_args[0]
        self.assertEqual(call_args[1], 'hl')
        self.assertEqual(call_args[2], 1440)

        # Check that the model column was added
        self.assertIn('hl_1440_L0', result_df.columns)

        # Check that parquet was saved
        mock_dump.assert_called_once()

    @patch('lib.alpha.model_calcs.dump_parquet_files')
    @patch('lib.alpha.model_calcs.Models')
    @patch('lib.alpha.model_calcs.DataLoader')
    def test_process_model_skips_existing(self, mock_data_loader, mock_models_class, mock_dump):
        """Test process_model skips calculation if signal already exists."""
        mock_models_instance = MagicMock()
        mock_models_class.return_value = mock_models_instance

        model_calcs = ModelCalcs(
            config=self.config,
            models_dir_manager=MagicMock()
        )

        # Add existing model column and reset index
        models_df = self.models_df.copy()
        models_df['hl_1440_L0'] = 0.2
        models_df = models_df.reset_index()

        forecast = {'name': 'hl', 'lags': 3}
        _ = model_calcs.process_model(
            forecast=forecast,
            horizon=1440,
            models_df=models_df,
            start_date=date(2024, 1, 1)
        )

        # Calculate should NOT be called
        mock_models_instance.calculate.assert_not_called()

        # But parquet should still be saved
        mock_dump.assert_called_once()

    @patch('lib.alpha.model_calcs.dump_parquet_files')
    @patch('lib.alpha.model_calcs.set_index')
    @patch('lib.alpha.model_calcs.Models')
    @patch('lib.alpha.model_calcs.DataLoader')
    def test_process_model_new_format(self, mock_data_loader, mock_models_class,
                                      mock_set_index, mock_dump_new):
        """Test process_model with new bar format."""
        mock_models_instance = MagicMock()
        mock_models_class.return_value = mock_models_instance
        mock_set_index.return_value = self.models_df

        model_calcs = ModelCalcs(
            config=self.config,
            models_dir_manager=MagicMock()
        )

        forecast = {'name': 'hl', 'lags': 3}
        models_df = self.models_df.copy()
        models_df['hl_1440_L0'] = 0.1

        _ = model_calcs.process_model(
            forecast=forecast,
            horizon=1440,
            models_df=models_df,
            start_date=date(2024, 1, 1)
        )

        # Check that new format dump was called
        mock_dump_new.assert_called_once()
        mock_set_index.assert_called_once()

    @patch('lib.alpha.model_calcs.Pool')
    @patch('lib.alpha.model_calcs.merge_on_index')
    @patch('lib.alpha.model_calcs.Models')
    @patch('lib.alpha.model_calcs.DataLoader')
    def test_process_models_single_date(self, mock_data_loader_class, mock_models_class,
                                        mock_merge, mock_pool_class):
        """Test process_models for a single date."""
        # Setup mocks
        mock_data_loader = MagicMock()
        mock_data_loader_class.return_value = mock_data_loader

        # Mock feature and bar data loading with proper minute-level data
        features_df = pd.DataFrame(index=self.index, data={'feature1': np.random.randn(len(self.index))})
        bars_df = self.models_df.copy()

        mock_data_loader.load_features.return_value = features_df
        mock_data_loader.load_bars.return_value = bars_df
        mock_merge.return_value = self.models_df
        
        # Mock Models class and its methods
        mock_models = MagicMock()
        mock_models_class.return_value = mock_models
        mock_models.extract_models_features.return_value = (['close_trade', 'high_trade_1440'], ['feature1'])

        with patch('lib.alpha.model_calcs.Universe') as mock_universe_class:
            mock_universe = MagicMock()
            mock_universe.load_universe_symbols.return_value = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
            mock_universe_class.return_value = mock_universe
            
            model_calcs = ModelCalcs(
                config=self.config,
                horizons=[1440],
                models_to_run=['hl'],
                pool_size=1,  # Single process for easier testing
                models_dir_manager=MagicMock()
            )

        # Mock process_model
        with patch.object(model_calcs, 'process_model') as mock_process:
            model_calcs.process_models(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 1)
            )

            # Check that features and bars were loaded
            mock_data_loader.load_features.assert_called_once()
            mock_data_loader.load_bars.assert_called_once()

            # Check that process_model was called
            mock_process.assert_called_once()
            call_args = mock_process.call_args[1]
            self.assertEqual(call_args['horizon'], 1440)
            self.assertEqual(call_args['forecast']['name'], 'hl')

    @patch('lib.alpha.model_calcs.merge_on_index')
    @patch('lib.alpha.model_calcs.Models')
    @patch('lib.alpha.model_calcs.DataLoader')
    def test_process_models_skip_missing_features(self, mock_data_loader_class,
                                                  mock_models_class, mock_merge):
        """Test that process_models skips when features are missing."""
        mock_data_loader = MagicMock()
        mock_data_loader_class.return_value = mock_data_loader
        
        # Mock Models class and its methods
        mock_models = MagicMock()
        mock_models_class.return_value = mock_models
        mock_models.extract_models_features.return_value = (['close_trade'], ['feature1'])

        # Return None for features
        mock_data_loader.load_features.return_value = None

        with patch('lib.alpha.model_calcs.Universe') as mock_universe_class:
            mock_universe = MagicMock()
            mock_universe.load_universe_symbols.return_value = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
            mock_universe_class.return_value = mock_universe
            
            model_calcs = ModelCalcs(
                config=self.config,
                horizons=[1440],
                models_to_run=['hl'],
                models_dir_manager=MagicMock()
            )

        # Mock process_model
        with patch.object(model_calcs, 'process_model') as mock_process:
            model_calcs.process_models(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 1)
            )

            # process_model should not be called
            mock_process.assert_not_called()

    @patch('lib.alpha.model_calcs.Pool')
    @patch('lib.alpha.model_calcs.merge_on_index')
    @patch('lib.alpha.model_calcs.Models')
    @patch('lib.alpha.model_calcs.DataLoader')
    def test_process_models_parallel(self, mock_data_loader_class, mock_models_class,
                                    mock_merge, mock_pool_class):
        """Test parallel processing of models."""
        # Setup mocks
        mock_data_loader = MagicMock()
        mock_data_loader_class.return_value = mock_data_loader

        features_df = pd.DataFrame(index=self.index, data={'feature1': np.random.randn(len(self.index))})
        bars_df = self.models_df.copy()

        mock_data_loader.load_features.return_value = features_df
        mock_data_loader.load_bars.return_value = bars_df
        mock_merge.return_value = self.models_df
        
        # Mock Models class and its methods
        mock_models = MagicMock()
        mock_models_class.return_value = mock_models
        mock_models.extract_models_features.return_value = (['close_trade', 'high_trade_1440'], ['feature1'])

        # Mock pool
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        with patch('lib.alpha.model_calcs.Universe') as mock_universe_class:
            mock_universe = MagicMock()
            mock_universe.load_universe_symbols.return_value = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
            mock_universe_class.return_value = mock_universe
            
            model_calcs = ModelCalcs(
                config=self.config,
                horizons=[1440],
                models_to_run=['hl', 'c2vwap'],
                pool_size=2,  # Parallel processing
                models_dir_manager=MagicMock()
            )

        model_calcs.process_models(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1)
        )

        # Check that pool was used
        mock_pool_class.assert_called_once_with(processes=2)
        mock_pool.starmap.assert_called_once()
        mock_pool.close.assert_called_once()
        mock_pool.join.assert_called_once()

        # Check starmap arguments
        starmap_args = mock_pool.starmap.call_args[0]
        self.assertEqual(starmap_args[0], model_calcs.process_model)
        self.assertEqual(len(starmap_args[1]), 2)  # Two models


class TestModelCalcsIntegration(unittest.TestCase):
    """Integration tests for ModelCalcs."""

    def setUp(self):
        """Set up test fixtures."""
        # Load test config with all required keys
        with open('test/fixtures/models/config_models_generation_test.json') as f:
            self.config = json.load(f)

    @patch('lib.alpha.model_calcs.dump_parquet_files')
    @patch('lib.alpha.model_calcs.date_range')
    @patch('lib.alpha.model_calcs.merge_on_index')
    @patch('lib.alpha.model_calcs.Models')
    @patch('lib.alpha.model_calcs.DataLoader')
    def test_end_to_end_processing(self, mock_data_loader_class, mock_models_class,
                                   mock_merge, mock_date_range, mock_dump):
        """Test end-to-end processing of models."""
        # Setup date range
        mock_date_range.return_value = [date(2024, 1, 1), date(2024, 1, 2)]

        # Setup data loader
        mock_data_loader = MagicMock()
        mock_data_loader_class.return_value = mock_data_loader

        # Create test data - need minute-level data for each day
        # For 2 days, we need 1440 minutes × 1 symbol = 1440 rows per day
        dates1 = pd.date_range(start='2024-01-01', end='2024-01-01 23:59:00', freq='min')
        dates2 = pd.date_range(start='2024-01-02', end='2024-01-02 23:59:00', freq='min')
        symbols = ['BTCUSDT_binance-futures']
        
        index1 = pd.MultiIndex.from_product([dates1, symbols], names=['ts', 'symbol_venue'])
        index2 = pd.MultiIndex.from_product([dates2, symbols], names=['ts', 'symbol_venue'])
        
        # Create data for each date
        n_rows = len(index1)
        features_df1 = pd.DataFrame(index=index1, data={'feature1': np.random.randn(n_rows)})
        features_df2 = pd.DataFrame(index=index2, data={'feature1': np.random.randn(n_rows)})
        
        bars_df1 = pd.DataFrame(index=index1, data={'close': np.random.uniform(100, 200, n_rows)})
        bars_df2 = pd.DataFrame(index=index2, data={'close': np.random.uniform(100, 200, n_rows)})
        
        models_df1 = pd.DataFrame(index=index1, data={'close': bars_df1['close'], 'feature1': features_df1['feature1'], 'fittable': True})
        models_df2 = pd.DataFrame(index=index2, data={'close': bars_df2['close'], 'feature1': features_df2['feature1'], 'fittable': True})

        # Mock to return different data for each date
        mock_data_loader.load_features.side_effect = [features_df1, features_df2]
        mock_data_loader.load_bars.side_effect = [bars_df1, bars_df2]
        mock_merge.side_effect = [models_df1, models_df2]

        # Setup models
        mock_models = MagicMock()
        mock_models_class.return_value = mock_models
        mock_models.extract_models_features.return_value = (['close'], ['feature1'])

        def add_model_signal(df, name, horizon):
            df[f'{name}_{horizon}_L0'] = 0.5
            return df

        mock_models.calculate.side_effect = add_model_signal

        # Create and run ModelCalcs
        with patch('lib.alpha.model_calcs.Universe') as mock_universe_class:
            mock_universe = MagicMock()
            mock_universe.load_universe_symbols.return_value = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
            mock_universe_class.return_value = mock_universe
            
            model_calcs = ModelCalcs(
                config=self.config,
                pool_size=1,
                models_dir_manager=MagicMock()
            )

        model_calcs.process_models(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2)
        )

        # Verify the flow
        self.assertEqual(mock_data_loader.load_features.call_count, 2)  # Two dates
        self.assertEqual(mock_data_loader.load_bars.call_count, 2)
        self.assertEqual(mock_models.calculate.call_count, 6)  # 3 models × 2 dates
        self.assertEqual(mock_dump.call_count, 6)  # 3 models × 2 dates


if __name__ == '__main__':
    unittest.main()
