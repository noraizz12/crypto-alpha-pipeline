"""Unit tests for the server_horizon_calcs module.

Tests cover the ServerHorizonCalcs class which manages real-time feature
and model calculations for specific time horizons in the live trading system.
"""

import unittest
from datetime import datetime as dt
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd

from lib.server import ServerHorizonCalcs


class TestServerHorizonCalcsInit(unittest.TestCase):
    """Test cases for ServerHorizonCalcs initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'REOPTIMIZE_INTERVAL_MINS': 120,
            'FCASTS': {
                '60': {
                    'models': [
                        {'name': 'hl', 'lags': 3, 'weight': 0.5},
                        {'name': 'c2vwap', 'lags': 2, 'weight': 0.3}
                    ]
                },
                '1440': {
                    'models': [
                        {'name': 'hl', 'lags': 5, 'weight': 1.0}
                    ]
                }
            }
        }

        # Create test features DataFrame
        dates = pd.date_range(start='2024-01-01 00:00', end='2024-01-01 02:00', freq='60min')
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        self.index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])
        self.features_df = pd.DataFrame(
            index=self.index,
            data={
                'close': np.random.randn(len(self.index)),
                'volume': np.random.randn(len(self.index)),
                'logret_60': np.random.randn(len(self.index))
            }
        )

    @patch('lib.server.server_horizon_calcs.Models')
    @patch('lib.server.server_horizon_calcs.Features')
    @patch('lib.server.server_horizon_calcs.Calcs')
    def test_init_basic(self, mock_calcs_class, mock_features_class, mock_models_class):
        """Test basic initialization of ServerHorizonCalcs."""
        server_calc = ServerHorizonCalcs(
            config=self.config,
            horizon=60,
            features_df=self.features_df
        )

        self.assertEqual(server_calc.config, self.config)
        self.assertEqual(server_calc.horizon, 60)
        self.assertEqual(server_calc.reopt_interval, 120)
        pd.testing.assert_frame_equal(server_calc.features_df, self.features_df)

        # Check last_processed_ts is set to max timestamp
        expected_last_ts = self.features_df.index.get_level_values('ts').max()
        self.assertEqual(server_calc.last_processed_ts, expected_last_ts)

        # Verify instances were created
        mock_calcs_class.assert_called_once_with(config=self.config, prod=True)
        mock_features_class.assert_called_once()
        mock_models_class.assert_called_once_with(config=self.config)

    @patch('lib.server.server_horizon_calcs.Models')
    @patch('lib.server.server_horizon_calcs.Features')
    @patch('lib.server.server_horizon_calcs.Calcs')
    def test_init_with_custom_dir_manager(self, mock_calcs_class, mock_features_class, mock_models_class):  # pylint: disable=unused-argument
        """Test initialization with custom directory manager."""
        mock_dir_manager = MagicMock()

        _ = ServerHorizonCalcs(
            config=self.config,
            horizon=1440,
            features_df=self.features_df,
            server_horizon_calcs_dir_manager=mock_dir_manager
        )

        # Check that Features was called with the custom dir manager
        mock_features_class.assert_called_once_with(
            config=self.config,
            frequency=1440,
            prod=True,
            features_dir_manager=mock_dir_manager
        )


class TestServerHorizonCalcsMethods(unittest.TestCase):
    """Test cases for ServerHorizonCalcs methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'REOPTIMIZE_INTERVAL_MINS': 60,
            'FCASTS': {
                '60': {
                    "features": [
                        "beta_1440",
                        "day_of_week"
                    ],
                    'models': [
                        {'name': 'hl', 'lags': 3, 'weight': 0.5},
                        {'name': 'c2vwap', 'lags': 2, 'weight': 0.0}
                    ]
                }
            }
        }

        # Create test features DataFrame
        dates = pd.date_range(start='2024-01-01 00:00', end='2024-01-01 02:00', freq='60min')
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        self.index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])
        self.features_df = pd.DataFrame(
            index=self.index,
            data={
                'close': np.random.randn(len(self.index)),
                'volume': np.random.randn(len(self.index))
            }
        )

    @patch('lib.server.server_horizon_calcs.Models')
    @patch('lib.server.server_horizon_calcs.Features')
    @patch('lib.server.server_horizon_calcs.Calcs')
    def test_get_max_ts(self, mock_calcs_class, mock_features_class, mock_models_class):  # pylint: disable=unused-argument
        """Test get_max_ts returns the maximum timestamp."""
        server_calc = ServerHorizonCalcs(
            config=self.config,
            horizon=60,
            features_df=self.features_df
        )

        max_ts = server_calc.get_max_ts()
        expected_max_ts = pd.Timestamp('2024-01-01 02:00:00')
        self.assertEqual(max_ts, expected_max_ts)

    @patch('lib.server.server_horizon_calcs.concat')
    @patch('lib.server.server_horizon_calcs.check_df_column_changes')
    @patch('lib.server.server_horizon_calcs.Models')
    @patch('lib.server.server_horizon_calcs.Features')
    @patch('lib.server.server_horizon_calcs.Calcs')
    def test_update_df(self, mock_calcs_class, mock_features_class, mock_models_class,
                       mock_check_df, mock_concat):  # pylint: disable=too-many-arguments
        """Test update_df appends new data correctly."""
        server_calc = ServerHorizonCalcs(
            config=self.config,
            horizon=60,
            features_df=self.features_df
        )

        # Create new data
        new_dates = pd.date_range(start='2024-01-01 03:00', end='2024-01-01 04:00', freq='60min')
        new_index = pd.MultiIndex.from_product([new_dates, ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']],
                                               names=['ts', 'symbol_venue'])
        new_df = pd.DataFrame(
            index=new_index,
            data={
                'close': np.random.randn(len(new_index)),
                'volume': np.random.randn(len(new_index))
            }
        )

        # Mock concat to return combined DataFrame
        combined_df = pd.concat([self.features_df, new_df])
        mock_concat.return_value = combined_df

        # Update DataFrame
        server_calc.update_df(new_df)

        # Verify checks were performed
        mock_check_df.assert_called_once()
        mock_concat.assert_called_once()

        # Verify features_df was updated
        pd.testing.assert_frame_equal(server_calc.features_df, combined_df)

    @patch('lib.server.server_horizon_calcs.Models')
    @patch('lib.server.server_horizon_calcs.Features')
    @patch('lib.server.server_horizon_calcs.Calcs')
    def test_update_last_processed_ts(self, mock_calcs_class, mock_features_class, mock_models_class):  # pylint: disable=unused-argument
        """Test update_last_processed_ts updates the timestamp."""
        server_calc = ServerHorizonCalcs(
            config=self.config,
            horizon=60,
            features_df=self.features_df
        )

        # Initially last_processed_ts should be max of features_df
        initial_ts = server_calc.last_processed_ts
        self.assertEqual(initial_ts, pd.Timestamp('2024-01-01 02:00:00'))

        # Add new data by modifying features_df directly
        new_dates = pd.date_range(start='2024-01-01 03:00', end='2024-01-01 03:00', freq='60min')
        new_index = pd.MultiIndex.from_product([new_dates, ['BTCUSDT_binance-futures']],
                                               names=['ts', 'symbol_venue'])
        new_rows = pd.DataFrame(index=new_index, data={'close': [100], 'volume': [1000]})
        server_calc.features_df = pd.concat([server_calc.features_df, new_rows])

        # Update last processed timestamp
        server_calc.update_last_processed_ts()

        # Should now be the new max
        self.assertEqual(server_calc.last_processed_ts, pd.Timestamp('2024-01-01 03:00:00'))

    @patch('lib.server.server_horizon_calcs.carry_forward')
    @patch('lib.server.server_horizon_calcs.Models')
    @patch('lib.server.server_horizon_calcs.Features')
    @patch('lib.server.server_horizon_calcs.Calcs')
    def test_calculate_features_carry_forward(self, mock_calcs_class, mock_features_class,
                                              mock_models_class, mock_carry_forward):  # pylint: disable=unused-argument
        """Test calculate_features performs carry forward operations."""
        mock_features_instance = MagicMock()
        mock_features_class.return_value = mock_features_instance

        server_calc = ServerHorizonCalcs(
            config=self.config,
            horizon=60,
            features_df=self.features_df
        )

        # Add boolean columns
        server_calc.features_df['fittable'] = True
        server_calc.features_df['tradeable'] = False

        # Store reference to the dataframe with bool columns
        df_with_bools = server_calc.features_df.copy()
        df_with_bools['fittable'] = True
        df_with_bools['tradeable'] = False

        # Mock carry_forward to return modified DataFrame
        mock_carry_forward.return_value = df_with_bools

        # Mock run_frequency to return the same DataFrame to preserve bool columns
        mock_features_instance.run_frequency.return_value = df_with_bools

        # Define carry forward map
        carry_forward_map = {60: ['fittable', 'tradeable']}

        # Calculate features
        server_calc.calculate_features(carry_forward_map)

        # Verify carry_forward was called
        mock_carry_forward.assert_called_once()
        call_args = mock_carry_forward.call_args[0]
        self.assertEqual(call_args[1], ['fittable', 'tradeable'])

        # Verify boolean columns are cast to bool
        self.assertEqual(server_calc.features_df['fittable'].dtype, bool)
        self.assertEqual(server_calc.features_df['tradeable'].dtype, bool)

    @patch('lib.server.server_horizon_calcs.Models')
    @patch('lib.server.server_horizon_calcs.Features')
    @patch('lib.server.server_horizon_calcs.Calcs')
    def test_calculate_features_horizon_1_skip(self, mock_calcs_class, mock_features_class, mock_models_class):  # pylint: disable=unused-argument
        """Test calculate_features skips feature calculation for horizon=1."""
        mock_features_instance = MagicMock()
        mock_features_class.return_value = mock_features_instance

        server_calc = ServerHorizonCalcs(
            config=self.config,
            horizon=1,  # 1-minute horizon
            features_df=self.features_df
        )

        carry_forward_map = {1: []}
        server_calc.calculate_features(carry_forward_map)

        # Features.run_frequency should NOT be called for horizon=1
        mock_features_instance.run_frequency.assert_not_called()

    @patch('lib.server.server_horizon_calcs.Models')
    @patch('lib.server.server_horizon_calcs.Features')
    @patch('lib.server.server_horizon_calcs.Calcs')
    def test_calculate_features_reopt_interval(self, mock_calcs_class, mock_features_class, mock_models_class):  # pylint: disable=unused-argument
        """Test calculate_features calculates volume forecast at reopt interval."""
        mock_calcs_instance = MagicMock()
        mock_calcs_class.return_value = mock_calcs_instance
        mock_features_instance = MagicMock()
        mock_features_class.return_value = mock_features_instance

        # Set horizon to match reopt interval
        server_calc = ServerHorizonCalcs(
            config=self.config,
            horizon=60,  # Matches REOPTIMIZE_INTERVAL_MINS
            features_df=self.features_df
        )

        # Mock return values
        mock_calcs_instance.calculate_volume_forecast.return_value = self.features_df
        # Mock run_frequency to return a dataframe with 'fittable' column
        features_df_with_fittable = self.features_df.copy()
        features_df_with_fittable['fittable'] = True
        mock_features_instance.run_frequency.return_value = features_df_with_fittable

        carry_forward_map = {60: []}
        server_calc.calculate_features(carry_forward_map)

        # Volume forecast should be calculated
        mock_calcs_instance.calculate_volume_forecast.assert_called_once_with(
            self.features_df, horizon=60
        )

        # Features should still be calculated
        mock_features_instance.run_frequency.assert_called_once()

    @patch('lib.server.server_horizon_calcs.set_index')
    @patch('lib.server.server_horizon_calcs.get_min_max_ts')
    @patch('lib.server.server_horizon_calcs.Models')
    @patch('lib.server.server_horizon_calcs.Features')
    @patch('lib.server.server_horizon_calcs.Calcs')
    def test_calculate_models(self, mock_calcs_class, mock_features_class, mock_models_class,
                              mock_get_min_max_ts, mock_set_index):  # pylint: disable=too-many-arguments
        """Test calculate_models processes new data correctly."""
        mock_models_instance = MagicMock()
        mock_models_class.return_value = mock_models_instance

        server_calc = ServerHorizonCalcs(
            config=self.config,
            horizon=60,
            features_df=self.features_df
        )

        # Set last_processed_ts to an earlier time
        server_calc.last_processed_ts = pd.Timestamp('2024-01-01 01:00:00')

        # Mock model calculation
        def mock_calculate(df, name, horizon):
            df[f'{name}_{horizon}_L0'] = 0.5
            return df

        mock_models_instance.calculate.side_effect = mock_calculate
        mock_get_min_max_ts.return_value = (pd.Timestamp('2024-01-01 02:00:00'),
                                            pd.Timestamp('2024-01-01 02:00:00'))

        # Mock set_index to return the DataFrame
        mock_set_index.side_effect = lambda df: df.set_index(['ts', 'symbol_venue'])

        # Calculate models
        server_calc.calculate_models()

        # Verify model calculation was called (only for non-zero weight model)
        self.assertEqual(mock_models_instance.calculate.call_count, 1)
        call_args = mock_models_instance.calculate.call_args[0]
        self.assertEqual(call_args[1], 'hl')
        self.assertEqual(call_args[2], 60)

        # Check that absolute value columns were created
        self.assertIn('hl_60_L0_abs', server_calc.features_df.columns)

    @patch('lib.server.server_horizon_calcs.set_index')
    @patch('lib.server.server_horizon_calcs.get_min_max_ts')
    @patch('lib.server.server_horizon_calcs.log_and_raise')
    @patch('lib.server.server_horizon_calcs.Models')
    @patch('lib.server.server_horizon_calcs.Features')
    @patch('lib.server.server_horizon_calcs.Calcs')
    def test_calculate_models_all_null_error(self, mock_calcs_class, mock_features_class,
                                             mock_models_class, mock_log_and_raise,
                                             mock_get_min_max_ts, mock_set_index):  # pylint: disable=too-many-arguments
        """Test calculate_models raises error when model returns all nulls."""
        mock_models_instance = MagicMock()
        mock_models_class.return_value = mock_models_instance

        server_calc = ServerHorizonCalcs(
            config=self.config,
            horizon=60,
            features_df=self.features_df
        )

        server_calc.last_processed_ts = pd.Timestamp('2024-01-01 01:00:00')

        # Mock model calculation to return all nulls
        def mock_calculate_null(df, name, horizon):
            df[f'{name}_{horizon}_L0'] = np.nan
            return df

        mock_models_instance.calculate.side_effect = mock_calculate_null
        mock_get_min_max_ts.return_value = (pd.Timestamp('2024-01-01 02:00:00'),
                                            pd.Timestamp('2024-01-01 02:00:00'))
        mock_log_and_raise.side_effect = ValueError("All values null")

        # Should raise error
        with self.assertRaises(ValueError):
            server_calc.calculate_models()

        mock_log_and_raise.assert_called_once()

    @patch('lib.server.server_horizon_calcs.generate_model_lags')
    @patch('lib.server.server_horizon_calcs.Models')
    @patch('lib.server.server_horizon_calcs.Features')
    @patch('lib.server.server_horizon_calcs.Calcs')
    def test_lag_models(self, mock_calcs_class, mock_features_class, mock_models_class,
                        mock_generate_lags):  # pylint: disable=unused-argument
        """Test lag_models generates lags for non-zero weight models."""
        server_calc = ServerHorizonCalcs(
            config=self.config,
            horizon=60,
            features_df=self.features_df
        )

        # Mock generate_model_lags to return updated DataFrame
        mock_generate_lags.return_value = (self.features_df, ['hl_60_L0', 'hl_60_L1', 'hl_60_L2'])

        # Generate lags
        server_calc.lag_models()

        # Should only be called for hl model (weight > 0), not c2vwap (weight = 0)
        mock_generate_lags.assert_called_once_with(
            df=self.features_df,
            model_name='hl_60',
            horizon=60,
            lags=3,
            prod=True
        )

    @patch('lib.server.server_horizon_calcs.Models')
    @patch('lib.server.server_horizon_calcs.Features')
    @patch('lib.server.server_horizon_calcs.Calcs')
    def test_calculate_and_lag_models(self, mock_calcs_class, mock_features_class, mock_models_class):  # pylint: disable=unused-argument
        """Test calculate_and_lag_models calls both methods."""
        server_calc = ServerHorizonCalcs(
            config=self.config,
            horizon=60,
            features_df=self.features_df
        )

        # Mock the individual methods
        with patch.object(server_calc, 'calculate_models') as mock_calc_models:
            with patch.object(server_calc, 'lag_models') as mock_lag_models:
                server_calc.calculate_and_lag_models()

                # Both methods should be called
                mock_calc_models.assert_called_once()
                mock_lag_models.assert_called_once()


class TestServerHorizonCalcsIntegration(unittest.TestCase):
    """Integration tests for ServerHorizonCalcs."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'REOPTIMIZE_INTERVAL_MINS': 60,
            'FCASTS': {
                '60': {
                    "features": [
                        "beta_1440",
                        "day_of_week"
                    ],
                    'models': [
                        {'name': 'hl', 'lags': 2, 'weight': 1.0}
                    ]
                }
            }
        }

        # Create test data
        dates = pd.date_range(start='2024-01-01 00:00', end='2024-01-01 03:00', freq='60min')
        symbols = ['BTCUSDT_binance-futures']
        self.index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])
        self.features_df = pd.DataFrame(
            index=self.index,
            data={
                'close': [100, 101, 102, 103],
                'volume': [1000, 1100, 1200, 1300],
                'fittable': [True, True, True, True]
            }
        )

    @patch('lib.server.server_horizon_calcs.generate_model_lags')
    @patch('lib.server.server_horizon_calcs.set_index')
    @patch('lib.server.server_horizon_calcs.get_min_max_ts')
    @patch('lib.server.server_horizon_calcs.carry_forward')
    @patch('lib.server.server_horizon_calcs.concat')
    @patch('lib.server.server_horizon_calcs.check_df_column_changes')
    @patch('lib.server.server_horizon_calcs.Models')
    @patch('lib.server.server_horizon_calcs.Features')
    @patch('lib.server.server_horizon_calcs.Calcs')
    def test_full_update_cycle(self, mock_calcs_class, mock_features_class, mock_models_class,
                               mock_check_df, mock_concat, mock_carry_forward,
                               mock_get_min_max_ts, mock_set_index, mock_generate_lags):  # pylint: disable=too-many-arguments,too-many-locals
        """Test a full update cycle with new data."""
        # Setup mocks
        mock_calcs_instance = MagicMock()
        mock_calcs_class.return_value = mock_calcs_instance
        mock_features_instance = MagicMock()
        mock_features_class.return_value = mock_features_instance
        mock_models_instance = MagicMock()
        mock_models_class.return_value = mock_models_instance

        # Initialize ServerHorizonCalcs
        server_calc = ServerHorizonCalcs(
            config=self.config,
            horizon=60,
            features_df=self.features_df[:2]  # Initial data
        )

        # Create new data
        new_df = self.features_df[2:]

        # Mock behaviors
        mock_concat.return_value = self.features_df  # Full DataFrame after concat
        mock_carry_forward.return_value = self.features_df
        mock_calcs_instance.calculate_volume_forecast.return_value = self.features_df
        mock_features_instance.run_frequency.return_value = self.features_df
        mock_get_min_max_ts.return_value = (pd.Timestamp('2024-01-01 02:00:00'),
                                            pd.Timestamp('2024-01-01 03:00:00'))

        def mock_model_calculate(df, name, horizon):
            df[f'{name}_{horizon}_L0'] = 0.1
            return df
        mock_models_instance.calculate.side_effect = mock_model_calculate

        mock_set_index.side_effect = lambda df: df.set_index(['ts', 'symbol_venue'])
        mock_generate_lags.return_value = (self.features_df, ['hl_60_L0', 'hl_60_L1'])

        # Perform full update cycle
        server_calc.update_df(new_df)
        server_calc.calculate_features({60: ['fittable']})
        server_calc.calculate_and_lag_models()
        server_calc.update_last_processed_ts()

        # Verify all steps were executed
        mock_check_df.assert_called_once()
        mock_concat.assert_called_once()
        mock_carry_forward.assert_called_once()
        mock_calcs_instance.calculate_volume_forecast.assert_called_once()  # Called because horizon = reopt_interval
        mock_features_instance.run_frequency.assert_called_once()
        mock_models_instance.calculate.assert_called_once()
        mock_generate_lags.assert_called_once()

        # Verify final state
        self.assertEqual(server_calc.last_processed_ts, pd.Timestamp('2024-01-01 03:00:00'))


if __name__ == '__main__':
    unittest.main()
