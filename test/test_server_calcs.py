"""Unit tests for the server_calcs module.

Tests cover the ServerCalcs class which orchestrates real-time calculations
across multiple time horizons for the live trading system.
"""
# pylint: disable=too-many-public-methods

import unittest
from datetime import datetime as dt
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from lib.server import ServerCalcs


class TestServerCalcsInit(unittest.TestCase):
    """Test cases for ServerCalcs initialization."""
    # pylint: disable=unused-argument

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'OPT_HORIZON': 1440,
            'ST_RISK_HORIZON': 15,
            'VOLUME_BUCKET_MINS': 60,
            'NEWS_SIMILARITY_THRESHOLD': 0.9,
            'REOPTIMIZE_INTERVAL_MINS': 120
        }

        self.eod_today = dt(2024, 1, 1, 23, 59, 59)
        self.feature_horizons = [15, 60, 120, 1440]
        self.short_term_horizons = [15, 60]
        self.model_horizons = [15, 60, 120, 1440]
        self.feature_flds_to_load = {
            60: ['beta_60', 'rsi_60'],
            1440: ['beta_1440']
        }

    @patch('lib.server.server_calcs.Forecasts')
    @patch('lib.server.server_calcs.Calcs')
    @patch('lib.server.server_calcs.LiveBars')
    @patch('lib.server.server_calcs.DataLoader')
    def test_init_basic(self, mock_data_loader_class, mock_live_bars_class,
                       mock_calcs_class, mock_forecasts_class):
        """Test basic initialization of ServerCalcs."""
        server_calcs = ServerCalcs(
            config=self.config,
            eod_today=self.eod_today,
            feature_horizons=self.feature_horizons,
            short_term_horizons=self.short_term_horizons,
            model_horizons=self.model_horizons,
            feature_flds_to_load=self.feature_flds_to_load
        )

        # Check attributes are set correctly
        self.assertEqual(server_calcs.config, self.config)
        self.assertEqual(server_calcs.eod_today, self.eod_today)
        self.assertEqual(server_calcs.feature_horizons, self.feature_horizons)
        self.assertEqual(server_calcs.short_term_horizons, self.short_term_horizons)
        self.assertEqual(server_calcs.model_horizons, self.model_horizons)
        self.assertEqual(server_calcs.feature_flds_to_load, self.feature_flds_to_load)

        # Check config values are extracted
        self.assertEqual(server_calcs.opt_horizon, 1440)
        self.assertEqual(server_calcs.st_risk_horizon, 15)
        self.assertEqual(server_calcs.volume_bucket_mins, 60)
        self.assertEqual(server_calcs.similarity_threshold, 0.9)

        # Check instances are created
        mock_data_loader_class.assert_called_once()
        mock_live_bars_class.assert_called_once()
        mock_calcs_class.assert_called_once_with(config=self.config, prod=True)
        mock_forecasts_class.assert_called_once()

    @patch('lib.server.server_calcs.Forecasts')
    @patch('lib.server.server_calcs.Calcs')
    @patch('lib.server.server_calcs.LiveBars')
    @patch('lib.server.server_calcs.DataLoader')
    def test_init_with_optional_params(self, mock_data_loader_class, mock_live_bars_class,
                                      mock_calcs_class, mock_forecasts_class):  # pylint: disable=unused-argument
        """Test initialization with optional parameters."""
        universe = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        mock_dir_manager = MagicMock()

        _ = ServerCalcs(
            config=self.config,
            eod_today=self.eod_today,
            feature_horizons=self.feature_horizons,
            short_term_horizons=self.short_term_horizons,
            model_horizons=self.model_horizons,
            feature_flds_to_load=self.feature_flds_to_load,
            server_calcs_dir_manager=mock_dir_manager,
            universe=universe
        )

        # Check optional parameters are passed correctly
        mock_live_bars_class.assert_called_once_with(
            live_bar_dir=mock_dir_manager.LIVE_DATA_DIR,
            universe=universe,
            use_new=True
        )

        # bars_type is no longer passed to Forecasts


class TestServerCalcsMethods(unittest.TestCase):
    """Test cases for ServerCalcs methods."""
    # pylint: disable=unused-argument,too-many-public-methods

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'OPT_HORIZON': 1440,
            'ST_RISK_HORIZON': 15,
            'VOLUME_BUCKET_MINS': 60,
            'NEWS_SIMILARITY_THRESHOLD': 0.9,
            'REOPTIMIZE_INTERVAL_MINS': 120
        }

        # Create test DataFrames
        dates = pd.date_range(start='2024-01-01 00:00', end='2024-01-01 02:00', freq='60min')
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        self.index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])

        self.test_df = pd.DataFrame(
            index=self.index,
            data={
                'close': np.random.randn(len(self.index)),
                'volume': np.abs(np.random.randn(len(self.index))),
                'logret_60': np.random.randn(len(self.index))
            }
        )

    @patch('lib.server.server_calcs.ServerHorizonCalcs')
    @patch('lib.server.server_calcs.Forecasts')
    @patch('lib.server.server_calcs.Calcs')
    @patch('lib.server.server_calcs.LiveBars')
    @patch('lib.server.server_calcs.DataLoader')
    def test_make_server_horizon_calc(self, mock_data_loader_class, mock_live_bars_class,
                                     mock_calcs_class, mock_forecasts_class,
                                     mock_server_horizon_calcs_class):  # pylint: disable=too-many-arguments,unused-argument
        """Test make_server_horizon_calc creates and stores horizon calc."""
        server_calcs = ServerCalcs(
            config=self.config,
            eod_today=dt.now(),
            feature_horizons=[60],
            short_term_horizons=[60],
            model_horizons=[60],
            feature_flds_to_load={}
        )

        server_calcs.make_server_horizon_calc(horizon=60, df=self.test_df)

        # Check ServerHorizonCalcs was created
        mock_server_horizon_calcs_class.assert_called_once_with(
            config=self.config,
            horizon=60,
            features_df=self.test_df,
            server_horizon_calcs_dir_manager=server_calcs.dir_manager
        )

        # Check it was stored
        self.assertIn(60, server_calcs.horizon_calcs)

    @patch('lib.server.server_calcs.Forecasts')
    @patch('lib.server.server_calcs.Calcs')
    @patch('lib.server.server_calcs.LiveBars')
    @patch('lib.server.server_calcs.DataLoader')
    def test_get_horizon_calc(self, mock_data_loader_class, mock_live_bars_class,
                             mock_calcs_class, mock_forecasts_class):  # pylint: disable=unused-argument
        """Test get_horizon_calc returns correct instance or raises error."""
        server_calcs = ServerCalcs(
            config=self.config,
            eod_today=dt.now(),
            feature_horizons=[60],
            short_term_horizons=[60],
            model_horizons=[60],
            feature_flds_to_load={}
        )

        # Add a mock horizon calc
        mock_hc = MagicMock()
        server_calcs.horizon_calcs[60] = mock_hc

        # Test successful get
        result = server_calcs.get_horizon_calc(60)
        self.assertEqual(result, mock_hc)

        # Test error for missing horizon
        with self.assertRaises(Exception):
            server_calcs.get_horizon_calc(120)

    @patch('lib.server.server_calcs.Forecasts')
    @patch('lib.server.server_calcs.Calcs')
    @patch('lib.server.server_calcs.LiveBars')
    @patch('lib.server.server_calcs.DataLoader')
    def test_get_last_processed_ts(self, mock_data_loader_class, mock_live_bars_class,
                                   mock_calcs_class, mock_forecasts_class):  # pylint: disable=unused-argument
        """Test get_last_processed_ts returns timestamp from 1-minute horizon."""
        server_calcs = ServerCalcs(
            config=self.config,
            eod_today=dt.now(),
            feature_horizons=[1, 60],
            short_term_horizons=[60],
            model_horizons=[60],
            feature_flds_to_load={}
        )

        # Mock 1-minute horizon calc
        mock_hc_1 = MagicMock()
        mock_hc_1.last_processed_ts = pd.Timestamp('2024-01-01 02:00:00')
        server_calcs.horizon_calcs[1] = mock_hc_1

        result = server_calcs.get_last_processed_ts()
        self.assertEqual(result, pd.Timestamp('2024-01-01 02:00:00'))

    @patch('lib.server.server_calcs.Forecasts')
    @patch('lib.server.server_calcs.Calcs')
    @patch('lib.server.server_calcs.LiveBars')
    @patch('lib.server.server_calcs.DataLoader')
    def test_get_horizon_df(self, mock_data_loader_class, mock_live_bars_class,
                           mock_calcs_class, mock_forecasts_class):  # pylint: disable=unused-argument
        """Test get_horizon_df returns features DataFrame for horizon."""
        server_calcs = ServerCalcs(
            config=self.config,
            eod_today=dt.now(),
            feature_horizons=[60],
            short_term_horizons=[60],
            model_horizons=[60],
            feature_flds_to_load={}
        )

        # Mock horizon calc
        mock_hc = MagicMock()
        mock_hc.features_df = self.test_df
        server_calcs.horizon_calcs[60] = mock_hc

        result = server_calcs.get_horizon_df(60)
        pd.testing.assert_frame_equal(result, self.test_df)

    @patch('lib.server.server_calcs.Forecasts')
    @patch('lib.server.server_calcs.Calcs')
    @patch('lib.server.server_calcs.LiveBars')
    @patch('lib.server.server_calcs.DataLoader')
    def test_get_latest_features(self, mock_data_loader_class, mock_live_bars_class,
                                mock_calcs_class, mock_forecasts_class):  # pylint: disable=unused-argument
        """Test get_latest_features combines features from all horizons."""
        server_calcs = ServerCalcs(
            config=self.config,
            eod_today=dt.now(),
            feature_horizons=[1, 60],
            short_term_horizons=[60],
            model_horizons=[60],
            feature_flds_to_load={}
        )

        # Create test data for different horizons
        last_ts = pd.Timestamp('2024-01-01 02:00:00')

        # Mock 1-minute horizon
        mock_hc_1 = MagicMock()
        mock_hc_1.last_processed_ts = last_ts
        df_1 = pd.DataFrame(index=self.index, data={'close': [1, 2, 3, 4, 5, 6], 'volume': [10, 20, 30, 40, 50, 60]})
        mock_hc_1.features_df = df_1
        server_calcs.horizon_calcs[1] = mock_hc_1

        # Mock 60-minute horizon
        mock_hc_60 = MagicMock()
        df_60_data = {'logret_60': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 
                      'beta_60': [1, 2, 3, 4, 5, 6]}
        df_60 = pd.DataFrame(index=self.index, data=df_60_data)
        mock_hc_60.features_df = df_60
        server_calcs.horizon_calcs[60] = mock_hc_60

        result = server_calcs.get_latest_features()

        # Should only have data for last timestamp
        self.assertEqual(len(result), 2)  # 2 symbols at last timestamp
        self.assertIn('close', result.columns)
        self.assertIn('volume', result.columns)
        self.assertIn('logret_60', result.columns)
        self.assertIn('beta_60', result.columns)

    @patch('lib.server.server_calcs.log_mem_usage')
    @patch('lib.server.server_calcs.check_df')
    @patch('lib.server.server_calcs.concat')
    @patch('lib.server.server_calcs.Forecasts')
    @patch('lib.server.server_calcs.Calcs')
    @patch('lib.server.server_calcs.LiveBars')
    @patch('lib.server.server_calcs.DataLoader')
    def test_update_horizon_rolling_fields_1min(self, mock_data_loader_class, mock_live_bars_class,
                                                mock_calcs_class, mock_forecasts_class,
                                                mock_concat, mock_check_df, mock_log_mem):  # pylint: disable=too-many-arguments,unused-argument
        """Test update_horizon_rolling_fields for 1-minute horizon."""
        server_calcs = ServerCalcs(
            config=self.config,
            eod_today=dt.now(),
            feature_horizons=[1],
            short_term_horizons=[],
            model_horizons=[1],
            feature_flds_to_load={}
        )

        # Mock horizon calc
        mock_hc = MagicMock()
        server_calcs.horizon_calcs[1] = mock_hc

        # Call with 1-minute horizon
        server_calcs.update_horizon_rolling_fields(
            horizon=1,
            new_bars_df=self.test_df,
            previous_bars_ts=pd.Timestamp('2024-01-01 01:00:00'),
            existing_flds=['close', 'volume'],
            one_min_bars_df_unstacked=self.test_df.unstack()
        )

        # For 1-minute, should just update_df
        mock_hc.update_df.assert_called_once()
        pd.testing.assert_frame_equal(mock_hc.update_df.call_args[0][0], self.test_df)

    @patch('lib.server.server_calcs.Pool')
    @patch('lib.server.server_calcs.Forecasts')
    @patch('lib.server.server_calcs.Calcs')
    @patch('lib.server.server_calcs.LiveBars')
    @patch('lib.server.server_calcs.DataLoader')
    def test_update_rolling_fields(self, mock_data_loader_class, mock_live_bars_class,
                                  mock_calcs_class, mock_forecasts_class, mock_pool_class):  # pylint: disable=unused-argument
        """Test update_rolling_fields processes all horizons in parallel."""
        server_calcs = ServerCalcs(
            config=self.config,
            eod_today=dt.now(),
            feature_horizons=[1, 60],
            short_term_horizons=[60],
            model_horizons=[60],
            feature_flds_to_load={}
        )

        # Mock horizon calcs
        mock_hc_1 = MagicMock()
        mock_hc_1.features_df = self.test_df
        server_calcs.horizon_calcs[1] = mock_hc_1
        server_calcs.horizon_calcs[60] = MagicMock()

        # Mock pool
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        # Call update_rolling_fields
        server_calcs.update_rolling_fields(new_bars_df=self.test_df)

        # Check pool was used correctly
        mock_pool_class.assert_called_once_with(2)  # 2 horizons
        mock_pool.starmap.assert_called_once()
        mock_pool.close.assert_called_once()
        mock_pool.join.assert_called_once()

    @patch('lib.server.server_calcs.get_min_max_ts')
    @patch('lib.server.server_calcs.carry_forward')
    @patch('lib.server.server_calcs.concat')
    @patch('lib.server.server_calcs.load_news')
    @patch('lib.server.server_calcs.today_date')
    @patch('lib.server.server_calcs.Forecasts')
    @patch('lib.server.server_calcs.Calcs')
    @patch('lib.server.server_calcs.LiveBars')
    @patch('lib.server.server_calcs.DataLoader')
    def test_update_bars_no_new_data(self, mock_data_loader_class, mock_live_bars_class,
                                    mock_calcs_class, mock_forecasts_class,
                                    mock_today_date, mock_load_news, mock_concat,
                                    mock_carry_forward, mock_get_min_max_ts):  # pylint: disable=too-many-arguments,unused-argument,too-many-locals
        """Test update_bars returns False when no new data."""
        server_calcs = ServerCalcs(
            config=self.config,
            eod_today=dt.now(),
            feature_horizons=[1],
            short_term_horizons=[],
            model_horizons=[1],
            feature_flds_to_load={}
        )

        # Mock horizon calc
        mock_hc = MagicMock()
        mock_hc.get_max_ts.return_value = pd.Timestamp('2024-01-01 02:00:00')
        server_calcs.horizon_calcs[1] = mock_hc

        # Mock live bars returning None
        mock_live_bars = MagicMock()
        mock_live_bars.load_live_bars.return_value = None
        server_calcs.live_bars = mock_live_bars

        result = server_calcs.update_bars()
        self.assertFalse(result)

    @patch('lib.server.server_calcs.Pool')
    @patch('lib.server.server_calcs.unique_list')
    @patch('lib.server.server_calcs.Forecasts')
    @patch('lib.server.server_calcs.Calcs')
    @patch('lib.server.server_calcs.LiveBars')
    @patch('lib.server.server_calcs.DataLoader')
    def test_calculate_features(self, mock_data_loader_class, mock_live_bars_class,
                               mock_calcs_class, mock_forecasts_class,
                               mock_unique_list, mock_pool_class):  # pylint: disable=too-many-arguments,unused-argument
        """Test calculate_features coordinates feature calculation."""
        mock_unique_list.side_effect = lambda x: list(set(x))

        server_calcs = ServerCalcs(
            config=self.config,
            eod_today=dt.now(),
            feature_horizons=[15, 60, 1440],
            short_term_horizons=[15, 60],
            model_horizons=[15, 60],
            feature_flds_to_load={60: ['beta_60']}
        )

        # Mock pool
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        # Call calculate_features
        server_calcs.calculate_features(st_only=False)

        # Check pool was created and used
        mock_pool.starmap.assert_called_once()
        call_args = mock_pool.starmap.call_args[0]
        self.assertEqual(call_args[0], server_calcs.calculate_horizon_feature)

        # Check carry forward map was built correctly
        # Should have entries for 1, 15, 60, 1440
        horizons_in_calls = [args[0] for args in call_args[1]]
        self.assertIn(1, horizons_in_calls)
        self.assertIn(1440, horizons_in_calls)

    @patch('lib.server.server_calcs.Pool')
    @patch('lib.server.server_calcs.Forecasts')
    @patch('lib.server.server_calcs.Calcs')
    @patch('lib.server.server_calcs.LiveBars')
    @patch('lib.server.server_calcs.DataLoader')
    def test_calculate_models(self, mock_data_loader_class, mock_live_bars_class,
                             mock_calcs_class, mock_forecasts_class, mock_pool_class):  # pylint: disable=unused-argument
        """Test calculate_models processes model horizons in parallel."""
        server_calcs = ServerCalcs(
            config=self.config,
            eod_today=dt.now(),
            feature_horizons=[15, 60],
            short_term_horizons=[15],
            model_horizons=[15, 60],
            feature_flds_to_load={}
        )

        # Mock pool
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        # Test full model calculation
        server_calcs.calculate_models(st_only=False)

        # Check pool was used with all model horizons
        mock_pool_class.assert_called_with(2)  # 2 model horizons

        # Test short-term only
        mock_pool_class.reset_mock()
        server_calcs.calculate_models(st_only=True)

        # Check pool was used with only short-term horizons
        mock_pool_class.assert_called_with(1)  # 1 short-term horizon

    @patch('lib.server.server_calcs.calculate_horizon_alphas')
    @patch('lib.server.server_calcs.log_and_raise')
    @patch('lib.server.server_calcs.Forecasts')
    @patch('lib.server.server_calcs.Calcs')
    @patch('lib.server.server_calcs.LiveBars')
    @patch('lib.server.server_calcs.DataLoader')
    def test_calculate_alphas(self, mock_data_loader_class, mock_live_bars_class,
                             mock_calcs_class, mock_forecasts_class,
                             mock_log_and_raise, mock_calc_horizon_alphas):  # pylint: disable=too-many-arguments,unused-argument
        """Test calculate_alphas generates alpha signals."""
        server_calcs = ServerCalcs(
            config=self.config,
            eod_today=dt.now(),
            feature_horizons=[60],
            short_term_horizons=[60],
            model_horizons=[60],
            feature_flds_to_load={}
        )

        # Mock horizon calc for getting last processed ts
        mock_hc_1 = MagicMock()
        mock_hc_1.last_processed_ts = pd.Timestamp('2024-01-01 02:00:00')
        server_calcs.horizon_calcs[1] = mock_hc_1

        # Mock get_latest_features
        with patch.object(server_calcs, 'get_latest_features') as mock_get_latest:
            mock_get_latest.return_value = self.test_df

            # Mock forecasts
            mock_forecasts = MagicMock()
            mock_alphas_df = self.test_df.copy()
            mock_alphas_df['alpha_60'] = 0.1
            mock_forecasts.compute_model_alphas_for_server.return_value = (mock_alphas_df, ['alpha_60'])
            server_calcs.forecasts = mock_forecasts

            # Mock horizon alpha calculation
            mock_calc_horizon_alphas.return_value = (mock_alphas_df, ['combined_alpha'])

            # Call calculate_alphas
            result_df, result_cols = server_calcs.calculate_alphas(horizons=[60])

            # Check results
            pd.testing.assert_frame_equal(result_df, mock_alphas_df)
            self.assertIn('alpha_60', result_cols)
            self.assertIn('combined_alpha', result_cols)

    @patch('lib.server.server_calcs.log_mem_usage')
    @patch('lib.server.server_calcs.Forecasts')
    @patch('lib.server.server_calcs.Calcs')
    @patch('lib.server.server_calcs.LiveBars')
    @patch('lib.server.server_calcs.DataLoader')
    def test_update(self, mock_data_loader_class, mock_live_bars_class,
                   mock_calcs_class, mock_forecasts_class, mock_log_mem):  # pylint: disable=unused-argument
        """Test update orchestrates full update cycle."""
        server_calcs = ServerCalcs(
            config=self.config,
            eod_today=dt.now(),
            feature_horizons=[60],
            short_term_horizons=[60],
            model_horizons=[60],
            feature_flds_to_load={}
        )

        # Mock horizon calc
        mock_hc = MagicMock()
        server_calcs.horizon_calcs[60] = mock_hc

        # Mock methods
        with patch.object(server_calcs, 'update_bars') as mock_update_bars:
            with patch.object(server_calcs, 'calculate_features') as mock_calc_features:
                with patch.object(server_calcs, 'calculate_models') as mock_calc_models:
                    # Test successful update
                    mock_update_bars.return_value = True
                    result = server_calcs.update(st_only=False)

                    self.assertTrue(result)
                    mock_update_bars.assert_called_once()
                    mock_calc_features.assert_called_once_with(st_only=False)
                    mock_calc_models.assert_called_once_with(st_only=False)
                    mock_hc.update_last_processed_ts.assert_called_once()

                    # Test no new data
                    mock_update_bars.return_value = False
                    result = server_calcs.update(st_only=True)

                    self.assertFalse(result)


class TestServerCalcsIntegration(unittest.TestCase):
    """Integration tests for ServerCalcs."""
    # pylint: disable=unused-argument

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'OPT_HORIZON': 1440,
            'ST_RISK_HORIZON': 15,
            'VOLUME_BUCKET_MINS': 60,
            'NEWS_SIMILARITY_THRESHOLD': 0.9,
            'REOPTIMIZE_INTERVAL_MINS': 120
        }

    @patch('lib.server.server_calcs.ServerHorizonCalcs')
    @patch('lib.server.server_calcs.log_mem_usage')
    @patch('lib.server.server_calcs.Pool')
    @patch('lib.server.server_calcs.Forecasts')
    @patch('lib.server.server_calcs.Calcs')
    @patch('lib.server.server_calcs.LiveBars')
    @patch('lib.server.server_calcs.DataLoader')
    def test_full_workflow(self, mock_data_loader_class, mock_live_bars_class,
                          mock_calcs_class, mock_forecasts_class, mock_pool_class,
                          mock_log_mem, mock_server_horizon_calcs_class):  # pylint: disable=too-many-arguments,unused-argument,too-many-locals
        """Test full workflow from initialization through alpha calculation."""
        # Create server calcs
        server_calcs = ServerCalcs(
            config=self.config,
            eod_today=dt(2024, 1, 1, 23, 59, 59),
            feature_horizons=[1, 60],
            short_term_horizons=[60],
            model_horizons=[60],
            feature_flds_to_load={60: ['beta_60']}
        )

        # Create test data
        dates = pd.date_range(start='2024-01-01 00:00', end='2024-01-01 01:00', freq='60min')
        symbols = ['BTCUSDT_binance-futures']
        index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])
        test_df = pd.DataFrame(index=index, data={'close': [100, 101], 'volume': [1000, 1100]})

        # Mock initial data loading
        server_calcs.make_server_horizon_calc(1, test_df)
        server_calcs.make_server_horizon_calc(60, test_df)

        # Mock pool for parallel processing
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        # Test that methods can be called in sequence
        with patch.object(server_calcs, 'update_bars') as mock_update_bars:
            mock_update_bars.return_value = True

            # Run update
            result = server_calcs.update(st_only=False)
            self.assertTrue(result)

            # Verify horizon calcs were created
            self.assertEqual(mock_server_horizon_calcs_class.call_count, 2)  # 1 and 60 minute horizons


if __name__ == '__main__':
    unittest.main()
