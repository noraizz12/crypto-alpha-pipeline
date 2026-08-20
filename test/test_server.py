"""Unit tests for the AlphaServer module."""
# pylint: disable=unused-argument,protected-access

import unittest
from datetime import datetime as dt
from datetime import timedelta as td, timezone
from unittest.mock import Mock, patch
import pandas as pd
from slack_sdk.webhook.async_client import AsyncWebhookClient

from lib.server import AlphaServer
from lib.util.directory import DirectoryManager


def get_test_config():
    """Get a complete test configuration with all required fields."""
    return {
        'OPT_HORIZON': 1440,
        'SHORT_TERM_MODEL_HORIZONS': [15, 60],
        'REOPTIMIZE_INTERVAL_MINS': 120,
        'VOLUME_BUCKET_MINS': 60,
        'MAX_PORTFOLIO_NOTIONAL': 10000000,
        'WAVE_INTERVAL_MINS': 1,
        'NEWS_SIMILARITY_THRESHOLD': 0.9,
        'DEFAULT_FEATURE_LOOKBACK_PERIODS': 90,
        'SYMBOL_UNIVERSE': ['BTC', 'ETH'],
        'FCASTS': {
            '1440': {
                'models': [{'name': 'test_model', 'weight': 1.0, 'lags': 1}],
                'features': ['logret_1440_lz', 'dvolume_1440_lz']
            },
            '60': {
                'models': [{'name': 'test_model_st', 'weight': 1.0, 'lags': 1}],
                'features': ['logret_60_lz', 'dvolume_60_lz']
            }
        },
        'REOPT_TIMES': ['00:00', '12:00'],
        'OPT_OFFSET_MINS': 2,
        'DYNAMIC_UNIVERSE': True,
        'ADV_LOOKBACK_DAYS': 45,
        'MIN_ADVP_FEATUREABLE': 2.5e7,
        'MIN_ADVP_FITTABLE': 1.5e7,
        'MIN_ADVP_TRADEABLE': 1.5e7,
        'MIN_ADVP_EXPANDABLE': 5.5e7,
        'MAX_MARKET_CAP_EXPANDABLE_FRAC': 1.0,
        'MAX_POSITION_VOLUME_FRACTION': 0.05,
        'BETA_LOOKBACK_PERIODS': 90,
        'MAX_MOVE_FILTER': 2.5,
        'MAX_DVOL_SIGMA': 2.0,
        'MIN_FUNDING_RATE': -0.005,
        'MAX_POSITION_PCT': 0.04,
        'EXCLUDE_NON_ALPHA_TRADES': False,
        'RISK_FLD': 'logret_HORIZON_trstd',
        'FILTER_DELISTING': True,
        'DELISTING_BUFFER_DAYS': 6,
        'FEATURE_SIGMA_BOUND': 5,
        'OLD_POSITION_DAYS': 30,
        'OLD_POSITION_RISK_MULT': 3.5,
        # Portfolio optimization parameters
        'KAPPA': 1e-7,
        'GAMMA': 2.5e-4,
        'MAX_ALPHA': 0.05,
        'MAX_PORTFOLIO_NOTIONAL_SLACK': 0.1,
        'MAX_TRADING_BIAS': 0.1,
        'BASE_AGGRESSION': 0.0009,
        'MIN_POSITION': 1000.0,
        'MAX_TRADE_DOLLARS': 800000,
        'MIN_TRADE_DOLLARS': 100.0,
        'MAX_TURNOVER': 0.8,
        'EXCHANGE_FEES': 0.00005,
        'FACTOR_SIGMAS': {'dollar_exposure': 0.10},
        'EXPOSURE_LIMITS': {'dollar_exposure': 0.05},
        'SCALE_ALPHA_OPT': False,
        'CENTER_ALPHA_OPT': True,
        'CONST_NOTIONAL': False,
        'CORRECT_TARGET_IMBALANCE': False,
        'ALPHA_MULT': 1.0,
        'ALPHA_TILT': 0.0,
        'MAX_VOLUME_FRACTION_PARTICIPATION': 0.02,
        'HORIZON_MODEL_FACTOR': 1.0,
        # CVXPY optimization parameters
        'OPT_RHO': 0.3,
        'OPT_RHO_INTERVAL': 10,
        'OPT_ALPHA': 1.6,
        # Forecasts parameters
        'MIN_TSTAT': 1.5,
        'ENFORCE_REV_MOM': True,
        # Missing liquidity thresholds
        'MIN_ADVP_PRICEABLE': 2.5e7,
        'MIN_ADVP_FEATUREABLE': 2.5e7,
        'FEATUREABLE_HIST_PERIODS': 30,
        'MIN_ADVP_FITTABLE': 1.5e7,
        'MIN_ADVP_TRADEABLE': 1.5e7,
        'MIN_ADVP_EXPANDABLE': 5.5e7,
        'MAX_MARKET_CAP_EXPANDABLE_FRAC': 1.0,
        'ADV_LOOKBACK_DAYS': 45,
        'BETA_LOOKBACK_PERIODS': 90,
        'MAX_POSITION_VOLUME_FRACTION': 0.05,
        'MAX_DVOL_SIGMA': 2.0,
        'MIN_FUNDING_RATE': -0.005,
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
        'FILTER_DELISTING': True,
        'DELISTING_BUFFER_DAYS': 6,
        'FEATURE_SIGMA_BOUND': 5,
        'OLD_POSITION_DAYS': 30,
        'OLD_POSITION_RISK_MULT': 3.5,
        'MAX_MOVE_FILTER': 2.5,
        'RISK_FLD': 'logret_HORIZON_trstd',
        'REOPT_TIMES': [],
        'OPT_OFFSET_MINS': 2,
        # Features configuration
        'FEATURES': {
            '60': {
                'prod': ['logret_60_lz', 'dvolume_60_lz'],
                'nonprod': []
            },
            '1440': {
                'prod': ['logret_1440_lz', 'dvolume_1440_lz'],
                'nonprod': []
            }
        }
    }


class TestAlphaServerInit(unittest.TestCase):
    """Test AlphaServer initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_test_config()
        self.universe_df = pd.DataFrame({
            'fittable': [True, True],
            'tradeable': [True, True],
            'expandable': [True, True],
            'priceable': [True, True],
            'advp': [1000000, 2000000],
            'marketcap': [1e9, 2e9],
            'symbol_venue': ['BTC_binance', 'ETH_binance']
        })

    def _setup_common_mocks(self, mock_data_loader, mock_data_loader_instance=None):
        """Set up common mocks for data loader."""
        if mock_data_loader_instance is None:
            mock_data_loader_instance = Mock()
        mock_data_loader_instance.load_universe_df.return_value = self.universe_df
        mock_data_loader_instance.load_universe_symbol_venues.return_value = ['BTC_binance', 'ETH_binance']
        mock_data_loader.return_value = mock_data_loader_instance
        return mock_data_loader_instance

    @patch('lib.server.server.get_config')
    @patch('lib.server.server.DataLoader')
    @patch('lib.portfolio_optimization.PortfolioOptimizer')
    @patch('lib.server.server.Forecasts')
    @patch('lib.server.server.Calcs')
    @patch('lib.data.loaders.load_positions')
    @patch('lib.server.server.AlphaServer.setup')
    def test_init_default_parameters(self, mock_setup, mock_load_positions, mock_calcs, mock_forecasts,
                                   mock_optimizer, mock_data_loader, mock_get_config):
        """Test initialization with default parameters."""
        # Ensure get_config always returns test config
        mock_get_config.return_value = (None, self.config)
        mock_load_positions.return_value = pd.DataFrame({'qty': [1]},
                                                       index=pd.MultiIndex.from_tuples([('BTC_binance',)], names=['symbol_venue']))

        # Mock data loader to avoid loading actual config
        self._setup_common_mocks(mock_data_loader)

        # Create server with minimal parameters - config should come from mock
        server = AlphaServer()

        # Verify the mocked config was used
        mock_get_config.assert_called()

        # Check key attributes initialized
        self.assertFalse(server.debug)
        self.assertFalse(server.trade_on_start)
        self.assertIsNotNone(server.universe)
        self.assertEqual(server.model_horizons, [60, 1440])
        self.assertEqual(server.opt_horizon, 1440)
        self.assertEqual(server.short_term_horizons, [60])
        
    @patch('lib.server.server.get_config')
    @patch('lib.server.server.DataLoader')
    @patch('lib.portfolio_optimization.PortfolioOptimizer')
    @patch('lib.server.server.Forecasts')
    @patch('lib.server.server.Calcs')
    @patch('lib.data.loaders.load_positions')
    @patch('lib.server.server.AlphaServer.setup')
    def test_init_with_custom_parameters(self, mock_setup, mock_load_positions, mock_calcs, mock_forecasts,
                                        mock_optimizer, mock_data_loader, mock_get_config):
        """Test initialization with custom parameters."""
        mock_get_config.return_value = (None, self.config)
        mock_load_positions.return_value = pd.DataFrame({'qty': [1]},
                                                       index=pd.MultiIndex.from_tuples([('BTC_binance',)], names=['symbol_venue']))

        # Mock data loader
        self._setup_common_mocks(mock_data_loader)

        # Custom parameters
        eod_today = dt.now(timezone.utc)
        slack_client = Mock(spec=AsyncWebhookClient)
        dir_manager = Mock(spec=DirectoryManager)
        dir_manager.RAW_TARGET_DIR = '/test/raw'
        dir_manager.TARGET_DIR = '/test/target'
        dir_manager.POSITION_DIR = '/test/pos'
        dir_manager.UTIL_DIR = '/test/util'
        dir_manager.NEWS_DIR = '/test/news'
        dir_manager.FITS_DIR = '/test/fits'
        dir_manager.PROD_FITS_DIR = '/test/prod_fits'
        
        server = AlphaServer(
            eod_today=eod_today,
            config_file='test_config.json',
            slack_client=slack_client,
            debug=True,
            trade_on_start=True,
            trailing_days_to_load=30,
            server_dir_manager=dir_manager,
            latest_alpha_dir='/test/alpha',
            raw_target_dir='/test/raw_custom',
            target_dir='/test/target_custom'
        )
        
        self.assertTrue(server.debug)
        self.assertTrue(server.trade_on_start)
        # bars_type no longer exists
        self.assertEqual(server.trailing_lookback_periods[1440], 30)
        self.assertEqual(server.raw_target_dir, '/test/raw_custom')
        self.assertEqual(server.target_dir, '/test/target_custom')
        self.assertEqual(server.latest_alpha_dir, '/test/alpha')
        

class TestAlphaServerMethods(unittest.TestCase):
    """Test AlphaServer methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = get_test_config()
        self.universe_df = pd.DataFrame({
            'fittable': [True, True],
            'tradeable': [True, True],
            'expandable': [True, True],
            'priceable': [True, True],
            'advp': [1000000, 2000000],
            'marketcap': [1e9, 2e9],
            'symbol_venue': ['BTC_binance', 'ETH_binance']
        })
        
    def _setup_common_mocks(self, mock_data_loader, mock_data_loader_instance=None):
        """Set up common mocks for data loader."""
        if mock_data_loader_instance is None:
            mock_data_loader_instance = Mock()
        mock_data_loader_instance.load_universe_df.return_value = self.universe_df
        mock_data_loader_instance.load_universe_symbol_venues.return_value = ['BTC_binance', 'ETH_binance']
        mock_data_loader.return_value = mock_data_loader_instance
        return mock_data_loader_instance
        
    @patch('lib.server.server.get_config')
    @patch('lib.server.server.DataLoader')
    @patch('lib.portfolio_optimization.PortfolioOptimizer')
    @patch('lib.server.server.Forecasts')
    @patch('lib.calcs.calcs.Calcs')
    @patch('lib.data.loaders.load_positions')
    @patch('lib.server.server.AlphaServer.setup')
    def test_message_methods(self, mock_setup, mock_load_positions, mock_calcs, mock_forecasts,
                           mock_optimizer, mock_data_loader, mock_get_config):
        """Test messaging methods."""
        mock_get_config.return_value = (None, self.config)
        mock_load_positions.return_value = None

        # Mock data loader using the helper
        self._setup_common_mocks(mock_data_loader)

        slack_client = Mock()
        server = AlphaServer(slack_client=slack_client, debug=True)  # Use debug=True to avoid slack sending
        
        # Test regular message (debug mode logs but doesn't send to slack)
        with patch('lib.server.server.logger') as mock_logger:
            server._message("Test message")
            mock_logger.info.assert_called_with("[SLACK] Test message")
            
        # Test error message
        with patch('lib.server.server.logger') as mock_logger:
            server._message("Error message", error=True)
            mock_logger.error.assert_called_with("[SLACK] Error message", key="server message")
            
        # Test DataFrame message
        test_df = pd.DataFrame({'col1': range(10), 'col2': range(10, 20)})
        with patch.object(server, '_message') as mock_message:
            server._message_df(test_df, "Test DF")
            mock_message.assert_called()
            
    @patch('lib.server.server.filter_universe')
    @patch('lib.server.server.get_config')
    @patch('lib.server.server.DataLoader')
    @patch('lib.portfolio_optimization.PortfolioOptimizer')
    @patch('lib.server.server.Forecasts')
    @patch('lib.server.server.Calcs')
    @patch('lib.server.server.load_positions')  # Patch where it's imported
    @patch('lib.server.server.AlphaServer.setup')
    def test_update_universe(self, mock_setup, mock_load_positions, mock_calcs_class, mock_forecasts,
                            mock_optimizer, mock_data_loader, mock_get_config, mock_filter_universe):
        """Test universe update functionality."""
        mock_get_config.return_value = (None, self.config)
        mock_load_positions.return_value = None

        # Mock data loader using the helper
        self._setup_common_mocks(mock_data_loader)

        # Mock calcs instance
        mock_calcs = Mock()
        mock_calcs_class.return_value = mock_calcs

        # Mock the filter_universe function
        mock_filter_universe.return_value = ['BTC_binance']
        
        server = AlphaServer()
        server.yesterday_date = dt.now().date() - td(days=1)
        server.pos_dir = '/test/pos'  # Use non-POSITION_DIR
        
        # Create test bars DataFrame with proper MultiIndex
        ts = pd.Timestamp('2023-01-01 12:00:00')
        bars_df = pd.DataFrame({
            'close_mid': [100, 200]
        }, index=pd.MultiIndex.from_tuples(
            [(ts, 'BTC_binance'), (ts, 'ETH_binance')],
            names=['ts', 'symbol_venue']
        ))
        
        # Update universe
        server.update_universe(bars_df)
        
        # Check filter_universe was called
        mock_filter_universe.assert_called_once()
        self.assertEqual(server.universe, ['BTC_binance'])
        
    @patch('lib.server.server.get_config')
    @patch('lib.server.server.DataLoader')
    @patch('lib.portfolio_optimization.PortfolioOptimizer')
    @patch('lib.server.server.Forecasts')
    @patch('lib.server.server.Calcs')
    @patch('lib.data.loaders.load_positions')
    @patch('lib.server.server.AlphaServer.setup')
    def test_get_position_age(self, mock_setup, mock_load_positions, mock_calcs, mock_forecasts,
                             mock_optimizer, mock_data_loader, mock_get_config):
        """Test position age calculation."""
        mock_get_config.return_value = (None, self.config)
        mock_load_positions.return_value = None

        # Mock data loader using the helper
        self._setup_common_mocks(mock_data_loader)

        server = AlphaServer()
        server.pos_dir = '/test/pos'  # Non-POSITION_DIR to skip PnL calculation
        
        # Test with position_age already in DataFrame
        df_with_age = pd.DataFrame({'position_age': [5, 10]})
        result = server.get_position_age(df_with_age)
        self.assertIn('position_age', result.columns)
        
        # Test without position_age
        df_without_age = pd.DataFrame({'symbol': ['BTC', 'ETH']})
        result = server.get_position_age(df_without_age)
        self.assertIn('position_age', result.columns)
        self.assertTrue((result['position_age'] == 0).all())

    @patch('lib.server.server.get_config')
    @patch('lib.server.server.DataLoader')
    @patch('lib.portfolio_optimization.PortfolioOptimizer')
    @patch('lib.server.server.Forecasts')
    @patch('lib.server.server.Calcs')
    @patch('lib.data.loaders.load_positions')
    @patch('lib.server.server.AlphaServer.setup')
    def test_get_position_age_with_cached_values(self, mock_setup, mock_load_positions, mock_calcs,
                                                  mock_forecasts, mock_optimizer, mock_data_loader,
                                                  mock_get_config):
        """Test position age uses cached values from BinancePnl calculation.

        This tests Sean's requirement: position_age should be calculated once at
        server startup using BinancePnl and cached in self.position_age_df.
        The get_position_age() method should join this cached data with incoming
        dataframes to apply the OLD_POSITION_RISK_MULT (3.5x) for positions > 30 days.
        """
        mock_get_config.return_value = (None, self.config)
        mock_load_positions.return_value = None

        # Mock data loader using the helper
        self._setup_common_mocks(mock_data_loader)

        server = AlphaServer()

        # Simulate cached position_age from BinancePnl calculation (done in setup())
        # This is what would be calculated from historical fill data
        server.position_age_df = pd.DataFrame({
            'position_age': [5, 31, 15]  # BTC=5 days, ETH=31 days (old!), SOL=15 days
        }, index=pd.Index(['BTC_binance', 'ETH_binance', 'SOL_binance'], name='symbol_venue'))

        # Test that get_position_age joins cached values correctly
        df = pd.DataFrame({
            'close_mid': [50000.0, 3000.0, 100.0]
        }, index=pd.Index(['BTC_binance', 'ETH_binance', 'SOL_binance'], name='symbol_venue'))

        result = server.get_position_age(df)

        # Verify position_age column was added from cache
        self.assertIn('position_age', result.columns)
        self.assertEqual(result.loc['BTC_binance', 'position_age'], 5)
        self.assertEqual(result.loc['ETH_binance', 'position_age'], 31)  # Old position!
        self.assertEqual(result.loc['SOL_binance', 'position_age'], 15)

        # Verify old position (31 days) exceeds OLD_POSITION_DAYS threshold (30)
        old_position_days = self.config['OLD_POSITION_DAYS']
        old_positions = result[result['position_age'] > old_position_days]
        self.assertEqual(len(old_positions), 1)
        self.assertIn('ETH_binance', old_positions.index)

    @patch('lib.server.server.get_config')
    @patch('lib.server.server.DataLoader')
    @patch('lib.portfolio_optimization.PortfolioOptimizer')
    @patch('lib.server.server.Forecasts')
    @patch('lib.server.server.Calcs')
    @patch('lib.data.loaders.load_positions')
    @patch('lib.server.server.AlphaServer.setup')
    def test_get_position_age_with_missing_symbols(self, mock_setup, mock_load_positions, mock_calcs,
                                                    mock_forecasts, mock_optimizer, mock_data_loader,
                                                    mock_get_config):
        """Test position age handles symbols not in cache (new positions)."""
        mock_get_config.return_value = (None, self.config)
        mock_load_positions.return_value = None

        self._setup_common_mocks(mock_data_loader)

        server = AlphaServer()

        # Cache only has BTC, but we query with BTC and a new symbol (DOGE)
        server.position_age_df = pd.DataFrame({
            'position_age': [20]
        }, index=pd.Index(['BTC_binance'], name='symbol_venue'))

        df = pd.DataFrame({
            'close_mid': [50000.0, 0.1]
        }, index=pd.Index(['BTC_binance', 'DOGE_binance'], name='symbol_venue'))

        result = server.get_position_age(df)

        # BTC should have cached age, DOGE should default to 0 (new position)
        self.assertEqual(result.loc['BTC_binance', 'position_age'], 20)
        self.assertEqual(result.loc['DOGE_binance', 'position_age'], 0)

    @patch('lib.server.server.get_config')
    @patch('lib.server.server.DataLoader')
    @patch('lib.portfolio_optimization.PortfolioOptimizer')
    @patch('lib.server.server.Forecasts')
    @patch('lib.server.server.Calcs')
    @patch('lib.server.server.load_positions')  # Patch where it's imported
    @patch('lib.server.server.AlphaServer.setup')
    def test_set_positions(self, mock_setup, mock_load_positions, mock_calcs, mock_forecasts,
                          mock_optimizer, mock_data_loader, mock_get_config):
        """Test position loading and validation."""
        mock_get_config.return_value = (None, self.config)

        # Mock data loader using the helper
        self._setup_common_mocks(mock_data_loader)

        server = AlphaServer(debug=True)
        server.today_date = dt.now().date()
        server.pos_dir = '/test/pos'  # Use non-POSITION_DIR
        
        # Create test DataFrame
        ts = pd.Timestamp('2023-01-01 12:00:00')
        df = pd.DataFrame({
            'close_mid': [50000.0, 3000.0]
        }, index=pd.MultiIndex.from_tuples(
            [(ts, 'BTC_binance'), (ts, 'ETH_binance')],
            names=['ts', 'symbol_venue']
        ))
        
        # Mock position loading with MultiIndex to match the df structure
        positions_df = pd.DataFrame({
            'qty': [0.5, 2.0]
        }, index=pd.MultiIndex.from_tuples(
            [(ts, 'BTC_binance'), (ts, 'ETH_binance')],
            names=['ts', 'symbol_venue']
        ))
        
        # Mock load_positions to return our test positions
        # Need to ensure the mock applies to all calls within set_positions
        mock_load_positions.return_value = positions_df
        
        # Call set_positions which internally calls load_positions
        result = server.set_positions(df)
            
        self.assertIn('qty', result.columns)
        self.assertIn('position', result.columns)
        
        # Debug - print to see what's happening
        if result.loc[(ts, 'BTC_binance'), 'qty'] == 0:
            # The mock isn't working, positions are being set to 0
            # Let's check if load_positions was called with the right parameters
            mock_load_positions.assert_called()
            
        # Check the positions were calculated correctly
        # qty * close_mid = position
        self.assertAlmostEqual(float(result.loc[(ts, 'BTC_binance'), 'qty']), 0.5, places=4)
        self.assertEqual(result.loc[(ts, 'BTC_binance'), 'close_mid'], 50000.0)
        self.assertAlmostEqual(float(result.loc[(ts, 'BTC_binance'), 'position']), 25000.0, places=1)
        self.assertAlmostEqual(float(result.loc[(ts, 'ETH_binance'), 'qty']), 2.0, places=4)
        self.assertEqual(result.loc[(ts, 'ETH_binance'), 'close_mid'], 3000.0)
        self.assertAlmostEqual(float(result.loc[(ts, 'ETH_binance'), 'position']), 6000.0, places=1)
        
        # Test exception when no positions loaded and debug=False
        server.debug = False
        mock_load_positions.return_value = None
        with self.assertRaises(Exception):
            server.set_positions(df)
        
    @patch('lib.server.server.get_config')
    @patch('lib.server.server.DataLoader')
    @patch('lib.portfolio_optimization.PortfolioOptimizer')
    @patch('lib.server.server.Forecasts')
    @patch('lib.server.server.Calcs')
    @patch('lib.data.loaders.load_positions')
    @patch('lib.server.server.AlphaServer.setup')
    @patch('lib.external.binance_utils.get_positions')
    def test_set_max_positions_from_binance(self, mock_get_positions, mock_setup, mock_load_positions,
                                           mock_calcs, mock_forecasts, mock_optimizer,
                                           mock_data_loader, mock_get_config):
        """Test loading position limits from Binance."""
        mock_get_config.return_value = (None, self.config)
        mock_load_positions.return_value = None

        # Mock data loader using the helper
        self._setup_common_mocks(mock_data_loader)

        server = AlphaServer()
        
        # Test successful load
        binance_df = pd.DataFrame({
            'symbol': ['BTCUSDT', 'ETHUSDT'],
            'maxNotionalValue': [1000000, 500000]
        })
        mock_get_positions.return_value = binance_df
        
        ts = pd.Timestamp('2023-01-01')
        df = pd.DataFrame({
            'close_mid': [50000.0, 3000.0]
        }, index=pd.MultiIndex.from_tuples(
            [(ts, 'BTC_binance'), (ts, 'ETH_binance')],
            names=['ts', 'symbol_venue']
        ))
        
        with patch('lib.util.dataframes.make_symbol_venue') as mock_make_symbol_venue:
            mock_make_symbol_venue.return_value = pd.DataFrame({
                'symbol_venue': ['BTC_binance', 'ETH_binance'],
                'maxNotionalValue': [1000000, 500000]
            })
            result = server.set_max_positions_from_binance(df)
            
        self.assertIn('max_position_value', result.columns)
        
        # Test failed load
        mock_get_positions.return_value = None
        result = server.set_max_positions_from_binance(df)
        self.assertTrue(result['max_position_value'].isna().all())
        

class TestAlphaServerTargetGeneration(unittest.TestCase):
    """Test target generation and dumping."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = get_test_config()
        # Override SHORT_TERM_MODEL_HORIZONS for this test
        self.config['SHORT_TERM_MODEL_HORIZONS'] = [60]
        self.universe_df = pd.DataFrame({
            'fittable': [True, True],
            'tradeable': [True, True],
            'expandable': [True, True],
            'priceable': [True, True],
            'advp': [1000000, 2000000],
            'marketcap': [1e9, 2e9],
            'symbol_venue': ['BTC_binance', 'ETH_binance']
        })
        
    def _setup_common_mocks(self, mock_data_loader, mock_data_loader_instance=None):
        """Set up common mocks for data loader."""
        if mock_data_loader_instance is None:
            mock_data_loader_instance = Mock()
        mock_data_loader_instance.load_universe_df.return_value = self.universe_df
        mock_data_loader_instance.load_universe_symbol_venues.return_value = ['BTC_binance', 'ETH_binance']
        mock_data_loader.return_value = mock_data_loader_instance
        return mock_data_loader_instance
        
    @patch('lib.server.server.get_config')
    @patch('lib.server.server.DataLoader')
    @patch('lib.portfolio_optimization.PortfolioOptimizer')
    @patch('lib.server.server.Forecasts')
    @patch('lib.server.server.Calcs')
    @patch('lib.data.loaders.load_positions')
    @patch('lib.server.server.AlphaServer.setup')
    def test_dump_targets(self, mock_setup, mock_load_positions, mock_calcs, mock_forecasts,
                         mock_optimizer, mock_data_loader, mock_get_config):
        """Test target file dumping."""
        mock_get_config.return_value = (None, self.config)
        mock_load_positions.return_value = None

        # Mock data loader using the helper
        self._setup_common_mocks(mock_data_loader)

        server = AlphaServer(debug=True)  # Debug mode to skip actual file writing
        server.raw_target_dir = '/test/raw'
        server.target_dir = '/test/target'
        server.short_term_horizons = [60]
        server.model_horizons = [60, 1440]
        server.model_alpha_cols = ['alpha_60', 'alpha_1440']
        
        # Create test targets DataFrame
        ts = pd.Timestamp('2023-01-01 12:00:00')
        targets_df = pd.DataFrame({
            'position': [25000.0, 6000.0],
            'target_position': [30000.0, 5000.0],
            'close_mid': [50000.0, 3000.0],
            'alpha_opt': [0.01, -0.005],
            'alpha_60': [0.005, -0.002],
            'alpha_1440': [0.005, -0.003],
            'risk_1440': [0.02, 0.03],
            'risk_15': [0.01, 0.015],
            'logret_1440_lz': [0.5, -0.3],
            'desired_trade_dollars': [5000.0, -1000.0],
            'desired_trade_dollars_abs': [5000.0, 1000.0],
            'lbound': [-100000, -50000],
            'ubound': [100000, 50000],
            'util': [0.001, -0.0005]
        }, index=pd.MultiIndex.from_tuples(
            [(ts, 'BTC_binance'), (ts, 'ETH_binance')],
            names=['ts', 'symbol_venue']
        ))
        
        # Test with optimization
        server.dump_targets(targets_df, optimize=True)
        
        # Test without optimization
        server.dump_targets(targets_df, optimize=False)
        
    @patch('lib.server.server.load_mktcap_and_groupings')
    @patch('lib.server.server.safe_mkdir')
    @patch('lib.server.server.get_config')
    @patch('lib.server.server.DataLoader')
    @patch('lib.portfolio_optimization.PortfolioOptimizer')
    @patch('lib.server.server.Forecasts')
    @patch('lib.server.server.Calcs')
    @patch('lib.data.loaders.load_positions')
    @patch('lib.server.server.AlphaServer.setup')
    def test_generate_targets_with_optimization(self, mock_setup, mock_load_positions, mock_calcs_class,
                                               mock_forecasts, mock_optimizer_class, mock_data_loader,
                                               mock_get_config, mock_safe_mkdir,
                                               mock_load_mktcap):
        """Test target generation with full optimization."""
        mock_get_config.return_value = (None, self.config)
        mock_load_positions.return_value = None
        mock_safe_mkdir.return_value = None  # Mock the directory creation

        # Mock load_mktcap_and_groupings to return DataFrame with category columns
        mock_mktcap_df = pd.DataFrame({
            'symbol': ['BTC', 'ETH'],
            'cat_defi': [False, True],
            'cat_layer1': [True, True],
            'cat_meme': [False, False]
        })
        mock_load_mktcap.return_value = mock_mktcap_df

        # Mock data loader using the helper
        self._setup_common_mocks(mock_data_loader)

        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.util_metrics = {}
        mock_optimizer_class.return_value = mock_optimizer
        
        # Mock calcs
        mock_calcs = Mock()
        mock_calcs_class.return_value = mock_calcs
        
        server = AlphaServer()
        server.model_horizons = [60, 1440]
        server.short_term_horizons = [60]
        server.dir_manager = Mock()
        server.dir_manager.UTIL_DIR = '/test/util'
        server.model_alpha_cols = []  # Empty list since we're not testing model-specific alphas
        
        # Replace the trade_bot with our mock
        server.trade_bot = mock_optimizer
        
        # Mock server_calcs
        server.server_calcs = Mock()
        ts = pd.Timestamp('2023-01-01 12:00:00')
        alpha_df = pd.DataFrame({
            'alpha_60': [0.01, -0.005],
            'alpha_60_rev': [0.005, -0.002],
            'alpha_60_mom': [0.005, -0.003],
            'alpha_1440': [0.02, -0.01],
            'alpha_1440_rev': [0.01, -0.005],
            'alpha_1440_mom': [0.01, -0.005],
            'expandable': [True, True],
            'tradeable': [True, True],
            'risk_1440': [0.02, 0.03],
            'risk_15': [0.01, 0.015],
            'relative_spread_1440_trmean': [0.0001, 0.0002],
            'dvolume_1440_trmean': [1000000, 500000],
            'dvolume_1440': [900000, 450000],
            'dvolume_120_forecast': [800000, 400000],  # Add volume forecast for optimization interval
            'lbound': [-100000, -50000],
            'ubound': [100000, 50000],
            'last_funding_rate': [0.0001, 0.0001],
            'logret_1440_lz': [0.5, -0.3],
            'delisting_date': [pd.Timestamp('2025-12-31'), pd.Timestamp('2025-12-31')]
        }, index=pd.MultiIndex.from_tuples(
            [(ts, 'BTC_binance'), (ts, 'ETH_binance')],
            names=['ts', 'symbol_venue']
        ))
        server.server_calcs.calculate_alphas.return_value = (alpha_df, ['alpha_60', 'alpha_1440'])
        
        # Mock helper methods
        def mock_set_positions(df):
            # Add required columns that set_positions would add
            df['qty'] = [0.1, 0.2]  # Example quantities
            df['close_mid'] = [50000.0, 3000.0]  # Example prices
            df['position'] = df['qty'] * df['close_mid']
            return df
        
        server.set_positions = Mock(side_effect=mock_set_positions)
        server.set_max_positions_from_binance = Mock(side_effect=lambda df: df)
        
        # Mock calculate_expandable_filter to pass through the dataframe
        mock_calcs.calculate_expandable_filter.side_effect = lambda df: df

        # Mock calculate_exclusions_and_bounds to add bounds
        def mock_calculate_bounds(df):
            # Add bounds that would be calculated
            if 'lbound' not in df.columns:
                df['lbound'] = [-100000, -50000]
            if 'ubound' not in df.columns:
                df['ubound'] = [100000, 50000]
            # Add columns expected by calculate_exclusions_and_bounds  
            if 'delisting_date' not in df.columns:
                df['delisting_date'] = pd.Timestamp('2025-12-31')  # Far future date
            if 'alpha_opt' not in df.columns:
                df['alpha_opt'] = df['alpha_60'] + df['alpha_1440']
            return df
        
        mock_calcs.calculate_exclusions_and_bounds.side_effect = mock_calculate_bounds
        
        # Mock optimization methods
        def mock_make_alpha_opt(df):
            # Add columns that make_alpha_opt would add
            df['alpha_opt'] = df['alpha_60'] + df['alpha_1440']
            df['alpha_rev'] = 0.0
            df['alpha_mom'] = 0.0
            return df, ['alpha_opt', 'alpha_rev', 'alpha_mom']
            
        def mock_optimize(alpha_df):
            alpha_df['target_opt'] = alpha_df['alpha_opt'] * 100000
            return alpha_df
            
        def mock_calculate_targets(timeslice_df):
            timeslice_df['target_position'] = timeslice_df.get('target_opt', 0)
            timeslice_df['desired_trade_dollars'] = timeslice_df['target_position'] - timeslice_df['position']
            timeslice_df['desired_trade_dollars_abs'] = timeslice_df['desired_trade_dollars'].abs()
            # Add util column expected by dump_targets
            timeslice_df['util'] = [0.001, -0.0005]
            return timeslice_df
            
        mock_optimizer.make_alpha_opt.side_effect = mock_make_alpha_opt
        mock_optimizer.optimize.side_effect = mock_optimize
        mock_optimizer.calculate_targets.side_effect = mock_calculate_targets
        mock_optimizer.dump_util_metrics = Mock()
        
        # Mock dump_targets
        server.dump_targets = Mock()
        
        # Generate targets with optimization
        server.generate_targets(optimize=True)
        
        # Verify calls
        server.server_calcs.calculate_alphas.assert_called_once_with(horizons=[60, 1440])
        mock_optimizer.optimize.assert_called_once()
        server.dump_targets.assert_called_once()
        mock_optimizer.dump_util_metrics.assert_called_once()
        
    @patch('lib.server.server.get_config')
    @patch('lib.server.server.DataLoader')
    @patch('lib.portfolio_optimization.PortfolioOptimizer')
    @patch('lib.server.server.Forecasts')
    @patch('lib.server.server.Calcs')
    @patch('lib.server.server.load_positions')  # Patch where it's imported
    @patch('lib.server.server.AlphaServer.setup')
    def test_generate_targets_without_optimization(self, mock_setup, mock_load_positions, mock_calcs_class,
                                                  mock_forecasts, mock_optimizer_class, mock_data_loader,
                                                  mock_get_config):
        """Test target generation without optimization (ST only)."""
        mock_get_config.return_value = (None, self.config)
        mock_load_positions.return_value = None

        # Mock data loader using the helper
        self._setup_common_mocks(mock_data_loader)

        # Mock optimizer
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        
        # Mock calcs
        mock_calcs = Mock()
        mock_calcs_class.return_value = mock_calcs
        
        server = AlphaServer()
        server.model_horizons = [60, 1440]
        server.short_term_horizons = [60]
        server.model_alpha_cols = []  # Empty list since we're not testing model-specific alphas
        
        # Replace the trade_bot and calcs with our mocks
        server.trade_bot = mock_optimizer
        server.calcs = mock_calcs
        
        # Set up previous alphas
        ts_prev = pd.Timestamp('2023-01-01 11:00:00')
        server.alphas_df = pd.DataFrame({
            'target_opt': [50000.0, -10000.0]
        }, index=pd.MultiIndex.from_tuples(
            [(ts_prev, 'BTC_binance'), (ts_prev, 'ETH_binance')],
            names=['ts', 'symbol_venue']
        ))
        
        # Mock server_calcs
        server.server_calcs = Mock()
        ts = pd.Timestamp('2023-01-01 12:00:00')
        alpha_df = pd.DataFrame({
            'alpha_60': [0.015, -0.008],
            'tradeable': [True, True],
            'expandable': [True, True]
        }, index=pd.MultiIndex.from_tuples(
            [(ts, 'BTC_binance'), (ts, 'ETH_binance')],
            names=['ts', 'symbol_venue']
        ))
        server.server_calcs.calculate_alphas.return_value = (alpha_df, ['alpha_60'])
        
        # Mock helper methods
        def mock_set_positions(df):
            # Add required columns that set_positions would add
            df['qty'] = [0.1, 0.2]  # Example quantities
            df['close_mid'] = [50000.0, 3000.0]  # Example prices
            df['position'] = df['qty'] * df['close_mid']
            return df
        
        def mock_calculate_exclusions_and_bounds(df):
            # Add required columns that calculate_exclusions_and_bounds needs/adds
            if 'dvolume_1440_trmean' not in df.columns:
                df['dvolume_1440_trmean'] = [1000000.0, 500000.0]
            if 'delisting_date' not in df.columns:
                df['delisting_date'] = pd.Timestamp('2025-12-31')
            if 'lbound' not in df.columns:
                df['lbound'] = [-100000, -50000]
            if 'ubound' not in df.columns:
                df['ubound'] = [100000, 50000]
            if 'relative_spread_1440_trmean' not in df.columns:
                df['relative_spread_1440_trmean'] = [0.0001, 0.0002]
            if 'last_funding_rate' not in df.columns:
                df['last_funding_rate'] = [0.0001, 0.0001]
            if 'logret_1440_lz' not in df.columns:
                df['logret_1440_lz'] = [0.5, -0.3]
            return df
        
        server.set_positions = Mock(side_effect=mock_set_positions)
        server.set_max_positions_from_binance = Mock(side_effect=lambda df: df)
        mock_calcs.calculate_expandable_filter.side_effect = lambda df: df
        mock_calcs.calculate_exclusions_and_bounds.side_effect = mock_calculate_exclusions_and_bounds
        mock_optimizer.calculate_targets.side_effect = lambda timeslice_df: timeslice_df
        server.dump_targets = Mock()

        # Generate targets without optimization
        server.generate_targets(optimize=False)
        
        # Verify calls
        server.server_calcs.calculate_alphas.assert_called_once_with(horizons=[60])
        mock_optimizer.optimize.assert_not_called()
        server.dump_targets.assert_called_once()
        

class TestAlphaServerLoop(unittest.TestCase):
    """Test server loop functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = get_test_config()
        # Override SHORT_TERM_MODEL_HORIZONS for this test
        self.config['SHORT_TERM_MODEL_HORIZONS'] = [60]
        self.universe_df = pd.DataFrame({
            'fittable': [True, True],
            'tradeable': [True, True],
            'expandable': [True, True],
            'priceable': [True, True],
            'advp': [1000000, 2000000],
            'marketcap': [1e9, 2e9],
            'symbol_venue': ['BTC_binance', 'ETH_binance']
        })
        
    def _setup_common_mocks(self, mock_data_loader, mock_data_loader_instance=None):
        """Set up common mocks for data loader."""
        if mock_data_loader_instance is None:
            mock_data_loader_instance = Mock()
        mock_data_loader_instance.load_universe_df.return_value = self.universe_df
        mock_data_loader_instance.load_universe_symbol_venues.return_value = ['BTC_binance', 'ETH_binance']
        mock_data_loader.return_value = mock_data_loader_instance
        return mock_data_loader_instance
        
    @patch('lib.server.server.get_config')
    @patch('lib.server.server.DataLoader')
    @patch('lib.portfolio_optimization.PortfolioOptimizer')
    @patch('lib.server.server.Forecasts')
    @patch('lib.server.server.Calcs')
    @patch('lib.server.server.load_positions')  # Patch where it's imported
    @patch('lib.server.server.AlphaServer.setup')
    def test_get_latest_target_age(self, mock_setup, mock_load_positions, mock_calcs, mock_forecasts,
                                   mock_optimizer, mock_data_loader, mock_get_config):
        """Test getting age of latest target file."""
        mock_get_config.return_value = (None, self.config)
        mock_load_positions.return_value = None

        # Mock data loader using the helper
        self._setup_common_mocks(mock_data_loader)

        server = AlphaServer()
        
        # Test with no target files
        with patch('lib.server.server.get_latest_target_files', return_value=[]):
            age = server.get_latest_target_age()
            self.assertIsNone(age)
            
        # Test with target files
        # Create a specific time for testing to avoid timing issues
        base_time = dt(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        time_10min_ago = base_time - td(minutes=10)
        time_30min_ago = base_time - td(minutes=30)
        
        timestamp_30min_ago = time_30min_ago.strftime('%Y%m%d_%H%M')
        timestamp_10min_ago = time_10min_ago.strftime('%Y%m%d_%H%M')
        
        # Patch today_date to return a consistent date for load_target_files
        with patch('lib.server.server.today_date', return_value=base_time.date()):
            with patch('lib.server.server.get_latest_target_files', return_value=[
                f'/test/targets.opt.{timestamp_30min_ago}.csv',
                f'/test/targets.opt.{timestamp_10min_ago}.csv'
            ]):
                with patch('os.path.getmtime') as mock_getmtime:
                    # Set up modification times - the 10min file was modified more recently
                    def getmtime_side_effect(path):
                        if timestamp_30min_ago in path:
                            return time_30min_ago.timestamp()
                        return time_10min_ago.timestamp()
                            
                    mock_getmtime.side_effect = getmtime_side_effect
                    
                    # Patch datetime in the server module and str_to_dt
                    with patch('lib.server.server.dt') as mock_dt:
                        with patch('lib.server.server.str_to_dt') as mock_str_to_dt:
                            # Mock dt.now to return our fixed time
                            mock_dt.now.return_value = base_time
                            # Keep timezone attribute
                            mock_dt.timezone = timezone
                            # Mock str_to_dt to convert the timestamp correctly
                            def str_to_dt_side_effect(ts_str):
                                if '202401' in ts_str and '1150' in ts_str:
                                    return time_10min_ago
                                elif '202401' in ts_str and '1130' in ts_str:
                                    return time_30min_ago
                                # Default to parsing the 10min ago timestamp
                                return time_10min_ago
                            mock_str_to_dt.side_effect = str_to_dt_side_effect
                            
                            age = server.get_latest_target_age()
                            self.assertIsNotNone(age)
                            # The latest file (by modification time) has timestamp_10min_ago in its name
                            # So age should be 10 minutes = 600 seconds
                            self.assertAlmostEqual(age, 600, delta=1)  # ~10 minutes
                
    @patch('lib.server.server.get_config')
    @patch('lib.server.server.DataLoader')
    @patch('lib.portfolio_optimization.PortfolioOptimizer')
    @patch('lib.server.server.Forecasts')
    @patch('lib.server.server.Calcs')
    @patch('lib.data.loaders.load_positions')
    @patch('lib.server.server.AlphaServer.setup')
    def test_update_method(self, mock_setup, mock_load_positions, mock_calcs, mock_forecasts,
                          mock_optimizer, mock_data_loader, mock_get_config):
        """Test update method delegation."""
        mock_get_config.return_value = (None, self.config)
        mock_load_positions.return_value = None

        # Mock data loader using the helper
        self._setup_common_mocks(mock_data_loader)

        server = AlphaServer()
        server.server_calcs = Mock()
        server.server_calcs.update.return_value = True
        
        # Test ST only update
        result = server.update(st_only=True)
        self.assertTrue(result)
        server.server_calcs.update.assert_called_once_with(True)
        
        # Test full update
        server.server_calcs.update.reset_mock()
        result = server.update(st_only=False)
        self.assertTrue(result)
        server.server_calcs.update.assert_called_once_with(False)
        

class TestAlphaServerIntegration(unittest.TestCase):
    """Integration tests for AlphaServer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = get_test_config()
        # Override settings for integration tests
        self.config['OPT_HORIZON'] = 60
        self.config['SHORT_TERM_MODEL_HORIZONS'] = [15]
        self.config['DEFAULT_FEATURE_LOOKBACK_PERIODS'] = 2
        self.config['SYMBOL_UNIVERSE'] = ['BTC']
        self.config['FCASTS'] = {
            '60': {
                'models': [{'name': 'test_model', 'weight': 1.0, 'lags': 1}],
                'features': ['logret_60_lz', 'dvolume_60_lz']
            },
            '15': {
                'models': [{'name': 'test_model_st', 'weight': 1.0, 'lags': 1}],
                'features': ['logret_15_lz', 'dvolume_15_lz']
            }
        }
        self.universe_df = pd.DataFrame({
            'fittable': [True, True],
            'tradeable': [True, True],
            'expandable': [True, True],
            'priceable': [True, True],
            'advp': [1000000, 2000000],
            'marketcap': [1e9, 2e9],
            'symbol_venue': ['BTC_binance', 'ETH_binance']
        })
        
    def _setup_common_mocks(self, mock_data_loader, mock_data_loader_instance=None):
        """Set up common mocks for data loader."""
        if mock_data_loader_instance is None:
            mock_data_loader_instance = Mock()
        mock_data_loader_instance.load_universe_df.return_value = self.universe_df
        mock_data_loader_instance.load_universe_symbol_venues.return_value = ['BTC_binance', 'ETH_binance']
        mock_data_loader.return_value = mock_data_loader_instance
        return mock_data_loader_instance
        
    @patch('lib.server.server.get_config')
    @patch('lib.server.server.DataLoader')
    @patch('lib.portfolio_optimization.PortfolioOptimizer')
    @patch('lib.server.server.Forecasts')
    @patch('lib.server.server.Calcs')
    @patch('lib.data.loaders.load_positions')
    @patch('lib.server.server_calcs.ServerCalcs')
    @patch('lib.server.server.AlphaServer.setup')
    def test_get_latest_alphas(self, mock_setup, mock_server_calcs_class, mock_load_positions, mock_calcs,
                               mock_forecasts, mock_optimizer, mock_data_loader, mock_get_config):
        """Test getting latest alpha signals."""
        mock_get_config.return_value = (None, self.config)
        mock_load_positions.return_value = None

        # Mock data loader using the helper
        self._setup_common_mocks(mock_data_loader)

        # Mock ServerCalcs
        mock_server_calcs = Mock()
        mock_server_calcs_class.return_value = mock_server_calcs
        
        server = AlphaServer()
        
        # Initialize alphas_df with proper MultiIndex
        server.alphas_df = pd.DataFrame(columns=['alpha_60'], 
                                       index=pd.MultiIndex.from_tuples([], names=['ts', 'symbol_venue']))
        
        # Initially empty
        latest = server.get_latest_alphas()
        self.assertTrue(latest.empty)
        
        # Add some alphas
        ts1 = pd.Timestamp('2023-01-01 10:00:00')
        ts2 = pd.Timestamp('2023-01-01 11:00:00')
        
        server.alphas_df = pd.DataFrame({
            'alpha_60': [0.01, 0.02, -0.01],
            'alpha_15': [0.005, 0.01, -0.005]
        }, index=pd.MultiIndex.from_tuples([
            (ts1, 'BTC_binance'),
            (ts2, 'BTC_binance'),
            (ts2, 'ETH_binance')
        ], names=['ts', 'symbol_venue']))
        
        # Get latest
        latest = server.get_latest_alphas()
        self.assertEqual(len(latest), 2)  # Only ts2 entries
        self.assertEqual(latest.index.get_level_values('ts').unique()[0], ts2)
        

if __name__ == '__main__':
    unittest.main()
