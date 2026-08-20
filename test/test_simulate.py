"""Unit tests for the simulate module.

This test suite validates the backtesting simulation engine functionality,
including initialization, data loading, execution simulation, and PnL tracking.
"""
# pylint: disable=protected-access

import unittest
from datetime import date
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from lib.sim.simulate import Simulate
from lib.portfolio_optimization import OptimizationError


def get_test_config():
    """Get a valid test configuration."""
    return {
        'FCASTS': {
            '1440': {
                'features': {'prod': ['feature1', 'feature2']},
                'models': [{'name': 'model1', 'weight': 1.0, 'lags': 1}]
            }
        },
        'REOPTIMIZE_INTERVAL_MINS': 120,
        'MAX_PORTFOLIO_NOTIONAL': 20000000,
        'OPT_HORIZON': 1440,
        'OPT_OFFSET_MINS': 2,
        'MAX_VOLUME_FRACTION_PARTICIPATION': 0.02,
        'MIN_TRADE_DOLLARS': 100,
        'DEFAULT_SLIPPAGE': 0.0001,
        'EXCHANGE_FEES': 0.00005,
        'RISK_HORIZON': 120,
        'RISK_FLD': 'logret_HORIZON_trstd',
        'BETA_LOOKBACK_PERIODS': 90,
        'DEFAULT_FEATURE_LOOKBACK_PERIODS': 90,
        'FEATURE_SIGMA_BOUND': 5,
        'SIM_EXECUTION_PRICE': 'vwap',
        'SKYROCKETING_SIGMA_LEVEL': None,
        'SKYROCKETING_SLIPPAGE': 0.001,
        'VWAP_HORIZON': 0,
        'SYMBOL_UNIVERSE': ['BTC', 'ETH', 'SOL'],
        'FACTOR_SIGMAS': {'dollar_exposure': 0.1},
        'SHORT_TERM_MODEL_HORIZONS': [15, 60],
        'DYNAMIC_UNIVERSE': False,
        'MIN_ADVP_PRICEABLE': 2.5e7,
        'MIN_ADVP_FEATUREABLE': 2.5e7,
        'FEATUREABLE_HIST_PERIODS': 30,
        'MIN_ADVP_FITTABLE': 1.5e7,
        'MIN_ADVP_TRADEABLE': 1.5e7,
        'MIN_ADVP_EXPANDABLE': 5.5e7,
        'ADV_LOOKBACK_DAYS': 45,
        'OLD_POSITION_DAYS': 30,
        'OLD_POSITION_RISK_MULT': 3.5,
        'MAX_MOVE_FILTER': 2.5,
        'MAX_POSITION_VOLUME_FRACTION': 0.05,
        'MAX_POSITION_PCT': 0.040,
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
        'REOPT_TIMES': [],
        'SKIP_TZ': ['America/New_York'],
        'MAX_DVOL_SIGMA': 2.0,
        'MIN_FUNDING_RATE': -0.005,
        'MAX_MARKET_CAP_EXPANDABLE_FRAC': 1.0
    }


class TestSimulateInit(unittest.TestCase):
    """Test Simulate class initialization."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = get_test_config()
        
    @patch('lib.sim.simulate.PortfolioOptimizer')
    @patch('lib.sim.simulate.Calcs')
    @patch('lib.sim.simulate.DataLoader')
    def test_init_default(self, mock_data_loader, mock_calcs, mock_optimizer):
        """Test initialization with default parameters."""
        sim = Simulate(
            config=self.config,
            sim_dir='/test/sim'
        )
        
        self.assertEqual(sim.config, self.config)
        self.assertEqual(sim.sim_dir, '/test/sim')
        self.assertFalse(sim.initialize_positions)
        self.assertFalse(sim.verbose)
        self.assertFalse(sim.sim_comp)
        self.assertEqual(sim.reopt_interval, 120)
        self.assertEqual(sim.max_portfolio_notional, 20000000)
        self.assertEqual(sim.vwap_horizon, 120)  # Defaults to reopt_interval
        
        # Check that components are initialized
        mock_optimizer.assert_called_once()
        mock_calcs.assert_called_once()
        mock_data_loader.assert_called_once()
        
    @patch('lib.sim.simulate.PortfolioOptimizer')
    @patch('lib.sim.simulate.Calcs')
    @patch('lib.sim.simulate.DataLoader')
    def test_init_with_parameters(self, _mock_data_loader, _mock_calcs, _mock_optimizer):
        """Test initialization with custom parameters."""
        self.config['VWAP_HORIZON'] = 60
        
        sim = Simulate(
            config=self.config,
            sim_dir='/test/sim',
            initialize_positions=True,
            verbose=True,
            sim_comp=True,
            skip_missing_alphas=True,
            config_grid='test_grid',
            alpha_condition='rev'
        )
        
        self.assertTrue(sim.initialize_positions)
        self.assertTrue(sim.verbose)
        self.assertTrue(sim.sim_comp)
        self.assertTrue(sim.skip_missing_alphas)
        # bars_type no longer exists
        self.assertEqual(sim.config_grid, 'test_grid')
        self.assertEqual(sim.alpha_condition, 'rev')
        self.assertEqual(sim.vwap_horizon, 60)
        
        
class TestSimulateRecordPnl(unittest.TestCase):
    """Test PnL recording functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.ts = pd.Timestamp('2023-01-01 12:00:00', tz='UTC')
        self.config = get_test_config()
        self.config['FCASTS'] = {}  # Empty for these tests
        
    @patch('lib.sim.simulate.PortfolioOptimizer')
    @patch('lib.sim.simulate.Calcs')
    @patch('lib.sim.simulate.DataLoader')
    def test_record_pnl_normal(self, _mock_data_loader, _mock_calcs, _mock_optimizer):
        """Test normal PnL recording."""
        sim = Simulate(self.config, '/test/sim')
        sim.trade_bot = Mock()
        sim.trade_bot.util_metrics = {
            'expected_utility': 100,
            'expected_risk': 0.01
        }
        
        # Create test DataFrame
        ts_df = pd.DataFrame({
            'ts': [self.ts] * 3,
            'symbol': ['BTC', 'ETH', 'SOL'],
            'pnl': [1000, -500, 200],
            'executed_dollars': [10000, -5000, 0],
            'position': [50000, -30000, 10000],
            'unexecuted_dollars': [100, 50, 0],
            'fill_slippage': [10, -5, 0]
        })
        
        result = sim._record_pnl(ts_df)
        
        # Check result format
        expected = f"{self.ts}|700|60000|-30000|10000|-5000|150|5"
        self.assertEqual(result, expected)
        
        # Check execution stats recorded
        self.assertEqual(len(sim.execution_stats_list), 1)
        stats = sim.execution_stats_list[0]
        self.assertEqual(stats['ts'], self.ts)
        self.assertEqual(stats['unexecuted'], 150)
        self.assertEqual(stats['fill_slip'], 5)
        
    @patch('lib.sim.simulate.PortfolioOptimizer')
    @patch('lib.sim.simulate.Calcs')
    @patch('lib.sim.simulate.DataLoader')
    @patch('lib.sim.simulate.log_and_raise', side_effect=Exception("Portfolio grew to 35000000, something is wrong..."))
    def test_record_pnl_suspicious_size(self, _mock_log_raise, _mock_data_loader, _mock_calcs, _mock_optimizer):
        """Test PnL recording with suspicious portfolio size."""
        sim = Simulate(self.config, '/test/sim')
        sim.trade_bot = Mock()
        sim.trade_bot.util_metrics = {}
        
        # Create test DataFrame with huge position
        ts_df = pd.DataFrame({
            'ts': [self.ts],
            'symbol': ['BTC'],
            'pnl': [1000],
            'executed_dollars': [0],
            'position': [35000000],  # Exceeds threshold
            'position_previous': [34000000],  # Add missing column
            'unexecuted_dollars': [0],
            'fill_slippage': [0],
            'symbol_venue': ['BTC_binance'],
            'close_mid': [50000],
            'close_mid_previous': [49000],
            'qty': [700],
            'qty_previous': [700],
            'limit_qty_pos': [10],
            'limit_qty_neg': [-10],
            'desired_trade_dollars_previous': [0],
            'target_opt': [35000000]
        })
        
        with self.assertRaises(Exception) as context:
            sim._record_pnl(ts_df)
        
        self.assertIn("Portfolio grew to", str(context.exception))
        
        
class TestSimulateCalculateFunding(unittest.TestCase):
    """Test funding calculation functionality."""
    
    def test_calculate_funding_for_sim(self):
        """Test funding rate calculation."""
        config = get_test_config()
        config['FCASTS'] = {}
        config['REOPTIMIZE_INTERVAL_MINS'] = 240  # 4 hours
        
        with patch('lib.sim.simulate.PortfolioOptimizer'), \
             patch('lib.sim.simulate.Calcs'), \
             patch('lib.sim.simulate.DataLoader'):
            sim = Simulate(config, '/test/sim')
        
        # Create test data
        base_time = pd.Timestamp('2023-01-01 00:00:00', tz='UTC')
        
        # Main DataFrame
        df = pd.DataFrame({
            'ts': [base_time + pd.Timedelta(hours=4), base_time + pd.Timedelta(hours=8)],
            'symbol_venue': ['BTC_binance', 'BTC_binance']
        }).set_index(['ts', 'symbol_venue'])
        
        # Funding DataFrame with rates at 00:00, 04:00, 08:00
        funding_times = [base_time, base_time + pd.Timedelta(hours=4), base_time + pd.Timedelta(hours=8)]
        funding_df = pd.DataFrame({
            'ts': funding_times,
            'symbol_venue': ['BTC_binance'] * 3,
            'last_funding_rate': [0.0001, 0.0002, 0.0003],
            'next_funding_time': funding_times
        }).set_index(['ts', 'symbol_venue'])
        
        origin = base_time
        alpha_times = pd.DatetimeIndex([base_time + pd.Timedelta(hours=4), base_time + pd.Timedelta(hours=8)])
        
        result = sim._calculate_funding_for_sim(df, funding_df, origin, alpha_times)
        
        # Check results
        self.assertIn('realized_funding_rate', result.columns)
        self.assertIn('percent_executed_at_funding', result.columns)
        
        # With label='right' and closed='right', periods are (start, end]
        # First period (00:00, 04:00] includes only 04:00 funding: 0.0002
        self.assertAlmostEqual(result.iloc[0]['realized_funding_rate'], 0.0002, places=6)
        
        # Second period (04:00, 08:00] includes only 08:00 funding: 0.0003
        self.assertAlmostEqual(result.iloc[1]['realized_funding_rate'], 0.0003, places=6)
        
        
class TestSimulateDataLoading(unittest.TestCase):
    """Test data loading functionality."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = get_test_config()
        
    @patch('lib.sim.simulate.load_mktcap_and_groupings')
    @patch('lib.universe.filter_universe')
    @patch('lib.sim.simulate.PortfolioOptimizer')
    @patch('lib.sim.simulate.Calcs')
    @patch('lib.sim.simulate.DataLoader')
    def test_sim_data_for_date_success(self, mock_data_loader_class, mock_calcs_class, _mock_optimizer, mock_filter_universe, mock_load_mktcap):
        """Test successful data loading for a date."""
        # Set up mocks
        mock_data_loader = Mock()
        mock_data_loader_class.return_value = mock_data_loader
        
        mock_calcs = Mock()
        mock_calcs_class.return_value = mock_calcs
        
        sim = Simulate(self.config, '/test/sim')
        sim.data_loader = mock_data_loader
        sim.calcs = mock_calcs
        
        test_date = date(2023, 1, 1)
        test_ts = pd.Timestamp('2023-01-01 12:00:00', tz='UTC')
        
        # Mock universe filtering
        bars_1_df = pd.DataFrame({
            'update_cnt': [100, 200],
            'volume': [1000, 2000],
            'fittable': [True, True],
            'advp': [20000000, 30000000]
        }, index=pd.MultiIndex.from_tuples([
            (test_ts, 'BTCUSDT_binance-futures'),
            (test_ts, 'ETHUSDT_binance-futures')
        ], names=['ts', 'symbol_venue']))
        
        # Create bars data with required columns
        bars_df = pd.DataFrame({
            'last_funding_rate': [0.0001, 0.0001],
            'next_funding_time': [test_ts, test_ts],
            'close_mid': [50000.0, 3000.0],
            'dvolume_120': [1000000.0, 500000.0],
            'logret_120': [0.001, 0.002],
            'vwap_120': [49950.0, 2995.0],
            'high_mid_120': [50100.0, 3010.0],
            'advp': [20000000.0, 30000000.0]
        }, index=pd.MultiIndex.from_tuples([
            (test_ts, 'BTCUSDT_binance-futures'),
            (test_ts, 'ETHUSDT_binance-futures')
        ], names=['ts', 'symbol_venue']))
        
        # Create daily bars data for when reopt_interval != 1440
        bars_1440_df = pd.DataFrame({
            'logret_1440': [0.005, 0.006],
            'logret_resid_eqmkt_1440': [0.003, 0.004]
        }, index=bars_df.index)
        
        # Create vwap bars data
        bars_vwap_df = pd.DataFrame({
            'vwap_120': [49900.0, 2990.0]
        }, index=bars_df.index)
        
        # More flexible mock that handles variable number of calls
        def load_bars_side_effect(*args, **kwargs):
            # Check if this is asking for specific columns
            cols = kwargs.get('cols', [])
            if cols == ['vwap_120']:
                return bars_vwap_df
            # Check horizon to determine which bars to return
            horizon = kwargs.get('horizon', 1)
            if horizon == 1:
                return bars_1_df
            if horizon == 1440:
                return bars_1440_df
            else:  # horizon == 120 (reopt_interval)
                return bars_df
        
        mock_data_loader.load_bars.side_effect = load_bars_side_effect
        
        mock_filter_universe.return_value = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        
        # Mock universe dataframe
        universe_df = pd.DataFrame({
            'symbol_venue': ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures'],
            'expandable': [True, True],
            'fittable': [True, True],
            'tradeable': [True, True],
            'advp': [20000000.0, 30000000.0],
            'mdvp': [16000000.0, 24000000.0],
            'dvolume_1440_trmean': [1000000.0, 2000000.0],
            'marketcap': [1e9, 5e8]
        })
        mock_data_loader.load_universe_df.return_value = universe_df
        
        # Mock alpha data
        alpha_df = pd.DataFrame({
            'alpha_1440': [0.01, 0.02]
        }, index=pd.MultiIndex.from_tuples([
            (test_ts, 'BTCUSDT_binance-futures'),
            (test_ts, 'ETHUSDT_binance-futures')
        ], names=['ts', 'symbol_venue']))
        
        mock_data_loader.load_alphas.return_value = alpha_df
        
        # Mock calculate_horizon_alphas - need to add _rev and _mom columns
        with patch('lib.sim.simulate.calculate_horizon_alphas') as mock_calc_alphas:
            # Add the required columns for non-sim_comp mode
            alpha_df['alpha_1440_rev'] = alpha_df['alpha_1440'] * 0.5
            alpha_df['alpha_1440_mom'] = alpha_df['alpha_1440'] * 0.5
            mock_calc_alphas.return_value = (alpha_df, None)
            
            # Mock features - use float for boolean columns to avoid dtype merge issues
            features_df = pd.DataFrame({
                'dvolume_1440_trmean': [1000000.0, 2000000.0],
                'tradeable': [1.0, 1.0],
                'expandable': [1.0, 1.0]
            }, index=alpha_df.index)
            
            mock_data_loader.load_features.return_value = features_df
            
            # Mock volume forecast
            mock_calcs.calculate_volume_forecast.side_effect = lambda df, horizon: df
            
            # Mock delisting dates
            mock_data_loader.add_delisting_dates.side_effect = lambda df, **kwargs: df

            # Mock market cap / groupings data
            mock_load_mktcap.return_value = pd.DataFrame({
                'symbol_venue': ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures'],
            })

            # Call method - pylint: disable=protected-access
            result = sim._sim_data_for_date(test_date)
            
            # Verify result
            self.assertIsNotNone(result)
            self.assertIn('symbol_venue', result.columns)
            self.assertIn('ts', result.columns)
            
            # Verify calls
            mock_data_loader.load_alphas.assert_called_once()
            mock_data_loader.load_features.assert_called_once()
            mock_calc_alphas.assert_called_once()
            
    @patch('lib.universe.filter_universe')
    @patch('lib.sim.simulate.PortfolioOptimizer')
    @patch('lib.sim.simulate.Calcs')
    @patch('lib.sim.simulate.DataLoader')
    def test_sim_data_for_date_no_alphas(self, mock_data_loader_class, mock_calcs_class, _mock_optimizer, mock_filter_universe):
        """Test data loading when no alphas available."""
        mock_data_loader = Mock()
        mock_data_loader_class.return_value = mock_data_loader
        
        mock_calcs = Mock()
        mock_calcs_class.return_value = mock_calcs
        
        sim = Simulate(self.config, '/test/sim')
        sim.data_loader = mock_data_loader
        sim.calcs = mock_calcs
        
        # Mock empty alpha response
        mock_data_loader.load_alphas.return_value = None
        
        # Mock universe filtering bars
        mock_data_loader.load_bars.return_value = pd.DataFrame()

        # Mock load_universe_df to return a DataFrame
        mock_universe_df = pd.DataFrame({
            'symbol_venue': ['BTC_binance', 'ETH_binance'],
            'tradeable': [True, True]
        })
        mock_data_loader.load_universe_df.return_value = mock_universe_df

        mock_filter_universe.return_value = ['BTC_binance']
        
        # pylint: disable=protected-access
        result = sim._sim_data_for_date(date(2023, 1, 1))
        
        self.assertIsNone(result)
        
        
class TestSimulateExecution(unittest.TestCase):
    """Test order execution simulation."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = get_test_config()
        self.config['FCASTS'] = {}
        
    @patch('lib.sim.simulate.PortfolioOptimizer')
    @patch('lib.sim.simulate.Calcs')
    @patch('lib.sim.simulate.DataLoader')
    def test_execute_normal_trade(self, _mock_data_loader, _mock_calcs, _mock_optimizer):
        """Test normal trade execution."""
        sim = Simulate(self.config, '/test/sim')
        sim.pnl_calculator = Mock()
        sim.pnl_calculator.fill_loader = Mock()
        sim.pnl_calculator.bar_loader = Mock()
        sim.pnl_calculator.fundings_loader = Mock()
        # Make fundings_loader return the DataFrame unchanged
        sim.pnl_calculator.fundings_loader.update_funding_df_from_sim_timeslice.side_effect = lambda df, *args: df
        
        ts = pd.Timestamp('2023-01-01 12:00:00', tz='UTC')
        
        # Create test DataFrame
        ts_df = pd.DataFrame({
            'ts': [ts],  # Add ts column needed by funding calculation
            'symbol': ['BTC'],
            'symbol_venue': ['BTC_binance'],
            'close_mid': [50000.0],
            'close_mid_previous': [49000.0],
            'vwap_120': [49500.0],
            'desired_trade_dollars_previous': [10000.0],
            'desired_trade_qty_previous': [0.2],
            'volume_120': [5.0],  # Small volume to trigger constraint
            'qty': [0.0],
            'position': [0.0],
            'cash': [100000.0],
            'expanding_previous': [1.0],
            'funding_income': [0.0],
            'position_ts_previous': [pd.NaT],  # No previous position
            'executed_position_ts': [ts],  # Current timestamp
            'fill_slippage': [0.0],  # Initialize slippage tracking
            'realized_funding_rate': [0.0],  # Initialize funding rate
            'percent_executed_at_funding': [0.0]  # Initialize funding percentage
        })
        
        # pylint: disable=protected-access
        result = sim._execute(ts_df, ts)
        
        # Check execution - with volume_120=5, max participation is 5*0.02=0.1
        self.assertAlmostEqual(result.iloc[0]['executed_qty'], 0.1, places=6)  # Limited by volume
        self.assertAlmostEqual(float(result.iloc[0]['trade_price']), 49500.0 * 1.0001, places=2)  # VWAP with slippage
        self.assertGreater(result.iloc[0]['fees'], 0)
        self.assertAlmostEqual(result.iloc[0]['position'], 50000.0 * 0.1, places=2)  # position = qty * close_mid
        
    @patch('lib.sim.simulate.PortfolioOptimizer')
    @patch('lib.sim.simulate.Calcs')
    @patch('lib.sim.simulate.DataLoader')
    @patch('lib.sim.simulate.log_and_raise', side_effect=Exception("Bad execution prices even with carry forward!"))
    def test_execute_liquidation(self, _mock_log_raise, _mock_data_loader, _mock_calcs, _mock_optimizer):
        """Test position liquidation for dead asset."""
        sim = Simulate(self.config, '/test/sim')
        sim.pnl_calculator = Mock()
        sim.pnl_calculator.fill_loader = Mock()
        sim.pnl_calculator.bar_loader = Mock()
        sim.pnl_calculator.fundings_loader = Mock()
        # Make fundings_loader return the DataFrame unchanged
        sim.pnl_calculator.fundings_loader.update_funding_df_from_sim_timeslice.side_effect = lambda df, *args: df
        
        ts = pd.Timestamp('2023-01-01 12:00:00', tz='UTC')
        
        # Create test DataFrame with NaN price (dead asset)
        ts_df = pd.DataFrame({
            'symbol': ['BTC'],
            'symbol_venue': ['BTC_binance'],
            'close_mid': [np.nan],
            'close_mid_previous': [np.nan],  # Also bad
            'vwap_120': [50000.0],
            'desired_trade_dollars_previous': [0.0],
            'desired_trade_qty_previous': [np.nan],  # This will trigger the check
            'volume_120': [1000.0],
            'qty': [1.0],  # Has position
            'position': [50000.0],
            'cash': [0.0],
            'expanding_previous': [0.0],
            'funding_income': [0.0],
            'ts': [ts],
            'realized_funding_rate': [0.0],  # Initialize funding rate
            'percent_executed_at_funding': [0.0]  # Initialize funding percentage
        })
        
        # Should raise due to bad prices
        with self.assertRaises(Exception) as context:
            # pylint: disable=protected-access
            sim._execute(ts_df, ts)
        
        self.assertIn("Bad execution prices", str(context.exception))
        
    @patch('lib.sim.simulate.PortfolioOptimizer')
    @patch('lib.sim.simulate.Calcs')
    @patch('lib.sim.simulate.DataLoader')
    def test_execute_skyrocketing_short(self, _mock_data_loader, _mock_calcs, _mock_optimizer):
        """Test forced liquidation of skyrocketing short position."""
        self.config['SKYROCKETING_SIGMA_LEVEL'] = 3.0
        sim = Simulate(self.config, '/test/sim')
        sim.pnl_calculator = Mock()
        sim.pnl_calculator.fill_loader = Mock()
        sim.pnl_calculator.bar_loader = Mock()
        sim.pnl_calculator.fundings_loader = Mock()
        # Make fundings_loader return the DataFrame unchanged
        sim.pnl_calculator.fundings_loader.update_funding_df_from_sim_timeslice.side_effect = lambda df, *args: df
        
        ts = pd.Timestamp('2023-01-01 12:00:00', tz='UTC')
        
        # Create test DataFrame with skyrocketing price
        ts_df = pd.DataFrame({
            'ts': [ts],  # Add ts column needed by funding calculation
            'symbol': ['BTC'],
            'symbol_venue': ['BTC_binance'],
            'close_mid': [60000.0],
            'close_mid_previous': [50000.0],
            'high_mid_120': [65000.0],  # Very high
            'logret_120_trstd_previous': [0.02],  # 2% volatility
            'vwap_120': [55000.0],
            'desired_trade_dollars_previous': [0.0],
            'desired_trade_qty_previous': [0.0],
            'volume_120': [1000.0],
            'qty': [-1.0],  # Short position
            'position': [-50000.0],
            'cash': [100000.0],
            'expanding_previous': [0.0],
            'funding_income': [0.0],
            'position_ts_previous': [ts - pd.Timedelta(hours=1)],  # Position opened 1 hour ago
            'executed_position_ts': [ts],  # Current timestamp
            'fill_slippage': [0.0],  # Initialize slippage tracking
            'realized_funding_rate': [0.0],  # Initialize funding rate
            'percent_executed_at_funding': [0.0]  # Initialize funding percentage
        })
        
        # pylint: disable=protected-access
        result = sim._execute(ts_df, ts)
        
        # Check forced liquidation
        self.assertEqual(result.iloc[0]['executed_qty'], 1.0)  # Cover short
        self.assertEqual(result.iloc[0]['qty'], 0.0)
        # Liquidation price = close_mid_previous * exp(volatility * sigma_level) * (1 + slippage)
        # = 50000 * exp(0.02 * 3.0) * 1.001 ≈ 53144.92
        self.assertAlmostEqual(float(result.iloc[0]['trade_price']), 53144.92, places=0)
        
        
class TestSimulateOptimization(unittest.TestCase):
    """Test target generation and optimization."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = get_test_config()
        self.config['FCASTS'] = {}
        
    @patch('lib.sim.simulate.PortfolioOptimizer')
    @patch('lib.sim.simulate.Calcs')
    @patch('lib.sim.simulate.DataLoader')
    def test_generate_targets_success(self, _mock_data_loader, mock_calcs_class, mock_optimizer_class):
        """Test successful target generation."""
        mock_calcs = Mock()
        mock_calcs_class.return_value = mock_calcs
        
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        
        sim = Simulate(self.config, '/test/sim')
        sim.calcs = mock_calcs
        sim.trade_bot = mock_optimizer
        
        ts = pd.Timestamp('2023-01-01 12:00:00', tz='UTC')
        
        # Create test DataFrame
        ts_df = pd.DataFrame({
            'symbol_venue': ['BTC_binance'],
            'alpha_1440': [0.01],
            'position': [0.0]
        }).set_index('symbol_venue')
        
        # Mock methods
        mock_calcs.calculate_risk.return_value = ts_df
        mock_optimizer.make_alpha_opt.return_value = (ts_df, None)
        mock_calcs.calculate_tradeable_filter.return_value = ts_df
        mock_calcs.calculate_expandable_filter.return_value = ts_df
        mock_calcs.calculate_exclusions_and_bounds.return_value = ts_df
        
        # Add expected columns
        ts_df['target_opt'] = 10000.0
        mock_optimizer.optimize.return_value = ts_df
        mock_optimizer.calculate_targets.return_value = ts_df
        
        ts_df['desired_trade_dollars'] = 10000.0
        ts_df['desired_trade_qty'] = 0.2
        mock_optimizer.generate_desired_trades.return_value = ts_df
        
        # pylint: disable=protected-access
        result = sim._generate_targets(ts_df, ts)
        
        # Check results
        self.assertIn('target_opt', result.columns)
        self.assertIn('desired_trade_dollars', result.columns)
        self.assertIn('unexecuted_dollars', result.columns)
        self.assertEqual(result.iloc[0]['unexecuted_dollars'], 0.0)
        
        # Verify calls
        mock_calcs.calculate_risk.assert_called_once()
        mock_optimizer.make_alpha_opt.assert_called_once()
        mock_optimizer.optimize.assert_called_once()
        
    @patch('lib.sim.simulate.PortfolioOptimizer')
    @patch('lib.sim.simulate.Calcs')
    @patch('lib.sim.simulate.DataLoader')
    def test_generate_targets_optimization_failure(self, _mock_data_loader, mock_calcs_class, mock_optimizer_class):
        """Test target generation when optimization fails."""
        mock_calcs = Mock()
        mock_calcs_class.return_value = mock_calcs
        
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        
        sim = Simulate(self.config, '/test/sim')
        sim.calcs = mock_calcs
        sim.trade_bot = mock_optimizer
        
        ts = pd.Timestamp('2023-01-01 12:00:00', tz='UTC')
        
        # Create test DataFrame
        ts_df = pd.DataFrame({
            'symbol_venue': ['BTC_binance'],
            'alpha_1440': [0.01],
            'position': [10000.0],
            'desired_trade_dollars_previous': [5000.0],
            'executed_dollars': [0.0],
            'close_mid': [50000.0]
        }).set_index('symbol_venue')
        
        # Mock optimization failure
        mock_calcs.calculate_risk.return_value = ts_df
        mock_optimizer.make_alpha_opt.side_effect = OptimizationError("Test error")
        
        # pylint: disable=protected-access
        result = sim._generate_targets(ts_df, ts)
        
        # Check fallback behavior
        self.assertEqual(result.iloc[0]['unexecuted_dollars'], 5000.0)
        self.assertEqual(result.iloc[0]['desired_trade_dollars'], 5000.0)
        self.assertEqual(result.iloc[0]['desired_trade_qty'], 0.1)
        
        
class TestSimulatePositionAge(unittest.TestCase):
    """Test position age calculation."""
    
    def test_calculate_position_age_new_position(self):
        """Test age calculation for new position."""
        config = get_test_config()
        config['FCASTS'] = {}
        
        with patch('lib.sim.simulate.PortfolioOptimizer'), \
             patch('lib.sim.simulate.Calcs'), \
             patch('lib.sim.simulate.DataLoader'):
            sim = Simulate(config, '/test/sim')
        
        current_ts = pd.Timestamp('2023-01-10 12:00:00', tz='UTC')
        
        # New position (no previous)
        ts_df = pd.DataFrame({
            'qty': [1.0],
            'executed_qty': [1.0],
            'position_ts_previous': [pd.NaT],
            'executed_position_ts': [current_ts]
        })
        
        result = sim._calculate_position_age(ts_df, current_ts)
        
        self.assertEqual(result.iloc[0]['position_ts'], current_ts)
        self.assertEqual(result.iloc[0]['position_age'], 0)
        
    def test_calculate_position_age_increased_position(self):
        """Test age calculation for increased position."""
        config = get_test_config()
        config['FCASTS'] = {}
        
        with patch('lib.sim.simulate.PortfolioOptimizer'), \
             patch('lib.sim.simulate.Calcs'), \
             patch('lib.sim.simulate.DataLoader'):
            sim = Simulate(config, '/test/sim')
        
        current_ts = pd.Timestamp('2023-01-10 12:00:00', tz='UTC')
        old_ts = pd.Timestamp('2023-01-05 12:00:00', tz='UTC')
        
        # Increased position
        ts_df = pd.DataFrame({
            'qty': [2.0],
            'executed_qty': [1.0],
            'position_ts_previous': [old_ts],
            'executed_position_ts': [current_ts]
        })
        
        result = sim._calculate_position_age(ts_df, current_ts)
        
        # Should be weighted average (existing 1 unit at old_ts, new 1 unit at current_ts)
        # Weighted average timestamp calculation
        self.assertIsInstance(result.iloc[0]['position_ts'], pd.Timestamp)
        # Position age should be between 0 and 5 days
        self.assertGreaterEqual(result.iloc[0]['position_age'], 0)
        self.assertLessEqual(result.iloc[0]['position_age'], 5)
        
        
class TestSimulateIntegration(unittest.TestCase):
    """Test full simulation integration."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = get_test_config()
        self.config['REOPTIMIZE_INTERVAL_MINS'] = 1440  # Daily
        
    @patch('lib.sim.simulate.FillPnl')
    @patch('lib.sim.simulate.FillBreakdown')
    @patch('lib.sim.simulate.PortfolioOptimizer')
    @patch('lib.sim.simulate.Calcs')
    @patch('lib.sim.simulate.DataLoader')
    def test_simulate_single_day(self, mock_data_loader_class, mock_calcs_class, 
                                 mock_optimizer_class, mock_breakdown_class, mock_pnl_class):
        """Test simulating a single day."""
        # Set up complex mocks
        mock_data_loader = Mock()
        mock_data_loader_class.return_value = mock_data_loader
        
        mock_calcs = Mock()
        mock_calcs_class.return_value = mock_calcs
        
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        mock_optimizer.util_metrics = {}
        
        mock_pnl = Mock()
        mock_pnl_class.return_value = mock_pnl
        mock_pnl.calculator = Mock()
        mock_pnl.calculator.fills_pnl_df = pd.DataFrame()
        mock_pnl.position_loader = Mock()
        mock_pnl.fill_loader = Mock()
        mock_pnl.bar_loader = Mock()
        mock_pnl.fundings_loader = Mock()
        
        mock_breakdown = Mock()
        mock_breakdown_class.return_value = mock_breakdown
        
        sim = Simulate(self.config, '/test/sim')
        
        # Mock _sim_data_for_date
        test_date = date(2023, 1, 1)
        test_ts = pd.Timestamp('2023-01-01 00:00:00', tz='UTC')
        
        sim_data = pd.DataFrame({
            'ts': [test_ts],
            'symbol': ['BTC'],
            'symbol_venue': ['BTC_binance'],
            'close_mid': [50000.0],
            'close_mid_previous': [49000.0],
            'vwap_1440': [49500.0],
            'volume_1440': [1000.0],
            'alpha_1440': [0.01],
            'dvolume_1440_trmean': [1000000.0],
            'tradeable': [True],
            'expandable': [True],
            'logret_1440_trstd': [0.02],
            'risk_1440': [0.02],
            'realized_funding_rate': [0.0],
            'percent_executed_at_funding': [0.0],
            'high_mid_1440': [50500.0]
        })
        
        # pylint: disable=protected-access
        with patch.object(sim, '_sim_data_for_date', return_value=sim_data):
            # Mock optimization results
            mock_calcs.calculate_risk.side_effect = lambda df, **kwargs: df
            mock_optimizer.make_alpha_opt.side_effect = lambda df, **kwargs: (df, None)
            mock_calcs.calculate_tradeable_filter.side_effect = lambda df: df
            mock_calcs.calculate_expandable_filter.side_effect = lambda df: df
            mock_calcs.calculate_exclusions_and_bounds.side_effect = lambda df: df
            
            def add_target_cols(df):
                df = df.copy()
                df['target_opt'] = 10000.0
                return df
            
            mock_optimizer.optimize.side_effect = add_target_cols
            mock_optimizer.calculate_targets.side_effect = lambda df: df
            
            def add_trade_cols(df):
                df = df.copy()
                df['desired_trade_dollars'] = 10000.0
                df['desired_trade_qty'] = 0.2
                return df
            
            mock_optimizer.generate_desired_trades.side_effect = add_trade_cols

            # Add mock fills to avoid "No fills generated" error
            sim.fills_dfs = [pd.DataFrame({
                'ts': [test_ts],
                'symbol_venue': ['BTC_binance'],
                'date': [test_date],
                'symbol': ['BTC'],
                'executed_qty': [0.1],
                'trade_price': [50000.0],
                'commission': [5.0]
            })]

            # Add mock fundings to avoid "No fundings generated" error
            sim.fundings_dfs = [pd.DataFrame({
                'ts': [test_ts],
                'symbol_venue': ['BTC_binance'],
                'funding_income': [10.0]
            })]

            # Mock PnL calculation
            mock_pnl.run_pnl_calculation.return_value = pd.DataFrame()
            mock_pnl.run_pnl_aggregation.return_value = pd.DataFrame()
            mock_pnl.run_pnl_aggregation_by_ts.return_value = pd.DataFrame({
                'ts': [test_ts],
                'pnl': [100.0],
                'long': [10000.0],
                'short': [0.0],
                'unexecuted': [0.0],
                'fill_slip': [0.0],
                'expected_utility': [0.0],
                'expected_risk': [0.0],
                'expected_factor_risk': [0.0],
                'expected_resid_risk': [0.0],
                'expected_return': [0.0],
                'expected_return_bps': [0.0],
                'expected_slippage': [0.0],
                'expected_fees': [0.0],
                'fees_usd': [0.5],
                'funding_income': [0.0]
            })
            
            # Mock file writing operations and PnL dumping
            with patch('pandas.DataFrame.to_parquet'), \
                 patch('pandas.DataFrame.to_csv'), \
                 patch.object(sim, '_dump_pnl_file'):
                # Run simulation
                result = sim.simulate(
                    start_date=test_date,
                    end_date=test_date
                )
            
            # Verify simulation ran
            self.assertIsInstance(result, pd.DataFrame)
            self.assertGreater(len(result), 0)
            
            # Verify key methods called
            mock_optimizer.optimize.assert_called()
            mock_pnl.run_pnl_calculation.assert_called_once()
            

if __name__ == '__main__':
    unittest.main()
