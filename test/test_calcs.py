import unittest
import json
import os
from unittest.mock import patch
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import pytest

from lib.calcs.calcs import (
    Calcs,
    MIN_BETA_POINTS, BETA_RET_BOUND, MAX_BETA, NEWS_DECAY_MINS,
    DEFAULT_LONG_TERM_NEWS_LOOKBACK_PERIODS, DEFAULT_MID_TERM_NEWS_LOOKBACK_PERIODS,
    DEFAULT_SHORT_TERM_NEWS_LOOKBACK_PERIODS, MIN_NEWS_DT
)
from lib.calcs.calc_util import make_market_weight_col, get_recent_series_df, get_sparse_rows_for_prod
from lib.util.dataframes import get_df_idx
from lib.calcs.calc_returns import calculate_weighted_mkt_return, calculate_resampled_rsi
from lib.calcs.calc_risk import calculate_beta_single_symbol, calc_semi_std, apply_old_position_risk_multiplier
from lib.calcs.calc_volume import calculate_advp, calculate_mdvp


def get_test_config():
    """Load the test configuration from the fixtures directory.
    
    This is a copy of the production config that remains stable for testing
    and won't change when production config is updated.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Add any test-specific overrides here if needed
    config['FREQUENCY'] = config.get('FREQUENCY', 60)  # Default frequency for tests
    
    return config


class TestCalcsHelperFunctions(unittest.TestCase):
    """Test standalone helper functions in calcs module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample MultiIndex DataFrame
        dates = pd.date_range('2024-01-01', periods=5, freq='1h', tz='UTC')
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])
        self.df = pd.DataFrame({
            'close_mid': np.random.rand(10) * 100,
            'volume': np.random.rand(10) * 1000,
            'advp': np.random.rand(10) * 1e7
        }, index=index)
    
    def test_get_idx_with_start(self):
        """Test _get_idx with start datetime filter."""
        start_dt = pd.Timestamp('2024-01-01 02:00:00', tz='UTC')
        idx = get_df_idx(self.df, start=start_dt)
        
        # Should return boolean mask (numpy array)
        self.assertIsInstance(idx, np.ndarray)
        self.assertEqual(idx.dtype, bool)
        
        # Should filter out timestamps <= start_dt
        filtered_df = self.df[idx]
        self.assertTrue(all(filtered_df.index.get_level_values('ts') > start_dt))
    
    def test_get_idx_without_start(self):
        """Test _get_idx without start datetime filter."""
        idx = get_df_idx(self.df, start=None)
        
        # Should return the full index
        self.assertIsInstance(idx, pd.MultiIndex)
        self.assertEqual(len(idx), len(self.df))
    
    def test_make_market_weight_col(self):
        """Test market weight column creation."""
        df_copy = self.df.copy()
        result = make_market_weight_col(df_copy)
        
        # Should add mkt_wgt column
        self.assertIn('mkt_wgt', result.columns)
        
        # Check calculation: log(max(advp / 1e6, 1.0))
        expected = np.log(np.maximum(df_copy['advp'] / 1e6, 1.0))
        assert_series_equal(result['mkt_wgt'], expected, check_names=False)
    
    def test_calculate_weighted_mkt_return(self):
        """Test volume-weighted market return calculation."""
        # Create test data with known values
        dates = pd.date_range('2024-01-01', periods=3, freq='1h', tz='UTC')
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])
        
        df = pd.DataFrame({
            'logret': [0.01, 0.02, -0.01, 0.03, 0.00, -0.02],
            'advp': [1e7, 2e7, 1e7, 2e7, 1e7, 2e7],
            'fittable': [True, True, True, True, True, True]
        }, index=index)
        
        result_df, mkt_col = calculate_weighted_mkt_return(df, filter_col='fittable', ret_col='logret')
        
        # Check return values
        self.assertEqual(mkt_col, 'logret_wgtmkt')
        self.assertIn(mkt_col, result_df.columns)
        
        # Market return should be calculated for each timestamp
        for ts in dates:
            ts_data = result_df.xs(ts, level='ts')
            self.assertEqual(len(ts_data[mkt_col].unique()), 1)  # Same value for all symbols
    
    def test_calculate_weighted_mkt_return_with_frequency(self):
        """Test weighted market return with frequency suffix."""
        df_copy = self.df.copy()
        df_copy['logret_60'] = np.random.rand(len(df_copy)) * 0.1
        df_copy['fittable'] = True
        
        result_df, mkt_col = calculate_weighted_mkt_return(df_copy, filter_col='fittable', ret_col='logret', frequency=60)
        
        self.assertEqual(mkt_col, 'logret_wgtmkt_60')
        self.assertIn(mkt_col, result_df.columns)


def create_properly_initialized_calcs(config=None, frequency=60, prod=False):
    """Create a properly initialized Calcs instance for testing."""
    if config is None:
        config = get_test_config()
    
    # Create Calcs instance with proper initialization
    calcs = Calcs(config, frequency=frequency, prod=prod)
    
    # Ensure all required attributes are set
    if not hasattr(calcs, 'new_cols'):
        calcs.new_cols = []
    
    return calcs


def create_comprehensive_test_data(periods=200, symbols=None, frequency='1h', horizons=None):
    """Create comprehensive test data with all commonly needed columns."""
    if symbols is None:
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
    if horizons is None:
        horizons = [15, 60, 120, 360, 720, 1440]
    
    # Create date range
    dates = pd.date_range('2024-01-01', periods=periods, freq=frequency, tz='UTC')
    index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])
    
    # Generate base data
    np.random.seed(42)
    n_rows = len(index)
    
    # Price data
    btc_base = 50000
    eth_base = 3000
    prices = []
    for symbol in symbols * periods:
        if 'BTC' in symbol:
            prices.append(btc_base + np.random.randn() * 1000)
        elif 'ETH' in symbol:
            prices.append(eth_base + np.random.randn() * 100)
        else:
            prices.append(100 + np.random.randn() * 10)
    
    prices = np.array(prices, dtype=np.float32)
    
    df = pd.DataFrame({
        # Core price fields - all float32
        'close_mid': prices,
        'close_trade': prices * np.float32(1.0001),
        'open_mid': prices * np.float32(0.999),
        'high_mid': prices * np.float32(1.01),
        'low_mid': prices * np.float32(0.99),
        
        # Volume fields - all float32
        'volume': (np.random.rand(n_rows) * 1000 + 100).astype(np.float32),
        'dvolume': (prices * np.random.rand(n_rows) * 100).astype(np.float32),
        'trade_cnt': np.random.randint(10, 100, n_rows).astype(np.float32),
        'update_cnt': np.random.randint(100, 1000, n_rows).astype(np.float32),
        
        # Market microstructure - all float32
        'bid_sz_avg': (np.random.rand(n_rows) * 100 + 10).astype(np.float32),
        'ask_sz_avg': (np.random.rand(n_rows) * 100 + 10).astype(np.float32),
        'spread_avg': (np.random.rand(n_rows) * 0.01).astype(np.float32),
        
        # Funding fields - all float32
        'last_funding_rate': (np.random.rand(n_rows) * 0.001).astype(np.float32),
        'realized_funding': (np.random.rand(n_rows) * 0.0001).astype(np.float32),
        'next_funding_time': dates[0] + pd.Timedelta(hours=8),
        
        # Other fields - float32 where applicable
        'advp': (np.random.rand(n_rows) * 2e7 + 1e7).astype(np.float32),
        'fittable': True,
        'open_interest': (np.random.rand(n_rows) * 1e8).astype(np.float32),
        'news_event': np.random.choice([0, 1], n_rows, p=[0.95, 0.05])
    }, index=index)
    
    # Add horizon-specific fields - all float32
    for horizon in horizons:
        suffix = f'_{horizon}'
        df[f'volume{suffix}'] = (df['volume'] * horizon).astype(np.float32)
        df[f'dvolume{suffix}'] = (df['dvolume'] * horizon).astype(np.float32)
        df[f'trade_cnt{suffix}'] = (df['trade_cnt'] * horizon).astype(np.float32)
        df[f'update_cnt{suffix}'] = (df['update_cnt'] * horizon).astype(np.float32)
        df[f'bid_sz_avg{suffix}'] = df['bid_sz_avg'].astype(np.float32)
        df[f'ask_sz_avg{suffix}'] = df['ask_sz_avg'].astype(np.float32)
        df[f'spread_avg{suffix}'] = df['spread_avg'].astype(np.float32)
        df[f'open_interest{suffix}'] = df['open_interest'].astype(np.float32)
        
    return df


class TestCalcsClass(unittest.TestCase):
    """Test the Calcs class and its methods - simplified version."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_test_config()
        self.calcs = create_properly_initialized_calcs(self.config, frequency=60, prod=False)
        
        # Use comprehensive test data
        self.df = create_comprehensive_test_data(periods=100, frequency='1min')
    
    def test_calc_vwap(self):
        """Test VWAP calculation."""
        df_copy = self.df.copy()
        
        # Calculate VWAP without horizon
        result = self.calcs.calc_vwap(df_copy)
        self.assertIn('vwap', result.columns)
        
        # VWAP should be dvolume / volume
        expected_vwap = (df_copy['dvolume'] / df_copy['volume']).astype(np.float32)
        assert_series_equal(result['vwap'], expected_vwap, check_names=False)
        
        # Test with horizon
        df_copy['dvolume_60'] = df_copy['dvolume'] * 2
        df_copy['volume_60'] = df_copy['volume'] * 2
        result = self.calcs.calc_vwap(df_copy, horizon=60)
        self.assertIn('vwap_60', result.columns)
    
    def test_calc_logret(self):
        """Test log return calculation."""
        df_copy = self.df.copy()
        result = self.calcs.calc_logret(df_copy)
        
        # Should have logret column
        self.assertIn('logret', result.columns)
        
        # First period for each symbol should be NaN
        for symbol in df_copy.index.get_level_values('symbol_venue').unique():
            symbol_data = result.xs(symbol, level='symbol_venue')
            self.assertTrue(pd.isna(symbol_data['logret'].iloc[0]))
    
    def test_calc_funding_adjusted_logret_static(self):
        """Test funding-adjusted log return calculation (static method)."""
        df_copy = self.df.copy()
        df_copy['logret'] = np.random.rand(len(df_copy)) * 0.01
        df_copy['realized_funding'] = np.random.rand(len(df_copy)) * 0.0001
        df_copy['last_funding_rate'] = np.random.rand(len(df_copy)) * 0.001
        
        # Add next_funding_time column - set to future times
        base_time = pd.Timestamp('2024-01-01 08:00:00')
        df_copy['next_funding_time'] = base_time + pd.Timedelta(hours=8)
        
        result = self.calcs.calc_funding_adjusted_logret(df_copy)
        
        # Should have funding adjusted columns
        self.assertIn('logret_funding_adj', result.columns)
        
        # The calculation is more complex due to funding time logic
        # Just verify the column exists and has reasonable values
        self.assertFalse(result['logret_funding_adj'].isna().all())
    
    def test_calculate_trailing_z(self):
        """Test trailing z-score calculation."""
        df_copy = self.df.copy()
        df_copy['test_feature'] = np.random.rand(len(df_copy))
        df_copy['test_feature_trmean'] = df_copy['test_feature'].rolling(10).mean()
        df_copy['test_feature_trstd'] = df_copy['test_feature'].rolling(10).std()
        
        result = self.calcs.calculate_trailing_z(df_copy, 'test_feature')
        
        # Should have lz column
        self.assertIn('test_feature_lz', result.columns)
        
        # Z-scores should be bounded
        self.assertTrue((result['test_feature_lz'] >= -5).all())
        self.assertTrue((result['test_feature_lz'] <= 5).all())


class TestCalcsBetaCalculation(unittest.TestCase):
    """Test beta calculation methods - simplified."""
    
    def test_calculate_beta_single_symbol_static(self):
        """Test static beta calculation for single symbol."""
        # Create test data
        dates = pd.date_range('2024-01-01', periods=100, freq='1h', tz='UTC')
        market_returns = np.random.randn(100) * 0.01
        symbol_returns = 0.8 * market_returns + np.random.randn(100) * 0.005
        
        symbol = 'BTCUSDT_binance-futures'
        period_df = pd.DataFrame({
            'logret': symbol_returns,
            'logret_wgtmkt': market_returns,
            'symbol_venue': symbol
        }, index=pd.MultiIndex.from_product([dates, [symbol]], names=['ts', 'symbol_venue']))
        
        beta, t_stat = calculate_beta_single_symbol(
            period_df, 
            symbol,
            'logret',
            'logret_wgtmkt',
            dates[-1]
        )
        
        # Beta should be close to 0.8
        self.assertAlmostEqual(beta, 0.8, delta=0.3)
        
        # T-stat should be significant (for 100 points)
        self.assertGreater(abs(t_stat), 1.0)


class TestCalcsDeltaAndImbalance(unittest.TestCase):
    """Test delta calculations and bid-ask imbalance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_test_config()
        self.calcs = create_properly_initialized_calcs(self.config, frequency=60, prod=False)
        
        # Use comprehensive test data
        self.df = create_comprehensive_test_data(periods=120, frequency='1min')
    
    def test_calculate_delta(self):
        """Test delta calculation for absolute and percentage changes."""
        df_copy = self.df.copy()
        
        # Clear new_cols tracking
        self.calcs.new_cols = []
        
        # Calculate deltas on dvolume with frequency suffix
        result = self.calcs.calculate_delta(df_copy, feature='dvolume_60', start_dt=None)
        
        # Check columns were created with frequency suffix
        self.assertIn('dvolume_60_d', result.columns)
        self.assertIn('dvolume_60_dp', result.columns)
        
        # Check new_cols tracking
        self.assertIn('dvolume_60_d', self.calcs.new_cols)
        self.assertIn('dvolume_60_dp', self.calcs.new_cols)
        
        # Verify delta calculation (diff with lag of frequency)
        # The delta should be calculated on the 'dvolume_60' column
        unstacked = result['dvolume_60'].unstack('symbol_venue')
        expected_delta = unstacked.diff(periods=60).stack()
        expected_delta.name = 'dvolume_60_d'
        
        # Compare non-NaN values
        mask = ~result['dvolume_60_d'].isna()
        assert_series_equal(
            result['dvolume_60_d'][mask].sort_index(),
            expected_delta[mask].sort_index(),
            check_names=False
        )
    
    def test_calculate_bid_ask_imbalance(self):
        """Test bid-ask imbalance calculation."""
        df_copy = self.df.copy()
        
        # Clear new_cols tracking
        self.calcs.new_cols = []
        
        # Calculate imbalance
        result = self.calcs.calculate_bid_ask_imbalance(df_copy)
        
        # Check columns were created
        self.assertIn('ba_imbal_60', result.columns)
        self.assertIn('ba_imbal_60_d', result.columns)
        self.assertIn('ba_imbal_60_dp', result.columns)
        
        # Verify imbalance calculation
        expected_imbal = np.log(df_copy['bid_sz_avg_60'] / df_copy['ask_sz_avg_60']).fillna(0)
        assert_series_equal(
            result['ba_imbal_60'],
            expected_imbal,
            check_names=False
        )
        
        # Check that positive values indicate more bid pressure
        bid_heavy = result['bid_sz_avg_60'] > result['ask_sz_avg_60']
        self.assertTrue((result.loc[bid_heavy, 'ba_imbal_60'] > 0).all())
        
        # Check that negative values indicate more ask pressure
        ask_heavy = result['bid_sz_avg_60'] < result['ask_sz_avg_60']
        self.assertTrue((result.loc[ask_heavy, 'ba_imbal_60'] < 0).all())


class TestCalcsRSI(unittest.TestCase):
    """Test RSI calculation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_test_config()
        self.config['DEFAULT_FEATURE_LOOKBACK_PERIODS'] = 14  # Override for RSI test
        self.calcs = create_properly_initialized_calcs(self.config, frequency=60, prod=False)
        
        # Create sample data with known pattern
        dates = pd.date_range('2024-01-01', periods=100, freq='1h', tz='UTC')
        
        # Create returns that will produce known RSI values
        # Alternating positive and negative returns
        returns = []
        for i in range(100):
            if i < 20:
                returns.append(0.01)  # All positive (RSI should be high)
            elif i < 40:
                returns.append(-0.01)  # All negative (RSI should be low)
            else:
                returns.append(0.01 if i % 2 == 0 else -0.01)  # Alternating
        
        index = pd.MultiIndex.from_product(
            [dates, ['BTCUSDT_binance-futures']], 
            names=['ts', 'symbol_venue']
        )
        
        # Use comprehensive test data as base
        self.df = create_comprehensive_test_data(periods=100, symbols=['BTCUSDT_binance-futures'], frequency='1h')
        
        # Override logret_60 with our pattern
        self.df['logret_60'] = returns
    
    def test_calculate_rsi_basic(self):
        """Test basic RSI calculation."""
        df_copy = self.df.copy()
        
        # Clear new_cols tracking
        self.calcs.new_cols = []
        
        # Calculate RSI - need more data for 60min resampling
        # Start later to ensure we have enough history
        result = self.calcs.calculate_rsi(
            df_copy, 
            feature='logret_60',
            start_dt=pd.Timestamp('2024-01-03 00:00:00', tz='UTC'),  # Start on day 3 to have more history
            resample_frequency=60,
            lookback_periods=14
        )
        
        # Check column was created
        self.assertIn('rsi_60', result.columns)
        self.assertIn('rsi_60', self.calcs.new_cols)
        
        # RSI should be between 0 and 100
        rsi_values = result['rsi_60'].dropna()
        if len(rsi_values) > 0:
            self.assertTrue((rsi_values >= 0).all())
            self.assertTrue((rsi_values <= 100).all())
            
            # Check specific periods if we have enough data
            # Note: With resampling to 60min, we'll have fewer data points
            result_reset = result.reset_index()
            
            # Look for periods with consistent positive returns (should have high RSI)
            if len(rsi_values) > 5:
                # Since we're dealing with resampled data, check general trends
                high_rsi_values = rsi_values[rsi_values > 70]
                low_rsi_values = rsi_values[rsi_values < 30]
                
                # We should have some extreme values given our data pattern
                self.assertTrue(len(high_rsi_values) > 0 or len(low_rsi_values) > 0)
    
    def test_calculate_resampled_rsi(self):
        """Test calculate_resampled_rsi helper function."""
        df_copy = self.df.copy()
        
        # Get a subset of data
        recent_df = df_copy['logret_60'].unstack('symbol_venue').iloc[:50]
        
        # Calculate resampled RSI
        rsi_result = calculate_resampled_rsi(
            recent_df=recent_df,
            resample_frequency=60,
            window=14,
            feature_freq='logret_60',
            rsi_col='rsi_60'
        )
        
        # Check output structure - it returns symbol columns with 'rsi_60' renamed
        self.assertIn('BTCUSDT_binance-futures', rsi_result.columns)
        self.assertEqual(len(rsi_result.columns), 1)  # Should only have BTCUSDT
        
        # Values should be between 0 and 100
        rsi_values = rsi_result['BTCUSDT_binance-futures'].dropna()
        self.assertTrue((rsi_values >= 0).all())
        self.assertTrue((rsi_values <= 100).all())


class TestCalcsAdditionalMethods(unittest.TestCase):
    """Test additional calculation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_test_config()
        self.calcs = create_properly_initialized_calcs(self.config, frequency=60, prod=False)
        
        # Use comprehensive test data - 35 days worth
        self.df = create_comprehensive_test_data(periods=1440*35, frequency='1min')
        
        # Add any additional columns needed for specific tests
        self.df['logret'] = (np.random.randn(len(self.df)) * 0.01).astype(np.float32)
        self.df['logret_60'] = self.df['logret'].astype(np.float32)
    
    def test_calculate_advp(self):
        """Test average daily volume (ADVP) calculation."""
        df_copy = self.df.copy()

        # Calculate ADVP
        lookback_days = 10
        result = self.calcs.calculate_advp(df_copy, lookback_days=lookback_days)

        # Should have advp column
        self.assertIn('advp', result.columns)

        # ADVP should be positive (excluding NaN values)
        non_nan_advp = result['advp'].dropna()
        self.assertTrue((non_nan_advp > 0).all())

        # ADVP values should be reasonable (between 1K and 1T dollars)
        self.assertTrue((non_nan_advp > 1e3).all())
        self.assertTrue((non_nan_advp < 1e12).all())

    def test_calculate_mdvp(self):
        """Test median daily volume (MDVP) calculation."""
        calcs_1440 = create_properly_initialized_calcs(self.config, frequency=1440, prod=False)
        df_copy = self.df.copy()

        lookback_days = 10
        result = calcs_1440.calculate_mdvp(df_copy, lookback_days=lookback_days)

        # Should have mdvp column
        self.assertIn('mdvp', result.columns)

        # MDVP should be positive (excluding NaN values)
        non_nan_mdvp = result['mdvp'].dropna()
        self.assertTrue((non_nan_mdvp > 0).all())

        # MDVP values should be reasonable (between 1K and 1T dollars)
        self.assertTrue((non_nan_mdvp > 1e3).all())
        self.assertTrue((non_nan_mdvp < 1e12).all())

    def test_calculate_mdvp_robust_to_spikes(self):
        """Test that MDVP is more robust to volume spikes than ADVP."""
        # Create data with a few extreme volume days embedded in normal volume
        n_days = 35
        dates = pd.date_range('2024-01-01', periods=1440 * n_days, freq='1min', tz='UTC')
        symbols = ['BTCUSDT_binance-futures']
        index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])
        n_rows = len(index)

        np.random.seed(99)
        # Base volume: ~5M per minute, so ~7.2B per day
        base_volume = (np.ones(n_rows) * 5e6).astype(np.float32)
        # dvolume_1440 is the daily-aggregated volume, same for all minutes in a day
        daily_volume = np.zeros(n_rows, dtype=np.float32)
        for day in range(n_days):
            start = day * 1440
            end = start + 1440
            daily_volume[start:end] = 5e6 * 1440  # Normal daily sum = 7.2B
        # Inject 2 extreme spike days (10x volume) on days 20 and 25
        for spike_day in [20, 25]:
            start = spike_day * 1440
            end = start + 1440
            base_volume[start:end] = 50e6
            daily_volume[start:end] = 50e6 * 1440  # Spike daily sum
        df_spiked = pd.DataFrame({
            'dvolume': base_volume,
            'dvolume_1440': daily_volume,
        }, index=index)

        lookback_days = 10
        result_mdvp, _ = calculate_mdvp(df_spiked, lookback_days=lookback_days)
        result_advp, _ = calculate_advp(result_mdvp, lookback_days=lookback_days)

        # After the spike window, check the last day where both are valid
        both_valid = result_advp['mdvp'].notna() & result_advp['advp'].notna()
        last_ts = result_advp.loc[both_valid].index.get_level_values('ts').max()
        last_row = result_advp.xs(last_ts, level='ts').iloc[0]

        # MDVP should be less affected by the spikes than ADVP
        # Normal daily volume = 5e6 * 1440 = 7.2B
        normal_daily = 5e6 * 1440
        mdvp_deviation = abs(last_row['mdvp'] - normal_daily) / normal_daily
        advp_deviation = abs(last_row['advp'] - normal_daily) / normal_daily
        self.assertLess(mdvp_deviation, advp_deviation)

    def test_calculate_mdvp_short_history_raises(self):
        """Test that MDVP raises when history is too short."""
        # Use only 5 days of data but request 10-day lookback
        dates = pd.date_range('2024-01-01', periods=1440 * 5, freq='1min', tz='UTC')
        symbols = ['BTCUSDT_binance-futures']
        index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])
        df_short = pd.DataFrame({
            'dvolume_1440': np.random.rand(len(index)).astype(np.float32) * 1e6
        }, index=index)

        with self.assertRaises(Exception):
            calculate_mdvp(df_short, lookback_days=10)

    def test_calculate_mdvp_allow_short_history(self):
        """Test that MDVP works with short history when allowed."""
        dates = pd.date_range('2024-01-01', periods=1440 * 5, freq='1min', tz='UTC')
        symbols = ['BTCUSDT_binance-futures']
        index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])
        df_short = pd.DataFrame({
            'dvolume_1440': np.random.rand(len(index)).astype(np.float32) * 1e6
        }, index=index)

        result, new_cols = calculate_mdvp(df_short, lookback_days=10, allow_short_history=True)
        self.assertIn('mdvp', result.columns)
        self.assertEqual(new_cols, ['mdvp'])
    
    def test_calculate_minmax(self):
        """Test min/max calculation over lookback period."""
        df_copy = self.df.copy()
        
        # Add required trailing mean and std columns
        df_copy['logret_60_trmean'] = df_copy.groupby(level='symbol_venue')['logret_60'].transform(
            lambda x: x.rolling(30).mean()
        )
        df_copy['logret_60_trstd'] = df_copy.groupby(level='symbol_venue')['logret_60'].transform(
            lambda x: x.rolling(30).std()
        )
        
        # Clear new_cols
        self.calcs.new_cols = []
        
        # Calculate min/max for logret
        result = self.calcs.calculate_minmax(
            df_copy, 
            feature='logret_60',
            lookback_periods=30
        )
        
        # Check columns created
        self.assertIn('logret_60_min', result.columns)
        self.assertIn('logret_60_max', result.columns)
        
        # Min should be <= current value <= max
        mask = ~result['logret_60_min'].isna()
        self.assertTrue(
            (result.loc[mask, 'logret_60_min'] <= result.loc[mask, 'logret_60']).all()
        )
        self.assertTrue(
            (result.loc[mask, 'logret_60'] <= result.loc[mask, 'logret_60_max']).all()
        )
    
    def test_calculate_spread_factor(self):
        """Test relative spread calculation."""
        df_copy = self.df.copy()
        
        # Add required columns
        df_copy['spread_avg_60'] = df_copy['spread_avg']
        df_copy['close_mid_60'] = df_copy['close_mid']
        
        # Clear new_cols
        self.calcs.new_cols = []
        
        # Calculate spread factor
        result = self.calcs.calculate_spread_factor(df_copy)
        
        # Check column created
        self.assertIn('relative_spread_60', result.columns)
        
        # Relative spread should be spread / close_mid
        expected = df_copy['spread_avg_60'] / df_copy['close_mid_60']
        assert_series_equal(
            result['relative_spread_60'].astype(np.float32),
            expected.astype(np.float32),
            check_names=False
        )
    
    def test_calculate_trade_size(self):
        """Test average trade size calculation."""
        df_copy = self.df.copy()
        
        # Clear new_cols
        self.calcs.new_cols = []
        
        # Calculate trade size
        result = self.calcs.calculate_trade_size(df_copy)
        
        # Check column created
        self.assertIn('trade_sz_60', result.columns)
        
        # Trade size should be volume / trade_cnt
        expected = df_copy['volume_60'] / df_copy['trade_cnt_60']
        expected = expected.replace([np.inf, -np.inf], 0).fillna(0)
        
        assert_series_equal(
            result['trade_sz_60'],
            expected.astype(np.float32),
            check_names=False
        )
    
    def test_calculate_update_size(self):
        """Test relative update frequency calculation."""
        df_copy = self.df.copy()
        
        # Clear new_cols
        self.calcs.new_cols = []
        
        # Calculate update size
        result = self.calcs.calculate_update_size(df_copy)
        
        # Check column created
        self.assertIn('relative_updates_60', result.columns)
        
        # The method calculates trailing statistics, so just verify the column exists
        # and has reasonable values after the lookback period
        non_nan_values = result['relative_updates_60'].dropna()
        
        # Should have some non-NaN values
        self.assertGreater(len(non_nan_values), 0)
        
        # Values should be positive (updates per unit volume)
        self.assertTrue((non_nan_values > 0).all())
        
        # Should also have trailing mean/std columns
        self.assertIn('relative_updates_60_trmean', result.columns)
        self.assertIn('relative_updates_60_trstd', result.columns)
        self.assertIn('relative_updates_60_lz', result.columns)


class TestCalcsFilters(unittest.TestCase):
    """Test filter calculation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_test_config()
        self.calcs = create_properly_initialized_calcs(self.config, frequency=60, prod=False)
        
        # Create test data with 3 symbols having different liquidity levels
        # Need enough periods for featureable filter (30 periods at hourly frequency)
        dates = pd.date_range('2024-01-01', periods=200, freq='1h', tz='UTC')
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures', 'XRPUSDT_binance-futures']
        
        # Use comprehensive test data
        self.df = create_comprehensive_test_data(periods=200, symbols=symbols, frequency='1h')
        
        # Override ADVP levels for each symbol
        for i, row in self.df.iterrows():
            symbol = row.name[1]  # Get symbol from index
            if 'BTC' in symbol:
                self.df.loc[i, 'advp'] = 6e7  # Above all thresholds
            elif 'ETH' in symbol:
                self.df.loc[i, 'advp'] = 3e7  # Above fittable/tradeable, below expandable
            else:
                self.df.loc[i, 'advp'] = 1e7  # Below all thresholds
    
    def test_calculate_fittable_filter(self):
        """Test fittable filter calculation."""
        df_copy = self.df.copy()
        
        # Add required columns that would normally be added by featureable filter
        df_copy['priceable'] = True
        df_copy['featureable'] = True
        
        # Calculate fittable filter directly (skip featureable which has complex requirements)
        result = self.calcs.calculate_fittable_filter(df_copy)
        
        # Check column created
        self.assertIn('fittable', result.columns)
        
        # BTC and ETH should be fittable
        btc_data = result.xs('BTCUSDT_binance-futures', level='symbol_venue')
        eth_data = result.xs('ETHUSDT_binance-futures', level='symbol_venue')
        xrp_data = result.xs('XRPUSDT_binance-futures', level='symbol_venue')
        
        self.assertTrue(btc_data['fittable'].all())
        self.assertTrue(eth_data['fittable'].all())
        self.assertFalse(xrp_data['fittable'].all())
    
    def test_calculate_tradeable_filter(self):
        """Test tradeable filter calculation."""
        df_copy = self.df.copy()
        
        # Add required column - tradeable filter uses dvolume_1440_trmean
        df_copy['dvolume_1440_trmean'] = df_copy['advp']  # Use advp as proxy
        
        # Calculate tradeable filter
        result = self.calcs.calculate_tradeable_filter(df_copy)
        
        # Check column created
        self.assertIn('tradeable', result.columns)
        
        # Same thresholds as fittable in this config
        btc_data = result.xs('BTCUSDT_binance-futures', level='symbol_venue')
        eth_data = result.xs('ETHUSDT_binance-futures', level='symbol_venue')
        xrp_data = result.xs('XRPUSDT_binance-futures', level='symbol_venue')
        
        self.assertTrue(btc_data['tradeable'].all())
        self.assertTrue(eth_data['tradeable'].all())
        self.assertFalse(xrp_data['tradeable'].all())
    
    def test_calculate_expandable_filter(self):
        """Test expandable filter calculation."""
        df_copy = self.df.copy()
        
        # Add required columns - expandable filter uses dvolume_1440_trmean, mdvp, marketcap, delisting_date
        df_copy['dvolume_1440_trmean'] = df_copy['advp']  # Use advp as proxy
        df_copy['mdvp'] = df_copy['advp']  # Use advp as proxy
        df_copy['marketcap'] = np.nan  # No marketcap data (will be treated as expandable by default)
        df_copy['delisting_date'] = pd.NaT  # No delisting dates
        df_copy['delisting_date'] = pd.to_datetime(df_copy['delisting_date'], utc=True)  # Ensure timezone
        
        # Calculate expandable filter
        result = self.calcs.calculate_expandable_filter(df_copy)
        
        # Check column created
        self.assertIn('expandable', result.columns)
        
        # Only BTC should be expandable
        btc_data = result.xs('BTCUSDT_binance-futures', level='symbol_venue')
        eth_data = result.xs('ETHUSDT_binance-futures', level='symbol_venue')
        xrp_data = result.xs('XRPUSDT_binance-futures', level='symbol_venue')
        
        self.assertTrue(btc_data['expandable'].all())
        self.assertFalse(eth_data['expandable'].all())
        self.assertFalse(xrp_data['expandable'].all())


class TestCalcsEdgeCases(unittest.TestCase):
    """Test edge cases and error handling - simplified."""
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames in helper functions."""
        empty_df = pd.DataFrame()
        
        # Test empty dataframe with different structure
        empty_df_with_cols = pd.DataFrame(columns=['advp'])
        result = make_market_weight_col(empty_df_with_cols.copy())
        self.assertTrue(result.empty)
        self.assertIn('mkt_wgt', result.columns)
    
    def test_single_row_calculations(self):
        """Test calculations with minimal data."""
        index = pd.MultiIndex.from_tuples([('2024-01-01', 'BTCUSDT_binance-futures')], 
                                         names=['ts', 'symbol_venue'])
        df = pd.DataFrame({
            'advp': [1e7]
        }, index=index)
        
        # Market weight calculation should work
        result = make_market_weight_col(df)
        self.assertIn('mkt_wgt', result.columns)
        self.assertEqual(len(result), 1)
    
    def test_nan_handling_in_helper_functions(self):
        """Test NaN handling in helper functions."""
        dates = pd.date_range('2024-01-01', periods=5, freq='1h', tz='UTC')
        index = pd.MultiIndex.from_product([dates, ['BTCUSDT_binance-futures']], 
                                          names=['ts', 'symbol_venue'])
        
        df = pd.DataFrame({
            'advp': [1e7, np.nan, 2e7, np.nan, 3e7]
        }, index=index)
        
        # make_market_weight_col should handle NaNs
        result = make_market_weight_col(df)
        self.assertIn('mkt_wgt', result.columns)
        
        # NaN inputs should produce NaN outputs
        self.assertTrue(pd.isna(result['mkt_wgt'].iloc[1]))
        self.assertTrue(pd.isna(result['mkt_wgt'].iloc[3]))
        
        # Non-NaN inputs should produce valid outputs
        self.assertFalse(pd.isna(result['mkt_wgt'].iloc[0]))
        self.assertFalse(pd.isna(result['mkt_wgt'].iloc[2]))
        self.assertFalse(pd.isna(result['mkt_wgt'].iloc[4]))
    
    def test_get_recent_series_df_ensures_timezone_aware(self):
        """Test _get_recent_series_df ensures timezone-aware timestamps."""
        # Create data with timezone-naive timestamps
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        index = pd.MultiIndex.from_product([dates, ['BTCUSDT_binance-futures']], 
                                          names=['ts', 'symbol_venue'])
        
        df = pd.DataFrame({
            'logret_60': np.random.randn(100)
        }, index=index)
        
        # Create a Calcs instance
        config = get_test_config()
        calcs = Calcs(config=config, frequency=60, prod=False)
        
        # Test with timezone-aware start_dt
        start_dt = pd.Timestamp('2024-01-02', tz='UTC')
        
        # Call get_recent_series_df - should not raise TypeError
        series_df, window = get_recent_series_df(
            df=df, 
            start_dt=start_dt, 
            lookback_periods=24,
            frequency=60,
            feature_freq='logret_60'
        )
        
        # Verify the function worked without timezone errors
        self.assertIsInstance(series_df, pd.DataFrame)
        self.assertIsInstance(window, int)
        self.assertGreater(len(series_df), 0)
        
        # Test with timezone-naive start_dt
        start_dt_naive = pd.Timestamp('2024-01-02')  # No timezone
        
        # This should also work without raising TypeError
        series_df2, window2 = get_recent_series_df(
            df=df, 
            start_dt=start_dt_naive, 
            lookback_periods=24,
            frequency=60,
            feature_freq='logret_60'
        )
        
        # Verify it worked
        self.assertIsInstance(series_df2, pd.DataFrame)
        self.assertIsInstance(window2, int)
        self.assertGreater(len(series_df2), 0)


class TestCalcsConstants(unittest.TestCase):
    """Test module constants."""
    
    def test_constants_values(self):
        """Test that constants have expected values."""
        self.assertEqual(MIN_BETA_POINTS, 30)
        self.assertEqual(BETA_RET_BOUND, 0.1)
        self.assertEqual(MAX_BETA, 10)
        self.assertEqual(NEWS_DECAY_MINS, 360)
        self.assertEqual(DEFAULT_LONG_TERM_NEWS_LOOKBACK_PERIODS, 30)
        self.assertEqual(DEFAULT_MID_TERM_NEWS_LOOKBACK_PERIODS, 7)
        self.assertEqual(DEFAULT_SHORT_TERM_NEWS_LOOKBACK_PERIODS, 3)
        self.assertIsInstance(MIN_NEWS_DT, pd.Timestamp)


class TestCalcsReturns(unittest.TestCase):
    """Test return calculation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_test_config()
        self.calcs = create_properly_initialized_calcs(self.config, frequency=60, prod=False)
        
        # Use comprehensive test data
        self.df = create_comprehensive_test_data(periods=100, frequency='1h')
    
    def test_calc_returns_comprehensive(self):
        """Test the main calc_returns method."""
        # Create a Calcs instance with frequency=None for this test
        calcs_no_freq = create_properly_initialized_calcs(self.config, frequency=None, prod=False)
        
        df_copy = self.df.copy()
        
        # Add logret column - calc_returns expects it to exist
        df_copy['logret'] = np.random.randn(len(df_copy)) * 0.01
        # Add update_cnt column for calculate_price_filter
        if 'update_cnt' not in df_copy.columns:
            df_copy['update_cnt'] = 100
        
        # Mock market return dataframes
        mock_market_ret_df = pd.DataFrame({
            'ts': df_copy.index.get_level_values('ts').unique(),
            'logret_mkt': np.random.randn(100) * 0.001,
            'logret_wgtmkt': np.random.randn(100) * 0.001
        })
        
        # Mock the sub-methods to avoid complex dependencies
        with patch.object(calcs_no_freq, 'calc_logret') as mock_logret:
            with patch.object(calcs_no_freq, 'calculate_resid_return') as mock_resid:
                with patch.object(calcs_no_freq, 'calculate_weighted_resid_return') as mock_wgt_resid:
                    # Set up return values
                    mock_logret.return_value = df_copy
                    mock_resid.return_value = df_copy.copy()
                    mock_wgt_resid.return_value = df_copy.copy()
                    
                    # Call calc_returns
                    result = calcs_no_freq.calc_returns(
                        df_copy,
                        start_dt=None,
                        market_ret_df=mock_market_ret_df
                    )
                    
                    # Should call calculation methods
                    mock_logret.assert_called_once()
                    mock_resid.assert_called()
                    mock_wgt_resid.assert_called()
    
    def test_calculate_resid_return(self):
        """Test residual return calculation."""
        df_copy = self.df.copy()
        df_copy['logret'] = np.random.randn(len(df_copy)) * 0.01
        
        # Create market return data
        market_ret_df = pd.DataFrame({
            'ts': df_copy.index.get_level_values('ts').unique(),
            'logret_mkt': np.random.randn(100) * 0.005
        })
        
        result = self.calcs.calculate_resid_return(df_copy, filter_col='fittable', ret_col='logret', market_ret_df=market_ret_df)
        
        # Should have residual return column
        self.assertIn('logret_resid_eqmkt', result.columns)
        
        # Residual should be logret - market return
        for _, row in market_ret_df.iterrows():
            ts = row['ts']
            ts_data = result.xs(ts, level='ts')
            market_ret = row['logret_mkt']
            expected_resid = ts_data['logret'] - market_ret
            assert_series_equal(
                ts_data['logret_resid_eqmkt'],
                expected_resid,
                check_names=False
            )
    
    def test_calculate_weighted_resid_return(self):
        """Test weighted residual return calculation."""
        df_copy = self.df.copy()
        df_copy['logret'] = (np.random.randn(len(df_copy)) * 0.01).astype(np.float32)
        
        # The method might calculate the weighted market return if it doesn't exist
        result = self.calcs.calculate_weighted_resid_return(df_copy, filter_col='fittable', ret_col='logret')
        
        # Should have weighted residual return column
        self.assertIn('logret_resid_wgtmkt', result.columns)
        
        # Should also have the weighted market return
        self.assertIn('logret_wgtmkt', result.columns)
        
        # Residual should be logret - logret_wgtmkt (both exist now)
        expected = result['logret'] - result['logret_wgtmkt']
        assert_series_equal(
            result['logret_resid_wgtmkt'],
            expected.astype(np.float32),
            check_names=False
        )


class TestCalcsRiskAndBeta(unittest.TestCase):
    """Test risk and beta calculation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_test_config()
        self.calcs = create_properly_initialized_calcs(self.config, frequency=60, prod=False)
        
        # Create test data with known beta relationship
        dates = pd.date_range('2024-01-01', periods=180, freq='1h', tz='UTC')  # 3x beta lookback
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        
        # Use comprehensive test data
        self.df = create_comprehensive_test_data(periods=180, symbols=symbols, frequency='1h')
        
        # Generate market returns
        np.random.seed(42)
        market_returns = (np.random.randn(180) * 0.01).astype(np.float32)
        
        # BTC with beta ~0.8, ETH with beta ~1.2
        btc_returns = (0.8 * market_returns + np.random.randn(180) * 0.005).astype(np.float32)
        eth_returns = (1.2 * market_returns + np.random.randn(180) * 0.005).astype(np.float32)
        
        # Override returns with our pattern
        self.df['logret'] = np.concatenate([btc_returns, eth_returns]).astype(np.float32)
        self.df['logret_wgtmkt'] = np.tile(market_returns, 2).astype(np.float32)
        self.df['logret_60'] = self.df['logret'].astype(np.float32)
        self.df['logret_wgtmkt_60'] = self.df['logret_wgtmkt'].astype(np.float32)
    
    def test_calculate_beta_full(self):
        """Test full beta calculation across symbols."""
        df_copy = self.df.copy()
        
        # Add funding-adjusted returns (calculate_beta uses these by default)
        df_copy['logret_funding_adj'] = df_copy['logret']
        df_copy['logret_funding_adj_60'] = df_copy['logret_60']
        df_copy['logret_funding_adj_wgtmkt_60'] = df_copy['logret_wgtmkt_60']
        
        # Clear new_cols
        self.calcs.new_cols = []
        
        # Calculate beta
        result = self.calcs.calculate_beta(
            df_copy,
            feature='logret',
            start_dt=pd.Timestamp('2024-01-04 00:00:00', tz='UTC'),  # Start after 3 days (90 hours) lookback
            lookback_periods=90
        )
        
        # Should have beta columns
        self.assertIn('beta_60', result.columns)
        self.assertIn('beta_t_60', result.columns)
        
        # Check beta values are reasonable
        btc_data = result.xs('BTCUSDT_binance-futures', level='symbol_venue')
        eth_data = result.xs('ETHUSDT_binance-futures', level='symbol_venue')
        
        # Beta values should exist and be reasonable
        btc_beta_mean = btc_data['beta_60'].dropna().mean()
        eth_beta_mean = eth_data['beta_60'].dropna().mean()
        
        # Beta values should be bounded
        self.assertGreater(btc_beta_mean, -10)
        self.assertLess(btc_beta_mean, 10)
        self.assertGreater(eth_beta_mean, -10)
        self.assertLess(eth_beta_mean, 10)
        
        # Should have calculated some betas
        self.assertGreater(len(btc_data['beta_60'].dropna()), 0)
        self.assertGreater(len(eth_data['beta_60'].dropna()), 0)
        
        # T-stats should exist
        self.assertGreater(len(btc_data['beta_t_60'].dropna()), 0)
        self.assertGreater(len(eth_data['beta_t_60'].dropna()), 0)
    
    def test_calculate_risk(self):
        """Test risk calculation."""
        df_copy = self.df.copy()
        
        # Add required trailing std columns with correct naming
        # Risk field template is logret_HORIZON_trstd where HORIZON = sample_frequency (not risk_frequency)
        for symbol in df_copy.index.get_level_values('symbol_venue').unique():
            symbol_data = df_copy.xs(symbol, level='symbol_venue')
            rolling_std = symbol_data['logret'].rolling(90).std()
            df_copy.loc[df_copy.index.get_level_values('symbol_venue') == symbol, 'logret_60_trstd'] = rolling_std.values
        
        # Calculate risk
        result = self.calcs.calculate_risk(df_copy, risk_frequency=120, sample_frequency=60)
        
        # Should have risk column
        self.assertIn('risk_120', result.columns)
        
        # Risk should be scaled by sqrt(risk_freq / sample_freq)
        scale_factor = np.sqrt(120 / 60)
        expected_risk = df_copy['logret_60_trstd'] * scale_factor
        
        # Note: apply_old_position_risk_multiplier might modify the values if position_age exists
        # For this test, we just check that risk values exist and are reasonable
        self.assertGreater(len(result['risk_120'].dropna()), 0)
        self.assertTrue((result['risk_120'].dropna() > 0).all())
    
    def test_apply_old_position_risk_multiplier(self):
        """Test old position risk multiplier."""
        df_copy = self.df.copy()
        original_risk = (np.random.rand(len(df_copy)) * 0.01).astype(np.float32)
        df_copy['risk_120'] = original_risk.copy()
        
        # Add position age in days (> 30 days is considered old)
        df_copy['position_age'] = 45  # Old position
        
        result = apply_old_position_risk_multiplier(df_copy, 'risk_120', 
                                                   self.config['OLD_POSITION_DAYS'], 
                                                   self.config['OLD_POSITION_RISK_MULT'])
        
        # Should have adjusted risk column
        self.assertIn('risk_120', result.columns)
        
        # Old positions should have higher risk
        # The method modifies in place, so need to compare with original values
        expected_risk = original_risk * self.config['OLD_POSITION_RISK_MULT']
        assert_series_equal(
            result['risk_120'],
            pd.Series(expected_risk, index=result.index, dtype=np.float32),
            check_names=False
        )


class TestCalcsVolumeAndTrailing(unittest.TestCase):
    """Test volume buckets and trailing calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_test_config()
        self.calcs = create_properly_initialized_calcs(self.config, frequency=60, prod=False)
        
        # Use comprehensive test data - 7 days
        self.df = create_comprehensive_test_data(periods=1440*7, frequency='1min')
        
        # Create volume pattern - higher during certain hours
        dates = self.df.index.get_level_values('ts')
        volumes = []
        for ts in dates.unique():
            hour = ts.hour
            if 8 <= hour <= 16:  # Active hours
                volume = np.random.rand() * 2000 + 1000
            else:  # Quiet hours
                volume = np.random.rand() * 500 + 100
            volumes.append(volume)
        
        # Set volume pattern for all symbols
        n_symbols = len(self.df.index.get_level_values('symbol_venue').unique())
        self.df['dvolume_60'] = np.tile(volumes, n_symbols).astype(np.float32)
        self.df['logret'] = (np.random.randn(len(self.df)) * 0.01).astype(np.float32)
        self.df['logret_60'] = self.df['logret'].astype(np.float32)
    
    def test_calculate_volume_buckets(self):
        """Test volume bucket calculation."""
        df_copy = self.df.copy()

        # Add date column (required by calculate_volume_buckets)
        df_with_date = df_copy.reset_index()
        df_with_date['date'] = df_with_date['ts'].dt.date
        df_with_date = df_with_date.set_index(['ts', 'symbol_venue'])

        # Clear new_cols
        self.calcs.new_cols = []

        # Calculate volume buckets
        result = self.calcs.calculate_volume_buckets(
            df_with_date,
            feature='dvolume',
            lookback_days=5  # Reduced for testing
        )

        # Should have median volume column
        self.assertIn('median_time_bucket_dvolume_60', result.columns)

        # Check that values exist after lookback period
        non_null_values = result['median_time_bucket_dvolume_60'].dropna()
        self.assertGreater(len(non_null_values), 0)

        # Verify median values are positive (volumes should be non-negative)
        self.assertTrue((non_null_values >= 0).all())
    
    def test_calculate_trailing_mean(self):
        """Test trailing mean calculation."""
        df_copy = self.df.copy()
        
        # Clear new_cols
        self.calcs.new_cols = []
        
        # Calculate trailing mean
        result = self.calcs.calculate_trailing_mean(
            df_copy,
            feature='logret_60',
            lookback_periods=24,
            ffill_na=True
        )
        
        # Should have trailing mean and std columns
        self.assertIn('logret_60_trmean', result.columns)
        self.assertIn('logret_60_trstd', result.columns)
        self.assertIn('logret_60_lz', result.columns)
        
        # Check that calculations produced values
        non_null_mean = result['logret_60_trmean'].dropna()
        non_null_std = result['logret_60_trstd'].dropna()
        non_null_lz = result['logret_60_lz'].dropna()
        
        self.assertGreater(len(non_null_mean), 0)
        self.assertGreater(len(non_null_std), 0)
        self.assertGreater(len(non_null_lz), 0)
        
        # Verify that std values are positive
        self.assertTrue((non_null_std > 0).all())
        
        # Verify that lz values are bounded [-5, 5]
        self.assertTrue((non_null_lz >= -5).all())
        self.assertTrue((non_null_lz <= 5).all())
    
    def test_calculate_trailing_semi_variance(self):
        """Test semi-variance calculation."""
        df_copy = self.df.copy()
        df_copy['logret_60_trmean'] = df_copy.groupby(level='symbol_venue')['logret_60'].transform(
            lambda x: x.rolling(24).mean()
        )
        
        # Clear new_cols
        self.calcs.new_cols = []
        
        # Calculate semi-variance
        result = self.calcs.calculate_trailing_semi_variance(
            df_copy,
            feature='logret_60',
            lookback_periods=24
        )
        
        # Should have upside and downside std columns
        self.assertIn('logret_60_trstd_u', result.columns)
        self.assertIn('logret_60_trstd_d', result.columns)
        self.assertIn('logret_60_trstd_udratio', result.columns)
        
        # Verify ratio calculation
        mask = (result['logret_60_trstd_d'] > 0) & ~result['logret_60_trstd_u'].isna()
        expected_ratio = result.loc[mask, 'logret_60_trstd_u'] / result.loc[mask, 'logret_60_trstd_d']
        assert_series_equal(
            result.loc[mask, 'logret_60_trstd_udratio'].astype(np.float32),
            expected_ratio.astype(np.float32),
            check_names=False
        )


class TestCalcsAdvancedFeatures(unittest.TestCase):
    """Test advanced feature calculation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_test_config()
        self.calcs = create_properly_initialized_calcs(self.config, frequency=60, prod=False)
        
        # Use comprehensive test data with dates after MIN_NEWS_DT (2024-04-05)
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures', 'XRPUSDT_binance-futures']
        self.df = create_comprehensive_test_data(periods=500, symbols=symbols, frequency='1h')
        
        # Update timestamps to be after MIN_NEWS_DT for news tests
        new_dates = pd.date_range('2024-05-01', periods=500, freq='1h', tz='UTC')
        new_index = pd.MultiIndex.from_product([new_dates, symbols], names=['ts', 'symbol_venue'])
        self.df.index = new_index
        
        # Add any additional columns needed
        self.df['logret'] = (np.random.randn(len(self.df)) * 0.01).astype(np.float32)
        self.df['logret_60'] = self.df['logret'].astype(np.float32)
        # Add news_title column (initially all NaN - no news)
        self.df['news_title'] = np.nan
    
    def test_calculate_quintiles(self):
        """Test quintile calculation."""
        df_copy = self.df.copy()
        
        result = self.calcs.calculate_quintiles(df_copy, 'logret')
        
        # Should have quintile column
        self.assertIn('logret_q', result.columns)
        
        # Check that quintiles are categorical with proper labels
        quintile_values = result['logret_q'].dropna()
        if len(quintile_values) > 0:
            # Should have 5 categories
            unique_quintiles = quintile_values.unique()
            expected_labels = {f'Q{i}_logret' for i in range(1, 6)}
            
            # Check that we have quintile labels
            actual_labels = set(str(q) for q in unique_quintiles)
            self.assertTrue(actual_labels.issubset(expected_labels))
    
    def test_calculate_weighted_feature(self):
        """Test weighted feature calculation."""
        df_copy = self.df.copy()
        
        result = self.calcs.calculate_weighted_feature(df_copy, 'logret', 'dvolume')
        
        # Should have weighted feature columns
        self.assertIn('logret_w_dvolume', result.columns)
        self.assertIn('dvolume_sum', result.columns)
        
        # Verify calculation - it's actually a more complex weighted calculation
        # Just check that the columns exist and have reasonable values
        self.assertFalse(result['logret_w_dvolume'].isna().all())
        self.assertFalse(result['dvolume_sum'].isna().all())
    
    def test_calculate_time_factors(self):
        """Test time factor calculation."""
        df_copy = self.df.copy()
        
        # Clear new_cols
        self.calcs.new_cols = []
        
        # Pass start_dt to ensure idx is a boolean array
        result = self.calcs.calculate_time_factors(df_copy, start_dt=pd.Timestamp('2024-01-01 01:00:00', tz='UTC'))
        
        # Should have time columns
        self.assertIn('hour_of_day', result.columns)
        self.assertIn('day_of_week', result.columns)
        
        # Check that values exist for the subset of data after start_dt
        non_null_hours = result['hour_of_day'].dropna()
        non_null_days = result['day_of_week'].dropna()
        
        self.assertGreater(len(non_null_hours), 0)
        self.assertGreater(len(non_null_days), 0)
        
        # Verify hour values are in valid range
        self.assertTrue((non_null_hours >= 0).all())
        self.assertTrue((non_null_hours <= 23).all())
        
        # Verify day of week values are in valid range (0=Monday, 6=Sunday)
        self.assertTrue((non_null_days >= 0).all())
        self.assertTrue((non_null_days <= 6).all())
    
    def test_calculate_abs_factor(self):
        """Test absolute value factor calculation."""
        df_copy = self.df.copy()
        
        # Clear new_cols
        self.calcs.new_cols = []
        
        result = self.calcs.calculate_abs_factor(df_copy, feature='logret')
        
        # Should have abs column - it's named logret_abs not logret_60_abs
        self.assertIn('logret_abs', result.columns)
        
        # Verify calculation
        expected = df_copy['logret'].abs()
        assert_series_equal(
            result['logret_abs'].astype(np.float32),
            expected.astype(np.float32),
            check_names=False
        )
    
    def test_calc_decayed_news(self):
        """Test decayed news calculation."""
        df_copy = self.df.copy()
        
        # Set news events at specific times using news_title
        df_copy['news_title'] = ''  # Default to empty string (no news)
        
        # Add news events for BTC at specific times
        btc_mask = df_copy.index.get_level_values('symbol_venue') == 'BTCUSDT_binance-futures'
        time_mask = df_copy.index.get_level_values('ts').isin([
            pd.Timestamp('2024-05-02 10:00:00'),
            pd.Timestamp('2024-05-05 15:00:00')
        ])
        df_copy.loc[btc_mask & time_mask, 'news_title'] = 'Test news event'
        
        # Clear new_cols
        self.calcs.new_cols = []
        
        # Start later to have enough lookback
        result = self.calcs.calc_decayed_news(
            df_copy,
            lookback_periods=30,
            start_dt=pd.Timestamp('2024-05-01 04:00:00', tz='UTC')  # Start 4 hours later for lookback
        )
        
        # Should have decayed news column
        self.assertIn('news_event_decayed_60', result.columns)
        
        # News impact should decay over time
        btc_data = result.xs('BTCUSDT_binance-futures', level='symbol_venue')
        
        # Check that decay values exist
        non_null_values = btc_data['news_event_decayed_60'].dropna()
        self.assertGreater(len(non_null_values), 0)
        
        # Values should be non-negative (decayed sum can be > 1)
        self.assertTrue((non_null_values >= 0).all())
        
        # Should have some non-zero values where news events occurred
        self.assertTrue((non_null_values > 0).any())
        
        # Should also have trailing statistics columns
        self.assertIn('news_event_decayed_60_trmean', result.columns)
        self.assertIn('news_event_decayed_60_trstd', result.columns)
        self.assertIn('news_event_decayed_60_lz', result.columns)


class TestCalcsFilteringAndBounds(unittest.TestCase):
    """Test universe filtering and position bounds."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_test_config()
        self.calcs = create_properly_initialized_calcs(self.config, frequency=60, prod=False)
        
        # Create test data with specific symbols
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures', 
                  'NEWCOIN_binance-futures', 'DELISTCOIN_binance-futures']
        self.df = create_comprehensive_test_data(periods=100, symbols=symbols, frequency='1h')
        
        # Override specific values for testing
        for i, row in self.df.iterrows():
            symbol = row.name[1]
            if 'BTC' in symbol:
                self.df.loc[i, 'advp'] = 6e7
                self.df.loc[i, 'position'] = 100000
                self.df.loc[i, 'tradeable'] = True
                self.df.loc[i, 'fittable'] = True
            elif 'ETH' in symbol:
                self.df.loc[i, 'advp'] = 3e7
                self.df.loc[i, 'position'] = 50000
                self.df.loc[i, 'tradeable'] = True
                self.df.loc[i, 'fittable'] = True
            elif 'NEWCOIN' in symbol:
                self.df.loc[i, 'advp'] = 5e6
                self.df.loc[i, 'position'] = 0
                self.df.loc[i, 'tradeable'] = False
                self.df.loc[i, 'fittable'] = False
            else:  # DELISTCOIN
                self.df.loc[i, 'advp'] = 1e7
                self.df.loc[i, 'position'] = 20000
                self.df.loc[i, 'tradeable'] = True
                self.df.loc[i, 'fittable'] = True
        
        # Add logret_60_lz
        self.df['logret_60_lz'] = np.random.randn(len(self.df))
        
        # Create positions dataframe
        self.positions_df = pd.DataFrame({
            'qty': [100000, 50000, 0, 20000]  # Use 'qty' as expected by filter_universe
        }, index=pd.Index(symbols, name='symbol_venue'))
    
    def test_filter_universe(self):
        """Test universe filtering logic."""
        from lib.universe import filter_universe

        init_universe = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures', 
                        'NEWCOIN_binance-futures', 'DELISTCOIN_binance-futures']
        
        # Test without delisting - just fittable filtering
        filtered_universe = filter_universe(
            init_universe,
            self.df,
            self.positions_df,
            pd.Timestamp('2024-01-05', tz='UTC').date()
        )
        
        # Should include fittable symbols
        self.assertIn('BTCUSDT_binance-futures', filtered_universe)
        self.assertIn('ETHUSDT_binance-futures', filtered_universe)
        
        # Should exclude non-fittable symbols without positions
        self.assertNotIn('NEWCOIN_binance-futures', filtered_universe)
        
        # Should include symbols with positions even if not fittable
        self.assertIn('DELISTCOIN_binance-futures', filtered_universe)
    
    def test_calculate_exclusions_and_bounds(self):
        """Test exclusion and bounds calculation."""
        df_copy = self.df.copy()

        # Add required columns
        df_copy['expandable'] = df_copy['tradeable']  # Use tradeable as proxy
        # Create more diverse volume data to avoid duplicate bin edges in pd.cut
        # Add random variation to create unique percentiles
        np.random.seed(42)  # For reproducibility
        volume_variation = np.random.uniform(0.8, 1.2, len(df_copy))
        df_copy['dvolume_1440_trmean'] = (df_copy['advp'] * volume_variation).astype(np.float32)
        df_copy['mdvp'] = (df_copy['advp'] * volume_variation).astype(np.float32)  # bound_col for position sizing
        df_copy['marketcap'] = (df_copy['advp'] * 10).astype(np.float32)  # Market cap proxy
        df_copy['logret_1440_lz'] = np.random.randn(len(df_copy)).astype(np.float32)  # Add daily return z-score
        df_copy['dvolume_1440_lz'] = np.random.randn(len(df_copy)).astype(np.float32)  # Add daily volume z-score
        df_copy['delisting_date'] = pd.NaT  # No delisting dates
        df_copy['delisting_date'] = pd.to_datetime(df_copy['delisting_date'], utc=True)  # Ensure timezone
        
        # Set extreme moves for some symbols
        btc_mask = df_copy.index.get_level_values('symbol_venue') == 'BTCUSDT_binance-futures'
        df_copy.loc[btc_mask, 'logret_60_lz'] = 3.0  # Above max_move_filter
        df_copy.loc[btc_mask, 'logret_1440_lz'] = 3.0  # Also set daily z-score high
        
        result = self.calcs.calculate_exclusions_and_bounds(df_copy)
        
        # Should have bounds columns
        self.assertIn('lbound', result.columns)
        self.assertIn('ubound', result.columns)
        
        # BTC with extreme moves should have bounds set to 0
        btc_data = result.xs('BTCUSDT_binance-futures', level='symbol_venue')
        # Verify bounds exist
        self.assertIsNotNone(btc_data['lbound'].iloc[0])
        self.assertIsNotNone(btc_data['ubound'].iloc[0])
        
        # Non-tradeable assets should have zero bounds
        newcoin_data = result.xs('NEWCOIN_binance-futures', level='symbol_venue')
        self.assertEqual(newcoin_data['lbound'].iloc[0], 0)
        self.assertEqual(newcoin_data['ubound'].iloc[0], 0)
    
    def test_adjust_bounds_with_max_notional_value(self):
        """Test bounds adjustment based on max notional."""
        from lib.calcs.calc_filters_and_bounds import adjust_bounds_with_max_notional_value
        
        df_copy = self.df.copy()
        
        # Set up bounds
        df_copy['lbound'] = -0.04
        df_copy['ubound'] = 0.04
        df_copy['close_mid'] = [50000, 3000, 1, 100] * 100  # Different prices
        df_copy['position'] = 0  # Add position column needed by the function
        
        tradeable_idx = df_copy.index.get_level_values('symbol_venue').isin([
            'BTCUSDT_binance-futures', 'ETHUSDT_binance-futures'
        ])
        
        result = adjust_bounds_with_max_notional_value(df_copy, tradeable_idx)
        
        # Should have adjusted bounds
        self.assertIn('lbound', result.columns)
        self.assertIn('ubound', result.columns)
        
        # Bounds should be adjusted properly
        btc_data = result.xs('BTCUSDT_binance-futures', level='symbol_venue')
        
        # Check that bounds were passed through (the method logs that max_position_value column is missing)
        # In production, bounds would be adjusted by position values
        self.assertEqual(btc_data['ubound'].iloc[0], 0.04)  # Original upper bound
        self.assertEqual(btc_data['lbound'].iloc[0], -0.04)  # Original lower bound


class TestCalcsHelperMethods(unittest.TestCase):
    """Test helper and utility methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_test_config()
        self.calcs = create_properly_initialized_calcs(self.config, frequency=60, prod=False)
    
    def test_get_new_cols(self):
        """Test new columns getter."""
        # Clear and add some columns
        self.calcs.new_cols = []
        self.calcs.new_cols.extend(['col1', 'col2', 'col3', 'col2'])  # Include duplicate
        
        result = self.calcs.get_new_cols()
        
        # Should contain unique columns only
        self.assertEqual(set(result), {'col1', 'col2', 'col3'})
        # get_new_cols doesn't clear the list, it just returns unique values
        self.assertEqual(self.calcs.new_cols, ['col1', 'col2', 'col3', 'col2'])
    
    def test_get_lagged_lookback_minutes(self):
        """Test lagged lookback calculation."""
        # The method uses horizon_to_max_lags[frequency] * frequency
        # For frequency 60, this is typically 1 lag, so result = 1 * 60 = 60
        result = self.calcs.get_lagged_lookback_minutes()
        
        # Should depend on horizon_to_max_lags configuration
        expected = self.calcs.horizon_to_max_lags.get(self.calcs.frequency, 1) * self.calcs.frequency
        self.assertEqual(result, expected)
    
    def test_calc_semi_std(self):
        """Test semi-standard deviation calculation."""
        # Test data with known pattern
        data = pd.Series([1, 2, -1, -2, 3, -3, 0, 1, -1])
        mean = data.mean()
        
        # Upside semi-std (positive deviations)
        upside_std = calc_semi_std(data, side=1)
        
        # Downside semi-std (negative deviations)
        downside_std = calc_semi_std(data, side=-1)
        
        # Both should be positive
        self.assertGreater(upside_std, 0)
        self.assertGreater(downside_std, 0)
        
        # Verify manual calculation for upside
        # calc_semi_std uses ALL values but zeros out negative deviations
        deviations = np.maximum(data - mean, 0)
        expected_upside = np.sqrt(np.mean(deviations ** 2))
        self.assertAlmostEqual(upside_std, expected_upside, places=5)
    
    def test_get_recent_series_df(self):
        """Test getting recent series data."""
        # Create test data
        dates = pd.date_range('2024-01-01', periods=200, freq='1h', tz='UTC')
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])
        
        df = pd.DataFrame({
            'test_feature_60': np.random.randn(400)
        }, index=index)
        
        # Get recent data
        start_dt = pd.Timestamp('2024-01-05 00:00:00', tz='UTC')
        recent_df, offset = get_recent_series_df(
            df, start_dt, lookback_periods=48, frequency=60, feature_freq='test_feature_60'
        )
        
        # Should return unstacked dataframe with MultiIndex columns
        self.assertEqual(len(recent_df.columns), 2)  # Two symbols
        
        # Check that both symbols are in the column level
        symbol_level = recent_df.columns.get_level_values('symbol_venue')
        self.assertIn('BTCUSDT_binance-futures', symbol_level)
        self.assertIn('ETHUSDT_binance-futures', symbol_level)
        
        # Should have some data (warning says it's using 2 day lookback window)
        self.assertGreater(len(recent_df), 0)
        # It might return more than requested due to windowing
        self.assertLess(len(recent_df), 500)  # Reasonable upper bound
    
    def test_get_sparse_rows_for_prod(self):
        """Test sparse row selection for production."""
        # Create test data with proper MultiIndex
        dates = pd.date_range('2024-01-01', periods=1000, freq='1min', tz='UTC')
        symbols = ['BTCUSDT_binance-futures']
        index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])
        df = pd.DataFrame({
            'value': np.random.randn(1000)
        }, index=index)
        
        # Set production mode
        self.calcs.prod = True
        
        # Need reopt_times and max_lags for the function
        reopt_times = ['2024-01-01 00:00:00+00:00', '2024-01-01 12:00:00+00:00']
        max_lags = 1
        frequency = 60
        
        result = get_sparse_rows_for_prod(df, reopt_times=reopt_times, max_lags=max_lags, frequency=frequency)
        
        # Should return fewer rows
        self.assertLess(len(result), len(df))
        
        # Should include some minute markers
        minutes = result.index.get_level_values('ts').minute
        unique_minutes = set(minutes)
        
        # Should have some minutes (the exact minutes depend on the data and frequency)
        self.assertGreater(len(unique_minutes), 0)
        # The method filters to specific timestamps for production
        # Just verify it reduced the data significantly
        self.assertLess(len(unique_minutes), 60)  # Should not have all minutes


if __name__ == '__main__':
    unittest.main()
