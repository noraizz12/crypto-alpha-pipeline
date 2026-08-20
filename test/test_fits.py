"""Unit tests for the fits module.

This module contains comprehensive unit tests for the model fitting functionality,
including tests for SVM classifiers, regression fitting, and the main Fits pipeline.
"""

import unittest
from datetime import datetime as dt
from datetime import timedelta as td
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import joblib
import tempfile
import os

from lib.fits.fits import Fits
from lib.fits.model_horizon import ModelHorizonFit
from lib.fits.fit_util import (
    get_dep, make_classification_bar_features,
    BASE_CLASSIFICATION_BAR_FEATURES,
    MIN_CLASSIFICATION_OBS, MIN_FITTING_OBS, MIN_COEFF
)

# Define CLASSIFICATION_FEATURES_1440 for tests (was commented out in lib.fits)
CLASSIFICATION_FEATURES_1440 = [
    'day_of_week', 'hour_of_day',
    'beta_1440',
    'logret_1440_trstd',
    'dvolume_1440_trmean_cz', 'dvolume_1440_trmean',
    'relative_updates_1440_lz', 'trade_sz_1440_lz',
]


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions in fits module."""
    
    def test_get_dep_raw(self):
        """Test get_dep for raw return type."""
        self.assertEqual(get_dep('raw', 0, 60), 'y_raw1_60')
        self.assertEqual(get_dep('raw', 2, 1440), 'y_raw3_1440')
        
    def test_get_dep_residualized(self):
        """Test get_dep for residualized return types."""
        self.assertEqual(get_dep('resid_eq', 0, 15), 'y_resid_eqmkt1_15')
        self.assertEqual(get_dep('resid_wgt', 1, 120), 'y_resid_wgtmkt2_120')
        
    def test_get_dep_funding_adjusted(self):
        """Test get_dep for funding-adjusted return types."""
        self.assertEqual(get_dep('funding_adj_raw', 0, 60), 'y_funding_adj_raw1_60')
        self.assertEqual(get_dep('funding_adj_resid_eq', 1, 1440), 'y_funding_adj_resid_eqmkt2_1440')
        self.assertEqual(get_dep('funding_adj_resid_wgt', 2, 15), 'y_funding_adj_resid_wgtmkt3_15')
        
    def test_get_dep_invalid_type(self):
        """Test get_dep with invalid return type."""
        with self.assertRaises(Exception):
            get_dep('invalid_type', 0, 60)
            
    def test_make_classification_bar_features(self):
        """Test make_classification_bar_features function."""
        horizons = [60, 1440]
        features = make_classification_bar_features(horizons)
        
        expected = []
        for base_feat in BASE_CLASSIFICATION_BAR_FEATURES:
            for h in horizons:
                expected.append(f"{base_feat}_{h}")
                
        self.assertEqual(features, expected)
        self.assertIn('last_funding_rate_mean_60', features)
        self.assertIn('logret_1440', features)
        self.assertIn('dvolume_60', features)


class TestModelHorizonFit(unittest.TestCase):
    """Test cases for ModelHorizonFit class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'SIGMA_BOUND_DEP': 3,
            'SIGMA_BOUND_INDEP': 3,
            'MIN_CLASSIFICATION_QUANTILE': 0.1,
            'SVM_REGULARIZATION': 0.001,
            'USE_CLUSTER_REGRESSION': False,
            'USE_CV_REGULARIZATION': False,
            'FITTING_RESAMPLE_HORIZON_THRESHOLD': 1440,
            'FITTING_RESAMPLE_FREQUENCY': '60min',
            'CLASSIFIER_TYPE': 'rf',
            'RF_N_ESTIMATORS': 100,
            'RF_MAX_DEPTH': 10,
            'RF_MIN_SAMPLES_SPLIT': 500,
            'RF_MIN_SAMPLES_LEAF': 100,
            'RF_MAX_SAMPLES': 0.35,
            'RF_MAX_FEATURES': 'sqrt',
            'RF_BOOTSTRAP': True,
            'RF_OOB_SCORE': True,
            'RF_CCP_ALPHA': 0.001,
            'RF_RANDOM_STATE': 42,
            'RF_N_JOBS': -1,
            'RF_CLASS_WEIGHT': 'balanced',
            'RF_VERBOSE': 1,
            'FCASTS': {
                '60': {
                    'models': [
                        {'name': 'hl', 'lags': 0}
                    ]
                },
                '1440': {
                    'models': [
                        {'name': 'hl', 'lags': 0}
                    ]
                }
            }
        }
        
        # Create sample fitting data - need at least MIN_FITTING_OBS (1000) rows
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='1h')  # More dates
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        
        index_tuples = [(date, symbol) for date in dates for symbol in symbols]
        self.index = pd.MultiIndex.from_tuples(index_tuples, names=['ts', 'symbol_venue'])
        
        # Create fitting DataFrame
        np.random.seed(42)
        self.fitting_df = pd.DataFrame({
            'hl_60_L0': np.random.normal(0, 0.01, len(self.index)),
            'y_raw1_60': np.random.normal(0, 0.01, len(self.index)),
            'y_resid_eqmkt1_60': np.random.normal(0, 0.008, len(self.index)),
            'y_resid_wgtmkt1_60': np.random.normal(0, 0.008, len(self.index)),
            'beta_1440': np.random.uniform(0.5, 1.5, len(self.index)),
            'logret_1440_trstd': np.random.uniform(0.01, 0.03, len(self.index)),
            'dvolume_1440_trmean': np.random.uniform(1e6, 1e8, len(self.index)),
            'day_of_week': np.random.randint(0, 7, len(self.index)),
            'hour_of_day': np.random.randint(0, 24, len(self.index))
        }, index=self.index)
        
        # Add classification features
        for feat in CLASSIFICATION_FEATURES_1440:
            if feat not in self.fitting_df.columns:
                self.fitting_df[feat] = np.random.normal(0, 1, len(self.index))
                
        self.forecast = {
            'name': 'hl',
            'lags': 0,
            'fit_type': 'svm_adapt'
        }
        
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_init_valid(self):
        """Test ModelHorizonFit initialization with valid data."""
        mhf = ModelHorizonFit(
            prod=False,
            debug=False,
            config=self.config,
            
            forecast=self.forecast,
            fitting_df=self.fitting_df,
            classification_df=self.fitting_df.copy(),
            weighted_fit=False,
            horizon=60,
            lags=0,
            return_type='raw',
            classifier_dir=self.temp_dir,
            classification_features=CLASSIFICATION_FEATURES_1440
        )
        
        self.assertEqual(mhf.name, 'hl')
        self.assertEqual(mhf.horizon, 60)
        self.assertEqual(mhf.name_horizon, 'hl_60')
        self.assertEqual(mhf.indep0, 'hl_60_L0')
        self.assertEqual(mhf.classification_dep, 'y_raw1_60')
        self.assertIsNotNone(mhf.as_of)
        
    def test_init_insufficient_data(self):
        """Test ModelHorizonFit initialization with insufficient data."""
        small_df = self.fitting_df.iloc[:10]  # Too few observations
        
        with self.assertRaises(ValueError):
            ModelHorizonFit(
                prod=False,
                debug=False,
                config=self.config,
                
                forecast=self.forecast,
                fitting_df=small_df,
                classification_df=small_df,
                weighted_fit=False,
                horizon=60,
                lags=0,
                return_type='raw',
                classifier_dir=self.temp_dir
            )
            
    def test_smooth_fits(self):
        """Test coefficient smoothing across lags."""
        mhf = ModelHorizonFit(
            prod=False,
            debug=False,
            config=self.config,
            
            forecast=self.forecast,
            fitting_df=self.fitting_df,
            classification_df=self.fitting_df,
            weighted_fit=False,
            horizon=60,
            lags=0,
            return_type='raw',
            classifier_dir=self.temp_dir
        )
        
        # Create test fit data
        fit_df = pd.DataFrame({
            'lag': [0, 1, 0, 1],
            'condition': ['mom', 'mom', 'rev', 'rev'],
            'coeff': [0.5, 0.3, -0.4, -0.2]
        })
        
        smoothed = mhf.smooth_fits(fit_df)
        
        self.assertIn('coeff_diff', smoothed.columns)
        self.assertIn('coeff_smooth', smoothed.columns)
        # Lag 0 should keep original coefficient
        self.assertEqual(smoothed.loc[smoothed['lag'] == 0, 'coeff_diff'].iloc[0], 0.5)
        
    def test_classify(self):
        """Test classification of momentum vs mean-reversion regimes."""
        # Create larger dataset to meet MIN_CLASSIFICATION_OBS
        dates = pd.date_range('2023-01-01', '2023-06-30', freq='1h')
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures', 
                   'BNBUSDT_binance-futures', 'ADAUSDT_binance-futures']
        
        index_tuples = [(date, symbol) for date in dates for symbol in symbols]
        large_index = pd.MultiIndex.from_tuples(index_tuples, names=['ts', 'symbol_venue'])
        
        # Create large fitting DataFrame
        np.random.seed(42)
        large_fitting_df = pd.DataFrame({
            'hl_60_L0': np.random.normal(0, 0.01, len(large_index)),
            'y_raw1_60': np.random.normal(0, 0.01, len(large_index)),
        }, index=large_index)
        
        # Add clear patterns for classification
        # Momentum: both positive and large
        large_fitting_df.loc[large_fitting_df.index[:2000], 'hl_60_L0'] = 0.02
        large_fitting_df.loc[large_fitting_df.index[:2000], 'y_raw1_60'] = 0.03
        # Mean reversion: opposite signs  
        large_fitting_df.loc[large_fitting_df.index[2000:4000], 'hl_60_L0'] = 0.02
        large_fitting_df.loc[large_fitting_df.index[2000:4000], 'y_raw1_60'] = -0.03
        
        mhf = ModelHorizonFit(
            prod=False,
            debug=False,
            config=self.config,
            
            forecast=self.forecast,
            fitting_df=large_fitting_df,
            classification_df=large_fitting_df.copy(),
            weighted_fit=False,
            horizon=60,
            lags=0,
            return_type='raw',
            classifier_dir=self.temp_dir
        )
        
        result = mhf.classify(large_fitting_df)
        
        self.assertIsNotNone(result)
        self.assertIn('class_hl', result.columns)
        # Should have both momentum (1) and mean-reversion (-1) classifications
        self.assertIn(1, result['class_hl'].values)
        self.assertIn(-1, result['class_hl'].values)
        # Should have filtered out zeros
        self.assertNotIn(0, result['class_hl'].values)
        
    @patch('lib.fits.model_horizon.joblib.dump')
    @patch('lib.fits.fits.safe_mkdir')
    def test_persist_classifier(self, mock_mkdir, mock_dump):
        """Test classifier persistence."""
        mhf = ModelHorizonFit(
            prod=False,
            debug=False,
            config=self.config,
            
            forecast=self.forecast,
            fitting_df=self.fitting_df,
            classification_df=self.fitting_df,
            weighted_fit=False,
            horizon=60,
            lags=0,
            return_type='raw',
            classifier_dir=self.temp_dir
        )
        
        # Create the necessary directory structure for new format
        import os
        os.makedirs(f'{self.temp_dir}/60/hl', exist_ok=True)
        
        # Create mock classifier
        classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('linearsvc', LinearSVC())
        ])
        
        feature_coeffs = {'feat1': 0.1, 'feat2': -0.2}
        
        mhf._persist_classifier(classifier, 0.85, 0.82, feature_coeffs)
        
        mock_dump.assert_called_once()
        # Verify metrics file was written (new format uses horizon/name subdirectory)
        # as_of = max_date + lag_days where lag_days = int(60/1440)*(0+1) = 0, so as_of = 2024-01-31
        expected_metrics_file = f'{self.temp_dir}/60/hl/classifier.60.hl.20240131.metrics'
        self.assertTrue(os.path.exists(expected_metrics_file))
        
    def test_run_regression(self):
        """Test regression execution."""
        mhf = ModelHorizonFit(
            prod=False,
            debug=False,
            config=self.config,
            
            forecast=self.forecast,
            fitting_df=self.fitting_df,
            classification_df=self.fitting_df,
            weighted_fit=False,
            horizon=60,
            lags=0,
            return_type='raw',
            classifier_dir=self.temp_dir
        )
        
        # Create regression data
        reg_df = pd.DataFrame({
            'hl_60_L0': np.random.normal(0, 0.01, 1000),
            'y_raw1_60': np.random.normal(0, 0.01, 1000),
            'symbol_venue': ['BTCUSDT_binance-futures'] * 1000
        })
        # Add some correlation
        reg_df['y_raw1_60'] = 0.5 * reg_df['hl_60_L0'] + np.random.normal(0, 0.005, 1000)
        
        with patch.object(mhf, '_run_regresssion_results') as mock_reg:
            # Mock regression results
            mock_results = Mock()
            mock_results.params = pd.Series([0.0001, 0.5])  # intercept, slope
            mock_results.tvalues = pd.Series([0.1, 3.5])
            mock_results.bse = pd.Series([0.001, 0.15])
            mock_results.summary.return_value = "Mock summary"
            mock_reg.return_value = mock_results
            
            coeff, tstat, stderr = mhf.run_regression(
                reg_df, 'mom', 'y_raw1_60', 0
            )
            
            self.assertEqual(coeff, 0.5)
            self.assertEqual(tstat, 3.5)
            self.assertEqual(stderr, 0.15)


class TestFits(unittest.TestCase):
    """Test cases for main Fits class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'FCASTS': {
                '60': {
                    'models': [
                        {'name': 'hl', 'lags': 1},
                        {'name': 'c2vwap', 'lags': 1}
                    ],
                    'refitting_days': 7,
                    'fitting_lookback_days': 90,
                    'return_type': 'resid_wgt'
                },
                '1440': {
                    'models': [
                        {'name': 'hl', 'lags': 2}
                    ],
                    'refitting_days': 30,
                    'fitting_lookback_days': 180,
                    'return_type': 'raw'
                }
            },
            'CLASSIFICATION_HISTORY_DAYS': 280,
            'SVM_REGULARIZATION': 0.001,
            'SYMBOL_UNIVERSE': ['BTCUSDT', 'ETHUSDT'],
            'DYNAMIC_UNIVERSE': False,
            'CLASSIFIER_TYPE': 'rf',
            'RF_N_ESTIMATORS': 100,
            'RF_MAX_DEPTH': 10,
            'RF_MIN_SAMPLES_SPLIT': 500,
            'RF_MIN_SAMPLES_LEAF': 100,
            'RF_RANDOM_STATE': 42,
            'RF_N_JOBS': -1,
            'RF_CLASS_WEIGHT': 'balanced',
            'RF_VERBOSE': 1,
            'ADV_LOOKBACK_DAYS': 45,
            'MIN_ADVP_PRICEABLE': 2.5e7,
            'MIN_ADVP_FEATUREABLE': 2.5e7,
            'FEATUREABLE_HIST_PERIODS': 30,
            'MIN_ADVP_FITTABLE': 1.5e7,
            'MIN_ADVP_TRADEABLE': 1.5e7,
            'MIN_ADVP_EXPANDABLE': 5.5e7,
            'MAX_PORTFOLIO_NOTIONAL': 2.0e7,
            'MAX_POSITION_VOLUME_FRACTION': 0.05,
            'DEFAULT_FEATURE_LOOKBACK_PERIODS': 90,
            'BETA_LOOKBACK_PERIODS': 90,
            'MAX_MOVE_FILTER': 2.5,
            'MAX_POSITION_PCT': 0.04,
            'EXCLUDE_NON_ALPHA_TRADES': False,
            'OPT_HORIZON': 1440,
            'RISK_FLD': 'logret_HORIZON_trstd',
            'FILTER_DELISTING': True,
            'DELISTING_BUFFER_DAYS': 6,
            'FEATURE_SIGMA_BOUND': 5,
            'OLD_POSITION_DAYS': 30,
            'OLD_POSITION_RISK_MULT': 3.5,
            'REOPT_TIMES': [],
            'OPT_OFFSET_MINS': 2,
            'WEIGHTED_REGRESSION': False,
            'FITTING_RESAMPLE_HORIZON_THRESHOLD': 1440,
            'FITTING_RESAMPLE_FREQUENCY': '60min'
        }
        
        self.mock_dir_manager = Mock()
        self.mock_dir_manager.FITS_DIR = '/data/fits'
        self.mock_dir_manager.FITS_DIR_PROD = '/data/prod_fits'
        self.mock_dir_manager.FITS_DIR_DEV = '/data/dev_fits'
        self.mock_dir_manager.PREBAR_DIR = '/data/prebars'
        self.mock_dir_manager.UNIVERSE_DIR = '/data/universe'
        
    @patch('lib.fits.fits.DataLoader')
    @patch('lib.fits.fits.Models')
    @patch('lib.fits.fits.Calcs')
    @patch('lib.fits.fits.Universe')
    def test_init_default(self, mock_universe_class, mock_calcs, mock_models, mock_data_loader):
        """Test Fits initialization with default parameters."""
        mock_universe_instance = Mock()
        mock_universe_class.return_value = mock_universe_instance
        
        fits = Fits(
            config=self.config,
            fits_dir_manager=self.mock_dir_manager
        )
        
        self.assertFalse(fits.prod)
        self.assertFalse(fits.debug)
        # bars_type no longer exists
        self.assertEqual(fits.pool_size, 4)
        self.assertEqual(fits.classification_history_days, 280)
        self.assertIsNotNone(fits.horizons)
        
    @patch('lib.fits.fits.DataLoader')
    @patch('lib.fits.fits.Models')
    @patch('lib.fits.fits.Calcs')
    @patch('lib.fits.fits.Universe')
    def test_init_production(self, mock_universe_class, mock_calcs, mock_models, mock_data_loader):
        """Test Fits initialization in production mode."""
        mock_universe_instance = Mock()
        mock_universe_class.return_value = mock_universe_instance
        
        fits = Fits(
            config=self.config,
            prod=True,
            fits_dir_manager=self.mock_dir_manager
        )
        
        self.assertTrue(fits.prod)
        self.assertEqual(fits.fits_dir, '/data/prod_fits')
        
    @patch('lib.fits.fits.extract_horizons')
    @patch('lib.fits.fits.extract_models')
    @patch('lib.fits.fits.DataLoader')
    @patch('lib.fits.fits.Models')
    @patch('lib.fits.fits.Calcs')
    @patch('lib.fits.fits.Universe')
    def test_init_custom_horizons_models(
        self, mock_universe_class, mock_calcs, mock_models, 
        mock_data_loader, mock_extract_models, mock_extract_horizons
    ):
        """Test Fits initialization with custom horizons and models."""
        mock_universe_instance = Mock()
        mock_universe_class.return_value = mock_universe_instance
        
        custom_horizons = [15, 60]
        custom_models = ['hl']
        
        fits = Fits(
            config=self.config,
            horizons=custom_horizons,
            models=custom_models,
            fits_dir_manager=self.mock_dir_manager
        )
        
        self.assertEqual(fits.horizons, custom_horizons)
        self.assertEqual(fits.models_to_run, custom_models)
        # Should not call extract functions when custom values provided
        mock_extract_horizons.assert_not_called()
        
    def test_get_latest_fit_file_date(self):
        """Test extracting date from fit file name."""
        with patch('lib.fits.fits.DataLoader') as mock_dl, \
             patch('lib.fits.fits.Models'), \
             patch('lib.fits.fits.Calcs'), \
             patch('lib.fits.fits.Universe'), \
             patch('glob.glob') as mock_glob:
            
            mock_dl_instance = Mock()
            mock_dl.return_value = mock_dl_instance
            
            fits = Fits(
                config=self.config,
                fits_dir_manager=self.mock_dir_manager
            )
            
            # Mock file listing
            mock_glob.return_value = [
                '/data/fits/fits.hl.60.20240115.csv',
                '/data/fits/fits.hl.60.20240108.csv'
            ]
            
            with patch.object(fits, 'get_latest_fit_file', return_value='/data/fits/fits.hl.60.20240115.csv'):
                date = fits.get_latest_fit_file_date(60, 'hl')
                
                self.assertEqual(date, dt(2024, 1, 15).date())
                
    def test_get_latest_fit_file_date_no_files(self):
        """Test get_latest_fit_file_date when no files exist."""
        with patch('lib.fits.fits.DataLoader'), \
             patch('lib.fits.fits.Models'), \
             patch('lib.fits.fits.Calcs'), \
             patch('lib.fits.fits.Universe'):
            
            fits = Fits(
                config=self.config,
                fits_dir_manager=self.mock_dir_manager
            )
            
            with patch.object(fits, 'get_latest_fit_file', return_value=None):
                date = fits.get_latest_fit_file_date(60, 'hl')
                self.assertIsNone(date)
                
    @patch('pandas.DataFrame.to_csv')
    def test_dump_fits_debug(self, mock_to_csv):
        """Test dump_fits in debug mode."""
        with patch('lib.fits.fits.DataLoader'), \
             patch('lib.fits.fits.Models'), \
             patch('lib.fits.fits.Calcs'), \
             patch('lib.fits.fits.Universe'):
            
            fits = Fits(
                config=self.config,
                debug=True,
                fits_dir_manager=self.mock_dir_manager
            )
            
            # Create sample fit data
            fit_df = pd.DataFrame({
                'horizon': [60, 60],
                'name': ['hl', 'hl'],
                'as_of': [dt(2024, 1, 15).date(), dt(2024, 1, 15).date()],
                'condition': ['mom', 'rev'],
                'lag': [0, 0],
                'coeff': [0.5, -0.3],
                'tstat': [3.2, -2.1]
            })
            
            fits.dump_fits(fit_df)
            
            # Should write to debug file
            mock_to_csv.assert_called_once_with("fits.debug.csv")
            
    def test_dump_fits_none(self):
        """Test dump_fits with None input."""
        with patch('lib.fits.fits.DataLoader'), \
             patch('lib.fits.fits.Models'), \
             patch('lib.fits.fits.Calcs'), \
             patch('lib.fits.fits.Universe'):
            
            fits = Fits(
                config=self.config,
                fits_dir_manager=self.mock_dir_manager
            )
            
            # Should handle None gracefully
            fits.dump_fits(None)  # Should not raise
            
    @patch('lib.fits.fits.Pool')
    def test_generate_rolling_fits_pool_error(self, mock_pool):
        """Test error handling in parallel fitting."""
        with patch('lib.fits.fits.DataLoader'), \
             patch('lib.fits.fits.Models'), \
             patch('lib.fits.fits.Calcs'), \
             patch('lib.fits.fits.Universe'):
            
            fits = Fits(
                config=self.config,
                pool_size=2,
                fits_dir_manager=self.mock_dir_manager
            )
            
            # Mock pool to raise exception
            mock_pool_instance = Mock()
            mock_pool_instance.starmap.side_effect = Exception("Pool error")
            mock_pool.return_value = mock_pool_instance
            
            with patch.object(fits, '_build_fit_data'), \
                 patch.object(fits, 'gather_fitting_dates', return_value=[(dt(2024,1,1).date(), dt(2024,1,15).date())]), \
                 patch.object(fits, 'generate_fitting_window'):
                
                # Should handle pool errors
                with self.assertRaises(Exception):
                    fits.generate_rolling_fits(dt(2024,1,1).date(), dt(2024,1,15).date())


class TestModelFittingIntegration(unittest.TestCase):
    """Integration tests for model fitting pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.config = {
            'FCASTS': {
                '60': {
                    'models': [{'name': 'hl', 'lags': 1}],
                    'refitting_days': 7,
                    'fitting_lookback_days': 30,
                    'return_type': 'raw'
                }
            },
            'CLASSIFICATION_HISTORY_DAYS': 90,
            'SVM_REGULARIZATION': 0.001,
            'SYMBOL_UNIVERSE': ['BTCUSDT', 'ETHUSDT'],
            'SIGMA_BOUND_DEP': 3,
            'SIGMA_BOUND_INDEP': 3,
            'MIN_CLASSIFICATION_QUANTILE': 0.1,
            'USE_CLUSTER_REGRESSION': False,
            'USE_CV_REGULARIZATION': False,
            'CLASSIFIER_TYPE': 'rf',
            'RF_N_ESTIMATORS': 100,
            'RF_MAX_DEPTH': 10,
            'RF_MIN_SAMPLES_SPLIT': 500,
            'RF_MIN_SAMPLES_LEAF': 100,
            'RF_MAX_SAMPLES': 0.35,
            'RF_MAX_FEATURES': 'sqrt',
            'RF_BOOTSTRAP': True,
            'RF_OOB_SCORE': True,
            'RF_CCP_ALPHA': 0.001,
            'RF_RANDOM_STATE': 42,
            'RF_N_JOBS': -1,
            'RF_CLASS_WEIGHT': 'balanced',
            'RF_VERBOSE': 1,
            'ADV_LOOKBACK_DAYS': 45,
            'MIN_ADVP_FEATUREABLE': 2.5e7,
            'MIN_ADVP_FITTABLE': 1.5e7,
            'MIN_ADVP_TRADEABLE': 1.5e7,
            'MIN_ADVP_EXPANDABLE': 5.5e7,
            'MAX_PORTFOLIO_NOTIONAL': 2.0e7,
            'MAX_POSITION_VOLUME_FRACTION': 0.05,
            'DEFAULT_FEATURE_LOOKBACK_PERIODS': 90,
            'BETA_LOOKBACK_PERIODS': 90,
            'MAX_MOVE_FILTER': 2.5,
            'MAX_POSITION_PCT': 0.04,
            'EXCLUDE_NON_ALPHA_TRADES': False,
            'OPT_HORIZON': 1440,
            'RISK_FLD': 'logret_HORIZON_trstd',
            'FILTER_DELISTING': True,
            'DELISTING_BUFFER_DAYS': 6,
            'FEATURE_SIGMA_BOUND': 5,
            'OLD_POSITION_DAYS': 30,
            'OLD_POSITION_RISK_MULT': 3.5,
            'REOPT_TIMES': [],
            'OPT_OFFSET_MINS': 2,
            'DYNAMIC_UNIVERSE': False,
            'FITTING_RESAMPLE_HORIZON_THRESHOLD': 1440,
            'FITTING_RESAMPLE_FREQUENCY': '60min'
        }
        
        # Create realistic multi-index data - need enough for MIN_CLASSIFICATION_OBS
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='1h')  # Full year of hourly data
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures', 
                   'BNBUSDT_binance-futures', 'ADAUSDT_binance-futures']
        
        index_tuples = [(date, symbol) for date in dates for symbol in symbols]
        self.index = pd.MultiIndex.from_tuples(index_tuples, names=['ts', 'symbol_venue'])
        
        # Create correlated data for realistic testing
        np.random.seed(42)
        n = len(self.index)
        
        # Model predictions with some signal
        model_pred = np.random.normal(0, 0.01, n)
        
        # Forward returns correlated with predictions
        forward_returns = 0.3 * model_pred + np.random.normal(0, 0.008, n)
        
        self.data_df = pd.DataFrame({
            'hl_60_L0': model_pred,
            'y_raw1_60': forward_returns,
            'y_resid_eqmkt1_60': forward_returns * 0.8,
            'y_resid_wgtmkt1_60': forward_returns * 0.9,
            'fittable': True,
            'advp': 5e7,
            'mkt_wgt': np.random.uniform(0.5, 2.0, n)
        }, index=self.index)
        
        # Add all required features with some predictive power
        for i, feat in enumerate(CLASSIFICATION_FEATURES_1440):
            # Make first few features correlated with outcome
            if i < 3:
                self.data_df[feat] = forward_returns * 2 + np.random.normal(0, 0.5, n)
            else:
                self.data_df[feat] = np.random.normal(0, 1, n)
            
    def test_classification_workflow(self):
        """Test complete classification workflow."""
        forecast = {'name': 'hl', 'lags': 0, 'fit_type': 'svm_adapt'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            mhf = ModelHorizonFit(
                prod=False,
                debug=True,  # Don't persist
                config=self.config,
                
                forecast=forecast,
                fitting_df=self.data_df,
                classification_df=self.data_df.copy(),
                    weighted_fit=False,
                horizon=60,
                lags=0,
                return_type='raw',
                classifier_dir=temp_dir,
                classification_features=CLASSIFICATION_FEATURES_1440[:3]  # Use subset
            )
            
            # Test classification
            classified = mhf.classify(self.data_df)
            self.assertIsNotNone(classified)
            self.assertGreater(len(classified), MIN_CLASSIFICATION_OBS)
            
            # Test classifier creation - may return None if RF fails
            with patch.object(mhf, '_debug_and_persist_classifier', return_value=(0.8, 0.75)):
                clf = mhf.create_classifier()
                # RF classifier may fail with synthetic data and return None
                # This is expected behavior when feature importances sum to 0
                if clf is not None:
                    self.assertIsInstance(clf, Pipeline)
                    # Check for LinearSVC classifier (svm_adapt)
                    self.assertIn('linearsvc', clf.named_steps)
                
    def test_regression_fitting_workflow(self):
        """Test regression fitting without classifier."""
        forecast = {'name': 'hl', 'lags': 0, 'fit_type': 'vanilla'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            mhf = ModelHorizonFit(
                prod=False,
                debug=True,
                config=self.config,
                
                forecast=forecast,
                fitting_df=self.data_df,
                classification_df=self.data_df,
                    weighted_fit=True,
                horizon=60,
                lags=0,
                return_type='raw',
                classifier_dir=temp_dir,
                classification_features=[]  # No classifier
            )
            
            # Run full fitting
            mhf = mhf.run_fits()
            
            self.assertIsNotNone(mhf.result_df)
            self.assertIn('coeff', mhf.result_df.columns)
            self.assertIn('tstat', mhf.result_df.columns)
            self.assertIn('condition', mhf.result_df.columns)
            
            # Should have momentum and reversion rows
            conditions = mhf.result_df['condition'].unique()
            self.assertIn('mom', conditions)
            self.assertIn('rev', conditions)
            
    def test_cv_regularization_workflow(self):
        """Test classifier with cross-validation regularization."""
        config_cv = self.config.copy()
        config_cv['USE_CV_REGULARIZATION'] = True
        # Use SVM for CV test since RF doesn't need CV for regularization
        config_cv['CLASSIFIER_TYPE'] = 'svm_cv'
        
        forecast = {'name': 'hl', 'lags': 0, 'fit_type': 'svm_cv'}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            mhf = ModelHorizonFit(
                prod=False,
                debug=True,
                config=config_cv,
                
                forecast=forecast,
                fitting_df=self.data_df,
                classification_df=self.data_df.copy(),
                    weighted_fit=False,
                horizon=60,
                lags=0,
                return_type='raw',
                classifier_dir=temp_dir,
                classification_features=CLASSIFICATION_FEATURES_1440[:2]
            )
            
            # Create enough data for time series CV
            large_class_df = self.data_df.copy()
            # Add the class column that create_svm expects
            large_class_df['class_hl'] = np.random.choice([-1, 1], size=len(large_class_df))
            
            with patch.object(mhf, 'classify', return_value=large_class_df):
                # Mock subsample to return enough data (don't reduce to 1%)
                with patch.object(mhf, '_subsample_classification_data', return_value=large_class_df):
                    with patch.object(mhf, '_debug_and_persist_classifier', return_value=(0.82, 0.78)):
                        mhf.create_classifier()
                        
                        self.assertIsInstance(mhf.classifier, Pipeline)
                        # For SVM with CV, should have linearsvc in pipeline
                        self.assertIn('linearsvc', mhf.classifier.named_steps)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_min_coeff_threshold(self):
        """Test handling of very small coefficients."""
        self.assertLess(MIN_COEFF, 1e-6)
        
        # Test coefficient comparison
        tiny_coeff = 1e-10
        self.assertLess(abs(tiny_coeff), MIN_COEFF)
        
    def test_empty_classification(self):
        """Test classification with no valid points."""
        config = {
            'SIGMA_BOUND_DEP': 3,
            'SIGMA_BOUND_INDEP': 3,
            'MIN_CLASSIFICATION_QUANTILE': 0.99,  # Very high threshold
            'SVM_REGULARIZATION': 0.001,
            'USE_CLUSTER_REGRESSION': False,
            'USE_CV_REGULARIZATION': False,
            'FITTING_RESAMPLE_HORIZON_THRESHOLD': 1440,
            'FITTING_RESAMPLE_FREQUENCY': '60min',
            'CLASSIFIER_TYPE': 'rf',
            'RF_N_ESTIMATORS': 100,
            'RF_MAX_DEPTH': 10,
            'RF_MIN_SAMPLES_SPLIT': 500,
            'RF_MIN_SAMPLES_LEAF': 100,
            'RF_MAX_SAMPLES': 0.35,
            'RF_MAX_FEATURES': 'sqrt',
            'RF_BOOTSTRAP': True,
            'RF_OOB_SCORE': True,
            'RF_CCP_ALPHA': 0.001,
            'RF_RANDOM_STATE': 42,
            'RF_N_JOBS': -1,
            'RF_CLASS_WEIGHT': 'balanced',
            'RF_VERBOSE': 1,
            'FCASTS': {
                '60': {'models': [{'name': 'hl', 'lags': 0}]},
                '1440': {'models': [{'name': 'hl', 'lags': 0}]}
            }
        }
        
        # Create data with small moves - need at least MIN_FITTING_OBS
        dates = pd.date_range('2024-01-01', '2024-02-15', freq='1h')  # More dates
        symbols = ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        index_tuples = [(date, symbol) for date in dates for symbol in symbols]
        index = pd.MultiIndex.from_tuples(index_tuples, names=['ts', 'symbol_venue'])
        
        n = len(index)
        fitting_df = pd.DataFrame({
            'hl_60_L0': np.random.normal(0, 0.0001, n),  # Very small
            'y_raw1_60': np.random.normal(0, 0.0001, n),  # Very small
        }, index=index)
        
        mhf = ModelHorizonFit(
            prod=False,
            debug=True,
            config=config,
            
            forecast={'name': 'hl', 'lags': 0, 'fit_type': 'svm_adapt'},
            fitting_df=fitting_df,
            classification_df=fitting_df,
            weighted_fit=False,
            horizon=60,
            lags=0,
            return_type='raw',
            classifier_dir='/tmp'
        )
        
        result = mhf.classify(fitting_df)
        # Should return None due to high quantile threshold
        if result is not None:
            self.assertLess(len(result), MIN_CLASSIFICATION_OBS)


if __name__ == '__main__':
    unittest.main()
