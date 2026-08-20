"""Unit tests for the tardis module.

This module tests the Tardis data downloader and processor which handles
downloading historical market data from Tardis.dev API and processing it
into standardized 1-minute OHLCV bars.
"""
# pylint: disable=protected-access
# Testing private methods is necessary for comprehensive unit testing
import os
from datetime import datetime as dt
from contextlib import ExitStack
from unittest.mock import Mock, patch
import logging

import pandas as pd
import pytest

from lib.bars.tardis import Tardis, TARDIS_BAR_START_DATE, DATA_TYPES


class TestTardis:
    """Test cases for the Tardis class."""
    
    @staticmethod
    def get_mocked_tardis_context():
        """Get context manager with mocked dependencies for Tardis initialization."""
        stack = ExitStack()
        
        # Mock DataLoader
        mock_dl_class = stack.enter_context(patch('lib.bars.tardis.DataLoader'))
        mock_data_loader = Mock()
        mock_dl_class.return_value = mock_data_loader
        
        # Mock Calcs
        stack.enter_context(patch('lib.bars.tardis.Calcs'))
        
        # Mock Universe
        mock_universe_class = stack.enter_context(patch('lib.bars.tardis.Universe'))
        mock_universe = Mock()
        mock_universe.load_universe_symbols.return_value = ['BTCUSDT', 'ETHUSDT']
        mock_universe_class.return_value = mock_universe
        
        return stack
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            'SYMBOL_UNIVERSE': ['BTC', 'ETH'],
            'ADV_LOOKBACK_DAYS': 45,
            'DYNAMIC_UNIVERSE': False,
            'FCASTS': {
                '1440': {
                    'models': [{'name': 'test_model', 'lags': 1}]
                }
            },
            'MIN_ADVP_PRICEABLE': 2.5e7,
            'MIN_ADVP_FEATUREABLE': 2.5e7,
            'FEATUREABLE_HIST_PERIODS': 30,
            'MIN_ADVP_FITTABLE': 1.5e7,
            'MIN_ADVP_TRADEABLE': 1.5e7,
            'MIN_ADVP_EXPANDABLE': 5.5e7,
            'MAX_MARKET_CAP_EXPANDABLE_FRAC': 1.0,
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
            'FEATURE_SIGMA_BOUND': 5,
            'MAX_DVOL_SIGMA': 2.0,
            'MIN_FUNDING_RATE': -0.005,
            'OLD_POSITION_DAYS': 30,
            'OLD_POSITION_RISK_MULT': 3.5,
            'MAX_MOVE_FILTER': 2.5,
            'BETA_LOOKBACK_PERIODS': 90,
            'DEFAULT_FEATURE_LOOKBACK_PERIODS': 90,
            'REOPT_TIMES': [],
            'OPT_OFFSET_MINS': 2
        }
    
    @pytest.fixture
    def mock_api_key(self):
        """Mock API key for testing."""
        return "test_api_key_123"
    
    @pytest.fixture
    def sample_book_df(self):
        """Create sample order book snapshot data."""
        # Create timestamps for every minute of 2024-01-01 (1440 minutes)
        base_ts = 1704067200000000  # 2024-01-01 00:00:00 in microseconds
        timestamps = [base_ts + i * 60 * 1000000 for i in range(1440)]  # One per minute
        
        data = {
            'timestamp': timestamps,
            'local_timestamp': [ts + 100000 for ts in timestamps],  # Add 100ms latency
            'symbol': ['BTCUSDT'] * 1440,
            'exchange': ['binance-futures'] * 1440,
            'bids[0].price': [42000.0 + (i % 10) for i in range(1440)],
            'bids[0].amount': [10.0 + (i % 5) for i in range(1440)],
            'bids[1].price': [41999.0 + (i % 10) for i in range(1440)],
            'bids[1].amount': [5.0 + (i % 3) for i in range(1440)],
            'bids[2].price': [41998.0 + (i % 10) for i in range(1440)],
            'bids[2].amount': [3.0 + (i % 2) for i in range(1440)],
            'bids[3].price': [41997.0 + (i % 10) for i in range(1440)],
            'bids[3].amount': [2.0 + (i % 2) for i in range(1440)],
            'bids[4].price': [41996.0 + (i % 10) for i in range(1440)],
            'bids[4].amount': [1.0 + (i % 2) for i in range(1440)],
            'asks[0].price': [42001.0 + (i % 10) for i in range(1440)],
            'asks[0].amount': [10.0 + (i % 5) for i in range(1440)],
            'asks[1].price': [42002.0 + (i % 10) for i in range(1440)],
            'asks[1].amount': [5.0 + (i % 3) for i in range(1440)],
            'asks[2].price': [42003.0 + (i % 10) for i in range(1440)],
            'asks[2].amount': [3.0 + (i % 2) for i in range(1440)],
            'asks[3].price': [42004.0 + (i % 10) for i in range(1440)],
            'asks[3].amount': [2.0 + (i % 2) for i in range(1440)],
            'asks[4].price': [42005.0 + (i % 10) for i in range(1440)],
            'asks[4].amount': [1.0 + (i % 2) for i in range(1440)],
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_trades_df(self):
        """Create sample trades data."""
        # Create timestamps for every minute of 2024-01-01 (1440 minutes)
        base_ts = 1704067200000000  # 2024-01-01 00:00:00 in microseconds
        # Add 30 seconds offset so trades occur mid-minute
        timestamps = [base_ts + i * 60 * 1000000 + 30 * 1000000 for i in range(1440)]
        
        data = {
            'timestamp': timestamps,
            'local_timestamp': [ts + 100000 for ts in timestamps],  # Add 100ms latency
            'symbol': ['BTCUSDT'] * 1440,
            'exchange': ['binance-futures'] * 1440,
            'price': [42000.0 + (i % 100) for i in range(1440)],
            'amount': [0.5 + (i % 10) * 0.1 for i in range(1440)],
            'side': ['buy' if i % 2 == 0 else 'sell' for i in range(1440)],
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_funding_df(self):
        """Create sample derivative ticker (funding) data."""
        data = {
            'timestamp': [1704067200000000, 1704067260000000, 1704067320000000],  # microseconds
            'local_timestamp': [1704067200100000, 1704067260100000, 1704067320100000],
            'symbol': ['BTCUSDT', 'BTCUSDT', 'BTCUSDT'],
            'exchange': ['binance-futures', 'binance-futures', 'binance-futures'],
            'funding_timestamp': [1704096000000000, 1704096000000000, 1704096000000000],  # 8 hours later
            'funding_rate': [0.0001, 0.0001, 0.00015],
            'open_interest': [1000000.0, 1010000.0, 1005000.0],
            'last_price': [42000.0, 42100.0, 42050.0],
            'index_price': [41990.0, 42090.0, 42040.0],
            'mark_price': [41995.0, 42095.0, 42045.0],
        }
        return pd.DataFrame(data)
    
    def test_init_default(self, mock_config, mock_api_key):
        """Test Tardis initialization with default parameters."""
        with patch('lib.bars.tardis.DataLoader') as mock_dl_class, \
             patch('lib.bars.tardis.Calcs'), \
             patch('lib.bars.tardis.get_bucket') as mock_get_bucket, \
             patch('lib.bars.tardis.Universe') as mock_universe_class:
            
            mock_data_loader = Mock()
            mock_data_loader.load_universe_symbols.return_value = ['BTCUSDT', 'ETHUSDT']
            mock_dl_class.return_value = mock_data_loader
            
            # Mock Universe to avoid metadata loading
            mock_universe = Mock()
            mock_universe.load_universe_symbols.return_value = ['BTCUSDT', 'ETHUSDT']
            mock_universe_class.return_value = mock_universe
            
            tardis = Tardis(
                api_key=mock_api_key,
                config=mock_config,
                s3=True,
                symbols=['BTCUSDT', 'ETHUSDT']
            )
            
            assert tardis.api_key == mock_api_key
            assert tardis.config == mock_config
            assert tardis.s3 is True
            assert tardis.pool_size == 1
            assert tardis.overwrite is False
            assert tardis.debug is False
            assert tardis.backfill is False
            assert tardis.force is False
            assert tardis.data_types == DATA_TYPES
            assert tardis.universe == mock_universe
            assert tardis.symbols == ['BTCUSDT', 'ETHUSDT']
            assert not tardis.availability_dict
            assert not tardis.metadata_dict
            mock_get_bucket.assert_called_once()
            # Should not call load_universe_symbols since symbols are provided explicitly
            mock_universe.load_universe_symbols.assert_not_called()
    
    def test_init_custom_params(self, mock_config, mock_api_key):
        """Test Tardis initialization with custom parameters."""
        with patch('lib.bars.tardis.DataLoader') as mock_dl_class, \
             patch('lib.bars.tardis.Calcs'):
            
            mock_dl_class.return_value = Mock()
            custom_perps = ['SOLUSDT', 'ADAUSDT']
            custom_data_types = ['trades', 'book_snapshot_5']
            custom_funding_dir = '/custom/funding'
            
            tardis = Tardis(
                api_key=mock_api_key,
                config=mock_config,
                s3=False,
                pool_size=4,
                symbols=custom_perps,
                overwrite=True,
                debug=True,
                backfill=True,
                force=True,
                data_types=custom_data_types,
                funding_output_dir=custom_funding_dir
            )
            
            assert tardis.symbols == custom_perps
            assert tardis.pool_size == 4
            assert tardis.overwrite is True
            assert tardis.debug is True
            assert tardis.backfill is True
            assert tardis.force is True
            assert tardis.data_types == custom_data_types
            assert tardis.funding_output_dir == custom_funding_dir
            assert tardis.bucket is None  # s3=False
    
    def test_tardis_data_file_name(self):
        """Test static method for generating Tardis filenames."""
        filename = Tardis._tardis_data_file_name(
            'binance-futures',
            'trades',
            dt(2024, 1, 1).date(),
            'BTCUSDT'
        )
        assert filename == "trades.BTCUSDT.binance-futures.20240101.csv.gz"
        
        # Test with different file format
        filename2 = Tardis._tardis_data_file_name(
            'deribit',
            'book_snapshot_5',
            dt(2024, 2, 15).date(),
            'BTC-PERPETUAL',
            'parquet'
        )
        assert filename2 == "book_snapshot_5.BTC-PERPETUAL.deribit.20240215.parquet.gz"
    
    # NOTE: _upload_to_s3 method doesn't exist in Tardis class anymore
    # These tests are commented out as the functionality may have been moved elsewhere
    # def test_upload_to_s3_success(self):
    #     """Test successful S3 upload."""
    #     with patch('lib.bars.tardis.upload_file') as mock_upload, \
    #          patch('lib.bars.tardis.BUCKET', 'test-bucket'):
    #         mock_upload.return_value = True
    #         
    #         # Should not raise
    #         Tardis._upload_to_s3('BTCUSDT', dt(2024, 1, 1), '/tmp/test.parquet')
    #         
    #         mock_upload.assert_called_once_with(
    #             '/tmp/test.parquet',
    #             '1_min/20240101/binance-futures/BTCUSDT.parquet',
    #             bucket_name='test-bucket'
    #         )
    # 
    # def test_upload_to_s3_failure(self):
    #     """Test failed S3 upload."""
    #     with patch('lib.bars.tardis.upload_file') as mock_upload:
    #         mock_upload.return_value = False
    #         
    #         with pytest.raises(RuntimeError, match="could not upload 1-min bars to s3"):
    #             Tardis._upload_to_s3('BTCUSDT', dt(2024, 1, 1), '/tmp/test.parquet')
    
    def test_download_symbol_from_s3(self, mock_config, mock_api_key, tmp_path):
        """Test downloading symbol data from S3."""
        with patch('lib.bars.tardis.DataLoader') as mock_dl_class, \
             patch('lib.bars.tardis.Calcs'), \
             patch('lib.bars.tardis.get_bucket') as mock_get_bucket, \
             patch('lib.bars.tardis.safe_mkdir'), \
             patch('lib.bars.tardis.DATA_DIR', str(tmp_path)), \
             patch('lib.bars.tardis.Universe') as mock_universe_class:
            
            mock_bucket = Mock()
            # Make download succeed
            mock_bucket.download_file.return_value = None
            mock_get_bucket.return_value = mock_bucket
            
            # Mock Universe to avoid metadata loading
            mock_universe = Mock()
            mock_universe.load_universe_symbols.return_value = ['BTCUSDT', 'ETHUSDT']
            mock_universe_class.return_value = mock_universe
            
            # Mock DataLoader
            mock_data_loader = Mock()
            mock_dl_class.return_value = mock_data_loader
            
            tardis = Tardis(mock_api_key, mock_config, s3=True)
            
            # Create the expected file after S3 download
            expected_file = tmp_path / 'tardis' / '20240101' / 'trades.BTCUSDT.binance-futures.20240101.csv.gz'
            expected_file.parent.mkdir(parents=True, exist_ok=True)
            expected_file.touch()
            
            result = tardis._download_symbol(
                'BTCUSDT',
                'trades',
                'binance-futures',
                dt(2024, 1, 1).date(),
                'tardis/20240101'
            )
            
            # Should try to download from S3
            mock_bucket.download_file.assert_called_once()
            assert result is True
    
    def test_download_symbol_local_exists(self, mock_config, mock_api_key, tmp_path, caplog):  # pylint: disable=unused-argument
        """Test behavior when file exists locally."""
        with patch('lib.bars.tardis.DataLoader') as mock_dl_class, \
             patch('lib.bars.tardis.Calcs'), \
             patch('os.path.isfile', return_value=True), \
             patch('lib.bars.tardis.safe_mkdir'), \
             patch('lib.bars.tardis.Universe') as mock_universe_class:
            
            # Mock Universe to avoid metadata loading
            mock_universe = Mock()
            mock_universe.load_universe_symbols.return_value = ['BTCUSDT', 'ETHUSDT']
            mock_universe_class.return_value = mock_universe
            
            # Mock DataLoader
            mock_data_loader = Mock()
            mock_dl_class.return_value = mock_data_loader
            
            tardis = Tardis(mock_api_key, mock_config, s3=False)
            
            with caplog.at_level(logging.INFO):
                result = tardis._download_symbol(
                    'BTCUSDT',
                    'trades',
                    'binance-futures',
                    dt(2024, 1, 1).date(),
                    'tardis/20240101'
                )
            
            assert result is True
            assert "Already downloaded" in caplog.text
    
    def test_download_symbol_tardis_api(self, mock_config, mock_api_key):
        """Test downloading from Tardis API."""
        with patch('lib.bars.tardis.DataLoader') as mock_dl_class, \
             patch('lib.bars.tardis.Calcs'), \
             patch('os.path.isfile', return_value=False), \
             patch('lib.bars.tardis.safe_mkdir'), \
             patch('lib.bars.tardis.datasets.download') as mock_download, \
             patch('lib.bars.tardis.DATA_DIR', '/test_data'), \
             patch('lib.bars.tardis.Universe') as mock_universe_class:
            
            # Mock Universe to avoid metadata loading
            mock_universe = Mock()
            mock_universe.load_universe_symbols.return_value = ['BTCUSDT', 'ETHUSDT']
            mock_universe_class.return_value = mock_universe
            
            # Mock DataLoader
            mock_data_loader = Mock()
            mock_dl_class.return_value = mock_data_loader
            
            tardis = Tardis(mock_api_key, mock_config, s3=False)
            
            result = tardis._download_symbol(
                'BTCUSDT',
                'trades',
                'binance-futures',
                dt(2024, 1, 1).date(),
                'tardis/20240101'
            )
            
            assert result is True
            mock_download.assert_called_once_with(
                exchange='binance-futures',
                data_types=['trades'],
                from_date='20240101',
                to_date='20240101',
                symbols=['BTCUSDT'],
                api_key=mock_api_key,
                download_dir='/test_data/tardis/20240101',
                format='csv',
                get_filename=Tardis._tardis_data_file_name
            )
    
    def test_download_symbol_api_error(self, mock_config, mock_api_key, caplog):
        """Test handling of Tardis API errors."""
        with patch('lib.bars.tardis.DataLoader') as mock_dl_class, \
             patch('lib.bars.tardis.Calcs'), \
             patch('os.path.isfile', return_value=False), \
             patch('lib.bars.tardis.safe_mkdir'), \
             patch('lib.bars.tardis.datasets.download') as mock_download, \
             patch('lib.bars.tardis.Universe') as mock_universe_class:
            
            # Mock API error with availability info
            error_msg = "Error: {\"datasetInfo\": {\"availableSince\": \"2023-01-01T00:00:00.000Z\", \"availableTo\": \"2024-12-31T23:59:59.999Z\"}}"
            mock_download.side_effect = Exception(error_msg)
            
            # Mock Universe to avoid metadata loading
            mock_universe = Mock()
            mock_universe.load_universe_symbols.return_value = ['BTCUSDT', 'ETHUSDT']
            mock_universe_class.return_value = mock_universe
            
            # Mock DataLoader
            mock_data_loader = Mock()
            mock_dl_class.return_value = mock_data_loader
            
            tardis = Tardis(mock_api_key, mock_config, s3=False)
            
            with caplog.at_level(logging.WARNING):
                result = tardis._download_symbol(
                    'BTCUSDT',
                    'trades',
                    'binance-futures',
                    dt(2024, 1, 1).date(),
                    'tardis/20240101'
                )
            
            assert result is False
            assert "download failed for BTCUSDT" in caplog.text
            # Check availability was updated
            assert 'BTCUSDT' in tardis.availability_dict
            assert tardis.availability_dict['BTCUSDT'][0] == dt(2023, 1, 1).date()
    
    def test_dump_prebars_data(self, mock_config, mock_api_key, tmp_path):
        """Test pre-bar data dumping."""
        with self.get_mocked_tardis_context(), \
             patch('lib.bars.tardis.dir_manager') as mock_dir_manager:
            
            mock_dir_manager.PREBAR_DIR = str(tmp_path / 'prebar')
            
            tardis = Tardis(mock_api_key, mock_config, s3=False)
            
            # Create sample bar data with 1440 rows (one per minute for 24 hours)
            timestamps = pd.date_range('2024-01-01', periods=1440, freq='min')
            sample_bars_df = pd.DataFrame({
                'close_mid': [42000 + i for i in range(1440)],
                'volume': [100 + i for i in range(1440)]
            })
            sample_bars_df['symbol'] = 'BTCUSDT'
            sample_bars_df['venue'] = 'binance-futures'
            sample_bars_df['ts'] = timestamps
            sample_bars_df.set_index(['ts', 'symbol', 'venue'], inplace=True)
            
            # Create the directory structure
            dir_path = tmp_path / 'prebar' / 'tardis' / '20240101' / 'binance-futures'
            dir_path.mkdir(parents=True, exist_ok=True)
            
            tardis._dump_prebars_data(
                sample_bars_df,
                'BTCUSDT',
                'binance-futures',
                dt(2024, 1, 1).date()
            )
            
            # Check file was created
            expected_file = dir_path / 'prebars.tardis.binance-futures.20240101.BTCUSDT.parquet'
            assert expected_file.exists()
    
    def test_update_availability_list(self, mock_config, mock_api_key):
        """Test updating availability from error messages."""
        with self.get_mocked_tardis_context():
            
            tardis = Tardis(mock_api_key, mock_config, s3=False)
            
            # Test valid JSON error
            error_msg = 'Error: {"datasetInfo": {"availableSince": "2023-06-01T00:00:00.000Z", "availableTo": "2024-12-31T23:59:59.999Z"}}'
            tardis._update_availability_list('ETHUSDT', error_msg)
            
            assert 'ETHUSDT' in tardis.availability_dict
            assert tardis.availability_dict['ETHUSDT'] == (
                dt(2023, 6, 1).date(),
                dt(2024, 12, 31).date()
            )
            
            # Test invalid JSON
            tardis._update_availability_list('BTCUSDT', 'Not a JSON error')
            assert 'BTCUSDT' not in tardis.availability_dict
    
    def test_download_metadata_info_success(self, mock_config, mock_api_key):
        """Test successful metadata download."""
        with self.get_mocked_tardis_context(), \
             patch('lib.bars.tardis.requests.get') as mock_get:
            
            mock_response = Mock()
            mock_response.json.return_value = {
                'availableSince': '2023-01-01T00:00:00.000Z',
                'availableTo': '2024-12-31T23:59:59.999Z',
                'symbol': 'BTCUSDT',
                'type': 'perpetual',
                'quoteCurrency': 'USDT'
            }
            mock_get.return_value = mock_response
            
            tardis = Tardis(mock_api_key, mock_config, s3=False)
            tardis._download_metadata_info('BTCUSDT', 'binance-futures')
            
            assert 'BTCUSDT' in tardis.availability_dict
            assert 'BTCUSDT' in tardis.metadata_dict
            # Note: availableTo is made inclusive by subtracting 1 day
            assert tardis.availability_dict['BTCUSDT'][1] == dt(2024, 12, 30).date()
    
    def test_download_metadata_info_error(self, mock_config, mock_api_key, caplog):
        """Test metadata download with API error."""
        with self.get_mocked_tardis_context(), \
             patch('lib.bars.tardis.requests.get') as mock_get:
            
            mock_response = Mock()
            mock_response.json.return_value = {'code': 'ERROR', 'message': 'Symbol not found'}
            mock_get.return_value = mock_response
            
            tardis = Tardis(mock_api_key, mock_config, s3=False)
            
            with caplog.at_level(logging.ERROR):
                tardis._download_metadata_info('INVALID', 'binance-futures')
            
            assert "fail to get instrument meta response" in caplog.text
            assert 'INVALID' not in tardis.availability_dict
    
    def test_merge_and_dump_funding_data(self, mock_config, mock_api_key, tmp_path):
        """Test merging and saving funding data."""
        with self.get_mocked_tardis_context(), \
             patch('lib.bars.tardis.concat') as mock_concat:
            
            tardis = Tardis(mock_api_key, mock_config, s3=False, 
                          funding_output_dir=str(tmp_path))
            
            # Create sample funding DataFrames with MultiIndex
            timestamps = pd.date_range('2024-01-01', periods=1440, freq='min')
            index1 = pd.MultiIndex.from_product([timestamps, ['BTCUSDT'], ['binance-futures']], 
                                               names=['ts', 'symbol', 'venue'])
            index2 = pd.MultiIndex.from_product([timestamps, ['ETHUSDT'], ['binance-futures']], 
                                               names=['ts', 'symbol', 'venue'])
            
            df1 = pd.DataFrame({
                'funding_rate': [0.0001] * 1440,
                'open_interest': [1000000] * 1440
            }, index=index1)
            
            df2 = pd.DataFrame({
                'funding_rate': [0.0002] * 1440,
                'open_interest': [2000000] * 1440
            }, index=index2)
            
            # Create merged DataFrame
            merged_df = pd.concat([df1, df2])
            mock_concat.return_value = merged_df
            
            tardis.symbol_funding_dfs = [df1, df2]
            
            tardis._merge_and_dump_funding_data(dt(2024, 1, 1).date())
            
            # Check file was created
            expected_file = tmp_path / 'funding.20240101.parquet'
            assert expected_file.exists()
    
    def test_merge_and_dump_funding_data_no_data(self, mock_config, mock_api_key):
        """Test error when no funding data available."""
        with self.get_mocked_tardis_context():
            
            tardis = Tardis(mock_api_key, mock_config, s3=False)
            tardis.symbol_funding_dfs = []
            
            with pytest.raises(RuntimeError, match="no funding dataframe generated"):
                tardis._merge_and_dump_funding_data(dt(2024, 1, 1).date())
    
    def test_download_and_dump_exchange_info(self, mock_config, mock_api_key, tmp_path):
        """Test downloading and saving exchange info."""
        with self.get_mocked_tardis_context(), \
             patch('lib.bars.tardis.dir_manager') as mock_dir_manager, \
             patch('lib.bars.tardis.get_exchange_info') as mock_get_info, \
             patch('os.path.exists', return_value=True):
            
            mock_dir_manager.EXCH_INFO_DIR = str(tmp_path)
            mock_get_info.return_value = {
                'symbols': [
                    {'symbol': 'BTCUSDT', 'status': 'TRADING'},
                    {'symbol': 'ETHUSDT', 'status': 'TRADING'}
                ]
            }
            
            tardis = Tardis(mock_api_key, mock_config, s3=False)
            
            with patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
                tardis._download_and_dump_exchange_info()
                mock_to_parquet.assert_called_once()
    
    def test_download_and_dump_binance_metadata(self, mock_config, mock_api_key, tmp_path):
        """Test downloading metadata for all symbols."""
        with patch('lib.bars.tardis.DataLoader') as mock_dl_class, \
             patch('lib.bars.tardis.Calcs'), \
             patch('lib.bars.tardis.dir_manager') as mock_dir_manager, \
             patch('os.path.exists', return_value=True), \
             patch('lib.bars.tardis.Universe') as mock_universe_class:
            
            mock_data_loader = Mock()
            mock_data_loader.load_universe_symbols.return_value = ['BTCUSDT', 'ETHUSDT']
            mock_dl_class.return_value = mock_data_loader
            
            # Mock Universe to return symbols
            mock_universe = Mock()
            mock_universe.load_universe_symbols.return_value = ['BTCUSDT', 'ETHUSDT']
            mock_universe_class.return_value = mock_universe
            
            mock_dir_manager.BINANCE_META_DIR = str(tmp_path)
            
            tardis = Tardis(mock_api_key, mock_config, s3=False, symbols=['BTCUSDT', 'ETHUSDT'])
            
            # Mock the download method
            tardis._download_metadata_info = Mock()
            
            with patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
                tardis._download_and_dump_current_binance_metadata('binance-futures', symbols=['BTCUSDT', 'ETHUSDT'])
                
                # Should call download for each symbol
                assert tardis._download_metadata_info.call_count == 2
                mock_to_parquet.assert_called_once()
    
    def test_generate_1m_fundings_for_symbol(self, mock_config, mock_api_key, 
                                            tmp_path, sample_funding_df):
        """Test generating 1-minute funding data."""
        with self.get_mocked_tardis_context(), \
             patch('lib.bars.tardis.DATA_DIR', str(tmp_path)), \
             patch('os.path.isfile', return_value=True), \
             patch('os.path.getsize', return_value=1000):
            
            tardis = Tardis(mock_api_key, mock_config, s3=False)
            
            # Create mock CSV file
            csv_file = tmp_path / 'tardis' / '20240101' / 'derivative_ticker.BTCUSDT.binance-futures.20240101.csv.gz'
            csv_file.parent.mkdir(parents=True, exist_ok=True)
            
            with patch('pandas.read_csv', return_value=sample_funding_df):
                tardis._generate_1m_fundings_for_symbol(
                    'BTCUSDT',
                    'binance-futures',
                    dt(2024, 1, 1).date(),
                    'tardis/20240101'
                )
            
            assert len(tardis.symbol_funding_dfs) == 1
            # Should have 1440 minutes of data
            assert len(tardis.symbol_funding_dfs[0]) == 1440
    
    def test_generate_1m_bars_for_symbol(self, mock_config, mock_api_key,
                                        tmp_path, sample_book_df, sample_trades_df):
        """Test generating 1-minute bars from raw data."""
        with patch('lib.bars.tardis.DataLoader'), \
             patch('lib.bars.tardis.Calcs') as mock_calcs_class, \
             patch('lib.bars.tardis.DATA_DIR', str(tmp_path)), \
             patch('lib.bars.tardis.dir_manager') as mock_dir_manager, \
             patch('os.path.isfile') as mock_isfile:
            
            mock_calcs = Mock()
            mock_calcs.calc_logret.side_effect = lambda df: df
            mock_calcs.calc_vwap.side_effect = lambda df: df
            mock_calcs_class.return_value = mock_calcs
            
            # Use a temporary directory for testing
            setattr(mock_dir_manager, 'TARDIS_MINUTE_BAR_DIR', str(tmp_path / 'bars'))
            mock_dir_manager.PREBAR_DIR = str(tmp_path / 'prebar')
            
            # Mock file existence checks
            def isfile_side_effect(path):
                if 'book_snapshot' in path or 'trades' in path:
                    return True
                return False
            mock_isfile.side_effect = isfile_side_effect
            
            with patch('os.path.getsize', return_value=1000), \
                 patch('pandas.read_csv') as mock_read_csv:
                
                # Return appropriate data based on filename
                def read_csv_side_effect(filename):
                    if 'book_snapshot' in filename:
                        return sample_book_df
                    if 'trades' in filename:
                        return sample_trades_df
                    return pd.DataFrame()
                
                mock_read_csv.side_effect = read_csv_side_effect
                # Need to patch Universe here too
                with patch('lib.bars.tardis.Universe') as mock_universe_class:
                    mock_universe = Mock()
                    mock_universe.load_universe_symbols.return_value = ['BTCUSDT', 'ETHUSDT']
                    mock_universe_class.return_value = mock_universe
                    
                    tardis = Tardis(mock_api_key, mock_config, s3=False, debug=False)
                    
                    barfile_dir = str(tmp_path / 'bars' / '20240101' / 'binance-futures')
                    os.makedirs(barfile_dir, exist_ok=True)
                    
                    with patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
                        tardis._generate_prebars_for_symbol(
                            'BTCUSDT',
                            'binance-futures',
                            dt(2024, 1, 1).date(),
                            'tardis/20240101',
                            barfile_dir
                        )
                        
                        # Should save bar file
                        mock_to_parquet.assert_called()
    
    def test_download_files_basic(self, mock_config, mock_api_key):
        """Test basic download_files functionality."""
        with self.get_mocked_tardis_context(), \
             patch('lib.bars.tardis.safe_mkdir'):
            
            tardis = Tardis(mock_api_key, mock_config, s3=False, backfill=True, symbols=['BTCUSDT', 'ETHUSDT'])
            
            # Mock the download and generation methods
            tardis._download_files_and_generate_bars = Mock()
            tardis._merge_and_dump_funding_data = Mock()
            
            tardis.download_files(
                dt(2024, 1, 1).date(),
                dt(2024, 1, 2).date()
            )
            
            # Should call for each date and symbol (2 dates * 2 symbols)
            assert tardis._download_files_and_generate_bars.call_count == 4
            # Should merge funding data if derivative_ticker in data_types
            assert tardis._merge_and_dump_funding_data.call_count == 2
    
    def test_download_files_multiprocessing(self, mock_config, mock_api_key):
        """Test download_files with multiprocessing."""
        with patch('lib.bars.tardis.DataLoader') as mock_dl_class, \
             patch('lib.bars.tardis.Calcs'), \
             patch('lib.bars.tardis.safe_mkdir'), \
             patch('lib.bars.tardis.Pool') as mock_pool_class, \
             patch('lib.bars.tardis.Universe') as mock_universe_class:
            
            mock_data_loader = Mock()
            mock_data_loader.load_universe_symbols.return_value = ['BTCUSDT', 'ETHUSDT']
            mock_dl_class.return_value = mock_data_loader
            
            # Mock Universe to avoid metadata loading
            mock_universe = Mock()
            mock_universe.load_universe_symbols.return_value = ['BTCUSDT', 'ETHUSDT']
            mock_universe_class.return_value = mock_universe
            
            mock_pool = Mock()
            mock_pool.starmap.return_value = [None, None]
            mock_pool_class.return_value = mock_pool
            
            tardis = Tardis(mock_api_key, mock_config, s3=False, 
                          pool_size=2, backfill=True,
                          symbols=['BTCUSDT', 'ETHUSDT'],
                          data_types=['trades', 'book_snapshot_5'])  # No derivative_ticker
            
            tardis.download_files(
                dt(2024, 1, 1).date(),
                dt(2024, 1, 1).date()
            )
            
            # Should create pool with 2 processes
            mock_pool_class.assert_called_once_with(processes=2)
            mock_pool.starmap.assert_called_once()
            mock_pool.close.assert_called_once()
            mock_pool.join.assert_called_once()
    
    def test_download_files_and_generate_bars(self, mock_config, mock_api_key):
        """Test download_files_and_generate_bars coordination."""
        with self.get_mocked_tardis_context():
            
            tardis = Tardis(mock_api_key, mock_config, s3=False)
            
            # Mock internal methods
            tardis._download_symbol = Mock(return_value=True)
            tardis._generate_prebars_for_symbol = Mock()
            tardis._generate_1m_fundings_for_symbol = Mock()
            
            # Test normal flow
            tardis._download_files_and_generate_bars(
                'BTCUSDT',
                dt(2024, 1, 1).date(),
                '/bars',
                'binance-futures',
                'tardis/20240101'
            )
            
            # Should download all data types
            assert tardis._download_symbol.call_count == len(tardis.data_types)
            
            # Should generate bars and funding
            tardis._generate_prebars_for_symbol.assert_called_once()
            tardis._generate_1m_fundings_for_symbol.assert_called_once()
    
    def test_download_files_and_generate_bars_availability(self, mock_config, 
                                                          mock_api_key, caplog):
        """Test respecting availability windows."""
        with self.get_mocked_tardis_context():
            
            tardis = Tardis(mock_api_key, mock_config, s3=False)
            
            # Set availability window
            tardis.availability_dict['BTCUSDT'] = (
                dt(2023, 1, 1).date(),
                dt(2023, 12, 31).date()
            )
            
            # Mock methods
            tardis._download_symbol = Mock()
            
            with caplog.at_level(logging.INFO):
                # Try to download after availability window
                tardis._download_files_and_generate_bars(
                    'BTCUSDT',
                    dt(2024, 1, 1).date(),
                    '/bars',
                    'binance-futures',
                    'tardis/20240101'
                )
            
            # Should skip download
            assert "Skipping BTCUSDT after" in caplog.text
            tardis._download_symbol.assert_not_called()
    
    def test_tardis_bar_start_date(self):
        """Test the default Tardis bar start date constant."""
        assert TARDIS_BAR_START_DATE == dt(2023, 1, 1).date()
