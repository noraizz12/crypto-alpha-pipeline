"""
Unit tests for lib/dataframes.py module.

Tests cover DataFrame manipulation, validation, and utility functions used
throughout the trading system for data processing and analysis.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import CategoricalDtype

from lib.util import dataframes
from lib.util.util import TARDIS_EXCHANGE


class TestDataFrameValidation:
    """Test DataFrame validation functions."""
    
    def test_check_dup_rows_no_duplicates(self):
        """Test check_dup_rows with no duplicate rows."""
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=['a', 'b', 'c'])
        
        # Should not raise
        dataframes.check_dup_rows(df)
    
    def test_check_dup_rows_with_duplicates(self):
        """Test check_dup_rows raises on duplicate rows."""
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=['a', 'b', 'b'])  # Duplicate index
        
        with pytest.raises(RuntimeError):
            dataframes.check_dup_rows(df)
    
    def test_check_dup_cols_no_duplicates(self):
        """Test check_dup_cols with no duplicate columns."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        # Should not raise
        dataframes.check_dup_cols(df)
    
    @patch('lib.util.dataframes.log_and_raise')
    def test_check_dup_cols_with_duplicates(self, mock_log_and_raise):
        """Test check_dup_cols raises on duplicate columns."""
        mock_log_and_raise.side_effect = Exception("Duplicate columns found")
        
        # Create DataFrame with duplicate columns
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'b'])
        
        with pytest.raises(Exception, match="Duplicate columns found"):
            dataframes.check_dup_cols(df)
    
    def test_check_df_valid(self):
        """Test check_df with valid DataFrame."""
        ts = pd.date_range('2023-01-01', periods=3, freq='min')
        df = pd.DataFrame({
            'ts': ts,
            'symbol_venue': ['BTCUSDT_binance-futures'] * 3,
            'value': [1, 2, 3]
        })
        df = df.set_index(['ts', 'symbol_venue'])
        
        # Should not raise
        dataframes.check_df(df)
    
    @patch('lib.util.dataframes.log_and_raise')
    def test_check_df_bad_index(self, mock_log_and_raise):
        """Test check_df raises on bad index."""
        mock_log_and_raise.side_effect = Exception("bad dataframe index")
        
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=['a', 'b', 'c'])
        
        with pytest.raises(Exception, match="bad dataframe index"):
            dataframes.check_df(df)
    
    def test_check_df_ignore_index(self):
        """Test check_df with ignore_index=True."""
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=['a', 'b', 'c'])
        
        # Should not raise even with wrong index
        dataframes.check_df(df, ignore_index=True)
    
    def test_check_missing_ts(self):
        """Test check_missing_ts identifies gaps in time series."""
        # Create DataFrame with gap - using UTC to match expected behavior
        ts1 = pd.date_range('2023-01-01 00:00', '2023-01-01 00:02', freq='min', tz='UTC')
        ts2 = pd.date_range('2023-01-01 00:04', '2023-01-01 00:05', freq='min', tz='UTC')
        ts = pd.DatetimeIndex(list(ts1) + list(ts2))
        
        df = pd.DataFrame({
            'ts': ts,
            'symbol_venue': ['BTCUSDT_binance-futures'] * len(ts),
            'value': range(len(ts))
        })
        df = df.set_index(['ts', 'symbol_venue'])
        
        missing = dataframes.check_missing_ts(df)
        
        assert len(missing) == 1
        assert missing[0] == pd.Timestamp('2023-01-01 00:03', tz='UTC')
    
    def test_check_missing_ts_no_gaps(self):
        """Test check_missing_ts with continuous data."""
        ts = pd.date_range('2023-01-01', periods=5, freq='min', tz='UTC')
        df = pd.DataFrame({
            'ts': ts,
            'symbol_venue': ['BTCUSDT_binance-futures'] * 5,
            'value': range(5)
        })
        df = df.set_index(['ts', 'symbol_venue'])
        
        missing = dataframes.check_missing_ts(df)
        assert len(missing) == 0


class TestIndexManipulation:
    """Test index manipulation functions."""
    
    def test_set_index_valid(self):
        """Test set_index with valid data."""
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=3, freq='min'),
            'symbol_venue': ['BTCUSDT_binance-futures'] * 3,
            'value': [1, 2, 3]
        })
        
        result = dataframes.set_index(df)
        
        assert result.index.names == ['ts', 'symbol_venue']
        assert len(result) == 3
        assert 'ts' not in result.columns
        assert 'symbol_venue' not in result.columns
    
    def test_set_index_nan_timestamp(self):
        """Test set_index raises on NaT timestamp."""
        df = pd.DataFrame({
            'ts': [pd.Timestamp('2023-01-01'), pd.NaT, pd.Timestamp('2023-01-03')],
            'symbol_venue': ['BTCUSDT_binance-futures'] * 3,
            'value': [1, 2, 3]
        })
        
        with pytest.raises(Exception, match="Found 1 NaT values in ts column"):
            dataframes.set_index(df)
    
    def test_set_index_multiple_nat_values(self):
        """Test set_index error message with multiple NaT values."""
        df = pd.DataFrame({
            'ts': [pd.NaT, pd.Timestamp('2023-01-02'), pd.NaT, pd.NaT],
            'symbol_venue': ['BTCUSDT_binance-futures'] * 4,
            'value': [1, 2, 3, 4]
        })
        
        with pytest.raises(Exception, match="Found 3 NaT values in ts column"):
            dataframes.set_index(df)
    
    def test_set_index_fast_mode(self):
        """Test set_index with fast=True skips integrity check."""
        # Create DataFrame with duplicate index (would fail integrity check)
        df = pd.DataFrame({
            'ts': [pd.Timestamp('2023-01-01')] * 2,
            'symbol_venue': ['BTCUSDT_binance-futures'] * 2,
            'value': [1, 2]
        })
        
        # Should not raise with fast=True even with duplicates
        result = dataframes.set_index(df, fast=True)
        assert result.index.names == ['ts', 'symbol_venue']
    

class TestDataTypeConversion:
    """Test data type conversion functions."""
    
    def test_shrink_floats(self):
        """Test shrink_floats converts float64 to float32."""
        df = pd.DataFrame({
            'float64_col': np.array([1.0, 2.0, 3.0], dtype=np.float64),
            'float32_col': np.array([4.0, 5.0, 6.0], dtype=np.float32),
            'int_col': [7, 8, 9]
        })
        
        result = dataframes.shrink_floats(df)
        
        assert result['float64_col'].dtype == np.float32
        assert result['float32_col'].dtype == np.float32
        assert result['int_col'].dtype == np.int64  # Unchanged
    
    def test_shrink_floats_no_float64(self):
        """Test shrink_floats with no float64 columns."""
        df = pd.DataFrame({
            'float32_col': np.array([1.0, 2.0, 3.0], dtype=np.float32),
            'int_col': [4, 5, 6]
        })
        
        result = dataframes.shrink_floats(df)
        
        assert result['float32_col'].dtype == np.float32
        assert result['int_col'].dtype == np.int64
    
    def test_categorize_column(self):
        """Test categorize_column filters and converts to categorical."""
        cat = CategoricalDtype(['BTCUSDT', 'ETHUSDT'], ordered=True)
        df = pd.DataFrame({
            'symbol': ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'BTCUSDT'],
            'value': [1, 2, 3, 4]
        })

        with patch('lib.util.dataframes.logger') as mock_logger:
            result = dataframes.convert_column_to_categorical(df, 'symbol', cat)

            # Should filter out XRPUSDT
            assert len(result) == 3
            assert 'XRPUSDT' not in result['symbol'].values
            assert result['symbol'].dtype == cat
            
            # Should log warning about size change
            assert any('Dataframe from 4 -> 3 converting symbol to categorical' in str(call)
                      for call in mock_logger.warning.call_args_list)
    
    def test_remove_infs(self):
        """Test remove_infs replaces infinity with NaN."""
        df = pd.DataFrame({
            'col1': [1.0, np.inf, -np.inf, 4.0],
            'col2': [np.inf, 2.0, 3.0, -np.inf]
        })
        
        result = dataframes.remove_infs(df)
        
        assert pd.isna(result.loc[1, 'col1'])
        assert pd.isna(result.loc[2, 'col1'])
        assert pd.isna(result.loc[0, 'col2'])
        assert pd.isna(result.loc[3, 'col2'])
        assert result.loc[0, 'col1'] == 1.0
        assert result.loc[1, 'col2'] == 2.0


class TestColumnManipulation:
    """Test column manipulation functions."""
    
    def test_make_symbol_venue(self):
        """Test make_symbol_venue creates combined column."""
        df = pd.DataFrame({
            'symbol': ['BTCUSDT', 'ETHUSDT'],
            'venue': ['binance-futures', 'binance-futures'],
            'value': [1, 2]
        })

        result = dataframes.make_symbol_venue(df)

        assert 'symbol_venue' in result.columns
        assert result['symbol_venue'].iloc[0] == 'BTCUSDT_binance-futures'
        assert result['symbol_venue'].iloc[1] == 'ETHUSDT_binance-futures'
    
    def test_make_symbol_venue_no_venue(self):
        """Test make_symbol_venue defaults to TARDIS_EXCHANGE."""
        df = pd.DataFrame({
            'symbol': ['BTCUSDT'],
            'value': [1]
        })

        result = dataframes.make_symbol_venue(df)

        assert result['symbol_venue'].iloc[0] == f'BTCUSDT_{TARDIS_EXCHANGE}'
        assert result['venue'].iloc[0] == TARDIS_EXCHANGE
    
    def test_flatten_cols(self):
        """Test flatten_cols converts MultiIndex columns to flat."""
        # Create DataFrame with MultiIndex columns
        arrays = [['A', 'A', 'B'], ['X', 'Y', 'X']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        df = pd.DataFrame(np.random.randn(3, 3), columns=index)
        
        result = dataframes.flatten_cols(df)
        
        assert list(result.columns) == ['A_X', 'A_Y', 'B_X']
        assert not isinstance(result.columns, pd.MultiIndex)
    
    def test_make_date(self):
        """Test make_date extracts date with 1-minute offset."""
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-02 00:00', periods=3, freq='30min'),
            'value': [1, 2, 3]
        })
        
        result = dataframes.make_date(df)
        
        # First timestamp at midnight should map to previous day
        assert result['date'].iloc[0] == pd.Timestamp('2023-01-01').normalize()
        # 00:30 - 1min = 00:29, which is 2023-01-02
        assert result['date'].iloc[1] == pd.Timestamp('2023-01-02').normalize()
        assert result['date'].iloc[2] == pd.Timestamp('2023-01-02').normalize()
    
    def test_make_date_existing_valid(self):
        """Test make_date skips if valid date column exists."""
        df = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=2, freq='h'),
            'date': pd.date_range('2023-01-01', periods=2, freq='D'),
            'value': [1, 2]
        })
        
        result = dataframes.make_date(df)
        
        # Should keep existing date column
        assert (result['date'] == df['date']).all()
    
    def test_make_date_from_multiindex(self):
        """Test make_date with timestamp in MultiIndex."""
        ts = pd.date_range('2023-01-02 00:00', periods=3, freq='30min', tz='UTC')
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=pd.MultiIndex.from_arrays([
            ts,
            ['A', 'B', 'C']
        ], names=['ts', 'symbol']))
        
        result = dataframes.make_date(df)
        
        # First timestamp at midnight should map to previous day
        assert result['date'].iloc[0] == pd.Timestamp('2023-01-01', tz='UTC').normalize()
        # 00:30 - 1min = 00:29, which is 2023-01-02
        assert result['date'].iloc[1] == pd.Timestamp('2023-01-02', tz='UTC').normalize()
        assert result['date'].iloc[2] == pd.Timestamp('2023-01-02', tz='UTC').normalize()
    
    def test_make_symbol(self):
        """Test make_symbol extracts symbol from symbol_venue."""
        df = pd.DataFrame({
            'symbol_venue': ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures'],
            'value': [1, 2]
        })
        
        result = dataframes.make_symbol(df)
        
        assert 'symbol' in result.columns
        assert result['symbol'].iloc[0] == 'BTCUSDT'
        assert result['symbol'].iloc[1] == 'ETHUSDT'
    
    def test_make_symbol_from_index(self):
        """Test make_symbol works with symbol_venue in index."""
        ts = pd.date_range('2023-01-01', periods=2, freq='min')
        df = pd.DataFrame({
            'value': [1, 2]
        }, index=pd.MultiIndex.from_arrays([
            ts,
            ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures']
        ], names=['ts', 'symbol_venue']))
        
        result = dataframes.make_symbol(df)
        
        assert 'symbol' in result.columns
        assert result['symbol'].iloc[0] == 'BTCUSDT'
    
    @patch('lib.util.dataframes.log_and_raise')
    def test_make_symbol_no_symbol_venue(self, mock_log_and_raise):
        """Test make_symbol raises when no symbol_venue found."""
        mock_log_and_raise.side_effect = Exception("No symbol in dataframe")
        
        df = pd.DataFrame({
            'value': [1, 2]
        })
        
        with pytest.raises(Exception, match="No symbol in dataframe"):
            dataframes.make_symbol(df)
    
    def test_make_quintile(self):
        """Test make_quintile creates quantile buckets."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        result = dataframes.make_quintile(df, 'value', q=5)
        
        assert 'value_quintile' in result.columns
        assert result['value_quintile'].nunique() == 5
        assert isinstance(result['value_quintile'].dtype, pd.CategoricalDtype)
    
    @patch('lib.util.dataframes.logger')
    def test_make_quintile_insufficient_unique(self, mock_logger):
        """Test make_quintile warns when insufficient unique values."""
        df = pd.DataFrame({
            'value': [1, 1, 2, 2]  # Only 2 unique values
        })
        
        result = dataframes.make_quintile(df, 'value', q=5)
        
        # Should still create quintiles, just fewer than requested
        assert 'value_quintile' in result.columns
        assert result['value_quintile'].nunique() <= 2
        
        # Should log warning
        mock_logger.warning.assert_called_with('value only has 2 unique values')


class TestSafeOperations:
    """Test safe operation functions."""
    
    def test_safe_get_existing_column(self):
        """Test safe_get returns column when it exists."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        result = dataframes.safe_get(df, 'col1')
        
        assert isinstance(result, pd.Series)
        assert (result == df['col1']).all()
    
    @patch('lib.util.dataframes.logger')
    def test_safe_get_missing_column(self, mock_logger):
        """Test safe_get returns None for missing column."""
        df = pd.DataFrame({
            'col1': [1, 2, 3]
        })
        
        result = dataframes.safe_get(df, 'col2')
        
        assert result is None
        mock_logger.warning.assert_called_with('there is no col2 inside the df')
    
    def test_safe_get_from_series_existing(self):
        """Test safe_get_from_series returns value when it exists."""
        series = pd.Series({'a': 1, 'b': 2, 'c': 3})
        
        result = dataframes.safe_get_from_series(series, 'b')
        
        assert result == 2
    
    @patch('lib.util.dataframes.logger')
    def test_safe_get_from_series_missing(self, mock_logger):
        """Test safe_get_from_series returns None for missing key."""
        series = pd.Series({'a': 1, 'b': 2})
        
        result = dataframes.safe_get_from_series(series, 'c')
        
        assert result is None
        mock_logger.warning.assert_called_with('there is no c inside the series')
    
    def test_safe_del_existing_column(self):
        """Test safe_del removes existing column."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        result = dataframes.safe_del(df, 'col2')
        
        assert 'col2' not in result.columns
        assert 'col1' in result.columns
    
    def test_safe_del_missing_column(self):
        """Test safe_del handles missing column gracefully."""
        df = pd.DataFrame({
            'col1': [1, 2, 3]
        })
        
        # Should not raise
        result = dataframes.safe_del(df, 'col2')
        
        assert 'col1' in result.columns
        assert len(result.columns) == 1
    
    def test_clean_poop(self):
        """Test clean_poop removes _poop suffixed columns."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2_poop': [4, 5, 6],
            'col3': [7, 8, 9],
            'col4_poop': [10, 11, 12]
        })
        
        result = dataframes.clean_poop(df)
        
        assert 'col1' in result.columns
        assert 'col3' in result.columns
        assert 'col2_poop' not in result.columns
        assert 'col4_poop' not in result.columns


class TestDataAnalysis:
    """Test data analysis functions."""
    
    @patch('lib.util.dataframes.logger')
    def test_log_df_mem_usage(self, mock_logger):
        """Test log_df_mem_usage logs memory statistics."""
        df = pd.DataFrame({
            'int_col': range(1000),
            'float_col': np.random.randn(1000),
            'str_col': ['test'] * 1000
        })
        
        dataframes.log_df_mem_usage(df, 'test_df')
        
        # Should log memory usage
        assert mock_logger.info.called
        call_args = str(mock_logger.info.call_args)
        assert 'test_df' in call_args
        assert 'MB' in call_args
    
    @patch('lib.util.dataframes.logger')
    def test_log_col_summary(self, mock_logger):
        """Test log_col_summary logs column statistics."""
        ts = pd.date_range('2023-01-01', periods=10, freq='min')
        df = pd.DataFrame({
            'value': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10]
        }, index=pd.MultiIndex.from_arrays([
            ts,
            ['BTCUSDT_binance-futures'] * 10
        ], names=['ts', 'symbol_venue']))
        
        dataframes.log_col_summary(df, 'value')
        
        # Should log summary statistics
        assert mock_logger.info.called
        call_args = str(mock_logger.info.call_args)
        assert 'value' in call_args
        assert 'NAs 1/10' in call_args
        assert 'min=' in call_args
        assert 'max=' in call_args
        assert 'std=' in call_args
    
    def test_long_short_sums(self):
        """Test long_short_sums calculates directional sums."""
        df = pd.DataFrame({
            'positions': [100, -50, 200, -75, 0, 150]
        })
        
        long_sum, short_sum = dataframes.long_short_sums(df, 'positions')
        
        assert long_sum == 450  # 100 + 200 + 150
        assert short_sum == -125  # -50 + -75
    
    def test_long_short_sums_no_values(self):
        """Test long_short_sums with no positive/negative values."""
        df = pd.DataFrame({
            'positions': [0, 0, 0]
        })
        
        long_sum, short_sum = dataframes.long_short_sums(df, 'positions')
        
        assert long_sum == 0
        assert short_sum == 0


class TestMergeConcat:
    """Test merge and concatenation operations."""
    
    def test_concat_basic(self):
        """Test basic DataFrame concatenation."""
        df1 = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}, index=[0, 1])
        df2 = pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]}, index=[2, 3])
        
        result = dataframes.concat([df1, df2])
        
        assert len(result) == 4
        assert list(result['col1']) == [1, 2, 5, 6]
        assert list(result['col2']) == [3, 4, 7, 8]
    
    def test_concat_empty_list(self):
        """Test concat with empty list returns None."""
        result = dataframes.concat([])
        assert result is None
    
    def test_concat_dtype_conversion(self):
        """Test concat attempts dtype standardization."""
        df1 = pd.DataFrame({'col1': [1, 2], 'col2': [3.0, 4.0]}, index=[0, 1])
        df2 = pd.DataFrame({'col1': [5, 6], 'col2': [7.0, 8.0]}, index=[2, 3])

        # Change dtype of second DataFrame
        df2['col2'] = df2['col2'].astype(np.float32)

        # Use strict=False to allow dtype conversion without raising
        result = dataframes.concat([df1, df2], strict=False)

        assert len(result) == 4
        # Should attempt to standardize dtypes

    def test_concat_float_to_int_conversion(self):
        """Test concat converts float columns to int when first df has int dtype.

        This test specifically targets line 797 in dataframes.concat() where
        float values should be converted to int type to match the first dataframe.
        """
        # First dataframe has integer column
        df1 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]}, index=[0, 1])
        df1['id'] = df1['id'].astype(np.int64)

        # Second dataframe has float column (but values are whole numbers)
        df2 = pd.DataFrame({'id': [3.0, 4.0], 'value': [30, 40]}, index=[2, 3])
        df2['id'] = df2['id'].astype(np.float64)

        # Concat should convert df2's float column to int to match df1
        result = dataframes.concat([df1, df2], strict=False, quiet=True)

        assert len(result) == 4
        # Verify the id column is int type (from first dataframe)
        assert pd.api.types.is_integer_dtype(result['id']), \
            f"Expected int dtype for 'id' column, got {result['id'].dtype}"
        # Verify all values are correct
        assert list(result['id']) == [1, 2, 3, 4]
        assert list(result['value']) == [10, 20, 30, 40]

    def test_concat_float_to_int_with_nan(self):
        """Test concat converts float with NaN to int by filling with 0.

        This test specifically targets line 797 in dataframes.concat() where
        NaN values in float columns should be filled with 0 before converting to int,
        since int types cannot represent NaN.
        """
        # First dataframe has integer column
        df1 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]}, index=[0, 1])
        df1['id'] = df1['id'].astype(np.int64)

        # Second dataframe has float column with NaN values
        df2 = pd.DataFrame({'id': [3.0, np.nan], 'value': [30, 40]}, index=[2, 3])
        df2['id'] = df2['id'].astype(np.float64)

        # Concat should convert df2's float column to int, filling NaN with 0
        result = dataframes.concat([df1, df2], strict=False, quiet=True)

        assert len(result) == 4
        # Verify the id column is int type
        assert pd.api.types.is_integer_dtype(result['id']), \
            f"Expected int dtype for 'id' column, got {result['id'].dtype}"
        # Verify NaN was filled with 0
        assert list(result['id']) == [1, 2, 3, 0]
        assert list(result['value']) == [10, 20, 30, 40]

    def test_concat_float_to_int_with_decimals(self):
        """Test concat truncates decimal values when converting float to int.

        This test verifies behavior when float values have decimal parts that
        need to be truncated during int conversion at line 797.
        """
        # First dataframe has integer column
        df1 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]}, index=[0, 1])
        df1['id'] = df1['id'].astype(np.int64)

        # Second dataframe has float column with decimal values
        df2 = pd.DataFrame({'id': [3.7, 4.9], 'value': [30, 40]}, index=[2, 3])
        df2['id'] = df2['id'].astype(np.float64)

        # Concat should convert df2's float column to int (truncating decimals)
        result = dataframes.concat([df1, df2], strict=False, quiet=True)

        assert len(result) == 4
        # Verify the id column is int type
        assert pd.api.types.is_integer_dtype(result['id']), \
            f"Expected int dtype for 'id' column, got {result['id'].dtype}"
        # Verify decimal values were truncated (not rounded)
        assert list(result['id']) == [1, 2, 3, 4]  # 3.7->3, 4.9->4 (truncated)

    def test_concat_float32_to_int32_with_inf(self):
        """Test concat handles infinity values when converting float32 to int32.

        This test reproduces the error: "cannot safely cast non-equivalent float32 to int32"
        which occurs when infinity values are present in float columns.
        Line 797 must replace inf/-inf with 0 before converting to int.
        """
        # First dataframe has int32 column
        df1 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]}, index=[0, 1])
        df1['id'] = df1['id'].astype(np.int32)

        # Second dataframe has float32 column with infinity values
        df2 = pd.DataFrame({'id': [3.0, np.inf, -np.inf, np.nan], 'value': [30, 40, 50, 60]}, index=[2, 3, 4, 5])
        df2['id'] = df2['id'].astype(np.float32)

        # Concat should convert df2's float column to int, replacing inf and NaN with 0
        result = dataframes.concat([df1, df2], strict=False, quiet=True)

        assert len(result) == 6
        # Verify the id column is int type
        assert pd.api.types.is_integer_dtype(result['id']), \
            f"Expected int dtype for 'id' column, got {result['id'].dtype}"
        # Verify inf, -inf, and NaN were all replaced with 0
        assert list(result['id']) == [1, 2, 3, 0, 0, 0]

    @patch('lib.util.dataframes.logger')
    def test_concat_mismatched_columns(self, mock_logger):
        """Test concat handles mismatched columns."""
        df1 = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}, index=[0, 1])
        df2 = pd.DataFrame({'col1': [5, 6], 'col3': [7, 8]}, index=[2, 3])  # Different column
        
        # Use strict=False to allow mismatched columns
        result = dataframes.concat([df1, df2], strict=False)
        
        # Should still concatenate, filling missing values with NaN
        assert len(result) == 4
        assert 'col1' in result.columns
        assert 'col2' in result.columns
        assert 'col3' in result.columns
        
        # Should log warning about dtype conversion issue
        assert mock_logger.warning.called
    
    def test_merge_on_index_left(self):
        """Test merge_on_index with left join."""
        ts = pd.date_range('2023-01-01', periods=3, freq='min')
        df1 = pd.DataFrame({
            'col1': [1, 2, 3]
        }, index=ts)
        
        df2 = pd.DataFrame({
            'col2': [4, 5]
        }, index=ts[:2])  # Only first 2 timestamps
        
        result = dataframes.merge_on_index(df1, df2, how='left')
        
        assert len(result) == 3
        assert list(result['col1']) == [1, 2, 3]
        assert list(result['col2'][:2]) == [4, 5]
        assert pd.isna(result['col2'].iloc[2])
    
    def test_merge_on_index_suffix_cleanup(self):
        """Test merge_on_index cleans up _poop suffix."""
        df1 = pd.DataFrame({
            'col1': [1, 2],
            'shared': [3, 4]
        }, index=['a', 'b'])
        
        df2 = pd.DataFrame({
            'col2': [5, 6],
            'shared': [7, 8]
        }, index=['a', 'b'])
        
        result = dataframes.merge_on_index(df1, df2)
        
        assert 'shared' in result.columns
        assert 'shared_poop' not in result.columns
        assert list(result['shared']) == [3, 4]  # Kept left side
    
    def test_merge_on_index_custom_suffixes(self):
        """Test merge_on_index with custom suffixes."""
        df1 = pd.DataFrame({
            'shared': [1, 2]
        }, index=['a', 'b'])
        
        df2 = pd.DataFrame({
            'shared': [3, 4]
        }, index=['a', 'b'])
        
        result = dataframes.merge_on_index(df1, df2, suffixes=('_left', '_right'))
        
        assert 'shared_left' in result.columns
        assert 'shared_right' in result.columns
        assert 'shared_poop' not in result.columns  # No cleanup with custom suffixes

    def test_merge_on_index_left_join_empty_df2(self):
        """Test left join with empty df2 adds df2 columns as NaN."""
        df1 = pd.DataFrame({
            'col1': [1, 2, 3]
        }, index=['a', 'b', 'c'])

        df2 = pd.DataFrame({
            'col2': pd.Series([], dtype=float)
        }, index=pd.Index([], dtype=str))

        result = dataframes.merge_on_index(df1, df2, how='left')

        assert len(result) == 3
        assert list(result['col1']) == [1, 2, 3]
        assert 'col2' in result.columns
        assert result['col2'].isna().all()

    def test_merge_on_index_right_join_empty_df1(self):
        """Test right join with empty df1 adds df1 columns as NaN."""
        df1 = pd.DataFrame({
            'col1': pd.Series([], dtype=float)
        }, index=pd.Index([], dtype=str))

        df2 = pd.DataFrame({
            'col2': [4, 5, 6]
        }, index=['a', 'b', 'c'])

        result = dataframes.merge_on_index(df1, df2, how='right')

        assert len(result) == 3
        assert list(result['col2']) == [4, 5, 6]
        assert 'col1' in result.columns
        assert result['col1'].isna().all()

    def test_merge_on_index_outer_join_empty_df1(self):
        """Test outer join with empty df1 adds df1 columns as NaN."""
        df1 = pd.DataFrame({
            'col1': pd.Series([], dtype=float)
        }, index=pd.Index([], dtype=str))

        df2 = pd.DataFrame({
            'col2': [4, 5]
        }, index=['a', 'b'])

        result = dataframes.merge_on_index(df1, df2, how='outer')

        assert len(result) == 2
        assert list(result['col2']) == [4, 5]
        assert 'col1' in result.columns
        assert result['col1'].isna().all()

    def test_merge_on_index_outer_join_empty_df2(self):
        """Test outer join with empty df2 adds df2 columns as NaN."""
        df1 = pd.DataFrame({
            'col1': [1, 2]
        }, index=['a', 'b'])

        df2 = pd.DataFrame({
            'col2': pd.Series([], dtype=float)
        }, index=pd.Index([], dtype=str))

        result = dataframes.merge_on_index(df1, df2, how='outer')

        assert len(result) == 2
        assert list(result['col1']) == [1, 2]
        assert 'col2' in result.columns
        assert result['col2'].isna().all()

    def test_merge_on_index_outer_join_both_empty(self):
        """Test outer join with both dfs empty returns empty df with all columns."""
        df1 = pd.DataFrame({
            'col1': pd.Series([], dtype=float)
        }, index=pd.Index([], dtype=str))

        df2 = pd.DataFrame({
            'col2': pd.Series([], dtype=float)
        }, index=pd.Index([], dtype=str))

        result = dataframes.merge_on_index(df1, df2, how='outer')

        assert len(result) == 0
        assert 'col1' in result.columns
        assert 'col2' in result.columns

    def test_merge_on_index_left_join_empty_df1(self):
        """Test left join with empty df1 returns empty df with all columns."""
        df1 = pd.DataFrame({
            'col1': pd.Series([], dtype=float)
        }, index=pd.Index([], dtype=str))

        df2 = pd.DataFrame({
            'col2': [4, 5]
        }, index=['a', 'b'])

        result = dataframes.merge_on_index(df1, df2, how='left')

        assert len(result) == 0
        assert 'col1' in result.columns
        assert 'col2' in result.columns

    def test_merge_on_index_right_join_empty_df2(self):
        """Test right join with empty df2 returns empty df with all columns."""
        df1 = pd.DataFrame({
            'col1': [1, 2]
        }, index=['a', 'b'])

        df2 = pd.DataFrame({
            'col2': pd.Series([], dtype=float)
        }, index=pd.Index([], dtype=str))

        result = dataframes.merge_on_index(df1, df2, how='right')

        assert len(result) == 0
        assert 'col1' in result.columns
        assert 'col2' in result.columns

    def test_merge_on_index_inner_join_empty(self):
        """Test inner join with empty df returns empty df with all columns."""
        df1 = pd.DataFrame({
            'col1': [1, 2]
        }, index=['a', 'b'])

        df2 = pd.DataFrame({
            'col2': pd.Series([], dtype=float)
        }, index=pd.Index([], dtype=str))

        result = dataframes.merge_on_index(df1, df2, how='inner')

        assert len(result) == 0
        assert 'col1' in result.columns
        assert 'col2' in result.columns

    def test_concat_mixed_timezones(self):
        """Test concat handles mixed timezone-aware and timezone-naive 'ts' dataframes."""
        import warnings
        
        # Create dataframes with different timezone situations for 'ts' column
        df1 = pd.DataFrame({
            'ts': pd.date_range('2024-01-01', periods=3, tz='UTC'),
            'value': [1, 2, 3]
        })
        
        df2 = pd.DataFrame({
            'ts': pd.date_range('2024-01-04', periods=3),  # timezone-naive
            'value': [4, 5, 6]
        })
        
        df3 = pd.DataFrame({
            'ts': pd.date_range('2024-01-07', periods=3, tz='Etc/GMT+5'),  # different timezone
            'value': [7, 8, 9]
        })
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = dataframes.concat([df1, df2, df3], fast=True, quiet=True)
            
            # Should not produce timezone concatenation warnings for 'ts' column
            tz_warnings = [warning for warning in w if 'timezone' in str(warning.message).lower()]
            # May have warnings for other columns, but not for 'ts'
            for warning in tz_warnings:
                assert 'ts' not in str(warning.message), f"Unexpected timezone warning for 'ts': {warning.message}"
        
        # Verify 'ts' column is timezone-aware UTC
        assert result['ts'].dt.tz is not None
        assert str(result['ts'].dt.tz) == 'UTC'
        
        # Verify data integrity
        assert len(result) == 9
        assert list(result['value']) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    def test_ensure_utc_multiple_columns(self):
        """Test ensure_utc_ts can handle multiple datetime columns."""
        df = pd.DataFrame({
            'ts': pd.date_range('2024-01-01', periods=3),  # timezone-naive
            'date': pd.date_range('2024-01-01', periods=3, tz='Etc/GMT+5'),
            'created_at': pd.date_range('2024-01-01', periods=3, tz='Etc/GMT-1'),
            'value': [1, 2, 3]
        })
        
        # Convert multiple columns
        result = dataframes.ensure_utc_ts(df, columns=['ts', 'date', 'created_at'])
        
        # All specified columns should be UTC
        assert str(result['ts'].dt.tz) == 'UTC'
        assert str(result['date'].dt.tz) == 'UTC'
        assert str(result['created_at'].dt.tz) == 'UTC'


class TestDataTransformation:
    """Test data transformation functions."""
    
    def test_unstack_feature(self):
        """Test unstack_feature pivots data correctly."""
        ts = pd.date_range('2023-01-01', periods=4, freq='min')
        df = pd.DataFrame({
            'ts': ts.repeat(2),
            'symbol_venue': ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures'] * 4,
            'feature1': range(8)
        })
        df = df.set_index(['ts', 'symbol_venue'])
        
        with patch('lib.util.dataframes.logger'):
            result = dataframes.unstack_feature(df, 'feature1')
        
        assert result.shape == (4, 2)  # 4 timestamps, 2 symbols
        assert result.columns.names == [None, 'symbol_venue']
        assert result.index.name == 'ts'
    
    @patch('lib.util.dataframes.log_and_raise')
    def test_unstack_feature_duplicates(self, mock_log_and_raise):
        """Test unstack_feature raises on duplicate indices."""
        mock_log_and_raise.side_effect = Exception("Could not unstack")
        
        ts = pd.Timestamp('2023-01-01')
        df = pd.DataFrame({
            'ts': [ts, ts],
            'symbol_venue': ['BTCUSDT_binance-futures'] * 2,  # Duplicate index
            'feature1': [1, 2]
        })
        df = df.set_index(['ts', 'symbol_venue'])
        
        with pytest.raises(Exception, match="Could not unstack"):
            dataframes.unstack_feature(df, 'feature1')
    
    @patch('lib.util.dataframes.logger')
    def test_carry_forward(self, mock_logger):
        """Test carry_forward fills values correctly."""
        ts = pd.date_range('2023-01-01', periods=5, freq='min')
        df = pd.DataFrame({
            'ts': ts,
            'symbol_venue': ['BTCUSDT_binance-futures'] * 5,
            'col1': [1.0, 2.0, np.nan, np.nan, 5.0],
            'col2': [10.0, np.nan, np.nan, np.nan, np.nan]
        })
        df = df.set_index(['ts', 'symbol_venue'])
        
        result = dataframes.carry_forward(df, ['col1', 'col2'])
        
        # col1 should forward fill the NaN values
        assert result['col1'].iloc[2] == 2.0
        assert result['col1'].iloc[3] == 2.0
        
        # col2 should forward fill from 10.0
        assert result['col2'].iloc[1] == 10.0
        assert result['col2'].iloc[4] == 10.0
        
        # Should log carry forward info
        assert mock_logger.info.called
    
    def test_get_min_max_ts(self):
        """Test get_min_max_ts returns timestamp range."""
        ts = pd.date_range('2023-01-01 10:00', '2023-01-01 12:00', freq='30min', tz='UTC')
        df = pd.DataFrame({
            'value': range(len(ts))
        }, index=pd.MultiIndex.from_arrays([
            ts,
            ['BTCUSDT_binance-futures'] * len(ts)
        ], names=['ts', 'symbol_venue']))
        
        min_ts, max_ts = dataframes.get_min_max_ts(df)
        
        assert min_ts == pd.Timestamp('2023-01-01 10:00', tz='UTC')
        assert max_ts == pd.Timestamp('2023-01-01 12:00', tz='UTC')


class TestSpecializedFunctions:
    """Test specialized DataFrame functions."""
    
    def test_remove_1_min_cols(self):
        """Test remove_1_min_cols removes _1 suffix."""
        df = pd.DataFrame({
            'col1_1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3_1': [7, 8, 9]
        })
        
        result = dataframes.remove_1_min_cols(df)
        
        assert 'col1' in result.columns
        assert 'col1_1' not in result.columns
        assert 'col2' in result.columns
        assert 'col3' in result.columns
        assert 'col3_1' not in result.columns
    
    def test_remove_1_min_cols_with_conflicts(self):
        """Test remove_1_min_cols handles naming conflicts."""
        df = pd.DataFrame({
            'col1_1': [1, 2, 3],
            'col1': [4, 5, 6],  # Already exists
            'col2_1': [7, 8, 9]
        })
        
        with patch('lib.util.dataframes.logger') as mock_logger:
            result = dataframes.remove_1_min_cols(df)
            
            # Should not rename col1_1 due to conflict
            assert 'col1_1' in result.columns
            assert list(result['col1']) == [4, 5, 6]  # Original preserved
            
            # Should rename col2_1
            assert 'col2' in result.columns
            assert 'col2_1' not in result.columns
            
            # Should log warning
            mock_logger.warning.assert_called_with('col1 already in dataframe!')
    
    def test_rolling_agg_ignore_internal_nan(self):
        """Test rolling aggregation preserves trailing NaNs."""
        ts = pd.date_range('2023-01-01', periods=10, freq='min')
        df = pd.DataFrame({
            'col1': [1, 2, 3, np.nan, 5, 6, 7, np.nan, np.nan, np.nan]
        }, index=ts)
        
        result = dataframes.rolling_agg_ignore_internal_nan(df, window_mins=3, method='sum')
        
        # Should calculate rolling sum
        assert result['col1'].iloc[2] == 6.0  # 1 + 2 + 3
        
        # Should preserve trailing NaNs
        assert pd.isna(result['col1'].iloc[7])
        assert pd.isna(result['col1'].iloc[8])
        assert pd.isna(result['col1'].iloc[9])
        
        # Should have float32 dtype
        assert result['col1'].dtype == np.float32
    
    def test_get_trailing_nan_mask(self):
        """Test get_trailing_nan_mask identifies end NaNs."""
        df = pd.DataFrame({
            'col1': [1, np.nan, 3, np.nan, np.nan],  # Trailing NaNs at end
            'col2': [1, 2, 3, 4, 5],  # No trailing NaNs
            'col3': [np.nan, np.nan, np.nan, np.nan, np.nan]  # All NaN
        })
        
        mask = dataframes.get_trailing_nan_mask(df)
        
        # col1: only last 2 are trailing NaNs
        assert not mask['col1'].iloc[0]
        assert not mask['col1'].iloc[1]  # Internal NaN, not trailing
        assert not mask['col1'].iloc[2]
        assert mask['col1'].iloc[3]
        assert mask['col1'].iloc[4]
        
        # col2: no trailing NaNs
        assert not any(mask['col2'])
        
        # col3: all are trailing NaNs
        assert all(mask['col3'])


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch('pandas.read_parquet')
    def test_has_sufficient_valid_data_success(self, mock_read_parquet):
        """Test has_sufficient_valid_data with enough data."""
        mock_df = pd.DataFrame({
            'close_mid': [50000.0] * 1436 + [np.nan] * 4  # 1436 valid, 4 NaN
        })
        mock_read_parquet.return_value = mock_df
        
        result = dataframes.has_sufficient_valid_data('/test/file.parquet')
        
        assert result is True
        mock_read_parquet.assert_called_once_with('/test/file.parquet', columns=['close_mid'])
    
    @patch('pandas.read_parquet')
    @patch('lib.util.dataframes.logger')
    def test_has_sufficient_valid_data_failure(self, mock_logger, mock_read_parquet):
        """Test has_sufficient_valid_data with insufficient data."""
        mock_df = pd.DataFrame({
            'close_mid': [50000.0] * 1000 + [np.nan] * 440  # Only 1000 valid
        })
        mock_read_parquet.return_value = mock_df
        
        result = dataframes.has_sufficient_valid_data('/test/file.parquet')
        
        assert result is False
        # Check the call was made with the right message (handling numpy int64)
        call_args = mock_logger.info.call_args[0][0]
        assert 'Not enough valid data num_valid_bar_data=' in call_args
        assert '1000' in call_args
        assert 'at close_mid from /test/file.parquet' in call_args
    
    def test_get_last_timeslice(self):
        """Test get_last_timeslice returns latest timestamp data."""
        ts = pd.date_range('2023-01-01', periods=3, freq='h')
        df = pd.DataFrame({
            'value': range(9)
        }, index=pd.MultiIndex.from_product([
            ts,
            ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures', 'XRPUSDT_binance-futures']
        ], names=['ts', 'symbol_venue']))
        
        result = dataframes.get_last_timeslice(df)
        
        assert len(result) == 3  # 3 symbols at last timestamp
        assert all(result.index.get_level_values('ts') == ts[-1])
        assert list(result['value']) == [6, 7, 8]
    
    def test_convert_to_dict(self):
        """Test convert_to_dict creates key-value mapping."""
        df = pd.DataFrame({
            'symbol': ['BTCUSDT', 'ETHUSDT', 'XRPUSDT'],
            'price': [50000, 3000, 0.5]
        })
        
        result = dataframes.convert_to_dict(df, 'symbol', 'price')
        
        assert isinstance(result, dict)
        assert result['BTCUSDT'] == 50000
        assert result['ETHUSDT'] == 3000
        assert result['XRPUSDT'] == 0.5
    
    def test_convert_to_dict_from_index(self):
        """Test convert_to_dict handles index columns."""
        df = pd.DataFrame({
            'price': [50000, 3000, 0.5]
        }, index=['BTCUSDT', 'ETHUSDT', 'XRPUSDT'])
        df.index.name = 'symbol'
        
        result = dataframes.convert_to_dict(df, 'symbol', 'price')
        
        assert result['BTCUSDT'] == 50000
        assert result['ETHUSDT'] == 3000
    
    def test_clip_col_by_iqr(self):
        """Test clip_col_by_iqr removes outliers."""
        # Create data with outliers
        data = list(range(1, 10)) + [100, -50]  # 100 and -50 are outliers
        df = pd.DataFrame({'values': data})
        
        result = dataframes.clip_col_by_iqr(df, 'values')
        
        # Check that outliers are clipped
        assert result['values'].max() < 100
        assert result['values'].min() > -50
        
        # Check that normal values are preserved
        assert 5 in result['values'].values
    
    def test_get_first_column_values(self):
        """Test get_first_column_values adds first value columns."""
        ts = pd.date_range('2023-01-01', periods=4, freq='6h')
        df = pd.DataFrame({
            'ts': ts.repeat(2),
            'symbol_venue': ['BTCUSDT_binance-futures', 'ETHUSDT_binance-futures'] * 4,
            'date': [pd.Timestamp('2023-01-01').date()] * 4 + [pd.Timestamp('2023-01-02').date()] * 4,
            'open': [100, 200, 110, 210, 120, 220, 130, 230],
            'close': [105, 205, 115, 215, 125, 225, 135, 235]
        })
        
        result = dataframes.make_daily_carry_forward_values(df, ['open', 'close'])
        
        assert 'first_open' in result.columns
        assert 'first_close' in result.columns
        
        # Check first values for BTCUSDT on 2023-01-01
        btc_day1 = result[(result['symbol_venue'] == 'BTCUSDT_binance-futures') & 
                         (result['date'] == pd.Timestamp('2023-01-01').date())]
        assert all(btc_day1['first_open'] == 100)
        assert all(btc_day1['first_close'] == 105)


class TestDataFrameStyling:
    """Test DataFrame styling functions."""
    
    def test_df_style(self):
        """Test df_style returns bold style."""
        style = dataframes.df_style('any_value')
        assert style == "font-weight: bold"
    
    def test_highlight_df_rows(self):
        """Test highlight_df_rows applies styling."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        }, index=['a', 'b', 'c'])
        
        # This function modifies DataFrame styling in-place
        # Just verify it doesn't raise an error
        dataframes.highlight_df_rows(df, ['a', 'c'])
        
        # Note: We can't easily test the actual styling output
        # as it requires a Jupyter/HTML environment


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_check_df_column_changes(self):
        """Test check_df_column_changes logs new columns."""
        df1 = pd.DataFrame({'col1': [1], 'col2': [2]})
        df2 = pd.DataFrame({'col1': [1], 'col2': [2], 'col3': [3], 'col4': [4]})
        
        with patch('lib.util.dataframes.logger') as mock_logger:
            dataframes.check_df_column_changes(df1, df2)
            
            # Should log added columns
            assert mock_logger.info.called
            call_args = str(mock_logger.info.call_args)
            assert 'col3' in call_args
            assert 'col4' in call_args
    
    def test_empty_dataframe_handling(self):
        """Test functions handle empty DataFrames gracefully."""
        empty_df = pd.DataFrame()
        
        # shrink_floats
        result = dataframes.shrink_floats(empty_df)
        assert len(result) == 0
        
        # remove_infs
        result = dataframes.remove_infs(empty_df)
        assert len(result) == 0
        
        # flatten_cols
        result = dataframes.flatten_cols(empty_df)
        assert len(result) == 0


class TestEnsureUTCTimezone:
    """Test ensure_utc_ts function with various index types."""
    
    def test_single_datetime_index_naive(self):
        """Test with single DatetimeIndex that is timezone-naive."""
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03'], name='ts'))
        
        result = dataframes.ensure_utc_ts(df)
        assert result.index.tz is not None
        assert str(result.index.tz) == 'UTC'
    
    def test_single_datetime_index_with_tz(self):
        """Test with single DatetimeIndex that has non-UTC timezone."""
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03'], 
                                  name='ts').tz_localize('Etc/GMT+5'))
        
        result = dataframes.ensure_utc_ts(df)
        assert str(result.index.tz) == 'UTC'
    
    def test_single_datetime_index_already_utc(self):
        """Test with single DatetimeIndex already in UTC."""
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03'], 
                                  name='ts').tz_localize('UTC'))
        
        result = dataframes.ensure_utc_ts(df)
        assert str(result.index.tz) == 'UTC'
    
    def test_single_non_datetime_index_named_ts(self):
        """Test with non-DatetimeIndex that happens to be named 'ts'."""
        # This is the bug case - a regular Index named 'ts' that can't be converted to datetime
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=pd.Index(['a', 'b', 'c'], name='ts'))
        
        # Should raise ValueError when trying to convert non-datetime strings
        with pytest.raises(ValueError, match="Given date string"):
            dataframes.ensure_utc_ts(df)
    
    def test_multiindex_with_datetime_ts(self):
        """Test with MultiIndex containing DatetimeIndex ts level."""
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=pd.MultiIndex.from_arrays([
            pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03']),
            ['A', 'B', 'C']
        ], names=['ts', 'symbol']))
        
        result = dataframes.ensure_utc_ts(df)
        ts_level = result.index.get_level_values('ts')
        assert ts_level.tz is not None
        assert str(ts_level.tz) == 'UTC'
    
    def test_multiindex_with_non_datetime_ts(self):
        """Test with MultiIndex where ts level is not DatetimeIndex."""
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=pd.MultiIndex.from_arrays([
            ['2023-01-01', '2023-01-02', '2023-01-03'],  # String, not datetime
            ['A', 'B', 'C']
        ], names=['ts', 'symbol']))
        
        # Should convert string dates to UTC DatetimeIndex
        result = dataframes.ensure_utc_ts(df)
        ts_level = result.index.get_level_values('ts')
        assert isinstance(ts_level, pd.DatetimeIndex)
        assert str(ts_level.tz) == 'UTC'
    
    def test_ts_as_column_naive(self):
        """Test with ts as a column that is timezone-naive."""
        df = pd.DataFrame({
            'ts': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'value': [1, 2, 3]
        })
        
        result = dataframes.ensure_utc_ts(df)
        assert result['ts'].dt.tz is not None
        assert str(result['ts'].dt.tz) == 'UTC'
    
    def test_ts_as_column_with_tz(self):
        """Test with ts as a column with non-UTC timezone."""
        df = pd.DataFrame({
            'ts': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']).tz_localize('Etc/GMT+5'),
            'value': [1, 2, 3]
        })
        
        result = dataframes.ensure_utc_ts(df)
        assert str(result['ts'].dt.tz) == 'UTC'
    
    def test_no_ts_column_or_index(self):
        """Test with DataFrame that has no ts column or index."""
        df = pd.DataFrame({
            'value': [1, 2, 3],
            'other': [4, 5, 6]
        })
        
        result = dataframes.ensure_utc_ts(df)
        assert result.equals(df)  # Should be unchanged
    
    def test_ts_as_non_datetime_column(self):
        """Test with ts as a non-datetime column (e.g., string)."""
        df = pd.DataFrame({
            'ts': ['2023-01-01', '2023-01-02', '2023-01-03'],  # String, not datetime
            'value': [1, 2, 3]
        })
        
        # Should convert to datetime and make timezone-aware
        result = dataframes.ensure_utc_ts(df)
        assert pd.api.types.is_datetime64_any_dtype(result['ts'])
        assert str(result['ts'].dt.tz) == 'UTC'
        # Verify the conversion worked correctly
        assert result['ts'].iloc[0] == pd.Timestamp('2023-01-01', tz='UTC')
        assert result['ts'].iloc[1] == pd.Timestamp('2023-01-02', tz='UTC')
        assert result['ts'].iloc[2] == pd.Timestamp('2023-01-03', tz='UTC')
    
    def test_non_datetime_single_index(self):
        """Test with non-datetime single index named 'ts'."""
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=pd.Index(['2023-01-01', '2023-01-02', '2023-01-03'], name='ts'))
        
        # Should convert index to datetime and make timezone-aware
        result = dataframes.ensure_utc_ts(df)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert str(result.index.tz) == 'UTC'
        assert result.index[0] == pd.Timestamp('2023-01-01', tz='UTC')
        assert result.index[1] == pd.Timestamp('2023-01-02', tz='UTC')
        assert result.index[2] == pd.Timestamp('2023-01-03', tz='UTC')
    
    def test_non_datetime_multiindex(self):
        """Test with non-datetime ts level in MultiIndex."""
        idx = pd.MultiIndex.from_arrays([
            ['2023-01-01', '2023-01-02', '2023-01-03'],  # String timestamps
            ['A', 'B', 'C']
        ], names=['ts', 'symbol'])
        
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=idx)
        
        # Should convert ts level to datetime and make timezone-aware
        result = dataframes.ensure_utc_ts(df)
        ts_level = result.index.get_level_values('ts')
        assert isinstance(ts_level, pd.DatetimeIndex)
        assert str(ts_level.tz) == 'UTC'
        assert ts_level[0] == pd.Timestamp('2023-01-01', tz='UTC')
        assert ts_level[1] == pd.Timestamp('2023-01-02', tz='UTC')
        assert ts_level[2] == pd.Timestamp('2023-01-03', tz='UTC')


class TestGetMinMaxTs:
    """Test get_min_max_ts function with various timezone scenarios."""
    
    def test_all_timezone_aware_utc(self):
        """Test with all timestamps timezone-aware in UTC."""
        ts = pd.date_range('2023-01-01', '2023-01-03', tz='UTC')
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=pd.MultiIndex.from_arrays([
            ts,
            ['A', 'B', 'C']
        ], names=['ts', 'symbol']))
        
        min_ts, max_ts = dataframes.get_min_max_ts(df)
        # Should remain in UTC
        assert str(min_ts.tz) == 'UTC'
        assert str(max_ts.tz) == 'UTC'
        assert min_ts == pd.Timestamp('2023-01-01', tz='UTC')
        assert max_ts == pd.Timestamp('2023-01-03', tz='UTC')
    
    def test_all_timezone_aware_non_utc(self):
        """Test with all timestamps timezone-aware in non-UTC timezone."""
        ts = pd.date_range('2023-01-01', '2023-01-03', tz='Etc/GMT+5')
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=pd.MultiIndex.from_arrays([
            ts,
            ['A', 'B', 'C']
        ], names=['ts', 'symbol']))
        
        min_ts, max_ts = dataframes.get_min_max_ts(df)
        # Should be converted to UTC
        assert str(min_ts.tz) == 'UTC'
        assert str(max_ts.tz) == 'UTC'
    
    def test_all_timezone_naive(self):
        """Test with all timestamps timezone-naive."""
        ts = pd.date_range('2023-01-01', '2023-01-03')
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=pd.MultiIndex.from_arrays([
            ts,
            ['A', 'B', 'C']
        ], names=['ts', 'symbol']))
        
        min_ts, max_ts = dataframes.get_min_max_ts(df)
        # Should be localized to UTC
        assert str(min_ts.tz) == 'UTC'
        assert str(max_ts.tz) == 'UTC'
        assert min_ts == pd.Timestamp('2023-01-01', tz='UTC')
        assert max_ts == pd.Timestamp('2023-01-03', tz='UTC')
    
    def test_mixed_timezone_aware_naive(self):
        """Test with mixed timezone-aware and timezone-naive timestamps."""
        # Create mixed timestamps - some with UTC, some naive
        ts1 = pd.Timestamp('2023-01-01').tz_localize('UTC')
        ts2 = pd.Timestamp('2023-01-02')  # Naive
        ts3 = pd.Timestamp('2023-01-03').tz_localize('UTC')
        
        # Create index manually to have mixed timezone awareness
        idx = pd.Index([ts1, ts2, ts3], name='ts')
        
        df = pd.DataFrame({
            'value': [1, 2, 3],
            'symbol': ['A', 'B', 'C']
        }, index=idx)
        df = df.set_index('symbol', append=True)
        
        # This should handle the mixed timezone case and always return UTC
        min_ts, max_ts = dataframes.get_min_max_ts(df)
        assert str(min_ts.tz) == 'UTC'
        assert str(max_ts.tz) == 'UTC'
        assert min_ts == pd.Timestamp('2023-01-01', tz='UTC')
        assert max_ts == pd.Timestamp('2023-01-03', tz='UTC')
    
    def test_non_datetime_index(self):
        """Test with non-DatetimeIndex ts level."""
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=pd.MultiIndex.from_arrays([
            ['2023-01-01', '2023-01-02', '2023-01-03'],  # Strings
            ['A', 'B', 'C']
        ], names=['ts', 'symbol']))
        
        # Should convert strings to UTC timestamps
        min_ts, max_ts = dataframes.get_min_max_ts(df)
        assert min_ts == pd.Timestamp('2023-01-01', tz='UTC')
        assert max_ts == pd.Timestamp('2023-01-03', tz='UTC')
