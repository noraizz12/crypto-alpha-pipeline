"""Unit tests for the files module.

This module tests file system utilities including directory management,
file operations, and data loading functions.
"""
import os
import json
from datetime import datetime as dt
from unittest.mock import Mock, patch
import logging

import pandas as pd
import pytest

from lib.util.files import (
    safe_mkdir, make_sim_dir, get_user, find_latest_file_date,
    write_df_to_csv, get_file_created_dt, load_jsonl, load_dict_file
)


class TestSafeMkdir:
    """Test cases for the safe_mkdir function."""
    
    def test_create_new_directory(self, tmp_path):
        """Test creating a new directory."""
        test_dir = tmp_path / "test_new_dir"
        assert not test_dir.exists()
        
        safe_mkdir(str(test_dir))
        
        assert test_dir.exists()
        assert test_dir.is_dir()
    
    def test_existing_directory(self, tmp_path):
        """Test that existing directory doesn't raise error."""
        test_dir = tmp_path / "test_existing_dir"
        test_dir.mkdir()
        assert test_dir.exists()
        
        # Should not raise
        safe_mkdir(str(test_dir))
        
        assert test_dir.exists()
        assert test_dir.is_dir()
    
    def test_nested_directories(self, tmp_path):
        """Test creating nested directories."""
        test_dir = tmp_path / "level1" / "level2" / "level3"
        assert not test_dir.exists()
        
        safe_mkdir(str(test_dir))
        
        assert test_dir.exists()
        assert test_dir.is_dir()
    
    def test_invalid_path(self):
        """Test that invalid paths still raise errors."""
        with pytest.raises(OSError):
            safe_mkdir("/dev/null/invalid/path")


class TestMakeSimDir:
    """Test cases for the make_sim_dir function."""
    
    @patch('lib.util.files.SIM_DIR', '/tmp/test_sims')
    def test_create_sim_dir_with_name(self, tmp_path):
        """Test creating simulation directory with specific name."""
        # Use tmp_path as SIM_DIR
        with patch('lib.util.files.SIM_DIR', str(tmp_path)):
            sim_name = "test_sim_2024"
            sim_dir = make_sim_dir(sim_name)
            
            assert sim_dir == str(tmp_path / sim_name)
            assert os.path.exists(sim_dir)
            assert os.path.isdir(sim_dir)
    
    @patch('lib.util.files.SIM_DIR', '/tmp/test_sims')
    def test_create_sim_dir_default_name(self, tmp_path):
        """Test creating simulation directory with timestamp name."""
        with patch('lib.util.files.SIM_DIR', str(tmp_path)):
            with patch('lib.util.files.dt') as mock_dt:
                mock_now = Mock()
                mock_now.strftime.return_value = "20240115_103045"
                mock_dt.now.return_value = mock_now
                
                sim_dir = make_sim_dir()
                
                assert sim_dir == str(tmp_path / "20240115_103045")
    
    @patch('lib.util.files.SIM_DIR', '/tmp/test_sims')
    def test_remove_existing_directory(self, tmp_path, caplog):
        """Test that existing directory is removed."""
        with patch('lib.util.files.SIM_DIR', str(tmp_path)):
            sim_name = "existing_sim"
            existing_dir = tmp_path / sim_name
            existing_dir.mkdir()
            
            # Create a file in the directory
            (existing_dir / "test.txt").write_text("test")
            
            with caplog.at_level(logging.WARNING):
                sim_dir = make_sim_dir(sim_name)
            
            assert os.path.exists(sim_dir)
            assert not (existing_dir / "test.txt").exists()
    
    def test_empty_name_raises(self):
        """Test that empty sim name raises error."""
        with pytest.raises(RuntimeError, match="not removing sim dir"):
            make_sim_dir("")
    
    @patch('lib.util.files.SIM_DIR', '/tmp/test_sims')
    def test_failed_removal_continues(self, tmp_path, caplog):
        """Test that failed directory removal doesn't stop creation."""
        with patch('lib.util.files.SIM_DIR', str(tmp_path)):
            with patch('shutil.rmtree') as mock_rmtree:
                mock_rmtree.side_effect = Exception("Permission denied")
                
                sim_name = "test_sim"
                with caplog.at_level(logging.WARNING):
                    sim_dir = make_sim_dir(sim_name)
                
                assert "Could not remove" in caplog.text
                assert os.path.exists(sim_dir)


class TestGetUser:
    """Test cases for the get_user function."""
    
    def test_get_user_standard_path(self):
        """Test extracting user from standard Unix path."""
        with patch('lib.util.files.ROOT_DIR', '/home/trader/projects/stat_arb'):
            user = get_user()
            assert user == "home"  # Index 1 after split is "home"
    
    def test_get_user_different_path(self):
        """Test extracting user from different path structure."""
        with patch('lib.util.files.ROOT_DIR', '/users/john/workspace'):
            user = get_user()
            assert user == "users"  # Index 1 after split is "users"
    
    def test_get_user_root_path(self):
        """Test behavior with root path."""
        with patch('lib.util.files.ROOT_DIR', '/root/project'):
            user = get_user()
            assert user == "root"  # Index 1 after split is "root"


class TestFindLatestFileDate:
    """Test cases for the find_latest_file_date function."""
    
    def test_find_latest_with_files(self, tmp_path, caplog):
        """Test finding latest file date from multiple files."""
        # Create test files with dates in filenames
        files = [
            tmp_path / "data_20240101.parquet",
            tmp_path / "data_20240105.parquet",
            tmp_path / "data_20240103.parquet"
        ]
        
        for f in files:
            f.touch()
        
        with caplog.at_level(logging.INFO):
            pattern = str(tmp_path / "data_*.parquet")
            result = find_latest_file_date(pattern)
        
        assert result == dt(2024, 1, 5)
        assert "Latest file time is 2024-01-05" in caplog.text
    
    def test_find_latest_no_files(self, tmp_path, caplog):
        """Test behavior when no files match pattern."""
        with caplog.at_level(logging.WARNING):
            pattern = str(tmp_path / "nonexistent_*.parquet")
            result = find_latest_file_date(pattern)
        
        assert result is None
        assert "No files found like" in caplog.text
    
    def test_find_latest_single_file(self, tmp_path):
        """Test with single matching file."""
        test_file = tmp_path / "report_20240215.csv"
        test_file.touch()
        
        pattern = str(tmp_path / "report_*.csv")
        result = find_latest_file_date(pattern)
        
        assert result == dt(2024, 2, 15)
    
    def test_find_latest_different_extensions(self, tmp_path):
        """Test that extension doesn't affect date extraction."""
        files = [
            tmp_path / "data_20240110.csv",
            tmp_path / "data_20240111.parquet",
            tmp_path / "data_20240109.json"
        ]
        
        for f in files:
            f.touch()
        
        # Test each pattern separately
        result_csv = find_latest_file_date(str(tmp_path / "data_*.csv"))
        assert result_csv == dt(2024, 1, 10)
        
        result_parquet = find_latest_file_date(str(tmp_path / "data_*.parquet"))
        assert result_parquet == dt(2024, 1, 11)


class TestWriteDfToCsv:
    """Test cases for the write_df_to_csv function."""
    
    def test_write_basic_dataframe(self, tmp_path, caplog):
        """Test writing a basic DataFrame to CSV."""
        df = pd.DataFrame({
            'symbol': ['BTC', 'ETH', 'SOL'],
            'price': [50000, 3000, 100]
        })
        
        filename = str(tmp_path / "test_data")
        
        with caplog.at_level(logging.INFO):
            write_df_to_csv(df, filename)
        
        # Check file was created with .csv extension
        csv_file = tmp_path / "test_data.csv"
        assert csv_file.exists()
        assert f"Writing {filename}.csv" in caplog.text
        
        # Read back and verify
        read_df = pd.read_csv(csv_file)
        pd.testing.assert_frame_equal(read_df, df)
    
    def test_write_none_dataframe(self, tmp_path, caplog):
        """Test that None DataFrame logs warning and doesn't create file."""
        filename = str(tmp_path / "none_data")
        
        with caplog.at_level(logging.WARNING):
            write_df_to_csv(None, filename)
        
        assert "Not writing empty dataframe" in caplog.text
        assert not (tmp_path / "none_data.csv").exists()
    
    def test_write_append_mode(self, tmp_path):
        """Test appending to existing CSV file."""
        df1 = pd.DataFrame({'col': [1, 2, 3]})
        df2 = pd.DataFrame({'col': [4, 5, 6]})
        
        filename = str(tmp_path / "append_test")
        
        # Write first DataFrame
        write_df_to_csv(df1, filename, append=False)
        
        # Append second DataFrame
        write_df_to_csv(df2, filename, append=True)
        
        # Read back and verify
        result = pd.read_csv(tmp_path / "append_test.csv")
        expected = pd.DataFrame({'col': [1, 2, 3, 4, 5, 6]})
        pd.testing.assert_frame_equal(result, expected)
    
    def test_write_append_to_new_file(self, tmp_path):
        """Test append mode creates new file if doesn't exist."""
        df = pd.DataFrame({'data': [1, 2, 3]})
        filename = str(tmp_path / "new_append")
        
        write_df_to_csv(df, filename, append=True)
        
        csv_file = tmp_path / "new_append.csv"
        assert csv_file.exists()
        
        # Should have header since it's a new file
        content = csv_file.read_text()
        assert "data" in content
    
    @patch('lib.util.files.LOCAL', True)
    def test_permission_error_local_mode(self, tmp_path, capsys):
        """Test that permission errors print DataFrame in LOCAL mode."""
        df = pd.DataFrame({'test': [1, 2, 3]})
        
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            mock_to_csv.side_effect = PermissionError("Access denied")
            
            # Should not raise in LOCAL mode
            write_df_to_csv(df, "test_file")
            
            # Should print DataFrame instead
            captured = capsys.readouterr()
            assert "test" in captured.out
            assert "1" in captured.out
    
    @patch('lib.util.files.LOCAL', False)
    def test_permission_error_production_mode(self):
        """Test that permission errors raise in production mode."""
        df = pd.DataFrame({'test': [1, 2, 3]})
        
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            mock_to_csv.side_effect = PermissionError("Access denied")
            
            with pytest.raises(PermissionError):
                write_df_to_csv(df, "test_file")


class TestGetFileCreatedDt:
    """Test cases for the get_file_created_dt function."""
    
    def test_get_created_time_existing_file(self, tmp_path):
        """Test getting creation time of existing file."""
        test_file = tmp_path / "test.txt"
        test_file.touch()
        
        # Mock getctime to return a known timestamp
        mock_timestamp = 1705325400.123  # 2024-01-15 10:30:00.123
        with patch('os.path.getctime', return_value=mock_timestamp):
            result = get_file_created_dt(str(test_file))
        
        assert result is not None
        # millis_to_dt expects milliseconds, so multiply by 1000
        assert isinstance(result, dt)
    
    def test_get_created_time_nonexistent_file(self, caplog):
        """Test behavior with non-existent file."""
        with caplog.at_level(logging.ERROR):
            result = get_file_created_dt("/nonexistent/file.txt")
        
        assert result is None
        assert "Could not get file time" in caplog.text
    
    def test_get_created_time_permission_error(self, tmp_path, caplog):
        """Test behavior with permission errors."""
        test_file = tmp_path / "test.txt"
        test_file.touch()
        
        with patch('os.path.getctime', side_effect=PermissionError("Access denied")):
            with caplog.at_level(logging.ERROR):
                result = get_file_created_dt(str(test_file))
        
        assert result is None
        assert "Could not get file time" in caplog.text


class TestLoadJsonl:
    """Test cases for the load_jsonl function."""
    
    def test_load_valid_jsonl(self, tmp_path):
        """Test loading valid JSONL file."""
        jsonl_file = tmp_path / "test.jsonl"
        
        # Create test JSONL content
        data = [
            {"id": 1, "name": "Alice", "score": 95},
            {"id": 2, "name": "Bob", "score": 87},
            {"id": 3, "name": "Charlie", "score": 92}
        ]
        
        with open(jsonl_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        # Load the file
        df = load_jsonl(str(jsonl_file))
        
        assert len(df) == 3
        assert list(df.columns) == ['id', 'name', 'score']
        assert df['name'].tolist() == ['Alice', 'Bob', 'Charlie']
    
    def test_load_jsonl_with_columns(self, tmp_path):
        """Test loading JSONL with specific columns."""
        jsonl_file = tmp_path / "test.jsonl"
        
        data = [
            {"id": 1, "name": "Alice", "score": 95, "grade": "A"},
            {"id": 2, "name": "Bob", "score": 87, "grade": "B"}
        ]
        
        with open(jsonl_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        # Load only specific columns
        df = load_jsonl(str(jsonl_file), columns=['id', 'score'])
        
        assert len(df) == 2
        assert set(df.columns) == {'id', 'score'}  # Use set to ignore order
        assert 'name' not in df.columns
        assert 'grade' not in df.columns
    
    def test_load_jsonl_missing_columns(self, tmp_path, capsys):
        """Test warning when requested columns don't exist."""
        jsonl_file = tmp_path / "test.jsonl"
        
        data = [{"id": 1, "value": 100}]
        
        with open(jsonl_file, 'w') as f:
            f.write(json.dumps(data[0]) + '\n')
        
        # Request non-existent columns
        df = load_jsonl(str(jsonl_file), columns=['id', 'missing_col'])
        
        captured = capsys.readouterr()
        assert "Warning: Columns not found: {'missing_col'}" in captured.out
        assert list(df.columns) == ['id']
    
    def test_load_jsonl_with_limit(self, tmp_path):
        """Test loading limited number of lines."""
        jsonl_file = tmp_path / "test.jsonl"
        
        # Create file with 10 lines
        with open(jsonl_file, 'w') as f:
            for i in range(10):
                f.write(json.dumps({"index": i}) + '\n')
        
        # Load only first 3 lines
        df = load_jsonl(str(jsonl_file), lines_to_read=3)
        
        assert len(df) == 3
        assert df['index'].tolist() == [0, 1, 2]
    
    def test_load_jsonl_invalid_json(self, tmp_path, capsys):
        """Test handling of invalid JSON lines."""
        jsonl_file = tmp_path / "test.jsonl"
        
        with open(jsonl_file, 'w') as f:
            f.write('{"valid": 1}\n')
            f.write('invalid json\n')  # Invalid
            f.write('{"valid": 2}\n')
        
        df = load_jsonl(str(jsonl_file))
        
        captured = capsys.readouterr()
        assert "Warning: Invalid JSON on line 2" in captured.out
        assert len(df) == 2
        assert df['valid'].tolist() == [1, 2]
    
    def test_load_jsonl_empty_file(self, tmp_path):
        """Test loading empty JSONL file."""
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.touch()
        
        with pytest.raises(Exception, match="Error loading file: No valid JSON objects found"):
            load_jsonl(str(jsonl_file))
    
    def test_load_jsonl_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_jsonl("/nonexistent/file.jsonl")
    
    def test_load_jsonl_encoding(self, tmp_path):
        """Test loading with different encoding."""
        jsonl_file = tmp_path / "test.jsonl"
        
        # Write with UTF-8 encoding
        data = {"text": "café"}
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data) + '\n')
        
        # Load with UTF-8 encoding
        df = load_jsonl(str(jsonl_file), encoding='utf-8')
        
        assert df['text'].iloc[0] == "café"


class TestLoadDictFile:
    """Test cases for the load_dict_file function."""
    
    def test_load_valid_dict_file(self, tmp_path):
        """Test loading valid dictionary file."""
        dict_file = tmp_path / "test.txt"
        
        # Create test dictionary content
        with open(dict_file, 'w') as f:
            f.write("{'id': 1, 'name': 'Alice', 'active': True}\n")
            f.write("{'id': 2, 'name': 'Bob', 'active': False}\n")
            f.write("{'id': 3, 'name': 'Charlie', 'active': True}\n")
        
        df = load_dict_file(str(dict_file))
        
        assert len(df) == 3
        assert list(df.columns) == ['id', 'name', 'active']
        assert df['name'].tolist() == ['Alice', 'Bob', 'Charlie']
    
    def test_load_dict_with_columns(self, tmp_path):
        """Test loading dictionary file with specific columns."""
        dict_file = tmp_path / "test.txt"
        
        with open(dict_file, 'w') as f:
            f.write("{'id': 1, 'value': 100, 'extra': 'data'}\n")
            f.write("{'id': 2, 'value': 200, 'extra': 'more'}\n")
        
        df = load_dict_file(str(dict_file), columns=['id', 'value'])
        
        assert len(df) == 2
        assert set(df.columns) == {'id', 'value'}  # Use set to ignore order
        assert 'extra' not in df.columns
    
    def test_load_dict_invalid_lines(self, tmp_path, capsys):
        """Test handling of invalid dictionary lines."""
        dict_file = tmp_path / "test.txt"
        
        with open(dict_file, 'w') as f:
            f.write("{'valid': 1}\n")
            f.write("not a dict\n")  # Invalid
            f.write("[1, 2, 3]\n")    # Valid Python but not a dict
            f.write("{'valid': 2}\n")
        
        df = load_dict_file(str(dict_file))
        
        captured = capsys.readouterr()
        assert "Warning: Invalid dictionary on line 2" in captured.out
        assert "Warning: Line 3 is not a dictionary" in captured.out
        assert len(df) == 2
        assert df['valid'].tolist() == [1, 2]
    
    def test_load_dict_with_limit(self, tmp_path):
        """Test loading limited number of lines."""
        dict_file = tmp_path / "test.txt"
        
        with open(dict_file, 'w') as f:
            for i in range(10):
                f.write(f"{{'index': {i}}}\n")
        
        df = load_dict_file(str(dict_file), lines_to_read=5)
        
        assert len(df) == 5
        assert df['index'].tolist() == [0, 1, 2, 3, 4]
    
    def test_load_dict_empty_file(self, tmp_path):
        """Test loading empty dictionary file."""
        dict_file = tmp_path / "empty.txt"
        dict_file.touch()
        
        with pytest.raises(ValueError, match="No valid dictionaries found"):
            load_dict_file(str(dict_file))
    
    def test_load_dict_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_dict_file("/nonexistent/file.txt")
    
    def test_load_dict_complex_types(self, tmp_path):
        """Test loading dictionaries with various Python types."""
        dict_file = tmp_path / "test.txt"
        
        with open(dict_file, 'w') as f:
            f.write("{'int': 42, 'float': 3.14, 'str': 'hello', 'bool': True, 'none': None}\n")
            f.write("{'list': [1, 2, 3], 'nested': {'a': 1, 'b': 2}}\n")
        
        df = load_dict_file(str(dict_file))
        
        assert len(df) == 2
        assert df.iloc[0]['int'] == 42
        assert df.iloc[0]['float'] == 3.14
        assert df.iloc[0]['str'] == 'hello'
        assert df.iloc[0]['bool'] is True
        assert pd.isna(df.iloc[0]['none'])
        assert df.iloc[1]['list'] == [1, 2, 3]
        assert df.iloc[1]['nested'] == {'a': 1, 'b': 2}
    
    def test_load_dict_malicious_code(self, tmp_path):
        """Test that malicious code is not executed."""
        dict_file = tmp_path / "test.txt"
        
        with open(dict_file, 'w') as f:
            # This would be dangerous with eval() but safe with literal_eval()
            f.write("{'data': __import__('os').system('echo hacked')}\n")
        
        # Should raise ValueError, not execute code
        with pytest.raises(ValueError):
            load_dict_file(str(dict_file))
