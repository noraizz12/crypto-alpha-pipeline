"""Test the to_datetime wrapper function."""

import unittest
import pandas as pd
import logging
from datetime import datetime as dt, timezone

from lib.util.time_util import to_datetime


class TestToDatetimeWrapper(unittest.TestCase):
    """Test the to_datetime wrapper function."""
    
    def test_single_string_conversion(self):
        """Test converting a single date string."""
        result = to_datetime('2024-01-01')
        self.assertIsInstance(result, pd.Timestamp)
        self.assertEqual(result.tz, timezone.utc)
        self.assertEqual(str(result), '2024-01-01 00:00:00+00:00')
    
    def test_list_conversion(self):
        """Test converting a list of date strings."""
        result = to_datetime(['2024-01-01', '2024-01-02'])
        self.assertIsInstance(result, pd.DatetimeIndex)
        self.assertEqual(result.tz, timezone.utc)
        self.assertEqual(len(result), 2)
    
    def test_series_conversion(self):
        """Test converting a pandas Series."""
        series = pd.Series(['2024-01-01', '2024-01-02'])
        result = to_datetime(series)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result.dt.tz, timezone.utc)
    
    def test_format_parameter(self):
        """Test passing format parameter."""
        result = to_datetime('20240101', format='%Y%m%d')
        self.assertEqual(result.tz, timezone.utc)
        self.assertEqual(str(result), '2024-01-01 00:00:00+00:00')
    
    def test_utc_false_warning(self):
        """Test that utc=False generates warning but still uses UTC."""
        with self.assertLogs('lib.util.time_util', level=logging.WARNING) as cm:
            result = to_datetime('2024-01-01', utc=False)
        
        # Check warning was logged
        self.assertIn('utc=False', cm.output[0])
        self.assertIn('overriding to utc=True', cm.output[0])
        
        # Check result is still UTC
        self.assertEqual(result.tz, timezone.utc)
    
    def test_unit_parameter(self):
        """Test unit parameter for timestamp conversion."""
        timestamp = 1704067200  # 2024-01-01 00:00:00 UTC
        result = to_datetime(timestamp, unit='s')
        self.assertEqual(result.tz, timezone.utc)
        self.assertEqual(result.year, 2024)
        self.assertEqual(result.month, 1)
        self.assertEqual(result.day, 1)
    
    def test_errors_parameter(self):
        """Test errors parameter."""
        # Test coerce
        result = to_datetime(['2024-01-01', 'invalid'], errors='coerce')
        self.assertEqual(result[0].tz, timezone.utc)
        self.assertTrue(pd.isna(result[1]))
        
        # Test raise (default)
        with self.assertRaises(ValueError):
            to_datetime('invalid date')
    
    def test_dayfirst_parameter(self):
        """Test dayfirst parameter."""
        # European date format
        result = to_datetime('01/02/2024', dayfirst=True)
        self.assertEqual(result.month, 2)  # February
        self.assertEqual(result.day, 1)
        self.assertEqual(result.tz, timezone.utc)
    
    def test_cache_parameter(self):
        """Test cache parameter passes through."""
        # Should work the same with cache=True or cache=False
        result1 = to_datetime(['2024-01-01'] * 100, cache=True)
        result2 = to_datetime(['2024-01-01'] * 100, cache=False)
        self.assertTrue((result1 == result2).all())
        self.assertEqual(result1.tz, timezone.utc)


if __name__ == '__main__':
    unittest.main()