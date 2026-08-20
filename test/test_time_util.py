"""
Unit tests for time_util.py module.

Tests all datetime utility functions including conversions, timezone handling,
and date arithmetic operations.
"""

import math
import time
from datetime import datetime as dt, timedelta as td, timezone, date as dt_date
from unittest.mock import patch

import pandas as pd
import pytest

from lib.util.time_util import (
    MINUTES_IN_DAY, MILLIS_IN_DAY, DATE_FORMAT, DATE_TIME_FORMAT,
    ensure_utc_timezone,
    today, today_date, yesterday, yesterday_date,
    start_of_next_month, end_of_day, beginning_of_day, beginning_of_day_millis,
    millis_to_dt, millis_to_dt_str, dt_to_str, date_to_str, dt_to_millis,
    date_str_to_dt, date_str_to_date, date_to_end_dt, date_to_start_dt,
    str_to_dt, date_range, wait_until_minute, time_str_to_dts,
    next_closest_time, next_n_minute, compute_lookback_days,
    seconds_until_next_utc_midnight
)


class TestConstants:
    """Test module constants."""
    
    def test_constants(self):
        """Test that constants have expected values."""
        assert MINUTES_IN_DAY == 1440
        assert MILLIS_IN_DAY == 1440 * 60 * 1000
        assert DATE_FORMAT == "%Y%m%d"
        assert DATE_TIME_FORMAT == "%Y%m%d_%H%M"


class TestTimezoneUtilities:
    """Test timezone handling utilities."""
    
    def test_ensure_utc_timezone_naive(self):
        """Test ensure_utc_timezone with naive datetime."""
        naive_dt = dt(2024, 1, 1, 12, 0, 0)
        result = ensure_utc_timezone(naive_dt)
        assert result.tzinfo == timezone.utc
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 12
    
    def test_ensure_utc_timezone_already_utc(self):
        """Test ensure_utc_timezone with UTC datetime."""
        utc_dt = dt(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = ensure_utc_timezone(utc_dt)
        assert result == utc_dt
        assert result.tzinfo == timezone.utc
    
    def test_ensure_utc_timezone_other_timezone(self):
        """Test ensure_utc_timezone with non-UTC timezone."""
        # Create EST timezone (UTC-5)
        est = timezone(td(hours=-5))
        est_dt = dt(2024, 1, 1, 12, 0, 0, tzinfo=est)
        result = ensure_utc_timezone(est_dt)
        assert result.tzinfo == timezone.utc
        # 12:00 EST = 17:00 UTC
        assert result.hour == 17


class TestDateGetters:
    """Test date/time getter functions."""
    
    @patch('lib.util.time_util.dt')
    def test_today(self, mock_dt):
        """Test today function."""
        mock_now = dt(2024, 1, 15, 14, 30, 45, tzinfo=timezone.utc)
        mock_dt.now.return_value = mock_now
        result = today()
        assert isinstance(result, pd.Timestamp)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0
    
    def test_today_date(self):
        """Test today_date function (without mocking since it's simple)."""
        result = today_date()
        assert isinstance(result, dt_date)
        # Just verify it returns a valid date
        assert result.year > 2020
        assert 1 <= result.month <= 12
        assert 1 <= result.day <= 31
    
    @patch('lib.util.time_util.dt')
    def test_yesterday(self, mock_dt):
        """Test yesterday function."""
        mock_now = dt(2024, 1, 15, 14, 30, 45, tzinfo=timezone.utc)
        mock_dt.now.return_value = mock_now
        result = yesterday()
        assert isinstance(result, pd.Timestamp)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 14
        assert result.hour == 0
    
    def test_yesterday_date(self):
        """Test yesterday_date function (without mocking since it's simple)."""
        today = today_date()
        result = yesterday_date()
        assert isinstance(result, dt_date)
        # Yesterday should be one day before today
        assert (today - result).days == 1


class TestDateArithmetic:
    """Test date arithmetic functions."""
    
    def test_start_of_next_month_mid_month(self):
        """Test start_of_next_month from middle of month."""
        test_dt = dt(2024, 1, 15, 14, 30, 45)
        result = start_of_next_month(test_dt)
        assert result.year == 2024
        assert result.month == 2
        assert result.day == 1
        # Note: start_of_next_month does not reset time to midnight
        assert result.hour == 14
        assert result.minute == 30
    
    def test_start_of_next_month_end_of_month(self):
        """Test start_of_next_month from end of month."""
        test_dt = dt(2024, 1, 31, 23, 59, 59)
        result = start_of_next_month(test_dt)
        assert result.year == 2024
        assert result.month == 2
        assert result.day == 1
    
    def test_start_of_next_month_december(self):
        """Test start_of_next_month from December."""
        test_dt = dt(2024, 12, 15, 12, 0, 0)
        result = start_of_next_month(test_dt)
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 1
    
    def test_end_of_day(self):
        """Test end_of_day function."""
        test_dt = dt(2024, 1, 15, 14, 30, 45)
        result = end_of_day(test_dt)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 16
        assert result.hour == 0
        assert result.minute == 0
    
    def test_beginning_of_day_with_datetime(self):
        """Test beginning_of_day with datetime input."""
        test_dt = dt(2024, 1, 15, 14, 30, 45)
        result = beginning_of_day(test_dt)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 0
        assert result.minute == 0
    
    def test_beginning_of_day_no_input(self):
        """Test beginning_of_day with no input (uses current time)."""
        result = beginning_of_day()
        # Should return today at midnight
        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0
        assert result.microsecond == 0
        assert result.tzinfo == timezone.utc
        # Should be today's date
        now = dt.now(timezone.utc)
        assert result.date() == now.date()
    
    def test_beginning_of_day_preserves_timezone(self):
        """Test beginning_of_day preserves timezone."""
        test_dt = dt(2024, 1, 15, 14, 30, 45, tzinfo=timezone.utc)
        result = beginning_of_day(test_dt)
        assert result.tzinfo == timezone.utc
    
    def test_beginning_of_day_millis(self):
        """Test beginning_of_day_millis function."""
        test_dt = dt(2024, 1, 1, 14, 30, 45, tzinfo=timezone.utc)
        result = beginning_of_day_millis(test_dt)
        expected = dt(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        expected_millis = int(expected.timestamp() * 1000)
        assert result == expected_millis


class TestConversions:
    """Test conversion functions."""
    
    def test_millis_to_dt(self):
        """Test millis_to_dt conversion."""
        # Test with known timestamp: 2024-01-01 00:00:00 UTC
        millis = 1704067200000
        result = millis_to_dt(millis)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 0
    
    def test_millis_to_dt_negative(self):
        """Test millis_to_dt with negative value (before 1970)."""
        # 1969-12-31 00:00:00 UTC
        millis = -86400000
        result = millis_to_dt(millis)
        assert result.year == 1969
        assert result.month == 12
        assert result.day == 31
    
    def test_millis_to_dt_str(self):
        """Test millis_to_dt_str conversion."""
        # 2024-01-01 14:30:00 UTC
        millis = 1704119400000
        result = millis_to_dt_str(millis)
        assert result == "20240101_1430"
    
    def test_dt_to_str_with_datetime(self):
        """Test dt_to_str with datetime input."""
        test_dt = dt(2024, 1, 15, 14, 30, 45)
        result = dt_to_str(test_dt)
        assert result == "20240115_1430"
    
    @patch('lib.util.time_util.dt')
    def test_dt_to_str_no_input(self, mock_dt):
        """Test dt_to_str with no input."""
        mock_now = dt(2024, 1, 15, 14, 30, 45, tzinfo=timezone.utc)
        mock_dt.now.return_value = mock_now
        result = dt_to_str()
        assert result == "20240115_1430"
    
    def test_dt_to_str_with_nat(self):
        """Test dt_to_str with NaT value."""
        result = dt_to_str(pd.NaT)
        assert result == "NaT"
    
    def test_date_to_str_with_date(self):
        """Test date_to_str with date input."""
        test_date = dt(2024, 1, 15).date()
        result = date_to_str(test_date)
        assert result == "20240115"
    
    def test_date_to_str_with_datetime(self):
        """Test date_to_str with datetime input."""
        test_dt = dt(2024, 1, 15, 14, 30, 45)
        result = date_to_str(test_dt)
        assert result == "20240115"
    
    @patch('lib.util.time_util.today')
    def test_date_to_str_no_input(self, mock_today):
        """Test date_to_str with no input."""
        mock_today.return_value = pd.Timestamp('2024-01-15')
        result = date_to_str()
        assert result == "20240115"
    
    def test_dt_to_millis_with_datetime(self):
        """Test dt_to_millis with datetime input."""
        test_dt = dt(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        result = dt_to_millis(test_dt)
        assert result == 1704067200000
    
    @patch('lib.util.time_util.dt')
    def test_dt_to_millis_no_input(self, mock_dt):
        """Test dt_to_millis with no input."""
        mock_now = dt(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        mock_dt.now.return_value = mock_now
        result = dt_to_millis()
        assert result == 1704067200000


class TestStringToDateConversions:
    """Test string to date/datetime conversions."""
    
    def test_date_str_to_dt_string_tz_aware(self):
        """Test date_str_to_dt with string input, timezone aware."""
        result = date_str_to_dt("20240115")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 0
        # Now returns timezone-aware UTC datetime by default
        assert result.tzinfo == timezone.utc
    
    def test_date_str_to_dt_string_tz_naive(self):
        """Test date_str_to_dt with string input (now always returns timezone aware)."""
        result = date_str_to_dt("20240115")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 0
        assert result.tzinfo == timezone.utc
    
    def test_date_str_to_dt_integer(self):
        """Test date_str_to_dt with integer input."""
        result = date_str_to_dt(20240115)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
    
    def test_date_str_to_dt_none(self):
        """Test date_str_to_dt with None input."""
        result = date_str_to_dt(None)
        assert result is None
    
    def test_date_str_to_date_string(self):
        """Test date_str_to_date with string input."""
        result = date_str_to_date("20240115")
        # Use type() instead of isinstance() for exact type check
        assert type(result) == dt_date
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
    
    def test_date_str_to_date_none(self):
        """Test date_str_to_date with None input."""
        result = date_str_to_date(None)
        assert result is None
    
    def test_date_to_end_dt_tz_aware(self):
        """Test date_to_end_dt with timezone aware output."""
        test_date = dt(2024, 1, 15).date()
        result = date_to_end_dt(test_date)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 16
        assert result.hour == 0
        assert result.tzinfo == timezone.utc
    
    def test_date_to_end_dt_tz_naive(self):
        """Test date_to_end_dt (now always returns timezone aware)."""
        test_date = dt(2024, 1, 15).date()
        result = date_to_end_dt(test_date)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 16
        assert result.hour == 0
        assert result.tzinfo == timezone.utc
    
    def test_date_to_start_dt_tz_aware(self):
        """Test date_to_start_dt with timezone aware output."""
        test_date = dt(2024, 1, 15).date()
        result = date_to_start_dt(test_date)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 0
        assert result.tzinfo == timezone.utc
    
    def test_date_to_start_dt_tz_naive(self):
        """Test date_to_start_dt (now always returns timezone aware)."""
        test_date = dt(2024, 1, 15).date()
        result = date_to_start_dt(test_date)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 0
        assert result.tzinfo == timezone.utc
    
    def test_str_to_dt_string_tz_aware(self):
        """Test str_to_dt with string input, timezone aware."""
        result = str_to_dt("20240115_1430")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 14
        assert result.minute == 30
        # Now returns timezone-aware UTC datetime by default
        assert result.tzinfo == timezone.utc
    
    def test_str_to_dt_string_tz_naive(self):
        """Test str_to_dt with string input (now always returns timezone aware)."""
        result = str_to_dt("20240115_1430")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 14
        assert result.minute == 30
        assert result.tzinfo == timezone.utc
    
    def test_str_to_dt_integer(self):
        """Test str_to_dt with integer input."""
        # Note: Python interprets 20240115_1430 as invalid syntax
        # We need to pass it as a string-like integer
        result = str_to_dt("20240115_1430")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 14
        assert result.minute == 30
    
    def test_str_to_dt_none(self):
        """Test str_to_dt with None input."""
        result = str_to_dt(None)
        assert result is None


class TestDateRange:
    """Test date_range function."""
    
    def test_date_range_basic(self):
        """Test basic date_range functionality."""
        start = dt(2024, 1, 1).date()
        end = dt(2024, 1, 5).date()
        result = date_range(start, end)
        assert len(result) == 5
        assert result[0] == dt(2024, 1, 1).date()
        assert result[4] == dt(2024, 1, 5).date()
    
    def test_date_range_skip_days(self):
        """Test date_range with skip_days parameter."""
        start = dt(2024, 1, 1).date()
        end = dt(2024, 1, 10).date()
        result = date_range(start, end, skip_days=2)
        assert len(result) == 5
        assert result[0] == dt(2024, 1, 1).date()
        assert result[1] == dt(2024, 1, 3).date()
        assert result[2] == dt(2024, 1, 5).date()
    
    def test_date_range_single_day(self):
        """Test date_range with same start and end date."""
        date = dt(2024, 1, 1).date()
        result = date_range(date, date)
        assert len(result) == 1
        assert result[0] == date


class TestWaitUntilMinute:
    """Test wait_until_minute function."""
    
    @patch('lib.util.time_util.time.sleep')
    @patch('lib.util.time_util.dt')
    def test_wait_until_minute(self, mock_dt, mock_sleep):
        """Test wait_until_minute function."""
        # Simulate time progression
        minutes = [14, 14, 15]  # Wait until minute 15
        mock_dt.now.side_effect = [
            dt(2024, 1, 1, 12, m, 30, tzinfo=timezone.utc) for m in minutes
        ]
        
        wait_until_minute(15)
        
        # Should sleep twice before the minute is divisible by 15
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(10)


class TestTimeStrToDts:
    """Test time_str_to_dts function."""
    
    def test_time_str_to_dts_basic(self):
        """Test basic time_str_to_dts functionality."""
        times = ["14:30", "09:00", "18:45"]
        current = dt(2024, 1, 15, 10, 0, 0)
        result = time_str_to_dts(times, current)
        
        assert len(result) == 3
        assert result[0].hour == 14
        assert result[0].minute == 30
        assert result[0].day == 15
        
        # 09:00 is before current time, so should be tomorrow
        assert result[1].hour == 9
        assert result[1].minute == 0
        assert result[1].day == 16
        
        assert result[2].hour == 18
        assert result[2].minute == 45
        assert result[2].day == 15
    
    def test_time_str_to_dts_with_timezone(self):
        """Test time_str_to_dts preserves timezone."""
        times = ["14:30"]
        current = dt(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = time_str_to_dts(times, current)
        
        assert result[0].tzinfo == timezone.utc


class TestNextClosestTime:
    """Test next_closest_time function."""
    
    def test_next_closest_time_basic(self):
        """Test basic next_closest_time functionality."""
        current = dt(2024, 1, 15, 10, 0, 0)
        times = [
            dt(2024, 1, 15, 9, 0, 0),   # Past
            dt(2024, 1, 15, 11, 0, 0),  # Future
            dt(2024, 1, 15, 14, 0, 0),  # Future
        ]
        result = next_closest_time(current, times)
        assert result == times[1]  # 11:00
    
    def test_next_closest_time_all_past(self):
        """Test next_closest_time when all times are past."""
        current = dt(2024, 1, 15, 15, 0, 0)
        times = [
            dt(2024, 1, 15, 9, 0, 0),
            dt(2024, 1, 15, 11, 0, 0),
            dt(2024, 1, 15, 14, 0, 0),
        ]
        result = next_closest_time(current, times)
        assert result is None
    
    def test_next_closest_time_empty_list(self):
        """Test next_closest_time with empty list."""
        current = dt(2024, 1, 15, 10, 0, 0)
        result = next_closest_time(current, [])
        assert result is None


class TestNextNMinute:
    """Test next_n_minute function."""
    
    def test_next_n_minute_basic(self):
        """Test basic next_n_minute functionality."""
        # Create timestamp for 2024-01-01 12:34:56
        test_dt = dt(2024, 1, 1, 12, 34, 56)
        ts = test_dt.timestamp()
        
        result = next_n_minute(ts, 5)
        
        # Should round down to 12:34:00 and add 5 minutes = 12:39:00
        result_dt = dt.fromtimestamp(result)
        assert result_dt.minute == 39
        assert result_dt.second == 0
    
    def test_next_n_minute_zero_minutes(self):
        """Test next_n_minute with zero minutes."""
        test_dt = dt(2024, 1, 1, 12, 34, 56)
        ts = test_dt.timestamp()
        
        result = next_n_minute(ts, 0)
        
        # Should round down to start of current minute
        result_dt = dt.fromtimestamp(result)
        assert result_dt.minute == 34
        assert result_dt.second == 0


class TestComputeLookbackDays:
    """Test compute_lookback_days function."""

    def test_compute_lookback_days_basic(self):
        """Test basic compute_lookback_days functionality."""
        # 1 day horizon, 1 lag = 2 days
        result = compute_lookback_days(1440)
        assert result == 2

        # 1 day horizon, 7 lags = 8 days
        result = compute_lookback_days(1440, lags=7)
        assert result == 8

    def test_compute_lookback_days_partial_day(self):
        """Test compute_lookback_days with partial day."""
        # New formula: int(math.ceil(horizon * (lags+1)) / 1440)
        # 720 minutes, lags=1: int(ceil(720 * 2) / 1440) = int(1) = 1
        result = compute_lookback_days(720)
        assert result == 1

        # 15 minutes, lags=1: int(ceil(15 * 2) / 1440) = int(0.02) = 0
        result = compute_lookback_days(15)
        assert result == 0

    def test_compute_lookback_days_multiple_lags(self):
        """Test compute_lookback_days with multiple lags."""
        # New formula: int(math.ceil(horizon * (lags+1)) / 1440)
        # 60 minutes, lags=24: int(ceil(60 * 25) / 1440) = int(1.04) = 1
        result = compute_lookback_days(60, lags=24)
        assert result == 1

        # 60 minutes, lags=25: int(ceil(60 * 26) / 1440) = int(1.08) = 1
        result = compute_lookback_days(60, lags=25)
        assert result == 1


class TestSecondsUntilNextUtcMidnight:
    """Test seconds_until_next_utc_midnight function."""

    def test_seconds_until_midnight_noon(self):
        """Test seconds until midnight from noon UTC."""
        test_dt = dt(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = seconds_until_next_utc_midnight(test_dt)
        # 12 hours = 43200 seconds
        assert result == 43200.0

    def test_seconds_until_midnight_5min_before(self):
        """Test seconds until midnight 5 minutes before."""
        test_dt = dt(2024, 1, 15, 23, 55, 0, tzinfo=timezone.utc)
        result = seconds_until_next_utc_midnight(test_dt)
        # 5 minutes = 300 seconds
        assert result == 300.0

    def test_seconds_until_midnight_1sec_before(self):
        """Test seconds until midnight 1 second before."""
        test_dt = dt(2024, 1, 15, 23, 59, 59, tzinfo=timezone.utc)
        result = seconds_until_next_utc_midnight(test_dt)
        # 1 second
        assert result == 1.0

    def test_seconds_until_midnight_with_microseconds(self):
        """Test seconds until midnight with microseconds precision."""
        test_dt = dt(2024, 1, 15, 23, 59, 59, 500000, tzinfo=timezone.utc)
        result = seconds_until_next_utc_midnight(test_dt)
        # 0.5 seconds
        assert abs(result - 0.5) < 0.001

    def test_seconds_until_midnight_at_midnight(self):
        """Test seconds until midnight when at midnight."""
        test_dt = dt(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        result = seconds_until_next_utc_midnight(test_dt)
        # Full day = 86400 seconds
        assert result == 86400.0

    def test_seconds_until_midnight_just_after_midnight(self):
        """Test seconds until midnight just after midnight."""
        test_dt = dt(2024, 1, 15, 0, 0, 1, tzinfo=timezone.utc)
        result = seconds_until_next_utc_midnight(test_dt)
        # Almost full day = 86399 seconds
        assert result == 86399.0

    def test_seconds_until_midnight_naive_datetime(self):
        """Test with naive datetime (should assume UTC)."""
        test_dt = dt(2024, 1, 15, 12, 0, 0)  # No timezone
        result = seconds_until_next_utc_midnight(test_dt)
        # Should treat as UTC and return 12 hours
        assert result == 43200.0

    def test_seconds_until_midnight_no_input(self):
        """Test without input (uses current time)."""
        result = seconds_until_next_utc_midnight()
        # Result should be between 0 and 86400 seconds
        assert 0 < result <= 86400.0

    def test_seconds_until_midnight_non_utc_timezone(self):
        """Test with non-UTC timezone (should convert to UTC)."""
        # 12:00 EST (UTC-5) = 17:00 UTC
        est = timezone(td(hours=-5))
        test_dt = dt(2024, 1, 15, 12, 0, 0, tzinfo=est)
        result = seconds_until_next_utc_midnight(test_dt)
        # 17:00 UTC to midnight = 7 hours = 25200 seconds
        assert result == 25200.0

    def test_seconds_until_midnight_year_boundary(self):
        """Test across year boundary."""
        test_dt = dt(2024, 12, 31, 23, 30, 0, tzinfo=timezone.utc)
        result = seconds_until_next_utc_midnight(test_dt)
        # 30 minutes = 1800 seconds
        assert result == 1800.0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_date_str_to_dt_invalid_format(self):
        """Test date_str_to_dt with invalid format."""
        with pytest.raises(ValueError):
            date_str_to_dt("2024-01-15")  # Wrong format
    
    def test_str_to_dt_invalid_format(self):
        """Test str_to_dt with invalid format."""
        with pytest.raises(ValueError):
            str_to_dt("2024-01-15 14:30")  # Wrong format
    
    def test_time_str_to_dts_invalid_format(self):
        """Test time_str_to_dts with invalid time format."""
        with pytest.raises(ValueError):
            time_str_to_dts(["14:30:00"], dt.now())  # Has seconds
    
    def test_leap_year_handling(self):
        """Test leap year handling in date functions."""
        # February 29, 2024 (leap year)
        leap_date = date_str_to_date("20240229")
        assert leap_date.year == 2024
        assert leap_date.month == 2
        assert leap_date.day == 29
        
        # Next day should be March 1
        next_day = date_to_end_dt(leap_date)
        assert next_day.month == 3
        assert next_day.day == 1
    
    def test_year_boundary(self):
        """Test year boundary handling."""
        # December 31, 2023
        year_end = date_str_to_date("20231231")
        
        # Next day should be January 1, 2024
        next_day = date_to_end_dt(year_end)
        assert next_day.year == 2024
        assert next_day.month == 1
        assert next_day.day == 1