import math
import time
import logging
import re
from datetime import datetime as dt, timedelta as td, timezone, date
from typing import Optional, List, Union, Any

import pandas as pd
import numpy as np


MINUTES_IN_DAY = 1440
MILLIS_IN_DAY = MINUTES_IN_DAY * 60 * 1000
DATE_FORMAT = "%Y%m%d"
DATE_TIME_FORMAT = "%Y%m%d_%H%M"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def to_datetime(arg: Any, **kwargs) -> Union[pd.Timestamp, pd.DatetimeIndex, pd.Series]:
    """Wrapper for pd.to_datetime that ensures UTC timezone is always set.
    
    This function wraps pandas.to_datetime() and automatically sets utc=True
    to ensure all datetime conversions result in UTC timezone-aware objects.
    
    Args:
        arg: The object to convert to datetime (same as pd.to_datetime).
        **kwargs: Additional keyword arguments to pass to pd.to_datetime.
                  If 'utc' is explicitly set to False, a warning is issued
                  but utc=True is still used.
    
    Returns:
        Same as pd.to_datetime - Timestamp, DatetimeIndex, or Series of datetime64.
    
    Examples:
        >>> to_datetime('2024-01-01')
        Timestamp('2024-01-01 00:00:00+0000', tz='UTC')
        
        >>> to_datetime(['2024-01-01', '2024-01-02'])
        DatetimeIndex(['2024-01-01', '2024-01-02'], dtype='datetime64[ns, UTC]', freq=None)
    """
    if 'utc' in kwargs and not kwargs['utc']:
        logger.warning("to_datetime called with utc=False, overriding to utc=True for consistency")
    
    kwargs['utc'] = True
    result = pd.to_datetime(arg, **kwargs)
    
    if hasattr(result, 'dtype'):
        # Ensure the result has nanosecond precision
        if 'datetime64' in str(result.dtype) and 'ns' not in str(result.dtype):
            result = result.astype('datetime64[ns, UTC]')
    return result


def ensure_utc_timezone(adt: dt) -> dt:
    """Ensure a datetime has UTC timezone, converting if necessary.
    
    Args:
        adt: Datetime object (timezone-aware or naive).
        
    Returns:
        datetime: Timezone-aware datetime in UTC.
        
    Notes:
        - If input is naive, assumes it represents UTC time
        - If input has non-UTC timezone, converts to UTC
        - If already UTC, returns unchanged
        - Useful for standardizing datetime inputs
    """
    if adt.tzinfo is None:
        # Naive datetime - assume it's UTC
        return adt.replace(tzinfo=timezone.utc)
    if adt.tzinfo != timezone.utc:
        # Has timezone but not UTC - convert to UTC
        return adt.astimezone(timezone.utc)
    # Already UTC
    return adt


def convert_to_datetime_series(series: pd.Series, unit: str = 'ms', context: str = "") -> pd.Series:
    """Convert a pandas Series to datetime64[ns, UTC] handling various input formats.
    
    This function intelligently handles different datetime representations:
    - Already datetime64 arrays (ensures UTC timezone)
    - Numeric values (treats as timestamps with specified unit)
    - Mixed object arrays
    - NaN/NaT values are preserved
    
    Args:
        series: Pandas Series that may contain datetime values in various formats.
        unit: Unit for numeric timestamp conversion ('ms', 'us', 'ns', 's').
              Only used if the series contains numeric values.
        context: Optional context string for logging (e.g., column name).
        
    Returns:
        pd.Series: Series with datetime64[ns, UTC] dtype.
        
    Notes:
        - If input is already datetime, ensures UTC timezone
        - If input is numeric, treats as timestamps in specified unit
        - If input is object dtype, attempts datetime conversion
        - Handles special case where 0 represents missing data (converts to NaT)
        
    Examples:
        >>> # From milliseconds
        >>> ms_series = pd.Series([1704067200000, None, 1704153600000])
        >>> convert_to_datetime_series(ms_series, unit='ms')
        0   2024-01-01 00:00:00+00:00
        1                          NaT
        2   2024-01-02 00:00:00+00:00
        dtype: datetime64[ns, UTC]
        
        >>> # Already datetime
        >>> dt_series = pd.Series(pd.date_range('2024-01-01', periods=3))
        >>> convert_to_datetime_series(dt_series)
        0   2024-01-01 00:00:00+00:00
        1   2024-01-02 00:00:00+00:00
        2   2024-01-03 00:00:00+00:00
        dtype: datetime64[ns, UTC]
    """
    if len(series) == 0:
        return pd.Series([], dtype='datetime64[ns, UTC]')
    
    # Check if it's already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        # Already datetime, just ensure UTC timezone
        if context:
            logger.debug(f"Series {context} is already datetime dtype: {series.dtype}")
        return to_datetime(series)

    # Check if it's numeric (likely timestamps)
    if pd.api.types.is_numeric_dtype(series):
        if context:
            logger.debug(f"Converting numeric series {context} to datetime with unit={unit}")
        
        # Replace 0 with NaN before conversion (0 often represents missing data)
        # But preserve actual zeros that might be valid timestamps
        series_copy = series.copy()
        
        # Only treat 0 as missing if it's likely not a valid timestamp
        # Unix epoch (0) is 1970-01-01, which is unlikely to be valid financial data
        if unit == 'ms':
            # For milliseconds, 0 would be 1970-01-01
            series_copy = series_copy.replace(0, np.nan)
        elif unit == 's':
            # For seconds, 0 would be 1970-01-01
            series_copy = series_copy.replace(0, np.nan)
            
        # Convert to datetime
        result = to_datetime(series_copy, unit=unit)
        
        # Additional check: replace any 1970-01-01 dates with NaT
        # These are likely from 0 values that represent missing data
        epoch_date = pd.Timestamp('1970-01-01', tz='UTC')
        if (result == epoch_date).any():
            if context:
                logger.debug(f"Replacing epoch dates (1970-01-01) with NaT in {context}")
            result = result.where(result != epoch_date, pd.NaT)
            
        return result
    
    # Check if it's object or string dtype
    elif series.dtype == 'object' or pd.api.types.is_string_dtype(series):
        if context:
            sample_value = series.dropna().iloc[0] if len(series.dropna()) > 0 else None
            sample_type = type(sample_value).__name__ if sample_value is not None else 'None'
            logger.debug(f"Converting object/string dtype series {context} to datetime, sample type: {sample_type}")

        # Try direct conversion
        try:
            return to_datetime(series)
        except Exception as e:
            logger.warning(f"Failed to convert object series {context} to datetime: {e}")
            # Try filling NaN/None with 0 first, then convert
            series_filled = series.fillna(0)
            result = to_datetime(series_filled, unit=unit)
            # Replace epoch dates back to NaT
            epoch_date = pd.Timestamp('1970-01-01', tz='UTC')
            result = result.where(result != epoch_date, pd.NaT)
            return result

    else:
        # Unexpected dtype
        logger.warning(f"Unexpected dtype for datetime conversion{f' ({context})' if context else ''}: {series.dtype}")
        try:
            return to_datetime(series)
        except Exception as e:
            logger.error(f"Failed to convert series {context} with dtype {series.dtype} to datetime: {e}")
            raise


def datetime_series_to_int64(series: pd.Series, context: str = "") -> pd.Series:
    """Convert a pandas Series of datetime values to int64 nanoseconds.

    This function handles various datetime representations safely:
    - datetime64 arrays at any resolution (ns, us, ms, s)
    - Object arrays containing Timestamp objects (need .value accessor)
    - Mixed or problematic dtypes (fallback to element-wise conversion)

    Args:
        series: Pandas Series containing datetime values.
        context: Optional context string for logging (e.g., symbol name).

    Returns:
        pd.Series: Series with int64 values representing nanoseconds since epoch.

    Raises:
        TypeError: If the series contains non-datetime values that cannot be converted.

    Notes:
        - NaN/NaT values are preserved as 0 in the output
        - Logs warnings for unexpected dtypes to help with debugging
        - Uses the fastest conversion method available for the data type
        - Normalizes to nanosecond resolution before int64 cast to handle
          pandas 3.0's default microsecond resolution correctly

    Examples:
        >>> dates = pd.Series(pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> datetime_series_to_int64(dates)
        0    1704067200000000000
        1    1704153600000000000
        2    1704240000000000000
        dtype: int64
    """
    if len(series) == 0:
        return pd.Series([], dtype='int64')

    # Check the dtype of the series
    if pd.api.types.is_datetime64_any_dtype(series):
        # Normalize to nanosecond resolution before int64 cast.
        # In pandas 3.0+, astype('int64') returns values in the column's
        # native resolution (us, ms, s), not always nanoseconds.
        try:
            return series.dt.as_unit('ns').astype('int64')
        except Exception as e:
            # If as_unit/astype fails, fall back to .value (always returns ns)
            logger.warning(f"Fast conversion failed for datetime64 dtype{f' ({context})' if context else ''}: {e}")
            return series.apply(lambda x: x.value if pd.notna(x) else 0)

    # Check if it's object dtype (or string dtype in pandas 3.0+)
    elif series.dtype == 'object' or pd.api.types.is_string_dtype(series):
        # Log the unexpected dtype for debugging
        sample_value = series.iloc[0] if len(series) > 0 else None
        sample_type = type(sample_value).__name__ if sample_value is not None else 'None'

        # Only log if we have actual values (not all NaN)
        non_null_count = series.notna().sum()
        if non_null_count > 0:
            logger.debug(f"Converting object dtype to int64{f' ({context})' if context else ''}: "
                        f"dtype={series.dtype}, sample_type={sample_type}, non_null_count={non_null_count}")

        # For object dtype containing Timestamps, use .value accessor (always returns ns)
        try:
            return series.apply(lambda x: x.value if pd.notna(x) and hasattr(x, 'value') else 0)
        except AttributeError as e:
            # If objects don't have .value attribute, try converting to datetime first
            logger.warning(f"Object dtype doesn't have .value attribute{f' ({context})' if context else ''}, "
                          f"attempting datetime conversion: {e}")
            datetime_series = pd.to_datetime(series, utc=True)
            return datetime_series.dt.as_unit('ns').astype('int64')

    else:
        # Unexpected dtype - log warning and attempt conversion
        logger.warning(f"Unexpected dtype for datetime conversion{f' ({context})' if context else ''}: {series.dtype}")
        try:
            # Try converting to datetime first, then to int64
            datetime_series = pd.to_datetime(series, utc=True)
            return datetime_series.dt.as_unit('ns').astype('int64')
        except Exception as e:
            logger.error(f"Failed to convert series to int64{f' ({context})' if context else ''}: {e}")
            raise TypeError(f"Cannot convert series with dtype {series.dtype} to int64 timestamps") from e


def datetime_series_to_epoch_seconds(series: pd.Series) -> pd.Series:
    """Convert a datetime Series to epoch seconds (float64), handling tz-naive/aware and any resolution.

    Works correctly regardless of pandas datetime resolution (ns, us, s) by using
    timedelta subtraction rather than raw int64 casting.

    Args:
        series: Pandas Series of datetime values (tz-naive or tz-aware).

    Returns:
        pd.Series of float64 epoch seconds. NaT values become 0.0.
    """
    if series.dt.tz is None:
        series = series.dt.tz_localize('UTC')
    epoch = pd.Timestamp('1970-01-01', tz='UTC')
    return (series - epoch).dt.total_seconds().fillna(0.0)


def today() -> pd.Timestamp:
    """Get current UTC date as a pandas Timestamp.
    
    Returns:
        pd.Timestamp: Current UTC date at midnight (00:00:00).
        
    Notes:
        - Returns a timezone-aware UTC timestamp
        - Time component is set to midnight
        - Equivalent to pd.to_datetime(today_date(), utc=True)
    """
    # pandas timestamp is almost interchangeable with python datetime!
    return pd.to_datetime(today_date(), utc=True)


def today_date() -> date:
    """Get current UTC date.
    
    Returns:
        datetime.date: Current UTC date without time component.
        
    Notes:
        - Always uses UTC timezone
        - Returns pure date object without timezone info
    """
    return dt.now(timezone.utc).date()


def yesterday() -> pd.Timestamp:
    """Get yesterday's UTC date as a pandas Timestamp.
    
    Returns:
        pd.Timestamp: Yesterday's UTC date at midnight (00:00:00).
        
    Notes:
        - Returns a timezone-aware UTC timestamp
        - Time component is set to midnight
        - Exactly 24 hours before today()
    """
    return today() - td(days=1)


def yesterday_date() -> date:
    """Get yesterday's UTC date.
    
    Returns:
        datetime.date: Yesterday's UTC date without time component.
        
    Notes:
        - Always uses UTC timezone for calculation
        - Returns pure date object without timezone info
    """
    return today_date() - td(days=1)


def start_of_next_month(adt: dt) -> dt:
    """Get the first day of the next month.
    
    Args:
        adt: Input datetime to calculate next month from.
        
    Returns:
        datetime: First day of the next month at 00:00:00.
        
    Notes:
        - Preserves timezone information from input
        - If input is timezone-naive, assumes UTC and returns timezone-aware UTC
        - Works correctly across month boundaries (e.g., Jan 31 -> Feb 1)
        - Time component is reset to midnight
    """
    adt = ensure_utc_timezone(adt)
    return (adt.replace(day=1) + td(days=32)).replace(day=1)


def end_of_day(adt: dt) -> dt:
    """Get the end of day (start of next day) for given datetime.
    
    Args:
        adt: Input datetime to get end of day for.
        
    Returns:
        datetime: Start of the next day (00:00:00 of following day).
        
    Notes:
        - Returns midnight of the following day
        - If input is timezone-naive, assumes UTC and returns timezone-aware UTC
        - Useful for date range queries with exclusive upper bound
    """
    adt = ensure_utc_timezone(adt)
    return beginning_of_day(adt) + td(days=1)


def beginning_of_day(adt: Optional[dt] = None) -> dt:
    """Get the beginning of day (midnight) for given datetime.
    
    Args:
        adt: Input datetime. If None, uses current UTC time.
        
    Returns:
        datetime: Input date at midnight (00:00:00) with UTC timezone.
        
    Notes:
        - Always returns timezone-aware UTC datetime
        - If no input provided, uses current UTC time
        - Time component is set to 00:00:00
        - If input is timezone-naive, assumes UTC
    """
    if adt is None:
        adt = dt.now(timezone.utc)
    else:
        adt = ensure_utc_timezone(adt)
    
    # Convert to date and back to datetime at midnight
    result = dt.combine(adt.date(), dt.min.time())
    # Ensure UTC timezone
    return result.replace(tzinfo=timezone.utc)


def beginning_of_day_millis(adt: Optional[dt] = None) -> int:
    """Get the beginning of day as milliseconds since epoch.
    
    Args:
        adt: Input datetime. If None, uses current UTC time.
        
    Returns:
        int: Milliseconds since Unix epoch for midnight of given day.
        
    Notes:
        - Useful for timestamp-based queries and calculations
        - Always returns positive integer
        - Timezone-aware calculations preserved
    """
    return dt_to_millis(beginning_of_day(adt))


### CONVERSIONS


def millis_to_dt(millis: int) -> dt:
    """Convert milliseconds since epoch to datetime.
    
    Args:
        millis: Milliseconds since Unix epoch (1970-01-01 00:00:00 UTC).
        
    Returns:
        datetime: Datetime representation of the timestamp.
        
    Notes:
        - Returns timezone-aware datetime in UTC
        - Uses pandas for conversion for consistency
        - Handles negative values for dates before 1970
    """
    return pd.to_datetime(millis, unit='ms', utc=True)


def millis_to_dt_str(millis: int) -> str:
    """Convert milliseconds since epoch to formatted date string.
    
    Args:
        millis: Milliseconds since Unix epoch.
        
    Returns:
        str: Formatted string in DATE_TIME_FORMAT (YYYYMMDD_HHMM).
        
    Notes:
        - Convenient for logging and file naming
        - Format: 'YYYYMMDD_HHMM' (e.g., '20240101_1430')
    """
    return dt_to_str(millis_to_dt(millis))


def dt_to_str(adt: Optional[dt] = None) -> str:
    """Convert datetime to formatted string.
    
    Args:
        adt: Input datetime. If None, uses current UTC time.
        
    Returns:
        str: Formatted string in DATE_TIME_FORMAT (YYYYMMDD_HHMM) or 'NaT' for null values.
        
    Notes:
        - Format: 'YYYYMMDD_HHMM' (e.g., '20240101_1430')
        - Returns 'NaT' for pandas null datetime values
        - Used for file naming and logging
    """
    if adt is None:
        adt = dt.now(timezone.utc)
    elif pd.isna(adt):
        return "NaT"
    return adt.strftime(DATE_TIME_FORMAT)


def date_to_str(adt: Optional[dt] | Optional[date] = None) -> str:
    """Convert date or datetime to formatted date string.
    
    Args:
        adt: Input date or datetime. If None, uses today's date.
        
    Returns:
        str: Formatted string in DATE_FORMAT (YYYYMMDD).
        
    Notes:
        - Format: 'YYYYMMDD' (e.g., '20240101')
        - Accepts both datetime and date objects
        - Default behavior uses today() if None provided
    """
    #this default is stupid and should change
    if adt is None:
        adt = today()
    return adt.strftime(DATE_FORMAT)


def dt_to_millis(adt: Optional[dt] = None) -> int:
    """Convert datetime to milliseconds since epoch.
    
    Args:
        adt: Input datetime. If None, uses current UTC time.
        
    Returns:
        int: Milliseconds since Unix epoch (1970-01-01 00:00:00 UTC).
        
    Notes:
        - Handles timezone-aware datetimes correctly
        - Always returns integer value
        - Precision limited to milliseconds (truncates microseconds)
    """
    if adt is None:
        adt = dt.now(timezone.utc)
    return int(adt.timestamp() * 1000)


def date_str_to_dt(dstr: str | int) -> Optional[dt]:
    """Convert date string to datetime.
    
    Args:
        dstr: Date string in YYYYMMDD format or integer representation.
        
    Returns:
        Optional[datetime]: UTC timezone-aware datetime at midnight, or None if input is None.
        
    Notes:
        - Expected format: 'YYYYMMDD' (e.g., '20240101')
        - Time component set to 00:00:00
        - Accepts both string and integer inputs
        - Always returns UTC timezone-aware datetime
    """
    if dstr is None:
        return None
    naive_dt = dt.strptime(str(dstr), DATE_FORMAT)
    return naive_dt.replace(tzinfo=timezone.utc)


def date_str_to_date(dstr: str | int) -> Optional[date]:
    """Convert date string to date object.
    
    Args:
        dstr: Date string in YYYYMMDD format or integer representation.
        
    Returns:
        Optional[datetime.date]: Date object, or None if input is None.
        
    Notes:
        - Expected format: 'YYYYMMDD' (e.g., '20240101')
        - Returns pure date object without time component
        - Accepts both string and integer inputs
    """
    dt_obj = date_str_to_dt(dstr)
    if dt_obj is None:
        return None
    return dt_obj.date()


def date_to_end_dt(adate: date) -> Optional[dt]:
    """Convert date to end-of-day datetime (start of next day).
    
    Args:
        adate: Input date object.
        
    Returns:
        datetime: UTC timezone-aware midnight of the following day.
        
    Notes:
        - Useful for date range queries with exclusive upper bound
        - Time set to 00:00:00 of next day
        - Always returns UTC timezone-aware datetime
    """
    result = dt.combine(adate + td(days=1), dt.min.time())
    return result.replace(tzinfo=timezone.utc)


def date_to_start_dt(adate: date) -> Optional[dt]:
    """Convert date to start-of-day datetime.
    
    Args:
        adate: Input date object.
        
    Returns:
        datetime: UTC timezone-aware midnight of the given day.
        
    Notes:
        - Time set to 00:00:00
        - Equivalent to beginning_of_day for date objects
        - Always returns UTC timezone-aware datetime
    """
    result = dt.combine(adate, dt.min.time())
    return result.replace(tzinfo=timezone.utc)


def str_to_dt(dstr: str | int) -> Optional[dt]:
    """Convert datetime string to datetime object.
    
    Args:
        dstr: DateTime string in YYYYMMDD_HHMM format or integer representation.
        
    Returns:
        Optional[datetime]: UTC timezone-aware parsed datetime, or None if input is None.
        
    Notes:
        - Expected format: 'YYYYMMDD_HHMM' (e.g., '20240101_1430')
        - Accepts both string and integer inputs
        - Always returns UTC timezone-aware datetime
    """
    if dstr is None:
        return None
    naive_dt = dt.strptime(str(dstr), DATE_TIME_FORMAT)
    return naive_dt.replace(tzinfo=timezone.utc)


def date_range(d1: date, d2: date, skip_days: Optional[int] = 1) -> List[date]:
    """Generate a list of dates between two dates.

    Args:
        d1: Start date (inclusive).
        d2: End date (inclusive).
        skip_days: Number of days to skip between dates. Defaults to 1.

    Returns:
        List[datetime.date]: List of date objects in the range.

    Notes:
        - Both start and end dates are inclusive
        - skip_days=1 returns every day, skip_days=2 returns every other day
        - Uses pandas date_range for robust date arithmetic
        - Handles timezone-aware datetime inputs by converting to naive dates
    """
    # Normalize inputs to plain dates to avoid timezone mismatch errors
    # datetime is a subclass of date, so check datetime first
    if isinstance(d1, dt):
        d1 = d1.date()
    if isinstance(d2, dt):
        d2 = d2.date()
    return [dd.date() for dd in pd.date_range(d1, d2, freq=f"{skip_days}D")]


def wait_until_minute(minute: int):
    """Block execution until the minute is divisible by specified value.
    
    Args:
        minute: Minute divisor to wait for (e.g., 5 waits for :00, :05, :10, etc.).
        
    Notes:
        - Blocks thread execution with 10-second polling interval
        - Uses UTC time for minute checking
        - Useful for scheduling tasks at regular intervals
        - Example: wait_until_minute(15) waits for :00, :15, :30, :45
    """
    while True:
        if dt.now(timezone.utc).minute % minute == 0:
            break
        time.sleep(10)


def time_str_to_dts(times: List[str], current_time: dt) -> List[dt]:
    """Convert list of time strings to datetime objects.
    
    Args:
        times: List of time strings in HH:MM format (e.g., ['14:30', '09:00']).
        current_time: Reference datetime for date and timezone information.
        
    Returns:
        List[datetime]: List of datetime objects with times set.
        
    Notes:
        - Times are set to today's date from current_time
        - If a time has already passed today, it's set to tomorrow
        - Preserves timezone information from current_time
        - Useful for scheduling future events
    """
    time_dts = []
    for time_str in times:
        # Parse each time string to a datetime object for today
        time_today = dt.strptime(time_str, "%H:%M")
        time_today = time_today.replace(year=current_time.year, month=current_time.month, day=current_time.day)
        # If current_time has timezone info, add it to time_today
        if current_time.tzinfo is not None:
            time_today = time_today.replace(tzinfo=current_time.tzinfo)
        # If the time is earlier in the day, assume it's for the next day
        if time_today <= current_time:
            time_today += td(days=1)
        time_dts.append(time_today)
    return time_dts


def next_closest_time(current_time: dt, times: List[dt]) -> dt:
    """Find the next closest future time from a list of times.
    
    Args:
        current_time: Reference time to compare against.
        times: List of datetime objects to search.
        
    Returns:
        datetime: The closest future time, or None if no future times exist.
        
    Notes:
        - Only considers times after current_time
        - Returns None if all times are in the past
        - Useful for finding next scheduled event
        - Times must be timezone-consistent with current_time
    """
    # Initialize the closest time far in the future
    closest_time = None
    min_time_diff = td(days=1)  # a large time delta initially

    for time_today in times:
        if time_today < current_time:
            continue

        # Calculate the time difference
        time_diff = time_today - current_time

        # Update the closest time if this time is closer
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_time = time_today
    return closest_time


def next_n_minute(ts: float, num_mins: int) -> int:
    """Calculate the timestamp for n minutes from now, aligned to minute boundary.
    
    Args:
        ts: Current timestamp from time.time() (seconds since epoch).
        num_mins: Number of minutes to add.
        
    Returns:
        int: Timestamp for n minutes from now, aligned to start of minute.
        
    Notes:
        - Rounds down to current minute start, then adds specified minutes
        - Returns integer seconds (not milliseconds)
        - Useful for scheduling tasks at exact minute boundaries
        - Example: next_n_minute(time.time(), 5) returns timestamp for 5 minutes from now at :00 seconds
    """
    # ts here based on time.time()
    return int(ts - (int(ts) % 60)) + (60 * num_mins)


def compute_lookback_days(horizon: int, lags: int = 1) -> int:
    """Calculate required lookback days for given horizon and lags.

    Args:
        horizon: Time horizon in minutes.
        lags: Number of lag periods to include. Defaults to 1.

    Returns:
        int: Number of days needed for lookback.

    Notes:
        - Converts minute-based horizons to days
        - Adds 1 extra day for safety margin
        - Formula: ceil((horizon * lags) / 1440) + 1
        - Used for determining data requirements in feature/model calculations
        - Example: horizon=1440 (1 day), lags=7 returns 8 days
    """
    return int(math.ceil(horizon * (lags+1)) / 1440)


def seconds_until_next_utc_midnight(now: Optional[dt] = None) -> float:
    """Calculate seconds until the next UTC midnight.

    Args:
        now: Current datetime. If None, uses current UTC time.

    Returns:
        float: Number of seconds until next UTC midnight (00:00:00).

    Notes:
        - Always calculates relative to UTC timezone
        - Returns positive float representing seconds remaining
        - If current time is exactly midnight, returns seconds until next midnight (~86400)
        - Useful for scheduling midnight tasks

    Examples:
        >>> # At 23:55:00 UTC
        >>> seconds_until_next_utc_midnight()
        300.0  # 5 minutes = 300 seconds

        >>> # At 12:00:00 UTC
        >>> seconds_until_next_utc_midnight()
        43200.0  # 12 hours = 43200 seconds
    """
    if now is None:
        now = dt.now(timezone.utc)
    else:
        now = ensure_utc_timezone(now)

    # Calculate next midnight
    next_midnight = (now + td(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    # Return seconds until midnight
    return (next_midnight - now).total_seconds()


def find_date_pattern(text: str) -> Optional[str]:
    """Find date pattern .YYYYMMDD. where YYYY starts with 2, return YYYYMMDD"""
    pattern = r'\.2\d{7}\.'
    matches = re.findall(pattern, text)
    # Return first match without dots, or None
    return matches[0][1:-1] if matches else None


def get_week_range(reference_date: dt = None) -> tuple[dt, dt]:
    """Calculate Monday-Monday week range containing the reference date.

    Args:
        reference_date: Date to find the week for. If None, uses current time.

    Returns:
        Tuple of (start_dt, end_dt) where:
        - start_dt: Monday 00:00:00 UTC at the start of the week
        - end_dt: Monday 00:00:00 UTC at the end of the week (exclusive)
    """
    if reference_date is None:
        reference_date = dt.now(timezone.utc)

    # Ensure timezone aware
    if reference_date.tzinfo is None:
        reference_date = reference_date.replace(tzinfo=timezone.utc)

    # Find the most recent Monday (including today if it's Monday)
    # weekday(): Monday=0, Tuesday=1, ..., Sunday=6
    days_since_monday = reference_date.weekday()

    # Start of week is the Monday before or on the reference date
    week_start = reference_date - td(days=days_since_monday)
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

    # End of week is the next Monday
    week_end = week_start + td(days=7)

    return week_start, week_end


def get_last_complete_week_range() -> tuple[dt, dt]:
    """Get the most recent complete Monday-Monday week.

    If today is Monday, returns the week that just ended.
    Otherwise, returns the last complete week.

    Returns:
        Tuple of (start_dt, end_dt) for the last complete week
    """
    now = dt.now(timezone.utc)

    # Get the week containing today
    week_start, week_end = get_week_range(now)

    # If the week hasn't ended yet, go back one week
    if week_end > now:
        week_start = week_start - td(days=7)
        week_end = week_end - td(days=7)

    return week_start, week_end


def get_friday_week_range(reference_date: dt = None) -> tuple[dt, dt]:
    """Calculate Friday-Friday week range containing the reference date.

    Args:
        reference_date: Date to find the week for. If None, uses current time.

    Returns:
        Tuple of (start_dt, end_dt) where:
        - start_dt: Friday 00:00:00 UTC at the start of the week
        - end_dt: Friday 00:00:00 UTC at the end of the week (exclusive)
    """
    if reference_date is None:
        reference_date = dt.now(timezone.utc)

    # Ensure timezone aware
    if reference_date.tzinfo is None:
        reference_date = reference_date.replace(tzinfo=timezone.utc)

    # Find the most recent Friday (including today if it's Friday)
    # weekday(): Monday=0, Tuesday=1, ..., Thursday=3, Friday=4, Saturday=5, Sunday=6
    days_since_friday = (reference_date.weekday() - 4) % 7

    # Start of week is the Friday before or on the reference date
    week_start = reference_date - td(days=days_since_friday)
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

    # End of week is the next Friday
    week_end = week_start + td(days=7)

    return week_start, week_end


def get_last_complete_friday_week_range() -> tuple[dt, dt]:
    """Get the most recent complete Friday-Friday week.

    If today is Friday, returns the week that just ended.
    Otherwise, returns the last complete week.

    Returns:
        Tuple of (start_dt, end_dt) for the last complete Friday-Friday week
    """
    now = dt.now(timezone.utc)

    # Get the week containing today
    week_start, week_end = get_friday_week_range(now)

    # If the week hasn't ended yet, go back one week
    if week_end > now:
        week_start = week_start - td(days=7)
        week_end = week_end - td(days=7)

    return week_start, week_end
