import logging
from datetime import timedelta as td, datetime as dt
from typing import List, Tuple, Optional, Literal, Any

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from .logging_util import KeyLogger
from .time_util import dt_to_str, to_datetime, beginning_of_day
from .util import TARDIS_EXCHANGE, log_and_raise, SYMBOL_VENUE

pd.options.display.max_rows = 1440

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)

DF_INDEX = ['ts', SYMBOL_VENUE]
MIN_VALID_BAR_DATA = 1440 - 5
INT32_MAX = np.iinfo(np.int32).max


FAST_CONCAT = True


def fillin_index_df(df: pd.DataFrame) -> pd.DataFrame:
    min_ts, max_ts = get_min_max_ts(df)
    start = beginning_of_day(min_ts) + td(minutes=1)
    end = beginning_of_day(max_ts) + td(days=1)
    rng = pd.date_range(start=start, end=end, freq='1Min')
    logger.info(f"Filling in index from {start} - {end} with {len(rng)} rows")
    df = df.reindex(pd.MultiIndex.from_product([rng, df.index.levels[1]], names=['ts', 'symbol_venue']))
    return df


def ensure_utc_ts(df: pd.DataFrame, columns: Optional[List[str]] = None, convert_to_datetime: bool = True) -> pd.DataFrame:
    """Ensure that specified datetime columns or indices are timezone-aware UTC.
    
    Converts non-datetime data to datetime, timezone-naive timestamps to UTC, 
    and converts other timezones to UTC.
    
    Args:
        df: DataFrame with datetime columns/indices that may be timezone-naive or non-datetime.
        columns: List of column names to process. Defaults to ['ts']. Checks both columns and index levels.
        convert_to_datetime: If True, converts non-datetime to datetime. If False, only handles timezone conversion.
        
    Returns:
        DataFrame with timezone-aware UTC datetime columns/indices.
        Modifies the DataFrame in place and returns it.
        
    Notes:
        - Processes specified columns in both regular columns and index levels
        - If column is not datetime type and convert_to_datetime=True, converts to datetime assuming UTC
        - If column is timezone-naive, assumes it represents UTC time
        - If column has non-UTC timezone, converts to UTC
        - If already UTC datetime, returns unchanged
        - Handles both column and MultiIndex cases
    """
    if columns is None:
        columns = ['ts']
    
    # Process each specified column
    for col_name in columns:
        # Check if column is in the index
        if col_name in df.index.names:
            if df.index.nlevels == 1:
                # Single index
                if not isinstance(df.index, pd.DatetimeIndex) and convert_to_datetime:
                    logger.warning(f"Ensuring column {col_name} is datetime")
                    # Convert to DatetimeIndex
                    df.index = to_datetime(df.index)
                elif isinstance(df.index, pd.DatetimeIndex):
                    if df.index.tz is None:
                        logger.warning(f"Adding TZ info to {col_name}")
                        df.index = df.index.tz_localize('UTC')
                    elif str(df.index.tz) != 'UTC':
                        logger.warning(f"Converting TZ info on {col_name} from {df.index.tz}")
                        df.index = df.index.tz_convert('UTC')
            else:
                # MultiIndex
                col_level = df.index.names.index(col_name)
                col_index = df.index.get_level_values(col_level)
                
                # If not a DatetimeIndex and we should convert
                if not isinstance(col_index, pd.DatetimeIndex) and convert_to_datetime:
                    logger.warning(f"Ensuring column {col_name} is datetime")
                    col_index = to_datetime(col_index)
                    needs_update = True
                elif isinstance(col_index, pd.DatetimeIndex):
                    needs_update = False
                    if col_index.tz is None:
                        logger.warning(f"Adding TZ info to {col_name}")
                        col_index = col_index.tz_localize('UTC')
                        needs_update = True
                    elif str(col_index.tz) != 'UTC':
                        logger.warning(f"Converting TZ info on {col_name} from {df.index.tz}")
                        col_index = col_index.tz_convert('UTC')
                        needs_update = True
                else:
                    needs_update = False
                
                # Only reconstruct the MultiIndex if we modified the index
                if needs_update:
                    arrays = []
                    for i, name in enumerate(df.index.names):
                        if i == col_level:
                            arrays.append(col_index)
                        else:
                            arrays.append(df.index.get_level_values(i))
                    df.index = pd.MultiIndex.from_arrays(arrays, names=df.index.names)
        
        # Check if column is a regular column
        elif col_name in df.columns:
            # Check if column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df[col_name]) and convert_to_datetime:
                # Convert to datetime
                df[col_name] = to_datetime(df[col_name])
            elif pd.api.types.is_datetime64_any_dtype(df[col_name]):
                # Already datetime, ensure UTC
                if df[col_name].dt.tz is None:
                    df[col_name] = df[col_name].dt.tz_localize('UTC')
                elif str(df[col_name].dt.tz) != 'UTC':
                    df[col_name] = df[col_name].dt.tz_convert('UTC')
    
    return df


def read_parquet(path, ensure_utc: bool = True, **kwargs) -> pd.DataFrame:
    """Read a parquet file and optionally ensure timezone-aware UTC timestamps.
    
    This is a wrapper around pd.read_parquet that automatically converts
    any 'ts' column or index to timezone-aware UTC.
    
    Args:
        path: String path to the parquet file or file-like object.
        ensure_utc: If True (default), ensures 'ts' is timezone-aware UTC.
                   Set to False to skip timezone conversion for performance
                   when loading many files that will be concatenated.
        **kwargs: Additional keyword arguments passed to pd.read_parquet.
                 Common args include columns, engine, use_pandas_metadata.
        
    Returns:
        DataFrame with timezone-aware UTC 'ts' column or index if ensure_utc=True.
        
    Notes:
        - Preserves all original pd.read_parquet functionality
        - Only modifies 'ts' timestamps to ensure UTC timezone when ensure_utc=True
        - Safe to use as drop-in replacement for pd.read_parquet
        - Use ensure_utc=False when loading many files for concatenation
    """
    df = pd.read_parquet(path, **kwargs)
    if ensure_utc:
        return ensure_utc_ts(df)
    return df


def check_dup_rows(df: pd.DataFrame):
    """Check that a DataFrame has no duplicate rows based on its index.
    
    Args:
        df: DataFrame to check for duplicate rows.
        
    Raises:
        AssertionError: If duplicate rows are found in the DataFrame index.
        
    Notes:
        - Uses pandas index.duplicated() to identify duplicate rows
        - Critical for data integrity in time series data
    """
    dup_rows = df[df.index.duplicated()]
    if len(dup_rows) != 0:
        logger.error(dup_rows.index)
        raise RuntimeError("corrupt af df")


def check_df(df: pd.DataFrame, ignore_index: bool = False):
    """Perform comprehensive validation checks on a DataFrame.
    
    Validates that the DataFrame has no duplicate rows or columns and has the
    expected index structure for trading data.
    
    Args:
        df: DataFrame to validate.
        ignore_index: If True, skips index validation. Defaults to False.
        
    Raises:
        Exception: If duplicate rows or columns are found.
        Exception: If index doesn't match expected ['ts', 'symbol_venue'] structure.
        
    Notes:
        - Expected index is ['ts', 'symbol_venue'] for time series trading data
        - Calls check_dup_rows() and check_dup_cols() for thorough validation
    """

    logger.info(f"Checking dataframe consistency (duplicates and indices) {ignore_index=}")
    check_dup_rows(df)
    check_dup_cols(df)
    if not ignore_index and DF_INDEX != df.index.names:
        print(df.index.names)
        raise log_and_raise(f"bad dataframe index, not {DF_INDEX}")


def check_dup_cols(df: pd.DataFrame):
    """Check that a DataFrame has no duplicate column names.
    
    Args:
        df: DataFrame to check for duplicate columns.
        
    Raises:
        Exception: If duplicate column names are found, listing the duplicates.
        
    Notes:
        - Critical for preventing data corruption when merging DataFrames
        - Prints duplicate column names before raising exception
    """
    dup_cols = df.columns.duplicated()
    if any(dup_cols):
        dupes = sorted(df.columns[dup_cols])
        print(dupes)
        raise log_and_raise(f"Duplicate columns found in dataframe load: {dupes}")


def check_missing_ts(df: pd.DataFrame, freq: Optional[str] = None) -> pd.DatetimeIndex:
    """Identify missing timestamps in a time series DataFrame.
    
    Checks for gaps in the timestamp index by comparing against a complete
    date range at the specified frequency.
    
    Args:
        df: DataFrame with timestamp index to check.
        freq: Pandas frequency string (e.g., 'min', 'H', 'D'). Defaults to 'min'.
        
    Returns:
        pd.DatetimeIndex: DatetimeIndex containing all missing timestamps.
        
    Notes:
        - Useful for detecting data gaps in time series
        - Default frequency is minute-level ('min')
        - Returns empty DatetimeIndex if no gaps found
    """
    min_ts, max_ts = get_min_max_ts(df)
    full_range_ts = pd.date_range(start=min_ts, end=max_ts, freq=freq if freq is not None else 'min')
    missing_ts = full_range_ts.difference(df.index.get_level_values('ts'))
    return missing_ts


def set_index(df: pd.DataFrame, fast: bool = FAST_CONCAT) -> pd.DataFrame:
    """Set the standard ['ts', 'symbol_venue'] MultiIndex on a DataFrame.
    
    Converts a DataFrame with 'ts' and 'symbol_venue' columns into a MultiIndex
    DataFrame, which is the standard format for time series trading data.
    
    Args:
        df: DataFrame with 'ts' and 'symbol_venue' columns.
        fast: If True, skips index integrity verification for performance. Defaults to False.
        
    Returns:
        DataFrame with ['ts', 'symbol_venue'] MultiIndex with UTC timezone-aware timestamps.
        
    Raises:
        Exception: If NaT values found in timestamp column.
        ValueError: If index cannot be set (e.g., duplicate index values).
        
    Notes:
        - Validates that no NaT timestamps exist before indexing
        - With fast=False, verifies index uniqueness (slower but safer)
        - Standard index structure for all trading DataFrames
        - Ensures ts index is timezone-aware UTC
    """
    # Check for NaT values in ts column
    if 'ts' in df.columns and df['ts'].isna().any():
        nat_count = df['ts'].isna().sum()
        sample_rows = df[df['ts'].isna()].head(5)
        logger.error(f"Found {nat_count} NaT values in ts column", key="nat_values_in_ts")
        print(f"Sample rows with NaT timestamps:\n{sample_rows}")
        raise log_and_raise(f"Found {nat_count} NaT values in ts column - cannot set index with invalid timestamps")

    # Ensure ts column is UTC before setting index
    df = ensure_utc_ts(df)

    try:
        df = df.set_index(DF_INDEX)
    except Exception as e:
        raise log_and_raise(f"Could not set index on dataframe! {e}", df=df)

    if not fast and not df.index.is_unique:
        duplicate_count = df.index.duplicated().sum()
        logger.error(f"Duplicate index entries found: {duplicate_count}", key="fail to set ts symbol_venue index for dataframe")
        print(df[df.index.duplicated(keep=False)].head(20))
        raise ValueError(f"Index has {duplicate_count} duplicate entries")

    df = df.sort_index()
    return df


def shrink_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Convert float64 columns to float32 to reduce memory usage.
    
    Automatically finds all float64 columns and converts them to float32,
    which uses half the memory with minimal precision loss for most trading data.
    
    Args:
        df: DataFrame potentially containing float64 columns.
        
    Returns:
        DataFrame with all float64 columns converted to float32.
        
    Notes:
        - Reduces memory usage by ~50% for float columns
        - float32 provides ~7 decimal digits of precision
        - Safe for most price/volume data in trading applications
        - No-op if no float64 columns exist
    """
    cols = df.select_dtypes(np.float64)
    col_names = [str(c) for c in cols.columns]
    if len(col_names) == 0:
        return df
    # logger.info(f"Shrinking columns {col_names}")
    df[cols.columns] = cols.astype(np.float32)
    return df


def safe_int32_cast(series: pd.Series) -> pd.Series:
    """Safely cast a float series to nullable Int32.

    Handles edge cases that cause pandas to fail when casting to Int32:
    - NaN values: replaced with 0
    - Infinity values: replaced with INT32_MAX (positive) or 0 (negative)
    - Values exceeding Int32 range: clipped to [0, INT32_MAX]
    - Float precision issues: resolved by converting through float64 and rounding

    Args:
        series: A pandas Series with float values (typically float32 or float64).

    Returns:
        A pandas Series with nullable Int32 dtype.

    Notes:
        - Designed for count data (trade_cnt, update_cnt) which should be non-negative
        - Clips negative values to 0 since counts cannot be negative
        - Uses nullable Int32 to preserve NaN semantics if needed in downstream processing
    """
    return (series.fillna(0)
            .replace([np.inf, -np.inf], [INT32_MAX, 0])
            .astype('float64').round()
            .clip(lower=0, upper=INT32_MAX).astype('Int32'))


def convert_column_to_categorical(df: pd.DataFrame, fld: str, cat: CategoricalDtype) -> pd.DataFrame:
    """Convert a column to categorical dtype with predefined categories.
    
    Filters the DataFrame to only include rows where the column value is in the
    allowed categories, then converts the column to categorical dtype.
    
    Args:
        df: DataFrame containing the column to categorize.
        fld: Column name to convert to categorical.
        cat: CategoricalDtype with predefined categories.
        
    Returns:
        DataFrame with the specified column converted to categorical dtype.
        
    Notes:
        - Rows with values not in categories are dropped (with warning)
        - Used primarily for symbol/symbol_venue standardization
        - Helps reduce memory usage and enforce data consistency
        - May reduce DataFrame size if invalid categories exist
    """

    pre_cat_count = df.shape[0]
    # filter by categories otherwise it will assign NaN and lead to errors since it's not inside categories
    # it should be only affecting symbol/symbol_venue as we record a dynamic universe rather than fixed categories
    df = df.loc[df[fld].isin(list(cat.categories))]
    post_cat_count = df.shape[0]
    if post_cat_count < pre_cat_count:
        logger.warning(f"Dataframe from {pre_cat_count} -> {post_cat_count} converting {fld} to categorical")
    df[fld] = df[fld].astype(cat)
    return df

def clean_poop(df: pd.DataFrame) -> pd.DataFrame:
    """Remove temporary columns ending with '_poop' suffix.
    
    Cleans up DataFrame by removing columns with '_poop' suffix, which are
    typically created during merge operations as temporary duplicates.
    
    Args:
        df: DataFrame potentially containing '_poop' suffixed columns.
        
    Returns:
        DataFrame with all '_poop' columns removed.
        
    Notes:
        - '_poop' suffix is used as default duplicate suffix in merge_on_index()
        - Safe deletion that won't error if column doesn't exist
    """
    for col in df.columns:
        if col.endswith('_poop'):
            df = safe_del(df=df, fld=col)
    return df


def log_df_mem_usage(df: pd.DataFrame, name: str):
    """Log detailed memory usage statistics for a DataFrame.
    
    Calculates and logs memory usage for each column and total DataFrame size
    in megabytes.
    
    Args:
        df: DataFrame to analyze.
        name: Descriptive name for the DataFrame in log output.
        
    Notes:
        - Uses deep=True for accurate memory calculation of object types
        - Logs both total memory and per-column breakdown
        - Useful for memory optimization and debugging
        - Output in MB for readability
    """
    mem_df = df.memory_usage(deep=True) / 1024 / 1024
    tot = round(mem_df.sum(), 0)
    logger.info(f"Dataframe mem {name}: {tot}MB \n {mem_df}")


def make_symbol_venue(df: pd.DataFrame) -> pd.DataFrame:
    """Create symbol_venue column by combining symbol and venue.
    
    Constructs the standard symbol_venue identifier format by concatenating
    symbol and venue with underscore separator.
    
    Args:
        df: DataFrame with 'symbol' column, optionally 'venue'.

    Returns:
        DataFrame with new/updated 'symbol_venue' column.
        
    Raises:
        KeyError: If 'symbol' column is missing.
        
    Notes:
        - Format: '{symbol}_{venue}' (e.g., 'BTCUSDT_binance-futures')
        - If 'venue' missing, defaults to TARDIS_EXCHANGE
        - Standard identifier format used throughout the system
    """


    if 'venue' not in df.columns:
        df['venue'] = TARDIS_EXCHANGE

    try:
        df['symbol_venue'] = (df['symbol'].astype(str) + '_' + df['venue'].astype(str))
    except KeyError as ke:
        logger.error(ke, key="fail to make symbol venue")
        print(df)
        raise
    return df


def make_aux_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add auxiliary fields commonly needed for trading data analysis.
    
    Creates symbol_venue, date, and time fields, then sets standard index.
    
    Args:
        df: DataFrame with basic trading data.
        
    Returns:
        DataFrame with auxiliary fields added and standard index set.
        
    Notes:
        - Creates symbol_venue identifier
        - Extracts date (with 1-minute offset for midnight handling)
        - Extracts time component for intraday analysis
        - Sets standard ['ts', 'symbol_venue'] MultiIndex
    """
    df = make_symbol_venue(df)
    # a bit of a hack, but since the bars end on midnight the next day without this it gets included in the next day file
    df = make_date(df)
    df['time'] = df['ts'].dt.time
    df = set_index(df)
    return df


def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns to single-level with underscore separator.
    
    Converts hierarchical column names to flat strings by joining levels
    with underscores.
    
    Args:
        df: DataFrame with MultiIndex columns.
        
    Returns:
        DataFrame with single-level column names.
        
    Notes:
        - Useful after pivot/groupby operations that create MultiIndex columns
        - Example: ('price', 'mean') becomes 'price_mean'
        - Preserves all information in a flat structure
    """
    df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df.columns]
    return df


def make_date(df: pd.DataFrame, no_day_boundary_shift: bool = False) -> Optional[pd.DataFrame]:
    """Extract date from timestamp with optional 1-minute offset for proper day assignment.

    Creates a 'date' column from timestamp data. By default, subtracts 1 minute to handle
    bars that end at midnight (which should belong to the previous day).

    Args:
        df: DataFrame with timestamp data (column or index).
        no_day_boundary_shift: If True, skip the 1-minute shift. Use this for data
            timestamped at 00:00:00 that should NOT be shifted to the previous day
            (e.g., portfolio snapshots, position data).

    Returns:
        DataFrame with 'date' column added.

    Notes:
        - Skips if valid 'date' column already exists
        - Default behavior subtracts 1 minute before normalizing (for bar data)
        - Set no_day_boundary_shift=True for portfolio/position data at midnight
        - Works with both column and index timestamps
        - Critical for proper daily aggregations
    """
    if df is None:
        logger.warning("Null dataframe passed to make_date, skipping")
        return None

    if df.empty:
        logger.warning("Empty dataframe passed to make_date, skipping")
        return df

    if 'date' in df.columns and not any(df['date'].isna()):
        return df

    if 'ts' in df.columns:
        if no_day_boundary_shift:
            df['date'] = df['ts'].dt.normalize()
        else:
            df['date'] = (df['ts'] - td(minutes=1)).dt.normalize()
    else:
        # Get timestamps and ensure it's a DatetimeIndex for normalize()
        ts_values = df.index.get_level_values('ts')
        if no_day_boundary_shift:
            df['date'] = to_datetime(ts_values).normalize()
        else:
            ts_shifted = to_datetime(ts_values - td(minutes=1))
            df['date'] = ts_shifted.normalize()
    return df


def make_symbol(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Extract symbol from symbol_venue column.
    
    Parses the symbol component from symbol_venue format by splitting on
    underscore and taking the first part.
    
    Args:
        df: DataFrame with symbol_venue data (column or index).
        
    Returns:
        DataFrame with 'symbol' column added.
        
    Raises:
        Exception: If neither symbol nor symbol_venue found.
        
    Notes:
        - Skips if valid 'symbol' column already exists
        - Extracts from 'BTCUSDT_binance-futures' -> 'BTCUSDT'
        - Works with both column and index symbol_venue
    """
    if df is None:
        return None
    if 'symbol' in df.columns and not any(df['symbol'].isna()):
        return df
    if df.empty:
        # Add empty symbol column with correct dtype for empty DataFrames
        df['symbol'] = pd.Series(dtype=str)
        return df
    if 'symbol_venue' in df.columns:
        df['symbol'] = df['symbol_venue'].astype(str).str.split('_').str[0]
    elif 'symbol_venue' in df.reset_index():
        df['symbol'] = df.index.get_level_values('symbol_venue').astype(str).str.split('_').str[0]
    else:
        raise log_and_raise("No symbol in dataframe")
    return df


def make_quintile(df: pd.DataFrame, col: str, q: int = 5) -> pd.DataFrame:
    """Create quintile buckets for a numeric column.
    
    Divides a column into q equal-sized buckets based on quantiles,
    useful for cross-sectional analysis and ranking.
    
    Args:
        df: DataFrame containing the column to bucket.
        col: Column name to create quintiles for.
        q: Number of quantiles to create. Defaults to 5 (quintiles).
        
    Returns:
        DataFrame with new '{col}_quintile' column added.
        
    Raises:
        Exception: If specified column doesn't exist.
        
    Notes:
        - Uses pd.qcut for equal-frequency buckets
        - Handles duplicates with 'drop' to avoid bin edge issues
        - Warns if fewer unique values than requested quintiles
        - New column name format: '{original}_quintile'
    """
    if col not in df.columns:
        raise log_and_raise(f"{col} not in dataframe!")
    if df[col].nunique() < q:
        logger.warning(f"{col} only has {df[col].nunique()} unique values")
    df[f'{col}_quintile'] = pd.qcut(df[col], q, duplicates='drop')

    return df


def safe_get_from_series(df: pd.Series, fld: str) -> Any:
    """Safely retrieve a value from a Series without raising KeyError.
    
    Args:
        df: Pandas Series to retrieve from.
        fld: Key/index to look up.
        
    Returns:
        Value at the specified key, or None if not found.
        
    Notes:
        - Logs warning when key not found
        - Useful for optional field access
        - Returns None instead of raising exception
    """
    try:
        return df[fld]
    except KeyError:
        logger.warning(f'there is no {fld} inside the series')
        return None


def safe_get(df: pd.DataFrame, fld: str) -> Optional[pd.Series]:
    """Safely retrieve a column from a DataFrame without raising KeyError.
    
    Args:
        df: DataFrame to retrieve column from.
        fld: Column name to look up.
        
    Returns:
        Series containing the column data, or None if not found.
        
    Notes:
        - Logs warning when column not found
        - Useful for optional column access
        - Returns None instead of raising exception
    """
    try:
        return df[fld]
    except KeyError:
        logger.warning(f'there is no {fld} inside the df')
        return None


def round_col_to_zero(df: pd.DataFrame, fld: str, tol: float = 1e-10) -> pd.DataFrame:
    """Round a column to zero if its absolute value is below a tolerance.

    Args:
        df: DataFrame containing the column to round.
        fld: Column name to round.
        tol: Tolerance level below which values are set to zero. Defaults to 1e-10.

    Returns:
        DataFrame with specified column rounded to zero where applicable.

    Notes:
        - Uses vectorized operations for better performance on large DataFrames
        - Useful for cleaning up small numerical errors in calculations
        - Does not raise an error if column does not exist
    """
    try:
        mask = df[fld].abs() < tol
        df.loc[mask, fld] = np.float32(0.0)
    except Exception:
        pass
    return df


def safe_del(df: pd.DataFrame, fld: str) -> pd.DataFrame:
    """Safely delete a column from a DataFrame without raising exceptions.
    
    Args:
        df: DataFrame to delete column from.
        fld: Column name to delete.
        
    Returns:
        DataFrame with column removed (if it existed).
        
    Notes:
        - Silent operation - no error if column doesn't exist
        - Modifies DataFrame in-place
        - Useful for cleanup operations where column may or may not exist
    """
    try:
        del df[fld]
    except Exception:
        pass
    return df


def remove_infs(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Replace infinite values with NaN in a DataFrame.
    
    Args:
        df: DataFrame potentially containing infinite values.
        
    Returns:
        DataFrame with all inf and -inf values replaced by NaN.
        
    Notes:
        - Handles both positive and negative infinity
        - Useful for numerical stability in calculations
        - NaN values can be handled by subsequent fillna operations
    """
    return df.replace([np.inf, -np.inf], np.nan)


def log_col_summary(df: pd.DataFrame, col: str):
    """Log comprehensive statistical summary for a DataFrame column.
    
    Logs key statistics including time range, missing values, min/max,
    standard deviation, and mean absolute value.
    
    Args:
        df: DataFrame containing the column.
        col: Column name to summarize.
        
    Notes:
        - Includes timestamp range if DataFrame has ts index
        - Shows NA count and percentage
        - Formats numbers to 4 decimal places
        - Useful for data quality checks and debugging
    """
    try:
        min_ts, max_ts = get_min_max_ts(df)
        min_max_ts = f" {dt_to_str(min_ts)}-{dt_to_str(max_ts)}"
    except KeyError:
        min_max_ts = ""
    na_cnt = len(df[df[col].isna()])
    zero_cnt = len(df[df[col] == 0])
    tot_cnt = len(df)
    if zero_cnt == tot_cnt or na_cnt == tot_cnt:
        logger.error(f"Column {col} has {zero_cnt=} or {na_cnt=} equal to {tot_cnt=} which is not expected")
    logger.info(f"{col}{min_max_ts}, NAs {na_cnt}/{tot_cnt} Zeros {zero_cnt} min={df[col].min():.4f} max={df[col].max():.4f} std={df[col].std():.4f} absize={df[col].abs().mean():.4f}")


def concat(dfs: List[pd.DataFrame], fast: bool = FAST_CONCAT, quiet: bool = False, check: bool = False, strict: bool = False) -> Optional[pd.DataFrame]:
    """Concatenate a list of DataFrames with consistent dtype handling.
    
    Attempts to standardize dtypes across all DataFrames before concatenation
    to avoid type conflicts. Also ensures datetime columns are timezone-aware
    to prevent concatenation warnings.
    
    Args:
        dfs: List of DataFrames to concatenate.
        fast: If True, skips integrity checks for performance. Defaults to False.
        quiet: If True, suppresses warning messages. Defaults to False.
        check: If True, validates result with check_df(). Defaults to False.
        
    Returns:
        Concatenated DataFrame, or None if empty list provided.
        
    Raises:
        ValueError: If concatenation fails due to incompatible data.
        
    Notes:
        - Ensures all datetime columns are timezone-aware UTC
        - Attempts to convert all DataFrames to dtype of first DataFrame
        - Logs column differences when dtype conversion fails
        - With fast=False, verifies no duplicate indices
        - Returns None for empty input list
    """
    df_cnt = len(dfs)
    logger.debug(f"Concatenating {df_cnt} dataframes...")
    if df_cnt == 0:
        logger.warning("Can't concat 0 dataframes...")
        return None

    first_df = dfs[0]
    dtypes = first_df.dtypes
    # if len(first_df) > 0:
    #     logger.info(f"Using dtypes from {first_df.index[0]} - {first_df.index[-1]}")
    # else:
    #     logger.info(f"Using dtypes from empty dataframe")

    missing_col_dfs = {}  # Track which dataframes are missing which columns

    # logger.info(f"Checking datatypes before concatenation...")
    # Single loop to handle both dtype conversion and timezone normalization
    for i in range(len(dfs)):
        df = dfs[i]
        
        # First, try to convert to consistent dtypes
        try:
            # Check if dtypes match, if not we need to convert
            if not df.dtypes.equals(dtypes):
                # Skip timezone-aware datetime columns in dtype conversion
                # as they need special handling
                cols_to_convert = []
                missing_cols = []
                
                # Check all columns in current dataframe
                for col in df.columns:
                    if col not in dtypes.index:
                        # Track missing column but continue processing
                        missing_cols.append(col)
                    else:
                        # Skip datetime columns with timezone info
                        if (pd.api.types.is_datetime64_any_dtype(df[col]) and hasattr(df[col], 'dt') and df[col].dt.tz is not None) or \
                           isinstance(dtypes[col], pd.DatetimeTZDtype):
                            continue
                        if df[col].dtype != dtypes[col]:
                            cols_to_convert.append(col)
                
                # Track missing columns for this dataframe
                if missing_cols:
                    for col in missing_cols:
                        if col not in missing_col_dfs:
                            missing_col_dfs[col] = []
                        missing_col_dfs[col].append(i)
                
                # Convert columns in place (only those that exist in both)
                for col in cols_to_convert:
                    new_type = dtypes[col]
                    if not quiet:
                        logger.warning(f"Converting column '{col}' from {dfs[i][col].dtype} to {new_type} in dataframe {i}")
                        if strict:
                            print(dfs[i])
                            raise ValueError(f"Column '{col}' has incorrect dtype {dfs[i][col].dtype}, expected {new_type}")

                    if str(new_type).lower().startswith('int'):
                        # Convert float → int, handling inf/-inf/nan
                        cleaned = remove_infs(dfs[i][col]).fillna(0)

                        # Check if it's a pandas nullable int type (Int32) or numpy int type (int32)
                        if hasattr(new_type, 'numpy_dtype'):
                            # Pandas nullable integer type - convert to numpy equivalent
                            target_dtype = new_type.numpy_dtype
                        else:
                            # Already a numpy type
                            target_dtype = new_type

                        # Direct assignment to change column dtype (not .loc which preserves dtype)
                        dfs[i][col] = cleaned.astype(target_dtype)
                    else:
                        dfs[i][col] = dfs[i][col].astype(new_type)
                    
        except Exception as e:
            logger.warning(f"Unknown exception changing dtype of dataframes {e}, moving on...")
            if strict:
                raise

        # Ensure 'ts' column/index is UTC (modifies in place, skip datetime conversion for speed)
        ensure_utc_ts(dfs[i], convert_to_datetime=False)
    
    # Log warnings about missing columns after processing all dataframes
    if missing_col_dfs and not quiet:
        for col, df_indices in missing_col_dfs.items():
            logger.warning(f"Column '{col}' not found in reference dataframe but exists in {len(df_indices)} dataframe(s) at indices: {df_indices[:10]}...")

    logger.info(f"Concatenating {len(dfs)} dataframes... {fast=}")
    try:
        df = pd.concat(dfs, verify_integrity=not fast)
    except ValueError as ve:
        logger.error(f"Could not concat dataframes... {ve}", key="fail to concat dataframes")
        raise

    if check:
        check_df(df)
    return df


def long_short_sums(df: pd.DataFrame, fld: str) -> Tuple[float, float]:
    """Calculate separate sums for positive and negative values in a column.
    
    Useful for analyzing long/short positions or directional components.
    
    Args:
        df: DataFrame containing the data.
        fld: Column name to sum by sign.
        
    Returns:
        Tuple of (long_sum, short_sum) where long_sum is sum of positive values
        and short_sum is sum of negative values.
        
    Notes:
        - Zero values are excluded from both sums
        - Common use: position analysis, PnL attribution
        - Returns (0, 0) if column contains no positive/negative values
    """
    long = df[df[fld] > 0][fld].sum()
    short = df[df[fld] < 0][fld].sum()
    return long, short


def merge_on_index(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        how: Literal["left", "right", "inner", "outer", "cross"] = 'left',
        suffixes: Optional[Tuple[str, str]] = None,
        debug: bool = False
) -> pd.DataFrame:
    """Merge two DataFrames on their indices with automatic cleanup.
    
    Performs index-based merge with optional duplicate column handling
    using '_poop' suffix for automatic cleanup.
    
    Args:
        df1: Left DataFrame to merge.
        df2: Right DataFrame to merge.
        how: Type of merge to perform. Defaults to 'left'.
        suffixes: Tuple of suffixes for overlapping columns. If None, uses ('', '_poop').
        debug: If True, logs row counts before and after merge. Defaults to False.
        
    Returns:
        Merged DataFrame with duplicate columns cleaned up if default suffixes used.
        
    Notes:
        - Default suffixes ('', '_poop') trigger automatic cleanup
        - Debug mode shows data flow for troubleshooting
        - Index alignment is critical for correct results
    """
    clean = False
    if suffixes is None:
        suffixes = ('', '_poop')
        clean = True
    if debug:
        left_cnt = len(df1)
        right_cnt = len(df2)

    # Convert Series to DataFrame for consistent handling
    if isinstance(df2, pd.Series):
        df2 = df2.to_frame()

    # Handle empty dataframe cases - let pd.merge handle most cases natively
    # Only special-case when we need to add columns from empty df to non-empty df
    if df1.empty or df2.empty:
        # For left join with df2 empty but df1 not empty: add df2 columns to df1
        if how == 'left' and df2.empty and not df1.empty:
            ret_df = df1.copy()
            for col in df2.columns:
                if col not in ret_df.columns:
                    ret_df[col] = np.nan
            if clean:
                ret_df = clean_poop(ret_df)
            return ret_df
        # For right join with df1 empty but df2 not empty: add df1 columns to df2
        if how == 'right' and df1.empty and not df2.empty:
            ret_df = df2.copy()
            for col in df1.columns:
                if col not in ret_df.columns:
                    ret_df[col] = np.nan
            if clean:
                ret_df = clean_poop(ret_df)
            return ret_df
        # For outer join: add missing columns from either side
        if how == 'outer':
            if df1.empty and df2.empty:
                all_cols = list(df1.columns) + [c for c in df2.columns if c not in df1.columns]
                ret_df = pd.DataFrame(columns=all_cols)
            elif df1.empty:
                ret_df = df2.copy()
                for col in df1.columns:
                    if col not in ret_df.columns:
                        ret_df[col] = np.nan
            else:  # df2.empty
                ret_df = df1.copy()
                for col in df2.columns:
                    if col not in ret_df.columns:
                        ret_df[col] = np.nan
            if clean:
                ret_df = clean_poop(ret_df)
            return ret_df
        # For all other cases (inner, left with df1 empty, right with df2 empty),
        # let pd.merge handle it - it will return empty df with proper columns

    try:
        ret_df = pd.merge(df1, df2, left_index=True, right_index=True, how=how, suffixes=suffixes)
    except ValueError as ve:
        df1.head().to_parquet("df1.parquet")
        df2.head().to_parquet("df2.parquet")
        raise log_and_raise(f"Cannot merge dataframes: {ve}")

    if clean:
        ret_df = clean_poop(ret_df)
    if debug:
        final_cnt = len(ret_df)
        logger.info(f"Merging dataframes left:{left_cnt}, right:{right_cnt} -> {final_cnt}")
    return ret_df


def df_style(val: Any) -> str:
    """Return bold font style for DataFrame styling.
    
    Args:
        val: Cell value (unused, required by pandas styling API).
        
    Returns:
        str: CSS style string for bold font.
        
    Notes:
        - Used with DataFrame.style.map() for visual formatting
        - Makes selected cells bold in Jupyter notebooks
    """
    return "font-weight: bold"


def highlight_df_rows(df: pd.DataFrame, rows: List[Any]) -> None:
    """Apply bold styling to specific rows in a DataFrame.
    
    Args:
        df: DataFrame to style.
        rows: Row indices or labels to highlight.
        
    Notes:
        - Modifies DataFrame styling in-place
        - Only visible in Jupyter notebooks or HTML output
        - Uses df_style() to apply bold font
    """
    df.style.map(df_style, subset=rows)


def unstack_feature(features_df: pd.DataFrame, feature: str, start: Optional[dt] = None) -> pd.DataFrame:
    """Pivot a feature from long to wide format using symbol_venue as columns.
    
    Transforms time series data from MultiIndex format to wide format with
    timestamps as rows and symbol_venues as columns.
    
    Args:
        features_df: MultiIndex DataFrame with ['ts', 'symbol_venue'] index.
        feature: Column name to unstack.
        start: Optional start time to filter data. Defaults to None.
        
    Returns:
        DataFrame with timestamps as index and symbol_venues as columns.
        
    Raises:
        Exception: If timestamps are invalid or unstacking fails.
        
    Notes:
        - Commonly used for cross-sectional analysis
        - Filters to timezone-naive start time if provided
        - Logs time range being unstacked
        - Shows duplicate indices if unstacking fails
    """
    if start is not None:
        features_df = features_df[features_df.index.get_level_values('ts') >= start]

    min_ts, max_ts = get_min_max_ts(features_df)
    if pd.isna(min_ts) or pd.isna(max_ts):
        raise log_and_raise(f"Bad NA timestamps for {feature=} in features {min_ts=} {max_ts=}", df=features_df)

    logger.info(f"Unstacking {feature} from {min_ts} to {max_ts}")
    try:
        df = features_df[[feature]].unstack()
    except ValueError as ve:
        raise log_and_raise(f"Could not unstack {feature}: {ve}", df=features_df[features_df.index.duplicated()])
    return df


def carry_forward(df: pd.DataFrame, flds: List[str]) -> pd.DataFrame:
    """Forward fill specified columns preserving MultiIndex structure.
    
    Performs forward filling on MultiIndex DataFrames by unstacking,
    filling, and restacking to maintain proper time series continuity.
    
    Args:
        df: MultiIndex DataFrame to forward fill.
        flds: List of column names to forward fill.
        
    Returns:
        DataFrame with specified columns forward filled.
        
    Notes:
        - Logs last valid value and timestamp for each field
        - Skips missing columns with error logging
        - Uses unstack/stack to handle MultiIndex properly
        - Applies float shrinking for memory efficiency
        - future_stack=True maintains compatibility
    """
    _, max_ts = get_min_max_ts(df)
    for fld in flds:
        if fld not in df.columns:
            logger.error(f"{fld} not in dataframe, can't carry forward, should fix", key="fail to carry forward")
            continue
        logger.info(f"Carrying forward {fld}")
        last_valid_idx = df[fld].replace({None: np.nan}).last_valid_index()
        if last_valid_idx is None:
            logger.error(f"{fld} has no valid index to carry forward!")

        last_valid_value = df.loc[last_valid_idx, fld]
        logger.info(f"Carrying forward {fld} with {last_valid_value=} from {last_valid_idx=} to {max_ts}")
        df[fld] = df[[fld]].unstack().ffill().stack(future_stack=True)
    df = shrink_floats(df)
    return df


def get_min_max_ts(df: pd.DataFrame) -> Tuple[dt, dt]:
    """Get minimum and maximum timestamps from DataFrame index.
    
    Args:
        df: DataFrame with 'ts' level in MultiIndex.
        
    Returns:
        Tuple of (min_timestamp, max_timestamp) always timezone-aware in UTC.
        
    Raises:
        KeyError: If 'ts' not found in index levels.
        
    Notes:
        - Assumes MultiIndex with 'ts' level
        - Always returns timezone-aware pandas Timestamp objects in UTC
        - Useful for determining data time range
        - Handles mixed timezone-aware and timezone-naive timestamps
    """
    lvs = df.index.get_level_values('ts')
    
    # Handle potential mixed timezone awareness by converting to UTC if needed
    if isinstance(lvs, pd.DatetimeIndex):
        # Check if we have mixed timezone awareness
        if lvs.tz is None:
            # All naive - try to localize to UTC
            try:
                lvs = lvs.tz_localize('UTC')
            except TypeError:
                # Mixed tz-aware and tz-naive - convert to strings then back to UTC
                lvs = to_datetime(lvs.astype(str))
        elif str(lvs.tz) != 'UTC':
            # Has timezone but not UTC - convert to UTC
            lvs = lvs.tz_convert('UTC')
    else:
        # Not a DatetimeIndex - check if it contains datetime objects
        try:
            # Try to get min/max directly
            min_ts, max_ts = lvs.min(), lvs.max()
            # Ensure timezone-aware
            return to_datetime(min_ts), to_datetime(max_ts)
        except TypeError as e:
            if "Cannot compare tz-naive and tz-aware" in str(e):
                # Mixed timezone awareness in a regular Index
                # Convert to DatetimeIndex with UTC
                lvs = to_datetime(lvs)
            else:
                raise
    
    # Get min/max timestamps
    min_ts, max_ts = lvs.min(), lvs.max()
    
    # Ensure both timestamps are timezone-aware in UTC
    # to_datetime handles both naive and aware timestamps correctly
    min_ts = to_datetime(min_ts)
    max_ts = to_datetime(max_ts)
    
    return min_ts, max_ts


def remove_1_min_cols(df: pd.DataFrame, skip: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove '_1' suffix from column names for backward compatibility.
    
    Handles transition from suffixed to non-suffixed column naming convention
    by removing '_1' suffix where safe to do so.
    
    Args:
        df: DataFrame with potentially suffixed columns.
        skip: List of column names to preserve with suffix. Defaults to None.
        
    Returns:
        DataFrame with '_1' suffixes removed where appropriate.
        
    Notes:
        - Only renames if target name doesn't already exist
        - Warns if rename would create duplicate column
        - Preserves columns in skip list
        - Common use: bar data compatibility
    """
    # remove the _1 b/c the current bar files won't have that suffix
    skip = [] if skip is None else skip
    for col in list(df.columns):
        if col.endswith('_1') and col not in skip:
            new_col = col[:-2]
            if new_col not in df.columns:
                df.rename(columns={col: new_col}, inplace=True)
            else:
                logger.warning(f"{new_col} already in dataframe!")
    return df


def rolling_agg_ignore_internal_nan(df: pd.DataFrame, window_mins: int, method: str) -> pd.DataFrame:
    """Calculate rolling aggregation that preserves trailing NaN patterns.
    
    Performs rolling window calculations while maintaining NaN values at the
    end of series, which is important for handling incomplete data.
    
    Args:
        df: DataFrame with time series data.
        window_mins: Window size in minutes.
        method: Aggregation method ('sum', 'mean', 'std', etc.).
        
    Returns:
        DataFrame with rolling aggregations applied, preserving trailing NaNs.
        
    Notes:
        - Uses time-based rolling window ('{window_mins}min')
        - min_periods=1 allows partial window calculations
        - Masks results where original data is NaN
        - Returns float32 for memory efficiency
    """
    ret_df = pd.DataFrame(columns=df.columns, index=df.index, dtype=np.float32)
    for col in df.columns:
        rolling_agg = df[col].rolling(f"{window_mins}min", min_periods=1).agg(method)
        ret_df[col] = rolling_agg.mask(df[col].isna()).astype(np.float32)

    return ret_df


def get_trailing_nan_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Create boolean mask identifying trailing NaN values in each column.
    
    Identifies consecutive NaN values at the end of each time series,
    which represent missing future data rather than gaps.
    
    Args:
        df: DataFrame to analyze for trailing NaNs.
        
    Returns:
        Boolean DataFrame where True indicates a trailing NaN.
        
    Notes:
        - Works column-wise to handle different end points
        - Uses reverse cumsum trick to identify trailing sequences
        - Critical for proper handling of incomplete data
        - Preserves non-trailing NaNs as False
    """
    # Find NaN values and reverse to work from end
    is_nan = df.isna()
    rev_is_nan = is_nan.iloc[::-1]
    # Create groups that break at first non-NaN
    groups = (~rev_is_nan).cumsum()
    # Get mask for consecutive end NaNs
    end_nan_mask = (groups == 0) & rev_is_nan
    end_nan_mask = end_nan_mask.iloc[::-1]
    return end_nan_mask


def check_df_column_changes(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """Log column differences between two DataFrames.
    
    Compares column sets and logs any new columns added in df2.
    
    Args:
        df1: Original DataFrame.
        df2: New DataFrame to compare.
        
    Notes:
        - Only logs additions, not deletions
        - Useful for tracking schema evolution
        - No return value, only logging side effect
    """
    old_cols = set(df1.columns)
    new_cols = set(df2.columns)
    added_cols = new_cols - old_cols
    if len(added_cols) > 0:
        logger.info(f"Adding columns {added_cols}")


def has_sufficient_valid_data(file_name: str, valid_col: str = 'close_mid', min_valid_bar_data: int = MIN_VALID_BAR_DATA) -> bool:
    """Check if a parquet file has sufficient non-null data in specified column.
    
    Validates data quality by ensuring minimum number of valid (non-null) values
    exist in a critical column, typically used for bar data validation.
    
    Args:
        file_name: Path to parquet file to check.
        valid_col: Column name to check for valid data. Defaults to 'close_mid'.
        min_valid_bar_data: Minimum required non-null values. Defaults to MIN_VALID_BAR_DATA (1435).
        
    Returns:
        True if sufficient valid data exists, False otherwise.
        
    Notes:
        - Efficient partial file read (only specified column)
        - Default threshold allows up to 5 missing minutes in a day
        - Logs details when data is insufficient
        - Common use: validating bar files before processing
    """
    df = pd.read_parquet(file_name, columns=[valid_col])
    num_valid_bar_data = df[valid_col].notnull().sum()
    if num_valid_bar_data < min_valid_bar_data:
        logger.info(f"Not enough valid data {num_valid_bar_data=} at {valid_col} from {file_name}")
        return False
    return True


def get_last_timeslice(df: pd.DataFrame) -> pd.DataFrame:
    """Extract all rows from the latest timestamp in a MultiIndex DataFrame.
    
    Args:
        df: MultiIndex DataFrame with 'ts' level.
        
    Returns:
        DataFrame containing only rows from the maximum timestamp.
        
    Notes:
        - Preserves MultiIndex structure
        - Returns all symbols for the latest time
        - Useful for getting current positions/values
    """
    return df[df.index.get_level_values('ts') == df.index.get_level_values('ts').max()]


def convert_to_dict(df: pd.DataFrame, key_col: str, val_col: str) -> dict:
    """Convert two DataFrame columns to a dictionary mapping.
    
    Args:
        df: DataFrame containing the data.
        key_col: Column name to use as dictionary keys.
        val_col: Column name to use as dictionary values.
        
    Returns:
        Dictionary mapping key_col values to val_col values.
        
    Notes:
        - Automatically handles index columns by resetting if needed
        - Last value wins if duplicate keys exist
        - Useful for lookups and configuration mappings
    """
    if key_col in df.index.names or val_col in df.index.names:
        df = df.reset_index()
    return dict(zip(df[key_col], df[val_col]))


def clip_col_by_iqr(df: pd.DataFrame, clip_col: str) -> pd.DataFrame:
    """Clip outliers in a column using Interquartile Range (IQR) method.
    
    Applies standard IQR outlier detection to cap extreme values at
    1.5 * IQR beyond the first and third quartiles.
    
    Args:
        df: DataFrame containing the column to clip.
        clip_col: Column name to apply outlier clipping to.
        
    Returns:
        DataFrame with specified column clipped to remove outliers.
        
    Notes:
        - Uses 1.5 * IQR rule (standard for box plots)
        - Modifies column in-place
        - Preserves data distribution while removing extremes
        - Lower bound: Q1 - 1.5 * IQR
        - Upper bound: Q3 + 1.5 * IQR
    """
    Q1 = df[clip_col].quantile(0.25)
    Q3 = df[clip_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[clip_col] = df[clip_col].clip(lower=lower_bound, upper=upper_bound)
    return df


def make_daily_carry_forward_values(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Add columns with first value per symbol_venue and date group.
    
    For each specified column, creates a new column containing the first
    value within each symbol_venue/date group.
    
    Args:
        df: DataFrame with symbol_venue and date columns.
        cols: List of column names to get first values for.
        
    Returns:
        DataFrame with new 'first_{col}' columns added.
        
    Notes:
        - New columns named as 'first_{original_column_name}'
        - Uses observed=False to handle all categorical levels
        - Useful for open prices, initial conditions
        - Values repeated for all rows in the group
    """
    grouper = df.groupby(['symbol_venue', 'date'], observed=False)
    for col in cols:
        df[f'first_{col}'] = grouper[col].transform('first')
    return df


def compare_dataframes(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        name1: str,
        name2: str,
        case: str,
        num_case: int = 3,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        show_diff_desc: bool = False,
) -> Tuple[bool, str, str]:
    """Compare two DataFrames and report differences in detail.
    
    Performs comprehensive comparison of DataFrames including shape, columns,
    data types, and values. Provides detailed difference reports for debugging.
    
    Args:
        df1: First DataFrame to compare.
        df2: Second DataFrame to compare.
        name1: Name for first DataFrame in reports.
        name2: Name for second DataFrame in reports.
        case: Description of the comparison case.
        num_case: Number of difference examples to show. Defaults to 3.
        rtol: Relative tolerance for numeric comparisons. Defaults to 1e-05.
        atol: Absolute tolerance for numeric comparisons. Defaults to 1e-08.
        show_diff_desc: If True, shows statistical description of differences.
        
    Returns:
        Tuple of (match_result, message, column_differences):
        - match_result: True if DataFrames match within tolerances
        - message: Detailed comparison report
        - column_differences: Comma-separated list of differing columns
        
    Notes:
        - Ignores 'file_created_*ts' columns
        - Converts log returns to basis points for display
        - Uses numpy.allclose for numeric comparisons
        - Shows difference percentages and examples
    """
    file_created_cols = [c for c in df1.columns if c.startswith('file_created_') and c.endswith('ts')]
    df1 = df1.drop(columns=file_created_cols, errors='ignore')
    file_created_cols = [c for c in df2.columns if c.startswith('file_created_') and c.endswith('ts')]
    df2 = df2.drop(columns=file_created_cols, errors='ignore')

    msg = f"\n{case}:"
    cols_compare = ""
    res = True
    if df1.equals(df2):
        msg = f"\n{case}: {name1} and {name2} are identical."
        return res, msg, cols_compare

    if df1.shape[0] != df2.shape[0]:
        res = False
        msg += f"\nShape mismatch: {name1} {df1.shape} vs {name2} {df2.shape}"
        return res, msg, cols_compare

    # Log start of column-by-column comparison
    logger.info(f"\nStarting column-by-column comparison for {case}")
    logger.info(f"Comparing {len(df1.columns)} columns between {name1} and {name2}")
    
    for column in df1.columns:
        if column not in df2.columns:
            logger.info(f"Column '{column}': MISMATCH - exists in {name1} but not in {name2}")
            msg += f"\nColumn mismatch: '{column}' exists in {name1} but not in {name2}"
            res = False
            continue

        # Start checking this column
        column_match = True
        column_issues = []

        if df1[column].dtype != df2[column].dtype:
            column_match = False
            column_issues.append(f"dtype mismatch: {df1[column].dtype} vs {df2[column].dtype}")
            res = False
            msg += f"\nData type mismatch in column '{column}': {name1} {df1[column].dtype}, {name2} {df2[column].dtype}"

        if not df1[column].equals(df2[column]):
            if pd.api.types.is_numeric_dtype(df1[column]) and not pd.api.types.is_bool_dtype(df1[column]):
                if not np.allclose(df1[column], df2[column], rtol=rtol, atol=atol, equal_nan=True):
                    column_match = False
                    column_issues.append("numeric values differ beyond tolerance")
                    cols_compare += f'{len(df1.index.names) + df1.columns.get_loc(column)},{column},'
                    msg += f"\nNumeric Value Differences in column: {column}"
                    res = False
                    diff_df = pd.merge(df1[[column]], df2[[column]], left_index=True, right_index=True, suffixes=(f"_{name1}", f"_{name2}"))
                    # convert to bps for logret_ diff compare since they are mostly pretty small and hard to see when print out
                    if 'logret' in column:
                        diff_df = diff_df * 10000.0
                    diff_df['diff(%)'] = (diff_df[f"{column}_{name1}"] - diff_df[f"{column}_{name2}"]).abs() / diff_df[f"{column}_{name1}"] * 100
                    diff_df.loc[(diff_df[f"{column}_{name1}"] == 0) & (diff_df[f"{column}_{name2}"] != 0), 'diff(%)'] = 100
                    msg += "\n" + "-" * 150
                    msg += f"\n{diff_df.sort_values('diff(%)', ascending=False).head(num_case).to_markdown(floatfmt=',.2f')}"
                    msg += "\n" + "-" * 150
                    if show_diff_desc:
                        msg += f"\n{diff_df['diff(%)'].describe().to_markdown(floatfmt=',.2f')}"
                        msg += "\n" + "-" * 150
                else:
                    # Values are close enough within tolerance
                    column_issues.append("numeric values match within tolerance")
            elif pd.api.types.is_datetime64_dtype(df1[column]):
                column_match = False
                column_issues.append("datetime values differ")
                msg += f"\nDatetime Value Differences in column: {column}"
            else:
                column_match = False
                column_issues.append("non-numeric values differ")
                cols_compare += f'{len(df1.index.names) + df1.columns.get_loc(column)},{column}'
                msg += f"\nNon-numeric Value Differences in column: {column}"
                res = False
                diff_mask = df1[column] != df2[column]
                diff_indices = diff_mask[diff_mask].index[:num_case]
                diff_df = pd.merge(df1[[column]], df2[[column]], left_index=True, right_index=True, suffixes=(f"_{name1}", f"_{name2}"))
                msg += "\n" + "-" * 150
                msg += f"\n{diff_df.loc[diff_indices].to_markdown()}"
                msg += "\n" + "-" * 150
                if show_diff_desc:
                    msg += f"\n{diff_df[[f'{column}_{name1}', f'{column}_{name2}']].describe().to_markdown()}"
                    msg += "\n" + "-" * 150

        nulls1 = df1[column].isnull().sum()
        nulls2 = df2[column].isnull().sum()
        if nulls1 != nulls2:
            column_match = False
            column_issues.append(f"null count mismatch: {nulls1} vs {nulls2}")
            res = False
            msg += f"\nNull value count in '{column}': {name1} {nulls1}, {name2} {nulls2}\n"
        
        # Log the result for this column
        if column_match and not column_issues:
            logger.info(f"Column '{column}': MATCH")
        else:
            if column_issues:
                logger.info(f"Column '{column}': MISMATCH - {'; '.join(column_issues)}")
            else:
                logger.info(f"Column '{column}': MISMATCH")

    extra_columns = set(df2.columns) - set(df1.columns)
    if extra_columns:
        res = False
        for col in extra_columns:
            logger.info(f"Column '{col}': MISMATCH - exists in {name2} but not in {name1}")
        msg += f"\nColumn mismatch: Columns in {name2} that are not in {name1}: {list(extra_columns)}"
    
    # Log summary of comparison
    total_columns = len(set(df1.columns) | set(df2.columns))
    mismatched_columns = cols_compare.count(',') // 2 if cols_compare else 0
    logger.info(f"\nColumn comparison summary for {case}:")
    logger.info(f"Total columns checked: {total_columns}")
    logger.info(f"Columns with mismatches: {mismatched_columns}")
    logger.info(f"Overall result: {'MATCH' if res else 'MISMATCH'}")

    return res, msg, cols_compare


def log_dataframe_summary(df: pd.DataFrame, name: str) -> None:
    """Log comprehensive summary of DataFrame for debugging.
    
    Logs detailed information about a DataFrame including shape, columns,
    data types, memory usage, summary statistics, and sample rows.
    
    Args:
        df: DataFrame to summarize.
        name: Descriptive name for the DataFrame in logs.
        
    Notes:
        - Includes memory usage with 'deep' calculation
        - Shows summary statistics for all column types
        - Displays first 5 rows as sample
    """
    logger.info(f"\n{name} DataFrame Summary:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info("Column Types and Non-Null Counts:")
    logger.info(df.info(memory_usage='deep'))
    logger.info("\nSummary Statistics:")
    logger.info(df.describe(include='all').to_string())
    logger.info("\nFirst 5 rows:")
    logger.info(df.head().to_string())


def get_df_idx(df: pd.DataFrame, start: Optional[dt] = None) -> pd.Series:
    """Get index mask for filtering dataframe by start time.

    Args:
        df: Input dataframe with MultiIndex containing 'ts' level
        start: Optional start datetime for filtering. If provided, returns mask
            for timestamps greater than this value

    Returns:
        Index or boolean mask for filtering the dataframe
    """
    if start is not None:
        idx = df.index.get_level_values('ts') > start
    else:
        idx = df.index
    return idx

def fast_concat(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    # Extract values and concatenate
    arrays = [df.values for df in dfs]
    combined_array = np.hstack(arrays)

    # Create column names
    col_names = []
    for i, df in enumerate(dfs):
        col_names.extend([f"{i}_{col}" for col in df.columns])

    return pd.DataFrame(combined_array, index=dfs[0].index, columns=col_names)

def cols_to_list_str(df: pd.DataFrame) -> str:
    cols = ",".join(sorted([str(c) for c in df.columns]))
    return cols


def convert_camel_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all DataFrame column names from camelCase to snake_case.

    Transforms column names following camelCase convention (e.g., 'positionAmt')
    to snake_case convention (e.g., 'position_amt').

    Args:
        df: DataFrame with potentially camelCase column names.

    Returns:
        DataFrame with all column names converted to snake_case.

    Notes:
        - Handles common patterns like 'myVariable' -> 'my_variable'
        - Handles acronyms like 'HTTPSConnection' -> 'https_connection'
        - Handles numbers like 'var2Name' -> 'var2_name'
        - Idempotent: running on snake_case names returns unchanged
        - Useful for standardizing data from external APIs (e.g., Binance)

    Examples:
        >>> df = pd.DataFrame({'userName': [1], 'positionAmt': [2]})
        >>> convert_camel_cols(df)
           user_name  position_amt
        0          1             2
    """
    import re

    def camel_to_snake(name: str) -> str:
        """Convert camelCase to snake_case."""
        # Insert underscore before uppercase letters that follow lowercase letters
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        # Insert underscore before uppercase letters that follow lowercase or other uppercase letters
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    df.columns = [camel_to_snake(col) for col in df.columns]
    return df


def convert_category_cols_to_bool(df: pd.DataFrame, fill_value: bool = False) -> pd.DataFrame:
    """Convert category_* columns to boolean type, filling NaN with specified value.

    When merging dataframes with boolean columns, pandas converts them to object type
    if NaN values are present (since standard bool type cannot represent NaN).
    This function identifies category_* columns, fills NaN values, and converts to bool.

    Args:
        df: DataFrame with category_* columns that may be object or boolean type
        fill_value: Value to use for NaN entries (default: False)

    Returns:
        DataFrame with category_* columns converted to boolean type

    Notes:
        - Identifies columns starting with 'category_'
        - Only converts object or boolean dtype columns (skips numeric columns)
        - Fills NaN values with fill_value (typically False)
        - Converts to boolean type to save memory and enable boolean operations
        - Modifies DataFrame in place and returns it
        - Safe to call on dataframes without category columns (no-op)

    Example:
        >>> models_df = pd.merge(bars_df, categorical_df, how='left', on='symbol')
        >>> models_df = convert_category_cols_to_bool(models_df)
        >>> # Now category_* columns are bool instead of object
    """
    category_cols = sorted([col for col in df.columns if col.startswith('category_')])
    logger.info(f"Converting categories to bool on {category_cols}")

    for col in category_cols:
        # Only convert if column is object or bool type (skip int/float types)
        dtype_kind = df[col].dtype.kind
        if dtype_kind in ('O', 'b', 'f'):  # 'O' = object, 'b' = boolean
            # logger.info(f"Converting {col} from {df[col].dtype} to bool")
            df[col] = df[col].fillna(fill_value).astype(bool)
        else:
            logger.info(f"Skipping {col} with dtype {df[col].dtype} (kind={dtype_kind})")
    return df