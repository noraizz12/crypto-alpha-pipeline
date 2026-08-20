import logging
import math
import os
import re
from typing import Any, List, Optional, Set, Literal

import numpy as np
import pandas as pd
import psutil

from .aws import load_aws_secrets
from .directory import ROOT_DIR
from .time_util import date_str_to_date, date_str_to_dt

LOCAL = os.environ.get('STATARB_ENV', 'local') == 'local'
if LOCAL:
    print("Running in LOCAL mode! (not PROD)")

PROD = os.environ.get('STATARB_ENV', None) == "prod"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

pd.options.display.width = 250
pd.options.display.max_columns = 25
pd.set_option('display.max_rows', 1440)

# Load Tardis API key from environment or AWS
TARDIS_API_KEY = os.environ.get('TARDIS_API_KEY', '')
if not TARDIS_API_KEY:
    try:
        secrets = load_aws_secrets(statarb_secretid='statarb/tardis_key')
        if secrets:
            TARDIS_API_KEY = secrets.get('tardis_key', '')
    except Exception:
        logger.warning("No TARDIS_API_KEY available - set via environment or AWS Secrets Manager")

# OMS connection settings (configure via environment for production)
STATARB_OMS_IP = os.environ.get('OMS_IP', '127.0.0.1')
STATARB_OMS_PORT = int(os.environ.get('OMS_PORT', '53118'))

# C++ OMS connection settings
STATARB_CPP_OMS_IP = os.environ.get('CPP_OMS_IP', '127.0.0.1')
STATARB_CPP_OMS_INGEST_PORT = int(os.environ.get('CPP_OMS_INGEST_PORT', '5550'))
STATARB_CPP_OMS_REPLY_PORT = int(os.environ.get('CPP_OMS_REPLY_PORT', '5551'))
STATARB_CPP_OMS_USER_DATA_PORT = int(os.environ.get('CPP_OMS_USER_DATA_PORT', '5555'))

ONE_PENNY = 0.01
MAX_WORKER_THREADPOOL = 24
STABLECOINS = ['USDC', 'USDT']

MARKETCAP_SYMBOL_START_DATE = date_str_to_date("20250320")
TRADING_START_DT = date_str_to_dt("20260108")

PNL_START_DATE = date_str_to_dt("20260108")

# one of https://api.tardis.dev/v1/exchanges with supportsDatasets:true - use 'id' value
TARDIS_EXCHANGE = "binance-futures"

SYMBOL_BASE = 'symbol'
SYMBOL_PAIR = 'pair'
SYMBOL_VENUE = 'symbol_venue'

MKT_SYMBOL = 'MKTIDX'


def log_mem_usage():
    """Log current memory usage of the process in gigabytes.
    
    Uses psutil to get resident set size (RSS) memory usage and logs it
    in a human-readable format.
    
    Notes:
        - RSS includes all memory the process has allocated
        - Useful for monitoring memory consumption during data processing
    """
    logger.info(f"Total Mem: {psutil.Process().memory_info().rss / (1024 * 1024 * 1024) :.2f} Gig.")


def get_config_universe(config: dict, symbol_type: Literal[SYMBOL_BASE, SYMBOL_PAIR, SYMBOL_VENUE] = SYMBOL_BASE) -> List[str]:
    """Get list of trading symbols from configuration.
    
    Retrieves the universe of symbols to trade based on configuration settings.
    Can return symbol-only or symbol_venue combinations.
    
    Args:
        config: Configuration dictionary.
        symbol_venues: If True, returns symbol_venue format (e.g., 'BTCUSDT_binance-futures').

    Returns:
        List of trading symbols in requested format.
        
    Notes:
        - All symbols are converted to USDT perpetual format
        - Results are sorted alphabetically
    """

    uni = config['SYMBOL_UNIVERSE']
    logger.info(f"Using config universe of {len(uni)} symbols")

    if symbol_type in (SYMBOL_PAIR, SYMBOL_VENUE):
        uni = symbols_to_pairs(uni)

    if symbol_type == SYMBOL_VENUE:
        all_perps = []
        all_perps += sorted([f"{ss}_{TARDIS_EXCHANGE}" for ss in uni])
        uni = all_perps
    return uni

def symbols_to_pairs(symbols: List[str]) -> List[str]:
    symbols = sorted([f"{ss}USDT" for ss in symbols])
    return symbols

def symbols_to_symbol_venues(symbols: List[str]) -> List[str]:
    if len(symbols) == 0:
        return symbols

    if not symbols[0].endswith("USDT"):
        symbols = symbols_to_pairs(symbols)

    symbol_venues = sorted([f"{ss}_{TARDIS_EXCHANGE}" for ss in symbols])
    return symbol_venues

def symbol_venue_to_symbol(symbol_venue: str) -> str:
    return symbol_venue.split('_')[0]

def symbol_venues_to_symbols(symbol_venues: Optional[List[str]]) -> Optional[List[str]]:
    if symbol_venues is None:
        return None
    return [symbol_venue_to_symbol(sv) for sv in symbol_venues]


def fmoney(amount: float, cents: bool = False, thousands_sep: bool = False) -> str:
    """Format a number as a dollar amount string.
    
    Converts numeric values to formatted dollar strings with optional
    cents and thousands separators.
    
    Args:
        amount: Numeric amount to format.
        cents: If True, includes cents (.00 format). Defaults to False.
        thousands_sep: If True, includes comma thousands separator. Defaults to False.
        
    Returns:
        Formatted dollar string (e.g., '$1,234' or '-$567.89').
        
    Notes:
        - Returns '$NaN' for None or NaN values
        - Handles negative values with '-$' prefix
        - Rounds to nearest dollar if cents=False
    """
    if amount is None or np.isnan(amount):
        return '$NaN'
    number = int(abs(round(amount, 0)))
    sgn = "-" if amount < 0 else ""

    number_str = f"{number:,}" if thousands_sep else str(number)
    frmt = f"{sgn}${number_str}"

    if cents:
        cent_amt = abs(round((amount - int(amount)) * 100, 0))
        frmt += f".{cent_amt:02.0f}"
    return frmt


def fpct(amt: float) -> str:
    """Format a number as a percentage string.
    
    Converts decimal values to percentage format with 2 decimal places.
    
    Args:
        amt: Decimal value to format (e.g., 0.125 for 12.5%).
        
    Returns:
        Formatted percentage string (e.g., '12.50%').
    """
    return f"{100.0 * amt:.2f}%"


def exception_msg(msg: str, e: Exception) -> str:
    """Format exception message with type and context.
    
    Creates a standardized error message format including exception type,
    context message, and exception details.
    
    Args:
        msg: Context message describing where/why the exception occurred.
        e: The exception that was caught.
        
    Returns:
        Formatted string: '{ExceptionType}, {context}: {exception details}'.
    """
    ex_type = type(e).__name__
    return f"{ex_type}, {msg}: {e}"


def unique_list(lst: List[Any] | Set[Any]) -> List[Any]:
    """Return unique elements from a list or set.
    
    Removes duplicates from the input collection while converting to list.
    
    Args:
        lst: List or set of elements.
        
    Returns:
        List containing unique elements only, preserving order of first occurrence.
        
    Notes:
        - Order is preserved (first occurrence of each element)
        - Works with any hashable elements
    """
    if isinstance(lst, set):
        # If input is a set, convert to sorted list for deterministic ordering
        return sorted(list(lst))
    
    # For lists, preserve order of first occurrence
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def round_up_to_multiple(number: float | int, multiple: int) -> int:
    """Round a number up to the nearest multiple.
    
    Rounds the input number up to the nearest integer multiple of the
    specified value.
    
    Args:
        number: Number to round up.
        multiple: Multiple to round to.
        
    Returns:
        Rounded up value as integer.
        
    Examples:
        >>> round_up_to_multiple(7, 5)
        10
        >>> round_up_to_multiple(10, 5)
        10
    """
    return math.ceil(int(number) / multiple) * multiple


def get_nested_dict_value(d: dict, outer_key: str, inner_key: str, default_val: Any):
    """Safely get value from nested dictionary with default.
    
    Retrieves value from a two-level nested dictionary, returning default
    if either key is missing.
    
    Args:
        d: Dictionary to search.
        outer_key: First level key.
        inner_key: Second level key.
        default_val: Value to return if key path not found.
        
    Returns:
        Value at d[outer_key][inner_key] or default_val.
        
    Examples:
        >>> d = {'a': {'b': 1}}
        >>> get_nested_dict_value(d, 'a', 'b', 0)
        1
        >>> get_nested_dict_value(d, 'a', 'c', 0)
        0
    """
    return d.get(outer_key, {}).get(inner_key, default_val)


def safe_del_from_dict(d: dict, key: str) -> None:
    """Safely delete key from dictionary without raising KeyError.
    
    Attempts to delete a key from dictionary, silently ignoring if
    the key doesn't exist.
    
    Args:
        d: Dictionary to modify.
        key: Key to delete.
        
    Notes:
        - Modifies dictionary in-place
        - No error if key doesn't exist
    """
    try:
        del d[key]
    except KeyError:
        pass


def assert_float_equal(a: float, b: float, msg: str, tolerance: float = 1e-7) -> None:
    """Assert two floats are equal within tolerance.
    
    Compares two floating point values for approximate equality, raising
    AssertionError if they differ by more than the specified tolerance.
    
    Args:
        a: First value to compare.
        b: Second value to compare.
        msg: Message to include in assertion error.
        tolerance: Maximum allowed absolute difference. Defaults to 1e-7.
        
    Raises:
        AssertionError: If |a - b| >= tolerance.
        
    Notes:
        - Uses absolute difference, not relative
        - Default tolerance suitable for most financial calculations
    """
    assert abs(a - b) < tolerance, f"{a} and {b} differ by more than {tolerance}, {msg}"


def chunk_list(input_list: list, N: int) -> List[list]:
    """Split a list into chunks of specified size.
    
    Divides input list into sublists of size N, with the last chunk
    potentially being smaller.
    
    Args:
        input_list: List to split into chunks.
        N: Maximum size of each chunk.
        
    Returns:
        List of list chunks.
        
    Examples:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    result = []
    for i in range(0, len(input_list), N):
        chunk = input_list[i:i + N]
        result.append(chunk)
    return result


def truncate_to_words(text: str, n: int) -> str:
    """Truncate text to first n words.
    
    Splits text by whitespace and returns only the first n words
    joined back together.
    
    Args:
        text: Text to truncate.
        n: Number of words to keep.
        
    Returns:
        String containing first n words.
        
    Notes:
        - Words are split by any whitespace
        - Returns fewer words if text has less than n words
    """
    words = text.split()
    first_n = words[:n]
    return ' '.join(first_n)


def delete_all_files_in_tree(directory: str, exclude_dirs: Optional[List[str]] = None) -> int:
    """Recursively delete all files in a directory tree, keeping directories intact.
    
    Traverses the directory tree and deletes all files while preserving the
    directory structure. Can optionally exclude specific directory names from
    file deletion while still traversing their subdirectories.
    
    Args:
        directory: Root directory to start deletion from.
        exclude_dirs: Optional list of directory names to exclude from deletion.
                     Files in directories with these names will be preserved,
                     but subdirectories will still be traversed.
        
    Returns:
        Number of files deleted.
        
    Notes:
        - Only deletes files, not directories
        - Recursively traverses all subdirectories
        - Logs each file deletion at debug level
        - Handles permission errors gracefully
        - Excluded directories are matched by basename only
        
    Example:
        # Delete all files except those in 'data' directories
        delete_all_files_in_tree('/path/to/fixtures', exclude_dirs=['data'])
        
        # This would preserve:
        # /path/to/fixtures/data/file.txt
        # /path/to/fixtures/subdir/data/file.txt
        # But delete:
        # /path/to/fixtures/file.txt
        # /path/to/fixtures/subdir/file.txt
    """
    deleted_count = 0
    exclude_dirs = exclude_dirs or []
    
    if not os.path.exists(directory):
        logger.warning(f"Directory {directory} does not exist")
        return deleted_count
    
    for root, dirs, files in os.walk(directory):
        # Check if current directory basename is in exclude list
        current_dir_name = os.path.basename(root)
        if current_dir_name in exclude_dirs:
            logger.debug(f"Skipping file deletion in excluded directory: {root}")
            continue
            
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                logger.debug(f"Deleted file: {file_path}")
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")
    
    logger.info(f"Deleted {deleted_count} files from {directory}")
    if exclude_dirs:
        logger.info(f"Excluded directories: {exclude_dirs}")
    return deleted_count


def remove_non_ascii_and_extra_whitespace(text: str) -> Optional[str]:
    """Clean text by removing non-ASCII characters and extra whitespace.
    
    Strips non-ASCII characters, converts newlines to spaces, and collapses
    multiple whitespace characters into single spaces.
    
    Args:
        text: Text to clean.
        
    Returns:
        Cleaned text string, or None if cleaning fails.
        
    Notes:
        - Useful for processing external text data
        - Replaces newlines with spaces
        - Collapses multiple spaces to single space
    """
    try:
        ret = text.encode('ascii', 'ignore').decode('ascii').replace('\n', ' ')
        ret = re.sub(r'\s+', ' ', ret)
        return ret
    except:
        return None


def get_env(key: str) -> Optional[str]:
    """Safely get environment variable value.
    
    Retrieves environment variable value without raising KeyError
    if the variable doesn't exist.
    
    Args:
        key: Environment variable name.
        
    Returns:
        Environment variable value or None if not set.
    """
    try:
        return os.environ[key]
    except KeyError:
        return None


def remove_after_last_occurrence(s: str, char: str) -> str:
    """Remove everything after the last occurrence of specified character.
    
    Truncates string at the last occurrence of the given character,
    excluding the character itself.
    
    Args:
        s: String to truncate.
        char: Character to find and truncate after.
        
    Returns:
        Truncated string, or original if character not found.
        
    Examples:
        >>> remove_after_last_occurrence('path/to/file.txt', '/')
        'path/to'
        >>> remove_after_last_occurrence('no_slash_here', '/')
        'no_slash_here'
    """
    return s.rpartition(char)[0] if char in s else s


def get_max_worker_for_threadpool(divisor: int = 4) -> int:
    """Calculate optimal number of workers for thread pool.
    
    Determines thread pool size based on CPU count, using 2/3 of available
    CPUs but capped at MAX_WORKER_THREADPOOL.
    
    Returns:
        Number of workers to use for thread pool.
        
    Notes:
        - Uses 2/3 of CPU cores for balance between parallelism and system resources
        - Capped at MAX_WORKER_THREADPOOL (24) to prevent oversubscription
    """
    return min(os.cpu_count() // divisor, MAX_WORKER_THREADPOOL)



def extract_horizon_from_feature(feature: str) -> Optional[int]:
    parts = feature.split('_')
    for part in parts:
        try:
            # Try to convert to int - if successful, it might be a horizon
            return int(part)
        except ValueError:
            # Not a number, continue to next part
            pass
    return None


DEFAULT_MODEL_LAGS = 1
MODELS = ["hl", "c2vwap", "slz", "vadj", "ba", "badj", "oi", "rsi"]
MODEL_HORIZONS = [15, 60, 120, 360, 720, 1440, 4320, 10080, 43200]


def log_and_raise(msg: str, df: Optional[pd.DataFrame | pd.Series] = None, df_name: Optional[str] = None) -> RuntimeError:
    """Log an error message and optionally save problematic data before raising.

    Logs error message and saves DataFrame/Series to parquet file for debugging
    before returning a RuntimeError to be raised.

    Args:
        msg: Error message to log and include in exception.
        df: Optional DataFrame or Series to save for debugging.

    Returns:
        RuntimeError with the provided message.

    Notes:
        - Saved files are named 'error_df_{timestamp}.parquet'
        - If save fails, prints DataFrame to console
        - Use as: raise log_and_raise("error message", df)
    """
    from .time_util import dt_to_str
    
    logger.error(msg)
    if df is not None:
        if df_name is not None:
            error_file = f'error_df_{df_name}_{dt_to_str()}.parquet'
        else:
            error_file = f'error_df_{dt_to_str()}.parquet'
        logger.error(f'Dumping {error_file}')
        try:
            df.to_parquet(error_file)
        except Exception as e:
            print(f"Could not dump error file! {e}")
        print(df)
    return RuntimeError(msg)

