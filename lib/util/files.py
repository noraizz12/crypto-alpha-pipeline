"""File system utilities for the trading system.

This module provides utilities for file and directory management, including:
- Safe directory creation and cleanup
- File pattern matching and date extraction
- CSV writing with append support
- JSONL and dictionary file loading
- File metadata retrieval

The module is designed to handle common file operations needed by the trading
system, with proper error handling and logging.

Key functions:
    - safe_mkdir: Create directories without raising if they exist
    - make_sim_dir: Create simulation directories with cleanup
    - find_latest_file_date: Find most recent file by date pattern
    - write_df_to_csv: Write DataFrames to CSV with append support
    - load_jsonl: Load JSON Lines format files
    - load_dict_file: Load files containing Python dictionaries
"""
import glob
import os
import re
import shutil
import logging
import json
import ast
from datetime import datetime as dt, timezone, date
from typing import Optional, List
import pandas as pd

from .time_util import DATE_TIME_FORMAT, DATE_FORMAT, millis_to_dt, date_str_to_date
from .directory import ROOT_DIR, SIM_DIR, dir_manager
from .util import LOCAL
from .util import log_and_raise

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def safe_mkdir(name: str) -> None:
    """Create a directory if it doesn't exist, without raising errors.
    
    Safe wrapper around os.makedirs that silently handles the case where
    the directory already exists. Useful for ensuring output directories
    exist without cluttering code with try/except blocks.
    
    Args:
        name: Path of the directory to create
        
    Side Effects:
        Creates the directory and any necessary parent directories
        
    Notes:
        - Uses os.makedirs internally, so creates parent directories
        - Silently succeeds if directory already exists
        - Will still raise for other errors (permissions, invalid path, etc.)
    """
    try:
        os.makedirs(name)
    except FileExistsError:
        pass


def make_sim_dir(sim_name: Optional[str] = None, base_dir: Optional[str] = None) -> str:
    """Create a simulation directory, removing any existing one.

    Creates a fresh directory for simulation outputs, ensuring a clean state
    by removing any existing directory with the same name. If no name is
    provided, uses current UTC timestamp.

    Args:
        sim_name: Name for the simulation directory. If None, uses current
                 UTC timestamp in DATE_TIME_FORMAT
        base_dir: Custom base directory for simulations. If None, uses SIM_DIR

    Returns:
        Full path to the created simulation directory

    Raises:
        RuntimeError: If sim_name is empty string (safety check)

    Side Effects:
        - Removes existing directory if present (with warning)
        - Creates new directory at base_dir/sim_name (or SIM_DIR/sim_name if base_dir is None)

    Example:
        >>> sim_dir = make_sim_dir("backtest_2024")
        >>> # sim_dir = "/path/to/sims/backtest_2024"
        >>> sim_dir = make_sim_dir("comp_test", base_dir="/path/to/sims/simcomp")
        >>> # sim_dir = "/path/to/sims/simcomp/comp_test"

    Notes:
        - Be careful as this DELETES existing directories
        - Empty string names are rejected for safety
    """
    if sim_name is None:
        sim_name = dt.now(timezone.utc).strftime(DATE_TIME_FORMAT)
    if base_dir is None:
        base_dir = SIM_DIR
    # Ensure base_dir exists
    os.makedirs(base_dir, exist_ok=True)
    sim_dir = base_dir + "/" + sim_name

    if len(sim_name) == 0:
        raise log_and_raise(f"not removing sim dir {sim_name}")

    try:
        shutil.rmtree(sim_dir)
    except Exception as e:
        logger.warning(f"Could not remove {sim_dir}: {e}")

    try:
        os.mkdir(sim_dir)
    except FileExistsError:
        pass

    return sim_dir


def get_user() -> str:
    """Extract the second path component from ROOT_DIR.
    
    Splits ROOT_DIR by '/' and returns the element at index 1.
    Originally intended to extract username from paths like /home/username/...,
    but actually returns the second path component (e.g., "home" from "/home/user/...").
    
    Returns:
        Second path component from ROOT_DIR after splitting by '/'
        
    Example:
        >>> # If ROOT_DIR = "/home/trader/stat_arb"
        >>> user = get_user()
        >>> # user = "home" (not "trader" as might be expected)
        
    Notes:
        - Returns element at index 1 after split, not the actual username
        - For path "/home/user/project", returns "home" not "user"
        - May not work as intended for extracting usernames
    """
    user = ROOT_DIR.split('/')[1]  # Gets second component, not username
    return user


def find_latest_file_date(pattern: str) -> Optional[dt]:
    """Find the date of the most recent file matching a glob pattern.
    
    Searches for files matching the given pattern and extracts the date
    from the most recent one. Assumes files have dates in YYYYMMDD format
    as the last 8 characters before the file extension.
    
    Args:
        pattern: Glob pattern to match files (e.g., "data/*.parquet")
        
    Returns:
        datetime object of the most recent file's date, or None if no
        files found
        
    Side Effects:
        Logs warning if no files found, info if date extracted
        
    Example:
        >>> # Files: data/bars_20240101.parquet, data/bars_20240102.parquet
        >>> latest = find_latest_file_date("data/bars_*.parquet")
        >>> # latest = datetime(2024, 1, 2)
        
    Notes:
        - Files must have 8-digit date (YYYYMMDD) before extension
        - Files are sorted lexicographically to find latest
        - Returns None rather than raising if no files found
    """
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        logger.warning(f"No files found like {pattern}")
        return None
    last_file = files[-1]

    dot_idx = last_file.rindex('.')
    file_stub = last_file[:dot_idx]

    ts = dt.strptime(file_stub[-8:], DATE_FORMAT)
    logger.info(f"Latest file time is {ts}")
    return ts


def find_latest_universe_date(universe_dir: Optional[str] = None) -> Optional[date]:
    """Find the date of the most recent universe file.
    
    Searches the universe directory for all universe files and extracts the date
    from the most recent one. Universe files follow the pattern universe.YYYYMMDD.parquet.
    
    Args:
        universe_dir: Directory containing universe files. Defaults to dir_manager.UNIVERSE_DIR
        
    Returns:
        date object of the most recent universe file's date, or None if no
        files found
        
    Side Effects:
        Logs warning if no files found, info if date extracted
        
    Example:
        >>> # Files: universe.20240101.parquet, universe.20240102.parquet
        >>> latest = find_latest_universe_date()  # Uses default UNIVERSE_DIR
        >>> # latest = date(2024, 1, 2)
        
    Notes:
        - Files must follow pattern universe.YYYYMMDD.parquet
        - Files are sorted lexicographically to find latest
        - Returns date object (not datetime) for consistency with universe loading
    """
    if universe_dir is None:
        universe_dir = dir_manager.UNIVERSE_DIR
    
    pattern = f"{universe_dir}/universe.*.parquet"
    files = sorted(glob.glob(pattern))
    
    if len(files) == 0:
        logger.warning(f"No universe files found in {universe_dir}")
        return None
    
    last_file = files[-1]
    
    # Extract date from filename (format: universe.YYYYMMDD.parquet)
    match = re.search(r'universe\.(\d{8})\.parquet', last_file)
    if match:
        date_str = match.group(1)
        universe_date = date_str_to_date(date_str)
        logger.info(f"Latest universe file date is {universe_date}")
        return universe_date
    
    logger.warning(f"Could not parse date from universe filename: {last_file}")
    return None


def write_df_to_csv(df: pd.DataFrame, filename: str, append: bool = False) -> None:
    """Write DataFrame to CSV file with optional append mode.
    
    Saves a pandas DataFrame to CSV format, with support for appending to
    existing files. Handles headers correctly when appending and provides
    special handling for permission errors in local development.
    
    Args:
        df: DataFrame to write (can be None, in which case nothing is written)
        filename: Base filename without extension (e.g., "results" not "results.csv")
        append: If True, append to existing file; if False, overwrite
        
    Side Effects:
        - Creates or modifies CSV file at {filename}.csv
        - Logs file writing activity
        - In LOCAL mode, prints DataFrame on permission error instead of raising
        
    Example:
        >>> df = pd.DataFrame({'symbol': ['BTC', 'ETH'], 'price': [50000, 3000]})
        >>> write_df_to_csv(df, "prices", append=False)
        >>> # Creates prices.csv
        >>> write_df_to_csv(df, "prices", append=True)
        >>> # Appends to prices.csv without repeating headers
        
    Notes:
        - Automatically adds .csv extension
        - Handles None DataFrames gracefully (logs warning)
        - Suppresses header when appending to existing file
        - In LOCAL mode, prints instead of failing on permission errors
    """
    if df is None:
        logger.warning(f"Not writing empty dataframe to {filename}")
        return

    filename = f"{filename}.csv"
    logger.info(f"Writing {filename}")

    header = True
    if append:
        mode = "a"
        if os.path.isfile(filename):
            header = False
    else:
        mode = "w"

    try:
        df.to_csv(filename, index=False, mode=mode, header=header)
        return
    except PermissionError:
        if LOCAL:
            print(df)
        else:
            raise


def get_file_created_dt(filename: str) -> Optional[dt]:
    """Get the creation timestamp of a file.
    
    Retrieves the file creation time and converts it to a datetime object.
    Uses os.path.getctime which gives creation time on Windows and last
    metadata change time on Unix.
    
    Args:
        filename: Path to the file
        
    Returns:
        datetime object of file creation/change time, or None on error
        
    Side Effects:
        Logs error if file metadata cannot be accessed
        
    Example:
        >>> created = get_file_created_dt("/path/to/data.csv")
        >>> # created = datetime(2024, 1, 15, 10, 30, 45)
        
    Notes:
        - Returns None on any error (file not found, permissions, etc.)
        - On Unix, returns last metadata change time, not creation time
        - Converts from seconds to milliseconds for millis_to_dt
    """
    try:
        return millis_to_dt(int(1000.0 * os.path.getctime(filename)))
    except Exception as e:
        logger.error(f"Could not get file time: {e}")
    return None


def load_jsonl(
        file_path: str,
        columns: Optional[List[str]] = None,
        encoding: str = 'utf-8',
        lines_to_read: Optional[int] = None
) -> pd.DataFrame:
    """Load a JSON Lines file into a pandas DataFrame.
    
    Reads a JSONL file where each line contains a valid JSON object.
    Supports partial loading, column selection, and error handling for
    malformed lines.
    
    Args:
        file_path: Path to the .jsonl file
        columns: List of columns to include in result. If None, includes all.
                Missing columns are warned about but don't cause failure.
        encoding: File encoding (default: 'utf-8')
        lines_to_read: Number of lines to read. If None, reads entire file.
                      Useful for sampling large files.
                      
    Returns:
        DataFrame containing the JSON data with requested columns
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If no valid JSON objects found in file
        Exception: For other errors during loading
        
    Example:
        >>> # Load first 1000 lines with specific columns
        >>> df = load_jsonl(
        ...     "trades.jsonl",
        ...     columns=['symbol', 'price', 'volume'],
        ...     lines_to_read=1000
        ... )
        
    Notes:
        - Skips lines with invalid JSON (logs warning)
        - Warns about requested columns not found in data
        - Returns only the intersection of requested and available columns
    """
    data = []
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            for i, line in enumerate(f):
                if lines_to_read is not None and i >= lines_to_read:
                    break

                try:
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {i + 1}: {e}")
                    continue

        if not data:
            raise ValueError("No valid JSON objects found in file")

        df = pd.DataFrame(data)

        # Select specific columns if requested
        if columns is not None:
            missing_cols = set(columns) - set(df.columns)
            if missing_cols:
                print(f"Warning: Columns not found: {missing_cols}")
            df = df[list(set(columns) & set(df.columns))]

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading file: {e}")


def load_dict_file(
        file_path: str,
        columns: Optional[List[str]] = None,
        encoding: str = 'utf-8',
        lines_to_read: Optional[int] = None
) -> pd.DataFrame:
    """Load a file of Python dictionary literals into a DataFrame.
    
    Reads a file where each line contains a string representation of a
    Python dictionary. Uses ast.literal_eval for safe parsing (no code
    execution). Useful for files with Python-style data dumps.
    
    Args:
        file_path: Path to the file containing dictionary strings
        columns: List of columns to include in result. If None, includes all.
                Missing columns are warned about but don't cause failure.
        encoding: File encoding (default: 'utf-8')
        lines_to_read: Number of lines to read. If None, reads entire file.
                      Useful for sampling large files.
                      
    Returns:
        DataFrame containing the dictionary data with requested columns
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If no valid dictionaries found in file
        Exception: For other errors during loading
        
    Example:
        >>> # File contains lines like: {'symbol': 'BTC', 'price': 50000}
        >>> df = load_dict_file(
        ...     "python_data.txt",
        ...     columns=['symbol', 'price']
        ... )
        
    Notes:
        - Uses ast.literal_eval for safe parsing (no exec)
        - Skips lines that aren't valid dictionary literals
        - Warns about non-dictionary objects and parse errors
        - Returns only the intersection of requested and available columns
    """
    data = []
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            for i, line in enumerate(f):
                if lines_to_read is not None and i >= lines_to_read:
                    break

                try:
                    # Use literal_eval to safely evaluate string as dict
                    dict_obj = ast.literal_eval(line.strip())
                    if not isinstance(dict_obj, dict):
                        print(f"Warning: Line {i + 1} is not a dictionary")
                        continue
                    data.append(dict_obj)
                except (SyntaxError, ValueError) as e:
                    print(f"Warning: Invalid dictionary on line {i + 1}: {e}")
                    continue

        if not data:
            raise ValueError("No valid dictionaries found in file")

        df = pd.DataFrame(data)

        # Select specific columns if requested
        if columns is not None:
            missing_cols = set(columns) - set(df.columns)
            if missing_cols:
                print(f"Warning: Columns not found: {missing_cols}")
            df = df[list(set(columns) & set(df.columns))]

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        logger.warning(f"error loading file {e}")
        raise
