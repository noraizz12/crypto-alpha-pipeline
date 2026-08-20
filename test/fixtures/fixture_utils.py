#!/usr/bin/env python3
"""Common utilities for fixture generation across all integration tests.

This module provides reusable functions for:
- Copying data files with symbol/date filtering
- Managing fixture directories
- Loading test configurations
"""

import os
import json
import shutil
import glob
from typing import List, Set, Dict, Optional
from datetime import datetime, timedelta

# Common test configuration defaults
DEFAULT_TEST_SYMBOLS = ['BTC', 'ETH', 'BNB']
DEFAULT_HISTORICAL_DAYS = 7
DEFAULT_BAR_FREQUENCIES = [1, 15, 60, 1440]


def load_test_config(config_path: str) -> dict:
    """Load test configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_test_config(config_path: str, config: dict) -> None:
    """Save test configuration."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def clean_fixture_directory(base_dir: str, subdirs: List[str]) -> None:
    """Clean fixture directory by removing old files and creating clean subdirectories."""
    print(f"\nCleaning fixture directory: {base_dir}")
    
    # Remove all parquet files in root directory
    for file in glob.glob(os.path.join(base_dir, "*.parquet")):
        os.remove(file)
        print(f"  ✓ Removed {os.path.basename(file)}")
    
    # Clean and recreate subdirectories
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"  ✓ Cleaned {subdir}")
        os.makedirs(dir_path, exist_ok=True)


def get_date_range(test_date: str, historical_days: int, forward_days: int = 0) -> List[str]:
    """Generate date range for test data.
    
    Args:
        test_date: Target test date in YYYYMMDD format
        historical_days: Number of days before test date to include
        forward_days: Number of days after test date to include
        
    Returns:
        List of date strings in YYYYMMDD format
    """
    test_dt = datetime.strptime(test_date, '%Y%m%d')
    start_date = test_dt - timedelta(days=historical_days)
    end_date = test_dt + timedelta(days=forward_days)
    
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime('%Y%m%d'))
        current_date += timedelta(days=1)
    
    return dates


def copy_files_with_filter(
    source_pattern: str,
    dest_dir: str,
    filter_func: Optional[callable] = None,
    rename_func: Optional[callable] = None
) -> int:
    """Copy files matching pattern with optional filtering and renaming.
    
    Args:
        source_pattern: Glob pattern for source files
        dest_dir: Destination directory
        filter_func: Optional function to filter files (returns True to include)
        rename_func: Optional function to rename files
        
    Returns:
        Number of files copied
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    copied = 0
    for source_file in glob.glob(source_pattern):
        if filter_func and not filter_func(source_file):
            continue
            
        filename = os.path.basename(source_file)
        if rename_func:
            filename = rename_func(filename)
            
        dest_file = os.path.join(dest_dir, filename)
        shutil.copy2(source_file, dest_file)
        copied += 1
        
    return copied


def copy_bar_files(
    source_base_dir: str,
    dest_base_dir: str,
    symbols: Set[str],
    dates: List[str],
    frequencies: List[int],
    exchange: str = "binance-futures"
) -> int:
    """Copy bar files from production to test fixtures."""
    print(f"\nCopying bar files...")
    print(f"Source: {source_base_dir}")
    print(f"Destination: {dest_base_dir}")
    print(f"Frequencies: {frequencies}")
    print(f"Symbols: {sorted(symbols)}")
    
    total_copied = 0
    
    for freq in frequencies:
        print(f"\n--- Frequency {freq} ---")
        for date in dates:
            source_date_dir = os.path.join(source_base_dir, str(freq), exchange, date)
            dest_date_dir = os.path.join(dest_base_dir, str(freq), exchange, date)
            
            if not os.path.exists(source_date_dir):
                print(f"  Warning: Source directory not found: {source_date_dir}")
                continue
                
            # Create destination directory
            os.makedirs(dest_date_dir, exist_ok=True)
            
            print(f"  Date {date}:")
            copied_count = 0
            
            for symbol in symbols:
                # Production bar files have USDT suffix
                source_filename = f"bars.{freq}.{exchange}.{date}.{symbol}USDT.parquet"
                source_file = os.path.join(source_date_dir, source_filename)
                dest_file = os.path.join(dest_date_dir, source_filename)
                
                if os.path.exists(source_file):
                    shutil.copy2(source_file, dest_file)
                    copied_count += 1
                else:
                    print(f"    ✗ {symbol}USDT (not found)")
            
            print(f"    → Copied {copied_count} files")
            total_copied += copied_count
    
    print(f"\nTotal: {total_copied} bar files copied")
    return total_copied


def copy_feature_files(
    source_base_dir: str,
    dest_base_dir: str,
    symbols: Set[str],
    dates: List[str],
    horizons: List[int],
    feature_types: List[str]
) -> int:
    """Copy feature files from production to test fixtures."""
    print(f"\nCopying feature files for horizons {horizons}...")
    
    total_copied = 0
    
    for horizon in horizons:
        for feature_type in feature_types:
            for date in dates:
                source_date_dir = os.path.join(source_base_dir, str(horizon), feature_type, date)
                dest_date_dir = os.path.join(dest_base_dir, str(horizon), feature_type, date)
                
                if not os.path.exists(source_date_dir):
                    continue
                    
                # Create destination directory
                os.makedirs(dest_date_dir, exist_ok=True)
                
                copied_count = 0
                for symbol in symbols:
                    source_filename = f"features.{horizon}.{feature_type}.{date}.{symbol}USDT.parquet"
                    source_file = os.path.join(source_date_dir, source_filename)
                    dest_file = os.path.join(dest_date_dir, source_filename)
                    
                    if os.path.exists(source_file):
                        shutil.copy2(source_file, dest_file)
                        copied_count += 1
                
                if copied_count > 0:
                    print(f"  {horizon}/{feature_type}/{date}: Copied {copied_count} files")
                    total_copied += copied_count
    
    print(f"\nTotal: {total_copied} feature files copied")
    return total_copied


def copy_model_files(
    source_base_dir: str,
    dest_base_dir: str,
    dates: List[str],
    horizons: List[int],
    models: List[str]
) -> int:
    """Copy model files from production to test fixtures."""
    print(f"\nCopying model files for horizons {horizons}, models {models}...")
    
    total_copied = 0
    
    for horizon in horizons:
        for model in models:
            for date in dates:
                source_filename = f"models.{horizon}.{model}.{date}.parquet"
                source_file = os.path.join(source_base_dir, str(horizon), model, source_filename)
                
                # Create destination directory
                dest_dir = os.path.join(dest_base_dir, str(horizon), model)
                os.makedirs(dest_dir, exist_ok=True)
                
                dest_file = os.path.join(dest_dir, source_filename)
                
                if os.path.exists(source_file):
                    shutil.copy2(source_file, dest_file)
                    total_copied += 1
                    print(f"  ✓ {horizon}/{model}/{date}")
                else:
                    print(f"  ✗ {horizon}/{model}/{date} (not found)")
    
    print(f"\nTotal: {total_copied} model files copied")
    return total_copied


def copy_forward_files(
    source_base_dir: str,
    dest_base_dir: str,
    symbols: Set[str],
    dates: List[str],
    horizons: List[int]
) -> int:
    """Copy forward files from production to test fixtures."""
    print(f"\nCopying forward files for horizons {horizons}...")
    
    total_copied = 0
    
    for horizon in horizons:
        for date in dates:
            source_date_dir = os.path.join(source_base_dir, str(horizon), date)
            dest_date_dir = os.path.join(dest_base_dir, str(horizon), date)
            
            if not os.path.exists(source_date_dir):
                print(f"  Warning: Source directory not found: {source_date_dir}")
                continue
                
            # Create destination directory
            os.makedirs(dest_date_dir, exist_ok=True)
            
            copied_count = 0
            for symbol in symbols:
                source_filename = f"forwards.{horizon}.{date}.{symbol}USDT.parquet"
                source_file = os.path.join(source_date_dir, source_filename)
                dest_file = os.path.join(dest_date_dir, source_filename)
                
                if os.path.exists(source_file):
                    shutil.copy2(source_file, dest_file)
                    copied_count += 1
            
            if copied_count > 0:
                print(f"  {horizon}/{date}: Copied {copied_count} files")
                total_copied += copied_count
    
    print(f"\nTotal: {total_copied} forward files copied")
    return total_copied


def copy_fit_files(
    source_base_dir: str,
    dest_base_dir: str,
    dates: List[str],
    horizons: List[int],
    models: List[str]
) -> int:
    """Copy fit files from production to test fixtures."""
    print(f"\nCopying fit files for horizons {horizons}, models {models}...")
    
    total_copied = 0
    
    for horizon in horizons:
        for model in models:
            # Create destination directories
            csv_dest_dir = os.path.join(dest_base_dir, str(horizon), model)
            svm_dest_dir = os.path.join(dest_base_dir, "svm", f"{model}_{horizon}")
            os.makedirs(csv_dest_dir, exist_ok=True)
            os.makedirs(svm_dest_dir, exist_ok=True)
            
            for date in dates:
                # Copy CSV fit files from flat structure
                csv_filename = f"fits.{horizon}.{model}.{date}.csv"
                csv_source = os.path.join(source_base_dir, csv_filename)
                csv_dest = os.path.join(csv_dest_dir, csv_filename)
                
                if os.path.exists(csv_source):
                    shutil.copy2(csv_source, csv_dest)
                    total_copied += 1
                    print(f"  ✓ {csv_filename}")
                    
                    # Also create dev-prefixed copy for Forecasts with prod=False
                    dev_csv_filename = f"fits.dev.{horizon}.{model}.{date}.csv"
                    dev_csv_dest = os.path.join(csv_dest_dir, dev_csv_filename)
                    shutil.copy2(csv_source, dev_csv_dest)
                    total_copied += 1
                    print(f"  ✓ {dev_csv_filename}")
                
                # Copy SVM files if they exist
                svm_features_file = f"svm.{model}_{horizon}.{date}.features"
                svm_joblib_file = f"svm.{model}_{horizon}.{date}.joblib"
                
                svm_features_source = os.path.join(source_base_dir, "svm", f"{model}_{horizon}", svm_features_file)
                svm_joblib_source = os.path.join(source_base_dir, "svm", f"{model}_{horizon}", svm_joblib_file)
                
                if os.path.exists(svm_features_source):
                    shutil.copy2(svm_features_source, os.path.join(svm_dest_dir, svm_features_file))
                    total_copied += 1
                    print(f"  ✓ {svm_features_file}")
                
                if os.path.exists(svm_joblib_source):
                    shutil.copy2(svm_joblib_source, os.path.join(svm_dest_dir, svm_joblib_file))
                    total_copied += 1
                    print(f"  ✓ {svm_joblib_file}")
    
    print(f"\nTotal: {total_copied} fit files copied")
    return total_copied


def copy_universe_files(
    source_dir: str,
    dest_dir: str,
    dates: List[str]
) -> int:
    """Copy universe files from production."""
    print(f"\nCopying universe files...")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    copied = 0
    for date in dates:
        filename = f"universe.{date}.parquet"
        source_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(dest_dir, filename)
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, dest_file)
            copied += 1
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (not found)")
    
    print(f"  → Copied {copied} universe files")
    return copied


def copy_secdata_files(
    source_base_dir: str,
    dest_base_dir: str,
    dates: List[str],
    data_types: List[str] = ['binance_meta', 'funding']
) -> int:
    """Copy secdata files (binance_meta, funding, etc.) from production."""
    print(f"\nCopying secdata files...")
    
    total_copied = 0
    
    for data_type in data_types:
        source_dir = os.path.join(source_base_dir, data_type)
        dest_dir = os.path.join(dest_base_dir, data_type)
        
        if not os.path.exists(source_dir):
            print(f"  Warning: Source directory not found: {source_dir}")
            continue
            
        os.makedirs(dest_dir, exist_ok=True)
        
        copied_count = 0
        for date in dates:
            # Different data types have different filename patterns
            if data_type == 'binance_meta':
                filename = f"meta.{date}.parquet"
            elif data_type == 'funding':
                filename = f"funding.{date}.parquet"
            else:
                filename = f"{data_type}.{date}.parquet"
                
            source_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(dest_dir, filename)
            
            if os.path.exists(source_file):
                shutil.copy2(source_file, dest_file)
                copied_count += 1
        
        if copied_count > 0:
            print(f"  {data_type}: Copied {copied_count} files")
            total_copied += copied_count
    
    print(f"\nTotal: {total_copied} secdata files copied")
    return total_copied


def update_config_for_test(
    config: dict,
    symbols: List[str] = None,
    dynamic_universe: bool = True,
    lookback_days: int = 10,
    classification_history_days: int = 2
) -> dict:
    """Update configuration for minimal test setup.
    
    Args:
        config: Configuration dictionary to update
        symbols: List of symbols to use (defaults to DEFAULT_TEST_SYMBOLS)
        dynamic_universe: Whether to use dynamic universe filtering
        lookback_days: Number of lookback days for various calculations
        classification_history_days: Days for classification history
        
    Returns:
        Updated configuration dictionary
    """
    if symbols is None:
        symbols = DEFAULT_TEST_SYMBOLS.copy()
    
    # Update universe settings
    config['SYMBOL_UNIVERSE'] = symbols
    config['DYNAMIC_UNIVERSE'] = dynamic_universe
    
    # Update lookback parameters for minimal data
    config['BETA_LOOKBACK_PERIODS'] = lookback_days
    config['DEFAULT_FEATURE_LOOKBACK_PERIODS'] = lookback_days
    config['ADV_LOOKBACK_DAYS'] = lookback_days
    config['MA_LOOKBACK_DAYS'] = lookback_days
    config['CLASSIFICATION_HISTORY_DAYS'] = classification_history_days
    
    return config


def print_test_summary(
    test_name: str,
    symbols: List[str],
    test_date: str,
    dates: List[str],
    horizons: List[int],
    models: List[str] = None
) -> None:
    """Print a summary of test configuration."""
    print(f"\n{'=' * 60}")
    print(f"{test_name.upper()} TEST FIXTURE GENERATOR")
    print(f"{'=' * 60}")
    print(f"\nTest configuration:")
    print(f"  Symbols: {sorted(symbols)}")
    print(f"  Test date: {test_date}")
    print(f"  Data dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print(f"  Horizons: {horizons}")
    if models:
        print(f"  Models: {models}")


def create_delisting_file(dest_dir: str) -> None:
    """Create an empty delisting.txt file."""
    delisting_file = os.path.join(dest_dir, 'delisting.txt')
    with open(delisting_file, 'w') as f:
        # Empty file or you can add test delisting entries if needed
        pass
    print(f"  ✓ Created delisting.txt")


def copy_alpha_files(
    source_dir: str,
    dest_dir: str,
    dates: List[str],
    horizons: List[int],
    models: List[str]
) -> int:
    """Copy alpha files from source to destination.
    
    Args:
        source_dir: Source alpha directory
        dest_dir: Destination directory
        dates: List of dates in YYYYMMDD format
        horizons: List of horizons to copy
        models: List of models to copy
        
    Returns:
        Number of files copied
    """
    print(f"\nCopying alpha files...")
    os.makedirs(dest_dir, exist_ok=True)
    
    total_copied = 0
    
    for horizon in horizons:
        for model in models:
            horizon_model_dir = os.path.join(dest_dir, str(horizon), model)
            os.makedirs(horizon_model_dir, exist_ok=True)
            
            copied_count = 0
            for date in dates:
                filename = f"alphas.{horizon}.{model}.{date}.parquet"
                source_file = os.path.join(source_dir, str(horizon), model, filename)
                dest_file = os.path.join(horizon_model_dir, filename)
                
                if os.path.exists(source_file):
                    shutil.copy2(source_file, dest_file)
                    copied_count += 1
                else:
                    print(f"  ⚠ Warning: {source_file} not found")
            
            if copied_count > 0:
                print(f"  ✓ {horizon}/{model}: {copied_count} files")
                total_copied += copied_count
    
    print(f"\nTotal alpha files copied: {total_copied}")
    return total_copied