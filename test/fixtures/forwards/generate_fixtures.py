#!/usr/bin/env python3
"""Generate forwards test fixtures from production data.

IMPORTANT: All data must be copied from ~/stat_arb/data/ (production data directories).
           Do not create synthetic or dummy data files.
           Do NOT create symlinks - copy actual files to maintain test isolation.

This script:
1. Copies bar and universe files from production for specified symbols
2. Generates forwards using the Forwards class
3. Creates master fixtures for test comparison

Usage:
    python generate_fixtures.py                    # Generate all fixtures
    python generate_fixtures.py --master-only      # Only regenerate master fixtures

Complete list of files that should be generated in test/fixtures/forwards/data/:
    bars/15/binance-futures/20250106/bars.15.binance-futures.20250106.BNBUSDT.parquet
    bars/15/binance-futures/20250106/bars.15.binance-futures.20250106.BTCUSDT.parquet
    bars/15/binance-futures/20250106/bars.15.binance-futures.20250106.ETHUSDT.parquet
    bars/15/binance-futures/20250107/bars.15.binance-futures.20250107.BNBUSDT.parquet
    bars/15/binance-futures/20250107/bars.15.binance-futures.20250107.BTCUSDT.parquet
    bars/15/binance-futures/20250107/bars.15.binance-futures.20250107.ETHUSDT.parquet
    bars/15/binance-futures/20250108/bars.15.binance-futures.20250108.BNBUSDT.parquet
    bars/15/binance-futures/20250108/bars.15.binance-futures.20250108.BTCUSDT.parquet
    bars/15/binance-futures/20250108/bars.15.binance-futures.20250108.ETHUSDT.parquet
    universe/universe.20250105.parquet  # Needed for forwards calculation on 20250106
    universe/universe.20250106.parquet
    universe/universe.20250107.parquet
    universe/universe.20250108.parquet
"""

import os
import sys
import argparse
import shutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

# Import common fixture utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# pylint: disable=import-error,wrong-import-position,wrong-import-order
from fixture_utils import (
    load_test_config, save_test_config,
    get_date_range, update_config_for_test, print_test_summary,
    copy_bar_files, copy_universe_files, copy_forward_files, DEFAULT_TEST_SYMBOLS
)

from lib.util.directory import dir_manager, DirectoryManager
from lib.util.time_util import date_str_to_date
from lib.fits.forwards import Forwards


# Test configuration
TEST_DATE = "20250108"  # Last date in the file list
HISTORICAL_DAYS = 2  # Only need dates from 20250106 to 20250108
TEST_HORIZONS = [15]  # Test with 15-minute forwards only
BAR_FREQUENCIES = [15]  # Only 15-minute bars as per the file list


def generate_master_fixtures(config: dict, dates: list, horizons: list) -> None:
    """Generate master forwards fixtures."""
    print("\n" + "=" * 60)
    print("GENERATING MASTER FIXTURES")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_input_dir = os.path.join(script_dir, "data")
    fixture_output_dir = os.path.join(script_dir, "master")
    
    # Clean master directory completely before generating new fixtures
    print("Cleaning master directory...")
    if os.path.exists(fixture_output_dir):
        shutil.rmtree(fixture_output_dir)
        print("  ✓ Removed entire master directory")
    os.makedirs(fixture_output_dir, exist_ok=True)
    
    # Create a test-specific directory manager
    test_dir_manager = DirectoryManager(
        data_dir=fixture_input_dir,
        trading_dir=os.path.join(script_dir, "trading")
    )
    
    # Convert date strings to datetime objects
    start_date = date_str_to_date(dates[0])
    end_date = date_str_to_date(dates[-1])
    
    try:
        # Create forwards calculator with correct parameters
        forwards_calc = Forwards(
            config=config,
            update=False,  # Don't use live data update mode
            horizons=horizons,
            debug=False,
            forwards_dir_manager=test_dir_manager,
        )
        
        # Generate forwards for the date range
        forwards_calc.generate_forwards(start_date=start_date, end_date=end_date)
        
        print(f"  ✓ Generated forwards for horizons {horizons}")
        
    except Exception as e:
        print(f"  ✗ Failed to generate master fixtures: {e}")
        raise
    
    # Copy generated forwards to master directory using fixture_utils
    forwards_dir = os.path.join(fixture_input_dir, "forwards")
    if os.path.exists(forwards_dir):
        # Get the symbols from config
        symbols = set(DEFAULT_TEST_SYMBOLS)  # BTC, ETH, BNB
        
        # Use copy_forward_files from fixture_utils
        copied = copy_forward_files(
            forwards_dir,
            fixture_output_dir,
            symbols,
            dates,
            horizons
        )
        
        print(f"  → Copied {copied} forward files to master")
    else:
        print("  ⚠ Warning: No forwards directory found after generation")


def main():
    """Main function to generate forwards test fixtures."""
    parser = argparse.ArgumentParser(description='Generate forwards test fixtures')
    parser.add_argument('--master-only', action='store_true', 
                        help='Only regenerate master fixtures (skip copying data)')
    args = parser.parse_args()
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config_forwards_generation_test.json")
    data_dir = os.path.join(script_dir, "data")
    
    # Load configuration
    config = load_test_config(config_path)
    
    # Use only the symbols specified in the docstring
    symbols = set(DEFAULT_TEST_SYMBOLS)  # Only BTC, ETH, BNB as documented
    config = update_config_for_test(config, symbols=list(symbols))
    
    # Calculate date range
    dates = get_date_range(TEST_DATE, HISTORICAL_DAYS)
    
    # Print test summary
    print_test_summary("forwards", list(symbols), TEST_DATE, dates, TEST_HORIZONS)
    
    if not args.master_only:
        # IMPORTANT: Delete ALL data in fixtures/forwards/data/ for a clean slate
        print("\nCleaning ALL data files...")
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            print(f"  ✓ Removed entire data directory: {data_dir}")
        
        # Recreate the data directory structure
        os.makedirs(data_dir, exist_ok=True)
        print(f"  ✓ Created fresh data directory: {data_dir}")
        
        # Now use fixture_utils functions to copy required files
        # Copy bar files (only 15-minute bars as per the file list)
        copy_bar_files(
            dir_manager.BAR_DIR,
            os.path.join(data_dir, "bars"),
            symbols,
            dates,
            BAR_FREQUENCIES
        )
        
        # Copy universe files
        # IMPORTANT: Include one extra day before (20250105) for forwards calculation
        # Forwards for date T need universe from date T-1
        universe_dates = ['20250105'] + dates  # Add the day before
        copy_universe_files(
            dir_manager.UNIVERSE_DIR,
            os.path.join(data_dir, "universe"),
            universe_dates
        )
        
        # Verify we have the expected number of files as per the docstring
        expected_bar_files = 9  # 3 dates * 3 symbols * 1 frequency (15-minute)
        expected_universe_files = 4  # 4 dates (including 20250105 for forwards calculation)
        expected_total = expected_bar_files + expected_universe_files  # 13 total
        
        actual_files = 0
        for _, _, files in os.walk(data_dir):
            actual_files += len([f for f in files if f.endswith('.parquet')])
        
        print(f"\nExpected files: {expected_total} ({expected_bar_files} bars + {expected_universe_files} universe)")
        print(f"Actual files copied: {actual_files}")
        
        if actual_files != expected_total:
            print(f"⚠️  WARNING: File count mismatch! Expected {expected_total} but copied {actual_files}")
    
    # Generate master fixtures
    generate_master_fixtures(config, dates, TEST_HORIZONS)
    
    # Save updated config
    save_test_config(config_path, config)
    print("\n✓ Updated config with test symbols")
    
    print("\n" + "=" * 60)
    print("✅ Fixture generation complete!")
    print("=" * 60)
    print("\nTo regenerate all fixtures:")
    print(f"  cd {script_dir}")
    print("  python generate_fixtures.py")
    print("\nTo regenerate only master fixtures:")
    print("  python generate_fixtures.py --master-only")


if __name__ == "__main__":
    main()
