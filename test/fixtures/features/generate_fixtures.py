#!/usr/bin/env python3
"""Generate feature test fixtures from production bar data.

IMPORTANT: Do NOT create symlinks - copy actual files to maintain test isolation.

This script:
1. Copies bar files from the production data directory
2. Filters to only the symbols specified in the config
3. Generates master fixture feature files for test comparison

Usage:
    python generate_fixtures.py                    # Generate all fixtures
    python generate_fixtures.py --master-only      # Only regenerate master fixtures

Complete list of files that should be generated in test/fixtures/features/data/:
    bars/1/binance-futures/20250606/bars.1.binance-futures.20250606.ADAUSDT.parquet
    bars/1/binance-futures/20250606/bars.1.binance-futures.20250606.BNBUSDT.parquet
    bars/1/binance-futures/20250606/bars.1.binance-futures.20250606.BTCUSDT.parquet
    bars/1/binance-futures/20250606/bars.1.binance-futures.20250606.ETCUSDT.parquet
    bars/1/binance-futures/20250606/bars.1.binance-futures.20250606.ETHUSDT.parquet
    bars/1/binance-futures/20250607/bars.1.binance-futures.20250607.ADAUSDT.parquet
    bars/1/binance-futures/20250607/bars.1.binance-futures.20250607.BNBUSDT.parquet
    bars/1/binance-futures/20250607/bars.1.binance-futures.20250607.BTCUSDT.parquet
    bars/1/binance-futures/20250607/bars.1.binance-futures.20250607.ETCUSDT.parquet
    bars/1/binance-futures/20250607/bars.1.binance-futures.20250607.ETHUSDT.parquet
    bars/1/binance-futures/20250607/bars.1.binance-futures.20250607.ZILUSDT.parquet
    bars/1/binance-futures/20250608/bars.1.binance-futures.20250608.ADAUSDT.parquet
    bars/1/binance-futures/20250608/bars.1.binance-futures.20250608.BNBUSDT.parquet
    bars/1/binance-futures/20250608/bars.1.binance-futures.20250608.BTCUSDT.parquet
    bars/1/binance-futures/20250608/bars.1.binance-futures.20250608.ETCUSDT.parquet
    bars/1/binance-futures/20250608/bars.1.binance-futures.20250608.ETHUSDT.parquet
    bars/1/binance-futures/20250608/bars.1.binance-futures.20250608.ZILUSDT.parquet
    bars/1/binance-futures/20250609/bars.1.binance-futures.20250609.ADAUSDT.parquet
    bars/1/binance-futures/20250609/bars.1.binance-futures.20250609.BNBUSDT.parquet
    bars/1/binance-futures/20250609/bars.1.binance-futures.20250609.BTCUSDT.parquet
    bars/1/binance-futures/20250609/bars.1.binance-futures.20250609.ETCUSDT.parquet
    bars/1/binance-futures/20250609/bars.1.binance-futures.20250609.ETHUSDT.parquet
    bars/1/binance-futures/20250609/bars.1.binance-futures.20250609.ZILUSDT.parquet
    bars/1440/binance-futures/20250606/bars.1440.binance-futures.20250606.ADAUSDT.parquet
    bars/1440/binance-futures/20250606/bars.1440.binance-futures.20250606.BNBUSDT.parquet
    bars/1440/binance-futures/20250606/bars.1440.binance-futures.20250606.BTCUSDT.parquet
    bars/1440/binance-futures/20250606/bars.1440.binance-futures.20250606.ETCUSDT.parquet
    bars/1440/binance-futures/20250606/bars.1440.binance-futures.20250606.ETHUSDT.parquet
    bars/1440/binance-futures/20250607/bars.1440.binance-futures.20250607.ADAUSDT.parquet
    bars/1440/binance-futures/20250607/bars.1440.binance-futures.20250607.BNBUSDT.parquet
    bars/1440/binance-futures/20250607/bars.1440.binance-futures.20250607.BTCUSDT.parquet
    bars/1440/binance-futures/20250607/bars.1440.binance-futures.20250607.ETCUSDT.parquet
    bars/1440/binance-futures/20250607/bars.1440.binance-futures.20250607.ETHUSDT.parquet
    bars/1440/binance-futures/20250607/bars.1440.binance-futures.20250607.ZILUSDT.parquet
    bars/1440/binance-futures/20250608/bars.1440.binance-futures.20250608.ADAUSDT.parquet
    bars/1440/binance-futures/20250608/bars.1440.binance-futures.20250608.BNBUSDT.parquet
    bars/1440/binance-futures/20250608/bars.1440.binance-futures.20250608.BTCUSDT.parquet
    bars/1440/binance-futures/20250608/bars.1440.binance-futures.20250608.ETCUSDT.parquet
    bars/1440/binance-futures/20250608/bars.1440.binance-futures.20250608.ETHUSDT.parquet
    bars/1440/binance-futures/20250608/bars.1440.binance-futures.20250608.ZILUSDT.parquet
    bars/1440/binance-futures/20250609/bars.1440.binance-futures.20250609.ADAUSDT.parquet
    bars/1440/binance-futures/20250609/bars.1440.binance-futures.20250609.BNBUSDT.parquet
    bars/1440/binance-futures/20250609/bars.1440.binance-futures.20250609.BTCUSDT.parquet
    bars/1440/binance-futures/20250609/bars.1440.binance-futures.20250609.ETCUSDT.parquet
    bars/1440/binance-futures/20250609/bars.1440.binance-futures.20250609.ETHUSDT.parquet
    bars/1440/binance-futures/20250609/bars.1440.binance-futures.20250609.ZILUSDT.parquet
    delisting.txt
"""

import os
import sys
import argparse
import shutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

# Import common fixture utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fixture_utils import (
    load_test_config, save_test_config, clean_fixture_directory,
    get_date_range, update_config_for_test, print_test_summary,
    copy_bar_files, create_delisting_file
)

from lib.util.directory import dir_manager, DirectoryManager
from lib.util.time_util import date_str_to_date
from lib.alpha.features import Features


# Test configuration
TEST_DATE = "20250609"
HISTORICAL_DAYS = 3  # Need history for feature lookback calculations
TEST_FREQUENCY = 1440  # Daily features for testing
TEST_SYMBOLS = ['ADA', 'BNB', 'BTC', 'ETC', 'ETH', 'ZIL']  # Exact symbols from file list


def generate_master_fixtures(config: dict, dates: list, horizons: list) -> None:
    """Generate master fixture feature files that tests will compare against."""
    print("\n" + "=" * 60)
    print("GENERATING MASTER FIXTURES")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_input_dir = os.path.join(script_dir, "data")
    fixture_output_dir = os.path.join(script_dir, "master")
    
    # Clean master directory before generating new fixtures
    print("Cleaning master directory...")
    clean_fixture_directory(fixture_output_dir, [""])
    
    # Create a test-specific directory manager
    test_dir_manager = DirectoryManager(
        data_dir=fixture_input_dir,
        trading_dir=os.path.join(script_dir, "trading")
    )
    
    # Process each date
    for date_str in dates:
        date = date_str_to_date(date_str)
        print(f"\nGenerating features for {date_str}...")
        
        try:
            # Create features calculator - only pass horizon as frequency
            for horizon in horizons:
                features_calc = Features(
                    config=config,
                    frequency=horizon,
                    prod=False,  # Generate all features, not just production
                    debug=False,
                    features=None,  # Use default features from config
                    output_dir=os.path.join(fixture_input_dir, "features"),
                    features_dir_manager=test_dir_manager,
                )
                
                # Run feature generation
                features_calc.run(start_date=date, end_date=date)
            
            print(f"  ✓ Generated master fixtures for {date_str}")
            
        except Exception as e:
            print(f"  ✗ Failed to generate master fixtures for {date_str}: {e}")
            raise
    
    # Move generated features to master directory
    
    # Clear master directory first
    if os.path.exists(fixture_output_dir):
        shutil.rmtree(fixture_output_dir)
    os.makedirs(fixture_output_dir, exist_ok=True)
    
    # Copy all generated feature files to master
    features_dir = os.path.join(fixture_input_dir, "features")
    if os.path.exists(features_dir):
        for horizon in horizons:
            horizon_dir = os.path.join(features_dir, str(horizon))
            if os.path.exists(horizon_dir):
                dest_horizon_dir = os.path.join(fixture_output_dir, str(horizon))
                shutil.copytree(horizon_dir, dest_horizon_dir)
                print(f"  → Copied horizon {horizon} features to master")


def main():
    """Main function to generate feature test fixtures."""
    parser = argparse.ArgumentParser(description='Generate feature test fixtures')
    parser.add_argument('--master-only', action='store_true', 
                        help='Only regenerate master fixtures (skip copying bars)')
    args = parser.parse_args()
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config_features_generation_test.json")
    data_dir = os.path.join(script_dir, "data")
    
    # Load configuration
    config = load_test_config(config_path)
    
    # Use exact symbols from the file list
    symbols = set(TEST_SYMBOLS)
    config = update_config_for_test(config, symbols=list(symbols), lookback_days=2)
    
    # Calculate date range
    dates = get_date_range(TEST_DATE, HISTORICAL_DAYS)
    test_dates = [TEST_DATE]  # Only generate features for the test date
    
    # Print test summary
    print_test_summary("features", sorted(list(symbols)), TEST_DATE, dates, [1, TEST_FREQUENCY])
    
    if not args.master_only:
        # IMPORTANT: Delete ALL data in fixtures/features/data/ for a clean slate
        print("\nCleaning ALL data files...")
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            print(f"  ✓ Removed entire data directory: {data_dir}")
        
        # Recreate the data directory structure
        os.makedirs(data_dir, exist_ok=True)
        print(f"  ✓ Created fresh data directory: {data_dir}")
        
        # Set up directories
        source_base_dir = dir_manager.BAR_DIR
        dest_base_dir = os.path.join(data_dir, "bars")
        
        # Copy bar files - need both 1-minute and daily bars for feature calculation
        copy_bar_files(source_base_dir, dest_base_dir, symbols, dates, [1, TEST_FREQUENCY])
        
        # Create delisting file
        create_delisting_file(data_dir)
        
        # Verify we have the expected files
        # expected_bar_files = 46  # From the docstring list (excluding delisting.txt)
        actual_bar_files = 0
        
        bars_dir = os.path.join(data_dir, "bars")
        if os.path.exists(bars_dir):
            for _, _, files in os.walk(bars_dir):
                actual_bar_files += len([f for f in files if f.endswith('.parquet')])
        
        # Count delisting.txt
        if os.path.exists(os.path.join(data_dir, "delisting.txt")):
            actual_bar_files += 1
        
        print("\nExpected files: 47 (46 bars + 1 delisting.txt)")
        print(f"Actual files copied: {actual_bar_files}")
        
        if actual_bar_files != 47:
            print(f"⚠️  WARNING: File count mismatch! Expected 47 but copied {actual_bar_files}")
    
    # Generate master fixtures
    generate_master_fixtures(config, test_dates, [TEST_FREQUENCY])
    
    # Save updated config
    save_test_config(config_path, config)
    print("\n✓ Updated config with minimal test symbols")
    
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
