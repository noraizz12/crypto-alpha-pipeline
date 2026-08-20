#!/usr/bin/env python3
"""Generate models test fixtures from production data.

IMPORTANT: All data must be copied from ~/stat_arb/data/ (production data directories).
           Do not create synthetic or dummy data files.
           Do NOT create symlinks - copy actual files to maintain test isolation.

This script:
1. Copies necessary bar and feature files from the production data directory
2. Filters to only the symbols specified in the config
3. Generates master fixture model files for test comparison

Usage:
    python generate_fixtures.py                    # Generate all fixtures
    python generate_fixtures.py --master-only      # Only regenerate master fixtures

Complete list of files that should be generated in test/fixtures/models/data/:
    bars/1440/binance-futures/20250106/bars.1440.binance-futures.20250106.BNBUSDT.parquet
    bars/1440/binance-futures/20250106/bars.1440.binance-futures.20250106.BTCUSDT.parquet
    bars/1440/binance-futures/20250106/bars.1440.binance-futures.20250106.ETHUSDT.parquet
    bars/1440/binance-futures/20250107/bars.1440.binance-futures.20250107.BNBUSDT.parquet
    bars/1440/binance-futures/20250107/bars.1440.binance-futures.20250107.BTCUSDT.parquet
    bars/1440/binance-futures/20250107/bars.1440.binance-futures.20250107.ETHUSDT.parquet
    bars/1440/binance-futures/20250108/bars.1440.binance-futures.20250108.BNBUSDT.parquet
    bars/1440/binance-futures/20250108/bars.1440.binance-futures.20250108.BTCUSDT.parquet
    bars/1440/binance-futures/20250108/bars.1440.binance-futures.20250108.ETHUSDT.parquet
    features/1440/[various feature names]/[dates]/[symbol files] (generated)
    universe/universe.20250106.parquet
    universe/universe.20250107.parquet
    universe/universe.20250108.parquet
"""

import os
import sys
import argparse
import shutil
import glob

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

# Import common fixture utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fixture_utils import (
    load_test_config, save_test_config,
    get_date_range, update_config_for_test, print_test_summary,
    copy_bar_files, copy_universe_files, copy_feature_files, DEFAULT_TEST_SYMBOLS
)

from lib.util.directory import dir_manager, DirectoryManager
from lib.util.time_util import date_str_to_date
from lib.alpha.model_calcs import ModelCalcs
from lib.alpha.features import Features


# Test configuration
TEST_DATE = "20250108"
HISTORICAL_DAYS = 8  # Need extra days for feature lookback (3 days lookback + 3 test days = 6, plus buffer)
TEST_HORIZON = 1440  # Daily models for testing


def generate_master_fixtures(config: dict, dates: list, horizon: int) -> None:
    """Generate master fixture model files that tests will compare against."""
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
    trading_dir = os.path.join(script_dir, "trading")
    os.makedirs(trading_dir, exist_ok=True)
    
    # Create empty delisting.txt file if it doesn't exist
    delisting_file = os.path.join(trading_dir, "delisting.txt")
    if not os.path.exists(delisting_file):
        with open(delisting_file, 'w') as f:
            f.write("")  # Empty file for test
    
    test_dir_manager = DirectoryManager(
        data_dir=fixture_input_dir,
        trading_dir=trading_dir
    )
    
    # Process each date
    for date_str in dates:
        date = date_str_to_date(date_str)
        print(f"\nGenerating models for {date_str}...")
        
        try:
            # Create models calculator
            model_calc = ModelCalcs(
                config=config,
                models_to_run=['hl'],  # Only hl model as per typical test
                horizons=[horizon],
                debug=False,
                pool_size=1,
                models_dir_manager=test_dir_manager,
            )
            
            # Run model generation for the specific date
            model_calc.process_models(start_date=date, end_date=date)
            
            print(f"  ✓ Generated master fixtures for {date_str}")
            
        except Exception as e:
            print(f"  ✗ Failed to generate master fixtures for {date_str}: {e}")
            raise
    
    # Move generated models to master directory
    # Create master horizon directory
    master_horizon_dir = os.path.join(fixture_output_dir, str(horizon))
    os.makedirs(master_horizon_dir, exist_ok=True)
    
    # Look for model files in both possible locations
    possible_model_dirs = [
        os.path.join(fixture_input_dir, "models", str(horizon), "hl"),
        os.path.join(fixture_input_dir, "tardis_models", str(horizon), "hl")
    ]
    
    model_moved = False
    for models_dir in possible_model_dirs:
        if os.path.exists(models_dir):
            # Move all model files for the test date
            for model_file in glob.glob(os.path.join(models_dir, f"models.{horizon}.hl.{dates[-1]}.parquet")):
                dest_path = os.path.join(master_horizon_dir, os.path.basename(model_file))
                shutil.move(model_file, dest_path)
                print(f"  → Moved {os.path.basename(model_file)} to master from {os.path.basename(models_dir)}")
                model_moved = True
                break
        if model_moved:
            break
    
    if not model_moved:
        print("  ⚠ Model file not found in any expected directory")
    
    # Clean up any remaining model directories from data
    for subdir in ["models", "tardis_models"]:
        model_data_dir = os.path.join(fixture_input_dir, subdir)
        if os.path.exists(model_data_dir):
            shutil.rmtree(model_data_dir)
            print(f"  ✓ Cleaned up {subdir} directory from data")


def main():
    """Main function to generate model test fixtures."""
    parser = argparse.ArgumentParser(description='Generate model test fixtures')
    parser.add_argument('--master-only', action='store_true', 
                        help='Only regenerate master fixtures (skip copying data)')
    args = parser.parse_args()
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config_models_generation_test.json")
    
    # Load configuration
    config = load_test_config(config_path)
    
    # Use test symbols from config
    model_test_date = config.get('MODEL_TEST_DATE', TEST_DATE)
    symbols = set(config.get('SYMBOL_UNIVERSE', DEFAULT_TEST_SYMBOLS))
    config = update_config_for_test(config, symbols=list(symbols), lookback_days=3)
    
    # Calculate date range
    dates = get_date_range(model_test_date, HISTORICAL_DAYS)
    test_dates = [model_test_date]  # Only generate models for the test date
    
    # Print test summary
    print_test_summary("models", list(symbols), model_test_date, dates, [TEST_HORIZON])
    
    if not args.master_only:
        # IMPORTANT: Delete ALL data in fixtures/models/data/ for a clean slate
        data_dir = os.path.join(script_dir, "data")
        print("\nStep 1: Cleaning ALL existing data files...")
        print("=" * 60)
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            print(f"  ✓ Removed entire data directory: {data_dir}")
        else:
            print(f"  ℹ Data directory did not exist: {data_dir}")
        
        # Recreate the data directory structure
        os.makedirs(data_dir, exist_ok=True)
        print(f"  ✓ Created fresh data directory: {data_dir}")
        
        print("\nStep 2: Copying production data files...")
        print("=" * 60)
        
        # Copy bar files (only 1440 as per file list in docstring)
        bar_count = copy_bar_files(
            dir_manager.BAR_DIR,
            os.path.join(data_dir, "bars"),
            symbols,
            dates,
            [TEST_HORIZON]  # Only 1440 minute bars as per the file list
        )
        
        # Copy specific feature files needed by the model
        print("\nCopying feature files...")
        
        # Get the features needed by the model from the config
        model_features = config.get('FCASTS', {}).get(str(TEST_HORIZON), {}).get('features', [])
        print(f"Model requires features: {model_features}")
        
        feature_count = 0
        for feature_name in model_features:
            # Handle HORIZON placeholder
            actual_feature_name = feature_name.replace('_HORIZON', f'_{TEST_HORIZON}')
            
            for date_str in dates:
                source_feature_dir = os.path.join(os.environ.get('ROOT_DIR', ''), "data/features", str(TEST_HORIZON), actual_feature_name, date_str)
                dest_feature_dir = os.path.join(data_dir, "features", str(TEST_HORIZON), actual_feature_name, date_str)
                
                if os.path.exists(source_feature_dir):
                    os.makedirs(dest_feature_dir, exist_ok=True)
                    
                    # Copy all symbol files for this feature and date
                    copied = 0
                    for symbol in symbols:
                        filename = f"features.{TEST_HORIZON}.{actual_feature_name}.{date_str}.{symbol}USDT.parquet"
                        source_file = os.path.join(source_feature_dir, filename)
                        dest_file = os.path.join(dest_feature_dir, filename)
                        
                        if os.path.exists(source_file):
                            shutil.copy2(source_file, dest_file)
                            copied += 1
                    
                    if copied > 0:
                        print(f"  {actual_feature_name}/{date_str}: Copied {copied} files")
                        feature_count += copied
        
        print(f"\nTotal: {feature_count} feature files copied")
        
        # Copy universe files from production data directory
        # Use the actual production universe directory, not the test directory manager's path
        prod_universe_dir = os.path.join(os.environ.get('ROOT_DIR', ''), "data/universe")
        universe_count = copy_universe_files(
            prod_universe_dir,
            os.path.join(data_dir, "universe"),
            dates
        )
        
        print("\nStep 3: Verifying generated files...")
        print("=" * 60)
        
        # List of expected files from docstring
        expected_files = [
            "bars/1440/binance-futures/20250106/bars.1440.binance-futures.20250106.BNBUSDT.parquet",
            "bars/1440/binance-futures/20250106/bars.1440.binance-futures.20250106.BTCUSDT.parquet",
            "bars/1440/binance-futures/20250106/bars.1440.binance-futures.20250106.ETHUSDT.parquet",
            "bars/1440/binance-futures/20250107/bars.1440.binance-futures.20250107.BNBUSDT.parquet",
            "bars/1440/binance-futures/20250107/bars.1440.binance-futures.20250107.BTCUSDT.parquet",
            "bars/1440/binance-futures/20250107/bars.1440.binance-futures.20250107.ETHUSDT.parquet",
            "bars/1440/binance-futures/20250108/bars.1440.binance-futures.20250108.BNBUSDT.parquet",
            "bars/1440/binance-futures/20250108/bars.1440.binance-futures.20250108.BTCUSDT.parquet",
            "bars/1440/binance-futures/20250108/bars.1440.binance-futures.20250108.ETHUSDT.parquet",
            # Features will be generated with various feature names, not just 'fittable'
            # We'll skip checking for specific feature files since they are generated
            "universe/universe.20250106.parquet",
            "universe/universe.20250107.parquet",
            "universe/universe.20250108.parquet"
        ]
        
        # Check each expected file exists
        missing_files = []
        for expected_file in expected_files:
            full_path = os.path.join(data_dir, expected_file)
            if not os.path.exists(full_path):
                missing_files.append(expected_file)
        
        # Count actual files
        actual_files = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.parquet'):
                    rel_path = os.path.relpath(os.path.join(root, f), data_dir)
                    actual_files.append(rel_path)
        
        # Verify counts
        expected_bar_files = 9  # 3 dates * 3 symbols * 1 frequency
        expected_feature_files = 9  # 3 dates * 3 symbols * 1 feature type
        expected_universe_files = 3  # 3 dates
        expected_total = expected_bar_files + expected_feature_files + expected_universe_files  # 21 total
        
        print(f"Expected file count: {expected_total}")
        print(f"  - Bar files: {expected_bar_files}")
        print(f"  - Feature files: {expected_feature_files}")
        print(f"  - Universe files: {expected_universe_files}")
        
        print(f"\nActual file count: {len(actual_files)}")
        print(f"  - Bar files copied: {bar_count}")
        print(f"  - Feature files generated: {feature_count}")
        print(f"  - Universe files copied: {universe_count}")
        
        # Check if feature directory was created
        features_dir = os.path.join(data_dir, "features", str(TEST_HORIZON))
        if os.path.exists(features_dir):
            print(f"  ✓ Features directory created: {features_dir}")
        
        if missing_files:
            print(f"\n⚠️  ERROR: Missing {len(missing_files)} expected files:")
            for f in missing_files:
                print(f"  ✗ {f}")
            raise RuntimeError(f"Failed to generate all expected files. Missing {len(missing_files)} files.")
        
        # Check for unexpected files
        unexpected_files = [f for f in actual_files if f not in expected_files]
        if unexpected_files:
            print(f"\n⚠️  WARNING: Found {len(unexpected_files)} unexpected files:")
            for f in unexpected_files:
                print(f"  ? {f}")
        
        if len(actual_files) == expected_total and not missing_files:
            print(f"\n✅ All {expected_total} expected files successfully generated!")
    
    # Generate master fixtures
    generate_master_fixtures(config, test_dates, TEST_HORIZON)
    
    # Save updated config
    save_test_config(config_path, config)
    print("\n✓ Updated config")
    
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
