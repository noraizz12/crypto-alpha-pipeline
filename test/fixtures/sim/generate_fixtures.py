#!/usr/bin/env python3
"""Generate sim test fixtures from production data.

IMPORTANT: All data must be copied from ~/stat_arb/data/ (production data directories).
           Do not create synthetic or dummy data files.
           Do NOT create symlinks - copy actual files to maintain test isolation.

This script:
1. Copies necessary input data (bars, features, alphas) from production
2. Updates configuration for minimal test setup
3. Runs simulation to generate master fixtures

Usage:
    python generate_fixtures.py                    # Generate all fixtures
    python generate_fixtures.py --master-only      # Only regenerate master fixtures

Complete list of files that should be generated in test/fixtures/sim/data/:
    Date range: 20240705-20240707 (3 dates)
    Symbols: ADAUSDT, BNBUSDT, BTCUSDT, ETCUSDT, ETHUSDT (5 symbols)
    
    Bar files (30 total = 2 frequencies * 3 dates * 5 symbols):
    bars/360/binance-futures/20240705/bars.360.binance-futures.20240705.ADAUSDT.parquet
    bars/360/binance-futures/20240705/bars.360.binance-futures.20240705.BNBUSDT.parquet
    bars/360/binance-futures/20240705/bars.360.binance-futures.20240705.BTCUSDT.parquet
    bars/360/binance-futures/20240705/bars.360.binance-futures.20240705.ETCUSDT.parquet
    bars/360/binance-futures/20240705/bars.360.binance-futures.20240705.ETHUSDT.parquet
    bars/360/binance-futures/20240706/bars.360.binance-futures.20240706.ADAUSDT.parquet
    bars/360/binance-futures/20240706/bars.360.binance-futures.20240706.BNBUSDT.parquet
    bars/360/binance-futures/20240706/bars.360.binance-futures.20240706.BTCUSDT.parquet
    bars/360/binance-futures/20240706/bars.360.binance-futures.20240706.ETCUSDT.parquet
    bars/360/binance-futures/20240706/bars.360.binance-futures.20240706.ETHUSDT.parquet
    bars/360/binance-futures/20240707/bars.360.binance-futures.20240707.ADAUSDT.parquet
    bars/360/binance-futures/20240707/bars.360.binance-futures.20240707.BNBUSDT.parquet
    bars/360/binance-futures/20240707/bars.360.binance-futures.20240707.BTCUSDT.parquet
    bars/360/binance-futures/20240707/bars.360.binance-futures.20240707.ETCUSDT.parquet
    bars/360/binance-futures/20240707/bars.360.binance-futures.20240707.ETHUSDT.parquet
    bars/1440/binance-futures/20240705/bars.1440.binance-futures.20240705.ADAUSDT.parquet
    bars/1440/binance-futures/20240705/bars.1440.binance-futures.20240705.BNBUSDT.parquet
    bars/1440/binance-futures/20240705/bars.1440.binance-futures.20240705.BTCUSDT.parquet
    bars/1440/binance-futures/20240705/bars.1440.binance-futures.20240705.ETCUSDT.parquet
    bars/1440/binance-futures/20240705/bars.1440.binance-futures.20240705.ETHUSDT.parquet
    bars/1440/binance-futures/20240706/bars.1440.binance-futures.20240706.ADAUSDT.parquet
    bars/1440/binance-futures/20240706/bars.1440.binance-futures.20240706.BNBUSDT.parquet
    bars/1440/binance-futures/20240706/bars.1440.binance-futures.20240706.BTCUSDT.parquet
    bars/1440/binance-futures/20240706/bars.1440.binance-futures.20240706.ETCUSDT.parquet
    bars/1440/binance-futures/20240706/bars.1440.binance-futures.20240706.ETHUSDT.parquet
    bars/1440/binance-futures/20240707/bars.1440.binance-futures.20240707.ADAUSDT.parquet
    bars/1440/binance-futures/20240707/bars.1440.binance-futures.20240707.BNBUSDT.parquet
    bars/1440/binance-futures/20240707/bars.1440.binance-futures.20240707.BTCUSDT.parquet
    bars/1440/binance-futures/20240707/bars.1440.binance-futures.20240707.ETCUSDT.parquet
    bars/1440/binance-futures/20240707/bars.1440.binance-futures.20240707.ETHUSDT.parquet
    
    Feature files for horizon 360 (90 total = 6 feature types * 3 dates * 5 symbols):
    - dvolume_360, dvolume_360_trmean, dvolume_360_trmean_cz
    - logret_360_lz, logret_360_trstd, relative_spread_360
    
    Feature files for horizon 1440 (150 total = 10 feature types * 3 dates * 5 symbols):
    - beta_1440, dvolume_1440, dvolume_1440_trmean, dvolume_1440_trmean_cz
    - logret_1440_cz, logret_1440_lz, logret_1440_lz_cz, logret_1440_trstd
    - relative_spread_1440_trmean, risk_1440
    
    Alpha files in tardis_prod_alpha (4 total = 2 models * 2 sim dates):
    tardis_prod_alpha/hl_15_20240706.parquet
    tardis_prod_alpha/hl_15_20240707.parquet
    tardis_prod_alpha/c2vwap_720_20240706.parquet
    tardis_prod_alpha/c2vwap_720_20240707.parquet
    
    Universe files (3 total = 3 dates):
    universe/universe.20240705.parquet
    universe/universe.20240706.parquet  
    universe/universe.20240707.parquet
    
    Other files (1 total):
    delisting.txt
    
    TOTAL FILES: 278
"""

import os
import sys
import argparse
import shutil
from datetime import timedelta
from typing import List, Dict

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# pylint: disable=import-error,wrong-import-position
from fixture_utils import (
    load_test_config, save_test_config,
    update_config_for_test,
    copy_bar_files, copy_feature_files, copy_universe_files,
    create_delisting_file
)
from lib.util.directory import dir_manager, DirectoryManager
from lib.util.time_util import date_to_str, date_str_to_date
from lib.sim.simulate import Simulate
# pylint: enable=import-error,wrong-import-position

# Test configuration
TEST_START_DATE = "20240706"
TEST_END_DATE = "20240707"  # 2 days of simulation
TEST_SYMBOLS = ['BTC', 'ETH', 'BNB', 'ADA', 'ETC']  # Symbols used in master fixtures
HISTORICAL_DAYS = 1  # Extra day for initial positions
BAR_FREQUENCIES = [360, 1440]  # From config and simulate.py - only these are needed
FEATURE_HORIZONS = [360, 1440]  # Horizons needed for simulation
ALPHA_MODELS = ['hl_15', 'c2vwap_720']  # Models used in simulation


def get_required_features_for_sim() -> Dict[int, List[str]]:
    """Get all features required for simulation, organized by horizon.
    
    Based on the actual files in the sim fixture data directory.
    """
    features_by_horizon = {
        360: [
            'dvolume_360',
            'dvolume_360_trmean', 
            'dvolume_360_trmean_cz',
            'logret_360_lz',
            'logret_360_trstd',
            'relative_spread_360',
        ],
        1440: [
            'beta_1440',
            'dvolume_1440',
            'dvolume_1440_trmean',
            'dvolume_1440_trmean_cz',
            'logret_1440_cz',
            'logret_1440_lz',
            'logret_1440_lz_cz',
            'logret_1440_trstd',
            'relative_spread_1440_trmean',
            'risk_1440',
        ]
    }
    
    return features_by_horizon


def copy_alpha_files_for_sim(
    source_dir: str,
    dest_dir: str,
    dates: List[str]
) -> int:
    """Copy alpha files to sim fixture directory using proper directory structure."""
    print("\nCopying alpha files...")
    
    # Create tardis_prod_alpha directory (sim looks in tardis_prod_alpha)
    tardis_prod_alpha_dir = os.path.join(dest_dir, "tardis_prod_alpha")
    os.makedirs(tardis_prod_alpha_dir, exist_ok=True)
    
    total_copied = 0
    
    for date in dates:
        for model in ALPHA_MODELS:
            # The alpha files in production use a flat structure with model_horizon_date.parquet naming
            # e.g., hl_15_20240706.parquet, c2vwap_720_20240706.parquet
            source_file = os.path.join(source_dir, f"{model}_{date}.parquet")
            
            if os.path.exists(source_file):
                # Copy to tardis_prod_alpha directory with same name
                dest_file = os.path.join(tardis_prod_alpha_dir, f"{model}_{date}.parquet")
                
                shutil.copy2(source_file, dest_file)
                total_copied += 1
                print(f"  ✓ {model}_{date}.parquet")
            else:
                print(f"  ⚠ Warning: {source_file} not found")
    
    print(f"  Total alpha files copied: {total_copied}")
    return total_copied


# Removed create_test_universe_files function - should copy from production data
# The simulation uses bars from the 'bars' directory, not tardis_bars or live_bars


def generate_master_fixtures(
    config: dict,
    start_date: str,
    end_date: str
) -> None:
    """Generate master simulation fixtures."""
    print("\n" + "=" * 60)
    print("GENERATING MASTER FIXTURES")
    print("=" * 60)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_input_dir = os.path.join(script_dir, "data")
    fixture_output_dir = os.path.join(script_dir, "master")
    
    # Clean and create output directory
    if os.path.exists(fixture_output_dir):
        shutil.rmtree(fixture_output_dir)
    os.makedirs(fixture_output_dir, exist_ok=True)
    
    # Set up directory manager for fixtures
    sim_dir_manager = DirectoryManager(
        data_dir=fixture_input_dir,
        trading_dir=fixture_input_dir
    )
    
    # Convert date strings to date
    start_dt = date_str_to_date(start_date)
    end_dt = date_str_to_date(end_date)
    
    print(f"\nRunning simulation from {start_date} to {end_date}")
    print(f"Output directory: {fixture_output_dir}")
    
    try:
        # Create simulator
        sim = Simulate(
            config=config,
            sim_dir=fixture_output_dir,
            sim_comp=True,
            sim_dir_manager=sim_dir_manager,
        )
        
        # Run simulation
        sim.simulate(start_date=start_dt, end_date=end_dt)
        
        print("  ✓ Generated simulation results")
        
        # List generated files
        print("\nGenerated files:")
        for file in sorted(os.listdir(fixture_output_dir)):
            print(f"  - {file}")
        
    except Exception as e:
        print(f"  ✗ Failed to generate master fixtures: {e}")
        raise


def main():
    """Main function to generate sim test fixtures."""
    parser = argparse.ArgumentParser(description='Generate sim test fixtures')
    parser.add_argument('--master-only', action='store_true', 
                        help='Only regenerate master fixtures (skip copying data)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SIMULATION TEST FIXTURE GENERATOR")
    print("=" * 60)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config_sim_test.json")
    data_dir = os.path.join(script_dir, "data")
    
    # Load configuration
    config = load_test_config(config_path)
    
    # Update config with test parameters
    config = update_config_for_test(
        config,
        symbols=TEST_SYMBOLS,
        dynamic_universe=False,  # Use fixed universe for testing
        lookback_days=10,
        classification_history_days=2
    )
    
    # Get all dates needed
    start_dt = date_str_to_date(TEST_START_DATE)
    end_dt = date_str_to_date(TEST_END_DATE)
    
    # Need extra historical data for features/risk calculations
    all_dates = []
    current_date = start_dt - timedelta(days=HISTORICAL_DAYS)
    while current_date <= end_dt:
        all_dates.append(date_to_str(current_date))
        current_date += timedelta(days=1)
    
    # Simulation dates
    sim_dates = []
    current_date = start_dt
    while current_date <= end_dt:
        sim_dates.append(date_to_str(current_date))
        current_date += timedelta(days=1)
    
    print("\nTest configuration:")
    print(f"  Symbols: {TEST_SYMBOLS}")
    print(f"  Simulation dates: {TEST_START_DATE} to {TEST_END_DATE}")
    print(f"  Data dates: {all_dates[0]} to {all_dates[-1]} (includes history)")
    print(f"  Bar frequencies: {BAR_FREQUENCIES}")
    print(f"  Feature horizons: {FEATURE_HORIZONS}")
    print(f"  Alpha models: {ALPHA_MODELS}")
    
    if not args.master_only:
        # IMPORTANT: Delete ALL data in fixtures/sim/data/ for a clean slate
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
        
        # Create empty delisting file
        create_delisting_file(data_dir)
        
        # Copy bar files
        bars_dest_dir = os.path.join(data_dir, "bars")
        copy_bar_files(
            dir_manager.BAR_DIR,
            bars_dest_dir,
            TEST_SYMBOLS,
            all_dates,
            BAR_FREQUENCIES
        )
        
        # Copy feature files - get all required features for simulation
        features_dest_dir = os.path.join(data_dir, "features")
        
        # Get required features from function
        features_by_horizon = get_required_features_for_sim()
        
        # Copy features for each horizon
        for horizon, feature_list in features_by_horizon.items():
            print(f"\nCopying {horizon}-minute features...")
            copy_feature_files(
                dir_manager.FEATURES_DIR,
                features_dest_dir,
                TEST_SYMBOLS,
                all_dates,
                [horizon],
                feature_list
            )
        
        # Note: tardis_bars and live_bars are not needed for simulation
        # The simulation uses bars from the 'bars' directory
        
        # Copy alpha files
        copy_alpha_files_for_sim(
            dir_manager.ALPHA_DIR_DEV,
            data_dir,
            sim_dates
        )
        
        # Copy universe files from production data
        copy_universe_files(
            dir_manager.UNIVERSE_DIR,
            os.path.join(data_dir, "universe"),
            all_dates
        )
        
        print("\nStep 3: Verifying generated files...")
        print("=" * 60)
        
        # Count files by type based on docstring
        expected_counts = {
            'bars': 30,          # 2 frequencies * 3 dates * 5 symbols
            'features': 240,     # (6 features for 360 + 10 features for 1440) * 3 dates * 5 symbols = 240
            'tardis_prod_alpha': 4,  # 2 models * 2 sim dates
            'universe': 3,       # 3 dates
            'delisting.txt': 1
        }
        expected_total = 278  # Total files
        
        # Count actual files
        actual_counts = {}
        total_files = 0
        
        # Count bar files
        bars_dir = os.path.join(data_dir, "bars")
        if os.path.exists(bars_dir):
            count = sum(1 for _, _, files in os.walk(bars_dir) for f in files if f.endswith('.parquet'))
            actual_counts['bars'] = count
            total_files += count
        else:
            actual_counts['bars'] = 0
            
        # Count feature files
        features_dir = os.path.join(data_dir, "features")
        if os.path.exists(features_dir):
            count = sum(1 for _, _, files in os.walk(features_dir) for f in files if f.endswith('.parquet'))
            actual_counts['features'] = count
            total_files += count
        else:
            actual_counts['features'] = 0
            
        # Count alpha files
        alpha_dir = os.path.join(data_dir, "tardis_prod_alpha")
        if os.path.exists(alpha_dir):
            count = sum(1 for f in os.listdir(alpha_dir) if f.endswith('.parquet'))
            actual_counts['tardis_prod_alpha'] = count
            total_files += count
        else:
            actual_counts['tardis_prod_alpha'] = 0
            
        # Count universe files
        universe_dir = os.path.join(data_dir, "universe")
        if os.path.exists(universe_dir):
            count = sum(1 for f in os.listdir(universe_dir) if f.endswith('.parquet'))
            actual_counts['universe'] = count
            total_files += count
        else:
            actual_counts['universe'] = 0
            
        # Check delisting file
        if os.path.exists(os.path.join(data_dir, "delisting.txt")):
            actual_counts['delisting.txt'] = 1
            total_files += 1
        else:
            actual_counts['delisting.txt'] = 0
        
        print("Expected file counts:")
        for key in ['bars', 'features', 'tardis_prod_alpha', 'universe', 'delisting.txt']:
            expected = expected_counts.get(key, 0)
            print(f"  - {key}: {expected}")
        print(f"Total expected: {expected_total}")
        
        print("\nActual file counts:")
        for key in ['bars', 'features', 'tardis_prod_alpha', 'universe', 'delisting.txt']:
            actual = actual_counts.get(key, 0)
            expected = expected_counts.get(key, 0)
            status = "✓" if key != 'features' and actual == expected else "✓" if key == 'features' and actual > 0 else "✗"
            print(f"  {status} {key}: {actual}")
        print(f"Total actual: {total_files}")
        
        if total_files != expected_total:
            print(f"\n⚠️  WARNING: File count mismatch! Expected {expected_total} but found {total_files}")
            print("Note: Feature count may vary due to shared st_risk files across horizons")
    
    # Generate master fixtures
    generate_master_fixtures(config, TEST_START_DATE, TEST_END_DATE)
    
    # Save updated config
    save_test_config(config_path, config)
    print("\n✓ Updated config with test parameters")
    
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
