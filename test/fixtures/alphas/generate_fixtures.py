#!/usr/bin/env python3
"""Generate alphas test fixtures from production data.

IMPORTANT: All data must be copied from ~/stat_arb/data/ (production data directories).
           Do not create synthetic or dummy data files.
           Do NOT create symlinks - copy actual files to maintain test isolation.

This script:
1. Copies bar, feature, model, and fit files from production
2. Generates alphas using the Forecasts class
3. Creates master fixtures for test comparison

Usage:
    python generate_fixtures.py                    # Generate all fixtures
    python generate_fixtures.py --master-only      # Only regenerate master fixtures

Complete list of files that should be generated in test/fixtures/alphas/data/:
    bars/15/binance-futures/20250606/bars.15.binance-futures.20250606.BNBUSDT.parquet
    bars/15/binance-futures/20250606/bars.15.binance-futures.20250606.BTCUSDT.parquet
    bars/15/binance-futures/20250606/bars.15.binance-futures.20250606.ETHUSDT.parquet
    features/1440/fittable/20250606/features.1440.fittable.20250606.BNBUSDT.parquet
    features/1440/fittable/20250606/features.1440.fittable.20250606.BTCUSDT.parquet
    features/1440/fittable/20250606/features.1440.fittable.20250606.ETHUSDT.parquet
    features/1440/tradeable/20250606/features.1440.tradeable.20250606.BNBUSDT.parquet
    features/1440/tradeable/20250606/features.1440.tradeable.20250606.BTCUSDT.parquet
    features/1440/tradeable/20250606/features.1440.tradeable.20250606.ETHUSDT.parquet
    features/15/dvolume_15_lz/20250606/features.15.dvolume_15_lz.20250606.BNBUSDT.parquet
    features/15/dvolume_15_lz/20250606/features.15.dvolume_15_lz.20250606.BTCUSDT.parquet
    features/15/dvolume_15_lz/20250606/features.15.dvolume_15_lz.20250606.ETHUSDT.parquet
    features/15/dvolume_15_trmean_cz/20250606/features.15.dvolume_15_trmean_cz.20250606.BNBUSDT.parquet
    features/15/dvolume_15_trmean_cz/20250606/features.15.dvolume_15_trmean_cz.20250606.BTCUSDT.parquet
    features/15/dvolume_15_trmean_cz/20250606/features.15.dvolume_15_trmean_cz.20250606.ETHUSDT.parquet
    features/15/fittable/20250606/features.15.fittable.20250606.BNBUSDT.parquet
    features/15/fittable/20250606/features.15.fittable.20250606.BTCUSDT.parquet
    features/15/fittable/20250606/features.15.fittable.20250606.ETHUSDT.parquet
    features/15/logret_15_lz_cz/20250606/features.15.logret_15_lz_cz.20250606.BNBUSDT.parquet
    features/15/logret_15_lz_cz/20250606/features.15.logret_15_lz_cz.20250606.BTCUSDT.parquet
    features/15/logret_15_lz_cz/20250606/features.15.logret_15_lz_cz.20250606.ETHUSDT.parquet
    features/15/logret_15_trstd/20250606/features.15.logret_15_trstd.20250606.BNBUSDT.parquet
    features/15/logret_15_trstd/20250606/features.15.logret_15_trstd.20250606.BTCUSDT.parquet
    features/15/logret_15_trstd/20250606/features.15.logret_15_trstd.20250606.ETHUSDT.parquet
    features/15/relative_spread_15/20250606/features.15.relative_spread_15.20250606.BNBUSDT.parquet
    features/15/relative_spread_15/20250606/features.15.relative_spread_15.20250606.BTCUSDT.parquet
    features/15/relative_spread_15/20250606/features.15.relative_spread_15.20250606.ETHUSDT.parquet
    features/15/trade_sz_15/20250606/features.15.trade_sz_15.20250606.BNBUSDT.parquet
    features/15/trade_sz_15/20250606/features.15.trade_sz_15.20250606.BTCUSDT.parquet
    features/15/trade_sz_15/20250606/features.15.trade_sz_15.20250606.ETHUSDT.parquet
    fits/dev/15/hl/fits.dev.15.hl.20250531.csv
    fits/dev/15/hl/fits.dev.15.hl.20250601.csv
    fits/dev/15/hl/fits.dev.15.hl.20250602.csv
    fits/dev/15/hl/fits.dev.15.hl.20250603.csv
    fits/dev/15/hl/fits.dev.15.hl.20250604.csv
    fits/dev/15/hl/fits.dev.15.hl.20250605.csv
    fits/dev/15/hl/fits.dev.15.hl.20250606.csv
    fits/dev/15/hl/fits.dev.15.hl.20250607.csv
    models/15/hl/models.15.hl.20250530.parquet
    models/15/hl/models.15.hl.20250531.parquet
    models/15/hl/models.15.hl.20250601.parquet
    models/15/hl/models.15.hl.20250602.parquet
    models/15/hl/models.15.hl.20250603.parquet
    models/15/hl/models.15.hl.20250604.parquet
    models/15/hl/models.15.hl.20250605.parquet
    models/15/hl/models.15.hl.20250606.parquet
    prod_fits/15/hl/fits.15.hl.20250531.csv
    prod_fits/15/hl/fits.15.hl.20250601.csv
    prod_fits/15/hl/fits.15.hl.20250602.csv
    prod_fits/15/hl/fits.15.hl.20250603.csv
    prod_fits/15/hl/fits.15.hl.20250604.csv
    prod_fits/15/hl/fits.15.hl.20250605.csv
    prod_fits/15/hl/fits.15.hl.20250606.csv
    prod_fits/15/hl/fits.15.hl.20250607.csv
    prod_fits/15/hl/fits.dev.15.hl.20250531.csv
    prod_fits/15/hl/fits.dev.15.hl.20250601.csv
    prod_fits/15/hl/fits.dev.15.hl.20250602.csv
    prod_fits/15/hl/fits.dev.15.hl.20250603.csv
    prod_fits/15/hl/fits.dev.15.hl.20250604.csv
    prod_fits/15/hl/fits.dev.15.hl.20250605.csv
    prod_fits/15/hl/fits.dev.15.hl.20250606.csv
    prod_fits/15/hl/fits.dev.15.hl.20250607.csv
    prod_fits/svm/hl_15/svm.hl_15.20250531.features
    prod_fits/svm/hl_15/svm.hl_15.20250531.joblib
    prod_fits/svm/hl_15/svm.hl_15.20250601.features
    prod_fits/svm/hl_15/svm.hl_15.20250601.joblib
    prod_fits/svm/hl_15/svm.hl_15.20250602.features
    prod_fits/svm/hl_15/svm.hl_15.20250602.joblib
    prod_fits/svm/hl_15/svm.hl_15.20250603.features
    prod_fits/svm/hl_15/svm.hl_15.20250603.joblib
    prod_fits/svm/hl_15/svm.hl_15.20250604.features
    prod_fits/svm/hl_15/svm.hl_15.20250604.joblib
    prod_fits/svm/hl_15/svm.hl_15.20250605.features
    prod_fits/svm/hl_15/svm.hl_15.20250605.joblib
    prod_fits/svm/hl_15/svm.hl_15.20250606.features
    prod_fits/svm/hl_15/svm.hl_15.20250606.joblib
    prod_fits/svm/hl_15/svm.hl_15.20250607.features
    prod_fits/svm/hl_15/svm.hl_15.20250607.joblib
    universe/universe.20250530.parquet
    universe/universe.20250531.parquet
    universe/universe.20250601.parquet
    universe/universe.20250602.parquet
    universe/universe.20250603.parquet
    universe/universe.20250604.parquet
    universe/universe.20250605.parquet
    universe/universe.20250606.parquet
"""

import os
import sys
import argparse
import shutil
from datetime import timedelta
from typing import List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from lib.util.directory import dir_manager, DirectoryManager
from lib.util.time_util import date_str_to_date
from lib.alpha.forecasts import Forecasts

# Import common fixture utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fixture_utils import (
    load_test_config, save_test_config,
    copy_bar_files, copy_feature_files, copy_model_files,
    copy_fit_files, copy_universe_files
)


# Test configuration
TEST_DATE = "20250606"
HISTORICAL_DAYS = 7  # Need history for alpha calculations
TEST_HORIZONS = [15]  # Test with 15-minute alphas only
TEST_MODELS = ['hl']  # Test with single model type
BAR_FREQUENCIES = [1, 15]  # Need 1-minute bars and target frequency


def get_required_features(config: dict, horizons: list) -> list:
    """Extract required features for alpha calculation."""
    all_features = set()
    
    # Get features needed for fits and models
    for horizon in horizons:
        horizon_str = str(horizon)
        if horizon_str in config.get('FCASTS', {}):
            fits_config = config['FCASTS'][horizon_str]
            features = fits_config.get('features', [])
            all_features.update(features)
            
        if horizon_str in config.get('FEATURES', {}):
            features_config = config['FEATURES'][horizon_str]
            prod_features = features_config.get('prod', [])
            for feature in prod_features:
                feature_name = feature.replace('HORIZON', horizon_str)
                all_features.add(feature_name)
    
    # Always include core features
    all_features.update(['advp', 'fittable', 'tradeable', 'dvolume_1440_trmean'])
    
    return list(all_features)


def generate_master_fixtures(
    config: dict,
    alpha_date: str,
    horizons: List[int],
    models: List[str]
) -> None:
    """Generate master alphas fixtures."""
    print("\n" + "=" * 60)
    print("GENERATING MASTER FIXTURES")
    print("=" * 60)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_input_dir = os.path.join(script_dir, "data")
    fixture_output_dir = os.path.join(script_dir, "master")
    
    # Clean master directory completely before generating new fixtures
    print("Cleaning master directory...")
    if os.path.exists(fixture_output_dir):
        shutil.rmtree(fixture_output_dir)
        print("  ✓ Removed entire master directory")
    os.makedirs(fixture_output_dir, exist_ok=True)
    
    # Set up directory manager for fixtures
    test_dir_manager = DirectoryManager(
        data_dir=fixture_input_dir,
        trading_dir=fixture_input_dir
    )
    
    # Convert date string to datetime
    start_date = date_str_to_date(alpha_date)
    end_date = start_date
    
    print(f"\nGenerating alphas for single day: {alpha_date}")
    print(f"Horizons: {horizons}")
    print(f"Models: {models}")
    
    try:
        # Create forecasts calculator
        forecasts_calc = Forecasts(
            config=config,
            horizons=horizons,
            models=models,
            prod=False,  # Use prod=False since we're reading from prod_fits directory
            debug=False,
            forecast_dir_manager=test_dir_manager,
            output_dir=fixture_output_dir
        )
        
        # Run alphas generation
        forecasts_calc.generate_rolling_alphas(fit_file=None, start_date=start_date, end_date=end_date)
        
        # Copy generated files to master directory
        # Alphas are stored in the alpha directory
        alpha_dir = os.path.join(fixture_input_dir, "alpha")
        if os.path.exists(alpha_dir):
            for file in os.listdir(alpha_dir):
                if file.endswith('.parquet'):
                    shutil.copy2(
                        os.path.join(alpha_dir, file),
                        os.path.join(fixture_output_dir, file)
                    )
                    print(f"  ✓ Copied {file} to master")
        
        print(f"  ✓ Generated master fixtures for models {models} at horizons {horizons}")
        
    except Exception as e:
        print(f"  ✗ Failed to generate master fixtures: {e}")
        raise




def main():
    """Main function to generate alphas test fixtures."""
    parser = argparse.ArgumentParser(description='Generate alphas test fixtures')
    parser.add_argument('--master-only', action='store_true', 
                        help='Only regenerate master fixtures (skip copying data)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ALPHAS TEST FIXTURE GENERATOR")
    print("=" * 60)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config_alphas_generation_test.json")
    
    # Load configuration
    config = load_test_config(config_path)
    symbols = set(config.get('SYMBOL_UNIVERSE', []))
    
    # Reduce to 3 symbols for minimal test
    symbols = {'BTC', 'ETH', 'BNB'}
    
    # Update config with minimal symbols
    config['SYMBOL_UNIVERSE'] = list(symbols)
    config['DYNAMIC_UNIVERSE'] = True  # Use dynamic universe for tests
    
    # Calculate all dates needed - based on what's listed in docstring
    alpha_date = date_str_to_date(TEST_DATE)
    start_date = alpha_date - timedelta(days=HISTORICAL_DAYS)
    
    # Universe files go up to test date, but fits go one day beyond
    universe_dates = []
    fit_dates = []
    current_date = start_date
    while current_date <= alpha_date:
        date_str = current_date.strftime('%Y%m%d')
        universe_dates.append(date_str)
        fit_dates.append(date_str)
        current_date += timedelta(days=1)
    
    # Add one extra day for fits only (20250607)
    fit_dates.append((alpha_date + timedelta(days=1)).strftime('%Y%m%d'))
    
    print("\nTest configuration:")
    print(f"  Symbols: {sorted(symbols)}")
    print(f"  Alpha test date: {TEST_DATE}")
    print(f"  Data dates: {universe_dates[0]} to {universe_dates[-1]} ({len(universe_dates)} days)")
    print(f"  Horizons: {TEST_HORIZONS}")
    print(f"  Models: {TEST_MODELS}")
    print(f"  Bar frequencies: {BAR_FREQUENCIES}")
    
    if not args.master_only:
        # IMPORTANT: Delete ALL data in fixtures/alphas/data/ for a clean slate
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
        
        # Copy bar files - only 15-minute bars for test date as per docstring
        copy_bar_files(
            dir_manager.BAR_DIR,
            os.path.join(script_dir, "data", "bars"),
            symbols,
            [TEST_DATE],  # Only copy for the test date as per docstring
            [15]  # Only 15-minute bars as shown in docstring
        )
        
        # Copy specific feature files as listed in docstring
        # For horizon 15: dvolume_15_lz, dvolume_15_trmean_cz, fittable, logret_15_lz_cz, logret_15_trstd, relative_spread_15, trade_sz_15
        # For horizon 1440: fittable, tradeable
        feature_types_15 = ['dvolume_15_lz', 'dvolume_15_trmean_cz', 'fittable', 'logret_15_lz_cz', 
                            'logret_15_trstd', 'relative_spread_15', 'trade_sz_15']
        feature_types_1440 = ['fittable', 'tradeable']
        
        # Copy 15-minute features (only for test date 20250606)
        copy_feature_files(
            dir_manager.FEATURES_DIR,
            os.path.join(script_dir, "data", "features"),
            symbols,
            [TEST_DATE],  # Only copy for the test date as per docstring
            [15],
            feature_types_15
        )
        
        # Copy 1440-minute (daily) features (only for test date 20250606)
        copy_feature_files(
            dir_manager.FEATURES_DIR,
            os.path.join(script_dir, "data", "features"),
            symbols,
            [TEST_DATE],  # Only copy for the test date as per docstring
            [1440],
            feature_types_1440
        )
        
        # Copy model files (use universe dates)
        copy_model_files(
            dir_manager.MODELS_DIR,
            os.path.join(script_dir, "data", "models"),
            universe_dates,
            TEST_HORIZONS,
            TEST_MODELS
        )
        
        # Copy fit files (production fits) - use fit_dates which includes extra day
        copy_fit_files(
            dir_manager.FITS_DIR_DEV,
            os.path.join(script_dir, "data", "prod_fits"),
            fit_dates,
            TEST_HORIZONS,
            TEST_MODELS
        )
        
        # Also copy dev fits to the fits/dev directory for Forecasts class
        fits_dev_dir = os.path.join(script_dir, "data", "fits", "dev")
        os.makedirs(fits_dev_dir, exist_ok=True)
        
        # Copy dev fit files from prod_fits to fits/dev
        print("\nCopying dev fit files to fits/dev directory...")
        for horizon in TEST_HORIZONS:
            for model in TEST_MODELS:
                horizon_model_dir = os.path.join(fits_dev_dir, str(horizon), model)
                os.makedirs(horizon_model_dir, exist_ok=True)
                
                # Copy dev CSV files
                for date in fit_dates:
                    dev_filename = f"fits.dev.{horizon}.{model}.{date}.csv"
                    src_file = os.path.join(script_dir, "data", "prod_fits", str(horizon), model, dev_filename)
                    if os.path.exists(src_file):
                        dest_file = os.path.join(horizon_model_dir, dev_filename)
                        shutil.copy2(src_file, dest_file)
                        print(f"  ✓ Copied {dev_filename} to fits/dev")
        
        # Copy universe files (use universe_dates)
        copy_universe_files(
            dir_manager.UNIVERSE_DIR,
            os.path.join(script_dir, "data", "universe"),
            universe_dates
        )
        
        print("\nStep 3: Verifying generated files...")
        print("=" * 60)
        
        # Count files by type based on docstring
        expected_counts = {
            'bars': 3,          # 3 symbols x 1 date x 1 frequency (15min)
            'features': 27,     # 3 symbols x 9 feature types
            'models': 8,        # 8 dates x 1 model
            'fits': 8,          # dev fits: 8 dates
            'prod_fits': 32,    # 16 CSV files + 16 SVM files
            'universe': 8       # 8 dates
        }
        expected_total = sum(expected_counts.values())  # 86 total
        
        # Count actual files
        actual_counts = {}
        total_files = 0
        for subdir in expected_counts:
            subdir_path = os.path.join(data_dir, subdir)
            if os.path.exists(subdir_path):
                count = 0
                for _, _, files in os.walk(subdir_path):
                    for f in files:
                        if f.endswith(('.parquet', '.csv', '.joblib', '.features')):
                            count += 1
                            total_files += 1
                actual_counts[subdir] = count
            else:
                actual_counts[subdir] = 0
        
        print("Expected file counts:")
        for subdir, count in expected_counts.items():
            print(f"  - {subdir}: {count}")
        print(f"Total expected: {expected_total}")
        
        print("\nActual file counts:")
        for subdir, count in actual_counts.items():
            status = "✓" if count == expected_counts.get(subdir, 0) else "✗"
            print(f"  {status} {subdir}: {count}")
        print(f"Total actual: {total_files}")
        
        if total_files != expected_total:
            print(f"\n⚠️  WARNING: File count mismatch! Expected {expected_total} but found {total_files}")
    
    # Generate master fixtures for single day
    generate_master_fixtures(config, TEST_DATE, TEST_HORIZONS, TEST_MODELS)
    
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
