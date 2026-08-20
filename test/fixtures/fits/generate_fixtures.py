#!/usr/bin/env python3
"""Generate fits test fixtures from production data.

IMPORTANT: All data must be copied from ~/stat_arb/data/ (production data directories).
           Do not create synthetic or dummy data files.
           Do NOT create symlinks - copy actual files to maintain test isolation.

This script:
1. Copies bar, feature, forward, and model files from production
2. Generates fits using the Fits class
3. Creates master fixtures for test comparison

Usage:
    python generate_fixtures.py                    # Generate all fixtures
    python generate_fixtures.py --master-only      # Only regenerate master fixtures

Complete list of files that should be generated in test/fixtures/fits/data/:
    bars/15/binance-futures/20250101/bars.15.binance-futures.20250101.BNBUSDT.parquet
    bars/15/binance-futures/20250101/bars.15.binance-futures.20250101.BTCUSDT.parquet
    bars/15/binance-futures/20250101/bars.15.binance-futures.20250101.ETHUSDT.parquet
    bars/15/binance-futures/20250102/bars.15.binance-futures.20250102.BNBUSDT.parquet
    bars/15/binance-futures/20250102/bars.15.binance-futures.20250102.BTCUSDT.parquet
    bars/15/binance-futures/20250102/bars.15.binance-futures.20250102.ETHUSDT.parquet
    bars/15/binance-futures/20250103/bars.15.binance-futures.20250103.BNBUSDT.parquet
    bars/15/binance-futures/20250103/bars.15.binance-futures.20250103.BTCUSDT.parquet
    bars/15/binance-futures/20250103/bars.15.binance-futures.20250103.ETHUSDT.parquet
    bars/15/binance-futures/20250104/bars.15.binance-futures.20250104.BNBUSDT.parquet
    bars/15/binance-futures/20250104/bars.15.binance-futures.20250104.BTCUSDT.parquet
    bars/15/binance-futures/20250104/bars.15.binance-futures.20250104.ETHUSDT.parquet
    bars/15/binance-futures/20250105/bars.15.binance-futures.20250105.BNBUSDT.parquet
    bars/15/binance-futures/20250105/bars.15.binance-futures.20250105.BTCUSDT.parquet
    bars/15/binance-futures/20250105/bars.15.binance-futures.20250105.ETHUSDT.parquet
    bars/15/binance-futures/20250106/bars.15.binance-futures.20250106.BNBUSDT.parquet
    bars/15/binance-futures/20250106/bars.15.binance-futures.20250106.BTCUSDT.parquet
    bars/15/binance-futures/20250106/bars.15.binance-futures.20250106.ETHUSDT.parquet
    bars/15/binance-futures/20250107/bars.15.binance-futures.20250107.BNBUSDT.parquet
    bars/15/binance-futures/20250107/bars.15.binance-futures.20250107.BTCUSDT.parquet
    bars/15/binance-futures/20250107/bars.15.binance-futures.20250107.ETHUSDT.parquet
    features/15/ba_imbal_15/20250101/features.15.ba_imbal_15.20250101.BNBUSDT.parquet
    features/15/ba_imbal_15/20250101/features.15.ba_imbal_15.20250101.BTCUSDT.parquet
    features/15/ba_imbal_15/20250101/features.15.ba_imbal_15.20250101.ETHUSDT.parquet
    features/15/ba_imbal_15/20250102/features.15.ba_imbal_15.20250102.BNBUSDT.parquet
    features/15/ba_imbal_15/20250102/features.15.ba_imbal_15.20250102.BTCUSDT.parquet
    features/15/ba_imbal_15/20250102/features.15.ba_imbal_15.20250102.ETHUSDT.parquet
    features/15/ba_imbal_15/20250103/features.15.ba_imbal_15.20250103.BNBUSDT.parquet
    features/15/ba_imbal_15/20250103/features.15.ba_imbal_15.20250103.BTCUSDT.parquet
    features/15/ba_imbal_15/20250103/features.15.ba_imbal_15.20250103.ETHUSDT.parquet
    features/15/ba_imbal_15/20250104/features.15.ba_imbal_15.20250104.BNBUSDT.parquet
    features/15/ba_imbal_15/20250104/features.15.ba_imbal_15.20250104.BTCUSDT.parquet
    features/15/ba_imbal_15/20250104/features.15.ba_imbal_15.20250104.ETHUSDT.parquet
    features/15/ba_imbal_15/20250105/features.15.ba_imbal_15.20250105.BNBUSDT.parquet
    features/15/ba_imbal_15/20250105/features.15.ba_imbal_15.20250105.BTCUSDT.parquet
    features/15/ba_imbal_15/20250105/features.15.ba_imbal_15.20250105.ETHUSDT.parquet
    features/15/ba_imbal_15/20250106/features.15.ba_imbal_15.20250106.BNBUSDT.parquet
    features/15/ba_imbal_15/20250106/features.15.ba_imbal_15.20250106.BTCUSDT.parquet
    features/15/ba_imbal_15/20250106/features.15.ba_imbal_15.20250106.ETHUSDT.parquet
    features/15/ba_imbal_15/20250107/features.15.ba_imbal_15.20250107.BNBUSDT.parquet
    features/15/ba_imbal_15/20250107/features.15.ba_imbal_15.20250107.BTCUSDT.parquet
    features/15/ba_imbal_15/20250107/features.15.ba_imbal_15.20250107.ETHUSDT.parquet
    features/15/dvolume_15_trmean/20250101/features.15.dvolume_15_trmean.20250101.BNBUSDT.parquet
    features/15/dvolume_15_trmean/20250101/features.15.dvolume_15_trmean.20250101.BTCUSDT.parquet
    features/15/dvolume_15_trmean/20250101/features.15.dvolume_15_trmean.20250101.ETHUSDT.parquet
    features/15/dvolume_15_trmean/20250102/features.15.dvolume_15_trmean.20250102.BNBUSDT.parquet
    features/15/dvolume_15_trmean/20250102/features.15.dvolume_15_trmean.20250102.BTCUSDT.parquet
    features/15/dvolume_15_trmean/20250102/features.15.dvolume_15_trmean.20250102.ETHUSDT.parquet
    features/15/dvolume_15_trmean/20250103/features.15.dvolume_15_trmean.20250103.BNBUSDT.parquet
    features/15/dvolume_15_trmean/20250103/features.15.dvolume_15_trmean.20250103.BTCUSDT.parquet
    features/15/dvolume_15_trmean/20250103/features.15.dvolume_15_trmean.20250103.ETHUSDT.parquet
    features/15/dvolume_15_trmean/20250104/features.15.dvolume_15_trmean.20250104.BNBUSDT.parquet
    features/15/dvolume_15_trmean/20250104/features.15.dvolume_15_trmean.20250104.BTCUSDT.parquet
    features/15/dvolume_15_trmean/20250104/features.15.dvolume_15_trmean.20250104.ETHUSDT.parquet
    features/15/dvolume_15_trmean/20250105/features.15.dvolume_15_trmean.20250105.BNBUSDT.parquet
    features/15/dvolume_15_trmean/20250105/features.15.dvolume_15_trmean.20250105.BTCUSDT.parquet
    features/15/dvolume_15_trmean/20250105/features.15.dvolume_15_trmean.20250105.ETHUSDT.parquet
    features/15/dvolume_15_trmean/20250106/features.15.dvolume_15_trmean.20250106.BNBUSDT.parquet
    features/15/dvolume_15_trmean/20250106/features.15.dvolume_15_trmean.20250106.BTCUSDT.parquet
    features/15/dvolume_15_trmean/20250106/features.15.dvolume_15_trmean.20250106.ETHUSDT.parquet
    features/15/dvolume_15_trmean/20250107/features.15.dvolume_15_trmean.20250107.BNBUSDT.parquet
    features/15/dvolume_15_trmean/20250107/features.15.dvolume_15_trmean.20250107.BTCUSDT.parquet
    features/15/dvolume_15_trmean/20250107/features.15.dvolume_15_trmean.20250107.ETHUSDT.parquet
    features/15/fittable/20250101/features.15.fittable.20250101.BNBUSDT.parquet
    features/15/fittable/20250101/features.15.fittable.20250101.BTCUSDT.parquet
    features/15/fittable/20250101/features.15.fittable.20250101.ETHUSDT.parquet
    features/15/fittable/20250102/features.15.fittable.20250102.BNBUSDT.parquet
    features/15/fittable/20250102/features.15.fittable.20250102.BTCUSDT.parquet
    features/15/fittable/20250102/features.15.fittable.20250102.ETHUSDT.parquet
    features/15/fittable/20250103/features.15.fittable.20250103.BNBUSDT.parquet
    features/15/fittable/20250103/features.15.fittable.20250103.BTCUSDT.parquet
    features/15/fittable/20250103/features.15.fittable.20250103.ETHUSDT.parquet
    features/15/fittable/20250104/features.15.fittable.20250104.BNBUSDT.parquet
    features/15/fittable/20250104/features.15.fittable.20250104.BTCUSDT.parquet
    features/15/fittable/20250104/features.15.fittable.20250104.ETHUSDT.parquet
    features/15/fittable/20250105/features.15.fittable.20250105.BNBUSDT.parquet
    features/15/fittable/20250105/features.15.fittable.20250105.BTCUSDT.parquet
    features/15/fittable/20250105/features.15.fittable.20250105.ETHUSDT.parquet
    features/15/fittable/20250106/features.15.fittable.20250106.BNBUSDT.parquet
    features/15/fittable/20250106/features.15.fittable.20250106.BTCUSDT.parquet
    features/15/fittable/20250106/features.15.fittable.20250106.ETHUSDT.parquet
    features/15/fittable/20250107/features.15.fittable.20250107.BNBUSDT.parquet
    features/15/fittable/20250107/features.15.fittable.20250107.BTCUSDT.parquet
    features/15/fittable/20250107/features.15.fittable.20250107.ETHUSDT.parquet
    features/15/logret_15_trstd/20250101/features.15.logret_15_trstd.20250101.BNBUSDT.parquet
    features/15/logret_15_trstd/20250101/features.15.logret_15_trstd.20250101.BTCUSDT.parquet
    features/15/logret_15_trstd/20250101/features.15.logret_15_trstd.20250101.ETHUSDT.parquet
    features/15/logret_15_trstd/20250102/features.15.logret_15_trstd.20250102.BNBUSDT.parquet
    features/15/logret_15_trstd/20250102/features.15.logret_15_trstd.20250102.BTCUSDT.parquet
    features/15/logret_15_trstd/20250102/features.15.logret_15_trstd.20250102.ETHUSDT.parquet
    features/15/logret_15_trstd/20250103/features.15.logret_15_trstd.20250103.BNBUSDT.parquet
    features/15/logret_15_trstd/20250103/features.15.logret_15_trstd.20250103.BTCUSDT.parquet
    features/15/logret_15_trstd/20250103/features.15.logret_15_trstd.20250103.ETHUSDT.parquet
    features/15/logret_15_trstd/20250104/features.15.logret_15_trstd.20250104.BNBUSDT.parquet
    features/15/logret_15_trstd/20250104/features.15.logret_15_trstd.20250104.BTCUSDT.parquet
    features/15/logret_15_trstd/20250104/features.15.logret_15_trstd.20250104.ETHUSDT.parquet
    features/15/logret_15_trstd/20250105/features.15.logret_15_trstd.20250105.BNBUSDT.parquet
    features/15/logret_15_trstd/20250105/features.15.logret_15_trstd.20250105.BTCUSDT.parquet
    features/15/logret_15_trstd/20250105/features.15.logret_15_trstd.20250105.ETHUSDT.parquet
    features/15/logret_15_trstd/20250106/features.15.logret_15_trstd.20250106.BNBUSDT.parquet
    features/15/logret_15_trstd/20250106/features.15.logret_15_trstd.20250106.BTCUSDT.parquet
    features/15/logret_15_trstd/20250106/features.15.logret_15_trstd.20250106.ETHUSDT.parquet
    features/15/logret_15_trstd/20250107/features.15.logret_15_trstd.20250107.BNBUSDT.parquet
    features/15/logret_15_trstd/20250107/features.15.logret_15_trstd.20250107.BTCUSDT.parquet
    features/15/logret_15_trstd/20250107/features.15.logret_15_trstd.20250107.ETHUSDT.parquet
    features/15/trade_sz_15/20250101/features.15.trade_sz_15.20250101.BNBUSDT.parquet
    features/15/trade_sz_15/20250101/features.15.trade_sz_15.20250101.BTCUSDT.parquet
    features/15/trade_sz_15/20250101/features.15.trade_sz_15.20250101.ETHUSDT.parquet
    features/15/trade_sz_15/20250102/features.15.trade_sz_15.20250102.BNBUSDT.parquet
    features/15/trade_sz_15/20250102/features.15.trade_sz_15.20250102.BTCUSDT.parquet
    features/15/trade_sz_15/20250102/features.15.trade_sz_15.20250102.ETHUSDT.parquet
    features/15/trade_sz_15/20250103/features.15.trade_sz_15.20250103.BNBUSDT.parquet
    features/15/trade_sz_15/20250103/features.15.trade_sz_15.20250103.BTCUSDT.parquet
    features/15/trade_sz_15/20250103/features.15.trade_sz_15.20250103.ETHUSDT.parquet
    features/15/trade_sz_15/20250104/features.15.trade_sz_15.20250104.BNBUSDT.parquet
    features/15/trade_sz_15/20250104/features.15.trade_sz_15.20250104.BTCUSDT.parquet
    features/15/trade_sz_15/20250104/features.15.trade_sz_15.20250104.ETHUSDT.parquet
    features/15/trade_sz_15/20250105/features.15.trade_sz_15.20250105.BNBUSDT.parquet
    features/15/trade_sz_15/20250105/features.15.trade_sz_15.20250105.BTCUSDT.parquet
    features/15/trade_sz_15/20250105/features.15.trade_sz_15.20250105.ETHUSDT.parquet
    features/15/trade_sz_15/20250106/features.15.trade_sz_15.20250106.BNBUSDT.parquet
    features/15/trade_sz_15/20250106/features.15.trade_sz_15.20250106.BTCUSDT.parquet
    features/15/trade_sz_15/20250106/features.15.trade_sz_15.20250106.ETHUSDT.parquet
    features/15/trade_sz_15/20250107/features.15.trade_sz_15.20250107.BNBUSDT.parquet
    features/15/trade_sz_15/20250107/features.15.trade_sz_15.20250107.BTCUSDT.parquet
    features/15/trade_sz_15/20250107/features.15.trade_sz_15.20250107.ETHUSDT.parquet
    forwards/15/20250101/forwards.15.20250101.BNBUSDT.parquet
    forwards/15/20250101/forwards.15.20250101.BTCUSDT.parquet
    forwards/15/20250101/forwards.15.20250101.ETHUSDT.parquet
    forwards/15/20250102/forwards.15.20250102.BNBUSDT.parquet
    forwards/15/20250102/forwards.15.20250102.BTCUSDT.parquet
    forwards/15/20250102/forwards.15.20250102.ETHUSDT.parquet
    forwards/15/20250103/forwards.15.20250103.BNBUSDT.parquet
    forwards/15/20250103/forwards.15.20250103.BTCUSDT.parquet
    forwards/15/20250103/forwards.15.20250103.ETHUSDT.parquet
    forwards/15/20250104/forwards.15.20250104.BNBUSDT.parquet
    forwards/15/20250104/forwards.15.20250104.BTCUSDT.parquet
    forwards/15/20250104/forwards.15.20250104.ETHUSDT.parquet
    forwards/15/20250105/forwards.15.20250105.BNBUSDT.parquet
    forwards/15/20250105/forwards.15.20250105.BTCUSDT.parquet
    forwards/15/20250105/forwards.15.20250105.ETHUSDT.parquet
    forwards/15/20250106/forwards.15.20250106.BNBUSDT.parquet
    forwards/15/20250106/forwards.15.20250106.BTCUSDT.parquet
    forwards/15/20250106/forwards.15.20250106.ETHUSDT.parquet
    forwards/15/20250107/forwards.15.20250107.BNBUSDT.parquet
    forwards/15/20250107/forwards.15.20250107.BTCUSDT.parquet
    forwards/15/20250107/forwards.15.20250107.ETHUSDT.parquet
    models/15/hl/models.15.hl.20250101.parquet
    models/15/hl/models.15.hl.20250102.parquet
    models/15/hl/models.15.hl.20250103.parquet
    models/15/hl/models.15.hl.20250104.parquet
    models/15/hl/models.15.hl.20250105.parquet
    models/15/hl/models.15.hl.20250106.parquet
    models/15/hl/models.15.hl.20250107.parquet
    universe/universe.20250101.parquet
    universe/universe.20250102.parquet
    universe/universe.20250103.parquet
    universe/universe.20250104.parquet
    universe/universe.20250105.parquet
    universe/universe.20250106.parquet
    universe/universe.20250107.parquet
"""

import os
import sys
import argparse
import shutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from lib.util.directory import dir_manager
from lib.util.time_util import date_str_to_date
from lib.fits.fits import Fits

# Import common fixture utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fixture_utils import (
    load_test_config, save_test_config,
    get_date_range, update_config_for_test, print_test_summary,
    copy_bar_files, copy_feature_files, copy_model_files,
    copy_forward_files, copy_universe_files,
    DEFAULT_TEST_SYMBOLS
)


# Test configuration
TEST_DATE = "20250107"  # Match the last date in the docstring
HISTORICAL_DAYS = 11  # To get dates from 20241227 to 20250107 (need extra days for classification window)
TEST_HORIZONS = [15]  # Test with 15-minute fits only
TEST_MODELS = ['hl']  # Test with single model type
BAR_FREQUENCIES = [15]  # Only 15-minute bars as shown in docstring


def generate_master_fixtures(config: dict, test_date: str, horizons: list, models: list) -> None:
    """Generate master fit fixtures."""
    print("\n" + "=" * 60)
    print("GENERATING MASTER FIXTURES")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_output_dir = os.path.join(script_dir, "master")
    
    # Clean master directory completely before generating new fixtures
    print("Cleaning master directory...")
    if os.path.exists(fixture_output_dir):
        shutil.rmtree(fixture_output_dir)
        print("  ✓ Removed entire master directory")
    os.makedirs(fixture_output_dir, exist_ok=True)
    
    
    # Convert date string to datetime
    fit_date = date_str_to_date(test_date)
    
    print(f"\nGenerating fits for single day: {test_date}")
    print(f"Horizons: {horizons}")
    print(f"Models: {models}")
    
    try:
        # Create fits calculator
        fits_calc = Fits(
            config=config,
            horizons=horizons,
            models=models,
            pool_size=1,
            debug=False,
            prod=True,  # Use prod=True to match production behavior
            base_fits_dir=fixture_output_dir,
        )
        
        # Run fits generation
        fits_calc.generate_rolling_fits(start_date=fit_date, end_date=fit_date)
        
        print("  ✓ Generated master fixtures for fits")
        
    except Exception as e:
        print(f"  ✗ Failed to generate master fixtures: {e}")
        raise
    
    # The fits are already saved to the master directory
    print(f"  → Master fixtures saved to {fixture_output_dir}")


def get_required_features(config: dict, horizons: list) -> list:
    """Extract required features for fits calculation."""
    all_features = set()
    
    # Get features needed for fits
    for horizon in horizons:
        horizon_str = str(horizon)
        if horizon_str in config.get('FCASTS', {}):
            fits_config = config['FCASTS'][horizon_str]
            # Get features used for classification
            features = fits_config.get('features', [])
            all_features.update(features)
            
        # Also get model features
        if horizon_str in config.get('FEATURES', {}):
            features_config = config['FEATURES'][horizon_str]
            prod_features = features_config.get('prod', [])
            for feature in prod_features:
                feature_name = feature.replace('HORIZON', horizon_str)
                all_features.add(feature_name)
    
    # Always include core features
    all_features.update(['advp', 'fittable', 'tradeable', 'dvolume_1440_trmean'])
    
    return list(all_features)


def main():
    """Main function to generate fits test fixtures."""
    parser = argparse.ArgumentParser(description='Generate fits test fixtures')
    parser.add_argument('--master-only', action='store_true', 
                        help='Only regenerate master fixtures (skip copying data)')
    args = parser.parse_args()
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config_fits_test.json")
    
    # Load configuration
    config = load_test_config(config_path)
    
    # Use minimal symbols for test
    symbols = set(DEFAULT_TEST_SYMBOLS)
    config = update_config_for_test(
        config, 
        symbols=list(symbols),
        lookback_days=10,
        classification_history_days=2
    )
    
    # Set the fit test date in config
    config['FIT_TEST_DATE'] = TEST_DATE
    
    # Calculate date range
    dates = get_date_range(TEST_DATE, HISTORICAL_DAYS)
    
    # Print test summary
    print_test_summary("fits", list(symbols), TEST_DATE, dates, TEST_HORIZONS, TEST_MODELS)
    
    if not args.master_only:
        # IMPORTANT: Delete ALL data in fixtures/fits/data/ for a clean slate
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
        
        # Copy bar files
        copy_bar_files(
            dir_manager.BAR_DIR,
            os.path.join(script_dir, "data", "bars"),
            symbols,
            dates,
            BAR_FREQUENCIES
        )
        
        # Copy specific feature files as listed in docstring
        # For horizon 15: ba_imbal_15, dvolume_15_trmean, fittable, logret_15_trstd, trade_sz_15
        feature_types_15 = ['ba_imbal_15', 'dvolume_15_trmean', 'fittable', 'logret_15_trstd', 'trade_sz_15']
        
        # Copy feature files
        copy_feature_files(
            dir_manager.FEATURES_DIR,
            os.path.join(script_dir, "data", "features"),
            symbols,
            dates,
            TEST_HORIZONS,
            feature_types_15
        )
        
        # Copy model files
        copy_model_files(
            dir_manager.MODELS_DIR,
            os.path.join(script_dir, "data", "models"),
            dates,
            TEST_HORIZONS,
            TEST_MODELS
        )
        
        # Copy forward files
        copy_forward_files(
            dir_manager.FORWARDS_DIR,
            os.path.join(script_dir, "data", "forwards"),
            symbols,
            dates,
            TEST_HORIZONS
        )
        
        # Copy universe files
        copy_universe_files(
            dir_manager.UNIVERSE_DIR,
            os.path.join(script_dir, "data", "universe"),
            dates
        )
        
        print("\nStep 3: Verifying generated files...")
        print("=" * 60)
        
        # Count files by type based on docstring
        # With HISTORICAL_DAYS = 11, we get 12 dates (from Dec 27 to Jan 7)
        expected_counts = {
            'bars': 36,          # 12 dates * 3 symbols * 1 frequency (15min)
            'features': 180,     # 5 feature types * 12 dates * 3 symbols
            'forwards': 36,      # 12 dates * 3 symbols
            'models': 12,        # 12 dates * 1 model
            'universe': 12       # 12 dates
        }
        expected_total = sum(expected_counts.values())  # 276 total
        
        # Count actual files
        actual_counts = {}
        total_files = 0
        for subdir in expected_counts:
            subdir_path = os.path.join(data_dir, subdir)
            if not os.path.exists(subdir_path):
                actual_counts[subdir] = 0
                continue
            count = 0
            for _, _, files in os.walk(subdir_path):
                parquet_files = [f for f in files if f.endswith('.parquet')]
                count += len(parquet_files)
                total_files += len(parquet_files)
            actual_counts[subdir] = count
        
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
