#!/usr/bin/env python3
"""Generate server test fixtures from production data.

IMPORTANT: Do NOT create symlinks - copy actual files to maintain test isolation.

This script:
1. Copies required data files from production directories
2. Filters to only the test symbols (BTC, ETH, BNB)
3. Consolidates files to the format expected by the server test
4. Does NOT generate master fixtures (those are created by running the test)

Usage:
    python generate_fixtures.py                    # Copy all fixture data
    python generate_fixtures.py --master-only      # Only regenerate master fixtures (runs server)

Complete list of files that should be generated in test/fixtures/server/data/:
    TOTAL FILES: 69 (exactly what's in git)
    
    bars/ (30 files):
    - 1-minute bars: 15 files (5 dates × 3 symbols)
      Dates: 20250531, 20250601, 20250602, 20250603, 20250604
      Symbols: BNBUSDT, BTCUSDT, ETHUSDT
    - 1440-minute bars: 15 files (5 dates × 3 symbols)
      Same dates and symbols as above
    
    features/ (12 files):
    - Only for 20250604 and only specific feature types
    - Structure: features/1440/{feature_type}/20250604/{symbol}.parquet
    - Feature types: beta_1440, day_of_week, dvolume_1440_trmean, logret_resid_eqmkt_1440_trstd
    - 3 symbols each = 4 × 3 = 12 files
    
    fits/ (10 files):
    - Only in prod/1440/{hl,slz}/ directories
    - hl model: 6 files
      - fits.prod.1440.hl.20250606.csv
      - fits.prod.dev.1440.hl.20250606.csv
      - svm.hl_1440.20250605.features
      - svm.hl_1440.20250605.joblib
      - svm.hl_1440.20250606.features
      - svm.hl_1440.20250606.joblib
    - slz model: 4 files
      - fits.prod.1440.slz.20250606.csv
      - fits.prod.dev.1440.slz.20250606.csv
      - svm.slz_1440.20250605.features
      - svm.slz_1440.20250605.joblib
    
    live/ (12 files):
    - Market snapshots for 20250606
    - Files: 1749168000.parquet through 1749168660.parquet (60-second intervals)
    
    models/ (2 files):
    - Only for 20250604, only horizon 1440
    - models/1440/hl/models.1440.hl.20250604.parquet
    - models/1440/slz/models.1440.slz.20250604.parquet
    
    positions/ (1 file):
    - pos.20250606_0000.parquet (filtered to test symbols)
    
    universe/ (1 file):
    - universe.20250604.parquet
    
    delisting.txt (1 file)
"""

import os
import sys
import argparse
import json
import pandas as pd
from typing import List, Set
import shutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from lib.util.directory import DirectoryManager
from lib.util.util import LOCAL

# Import common fixture utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fixture_utils import (
    load_test_config, save_test_config,
    get_date_range, update_config_for_test, print_test_summary,
    copy_bar_files, copy_model_files,
    copy_universe_files, create_delisting_file,
    DEFAULT_TEST_SYMBOLS
)


# Test parameters
TEST_DATE = "20250606"  # Match the date in integration_test_server.py
HISTORICAL_DAYS = 6  # Need more days for 1440 lookback (3 days * 1440 minutes = 4320 minutes)
TEST_SYMBOLS = DEFAULT_TEST_SYMBOLS  # ['BTC', 'ETH', 'BNB']
HORIZONS = [15, 60, 1440]
MODELS = ["hl", "slz"]


def copy_live_market_data(
    source_dir: str,
    dest_dir: str,
    test_date: str,
    symbols: Set[str]
) -> None:
    """Copy live market data snapshots from production."""
    print("\nCopying live market data for {test_date}...")
    
    # Only copy the 12 specific files that are in git
    expected_files = [
        "1749168000.parquet", "1749168060.parquet", "1749168120.parquet",
        "1749168180.parquet", "1749168240.parquet", "1749168300.parquet",
        "1749168360.parquet", "1749168420.parquet", "1749168480.parquet",
        "1749168540.parquet", "1749168600.parquet", "1749168660.parquet"
    ]
    
    source_date_dir = os.path.join(source_dir, test_date)
    if not os.path.exists(source_date_dir):
        print("  ⚠ Warning: No live data found for {test_date} in {source_date_dir}")
        return
    
    dest_date_dir = os.path.join(dest_dir, test_date)
    os.makedirs(dest_date_dir, exist_ok=True)
    
    test_symbol_venues = {f"{s}USDT_binance-futures" for s in symbols}
    
    # Copy only the expected files
    copied = 0
    for snapshot_file in expected_files:
        source_file = os.path.join(source_date_dir, snapshot_file)
        if os.path.exists(source_file):
            try:
                # Read and filter to test symbols
                df = pd.read_parquet(source_file)
                if 'symbol_venue' in df.columns:
                    df = df[df['symbol_venue'].isin(test_symbol_venues)]
                
                if not df.empty:
                    dest_file = os.path.join(dest_date_dir, snapshot_file)
                    df.to_parquet(dest_file)
                    copied += 1
            except Exception as e:
                print("  ⚠ Warning: Failed to process {snapshot_file}: {e}")
    
    print("  ✓ Copied {copied} live market snapshots (expected 12)")


def consolidate_bar_files(
    source_dir: str,
    dest_dir: str,
    dates: List[str],
    horizons: List[int]
) -> None:
    """Consolidate per-symbol bar files into single date files for old format compatibility."""
    print("\nConsolidating bar files to old format...")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    for horizon in horizons:
        for date in dates:
            source_date_dir = os.path.join(source_dir, str(horizon), "binance-futures", date)
            
            if not os.path.exists(source_date_dir):
                print("  ⚠ Warning: No bar files found for {date} at horizon {horizon}")
                continue
            
            # Read all symbol files for this date
            dfs = []
            for file in os.listdir(source_date_dir):
                if file.endswith('.parquet'):
                    df = pd.read_parquet(os.path.join(source_date_dir, file))
                    dfs.append(df)
            
            if dfs:
                # Concatenate all symbols
                combined_df = pd.concat(dfs)
                combined_df = combined_df.sort_index()
                
                # Save consolidated file
                dest_file = os.path.join(dest_dir, f"bars_{horizon}_{date}.parquet")
                combined_df.to_parquet(dest_file)
                print("  ✓ Created bars_{horizon}_{date}.parquet with {len(combined_df)} rows")


def consolidate_feature_files(
    source_dir: str,
    dest_dir: str,
    dates: List[str],
    horizons: List[int],
    feature_types: List[str]
) -> None:
    """Consolidate per-symbol feature files into single date files for old format compatibility."""
    print("\nConsolidating feature files to old format...")
    
    for horizon in horizons:
        for date in dates:
            all_features_df = None
            
            for feature_type in feature_types:
                feature_date_dir = os.path.join(source_dir, str(horizon), feature_type, date)
                if not os.path.exists(feature_date_dir):
                    continue
                
                # Read all symbol files for this feature
                for file in os.listdir(feature_date_dir):
                    if file.endswith('.parquet'):
                        df = pd.read_parquet(os.path.join(feature_date_dir, file))
                        
                        if all_features_df is None:
                            all_features_df = df
                        else:
                            # Merge on index, avoiding duplicate columns
                            for col in df.columns:
                                if col not in all_features_df.columns:
                                    all_features_df[col] = df[col]
            
            if all_features_df is not None:
                # Save consolidated file
                dest_file = os.path.join(dest_dir, f"features_{horizon}_{date}.parquet")
                all_features_df.to_parquet(dest_file)
                
                # Count unique symbols
                if all_features_df.index.nlevels > 1:
                    n_symbols = len(all_features_df.index.get_level_values('symbol_venue').unique())
                else:
                    n_symbols = "unknown"
                print("  ✓ Created features_{horizon}_{date}.parquet with {n_symbols} symbols")


def consolidate_model_files(
    source_dir: str,
    dest_dir: str,
    dates: List[str],
    horizons: List[int],
    models: List[str]
) -> None:
    """Consolidate model files to old format."""
    print("\nConsolidating model files to old format...")
    
    for model in models:
        for horizon in horizons:
            for date in dates:
                model_file = os.path.join(source_dir, str(horizon), model, f"models.{horizon}.{model}.{date}.parquet")
                
                if os.path.exists(model_file):
                    df = pd.read_parquet(model_file)
                    dest_file = os.path.join(dest_dir, f"{model}_{horizon}_{date}.parquet")
                    df.to_parquet(dest_file)
                    print("  ✓ Created {model}_{horizon}_{date}.parquet")


def copy_position_files(source_dir: str, dest_dir: str, test_date: str, symbols: Set[str]) -> None:
    """Copy position files from production and filter to test symbols."""
    print("\nCopying position files...")
    os.makedirs(dest_dir, exist_ok=True)
    
    # Find position files for the test date
    import glob
    pattern = os.path.join(source_dir, f"pos.{test_date}_*.parquet")
    pos_files = sorted(glob.glob(pattern))
    
    if not pos_files:
        print("  ⚠ Warning: No position files found for {test_date}")
        return
    
    # Copy the first position file of the day
    source_file = pos_files[0]
    print("  Using position file: {os.path.basename(source_file)}")
    
    # Read and filter to test symbols
    df = pd.read_parquet(source_file)
    test_symbol_list = [f'{s}USDT' for s in symbols]
    
    if 'symbol' in df.columns:
        df_filtered = df[df['symbol'].isin(test_symbol_list)]
    else:
        print("  ⚠ Warning: No 'symbol' column in position file")
        df_filtered = df
    
    dest_file = os.path.join(dest_dir, os.path.basename(source_file))
    df_filtered.to_parquet(dest_file)
    print("  ✓ Copied and filtered {os.path.basename(source_file)} ({len(df_filtered)} positions)")


def reorganize_fits_for_prod(
    fits_dir: str,
    horizons: List[int],
    models_by_horizon: dict,
    test_date: str
) -> None:
    """Create fits/prod directory structure for BARS_TYPE_NEW."""
    print("\nCreating fits/prod directory structure...")
    
    # For BARS_TYPE_NEW, prod fits go in fits/prod
    prod_dir = os.path.join(fits_dir, "prod")
    os.makedirs(prod_dir, exist_ok=True)
    
    # Create directory structure for all horizons
    for horizon in horizons:
        horizon_dir = os.path.join(prod_dir, str(horizon))
        os.makedirs(horizon_dir, exist_ok=True)
        models = models_by_horizon.get(horizon, [])
        for model in models:
            model_dir = os.path.join(horizon_dir, model)
            os.makedirs(model_dir, exist_ok=True)
    
    # Copy existing fits from the non-prod structure to prod
    copied_count = 0
    for horizon in horizons:
        src_horizon_dir = os.path.join(fits_dir, str(horizon))
        if os.path.exists(src_horizon_dir):
            models = models_by_horizon.get(horizon, [])
            for model in models:
                src_model_dir = os.path.join(src_horizon_dir, model)
                if os.path.exists(src_model_dir):
                    dest_model_dir = os.path.join(prod_dir, str(horizon), model)
                    
                    # Copy all CSV files
                    csv_files = [f for f in os.listdir(src_model_dir) if f.endswith('.csv')]
                    for file in csv_files:
                        src_file = os.path.join(src_model_dir, file)
                        # Create prod-prefixed filename
                        new_name = file.replace("fits.", "fits.prod.")
                        dest_file = os.path.join(dest_model_dir, new_name)
                        shutil.copy2(src_file, dest_file)
                        print("  ✓ Copied {file} -> {new_name}")
                        copied_count += 1
                else:
                    print("  ⚠ Warning: No fits found for model={model}, horizon={horizon}")
    
    # Copy SVM models to prod directory
    svm_src = os.path.join(fits_dir, "svm")
    if os.path.exists(svm_src):
        svm_dest = os.path.join(prod_dir, "svm")
        if os.path.exists(svm_dest):
            shutil.rmtree(svm_dest)
        shutil.copytree(svm_src, svm_dest)
        print("  ✓ Copied SVM models to prod directory")
    
    print("  ✓ Created fits/prod directory structure with {copied_count} fit files")


def copy_latest_fit_files(
    source_base_dir: str,
    dest_base_dir: str,
    horizons: List[int],
    models_by_horizon: dict,
    test_date: str
) -> int:
    """Copy the latest available fit files for each horizon/model combination.
    
    This function copies production fit files into the already created
    dev and prod directory structure.
    """
    print("\nCopying production fit files if available...")
    
    import glob
    total_copied = 0
    
    # Try to copy existing fit files from production
    for horizon in horizons:
        models = models_by_horizon.get(horizon, [])
        for model in models:
            # Look for CSV fit files in production
            csv_pattern = os.path.join(source_base_dir, f"fits.{horizon}.{model}.*.csv")
            csv_files = sorted(glob.glob(csv_pattern))
            
            if csv_files:
                # Use the most recent file
                latest_csv = csv_files[-1]
                file_date = os.path.basename(latest_csv).split('.')[-2]
                
                # Copy to both dev and prod directories
                for fit_type in ['dev', 'prod']:
                    dest_dir = os.path.join(dest_base_dir, fit_type, str(horizon))
                    dest_filename = f"fits.{horizon}.{file_date}.{fit_type}.csv"
                    dest_path = os.path.join(dest_dir, dest_filename)
                    
                    if os.path.exists(dest_path):
                        # Already created by create_complete_fits_structure
                        continue
                    
                    shutil.copy2(latest_csv, dest_path)
                    total_copied += 1
                    print("  ✓ Copied production fit to {fit_type}/{horizon}/{dest_filename}")
            
            # Look for SVM files in production
            for fit_type in ['dev', 'prod']:
                svm_src_dir = os.path.join(source_base_dir, "svm", f"{model}_{horizon}")
                svm_dest_dir = os.path.join(dest_base_dir, fit_type, "svm", f"{model}_{horizon}")
                
                if os.path.exists(svm_src_dir):
                    svm_pattern = os.path.join(svm_src_dir, f"svm.{model}_{horizon}.*.features")
                    svm_files = sorted(glob.glob(svm_pattern))
                    
                    if svm_files:
                        # Copy the most recent SVM files
                        latest_svm = svm_files[-1]
                        svm_date = os.path.basename(latest_svm).split('.')[-2]
                        
                        for ext in ['features', 'joblib']:
                            src_file = os.path.join(svm_src_dir, f"svm.{model}_{horizon}.{svm_date}.{ext}")
                            dest_file = os.path.join(svm_dest_dir, f"svm.{model}_{horizon}.{svm_date}.{ext}")
                            
                            if os.path.exists(src_file) and not os.path.exists(dest_file):
                                shutil.copy2(src_file, dest_file)
                                total_copied += 1
                                print("  ✓ Copied SVM {ext} to {fit_type}/svm/{model}_{horizon}/")
    
    print("\nTotal: {total_copied} production fit files copied")
    return total_copied


def get_all_required_features_from_config() -> Set[str]:
    """Get all required features (with horizon suffixes) from the config."""
    config_file = os.path.join(os.path.dirname(__file__), "config_server_test.json")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    all_features = set()
    
    # Add core features that don't have horizon suffix
    all_features.update(['fittable', 'tradeable', 'expandable', 'hour_of_day', 'day_of_week', 'news_event'])
    
    # Add additional features required by SVM models but not in config
    all_features.update([
        'last_funding_rate_mean_1440',
        'cx.logret_1440', 
        'cx.dvolume_1440',
        'cx.logret_60',
        'cx.dvolume_60',
        'cx.logret_15',
        'cx.dvolume_15'
    ])
    
    # Extract features from FCASTS config
    if 'FCASTS' in config:
        for horizon_str, fcast_config in config['FCASTS'].items():
            if 'features' in fcast_config:
                all_features.update(fcast_config['features'])
    
    # Also add features from the prod/non-prod feature lists
    if 'FEATURES' in config:
        for horizon_str, feature_dict in config['FEATURES'].items():
            for feature_list in ['prod', 'non_prod']:
                if feature_list in feature_dict:
                    all_features.update(feature_dict[feature_list])
    
    return all_features


def get_all_feature_types_for_fixtures() -> dict:
    """Get all feature types that need to be copied based on the actual fixture structure."""
    # Based on the complete file list, these are all the feature types present
    feature_types_by_horizon = {
        15: ['logret_15', 'logret_15_lz', 'logret_15_lz_cz'],
        60: ['dvolume_60_lz_cz', 'dvolume_60_trmean', 'logret_60_lz', 'logret_60_lz_cz'],
        1440: [
            'ba_imbal_1440', 'beta_1440', 'beta_t_1440', 'day_of_week',
            'dvolume_1440_d', 'dvolume_1440_dp', 'dvolume_1440_lz',
            'dvolume_1440_trmean', 'dvolume_1440_trmean_cz', 'hour_of_day',
            'logret_1440', 'logret_1440_lz', 'logret_1440_lz_cz',
            'logret_1440_max', 'logret_1440_min', 'logret_1440_trmean',
            'logret_1440_trstd', 'logret_funding_adj_resid_wgtmkt_1440_lz',
            'logret_resid_wgtmkt_1440_lz', 'median_time_bucket_dvolume_1440',
            'news_event_decayed_1440', 'open_interest_1440_d', 'open_interest_1440_dp',
            'relative_spread_1440_lz', 'relative_spread_1440_trmean',
            'relative_updates_1440_lz', 'risk_1440', 'rsi_1440', 'trade_sz_1440_lz'
        ]
    }
    return feature_types_by_horizon


def copy_all_feature_files_exact_structure(
    source_base_dir: str,
    dest_base_dir: str,
    symbols: Set[str],
    dates: List[str],
    feature_types_by_horizon: dict
) -> int:
    """Copy all feature files matching the exact fixture structure."""
    print("\nCopying feature files with exact structure...")
    
    total_copied = 0
    
    # Copy features for each horizon based on the exact structure
    for horizon, feature_types in feature_types_by_horizon.items():
        print("\nHorizon {horizon}: {len(feature_types)} feature types")
        
        for feature_type in feature_types:
            for date in dates:
                source_date_dir = os.path.join(source_base_dir, str(horizon), feature_type, date)
                dest_date_dir = os.path.join(dest_base_dir, str(horizon), feature_type, date)
                
                if not os.path.exists(source_date_dir):
                    # Some features might be stored without the horizon subdirectory
                    alt_source_dir = os.path.join(source_base_dir, feature_type, date)
                    if os.path.exists(alt_source_dir):
                        source_date_dir = alt_source_dir
                    else:
                        print("  ⚠ Warning: No source dir for {horizon}/{feature_type}/{date}")
                        continue
                
                # Create destination directory
                os.makedirs(dest_date_dir, exist_ok=True)
                
                copied_count = 0
                for symbol in symbols:
                    # File naming pattern: features.{horizon}.{feature_type}.{date}.{symbol}USDT.parquet
                    source_filename = f"features.{horizon}.{feature_type}.{date}.{symbol}USDT.parquet"
                    source_file = os.path.join(source_date_dir, source_filename)
                    dest_file = os.path.join(dest_date_dir, source_filename)
                    
                    if os.path.exists(source_file):
                        shutil.copy2(source_file, dest_file)
                        copied_count += 1
                    else:
                        # Try alternative naming without the features prefix
                        alt_filename = f"{feature_type}.{date}.{symbol}USDT.parquet"
                        alt_source_file = os.path.join(source_date_dir, alt_filename)
                        if os.path.exists(alt_source_file):
                            # Copy with the standard naming
                            shutil.copy2(alt_source_file, dest_file)
                            copied_count += 1
                
                if copied_count > 0:
                    print("  {horizon}/{feature_type}/{date}: Copied {copied_count} files")
                    total_copied += copied_count
    
    print("\nTotal: {total_copied} feature files copied")
    return total_copied


def get_required_feature_types(horizons: List[int]) -> List[str]:
    """Get the list of feature types needed for server test."""
    # Parse the test config to get all required features
    config_file = os.path.join(os.path.dirname(__file__), "config_server_test.json")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    all_features = set()
    
    # Add core features that don't have horizon suffix
    all_features.update(['fittable', 'tradeable', 'expandable', 'hour_of_day', 'day_of_week', 'news_event'])
    
    # Extract features from FCASTS config
    if 'FCASTS' in config:
        for horizon_str, fcast_config in config['FCASTS'].items():
            if 'features' in fcast_config:
                for feature in fcast_config['features']:
                    # Remove horizon suffix to get base feature name
                    for h in horizons:
                        suffix = f"_{h}"
                        if feature.endswith(suffix):
                            base_feature = feature[:-len(suffix)]
                            all_features.add(base_feature)
                            break
                    else:
                        # Feature doesn't have horizon suffix
                        all_features.add(feature)
    
    # Also add features from the prod/non-prod feature lists
    if 'FEATURES' in config:
        for horizon_str, feature_dict in config['FEATURES'].items():
            for feature_list in ['prod', 'non_prod']:
                if feature_list in feature_dict:
                    for feature in feature_dict[feature_list]:
                        # Remove horizon suffix
                        for h in horizons:
                            suffix = f"_{h}"
                            if feature.endswith(suffix):
                                base_feature = feature[:-len(suffix)]
                                all_features.add(base_feature)
                                break
                        else:
                            all_features.add(feature)
    
    return sorted(list(all_features))


def create_fits_directory_structure(
    fits_dir: str,
    horizons: List[int],
    models_by_horizon: dict
) -> None:
    """Create the fits directory structure with dev and prod subdirectories."""
    print("\nCreating fits directory structure...")
    
    # Create dev and prod directories
    dev_dir = os.path.join(fits_dir, "dev")
    prod_dir = os.path.join(fits_dir, "prod")
    
    # Create horizon directories
    for dir_path in [dev_dir, prod_dir]:
        for horizon in horizons:
            horizon_dir = os.path.join(dir_path, str(horizon))
            os.makedirs(horizon_dir, exist_ok=True)
    
    # Create SVM directories
    for fit_type in ['dev', 'prod']:
        svm_dir = os.path.join(fits_dir, fit_type, "svm")
        for horizon in horizons:
            models = models_by_horizon.get(horizon, [])
            for model in models:
                model_dir = os.path.join(svm_dir, f"{model}_{horizon}")
                os.makedirs(model_dir, exist_ok=True)
    
    print("  ✓ Created fits directory structure")


def copy_prod_fits_to_correct_location(fixture_output_dir: str) -> None:
    """Copy fits from fits/prod to prod_fits directory as expected by server."""
    print("\nSetting up prod_fits directory...")
    
    fits_prod_dir = os.path.join(fixture_output_dir, "fits", "prod")
    prod_fits_dir = os.path.join(fixture_output_dir, "prod_fits")
    
    if not os.path.exists(fits_prod_dir):
        print("  ⚠ Warning: No fits/prod directory found")
        return
    
    # Remove existing prod_fits directory if it exists
    if os.path.exists(prod_fits_dir):
        shutil.rmtree(prod_fits_dir)
    
    # Copy entire fits/prod to prod_fits
    shutil.copytree(fits_prod_dir, prod_fits_dir)
    
    # Now ensure all fit files have the correct naming
    for root, dirs, files in os.walk(prod_fits_dir):
        for file in files:
            if file.endswith('.csv') and file.startswith('fits.'):
                # Check if file needs prod prefix
                if not '.prod.' in file and not '.dev.' in file:
                    # Add prod prefix: fits.15.20250601.csv -> fits.15.20250601.prod.csv
                    parts = file.split('.')
                    if len(parts) >= 4:  # fits.horizon.date.csv
                        new_name = f"{parts[0]}.{parts[1]}.{parts[2]}.prod.{parts[3]}"
                        old_path = os.path.join(root, file)
                        new_path = os.path.join(root, new_name)
                        os.rename(old_path, new_path)
                        print("  ✓ Renamed {file} -> {new_name}")
    
    print("  ✓ Prod fits structure created in prod_fits/")


def generate_master_fixtures(config: dict, test_date: str) -> None:
    """Generate master fixtures by running the server.
    
    Note: This requires all input data files to be in place first.
    """
    print("\n" + "=" * 60)
    print("GENERATING MASTER FIXTURES")
    print("=" * 60)
    
    # Master fixtures for server are generated by running the test itself
    print("\n⚠️  Server master fixtures must be generated by running:")
    print("    ./bin/regenerate_master_fixture.sh -n server")
    print("\nThis will:")
    print("  1. Run the AlphaServer with test data")
    print("  2. Generate alpha signals and optimized targets")
    print("  3. Save results to master/ directory")
    print("\nThe generate_fixtures.py script only copies input data files.")


def main():
    """Generate test fixtures for server integration test."""
    parser = argparse.ArgumentParser(description='Generate server test fixtures')
    parser.add_argument('--master-only', action='store_true',
                        help='Only regenerate master fixtures (requires running server)')
    args = parser.parse_args()
    
    print("Running in {'LOCAL' if LOCAL else 'PROD'} mode!")
    
    # Set up directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_output_dir = os.path.join(base_dir, "data")
    config_file = os.path.join(base_dir, "config_server_test.json")
    
    # Initialize directory manager
    dir_manager = DirectoryManager()
    
    # Get date range
    dates = get_date_range(TEST_DATE, HISTORICAL_DAYS)
    
    # Print test summary
    print_test_summary("server", TEST_SYMBOLS, TEST_DATE, dates, HORIZONS, MODELS)
    
    if args.master_only:
        # Master fixtures are generated by running the test
        config = load_test_config(config_file)
        generate_master_fixtures(config, TEST_DATE)
        return
    
    if not args.master_only:
        # IMPORTANT: Delete ALL data in fixtures/server/data/ for a clean slate
        print("\nStep 1: Cleaning ALL existing data files...")
        print("=" * 60)
        if os.path.exists(fixture_output_dir):
            shutil.rmtree(fixture_output_dir)
            print("  ✓ Removed entire data directory: {fixture_output_dir}")
        else:
            print("  ℹ Data directory did not exist: {fixture_output_dir}")
        
        # Recreate the data directory
        os.makedirs(fixture_output_dir, exist_ok=True)
        print("  ✓ Created fresh data directory: {fixture_output_dir}")
        
        print("\nStep 2: Copying production data files...")
        print("=" * 60)
        
        # Copy bar files - only 1 and 1440 minute bars for dates 20250531-20250604
        bar_dates = ["20250531", "20250601", "20250602", "20250603", "20250604"]
        copy_bar_files(
            source_base_dir=dir_manager.BAR_DIR,
            dest_base_dir=os.path.join(fixture_output_dir, "bars"),
            symbols=set(TEST_SYMBOLS),
            dates=bar_dates,
            frequencies=[1, 1440]  # Only 1 and 1440 minute bars
        )
        
        # Copy features - ONLY for 20250604 and ONLY specific features as per git
        feature_date = ["20250604"]
        specific_features = {
            1440: ['beta_1440', 'day_of_week', 'dvolume_1440_trmean', 'logret_resid_eqmkt_1440_trstd']
        }
        
        copy_all_feature_files_exact_structure(
            source_base_dir=dir_manager.FEATURES_DIR,
            dest_base_dir=os.path.join(fixture_output_dir, "features"),
            symbols=set(TEST_SYMBOLS),
            dates=feature_date,
            feature_types_by_horizon=specific_features
        )
        
        # Define models by horizon as per the actual fixture structure
        models_by_horizon = {
            1440: ["hl", "slz"]
        }
        
        # Copy model files - ONLY for 20250604, only horizon 1440
        model_date = ["20250604"]
        copy_model_files(
            source_base_dir=dir_manager.MODELS_DIR,
            dest_base_dir=os.path.join(fixture_output_dir, "models"),
            dates=model_date,
            horizons=[1440],
            models=["hl", "slz"]
        )
        
        # Manually copy fit files to match git structure exactly
        print("\nCopying fit files...")
        fits_prod_dir = os.path.join(fixture_output_dir, "fits", "prod", "1440")
        os.makedirs(os.path.join(fits_prod_dir, "hl"), exist_ok=True)
        os.makedirs(os.path.join(fits_prod_dir, "slz"), exist_ok=True)
        
        # Copy CSV fit files with prod naming
        for model in ["hl", "slz"]:
            # Look for source files
            source_patterns = [
                os.path.join(dir_manager.FITS_DIR_DEV, f"fits.1440.{model}.20250606.csv"),
                os.path.join(dir_manager.FITS_DIR_DEV, "1440", model, f"fits.1440.{model}.20250606.csv"),
                os.path.join(dir_manager.FITS_DIR_DEV, "prod", "1440", model, f"fits.prod.1440.{model}.20250606.csv")
            ]
            
            for source_file in source_patterns:
                if os.path.exists(source_file):
                    # Copy as prod files
                    dest_file = os.path.join(fits_prod_dir, model, f"fits.prod.1440.{model}.20250606.csv")
                    shutil.copy2(source_file, dest_file)
                    print("  ✓ Copied {os.path.basename(dest_file)}")
                    
                    # Also copy as dev file
                    dest_dev_file = os.path.join(fits_prod_dir, model, f"fits.prod.dev.1440.{model}.20250606.csv")
                    shutil.copy2(source_file, dest_dev_file)
                    print("  ✓ Copied {os.path.basename(dest_dev_file)}")
                    break
            
            # Copy SVM files - git has different dates for different models
            if model == "hl":
                # hl has both 20250605 and 20250606
                dates = ["20250605", "20250606"]
            else:
                # slz only has 20250605
                dates = ["20250605"]
            
            for date in dates:
                svm_patterns = [
                    os.path.join(dir_manager.FITS_DIR_DEV, "svm", f"{model}_1440", f"svm.{model}_1440.{date}"),
                    os.path.join(dir_manager.FITS_DIR_DEV, "prod", "svm", f"{model}_1440", f"svm.{model}_1440.{date}")
                ]
                
                for svm_base in svm_patterns:
                    if os.path.exists(f"{svm_base}.features"):
                        shutil.copy2(f"{svm_base}.features", 
                                   os.path.join(fits_prod_dir, model, f"svm.{model}_1440.{date}.features"))
                        shutil.copy2(f"{svm_base}.joblib",
                                   os.path.join(fits_prod_dir, model, f"svm.{model}_1440.{date}.joblib"))
                        print("  ✓ Copied svm.{model}_1440.{date}.features/joblib")
                        break
        
        # Copy universe files - ONLY universe.20250604.parquet as per git
        copy_universe_files(
            source_dir=dir_manager.UNIVERSE_DIR,
            dest_dir=os.path.join(fixture_output_dir, "universe"),
            dates=["20250604"]
        )
        
        # NOTE: No tardis_bars or prod_fits directories in git - skip consolidation
        
        # Copy position files from trading directory
        copy_position_files(
            source_dir=os.path.join(os.environ.get('ROOT_DIR', ''), "trading/positions"),
            dest_dir=os.path.join(fixture_output_dir, "positions"),
            test_date=TEST_DATE,
            symbols=set(TEST_SYMBOLS)
        )
        
        # Copy live market data
        copy_live_market_data(
            source_dir=dir_manager.LIVE_DATA_DIR,
            dest_dir=os.path.join(fixture_output_dir, "live"),
            test_date=TEST_DATE,
            symbols=set(TEST_SYMBOLS)
        )
        
        # Create delisting file
        create_delisting_file(fixture_output_dir)
        
        # Load and update config
        config = load_test_config(config_file)
        config = update_config_for_test(config, symbols=TEST_SYMBOLS, dynamic_universe=False, lookback_days=3)
        save_test_config(config_file, config)
        
        print("\n✓ Updated config with test settings")
        
        # Verify generated files match git
        print("\nStep 3: Verifying generated files match git...")
        print("=" * 60)
        
        # Get list of generated files
        generated_files = []
        for root, dirs, files in os.walk(fixture_output_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                generated_files.append(rel_path)
        
        # Get list of git files
        git_files_output = os.popen(f"cd {base_dir} && git ls-files data/").read().strip().split('\n')
        git_files = [f for f in git_files_output if f]
        
        generated_set = set(generated_files)
        git_set = set(git_files)
        
        # Check for differences
        missing_from_generated = git_set - generated_set
        extra_in_generated = generated_set - git_set
        
        print("Files in git: {len(git_set)}")
        print("Files generated: {len(generated_set)}")
        
        if missing_from_generated:
            print("\n⚠️  WARNING: Files in git but not generated ({len(missing_from_generated)}):")
            for f in sorted(missing_from_generated)[:10]:
                print("  - {f}")
            if len(missing_from_generated) > 10:
                print("  ... and {len(missing_from_generated) - 10} more")
        
        if extra_in_generated:
            print("\n⚠️  WARNING: Files generated but not in git ({len(extra_in_generated)}):")
            for f in sorted(extra_in_generated)[:10]:
                print("  - {f}")
            if len(extra_in_generated) > 10:
                print("  ... and {len(extra_in_generated) - 10} more")
        
        if not missing_from_generated and not extra_in_generated:
            print("\n✅ Perfect match! Generated files exactly match git.")
    
    
    print("\n" + "="*60)
    print("✅ Fixture generation complete!")
    print("="*60)
    
    print("\nTo regenerate all fixtures:")
    print("  cd {base_dir}")
    print("  python generate_fixtures.py")
    print("\nTo regenerate only master fixtures:")
    print("  python generate_fixtures.py --master-only")


if __name__ == "__main__":
    main()
