#!/usr/bin/env python3
"""Generate bar test fixtures from production prebar data.

IMPORTANT: All data must be copied from ~/stat_arb/data/ (production data directories).
           Do not create synthetic or dummy data files.
           Do NOT create symlinks - copy actual files to maintain test isolation.

This script:
1. Copies prebar files from the production data directory
2. Copies required metadata files
3. Generates master fixture bar files for test comparison

Usage:
    python generate_fixtures.py                    # Generate all fixtures
    python generate_fixtures.py --master-only      # Only regenerate master fixtures

Complete list of files that should be generated in test/fixtures/bars/data/:
    prebars/live/20250528/binance-futures/prebars.live.binance-futures.20250528.BNBUSDT.parquet
    prebars/live/20250528/binance-futures/prebars.live.binance-futures.20250528.BTCUSDT.parquet
    prebars/live/20250528/binance-futures/prebars.live.binance-futures.20250528.ETHUSDT.parquet
    prebars/live/20250529/binance-futures/prebars.live.binance-futures.20250529.BNBUSDT.parquet
    prebars/live/20250529/binance-futures/prebars.live.binance-futures.20250529.BTCUSDT.parquet
    prebars/live/20250529/binance-futures/prebars.live.binance-futures.20250529.ETHUSDT.parquet
    prebars/live/20250530/binance-futures/prebars.live.binance-futures.20250530.BNBUSDT.parquet
    prebars/live/20250530/binance-futures/prebars.live.binance-futures.20250530.BTCUSDT.parquet
    prebars/live/20250530/binance-futures/prebars.live.binance-futures.20250530.ETHUSDT.parquet
    prebars/live/20250531/binance-futures/prebars.live.binance-futures.20250531.BNBUSDT.parquet
    prebars/live/20250531/binance-futures/prebars.live.binance-futures.20250531.BTCUSDT.parquet
    prebars/live/20250531/binance-futures/prebars.live.binance-futures.20250531.ETHUSDT.parquet
    prebars/live/20250601/binance-futures/prebars.live.binance-futures.20250601.BNBUSDT.parquet
    prebars/live/20250601/binance-futures/prebars.live.binance-futures.20250601.BTCUSDT.parquet
    prebars/live/20250601/binance-futures/prebars.live.binance-futures.20250601.ETHUSDT.parquet
    prebars/live/20250602/binance-futures/prebars.live.binance-futures.20250602.BNBUSDT.parquet
    prebars/live/20250602/binance-futures/prebars.live.binance-futures.20250602.BTCUSDT.parquet
    prebars/live/20250602/binance-futures/prebars.live.binance-futures.20250602.ETHUSDT.parquet
    prebars/live/20250603/binance-futures/prebars.live.binance-futures.20250603.BNBUSDT.parquet
    prebars/live/20250603/binance-futures/prebars.live.binance-futures.20250603.BTCUSDT.parquet
    prebars/live/20250603/binance-futures/prebars.live.binance-futures.20250603.ETHUSDT.parquet
    prebars/live/20250604/binance-futures/prebars.live.binance-futures.20250604.BNBUSDT.parquet
    prebars/live/20250604/binance-futures/prebars.live.binance-futures.20250604.BTCUSDT.parquet
    prebars/live/20250604/binance-futures/prebars.live.binance-futures.20250604.ETHUSDT.parquet
    prebars/live/20250605/binance-futures/prebars.live.binance-futures.20250605.BNBUSDT.parquet
    prebars/live/20250605/binance-futures/prebars.live.binance-futures.20250605.BTCUSDT.parquet
    prebars/live/20250605/binance-futures/prebars.live.binance-futures.20250605.ETHUSDT.parquet
    prebars/live/20250606/binance-futures/prebars.live.binance-futures.20250606.BNBUSDT.parquet
    prebars/live/20250606/binance-futures/prebars.live.binance-futures.20250606.BTCUSDT.parquet
    prebars/live/20250606/binance-futures/prebars.live.binance-futures.20250606.ETHUSDT.parquet
    prebars/live/20250607/binance-futures/prebars.live.binance-futures.20250607.BNBUSDT.parquet
    prebars/live/20250607/binance-futures/prebars.live.binance-futures.20250607.BTCUSDT.parquet
    prebars/live/20250607/binance-futures/prebars.live.binance-futures.20250607.ETHUSDT.parquet
    prebars/live/20250608/binance-futures/prebars.live.binance-futures.20250608.BNBUSDT.parquet
    prebars/live/20250608/binance-futures/prebars.live.binance-futures.20250608.BTCUSDT.parquet
    prebars/live/20250608/binance-futures/prebars.live.binance-futures.20250608.ETHUSDT.parquet
    prebars/live/20250609/binance-futures/prebars.live.binance-futures.20250609.BNBUSDT.parquet
    prebars/live/20250609/binance-futures/prebars.live.binance-futures.20250609.BTCUSDT.parquet
    prebars/live/20250609/binance-futures/prebars.live.binance-futures.20250609.ETHUSDT.parquet
    secdata/binance_meta/meta.20250528.parquet
    secdata/binance_meta/meta.20250529.parquet
    secdata/binance_meta/meta.20250530.parquet
    secdata/binance_meta/meta.20250531.parquet
    secdata/binance_meta/meta.20250601.parquet
    secdata/binance_meta/meta.20250602.parquet
    secdata/binance_meta/meta.20250603.parquet
    secdata/binance_meta/meta.20250604.parquet
    secdata/binance_meta/meta.20250605.parquet
    secdata/binance_meta/meta.20250606.parquet
    secdata/binance_meta/meta.20250607.parquet
    secdata/binance_meta/meta.20250608.parquet
    secdata/binance_meta/meta.20250609.parquet
"""

import os
import sys
import argparse
import shutil
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

# Import common fixture utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fixture_utils import (
    load_test_config, save_test_config, clean_fixture_directory,
    get_date_range, update_config_for_test, print_test_summary,
    copy_secdata_files
)

from lib.util.directory import dir_manager, DirectoryManager
from lib.util.time_util import date_to_str, date_str_to_date
from lib.bars.bar_generator import BarGenerator
from lib.util.util import TARDIS_EXCHANGE


# Test configuration
TEST_DATE = "20250609"
HISTORICAL_DAYS = 12  # Need more days to satisfy bar generator lookback requirements
TEST_FREQUENCY = 15  # 15-minute bars for testing
TEST_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Only test 3 symbols


# Removed copy_prebar_files function - now inline in main() to use fixture_utils more directly


def generate_master_fixtures(config: dict, dates: list) -> None:
    """Generate master fixture bar files that tests will compare against."""
    print("\n" + "=" * 60)
    print("GENERATING MASTER FIXTURES")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_input_dir = os.path.join(script_dir, "data", "prebars", "live")
    fixture_output_dir = os.path.join(script_dir, "master")
    
    # Clean master directory before generating new fixtures
    print("Cleaning master directory...")
    clean_fixture_directory(fixture_output_dir, [""])
    
    # Create a test-specific directory manager
    test_dir_manager = DirectoryManager(
        data_dir=os.path.join(script_dir, "data"),
        trading_dir=os.path.join(script_dir, "trading")
    )
    
    # Process each date
    for date_str in dates:
        date = date_str_to_date(date_str)
        print(f"\nGenerating bars for {date_str}...")
        
        try:
            # Create bar generator
            bar_generator = BarGenerator(
                config=config,
                venue="binance-futures",
                start_date=date,
                end_date=date,
                chunk_days=7,  # Reduced from 30 for test efficiency
                horizons=[TEST_FREQUENCY],  # Only generate 15-minute bars for test
                debug=False,
                pool_size=1,
                output_dir=fixture_output_dir,
                bars_dir_manager=test_dir_manager,
            )
            
            # Run bar generation
            bar_generator.run()
            
            # Consolidate the output files into a single master fixture
            consolidate_master_fixture(fixture_output_dir, TEST_FREQUENCY, date, TEST_SYMBOLS)
            
            print(f"  ✓ Generated master fixture for {date_str}")
            
        except Exception as e:
            print(f"  ✗ Failed to generate master fixture for {date_str}: {e}")
            raise


def consolidate_master_fixture(output_dir: str, horizon: int, date, symbols: list) -> None:
    """Consolidate individual symbol bar files into single master fixture file."""
    date_str = date_to_str(date)
    bar_files = []
    
    # Read individual symbol files
    bar_dir = os.path.join(output_dir, str(horizon), TARDIS_EXCHANGE, date_str)
    
    if os.path.exists(bar_dir):
        for symbol in symbols:
            # Files are created with the full symbol name (already includes USDT)
            file_path = os.path.join(bar_dir, f"bars.{horizon}.{TARDIS_EXCHANGE}.{date_str}.{symbol}.parquet")
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                # The bar files only have 'ts' as index, need to add symbol_venue
                df = df.reset_index()
                df['symbol'] = symbol
                df['venue'] = TARDIS_EXCHANGE
                df['symbol_venue'] = f"{symbol}_{TARDIS_EXCHANGE}"
                # Set the standard MultiIndex
                df = df.set_index(['ts', 'symbol_venue'])
                bar_files.append(df)
    
    if bar_files:
        # Combine all symbols
        combined_df = pd.concat(bar_files)
        
        # Save as master fixture
        master_file = os.path.join(output_dir, f"bars_{horizon}_{date_str}.parquet")
        combined_df.to_parquet(master_file)
        print(f"    → Created {master_file}")


def main():
    """Main function to generate bar test fixtures."""
    parser = argparse.ArgumentParser(description='Generate bar test fixtures')
    parser.add_argument('--master-only', action='store_true', 
                        help='Only regenerate master fixtures (skip copying prebars)')
    args = parser.parse_args()
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config_bar_generation_test.json")
    data_dir = os.path.join(script_dir, "data")
    
    # Load configuration
    config = load_test_config(config_path)
    
    # Use minimal symbols for test
    symbols = set(TEST_SYMBOLS)
    config = update_config_for_test(config, symbols=list(symbols))
    
    # Calculate date range
    dates = get_date_range(TEST_DATE, HISTORICAL_DAYS)
    test_dates = [TEST_DATE]  # Only generate bars for the test date
    
    # Print test summary
    print_test_summary("bars", list(symbols), TEST_DATE, dates, [1, TEST_FREQUENCY])
    
    if not args.master_only:
        # IMPORTANT: Delete ALL data in fixtures/bars/data/ for a clean slate
        print("\nCleaning ALL data files...")
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            print(f"  ✓ Removed entire data directory: {data_dir}")
        
        # Recreate the data directory structure
        os.makedirs(data_dir, exist_ok=True)
        print(f"  ✓ Created fresh data directory: {data_dir}")
        
        # Set up directories
        source_base_dir = os.path.join(dir_manager.PREBAR_DIR, "live")
        dest_base_dir = os.path.join(data_dir, "prebars", "live")
        
        # Use fixture_utils function to copy prebar files with proper structure
        print("\nCopying prebar files...")
        print(f"Source: {source_base_dir}")
        print(f"Destination: {dest_base_dir}")
        
        total_copied = 0
        for date in dates:
            source_date_dir = os.path.join(source_base_dir, date, "binance-futures")
            dest_date_dir = os.path.join(dest_base_dir, date, "binance-futures")
            
            if not os.path.exists(source_date_dir):
                print(f"\nWarning: Source directory not found: {source_date_dir}")
                continue
                
            # Create destination directory
            os.makedirs(dest_date_dir, exist_ok=True)
            
            print(f"\nDate {date}:")
            copied_count = 0
            
            for symbol in symbols:
                # Production prebar files already have USDT suffix in symbol
                source_filename = f"prebars.live.binance-futures.{date}.{symbol}.parquet"
                source_file = os.path.join(source_date_dir, source_filename)
                dest_file = os.path.join(dest_date_dir, source_filename)
                
                if os.path.exists(source_file):
                    shutil.copy2(source_file, dest_file)
                    copied_count += 1
                    print(f"  ✓ {source_filename}")
                else:
                    print(f"  ✗ {source_filename} (not found)")
            
            print(f"  → Copied {copied_count} files")
            total_copied += copied_count
        
        print(f"\nTotal: {total_copied} prebar files copied")
        
        # Verify we have all 39 prebar files (13 dates * 3 symbols)
        expected_prebar_files = len(dates) * len(symbols)
        if total_copied != expected_prebar_files:
            print(f"\n⚠️  WARNING: Expected {expected_prebar_files} prebar files but only copied {total_copied}")
        
        # Copy actual tardis prebars for ADVP calculation (DataLoader.load_prebar_advp_map uses tardis)
        tardis_source_base_dir = os.path.join(dir_manager.PREBAR_DIR, "tardis")
        tardis_dest_base_dir = os.path.join(data_dir, "prebars", "tardis")
        print("\nCopying tardis prebar files (for ADVP calculation)...")
        print(f"Source: {tardis_source_base_dir}")
        print(f"Destination: {tardis_dest_base_dir}")
        
        tardis_total_copied = 0
        for date in dates:
            source_date_dir = os.path.join(tardis_source_base_dir, date, "binance-futures")
            dest_date_dir = os.path.join(tardis_dest_base_dir, date, "binance-futures")
            
            if not os.path.exists(source_date_dir):
                print(f"\nWarning: Tardis source directory not found: {source_date_dir}")
                continue
                
            # Create destination directory
            os.makedirs(dest_date_dir, exist_ok=True)
            
            print(f"\nDate {date} (tardis):")
            copied_count = 0
            
            for symbol in symbols:
                # Production tardis prebar files
                source_filename = f"prebars.tardis.binance-futures.{date}.{symbol}.parquet"
                source_file = os.path.join(source_date_dir, source_filename)
                dest_file = os.path.join(dest_date_dir, source_filename)
                
                if os.path.exists(source_file):
                    shutil.copy2(source_file, dest_file)
                    copied_count += 1
                    print(f"  ✓ {source_filename}")
                else:
                    print(f"  ✗ {source_filename} (not found)")
            
            print(f"  → Copied {copied_count} tardis files")
            tardis_total_copied += copied_count
        
        print(f"\nTotal: {tardis_total_copied} tardis prebar files copied")
        
        # Verify we have all tardis files
        expected_tardis_files = len(dates) * len(symbols)
        if tardis_total_copied != expected_tardis_files:
            print(f"\n⚠️  WARNING: Expected {expected_tardis_files} tardis prebar files but only copied {tardis_total_copied}")
        
        # Use fixture_utils to copy metadata files
        copy_secdata_files(
            os.path.dirname(dir_manager.BINANCE_META_DIR),  # Get parent secdata directory
            os.path.join(data_dir, "secdata"),
            dates,
            ['binance_meta']
        )
        
        # Verify we have all 13 meta files (one per date)
        meta_files_count = len(os.listdir(os.path.join(data_dir, "secdata", "binance_meta")))
        if meta_files_count != len(dates):
            print(f"\n⚠️  WARNING: Expected {len(dates)} meta files but found {meta_files_count}")
    
    # Generate master fixtures
    generate_master_fixtures(config, test_dates)
    
    # Save updated config
    save_test_config(config_path, config)
    print("\n✓ Updated config with minimal test symbols")
    
    # Final verification: count all generated files
    print("\n" + "=" * 60)
    print("VERIFYING GENERATED FILES")
    print("=" * 60)
    
    expected_files = 52  # As listed in the docstring
    actual_files = 0
    
    # Count prebars files
    prebars_dir = os.path.join(data_dir, "prebars", "live")
    if os.path.exists(prebars_dir):
        for _, _, files in os.walk(prebars_dir):
            actual_files += len([f for f in files if f.endswith('.parquet')])
    
    # Count secdata files
    secdata_dir = os.path.join(data_dir, "secdata", "binance_meta")
    if os.path.exists(secdata_dir):
        actual_files += len([f for f in os.listdir(secdata_dir) if f.endswith('.parquet')])
    
    print(f"\nExpected files: {expected_files}")
    print(f"Actual files: {actual_files}")
    
    if actual_files == expected_files:
        print("✅ All expected files generated successfully!")
    else:
        print(f"⚠️  WARNING: File count mismatch! Expected {expected_files} but generated {actual_files}")
    
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
