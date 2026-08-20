#!/usr/bin/env python3
"""
Count prebars, bars, and features files by date range.
Shows feature counts for each horizon in the table.
Features count uses dvolume_HORIZON_trmean as the representative feature.
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import ROOT_DIR from environment or directory module
from lib.util.directory import ROOT_DIR

DATA_DIR = f"{ROOT_DIR}/data"

# Available horizons in the system
HORIZONS = [15, 60, 120, 360, 720, 1440, 4320, 10080, 43200]


def daterange(start_date: datetime, end_date: datetime):
    """Generate dates between start and end date."""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def count_files_in_dir(directory: Path) -> int:
    """Count parquet files in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.parquet")))


def count_prebars_for_date(date_str: str) -> Dict[str, int]:
    """Count prebars files for a specific date."""
    counts = {
        'tardis': 0,
        'live': 0,
        'total': 0
    }
    
    # Check tardis prebars
    tardis_dir = Path(DATA_DIR) / "prebars" / "tardis" / date_str / "binance-futures"
    if tardis_dir.exists():
        counts['tardis'] = count_files_in_dir(tardis_dir)
    
    # Check live prebars
    live_dir = Path(DATA_DIR) / "prebars" / "live" / date_str / "binance-futures"
    if live_dir.exists():
        counts['live'] = count_files_in_dir(live_dir)
    
    counts['total'] = counts['tardis'] + counts['live']
    return counts


def count_bars_for_date(date_str: str) -> Dict[str, int]:
    """Count bar files for a specific date (1440 minute horizon only)."""
    # Only count 1440 minute (daily) bars for the main bars column
    bar_dir = Path(DATA_DIR) / "bars" / "1440" / "binance-futures" / date_str
    count = count_files_in_dir(bar_dir)
    
    counts = {
        'total': count,
        'horizon_1440': count
    }
    
    # Still collect all horizons for detailed reporting if needed
    for h in HORIZONS:
        if h != 1440:  # Already counted above
            bar_dir = Path(DATA_DIR) / "bars" / str(h) / "binance-futures" / date_str
            counts[f'horizon_{h}'] = count_files_in_dir(bar_dir)
    
    return counts


def count_features_for_date(date_str: str) -> Dict[str, int]:
    """
    Count feature files for a specific date using dvolume_HORIZON_trmean as representative.
    This gives us a reliable count of unique symbols that have features.
    """
    horizons_to_check = HORIZONS
    
    counts = {}
    total = 0
    
    for h in horizons_to_check:
        # Use dvolume_HORIZON_trmean as the representative feature
        feature_name = f"dvolume_{h}_trmean"
        feature_dir = Path(DATA_DIR) / "features" / str(h) / feature_name / date_str
        count = count_files_in_dir(feature_dir)
        counts[f'horizon_{h}'] = count
        total += count
    
    counts['total'] = total
    return counts


def print_summary_table(results: List[Dict]):
    """Print results in a formatted table with feature counts per horizon."""
    
    # Calculate column width for better formatting
    total_width = 210
    
    # Print header
    print("\n" + "="*total_width)
    print(f"DATA FILE COUNTS BY DATE - FEATURES BY HORIZON")
    print("="*total_width)
    
    # Build header with horizons
    header = f"{'Date':<12} {'Tardis':<10} {'Live':<10} {'Bars':<10}"
    for h in HORIZONS:
        header += f" {f'{h}min':<8}"
    
    print(header)
    print("-"*total_width)
    
    # Print data rows
    for r in results:
        row = f"{r['date']:<12} {r['prebars']['tardis']:<10} {r['prebars']['live']:<10} {r['bars']['total']:<10}"
        for h in HORIZONS:
            count = r['features'].get(f'horizon_{h}', 0)
            row += f" {count:<8}"
        print(row)
    
    # Print totals
    print("-"*total_width)
    total_prebars_tardis = sum(r['prebars']['tardis'] for r in results)
    total_prebars_live = sum(r['prebars']['live'] for r in results)
    total_bars = sum(r['bars']['total'] for r in results)
    
    row = f"{'TOTAL':<12} {total_prebars_tardis:<10} {total_prebars_live:<10} {total_bars:<10}"
    for h in HORIZONS:
        total_h = sum(r['features'].get(f'horizon_{h}', 0) for r in results)
        row += f" {total_h:<8}"
    print(row)
    print("="*total_width)
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print(f"  Date Range: {results[0]['date']} to {results[-1]['date']}")
    print(f"  Total Days: {len(results)}")
    print(f"  Total Tardis Prebars: {total_prebars_tardis:,}")
    print(f"  Total Live Prebars: {total_prebars_live:,}")
    print(f"  Total Bar Files (1440 min): {total_bars:,}")
    
    if results:
        # Show per-horizon breakdown for bars if we want detailed info
        # print("\n  Bar Files by Horizon:")
        # for h in HORIZONS:
        #     total = sum(r['bars'].get(f'horizon_{h}', 0) for r in results)
        #     if total > 0:
        #         print(f"    - {h:5} min: {total:,}")
        pass
        
        print("\n  Feature Files by Horizon:")
        for h in HORIZONS:
            total = sum(r['features'].get(f'horizon_{h}', 0) for r in results)
            if total > 0:
                print(f"    - {h:5} min: {total:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Count prebars, bars, and features files by date range. Shows feature counts for each horizon.'
    )
    parser.add_argument(
        '--from',
        dest='from_date',
        required=True,
        help='Start date (YYYYMMDD)'
    )
    parser.add_argument(
        '--to',
        dest='to_date',
        required=True,
        help='End date (YYYYMMDD)'
    )
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Output in CSV format'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.from_date, '%Y%m%d')
        end_date = datetime.strptime(args.to_date, '%Y%m%d')
    except ValueError:
        print("Error: Invalid date format. Use YYYYMMDD")
        sys.exit(1)
    
    if start_date > end_date:
        print("Error: Start date must be before or equal to end date")
        sys.exit(1)
    
    # Process each date
    results = []
    print(f"\nProcessing dates from {args.from_date} to {args.to_date}...")
    
    for date in daterange(start_date, end_date):
        date_str = date.strftime('%Y%m%d')
        
        # Count files for this date
        prebars_counts = count_prebars_for_date(date_str)
        bars_counts = count_bars_for_date(date_str)
        features_counts = count_features_for_date(date_str)
        
        results.append({
            'date': date_str,
            'prebars': prebars_counts,
            'bars': bars_counts,
            'features': features_counts
        })
        
        # Show progress
        if len(results) % 10 == 0:
            print(f"  Processed {len(results)} dates...")
    
    # Output results
    if args.csv:
        # CSV output with horizon breakdowns
        horizon_headers = []
        for h in HORIZONS:
            horizon_headers.append(f"Bars_{h}")
        for h in HORIZONS:
            horizon_headers.append(f"Features_{h}")
        
        print(f"Date,Prebars_Tardis,Prebars_Live,Prebars_Total,Bars_Total,Features_Total,"
              f"{','.join(horizon_headers)}")
        
        for r in results:
            bars_by_horizon = [str(r['bars'].get(f'horizon_{h}', 0)) for h in HORIZONS]
            features_by_horizon = [str(r['features'].get(f'horizon_{h}', 0)) for h in HORIZONS]
            
            print(f"{r['date']},{r['prebars']['tardis']},{r['prebars']['live']},"
                  f"{r['prebars']['total']},{r['bars']['total']},{r['features']['total']},"
                  f"{','.join(bars_by_horizon)},{','.join(features_by_horizon)}")
    else:
        # Table output
        print_summary_table(results)


if __name__ == '__main__':
    main()