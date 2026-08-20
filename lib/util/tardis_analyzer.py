"""Module to analyze tardis prebar file date ranges and missing dates for universe symbols."""

import glob
import logging
import os
from datetime import datetime, timedelta, date
from typing import List, Tuple, Optional, Dict

import pandas as pd

from lib.data.dataloader import DataLoader
from lib.data.loaders import load_metadata
from lib.universe import Universe
from lib.util.directory import DirectoryManager
from lib.util.files import find_latest_universe_date
from lib.util.time_util import date_to_str

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TardisDataAnalyzer:
    """Analyzes tardis prebar files for date coverage and missing dates."""
    
    def __init__(self, config: dict, universe_date: Optional[datetime.date] = None):
        """
        Initialize the analyzer.
        
        Args:
            config: Trading system configuration dict
            universe_date: Date to load universe for (uses latest if None)
        """
        self.config = config
        
        # Get DirectoryManager instance
        self.dir_manager = DirectoryManager()
        
        # Use DirectoryManager for paths
        self.universe_date = universe_date if universe_date is not None else find_latest_universe_date(self.dir_manager.UNIVERSE_DIR)
        self.tardis_dir = self.dir_manager.TARDIS_PREBAR_DIR
        self.universe = Universe(config=config)
        
        # Load binance metadata for symbol availability dates
        self.symbol_metadata = self._load_symbol_metadata()
        
    def _load_symbol_metadata(self) -> Dict[str, dict]:
        """
        Load binance metadata to get availability dates for symbols.
        
        Returns:
            Dictionary mapping symbol to metadata dict with availableSince and availableTo dates
        """
        try:
            # Use DataLoader to load the latest binance metadata
            data_loader = DataLoader(config=self.config)
            metadata_df = load_metadata(latest=True, metadata_dir=data_loader.dir_manager.BINANCE_META_DIR)
            
            if metadata_df is None:
                logger.warning("No binance metadata found")
                return {}
            
            # Convert to dictionary for easy lookup
            # Group by symbol (remove venue suffix for matching)
            symbol_meta = {}
            for symbol_venue, row in metadata_df.iterrows():
                if isinstance(symbol_venue, tuple):
                    symbol_venue = symbol_venue[1]  # Extract from tuple if MultiIndex
                
                # Extract just the symbol part (e.g., BTCUSDT from BTCUSDT_binance-futures)
                symbol = symbol_venue.split('_')[0] if '_' in symbol_venue else symbol_venue
                
                if symbol not in symbol_meta:
                    # Handle NaT (Not a Time) values
                    available_since = row['availableSince']
                    available_to = row['availableTo']
                    
                    # Convert pandas NaT to None for easier handling
                    if pd.isna(available_since):
                        available_since = None
                    if pd.isna(available_to):
                        available_to = None
                        
                    symbol_meta[symbol] = {
                        'availableSince': available_since,
                        'availableTo': available_to
                    }
            
            logger.info(f"Loaded metadata for {len(symbol_meta)} symbols")
            return symbol_meta
            
        except Exception as e:
            logger.warning(f"Failed to load binance metadata: {e}")
            return {}
    
    def load_universe_symbols(self) -> List[str]:
        """
        Load universe symbols using Universe class.
        
        Returns:
            List of symbol pairs (e.g., ['BTCUSDT', 'ETHUSDT', ...])
        """
        symbols = self.universe.load_universe_symbols(
            universe_source='file',
            universe_date=self.universe_date,
            symbol_type='pair'  # Get pairs like BTCUSDT - use 'pair' not 'symbol_pair'
        )
        return symbols if symbols else []
        
    def get_symbol_date_range(self, symbol: str) -> Tuple[Optional[datetime.date], Optional[datetime.date], List[datetime.date]]:
        """
        Get start date, end date, and missing dates for a symbol.
        
        Args:
            symbol: Symbol pair (e.g., 'BTCUSDT')
            
        Returns:
            Tuple of (start_date, end_date, missing_dates)
        """
        # Get metadata for this symbol to know availability range
        symbol_meta = self.symbol_metadata.get(symbol, {})
        available_since = symbol_meta.get('availableSince')
        available_to = symbol_meta.get('availableTo')
        
        # Find all files for this symbol
        pattern = os.path.join(self.tardis_dir, "*", "binance-futures", f"prebars.tardis.binance-futures.*.{symbol}.parquet")
        files = glob.glob(pattern)
        
        if not files:
            return None, None, []
            
        # Extract dates from filenames
        dates = []
        for filepath in files:
            # Extract date from filename format: prebars.tardis.binance-futures.YYYYMMDD.SYMBOL.parquet
            parts = os.path.basename(filepath).split('.')
            if len(parts) >= 5:
                date_str = parts[3]  # YYYYMMDD format
                try:
                    date = datetime.strptime(date_str, "%Y%m%d").date()
                    dates.append(date)
                except ValueError:
                    logger.warning(f"Could not parse date from filename: {filepath}")
                    
        if not dates:
            return None, None, []
            
        dates.sort()
        start_date = dates[0]
        end_date = dates[-1]
        
        # Determine the expected date range based on availability
        if available_since:
            # Use the later of: actual start date or availability start date
            expected_start = max(start_date, available_since)
        else:
            expected_start = start_date
            
        if available_to:
            # Use the earlier of: actual end date or availability end date
            expected_end = min(end_date, available_to)
        else:
            expected_end = end_date
        
        # Find missing dates between expected start and end
        missing_dates = []
        current_date = expected_start
        date_set = set(dates)
        
        while current_date <= expected_end:
            if current_date not in date_set:
                # Only count as missing if it's within the symbol's availability period
                if available_since and current_date < available_since:
                    pass  # Skip dates before symbol was available
                elif available_to and current_date > available_to:
                    pass  # Skip dates after symbol was delisted
                else:
                    missing_dates.append(current_date)
            current_date += timedelta(days=1)
            
        return start_date, end_date, missing_dates
        
    def find_symbol_coverage(self) -> List[dict]:
        """
        Find coverage information for all universe symbols.
        
        Returns:
            List of dicts with keys: symbol, start_date, end_date, missing_count, missing_dates, total_days, coverage_pct
        """
        symbols = self.load_universe_symbols()
        
        if not symbols:
            logger.warning("No symbols loaded from universe")
            return []
            
        results = []
        
        print(f"Processing {len(symbols)} symbols...")
        for i, symbol in enumerate(symbols, 1):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(symbols)} symbols...")
            start_date, end_date, missing_dates = self.get_symbol_date_range(symbol)
            
            if start_date and end_date:
                # Get metadata for proper coverage calculation
                symbol_meta = self.symbol_metadata.get(symbol, {})
                available_since = symbol_meta.get('availableSince')
                available_to = symbol_meta.get('availableTo')
                
                # Calculate expected range for coverage
                expected_start = max(start_date, available_since) if available_since else start_date
                expected_end = min(end_date, available_to) if available_to else end_date
                
                # Calculate total expected days (within availability period)
                total_expected_days = (expected_end - expected_start).days + 1
                missing_count = len(missing_dates)
                
                # Calculate coverage based on expected days, not total days
                coverage_pct = ((total_expected_days - missing_count) / total_expected_days * 100) if total_expected_days > 0 else 0
                
                results.append({
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date,
                    'available_since': available_since,
                    'available_to': available_to,
                    'total_days': total_expected_days,
                    'missing_count': missing_count,
                    'coverage_pct': round(coverage_pct, 2),
                    'missing_dates': missing_dates[:10] if missing_dates else []  # Show first 10 missing dates
                })
            else:
                results.append({
                    'symbol': symbol,
                    'start_date': None,
                    'end_date': None,
                    'total_days': 0,
                    'missing_count': 0,
                    'coverage_pct': 0.0,
                    'missing_dates': []
                })
        
        # Sort by coverage percentage (ascending to show problematic symbols first)
        results.sort(key=lambda x: x['coverage_pct'])
        
        return results
        
    def print_summary(self, coverage_data: List[dict]) -> None:
        """
        Print a summary of the analysis.
        
        Args:
            coverage_data: List of dicts from find_symbol_coverage()
        """
        if not coverage_data:
            print("No data to summarize")
            return
            
        print("\n" + "="*80)
        print("TARDIS PREBAR DATA COVERAGE ANALYSIS")
        print("="*80)
        
        # Overall statistics
        total_symbols = len(coverage_data)
        symbols_with_data = len([s for s in coverage_data if s['start_date'] is not None])
        symbols_without_data = total_symbols - symbols_with_data
        
        print(f"\nTotal symbols in universe: {total_symbols}")
        print(f"Symbols with data: {symbols_with_data}")
        print(f"Symbols without data: {symbols_without_data}")
        
        if symbols_with_data > 0:
            symbols_with_coverage = [s for s in coverage_data if s['coverage_pct'] > 0]
            if symbols_with_coverage:
                avg_coverage = sum(s['coverage_pct'] for s in symbols_with_coverage) / len(symbols_with_coverage)
                print(f"Average coverage: {avg_coverage:.2f}%")
            
            # Date range statistics
            valid_dates = [s for s in coverage_data if s['start_date'] is not None]
            if valid_dates:
                earliest_start = min(s['start_date'] for s in valid_dates)
                latest_end = max(s['end_date'] for s in valid_dates)
                print(f"\nData range: {earliest_start} to {latest_end}")
            
            # Show symbols with poor coverage
            poor_coverage = [s for s in coverage_data if 0 < s['coverage_pct'] < 90]
            if poor_coverage:
                print(f"\nSymbols with <90% coverage ({len(poor_coverage)} symbols):")
                print("-" * 40)
                for symbol_data in poor_coverage[:10]:
                    print(f"{symbol_data['symbol']:12} {symbol_data['coverage_pct']:6.2f}% ({symbol_data['missing_count']} missing days)")
                    if symbol_data['missing_dates']:
                        missing_str = ', '.join([date_to_str(d) for d in symbol_data['missing_dates'][:5]])
                        if len(symbol_data['missing_dates']) > 5:
                            missing_str += f", ... ({len(symbol_data['missing_dates'])-5} more)"
                        print(f"             Missing: {missing_str}")
                        
            # Show symbols with no data
            if symbols_without_data > 0:
                no_data_symbols = [s['symbol'] for s in coverage_data if s['start_date'] is None]
                print(f"\nSymbols with no data:")
                print("-" * 40)
                for i in range(0, min(len(no_data_symbols), 20), 5):
                    print(", ".join(no_data_symbols[i:i+5]))
                if len(no_data_symbols) > 20:
                    print(f"... and {len(no_data_symbols)-20} more")
        
        print("\n" + "="*80)