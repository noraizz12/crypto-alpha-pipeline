#!/usr/bin/env python3
"""
Analyze factor exposures and P&L attribution for trading simulations

Usage:
    python analyze_factors.py <sim_name> --factors beta_1440,logret_1440_trmean --horizon 1440
"""

import argparse
import logging.config
import os
import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.sim.factor_analysis import FactorAnalyzer
from lib.data import DataLoader
from lib.util.directory import SIM_DIR
from lib.util.logging_util import get_logging_config
from lib.alpha.features import get_available_features_for_horizons

# Setup logging
logging.config.dictConfig(get_logging_config("analyze_factors"))
logger = logging.getLogger(__name__)


def load_simulation_data(sim_name: str) -> pd.DataFrame:
    """Load simulation data from parquet files"""
    sim_path = Path(SIM_DIR) / sim_name
    
    if not sim_path.exists():
        raise ValueError(f"Simulation directory not found: {sim_path}")
    
    # Find all sim.*.parquet files
    sim_files = sorted(sim_path.glob("sim.*.parquet"))
    
    if not sim_files:
        raise ValueError(f"No simulation files found in {sim_path}")
    
    logger.info(f"Found {len(sim_files)} simulation files")
    
    # Load and concatenate all files
    dfs = []
    for file in sim_files:
        logger.info(f"Loading {file.name}")
        df = pd.read_parquet(file)
        dfs.append(df)
    
    sim_df = pd.concat(dfs)
    logger.info(f"Loaded {len(sim_df)} records from simulation")
    
    # Reset index to ensure ts and symbol_venue are columns
    sim_df = sim_df.reset_index()
    
    return sim_df




def calculate_portfolio_metrics(sim_df: pd.DataFrame) -> dict:
    """Calculate portfolio-level metrics including net exposure"""
    # Calculate gross and net portfolio notional
    sim_df['gross_notional'] = sim_df['position'].abs()
    sim_df['net_notional'] = sim_df['position']
    
    # Group by timestamp to get portfolio-level metrics
    portfolio_metrics = sim_df.groupby('ts').agg({
        'gross_notional': 'sum',
        'net_notional': 'sum'
    }).reset_index()
    
    # Calculate net exposure as percentage of gross
    portfolio_metrics['net_exposure_pct'] = (
        portfolio_metrics['net_notional'].abs() / 
        (portfolio_metrics['gross_notional'] + 1e-10) * 100
    )
    
    # Calculate average metrics
    avg_gross_notional = portfolio_metrics['gross_notional'].mean()
    avg_net_exposure_pct = portfolio_metrics['net_exposure_pct'].mean()
    
    return {
        'avg_gross_notional': avg_gross_notional,
        'avg_net_exposure_pct': avg_net_exposure_pct
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze factor exposures and P&L for trading simulations')
    parser.add_argument('sim_name', help='Name of the simulation directory under SIM_DIR')
    parser.add_argument('--factors', required=True, help='Comma-separated list of factors to analyze, or "all" to analyze all available features for the horizon')
    parser.add_argument('--horizon', type=int, default=1440, help='Horizon for loading features (default: 1440)')
    parser.add_argument('--output', help='Output CSV file for results (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Parse factors
    if args.factors.lower() == 'all':
        # Get all available features for the specified horizon
        logger.info(f"Getting all available features for horizon {args.horizon}")
        factors = get_available_features_for_horizons([args.horizon])
        if not factors:
            logger.error(f"No features found for horizon {args.horizon}")
            sys.exit(1)
        logger.info(f"Found {len(factors)} features for horizon {args.horizon}")
    else:
        factors = [f.strip() for f in args.factors.split(',')]
    
    logger.info(f"Analyzing {len(factors)} factors: {', '.join(factors[:5])}{'...' if len(factors) > 5 else ''}")
    
    try:
        # Load simulation data
        logger.info(f"Loading simulation: {args.sim_name}")
        sim_df = load_simulation_data(args.sim_name)
        
        # Calculate portfolio metrics
        portfolio_metrics = calculate_portfolio_metrics(sim_df)
        
        # Initialize analyzer
        data_loader = DataLoader()
        analyzer = FactorAnalyzer(data_loader)
        
        # Analyze each factor
        results = analyzer.analyze_multiple_factors(sim_df, factors, args.horizon)
        
        # Collect results
        all_stats = []
        for factor_name, result in results.items():
            if 'error' in result:
                logger.error(f"Error analyzing {factor_name}: {result['error']}")
                continue
            
            stats = result['stats']
            stats['factor'] = factor_name
            all_stats.append(stats)
        
        # Save to CSV if requested
        if args.output and all_stats:
            # Combine all stats
            combined_stats = pd.concat(all_stats, ignore_index=True)
            
            # Pivot to have factors as columns
            pivoted = combined_stats.pivot(index='metric', columns='factor', values='value')
            pivoted.to_csv(args.output)
            logger.info(f"Results saved to {args.output}")
        
        # Print summary table
        if all_stats:
            print(f"\n{'='*150}")
            print("FACTOR ANALYSIS SUMMARY (SORTED BY SHARPE RATIO)")
            print(f"{'='*150}")
            print(f"{'Rank':<6} {'Factor':<35} {'Total P&L':>15} {'Ann. Return %':>15} {'Ann. Vol %':>15} {'Sharpe':>10} {'Net Exp %':>15} {'Port Bias':>12}")
            print(f"{'-'*150}")
            
            # Create summary list for sorting
            summary_data = []
            for stats in all_stats:
                factor = stats['factor'].iloc[0]
                total_pnl = stats[stats['metric'] == 'total_pnl']['value'].iloc[0]
                ann_ret = stats[stats['metric'] == 'annualized_return']['value'].iloc[0]
                ann_vol = stats[stats['metric'] == 'annualized_volatility']['value'].iloc[0]
                sharpe = stats[stats['metric'] == 'sharpe_ratio']['value'].iloc[0]
                
                # Get average net exposure percentage
                avg_net_exposure_pct = 0.0
                net_exposure_row = stats[stats['metric'] == 'avg_net_exposure_pct']
                if not net_exposure_row.empty:
                    avg_net_exposure_pct = net_exposure_row['value'].iloc[0]
                
                # Get portfolio factor bias (z-score)
                portfolio_factor_bias = 0.0
                bias_row = stats[stats['metric'] == 'avg_portfolio_factor_zscore']
                if not bias_row.empty:
                    portfolio_factor_bias = bias_row['value'].iloc[0]
                
                # Annualized metrics are in dollar terms, convert to percentage of average portfolio
                # Only convert if we have a valid gross notional
                if portfolio_metrics['avg_gross_notional'] > 0:
                    ann_ret_pct = (ann_ret / portfolio_metrics['avg_gross_notional']) * 100
                    ann_vol_pct = (ann_vol / portfolio_metrics['avg_gross_notional']) * 100
                else:
                    ann_ret_pct = 0.0
                    ann_vol_pct = 0.0
                
                summary_data.append({
                    'factor': factor,
                    'total_pnl': total_pnl,
                    'ann_ret_pct': ann_ret_pct,
                    'ann_vol_pct': ann_vol_pct,
                    'sharpe': sharpe,
                    'avg_net_exposure_pct': avg_net_exposure_pct,
                    'portfolio_factor_bias': portfolio_factor_bias
                })
            
            # Sort by Sharpe ratio (descending)
            summary_data.sort(key=lambda x: x['sharpe'], reverse=True)
            
            # Print sorted results
            for i, data in enumerate(summary_data, 1):
                print(f"{i:<6} {data['factor']:<35} ${data['total_pnl']:>13,.0f} {data['ann_ret_pct']:>14.2f}% {data['ann_vol_pct']:>13.2f}% {data['sharpe']:>9.3f} {data['avg_net_exposure_pct']:>14.2f}% {data['portfolio_factor_bias']:>11.3f}")
            
            print(f"{'-'*150}")
            print(f"Total factors analyzed: {len(summary_data)}")
            print(f"Portfolio Average Net Exposure: {portfolio_metrics['avg_net_exposure_pct']:.2f}% of Gross Portfolio Notional")
            print("\nPort Bias: Portfolio's average z-score for the factor (vs universe). 0=neutral, >0=long bias, <0=short bias")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()