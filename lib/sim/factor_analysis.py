import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import date

from lib.data import DataLoader

logger = logging.getLogger(__name__)


class FactorAnalyzer:
    """Analyzes factor exposures and P&L attribution for trading simulations"""
    
    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Initialize the FactorAnalyzer
        
        Args:
            data_loader: DataLoader instance for loading feature data
        """
        self.data_loader = data_loader or DataLoader()
        self.factor_data = None
        self.factor_exposures = None
        self.factor_pnl = None
        
    def analyze_factor(self, 
                      sim_df: pd.DataFrame,
                      factor_name: str,
                      horizon: int = 1440) -> Dict[str, pd.DataFrame]:
        """
        Analyze factor exposures and P&L for a simulation
        
        Args:
            sim_df: Simulation dataframe with columns [ts, symbol_venue, position, close_mid, ...]
            factor_name: Name of the feature to use as a factor
            horizon: Horizon for loading features (default 1440)
            
        Returns:
            Dictionary with:
                - 'exposures': DataFrame with factor exposures by symbol/time
                - 'pnl': DataFrame with aggregated factor P&L over time
                - 'stats': DataFrame with summary statistics
        """
        logger.info(f"Analyzing factor '{factor_name}' with horizon {horizon}")
        
        # Load factor data
        self._load_factor_data(sim_df, factor_name, horizon)
        
        if self.factor_data is None:
            raise ValueError(f"Failed to load factor data for '{factor_name}'")
        
        # Calculate factor exposures
        self._calculate_factor_exposures(sim_df, factor_name)
        
        # Calculate summary statistics
        stats = self._calculate_factor_stats()
        
        return {
            'exposures': self.factor_exposures,
            'pnl': self.factor_pnl,
            'stats': stats
        }
    
    def analyze_multiple_factors(self,
                               sim_df: pd.DataFrame,
                               factor_names: List[str],
                               horizon: int = 1440) -> Dict[str, Dict]:
        """
        Analyze multiple factors for a simulation
        
        Args:
            sim_df: Simulation dataframe
            factor_names: List of factor names to analyze
            horizon: Horizon for loading features
            
        Returns:
            Dictionary with factor names as keys and analysis results as values
        """
        results = {}
        
        for factor_name in factor_names:
            try:
                logger.info(f"Analyzing factor: {factor_name}")
                results[factor_name] = self.analyze_factor(sim_df, factor_name, horizon)
            except Exception as e:
                logger.error(f"Error analyzing factor '{factor_name}': {e}")
                results[factor_name] = {'error': str(e)}
        
        return results
    
    def _load_factor_data(self, sim_df: pd.DataFrame, factor_name: str, horizon: int):
        """Load feature data for the specified factor"""
        try:
            # Get date range from simulation data
            ts_values = pd.to_datetime(sim_df['ts'])
            portfolio_symbols = sim_df['symbol_venue'].unique().tolist()
            
            start_date = ts_values.min().date()
            end_date = ts_values.max().date()
            
            logger.info(f"Loading factor '{factor_name}' for ALL symbols from {start_date} to {end_date}")
            
            # Load features for ALL available symbols (not just portfolio)
            # This gives us universe-wide factor standardization
            all_features_df = self.data_loader.load_features(
                horizons=[horizon],
                start_date=start_date,
                end_date=end_date,
                symbol_venues=None,  # None means load all available symbols
                cols=[factor_name]
            )
            
            if all_features_df is None or all_features_df.empty:
                logger.warning("No feature data loaded")
                return
            
            if factor_name not in all_features_df.columns:
                available = [col for col in all_features_df.columns if not col.startswith('_')]
                logger.error(f"Factor '{factor_name}' not found. Available: {available[:10]}...")
                return
            
            # Store both full universe data and portfolio subset
            self.universe_factor_data = all_features_df[[factor_name]].copy()
            universe_symbols = all_features_df.index.get_level_values('symbol_venue').unique()
            logger.info(f"Loaded universe factor data with {len(self.universe_factor_data)} records for {len(universe_symbols)} symbols")
            
            # Extract portfolio subset for analysis
            self.factor_data = all_features_df.loc[all_features_df.index.get_level_values('symbol_venue').isin(portfolio_symbols)][[factor_name]].copy()
            logger.info(f"Extracted portfolio factor data with {len(self.factor_data)} records for {len(portfolio_symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error loading factor data: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_factor_exposures(self, sim_df: pd.DataFrame, factor_name: str):
        """Calculate factor exposures and P&L attribution"""
        try:
            logger.info("Calculating factor exposures...")
            
            # Reset indices for merging
            sim_data = sim_df.reset_index()
            factor_data = self.factor_data.reset_index()
            
            # Ensure timestamps have the same timezone
            if hasattr(sim_data['ts'].dtype, 'tz'):
                if sim_data['ts'].dt.tz is None:
                    sim_data['ts'] = pd.to_datetime(sim_data['ts']).dt.tz_localize('UTC')
            else:
                sim_data['ts'] = pd.to_datetime(sim_data['ts']).dt.tz_localize('UTC')
                
            if hasattr(factor_data['ts'].dtype, 'tz'):
                if factor_data['ts'].dt.tz is None:
                    factor_data['ts'] = pd.to_datetime(factor_data['ts']).dt.tz_localize('UTC')
            else:
                factor_data['ts'] = pd.to_datetime(factor_data['ts']).dt.tz_localize('UTC')
            
            # Merge on ts and symbol_venue
            merged_df = sim_data.merge(
                factor_data,
                on=['ts', 'symbol_venue'],
                how='left',
                suffixes=('_sim', '_factor')
            )
            
            logger.info(f"Merged data shape: {merged_df.shape}")
            
            # Handle column naming after merge
            factor_col_name = factor_name
            if f"{factor_name}_factor" in merged_df.columns:
                factor_col_name = f"{factor_name}_factor"
            elif factor_name not in merged_df.columns and f"{factor_name}_sim" in merged_df.columns:
                logger.warning(f"Factor '{factor_name}' not found in feature data, using simulation data")
                factor_col_name = f"{factor_name}_sim"
            
            # Handle missing factor values
            missing_factor = merged_df[factor_col_name].isna().sum()
            if missing_factor > 0:
                logger.warning(f"Missing factor values for {missing_factor} records ({100*missing_factor/len(merged_df):.1f}%)")
                merged_df[factor_col_name] = merged_df[factor_col_name].fillna(0)
            
            # Sort by symbol and time
            merged_df = merged_df.sort_values(['symbol_venue', 'ts'])
            
            # Calculate returns and P&L
            merged_df['prev_position'] = merged_df.groupby('symbol_venue')['position'].shift(1)
            merged_df['prev_close'] = merged_df.groupby('symbol_venue')['close_mid'].shift(1)
            merged_df['price_return'] = (merged_df['close_mid'] - merged_df['prev_close']) / merged_df['prev_close']
            merged_df['price_return'] = merged_df['price_return'].fillna(0)
            
            # Standard P&L = previous position * price change
            merged_df['standard_pnl'] = merged_df['prev_position'].fillna(0) * (merged_df['close_mid'] - merged_df['prev_close']).fillna(0)
            
            # Factor exposure = factor value * position (in dollars)
            merged_df['factor_exposure'] = merged_df[factor_col_name] * merged_df['position']
            
            # For factor attribution, we want to isolate the P&L from factor exposure
            # Factor P&L = standardized factor value * position * return
            # Standardize factor values using UNIVERSE data, not just portfolio
            if hasattr(self, 'universe_factor_data') and self.universe_factor_data is not None:
                logger.info("Using universe-wide factor standardization")
                # Calculate universe mean and std for each timestamp
                universe_stats = self.universe_factor_data.reset_index().groupby('ts')[factor_name].agg(['mean', 'std'])
                universe_stats.columns = ['universe_mean', 'universe_std']
                
                # Merge universe stats with portfolio data
                merged_df = merged_df.merge(universe_stats, left_on='ts', right_index=True, how='left')
                
                # Calculate z-score using universe statistics
                merged_df['factor_zscore'] = (merged_df[factor_col_name] - merged_df['universe_mean']) / (merged_df['universe_std'] + 1e-10)
                
                # Log some statistics about the portfolio vs universe
                portfolio_mean_zscore = merged_df.groupby('ts')['factor_zscore'].mean().mean()
                logger.info(f"Portfolio average z-score vs universe: {portfolio_mean_zscore:.4f}")
                
                # Store for later use in stats
                self.portfolio_mean_zscore = portfolio_mean_zscore
            else:
                logger.warning("No universe data available, using portfolio-only standardization")
                # Fall back to portfolio-only standardization
                factor_mean = merged_df.groupby('ts')[factor_col_name].transform('mean')
                factor_std = merged_df.groupby('ts')[factor_col_name].transform('std')
                merged_df['factor_zscore'] = (merged_df[factor_col_name] - factor_mean) / (factor_std + 1e-10)
            
            # Factor P&L = z-scored factor * previous position * return
            merged_df['factor_pnl'] = merged_df['factor_zscore'] * merged_df['prev_position'].fillna(0) * merged_df['price_return']
            merged_df['factor_pnl'] = merged_df['factor_pnl'].fillna(0)
            
            # Store exposures
            self.factor_exposures = merged_df[['ts', 'symbol_venue', factor_col_name, 
                                             'position', 'factor_exposure', 'factor_pnl']].copy()
            self.factor_exposures.rename(columns={factor_col_name: factor_name}, inplace=True)
            
            # Calculate net position for positive and negative factor values
            pos_factor_mask = self.factor_exposures[factor_name] > 0
            neg_factor_mask = self.factor_exposures[factor_name] < 0
            
            # Aggregate by timestamp
            agg_dict = {
                'factor_exposure': 'sum',
                'factor_pnl': 'sum',
                'position': lambda x: x.abs().sum()
            }
            
            self.factor_pnl = self.factor_exposures.groupby('ts').agg(agg_dict).reset_index()
            
            # Calculate net positions for positive and negative factor values
            pos_factor_positions = self.factor_exposures[pos_factor_mask].groupby('ts')['position'].sum()
            neg_factor_positions = self.factor_exposures[neg_factor_mask].groupby('ts')['position'].sum()
            
            # Merge the position sums
            self.factor_pnl = self.factor_pnl.merge(pos_factor_positions.rename('pos_factor_position'), 
                                                  on='ts', how='left')
            self.factor_pnl = self.factor_pnl.merge(neg_factor_positions.rename('neg_factor_position'), 
                                                  on='ts', how='left')
            
            # Fill NaN values with 0
            self.factor_pnl['pos_factor_position'] = self.factor_pnl['pos_factor_position'].fillna(0)
            self.factor_pnl['neg_factor_position'] = self.factor_pnl['neg_factor_position'].fillna(0)
            
            # Calculate net position (long - short)
            self.factor_pnl['net_position'] = self.factor_pnl['pos_factor_position'] + self.factor_pnl['neg_factor_position']
            
            # Calculate cumulative factor P&L
            self.factor_pnl['cumulative_factor_pnl'] = self.factor_pnl['factor_pnl'].cumsum()
            
            # Calculate factor exposure as % of gross position
            self.factor_pnl['factor_exposure_pct'] = (
                self.factor_pnl['factor_exposure'].abs() / 
                (self.factor_pnl['position'] + 1e-10) * 100
            )
            
            # Calculate net exposure as % of gross position (based on net positions)
            self.factor_pnl['net_exposure_pct'] = (
                self.factor_pnl['net_position'] / 
                (self.factor_pnl['position'] + 1e-10) * 100
            )
            
            logger.info(f"Calculated factor exposures for {len(self.factor_pnl)} timestamps")
            
        except Exception as e:
            logger.error(f"Error calculating factor exposures: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_factor_stats(self) -> pd.DataFrame:
        """Calculate summary statistics for the factor analysis"""
        if self.factor_pnl is None or self.factor_pnl.empty:
            return pd.DataFrame()
        
        # Calculate daily statistics
        daily_pnl = self.factor_pnl.copy()
        daily_pnl['date'] = pd.to_datetime(daily_pnl['ts']).dt.date
        daily_stats = daily_pnl.groupby('date')['factor_pnl'].sum()
        
        # Calculate metrics
        total_pnl = self.factor_pnl['cumulative_factor_pnl'].iloc[-1]
        avg_daily_pnl = daily_stats.mean()
        std_daily_pnl = daily_stats.std()
        
        # Annualized metrics (assuming 365 trading days)
        annualized_return = avg_daily_pnl * 365
        annualized_volatility = std_daily_pnl * np.sqrt(365)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Exposure metrics
        avg_exposure = self.factor_pnl['factor_exposure'].abs().mean()
        max_exposure = self.factor_pnl['factor_exposure'].abs().max()
        avg_exposure_pct = self.factor_pnl['factor_exposure_pct'].mean()
        avg_net_exposure_pct = self.factor_pnl['net_exposure_pct'].mean()
        
        # Calculate portfolio factor bias if we have universe data
        avg_portfolio_factor_zscore = 0.0
        if hasattr(self, 'portfolio_mean_zscore'):
            avg_portfolio_factor_zscore = self.portfolio_mean_zscore
        
        # Create stats dataframe
        stats = pd.DataFrame({
            'metric': [
                'total_pnl',
                'avg_daily_pnl',
                'std_daily_pnl',
                'annualized_return',
                'annualized_volatility',
                'sharpe_ratio',
                'avg_exposure',
                'max_exposure',
                'avg_exposure_pct',
                'avg_net_exposure_pct',
                'avg_portfolio_factor_zscore'
            ],
            'value': [
                total_pnl,
                avg_daily_pnl,
                std_daily_pnl,
                annualized_return,
                annualized_volatility,
                sharpe_ratio,
                avg_exposure,
                max_exposure,
                avg_exposure_pct,
                avg_net_exposure_pct,
                avg_portfolio_factor_zscore
            ]
        })
        
        return stats