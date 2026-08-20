#!/usr/bin/env python3
"""
Interactive Simulation Comparison Dashboard V2 - Fixed PnL scale and added notional charts

Usage:
    python sim_comparison_dash_v2.py --sims breakdown2 --breakdown
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional
from datetime import datetime
import logging

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
import cvxpy as cp
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.util.directory import SIM_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimulationDashboard:
    """Interactive dashboard for simulation analysis."""
    
    def __init__(self, sim_names: List[str], sim_dir: str = SIM_DIR, include_breakdown: bool = False):
        self.sim_names = sim_names
        self.sim_dir = sim_dir
        self.include_breakdown = include_breakdown
        self.breakdown_data = {}
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        
    def analyze_breakdown_data(self, sim_name: str) -> Dict:
        """Analyze breakdown2 style simulation with model/horizon splits."""
        sim_path = os.path.join(self.sim_dir, sim_name)
        breakdown_data = {}
        
        # Look for model/horizon specific calculator files
        import glob
        import re
        model_files = glob.glob(os.path.join(sim_path, 'pnl.model_*.horizon_*.calculator.csv'))
        
        for model_file in model_files:
            # Extract model and horizon from filename using regex
            basename = os.path.basename(model_file)
            match = re.search(r'pnl\.model_(\w+)\.horizon_(\d+)\.calculator\.csv', basename)
            
            if match:
                model_name = match.group(1)
                horizon = match.group(2)
                key = f"{model_name}_{horizon}"
                
                try:
                    df = pd.read_csv(model_file, index_col=0, parse_dates=['ts'])
                    
                    # Calculate metrics
                    if 'pnl' in df.columns and len(df) > 0:
                        # Get daily PnL by taking the difference in cumulative PnL
                        # The PnL column in calculator files is cumulative
                        df['date'] = df['ts'].dt.date
                        daily_last_pnl = df.groupby('date')['pnl'].last()
                        daily_first_pnl = df.groupby('date')['pnl'].first()
                        daily_pnl = daily_last_pnl - daily_first_pnl
                        
                        if len(daily_pnl) > 1:
                            # Calculate metrics directly
                            daily_returns = daily_pnl.values
                            sharpe = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
                            
                            # Calculate max drawdown
                            cumsum = np.cumsum(daily_returns)
                            running_max = np.maximum.accumulate(cumsum)
                            drawdown = cumsum - running_max
                            max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
                            max_dd_perc = max_dd / running_max[np.argmin(drawdown)] if running_max[np.argmin(drawdown)] != 0 else 0
                            
                            # Calculate cumulative PnL over time
                            # Since PnL in the file is already cumulative, just resample to daily
                            cumulative_pnl = df.set_index('ts')['pnl'].resample('D').last()
                            
                            # Calculate gross notional and turnover
                            df['gross_notional'] = df['long'] + df['short'].abs()
                            df['turnover'] = df['traded_long'] + df['traded_short'].abs()
                            daily_turnover = df.groupby(df['ts'].dt.date)['turnover'].sum()
                            daily_notional = df.groupby(df['ts'].dt.date)['gross_notional'].mean()
                            
                            # Calculate turnover ratio (turnover / gross notional)
                            daily_turnover_ratio = daily_turnover / daily_notional
                            avg_turnover_ratio = daily_turnover_ratio.mean()
                            
                            # Calculate notional time series
                            notional_ts = df.set_index('ts')['gross_notional'].resample('D').mean()
                            
                            breakdown_data[key] = {
                                'model': model_name,
                                'horizon': int(horizon),
                                'sharpe': sharpe,
                                'total_pnl': daily_pnl.sum(),
                                'daily_avg': daily_pnl.mean(),
                                'volatility': daily_pnl.std(),
                                'win_rate': (daily_pnl > 0).sum() / len(daily_pnl) if len(daily_pnl) > 0 else 0,
                                'max_drawdown': max_dd_perc,
                                'avg_daily_turnover': daily_turnover.mean(),
                                'avg_gross_notional': daily_notional.mean(),
                                'avg_turnover_ratio': avg_turnover_ratio,
                                'daily_pnl': daily_pnl,
                                'cumulative_pnl': cumulative_pnl,
                                'notional_ts': notional_ts
                            }
                        else:
                            logger.warning(f"Insufficient data for {key}: only {len(daily_pnl)} days")
                except Exception as e:
                    logger.error(f"Failed to process {model_file}: {e}")
        
        return breakdown_data
    
    def load_all_data(self):
        """Load data for all simulations."""
        for sim_name in self.sim_names:
            logger.info(f"Loading simulation: {sim_name}")
            if self.include_breakdown and 'breakdown' in sim_name.lower():
                self.breakdown_data[sim_name] = self.analyze_breakdown_data(sim_name)
    
    def create_model_performance_heatmap(self) -> go.Figure:
        """Create heatmap of model performance across horizons."""
        if not self.breakdown_data:
            return go.Figure()
        
        # Organize data for heatmap
        models = set()
        horizons = set()
        
        for sim_data in self.breakdown_data.values():
            for key, data in sim_data.items():
                models.add(data['model'])
                horizons.add(data['horizon'])
        
        models = sorted(list(models))
        horizons = sorted(list(horizons))
        
        # Create heatmap data
        z_sharpe = []
        z_pnl = []
        
        for model in models:
            sharpe_row = []
            pnl_row = []
            for horizon in horizons:
                # Average across all simulations
                sharpes = []
                pnls = []
                
                for sim_data in self.breakdown_data.values():
                    key = f"{model}_{horizon}"
                    if key in sim_data:
                        sharpes.append(sim_data[key]['sharpe'])
                        pnls.append(sim_data[key]['total_pnl'])
                
                sharpe_row.append(np.mean(sharpes) if sharpes else 0)
                pnl_row.append(np.mean(pnls) if pnls else 0)
            
            z_sharpe.append(sharpe_row)
            z_pnl.append(pnl_row)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Sharpe Ratio by Model/Horizon', 'Total PnL by Model/Horizon'),
            horizontal_spacing=0.15
        )
        
        # Sharpe heatmap
        fig.add_trace(
            go.Heatmap(
                z=z_sharpe,
                x=[f"{h}min" for h in horizons],
                y=models,
                colorscale='RdBu',
                zmid=0,
                text=[[f"{val:.2f}" for val in row] for row in z_sharpe],
                texttemplate="%{text}",
                textfont={"size": 10},
                showscale=True,
                colorbar=dict(x=0.45, title="Sharpe")
            ),
            row=1, col=1
        )
        
        # PnL heatmap (show in thousands for better readability)
        fig.add_trace(
            go.Heatmap(
                z=[[val/1000 for val in row] for row in z_pnl],
                x=[f"{h}min" for h in horizons],
                y=models,
                colorscale='RdBu',
                zmid=0,
                text=[[f"${val/1000:.0f}K" for val in row] for row in z_pnl],
                texttemplate="%{text}",
                textfont={"size": 10},
                showscale=True,
                colorbar=dict(title="PnL ($K)")
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Model Performance Analysis by Horizon',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_model_horizon_cumulative_pnl_chart(self) -> go.Figure:
        """Create cumulative PnL chart for each model_horizon combination."""
        fig = go.Figure()
        
        if not self.breakdown_data:
            return fig
        
        # Plot cumulative PnL for each model_horizon
        colors = px.colors.qualitative.Plotly
        color_idx = 0
        
        for sim_name, breakdown_data in self.breakdown_data.items():
            for key, data in sorted(breakdown_data.items()):
                if 'cumulative_pnl' in data and len(data['cumulative_pnl']) > 0:
                    fig.add_trace(go.Scatter(
                        x=data['cumulative_pnl'].index,
                        y=data['cumulative_pnl'].values / 1000,  # Convert to thousands
                        mode='lines',
                        name=key,
                        line=dict(width=2, color=colors[color_idx % len(colors)]),
                        legendgroup=key,
                        showlegend=True
                    ))
                    color_idx += 1
        
        fig.update_layout(
            title='Cumulative PnL by Model-Horizon Combination',
            xaxis_title='Date',
            yaxis_title='Cumulative PnL ($K)',
            hovermode='x unified',
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            )
        )
        
        return fig
    
    def create_model_correlation_clustered(self) -> go.Figure:
        """Create hierarchically clustered correlation matrix with dendrogram."""
        if not self.breakdown_data:
            return go.Figure()
        
        # Collect daily PnL series for each model/horizon
        pnl_series = {}
        for sim_name, breakdown_data in self.breakdown_data.items():
            for key, data in breakdown_data.items():
                if 'daily_pnl' in data and len(data['daily_pnl']) > 0:
                    pnl_series[key] = data['daily_pnl']
        
        if len(pnl_series) < 2:
            return go.Figure()
        
        # Create DataFrame with all series aligned by date
        pnl_df = pd.DataFrame(pnl_series)
        
        # Calculate correlation matrix
        corr_matrix = pnl_df.corr()
        
        # Convert correlation to distance for clustering
        # Distance = 1 - correlation (so perfect correlation = 0 distance)
        distance_matrix = 1 - corr_matrix.abs()
        
        # Perform hierarchical clustering
        condensed_distances = squareform(distance_matrix)
        linkage = hierarchy.linkage(condensed_distances, method='average')
        
        # Create dendrogram
        dendro = hierarchy.dendrogram(linkage, labels=corr_matrix.columns.tolist(), no_plot=True)
        
        # Reorder correlation matrix based on dendrogram
        cluster_order = dendro['leaves']
        ordered_labels = [corr_matrix.columns[i] for i in cluster_order]
        corr_matrix_ordered = corr_matrix.iloc[cluster_order, cluster_order]
        
        # Create figure with dendrogram and heatmap
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.2, 0.8],
            vertical_spacing=0.02,
            subplot_titles=('Hierarchical Clustering Dendrogram', 'Correlation Matrix (Clustered)')
        )
        
        # Add dendrogram
        dendro_data = ff.create_dendrogram(
            distance_matrix.values,
            labels=corr_matrix.columns.tolist(),
            orientation='bottom'
        )
        
        for trace in dendro_data['data']:
            fig.add_trace(trace, row=1, col=1)
        
        # Add heatmap
        heatmap = go.Heatmap(
            z=corr_matrix_ordered.values,
            x=ordered_labels,
            y=ordered_labels,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=[[f"{val:.2f}" for val in row] for row in corr_matrix_ordered.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation", y=0.3, len=0.6)
        )
        
        fig.add_trace(heatmap, row=2, col=1)
        
        fig.update_layout(
            title='Model-Horizon Correlation Analysis (Hierarchically Clustered)',
            height=800,
            showlegend=False
        )
        
        fig.update_xaxes(showgrid=False, row=1, col=1)
        fig.update_yaxes(showgrid=False, row=1, col=1)
        fig.update_xaxes(showgrid=False, row=2, col=1)
        fig.update_yaxes(showgrid=False, row=2, col=1)
        
        return fig
    
    def create_model_correlation_matrix(self) -> go.Figure:
        """Create correlation matrix of daily returns across model/horizon combinations."""
        if not self.breakdown_data:
            return go.Figure()
        
        # Collect daily PnL series for each model/horizon
        pnl_series = {}
        for sim_name, breakdown_data in self.breakdown_data.items():
            for key, data in breakdown_data.items():
                if 'daily_pnl' in data and len(data['daily_pnl']) > 0:
                    pnl_series[key] = data['daily_pnl']
        
        if len(pnl_series) < 2:
            return go.Figure()
        
        # Create DataFrame with all series aligned by date
        pnl_df = pd.DataFrame(pnl_series)
        
        # Calculate correlation matrix
        corr_matrix = pnl_df.corr()
        
        # Sort by average correlation to group similar strategies
        # Calculate mean correlation for each strategy (excluding self-correlation)
        avg_corr = {}
        for col in corr_matrix.columns:
            # Get correlations excluding the diagonal (self-correlation)
            other_corrs = corr_matrix[col].drop(col)
            avg_corr[col] = other_corrs.mean()
        
        # Sort columns/rows by average correlation
        sorted_cols = sorted(corr_matrix.columns, key=lambda x: avg_corr[x], reverse=True)
        corr_matrix = corr_matrix.loc[sorted_cols, sorted_cols]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=[[f"{val:.2f}" for val in row] for row in corr_matrix.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Model-Horizon Daily Return Correlations (Sorted by Average Correlation)',
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'},
            height=600,
            width=800
        )
        
        # Add diagonal line to make it clear these are 1.0
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        
        return fig
    
    def calculate_optimal_portfolio(self) -> Dict:
        """Calculate optimal portfolio weights to maximize Sharpe ratio."""
        if not self.breakdown_data:
            return {}
        
        # Collect daily PnL series for each model/horizon
        pnl_series = {}
        strategy_names = []
        for sim_name, breakdown_data in self.breakdown_data.items():
            for key, data in breakdown_data.items():
                if 'daily_pnl' in data and len(data['daily_pnl']) > 0:
                    pnl_series[key] = data['daily_pnl']
                    strategy_names.append(key)
        
        if len(pnl_series) < 2:
            return {}
        
        # Create DataFrame with all series aligned by date
        pnl_df = pd.DataFrame(pnl_series)
        
        # Remove any rows with NaN values
        pnl_df = pnl_df.dropna()
        
        # Need to load the raw data to calculate returns from PnL and gross notional
        returns_series = {}
        all_strategy_names = []
        
        for sim_name, breakdown_data in self.breakdown_data.items():
            for key, data in breakdown_data.items():
                # Load the raw calculator data for this model/horizon
                model_name, horizon = key.split('_')
                calc_file = os.path.join(self.sim_dir, sim_name, f'pnl.model_{model_name}.horizon_{horizon}.calculator.csv')
                
                if os.path.exists(calc_file):
                    df = pd.read_csv(calc_file, parse_dates=['ts'])
                    
                    # Calculate gross notional = abs(long) + abs(short)
                    if 'long' in df.columns and 'short' in df.columns:
                        df['gross_notional'] = df['long'].abs() + df['short'].abs()
                        
                        # First, calculate the daily PnL from cumulative if needed
                        # Sort by timestamp to ensure proper ordering
                        df = df.sort_values('ts')
                        
                        # Check if PnL appears to be cumulative (each value >= previous)
                        if 'pnl' in df.columns and len(df) > 1:
                            pnl_diff = df['pnl'].diff()
                            # If most differences are positive and PnL is monotonically increasing, it's likely cumulative
                            if (pnl_diff > 0).sum() > len(pnl_diff) * 0.9:
                                df['daily_pnl'] = df['pnl'].diff().fillna(df['pnl'].iloc[0])
                            else:
                                df['daily_pnl'] = df['pnl']
                        
                        # Group by date to get daily aggregates
                        # For gross_notional, we want the end-of-day value
                        daily_data = df.groupby(df['ts'].dt.date).agg({
                            'daily_pnl': 'sum',
                            'gross_notional': 'last'  # Use end-of-day gross notional
                        })
                        
                        # Calculate daily returns = daily PnL / gross_notional
                        daily_returns = daily_data['daily_pnl'] / daily_data['gross_notional'].replace(0, np.nan)
                        daily_returns = daily_returns.dropna()
                        
                        if len(daily_returns) > 0:
                            returns_series[key] = daily_returns
                            all_strategy_names.append(key)
        
        if len(returns_series) < 2:
            return {}
        
        # Create DataFrame with all series aligned by date
        returns_df = pd.DataFrame(returns_series)
        returns_df = returns_df.dropna()
        
        # Get returns matrix
        returns = returns_df.values
        n_assets = returns.shape[1]
        
        # Calculate mean returns and covariance
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)
        
        # Add small regularization to ensure positive definite
        cov_matrix = cov_matrix + np.eye(n_assets) * 1e-8
        
        try:
            # Use CVXPY for convex optimization
            # Instead of directly maximizing Sharpe ratio (non-convex), 
            # we'll use the standard mean-variance optimization and sweep over risk aversion
            weights = cp.Variable(n_assets)
            
            best_sharpe = -np.inf
            best_weights = None
            best_stats = {}
            
            # Try different risk aversion parameters to find the one maximizing Sharpe
            risk_aversions = np.logspace(-4, 2, 100)
            
            for risk_aversion in risk_aversions:
                # Portfolio return
                portfolio_return = mean_returns @ weights
                
                # Portfolio variance
                portfolio_variance = cp.quad_form(weights, cov_matrix)
                
                # Objective: maximize return - risk_aversion * variance (standard mean-variance)
                objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
                
                # Constraints
                constraints = [
                    cp.sum(weights) == 1,  # Weights sum to 1
                    weights >= 0,          # Long-only constraint
                ]
                
                # Solve the problem
                prob = cp.Problem(objective, constraints)
                try:
                    prob.solve(solver=cp.ECOS, abstol=1e-8, reltol=1e-8)
                except:
                    # Fallback to SCS solver if ECOS fails
                    prob.solve(solver=cp.SCS, eps=1e-6)
                
                if prob.status in ['optimal', 'optimal_inaccurate']:
                    # Calculate Sharpe ratio for this solution
                    w = weights.value
                    if w is not None and not np.any(np.isnan(w)):
                        port_return = np.dot(w, mean_returns)
                        port_variance = np.dot(w, np.dot(cov_matrix, w))
                        port_std = np.sqrt(port_variance)
                        
                        if port_std > 1e-8:  # Avoid division by zero
                            sharpe = np.sqrt(252) * port_return / port_std
                            
                            if sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_weights = w.copy()
                                best_stats = {
                                    'return': port_return,
                                    'std': port_std,
                                    'risk_aversion': risk_aversion
                                }
            
            if best_weights is None:
                raise ValueError("No optimal solution found")
            
            # Use the best weights found
            optimal_weights = best_weights
            
            # Calculate portfolio statistics using the best solution
            portfolio_mean = best_stats['return']
            portfolio_std = best_stats['std']
            portfolio_sharpe = best_sharpe
            
            # Create results dictionary
            results = {
                'weights': dict(zip(all_strategy_names, optimal_weights)),
                'portfolio_sharpe': portfolio_sharpe,
                'portfolio_daily_mean': portfolio_mean,
                'portfolio_daily_std': portfolio_std,
                'portfolio_annual_return': portfolio_mean * 252,  # Annualized return
                'portfolio_annual_vol': portfolio_std * np.sqrt(252),  # Annualized volatility
                'status': 'optimal',
                'risk_aversion': best_stats['risk_aversion']
            }
            
            # Calculate historical performance with optimal weights
            optimal_returns = np.dot(returns, optimal_weights)
            cumulative_returns = (1 + optimal_returns).cumprod() - 1
            
            results['optimal_cumulative_returns'] = pd.Series(
                cumulative_returns, 
                index=returns_df.index
            )
            
            # Add individual Sharpe ratios for comparison
            individual_sharpes = {}
            for i, name in enumerate(all_strategy_names):
                if np.std(returns[:, i]) > 1e-8:
                    individual_sharpes[name] = np.sqrt(252) * np.mean(returns[:, i]) / np.std(returns[:, i])
                else:
                    individual_sharpes[name] = 0.0
            results['individual_sharpes'] = individual_sharpes
            
            return results
                
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            # Fallback to equal weights
            equal_weight = 1.0 / n_assets
            equal_weights = np.ones(n_assets) * equal_weight
            
            portfolio_mean = np.dot(equal_weights, mean_returns)
            portfolio_std = np.sqrt(np.dot(equal_weights, np.dot(cov_matrix, equal_weights)))
            portfolio_sharpe = np.sqrt(252) * portfolio_mean / portfolio_std
            
            return {
                'weights': dict(zip(all_strategy_names, equal_weights)),
                'portfolio_sharpe': portfolio_sharpe,
                'portfolio_daily_mean': portfolio_mean,
                'portfolio_daily_std': portfolio_std,
                'portfolio_annual_return': portfolio_mean * 252,
                'portfolio_annual_vol': portfolio_std * np.sqrt(252),
                'status': 'equal_weight_fallback',
                'error': str(e)
            }
    
    def create_optimal_portfolio_chart(self) -> go.Figure:
        """Create visualization of optimal portfolio weights and performance."""
        opt_results = self.calculate_optimal_portfolio()
        
        if not opt_results or 'weights' not in opt_results:
            return go.Figure().add_annotation(
                text="Insufficient data for portfolio optimization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Optimal Portfolio Weights',
                'Portfolio Performance Metrics',
                'Cumulative PnL: Optimal vs Individual Strategies',
                'Weight Distribution by Model Type'
            ),
            specs=[[{"type": "bar"}, {"type": "table"}],
                   [{"type": "scatter"}, {"type": "pie"}]],
            row_heights=[0.4, 0.6],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Sort weights for display
        weights = opt_results['weights']
        sorted_strategies = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        # 1. Bar chart of weights
        strategies = [s[0] for s in sorted_strategies]
        weight_values = [s[1] for s in sorted_strategies]
        
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=weight_values,
                text=[f"{w:.1%}" for w in weight_values],
                textposition='auto',
                marker_color=['green' if w > 0.05 else 'lightgray' for w in weight_values]
            ),
            row=1, col=1
        )
        
        # 2. Performance metrics table
        metrics_data = [
            ['Metric', 'Value'],
            ['Optimal Portfolio Sharpe', f"{opt_results['portfolio_sharpe']:.3f}"],
            ['Annual Return', f"{opt_results['portfolio_annual_return']*100:.1f}%"],
            ['Annual Volatility', f"{opt_results['portfolio_annual_vol']*100:.1f}%"],
            ['Daily Mean PnL', f"${opt_results['portfolio_daily_mean']:,.0f}"],
            ['Daily Std PnL', f"${opt_results['portfolio_daily_std']:,.0f}"],
            ['Number of Strategies', f"{len([w for w in weight_values if w > 0.001])}"],
            ['Optimization Status', opt_results['status']]
        ]
        
        # Add individual Sharpe ratios if available
        if 'individual_sharpes' in opt_results:
            metrics_data.append(['---', '---'])
            metrics_data.append(['Individual Sharpe Ratios:', ''])
            for strat, sharpe in sorted(opt_results['individual_sharpes'].items(), 
                                       key=lambda x: x[1], reverse=True)[:10]:  # Top 10
                if weights.get(strat, 0) > 0.01:  # Only show if weight > 1%
                    metrics_data.append([f"  {strat}", f"{sharpe:.3f}"])
        
        if 'risk_aversion' in opt_results:
            metrics_data.append(['Risk Aversion Parameter', f"{opt_results['risk_aversion']:.2e}"])
        
        fig.add_trace(
            go.Table(
                cells=dict(
                    values=list(zip(*metrics_data)),
                    align='left',
                    font=dict(size=12),
                    height=25
                )
            ),
            row=1, col=2
        )
        
        # 3. Cumulative returns comparison
        if 'optimal_cumulative_returns' in opt_results:
            # Add optimal portfolio line
            fig.add_trace(
                go.Scatter(
                    x=opt_results['optimal_cumulative_returns'].index,
                    y=opt_results['optimal_cumulative_returns'].values * 100,  # Convert to percentage
                    mode='lines',
                    name='Optimal Portfolio',
                    line=dict(width=3, color='red')
                ),
                row=2, col=1
            )
            
            # Add individual strategy lines (faded)
            for strategy, weight in weights.items():
                if weight > 0.01:  # Only show strategies with >1% weight
                    for sim_name, breakdown_data in self.breakdown_data.items():
                        if strategy in breakdown_data:
                            cumulative_pnl = breakdown_data[strategy].get('cumulative_pnl')
                            if cumulative_pnl is not None and len(cumulative_pnl) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=cumulative_pnl.index,
                                        y=cumulative_pnl.values / 1000,
                                        mode='lines',
                                        name=f"{strategy} ({weight:.1%})",
                                        line=dict(width=1),
                                        opacity=0.4
                                    ),
                                    row=2, col=1
                                )
                                break
        
        # 4. Pie chart of weights by model type
        model_weights = {}
        for strategy, weight in weights.items():
            model = strategy.split('_')[0]
            model_weights[model] = model_weights.get(model, 0) + weight
        
        fig.add_trace(
            go.Pie(
                labels=list(model_weights.keys()),
                values=list(model_weights.values()),
                textinfo='label+percent',
                hole=0.4
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Optimal Portfolio Analysis (Sharpe: {opt_results["portfolio_sharpe"]:.3f})',
            height=900,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_yaxes(title_text="Weight", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative PnL ($K)", row=2, col=1)
        
        return fig
    
    def create_model_horizon_gross_notional_chart(self) -> go.Figure:
        """Create gross notional size chart for each model_horizon combination."""
        fig = go.Figure()
        
        if not self.breakdown_data:
            return fig
        
        # Plot gross notional for each model_horizon
        colors = px.colors.qualitative.Plotly
        color_idx = 0
        
        for sim_name, breakdown_data in self.breakdown_data.items():
            for key, data in sorted(breakdown_data.items()):
                if 'notional_ts' in data and len(data['notional_ts']) > 0:
                    fig.add_trace(go.Scatter(
                        x=data['notional_ts'].index,
                        y=data['notional_ts'].values / 1e6,  # Convert to millions
                        mode='lines',
                        name=key,
                        line=dict(width=2, color=colors[color_idx % len(colors)]),
                        legendgroup=key,
                        showlegend=True
                    ))
                    color_idx += 1
        
        fig.update_layout(
            title='Gross Notional Size by Model-Horizon Combination',
            xaxis_title='Date',
            yaxis_title='Gross Notional ($M)',
            hovermode='x unified',
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            )
        )
        
        return fig
    
    def create_model_horizon_performance_table(self) -> pd.DataFrame:
        """Create detailed performance table for each model_horizon combination."""
        rows = []
        
        for sim_name, breakdown_data in self.breakdown_data.items():
            for key, data in sorted(breakdown_data.items()):
                row = {
                    'Model_Horizon': key,
                    'Model': data['model'],
                    'Horizon (min)': data['horizon'],
                    'Sharpe Ratio': data['sharpe'],
                    'Total PnL': data['total_pnl'],
                    'Daily Avg PnL': data['daily_avg'],
                    'Volatility': data['volatility'],
                    'Win Rate': data['win_rate'],
                    'Max Drawdown': data['max_drawdown'],
                    'Avg Turnover Ratio': data['avg_turnover_ratio'],
                    'Avg Gross Notional': data['avg_gross_notional']
                }
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            # Sort by sharpe ratio descending
            df = df.sort_values('Sharpe Ratio', ascending=False)
            return df
        
        return pd.DataFrame()
    
    def create_layout(self):
        """Create the Dash layout."""
        self.app.layout = html.Div([
            html.Div([
                html.H1("Simulation Comparison Dashboard", 
                       style={'textAlign': 'center', 'marginBottom': '20px'}),
                html.P(f"Comparing simulations: {', '.join(self.sim_names)}", 
                       style={'textAlign': 'center', 'color': '#666', 'marginBottom': '40px'}),
            ]),
            
            # Model breakdown section
            html.Div(id='breakdown-section', children=[
                html.H3("Model-Horizon Breakdown Analysis"),
                
                # Model performance heatmap
                html.Div([
                    dcc.Graph(id='model-heatmap')
                ], style={'marginBottom': '40px'}),
                
                # Model-horizon cumulative PnL
                html.Div([
                    dcc.Graph(id='model-horizon-pnl-chart')
                ], style={'marginBottom': '40px'}),
                
                # Model-horizon gross notional
                html.Div([
                    dcc.Graph(id='model-horizon-notional-chart')
                ], style={'marginBottom': '40px'}),
                
                # Model correlation matrix
                html.Div([
                    dcc.Graph(id='model-correlation-matrix')
                ], style={'marginBottom': '40px'}),
                
                # Optimal portfolio analysis
                html.Div([
                    dcc.Graph(id='optimal-portfolio-chart')
                ], style={'marginBottom': '40px'}),
                
                # Model-horizon performance table
                html.Div([
                    html.H4("Model-Horizon Performance Details"),
                    html.Div(id='model-horizon-table')
                ], style={'marginBottom': '40px'}),
            ] if self.include_breakdown else []),
            
            # Refresh interval
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # 60 seconds
                n_intervals=0
            )
        ], style={'padding': '20px', 'maxWidth': '1400px', 'margin': '0 auto'})
    
    def register_callbacks(self):
        """Register Dash callbacks."""
        
        @self.app.callback(
            [Output('model-heatmap', 'figure'),
             Output('model-horizon-pnl-chart', 'figure'),
             Output('model-horizon-notional-chart', 'figure'),
             Output('model-correlation-matrix', 'figure'),
             Output('optimal-portfolio-chart', 'figure'),
             Output('model-horizon-table', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            # Create charts
            model_heatmap = self.create_model_performance_heatmap()
            model_horizon_pnl = self.create_model_horizon_cumulative_pnl_chart()
            model_horizon_notional = self.create_model_horizon_gross_notional_chart()
            model_correlation = self.create_model_correlation_matrix()
            optimal_portfolio = self.create_optimal_portfolio_chart()
            
            # Model horizon table
            model_df = self.create_model_horizon_performance_table()
            # Format numeric columns
            if not model_df.empty:
                model_df['Sharpe Ratio'] = model_df['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
                model_df['Total PnL'] = model_df['Total PnL'].apply(lambda x: f"${x:,.0f}")
                model_df['Daily Avg PnL'] = model_df['Daily Avg PnL'].apply(lambda x: f"${x:,.0f}")
                model_df['Volatility'] = model_df['Volatility'].apply(lambda x: f"${x:,.0f}")
                model_df['Win Rate'] = model_df['Win Rate'].apply(lambda x: f"{x*100:.1f}%")
                model_df['Max Drawdown'] = model_df['Max Drawdown'].apply(lambda x: f"{x*100:.1f}%")
                model_df['Avg Turnover Ratio'] = model_df['Avg Turnover Ratio'].apply(lambda x: f"{x*100:.1f}%")
                model_df['Avg Gross Notional'] = model_df['Avg Gross Notional'].apply(lambda x: f"${x/1e6:,.1f}M")
            
            model_table = dash_table.DataTable(
                data=model_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in model_df.columns],
                style_cell={'textAlign': 'left'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                        'if': {'column_id': 'Sharpe Ratio'},
                        'fontWeight': 'bold'
                    }
                ],
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                sort_action="native"
            ) if not model_df.empty else html.P("No model breakdown data available")
            
            return model_heatmap, model_horizon_pnl, model_horizon_notional, model_correlation, optimal_portfolio, model_table
    
    def run(self, port: int = 8057):
        """Run the Dash server."""
        self.load_all_data()
        self.create_layout()
        self.register_callbacks()
        
        logger.info(f"Starting Dash server on port {port}")
        self.app.run(host='0.0.0.0', port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(description='Interactive simulation comparison dashboard')
    parser.add_argument('--sims', nargs='+', required=True, 
                        help='List of simulation names to compare')
    parser.add_argument('--sim-dir', default=SIM_DIR,
                        help='Base directory containing simulations')
    parser.add_argument('--breakdown', action='store_true',
                        help='Include model/horizon breakdown analysis')
    parser.add_argument('--port', type=int, default=8057,
                        help='Port to run the server on (default: 8057)')
    
    args = parser.parse_args()
    
    # Create and run dashboard
    dashboard = SimulationDashboard(args.sims, args.sim_dir, args.breakdown)
    dashboard.run(port=args.port)


if __name__ == '__main__':
    main()