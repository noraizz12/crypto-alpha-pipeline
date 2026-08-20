"""Statistical arbitrage trading system library.

This package contains all the core functionality for the stat arb trading system,
including bar generation, feature engineering, alpha generation, portfolio optimization,
simulation, and live trading.
"""

from .portfolio_optimization import PortfolioOptimizer, TargetSolver
from .universe import Universe

__all__ = [
    'PortfolioOptimizer',
    'TargetSolver',
    'Universe'
]
