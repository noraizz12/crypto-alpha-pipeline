"""Portfolio optimization using convex optimization for alpha-driven trading.

This module implements portfolio optimization for the statistical arbitrage system
using CVXPY for convex optimization. It handles:

1. **Mean-Variance Optimization**: Maximizes expected returns while controlling risk
2. **Transaction Cost Modeling**: Incorporates slippage and fees into optimization
3. **Constraint Management**: Enforces position limits, exposure limits, and turnover
4. **Multi-Level Solving**: Iteratively loosens constraints to find feasible solutions
5. **Factor Risk Decomposition**: Separates systematic and idiosyncratic risk

The optimization problem solved is:
    maximize: alpha - kappa * risk^2 - gamma * slippage - fees
    subject to: position bounds, exposure limits, turnover constraints

Key classes:
    - TargetSolver: Solves the convex optimization problem for given parameters
    - PortfolioOptimizer: Orchestrates the full optimization pipeline

The module integrates with the broader trading system by:
    - Taking alpha signals from multiple forecast models
    - Outputting optimal position targets for execution
    - Respecting real-world trading constraints (liquidity, risk limits)
    - Adapting to market conditions through dynamic parameter adjustment

Example:
    >>> optimizer = PortfolioOptimizer(config)
    >>> alpha_df = pd.DataFrame({
    ...     'alpha_opt': [0.001, -0.002, 0.003],
    ...     'position': [100000, -50000, 0],
    ...     'risk_1440': [0.01, 0.02, 0.015],
    ...     'lbound': [-200000, -150000, -100000],
    ...     'ubound': [200000, 150000, 100000]
    ... })
    >>> optimized_df = optimizer.optimize(alpha_df)
    >>> targets = optimizer.calculate_targets(optimized_df)
"""

from datetime import datetime as dt
import logging
from typing import Tuple, Optional, List, Dict, Any

import cvxpy as cp
import numpy as np
import pandas as pd

from lib.util.dataframes import shrink_floats, long_short_sums, remove_infs, check_dup_rows
from lib.util.time_util import dt_to_str
from lib.util.directory import SIM_DIR
from lib.util.util import fmoney
from lib.util.util import log_and_raise
from lib.util.logging_util import KeyLogger
from lib.util.config import extract_horizons

original_logger = logging.getLogger(__name__)
original_logger.setLevel(logging.INFO)
logger = KeyLogger(original_logger)

MIN_PORTFOLIO_PCT = 0.92
VERBOSE_OPT = True
MAX_ITER = 100000
LOOSEN_ALERT_LEVEL = 2
MAX_LOOSEN_LEVEL = 5
SMALL_PORTFOLIO_TRADE_MULT = 5.0

class OptimizationError(Exception):
    """Exception raised when portfolio optimization fails.
    
    This exception is raised when the convex optimization problem cannot be solved
    successfully, either due to infeasibility, numerical issues, or constraint violations.
    """
    pass


class TargetSolver:
    """Solves convex optimization problems for portfolio construction.
    
    This class encapsulates the CVXPY optimization problem and provides methods to:
    - Solve for optimal portfolio weights given alpha signals and constraints
    - Dynamically adjust risk aversion (kappa) to achieve target portfolio size
    - Calculate expected utility metrics from the solution
    
    The optimization objective is:
        maximize: alpha * weights - kappa * risk - gamma * slippage - fees
        
    Where:
        - alpha: Expected returns vector
        - weights: Portfolio weights (positions in dollars)
        - kappa: Risk aversion parameter
        - risk: Quadratic risk (factor + residual)
        - gamma: Transaction cost coefficient
        - slippage: Market impact costs
        - fees: Exchange and funding fees
    
    Attributes:
        config: Configuration dictionary
        weights: CVXPY Variable for portfolio positions
        ret: Expected portfolio return expression
        risk: Total portfolio risk (factor + residual)
        kappa: Risk aversion parameter (adjustable)
        gamma: Transaction cost parameter
        prob: CVXPY Problem object
        solved: Whether optimization was successfully solved
    """
    def __init__(
            self,
            config: dict,
            weights: cp.Variable,
            ret: cp.Expression,
            factor_loadings: np.array,
            factor_risk,
            resid_risk,
            slippage,
            fees,
            kappa: float,
            gamma: float,
            constraints: List[cp.Constraint]
    ):
        """Initialize the optimization solver with problem components.
        
        Args:
            config: Configuration dictionary containing optimization parameters
            weights: CVXPY Variable representing portfolio positions
            ret: Expression for expected portfolio return
            factor_loadings: Matrix of factor exposures (n_assets x n_factors)
            factor_risk: Quadratic form for systematic risk
            resid_risk: Quadratic form for idiosyncratic risk
            slippage: Expression for market impact costs (optional)
            fees: Expression for trading fees
            kappa: Initial risk aversion parameter
            gamma: Transaction cost coefficient
            constraints: List of optimization constraints
        """
        self.config = config
        self.rho = self.config['OPT_RHO']
        self.rho_interval = self.config['OPT_RHO_INTERVAL']
        self.alpha = self.config['OPT_ALPHA']

        self.weights = weights

        self.ret = ret
        self.factor_loadings = factor_loadings
        self.factor_risk = factor_risk
        self.resid_risk = resid_risk
        self.risk = self.resid_risk + self.factor_risk
        self.slippage = slippage
        self.fees = fees

        self.gamma = cp.Parameter(nonneg=True)
        self.gamma.value = gamma

        self.starting_kappa = kappa
        self.kappa = cp.Parameter(nonneg=True)
        self.kappa.value = kappa

        obj_func = self.ret - (self.kappa * self.risk) #- self.fees
        if slippage is not None:
            obj_func -= self.gamma * self.slippage
        self.prob = cp.Problem(cp.Maximize(obj_func), constraints=constraints)

        self.solved = False

    def solve(self, verbose: Optional[bool] = None) -> float:
        """Solve the optimization problem using OSQP solver.
        
        Attempts to find optimal portfolio weights that maximize the objective
        function subject to constraints. Validates the solution for feasibility
        and non-zero portfolio size.
        
        Args:
            verbose: Whether to print solver output (defaults to VERBOSE_OPT)
            
        Returns:
            Total portfolio size (sum of absolute position values)
            
        Raises:
            OptimizationError: If optimization fails or returns invalid solution
        """
        if verbose is None:
            verbose = VERBOSE_OPT

        logger.info(f"Solving problem {verbose=}")
        try:
            self.prob.solve(
                verbose=verbose,
                solver='OSQP',
                max_iter=50000,              # Reduced from 100000, still conservative
                eps_abs=1e-4,                # Explicit tolerance for $100 accuracy
                eps_rel=1e-4,                # Relative tolerance matching absolute
                adaptive_rho=True,           # Enable automatic penalty tuning
                adaptive_rho_interval=50,    # Update rho every 50 iterations
                rho=0.1,                     # Default rho for portfolio problems
                warm_start=True,             # Reuse previous solution as starting point
                polishing=True,                 # Refine solution for better accuracy
                polish_refine_iter=3,        # Polish iterations
                scaled_termination=True,     # Use scaled termination criteria
                check_termination=25         # Check convergence every 25 iterations
            )
        except cp.error.SolverError as e:
            raise OptimizationError(f"Error solving problem {e=}")

        solved = True
        error_msg = ""
        if self.prob.status not in ('optimal', 'solved'):
            error_msg += f"fail to optimize with optimal or solved, optimal status: {self.prob.status}"
            print(self.ret)
            print(self.risk)
            print(self.slippage)
            solved = False

        portfolio_size = np.nan
        if self.weights.value is not None:
            portfolio_size = np.sum(np.abs(self.weights.value))
            if pd.isna(portfolio_size) or portfolio_size == 0:
                error_msg += "nan in optimization or zero portfolio_size"
                print("Weights")
                print(self.weights.value)
                print()
                print("Return")
                print(self.ret)
                print()
                print("Risk")
                print(self.risk)
                print()
                print("Slippage")
                print(self.slippage)
                solved = False

        if not solved:
            raise OptimizationError(error_msg)

        self.solved = True
        return portfolio_size

    def solve_for_size(self, alpha_df: pd.DataFrame, min_portfolio_size: float, max_portfolio_size: float) -> pd.DataFrame:
        """Iteratively adjust kappa to achieve target portfolio size.
        
        Uses a binary search-like algorithm to find the risk aversion parameter
        (kappa) that results in a portfolio size within the specified bounds.
        Higher kappa reduces portfolio size by penalizing risk more.
        
        Args:
            alpha_df: DataFrame with alpha signals and constraints
            min_portfolio_size: Minimum acceptable portfolio size in dollars
            max_portfolio_size: Maximum acceptable portfolio size in dollars
            
        Returns:
            The input alpha_df unchanged (for compatibility)
            
        Note:
            The optimization results are stored in self.weights.value
            and can be retrieved using set_check_and_log_solution()
        """
        self.solved = False
        step_factor = 10
        getting_smaller = True
        for ii in range(30):
            if step_factor < 1e-16:
                step_factor = 10
            if self.kappa.value < 1e-16 or self.kappa.value > 1e16:
                self.kappa.value = self.starting_kappa
                break

            try:
                portfolio_size = self.solve()

                if portfolio_size > max_portfolio_size:
                    expected_return = self.ret.value
                    expected_risk = self.risk.value * self.kappa.value
                    expected_slippage = self.slippage.value * self.gamma.value
                    logger.info(f"Portfolio Too Big ${portfolio_size:.0f} {expected_return=:.2f} {expected_risk=:.2f} {expected_slippage=:.2f}")

                    self.kappa.value *= (1 + step_factor)
                    logger.info(f"New Kappa: {self.kappa.value} step: {step_factor}")
                    if not getting_smaller:
                        getting_smaller = True
                        step_factor /= 2
                    continue

                if portfolio_size < min_portfolio_size:
                    expected_return = self.ret.value
                    expected_risk = self.risk.value * self.kappa.value
                    expected_slippage = self.slippage.value * self.gamma.value
                    logger.info(f"Portfolio Too small ${portfolio_size:.0f} {expected_return=:.2f} {expected_risk=:.2f} {expected_slippage=:.2f}")

                    self.kappa.value /= (1 + step_factor)
                    logger.info(f"New Kappa: {self.kappa.value} step: {step_factor}")
                    if getting_smaller:
                        getting_smaller = False
                        step_factor /= 2
                    continue

                logger.info(f"Portfolio Size Found: ${portfolio_size:.0f}")
                break
            except (OptimizationError, cp.SolverError) as e:
                logger.warning(str(e))
                continue
            except Exception as e:
                logger.error(str(e), key="see error in optimization")
                break

        return alpha_df

    def set_check_and_log_solution(self, alpha_df: pd.DataFrame, min_position: float) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
        """Extract optimization solution and calculate utility metrics.
        
        Transfers the optimized weights to the DataFrame, applies minimum
        position filters, and calculates comprehensive portfolio metrics
        including expected return, risk decomposition, and trading costs.
        
        Args:
            alpha_df: DataFrame to populate with target positions
            min_position: Minimum position size threshold (positions below
                         this are zeroed out)
                         
        Returns:
            Tuple of:
                - alpha_df with 'target_opt' column containing optimal positions
                - Dictionary of utility metrics if successful, None otherwise
                
        Side Effects:
            - Sets alpha_df['target_opt'] with optimal positions
            - Logs detailed portfolio metrics
            - May set self.solved = False if portfolio is invalid
        """
        if not self.solved:
            logger.error("Problem not solved", key="see unsolved optimization")
            alpha_df['target_opt'] = 0.0
            return alpha_df, None

        alpha_df['target_opt'] = self.weights.value
        alpha_df.loc[alpha_df['target_opt'].abs() < min_position, 'target_opt'] = 0
        if alpha_df['target_opt'].abs().sum() == 0:
            logger.error("Zero Portfolio Generated...", key="zero portfolio notional generated")
            self.solved = False
            return alpha_df, None

        expected_long, expected_short = long_short_sums(alpha_df, 'target_opt')
        portfolio_notional = expected_long - expected_short
        long_alpha = np.dot(alpha_df['target_opt'].clip(lower=0).values, alpha_df['alpha_opt'])
        short_alpha = np.dot(alpha_df['target_opt'].clip(upper=0).values, alpha_df['alpha_opt'])
        expected_risk = self.risk.value * self.kappa.value
        expected_factor_risk = self.factor_risk.value * self.kappa.value
        expected_resid_risk = self.resid_risk.value * self.kappa.value
        expected_return = self.ret.value
        expected_return_bps = 10000 * expected_return / portfolio_notional
        expected_slippage = self.slippage.value * self.gamma.value if self.slippage is not None else 0.0
        expected_fees = self.fees.value
        expected_utility = self.ret.value - self.kappa.value * self.risk.value - expected_slippage - expected_fees
        expected_traded_dollars = np.sum(np.abs(self.weights.value - alpha_df['position'].values))

        logger.info(f"ERet (bps): {expected_return_bps:.2f}, ERet: {expected_return:.0f} ERiskSq: {expected_risk:.2f} ESlip: {expected_slippage:.2f} Util: {expected_utility:.0f}")
        logger.info(f"ETraded: ${expected_traded_dollars:.0f} ELong/EShort: ${expected_long:.0f} / ${expected_short:.0f} EFactorRiskSq: {expected_factor_risk:.2f} EResidRiskSq: {expected_resid_risk:.2f}")
        logger.info(f"Alpha Long/Short: {long_alpha:.4f} / {short_alpha:.4f}")

        util_metrics = {
            'expected_utility': expected_utility,
            'expected_risk': expected_risk,
            'expected_factor_risk': expected_factor_risk,
            'expected_resid_risk': expected_resid_risk,
            'expected_return': expected_return,
            'expected_return_bps': expected_return_bps,
            'expected_slippage': expected_slippage,
            'expected_fees': expected_fees
        }
        return alpha_df, util_metrics


class PortfolioOptimizer:
    """Main portfolio optimization orchestrator for the trading system.
    
    This class manages the complete portfolio optimization pipeline:
    1. Combines alpha signals from multiple forecast horizons
    2. Prepares risk and constraint data
    3. Sets up and solves the optimization problem
    4. Post-processes results into executable trades
    
    The optimizer handles real-world complexities including:
    - Multi-horizon alpha aggregation with appropriate scaling
    - Dynamic constraint loosening for feasibility
    - Portfolio balance and size adjustments
    - Trade utility calculation for prioritization
    - Factor exposure monitoring and limits
    
    Attributes:
        config: Configuration dictionary with all parameters
        horizons: List of forecast horizons to use
        last_kappa: Current risk aversion parameter (persists across optimizations)
        gamma: Transaction cost coefficient
        max_portfolio_size: Maximum allowed portfolio notional
        max_alpha: Cap on individual alpha signals
        util_metrics: Dictionary of utility metrics from last optimization
    """
    def __init__(self, config: dict, horizons: Optional[List[int]] = None,
                 scale: Optional[float] = None):
        """Initialize the portfolio optimizer with configuration.

        Args:
            config: Configuration dictionary containing all optimization parameters
                   including risk limits, constraints, and model parameters
            horizons: Optional list of forecast horizons to use. If None,
                     extracts from config
            scale: Optional scale factor (0.0-1.0) to apply to target positions.
                   If None, no scaling is applied.
        """
        self.config = config
        self.scale = scale

        self.horizons = extract_horizons(self.config) if horizons is None else horizons
        self.last_kappa = self.config['KAPPA']
        self.gamma = self.config['GAMMA']
        self.max_portfolio_size = self.config['MAX_PORTFOLIO_NOTIONAL']
        self.max_alpha = self.config['MAX_ALPHA']
        self.max_portfolio_notional_slack = self.config['MAX_PORTFOLIO_NOTIONAL_SLACK']
        self.base_aggression = self.config['BASE_AGGRESSION']
        self.opt_interval_mins = self.config['REOPTIMIZE_INTERVAL_MINS']
        self.opt_horizon = self.config['OPT_HORIZON']
        self.min_position = self.config['MIN_POSITION']
        self.max_position_pct = self.config['MAX_POSITION_PCT']
        self.max_trade_dollars = self.config['MAX_TRADE_DOLLARS']
        self.min_trade_dollars = self.config['MIN_TRADE_DOLLARS']
        self.max_turnover = self.config['MAX_TURNOVER']
        self.short_term_model_horizons = self.config['SHORT_TERM_MODEL_HORIZONS']
        self.exchange_fees = self.config['EXCHANGE_FEES']
        self.factor_sigmas = config['FACTOR_SIGMAS']
        self.exposure_limits = self.config['EXPOSURE_LIMITS']
        self.scale_alpha_opt = self.config['SCALE_ALPHA_OPT']
        self.center_alpha_opt = self.config['CENTER_ALPHA_OPT']
        self.const_notional = self.config['CONST_NOTIONAL']
        self.correct_target_imbalance = self.config['CORRECT_TARGET_IMBALANCE']
        self.alpha_mult = self.config['ALPHA_MULT']
        self.alpha_tilt = self.config['ALPHA_TILT']
        self.max_volume_fraction_participation = self.config['MAX_VOLUME_FRACTION_PARTICIPATION'] * 2.0
        self.horizon_model_factor = self.config['HORIZON_MODEL_FACTOR']

        self.optimizations_per_horizon = self.opt_horizon / self.opt_interval_mins
        self.alpha_std_bound = 5
        self.util_metrics = {}

    @staticmethod
    def generate_inequality_constraints(alpha_df: pd.DataFrame, exposure_limits: Dict[str, float], max_portfolio_size: float):
        """Generate constraint matrices for factor exposure limits.
        
        Creates constraint matrices A and b for linear inequalities of the form:
        A @ weights <= b (for positive exposures)
        A @ weights >= -b (for negative exposures)
        
        Args:
            alpha_df: DataFrame with factor loadings for each asset
            exposure_limits: Dictionary mapping factor names to exposure fractions
            max_portfolio_size: Maximum portfolio size for scaling limits
            
        Returns:
            Tuple of (A_pos, A_neg, b_pos, b_neg) constraint matrices
            
        Note:
            Currently scales exposure limits by portfolio size, but TODO
            suggests converting to percentage limits
        """
        n = len(alpha_df)
        A_pos = np.zeros((n, len(exposure_limits)))
        b_pos = []
        A_neg = np.zeros((n, len(exposure_limits)))
        b_neg = []

        alpha_df['dollar_exposure'] = np.ones(n)

        ii = 0
        for factor, limit in exposure_limits.items():
            A_pos[:, ii] = alpha_df[factor].values
            A_neg[:, ii] = alpha_df[factor].values
            # XXX TODO: make these percent limits, not dollar limits.
            limit = limit * max_portfolio_size
            logger.info(f"Limiting {factor} dollar exposure to {limit}")
            b_pos.append(limit)
            b_neg.append(-limit)
            ii += 1
        b_pos = np.array(b_pos)
        b_neg = np.array(b_neg)
        return A_pos, A_neg, b_pos, b_neg

    @staticmethod
    def generate_factor_matrix(alpha_df: pd.DataFrame, factor_sigmas: Dict[str, float]) -> Tuple[np.array, np.array]:
        """Construct factor loading and covariance matrices for risk modeling.
        
        Creates matrices for the factor risk model:
        Factor Risk = weights' @ F_loadings @ F_cov @ F_loadings' @ weights
        
        Args:
            alpha_df: DataFrame containing factor exposures for each asset
            factor_sigmas: Dictionary mapping factor names to their volatilities
                          (factors with sigma=0 are skipped)
                          
        Returns:
            Tuple of:
                - F_loadings: Factor loading matrix (n_assets x n_factors)
                - F_cov: Diagonal factor covariance matrix (n_factors x n_factors)
                
        Note:
            Assumes factors are uncorrelated (diagonal covariance matrix)
        """
        n = len(alpha_df)
        m = len(factor_sigmas)
        F_loadings = np.zeros((n, m))
        F_cov = np.eye(m)
        if 'dollar_exposure' not in alpha_df.columns:
            alpha_df['dollar_exposure'] = np.ones(n)

        ii = 0
        for factor, sigma in factor_sigmas.items():
            if sigma == 0:
                continue

            if factor not in alpha_df.columns:
                logger.warning(f"Factor {factor} not in alpha_df columns: {alpha_df.columns}")
                alpha_df[factor] = np.float32(0.0)

            if factor.startswith('category'):
                logger.info(f"Calculating {factor}, mean: {alpha_df[factor].mean()} std: {alpha_df[factor].std()} min: {alpha_df[factor].min()} max: {alpha_df[factor].max()}")
                F_loadings[:, ii] = alpha_df[factor].astype(float).fillna(0).values
            else:
                logger.info(f"Calculating {factor}, mean: {alpha_df[factor].mean()} std: {alpha_df[factor].std()} min: {alpha_df[factor].min()} max: {alpha_df[factor].max()}")
                F_loadings[:, ii] = alpha_df[factor].fillna(0).values

            F_cov[ii, ii] = np.square(sigma)
            ii += 1

        return F_loadings, F_cov

    @staticmethod
    def check_factor_exposures(factor_sigmas: Dict[str, float], factor_loadings: np.array, targets_s: pd.Series) -> None:
        """Calculate and log factor exposures and risks for the portfolio.
        
        For each risk factor, computes the portfolio's dollar exposure and
        associated risk contribution. Useful for monitoring systematic risks.
        
        Args:
            factor_sigmas: Dictionary of factor volatilities
            factor_loadings: Factor loading matrix from generate_factor_matrix
            targets_s: Series of target positions
            
        Raises:
            Exception: If NaN risk is detected for any factor
            
        Side Effects:
            Logs factor exposures and risk contributions
        """
        ii = 0
        for factor, sigma in factor_sigmas.items():
            if sigma == 0:
                continue
            f_exposure = np.dot(factor_loadings[:, ii], targets_s.values)
            if pd.isna(f_exposure):
                raise log_and_raise(f"Nan risk for {factor}", df=targets_s)

            f_risk = np.sqrt(sigma * f_exposure ** 2)
            logger.info(f"Factor Exposure: {factor} ${f_exposure:.0f}, risk: {f_risk:.0f}")
            ii += 1

    def check_bad_risk_data(self, alpha_df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing or invalid risk data before optimization.
        
        Identifies assets with NaN risk values and applies fallback strategies:
        - Sets alpha and spread to 0 for bad data
        - Uses previous period risk values if available
        - Falls back to universe average risk
        - Adjusts position bounds to prevent trading
        
        Args:
            alpha_df: DataFrame with risk and alpha data
            
        Returns:
            alpha_df with cleaned risk data and adjusted bounds
            
        Side Effects:
            Modifies alpha_df in place
            Logs warnings for affected securities
        """
        risk_fld = f'risk_{self.opt_horizon}'
        bad_risk_idx = alpha_df[risk_fld].isna()
        if any(bad_risk_idx):
            bad_cnt = len(alpha_df[bad_risk_idx])
            logger.info(f"Some bad risk data {risk_fld} found prior to optimization on {bad_cnt} securities, setting values to 0...")
            do_i_care_loc = (alpha_df['lbound'] != 0) & (alpha_df['ubound'] != 0)
            print_df = alpha_df[bad_risk_idx & do_i_care_loc]
            if len(print_df) > 0:
                try:
                    symbols = print_df['symbol_venue'].unique()
                    logger.warning(f"Bad risk data on {symbols}")
                    flds = ['symbol_venue', 'alpha_opt', 'ubound', 'lbound', 'position', 'dvolume_1440_trmean', risk_fld, 'close_mid']
                    print(print_df[flds])
                except:
                    pass

            for fld in ['alpha_opt', 'relative_spread_1440_trmean']:
                alpha_df.loc[bad_risk_idx, fld] = np.float32(0)

            for fld in ['dvolume_1440_trmean', risk_fld]:
                prev_fld = f"{fld}_previous"
                if prev_fld not in alpha_df.columns:
                    alpha_df.loc[bad_risk_idx, fld] = alpha_df[fld].dropna().mean()
                else:
                    # better than 0!
                    avg_risk = alpha_df[prev_fld].dropna().mean()
                    if np.isnan(avg_risk):
                        avg_risk = 0.0
                    logger.info(f"Overriding {fld} on bad data to {avg_risk}")
                    alpha_df.loc[bad_risk_idx, fld] = alpha_df[prev_fld].fillna(avg_risk)

            # wary of this as it may conflict with optimization bounds
            alpha_df.loc[bad_risk_idx, 'lbound'] = alpha_df['position'].fillna(0).clip(upper=0)
            alpha_df.loc[bad_risk_idx, 'ubound'] = alpha_df['position'].fillna(0).clip(lower=0)
        return alpha_df

    def calc_max_trade_dollars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate maximum allowed trade size for each asset.
        
        Trade limits are based on:
        - Fraction of forecast volume (liquidity constraint)
        - Absolute maximum trade size
        - Current position size (allows unwinding)
        
        Args:
            df: DataFrame with volume forecasts and positions
            
        Returns:
            df with 'max_trade_dollars' column added
        """
        df['max_trade_dollars'] = (self.max_volume_fraction_participation * df[f'dvolume_{self.opt_interval_mins}_forecast']).clip(upper=self.max_trade_dollars)
        df['position_abs'] = df['position'].abs()

        # rethink this....
        df['max_trade_dollars'] = df[['position_abs', 'max_trade_dollars']].max(axis=1)
        return df

    def calc_total_max_trade_dollars(self, df: pd.DataFrame, loosen_level: int) -> float:
        """Calculate total portfolio turnover limit for this optimization.
        
        Dynamically adjusts turnover based on:
        - Current portfolio size vs maximum
        - Loosen level (higher levels allow more trading)
        - Minimum needed to satisfy constraints
        
        Args:
            df: DataFrame with positions and bounds
            loosen_level: Constraint loosening level (0-5)
            
        Returns:
            Maximum total dollars that can be traded
        """
        current_portfolio_size = df['position'].abs().sum()
        upper_needed_dollars = (df['position'] - df['ubound']).clip(lower=0).sum()
        lower_needed_dollars = abs((df['position'] - df['lbound']).clip(upper=0).sum())
        total_needed_dollars = (upper_needed_dollars + lower_needed_dollars) * 1.05

        max_traded = (self.max_turnover * self.max_portfolio_size) / self.optimizations_per_horizon
        if current_portfolio_size < self.max_portfolio_size * .8:
            logger.warning(f"{current_portfolio_size=} less than 80% of max, allowing {SMALL_PORTFOLIO_TRADE_MULT}x trading...")
            max_traded *= SMALL_PORTFOLIO_TRADE_MULT
        elif loosen_level == 1:
            max_traded *= 1.5
        elif loosen_level == 2:
            max_traded *= 2.0
        elif loosen_level == 3:
            max_traded *= 3.0

        if total_needed_dollars > max_traded:
            logger.warning(f"Due to constraints, need to be able to trade {total_needed_dollars} > {max_traded=}, increasing max_traded")
            max_traded = total_needed_dollars

        logger.info(f"Using max traded amount {max_traded}")
        return max_traded

    def log_and_check_opt_info(self, df: pd.DataFrame, loosen_level: int) -> None:
        """Log optimization inputs and validate data quality.
        
        Computes and logs summary statistics for the optimization including:
        - Alpha signal distribution
        - Position bounds
        - Risk metrics
        - Liquidity measures
        
        Args:
            df: DataFrame with optimization inputs
            loosen_level: Current constraint loosening level
            
        Raises:
            OptimizationError: If inputs are invalid (zero alpha, bounds, etc.)
            
        Side Effects:
            Logs detailed optimization statistics
        """
        tradeable_df = df[df['tradeable']]
        avg_alpha = tradeable_df['alpha_opt'].abs().mean(skipna=False)
        long_alpha, short_alpha = long_short_sums(tradeable_df, 'alpha_opt')
        min_alpha = tradeable_df['alpha_opt'].min()
        max_alpha = tradeable_df['alpha_opt'].max()
        avg_ubound = tradeable_df['ubound'].mean(skipna=False)
        avg_lbound = tradeable_df['lbound'].mean(skipna=False)
        avg_risk = tradeable_df[f'risk_{self.opt_horizon}'].abs().mean(skipna=False)
        avg_spread = tradeable_df['relative_spread_1440_trmean'].mean(skipna=False)
        avg_dvolume = tradeable_df['dvolume_1440_trmean'].mean(skipna=False)
        optimize_ts = df.index.get_level_values('ts').max()

        logger.info(f"Optimizing {optimize_ts=} with {len(tradeable_df)} coins, Avg Bounds: {fmoney(avg_lbound)} / {fmoney(avg_ubound)} {avg_risk=:.6f} {avg_spread=:.6f} {fmoney(avg_dvolume)}")
        logger.info(f"Opt Alphas average: {avg_alpha:.4f} long tot: {long_alpha:.4f}, short tot: {short_alpha:.4f}, min: {min_alpha:.4f}, max: {max_alpha:.4f}")
        if avg_alpha == 0 or avg_ubound == 0 or avg_lbound == 0 or np.isnan(avg_alpha):
            raise OptimizationError(f'{"warning: " if loosen_level < LOOSEN_ALERT_LEVEL else ""}{loosen_level=} see bad optimization input {avg_alpha=}, {avg_ubound=}, {avg_lbound=}')

        for fld in ['ubound', 'lbound']:
            assert not df[fld].isna().any()

    def calc_slippage(self, df: pd.DataFrame, trade_dollars_abs: cp.Expression) -> float:
        """Calculate market impact costs (slippage) for trades.
        
        Uses a volume-based model where slippage increases for:
        - Less liquid assets (lower volume)
        - Larger trades relative to typical volume
        
        Args:
            df: DataFrame with volume data
            trade_dollars_abs: CVXPY expression for absolute trade values
            
        Returns:
            CVXPY expression for total slippage cost
            
        Note:
            Uses log-volume ratios to handle wide range of liquidities
        """
        # consider exp decay volume instead fo past 24 hour
        dvolume_1440_mean = np.log(df.loc[df['tradeable'], 'dvolume_1440_trmean']).mean()
        df['slippage_volume_frac'] = remove_infs(dvolume_1440_mean / np.log(df['dvolume_1440'])).fillna(0)  # .clip(lower=0.1)
        slippage = cp.sum(cp.multiply(trade_dollars_abs, df['slippage_volume_frac'].values))
        return slippage

    def calc_fees(self, df: pd.DataFrame, trade_dollars_abs: cp.Expression, position_dollars: cp.Expression) -> float:
        """Calculate trading fees including exchange and funding costs.
        
        Fee components:
        - Exchange fees: Proportional to trade size
        - Funding fees: Based on funding rates and trade direction
        
        Args:
            df: DataFrame with funding rate data
            trade_dollars: Signed trade values (for funding direction)
            trade_dollars_abs: Absolute trade values (for exchange fees)
            
        Returns:
            CVXPY expression for total fee cost
            
        Note:
            Comment suggests funding fee logic may need revision
        """
        # this is screwed up and seems to go in the wrong
        # direction but makes sim look buch better.  should be a forecast
        total_trade_dollars = cp.sum(trade_dollars_abs)
        funding_fees = cp.sum(cp.multiply(position_dollars, -df['last_funding_rate'].fillna(0).values))
        exchange_fees = total_trade_dollars * self.exchange_fees
        fees = exchange_fees + funding_fees
        return fees

    def calc_factor_risk(self, df: pd.DataFrame, weights: cp.Variable) -> Tuple[Any, np.array]:
        """Calculate systematic (factor) risk for the portfolio.
        
        Computes quadratic risk from factor exposures:
        risk = weights' @ F @ Cov @ F' @ weights
        
        Args:
            df: DataFrame with factor exposures
            weights: CVXPY variable for portfolio positions
            
        Returns:
            Tuple of:
                - factor_risk: CVXPY quadratic form expression
                - factor_loadings: Loading matrix for exposure analysis
        """
        logger.info("Calculating factor risk...")
        factor_sigmas = self.factor_sigmas.copy()
        factor_loadings, factor_cov = self.generate_factor_matrix(df, factor_sigmas)
        factor_risk = cp.quad_form(factor_loadings.T @ weights, factor_cov)
        return factor_risk, factor_loadings

    def calc_resid_risk(self, df: pd.DataFrame, weights: cp.Variable):
        """Calculate idiosyncratic (residual) risk for the portfolio.
        
        Uses diagonal covariance matrix (assumes no correlation between
        asset-specific risks). Risk values come from historical volatility.
        
        Args:
            df: DataFrame with individual asset risk values
            weights: CVXPY variable for portfolio positions
            
        Returns:
            CVXPY quadratic form for residual risk
            
        Raises:
            ValueError: If covariance matrix construction fails
        """
        corr_mat = np.eye(len(df))
        std_diag = np.diag(df[f'risk_{self.opt_horizon}'].values)

        Sigma = std_diag @ corr_mat @ std_diag
        try:
            resid_risk = cp.quad_form(weights, Sigma)
        except ValueError as ve:
            print(Sigma)
            raise log_and_raise(f"Could not calculate resid COV matrix {ve}", df=df[f'risk_{self.opt_horizon}'])

        return resid_risk

    def calc_constraints(self, df: pd.DataFrame, loosen_level: int, weights: cp.Variable, trade_dollars_abs: cp.Expression) -> List[cp.Constraint]:
        """Generate optimization constraints based on loosen level.
        
        Constraints include:
        - Position bounds (always enforced)
        - Total turnover limit (always enforced)
        - Individual trade size limits (relaxed at higher levels)
        - Portfolio size limits (commented out)
        - Factor exposure limits (commented out)
        
        Args:
            df: DataFrame with bounds and limits
            loosen_level: How much to relax constraints (0=strictest)
            weights: Portfolio position variables
            trade_dollars_abs: Absolute trade size expressions
            
        Returns:
            List of CVXPY constraints
        """
        total_max_trade_dollars = self.calc_total_max_trade_dollars(df, loosen_level=loosen_level)
        total_trade_dollars = cp.sum(trade_dollars_abs)
        # max_portfolio_size = self.max_portfolio_size * (1 + self.max_portfolio_notional_slack)

        # set trade dollars big enough to get in bounds...
        # A_pos, A_neg, b_pos, b_neg = self.generate_inequality_constraints(df, self.exposure_limits, self.max_portfolio_size)
        # pos_mat = A_pos.T @ weights
        # neg_mat = A_neg.T @ weights

        constraints = [
            weights <= df['ubound'],
            weights >= df['lbound'],
            total_trade_dollars <= total_max_trade_dollars
        ]

        if loosen_level <= 1:
            constraints += [
                trade_dollars_abs <= df['max_trade_dollars'],
            ]

        # if loosen_level <= 2:
        #     constraints += [
        #         cp.sum(cp.abs(weights)) <= max_portfolio_size,
        #     ]
        #
        # if loosen_level <= 3:
        #     constraints += [
        #         total_trade_dollars <= total_max_trade_dollars,
        #     ]
        # if loosen_level <= 4:
        #     constraints += [
        #         pos_mat <= b_pos,
        #         neg_mat >= b_neg,
        #     ]

        return constraints

    def setup_optimization(self, alpha_df: pd.DataFrame, loosen_level: int = 0) -> TargetSolver:
        """Set up the convex optimization problem.
        
        Prepares all components needed for optimization:
        - Cleans and validates input data
        - Constructs objective function (return - risk - costs)
        - Builds constraint set based on loosen level
        - Creates TargetSolver instance
        
        Args:
            alpha_df: DataFrame with alpha signals, positions, and constraints
            loosen_level: Constraint relaxation level (0=strictest, 5=most relaxed)
            
        Returns:
            TargetSolver instance ready to solve
            
        Raises:
            OptimizationError: If input data is invalid
            AssertionError: If loosen_level exceeds maximum
        """
        logger.info(f"Setting up optimization at horizon={self.opt_horizon} and {loosen_level=}")
        assert loosen_level <= MAX_LOOSEN_LEVEL

        alpha_df['target_opt'] = 0.0
        alpha_df['position'] = alpha_df['position'].fillna(0)

        alpha_df = self.check_bad_risk_data(alpha_df)
        self.log_and_check_opt_info(alpha_df, loosen_level)

        mu = alpha_df['alpha_opt'].values
        weights = cp.Variable(len(alpha_df))
        kappa = self.last_kappa
        ret = mu.T @ weights

        resid_risk = self.calc_resid_risk(alpha_df, weights=weights)
        factor_risk, factor_loadings = self.calc_factor_risk(alpha_df, weights)
        alpha_df = self.calc_max_trade_dollars(alpha_df)
        trade_dollars = weights.T - alpha_df['position'].values
        trade_dollars_abs = cp.abs(trade_dollars)

        # consider higher fees for things we think we'll have to be more aggressive on...
        slippage = None
        if self.gamma > 0:
            slippage = self.calc_slippage(alpha_df, trade_dollars_abs=trade_dollars_abs)
        fees = self.calc_fees(alpha_df, trade_dollars_abs=trade_dollars_abs, position_dollars=alpha_df['position'].values)
        constraints = self.calc_constraints(alpha_df, loosen_level=loosen_level, weights=weights, trade_dollars_abs=trade_dollars_abs)

        solver = TargetSolver(
            config=self.config,
            weights=weights,
            ret=ret,
            factor_loadings=factor_loadings,
            factor_risk=factor_risk,
            resid_risk=resid_risk,
            slippage=slippage,
            fees=fees,
            kappa=kappa,
            gamma=self.gamma,
            constraints=constraints
        )
        return solver

    def optimize(self, alpha_df: pd.DataFrame) -> pd.DataFrame:
        """Run portfolio optimization with progressive constraint loosening.
        
        Attempts to solve the optimization problem, starting with strict
        constraints and progressively loosening them if infeasible. This
        ensures we find a solution while respecting constraints as much
        as possible.
        
        Args:
            alpha_df: DataFrame with alpha signals and current positions
            
        Returns:
            alpha_df with 'target_opt' column containing optimal positions
            
        Raises:
            OptimizationError: If no feasible solution found at any level
            
        Side Effects:
            - Updates self.util_metrics with optimization statistics
            - Updates self.last_kappa with final risk aversion value
            - Logs optimization results and factor exposures
        """
        self.util_metrics = {}
        solver = None
        for loosen_level in range(MAX_LOOSEN_LEVEL + 1):
            solver = self.setup_optimization(alpha_df, loosen_level=loosen_level)
            if self.const_notional:
                alpha_df = solver.solve_for_size(
                    alpha_df=alpha_df,
                    min_portfolio_size=self.max_portfolio_size * MIN_PORTFOLIO_PCT,
                    max_portfolio_size=self.max_portfolio_size
                )
            else:
                try:
                    solver.solve(verbose=True)
                except Exception as e:
                    logger.warning(f"Failed to solve {e}")
                    continue
                if solver.solved:
                    break

        if solver.solved:
            alpha_df, util_metrics = solver.set_check_and_log_solution(alpha_df=alpha_df, min_position=self.min_position)
            if util_metrics is not None:
                self.util_metrics = util_metrics
            if solver.solved:
                logger.info("Optimization Solved...")
                self.check_factor_exposures(factor_sigmas=self.factor_sigmas, factor_loadings=solver.factor_loadings, targets_s=alpha_df['target_opt'])
                self.last_kappa = solver.kappa.value

        if not solver.solved:
            logger.info("Running optimization as verbose to debug...")
            solver.solve(verbose=True)
            raise OptimizationError("Could not optimize")

        return alpha_df

    def dump_util_metrics(self, util_file_path: str, opt_ts: dt):
        """Save optimization utility metrics to CSV file.
        
        Writes a single-row CSV with optimization statistics including
        expected utility, risk decomposition, returns, and costs.
        
        Args:
            util_file_path: Path to output CSV file
            opt_ts: Optimization timestamp
            
        Side Effects:
            Creates/overwrites CSV file with metrics
            Logs warning if write fails
        """
        if self.util_metrics:
            headers = ['timestamp'] + list(self.util_metrics.keys())
            values = [str(opt_ts)] + [str(value) for value in self.util_metrics.values()]
            try:
                with open(util_file_path, 'w') as file:
                    file.write(','.join(headers) + '\n')
                    file.write(','.join(values) + '\n')
            except Exception as e:
                logger.warning(f'fail to dump util metrics at {opt_ts} since {e}')

    def calculate_trade_utility(self, alpha_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate utility improvement for each trade.
        
        Computes the marginal utility of moving from current position to
        target position, accounting for alpha, risk, funding, and fees.
        Used for trade prioritization.
        
        Args:
            alpha_df: DataFrame with positions, targets, and parameters
            
        Returns:
            alpha_df with 'util' and 'util_per_dollar' columns added
            
        Side Effects:
            Logs total utility of proposed trades
        """
        if 'alpha_opt' not in alpha_df.columns:
            logger.info('Cant calculate trade utility')
            return alpha_df
        # add slippage and/or factor risk?
        alpha_df['util'] = (
                (alpha_df['target_position'] * alpha_df['alpha_opt'] - self.last_kappa * alpha_df['target_position'] * alpha_df[f'risk_{self.opt_horizon}'] ** 2 - alpha_df['target_position'] * alpha_df['last_funding_rate'] - alpha_df[
                    'desired_trade_dollars'] * self.exchange_fees) -
                (alpha_df['position'] * alpha_df['alpha_opt'] - self.last_kappa * alpha_df['position'] * alpha_df[f'risk_{self.opt_horizon}'] ** 2 - alpha_df['position'] * alpha_df['last_funding_rate'])
        )
        alpha_df['util_per_dollar'] = alpha_df['util'] / (alpha_df['target_position'] - alpha_df['position']).abs()
        total_util = alpha_df['util'].sum()
        logger.info(f"Total utility of trades: {total_util}")
        return alpha_df

    def adjust_portfolio_balance_and_size(self, timeslice_df: pd.DataFrame) -> pd.DataFrame:
        """Enforce portfolio balance and size constraints.
        
        Adjusts target positions to ensure:
        - Net long/short bias stays within limits
        - Total portfolio size doesn't exceed maximum
        
        Scales down positions proportionally when limits are exceeded.
        
        Args:
            timeslice_df: DataFrame with target positions
            
        Returns:
            timeslice_df with adjusted target positions
            
        Side Effects:
            Logs adjustments made
            Modifies target_position column in place
        """
        long_dollars, short_dollars = long_short_sums(timeslice_df, 'target_position')
        long_bias = long_dollars + short_dollars
        total_target_notional = long_dollars - short_dollars

        total_slack = total_target_notional * self.max_portfolio_notional_slack
        logger.info(f"Target Long/Short: ${long_dollars:.0f} / ${short_dollars:.0f}, allowable bias {fmoney(total_slack)}")

        if self.correct_target_imbalance:
            if long_bias > total_slack:
                overage = long_bias - total_slack
                fraction_to_keep = 1.0 - (overage / long_dollars)
                assert 0 <= fraction_to_keep <= 1.0
                logger.info(f"Unbalanced Portfolio Long Bias:${long_bias:.0f} reducing ${long_dollars:.0f} -> ${(long_dollars * fraction_to_keep):.0f}, overage:{fmoney(overage)}, {fraction_to_keep=}")
                timeslice_df.loc[timeslice_df['target_position'] > 0, 'target_position'] = timeslice_df['target_position'] * fraction_to_keep
                logger.info(f"After long bias adjust, Desired Traded ${self.calculate_desired_trade_dollars(timeslice_df):.0f}")

            elif long_bias < -total_slack:
                overage = long_bias + total_slack
                fraction_to_keep = 1.0 - (overage / short_dollars)
                assert 0 <= fraction_to_keep <= 1.0
                logger.info(f"Unbalanced Portfolio Short Bias ${long_bias:.0f} reducing ${short_dollars:.0f} -> ${(short_dollars * fraction_to_keep):.0f}, overage:{fmoney(overage)}, {fraction_to_keep}")
                timeslice_df.loc[timeslice_df['target_position'] < 0, 'target_position'] = timeslice_df['target_position'] * fraction_to_keep
                logger.info(f"After short bias adjust, Desired Traded ${self.calculate_desired_trade_dollars(timeslice_df):.0f}")

        if total_target_notional > self.max_portfolio_size:
            overage = total_target_notional - self.max_portfolio_size
            fraction_to_keep = 1.0 - (overage / total_target_notional)
            assert 0 <= fraction_to_keep <= 1.0
            logger.info(f"Portfolio too big by ${overage:.0f} reducing ${total_target_notional:.0f} -> ${self.max_portfolio_size:.0f}")
            timeslice_df['target_position'] = timeslice_df['target_position'] * fraction_to_keep
            logger.info(f"After max portfolio adjust, Desired Traded ${self.calculate_desired_trade_dollars(timeslice_df):.0f}")

        if self.scale is not None:
            timeslice_df['target_position'] = timeslice_df['target_position'] * self.scale
            logger.info(f"Applied scale={self.scale:.2f}, Desired Traded ${self.calculate_desired_trade_dollars(timeslice_df):.0f}")

        return timeslice_df

    def calculate_desired_trade_dollars(self, timeslice_df: pd.DataFrame) -> float:
        desired_trade_dollars = (timeslice_df['target_position'] - timeslice_df['position']).abs().sum()
        return desired_trade_dollars

    def calculate_targets(self, timeslice_df: pd.DataFrame, dump_state: bool = False) -> pd.DataFrame:
        """Convert optimization results to executable trade targets.
        
        Post-processes optimization output by:
        - Applying trade size limits
        - Enforcing position limits
        - Filtering small trades
        - Balancing portfolio
        - Calculating trade utility
        
        Args:
            timeslice_df: DataFrame with optimization results
            dump_state: Whether to save state to CSV for debugging
            
        Returns:
            timeslice_df with final target positions and trade metrics
            
        Side Effects:
            Logs trade statistics at each step
            May write debug CSV if dump_state=True
        """
        if dump_state:
            ts = dt_to_str(timeslice_df.index.get_level_values('ts').max())
            filename = SIM_DIR + f"/ts_{ts}.csv"
            logger.info(f"Dumping {filename}")
            timeslice_df.to_csv(filename)

        timeslice_df['desired_trade_dollars'] = timeslice_df['target_opt'] - timeslice_df['position']
        logger.info(f"Before process, Desired Traded ${timeslice_df['desired_trade_dollars'].abs().sum():.0f}")

        timeslice_df['desired_trade_dollars'] = timeslice_df['desired_trade_dollars'].clip(lower=-self.max_trade_dollars, upper=self.max_trade_dollars)
        timeslice_df['desired_target_dollars'] = timeslice_df['desired_trade_dollars'] + timeslice_df['position']
        timeslice_df['target_position'] = timeslice_df['position'] + timeslice_df['desired_trade_dollars']
        logger.info(f"After max trade clip, Desired Traded ${self.calculate_desired_trade_dollars(timeslice_df):.0f}")

        max_position = self.max_position_pct * timeslice_df['target_position'].abs().sum()
        timeslice_df['target_position'] = timeslice_df['target_position'].clip(lower=-max_position, upper=max_position)
        timeslice_df['desired_trade_dollars'] = timeslice_df['target_position'] - timeslice_df['position']

        #pulls exposure in to max_position_pct based off the actual portfolio size, not the ideal size
        logger.info(f"After max position clip of {max_position}, Desired Traded ${self.calculate_desired_trade_dollars(timeslice_df):.0f}")

        # we want to trade but not a big enough delta from the existing position
        timeslice_df.loc[timeslice_df['target_position'].abs() < self.min_position, 'target_position'] = 0
        timeslice_df.loc[(timeslice_df['desired_trade_dollars'].abs() < self.min_trade_dollars) & (timeslice_df['target_position'] != 0), 'target_position'] = timeslice_df['position']
        logger.info(f"After min position drop, Desired Traded ${self.calculate_desired_trade_dollars(timeslice_df):.0f}")

        timeslice_df = self.adjust_portfolio_balance_and_size(timeslice_df)

        timeslice_df['desired_trade_dollars'] = timeslice_df['target_position'] - timeslice_df['position']
        timeslice_df['desired_trade_dollars_abs'] = timeslice_df['desired_trade_dollars'].abs()
        desired_trade_dollars = timeslice_df['desired_trade_dollars_abs'].sum()
        logger.info(f"Desired Traded ${desired_trade_dollars:.0f}")

        timeslice_df = shrink_floats(timeslice_df)
        check_dup_rows(timeslice_df)

        target_symbols = len(timeslice_df[timeslice_df['target_position'] != 0])
        logger.info(f"Target symbols: {target_symbols}")

        timeslice_df = self.calculate_trade_utility(timeslice_df)
        timeslice_df = self.calculate_expanding(timeslice_df)

        self.check_reversal(timeslice_df)

        return timeslice_df

    @staticmethod
    def check_reversal(df: pd.DataFrame) -> None:
        logger.info(f"contracting alpha trades...")
        if 'alpha_opt' not in df.columns:
            logger.info(f"alpha_opt not in df columns, probably server in between opts")
            return

        try:
            reversing_df = df[(df["position"].apply(np.sign) != df["alpha_opt"].apply(np.sign)) & (df['expanding'] == -1)]
            reversing_df = reversing_df.sort_values(by='position_abs', ascending=False)[['expanding', 'position', 'desired_trade_dollars', 'alpha_opt']].sort_values(by=['expanding', 'desired_trade_dollars'])
            logger.info(reversing_df)
        except Exception as e:
            logger.warning(f"Could not do reversal check {e}")

    def calculate_expanding(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[((df['position'] >= 0) & (df['desired_trade_dollars'] > 0)) |
               ((df['position'] <= 0) & (df['desired_trade_dollars'] < 0)), 'expanding'] = 1
        df.loc[((df['position'] > 0) & (df['desired_trade_dollars'] < 0)) |
               ((df['position'] < 0) & (df['desired_trade_dollars'] > 0)), 'expanding'] = -1
        df['expanding'] = df['expanding'].fillna(0).astype('Int32')
        return df

    def generate_desired_trades(self, df: pd.DataFrame, util_trades: bool = False) -> pd.DataFrame:
        """Generate final trade list with execution details.
        
        Converts target positions to trades by:
        - Calculating trade direction (expanding vs reducing)
        - Optionally prioritizing by utility (not currently used)
        - Computing trade quantities from dollar amounts
        
        Args:
            df: DataFrame with target positions and current positions
            util_trades: Whether to prioritize trades by utility (experimental)
            
        Returns:
            df with trade details added:
                - expanding: 1 if adding to position, -1 if reducing
                - desired_trade_qty: Number of units to trade
                
        Raises:
            RuntimeError: If util_trades=True (not fully implemented)
        """
        df = self.calculate_expanding(df)

        if util_trades:
            slice_fraction = self.opt_interval_mins / self.opt_horizon
            total_desired_dollars = df['desired_trade_dollars'].abs().sum()
            total_slice_dollars = total_desired_dollars * slice_fraction
            df['desired_utility_trade_dollars'] = np.float32(0)
            # Sort indices by priority (descending)
            sorted_indices = df['util_per_dollar'].sort_values(ascending=False).index
            # Allocate amounts based on priority of utility
            logger.info(f"Trading {slice_fraction=} {fmoney(total_slice_dollars)} in this interval...")
            remaining = total_slice_dollars
            for idx in sorted_indices:
                allocation = df.at[idx, 'desired_trade_dollars']
                util = df.at[idx, 'util_per_dollar']
                logger.info(f"Allocation {util=} {allocation} {remaining=}")
                df.at[idx, 'desired_utility_trade_dollars'] = allocation
                remaining -= abs(allocation)
                if remaining <= 0:
                    break
            df['desired_trade_dollars'] = df['desired_utility_trade_dollars']
            new_total_desired_dollars = df['desired_trade_dollars'].abs().sum()
            logger.info(f"After utility ranking trade dollars from {fmoney(total_desired_dollars)} -> {fmoney(new_total_desired_dollars)}")
            # don't use me yet
            raise RuntimeError()

        df['desired_trade_qty'] = df['desired_trade_dollars'] / df['close_mid']
        return df

    def make_alpha_opt(self, df: pd.DataFrame, horizons: Optional[List[int]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Combine multi-horizon alphas into optimization alpha.
        
        Aggregates alpha signals from different time horizons with:
        - Horizon-based scaling (longer horizons scaled down)
        - Clipping to maximum alpha values
        - Optional standardization and bounds
        - Mean-centering with optional tilt
        
        Also separates reversal and momentum components for analysis.
        
        Args:
            df: DataFrame with per-horizon alpha columns
            horizons: List of horizons to include (defaults to non-ST horizons)
            
        Returns:
            Tuple of:
                - df with alpha_opt, alpha_rev, alpha_mom columns added
                - List of new column names
                
        Raises:
            OptimizationError: If no valid alpha signals found
            
        Side Effects:
            Logs alpha statistics and warnings
        """
        new_cols = ['alpha_opt', 'alpha_rev', 'alpha_mom']
        for col in new_cols:
            df[col] = np.float32(0)

        if horizons is None:
            horizons = set(self.horizons) - set(self.short_term_model_horizons)
        logger.info(f"Making alpha_opt from {horizons}")
        for horizon in horizons:
            mult = 1.0
            if horizon > self.opt_horizon:
                mult = self.opt_horizon / horizon

            mult = mult * min(horizon * self.horizon_model_factor, 1.0)

            min_alpha = df[f'alpha_{horizon}'].min()
            max_alpha = df[f'alpha_{horizon}'].max()
            avg_alpha = df[f'alpha_{horizon}'].fillna(0).abs().mean() * mult
            if avg_alpha == 0:
                logger.warning(f"All alphas 0 at {horizon=}!")
                continue

            logger.info(f"Adding alpha at {horizon=} to alpha_opt with {mult=}, {avg_alpha=} {min_alpha=} {max_alpha=}")
            if min_alpha < -1.0 or max_alpha > 1:
                logger.error(f"Suspiciously big alphas {max_alpha}!")
                # raise RuntimeError()

            df['alpha_opt'] += df[f'alpha_{horizon}'].fillna(0) * mult
            df['alpha_rev'] += df[f"alpha_{horizon}_rev"] * mult
            df['alpha_mom'] += df[f"alpha_{horizon}_mom"] * mult

        if df['alpha_opt'].fillna(0).abs().sum() == 0:
            logger.error(f"No alpha to optimize on at {df.reset_index()['ts'].max()}", key="fail to generate alpha in optimization")
            raise OptimizationError()

        if self.alpha_mult is not None:
            df['alpha_opt'] *= self.alpha_mult

        min_raw_alpha = df['alpha_opt'].min()
        max_raw_alpha = df['alpha_opt'].max()
        logger.info(f"Raw alpha opt from {min_raw_alpha} - {max_raw_alpha}")

        uni_idx = df['alpha_opt'] != 0
        alpha_std = df[uni_idx]['alpha_opt'].std()
        alpha_median = df[uni_idx]['alpha_opt'].median()

        df['alpha_opt'] = df['alpha_opt'].clip(lower=-self.max_alpha, upper=self.max_alpha)
        df.loc[uni_idx, 'alpha_opt'] = (df['alpha_opt'] - alpha_median)

        if self.scale_alpha_opt:
            logger.info(f"Scaling alpha_opt by {alpha_std=}")
            df['alpha_opt'] = df['alpha_opt'] / alpha_std

        if self.alpha_std_bound is not None:
            alpha_bnd = self.alpha_std_bound * alpha_std
            df['alpha_opt'] = df['alpha_opt'].clip(lower=-alpha_bnd, upper=alpha_bnd)

        if self.alpha_tilt != 0:
            df['alpha_opt'] = df['alpha_opt'] + self.alpha_tilt

        return df, new_cols
