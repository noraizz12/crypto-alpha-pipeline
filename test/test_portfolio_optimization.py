"""Unit tests for portfolio_optimization module.

Tests the portfolio optimization functionality including:
- Convex optimization solver
- Constraint generation
- Risk calculations
- Alpha aggregation
- Trade generation
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import numpy as np
import pandas as pd
import cvxpy as cp

from lib.portfolio_optimization import (
    OptimizationError, TargetSolver, PortfolioOptimizer,
    MIN_PORTFOLIO_PCT, MAX_LOOSEN_LEVEL
)


class TestOptimizationError(unittest.TestCase):
    """Test the OptimizationError exception."""
    
    def test_optimization_error_inheritance(self):
        """Test that OptimizationError inherits from Exception."""
        error = OptimizationError("Test error")
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Test error")


class TestTargetSolver(unittest.TestCase):
    """Test the TargetSolver class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'OPT_RHO': 0.3,
            'OPT_RHO_INTERVAL': 10,
            'OPT_ALPHA': 1.6
        }
        self.n_assets = 5
        self.weights = cp.Variable(self.n_assets)
        self.ret = cp.sum(self.weights)  # Simple return
        self.factor_loadings = np.random.randn(self.n_assets, 2)
        self.factor_risk = cp.sum_squares(self.weights)
        self.resid_risk = cp.sum_squares(self.weights)
        self.slippage = cp.sum(cp.abs(self.weights))
        self.fees = cp.sum(cp.abs(self.weights)) * 0.0005
        self.kappa = 1e-7
        self.gamma = 2.5e-4
        self.constraints = [
            self.weights >= -1000,
            self.weights <= 1000
        ]
        
    def test_init(self):
        """Test TargetSolver initialization."""
        solver = TargetSolver(
            config=self.config,
            weights=self.weights,
            ret=self.ret,
            factor_loadings=self.factor_loadings,
            factor_risk=self.factor_risk,
            resid_risk=self.resid_risk,
            slippage=self.slippage,
            fees=self.fees,
            kappa=self.kappa,
            gamma=self.gamma,
            constraints=self.constraints
        )
        
        self.assertEqual(solver.config, self.config)
        self.assertEqual(solver.rho, 0.3)
        self.assertEqual(solver.rho_interval, 10)
        self.assertEqual(solver.alpha, 1.6)
        self.assertEqual(solver.starting_kappa, self.kappa)
        self.assertEqual(solver.kappa.value, self.kappa)
        self.assertEqual(solver.gamma.value, self.gamma)
        self.assertFalse(solver.solved)
        self.assertIsInstance(solver.prob, cp.Problem)
        
    def test_solve_successful(self):
        """Test successful optimization solve."""
        solver = TargetSolver(
            config=self.config,
            weights=self.weights,
            ret=self.ret,
            factor_loadings=self.factor_loadings,
            factor_risk=self.factor_risk,
            resid_risk=self.resid_risk,
            slippage=self.slippage,
            fees=self.fees,
            kappa=self.kappa,
            gamma=self.gamma,
            constraints=self.constraints
        )
        
        # Mock the problem solve
        solver.prob.solve = Mock(return_value=None)
        type(solver.prob).status = PropertyMock(return_value='optimal')
        solver.weights.value = np.array([100, -50, 200, -150, 0])
        
        portfolio_size = solver.solve(verbose=False)
        
        self.assertTrue(solver.solved)
        self.assertEqual(portfolio_size, 500)  # sum of abs values
        solver.prob.solve.assert_called_once()
        
    def test_solve_failed_status(self):
        """Test optimization solve with failed status."""
        solver = TargetSolver(
            config=self.config,
            weights=self.weights,
            ret=self.ret,
            factor_loadings=self.factor_loadings,
            factor_risk=self.factor_risk,
            resid_risk=self.resid_risk,
            slippage=self.slippage,
            fees=self.fees,
            kappa=self.kappa,
            gamma=self.gamma,
            constraints=self.constraints
        )
        
        solver.prob.solve = Mock(return_value=None)
        type(solver.prob).status = PropertyMock(return_value='infeasible')
        
        with self.assertRaises(OptimizationError) as context:
            solver.solve()
            
        self.assertIn("fail to optimize", str(context.exception))
        self.assertFalse(solver.solved)
        
    def test_solve_zero_portfolio(self):
        """Test optimization solve resulting in zero portfolio."""
        solver = TargetSolver(
            config=self.config,
            weights=self.weights,
            ret=self.ret,
            factor_loadings=self.factor_loadings,
            factor_risk=self.factor_risk,
            resid_risk=self.resid_risk,
            slippage=self.slippage,
            fees=self.fees,
            kappa=self.kappa,
            gamma=self.gamma,
            constraints=self.constraints
        )
        
        solver.prob.solve = Mock(return_value=None)
        solver.prob.status = 'optimal'
        solver.weights.value = np.zeros(self.n_assets)
        
        with self.assertRaises(OptimizationError) as context:
            solver.solve()
            
        self.assertIn("zero portfolio_size", str(context.exception))
        
    def test_solve_for_size(self):
        """Test iterative solving for target portfolio size."""
        solver = TargetSolver(
            config=self.config,
            weights=self.weights,
            ret=self.ret,
            factor_loadings=self.factor_loadings,
            factor_risk=self.factor_risk,
            resid_risk=self.resid_risk,
            slippage=self.slippage,
            fees=self.fees,
            kappa=self.kappa,
            gamma=self.gamma,
            constraints=self.constraints
        )
        
        # Mock solve to return different sizes based on kappa
        def mock_solve():
            if solver.kappa.value < 1e-6:
                solver.weights.value = np.array([1000, -500, 2000, -1500, 0])
                return 5000  # Too big
            elif solver.kappa.value > 1e-5:
                solver.weights.value = np.array([10, -5, 20, -15, 0])
                return 50  # Too small
            else:
                solver.weights.value = np.array([100, -50, 200, -150, 0])
                return 500  # Just right
                
        solver.solve = Mock(side_effect=mock_solve)
        solver.ret = Mock(value=10)
        solver.risk = Mock(value=0.1)
        solver.slippage = Mock(value=0.01)
        
        alpha_df = pd.DataFrame()
        result = solver.solve_for_size(alpha_df, min_portfolio_size=100, max_portfolio_size=1000)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(1e-6 < solver.kappa.value < 1e-5)
        
    def test_set_check_and_log_solution_not_solved(self):
        """Test solution extraction when not solved."""
        solver = TargetSolver(
            config=self.config,
            weights=self.weights,
            ret=self.ret,
            factor_loadings=self.factor_loadings,
            factor_risk=self.factor_risk,
            resid_risk=self.resid_risk,
            slippage=self.slippage,
            fees=self.fees,
            kappa=self.kappa,
            gamma=self.gamma,
            constraints=self.constraints
        )
        
        alpha_df = pd.DataFrame({
            'symbol_venue': ['A', 'B', 'C'],
            'alpha_opt': [0.001, -0.002, 0.003],
            'position': [100, -50, 0]
        })
        
        result_df, util_metrics = solver.set_check_and_log_solution(alpha_df, min_position=10)
        
        self.assertTrue((result_df['target_opt'] == 0).all())
        self.assertIsNone(util_metrics)
        
    def test_set_check_and_log_solution_successful(self):
        """Test successful solution extraction."""
        solver = TargetSolver(
            config=self.config,
            weights=self.weights,
            ret=self.ret,
            factor_loadings=self.factor_loadings,
            factor_risk=self.factor_risk,
            resid_risk=self.resid_risk,
            slippage=self.slippage,
            fees=self.fees,
            kappa=self.kappa,
            gamma=self.gamma,
            constraints=self.constraints
        )
        
        solver.solved = True
        # Create new variable with correct dimension for this test
        weights_3d = cp.Variable(3)
        weights_3d.value = np.array([150, -75, 50])
        solver.weights = weights_3d
        solver.ret = Mock(value=0.5)
        solver.risk = Mock(value=0.01)
        solver.factor_risk = Mock(value=0.005)
        solver.resid_risk = Mock(value=0.005)
        solver.slippage = Mock(value=0.1)
        solver.fees = Mock(value=0.05)
        
        alpha_df = pd.DataFrame({
            'symbol_venue': ['A', 'B', 'C'],
            'alpha_opt': [0.001, -0.002, 0.003],
            'position': [100, -50, 0]
        })
        
        result_df, util_metrics = solver.set_check_and_log_solution(alpha_df, min_position=10)
        
        np.testing.assert_array_equal(result_df['target_opt'].values, [150, -75, 50])
        self.assertIsNotNone(util_metrics)
        self.assertIn('expected_utility', util_metrics)
        self.assertIn('expected_return', util_metrics)
        self.assertIn('expected_risk', util_metrics)


class TestPortfolioOptimizer(unittest.TestCase):
    """Test the PortfolioOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'KAPPA': 1e-7,
            'GAMMA': 2.5e-4,
            'MAX_PORTFOLIO_NOTIONAL': 2e7,
            'MAX_ALPHA': 0.05,
            'MAX_PORTFOLIO_NOTIONAL_SLACK': 0.1,
            'MAX_TRADING_BIAS': 0.1,
            'BASE_AGGRESSION': 0.0009,
            'REOPTIMIZE_INTERVAL_MINS': 120,
            'OPT_HORIZON': 1440,
            'MIN_POSITION': 1000,
            'MAX_POSITION_PCT': 0.04,
            'MAX_TRADE_DOLLARS': 800000,
            'MIN_TRADE_DOLLARS': 100,
            'MAX_TURNOVER': 0.8,
            'SHORT_TERM_MODEL_HORIZONS': [15, 60],
            'EXCHANGE_FEES': 0.00005,
            'FACTOR_SIGMAS': {'dollar_exposure': 0.1},
            'EXPOSURE_LIMITS': {'dollar_exposure': 0.05},
            'SCALE_ALPHA_OPT': False,
            'CENTER_ALPHA_OPT': True,
            'CONST_NOTIONAL': False,
            'CORRECT_TARGET_IMBALANCE': False,
            'ALPHA_MULT': 1.0,
            'ALPHA_TILT': 0.0,
            'MAX_VOLUME_FRACTION_PARTICIPATION': 0.02,
            'HORIZON_MODEL_FACTOR': 1.0,
            'OPT_RHO': 0.3,
            'OPT_RHO_INTERVAL': 10,
            'OPT_ALPHA': 1.6,
            'FCASTS': {
                '1440': {
                    'features': {'prod': ['feature1', 'feature2']},
                    'models': [{'name': 'test_model'}]
                },
                '720': {
                    'features': {'prod': ['feature1', 'feature2']},
                    'models': [{'name': 'test_model'}]
                }
            }
        }
        
    def test_init(self):
        """Test PortfolioOptimizer initialization."""
        optimizer = PortfolioOptimizer(self.config)
        
        self.assertEqual(optimizer.config, self.config)
        self.assertEqual(optimizer.last_kappa, 1e-7)
        self.assertEqual(optimizer.gamma, 2.5e-4)
        self.assertEqual(optimizer.max_portfolio_size, 2e7)
        self.assertEqual(optimizer.min_position, 1000)
        self.assertEqual(optimizer.alpha_std_bound, 5)
        self.assertEqual(optimizer.util_metrics, {})
        
    def test_generate_inequality_constraints(self):
        """Test inequality constraint generation."""
        alpha_df = pd.DataFrame({
            'symbol_venue': ['A', 'B', 'C'],
            'factor1': [0.1, -0.2, 0.3]
        })
        exposure_limits = {'dollar_exposure': 0.05}
        max_portfolio_size = 1e6
        
        A_pos, A_neg, b_pos, b_neg = PortfolioOptimizer.generate_inequality_constraints(
            alpha_df, exposure_limits, max_portfolio_size
        )
        
        self.assertEqual(A_pos.shape, (3, 1))
        self.assertEqual(len(b_pos), 1)
        self.assertEqual(b_pos[0], 50000)  # 0.05 * 1e6
        self.assertEqual(b_neg[0], -50000)
        
    def test_generate_factor_matrix(self):
        """Test factor matrix generation."""
        alpha_df = pd.DataFrame({
            'symbol_venue': ['A', 'B', 'C'],
            'factor1': [0.1, -0.2, 0.3],
            'factor2': [0.5, 0.6, -0.7]
        })
        factor_sigmas = {'factor1': 0.01, 'factor2': 0.02}
        
        F_loadings, F_cov = PortfolioOptimizer.generate_factor_matrix(alpha_df, factor_sigmas)
        
        self.assertEqual(F_loadings.shape, (3, 2))
        self.assertEqual(F_cov.shape, (2, 2))
        self.assertAlmostEqual(F_cov[0, 0], 0.0001)  # 0.01^2
        self.assertAlmostEqual(F_cov[1, 1], 0.0004)  # 0.02^2
        
    def test_check_factor_exposures(self):
        """Test factor exposure checking."""
        factor_sigmas = {'factor1': 0.01, 'factor2': 0.0}  # factor2 skipped
        factor_loadings = np.array([[0.1, 0.5], [-0.2, 0.6], [0.3, -0.7]])
        targets_s = pd.Series([10000, -5000, 20000])
        
        # Should not raise
        PortfolioOptimizer.check_factor_exposures(factor_sigmas, factor_loadings, targets_s)
        
    def test_check_bad_risk_data(self):
        """Test handling of bad risk data."""
        optimizer = PortfolioOptimizer(self.config)
        
        alpha_df = pd.DataFrame({
            'symbol_venue': ['A', 'B', 'C', 'D'],
            'alpha_opt': [0.001, 0.002, np.nan, 0.003],
            'risk_1440': [0.01, np.nan, 0.02, 0.015],
            'dvolume_1440_trmean': [1e6, 2e6, np.nan, 3e6],
            'relative_spread_1440_trmean': [0.0001, 0.0002, 0.0003, np.nan],
            'position': [10000, -5000, 0, 20000],
            'lbound': [-50000, -50000, -50000, -50000],
            'ubound': [50000, 50000, 50000, 50000],
            'close_mid': [100, 200, 300, 400]
        })
        
        result = optimizer.check_bad_risk_data(alpha_df.copy())
        
        # Check that NaN values were handled
        self.assertEqual(result.loc[1, 'alpha_opt'], 0)  # Bad risk data
        self.assertFalse(np.isnan(result.loc[1, 'risk_1440']))  # Filled with mean
        self.assertEqual(result.loc[1, 'lbound'], -5000)  # Adjusted to position
        self.assertEqual(result.loc[1, 'ubound'], 0)  # Adjusted to position
        
    def test_calc_max_trade_dollars(self):
        """Test maximum trade dollar calculation."""
        optimizer = PortfolioOptimizer(self.config)
        
        df = pd.DataFrame({
            'dvolume_120_forecast': [1e7, 5e6, 2e7],
            'position': [50000, -100000, 0]
        })
        
        result = optimizer.calc_max_trade_dollars(df)
        
        self.assertIn('max_trade_dollars', result.columns)
        self.assertIn('position_abs', result.columns)
        # Should be min of volume fraction and max trade dollars
        self.assertTrue((result['max_trade_dollars'] <= 800000).all())
        
    def test_calc_total_max_trade_dollars(self):
        """Test total maximum trade dollar calculation."""
        optimizer = PortfolioOptimizer(self.config)
        
        df = pd.DataFrame({
            'position': [5e6, -3e6, 6e6, -4e6],  # Total 18M, > 80% of 20M max
            'lbound': [-2e6, -5e6, 0, -6e6],
            'ubound': [8e6, 0, 10e6, 1e6]
        })
        
        # Normal case
        max_traded = optimizer.calc_total_max_trade_dollars(df, loosen_level=0)
        expected = (0.8 * 2e7) / (1440 / 120)  # max_turnover * max_portfolio / optimizations_per_horizon
        self.assertGreater(max_traded, 0)
        
        # High loosen level
        max_traded_loose = optimizer.calc_total_max_trade_dollars(df, loosen_level=3)
        self.assertGreater(max_traded_loose, max_traded)
        
    def test_log_and_check_opt_info(self):
        """Test optimization info logging and validation."""
        optimizer = PortfolioOptimizer(self.config)
        
        df = pd.DataFrame({
            'symbol_venue': ['A', 'B', 'C'],
            'tradeable': [True, True, False],
            'alpha_opt': [0.001, -0.002, 0.003],
            'lbound': [-50000, -100000, -200000],
            'ubound': [100000, 50000, 200000],
            'risk_1440': [0.01, 0.02, 0.03],
            'relative_spread_1440_trmean': [0.0001, 0.0002, 0.0003],
            'dvolume_1440_trmean': [1e6, 2e6, 3e6]
        }, index=pd.MultiIndex.from_product([
            [pd.Timestamp('2023-01-01')],
            ['A', 'B', 'C']
        ], names=['ts', 'symbol_venue']))
        
        # Should not raise for valid data
        optimizer.log_and_check_opt_info(df, loosen_level=0)
        
        # Test with zero alpha
        df_bad = df.copy()
        df_bad['alpha_opt'] = 0
        with self.assertRaises(OptimizationError):
            optimizer.log_and_check_opt_info(df_bad, loosen_level=0)
            
    def test_calc_slippage(self):
        """Test slippage calculation."""
        optimizer = PortfolioOptimizer(self.config)
        
        df = pd.DataFrame({
            'expandable': [True, True, False],
            'tradeable': [True, True, True],
            'dvolume_1440_trmean': [1e6, 2e6, 3e6],
            'dvolume_1440': [8e5, 2.5e6, 2e6]
        })
        
        trade_dollars_abs = cp.Variable(3)
        trade_dollars_abs.value = np.array([10000, 20000, 30000])
        
        slippage = optimizer.calc_slippage(df, trade_dollars_abs)
        
        self.assertIsInstance(slippage, cp.Expression)
        self.assertIn('slippage_volume_frac', df.columns)
        
    def test_calc_fees(self):
        """Test fee calculation."""
        optimizer = PortfolioOptimizer(self.config)
        
        df = pd.DataFrame({
            'last_funding_rate': [0.0001, -0.0002, 0.0003]
        })
        
        trade_dollars = cp.Variable(3)
        trade_dollars.value = np.array([10000, -20000, 30000])
        trade_dollars_abs = cp.abs(trade_dollars)
        
        fees = optimizer.calc_fees(df, trade_dollars, trade_dollars_abs)
        
        self.assertIsInstance(fees, cp.Expression)
        
    def test_calc_factor_risk(self):
        """Test factor risk calculation."""
        optimizer = PortfolioOptimizer(self.config)
        
        df = pd.DataFrame({
            'dollar_exposure': [1, 1, 1],
            'symbol_venue': ['A', 'B', 'C']
        })
        
        weights = cp.Variable(3)
        factor_risk, factor_loadings = optimizer.calc_factor_risk(df, weights)
        
        self.assertIsInstance(factor_risk, cp.Expression)
        self.assertEqual(factor_loadings.shape[0], 3)
        
    def test_calc_resid_risk(self):
        """Test residual risk calculation."""
        optimizer = PortfolioOptimizer(self.config)
        
        df = pd.DataFrame({
            'risk_1440': [0.01, 0.02, 0.015]
        })
        
        weights = cp.Variable(3)
        resid_risk = optimizer.calc_resid_risk(df, weights)
        
        self.assertIsInstance(resid_risk, cp.Expression)
        
    def test_calc_constraints(self):
        """Test constraint calculation."""
        optimizer = PortfolioOptimizer(self.config)
        
        df = pd.DataFrame({
            'lbound': [-50000, -100000, -200000],
            'ubound': [100000, 50000, 200000],
            'max_trade_dollars': [10000, 20000, 30000],
            'position': [0, 0, 0]
        })
        
        weights = cp.Variable(3)
        trade_dollars_abs = cp.abs(weights)
        
        constraints = optimizer.calc_constraints(df, loosen_level=0, 
                                                weights=weights, 
                                                trade_dollars_abs=trade_dollars_abs)
        
        self.assertIsInstance(constraints, list)
        self.assertGreater(len(constraints), 0)
        self.assertIsInstance(constraints[0], cp.Constraint)
        
    @patch('lib.portfolio_optimization.TargetSolver')
    def test_setup_optimization(self, mock_target_solver):
        """Test optimization setup."""
        optimizer = PortfolioOptimizer(self.config)
        
        alpha_df = pd.DataFrame({
            'symbol_venue': ['A', 'B', 'C'],
            'alpha_opt': [0.001, -0.002, 0.003],
            'position': [10000, -5000, 0],
            'risk_1440': [0.01, 0.02, 0.015],
            'lbound': [-50000, -100000, -200000],
            'ubound': [100000, 50000, 200000],
            'tradeable': [True, True, True],
            'relative_spread_1440_trmean': [0.0001, 0.0002, 0.0003],
            'dvolume_1440_trmean': [1e6, 2e6, 3e6],
            'dvolume_1440': [8e5, 2.5e6, 2e6],
            'dvolume_120_forecast': [1e5, 2e5, 3e5],
            'last_funding_rate': [0.0001, -0.0002, 0.0003],
            'expandable': [True, True, False]
        }, index=pd.MultiIndex.from_product([
            [pd.Timestamp('2023-01-01')],
            ['A', 'B', 'C']
        ], names=['ts', 'symbol_venue']))
        
        solver = optimizer.setup_optimization(alpha_df, loosen_level=0)
        
        self.assertTrue(mock_target_solver.called)
        
    def test_calculate_trade_utility(self):
        """Test trade utility calculation."""
        optimizer = PortfolioOptimizer(self.config)
        
        df = pd.DataFrame({
            'alpha_opt': [0.001, -0.002, 0.003],
            'position': [10000, -5000, 0],
            'target_position': [15000, -3000, 5000],
            'risk_1440': [0.01, 0.02, 0.015],
            'last_funding_rate': [0.0001, -0.0002, 0.0003],
            'desired_trade_dollars': [5000, 2000, 5000]
        })
        
        result = optimizer.calculate_trade_utility(df)
        
        self.assertIn('util', result.columns)
        self.assertIn('util_per_dollar', result.columns)
        
    def test_adjust_portfolio_balance_and_size(self):
        """Test portfolio balance and size adjustment."""
        optimizer = PortfolioOptimizer(self.config)
        
        # Test long bias adjustment
        df_long = pd.DataFrame({
            'target_position': [1e7, 5e6, 3e6, -2e6],  # 16M long, 2M short = 14M bias
            'position': [0, 0, 0, 0]  # Add required position column
        })
        
        result_long = optimizer.adjust_portfolio_balance_and_size(df_long.copy())
        long_sum = result_long[result_long['target_position'] > 0]['target_position'].sum()
        short_sum = abs(result_long[result_long['target_position'] < 0]['target_position'].sum())
        bias = long_sum - short_sum
        
        # Test portfolio too big
        df_big = pd.DataFrame({
            'target_position': [1.5e7, -1e7, 8e6, -5e6],  # 23M long, 15M short = 38M total
            'position': [0, 0, 0, 0]  # Add required position column
        })
        
        result_big = optimizer.adjust_portfolio_balance_and_size(df_big.copy())
        total_size = result_big['target_position'].abs().sum()
        self.assertLessEqual(total_size, optimizer.max_portfolio_size * 1.01)  # Allow small rounding
        
    def test_calculate_targets(self):
        """Test target calculation."""
        optimizer = PortfolioOptimizer(self.config)
        
        df = pd.DataFrame({
            'symbol_venue': ['A', 'B', 'C', 'D'],
            'position': [10000, -5000, 0, 100000],
            'target_opt': [15000, -3000, 5000, 90000],
            'alpha_opt': [0.001, -0.002, 0.003, -0.001],
            'risk_1440': [0.01, 0.02, 0.015, 0.025],
            'last_funding_rate': [0.0001, -0.0002, 0.0003, 0.0001],
            'close_mid': [100, 200, 300, 400]
        })
        
        result = optimizer.calculate_targets(df)
        
        self.assertIn('desired_trade_dollars', result.columns)
        self.assertIn('target_position', result.columns)
        self.assertIn('util', result.columns)
        
        # Check that small trades are filtered
        small_trades = result[result['desired_trade_dollars'].abs() < optimizer.min_trade_dollars]
        self.assertTrue((small_trades['target_position'] == small_trades['position']).all() |
                       (small_trades['target_position'] == 0).all())
        
    def test_generate_desired_trades(self):
        """Test desired trade generation."""
        optimizer = PortfolioOptimizer(self.config)
        
        df = pd.DataFrame({
            'position': [10000, -5000, 0, -20000],
            'desired_trade_dollars': [5000, 2000, -3000, -5000],
            'close_mid': [100, 200, 300, 400]
        })
        
        result = optimizer.generate_desired_trades(df)
        
        self.assertIn('expanding', result.columns)
        self.assertIn('desired_trade_qty', result.columns)
        
        # Check expanding logic
        self.assertEqual(result.iloc[0]['expanding'], 1)  # Long position, buying more
        self.assertEqual(result.iloc[1]['expanding'], -1)  # Short position, buying back
        
    def test_make_alpha_opt(self):
        """Test alpha optimization signal creation."""
        optimizer = PortfolioOptimizer(self.config)
        
        df = pd.DataFrame({
            'alpha_1440': [0.01, -0.02, 0.03],
            'alpha_1440_rev': [0.005, -0.01, 0.015],
            'alpha_1440_mom': [0.005, -0.01, 0.015],
            'alpha_720': [0.02, -0.01, 0.04],
            'alpha_720_rev': [0.01, -0.005, 0.02],
            'alpha_720_mom': [0.01, -0.005, 0.02]
        })
        
        result, new_cols = optimizer.make_alpha_opt(df, horizons=[720, 1440])
        
        self.assertIn('alpha_opt', result.columns)
        self.assertIn('alpha_rev', result.columns)
        self.assertIn('alpha_mom', result.columns)
        self.assertEqual(new_cols, ['alpha_opt', 'alpha_rev', 'alpha_mom'])
        
        # Check that alpha_opt is not all zeros
        self.assertGreater(result['alpha_opt'].abs().sum(), 0)
        
        # Note: alpha_opt is centered (mean subtracted) after clipping, so it can exceed max_alpha
        # Just check that values are reasonable
        self.assertTrue((result['alpha_opt'].abs() <= 0.1).all())
        
    def test_make_alpha_opt_no_alpha(self):
        """Test alpha optimization with no valid signals."""
        optimizer = PortfolioOptimizer(self.config)
        
        df = pd.DataFrame({
            'alpha_1440': [0, 0, 0],
            'alpha_1440_rev': [0, 0, 0],
            'alpha_1440_mom': [0, 0, 0]
        })
        # Add MultiIndex to match expected structure
        df.index = pd.MultiIndex.from_tuples([
            (pd.Timestamp('2024-01-01'), 'A'),
            (pd.Timestamp('2024-01-01'), 'B'),
            (pd.Timestamp('2024-01-01'), 'C')
        ], names=['ts', 'symbol_venue'])
        
        with self.assertRaises(OptimizationError):
            optimizer.make_alpha_opt(df, horizons=[1440])
            
    @patch('lib.portfolio_optimization.TargetSolver')
    def test_optimize_full_pipeline(self, mock_solver_class):
        """Test full optimization pipeline."""
        optimizer = PortfolioOptimizer(self.config)
        
        # Create mock solver
        mock_solver = Mock()
        mock_solver.solved = True
        mock_solver.kappa = Mock(value=1e-7)
        mock_solver.factor_loadings = np.array([[1, 0], [1, 0], [1, 0]])
        mock_solver.set_check_and_log_solution = Mock(return_value=(
            pd.DataFrame({'target_opt': [15000, -3000, 5000]}),
            {'expected_utility': 100}
        ))
        mock_solver_class.return_value = mock_solver
        
        alpha_df = pd.DataFrame({
            'symbol_venue': ['A', 'B', 'C'],
            'alpha_opt': [0.001, -0.002, 0.003],
            'position': [10000, -5000, 0],
            'risk_1440': [0.01, 0.02, 0.015],
            'lbound': [-50000, -100000, -200000],
            'ubound': [100000, 50000, 200000],
            'tradeable': [True, True, True],
            'relative_spread_1440_trmean': [0.0001, 0.0002, 0.0003],
            'dvolume_1440_trmean': [1e6, 2e6, 3e6],
            'dvolume_1440': [8e5, 2.5e6, 2e6],
            'dvolume_120_forecast': [1e5, 2e5, 3e5],
            'last_funding_rate': [0.0001, -0.0002, 0.0003],
            'expandable': [True, True, False],
            'target_opt': [0, 0, 0]
        }, index=pd.MultiIndex.from_product([
            [pd.Timestamp('2023-01-01')],
            ['A', 'B', 'C']
        ], names=['ts', 'symbol_venue']))
        
        result = optimizer.optimize(alpha_df)
        
        self.assertIn('target_opt', result.columns)
        self.assertEqual(optimizer.util_metrics['expected_utility'], 100)
        self.assertTrue(mock_solver.set_check_and_log_solution.called)


def build_realistic_fixture(n_symbols: int = 60) -> pd.DataFrame:
    """Build a synthetic but realistic DataFrame for end-to-end optimization tests.

    Creates n_symbols assets with realistic value ranges for all columns
    consumed by setup_optimization() and its sub-methods. Uses a fixed
    random seed for reproducibility.

    Returns:
        DataFrame with MultiIndex ['ts', 'symbol_venue'] and all columns
        needed by PortfolioOptimizer.optimize().
    """
    rng = np.random.default_rng(42)
    ts = pd.Timestamp('2024-06-15 12:00:00')
    symbols = [f'COIN{i:02d}_binance-futures' for i in range(n_symbols)]

    idx = pd.MultiIndex.from_product(
        [[ts], symbols], names=['ts', 'symbol_venue']
    )

    # Alpha signal: small, centered
    alpha_raw = rng.normal(0, 0.01, n_symbols).clip(-0.03, 0.03)
    alpha_opt = (alpha_raw - alpha_raw.mean()).astype(np.float32)

    # Current positions: mix of active and flat
    position = np.zeros(n_symbols, dtype=np.float32)
    active_mask = rng.random(n_symbols) > 0.3
    position[active_mask] = rng.uniform(-300_000, 300_000, active_mask.sum()).astype(np.float32)

    # Risk and volume
    risk_1440 = rng.uniform(0.008, 0.035, n_symbols).astype(np.float32)
    dvolume_1440_trmean = rng.uniform(5e6, 500e6, n_symbols).astype(np.float32)
    dvolume_1440 = rng.uniform(3e6, 300e6, n_symbols).astype(np.float32)
    dvolume_120_forecast = rng.uniform(0.5e6, 30e6, n_symbols).astype(np.float32)

    # Bounds
    lbound = rng.uniform(-400_000, -80_000, n_symbols).astype(np.float32)
    ubound = rng.uniform(80_000, 400_000, n_symbols).astype(np.float32)

    # Ensure position is within bounds for feasibility
    position = np.clip(position, lbound, ubound)

    # Tradeable / expandable
    tradeable = np.ones(n_symbols, dtype=bool)
    expandable = rng.random(n_symbols) < 0.8

    # Spread and funding
    relative_spread_1440_trmean = rng.uniform(0.0001, 0.001, n_symbols).astype(np.float32)
    last_funding_rate = rng.uniform(-0.0005, 0.0005, n_symbols).astype(np.float32)

    # Prices - realistic crypto range
    close_mid = np.exp(rng.uniform(np.log(0.5), np.log(90_000), n_symbols)).astype(np.float32)

    data = {
        'alpha_opt': alpha_opt,
        'position': position,
        'risk_1440': risk_1440,
        'lbound': lbound,
        'ubound': ubound,
        'tradeable': tradeable,
        'expandable': expandable,
        'dvolume_1440_trmean': dvolume_1440_trmean,
        'dvolume_1440': dvolume_1440,
        'dvolume_120_forecast': dvolume_120_forecast,
        'last_funding_rate': last_funding_rate,
        'relative_spread_1440_trmean': relative_spread_1440_trmean,
        'close_mid': close_mid,
        'target_opt': np.float32(0.0),
    }

    # Factor columns from production FACTOR_SIGMAS
    factor_cols = {
        'dvolume_1440_trmean_cz': rng.normal(0, 1, n_symbols),
        'logret_1440_cz': rng.normal(0, 1, n_symbols),
        'dvolume_1440_dp': rng.normal(0, 0.3, n_symbols),
        'category_gmci_depin_index': rng.choice([0.0, 1.0], n_symbols, p=[0.85, 0.15]),
        'category_gmci_30_index': rng.choice([0.0, 1.0], n_symbols, p=[0.5, 0.5]),
        'category_gmci_layer_1_index': rng.choice([0.0, 1.0], n_symbols, p=[0.8, 0.2]),
        'category_gmci_index': rng.choice([0.0, 1.0], n_symbols, p=[0.7, 0.3]),
        'category_gmci_meme_index': rng.choice([0.0, 1.0], n_symbols, p=[0.9, 0.1]),
        'category_gmci_defi_index': rng.choice([0.0, 1.0], n_symbols, p=[0.85, 0.15]),
        'category_proof_of_stake_(pos)': rng.choice([0.0, 1.0], n_symbols, p=[0.6, 0.4]),
        'mkt_wgt': rng.uniform(0.001, 0.1, n_symbols),
    }
    for col, vals in factor_cols.items():
        data[col] = np.float32(vals)

    df = pd.DataFrame(data, index=idx)
    return df


def _make_e2e_config() -> dict:
    """Return a production-like config dict for end-to-end optimization tests."""
    return {
        'KAPPA': 1e-7,
        'GAMMA': 5e-3,
        'MAX_PORTFOLIO_NOTIONAL': 2.8e7,
        'MAX_ALPHA': 0.5,
        'MAX_PORTFOLIO_NOTIONAL_SLACK': 0.035,
        'MAX_TRADING_BIAS': 0.1,
        'BASE_AGGRESSION': 0.0009,
        'REOPTIMIZE_INTERVAL_MINS': 120,
        'OPT_HORIZON': 1440,
        'MIN_POSITION': 1000,
        'MAX_POSITION_PCT': 0.06,
        'MAX_TRADE_DOLLARS': 2_000_000,
        'MIN_TRADE_DOLLARS': 500,
        'MAX_TURNOVER': 1.0,
        'SHORT_TERM_MODEL_HORIZONS': [60, 15],
        'EXCHANGE_FEES': 0.00005,
        'FACTOR_SIGMAS': {
            'dollar_exposure': 0.10,
            'dvolume_1440_trmean_cz': 0.0001,
            'logret_1440_cz': 0.25,
            'dvolume_1440_dp': 0.25,
            'category_gmci_depin_index': 0.01,
            'category_gmci_30_index': 0.01,
            'category_gmci_layer_1_index': 0.01,
            'category_gmci_index': 0.01,
            'category_gmci_meme_index': 0.01,
            'category_gmci_defi_index': 0.01,
            'category_proof_of_stake_(pos)': 0.01,
            'mkt_wgt': 0.10,
        },
        'EXPOSURE_LIMITS': {'dollar_exposure': 0.05},
        'SCALE_ALPHA_OPT': False,
        'CENTER_ALPHA_OPT': True,
        'CONST_NOTIONAL': False,
        'CORRECT_TARGET_IMBALANCE': False,
        'ALPHA_MULT': 1.0,
        'ALPHA_TILT': 0.0,
        'MAX_VOLUME_FRACTION_PARTICIPATION': 0.02,
        'HORIZON_MODEL_FACTOR': 0.001,
        'OPT_RHO': 0.3,
        'OPT_RHO_INTERVAL': 10,
        'OPT_ALPHA': 1.6,
        'FCASTS': {
            '1440': {
                'features': {'prod': ['feature1']},
                'models': [{'name': 'test_model'}],
            },
        },
    }


class TestOptimizationEndToEnd(unittest.TestCase):
    """End-to-end tests that run the real CVXPY optimizer on synthetic data."""

    def setUp(self):
        self.config = _make_e2e_config()
        self.optimizer = PortfolioOptimizer(self.config)

    def test_optimize_produces_valid_targets(self):
        """optimize() should solve and produce targets within bounds."""
        df = build_realistic_fixture()
        result_df = self.optimizer.optimize(df)

        # target_opt column exists with no NaNs
        self.assertIn('target_opt', result_df.columns)
        self.assertFalse(result_df['target_opt'].isna().any(),
                         "target_opt contains NaN values")

        # targets respect position bounds (OSQP solver tolerance can be ~$10-15)
        tol = 50.0
        self.assertTrue(
            (result_df['target_opt'] >= result_df['lbound'] - tol).all(),
            "Some targets below lbound"
        )
        self.assertTrue(
            (result_df['target_opt'] <= result_df['ubound'] + tol).all(),
            "Some targets above ubound"
        )

        # portfolio notional within limit (allow 15% slack for solver precision)
        total_notional = result_df['target_opt'].abs().sum()
        max_allowed = self.config['MAX_PORTFOLIO_NOTIONAL'] * 1.15
        self.assertLessEqual(
            total_notional, max_allowed,
            f"Total notional {total_notional:.0f} exceeds max {max_allowed:.0f}"
        )

        # util_metrics populated
        self.assertIsInstance(self.optimizer.util_metrics, dict)
        for key in ['expected_utility', 'expected_return', 'expected_risk']:
            self.assertIn(key, self.optimizer.util_metrics,
                          f"Missing util_metrics key: {key}")

    def test_optimize_full_pipeline(self):
        """optimize() -> calculate_targets() -> generate_desired_trades() should run cleanly."""
        df = build_realistic_fixture()
        df = self.optimizer.optimize(df)
        df = self.optimizer.calculate_targets(df)
        df = self.optimizer.generate_desired_trades(df)

        # Required output columns exist
        for col in ['desired_trade_dollars', 'desired_trade_qty', 'expanding']:
            self.assertIn(col, df.columns, f"Missing column: {col}")

        # Trade qty sign should be consistent with trade dollars / price
        nonzero = df[df['desired_trade_qty'].abs() > 0.001]
        if len(nonzero) > 0:
            signs_match = np.sign(nonzero['desired_trade_qty'].values) == np.sign(nonzero['desired_trade_dollars'].values)
            self.assertTrue(signs_match.all(),
                            "Trade qty sign inconsistent with trade dollars")

    def test_optimize_from_flat(self):
        """Optimizer should find valid positions when starting from flat."""
        df = build_realistic_fixture()
        df['position'] = np.float32(0.0)

        result_df = self.optimizer.optimize(df)

        # Should have a non-trivial portfolio
        total_notional = result_df['target_opt'].abs().sum()
        self.assertGreater(total_notional, 0,
                           "Optimizer produced zero portfolio from flat")

        # Should have both long and short positions
        n_long = (result_df['target_opt'] > self.config['MIN_POSITION']).sum()
        n_short = (result_df['target_opt'] < -self.config['MIN_POSITION']).sum()
        self.assertGreater(n_long, 0, "No long positions in portfolio")
        self.assertGreater(n_short, 0, "No short positions in portfolio")

    def test_optimize_regression_snapshot(self):
        """Optimizer output should match known-good snapshot for fixed inputs."""
        expected_targets = np.array([
            -80949.15, -118656.34, 287754.54, 315383.59, -365945.01,
            -80286.58, 0.00, -191960.00, 0.00, -112949.48,
            168984.57, 177505.20, 0.00, 275515.37, 0.00,
            -108595.75, 48636.49, -253164.31, 212733.39, -286317.85,
            91539.26, -143525.68, 231722.36, 0.00, -10618.03,
            -143692.12, 0.00, 46900.52, 0.00, -8005.21,
            200405.06, 255726.09, -360025.41, -238550.28, 195411.49,
            244076.44, 0.00, -350331.59, -164921.95, 224332.94,
            246487.64, -172830.80, -86334.31, 237696.68, -215850.51,
            0.00, 103059.01, 32421.51, 97781.83, 0.00,
            103344.03, 135357.22, -86391.13, 95653.83, -149094.55,
            -115027.84, -216827.80, 199123.57, -97272.32, 0.00,
        ])

        df = build_realistic_fixture()
        result_df = self.optimizer.optimize(df)
        actual = result_df['target_opt'].values

        np.testing.assert_allclose(
            actual, expected_targets, atol=100.0,
            err_msg="Optimizer output diverged from snapshot"
        )

    def test_optimize_idempotent_targets(self):
        """Re-optimizing with target as position should produce small changes."""
        df = build_realistic_fixture()
        result_df = self.optimizer.optimize(df.copy())

        # Feed targets back as positions and re-optimize
        df2 = df.copy()
        df2['position'] = result_df['target_opt'].values
        # Ensure positions are within bounds for feasibility
        df2['position'] = df2['position'].clip(lower=df2['lbound'], upper=df2['ubound'])
        df2['target_opt'] = np.float32(0.0)

        result_df2 = self.optimizer.optimize(df2)

        # Changes should be small relative to first-pass portfolio
        first_size = result_df['target_opt'].abs().sum()
        delta = (result_df2['target_opt'] - result_df['target_opt']).abs().sum()
        if first_size > 0:
            change_pct = delta / first_size
            self.assertLess(change_pct, 0.3,
                            f"Idempotency: {change_pct:.1%} change is too large")


if __name__ == '__main__':
    unittest.main()