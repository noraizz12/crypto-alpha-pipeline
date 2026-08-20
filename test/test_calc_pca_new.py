"""Tests for calculate_pca function in calc_risk.py.

This module tests the PCA-based residualized returns calculation, verifying:
1. Correct mathematical computation of factor scores and residuals
2. Eigenvalue-based component filtering
3. Proper handling of production vs non-production modes
4. Edge cases like insufficient data
5. Statistical properties of residuals (near-zero mean correlation)
"""

import unittest
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from lib.calcs.calc_risk import calculate_pca


def create_pca_test_data(
    n_periods: int = 200,
    n_symbols: int = 10,
    frequency: str = '1h',
    seed: int = 42
) -> pd.DataFrame:
    """Create synthetic test data with known factor structure.

    Creates returns data where:
    - First factor explains ~50% of variance (market factor)
    - Second factor explains ~20% (sector factor)
    - Remaining is idiosyncratic noise

    Args:
        n_periods: Number of time periods
        n_symbols: Number of symbols
        frequency: Pandas frequency string
        seed: Random seed for reproducibility

    Returns:
        DataFrame with multi-index (ts, symbol_venue) and logret column
    """
    np.random.seed(seed)

    dates = pd.date_range('2024-01-01', periods=n_periods, freq=frequency, tz='UTC')
    symbols = [f'SYM{i}USDT_binance-futures' for i in range(n_symbols)]

    # Create factor structure
    # Factor 1: Market factor (all symbols have positive loading ~0.7-1.0)
    market_factor = np.random.randn(n_periods) * 0.02  # Market returns
    market_loadings = np.random.uniform(0.7, 1.0, n_symbols)

    # Factor 2: Sector factor (half positive, half negative loading)
    sector_factor = np.random.randn(n_periods) * 0.01
    sector_loadings = np.array([1.0 if i < n_symbols // 2 else -1.0 for i in range(n_symbols)])
    sector_loadings *= np.random.uniform(0.3, 0.5, n_symbols)

    # Idiosyncratic returns (should be uncorrelated across symbols)
    idio_returns = np.random.randn(n_periods, n_symbols) * 0.005

    # Combine: total_return = market_loading * market + sector_loading * sector + idio
    returns_matrix = (
        np.outer(market_factor, market_loadings) +
        np.outer(sector_factor, sector_loadings) +
        idio_returns
    )

    # Build dataframe
    index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])

    # Flatten returns in the right order (time-major)
    returns_flat = returns_matrix.flatten(order='C')

    df = pd.DataFrame({
        'logret_60': returns_flat.astype(np.float32),
        'logret_1440': returns_flat.astype(np.float32),  # Same for daily
    }, index=index)

    return df


class TestCalculatePcaNewBasic(unittest.TestCase):
    """Basic functionality tests for calculate_pca."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = create_pca_test_data(n_periods=200, n_symbols=10, frequency='1h')

    def test_returns_correct_columns_non_prod(self):
        """Test that correct columns are created in non-prod mode."""
        frequency = 60
        start_dt = pd.Timestamp('2024-01-05 00:00:00', tz='UTC')

        result_df, new_cols = calculate_pca(
            self.df.copy(),
            frequency=frequency,
            start_dt=start_dt,
            lookback_periods=90,
            prod=False
        )

        # Check expected columns
        expected_cols = [
            f'pca_resid_{frequency}',
            f'pca_factor_ret_{frequency}',
            f'pca_n_components_{frequency}',
            f'pca_explained_var_{frequency}'
        ]

        self.assertEqual(set(new_cols), set(expected_cols))

        # Verify columns exist in dataframe
        for col in expected_cols:
            self.assertIn(col, result_df.columns)

    def test_returns_correct_columns_prod(self):
        """Test that only residual column is created in prod mode."""
        frequency = 60

        result_df, new_cols = calculate_pca(
            self.df.copy(),
            frequency=frequency,
            prod=True,
            lookback_periods=90
        )

        # In prod mode, only residual column
        self.assertEqual(new_cols, [f'pca_resid_{frequency}'])
        self.assertIn(f'pca_resid_{frequency}', result_df.columns)

    def test_residuals_have_correct_dtype(self):
        """Test that residuals are float32."""
        frequency = 60
        start_dt = pd.Timestamp('2024-01-05 00:00:00', tz='UTC')

        result_df, _ = calculate_pca(
            self.df.copy(),
            frequency=frequency,
            start_dt=start_dt,
            lookback_periods=90,
            prod=False
        )

        resid_col = f'pca_resid_{frequency}'
        self.assertEqual(result_df[resid_col].dtype, np.float32)

    def test_residuals_bounded(self):
        """Test that residuals are reasonably bounded."""
        frequency = 60
        start_dt = pd.Timestamp('2024-01-05 00:00:00', tz='UTC')

        result_df, _ = calculate_pca(
            self.df.copy(),
            frequency=frequency,
            start_dt=start_dt,
            lookback_periods=90,
            prod=False
        )

        resid_col = f'pca_resid_{frequency}'
        residuals = result_df[resid_col].dropna()

        # Residuals should be smaller than original returns in most cases
        # Since we're removing systematic component
        self.assertTrue(len(residuals) > 0)

        # Residuals should be bounded (not exploding)
        self.assertTrue(residuals.abs().max() < 1.0)  # Less than 100% return


class TestCalculatePcaNewMathematical(unittest.TestCase):
    """Mathematical correctness tests for calculate_pca."""

    def setUp(self):
        """Set up test fixtures with known factor structure."""
        self.df = create_pca_test_data(n_periods=300, n_symbols=20, frequency='1h')

    def test_residuals_near_zero_mean(self):
        """Test that residuals have near-zero cross-sectional mean."""
        frequency = 60
        start_dt = pd.Timestamp('2024-01-05 00:00:00', tz='UTC')

        result_df, _ = calculate_pca(
            self.df.copy(),
            frequency=frequency,
            start_dt=start_dt,
            lookback_periods=90,
            prod=False
        )

        resid_col = f'pca_resid_{frequency}'

        # Get cross-sectional mean at each timestamp
        resid_unstacked = result_df[resid_col].unstack()
        cs_means = resid_unstacked.mean(axis=1).dropna()

        # Cross-sectional mean should be near zero
        self.assertAlmostEqual(cs_means.mean(), 0.0, places=3)

    def test_residuals_lower_correlation_than_returns(self):
        """Test that residuals have lower pairwise correlation than original returns."""
        frequency = 60
        start_dt = pd.Timestamp('2024-01-05 00:00:00', tz='UTC')

        df_copy = self.df.copy()
        result_df, _ = calculate_pca(
            df_copy,
            frequency=frequency,
            start_dt=start_dt,
            lookback_periods=90,
            prod=False
        )

        ret_col = f'logret_{frequency}'
        resid_col = f'pca_resid_{frequency}'

        # Calculate mean pairwise correlation of original returns
        orig_unstacked = result_df[ret_col].unstack()
        orig_corr = orig_unstacked.corr()
        n = len(orig_corr)
        triu_idx = np.triu_indices(n, k=1)
        orig_mean_corr = orig_corr.values[triu_idx].mean()

        # Calculate mean pairwise correlation of residuals
        resid_unstacked = result_df[resid_col].unstack()
        resid_corr = resid_unstacked.corr()
        resid_mean_corr = resid_corr.values[triu_idx].mean()

        # Residuals should have lower correlation
        self.assertLess(abs(resid_mean_corr), abs(orig_mean_corr))

    def test_variance_decomposition(self):
        """Test that systematic + residual variance approximates total variance."""
        frequency = 60
        start_dt = pd.Timestamp('2024-01-05 00:00:00', tz='UTC')

        df_copy = self.df.copy()
        result_df, _ = calculate_pca(
            df_copy,
            frequency=frequency,
            start_dt=start_dt,
            lookback_periods=90,
            prod=False
        )

        ret_col = f'logret_{frequency}'
        resid_col = f'pca_resid_{frequency}'

        # Calculate variances
        returns = result_df[ret_col].dropna()
        residuals = result_df[resid_col].dropna()

        # Systematic variance = total - residual (approximately)
        total_var = returns.var()
        resid_var = residuals.var()

        # Residual variance should be less than total variance
        self.assertLess(resid_var, total_var)

        # Explained variance ratio should be positive
        explained_ratio = 1 - resid_var / total_var
        self.assertGreater(explained_ratio, 0)

    def test_component_filtering(self):
        """Test that eigenvalue threshold correctly filters components."""
        frequency = 60
        start_dt = pd.Timestamp('2024-01-05 00:00:00', tz='UTC')

        # Test with high threshold (should keep fewer components)
        result_high, _ = calculate_pca(
            self.df.copy(),
            frequency=frequency,
            start_dt=start_dt,
            lookback_periods=90,
            prod=False,
            eigenvalue_threshold=0.10  # 10% threshold
        )

        # Test with low threshold (should keep more components)
        result_low, _ = calculate_pca(
            self.df.copy(),
            frequency=frequency,
            start_dt=start_dt,
            lookback_periods=90,
            prod=False,
            eigenvalue_threshold=0.01  # 1% threshold
        )

        n_comp_high_col = f'pca_n_components_{frequency}'
        n_comp_high = result_high[n_comp_high_col].dropna().iloc[-1]
        n_comp_low = result_low[n_comp_high_col].dropna().iloc[-1]

        # Higher threshold should result in fewer components
        self.assertLessEqual(n_comp_high, n_comp_low)


class TestCalculatePcaNewEdgeCases(unittest.TestCase):
    """Edge case tests for calculate_pca."""

    def test_insufficient_data_warning(self):
        """Test handling of insufficient historical data."""
        # Create very short data
        df = create_pca_test_data(n_periods=20, n_symbols=50, frequency='1h')

        frequency = 60
        start_dt = pd.Timestamp('2024-01-01 05:00:00', tz='UTC')

        # Should not crash, just return with warnings
        result_df, new_cols = calculate_pca(
            df.copy(),
            frequency=frequency,
            start_dt=start_dt,
            lookback_periods=90,  # More than available data
            prod=False
        )

        # Function should still return the dataframe
        self.assertIsNotNone(result_df)

    def test_single_symbol_handling(self):
        """Test handling of single symbol (should fail gracefully)."""
        df = create_pca_test_data(n_periods=200, n_symbols=1, frequency='1h')

        frequency = 60
        start_dt = pd.Timestamp('2024-01-05 00:00:00', tz='UTC')

        # Should not crash
        result_df, new_cols = calculate_pca(
            df.copy(),
            frequency=frequency,
            start_dt=start_dt,
            lookback_periods=90,
            prod=False
        )

        self.assertIsNotNone(result_df)

    def test_nan_handling(self):
        """Test handling of NaN values in returns."""
        df = create_pca_test_data(n_periods=200, n_symbols=10, frequency='1h')

        # Insert some NaN values
        df.iloc[100:110, 0] = np.nan

        frequency = 60
        start_dt = pd.Timestamp('2024-01-05 00:00:00', tz='UTC')

        # Should not crash
        result_df, new_cols = calculate_pca(
            df.copy(),
            frequency=frequency,
            start_dt=start_dt,
            lookback_periods=90,
            prod=False
        )

        self.assertIsNotNone(result_df)
        # Should have some non-NaN residuals
        resid_col = f'pca_resid_{frequency}'
        self.assertGreater(result_df[resid_col].notna().sum(), 0)

    def test_production_mode_single_timestamp(self):
        """Test that production mode only computes for latest timestamp."""
        df = create_pca_test_data(n_periods=200, n_symbols=10, frequency='1h')

        frequency = 60

        result_df, _ = calculate_pca(
            df.copy(),
            frequency=frequency,
            prod=True,
            lookback_periods=90
        )

        resid_col = f'pca_resid_{frequency}'

        # In prod mode, should have residuals for only one timestamp
        resid_ts = result_df[result_df[resid_col].notna()].index.get_level_values('ts').unique()

        # Should be only the last timestamp
        self.assertEqual(len(resid_ts), 1)
        self.assertEqual(resid_ts[0], df.index.get_level_values('ts').max())


class TestCalculatePcaNewDailyFrequency(unittest.TestCase):
    """Test calculate_pca with daily (1440-minute) frequency."""

    def setUp(self):
        """Set up test fixtures with daily data."""
        # Create daily data spanning 100 days
        dates = pd.date_range('2024-01-01', periods=100, freq='D', tz='UTC')
        symbols = [f'SYM{i}USDT_binance-futures' for i in range(15)]

        np.random.seed(42)
        n_periods = len(dates)
        n_symbols = len(symbols)

        # Simple factor structure
        market_factor = np.random.randn(n_periods) * 0.03
        market_loadings = np.random.uniform(0.6, 1.2, n_symbols)
        idio_returns = np.random.randn(n_periods, n_symbols) * 0.01

        returns_matrix = np.outer(market_factor, market_loadings) + idio_returns

        index = pd.MultiIndex.from_product([dates, symbols], names=['ts', 'symbol_venue'])
        returns_flat = returns_matrix.flatten(order='C')

        self.df = pd.DataFrame({
            'logret_1440': returns_flat.astype(np.float32),
        }, index=index)

    def test_daily_frequency_calculation(self):
        """Test PCA calculation with daily returns."""
        frequency = 1440
        start_dt = pd.Timestamp('2024-02-01', tz='UTC')

        result_df, new_cols = calculate_pca(
            self.df.copy(),
            frequency=frequency,
            start_dt=start_dt,
            lookback_periods=30,  # 30 days lookback
            prod=False
        )

        resid_col = f'pca_resid_{frequency}'
        self.assertIn(resid_col, result_df.columns)

        # Should have residuals
        residuals = result_df[resid_col].dropna()
        self.assertGreater(len(residuals), 0)

    def test_daily_explained_variance_reasonable(self):
        """Test that explained variance is reasonable for daily data."""
        frequency = 1440
        start_dt = pd.Timestamp('2024-02-01', tz='UTC')

        result_df, _ = calculate_pca(
            self.df.copy(),
            frequency=frequency,
            start_dt=start_dt,
            lookback_periods=30,
            prod=False
        )

        var_col = f'pca_explained_var_{frequency}'
        explained_var = result_df[var_col].dropna()

        # Explained variance should be between 0 and 1
        self.assertTrue((explained_var >= 0).all())
        self.assertTrue((explained_var <= 1).all())

        # Given our factor structure, should explain significant variance
        self.assertGreater(explained_var.mean(), 0.3)


class TestCalculatePcaNewIntegration(unittest.TestCase):
    """Integration tests verifying the full workflow."""

    def test_full_workflow_non_prod(self):
        """Test complete non-production workflow."""
        df = create_pca_test_data(n_periods=300, n_symbols=20, frequency='1h')
        frequency = 60
        start_dt = pd.Timestamp('2024-01-05 00:00:00', tz='UTC')

        result_df, new_cols = calculate_pca(
            df.copy(),
            frequency=frequency,
            start_dt=start_dt,
            lookback_periods=90,
            prod=False,
            eigenvalue_threshold=0.01
        )

        # Verify all columns exist
        self.assertEqual(len(new_cols), 4)

        # Verify data integrity
        resid_col = f'pca_resid_{frequency}'
        n_comp_col = f'pca_n_components_{frequency}'
        var_col = f'pca_explained_var_{frequency}'
        factor_col = f'pca_factor_ret_{frequency}'

        # Check residuals exist
        residuals = result_df[resid_col].dropna()
        self.assertGreater(len(residuals), 0)

        # Check n_components is positive integer-valued
        n_comps = result_df[n_comp_col].dropna()
        self.assertTrue((n_comps > 0).all())
        self.assertTrue((n_comps == n_comps.astype(int)).all())

        # Check explained variance is in [0, 1]
        exp_var = result_df[var_col].dropna()
        self.assertTrue((exp_var >= 0).all())
        self.assertTrue((exp_var <= 1).all())

        # Check factor returns exist
        factor_rets = result_df[factor_col].dropna()
        self.assertGreater(len(factor_rets), 0)

    def test_full_workflow_prod(self):
        """Test complete production workflow."""
        df = create_pca_test_data(n_periods=200, n_symbols=15, frequency='1h')
        frequency = 60

        result_df, new_cols = calculate_pca(
            df.copy(),
            frequency=frequency,
            lookback_periods=90,
            prod=True,
            eigenvalue_threshold=0.01
        )

        # Only residual column in prod mode
        self.assertEqual(len(new_cols), 1)
        self.assertEqual(new_cols[0], f'pca_resid_{frequency}')

        # Verify residuals exist for latest timestamp
        resid_col = f'pca_resid_{frequency}'
        residuals = result_df[resid_col].dropna()
        self.assertGreater(len(residuals), 0)


if __name__ == '__main__':
    unittest.main()
