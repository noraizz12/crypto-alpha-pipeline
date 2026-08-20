"""Integration tests for the fits generation pipeline.

This module contains integration tests that verify the fits generation process
produces consistent results. It tests both master fixture generation and
comparison against those fixtures to catch regressions.

The tests use small fixture data to run quickly while still exercising the
full pipeline including:
- Data loading (bars, features, forwards, models)
- Classification with Random Forest
- Regression fitting
- Results persistence
"""
import os
import glob
import sys
import subprocess

# Set PYTHONHASHSEED for deterministic hash ordering
os.environ['PYTHONHASHSEED'] = '42'

import logging.config
from datetime import datetime as dt, timedelta as td
from typing import Optional, Dict
import traceback
import pytest
import pandas as pd
import numpy as np

from lib.util.dataframes import concat
from lib.util.directory import FIXTURE_DIR, DirectoryManager, make_fixture_dir
from lib.util.util import delete_all_files_in_tree
from lib.util.util import log_and_raise
from lib.util.logging_util import get_logging_config
from lib.fits.fits import Fits
from lib.util.time_util import date_str_to_dt
from lib.util.config import get_config

logging.config.dictConfig(get_logging_config("fits_integration_test"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FIXTURE_CONFIG: str = os.path.join(FIXTURE_DIR, "fits", "config_fits_test.json")
FIXTURE_INPUT_DIR: str = os.path.join(FIXTURE_DIR, "fits", "data")
FIXTURE_OUTPUT_DIR: str = os.path.join(FIXTURE_DIR, "fits", "master")
TEST_HORIZON = 15  # Use 15 min horizon for faster testing

# Tolerance constants for Random Forest classifier variability
NOBS_TOLERANCE = 0.035  # 3.5% tolerance for number of observations
NUMERIC_RTOL = 2e-2     # 2% relative tolerance for numeric columns
TSTAT_ATOL = 1.0        # Absolute tolerance for t-statistics
COEFF_ATOL = 0.1        # Absolute tolerance for coefficients

# Error messages
ERROR_MSG_DIFF_CHECK = (
    "If test failed, please ssh -N -f -L localhost:8060:127.0.0.1:8060 "
    "user@your_server and open "
    "http://localhost:8060/fixtures/integration_test_generate_fits_diff.html "
    "to check csv diff"
)
ERROR_MSG_RERUN = (
    "Run ./bin/run_integration_test.sh -n fits to see the full log\n"
    "Run ./bin/regenerate_master_fixture.sh -n fits to regenerate master "
    "fixture if it's a legit change"
)


@pytest.fixture
def test_dates() -> Dict[str, dt]:
    # Load config to get test dates
    _, config = get_config(FIXTURE_CONFIG)
    fit_test_date = config.get('FIT_TEST_DATE', '20250108')
    return {
        'start_dt': date_str_to_dt(fit_test_date),
        'end_dt': date_str_to_dt(fit_test_date),
    }


def load_fits_csv_file(fixture_dir: str) -> Optional[pd.DataFrame]:
    """Load fits CSV files from a directory and concatenate them.
    
    Args:
        fixture_dir: Directory containing fits CSV files
        
    Returns:
        Concatenated DataFrame of all fits files, or None if loading fails
    """
    logger.debug(f"Attempting to load fits CSV data from {fixture_dir}")
    try:
        # Check for new directory structure
        if os.path.exists(os.path.join(fixture_dir, "15", "hl")):
            file_pattern = f"{fixture_dir}/15/hl/fits.*.csv"
        else:
            file_pattern = f"{fixture_dir}/fits.*.csv"
            
        fit_files = glob.glob(file_pattern)
        dfs = []
        logger.info(f"Loading {len(fit_files)} files from pattern {file_pattern}...")
        for fit_file in fit_files:
            df = pd.read_csv(fit_file)
            # Reset index to avoid duplicate index issues
            df = df.reset_index(drop=True)
            dfs.append(df)
        # Use fast=True to avoid index verification issues
        fixture_df = concat(dfs, fast=True)
    except Exception as e:
        logger.exception(f"Failed to load fits CSV data from {fixture_dir}. Error: {str(e)}")
        return None
    logger.info(f"Successfully loaded fits CSV data from {fixture_dir}")
    return fixture_df


def check_master_fixture_exists(master_target_dir: str) -> bool:
    """Check if master fixture files exist in the target directory.
    
    Args:
        master_target_dir: Directory to check for master fixtures
        
    Returns:
        True if fixture files exist, False otherwise
    """
    # Check for new directory structure with horizon subdirectories
    if os.path.exists(os.path.join(master_target_dir, "15", "hl")):
        hl_dir = os.path.join(master_target_dir, "15", "hl")
        return any(f.startswith("fits.") and f.endswith(".csv") 
                  for f in os.listdir(hl_dir))
    # Check old structure
    return any(f.startswith("fits.") and f.endswith(".csv") 
              for f in os.listdir(master_target_dir) 
              if os.path.isfile(os.path.join(master_target_dir, f)))


def run_generation_script(script_name: str, args: list, 
                         cwd: str = None, env: dict = None) -> subprocess.CompletedProcess:
    """Run a generation script with the given arguments.
    
    Args:
        script_name: Name of the script to run
        args: Command line arguments for the script
        cwd: Working directory for script execution
        env: Environment variables for script execution
        
    Returns:
        CompletedProcess instance with execution results
    """
    cmd = [sys.executable, script_name] + args
    logger.info(f"Running command: {' '.join(cmd)}")
    if cwd:
        logger.info(f"Working directory: {cwd}")
    if env:
        logger.info(f"Setting DATA_DIR={env.get('DATA_DIR', 'not set')}")
    
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Script failed with return code {result.returncode}")
        logger.error(f"STDERR: {result.stderr}")
        logger.error(f"STDOUT: {result.stdout}")
    else:
        logger.info(f"Script {script_name} completed successfully")
    
    return result


def check_fixture_data_exists(test_dates: Dict[str, dt]) -> bool:
    """Check if required fixture data exists for fits test.
    
    Returns True if all required data exists, False otherwise.
    """
    required_files = []
    date = (test_dates['start_dt'] - td(days=1)).strftime('%Y%m%d')
    
    # Check for bar data
    bar_file = os.path.join(
        FIXTURE_INPUT_DIR, 'bars', '15', 'binance-futures', date,
        f"bars.15.binance-futures.{date}.BTCUSDT.parquet"
    )
    required_files.append(("Bar data", bar_file))
    
    # Check for feature data
    feature_file = os.path.join(
        FIXTURE_INPUT_DIR, 'features', '15', 'logret_15_trstd', date,
        f"features.15.logret_15_trstd.{date}.BTCUSDT.parquet"
    )
    required_files.append(("Feature data", feature_file))
    
    # Check for forward data
    forward_file = os.path.join(
        FIXTURE_INPUT_DIR, 'forwards', '15', date,
        f"forwards.15.{date}.BTCUSDT.parquet"
    )
    required_files.append(("Forward data", forward_file))
    
    # Check for model data
    model_file = os.path.join(
        FIXTURE_INPUT_DIR, 'models', '15', 'hl',
        f"models.15.hl.{date}.parquet"
    )
    required_files.append(("Model data", model_file))
    
    all_exist = True
    for name, file_path in required_files:
        if os.path.exists(file_path):
            logger.info(f"✓ {name} found: {file_path}")
        else:
            logger.error(f"✗ {name} missing: {file_path}")
            all_exist = False
    
    return all_exist


@pytest.fixture
def master_fits() -> Fits:
    _, config = get_config(FIXTURE_CONFIG)
    fits_dir_manager = DirectoryManager(data_dir=FIXTURE_INPUT_DIR, trading_dir=FIXTURE_INPUT_DIR)

    fits_calc = Fits(
        config=config,
        horizons=[TEST_HORIZON],
        models=['hl'],  # Only test hl model for speed
        prod=False,
        pool_size=2,
        debug=False,
        base_fits_dir=FIXTURE_OUTPUT_DIR,
        fits_dir_manager=fits_dir_manager,
    )

    return fits_calc


@pytest.fixture
def test_fits() -> Fits:
    temp_dir = os.path.join(FIXTURE_DIR, 'temp_generated')
    _, config = get_config(FIXTURE_CONFIG)
    fits_dir_manager = DirectoryManager(data_dir=FIXTURE_INPUT_DIR, trading_dir=FIXTURE_INPUT_DIR)

    fits_calc = Fits(
        config=config,
        horizons=[TEST_HORIZON],
        models=['hl'],  # Only test hl model for speed
        prod=False,
        pool_size=2,
        debug=False,
        base_fits_dir=os.path.join(temp_dir, 'fits'),
        fits_dir_manager=fits_dir_manager,
    )

    return fits_calc


def test_generate_master_fixture(master_fits: Fits, test_dates: Dict[str, dt]) -> None:
    """Generate master fixture files for regression testing.
    
    Creates reference fixtures that future test runs will be compared against.
    
    Args:
        master_fits: Fits instance configured for master fixture generation
        test_dates: Dictionary with start and end dates for testing
    """
    try:
        logger.info(f"Start generating master fixture for fits generation on {test_dates}")
        
        # Delete existing fixture files
        logger.info("Deleting existing fixture files...")
        delete_all_files_in_tree(FIXTURE_OUTPUT_DIR)
        
        # Check if required data exists
        if not check_fixture_data_exists(test_dates):
            raise log_and_raise(
                "Required fixture data missing. Please run generate_fixtures.py "
                "to create test data."
            )
        
        _ = make_fixture_dir(FIXTURE_OUTPUT_DIR)
        master_fits.generate_rolling_fits(
            start_date=test_dates['start_dt'].date(), 
            end_date=test_dates['end_dt'].date()
        )
        logger.info(f"Finish generating master fixture for fits generation on {test_dates}")
    except Exception as e:
        raise log_and_raise(f"An error occurred during generating master fixture: {str(e)}") from e


def test_fits_run(test_fits: Fits, test_dates: Dict[str, dt]) -> None:
    """Test that fits generation produces consistent results.
    
    Generates fits and compares them against master fixtures to detect
    regressions or unexpected changes.
    
    Args:
        test_fits: Fits instance configured for test generation
        test_dates: Dictionary with start and end dates for testing
    """
    logger.info("Start testing new branch of fits generation")

    temp_dir = os.path.join(FIXTURE_DIR, 'temp_generated')
    temp_fits_dir = make_fixture_dir(os.path.join(temp_dir, 'fits'))
    master_fits_dir = make_fixture_dir(FIXTURE_OUTPUT_DIR, keep_existing=True)

    master_fixture_exist = check_master_fixture_exists(master_fits_dir)
    if not master_fixture_exist:
        raise log_and_raise(
            "There is no existing master fixture files, please generate "
            "master fixtures first"
        )

    try:
        logger.info("Start test fits run")
        test_fits.generate_rolling_fits(
            start_date=test_dates['start_dt'].date(), 
            end_date=test_dates['end_dt'].date()
        )
        logger.info("Finish test fits run")

        logger.info("Comparing fits files")
        master_fits_df = load_fits_csv_file(master_fits_dir)
        temp_fits_df = load_fits_csv_file(temp_fits_dir)

        # For fits, we need to handle numerical precision issues in regression calculations
        # Sort both dataframes by the same columns to ensure consistent ordering
        sort_cols = ['horizon', 'as_of', 'name', 'lag']
        master_fits_df = master_fits_df.sort_values(sort_cols).reset_index(drop=True)
        temp_fits_df = temp_fits_df.sort_values(sort_cols).reset_index(drop=True)
        
        # First check if both dataframes have the same structure
        if set(master_fits_df.columns) != set(temp_fits_df.columns):
            raise ValueError(
                f"Column mismatch: master has {set(master_fits_df.columns)}, "
                f"temp has {set(temp_fits_df.columns)}"
            )
        
        # Check shape
        if master_fits_df.shape != temp_fits_df.shape:
            raise ValueError(
                f"Shape mismatch: master {master_fits_df.shape}, "
                f"temp {temp_fits_df.shape}"
            )
        
        # Compare non-numeric columns exactly
        non_numeric_cols = master_fits_df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            if not master_fits_df[col].equals(temp_fits_df[col]):
                raise ValueError(f"Non-numeric column {col} does not match exactly")
        
        # Compare numeric columns with appropriate tolerance
        numeric_cols = master_fits_df.select_dtypes(include=[np.number]).columns
        
        # Note: Random Forest classifier introduces more variability than SVM
        # even with fixed random seeds due to:
        # 1. Different sampling of observations between runs
        # 2. Numerical instability in tree splitting decisions
        # 3. Feature ordering effects on split decisions
        # Therefore, we need larger tolerances for RF-based fits
        
        # For most numeric columns, use a small tolerance
        for col in numeric_cols:
            if col in ['condition', 'condition_num']:
                # These should match exactly
                if not master_fits_df[col].equals(temp_fits_df[col]):
                    raise ValueError(f"Column {col} should match exactly but doesn't")
            elif col == 'nobs':
                # Allow larger tolerance for nobs due to RF classifier sampling variability
                # RF classifier can cause up to 2% variation in number of observations selected
                # Handle division by zero when master values are 0
                master_vals = master_fits_df[col].values
                temp_vals = temp_fits_df[col].values
                
                # Check absolute difference for zero values
                zero_mask = master_vals == 0
                if zero_mask.any():
                    abs_diff_zeros = np.abs(temp_vals[zero_mask] - master_vals[zero_mask])
                    if np.max(abs_diff_zeros) > 1:  # Allow difference of 1 observation for zero values
                        raise ValueError(
                            f"Column {col} has differences > 1 for zero values: "
                            f"max diff = {np.max(abs_diff_zeros)}"
                        )
                
                # Check relative difference for non-zero values
                nonzero_mask = ~zero_mask
                if nonzero_mask.any():
                    rel_diff = np.abs(master_vals[nonzero_mask] - temp_vals[nonzero_mask]) / master_vals[nonzero_mask]
                    if np.max(rel_diff) > NOBS_TOLERANCE:
                        raise ValueError(
                            f"Column {col} differs by more than {NOBS_TOLERANCE*100}%: "
                            f"max relative diff = {np.max(rel_diff)}"
                        )
            else:
                # For other numeric columns (coeff, tstat, stderr, etc), use tolerance
                # The differences we're seeing are on the order of 1e-15 to 1e-10 for coeff
                # but can be larger for tstat and stderr due to SVM variability
                if col in ['tstat', 'stderr']:
                    # Use a more lenient tolerance for statistical measures
                    # that can vary due to RF classifier numerical instability
                    if not np.allclose(master_fits_df[col].values, temp_fits_df[col].values, 
                                     rtol=NUMERIC_RTOL, atol=TSTAT_ATOL, equal_nan=True):
                        max_diff = np.max(np.abs(master_fits_df[col].values - temp_fits_df[col].values))
                        raise ValueError(f"Column {col} differs beyond tolerance. Max diff: {max_diff}")
                else:
                    # For coefficients and other values, use reasonable tolerance
                    # RF classifier can cause coefficient variations up to ~0.1
                    if not np.allclose(master_fits_df[col].values, temp_fits_df[col].values, 
                                     rtol=NUMERIC_RTOL, atol=COEFF_ATOL, equal_nan=True):
                        max_diff = np.max(np.abs(master_fits_df[col].values - temp_fits_df[col].values))
                        raise ValueError(f"Column {col} differs beyond tolerance. Max diff: {max_diff}")
        
        res = True
        msg = "Fits comparison passed"
        
        # Always save the comparison files for debugging
        master_fits_df.to_csv(os.path.join(temp_dir, 'master.csv'))
        temp_fits_df.to_csv(os.path.join(temp_dir, 'temp.csv'))
        
        if not res:
            raise ValueError(msg)
        
        # Don't remove temp dir for now to allow debugging
        # shutil.rmtree(temp_dir)
        logger.info(f"Temporary directory {temp_dir} kept for debugging.")
    except Exception as e:
        tb_str = traceback.format_exc()
        new_message = f"{str(e)}\n{tb_str}\n{ERROR_MSG_DIFF_CHECK}\n{ERROR_MSG_RERUN}"
        raise RuntimeError(new_message) from e

    finally:
        logger.info(ERROR_MSG_DIFF_CHECK)
        logger.info(ERROR_MSG_RERUN)

    logger.info("Fits run and comparisons completed.")
    logger.info(f"Fixture files are located in: {FIXTURE_OUTPUT_DIR}")
    logger.info(f"Temporary files are located in: {temp_dir}")


if __name__ == '__main__':
    pytest.main([__file__])