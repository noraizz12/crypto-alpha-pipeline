import os
import glob
import sys
import subprocess

import logging.config
from datetime import datetime as dt, timedelta as td
from typing import Optional, Dict
import shutil
import traceback
import pytest
import pandas as pd

# BARS_TYPE_NEW is no longer needed as it's the only option
from lib.util.dataframes import concat
from lib.util.directory import FIXTURE_DIR, DirectoryManager, make_fixture_dir
from lib.util.util import delete_all_files_in_tree
from lib.util.util import log_and_raise
from lib.util.logging_util import get_logging_config
from lib.util.dataframes import compare_dataframes
from lib.alpha.model_calcs import ModelCalcs
from lib.util.time_util import date_str_to_dt
from lib.util.config import get_config

logging.config.dictConfig(get_logging_config("models_integration_test"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FIXTURE_CONFIG: str = os.path.join(FIXTURE_DIR, "models", "config_models_generation_test.json")
FIXTURE_INPUT_DIR: str = os.path.join(FIXTURE_DIR, "models", "data")
FIXTURE_OUTPUT_DIR: str = os.path.join(FIXTURE_DIR, "models", "master")
TEST_HORIZON = 1440

LOG_MSG1 = "If test failed, please ssh -N -f -L localhost:8060:127.0.0.1:8060 user@your_server and open http://localhost:8060/fixtures/integration_test_generate_models_diff.html to check csv diff"
LOG_MSG2 = "Run ./bin/run_integration_test.sh -n models to see the full log\n"
LOG_MSG2 += "Run ./bin/regenerate_master_fixture.sh -n models to regenerate master fixture if it's a legit change"


@pytest.fixture
def test_dates() -> Dict[str, dt]:
    # Load config to get test dates
    _, config = get_config(FIXTURE_CONFIG)
    model_test_date = config.get('MODEL_TEST_DATE', '20250108')
    return {
        'start_dt': date_str_to_dt(model_test_date),
        'end_dt': date_str_to_dt(model_test_date),  # Single day test
    }


def load_models_parquet_file(fixture_dir: str) -> Optional[pd.DataFrame]:
    logger.debug(f"Attempting to load models parquet data from {fixture_dir}")
    try:
        # Search for parquet files in subdirectories too
        file_pattern = f"{fixture_dir}/**/*.parquet"
        model_files = glob.glob(file_pattern, recursive=True)
        dfs = []
        logger.info(f"Loading {len(model_files)} files...")
        for model_file in model_files:
            df = pd.read_parquet(model_file)
            dfs.append(df)
        fixture_df = concat(dfs)
    except Exception as e:
        logger.exception(f"Failed to load models parquet data from {fixture_dir}. Error: {str(e)}")
        return None
    logger.info(f"Successfully loaded models parquet data from {fixture_dir}")
    return fixture_df


def check_master_fixture_exist(master_target_dir):
    # Check for parquet files in the directory tree
    for root, dirs, files in os.walk(master_target_dir):
        if any(f.endswith(".parquet") for f in files):
            return True
    return False


def run_generation_script(script_name: str, args: list, cwd: str = None) -> subprocess.CompletedProcess:
    """Run a generation script with the given arguments."""
    cmd = [sys.executable, script_name] + args
    logger.info(f"Running command: {' '.join(cmd)}")
    if cwd:
        logger.info(f"Working directory: {cwd}")

    result = subprocess.run(
        cmd,
        cwd=cwd,
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


def remove_old_fixture_data() -> None:
    """Remove all files in FIXTURE_INPUT_DIR to clean up before regenerating fixtures."""
    logger.info(f"Removing old fixture data from {FIXTURE_INPUT_DIR}")

    if not os.path.exists(FIXTURE_INPUT_DIR):
        logger.warning(f"Fixture input directory does not exist: {FIXTURE_INPUT_DIR}")
        return

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(FIXTURE_INPUT_DIR):
        for filename in files:
            file_path = os.path.join(root, filename)
            logger.info(f"Removing file: {file_path}")
            os.remove(file_path)

    logger.info("Old fixture data removed successfully")


def generate_models_fixture_data(test_dates: Dict[str, dt]) -> None:
    """Generate all fixture data needed for models integration test.
    
    This function generates minimal test data by running the actual generation
    scripts with a test config that has only 6 symbols (ADA, BNB, BTC, ETC, ETH, ZIL).
    """
    logger.info("Generating fixture data for models integration test")

    remove_old_fixture_data()
    # Calculate date range needed
    start_date = test_dates['start_dt'].date() - td(days=3)  # Extra days for lookback
    end_date = test_dates['end_dt'].date()

    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')

    logger.info(f"Generating data from {start_str} to {end_str}")

    # Set working directory to src
    cwd = os.path.dirname(os.path.abspath(__file__)).replace('/test', '')

    # Path to test config
    test_config = 'test/fixtures/models/config_models_generation_test.json'

    # Step 1: Generate bars
    logger.info("Step 1: Generating bars...")
    result = run_generation_script(
        'generate_bars.py',
        ['--from', start_str,
         '--to', end_str,
         '--config', test_config,
         '--output-dir', 'test/fixtures/models/data/live_bars',
         '--horizons', '1440',  # Only generate daily bars for speed
         '-q'],  # Quiet mode
        cwd=cwd
    )

    if result.returncode != 0:
        raise RuntimeError(f"Bar generation failed: {result.stderr}")

    # Step 2: Generate master
    logger.info("Step 2: Generating master...")
    result = run_generation_script(
        'generate_features.py',
        ['-f', start_str,
         '-t', end_str,
         '-c', test_config,
         '-z', '1440',
         '-o', 'test/fixtures/models/data/features',
         '-q'],  # Quiet mode
        cwd=cwd
    )

    if result.returncode != 0:
        logger.warning(f"Feature generation had issues: {result.stderr}")
        # Continue anyway as we might have enough master

    logger.info("Fixture data generation complete")


@pytest.fixture
def master_model_calc() -> ModelCalcs:
    _, config = get_config(FIXTURE_CONFIG)
    models_dir_manager = DirectoryManager(data_dir=FIXTURE_INPUT_DIR, trading_dir=FIXTURE_INPUT_DIR)

    model_calc = ModelCalcs(
        config=config,
        models_to_run=['hl'],  # Only test hl model for speed
        horizons=[TEST_HORIZON],
        debug=False,
        pool_size=2,
        output_dir=FIXTURE_OUTPUT_DIR,
        models_dir_manager=models_dir_manager,
    )

    return model_calc


@pytest.fixture
def test_model_calc() -> ModelCalcs:
    temp_dir = os.path.join(FIXTURE_DIR, 'temp_generated')
    _, config = get_config(FIXTURE_CONFIG)
    models_dir_manager = DirectoryManager(data_dir=FIXTURE_INPUT_DIR, trading_dir=FIXTURE_INPUT_DIR)

    model_calc = ModelCalcs(
        config=config,
        models_to_run=['hl'],  # Only test hl model for speed
        horizons=[TEST_HORIZON],
        debug=False,
        pool_size=2,
        output_dir=os.path.join(temp_dir, 'models'),
        models_dir_manager=models_dir_manager,
    )

    return model_calc


def test_generate_master_fixture(master_model_calc: ModelCalcs, test_dates: Dict[str, dt]) -> None:
    try:
        logger.info(f"Start generating master fixture for model generation on {test_dates}")
        
        # Delete existing fixture files
        logger.info("Deleting existing fixture files...")
        delete_all_files_in_tree(FIXTURE_OUTPUT_DIR)
        
        # Check if required bar data exists (features will be generated on demand)
        bars_file = os.path.join(FIXTURE_INPUT_DIR, 'bars', '1440', 'binance-futures', test_dates['start_dt'].strftime('%Y%m%d'),
                                f"bars.1440.binance-futures.{test_dates['start_dt'].strftime('%Y%m%d')}.BTCUSDT.parquet")
        if not os.path.exists(bars_file):
            raise log_and_raise("Bar data missing. Please run generate_fixtures.py to create test data.")
        else:
            logger.info("Test data found, proceeding with model generation")

        _ = make_fixture_dir(FIXTURE_OUTPUT_DIR)
        master_model_calc.process_models(start_date=test_dates['start_dt'].date(), end_date=test_dates['end_dt'].date())
        logger.info(f"Finish generating master fixture for model generation on {test_dates}")
    except Exception as e:
        raise log_and_raise(f"An error occurred during generating master fixture: {str(e)}") from e


def test_model_run(test_model_calc: ModelCalcs, test_dates: Dict[str, dt]) -> None:
    logger.info("Start testing new branch of model generation")

    temp_dir = os.path.join(FIXTURE_DIR, 'temp_generated')
    temp_models_dir = make_fixture_dir(os.path.join(temp_dir, 'models'))
    master_models_dir = make_fixture_dir(FIXTURE_OUTPUT_DIR, keep_existing=True)

    master_fixture_exist = check_master_fixture_exist(master_models_dir)
    if not master_fixture_exist:
        raise log_and_raise("There is no existing master fixture files, please generate master fixtures first")

    try:
        logger.info("Start test model run")
        test_model_calc.process_models(start_date=test_dates['start_dt'].date(), end_date=test_dates['end_dt'].date())
        logger.info("Finish test model run")

        logger.info("Comparing models files")
        master_models_df = load_models_parquet_file(master_models_dir)
        temp_models_df = load_models_parquet_file(temp_models_dir)

        res, msg, _ = compare_dataframes(master_models_df, temp_models_df, "Fixture", "Temp", "Models")
        if not res:
            master_models_df.to_csv(os.path.join(temp_dir, 'master.csv'))
            temp_models_df.to_csv(os.path.join(temp_dir, 'temp.csv'))
            raise ValueError(msg)

        shutil.rmtree(temp_dir)
        logger.info(f"Temporary directory {temp_dir} has been removed.")
    except Exception as e:
        tb_str = traceback.format_exc()
        new_message = f"{str(e)}\n{tb_str}\n{LOG_MSG1}\n{LOG_MSG2}"
        raise RuntimeError(new_message) from e

    finally:
        logger.info(LOG_MSG1)
        logger.info(LOG_MSG2)

    logger.info("Models run and comparisons completed.")
    logger.info(f"Fixture files are located in: {FIXTURE_OUTPUT_DIR}")
    logger.info(f"Temporary files are located in: {temp_dir}")


if __name__ == '__main__':
    pytest.main([__file__])
