import glob
import logging.config
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime as dt, timedelta as td
from typing import Optional, Dict

import pandas as pd
import pytest

from lib.util.config import get_config
from lib.util.dataframes import concat
from lib.util.directory import FIXTURE_DIR, DirectoryManager, make_fixture_dir
from lib.alpha.forecasts import Forecasts
from lib.util.time_util import date_str_to_dt
from lib.util.util import delete_all_files_in_tree
from lib.util.util import log_and_raise
from lib.util.logging_util import get_logging_config
from lib.util.dataframes import compare_dataframes

logging.config.dictConfig(get_logging_config("alphas_integration_test"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FIXTURE_CONFIG: str = os.path.join(FIXTURE_DIR, "alphas", "config_alphas_generation_test.json")
FIXTURE_INPUT_DIR: str = os.path.join(FIXTURE_DIR, "alphas", "data")
FIXTURE_OUTPUT_DIR: str = os.path.join(FIXTURE_DIR, "alphas", "master")
TEST_HORIZON = 15  # Use 15 min horizon for faster testing

LOG_MSG1 = "If test failed, please ssh -N -f -L localhost:8060:127.0.0.1:8060 user@your_server and open http://localhost:8060/fixtures/integration_test_generate_alphas_diff.html to check csv diff"
LOG_MSG2 = "Run ./bin/run_integration_test.sh -n alphas to see the full log\n"
LOG_MSG2 += "Run ./bin/regenerate_master_fixture.sh -n alphas to regenerate master fixture if it's a legit change"


@pytest.fixture
def test_dates() -> Dict[str, dt]:
    _, config = get_config(FIXTURE_CONFIG)
    # Get test date from config or use default
    test_date = config.get('ALPHA_TEST_DATE', '20250108')
    return {
        'start_dt': date_str_to_dt(test_date),
        'end_dt': date_str_to_dt(test_date),
    }


def load_alphas_parquet_file(fixture_dir: str) -> Optional[pd.DataFrame]:
    logger.debug(f"Attempting to load alphas parquet data from {fixture_dir}")
    try:
        # Search for parquet files recursively
        file_pattern = f"{fixture_dir}/**/*.parquet"
        alpha_files = glob.glob(file_pattern, recursive=True)
        
        # If no files found at top level, check subdirectories
        if not alpha_files:
            file_pattern = f"{fixture_dir}/*.parquet"
            alpha_files = glob.glob(file_pattern)
            
        dfs = []
        logger.info(f"Loading {len(alpha_files)} files...")
        for alpha_file in alpha_files:
            df = pd.read_parquet(alpha_file)
            dfs.append(df)
        fixture_df = concat(dfs)
    except Exception as e:
        logger.exception(f"Failed to load alphas parquet data from {fixture_dir}. Error: {str(e)}")
        return None
    logger.info(f"Successfully loaded alphas parquet data from {fixture_dir}")
    return fixture_df


def check_master_fixture_exist(master_target_dir):
    # Check for parquet files in the root directory
    if any(f.endswith(".parquet") for f in os.listdir(master_target_dir)):
        return True
    
    # Also check for horizon subdirectories (new format)
    for item in os.listdir(master_target_dir):
        item_path = os.path.join(master_target_dir, item)
        if os.path.isdir(item_path):
            # Check for model subdirectories
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    # Check for parquet files in model subdirectories
                    if any(f.endswith(".parquet") for f in os.listdir(subitem_path)):
                        return True
    return False


def check_fixture_data_exists(test_dates: Dict[str, dt]) -> bool:
    """Check if all required fixture data exists."""
    required_dirs = ['bars', 'features', 'models', 'prod_fits', 'universe']
    
    for dir_name in required_dirs:
        dir_path = os.path.join(FIXTURE_INPUT_DIR, dir_name)
        if not os.path.exists(dir_path) or not os.listdir(dir_path):
            logger.info(f"Missing or empty fixture directory: {dir_name}")
            return False
    
    # Check for specific horizon/model directories for alphas
    horizon = TEST_HORIZON
    model = 'hl'
    
    # Check for model files
    model_dir = os.path.join(FIXTURE_INPUT_DIR, 'models', str(horizon), model)
    if not os.path.exists(model_dir) or not any(f.endswith('.parquet') for f in os.listdir(model_dir)):
        logger.info(f"Missing model files for horizon {horizon}, model {model}")
        return False
    
    # Check for fit files
    fit_dir = os.path.join(FIXTURE_INPUT_DIR, 'prod_fits', str(horizon), model)
    if not os.path.exists(fit_dir) or not any(f.endswith('.csv') for f in os.listdir(fit_dir)):
        logger.info(f"Missing fit files for horizon {horizon}, model {model}")
        return False
    
    logger.info("All required fixture data exists")
    return True


@pytest.fixture
def master_forecasts(test_dates: Dict[str, dt]) -> Forecasts:
    """Create Forecasts instance for master fixture generation."""
    _, config = get_config(FIXTURE_CONFIG)
    
    # Use the alphas test data directory
    forecasts_dir_manager = DirectoryManager(data_dir=FIXTURE_INPUT_DIR, trading_dir=FIXTURE_INPUT_DIR)
    
    # Don't modify FCASTS in the config - Forecasts.generate_rolling_alphas() always needs horizon 1440
    # Just ensure we have the minimal 1440 config that was added to the fixture config
    
    forecasts = Forecasts(
        config=config,
        prod=False,  # Use prod=False to read from prod_fits directory
        debug=False,
        horizons=[TEST_HORIZON],
        models=['hl'],
        output_dir=FIXTURE_OUTPUT_DIR,
        forecast_dir_manager=forecasts_dir_manager,
    )
    
    return forecasts


@pytest.fixture
def test_forecasts(test_dates: Dict[str, dt]) -> Forecasts:
    """Create Forecasts instance for test run."""
    temp_dir = os.path.join(FIXTURE_DIR, 'temp_generated')
    _, config = get_config(FIXTURE_CONFIG)
    
    # Use the alphas test data directory
    forecasts_dir_manager = DirectoryManager(data_dir=FIXTURE_INPUT_DIR, trading_dir=FIXTURE_INPUT_DIR)
    
    # Don't modify FCASTS in the config - Forecasts.generate_rolling_alphas() always needs horizon 1440
    # Just ensure we have the minimal 1440 config that was added to the fixture config
    
    forecasts = Forecasts(
        config=config,
        prod=False,  # Use prod=False to read from prod_fits directory
        debug=False,
        horizons=[TEST_HORIZON],
        models=['hl'],
        output_dir=os.path.join(temp_dir, 'alpha'),
        forecast_dir_manager=forecasts_dir_manager,
    )
    
    return forecasts


def test_generate_master_fixture(master_forecasts: Forecasts, test_dates: Dict[str, dt]) -> None:
    """Generate master fixture for alphas using Forecasts class."""
    try:
        logger.info(f"Start generating master fixture for alphas generation on {test_dates}")
        
        # Delete existing fixture files
        logger.info("Deleting existing fixture files...")
        delete_all_files_in_tree(FIXTURE_OUTPUT_DIR)
        
        # Check if required data exists, if not generate it
        if not check_fixture_data_exists(test_dates):
            logger.info("Fixture data missing, generating test data...")
            # Run the fixture generation script
            script_path = os.path.join(FIXTURE_DIR, "alphas", "generate_fixtures.py")
            result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to generate fixture data: {result.stderr}")
        else:
            logger.info("Test data already exists, skipping generation")
        
        _ = make_fixture_dir(FIXTURE_OUTPUT_DIR)
        
        # Run generate_rolling_alphas
        master_forecasts.generate_rolling_alphas(
            fit_file=None,
            start_date=test_dates['start_dt'].date(),
            end_date=test_dates['end_dt'].date(),
            verbose=False,
            chunk_days=90
        )
        
        logger.info(f"Finish generating master fixture for alphas generation on {test_dates}")
    except Exception as e:
        raise log_and_raise(f"An error occurred during generating master fixture: {str(e)}") from e


def test_alphas_run(test_forecasts: Forecasts, test_dates: Dict[str, dt]) -> None:
    logger.info("Start testing new branch of alphas generation")

    temp_dir = os.path.join(FIXTURE_DIR, 'temp_generated')
    temp_alphas_dir = make_fixture_dir(os.path.join(temp_dir, 'alpha'))
    master_alphas_dir = make_fixture_dir(FIXTURE_OUTPUT_DIR, keep_existing=True)

    master_fixture_exist = check_master_fixture_exist(master_alphas_dir)
    if not master_fixture_exist:
        raise log_and_raise("There is no existing master fixture files, please generate master fixtures first")

    try:
        logger.info("Start test alphas run")
        
        # Run generate_rolling_alphas
        test_forecasts.generate_rolling_alphas(
            fit_file=None,
            start_date=test_dates['start_dt'].date(),
            end_date=test_dates['end_dt'].date(),
            verbose=False,
            chunk_days=90,
            pool_size=1
        )
        
        logger.info("Finish test alphas run")

        logger.info("Comparing alphas files")
        master_alphas_df = load_alphas_parquet_file(master_alphas_dir)
        temp_alphas_df = load_alphas_parquet_file(temp_alphas_dir)

        res, msg, _ = compare_dataframes(master_alphas_df, temp_alphas_df, "Fixture", "Temp", "Alphas")
        if not res:
            master_alphas_df.to_csv(os.path.join(temp_dir, 'master.csv'))
            temp_alphas_df.to_csv(os.path.join(temp_dir, 'temp.csv'))
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

    logger.info("Alphas run and comparisons completed.")
    logger.info(f"Fixture files are located in: {FIXTURE_OUTPUT_DIR}")
    logger.info(f"Temporary files are located in: {temp_dir}")


if __name__ == '__main__':
    pytest.main([__file__])