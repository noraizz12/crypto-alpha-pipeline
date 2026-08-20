import os
import glob
import sys
import subprocess

import logging.config
from datetime import datetime as dt
from datetime import timedelta as td
from typing import Optional
import shutil
import traceback
import pytest
import pandas as pd

from lib.util.dataframes import concat
from lib.util.directory import FIXTURE_DIR, DirectoryManager, make_fixture_dir
from lib.util.util import delete_all_files_in_tree
from lib.util.util import log_and_raise
from lib.util.logging_util import get_logging_config
from lib.util.dataframes import compare_dataframes
from lib.server.server import AlphaServer
from lib.util.time_util import date_str_to_dt

logging.config.dictConfig(get_logging_config("integration_test_server"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FIXTURE_CONFIG: str = os.path.join(FIXTURE_DIR, "server", "config_server_test.json")
FIXTURE_INPUT_DIR: str = os.path.join(FIXTURE_DIR, "server", "data")
FIXTURE_OUTPUT_DIR: str = os.path.join(FIXTURE_DIR, "server", "master")

LOG_MSG1 = "If test failed, please ssh -N -f -L localhost:8060:127.0.0.1:8060 user@your_server and open http://localhost:8060/fixtures/integration_test_server_diff.html to check csv diff"
LOG_MSG2 = "Run ./bin/run_integration_test.sh -n server to see the full log\n"
LOG_MSG2 += "Run ./bin/regenerate_master_fixture.sh -n server to regenerate master fixture if it's a legit change"


@pytest.fixture
def test_dates() -> dt:
    return date_str_to_dt('20250606')


def load_raw_target_parquet_file(fixture_dir: str) -> Optional[pd.DataFrame]:
    logger.debug(f"Attempting to load raw target parquet data from {fixture_dir}")
    try:
        file_pattern = f"{fixture_dir}/alpha.*.parquet"
        raw_target_files = glob.glob(file_pattern)
        dfs = []
        logger.info(f"Loading {len(raw_target_files)} files...")
        for raw_target_file in raw_target_files:
            df = pd.read_parquet(raw_target_file)
            dfs.append(df)
        fixture_df = concat(dfs)
    except Exception as e:
        logger.exception(f"Failed to load raw target parquet data from {fixture_dir}. Error: {str(e)}")
        return None
    logger.info(f"Successfully loaded raw target parquet data from {fixture_dir}")
    return fixture_df


def load_target_csv_file(fixture_dir: str) -> Optional[pd.DataFrame]:
    logger.debug(f"Attempting to load target csv data from {fixture_dir}")
    try:
        file_pattern = f"{fixture_dir}/targets.*.csv"
        target_files = glob.glob(file_pattern)
        dfs = []
        logger.info(f"Loading {len(target_files)} files...")
        for target_file in target_files:
            df = pd.read_csv(target_file)
            dfs.append(df)
        fixture_df = concat(dfs)
    except Exception as e:
        logger.exception(f"Failed to load target csv data from {fixture_dir}. Error: {str(e)}")
        return None
    logger.info(f"Successfully loaded target csv data from {fixture_dir}")
    return fixture_df


def check_master_fixture_exist(master_target_dir):
    return any(f.startswith("targets.") and f.endswith(".csv") for f in os.listdir(master_target_dir))


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


def generate_server_fixture_data(test_date: dt) -> None:
    """Generate feature and model files for server test with all required columns."""
    logger.info("Generating fixture data for server integration test")
    
    # Calculate date range needed (need some lookback for features)
    end_date = test_date.date()
    start_date = end_date - td(days=5)  # 5 days lookback for features
    
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    logger.info(f"Generating data from {start_str} to {end_str}")
    
    # Set working directory to src
    cwd = os.path.dirname(os.path.abspath(__file__)).replace('/test', '')
    
    # Path to test config
    test_config = 'test/fixtures/server/config_server_test.json'
    
    # Step 1: Generate features for all required horizons
    horizons = ['15', '60', '1440']  # Server test uses these horizons
    
    for horizon in horizons:
        logger.info(f"Step 1: Generating features for horizon {horizon}...")
        result = run_generation_script(
            'generate_features.py',
            ['-f', start_str,
             '-t', end_str,
             '-c', test_config,
             '-z', horizon,
             '-o', 'test/fixtures/server/data/features',
             '-q'],  # Quiet mode
            cwd=cwd
        )
        
        if result.returncode != 0:
            logger.warning(f"Feature generation for horizon {horizon} had issues: {result.stderr}")
            # Continue anyway as we might have enough data
    
    # Step 2: Generate models to ensure they have all feature columns
    logger.info("Step 2: Generating models...")
    result = run_generation_script(
        'generate_models.py',
        ['--from', start_str,
         '--to', end_str,
         '--config', test_config,
         '--output-dir', 'test/fixtures/server/data/models',
         '--models', 'hl,slz',  # Server test uses these models
         '--horizons', '15,60,1440',
         '-q'],  # Quiet mode
        cwd=cwd
    )
    
    if result.returncode != 0:
        logger.warning(f"Model generation had issues: {result.stderr}")
        # Continue anyway
    
    logger.info("Fixture data generation complete")


@pytest.fixture
def master_server(test_dates: dt) -> AlphaServer:
    server_dir_manager = DirectoryManager(data_dir=FIXTURE_INPUT_DIR, trading_dir=FIXTURE_INPUT_DIR)
    server = AlphaServer(
        eod_today=test_dates,
        config_file=FIXTURE_CONFIG,
        slack_client=None,
        debug=False,
        trailing_days_to_load=2,
        server_dir_manager=server_dir_manager,
        latest_alpha_dir=os.path.join(FIXTURE_OUTPUT_DIR, "alpha"),
        raw_target_dir=os.path.join(FIXTURE_OUTPUT_DIR, 'raw_targets'),
        target_dir=os.path.join(FIXTURE_OUTPUT_DIR, 'targets'),
    )

    return server


@pytest.fixture
def test_server(test_dates: dt) -> AlphaServer:
    temp_dir = os.path.join(FIXTURE_DIR, 'temp_generated')
    temp_alpha_dir = os.path.join(temp_dir, 'alpha')
    temp_target_dir = os.path.join(temp_dir, 'targets')
    temp_raw_target_dir = os.path.join(temp_dir, 'raw_targets')
    server_dir_manager = DirectoryManager(data_dir=FIXTURE_INPUT_DIR, trading_dir=FIXTURE_INPUT_DIR)
    server = AlphaServer(
        eod_today=test_dates,
        config_file=FIXTURE_CONFIG,
        slack_client=None,
        debug=False,
        trailing_days_to_load=2,
        server_dir_manager=server_dir_manager,
        latest_alpha_dir=temp_alpha_dir,
        raw_target_dir=temp_raw_target_dir,
        target_dir=temp_target_dir,
    )

    return server


def test_generate_master_fixture(master_server: AlphaServer, test_dates: dt) -> None:
    try:
        logger.info(f"Start generating master fixture for server on {test_dates}")
        
        # NOTE: Feature regeneration is commented out for now as it takes too long
        # The features have already been regenerated with all required columns
        # Uncomment the following line if you need to regenerate features again:
        # generate_server_fixture_data(test_dates)
        
        # Delete existing fixture files
        logger.info("Deleting existing fixture files...")
        delete_all_files_in_tree(FIXTURE_OUTPUT_DIR)
        
        # Ensure directories exist
        _ = make_fixture_dir(os.path.join(FIXTURE_OUTPUT_DIR, 'alpha'))
        _ = make_fixture_dir(os.path.join(FIXTURE_OUTPUT_DIR, 'targets'))
        _ = make_fixture_dir(os.path.join(FIXTURE_OUTPUT_DIR, 'raw_targets'))
        
        master_server.update(st_only=False)
        master_server.generate_targets(optimize=True)
        logger.info(f"Finish generating master fixture for server on {test_dates}")
    except Exception as e:
        raise log_and_raise(f"An error occurred during generating master fixture: {str(e)}")


def test_server_run(test_server: AlphaServer) -> None:
    logger.info("Start testing new branch of server")

    temp_dir = os.path.join(FIXTURE_DIR, 'temp_generated')
    _ = make_fixture_dir(os.path.join(temp_dir, 'alpha'))
    temp_target_dir = make_fixture_dir(os.path.join(temp_dir, 'targets'))
    _ = make_fixture_dir(os.path.join(temp_dir, 'raw_targets'))
    _ = make_fixture_dir(os.path.join(FIXTURE_INPUT_DIR, 'live_bars'), keep_existing=True)
    _ = make_fixture_dir(os.path.join(FIXTURE_OUTPUT_DIR, 'alpha'), keep_existing=True)
    master_target_dir = make_fixture_dir(os.path.join(FIXTURE_OUTPUT_DIR, 'targets'), keep_existing=True)
    _ = make_fixture_dir(os.path.join(FIXTURE_OUTPUT_DIR, 'raw_targets'), keep_existing=True)

    master_fixture_exist = check_master_fixture_exist(master_target_dir)
    if not master_fixture_exist:
        raise log_and_raise("There is no existing master fixture files, please generate master fixtures first")

    try:
        res = True
        msg = ""
        logger.info("Start test server run")
        test_server.update(st_only=False)
        test_server.generate_targets(optimize=True)
        logger.info("Finish test server run")

        logger.info("Comparing targets files")
        master_target_df = load_target_csv_file(master_target_dir)
        temp_target_df = load_target_csv_file(temp_target_dir)

        res_pnl, msg_pnl, _ = compare_dataframes(master_target_df, temp_target_df, "Fixture", "Temp", "Target", rtol=1e-03)
        res &= res_pnl
        msg += msg_pnl

        if not res:
            master_target_df.to_csv(os.path.join(temp_dir, 'master.csv'))
            temp_target_df.to_csv(os.path.join(temp_dir, 'temp.csv'))
            raise Exception(msg)
        else:
            shutil.rmtree(temp_dir)
            logger.info(f"Temporary directory {temp_dir} has been removed.")
    except Exception as e:
        tb_str = traceback.format_exc()
        new_message = f"{str(e)}\n{tb_str}\n{LOG_MSG1}\n{LOG_MSG2}"
        raise RuntimeError(new_message) from e

    finally:
        logger.info(LOG_MSG1)
        logger.info(LOG_MSG2)

    logger.info("Server run and comparisons completed.")
    logger.info(f"Fixture files are located in: {FIXTURE_OUTPUT_DIR}")
    logger.info(f"Temporary files are located in: {temp_dir}")


if __name__ == '__main__':
    pytest.main([__file__])
