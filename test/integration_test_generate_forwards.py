import glob
import logging.config
import os
import shutil
import traceback
from datetime import datetime as dt, timedelta as td, date
from typing import Optional, Dict

import pandas as pd
import pytest

from lib.util.config import get_config
from lib.data.dataloader import DataLoader
from lib.util.dataframes import compare_dataframes
from lib.util.dataframes import concat
from lib.util.directory import FIXTURE_DIR, DirectoryManager, make_fixture_dir
from lib.fits.forwards import Forwards
from lib.util.time_util import date_str_to_dt, date_str_to_date
from lib.util.util import delete_all_files_in_tree
from lib.util.util import log_and_raise
from lib.util.logging_util import get_logging_config

logging.config.dictConfig(get_logging_config("forwards_integration_test"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FIXTURE_CONFIG: str = os.path.join(FIXTURE_DIR, "forwards",  "config_forwards_generation_test.json")
FIXTURE_INPUT_DIR: str = os.path.join(FIXTURE_DIR, "forwards", "data")
FIXTURE_OUTPUT_DIR: str = os.path.join(FIXTURE_DIR, "forwards", "master")
TEST_HORIZONS = [15]  # Use 15-minute horizon to minimize fixture data

LOG_MSG1 = "If test failed, please ssh -N -f -L localhost:8060:127.0.0.1:8060 user@your_server and open http://localhost:8060/fixtures/integration_test_generate_forwards_diff.html to check csv diff"
LOG_MSG2 = "Run ./bin/run_integration_test.sh -n master to see the full log\n"
LOG_MSG2 += "Run ./bin/regenerate_master_fixture.sh -n master to regenerate master fixture if it's a legit change"


@pytest.fixture
def test_dates() -> Dict[str, dt]:
    # Load config to get test dates
    _, config = get_config(FIXTURE_CONFIG)
    forward_test_date = config.get('FORWARD_TEST_DATE', '20250106')
    return {
        'start_dt': date_str_to_dt(forward_test_date),
        'end_dt': date_str_to_dt(forward_test_date),  # Single day test
    }


def load_forwards_parquet_file(fixture_dir: str, horizon: int = 15, start_date: dt.date = None, end_date: dt.date = None) -> Optional[pd.DataFrame]:
    logger.debug(f"Attempting to load master parquet data from {fixture_dir}")
    try:
        # Use DataLoader to properly load forward returns
        _, config = get_config(FIXTURE_CONFIG)
        # For master fixtures, the directory structure is different
        if 'master' in fixture_dir:
            data_dir = fixture_dir
        else:
            # For temp fixtures, we need to point to the parent data dir
            data_dir = os.path.dirname(fixture_dir)
        temp_dir_manager = DirectoryManager(data_dir=data_dir, trading_dir=data_dir)
        data_loader = DataLoader(config, temp_dir_manager)
        
        # If dates not provided, infer from directory structure
        if start_date is None or end_date is None:
            # Check if we're dealing with master fixtures or regular data
            if 'master' in fixture_dir:
                horizon_dir = f"{data_dir}/{horizon}"
            else:
                horizon_dir = f"{data_dir}/forwards/{horizon}"
                
            if os.path.exists(horizon_dir):
                date_dirs = sorted([d for d in os.listdir(horizon_dir) if os.path.isdir(os.path.join(horizon_dir, d))])
                if date_dirs:
                    start_date = date_str_to_date(date_dirs[0])
                    end_date = date_str_to_date(date_dirs[-1])
                else:
                    logger.error(f"No date directories found in {horizon_dir}")
                    return None
            else:
                logger.error(f"Horizon directory not found: {horizon_dir}")
                return None
        
        logger.info(f"Loading forwards with DataLoader from {data_dir} for dates {start_date} to {end_date}")
        
        # For master fixtures, we need to override the FORWARDS_DIR
        if 'master' in fixture_dir:
            # Temporarily override the FORWARDS_DIR
            original_forwards_dir = temp_dir_manager.FORWARDS_DIR
            temp_dir_manager.FORWARDS_DIR = data_dir
            
        fixture_df = data_loader.load_forward_returns(
            start_date=start_date,
            end_date=end_date,
            horizon=horizon
        )
        
        # Restore original value
        if 'master' in fixture_dir:
            temp_dir_manager.FORWARDS_DIR = original_forwards_dir
        
        if fixture_df is None:
            # Try old format
            file_pattern = f"{fixture_dir}/forward_{horizon}_*.parquet"
            forwards_files = sorted(glob.glob(file_pattern))
            
            if forwards_files:
                dfs = []
                for forwards_file in forwards_files:
                    df = pd.read_parquet(forwards_file)
                    dfs.append(df)
                if dfs:
                    fixture_df = concat(dfs)
        
        return fixture_df
        
    except Exception as e:
        logger.exception(f"Failed to load master parquet data from {fixture_dir}. Error: {str(e)}")
        return None


def check_master_fixture_exist(master_target_dir):
    # Check for new directory structure
    if os.path.exists(os.path.join(master_target_dir, "15")):
        horizon_dir = os.path.join(master_target_dir, "15")
        for date_dir in os.listdir(horizon_dir):
            date_path = os.path.join(horizon_dir, date_dir)
            if os.path.isdir(date_path) and any(f.endswith(".parquet") for f in os.listdir(date_path)):
                return True
    # Check old structure
    return any(f.endswith(".parquet") for f in os.listdir(master_target_dir) if os.path.isfile(os.path.join(master_target_dir, f)))




def test_generate_master_fixture(test_dates: Dict[str, dt]) -> None:
    try:
        logger.info(f"Start generating master fixture for master generation on {test_dates}")
        
        # Delete existing fixture files
        logger.info("Deleting existing fixture files...")
        delete_all_files_in_tree(FIXTURE_OUTPUT_DIR)
        
        # Check if required data exists
        bars_file = os.path.join(FIXTURE_INPUT_DIR, 'bars', '15', 'binance-futures', test_dates['start_dt'].strftime('%Y%m%d'), f"bars.15.binance-futures.{test_dates['start_dt'].strftime('%Y%m%d')}.BTCUSDT.parquet")
        if not os.path.exists(bars_file):
            raise log_and_raise("Bar data missing. Please run generate_fixtures.py to create test data.")
        else:
            logger.info("Test data found, proceeding with forwards generation")
        
        _ = make_fixture_dir(FIXTURE_OUTPUT_DIR, keep_existing=True)
        
        # Create Forwards instance and generate forward returns
        _, config = get_config(FIXTURE_CONFIG)
        forwards_dir_manager = DirectoryManager(data_dir=FIXTURE_INPUT_DIR, trading_dir=FIXTURE_INPUT_DIR)
        
        forwards_calc = Forwards(
            config=config,
            update=False,
            horizons=TEST_HORIZONS,
            debug=False,
            forwards_dir_manager=forwards_dir_manager,
            output_dir=FIXTURE_OUTPUT_DIR
        )
        
        # Run forward returns calculation
        logger.info(f"Generating forward returns for {test_dates['start_dt']} to {test_dates['end_dt']}")
        forwards_calc.generate_forwards(
            start_date=test_dates['start_dt'].date(), 
            end_date=test_dates['end_dt'].date()
        )
        
        logger.info(f"Finish generating master fixture for master generation on {test_dates}")
    except Exception as e:
        raise log_and_raise(f"An error occurred during generating master fixture: {str(e)}") from e


def test_forwards_run(test_dates: Dict[str, dt]) -> None:
    logger.info("Start testing new branch of master generation")

    temp_dir = os.path.join(FIXTURE_DIR, 'temp_generated')
    master_forwards_dir = make_fixture_dir(FIXTURE_OUTPUT_DIR, keep_existing=True)

    master_fixture_exist = check_master_fixture_exist(master_forwards_dir)
    if not master_fixture_exist:
        raise log_and_raise("There is no existing master fixture files, please generate master fixtures first")

    try:
        logger.info("Start test master run")
        
        # Create output directory for test
        temp_forwards_dir = os.path.join(temp_dir, 'master')
        os.makedirs(temp_forwards_dir, exist_ok=True)
        
        # Create Forwards instance and generate forward returns
        _, config = get_config(FIXTURE_CONFIG)
        forwards_dir_manager = DirectoryManager(data_dir=FIXTURE_INPUT_DIR, trading_dir=FIXTURE_INPUT_DIR)
        
        # Create a temporary directory manager that outputs to temp directory
        # Note: Forwards are saved to data_dir, not trading_dir
        temp_data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(temp_data_dir, exist_ok=True)
        
        # Symlink the input data directories for reading
        bars_link = os.path.join(temp_data_dir, 'bars')
        universe_link = os.path.join(temp_data_dir, 'universe')
        
        # Remove existing symlinks if they exist
        if os.path.islink(bars_link):
            os.unlink(bars_link)
        if os.path.islink(universe_link):
            os.unlink(universe_link)
            
        os.symlink(os.path.join(FIXTURE_INPUT_DIR, 'bars'), bars_link)
        os.symlink(os.path.join(FIXTURE_INPUT_DIR, 'universe'), universe_link)
        
        # Create output directories
        os.makedirs(os.path.join(temp_data_dir, 'forwards'), exist_ok=True)
        
        temp_dir_manager = DirectoryManager(data_dir=temp_data_dir, trading_dir=temp_dir)
        
        forwards_calc = Forwards(
            config=config,
            update=False,
            horizons=TEST_HORIZONS,
            debug=False,
            forwards_dir_manager=temp_dir_manager,
        )
        
        # Run forward returns calculation
        logger.info(f"Generating forward returns for {test_dates['start_dt']} to {test_dates['end_dt']}")
        forwards_calc.generate_forwards(
            start_date=test_dates['start_dt'].date(), 
            end_date=test_dates['end_dt'].date()
        )
        
        logger.info("Successfully generated forward returns")
        logger.info("Finish test master run")

        logger.info("Comparing master files")
        logger.info(f"Loading master forwards from: {master_forwards_dir}")
        try:
            master_forwards_df = load_forwards_parquet_file(
                master_forwards_dir, 
                horizon=15,
                start_date=test_dates['start_dt'].date(),
                end_date=test_dates['end_dt'].date()
            )
        except Exception as e:
            logger.error(f"Failed to load master forwards: {e}")
            raise
            
        # For new bars type, forwards are saved in data/forwards subdirectory
        temp_forwards_dir = os.path.join(temp_dir, 'data', 'forwards')
        logger.info(f"Loading temp forwards from: {temp_forwards_dir}")
        try:
            temp_forwards_df = load_forwards_parquet_file(
                temp_forwards_dir, 
                horizon=15,
                start_date=test_dates['start_dt'].date(),
                end_date=test_dates['end_dt'].date()
            )
        except Exception as e:
            logger.error(f"Failed to load temp forwards: {e}")
            raise

        if master_forwards_df is None or temp_forwards_df is None:
            raise ValueError("Failed to load forward returns for comparison")

        logger.info(f"Master forwards shape: {master_forwards_df.shape}")
        logger.info(f"Temp forwards shape: {temp_forwards_df.shape}")
        
        # Ensure both dataframes are sorted the same way for comparison
        master_forwards_df = master_forwards_df.sort_index()
        temp_forwards_df = temp_forwards_df.sort_index()
        
        res, msg, _ = compare_dataframes(master_forwards_df, temp_forwards_df, "Fixture", "Temp", "Forwards", 
                                        rtol=1e-10, atol=1e-10)
        if not res:
            master_forwards_df.to_csv(os.path.join(temp_dir, 'master.csv'))
            temp_forwards_df.to_csv(os.path.join(temp_dir, 'temp.csv'))
            logger.error(f"Comparison failed: {msg}")
            logger.error(f"Master columns: {sorted(master_forwards_df.columns.tolist())}")
            logger.error(f"Temp columns: {sorted(temp_forwards_df.columns.tolist())}")
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

    logger.info("Forwards run and comparisons completed.")
    logger.info(f"Fixture files are located in: {FIXTURE_OUTPUT_DIR}")
    logger.info(f"Temporary files are located in: {temp_dir}")


if __name__ == '__main__':
    pytest.main([__file__])