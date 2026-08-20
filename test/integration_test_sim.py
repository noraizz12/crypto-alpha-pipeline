import logging.config
import os
import shutil
import sys
from datetime import datetime as dt, timezone, date
from typing import Any, Dict, List, Optional
import traceback
import pandas as pd
import pytest

from lib.util.config import get_config
from lib.sim.simulate import Simulate
from lib.util.time_util import DATE_TIME_FORMAT, date_range, date_str_to_dt, date_to_str
from lib.util.directory import FIXTURE_DIR, DirectoryManager, make_fixture_dir
from lib.util.util import get_config_universe, delete_all_files_in_tree
from lib.util.util import log_and_raise, SYMBOL_VENUE
from lib.util.logging_util import get_logging_config
from lib.util.dataframes import compare_dataframes

logging.config.dictConfig(get_logging_config("test_pnl_calculation"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FIXTURE_CONFIG: str = os.path.join(FIXTURE_DIR, "sim", "config_sim_test.json")
FIXTURE_INPUT_DIR: str = os.path.join(FIXTURE_DIR, "sim", "data")
FIXTURE_OUTPUT_DIR: str = os.path.join(FIXTURE_DIR, "sim", "master")

LOG_MSG1 = "If test failed, please ssh -N -f -L localhost:8060:127.0.0.1:8060 user@your_server and open http://localhost:8060/fixtures/integration_test_sim_diff.html to check csv diff"
LOG_MSG2 = "Run ./bin/run_integration_test.sh -n sim to see the full log\n"
LOG_MSG2 += "Run ./bin/regenerate_master_fixture.sh -n sim to regenerate master fixture if it's a legit change"


@pytest.fixture
def test_dates() -> Dict[str, Any]:
    return {
        'start_dt': date_str_to_dt('20240706'),
        'end_dt': date_str_to_dt('20240707'),  # 2 days of simulation
    }


@pytest.fixture()
def test_config() -> Dict[str, Any]:
    return get_config(FIXTURE_CONFIG)[1]


@pytest.fixture()
def test_sim_name() -> str:
    return 'master'


@pytest.fixture(autouse=True)
def cleanup_temp_dir():
    """Ensure temp directory is cleaned up before and after each test."""
    temp_dir = os.path.join(FIXTURE_DIR, 'temp_generated')
    
    # Clean up before test
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up existing temp directory before test: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory before test: {e}")
    
    yield  # Run the test
    
    # Clean up after test
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp directory after test: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory after test: {e}")


def test_symbol_venue_universe(test_config: Dict[str, Any]) -> None:
    symbol_venues = get_config_universe(test_config, symbol_type=SYMBOL_VENUE)
    assert isinstance(symbol_venues, list)
    assert len(symbol_venues) > 0
    assert all(isinstance(sv, str) and '_' in sv for sv in symbol_venues)


def make_test_sim_dir(output_dir: str, sim_name: Optional[str] = None, keep_existing: bool = False) -> str:
    if sim_name is None:
        sim_name = dt.now(timezone.utc).strftime(DATE_TIME_FORMAT)
    sim_dir = output_dir if sim_name == 'master' else output_dir + "/" + sim_name
    logger.info("-----------------")
    if len(sim_name) == 0:
        logger.error("not removing sim dir")
        sys.exit(1)

    return make_fixture_dir(sim_dir, keep_existing)


def load_daily_sim_parquet_file(fixture_dir: str, single_date: date) -> Optional[pd.DataFrame]:
    file_path = os.path.join(fixture_dir, f"sim.{date_to_str(single_date)}.parquet")
    logger.debug(f"Attempting to load {single_date} daily sim parquet data from {file_path}")
    try:
        fixture_df = pd.read_parquet(file_path)
    except Exception as e:
        logger.exception(f"Failed to load daily sim parquet data from {file_path}. Error: {str(e)}")
        return None
    logger.info(f"Successfully loaded daily sim parquet data from {file_path}")
    return fixture_df


def load_sim_csv_file(file_name: str, fixture_dir: str, sep: str = ',') -> Optional[pd.DataFrame]:
    file_path = os.path.join(fixture_dir, f"{file_name}.csv")
    logger.debug(f"Attempting to load {file_name} sim csv data from {file_path}")
    try:
        fixture_df = pd.read_csv(file_path, sep=sep, index_col=0)
    except Exception as e:
        logger.exception(f"Failed to load sim csv data from {file_path}. Error: {str(e)}")
        return None
    logger.info(f"Successfully loaded sim csv data from {file_path}")
    return fixture_df


def check_master_fixture_exist(master_sim_dir):
    return any(f.startswith("sim.") and f.endswith(".parquet") for f in os.listdir(master_sim_dir))


@pytest.fixture
def master_simulator(
        test_config: Dict[str, Any],
        test_sim_name: str,
    ) -> Simulate:

    master_sim_dir = make_test_sim_dir(FIXTURE_OUTPUT_DIR, test_sim_name)
    sim_dir_manager = DirectoryManager(data_dir=FIXTURE_INPUT_DIR, trading_dir=FIXTURE_INPUT_DIR)
    sim = Simulate(
        config=test_config,
        sim_dir=master_sim_dir,
        sim_comp=True,
        sim_dir_manager=sim_dir_manager,
    )

    return sim


@pytest.fixture
def test_simulator(
        test_config: Dict[str, Any],
        test_sim_name: str,
    ) -> Simulate:

    temp_dir = os.path.join(FIXTURE_DIR, 'temp_generated')
    temp_sim_dir = make_test_sim_dir(temp_dir, test_sim_name)
    sim_dir_manager = DirectoryManager(data_dir=FIXTURE_INPUT_DIR, trading_dir=FIXTURE_INPUT_DIR)
    sim = Simulate(
        config=test_config,
        sim_dir=temp_sim_dir,
        sim_comp=True,
        sim_dir_manager=sim_dir_manager,
    )

    return sim


def test_generate_master_fixture(
        master_simulator: Simulate,
        test_dates: Dict[str, dt],
        test_sim_name: str,
    ) -> None:

    start_dt = test_dates['start_dt']
    end_dt = test_dates['end_dt']
    start_date = start_dt.date()
    end_date = end_dt.date()
    try:
        logger.info(f"Start generating master fixture for simulation {test_sim_name} from {start_date} to {end_date}")
        
        # Delete existing fixture files
        logger.info("Deleting existing fixture files...")
        delete_all_files_in_tree(FIXTURE_OUTPUT_DIR)
        
        master_simulator.simulate(start_date=start_date, end_date=end_date)
        logger.info(f"Finish generating master fixture for simulation {test_sim_name} from {start_date} to {end_date}")
    except Exception as e:
        raise log_and_raise(f"An error occurred during generating master fixture: {str(e)}")


def test_simulate_calculation(
        test_simulator: Simulate,
        test_dates: Dict[str, dt],
        test_sim_name: str,
    ) -> None:
    logger.info("Start testing new branch of simulation")
    start_dt = test_dates['start_dt']
    end_dt = test_dates['end_dt']

    start_date = start_dt.date()
    end_date = end_dt.date()

    temp_dir = os.path.join(FIXTURE_DIR, 'temp_generated')
    temp_sim_dir = make_test_sim_dir(temp_dir, test_sim_name)
    _ = make_fixture_dir(os.path.join(FIXTURE_INPUT_DIR, 'live_bars'), keep_existing=True)

    master_sim_dir = make_test_sim_dir(FIXTURE_OUTPUT_DIR, test_sim_name, keep_existing=True)

    master_fixture_exist = check_master_fixture_exist(master_sim_dir)
    if not master_fixture_exist:
        raise log_and_raise("There is no existing master fixture files, please generate master fixtures first")

    try:
        res = True
        msg = ""
        logger.info("Start test simulation run")
        test_simulator.simulate(start_date=start_date, end_date=end_date)
        logger.info("Finish test simulation run")

        for single_date in date_range(start_date, end_date):
            logger.info(f"\nProcessing date: {single_date}")

            master_sim_df = load_daily_sim_parquet_file(master_sim_dir, single_date)
            temp_sim_df = load_daily_sim_parquet_file(temp_sim_dir, single_date)

            res_pnl, msg_pnl, _ = compare_dataframes(master_sim_df, temp_sim_df, "Fixture", "Temp", f"{single_date} Sim", rtol=1, atol=1e-04)
            if not res_pnl:
                res &= res_pnl
                msg += msg_pnl

            logger.info(f"Simulation run and comparison for {single_date} completed successfully.")

        logger.info("Comparing pnl files for sim")

        master_sim_pnl_calculator_df = load_sim_csv_file('pnl.calculator', master_sim_dir)
        temp_sim_pnl_calculator_df = load_sim_csv_file('pnl.calculator', temp_sim_dir)

        res_pnl, msg_pnl, _ = compare_dataframes(master_sim_pnl_calculator_df, temp_sim_pnl_calculator_df, "Fixture", "Temp", "Sim PnL Calculator", rtol=1, atol=1e-03)
        if not res_pnl:
            res &= res_pnl
            msg += msg_pnl

        if not res:
            master_sim_df.to_csv(os.path.join(temp_dir, 'master.csv'))
            temp_sim_df.to_csv(os.path.join(temp_dir, 'temp.csv'))
            raise ValueError(msg)
        
    except Exception as e:
        # Save CSV files for debugging even on exception
        if 'master_sim_df' in locals() and 'temp_sim_df' in locals():
            master_sim_df.to_csv(os.path.join(temp_dir, 'master.csv'))
            temp_sim_df.to_csv(os.path.join(temp_dir, 'temp.csv'))
        tb_str = traceback.format_exc()
        new_message = f"{str(e)}\n{tb_str}\n{LOG_MSG1}\n{LOG_MSG2}"
        raise RuntimeError(new_message) from e

    finally:
        # Always clean up temp directory
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Temporary directory {temp_dir} has been removed.")
            except Exception as cleanup_error:
                logger.warning(f"Failed to remove temporary directory {temp_dir}: {cleanup_error}")
        
        logger.info(LOG_MSG1)
        logger.info(LOG_MSG2)

    logger.info("Simulation run and comparisons completed.")
    logger.info(f"Fixture files are located in: {FIXTURE_OUTPUT_DIR}")


if __name__ == '__main__':
    pytest.main([__file__])
