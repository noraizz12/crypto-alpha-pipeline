import logging.config
import os
import shutil
from argparse import Namespace
from datetime import date
from datetime import datetime as dt, date
from datetime import timedelta as td
from typing import List, Dict, Any
import traceback
import pandas as pd
import pytest

from lib.bars.bar_generator import BarGenerator
from lib.util.config import get_config
from lib.util.time_util import date_str_to_dt, date_to_str, date_range
from lib.util.directory import FIXTURE_DIR, DirectoryManager
from lib.util.util import TARDIS_EXCHANGE, delete_all_files_in_tree
from lib.util.util import log_and_raise
from lib.util.logging_util import get_logging_config
from lib.util.dataframes import compare_dataframes, log_dataframe_summary

logging.config.dictConfig(get_logging_config("test_bar_generation"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TEST_FREQUENCY: int = 15
FIXTURE_CONFIG: str = os.path.join(FIXTURE_DIR, "bars", "config_bar_generation_test.json")
FIXTURE_INPUT_DIR = os.path.join(FIXTURE_DIR, "bars", "data", "prebars", "live")
FIXTURE_OUTPUT_DIR = os.path.join(FIXTURE_DIR, "bars", "master")
FIXTURE_DATA_DIR = os.path.join(FIXTURE_DIR, "bars", "data")

LOG_MSG1 = "If test failed, please ssh -N -f -L localhost:8060:127.0.0.1:8060 user@your_server and open http://localhost:8060/fixtures/integration_test_generate_bars_diff.html to check csv diff"
LOG_MSG2 = "Run ./bin/run_integration_test.sh -n bars to see the full log\n"
LOG_MSG2 += "Run ./bin/regenerate_master_fixture.sh -n bars to regenerate master fixture if it's a legit change"


# DirectoryManager will be created in each function to ensure proper test isolation


def generate_bars_for_test(start_date: date, end_date: date, config: Dict[str, Any], input_dir: str, output_dir: str) -> None:
    # Create DirectoryManager for this test
    bar_test_dir_manager = DirectoryManager(data_dir=FIXTURE_DATA_DIR, trading_dir=FIXTURE_DATA_DIR)
    
    args = Namespace(
        venue=TARDIS_EXCHANGE,
        debug=False,  # Need False to actually write files
        pool_size=6,
    )
    bar_generator = BarGenerator(
        config=config,
        venue=args.venue,
        start_date=start_date,
        end_date=end_date,
        chunk_days=7,  # Reduced from 30 since we only test 2 days
        horizons=[TEST_FREQUENCY],
        debug=args.debug,
        pool_size=args.pool_size,
        output_dir=output_dir,
        bars_dir_manager=bar_test_dir_manager,
    )
    logger.info(f"BarGenerator created with output_dir: {bar_generator.output_dir}")
    bar_generator.run()


@pytest.fixture(scope="session", autouse=True)
def check_fixtures():
    missing_dirs = []
    for dir_path in [FIXTURE_INPUT_DIR, FIXTURE_OUTPUT_DIR]:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)

    if missing_dirs:
        pytest.skip(
            f"Fixture directories not found: {', '.join(missing_dirs)}. Please set up fixtures before running tests.")


@pytest.fixture()
def test_dates() -> Dict[str, dt]:
    # Only test 1 day since that's sufficient to verify functionality
    return {
        'start_dt': date_str_to_dt('20250609'),
        'end_dt': date_str_to_dt('20250609'),
    }


@pytest.fixture()
def test_coins() -> List[str]:
    # Only test 3 coins to speed up the test
    return [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
    ]


@pytest.fixture()
def test_config() -> Dict[str, Any]:
    return get_config(FIXTURE_CONFIG)[1]


def load_fixture_bars(date: date, test_coins: List[str]) -> pd.DataFrame:
    bars_list: List[pd.DataFrame] = []
    date_dir = os.path.join(FIXTURE_INPUT_DIR, date.strftime('%Y%m%d'))
    if not os.path.exists(date_dir):
        logger.warning(f"Directory not found for date: {date}")
        return pd.DataFrame()
    
    # No mapping needed - use the full symbol names directly
    for coin in test_coins:
        # Build the prebar file name pattern using the full symbol
        file_name = f"prebars.live.binance-futures.{date.strftime('%Y%m%d')}.{coin}.parquet"
        file_path = os.path.join(date_dir, "binance-futures", file_name)
        if not os.path.exists(file_path):
            logger.warning(f"Fixture input file not found: {file_path}")
            continue
        df = pd.read_parquet(file_path)
        bars_list.append(df)
    if not bars_list:
        logger.warning(f"No input fixture files found for date {date}")
        return pd.DataFrame()
    return pd.concat(bars_list)


def consolidate_bar_files(output_dir: str, horizon: int, date: date, symbols: List[str]) -> pd.DataFrame:
    """Consolidate individual symbol bar files from new format into single dataframe."""
    date_str = date_to_str(date)
    bar_files = []
    
    # Look for files in the new hierarchical structure
    bar_dir = os.path.join(output_dir, str(horizon), TARDIS_EXCHANGE, date_str)
    
    if os.path.exists(bar_dir):
        for symbol in symbols:
            file_path = os.path.join(bar_dir, f"bars.{horizon}.{TARDIS_EXCHANGE}.{date_str}.{symbol}.parquet")
            if os.path.exists(file_path):
                logger.info(f"Reading bar file: {file_path}")
                df = pd.read_parquet(file_path)
                # The bar files only have 'ts' as index, need to add symbol_venue
                df = df.reset_index()
                df['symbol'] = symbol
                df['venue'] = TARDIS_EXCHANGE
                df['symbol_venue'] = f"{symbol}_{TARDIS_EXCHANGE}"
                # Set the standard MultiIndex
                df = df.set_index(['ts', 'symbol_venue'])
                bar_files.append(df)
            else:
                logger.warning(f"Bar file not found: {file_path}")
    else:
        logger.warning(f"Bar directory not found: {bar_dir}")
    
    if bar_files:
        return pd.concat(bar_files)
    else:
        return pd.DataFrame()


def test_generate_master_fixture(test_dates: Dict[str, dt], test_config: Dict[str, Any]) -> None:
    try:
        start_date: date = test_dates['start_dt'].date()
        end_date: date = test_dates['end_dt'].date()
        
        # Check if master fixtures already exist
        freq_dir = os.path.join(FIXTURE_OUTPUT_DIR, str(TEST_FREQUENCY), "binance-futures", date_to_str(start_date))
        if os.path.exists(freq_dir) and os.listdir(freq_dir):
            logger.info(f"Master fixture files already exist for {start_date}, skipping generation")
            return
        
        # Delete existing fixture files
        logger.info("Deleting existing fixture files...")
        delete_all_files_in_tree(FIXTURE_OUTPUT_DIR)
        
        unexpected_output_dir = FIXTURE_DIR
        for single_date in date_range(start_date, end_date):
            logger.info(f"\nProcessing date: {single_date}")
            fixture_file = os.path.join(FIXTURE_OUTPUT_DIR, f"bars_{TEST_FREQUENCY}_{date_to_str(single_date)}.parquet")
            logger.info(f"No existing bar files. Generating bars for: {single_date} to {single_date + td(days=1)}")
            logger.info(f"Output file: {fixture_file}")
            generate_bars_for_test(single_date, single_date, test_config, FIXTURE_INPUT_DIR, FIXTURE_OUTPUT_DIR)

            # Consolidate the bar files from new format
            symbols = test_config.get('SYMBOL_UNIVERSE', ['ADA', 'BNB', 'BTC', 'ETH', 'XRP'])
            fixture_df = consolidate_bar_files(FIXTURE_OUTPUT_DIR, TEST_FREQUENCY, single_date, symbols)
            
            if fixture_df.empty:
                logger.error(f"Failed to generate bar files for {single_date}")
                pytest.fail(f"Failed to generate bar files for {single_date}")
            
            # Save the consolidated file for test compatibility
            fixture_df.to_parquet(fixture_file)
            logger.info(f"Saved consolidated bar file: {fixture_file}")
            
            assert not fixture_df.empty, f"Generated data is empty for {single_date} in file {fixture_file}"

            logger.info(f"Successfully generated bars for {single_date} in fixture directory")
            # log_dataframe_summary(fixture_df, "Generated Fixture Data")
    except Exception as e:
        raise log_and_raise(f"An error occurred during generating master fixture: {str(e)}")


def test_bar_generation(test_dates: Dict[str, date], test_coins: List[str], test_config: Dict[str, Any]) -> None:
    logger.info("Start testing bar generation")
    start_date: date = test_dates['start_dt'].date()
    end_date: date = test_dates['end_dt'].date()
    temp_dir = os.path.join(FIXTURE_DIR, 'temp_generated')
    unexpected_output_dir = FIXTURE_DIR

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(FIXTURE_OUTPUT_DIR, exist_ok=True)

    logger.info(f"Fixture input directory: {FIXTURE_INPUT_DIR}")
    logger.info(f"Fixture output directory: {FIXTURE_OUTPUT_DIR}")
    logger.info(f"Temporary directory: {temp_dir}")
    logger.info(f"Unexpected output directory: {unexpected_output_dir}")

    def bar_files_exist():
        # Check for bar files in the new directory structure
        freq_dir = os.path.join(FIXTURE_OUTPUT_DIR, str(TEST_FREQUENCY), "binance-futures")
        if os.path.exists(freq_dir):
            # Check if any date subdirectories exist
            subdirs = [d for d in os.listdir(freq_dir) if os.path.isdir(os.path.join(freq_dir, d))]
            return len(subdirs) > 0
        return False

    fixture_bars_exist = bar_files_exist()
    logger.info(f"Existing bar files found in fixture directory: {'Yes' if fixture_bars_exist else 'No'}")

    try:
        logger.info(f"Test start date: {start_date}")
        logger.info(f"Test end date: {end_date}")

        for single_date in date_range(start_date, end_date):
            logger.info(f"\nProcessing date: {single_date}")

            # Use load_fixture_bars to load existing fixture data
            existing_fixture_data = load_fixture_bars(single_date, test_coins)
            if not existing_fixture_data.empty:
                logger.info("Existing fixture data found:")
                # log_dataframe_summary(existing_fixture_data, "Existing Fixture Data")

            fixture_file = os.path.join(FIXTURE_OUTPUT_DIR, f"bars_{TEST_FREQUENCY}_{date_to_str(single_date)}.parquet")
            temp_file = os.path.join(temp_dir, f"bars_{TEST_FREQUENCY}_{date_to_str(single_date)}.parquet")

            # Generate bars in fixture directory if no bar files exist
            if not fixture_bars_exist:
                raise log_and_raise("There is no existing master fixture files, please generate master fixtures first")

            # Generate bars in temp directory for comparison
            logger.info(f"Generating bars for comparison: {single_date} to {single_date + td(days=1)}")
            logger.info(f"Output file: {temp_file}")
            generate_bars_for_test(single_date, single_date, test_config, FIXTURE_INPUT_DIR, temp_dir)

            # Consolidate the bar files from new format
            symbols = test_config.get('SYMBOL_UNIVERSE', ['ADA', 'BNB', 'BTC', 'ETH', 'XRP'])
            temp_df = consolidate_bar_files(temp_dir, TEST_FREQUENCY, single_date, symbols)
            
            if temp_df.empty:
                logger.error(f"Failed to generate bar files for {single_date} in temp directory")
                pytest.fail(f"Failed to generate bar files for {single_date}")
            
            # Save the consolidated file for comparison
            temp_df.to_parquet(temp_file)
            logger.info(f"Saved consolidated temp file: {temp_file}")
            
            assert not temp_df.empty, f"Generated data is empty for {single_date} in file {temp_file}"

            # Compare the generated files if bar files already existed
            if fixture_bars_exist:
                logger.info(f"Comparing files for {single_date}")
                fixture_df = pd.read_parquet(fixture_file)
                # Sort both dataframes before comparison to ensure consistent ordering
                fixture_df = fixture_df.sort_index()
                temp_df = temp_df.sort_index()
                res, msg, _ = compare_dataframes(fixture_df, temp_df, "Fixture", "Temp", "Bars", rtol=1e-04, atol=1e-06)
                if not res:
                    fixture_df.to_csv(os.path.join(temp_dir, 'master.csv'))
                    temp_df.to_csv(os.path.join(temp_dir, 'temp.csv'))
                    raise Exception(msg)
                else:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Temporary directory {temp_dir} has been removed.")

            logger.info(f"Bar generation and comparison for {single_date} completed successfully.")

    except Exception as e:
        tb_str = traceback.format_exc()
        new_message = f"{str(e)}\n{tb_str}\n{LOG_MSG1}\n{LOG_MSG2}"
        raise RuntimeError(new_message) from e

    finally:
        logger.info(LOG_MSG1)
        logger.info(LOG_MSG2)

    logger.info("All bar generations and comparisons completed.")
    logger.info(f"Fixture files are located in: {FIXTURE_OUTPUT_DIR}")
    logger.info(f"Temporary files are located in: {temp_dir}")


if __name__ == '__main__':
    pytest.main([__file__])
