import os
import glob

import logging.config
from datetime import datetime as dt
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
from lib.alpha.features import Features
from lib.util.time_util import date_str_to_dt
from lib.util.config import get_config

logging.config.dictConfig(get_logging_config("features_integration_test"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FIXTURE_CONFIG: str = os.path.join(FIXTURE_DIR, "features", "config_features_generation_test.json")
FIXTURE_INPUT_DIR: str = os.path.join(FIXTURE_DIR, "features", "data")
FIXTURE_OUTPUT_DIR: str = os.path.join(FIXTURE_DIR, "features", "master")
TEST_HORIZON = 1440

LOG_MSG1 = "If test failed, please ssh -N -f -L localhost:8060:127.0.0.1:8060 user@your_server and open http://localhost:8060/fixtures/integration_test_generate_features_diff.html to check csv diff"
LOG_MSG2 = "Run ./bin/run_integration_test.sh -n feature to see the full log\n"
LOG_MSG2 += "Run ./bin/regenerate_master_fixture.sh -n feature to regenerate master fixture if it's a legit change"


@pytest.fixture
def test_dates() -> Dict[str, dt]:
    return {
        'start_dt': date_str_to_dt('20250609'),
        'end_dt': date_str_to_dt('20250609'),
    }


def load_features_parquet_file(fixture_dir: str) -> Optional[pd.DataFrame]:
    logger.debug(f"Attempting to load master parquet data from {fixture_dir}")
    try:
        # New directory structure: fixture_dir/horizon/feature/date/files.parquet
        file_pattern = f"{fixture_dir}/*/*/*/*features*.parquet"
        raw_target_files = glob.glob(file_pattern)
        if not raw_target_files:
            # Try old pattern
            file_pattern = f"{fixture_dir}/features_*.parquet"
            raw_target_files = glob.glob(file_pattern)
        
        dfs = []
        logger.info(f"Loading {len(raw_target_files)} files...")
        # Group by feature to avoid index conflicts
        feature_dfs = {}
        for raw_target_file in raw_target_files:
            # Extract feature name and symbol from path
            parts = raw_target_file.split('/')
            if len(parts) >= 4:
                feature_name = parts[-3]  # feature name is in the third-to-last position
                filename = parts[-1]  # e.g., features.1440.logret_1440_min.20240910.BTCUSDT.parquet
                # Extract symbol from filename
                symbol_match = filename.split('.')[-2]  # Get BTCUSDT
                symbol_venue = f"{symbol_match}_binance-futures"
            else:
                feature_name = 'unknown'
                symbol_venue = 'unknown'
            
            df = pd.read_parquet(raw_target_file)
            # Add symbol_venue to the dataframe
            df['symbol_venue'] = symbol_venue
            df = df.reset_index()
            df = df.set_index(['ts', 'symbol_venue'])
            
            if feature_name not in feature_dfs:
                feature_dfs[feature_name] = []
            feature_dfs[feature_name].append(df)
        
        # Concatenate within each feature, then merge across features
        merged_dfs = []
        for feature_name, feature_df_list in feature_dfs.items():
            feature_df = concat(feature_df_list)
            # Rename the value column to the feature name
            if len(feature_df.columns) == 1:
                feature_df.columns = [feature_name]
            merged_dfs.append(feature_df)
        
        # Merge all features horizontally
        if merged_dfs:
            fixture_df = merged_dfs[0]
            for df in merged_dfs[1:]:
                fixture_df = fixture_df.join(df, how='outer')
        else:
            fixture_df = pd.DataFrame()
            
    except Exception as e:
        logger.exception(f"Failed to load master parquet data from {fixture_dir}. Error: {str(e)}")
        return None
    logger.info(f"Successfully loaded master parquet data from {fixture_dir}")
    return fixture_df


def check_master_fixture_exist(master_target_dir):
    # Check if there are any parquet files in the horizon subdirectories
    if not os.path.exists(master_target_dir):
        return False
    
    for horizon_dir in os.listdir(master_target_dir):
        horizon_path = os.path.join(master_target_dir, horizon_dir)
        if os.path.isdir(horizon_path):
            # Check for feature subdirectories
            for feature_dir in os.listdir(horizon_path):
                feature_path = os.path.join(horizon_path, feature_dir)
                if os.path.isdir(feature_path):
                    # Check for date subdirectories  
                    for date_dir in os.listdir(feature_path):
                        date_path = os.path.join(feature_path, date_dir)
                        if os.path.isdir(date_path):
                            # Check for parquet files
                            if any(f.endswith(".parquet") for f in os.listdir(date_path)):
                                return True
    return False


@pytest.fixture
def master_feature() -> Features:
    _, config = get_config(FIXTURE_CONFIG)
    features_dir_manager = DirectoryManager(data_dir=FIXTURE_INPUT_DIR, trading_dir=FIXTURE_INPUT_DIR)

    feature_calc = Features(
        config=config,
        frequency=TEST_HORIZON,
        prod=False,
        debug=False,
        output_dir=FIXTURE_OUTPUT_DIR,
        features_dir_manager=features_dir_manager,
    )

    return feature_calc


@pytest.fixture
def test_feature() -> Features:
    temp_dir = os.path.join(FIXTURE_DIR, 'temp_generated')
    _, config = get_config(FIXTURE_CONFIG)
    features_dir_manager = DirectoryManager(data_dir=FIXTURE_INPUT_DIR, trading_dir=FIXTURE_INPUT_DIR)

    feature_calc = Features(
        config=config,
        frequency=TEST_HORIZON,
        prod=False,
        debug=False,
        output_dir=os.path.join(temp_dir, 'master'),
        features_dir_manager=features_dir_manager,
    )

    return feature_calc


def test_generate_master_fixture(master_feature: Features, test_dates: Dict[str, dt]) -> None:
    try:
        logger.info(f"Start generating master fixture for feature generation on {test_dates}")
        
        # Delete existing fixture files
        logger.info("Deleting existing fixture files...")
        delete_all_files_in_tree(FIXTURE_OUTPUT_DIR)
        
        _ = make_fixture_dir(FIXTURE_OUTPUT_DIR)
        master_feature.run(start_date=test_dates['start_dt'].date(), end_date=test_dates['end_dt'].date())
        logger.info(f"Finish generating master fixture for feature generation on {test_dates}")
    except Exception as e:
        raise log_and_raise(f"An error occurred during generating master fixture: {str(e)}")


def test_feature_run(test_feature: Features, test_dates: Dict[str, dt]) -> None:
    logger.info("Start testing new branch of feature generation")

    temp_dir = os.path.join(FIXTURE_DIR, 'temp_generated')
    temp_features_dir = make_fixture_dir(os.path.join(temp_dir, 'master'))
    master_features_dir = make_fixture_dir(os.path.join(FIXTURE_OUTPUT_DIR), keep_existing=True)

    master_fixture_exist = check_master_fixture_exist(master_features_dir)
    if not master_fixture_exist:
        raise log_and_raise("There is no existing master fixture files, please generate master fixtures first")

    try:
        logger.info("Start test feature run")
        test_feature.run(start_date=test_dates['start_dt'].date(), end_date=test_dates['end_dt'].date())
        logger.info("Finish test feature run")

        logger.info("Comparing master files")
        master_raw_features_df = load_features_parquet_file(master_features_dir)
        temp_raw_features_df = load_features_parquet_file(temp_features_dir)

        res, msg, _ = compare_dataframes(master_raw_features_df, temp_raw_features_df, "Fixture", "Temp", "Features", rtol=1e-03)
        if not res:
            master_raw_features_df.to_csv(os.path.join(temp_dir, 'master.csv'))
            temp_raw_features_df.to_csv(os.path.join(temp_dir, 'temp.csv'))
            raise ValueError(msg)
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

    logger.info("Features run and comparisons completed.")
    logger.info(f"Fixture files are located in: {FIXTURE_OUTPUT_DIR}")
    logger.info(f"Temporary files are located in: {temp_dir}")


if __name__ == '__main__':
    pytest.main([__file__])
