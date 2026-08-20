"""Directory management singleton for the stat_arb project.

This module provides a singleton DirectoryManager class that manages all
directory paths used throughout the project. It ensures consistent path
references and allows for easy configuration of data and trading directories.
"""
import glob
import logging
import os
import shutil
from typing import Optional, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Get root directory from environment - must be set
try:
    ROOT_DIR = os.environ['ROOT_DIR']
except KeyError:
    print("Please set ROOT_DIR!")
    raise

# Core directories
DATA_DIR = ROOT_DIR + "/data"
TRADING_DIR = ROOT_DIR + "/trading"

# Source and configuration directories
SRC_DIR = ROOT_DIR + "/src"
CONFIG_DIR = SRC_DIR + "/config"
TEST_DIR = SRC_DIR + "/test"
FIXTURE_DIR = TEST_DIR + "/fixtures"

# Output directories
LOG_DIR = ROOT_DIR + "/log"
REPORT_DIR = ROOT_DIR + "/reports"
SIM_DIR = ROOT_DIR + "/sims"

NOTRADE_FILE = f"{TRADING_DIR}/notrade.txt"


class DirectoryManager:
    """Singleton class for managing project directory paths.

    This class provides centralized management of all directory paths used
    throughout the stat_arb project. It uses the singleton pattern to ensure
    only one instance exists and can be reconfigured if needed.
    """

    _instance = None
    _current_data_dir = None
    _current_trading_dir = None

    def __new__(cls, data_dir: str = DATA_DIR, trading_dir: str = TRADING_DIR):
        """Create or return the singleton instance.

        Args:
            data_dir: Root directory for data files (default: ROOT_DIR/data)
            trading_dir: Root directory for trading files (default: ROOT_DIR/trading)

        Returns:
            The singleton DirectoryManager instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._current_data_dir = data_dir
            cls._current_trading_dir = trading_dir
        elif data_dir != cls._current_data_dir or trading_dir != cls._current_trading_dir:
            cls._current_data_dir = data_dir
            cls._current_trading_dir = trading_dir
            cls._instance.__init__(data_dir, trading_dir)
        return cls._instance

    def __init__(self, data_dir: str = DATA_DIR, trading_dir: str = TRADING_DIR):
        """Initialize directory paths.

        Args:
            data_dir: Root directory for data files
            trading_dir: Root directory for trading files
        """
        # Data directories
        self.DATA_DIR = data_dir
        self.DEFI_DIR = f"{data_dir}/defi"
        self.UNIVERSE_DIR = f"{data_dir}/universe"
        self.MARKETCAP_DIR = f"{data_dir}/secdata/marketcap"

        # Bar data directories
        self.BAR_DIR = f"{data_dir}/bars"
        self.PREBAR_DIR = f"{data_dir}/prebars"
        self.TARDIS_PREBAR_DIR = f"{self.PREBAR_DIR}/tardis/"
        self.LIVE_PREBAR_DIR = f"{self.PREBAR_DIR}/live/"

        # Security data directories
        self.FUNDING_DIR = f"{data_dir}/secdata/funding"
        self.EXCH_INFO_DIR = f"{data_dir}/secdata/exchange"
        self.BINANCE_META_DIR = f"{data_dir}/secdata/binance_meta"

        # Model and feature directories
        self.MODELS_DIR = f"{data_dir}/models"
        self.FEATURES_DIR = f"{data_dir}/features"
        self.FORWARDS_DIR = f"{data_dir}/forwards"

        # Alpha directories
        self.ALPHA_DIR_DEV = f"{data_dir}/alpha/dev"
        self.ALPHA_DIR_PROD = f"{data_dir}/alpha/prod"

        # Fits directories
        self.FITS_DIR_DEV = f"{data_dir}/fits/dev"
        self.FITS_DIR_PROD = f"{data_dir}/fits/prod"

        # Other data directories
        self.NEWS_DIR = f"{data_dir}/news"
        self.NEWS_DIR_NEW = f"{data_dir}/news_new"
        self.LIVE_DATA_DIR = f"{data_dir}/live"
        self.LIVE_DATA_NEW_DIR = f"{data_dir}/live_cpp/"

        # Trading directories
        self.TRADING_DIR = trading_dir
        self.UTIL_DIR = f"{trading_dir}/util"
        self.POSITION_DIR = f"{trading_dir}/positions"
        self.BINANCE_POSITION_DIR = f"{trading_dir}/binance_positions"
        self.BALANCES_DIR = f"{trading_dir}/balances"
        self.FILLS_DIR = f"{trading_dir}/fills"
        self.FUNDING_INCOME_DIR = f"{trading_dir}/funding"
        self.RAW_TARGET_DIR = f"{trading_dir}/raw_targets"
        self.BINANCE_FILLS_DIR = f"{trading_dir}/binance_fills"
        self.ORDERS_DIR = f"{trading_dir}/orders"
        self.TARGET_DIR = f"{trading_dir}/targets"
        self.RAW_OMS_DIR = f"{trading_dir}/raw_oms"
        self.NOTRADE_FILE = f"{trading_dir}/notrade.txt"

    def get_all_directories(self) -> dict:
        """Get all configured directories as a dictionary.

        Returns:
            Dictionary mapping directory names to their paths
        """
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith('_') and attr.isupper() and isinstance(getattr(self, attr), str)
        }


# Create the singleton instance
dir_manager = DirectoryManager()


def get_sim_dirs(sim_dir: Optional[str] = None) -> List[str]:
    """Get list of simulation directories sorted by creation time.

    Scans the simulations directory for valid simulation results, filtering
    out directories without PnL files.

    Args:
        sim_dir: Directory to scan. Defaults to SIM_DIR if None.

    Returns:
        List of simulation directory names sorted by creation time.

    Notes:
        - Only includes directories containing pnl.*csv files
        - Logs warnings for directories without PnL files
        - Returns directories sorted oldest to newest
    """
    sim_dir = SIM_DIR if sim_dir is None else sim_dir
    subdirectories = []
    with os.scandir(sim_dir) as entries:
        for entry in entries:
            if entry.is_dir():
                if not glob.glob(f"{sim_dir}/{entry.name}/pnl.*csv"):
                    logger.warning(f"Sim {entry.name} has not pnl file...")
                    continue
                subdirectories.append(entry)
    subdirectories.sort(key=os.path.getctime)
    logger.info(f"Found {len(subdirectories)} sim directories...")
    dirs = [d.name for d in subdirectories]
    return dirs


def make_fixture_dir(output_dir: str, keep_existing: bool = False) -> str:
    """Create directory for test fixtures, optionally removing existing.

    Creates a directory for storing test fixture files, with option to
    clean out existing contents first.

    Args:
        output_dir: Path to directory to create.
        keep_existing: If False, removes existing directory first. Defaults to False.

    Returns:
        The output_dir path.

    Notes:
        - Used for test fixture generation
        - Silently handles if directory already exists
        - Be careful with keep_existing=False as it deletes contents
    """
    if not keep_existing:
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass
    try:
        os.makedirs(output_dir, exist_ok=True)
    except FileExistsError:
        pass
    return output_dir
