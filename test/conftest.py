"""
Pytest configuration and shared fixtures for unit tests
"""
import os
import sys
import warnings
from datetime import date

import pytest

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Disable certain warnings during tests
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture(scope="session")
def test_config():
    """Provide a test configuration dictionary"""
    return {
        'SYMBOL_UNIVERSE': ['BTC', 'ETH', 'SOL', 'MATIC', 'LINK'],
        'MIN_ADVP_TRADEABLE': 1.5e7,
        'MIN_ADVP_FITTABLE': 1.5e7,
        'MIN_ADVP_EXPANDABLE': 5.5e7,
        'MIN_ADVP_PRICEABLE': 2.5e7,
        'EXCHANGE_FEES': 0.00005,
        'DEFAULT_SLIPPAGE': 0.0001,
        'MAX_PORTFOLIO_NOTIONAL': 2.0e7,
        'MAX_POSITION_PCT': 0.04,
        'FILTER_DELISTING': True,
        'DELISTING_BUFFER_DAYS': 6,
        'NEWS_SIMILARITY_THRESHOLD': 0.9,
    }


@pytest.fixture(scope="session")
def test_dates():
    """Provide common test dates"""
    return {
        'start': date(2024, 1, 1),
        'end': date(2024, 1, 31),
        'mid': date(2024, 1, 15),
    }


@pytest.fixture(scope="function")
def mock_environment(monkeypatch):
    """Set up mock environment variables"""
    monkeypatch.setenv("USER", "test_user")


@pytest.fixture(scope="function", autouse=True)
def reset_directory_manager():
    """Reset DirectoryManager singleton before each test to ensure test isolation"""
    from lib.util.directory import DirectoryManager
    # Store the original instance
    original_instance = DirectoryManager._instance
    original_data_dir = DirectoryManager._current_data_dir
    original_trading_dir = DirectoryManager._current_trading_dir
    
    # Reset before test
    DirectoryManager._instance = None
    DirectoryManager._current_data_dir = None
    DirectoryManager._current_trading_dir = None
    
    yield
    
    # Restore after test
    DirectoryManager._instance = original_instance
    DirectoryManager._current_data_dir = original_data_dir
    DirectoryManager._current_trading_dir = original_trading_dir


@pytest.fixture(scope="session")
def sample_symbol_venues():
    """Provide sample symbol_venue combinations"""
    return [
        'BTCUSDT_binance-futures',
        'ETHUSDT_binance-futures',
        'SOLUSDT_binance-futures',
        'MATICUSDT_binance-futures',
        'LINKUSDT_binance-futures',
    ]


# Custom markers for categorizing tests
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "requires_data: marks tests that require actual data files")
