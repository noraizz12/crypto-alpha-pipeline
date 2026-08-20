# Unit Tests for data.py

This directory contains comprehensive unit tests for the `lib/data.py` module.

## Test Files

- **`test_data.py`**: Main unit tests covering core functionality
  - DataLoader class and methods
  - File loading functions (load_data_files, load_data_files_newformat)
  - News extraction and deduplication
  - Parquet file operations
  - Various utility functions

- **`test_data_edge_cases.py`**: Edge cases and error handling
  - Empty data handling
  - Invalid input validation
  - Error recovery scenarios
  - Performance and concurrency tests
  - Memory management tests

- **`conftest.py`**: Pytest configuration and shared fixtures

## Running the Tests

### Run all unit tests for data.py:
```bash
pytest test/test_data.py -v
```

### Run edge case tests:
```bash
pytest test/test_data_edge_cases.py -v
```

### Run both test files:
```bash
pytest test/test_data*.py -v
```

### Run with coverage:
```bash
pytest test/test_data*.py --cov=lib.data --cov-report=html
```

### Run specific test class:
```bash
pytest test/test_data.py::TestDataLoader -v
```

### Run specific test:
```bash
pytest test/test_data.py::TestDataLoader::test_dataloader_init -v
```

## Test Categories

Tests are organized into several categories:

1. **Core Functionality Tests** (`test_data.py`)
   - DataLoader initialization and methods
   - File loading and saving
   - Data transformation functions
   - Utility functions

2. **Edge Cases** (`test_data_edge_cases.py`)
   - Empty/missing data
   - Invalid inputs
   - Error conditions
   - Performance scenarios

3. **Integration Tests** (marked with `@pytest.mark.integration`)
   - Tests that involve multiple components
   - Full workflow tests

## Fixtures

Common fixtures are provided in `conftest.py`:
- `test_config`: Standard configuration dictionary
- `test_dates`: Common date ranges for testing
- `mock_environment`: Mock environment variables
- `sample_symbol_venues`: Sample trading symbols

## Adding New Tests

When adding new tests:

1. Use appropriate fixtures from `conftest.py`
2. Mock external dependencies (file I/O, network calls)
3. Test both success and failure cases
4. Use descriptive test names
5. Add docstrings explaining what each test verifies

## Mocking Guidelines

- Mock file system operations with `patch('lib.data.glob.glob')`
- Mock pandas read operations with `patch('lib.data.pd.read_parquet')`
- Mock datetime for time-sensitive tests
- Always clean up resources in tests

## Test Coverage Goals

Aim for:
- 80%+ line coverage for core functions
- 100% coverage for critical path functions
- Edge case coverage for all public APIs