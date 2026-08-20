# CLAUDE.md

This file provides guidance when working with code in this repository.

## System Overview

This is a **market data engineering system** for cryptocurrency futures, implementing data ingestion, feature engineering, and signal generation for Binance futures markets.

## Code Style Guidelines

- **Pylint Code Quality**: Ensure a pylint score of **at least 9.0** (preferably 10.0)
  - Fix all errors (E) and warnings (W)
  - Use proper import ordering: standard library → third-party → first-party (lib imports)
  - Always include final newlines in files
  - Use `raise ... from e` for proper exception chaining
  - Include a `--debug` option on all scripts that does not persist anything

- **Type Hints**: Always use type hints for function arguments and return types
  ```python
  def calculate_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
      """Calculate returns over specified horizon"""
      ...
  ```

- **DataFrame Conventions**:
  - Index should be `['ts', 'symbol_venue']` when possible
  - Default resolution is 1-minute
  - Cumulative data columns end in `_cum`
  - Default to `np.float32` for floats
  - Add `_df` suffix to DataFrame variable names

- Avoid functions over 100 lines
- Do not use `hasattr`

## Development Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment
export ROOT_DIR=$(pwd)
cp .env.example .env
# Edit .env with your configuration
```

## Key Commands

**Testing:**
```bash
# Run all tests
pytest test/ -v

# Run only unit tests
./bin/run_tests.sh -u

# Run only integration tests
./bin/run_tests.sh -i

# Run specific integration test
./bin/run_integration_test.sh -n bars
```

**Data Pipeline:**
```bash
# Generate bars
python generate_bars.py --from 20240101 --to 20240131 --debug

# Generate features
python generate_features.py --from 20240101 --to 20240131 --debug

# Generate forward returns
python generate_forwards.py --from 20240101 --to 20240131 --debug
```

## Git Workflow

**Branch Strategy:**
- **master**: Production-ready code
- **develop**: Integration branch
- **feature/**: New features
- **fix/**: Bug fixes

```bash
# Create feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/your-feature

# Make changes and commit
git add <files>
git commit -m "Description of changes"

# Push and create PR
git push -u origin feature/your-feature
```

## Architecture Overview

### Data Flow
```
Raw Data → Prebars → Bars → Features → Forward Returns → Models → Alphas → Targets
```

### Key Directories
- `lib/bars/` - Bar aggregation from raw ticks
- `lib/calcs/` - Feature calculations
- `lib/data/` - Data loading utilities
- `lib/external/` - Exchange API clients
- `lib/fits/` - Model training
- `lib/alpha/` - Signal generation
- `lib/trader/` - Order management
- `lib/util/` - Shared utilities

### Configuration
All parameters are in `config/config.json`. See `config/config.example.json` for reference.

## Testing Framework

- **Unit Tests**: `test/test_*.py` - Fast, isolated component tests
- **Integration Tests**: `test/integration_test_*.py` - Full pipeline validation
- Uses pytest with fixture-based comparisons
