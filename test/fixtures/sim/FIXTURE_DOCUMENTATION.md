# Simulation Fixture Documentation

## Purpose
The simulation fixtures test the backtesting engine that simulates trading strategies using historical alpha signals and market data.

## Directory Structure

### Input Files (data/)
These files contain the input data required for simulation and are **NEVER modified** during regeneration:

#### Alpha Signals
- **tardis_prod_alpha/{model_name}_{horizon}_{date}.parquet**: Production alpha signals
  - Contains combined alpha signals for trading
  - Example: `hl_15_20240706.parquet`, `c2vwap_720_20240707.parquet`

#### Market Data
- **bars/{horizon}/{exchange}/{date}/bars.{horizon}.{exchange}.{date}.{symbol}.parquet**: Historical bar data
  - Hierarchical structure by horizon/exchange/date
  - Used for execution prices and slippage modeling
  - Multiple horizons needed (1min, 360min, 1440min)
  - Example: `bars/1440/binance-futures/20240706/bars.1440.binance-futures.20240706.BTCUSDT.parquet`

#### Features
- **features/{horizon}/{feature_name}/{date}/features.{horizon}.{feature_name}.{date}.{symbol}.parquet**: Feature data
  - Hierarchical structure by horizon/feature/date
  - Per-symbol feature files for each feature type
  - Used for risk calculations and constraints
  - Example: `features/1440/beta_1440/20240706/features.1440.beta_1440.20240706.BTCUSDT.parquet`

#### Universe
- **universe/universe.{date}.parquet**: Trading universe definition
  - Contains metadata for all tradeable symbols on each date

#### Other
- **delisting.txt**: Symbols to exclude from trading
- **live_bars/**: Alternative bar data source (may be empty)

### Output Files (master/)
These files contain the simulation results that are compared against:

#### Simulation Results
- **sim.{date}.parquet**: Daily simulation results
  - Contains trades, positions, and PnL for each day
  - Example: `sim.20240706.parquet`

#### PnL Analysis
- **pnl.breakdown.csv**: PnL breakdown by various factors
- **pnl.calculator.csv**: Detailed PnL calculations
- **pnl_details.parquet**: Transaction-level PnL details
- **summary.txt**: High-level simulation summary

## Regeneration Behavior

When running `./bin/regenerate_master_fixture.sh -n sim`:

1. **data/ directory**: Remains completely untouched
   - All input files (alphas, bars, features) are preserved
   - No files are added, modified, or deleted

2. **master/ directory**: Completely regenerated
   - All existing simulation results are deleted
   - The simulation pipeline is run:
     - Reads alpha signals from data/prod_alpha/
     - Uses bar data for execution modeling
     - Applies portfolio constraints and risk limits
     - Simulates trades with realistic fees and slippage
     - Calculates detailed PnL attribution
     - Writes all results to master/

**Important**: Simulation requires:
- Alpha files in `tardis_prod_alpha/`
- 1-minute bars for execution (if configured) in `bars/1/`
- Bars for configured horizons (360min, 1440min) in respective `bars/{horizon}/` directories
- Feature data for risk calculations in `features/`
- Universe files for the simulation period in `universe/`

## Required Columns in Simulation Files

### sim.{date}.parquet
- `ts`: Timestamp
- `symbol`: Trading symbol
- `position`: Current position (shares)
- `position_dollars`: Position value in USD
- `trade`: Trade size (shares)
- `trade_dollars`: Trade value in USD
- `price`: Execution price
- `alpha`: Alpha signal used
- `target`: Target position
- `pnl`: Realized PnL
- `unrealized_pnl`: Unrealized PnL
- `total_pnl`: Total PnL
- `fees`: Transaction fees
- `slippage`: Estimated slippage cost

### pnl.breakdown.csv
- `category`: Breakdown category (symbol, date, model, etc.)
- `value`: Category value
- `pnl`: Total PnL for category
- `trades`: Number of trades
- `turnover`: Total turnover
- `sharpe`: Sharpe ratio
- `max_drawdown`: Maximum drawdown

### pnl_details.parquet
- `ts`: Trade timestamp
- `symbol`: Trading symbol
- `trade`: Trade size
- `price`: Execution price
- `fees`: Transaction fees
- `slippage`: Slippage cost
- `pnl`: Trade PnL
- `position_before`: Position before trade
- `position_after`: Position after trade

## Simulation Parameters (from config)
- `MAX_PORTFOLIO_NOTIONAL`: Maximum portfolio size
- `MAX_POSITION_PCT`: Maximum position as % of portfolio
- `MAX_TRADE_DOLLARS`: Maximum single trade size
- `EXCHANGE_FEES`: Exchange fee rate
- `DEFAULT_SLIPPAGE`: Default slippage assumption
- `MAX_VOLUME_FRACTION_PARTICIPATION`: Max % of volume to trade

## Test Configuration
- Config file: `config_sim_test.json`
- Test dates: 20240706 to 20240707 (2 simulation days)
- Historical data: 20240705 to 20240707 (includes 1 extra day for features)
- Test symbols: BTC, ETH, BNB, ADA, ETC (5 symbols)
- Alpha models: hl_15, c2vwap_720
- Bar frequencies: 360, 1440 minutes only
- Feature horizons: 360, 1440 minutes
- Execution price: VWAP
- Portfolio size: Configured in config_sim_test.json

## File Organization Notes

The fixture data follows the standard data organization pattern:
- **Bars**: `bars/{horizon}/{exchange}/{date}/` with per-symbol files
- **Features**: `features/{horizon}/{feature_name}/{date}/` with per-symbol files
- **Universe**: `universe/universe.{date}.parquet` files
- **Alpha signals**: `tardis_prod_alpha/{model}_{horizon}_{date}.parquet` files

This hierarchical structure matches the production data layout and makes it easy to locate specific data files.

## Additional Notes

The fixture expects 278 total files:
- **bars/**: 30 files (2 frequencies × 3 dates × 5 symbols)
- **features/360/**: 90 files (6 feature types × 3 dates × 5 symbols)
  - Feature types: dvolume_360, dvolume_360_trmean, dvolume_360_trmean_cz, logret_360_lz, logret_360_trstd, relative_spread_360
- **features/1440/**: 150 files (10 feature types × 3 dates × 5 symbols)
  - Feature types: beta_1440, dvolume_1440, dvolume_1440_trmean, dvolume_1440_trmean_cz, logret_1440_cz, logret_1440_lz, logret_1440_lz_cz, logret_1440_trstd, relative_spread_1440_trmean, risk_1440
- **tardis_prod_alpha/**: 4 files (2 models × 2 simulation dates)
- **universe/**: 3 files (one per date)
- **delisting.txt**: 1 file

## Usage Notes

1. **Running Tests**: The integration test runs full backtests and compares results
2. **Execution Modeling**: Uses bars at configured frequencies (360, 1440) for execution
3. **Portfolio Constraints**: Applies position limits, volume constraints, and risk management
4. **PnL Attribution**: Breaks down returns by symbol, date, and alpha source
5. **Master Fixtures**: Generated by running the Simulate class on the test data