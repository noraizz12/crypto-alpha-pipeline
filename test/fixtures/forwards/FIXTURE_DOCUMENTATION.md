# Forwards Fixture Documentation

## Purpose
The forwards fixtures test the forward return calculation pipeline that computes future returns for model training and evaluation.

## Fixture Generation
Run `python generate_fixtures.py` to generate test fixtures from production data.
- Use `--master-only` flag to only regenerate master fixtures without copying data

## Directory Structure
- **data/**: Contains input test data (copied from production)
  - **bars/{frequency}/binance-futures/{date}/**: Input bar files
  - **forwards/**: Pre-calculated forward returns (optional)
  - **universe/**: Universe definition files
- **master/**: Contains expected output files
  - **{horizon}/{date}/**: Generated forward files organized by horizon and date
    - Example: `15/20250107/forwards.15.20250107.BTCUSDT.parquet`

## Test Configuration
- Config file: `config_forwards_generation_test.json`
- Test symbols: BTC, ETH, BNB (from DEFAULT_TEST_SYMBOLS)
- Test date: 20250108 (last date in the file list)
- Test horizons: [15] minutes only
- Bar frequencies: [15] minutes only (as per file list)
- Bar type: BARS_TYPE_NEW
- Historical days: 2 (20250106 to 20250108)

## Required Columns in Forward Files

### Index Fields
- `ts`: Timestamp (datetime index)
- `symbol_venue`: Symbol and venue combination

### Forward Return Columns
For each forward-looking period:
- `fwd_logret_{period}`: Forward log return over period
- `fwd_logret_funding_adj_{period}`: Funding-adjusted forward log return
- `fwd_logret_resid_eqmkt_{period}`: Market-residualized forward return (equal-weighted)
- `fwd_logret_resid_wgtmkt_{period}`: Market-residualized forward return (volume-weighted)
- `fwd_logret_funding_adj_resid_eqmkt_{period}`: Funding-adjusted + equal-weighted residual
- `fwd_logret_funding_adj_resid_wgtmkt_{period}`: Funding-adjusted + volume-weighted residual

### Metadata
- `close_mid`: Current mid price
- `volume`: Current volume
- `dvolume`: Current dollar volume

## Forward Periods
Common forward periods include:
- 15 minutes
- 60 minutes
- 120 minutes
- 360 minutes
- 720 minutes
- 1440 minutes (1 day)
- 4320 minutes (3 days)
- 10080 minutes (1 week)

## Additional Notes
- The test expects 12 total files: 9 bar files (3 dates × 3 symbols × 1 frequency) + 3 universe files
- Forward files are generated using the Forwards class with update=False
- Generated forward files are copied from data/forwards/ to master/ directory
- The Forwards class handles all the forward return calculations based on the bar data

## Usage Notes

1. **Running Tests**: The integration test compares generated forwards in master/ against expected values
2. **Data Requirements**: Ensure sufficient bar data exists for all forward periods
3. **Market Residualization**: Forward returns include market-adjusted versions for alpha generation