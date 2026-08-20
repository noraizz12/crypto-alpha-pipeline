# Fits Fixture Documentation

## Purpose
The fits fixtures test the model fitting pipeline that trains alpha models and SVM classifiers on historical data.

## Fixture Generation
Run `python generate_fixtures.py` to generate test fixtures from production data.
- Use `--master-only` flag to only regenerate master fixtures without copying data

## Directory Structure
- **data/**: Contains input test data (copied from production)
  - **bars/{frequency}/binance-futures/{date}/**: Input bar files
  - **features/{horizon}/{feature_type}/{date}/**: Feature files
  - **models/{horizon}/{model}/**: Model prediction files
  - **forwards/{horizon}/{date}/**: Forward return files
  - **universe/**: Universe definition files
  - **prod_fits/**: Pre-calculated fits from production
- **master/**: Contains expected output files
  - **prod/{horizon}/{model}/**: Generated fit files
    - CSV files with regression coefficients
    - SVM model files (.joblib and .features)

## Test Configuration
- Config file: `config_fits_test.json`
- Test symbols: BTC, ETH, BNB (from DEFAULT_TEST_SYMBOLS)
- Test date: 20250107 (last date in the file list)
- Test horizon: 15 minutes only
- Test models: ['hl'] (high-low model only)
- Bar frequency: 15 minutes only (as per file list)
- Bar type: BARS_TYPE_NEW
- Historical days: 6 (20250101 to 20250107)
- Lookback parameters: 10 days (set in config update)
- Classification history: 2 days (set in config update)

## Required Columns in Fits CSV Files

### Model Coefficient Files
- `symbol`: Trading symbol
- `coeff`: Fitted coefficient
- `tstat`: T-statistic of coefficient
- `pvalue`: P-value of coefficient
- `r2`: R-squared of fit
- `n_obs`: Number of observations
- Additional model-specific metrics

## SVM Feature Files Format
The .features files contain feature importance scores:
```
insample: 0.5411003278579438
outsample: 0.5411001856726569
beta_1440: 0.0
day_of_week: 0.0021582174383647824
hour_of_day: 0.0
dvolume_1440_trmean_cz: 0.005094109160709076
dvolume_1440_lz: 0.0
logret_1440_trstd: 0.0
logret_1440_lz_cz: -0.009791700480988768
logret_1440_lz_abs: -0.0016169702200007918
trade_sz_1440_lz: -0.006669037615596971
relative_updates_1440_lz: -0.020493332596356363
hl_1440_L0: -0.06603587901893701
hl_1440_L0_abs: 0.0
logret_1440: 0.01254803497102165
dvolume_1440: 0.0
```

## Critical SVM Features
The SVM models expect these features to be present in the data:
- `beta_{horizon}`
- `day_of_week`
- `hour_of_day`
- `dvolume_{horizon}_trmean_cz`
- `dvolume_{horizon}_lz`
- `logret_{horizon}_trstd`
- `logret_{horizon}_lz_cz`
- `logret_{horizon}_lz_abs`
- `trade_sz_{horizon}_lz`
- `relative_updates_{horizon}_lz` (CRITICAL - often missing)
- `{model_name}_{horizon}_L{lag}` (model predictions)
- `{model_name}_{horizon}_L{lag}_abs` (absolute predictions)
- `logret_{horizon}` (raw returns)
- `dvolume_{horizon}` (dollar volume)

## Additional Notes
- The test expects 161 total files: 21 bars + 105 features + 21 forwards + 7 models + 7 universe files
- Feature types for 15-minute horizon: ba_imbal_15, dvolume_15_trmean, fittable, logret_15_trstd, trade_sz_15
- Master fixtures are generated only for the test date (20250107)
- Fits generation uses prod=True to match production behavior

## Usage Notes

1. **Running Tests**: The integration test trains models and compares against expected coefficients
2. **SVM Training**: Requires sufficient data variety for classification boundaries
3. **Feature Requirements**: SVM models need specific features present in the model files
4. **Coefficient Stability**: Small variations in coefficients are expected due to numerical precision