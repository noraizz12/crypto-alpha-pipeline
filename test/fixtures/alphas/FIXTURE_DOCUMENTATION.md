# Alphas Fixture Documentation

## Purpose
The alphas fixtures test the alpha generation pipeline that combines multiple model predictions into trading signals.

## Fixture Generation
Run `python generate_fixtures.py` to generate test fixtures from production data.
- Use `--master-only` flag to only regenerate master fixtures without copying data

## Directory Structure
- **data/**: Contains input test data (copied from production)
  - **bars/{frequency}/binance-futures/{date}/**: Input bar files (1 and 15 minute)
  - **features/{horizon}/{feature_type}/{date}/**: Feature files for horizons 15 and 1440
  - **models/15/hl/**: Model prediction files for 15-minute hl model
  - **fits/**: Fits and SVM models with dev/ subdirectory structure
    - **dev/15/hl/**: Dev-prefixed fit files for prod=False mode
    - **svm/hl_15/**: SVM model files
  - **universe/**: Universe definition files
- **master/**: Contains expected output files
  - **{horizon}/{model}/**: Generated alpha files organized by horizon and model
    - Example: `15/hl/alphas.dev.15.hl.20250606.parquet`
  - **fits.csv**: Summary of model fits

## Test Configuration
- Config file: `config_alphas_generation_test.json`
- Test symbols: BTC, ETH, BNB (minimal set of 3)
- Test date: 20250606
- Test horizons: 15 minutes only
- Test models: ['hl'] (high-low model only)
- Bar frequencies: 15-minute bars only for test date
- Bar type: BARS_TYPE_NEW
- Historical days: 7 (20250530 to 20250606 for universe/models, plus 20250607 for fits)
- Uses prod=False mode (reads from fits/dev structure)

## Required Columns in Alpha Files

### Index Fields
- `ts`: Timestamp (datetime index)
- `symbol_venue`: Symbol and venue combination

### All Feature Columns
Alpha files inherit all columns from feature files

### Model Predictions
For each model and lag:
- `{model_name}_{horizon}_L{lag}`: Raw prediction
- `{model_name}_{horizon}_L{lag}_coeff`: Applied coefficient
- `{model_name}_{horizon}_L{lag}_condition`: Condition (rev/mom)
- `{model_name}_{horizon}_L{lag}_err`: Prediction error
- `{model_name}_{horizon}_L{lag}_weight`: Model weight

### Combined Alpha Signals
- `alpha_{model_name}_{horizon}`: Combined alpha for model
- `alpha_{model_name}_{horizon}_rev`: Reversal component
- `alpha_{model_name}_{horizon}_mom`: Momentum component

### Classification
- `class`: Classification result (-1, 0, 1)
- Applied to filter/weight alpha signals

### Risk Metrics
- `risk_{horizon}`: Volatility-based risk
- `st_risk_{horizon}`: Short-term risk

## Alpha Calculation Process
1. Load features and model predictions
2. Apply fitted coefficients to predictions
3. Apply classification filters
4. Combine multiple lags per model
5. Separate reversal and momentum components
6. Calculate final alpha signals

## Additional Notes
- The test expects 86 total files: 3 bars + 27 features + 8 models + 8 dev fits + 32 prod_fits + 8 universe files
- Feature types for 15-minute: dvolume_15_lz, dvolume_15_trmean_cz, fittable, logret_15_lz_cz, logret_15_trstd, relative_spread_15, trade_sz_15
- Feature types for 1440-minute: fittable, tradeable (only for test date)
- Prod_fits include both regular and dev CSV files, plus SVM model files (.joblib and .features)
- Dev fit files are copied to fits/dev/ directory for Forecasts class to use

## Usage Notes

1. **Running Tests**: The integration test compares generated alphas against expected values
2. **Fits Dependency**: Requires valid fits files with coefficients and SVM models
3. **Signal Combination**: Multiple model lags are combined with appropriate weights
4. **SVM Filtering**: Classification results modify alpha signals based on market regime