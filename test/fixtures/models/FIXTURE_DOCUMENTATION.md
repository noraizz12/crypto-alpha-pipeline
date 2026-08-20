# Models Fixture Documentation

## Purpose
The models fixtures test the model generation pipeline that creates alpha predictions using features and historical returns.

## Fixture Generation
Run `python generate_fixtures.py` to generate test fixtures from production data.
- Use `--master-only` flag to only regenerate master fixtures without copying data

## Directory Structure
- **data/**: Contains input test data (copied from production)
  - **bars/{frequency}/binance-futures/{date}/**: Input bar files
  - **features/{horizon}/{feature_type}/{date}/**: Feature files
  - **universe/**: Universe definition files
- **master/**: Contains expected output files
  - **{horizon}/{model}/{filename}**: Generated model files organized by horizon and model
    - Example: `1440/hl/models.1440.hl.20250108.parquet`

## Test Configuration
- Config file: `config_models_generation_test.json`
- Test symbols: BTC, ETH, BNB (loaded from config SYMBOL_UNIVERSE)
- Test date: 20250108 (or from config MODEL_TEST_DATE)
- Test horizon: 1440 minutes (daily) only
- Test models: ['hl'] (high-low model only)
- Bar frequency: 1440 minutes only (as per file list)
- Bar type: BARS_TYPE_NEW
- Historical days: 2 (20250106 to 20250108)

## Required Columns in Model Files

### Index Fields
- `ts`: Timestamp (datetime index)
- `symbol_venue`: Symbol and venue combination

### All Feature Columns
Model files should contain ALL columns from the corresponding feature files, including:
- All return-based features (`logret_*`, `logret_resid_*`, etc.)
- All volume features (`dvolume_*`, `trade_sz_*`, etc.)
- All microstructure features (`relative_spread_*`, `ba_imbal_*`, `relative_updates_*`, etc.)
- All cross-sectional features (`*_cz`, `cx.*`)
- Market risk features (`beta_*`, `risk_*`)
- Technical indicators (`rsi_*`)
- Time features (`hour_of_day`, `day_of_week`)
- Trading filters (`fittable`, `tradeable`, `expandable`)

### Model-Specific Predictions
For each model and lag combination:
- `{model_name}_{horizon}_L{lag}`: Raw model prediction
- `{model_name}_{horizon}_L{lag}_coeff`: Model coefficient
- `{model_name}_{horizon}_L{lag}_condition`: Condition flag (rev/mom)
- `{model_name}_{horizon}_L{lag}_err`: Prediction error
- `{model_name}_{horizon}_L{lag}_weight`: Model weight

### Classification (if applicable)
- `class`: Classification result (-1, 0, 1)

## Important Notes
1. **Model files MUST contain all features used by downstream SVM models**
2. The `relative_updates_{horizon}_lz` feature is particularly important for SVM classification
3. Missing features will cause the server integration test to fail
4. Model files are essentially feature files plus model predictions

## Additional Notes
- The test expects 21 total files: 9 bar files + 9 feature files + 3 universe files
- Master fixtures are generated only for the test date (20250108)
- Model generation uses ModelCalcs which requires both bar and feature data
- Generated model files are moved from either "models" or "tardis_models" directory to master/

## Usage Notes

1. **Running Tests**: The integration test compares generated models in master/ against expected values
2. **Production Data**: The fixture uses real production data copied from ~/stat_arb/data/
3. **Feature Completeness**: Model files must contain ALL features, not just predictions, for downstream SVM compatibility