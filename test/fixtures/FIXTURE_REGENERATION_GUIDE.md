# Fixture Regeneration Guide

## Overview
This guide explains how to regenerate test fixtures and which files should NOT be deleted during regeneration.

## Regeneration Commands
Each fixture type can be regenerated using:
```bash
./bin/regenerate_master_fixture.sh -n <fixture_name>
```

Where `<fixture_name>` is one of:
- `bars` - Bar generation fixtures
- `features` - Feature engineering fixtures  
- `models` - Model generation fixtures
- `fits` - Model fitting/SVM fixtures
- `alphas` - Alpha signal fixtures
- `forwards` - Forward return fixtures
- `server` - Server/target generation fixtures
- `sim` - Simulation fixtures
- `pnl` - PnL calculation fixtures

## Directory Structure
All fixtures follow a consistent pattern:
- **data/** - Input test data (preserved during regeneration)
- **master/** - Output files (deleted and regenerated)
- **config_*.json** - Test configuration files (preserved)

## Important: What Gets Regenerated
- **Standard fixtures**: Only files in `master/` directories are regenerated
- **Models fixture**: SPECIAL CASE - regenerates both `data/` and `master/` directories

## Preserved Files
The following are NEVER deleted during standard regeneration:
- All files in **data/** directories (except for models fixture)
- Configuration files (**config_*.json**)
- Python cache directories (**__pycache__/**)
- Any files outside the master/ directory

## Model Fixture Exception
The models fixture (`./bin/regenerate_master_fixture.sh -n models`) is unique:
- It regenerates BOTH input data (data/) and output (master/)
- This is because it generates synthetic test data from scratch
- All other fixtures preserve their input data

## Regeneration Process
When you run the regeneration script:
1. Files in **master/** directories are deleted
2. The test's `test_generate_master_fixture` function is executed
3. New files are generated in **master/** directories
4. Input files in **data/** directories remain untouched

## Integration Test Behavior
All integration tests use `delete_all_files_in_tree()` on their output directories:
- Deletes all files in the `master/` directory tree
- Preserves directory structure
- The `data/` directories are at a sibling level and are never touched
- Exception: Models test also clears and regenerates its `data/` directory

## Troubleshooting

### Missing Input Files
If regeneration fails due to missing input files:
1. Check the fixture's FIXTURE_DOCUMENTATION.md for required input files
2. Verify all files marked as "NOT REGENERATED" exist in data/
3. Some files may need to be manually created or copied from production

### Server Test Specific Issues
The server test is particularly sensitive to:
1. **Model files must contain ALL feature columns** - not just predictions
2. **SVM models expect specific features** - especially `relative_updates_{horizon}_lz`
3. **prod_fits must match the model files** - feature names must align

## Best Practices
1. Always check FIXTURE_DOCUMENTATION.md before regenerating
2. Back up any manually created files before regeneration
3. Verify input data exists before running regeneration
4. Check git status after regeneration to ensure only expected files changed