# Features Fixture Documentation

## Purpose
The features fixtures test the feature engineering pipeline that calculates trading signals and technical indicators from bar data.

## Fixture Generation
Run `python generate_fixtures.py` to generate test fixtures from production data.
- Use `--master-only` flag to only regenerate master fixtures without copying data

## Directory Structure
- **data/**: Contains input test data (copied from production)
  - **bars/{frequency}/binance-futures/{date}/**: Input bar files
  - **delisting.txt**: List of delisted symbols to exclude
- **master/**: Contains expected output files
  - **{horizon}/{feature_type}/{date}/**: Generated feature files organized by horizon, feature type, and date
    - Example: `1440/logret_1440_lz/20250609/features.1440.logret_1440_lz.20250609.BTCUSDT.parquet`

## Test Configuration
- Config file: `config_features_generation_test.json`
- Test symbols: ADA, BNB, BTC, ETC, ETH, ZIL (6 symbols from the file list)
- Test date: 20250609
- Test horizon: 1440 minutes (daily)
- Bar frequencies: [1, 1440] minutes
- Bar type: BARS_TYPE_NEW
- Historical days: 3 (20250606 to 20250609) for feature lookback calculations

## Required Columns in Feature Files
Based on the feature documentation in CLAUDE.md, feature files should contain:

### Index Fields
- `ts`: Timestamp (datetime index)
- `symbol_venue`: Symbol and venue combination

### Return-Based Features
- `logret_{HORIZON}`: Log return over specified horizon
- `logret_funding_adj_{HORIZON}`: Funding-adjusted log return
- `logret_{HORIZON}_trmean`: Trailing mean of log returns
- `logret_{HORIZON}_trstd`: Trailing standard deviation
- `logret_{HORIZON}_lz`: Standardized return (z-score)
- `logret_{HORIZON}_min`: Minimum return over lookback
- `logret_{HORIZON}_max`: Maximum return over lookback
- `logret_{HORIZON}_min_cz`: Cross-sectional z-score of min return
- `logret_{HORIZON}_max_cz`: Cross-sectional z-score of max return
- `logret_{HORIZON}_abs`: Absolute value of log return
- `logret_{HORIZON}_lz_abs`: Absolute standardized return
- `logret_{HORIZON}_trstd_u`: Upside semi-deviation
- `logret_{HORIZON}_trstd_d`: Downside semi-deviation
- `logret_{HORIZON}_trstd_udratio`: Upside/downside variance ratio

### Market-Adjusted Returns
- `logret_resid_eqmkt_{HORIZON}`: Return minus equal-weighted market
- `logret_resid_eqmkt_{HORIZON}_trmean`: Trailing mean
- `logret_resid_eqmkt_{HORIZON}_trstd`: Trailing std
- `logret_resid_eqmkt_{HORIZON}_lz`: Standardized residual
- `logret_resid_wgtmkt_{HORIZON}`: Return minus volume-weighted market
- `logret_resid_wgtmkt_{HORIZON}_trmean`: Trailing mean
- `logret_resid_wgtmkt_{HORIZON}_trstd`: Trailing std
- `logret_resid_wgtmkt_{HORIZON}_lz`: Standardized weighted residual

### Cross-Sectional Features
- `logret_{HORIZON}_cz`: Cross-sectional z-score of returns
- `logret_{HORIZON}_lz_cz`: Cross-sectional z-score of longitudinal z-scores
- `cx.logret_{HORIZON}`: Universe mean return z-score
- `cx.dvolume_{HORIZON}`: Universe mean dollar volume z-score

### Volume Features
- `dvolume_{HORIZON}`: Dollar volume over horizon
- `dvolume_{HORIZON}_trmean`: Trailing mean dollar volume
- `dvolume_{HORIZON}_trstd`: Trailing std dollar volume
- `dvolume_{HORIZON}_lz`: Standardized dollar volume
- `dvolume_{HORIZON}_d`: Absolute change in dollar volume
- `dvolume_{HORIZON}_dp`: Percentage change in dollar volume
- `dvolume_{HORIZON}_trmean_cz`: Cross-sectional z-score of mean volume
- `dvolume_{HORIZON}_lz_cz`: Cross-sectional z-score of standardized volume
- `median_time_bucket_dvolume_{HORIZON}`: Median volume for time-of-day

### Microstructure Features
- `relative_spread_{HORIZON}`: Bid-ask spread / mid price
- `relative_spread_{HORIZON}_trmean`: Trailing mean
- `relative_spread_{HORIZON}_trstd`: Trailing std
- `relative_spread_{HORIZON}_lz`: Standardized spread
- `ba_imbal_{HORIZON}`: Log(bid_size / ask_size)
- `ba_imbal_{HORIZON}_d`: Change in imbalance
- `ba_imbal_{HORIZON}_dp`: Percentage change
- `trade_sz_{HORIZON}`: Average trade size
- `trade_sz_{HORIZON}_trmean`: Trailing mean
- `trade_sz_{HORIZON}_trstd`: Trailing std
- `trade_sz_{HORIZON}_lz`: Standardized trade size
- `relative_updates_{HORIZON}`: Update count / volume ratio
- `relative_updates_{HORIZON}_trmean`: Trailing mean
- `relative_updates_{HORIZON}_lz`: Standardized updates

### Risk & Technical Features
- `beta_{HORIZON}`: Rolling beta to market
- `beta_t_{HORIZON}`: T-statistic of beta
- `risk_{HORIZON}`: Volatility-based risk metric
- `rsi_{HORIZON}`: Relative Strength Index
- `hour_of_day`: Hour in UTC (0-23)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)

### News & Market Features
- `news_event`: Binary news occurrence
- `news_event_decayed_{HORIZON}`: Exponentially decayed news impact
- `open_interest_{HORIZON}_d`: Change in open interest
- `open_interest_{HORIZON}_dp`: Percentage change

### Trading Filters
- `fittable`: Meets liquidity for model fitting
- `tradeable`: Meets liquidity for trading
- `expandable`: Meets liquidity for expansion
- `advp`: Average daily volume in USD
- `close_mid`: Mid price at bar close
- `vwap`: Volume-weighted average price

## Additional Notes
- The test expects 47 files total: 46 bar files + 1 delisting.txt file
- Master fixtures are generated only for the test date (20250609) but require historical bar data for feature calculations
- Features are generated with prod=False to generate all features, not just production features

## Usage Notes

1. **Running Tests**: The integration test compares generated features in master/ against expected values
2. **Updating Fixtures**: Only regenerate when feature calculation logic changes intentionally
3. **Debugging Failures**: Check differences between old and new master/ files to understand what changed