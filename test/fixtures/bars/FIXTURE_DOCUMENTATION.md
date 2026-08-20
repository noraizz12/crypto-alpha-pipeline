# Bars Fixture Documentation

## Purpose
The bars fixtures are used to test the bar generation pipeline that aggregates raw trade data into OHLCV bars at various time horizons.

## Fixture Generation
Run `python generate_fixtures.py` to generate test fixtures from production data.
- Use `--master-only` flag to only regenerate master fixtures without copying data

## Directory Structure
- **data/**: Contains input test data (copied from production)
  - **bars/{frequency}/binance-futures/{date}/**: Bar files organized by frequency and date
  - **prebars/**: Pre-bar data for Live bars
  - **secdata/**: Security data including binance_meta
  - **binance_meta/**: Metadata files
- **master/**: Contains expected output files
  - **{frequency}/binance-futures/{date}/**: Generated bar files
  - **bars_{frequency}_{date}.parquet**: Aggregated bar data

## Test Configuration
- Config file: `config_bar_generation_test.json`
- Test symbols: BTC, ETH, BNB (minimal set of 3)
- Test date: 20250609
- Test frequencies: [1, 15] minutes
- Bar type: BARS_TYPE_NEW (using bars directory structure)

## Required Columns in Bar Files
Based on the bar documentation in CLAUDE.md, bar files should contain:

### Index Fields
- `ts`: Timestamp (datetime index) - Bar start time
- `symbol_venue`: Symbol and venue combination (e.g., "BTCUSDT_binance-futures")

### Core Price Fields  
- `close_mid`: Closing mid price (bid-ask midpoint)
- `close_trade`: Closing trade price
- `close_wgt_mid`: Closing weighted mid price
- `high_mid`: High mid price during the bar
- `low_mid`: Low mid price during the bar
- `high_trade`: High trade price during the bar
- `low_trade`: Low trade price during the bar

### Volume & Trading Activity
- `volume`: Total trading volume (base currency)
- `dvolume`: Dollar volume (price × volume)
- `trade_cnt`: Number of trades
- `update_cnt`: Number of order book updates
- `bid_trade_dollars`: USD volume of bid-side trades
- `ask_trade_dollars`: USD volume of ask-side trades

### Market Microstructure
- `bid_sz_avg`: Average bid size
- `ask_sz_avg`: Average ask size  
- `spread_avg`: Average bid-ask spread
- `book_latency`: Average order book update latency
- `trade_latency`: Average trade execution latency

### Returns & Performance
- `logret`: Log returns
- `logret_resid_eqmkt`: Market-residualized log returns (equal-weighted)
- `logret_resid_wgtmkt`: Market-residualized log returns (value-weighted)
- `logret_funding_adj`: Funding rate adjusted log returns
- `logret_funding_adj_resid_eqmkt`: Funding-adjusted + equal-weighted market-residualized
- `logret_funding_adj_resid_wgtmkt`: Funding-adjusted + value-weighted market-residualized

### Derivatives-Specific Fields
- `index_price`: Underlying spot index price
- `index_price_dvolume`: Index price × volume
- `index_mid_price_diff`: Basis (futures_price - index_price)
- `estimated_settle_price`: Exchange's estimated settlement price
- `last_funding_rate`: Most recent funding rate
- `next_funding_time`: Timestamp of next funding settlement
- `open_interest`: Total open interest

### Calculated Metrics
- `vwap`: Volume-weighted average price
- `twap`: Time-weighted average price
- `advp`: Average daily volume participation
- `fittable`: Boolean flag for model fitting suitability

### Rolling Horizon Fields
For each aggregatable field, rolling calculations over horizons [15, 60, 120, 360, 720, 1440, 4320, 10080, 43200]:
- `{field}_{horizon}`: Rolling aggregations (sum, mean, std, min, max as appropriate)
- `vwap_{horizon}`: Rolling VWAP
- `index_price_vwap_{horizon}`: Index-price VWAP

## Additional Notes
- The test uses a 13-day date range (20250528 to 20250609) to satisfy bar generator lookback requirements
- Master fixtures are generated only for the test date (20250609) 
- The script verifies that all 52 expected files are generated (39 prebar files + 13 meta files)