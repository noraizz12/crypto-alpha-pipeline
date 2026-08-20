# Crypto Market Data Engineering System

A production-grade market data engineering system for cryptocurrency futures, demonstrating real-time data ingestion, normalisation, feature engineering, and signal generation for Binance perpetual futures.

## Overview

This system implements a complete data pipeline for crypto derivatives trading:

1. **Data Ingestion** — Real-time and historical market data from Binance Futures and Tardis
2. **Bar Aggregation** — OHLCV bars at multiple timeframes (1m to daily)
3. **Feature Engineering** — 50+ technical and microstructure features
4. **Alpha Generation** — Multi-horizon predictive signals using machine learning
5. **Execution Interface** — WebSocket-based order management with Binance API

## Architecture

```
                              ┌─────────────────────────────────────────────────────┐
                              │                  Data Sources                        │
                              │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
                              │  │   Tardis    │  │   Binance   │  │  CoinGecko  │  │
                              │  │  (History)  │  │  (REST/WS)  │  │   (Spot)    │  │
                              │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
                              └─────────┼─────────────────┼─────────────────┼────────┘
                                        │                 │                 │
                                        ▼                 ▼                 ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                              INGESTION LAYER                                           │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐         │
│  │  lib/bars/tardis.py  │  │  lib/external/       │  │  download_*.py       │         │
│  │  - Tick data parsing │  │  binance_utils.py    │  │  - Scheduled pulls   │         │
│  │  - Prebar generation │  │  - REST API client   │  │  - Rate limiting     │         │
│  │  - Gap detection     │  │  - HMAC signing      │  │  - Retry logic       │         │
│  └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘         │
└─────────────┼──────────────────────────┼──────────────────────────┼────────────────────┘
              │                          │                          │
              ▼                          ▼                          ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                            PROCESSING LAYER                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────┐     │
│  │                        Bar Generation (lib/bars/)                             │     │
│  │  - OHLCV aggregation at 1m, 15m, 60m, 360m, 720m, 1440m                       │     │
│  │  - Volume-weighted prices (VWAP, TWAP)                                        │     │
│  │  - Microstructure metrics (spread, trade size, update frequency)              │     │
│  │  - Funding rate integration                                                   │     │
│  └──────────────────────────────────────────────────────────────────────────────┘     │
│                                        │                                               │
│                                        ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐     │
│  │                     Feature Engineering (lib/calcs/)                          │     │
│  │  - Log returns and residuals                                                  │     │
│  │  - Rolling statistics (mean, std, z-scores)                                   │     │
│  │  - Volume profiles and dollar volume                                          │     │
│  │  - Bid-ask imbalance and spread metrics                                       │     │
│  │  - Open interest changes                                                      │     │
│  │  - Time-of-day/day-of-week seasonality                                        │     │
│  └──────────────────────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                              STORAGE LAYER                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐            │
│  │       Parquet       │  │     ClickHouse      │  │         S3          │            │
│  │  - Local storage    │  │  - Time-series DB   │  │  - Archive storage  │            │
│  │  - Fast columnar    │  │  - Real-time query  │  │  - Backup/restore   │            │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘            │
└───────────────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                          SIGNAL GENERATION LAYER                                       │
│  ┌──────────────────────────────────────────────────────────────────────────────┐     │
│  │                      Model Training (lib/fits/)                               │     │
│  │  - Random Forest classifiers                                                  │     │
│  │  - Multi-horizon predictions (15m to 30d)                                     │     │
│  │  - Cross-validation and out-of-bag scoring                                    │     │
│  └──────────────────────────────────────────────────────────────────────────────┘     │
│                                        │                                               │
│                                        ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐     │
│  │                     Alpha Generation (lib/alpha/)                             │     │
│  │  - Combine predictions across horizons                                        │     │
│  │  - Risk-adjusted signal scaling                                               │     │
│  │  - Alpha decay and freshness tracking                                         │     │
│  └──────────────────────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION LAYER                                              │
│  ┌──────────────────────────────────────────────────────────────────────────────┐     │
│  │                    Order Management (lib/trader/)                             │     │
│  │  - WebSocket user data stream (binance_oms.py)                                │     │
│  │  - Order/fill event processing                                                │     │
│  │  - Position reconciliation                                                    │     │
│  │  - CVXPY portfolio optimization                                               │     │
│  └──────────────────────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

## Features

### Data Ingestion
- REST API integration with Binance Futures (HMAC-SHA256 authentication)
- WebSocket streaming for real-time user data (orders, fills, positions)
- Historical data download from Tardis exchange data service
- Rate limiting and automatic retry with exponential backoff
- Graceful reconnection for WebSocket disconnections

### Data Quality
- **Duplicate Detection**: Timestamp-based deduplication in bar aggregation
- **Gap Detection**: Missing bar identification across time series
- **Stale Data Detection**: Configurable staleness thresholds for live data
- **Timestamp Validation**: UTC normalisation and sanity checks
- **Value Validation**: Price/volume bounds checking, impossible trade detection

### Feature Engineering
- 50+ features across multiple timeframes
- Log returns with market-residualized variants
- Rolling statistics (trimmed mean, std, z-scores)
- Volume and dollar volume profiles
- Microstructure metrics (spread, trade size, order flow)
- Funding rate and open interest dynamics

### Storage
- Parquet files for efficient columnar storage
- ClickHouse integration for real-time time-series queries
- S3 support for cloud-based archival

## Running Locally

### Prerequisites
- Python 3.12+
- Binance Futures account with API access
- (Optional) ClickHouse for time-series storage
- (Optional) Tardis API key for historical data

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/crypto-market-data.git
cd crypto-market-data

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and configuration

# Set ROOT_DIR
export ROOT_DIR=$(pwd)
```

### Running the Data Pipeline

```bash
# Generate OHLCV bars from Tardis data
python generate_bars.py --from 20240101 --to 20240131 --debug

# Generate features from bars
python generate_features.py --from 20240101 --to 20240131 --debug

# Generate forward returns for model training
python generate_forwards.py --from 20240101 --to 20240131 --debug
```

### Running Tests

```bash
# Run all tests
pytest test/ -v

# Run specific test file
pytest test/test_bars.py -v

# Run with coverage
pytest test/ --cov=lib --cov-report=html
```

## Configuration

Configuration is managed via `config/config.json`. Key parameters:

| Parameter | Description |
|-----------|-------------|
| `SYMBOL_UNIVERSE` | List of trading pairs to process |
| `ADV_LOOKBACK_DAYS` | Days of history for volume calculations |
| `RF_N_ESTIMATORS` | Random Forest ensemble size |
| `RF_MAX_DEPTH` | Maximum tree depth |
| `EXCHANGE_FEES` | Fee rate for cost calculations |
| `FEATURES` | Feature lists per horizon |

See `config/config.example.json` for a complete example.

## Project Structure

```
├── lib/
│   ├── bars/           # Bar aggregation from raw ticks
│   ├── calcs/          # Feature calculations
│   ├── data/           # Data loading utilities
│   ├── external/       # Exchange API clients
│   ├── fits/           # Model training
│   ├── alpha/          # Signal generation
│   ├── trader/         # Order management
│   ├── pnl/            # P&L calculation
│   ├── sim/            # Backtesting
│   └── util/           # Shared utilities
├── config/             # Configuration files
├── test/               # Test suite
├── bin/                # Shell scripts
└── docs/               # Documentation
```

## Example Output

Sample bar data generated by the pipeline:

```
                             symbol_venue     open     high      low    close   volume    dvolume  spread_avg
ts
2024-01-15 00:00:00+00:00  BTCUSDT_binance  42850.1  42875.0  42840.0  42860.5   125.32   5372841     0.00012
2024-01-15 00:01:00+00:00  BTCUSDT_binance  42860.5  42890.0  42855.0  42882.3    89.45   3838915     0.00011
2024-01-15 00:02:00+00:00  BTCUSDT_binance  42882.3  42900.0  42870.0  42895.0   156.78   6726654     0.00013
```

## Design Decisions

**Why Parquet for storage?**
Columnar format optimised for analytical queries, efficient compression, and fast reads for time-series data. Native pandas integration.

**Why ClickHouse?**
Purpose-built for time-series data with excellent compression and query performance. Handles billions of rows efficiently.

**Why CVXPY for portfolio optimization?**
Industry-standard convex optimization library with support for multiple solvers. Clean API for expressing trading constraints.

**Why WebSocket for order management?**
Lower latency than polling REST endpoints. Binance user data stream provides real-time order/fill updates.

## Limitations

This is a demonstration project and has the following limitations:

- **Not production-ready**: Simplified error handling in some areas
- **No backtesting validation**: Model performance not verified on out-of-sample data
- **Single exchange**: Only supports Binance Futures
- **No HFT capability**: Designed for minute-level, not microsecond-level trading
- **Simplified risk management**: Basic position limits only

## Possible Improvements

- [ ] Add support for additional exchanges (OKX, Bybit, dYdX)
- [ ] Implement Kafka/Redpanda for event streaming
- [ ] Add sequence number gap detection for WebSocket messages
- [ ] Implement proper schema versioning for data formats
- [ ] Add latency metrics and monitoring dashboard
- [ ] Implement proper order book reconstruction from L2 data
- [ ] Add Grafana dashboards for system observability
- [ ] Implement data replay/backfill for development

## Testing

The project includes comprehensive tests:

- **Unit tests** for data processing functions
- **Integration tests** for the full pipeline
- **Fixture-based testing** with csvdiff comparison

```bash
# Run unit tests only
./bin/run_tests.sh -u

# Run integration tests
./bin/run_tests.sh -i

# Run specific integration test
./bin/run_integration_test.sh -n bars
```

## License

MIT License - see LICENSE file for details.

---

*This project demonstrates crypto market data engineering concepts for educational purposes.*
