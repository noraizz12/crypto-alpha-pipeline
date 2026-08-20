# System Architecture

This document provides detailed technical documentation of the system architecture.

## Data Pipeline Overview

The system processes market data through a series of transformation stages:

```
Raw Tick Data → Prebars → Bars → Features → Forward Returns → Models → Alphas → Targets
```

Each stage is idempotent and can be re-run independently.

## Module Structure

### lib/bars/ - Bar Aggregation

**Purpose**: Transform raw tick-level data into OHLCV bars at multiple timeframes.

**Key Components**:
- `bar_generator.py` - Main orchestration for bar generation
- `bar_resampler.py` - Time-based resampling logic
- `tardis.py` - Tardis exchange data integration

**Data Flow**:
```
Tardis/Binance Raw Data (parquet)
    ↓
PreBar Generation (1-minute aggregation)
    ↓
Bar Resampling (15m, 60m, 360m, 720m, 1440m, etc.)
    ↓
Rolling Aggregations (sums, means, stds across horizons)
    ↓
Output Parquet Files (data/bars/)
```

**Key Fields Generated**:
- OHLCV: open, high, low, close, volume
- Microstructure: spread_avg, trade_cnt, update_cnt
- Volume: dvolume (dollar volume), bid_trade_dollars, ask_trade_dollars
- Derived: logret, twap, vwap

### lib/calcs/ - Feature Calculations

**Purpose**: Transform bar data into predictive features.

**Key Modules**:
- `calc_returns.py` - Log returns and residualized returns
- `calc_volatility.py` - Volatility metrics
- `calc_volume.py` - Volume-based features
- `calc_microstructure.py` - Bid-ask spread and order flow
- `calc_momentum.py` - Momentum and mean-reversion signals

**Feature Categories**:

| Category | Examples | Description |
|----------|----------|-------------|
| Returns | `logret_HORIZON_lz` | Z-scored log returns |
| Volume | `dvolume_HORIZON_trmean` | Trimmed mean dollar volume |
| Spread | `relative_spread_HORIZON` | Bid-ask spread normalized |
| Momentum | `rsi_HORIZON` | RSI indicator |
| Flow | `ba_imbal_HORIZON` | Bid-ask imbalance |

### lib/external/ - Exchange Integration

**Purpose**: Interface with external data sources and exchanges.

**Key Components**:

**binance_utils.py** - Binance REST API wrapper:
- HMAC-SHA256 request signing
- Position and balance queries
- Funding rate data
- Trade history

**Key API Endpoints**:
```python
PAPI_URL = "papi.binance.com"   # Portfolio Margin
FAPI_URL = "fapi.binance.com"   # Futures
SAPI_URL = "api.binance.com"    # Spot
```

### lib/trader/ - Order Management

**Purpose**: Real-time order execution and position management.

**Key Components**:

**binance_oms.py** - WebSocket OMS:
- User data stream connection
- Listen key management
- Order/fill event processing
- ZMQ event publishing

**Connection Flow**:
```
Create Listen Key (REST POST /papi/v1/listenKey)
    ↓
Connect WebSocket (wss://fstream.binance.com/pm/ws/{listenKey})
    ↓
Receive Events (ORDER_TRADE_UPDATE, ACCOUNT_UPDATE, etc.)
    ↓
Publish via ZMQ (tcp://*:5555)
    ↓
Refresh Listen Key (every 30 min)
```

### lib/fits/ - Model Training

**Purpose**: Train predictive models on historical data.

**Approach**:
- Random Forest classifiers per horizon
- Features selected per timeframe
- Out-of-bag scoring for validation
- Model persistence via pickle

**Training Pipeline**:
```
Load Features + Forward Returns
    ↓
Feature Selection (per horizon)
    ↓
Train RF Classifier
    ↓
Compute OOB Score
    ↓
Save Model + Fit Statistics
```

### lib/alpha/ - Signal Generation

**Purpose**: Combine model predictions into tradeable signals.

**Signal Combination**:
- Multiple models per horizon
- Weighted ensemble averaging
- Risk-adjusted scaling
- Alpha decay tracking

### lib/util/ - Shared Utilities

**Key Modules**:
- `aws.py` - S3 and Secrets Manager
- `clickhouse.py` - Database operations
- `config.py` - Configuration management
- `dataframes.py` - DataFrame utilities
- `time_util.py` - Time handling
- `slack.py` - Notifications
- `opsgenie.py` - Alerting

## Data Storage

### File Formats

**Parquet Files**:
- Primary storage format
- Columnar, compressed
- Native pandas support

**Naming Convention**:
```
{datatype}_{date}_{horizon}.parquet
Example: bars_20240115_60.parquet
```

### Directory Structure

```
data/
├── prebars/           # Raw tick aggregations
│   └── tardis/        # Tardis source data
├── bars/              # OHLCV bars
├── features/          # Calculated features
├── forwards/          # Forward returns
├── models/            # Trained models
├── fits/              # Fit statistics
│   ├── dev/           # Development fits
│   └── prod/          # Production fits
├── alpha/             # Alpha signals
│   ├── dev/
│   └── prod/
└── universe/          # Trading universe

trading/
├── positions/         # Position snapshots
├── fills/             # Trade fills
├── orders/            # Order records
├── targets/           # Position targets
└── balances/          # Account balances
```

## Configuration

Configuration is centralized in `config/config.json`:

**Trading Parameters**:
```json
{
  "MAX_PORTFOLIO_NOTIONAL": 100000,
  "MAX_POSITION_PCT": 0.10,
  "EXCHANGE_FEES": 0.0001
}
```

**Model Parameters**:
```json
{
  "RF_N_ESTIMATORS": 100,
  "RF_MAX_DEPTH": 8,
  "RF_MIN_SAMPLES_SPLIT": 500
}
```

**Feature Configuration**:
```json
{
  "FEATURES": {
    "1440": {
      "prod": ["logret_HORIZON_lz", "dvolume_HORIZON_lz"]
    }
  }
}
```

## Error Handling

### Reconnection Strategy

WebSocket connections use exponential backoff:

```python
RECONNECT_DELAYS = [1, 2, 5, 10, 15]  # seconds
```

### Data Validation

- Timestamp bounds checking
- NaN/Inf handling
- Volume sanity checks
- Price movement filters

## Monitoring

### Logging

Structured logging with `KeyLogger`:
```python
logger.info(f"Processing {symbol}", key=symbol)
```

### Alerting

OpsGenie integration for critical alerts:
- High/Medium/Low/Critical priorities
- Automatic alert deduplication
- On-call schedule integration

## Testing Strategy

### Unit Tests

- Pure function testing
- Mock external dependencies
- Fast execution (<1s per test)

### Integration Tests

- Full pipeline validation
- Fixture-based comparison
- CSV diff for data validation

```bash
./bin/run_integration_test.sh -n bars
```

## Deployment Considerations

### Environment Variables

Required:
- `ROOT_DIR` - Project root directory
- `BINANCE_API_KEY` - API authentication
- `BINANCE_SECRET` - API secret

Optional:
- `CLICKHOUSE_HOST` - Database host
- `TARDIS_API_KEY` - Historical data
- `SLACK_WEBHOOK` - Notifications

### Process Management

- Kill files for graceful shutdown
- Lock files to prevent multiple instances
- Cron integration for scheduled jobs
