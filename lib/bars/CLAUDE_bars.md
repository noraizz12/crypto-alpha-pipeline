# lib/bars/ - Bar Data Aggregation and Resampling

**Purpose:** OHLCV bar aggregation and resampling from raw tick data. Converts 1-minute prebars into multi-horizon bars with comprehensive market metrics.

## Key Files

### bar_generator.py (360 lines)
Multi-horizon bar aggregation

**Key Components:**
- Aggregates 1-minute prebars into all configured horizons
- Calculates rolling statistics (sum, mean, min, max, std) for 30+ fields
- Computes VWAP, market-adjusted returns, funding-adjusted returns
- Key class: `BarGenerator` - Orchestrates full bar generation pipeline
- Defines `BAR_FLDS_AGG_DICT` - Aggregation rules for each field

### bar_resampler.py (400+ lines)
1-minute bar loading and merging

**Key Components:**
- Loads prebars from multiple sources (TARDIS, live)
- Merges funding rate data with price/volume data
- Handles S3 integration and caching for performance
- Key class: `BarResampler` - Loads and consolidates 1-min bars

### tardis.py (800+ lines)
Raw tick data download and processing

**Key Components:**
- Downloads trades, order book snapshots, derivative tickers from Tardis API
- Aggregates ticks into 1-minute prebars
- Generates funding rate and open interest data
- Key class: `Tardis` - Handles Tardis data pipeline

### live_bars_converter.py
Real-time bar generation for live trading

## Key Functionality

- **Data Acquisition:** TARDIS API → Raw tick data (trades, book snapshots)
- **Prebar Generation:** Raw ticks → 1-minute OHLCV prebars (22 columns)
- **Multi-Horizon Aggregation:** 1-min prebars → [15, 60, 120, ...] minute bars (39 columns)
- **Market Returns:** Calculate equal-weighted and volume-weighted market returns
- **Data Quality:** Enforce 1440 rows/day, handle missing data, forward fill gaps
