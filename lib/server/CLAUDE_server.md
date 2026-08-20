# lib/server/ - Real-Time Alpha Server

**Purpose:** Real-time alpha signal generation server for live trading. Runs continuously to compute alphas, optimize portfolio, and publish target positions.

## Key Files

### server.py (1500+ lines)
Main alpha generation server

**Key Components:**
- Key class: `AlphaServer` - Real-time signal generation and optimization
- Key method: `loop()` - Continuous operation loop
- Responsibilities:
  - Load real-time bars, features, and fitted models
  - Calculate multi-horizon alpha signals
  - Run portfolio optimization with risk constraints
  - Track position age for risk management
  - Publish target position files for trader
  - Send Slack notifications for monitoring
- Operates on two timescales:
  - Major reoptimizations (every 2 hours) - Full portfolio optimization
  - Minor updates (every minute) - Short-term alpha updates only

### server_calcs.py (500+ lines)
Server-specific calculations

**Key Components:**
- Key class: `ServerCalcs` - Calculations optimized for real-time use
- Implements incremental feature updates
- Handles sparse data for production efficiency
- Manages position age tracking

### server_horizon_calcs.py (400+ lines)
Per-horizon calculations

**Key Components:**
- Horizon-specific alpha and risk calculations
- Supports different calculation modes per horizon

## Key Functionality

- **Data Ingestion:** Real-time bars → features → models (every minute)
- **Alpha Calculation:** Multi-horizon signals (15min to monthly)
- **Portfolio Optimization:** CVXPY solver → optimal positions
- **Target Publishing:** Writes targets/{date}/targets.{timestamp}.csv
- **Monitoring:** Slack alerts, position tracking, risk monitoring
- **Dynamic Universe:** Liquidity filtering, delisting detection

## Execution Flow

1. Wait for next reoptimization time (e.g., 00:00, 02:00, 04:00, ...)
2. Load latest bars and features (last 2-3 days of data)
3. Load fitted models and classifiers
4. Calculate alpha signals for all horizons
5. Load current positions from Binance
6. Run portfolio optimizer (CVXPY) with risk constraints
7. Write target positions to targets/ directory
8. Repeat every reoptimization interval
