# lib/sim/ - Backtesting Simulation Engine

**Purpose:** Backtesting simulation engine for strategy validation. Models realistic trading conditions including slippage, fees, and market constraints.

## Key Files

### simulate.py (1500+ lines)
Main simulation engine

**Key Components:**
- Key class: `Simulate` - Backtesting orchestrator
- Key method: `simulate()` - Run backtest for date range
- Simulates:
  - Portfolio optimization at reoptimization intervals
  - Order execution with slippage models
  - Exchange fees and funding payments
  - Position tracking and PnL calculation
  - Volume constraints and liquidity limits
- Supports VWAP or open price execution
- Generates detailed execution statistics

### sim_util.py
Simulation utilities

**Key Functions:**
- `get_return_metrics_str()` - Performance metrics formatting
- Sharpe ratio, drawdown, and return statistics

### factor_analysis.py
Factor exposure analysis

**Key Components:**
- Analyzes strategy exposures to market factors
- Factor attribution for returns

## Key Functionality

- **Historical Replay:** Simulates trading on historical data minute-by-minute
- **Portfolio Optimization:** Same optimizer as production (CVXPY)
- **Execution Modeling:** Slippage = f(volume participation, volatility)
- **PnL Calculation:** Tracks positions, cash, fees, funding, realized/unrealized PnL
- **Performance Metrics:** Sharpe, max drawdown, turnover, win rate
- **Grid Search:** Parameter optimization across configurations

## Simulation Output

**Saved to:** `$ROOT_DIR/sims/{sim_name}/`

**Files:**
- `sim.{date}.parquet` - Daily simulation results with positions and PnL
- `pnl.{date}.csv` - Detailed PnL breakdown by symbol and model
- `trades.{date}.csv` - All simulated trades
- `config.json` - Configuration snapshot
