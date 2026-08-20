# lib/pnl/ - PnL Calculation and Attribution

**Purpose:** PnL calculation, attribution, and monitoring. Reconciles trading activity with exchange positions and balances.

## Key Files

### fill_pnl.py (800+ lines)
Main PnL calculator

**Key Components:**
- Key class: `FillPnl` - Calculates PnL from fill records
- Methods:
  - `calculate_pnl()` - Total PnL attribution
  - `calculate_realized_pnl()` - Realized gains/losses from fills
  - `calculate_unrealized_pnl()` - Mark-to-market on open positions
- Tracks fees, funding payments, and realized vs. unrealized PnL

### fill_pnl_breakdown.py (600+ lines)
Detailed PnL attribution

**Key Components:**
- Key class: `FillBreakdown` - Attributes PnL to models, horizons, symbols
- Breaks down by:
  - Model (hl, c2vwap, slz, etc.)
  - Horizon (15, 60, 1440, etc.)
  - Symbol (BTCUSDT, ETHUSDT, etc.)
  - Time period (daily, weekly, monthly)

### fill_pnl_symbol.py
Per-symbol PnL calculations

### pnl_monitor.py
Real-time PnL monitoring

**Key Components:**
- Tracks live PnL during trading
- Sends alerts if PnL hits thresholds

### pnl_util.py
PnL calculation utilities

## Key Functionality

- **Fill-Based PnL:** Uses fills/ data as source of truth
- **Position Reconciliation:** Matches against positions/ snapshots
- **Attribution:** PnL breakdown by model, horizon, symbol, timeframe
- **Funding Tracking:** Perpetual futures funding payments
- **Fee Accounting:** Exchange fees in multiple currencies (BNB, USDT)
- **Monitoring:** Real-time PnL alerts and dashboards
