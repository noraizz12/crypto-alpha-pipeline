# lib/calcs/ - Feature Calculations and Technical Indicators

**Purpose:** Comprehensive feature calculation library implementing technical indicators, market microstructure metrics, risk measures, and cross-sectional statistics.

## Key Files

### calcs.py (1500+ lines)
Main calculation orchestrator

**Key Components:**
- Key class: `Calcs` - Central hub for all feature calculations
- Integrates all calc_*.py modules
- Handles both batch (historical) and incremental (live) calculations
- Manages lookback periods and horizon-specific logic

### calc_returns.py
Return calculations

**Key Functions:**
- `calc_logret()` - Log returns
- `calc_vwap()` - Volume-weighted average price
- `calculate_resid_return()` - Market-residualized returns (equal-weighted, volume-weighted)
- `calculate_rsi()` - Relative Strength Index
- `calculate_funding_adjusted_logret()` - Returns adjusted for funding payments

### calc_risk.py
Risk and volatility metrics

**Key Functions:**
- `calculate_beta()` - Rolling beta to market (90-period)
- `calculate_risk()` - Volatility-based risk metric scaled by √horizon
- `calculate_trailing_semi_variance()` - Upside/downside semi-deviation
- `calculate_pca()` - Principal component analysis for risk factors

### calc_volume.py
Volume-based features

**Key Functions:**
- `calculate_advp()` - Average daily volume participation
- `calculate_volume_buckets()` - Time-of-day volume patterns
- `calculate_volume_forecast()` - Expected volume prediction

### calc_ob.py
Order book and microstructure

**Key Functions:**
- `calculate_bid_ask_imbalance()` - Log(bid_size / ask_size)
- `calculate_spread_factor()` - Relative spread (spread / mid_price)
- `calculate_trade_size()` - Average trade size metrics
- `calculate_update_size()` - Order book update frequency

### calc_filters_and_bounds.py
Universe filtering and position limits

**Key Functions:**
- `calculate_fittable_filter()` - Liquidity threshold for model fitting (ADVP > $30M)
- `calculate_tradeable_filter()` - Liquidity threshold for trading (ADVP > $20M)
- `calculate_expandable_filter()` - Liquidity threshold for position expansion (ADVP > $30M)
- `calculate_exclusions_and_bounds()` - Position bounds from risk constraints

### calc_util.py
Utility calculations

**Key Functions:**
- `calculate_trailing_z()` - Longitudinal z-scores
- `calculate_trailing_mean()` - Rolling mean calculations
- `calculate_minmax()` - Min/max over lookback periods
- `calculate_quintiles()` - Cross-sectional quintile ranks

## Key Functionality

- **Returns:** Raw returns → Market-adjusted returns → Funding-adjusted returns
- **Technical Indicators:** RSI, moving averages, min/max, semi-variance
- **Microstructure:** Spreads, imbalances, trade sizes, order flow
- **Risk:** Beta, volatility, PCA factors
- **Filters:** Liquidity-based universe selection
- **Cross-Sectional:** Z-scores, rankings, market-relative metrics
