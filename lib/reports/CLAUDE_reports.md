# lib/reports/ - Trading Reports and Analytics

**Purpose:** Trading reports, performance dashboards, and analytics. Generates comprehensive analysis of strategy performance.

## Key Files

### trading_reports.py (1000+ lines)
Live trading performance reports

**Key Components:**
- Daily trading summary
- Position analysis and turnover
- Fill quality metrics
- Model performance attribution

### sim_trading_reports.py (800+ lines)
Simulation result analysis

**Key Components:**
- Backtest performance metrics
- Comparison across simulations
- Factor exposure analysis
- Turnover and cost analysis

### markouts.py (500+ lines)
Alpha signal analysis

**Key Components:**
- Measures alpha predictive power
- Forward return correlations
- Signal decay analysis
- Information coefficient calculation

### slippage_reports.py
Execution quality analysis

**Key Components:**
- Analyzes actual vs. expected slippage
- Order execution statistics
- Market impact measurement

### base_dash_app.py
Dash/Plotly dashboard infrastructure

### hist_trading_reports.py
Historical trading analysis

### prod_fits_reports.py
Model fitting diagnostics

## Key Functionality

- **Performance Metrics:** Sharpe, Sortino, max drawdown, return decomposition
- **Attribution:** PnL by model, horizon, symbol, timeframe
- **Execution Analysis:** Slippage, market impact, fill rates
- **Model Diagnostics:** Alpha decay, IC, t-statistics, feature importance
- **Dashboards:** Interactive Plotly visualizations
