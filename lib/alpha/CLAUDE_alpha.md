# lib/alpha/ - Alpha Signal Generation and Feature Engineering

**Purpose:** Alpha signal generation and feature engineering pipeline. Transforms raw bar data into predictive features and combines model outputs into tradeable alpha signals.

## Key Files

### features.py (1000+ lines)
Feature engineering pipeline

**Key Components:**
- Implements 50+ features across categories: returns, volume, microstructure, risk, news
- Handles parallel processing across time horizons [15, 60, 120, 360, 720, 1440, 4320, 10080, 43200]
- Manages automatic lookback periods and chunked processing for large date ranges
- Key class: `Features` - Main pipeline for feature calculation and persistence
- Key function: `generate_rolling_features()` - Parallel multi-horizon processing

### forecasts.py (800+ lines)
Alpha signal combination and model ensemble

**Key Components:**
- Combines individual model alphas into final portfolio alphas
- Implements `calculate_horizon_alphas()` - Weighted combination of model signals
- Handles alpha scaling, clipping, centering, and normalization
- Supports conditional alphas (momentum vs. mean-reversion regimes)
- Key class: `Forecasts` - Loads fitted models and generates predictions

### models.py (600+ lines)
Alpha model implementations

**Key Components:**
- Implements 9 alpha models: hl, c2vwap, slz, vadj, ba, badj, oi, rsi, ip
- Each model captures different market inefficiencies (mean reversion, momentum, microstructure)
- Key class: `Models` - Container for all alpha model calculations

### model_calcs.py
Lagged feature generation and model utilities

**Key Components:**
- Creates lagged versions of features for predictive models
- Handles multi-lag feature construction

## Key Functionality

- **Feature Engineering:** Raw bars → 50+ engineered features per horizon
- **Alpha Generation:** Features + fitted models → alpha signals
- **Ensemble Methods:** Multiple models → combined weighted alpha
- **Production Support:** Incremental updates for real-time trading
