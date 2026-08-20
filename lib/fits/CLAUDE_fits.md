# lib/fits/ - Model Training and Fitting Pipeline

**Purpose:** Model training and fitting pipeline. Implements the two-stage process: regime classification (momentum vs. mean-reversion) followed by conditional linear regressions.

## Key Files

### fits.py (800+ lines)
Main fitting orchestrator

**Key Components:**
- Key class: `Fits` - Manages entire fitting pipeline
- Handles rolling window generation for time-series cross-validation
- Supports both development and production fitting schedules
- Manages parallel fitting across horizons and models
- Key method: `run_fitting()` - Execute fitting for date range

### model_horizon.py (1000+ lines)
Single model/horizon fitting

**Key Components:**
- Key class: `ModelHorizonFit` - Fits one model at one horizon
- Implements two-stage approach:
  1. Random Forest classifier for regime detection (momentum/reversal)
  2. Regularized linear regression per regime
- Handles feature selection with L1 regularization
- Computes HAC-robust standard errors
- Saves fitted coefficients and classifiers

### forwards.py (400+ lines)
Forward return calculation

**Key Components:**
- Generates target variables (y) for model training
- Calculates returns at multiple horizons (horizon 1 and horizon 2)
- Supports raw, market-residualized, and funding-adjusted return types
- Key class: `Forwards` - Computes forward-looking returns

### fit_util.py
Fitting utilities

**Key Functions:**
- `make_classification_bar_features()` - Feature engineering for classifiers
- Feature importance extraction from fitted models

## Key Functionality

- **Regime Classification:** Random Forest (150 trees) → Momentum/Reversal labels
- **Conditional Regression:** Separate linear models per regime
- **Feature Selection:** L1 regularization, t-statistic filtering (t > 1.0)
- **Robust Inference:** HAC standard errors for time-series data
- **Rolling Windows:** Time-series cross-validation for model validation
- **Production Deployment:** Refit schedules, model persistence

## Fitting Process

1. Load bars, features, and forward returns for training period
2. Train Random Forest classifier on lagged features → regime labels
3. Fit regularized regression separately for momentum and reversal regimes
4. Save coefficients (CSV) and classifier (pickle) to fits/ and models/ directories
5. Schedule refits based on `refitting_days` in config
