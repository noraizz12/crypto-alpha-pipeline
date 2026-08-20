# Overnight 15-Minute Reversal Signal

## TL;DR

Coins with abnormally large moves in the last 15 minutes of the day (23:46-00:00 UTC) tend to reverse in the first 15 minutes of the next day (00:00-00:15 UTC). The effect is statistically robust (FM t=5.91) but **does not survive transaction costs as a standalone strategy**. Best used as a component signal in the multi-alpha framework.

**Signal**: `-rank_cs(logret_15 / logret_15_trstd)` at end of day

---

## 1. Signal Description

| Property | Value |
|----------|-------|
| Predictor | `-rank_cs(logret_15 / logret_15_trstd)` at 00:00 UTC |
| Target | Forward 15-min residual return (`y_resid_eqmkt1_15`) |
| Universe | All Binance perpetual futures (~80-200 symbols depending on date) |
| Sample | January-December 2025, 365 trading days |
| Fama-MacBeth t-stat | **5.91** |
| % days positive | **67%** |
| D10-D1 spread | **12.5 bps** in forward residual returns |

**Components**:
- `logret_15`: rolling 15-minute log return (backward-looking, no look-ahead)
- `logret_15_trstd`: trailing 90-period standard deviation of 15-min returns
- `rank_cs()`: cross-sectional rank scaled to [-0.5, +0.5]
- Negative sign: short big up-movers, long big down-movers

---

## 2. Research Progression

Nine experiments were conducted, each building on the last. The progression moved from raw returns to vol-adjusted returns to forward residual returns, testing conditioning variables and composite predictors.

### Experiment 1: Raw Returns

Pooled OLS of next-day first-15-min return on current-day last-15-min return.

| Metric | Value |
|--------|-------|
| Pooled beta | -0.083 (t = -8.1) |
| FM t-stat | -3.76 |
| % days negative beta | 63% |

All deciles showed negative next-open returns, indicating a general negative drift at the open that obscured the reversal signal.

### Experiment 2: Horizon Decay

| Horizon | FM t-stat |
|---------|-----------|
| 15-min | -3.76 |
| 60-min | -3.30 |

The reversal is short-lived and concentrated in the first 15 minutes. By 60 minutes, the signal is half as strong.

### Experiment 3: Volatility Adjustment

Normalizing by trailing volatility (`logret_15 / logret_15_trstd`) nearly doubled the FM t-stat.

| Predictor | FM t-stat |
|-----------|-----------|
| Raw | -3.76 |
| Vol-adjusted | **-6.99** |

### Experiment 4-6: Conditioning Variables

- **Volume**: High-volume EOD moves reverse most (FM t=-4.05 vs -3.01 for low volume)
- **Asymmetry**: Up moves reverse reliably (FM t=-3.61); down moves do not (FM t=-0.79)
- **Daily return**: Daily winners that rally further into the close show the strongest reversal (Q5: FM t=-4.91, D1-D5 spread = 24 bps)

### Experiment 7: Forward Residual Returns

Switching the dependent variable from vol-adjusted returns to `y_resid_eqmkt1_15` eliminated the negative drift artifact and confirmed true cross-sectional reversal.

| Decile | Mean EOD vadj | Mean Fwd Resid (bps) |
|--------|--------------|---------------------|
| 0 (big down) | -1.29 | +6.0 |
| 5 | +0.09 | +4.2 |
| 9 (big up) | +1.41 | **-6.6** |

### Experiment 8-9: Composite Predictors

Tested 11 predictor formulations. The simple **cross-sectional rank** beat all composite predictors:

| Rank | Predictor | FM t-stat |
|------|-----------|-----------|
| 1 | **-rank_cs(eod_vadj)** | **5.91** |
| 2 | -eod_vadj * (1 + daily_cs+ * 10) | 5.77 |
| 3 | -rank * (1+dvol+) * (1+daily+*10) | 5.65 |
| 6 | -eod_vadj (baseline z-score) | 5.26 |
| 9 | -max(eod_vadj, 0) (up-only) | 1.91 |

**Why rank wins**: eliminates outlier sensitivity, inherently demeaned and bounded, cleaner monotonic sort. Neither volume nor daily-return conditioning added significant independent explanatory power beyond the rank.

### Monthly Stability

| Month | FM t-stat | % Days Positive |
|-------|-----------|-----------------|
| Jan | +1.70 | 65% |
| Feb | +0.90 | 61% |
| Mar | +2.22 | 65% |
| Apr | +4.21 | 77% |
| May | +0.31 | 58% |
| Jun | +3.59 | 83% |
| Jul | +5.72 | 90% |
| Aug | +2.93 | 71% |
| Sep | +0.12 | 50% |
| Oct | +1.14 | 71% |
| Nov | +1.51 | 57% |
| Dec | +0.90 | 60% |

Positive in all 12 months. Strongest Apr-Aug. Weakest in May and Sep but never negative.

---

## 3. Backtest Results

A full backtest was run with rolling 90-day beta estimation, 15 entry waves across the last 15 minutes of each day, and exit at the average `close_mid` over minutes 15-30 of the next day.

**Script**: `adhoc/overnight_reversal_15m_backtest.py`

### Strategy Mechanics

- **Capital**: $1,000,000
- **Entry window**: Rows 1425-1439 of day D's bar file (23:46-00:00 UTC), one wave per minute
- **Exit window**: Average close_mid over rows 14-29 of day (D+1) bar file (00:15-00:30 UTC)
- **Wave capital**: $66,667 per wave (15 waves = $1M total)
- **Positions**: Dollar-neutral within each wave, weights proportional to predicted alpha
- **Coefficient estimation**: Pooled OLS over trailing 90 days (signal at midnight vs realized fwd resid); skip day if beta <= 0
- **Warmup**: 90 days (first trade ~April 1)

### Performance Summary

| Metric | Value |
|--------|-------|
| Period | 2025-04-01 to 2025-12-31 |
| Traded days | 274 |
| Skipped days | 0 |
| Total PnL (gross) | **$82,204** |
| Avg daily PnL | $300 |
| Std daily PnL | $1,011 |
| Sharpe ratio (ann) | **5.67** |
| Win rate | **66.8%** |
| Best day | $3,300 |
| Worst day | -$5,576 |
| Max drawdown | -$8,094 (Dec 14) |
| Avg daily notional | $983,184 |
| Avg trades/day | 1,256 |
| Avg trailing beta | 0.001115 |

### Monthly PnL

| Month | Total PnL | Avg Daily |
|-------|-----------|-----------|
| Apr | $15,970 | $532 |
| May | $10,465 | $338 |
| Jun | $3,731 | $124 |
| Jul | $20,238 | $653 |
| Aug | $14,291 | $461 |
| Sep | -$772 | -$26 |
| Oct | $3,275 | $106 |
| Nov | $13,700 | $457 |
| Dec | $1,307 | $44 |

### Cumulative PnL

Chart saved to: `adhoc/overnight_reversal_results/cumulative_pnl.png`

Steadily upward-sloping equity curve with a small drawdown mid-year (Sep-Oct), consistent with the signal strength patterns found in the research phase.

---

## 4. Transaction Cost Analysis

### Gross Alpha

The backtest produces ~3.1 bps/day of gross alpha per dollar of notional ($300 daily PnL on ~$983K notional). Since leverage scales both PnL and costs proportionally, the breakeven is independent of leverage.

### Binance USDS-M Futures Taker Fees

| Tier | Taker Fee (one-way) | Round-trip Cost |
|------|--------------------:|----------------:|
| Regular (VIP 0) | 5.0 bps | ~10.0 bps |
| With BNB discount | 4.5 bps | ~9.0 bps |
| VIP 9 (highest) | 1.7 bps | ~3.4 bps |

### Net Economics

| Scenario | Gross Alpha | Cost/Day | Net PnL/Day | Net PnL/Year |
|----------|-------------|----------|-------------|--------------|
| Regular taker (5 bps) | $300 | -$983 | **-$683** | -$187K |
| VIP 9 taker (1.7 bps) | $300 | -$334 | **-$34** | -$9K |
| Maker (2 bps) | $300 | -$393 | **-$93** | -$25K |
| Breakeven | $300 | -$300 | $0 | $0 |

**Breakeven round-trip cost**: ~3.1 bps (= $300 / $983K * 10,000). This is below all standard Binance fee tiers.

### With 5x Leverage

Leverage does not help — it scales both PnL and costs by the same factor:

| Scenario (5x) | Gross PnL/Yr | Cost/Yr | Net PnL/Yr |
|----------------|-------------|---------|------------|
| Regular taker | $411K | -$674K | **-$263K** |
| VIP 9 taker | $411K | -$457K | **-$46K** |

---

## 5. Conclusions

### The Signal is Real

- FM t-stat of 5.91 across 365 days
- Positive in all 12 months
- 12.5 bps D10-D1 spread
- Consistent with economic intuition: profit-taking/short-covering on EOD rallies

### But Not Tradeable Standalone

- Gross alpha of ~3 bps/day on notional is below the minimum achievable transaction cost on Binance futures (~3.4 bps round-trip at VIP 9)
- No fee tier makes this profitable as an isolated strategy
- Leverage does not change the breakeven

### Recommended Use

1. **Component signal**: Integrate into the multi-alpha framework where the marginal cost of adding a position is lower than the standalone round-trip cost
2. **Timing overlay**: Use the signal to tilt existing positions rather than as a standalone entry/exit mechanism
3. **Further research**:
   - Test on earlier years for out-of-sample validation
   - Investigate whether the signal can be captured via limit orders (maker fees) rather than market orders
   - Explore longer holding periods that amortize transaction costs over larger expected moves

---

## 6. Look-Ahead Bias Checklist

| Component | Data Used | Known At Decision Time? |
|-----------|-----------|------------------------|
| `logret_15` at minute t | Rolling sum of past 15 1-min returns | Yes |
| `logret_15_trstd` at minute t | Trailing 90-period std | Yes |
| Cross-sectional rank | All symbols' values at minute t | Yes |
| Trailing beta (90-day) | Signals + realized fwd returns from D-90 to D-1 | Yes |
| Entry price (`close_mid`) | Price at trade time | Yes |
| Exit price (avg `close_mid` rows 14-29) | Future prices | No (used only for PnL, not signal) |

---

## 7. Key Files

### Research Scripts (in `adhoc/`)

| Script | Experiment |
|--------|-----------|
| `overnight_reversal_15m.py` | Raw returns |
| `overnight_reversal_60m.py` | 60-min horizon comparison |
| `overnight_reversal_15m_voladj.py` | Volatility adjustment |
| `overnight_reversal_15m_voladj_volume.py` | Volume conditioning |
| `overnight_reversal_15m_asymmetry.py` | Up/down asymmetry |
| `overnight_reversal_15m_by_daily.py` | Daily return conditioning |
| `overnight_reversal_15m_fwd_resid.py` | Forward residual returns |
| `overnight_reversal_15m_fwd_resid_vol.py` | Volume + fwd resid |
| `overnight_reversal_15m_best_predictor.py` | Composite predictor optimization |
| `overnight_reversal_15m_backtest.py` | Full backtest with rolling beta |

### Data Sources

| Data | Path |
|------|------|
| 15-min bars | `data/bars/15/binance-futures/{YYYYMMDD}/bars.15.binance-futures.{YYYYMMDD}.{SYMBOL}.parquet` |
| Trailing vol feature | `data/features/15/logret_15_trstd/{YYYYMMDD}/features.15.logret_15_trstd.{YYYYMMDD}.{SYMBOL}.parquet` |
| Forward returns | `data/forwards/15/{YYYYMMDD}/forwards.15.{YYYYMMDD}.{SYMBOL}.parquet` |

### Output

| File | Description |
|------|-------------|
| `adhoc/overnight_reversal_results/cumulative_pnl.png` | Cumulative PnL chart from backtest |
| `adhoc/overnight_reversal_15m_findings.md` | Original detailed research findings |
