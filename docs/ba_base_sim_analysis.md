# ba_base Simulation Analysis

**Sim:** `sims/ba_base` | **Period:** 2025-02-01 to 2026-01-31 (365 days) | **Config:** `config.json` (no beta/vol bounds)

## Overall Performance

| Metric | Value |
|--------|-------|
| Cumulative PnL | $12,893,254 |
| Annualized Sharpe | 1.95 |
| Avg Daily PnL | $32,733 |
| Daily Std | $320,921 |
| Skewness | -1.00 |
| Kurtosis | 5.54 |
| Win Rate | 55.1% |
| Max Drawdown | -$3,349,942 |

## Daily PnL Distribution

| Percentile | Value |
|-----------|-------|
| 1st | -$961,789 |
| 5th | -$457,676 |
| 10th | -$305,258 |
| 25th | -$124,780 |
| 50th (median) | $30,081 |
| 75th | $212,446 |
| 90th | $407,186 |
| 95th | $486,924 |
| 99th | $718,153 |

Notable: The distribution is negatively skewed (-1.0) with fat tails (kurtosis 5.5). The left tail is fatter — the 1st percentile loss ($962K) is 1.3x the 99th percentile gain ($718K).

## Monthly Breakdown

| Month | PnL | Avg Daily | Daily Std | Ann Sharpe | Worst Day | Best Day |
|-------|-----|-----------|-----------|------------|-----------|----------|
| 2025-02 | -$1,324,088 | -$49,040 | $492,222 | -1.90 | -$961,576 | $597,496 |
| 2025-03 | $2,910,994 | $93,903 | $255,430 | 7.02 | -$379,196 | $542,478 |
| 2025-04 | -$924,439 | -$30,815 | $409,702 | -1.44 | -$1,148,419 | $538,081 |
| 2025-05 | $2,424,830 | $78,220 | $250,968 | 5.95 | -$388,532 | $691,386 |
| 2025-06 | $2,570,533 | $85,684 | $201,006 | 8.14 | -$399,356 | $391,172 |
| 2025-07 | -$11,896 | -$384 | $249,141 | -0.03 | -$460,740 | $747,108 |
| 2025-08 | $1,986,766 | $64,089 | $168,823 | 7.25 | -$237,758 | $589,051 |
| 2025-09 | -$324,864 | -$10,829 | $174,510 | -1.19 | -$430,826 | $431,326 |
| 2025-10 | $2,260,694 | $72,926 | $345,472 | 4.03 | -$345,443 | $1,206,008 |
| 2025-11 | $330,115 | $11,004 | $522,482 | 0.40 | -$2,016,793 | $701,148 |
| 2025-12 | $209,583 | $6,761 | $297,550 | 0.43 | -$666,145 | $784,827 |
| 2026-01 | $1,806,550 | $58,276 | $297,394 | 3.74 | -$822,480 | $432,021 |

Notable: Strong months (Mar, May, Jun, Aug, Oct) have Sharpe > 4. Losing months (Feb, Apr, Jul, Sep) cluster in Q1 and Q3. Nov/Dec are marginal despite the two worst single days falling in November.

---

## Worst 20 Days

| Date | Portfolio PnL | Worst Contributors |
|------|--------------|-------------------|
| 2025-11-07 | **-$2,016,793** | HUSDT -$710K, ZECUSDT -$423K, FILUSDT -$343K |
| 2025-04-23 | **-$1,148,419** | TRUMPUSDT -$475K, SUIUSDT -$134K, WIFUSDT -$107K |
| 2025-04-11 | **-$1,021,119** | FARTCOINUSDT -$282K, POPCATUSDT -$235K, ALCHUSDT -$195K |
| 2025-11-14 | -$962,150 | ZECUSDT -$206K, SOONUSDT -$184K, DASHUSDT -$146K |
| 2025-02-08 | -$961,576 | POPCATUSDT -$383K, FARTCOINUSDT -$378K, PNUTUSDT -$165K |
| 2025-02-25 | -$864,964 | ACTUSDT -$155K, FARTCOINUSDT -$150K, POPCATUSDT -$97K |
| 2026-01-14 | -$822,480 | RIVERUSDT -$242K, ZECUSDT -$182K, ICPUSDT -$173K |
| 2025-04-09 | -$817,214 | FARTCOINUSDT -$504K, ALCHUSDT -$224K, BERAUSDT -$159K |
| 2025-02-22 | -$724,105 | FARTCOINUSDT -$190K, RUNEUSDT -$153K, GOATUSDT -$120K |
| 2025-02-10 | -$722,889 | FARTCOINUSDT -$339K, RUNEUSDT -$133K, LTCUSDT -$118K |
| 2025-12-31 | -$666,145 | LIGHTUSDT -$868K, ADAUSDT -$56K, PUMPUSDT -$35K |
| 2025-02-28 | -$647,810 | PNUTUSDT -$280K, NEIROUSDT -$231K, POPCATUSDT -$142K |
| 2025-02-26 | -$607,191 | PNUTUSDT -$171K, RUNEUSDT -$127K, POPCATUSDT -$126K |
| 2025-02-14 | -$590,484 | PNUTUSDT -$356K, POPCATUSDT -$189K, NEIROUSDT -$138K |
| 2025-02-20 | -$552,936 | TAOUSDT -$122K, MKRUSDT -$93K, POPCATUSDT -$80K |
| 2025-11-04 | -$541,224 | ICPUSDT -$157K, TAOUSDT -$144K, AIAUSDT -$138K |
| 2025-12-20 | -$513,612 | GIGGLEUSDT -$168K, HUSDT -$60K, ZECUSDT -$55K |
| 2025-04-26 | -$462,567 | WLDUSDT -$149K, NEIROUSDT -$76K, WIFUSDT -$53K |
| 2025-07-25 | -$460,740 | ENAUSDT -$296K, SYRUPUSDT -$167K, MKRUSDT -$69K |
| 2025-12-24 | -$440,317 | PIPPINUSDT -$106K, ZECUSDT -$99K, FARTCOINUSDT -$44K |

---

## Best 20 Days

| Date | Portfolio PnL | Best Contributors |
|------|--------------|-------------------|
| 2025-10-10 | **$1,206,008** | MYXUSDT $702K, PENGUUSDT $353K, SOMIUSDT $301K |
| 2025-10-30 | $925,426 | PUMPUSDT $195K, ENAUSDT $140K, TRUMPUSDT $127K |
| 2025-12-01 | $784,827 | ZECUSDT $280K, GIGGLEUSDT $243K |
| 2025-07-18 | $747,108 | UNIUSDT $250K, FARTCOINUSDT $98K, HBARUSDT $87K |
| 2025-11-13 | $701,148 | COAIUSDT $246K, AIAUSDT $160K, ZENUSDT $109K |
| 2025-05-30 | $691,386 | OMUSDT $155K, FARTCOINUSDT $125K, KAITOUSDT $120K |
| 2025-11-20 | $644,314 | SOONUSDT $153K, 4USDT $139K, COAIUSDT $108K |
| 2025-10-22 | $634,166 | ASTERUSDT $207K, ENAUSDT $128K, MYXUSDT $96K |
| 2025-02-27 | $597,496 | RUNEUSDT $254K, TAOUSDT $110K, PNUTUSDT $76K |
| 2025-08-14 | $589,051 | FARTCOINUSDT $161K, ADAUSDT $136K, PENGUUSDT $132K |
| 2025-11-08 | $579,535 | ZECUSDT $252K, FILUSDT $158K, BLESSUSDT $132K |
| 2025-02-09 | $578,132 | POPCATUSDT $183K, FARTCOINUSDT $165K, WIFUSDT $141K |
| 2025-12-02 | $560,955 | ADAUSDT $130K, SOLUSDT $122K, AAVEUSDT $118K |
| 2025-03-18 | $542,478 | AI16ZUSDT $163K, VIRTUALUSDT $69K, TRXUSDT $66K |
| 2025-04-04 | $538,081 | FARTCOINUSDT $153K, ALCHUSDT $132K, LAYERUSDT $80K |
| 2025-02-11 | $528,850 | AI16ZUSDT $337K, WIFUSDT $74K, POPCATUSDT $71K |
| 2025-03-01 | $510,742 | NEIROUSDT $310K, POPCATUSDT $145K, FARTCOINUSDT $143K |
| 2025-11-21 | $502,221 | ZECUSDT $278K, FARTCOINUSDT $253K, BCHUSDT $161K |
| 2025-04-06 | $487,082 | NEIROUSDT $148K, ETHUSDT $144K, FARTCOINUSDT $133K |
| 2025-03-04 | $486,026 | WIFUSDT $145K, ALCHUSDT $82K, ADAUSDT $77K |

---

## Repeat Offender Names

Names appearing with >$50K loss on the worst 30 days, sorted by frequency:

| Name | # Bad Days | Total Loss | Avg Loss | Avg \|Beta\| | Avg Vol |
|------|-----------|-----------|----------|-------------|---------|
| **FARTCOINUSDT** | **12** | **-$2,411,302** | -$200,942 | 0.369 | 0.0394 |
| **WIFUSDT** | **11** | -$1,083,705 | -$98,519 | 0.679 | 0.0223 |
| **NEIROUSDT** | **11** | -$1,057,654 | -$96,150 | 0.673 | 0.0272 |
| **PNUTUSDT** | **9** | **-$1,588,676** | -$176,520 | 0.640 | 0.0355 |
| **POPCATUSDT** | **9** | **-$1,462,121** | -$162,458 | 0.482 | 0.0356 |
| **ZECUSDT** | **6** | **-$1,270,029** | -$211,672 | 0.194 | 0.0222 |
| TAOUSDT | 6 | -$548,615 | -$91,436 | 0.646 | 0.0231 |
| VIRTUALUSDT | 5 | -$381,666 | -$76,333 | 0.617 | 0.0246 |
| ALCHUSDT | 4 | -$633,904 | -$158,476 | 0.245 | 0.0502 |
| MOODENGUSDT | 4 | -$325,836 | -$81,459 | 0.721 | 0.0275 |
| TRUMPUSDT | 3 | -$626,161 | -$208,720 | 0.663 | 0.0332 |
| RUNEUSDT | 3 | -$413,583 | -$137,861 | 0.443 | 0.0249 |
| ICPUSDT | 3 | -$396,892 | -$132,297 | 0.681 | 0.0427 |
| GOATUSDT | 3 | -$281,503 | -$93,834 | 0.681 | 0.0239 |
| ZENUSDT | 3 | -$248,789 | -$82,930 | 0.280 | 0.0253 |
| HUSDT | 2 | -$770,003 | -$385,002 | 0.099 | 0.0807 |
| AVNTUSDT | 2 | -$419,197 | -$209,599 | 0.434 | 0.0369 |
| ENAUSDT | 2 | -$404,239 | -$202,120 | 0.662 | 0.0306 |
| AIAUSDT | 2 | -$351,538 | -$175,769 | 0.179 | 0.1275 |
| DASHUSDT | 2 | -$229,224 | -$114,612 | 0.199 | 0.0255 |

### Key observations on repeat offenders

- **FARTCOINUSDT** is the #1 offender by far: 12 appearances, -$2.4M total. Low beta (0.37), very high vol (0.039). Heavily meme-driven.
- The meme cluster (FARTCOIN, WIF, NEIRO, PNUT, POPCAT) collectively accounts for 52 appearances and -$7.6M in losses on worst days.
- **ZECUSDT** stands out: only 6 appearances but -$1.27M total. Very low beta (0.19), meaning almost pure idiosyncratic risk. All 6 appearances are in Nov 2025 - Jan 2026.
- **HUSDT** has the highest per-event loss (-$385K avg) with the lowest beta (0.10) and highest vol (0.081).

---

## Hero Names

Names appearing with >$50K gain on the best 30 days:

| Name | # Good Days | Total Gain | Avg Gain | Avg \|Beta\| |
|------|------------|-----------|----------|-------------|
| **FARTCOINUSDT** | **16** | $2,087,539 | $130,471 | 0.468 |
| WIFUSDT | 8 | $927,750 | $115,969 | 0.707 |
| TAOUSDT | 7 | $727,585 | $103,941 | 0.648 |
| ZECUSDT | 6 | $1,101,483 | $183,580 | 0.205 |
| POPCATUSDT | 6 | $739,787 | $123,298 | 0.502 |
| GOATUSDT | 6 | $646,870 | $107,812 | 0.699 |
| ENAUSDT | 5 | $656,382 | $131,276 | 0.609 |
| ASTERUSDT | 5 | $577,464 | $115,493 | 0.402 |
| PNUTUSDT | 5 | $533,325 | $106,665 | 0.633 |
| AI16ZUSDT | 4 | $674,814 | $168,703 | 0.404 |
| NEIROUSDT | 4 | $604,995 | $151,249 | 0.661 |

### Dual-role names (both offender and hero)

| Name | # Bad | Bad Total | # Good | Good Total | **Net** |
|------|-------|----------|--------|-----------|---------|
| FARTCOINUSDT | 12 | -$2,411K | 16 | $2,088K | **-$324K** |
| PNUTUSDT | 9 | -$1,589K | 5 | $533K | **-$1,055K** |
| POPCATUSDT | 9 | -$1,462K | 6 | $740K | **-$722K** |
| NEIROUSDT | 11 | -$1,058K | 4 | $605K | **-$453K** |
| WIFUSDT | 11 | -$1,084K | 8 | $928K | **-$156K** |
| ZECUSDT | 6 | -$1,270K | 6 | $1,101K | **-$169K** |
| HUSDT | 2 | -$770K | 2 | $114K | **-$656K** |
| TAOUSDT | 6 | -$549K | 7 | $728K | **+$179K** |
| GOATUSDT | 3 | -$282K | 6 | $647K | **+$365K** |
| ENAUSDT | 2 | -$404K | 5 | $656K | **+$252K** |

Key insight: Most repeat offenders are net negative on tail days. The biggest net losers are **PNUTUSDT (-$1.06M)**, **POPCATUSDT (-$722K)**, **HUSDT (-$656K)**, **NEIROUSDT (-$453K)**, and **FARTCOINUSDT (-$324K)**. These names cost us on extreme days even though they generate alpha on normal days.

---

## Characteristics of Big Loss Events

### Beta distribution

| Beta Range | # Events | Total Loss | Avg Loss/Event | Avg Vol |
|-----------|---------|-----------|---------------|---------|
| [0.0, 0.2) | 22 | -$4.80M | **-$218K** | 0.056 |
| [0.2, 0.4) | 25 | -$4.15M | -$166K | 0.038 |
| [0.4, 0.6) | 43 | -$5.32M | -$124K | 0.031 |
| [0.6, 0.8) | 62 | -$6.62M | -$107K | 0.026 |
| [0.8, 1.0) | 15 | -$1.67M | -$112K | 0.028 |
| [1.0, 2.0) | 6 | -$0.40M | -$66K | 0.016 |

Key insight: Low-beta names ([0, 0.4)) have 47 events with **-$8.95M total loss** and the highest per-event loss (-$218K and -$166K). High-beta names ([0.6, 1.0)) have more events (77) but contribute -$8.30M with lower per-event impact. Low-beta losses hit harder per event.

### Vol characteristics on blowup day

For 89 big loss events (>$50K loss per name on worst 20 days):

**Vol ratio (name vol / cross-sectional mean vol) on blowup day:**
- Mean: 1.94x
- Median: 1.56x
- 73% of events had vol_ratio > 1.3
- 56% had vol_ratio > 1.5
- 31% had vol_ratio > 2.0

**Vol change (D0 vs D-3):**
- Mean: 1.49x
- Median: 1.16x
- 46% of events had vol_ratio increase > 1.2x over 3 days
- 27% had increase > 1.5x
- 18% had increase > 2.0x

**Vol ratio on D-1 (day before blowup):**
- Mean: 1.49x
- 49% already had vol_ratio > 1.3 the day before

---

## Vol Trajectory Examples

Selected blowup events showing vol leading up to loss:

### 2025-11-07 (-$2.0M, worst day of the year)
| Name | Loss | Beta | D-5 | D-3 | D-1 | D0 | D0/D-3 | VR |
|------|------|------|-----|-----|-----|-----|--------|-----|
| HUSDT | -$710K | 0.10 | 0.033 | 0.028 | 0.095 | 0.112 | 4.1x | 4.0x |
| ZECUSDT | -$423K | 0.13 | 0.027 | 0.044 | 0.038 | 0.030 | 0.7x | 1.1x |
| FILUSDT | -$343K | 0.89 | 0.016 | 0.025 | 0.036 | 0.073 | 3.0x | 2.7x |
| AIAUSDT | -$213K | 0.16 | 0.045 | 0.087 | 0.147 | 0.168 | 1.9x | 6.1x |

HUSDT vol went from 0.028 to 0.112 (4x) in 3 days. AIAUSDT was already running hot at 6x cross-section.

### 2025-04-09 (-$817K)
| Name | Loss | Beta | D-5 | D-3 | D-1 | D0 | D0/D-3 | VR |
|------|------|------|-----|-----|-----|-----|--------|-----|
| FARTCOINUSDT | -$504K | 0.35 | 0.055 | 0.044 | 0.043 | 0.059 | 1.4x | 2.3x |
| ALCHUSDT | -$224K | 0.13 | 0.040 | 0.036 | 0.047 | 0.039 | 1.1x | 1.5x |
| BERAUSDT | -$159K | 0.38 | 0.023 | 0.024 | 0.033 | 0.073 | 3.1x | 2.8x |

BERAUSDT vol tripled in 3 days. FARTCOINUSDT was chronically elevated at 2.3x cross-section.

### 2025-12-31 (-$666K)
| Name | Loss | Beta | D-5 | D-3 | D-1 | D0 | D0/D-3 | VR |
|------|------|------|-----|-----|-----|-----|--------|-----|
| LIGHTUSDT | -$868K | 0.09 | 0.121 | 0.038 | 0.034 | 0.155 | 4.1x | 11.7x |

Single name LIGHTUSDT drove the entire loss. Vol spiked from 0.034 to 0.155 on the day, but was at 0.121 five days prior — it had briefly calmed then exploded again.

---

## Summary & Implications for Risk Management

### The core problem
The worst days are driven by a small number of idiosyncratic blowups, not broad market moves. On the worst 20 days, 89 individual name-level events (>$50K loss) account for the bulk of portfolio losses. These are concentrated in:
1. **Meme/micro-cap names** (FARTCOIN, WIF, PNUT, POPCAT, NEIRO) — structurally high-vol, moderate beta
2. **Low-beta high-idio names** (ZEC, HUSDT, ALCH, DASH, AUCTION) — very low beta, pure idiosyncratic risk
3. **Occasional large-cap events** (TRUMP, ICP, FIL) — higher beta but vol spikes

### What signals predict blowups?
1. **Elevated vol level (cross-sectional)**: 73% of events had vol_ratio > 1.3x on the day. But 49% were already elevated on D-1, meaning the signal is often **already visible** before the loss.
2. **Vol spike (D0 vs D-3)**: 46% had vol > 1.2x their own D-3 level. This captures acute risk-up, but only catches half the events.
3. **Low beta**: Events in beta < 0.4 have 2x the per-event loss (-$190K avg) vs beta > 0.6 (-$107K avg).
4. **The combination matters**: Low beta + high vol_ratio captures the most destructive events (HUSDT at beta 0.10 / VR 4.0, AIAUSDT at beta 0.16 / VR 6.1).

### Why static vol_ratio bounds failed (beta_vol1/2/3 sims)
The BETA_VOL_BOUNDS approach constrained names when `|beta| < threshold AND vol_ratio > threshold`. The problem: many of these names are **structurally** high vol_ratio (e.g., FARTCOINUSDT is chronically at 2-3x). The bounds fired on ~80% of days, suppressing alpha capture broadly. The damage on "normal" days (-$2.7M) dwarfed the savings on worst days (+$1.2M).

### What a transient signal needs
A successful approach must distinguish:
- **Chronic high vol** (normal state for meme coins, alpha still works) → don't constrain
- **Acute vol spike** (precursor to blowup) → constrain

Potential transient signals to explore:
- **Vol acceleration**: vol_today / vol_rolling_5d_mean > threshold (captures acute spikes vs own baseline)
- **Vol rank change**: name's cross-sectional vol rank jumping significantly (e.g., from 50th to 90th percentile in 1-3 days)
- **Absolute vol spike**: vol / vol_3d_ago > threshold (D0/D-3 ratio caught 46% of events at 1.2x)
- **Combination**: low beta + vol acceleration + vol_ratio (three-feature gate)
