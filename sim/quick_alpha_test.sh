#!/bin/bash
# Quick test of select alphas over multiple days

START_DATE="${1:-20241215}"
END_DATE="${2:-20241220}"
MAX_POS="${3:-200000}"
REBALANCE="${4:-120}"

echo "Testing select alphas from $START_DATE to $END_DATE"
echo "Rebalance every $REBALANCE minutes"
echo "Alpha,Sharpe,Annual_Return%,Total_PnL,Days"

# Test a subset of promising alphas
for alpha in hl_720 hl_1440 c2vwap_720 c2vwap_1440 slz_360 vadj_1440 oi_720 badj_720; do
    output=$(python sim/simple_sim.py \
        --start-date $START_DATE \
        --end-date $END_DATE \
        --alphas $alpha \
        --max-position $MAX_POS \
        --rebalance-minutes $REBALANCE \
        2>&1)
    
    if echo "$output" | grep -q "=== Simple Simulation Results ==="; then
        sharpe=$(echo "$output" | grep "Sharpe Ratio:" | sed 's/.*: //')
        annual_return=$(echo "$output" | grep "Annualized Return:" | sed 's/.*: //' | sed 's/%//')
        total_pnl=$(echo "$output" | grep "Total PnL:" | sed 's/.*\$//' | tr -d ',')
        days=$(echo "$output" | grep "Days simulated:" | sed 's/.*: //')
        
        echo "$alpha,$sharpe,$annual_return,$total_pnl,$days"
    else
        echo "$alpha,ERROR,0,0,0"
    fi
done | column -t -s,