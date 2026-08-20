#!/usr/bin/env bash
. "$ROOT_DIR"/src/bin/include.sh
init_env

# Default lookback is 7 days if not specified
LOOKBACK=${1:-7}

python "$SRC_DIR"/sim.py --simcomp --init-positions --sim-dir="$SIM_DIR/simcomp" --sim-lookback="$LOOKBACK" 1> "$LOG_DIR"/sim_comparison_${LOOKBACK}d.log 2> "$LOG_DIR"/sim_comparison_${LOOKBACK}d.err &
PID1=$!
wait $PID1
