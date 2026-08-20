#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

python "$SRC_DIR"/sim.py --sim-lookback 365 --skip-missing-alphas --sim-dir="$SIM_DIR/long" 1> "$LOG_DIR"/sim_longterm.log 2> "$LOG_DIR"/sim_longterm.err &
PID1=$!
wait $PID1
