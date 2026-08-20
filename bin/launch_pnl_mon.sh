#!/usr/bin/env bash
. $ROOT_DIR/src/bin/include.sh
init_env
export STATARB_ENV=prod

LOCK_FILE="$TRADING_DIR"/pnl_mon.lock
if ! check_lock "$LOCK_FILE" "pnl_mon"; then
  echo "Failed Lock Check"
  exit 1
fi
touch "$LOCK_FILE"

python "$SRC_DIR"/pnl_mon.py --loop=8 "$@" 1> "$LOG_DIR"/pnl_mon.log 2> "$LOG_DIR"/pnl_mon.err

rm -f "$LOCK_FILE"


