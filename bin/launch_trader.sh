#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

LOCK_FILE="$TRADING_DIR"/trader.lock
if ! check_lock "$LOCK_FILE" "trader"; then
  exit 1
fi
touch "$LOCK_FILE"

export STATARB_ENV=prod
python "$SRC_DIR"/trader.py "$@" 1> "$LOG_DIR"/trader.log 2> "$LOG_DIR"/trader.err

rm -f "$LOCK_FILE"


