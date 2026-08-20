#!/usr/bin/env bash
. $ROOT_DIR/src/bin/include.sh
init_env
export STATARB_ENV=prod

LOCK_FILE="$TRADING_DIR"/binance.lock
if ! check_lock "$LOCK_FILE" "download_binance"; then
  exit 1
fi
touch "$LOCK_FILE"

python "$SRC_DIR"/download_binance_data.py --loop=10 "$@" 1> "$LOG_DIR"/binance.log 2> "$LOG_DIR"/binance.err

rm -f "$LOCK_FILE"
