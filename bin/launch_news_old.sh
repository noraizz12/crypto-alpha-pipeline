#!/usr/bin/env bash
. $ROOT_DIR/src/bin/include.sh
init_env

LOCK_FILE="$TRADING_DIR"/newsold.lock
if ! check_lock "$LOCK_FILE" "newsold"; then
  exit 1
fi
touch "$LOCK_FILE"

export STATARB_ENV=prod
python "$SRC_DIR"/lib/external/news_server.py "$@" 1> "$LOG_DIR"/newsold.log 2> "$LOG_DIR"/newsold.err
PID=$!
echo $PID >> "$LOCK_FILE"

rm -f "$LOCK_FILE"