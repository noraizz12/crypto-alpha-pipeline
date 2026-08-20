#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

LOCK_FILE="$TRADING_DIR"/server.lock
if ! check_lock "$LOCK_FILE" "server"; then
  exit 1
fi
touch "$LOCK_FILE"

export STATARB_ENV=prod

python "$SRC_DIR"/server.py "$@" 1> "$LOG_DIR"/server.log 2> "$LOG_DIR"/server.err
PID=$!
echo $PID >> "$LOCK_FILE"

rm -f "$LOCK_FILE"