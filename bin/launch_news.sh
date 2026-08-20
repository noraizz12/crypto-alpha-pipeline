#!/usr/bin/env bash
. $ROOT_DIR/src/bin/include.sh
init_env

LOCK_FILE="$TRADING_DIR"/news.lock
if ! check_lock "$LOCK_FILE" "news"; then
  exit 1
fi
touch "$LOCK_FILE"

export STATARB_ENV=prod
python "$SRC_DIR"/lib/news/news_server.py "$@" 1> "$LOG_DIR"/news.log 2> "$LOG_DIR"/news.err
PID=$!
echo $PID >> "$LOCK_FILE"

rm -f "$LOCK_FILE"