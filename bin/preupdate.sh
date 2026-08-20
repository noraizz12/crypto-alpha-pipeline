#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

echo "Logging to $LOG_DIR/preupdate.log"
echo "starting update env: $STATARB_ENV ..." > "$LOG_DIR"/preupdate.log

python "$SRC_DIR"/generate_live_bars.py --update || failed=true

if [ "$STATARB_ENV" = "prod" ]; then
  echo "Downloading coingecko..." >> "$LOG_DIR"/preupdate.log
  python "$SRC_DIR"/download_coingecko_data.py >> "$LOG_DIR"/preupdate.log 2>&1
fi
