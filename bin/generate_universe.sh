#!/usr/bin/env bash
. $ROOT_DIR/src/bin/include.sh
init_env

echo "Logging to $LOG_DIR/update_universe.log"
echo "starting update universe..." > $LOG_DIR/update_universe.log

python "$SRC_DIR"/generate_universe.py --update >> "$LOG_DIR"/update_universe.log

echo "Done daily update universe.." >> $LOG_DIR/update_universe.log