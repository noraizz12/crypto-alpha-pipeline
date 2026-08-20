#!/usr/bin/env bash
. $ROOT_DIR/src/bin/include.sh
init_env

echo "Logging to $LOG_DIR/download_defillama.log"
echo "starting downloading defillama..." > $LOG_DIR/download_defillama.log

python "$SRC_DIR"/download_defi_llama.py $UPDATE >> "$LOG_DIR"/download_defillama.log

echo "Done daily download defillama.." >> $LOG_DIR/download_defillama.log