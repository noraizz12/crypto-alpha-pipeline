#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

python "$SRC_DIR"/generate_fits.py --prod 1> "$LOG_DIR"/fits.log 2> "$LOG_DIR"/fits.err
