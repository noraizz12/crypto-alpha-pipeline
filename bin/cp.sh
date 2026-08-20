#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

python "$SRC_DIR"/cat_parquet.py "$@" | column -t -s,