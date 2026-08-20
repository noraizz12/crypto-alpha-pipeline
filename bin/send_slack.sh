#!/usr/bin/env bash
. $ROOT_DIR/src/bin/include.sh
init_env

python "$SRC_DIR"/send_slack.py "$@"
