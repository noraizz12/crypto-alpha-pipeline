#!/usr/bin/env bash
. $ROOT_DIR/src/bin/include.sh
init_env

cat $SRC_DIR/cron/prod.cron | crontab -
