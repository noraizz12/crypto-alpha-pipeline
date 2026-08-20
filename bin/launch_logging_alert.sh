#!/usr/bin/env bash
. $ROOT_DIR/src/bin/include.sh
init_env

# Run the Python script
python "$SRC_DIR"/alert_logging_error.py --file "$LOG_DIR"/trader.err 1> "$LOG_DIR"/alert_logging_error_trader.log 2> "$LOG_DIR"/alert_logging_error_trader.err &
python "$SRC_DIR"/alert_logging_error.py --file /tmp/cron.update.err 1> "$LOG_DIR"/alert_update_error.log 2> "$LOG_DIR"/alert_update_error.err & 
python "$SRC_DIR"/alert_logging_error.py --file /tmp/cron.update_tardis.err 1> "$LOG_DIR"/alert_update_tardis_error.log 2> "$LOG_DIR"/alert_update_tardis_error.err & 
python "$SRC_DIR"/alert_logging_error.py --file "$LOG_DIR"/server.err 1> "$LOG_DIR"/alert_logging_error_server.log 2> "$LOG_DIR"/alert_logging_error_server.err & 
python "$SRC_DIR"/alert_logging_error.py --file "$LOG_DIR"/news.err 1> "$LOG_DIR"/alert_logging_error_news.log 2> "$LOG_DIR"/alert_logging_error_news.err &
