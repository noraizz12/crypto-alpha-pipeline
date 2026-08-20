#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

CLI_STATARB_ENV="dev"        # default value for cli_live_env

if [[ "$1" == "--cli_live_env" ]]; then
    if [[ -z "$2" ]]; then
        echo "Error: --cli_live_env requires a value"
        exit 1
    fi
    
    CLI_STATARB_ENV="$2"
    
    # Validate the value
    if [[ ! "$CLI_STATARB_ENV" =~ ^(prod|dev)$ ]]; then
        echo "Error: --cli_live_env must be 'prod' or 'dev'"
        exit 1
    fi
fi

export STATARB_ENV="$CLI_STATARB_ENV"

echo "Logging to $LOG_DIR/update_backfill.log"
echo "starting update backfill..." > $LOG_DIR/update_backfill.log

CONFIG=""
if [ "$CLI_STATARB_ENV" = "dev" ]; then
  CONFIG="--config=$SRC_DIR/config/config_dev.json"
fi

python "$SRC_DIR"/generate_bars.py $CONFIG --backfill --bars-type=new >> "$LOG_DIR"/update_backfill.log 2>&1

echo "Done daily update.." >> $LOG_DIR/update_backfill.log
