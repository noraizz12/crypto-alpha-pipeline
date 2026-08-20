#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

FROM_DATE=""
TO_DATE=""

while getopts f:t: flag
do
    case "${flag}" in
        f) FROM_DATE=${OPTARG};;
        t) TO_DATE=${OPTARG};;
    esac
done

CMD="python $SRC_DIR/backfill_binance_fills.py"

if [ -n "$FROM_DATE" ]; then
    CMD="$CMD -f $FROM_DATE"
fi

if [ -n "$TO_DATE" ]; then
    CMD="$CMD -t $TO_DATE"
fi

# Run the command and redirect output and errors to log files
$CMD 1> "$LOG_DIR"/backfill_fills.log 2> "$LOG_DIR"/backfill_fills.err