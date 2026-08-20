#!/bin/bash
# Monitor Opsgenie on-call schedule and send Slack notifications when it changes
. $ROOT_DIR/src/bin/include.sh
init_env

python "$ROOT_DIR"/src/monitor_oncall.py "$@"
