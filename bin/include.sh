#!/usr/bin/env bash

# Common utilities for shell scripts

# Set default environment variables
export PYTHONUNBUFFERED=1

# Function to initialize environment and directories
init_env() {
    echo "User is $USER"

    if [ -z "$USER" ]; then
        export USER=$(id -un)
        echo "Empty USER ENV, set User to $USER"
    fi

    export STATARB_ENV=dev
    . "$ROOT_DIR"/env.sh

    if [ "$ROOT_DIR" = "" ]; then
      echo "Must set ROOT_DIR!"
      exit 1
    fi

    export SRC_DIR=$ROOT_DIR/src

    # Determine virtual environment directory (.venv for uv, venv for pip)
    if [ -d "$SRC_DIR/.venv" ]; then
        export VIRTUALENV_DIR="$SRC_DIR/.venv"
    elif [ -d "$SRC_DIR/venv" ]; then
        export VIRTUALENV_DIR="$SRC_DIR/venv"
    else
        echo "Warning: No virtual environment found at $SRC_DIR/.venv or $SRC_DIR/venv"
        export VIRTUALENV_DIR=""
    fi

    # Activate virtual environment if found
    if [ -n "$VIRTUALENV_DIR" ] && [ -f "$VIRTUALENV_DIR/bin/activate" ]; then
        . "$VIRTUALENV_DIR/bin/activate"
    fi

    export LOG_DIR=$ROOT_DIR/log
    export SIM_DIR=$ROOT_DIR/sims
    export TRADING_DIR=$ROOT_DIR/trading
    export PYTHONPATH=$SRC_DIR:$PYTHONPATH
    export PYTHONUNBUFFERED=1
}

# Function to send Slack notification with summary
send_slack_notification() {
    local sim_name="$1"
    local summary_file="$2"
    local output_file="$3"
    local error_file="$4"
    local exit_code="$5"

    if [[ $exit_code -eq 0 ]]; then
        local status="✅ COMPLETED"
    else
        local status="❌ FAILED"
    fi

    local message="*Simulation ${status}*: \`${sim_name}\`"

    # Add summary content if available
    if [[ -f "$summary_file" ]]; then
        message+="\n\n*Summary:*\n\`\`\`$(cat "$summary_file")\`\`\`"
    else
        message+="\n\n*Summary file not found at:* \`${summary_file}\`"
    fi

    message+="\n\n*Logs:*\n• Output: \`${output_file}\`\n• Errors: \`${error_file}\`"

    # Send to Slack channel using environment webhook
    local webhook_url="${SLACK_JOBS_WEBHOOK:-}"
    if [[ -n "$webhook_url" ]]; then
        python -c "
from slack_sdk.webhook.client import WebhookClient
import os
webhook_url = os.environ.get('SLACK_JOBS_WEBHOOK', '')
if webhook_url:
    webhook = WebhookClient(webhook_url)
    webhook.send(text='${message}')
"
    fi
}

# Function to check lock file
check_lock() {
    local LOCK_FILE="$1"
    local PROCESS_NAME="$2"

    if [ -e "$LOCK_FILE" ]; then
        pid=$(ps aux | grep python | grep "$PROCESS_NAME" | grep -v alert_logging_error | awk '{print $2}')
        if [ -z "$pid" ] || ! ps -p "$pid" > /dev/null 2>&1; then
            echo "Process $pid not existing, so remove $LOCK_FILE"
            rm -f "$LOCK_FILE"
        else
            echo "$PROCESS_NAME already appears to be running, remove $LOCK_FILE"
            return 1
        fi
    fi
    return 0
}
