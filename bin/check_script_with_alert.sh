#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

# Get the PID of the parent process (the shell running the pipeline)
parent_pid=$PPID
# Retrieve the command name of the last executed script dynamically
previous_command=$(ps -o args= --ppid "$parent_pid" | head -n 1)
echo "previous_command = $previous_command"
previous_pid=$(ps --ppid "$parent_pid" -o pid=,args= | grep -m 1 "$previous_command" | awk '{print $1}')

output=""
while IFS= read -r line; do
  output+="$line"$'\n'
  exit_status=$(awk '{print $38}' /proc/$previous_pid/stat 2>/dev/null)
done

# Check the exit status
if [[ -n $exit_status ]]; then
    echo "Error: $previous_command failed. Triggering Opsgenie alert..."
    python "$SRC_DIR"/lib/opsgenie.py --key "Script Failed: ${previous_command}"
    exit 1
else
    echo "$previous_command executed successfully."
    python "$SRC_DIR"/monit_slack_alert.py --case "Script Completed: ${previous_command}" --msg-only
fi

