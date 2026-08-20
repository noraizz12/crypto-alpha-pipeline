#!/usr/bin/env bash

# Script to run simulations in the background with output collection
# Usage: ./bin/run_sim_bg.sh [additional_sim_args...]
# Example: ./bin/run_sim_bg.sh --name my_sim --from 20240101 --to 20240131

# Get directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/../bin/include.sh"
init_env

# Generate unique timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Extract sim name from arguments if provided, otherwise use timestamp
SIM_NAME="sim_${TIMESTAMP}"
for arg in "$@"; do
    if [[ "$prev_arg" == "--name" ]] || [[ "$prev_arg" == "-n" ]]; then
        SIM_NAME="$arg"
        break
    fi
    prev_arg="$arg"
done

# Create unique output filename
OUTPUT_FILE="${LOG_DIR}/sim_${SIM_NAME}_${TIMESTAMP}.log"
ERROR_FILE="${LOG_DIR}/sim_${SIM_NAME}_${TIMESTAMP}.err"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

echo "Starting simulation with arguments: $@"
echo "Timestamp: $TIMESTAMP"
echo "Output file: $OUTPUT_FILE"
echo "Error file: $ERROR_FILE"
echo "PID will be written to: ${LOG_DIR}/sim_${SIM_NAME}_${TIMESTAMP}.pid"

# Run simulation in background and capture PID
python "$SRC_DIR"/sim.py "$@" 1> "$OUTPUT_FILE" 2> "$ERROR_FILE" &
SIM_PID=$!

# Save PID to file for monitoring
echo $SIM_PID > "${LOG_DIR}/sim_${SIM_NAME}_${TIMESTAMP}.pid"

echo "Simulation started with PID: $SIM_PID"
echo "Monitor progress with: tail -f $OUTPUT_FILE"
echo "Check errors with: tail -f $ERROR_FILE"
echo "Kill simulation with: kill $SIM_PID"

# Wait for simulation to complete and send Slack notification
(
    wait $SIM_PID
    EXIT_CODE=$?
    
    # Find the summary.txt file in the sim results directory
    SUMMARY_FILE="../sims/${SIM_NAME}/summary.txt"
    
    # Send Slack notification with results
    send_slack_notification "$SIM_NAME" "$SUMMARY_FILE" "$OUTPUT_FILE" "$ERROR_FILE" "$EXIT_CODE"
    
    # Clean up PID file
    rm -f "${LOG_DIR}/sim_${SIM_NAME}_${TIMESTAMP}.pid"
) &