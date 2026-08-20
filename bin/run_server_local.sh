#!/bin/bash

# Run server with local data configuration for fast testing.
# This script runs the server with a limited universe and optimized settings.

# Set environment for local mode
export LOCAL_MODE=1

# Get current date or use provided date
if [ $# -gt 0 ]; then
    DATE_STR=$1
else
    # Default to yesterday to ensure data exists
    DATE_STR=$(date -d "yesterday" +%Y%m%d)
fi

echo "Running server for date: $DATE_STR"
echo "Using config: config/config_local.json"
echo "Universe limited to 5 symbols for fast execution"

# Run the server with local config
CMD="python server.py -c config/config_local.json -s $DATE_STR -b new"

echo "Command: $CMD"

# Execute the command
$CMD