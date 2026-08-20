#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

# Function to display usage
usage() {
    echo "Usage: $0 <report_type> [options]"
    echo "Report types:"
    echo "  trading      - Trading Reports Dashboard (default port: 8050)"
    echo "  historical   - Historical Reports Dashboard (default port: 8051)"
    echo "  slippage     - Slippage Reports Dashboard (default port: 8052)"
    echo "  fits         - Prod Fits Reports Dashboard (default port: 8053)"
    echo "  sim          - Sim Trading Reports Dashboard (default port: 8054)"
    echo "  universe     - Universe Report Dashboard (default port: 8056)"
    echo "  hist_sim     - Historical Sim Reports Dashboard (default port: 8057)"
    echo "  alpha        - Alpha Reports Dashboard (default port: 8054)"
    echo "  execution    - Execution Monitoring Dashboard (default port: 8058)"
    echo ""
    echo "Options:"
    echo "  -p, --port PORT    Override default port"
    echo "  -d, --debug        Run in debug mode"
    echo "  -i, --interval SEC Set refresh interval in seconds"
    echo "  -h, --help         Show this help message"
    exit 1
}

# Check if report type is provided
if [ -z "$1" ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
fi

REPORT_TYPE=$1
shift  # Remove report type from arguments

echo "Launching report $REPORT_TYPE"

# Set defaults based on report type
case "$REPORT_TYPE" in
    trading)
        APP_FILE="trading_reports_app.py"
        DEFAULT_PORT=8050
        LOCK_NAME="trading_reports"
        ;;
    historical)
        APP_FILE="historical_reports_app.py"
        DEFAULT_PORT=8051
        LOCK_NAME="historical_reports"
        ;;
    slippage)
        APP_FILE="slippage_reports_app.py"
        DEFAULT_PORT=8052
        LOCK_NAME="slippage_reports"
        ;;
    fits)
        APP_FILE="fits_reports_prod_app.py"
        DEFAULT_PORT=8053
        LOCK_NAME="fits_reports"
        ;;
    sim)
        APP_FILE="pnl_comparison_app.py"
        DEFAULT_PORT=8055
        LOCK_NAME="sim_reports"
        ;;
    universe)
        APP_FILE="universe_report.py"
        DEFAULT_PORT=8056
        LOCK_NAME="universe_reports"
        ;;
    hist_sim)
        APP_FILE="hist_sim_reports_app.py"
        DEFAULT_PORT=8057
        LOCK_NAME="hist_sim_reports"
        ;;
    alpha)
        APP_FILE="alpha_report.py"
        DEFAULT_PORT=8054
        LOCK_NAME="alpha_reports"
        ;;
    execution)
        APP_FILE="execution_reports_app.py"
        DEFAULT_PORT=8058
        LOCK_NAME="execution_reports"
        ;;
    *)
        echo "Error: Unknown report type '$REPORT_TYPE'"
        echo ""
        usage
        ;;
esac

# Check if app file exists
if [ ! -f "$SRC_DIR/reports/$APP_FILE" ]; then
    echo "Error: Report app file not found: $SRC_DIR/reports/$APP_FILE"
    echo "This report type may not be implemented yet."
    exit 1
fi

# Parse command line arguments
PORT=$DEFAULT_PORT
ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -d|--debug)
            ARGS="$ARGS --debug"
            shift
            ;;
        -i|--interval)
            ARGS="$ARGS --interval $2"
            shift 2
            ;;
        *)
            ARGS="$ARGS $1"
            shift
            ;;
    esac
done

# Set up lock file
LOCK_FILE="$TRADING_DIR"/${LOCK_NAME}.lock
if ! check_lock "$LOCK_FILE" "$LOCK_NAME"; then
  echo "Lock file found, not running $Remove"
  exit 1
fi
touch "$LOCK_FILE"

# Launch the report app
echo "Starting ${REPORT_TYPE^} Reports Dashboard on port $PORT..."
python "$SRC_DIR"/reports/$APP_FILE -p $PORT $ARGS 1> "$LOG_DIR"/${LOCK_NAME}.log 2> "$LOG_DIR"/${LOCK_NAME}.err &
PID=$!
echo $PID >> "$LOCK_FILE"

echo "${REPORT_TYPE^} Reports Dashboard started with PID $PID"
echo "Access at http://localhost:$PORT"
echo "Logs: $LOG_DIR/${LOCK_NAME}.log"
echo "To stop: ./bin/stop.sh $LOCK_NAME"

# Wait for process to finish
wait $PID

# Clean up lock file
rm -f "$LOCK_FILE"