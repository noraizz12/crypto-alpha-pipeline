#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

SERVER_KILL_FILE=$TRADING_DIR/server.kill
SERVER_UPDATE_FILE=$TRADING_DIR/server.update
TRADER_KILL_FILE=$TRADING_DIR/trader.kill

restart_service() {
    local service_name=$1
    local sleep_secs=$2
    shift 2

    local kill_files=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        kill_files+=("$1")
        shift
    done
    [[ "$1" == "--" ]] && shift
    local launch_cmd=("$@")

    echo "touching ${service_name} kill... $(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_DIR"/update.log
    for file in "${kill_files[@]}"; do
        touch "$file"
    done
    sleep $sleep_secs
    for file in "${kill_files[@]}"; do
        rm -f "$file"
    done
    nohup "${launch_cmd[@]}" &
}

echo "Logging to $LOG_DIR/update.log"
echo "starting update env: $STATARB_ENV ..." > "$LOG_DIR"/update.log

download_failed=false
bars_failed=false

CONFIG=""
if [ "$STATARB_ENV" = "dev" ]; then
  CONFIG="--config=$ROOT_DIR/src/config/config_dev.json"
fi

echo "Using config [$CONFIG]" >> "$LOG_DIR"/update.log

# download and generate tardis bars
if [ "$STATARB_ENV" = "prod" ]; then
  # Wait for Tardis data to be ready (polls API until yesterday's data is available)
  # Max wait is 6 hours, so if started at 1 AM, will timeout by 7 AM and continue anyway
  echo "Waiting for Tardis data to be ready... $(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_DIR"/update.log
  python "$SRC_DIR"/wait_for_tardis_ready.py >> "$LOG_DIR"/update.log 2>&1 || {
    echo "WARNING: Tardis wait script timed out or failed, proceeding anyway... $(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_DIR"/update.log
  }
  echo "Starting tardis download... $(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_DIR"/update.log
  python "$SRC_DIR"/download_tardis.py --update --s3 $CONFIG || download_failed=true
fi
python "$SRC_DIR"/generate_bars.py --update --force $CONFIG || bars_failed=true

{
  python "$SRC_DIR"/process_news_files.py
  python "$SRC_DIR"/lib/news/process_new_news_files.py

  echo "bars data generation" >> "$LOG_DIR"/update.log
  python "$SRC_DIR"/generate_features.py --update $CONFIG
  python "$SRC_DIR"/generate_universe.py --update --prod $CONFIG

  # Run forwards and models in parallel
  python "$SRC_DIR"/generate_forwards.py --update $CONFIG &
  forwards_pid=$!
  python "$SRC_DIR"/generate_models.py --update $CONFIG &
  models_pid=$!
  wait $forwards_pid $models_pid
} >> "$LOG_DIR"/update.log 2>&1

# generate prod_fits and prod_alpha for prod only
if [ "$STATARB_ENV" = "prod" ]; then
  python "$SRC_DIR"/generate_fits.py --update --prod >> "$LOG_DIR"/update.log 2>&1

  # Start server restart in background (can run while alphas generate)
  echo "Restarting server..." >> "$LOG_DIR"/update.log
  restart_service "server" 60 "$SERVER_UPDATE_FILE" "$SERVER_KILL_FILE" -- "$SRC_DIR/bin/launch_server.sh" -t &
  server_restart_pid=$!

  python "$SRC_DIR"/generate_alphas.py --update --prod >> "$LOG_DIR"/update.log 2>&1

  # Wait for server restart to complete
  wait $server_restart_pid

  # Restart trader after server (uses same kill file pattern)
  echo "Restarting trader..." >> "$LOG_DIR"/update.log
  restart_service "trader" 60 "$TRADER_KILL_FILE" -- "$SRC_DIR/bin/launch_trader.sh"

  echo "Restarting reports..." >> "$LOG_DIR"/update.log
  touch "$TRADING_DIR/reports.kill"
  sleep 160
  rm -f "$TRADING_DIR/reports.kill"

  nohup "$SRC_DIR/bin/launch_reports.sh" trading >> "$LOG_DIR"/update.log 2>&1 &
  nohup "$SRC_DIR/bin/launch_reports.sh" historical >> "$LOG_DIR"/update.log 2>&1 &
  nohup "$SRC_DIR/bin/launch_reports.sh" slippage >> "$LOG_DIR"/update.log 2>&1 &
  # sim report removed - will launch after sims complete
  nohup "$SRC_DIR/bin/launch_reports.sh" fits >> "$LOG_DIR"/update.log 2>&1 &
  nohup "$SRC_DIR/bin/launch_reports.sh" universe >> "$LOG_DIR"/update.log 2>&1 &
  nohup "$SRC_DIR/bin/launch_reports.sh" alpha >> "$LOG_DIR"/update.log 2>&1 &
  nohup "$SRC_DIR/bin/launch_reports.sh" execution >> "$LOG_DIR"/update.log 2>&1 &
fi

echo "updating non-prod alphas and fits, put it here so not block server restart... $(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_DIR"/update.log
{
  # already log the bars case when we generate prod data
  python "$SRC_DIR"/generate_fits.py --update $CONFIG
  python "$SRC_DIR"/generate_alphas.py --update $CONFIG
} >> "$LOG_DIR"/update.log 2>&1

if [ "$STATARB_ENV" = "prod" ]; then
  . $SRC_DIR/bin/sim_comparison.sh 7
  . $SRC_DIR/bin/sim_comparison.sh 1
  . $SRC_DIR/bin/sim_longterm.sh

  echo "Launching sim report after sims complete... $(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_DIR"/update.log
  nohup "$SRC_DIR/bin/launch_reports.sh" sim >> "$LOG_DIR"/update.log 2>&1 &
  nohup "$SRC_DIR/bin/launch_reports.sh" hist_sim >> "$LOG_DIR"/update.log 2>&1 &
fi

echo "Done daily update.." >> "$LOG_DIR"/update.log
