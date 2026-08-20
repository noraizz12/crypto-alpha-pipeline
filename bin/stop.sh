#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

remove_lock_file() {
    local lock_file=""
    case "$1" in
        trader)
            lock_file="$TRADING_DIR/trader.lock"
            ;;
        server)
            lock_file="$TRADING_DIR/server.lock"
            ;;
        news)
            lock_file="$TRADING_DIR/news.lock"
            ;;
        binance_pnl)
            lock_file="$TRADING_DIR/binance.lock"
            ;;
        trading_reports|historical_reports|slippage_reports|fits_reports|sim_reports)
            lock_file="$TRADING_DIR/$1.lock"
            ;;
        *)
            echo "No specific lock file for parameter: $1"
            return 0
            ;;
    esac

    if [ -n "$lock_file" ] && [ -f "$lock_file" ]; then
        echo "Removing lock file: $lock_file"
        rm -f "$lock_file"
    elif [ -n "$lock_file" ]; then
        echo "Lock file not found: $lock_file"
    fi
}

if [ -z "$1" ]; then
    echo "ERROR must specify param to match"
else
    if [ -z "$2" ]; then
        pid=$(ps aux | grep python | grep "$1" | awk '{print $2}')        
    else
        pid=$(ps aux | grep python | grep "$1" | grep "$2" | awk '{print $2}')
    fi
    kill -2 "$pid"
    start=$SECONDS
    end=$SECONDS
    while (( end - start < 90 )); do
        sleep 1
        if ! ps -p "$pid" > /dev/null; then
            echo "Process $pid stopped successfully"
            remove_lock_file "$1"
            exit 0
        fi
        end=$SECONDS
    done 
    kill -9 "$pid"
    if ! ps -p "$pid" > /dev/null; then
        echo "Process $pid forcefully stopped"
        remove_lock_file "$1"
        exit 0
    fi
fi
