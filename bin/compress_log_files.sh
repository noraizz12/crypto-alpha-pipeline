#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

# Function to compress log files with datetime strings
compress_logs() {
    if [ ! -d "$LOG_DIR" ]; then
        echo "LOG_DIR $LOG_DIR does not exist"
        return 1
    fi
    
    cd "$LOG_DIR"
    
    # Get today's and yesterday's date in YYYYMMDD format
    today=$(date +%Y%m%d)
    yesterday=$(date -d "yesterday" +%Y%m%d)
    
    # Find and compress log files with pattern *.YYYYMMDD_HHSS.log
    find . -name "*.log" -type f | grep -E '\.[0-9]{8}_[0-9]{4}\.log$' | while read -r logfile; do
        if [ -f "$logfile" ]; then
            # Extract date from filename
            file_date=$(echo "$logfile" | grep -oE '[0-9]{8}' | tail -1)
            
            # Skip if file date is today or yesterday
            if [ "$file_date" = "$today" ] || [ "$file_date" = "$yesterday" ]; then
                echo "Skipping recent file: $logfile (date: $file_date)"
                continue
            fi
            
            gzip "$logfile"
            echo "Compressed: $logfile"
        fi
    done
}

# Run compression
compress_logs