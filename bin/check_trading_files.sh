#!/bin/bash
. $ROOT_DIR/src/bin/include.sh
init_env

cd $ROOT_DIR

yesterday=$(date -d "yesterday" +%Y%m%d)

echo -e "\nAlpha files check"
find -L ./data/prod_alpha -name "*.parquet" | sort -r | awk -F'[/_.]' '
{
    # Extract the type (first two parts of the filename)
    type = $6 "_" $7
    # Extract the date (last part before .parquet)
    date = $8
    
    # If we haven'"'"'t seen this type before or if this date is later, update
    if (!(type in latest) || date > latest[type]) {
        latest[type] = date
        latest_file[type] = $0
    }
}
END {
    # Print the results
    for (type in latest) {
        print "Type: " type ", Latest Date: " latest[type] ", File: " latest_file[type]
    }
}'

latest_date=$(find -L ./data/prod_alpha -name "*.parquet" | awk -F'[/_.]' '{print $8}' | sort -nr | head -1)
if [ "$latest_date" != "$yesterday" ]; then
    echo "Not see ${yesterday} alpha only with latest data from ${latest_date}"
    python "$SRC_DIR"/lib/opsgenie.py --key "Not see ${yesterday} alpha only with latest data from ${latest_date}"
fi

echo -e "\nBar files check"
find -L ./data/bars -name "bars_*.parquet" | sort -r | awk -F'[/_.]' '
{
    # Extract the type (the number after "bars_")
    type = $7
    # Extract the date (last part before .parquet)
    date = $8
    
    # If we haven'"'"'t seen this type before or if this date is later, update
    if (!(type in latest) || date > latest[type]) {
        latest[type] = date
        latest_file[type] = $0
    }
}
END {
    # Print the results
    for (type in latest) {
        print "Type: bars_" type ", Latest Date: " latest[type] ", File: " latest_file[type]
    }
}'

latest_bar_date=$(find -L ./data/bars -name "bars_*.parquet" | awk -F'[/_.]' '{print $8}' | sort -nr | head -1)
if [ "$latest_bar_date" != "$yesterday" ]; then
    echo "Not see ${yesterday} bars only with latest data from ${latest_bar_date}"
    python "$SRC_DIR"/opsgenie_alert.py --key "Not see ${yesterday} bars only with latest data from ${latest_bar_date}"
fi

echo -e "\nFeature files check"
find -L ./data/features -name "features_*.parquet" | sort -r | awk -F'[/_.]' '
{
    # Extract the type (the number after "features_")
    type = $6
    # Extract the date (last part before .parquet)
    date = $7
    
    # If we haven'"'"'t seen this type before or if this date is later, update
    if (!(type in latest) || date > latest[type]) {
        latest[type] = date
        latest_file[type] = $0
    }
}
END {
    # Print the results
    for (type in latest) {
        print "Type: features_" type ", Latest Date: " latest[type] ", File: " latest_file[type]
    }
}'

latest_feature_date=$(find -L ./data/features -name "features_*.parquet" | awk -F'[/_.]' '{print $7}' | sort -nr | head -1)
if [ "$latest_feature_date" != "$yesterday" ]; then
    echo "Not see ${yesterday} features only with latest data from ${latest_feature_date}"
    python "$SRC_DIR"/opsgenie_alert.py --key "Not see ${yesterday} features only with latest data from ${latest_feature_date}"
fi


echo -e "\nForwards files check"
find -L ./data/forwards -name "forward*.parquet" | sort -r | awk -F'[/_.]' '
{
    # Extract the type (after "forward" or "forwards")
    type = $(NF-2)
    # Extract the date (last field before .parquet)
    date = $(NF-1)
    
    # If we haven'"'"'t seen this type before or if this date is later, update
    if (!(type in latest) || date > latest[type]) {
        latest[type] = date
        latest_file[type] = $0
    }
}
END {
    # Print the results
    for (type in latest) {
        print "Type: forward " type ", Latest Date: " latest[type] ", File: " latest_file[type]
    }
}'

echo -e "\nFits files check"
find -L ./data/fits/prod -name "fits.*.prod.csv" | sort -r | awk -F'[/.]' '
{
    # Extract the type (after "fits.")
    type = $6
    # Extract the date (before ".prod")
    date = $7
    
    # If we haven'"'"'t seen this type before or if this date is later, update
    if (!(type in latest) || date > latest[type]) {
        latest[type] = date
        latest_file[type] = $0
    }
}
END {
    # Print the results
    for (type in latest) {
        print "Type: fits." type ", Latest Date: " latest[type] ", File: " latest_file[type]
    }
}'
# Find the most recent configuration file
latest_config=$(ls ./data/fits/prod/config* | grep -E '[0-9]{8}' | sort -t. -k3 -n | tail -n 1)

# Check if a config was found
if [ -n "$latest_config" ]; then
    echo "$latest_config"
fi

latest_fits_date=$(find -L ./data/fits/prod -name "fits.*.prod.csv" | awk -F'[/.]' '{print $7}' | sort -nr | head -1)
if [[ "$latest_fits_date" <= "$yesterday" ]]; then
    echo "Not see ${yesterday} fits only with latest data from ${latest_fits_date}"
    python "$SRC_DIR"/opsgenie_alert.py --key "Not see ${yesterday} fits only with latest data from ${latest_fits_date}"
fi

echo -e "\nModel files check"
find -L ./data/models -name "*.parquet" | sort -r | awk -F'[/_]' '
{
    # Extract the type (everything before the second underscore)
    type = $4 "_" $5
    # Extract the date (after the second underscore)
    date = $6
    # Remove the .parquet extension from the date
    gsub(".parquet", "", date)
    
    # If we haven'"'"'t seen this type before or if this date is later, update
    if (!(type in latest) || date > latest[type]) {
        latest[type] = date
        latest_file[type] = $0
    }
}
END {
    # Print the results
    for (type in latest) {
        print "Type: " type ", Latest Date: " latest[type] ", File: " latest_file[type]
    }
}'

latest_model_date=$(find -L ./data/models -name "*.parquet" | awk -F'[/_]' '{gsub(".parquet", "", $6); print $6}' | sort -nr | head -1)
if [ "$latest_model_date" != "$yesterday" ]; then
    echo "Not see ${yesterday} models only with latest data from ${latest_model_date}"
    python "$SRC_DIR"/opsgenie_alert.py --key "Not see ${yesterday} models only with latest data from ${latest_model_date}"
fi

echo -e "\nNews files check"
# Find the most recent news file
latest_news=$(ls ./data/news/news.*.csv | sort -t. -k2 -n | tail -n 1)

# Check if a file was found
if [ -n "$latest_news" ]; then
    echo "$latest_news"
fi

echo -e "\nLive files check"
today=$(date +%Y%m%d)
# Count the files for today and store the result in a variable
file_count=$(ls ./data/live/*$today -ahl 2>/dev/null | wc -l)

# Echo the result
echo "Number of live files matching today's date ($today): $file_count"
