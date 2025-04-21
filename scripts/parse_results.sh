#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide an ID argument. Usage: ./extract_recall_metrics.sh --id100"
    exit 1
fi

# Extract the ID from the argument
id=${1#--id}

# Set the path to the log directory
log_dir="../log"

# Function to extract values from a line
extract_values() {
    echo "$1" | awk -F'[=,]' '{print $8 ", " $9 ", " $10}'
}

# Process each log file in the log directory
for file in "$log_dir"/id${id}*.log; do
    if [ -f "$file" ]; then
        # Extract the lines containing the metrics
        mp3d_line=$(grep "mrr, recall1, recall5, recall10, recall20 =" "$file" | head -n 1)
        hm3d_line=$(grep "mrr, recall1, recall5, recall10, recall20 =" "$file" | tail -n 1)

        # Extract the required values
        mp3d_values=$(extract_values "$mp3d_line")
        hm3d_values=$(extract_values "$hm3d_line")

        # Print the results without the file name
        echo "$mp3d_values, $hm3d_values"
    fi
done
