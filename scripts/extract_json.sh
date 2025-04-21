#!/bin/bash

# Usage: ./extract_json.sh output.jsonl target.json extracted.json

output_jsonl=$1
target_json=$2
extracted_json=$3

# Extract the instruction_ids from the output JSONL and flatten into a single array
instruction_ids=$(jq -s '[.[] | .results[] | .instruction_id] | unique' "$output_jsonl")

# Check if the instruction_ids extraction was successful
if [ -z "$instruction_ids" ]; then
    echo "Error: Could not extract instruction IDs from the output JSONL."
    exit 1
fi

# Extract objects from the target JSON with the same instruction_ids
jq --argjson ids "$instruction_ids" '
    .[] | 
    select(type == "object" and has("instruction_id") and (.instruction_id as $id | $ids | index($id)))
' "$target_json" | jq -s . > "$extracted_json"

echo "Extraction completed. Results saved to $extracted_json."

