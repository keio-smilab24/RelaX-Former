#!/bin/bash
# author: daichiyashima
# date: 2024-03-30
# usage: ./cherrypick.sh 'path/to/json_file.json' 'log/output.log'

json_file=$1
log_file=$2

# change all to any to get groups with at least one rank > n and < m
jq_script='
    .[] |                       
    group_by(.instruction_id) |
    map(                      
        select(              
            all(.[];        
            .ranks[] | tonumber > 10 
            )
        )
    ) | map(select(. != null)) | {results: add} 
'

jq "$jq_script" "$json_file" |& tee "$log_file"
