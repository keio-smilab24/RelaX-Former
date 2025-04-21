#!/bin/bash

# Define the exclusion list
exclude_list=(
    "TEEsavR23oF-W7k2QWzBrFY-7UrtFsADwob-bxsVRursffK"
    "66seV3BWPoX-L53DsRRk4Ch-HaxA7YrQdEC-yr17PDCnDDW-mma8eWq3nNQ-XNeHsjL6nBB"
    "4ok3usBNeis-rsggHU7g7dh-cvZr5TUy5C5"
    "c5eTyR3Rxyh-LT9Jq6dN3Ea-h1zeeAwLh9Z-7MXmsvcQjpJ-q5QZSEeHe5g-BAbdmeyTvMZ"
    "Qpor2mEya8F-cYkrGrCg2kB-7Ukhou1GxYi"
    "58NLZxWBSpk-MHPLjHsuG27-bzCsHPLDztK-5cdEh9F2hJL-hyFzGGJCSYs-u1bkiGRVyu9"
    "nrA1tAA17Yp-LEFTm3JecaC-fsQtJ8t3nTf"
    "5jp3fCRSRjc-k1cupFYWXJ6-svBbv1Pavdk-F7EAMsdDASd-kJJyRFXVpx2"
    "XMHNu9rRQ1y-BFRyYbPCCPE-AWUFxHEyV3T-z9YwN9M8FpG"
    "VBzV5z6i1WS-7GAhQPFzMot-3t8DB4Uzvkt-y9hTuugGdiq-tQ5s4ShP627"
    "mL8ThkuaVTM-CrMo8WxCyVb-X7gTkoDHViv"
    "GLAQ4DNUx5U-QHhQZWdMpGJ-T6nG3E2Uui9"
    "FnSn2KSrALj-bCPU9suPUw9"
    "6D36GQHuP8H-YRUkbU5xsYj-LNg5mXe1BDj-Nfvxx8J5NCo-HMkoS756sz6-SByzJLxpRGn"
    "wcojb4TFT35-ziup5kvtCCR-a8BtkwhxdRV-dHwjuKfkRUR-jgPBycuV1Jq-6s7QHgap2fW"
    "rXXL6twQiWc-hkr2MGpHD6B-q3hn1WQ12rz-u8ug2rtNARf-uSKXQ5fFg6u-DYehNKdT76V"
    "eF36g7L6Z9M-XB4GS9ShBRE-QaLdnwvtxbs-vBMLrTe4uLA"
    "uLz9jNga3kC-vd3HHTEpmyA-dVW2D7TDctW-X4qjx5vquwH-rJhMRvNn4DS"
)

# Extract individual environment values
environments=()
for item in "${exclude_list[@]}"; do
    IFS='-' read -ra ADDR <<< "$item"
    for i in "${ADDR[@]}"; do
        environments+=("$i")
    done
done

# Convert array to jq filter
jq_filter=$(printf " and \$env != \"%s\"" "${environments[@]}")
jq_filter=${jq_filter:5} # Remove leading " and "

# Run jq command with dynamic filter
jq "[.[] | select(.environment as \$env | $jq_filter) | .instruction] | unique | length" hm3d_database.json
