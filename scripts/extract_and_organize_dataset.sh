#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/../data"

# Function to extract a tar.gz file while removing the top-level directory
extract_tar() {
    local tar_file="$1"
    local target_dir="$2"
    echo "Extracting ${tar_file} into ${target_dir}"
    mkdir -p "${target_dir}"
    tar --strip-components=1 -xzf "${tar_file}" -C "${target_dir}"
}

# Create the directory structure and extract files
echo "Creating directory structure and extracting files..."

# Extract EXTRACTED_IMGS_
if [ -f "${BASE_DIR}/EXTRACTED_IMGS_.tar.gz" ]; then
    extract_tar "${BASE_DIR}/EXTRACTED_IMGS_.tar.gz" "${BASE_DIR}/EXTRACTED_IMGS_"
else
    echo "EXTRACTED_IMGS_.tar.gz not found."
fi

# Extract gpt4v_embeddings
if [ -f "${BASE_DIR}/gpt4v_embeddings.tar.gz" ]; then
    extract_tar "${BASE_DIR}/gpt4v_embeddings.tar.gz" "${BASE_DIR}/gpt4v_embeddings"
else
    echo "gpt4v_embeddings.tar.gz not found."
fi

# Extract gpt3_embeddings into hm3d
if [ -f "${BASE_DIR}/gpt3_embeddings.tar.gz" ]; then
    extract_tar "${BASE_DIR}/gpt3_embeddings.tar.gz" "${BASE_DIR}/hm3d/gpt3_embeddings"
else
    echo "gpt3_embeddings.tar.gz not found."
fi

# Extract gpt4v_embeddings_hm3d into hm3d
if [ -f "${BASE_DIR}/gpt4v_embeddings_hm3d.tar.gz" ]; then
    extract_tar "${BASE_DIR}/gpt4v_embeddings_hm3d.tar.gz" "${BASE_DIR}/hm3d/gpt4v_embeddings"
else
    echo "gpt4v_embeddings_hm3d.tar.gz not found."
fi

if [ -f "${BASE_DIR}/hm3d_database.json" ]; then
    mkdir -p "${BASE_DIR}/hm3d/hm3d_dataset"
    echo "Moving hm3d_database.json"
    mv "${BASE_DIR}/hm3d_database.json" "${BASE_DIR}/hm3d/hm3d_dataset/"
else
    echo "hm3d_database.json not found."
fi

if [ -f "${BASE_DIR}/ver.3.tar.gz" ]; then
    extract_tar "${BASE_DIR}/ver.3.tar.gz" "${BASE_DIR}/hm3d/ver.3"
else
    echo "ver.3.tar.gz not found."
fi

if [ -f "${BASE_DIR}/ver.4.tar.gz" ]; then
    extract_tar "${BASE_DIR}/ver.4.tar.gz" "${BASE_DIR}/hm3d/ver.4"
else
    echo "ver.4.tar.gz not found."
fi

if [ -f "${BASE_DIR}/ltrpo_database.json" ]; then
    mkdir -p "${BASE_DIR}/ltrpo_dataset"
    echo "Moving ltrpo_database.json"
    mv "${BASE_DIR}/ltrpo_database.json" "${BASE_DIR}/ltrpo_dataset/"
else
    echo "ltrpo_database.json not found."
fi

echo "All files have been extracted and organized."

