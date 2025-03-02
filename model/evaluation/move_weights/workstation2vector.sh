#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <name-of-the-run>"
    exit 1
fi

# Assign the first argument to the REMOTE_DIR variable
MODEL_DIR="$1"
REMOTE_DIR="/home/sdl/3d_molecule_save"


# Define remote server details
REMOTE_USER="sdl"
REMOTE_HOST="3.144.192.155"


# Define local directory
LOCAL_DIR="/scratch/ssd004/datasets/cellxgene/3d_molecule_save/${MODEL_DIR}"

mkdir -p ${LOCAL_DIR}

# Construct the full remote directory path
REMOTE_PATH="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${MODEL_DIR}/*_best.pt"

# Copy the directory from the remote server to the local directory
scp -P 6000 -r ${REMOTE_PATH} ${LOCAL_DIR}

# Check if the copy was successful
if [ $? -eq 0 ]; then
    echo "Directory copied successfully to ${LOCAL_DIR}"
else
    echo "Directory copy failed"
fi
