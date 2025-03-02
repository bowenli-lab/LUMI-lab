#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <remote-path> <local-path>"
    exit 1
fi

# Assign the first argument to the REMOTE_DIR variable
LOCAL_PATH="$2"
REMOTE_PATH="$1"



# Define remote server details
REMOTE_USER="sdl"
REMOTE_HOST="3.144.192.155"


# Define local directory

# Construct the full remote directory path
REMOTE_FULL_PATH="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

# Copy the directory from the remote server to the local directory
scp -P 6000 -r ${REMOTE_FULL_PATH} ${LOCAL_PATH}

# Check if the copy was successful
if [ $? -eq 0 ]; then
    echo "file copied successfully to ${LOCAL_PATH}"
else
    echo "file copy failed"
fi
