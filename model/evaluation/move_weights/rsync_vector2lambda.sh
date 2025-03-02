#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_file> <destination_directory>"
    exit 1
fi

# Variables
SOURCE_FILE=$1
DEST_PATH=$2
DEST_HOST="209.20.158.223"
DEST_USER="ubuntu"
IDENTITY_FILE="~/.ssh/lambda-key.pem"
CHUNK_SIZE=2000M

# Split the file into chunks
split -b $CHUNK_SIZE $SOURCE_FILE ${SOURCE_FILE}.part

# Get the list of chunk files
CHUNKS=(${SOURCE_FILE}.part*)

# Create a remote directory for the chunks
ssh -i $IDENTITY_FILE $DEST_USER@$DEST_HOST "mkdir -p $DEST_PATH/tmp_chunks"

# Function to transfer a chunk
transfer_chunk() {
    local CHUNK_FILE=$1
    rsync -avz -e "ssh -i $IDENTITY_FILE" $CHUNK_FILE $DEST_USER@$DEST_HOST:$DEST_PATH/tmp_chunks/
    if [ $? -eq 0 ]; then
        echo "Chunk $CHUNK_FILE transferred successfully."
    else
        echo "Failed to transfer chunk $CHUNK_FILE."
    fi
}

# Transfer chunks concurrently
for CHUNK in "${CHUNKS[@]}"; do
    transfer_chunk $CHUNK &
done

# Wait for all transfers to complete
wait

# Reassemble the file on the destination
ssh -i $IDENTITY_FILE $DEST_USER@$DEST_HOST "cat $DEST_PATH/tmp_chunks/${SOURCE_FILE}.part* > $DEST_PATH/$(basename $SOURCE_FILE)"

# Cleanup the chunk files
ssh -i $IDENTITY_FILE $DEST_USER@$DEST_HOST "rm -rf $DEST_PATH/tmp_chunks"

# Cleanup local chunk files
rm ${SOURCE_FILE}.part*

echo "File transfer complete."