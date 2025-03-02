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

# Copy the file
scp -i $IDENTITY_FILE -r $SOURCE_FILE $DEST_USER@$DEST_HOST:$DEST_PATH

if [ $? -eq 0 ]; then
    echo "File copied successfully to $DEST_HOST:$DEST_PATH"
else
    echo "Failed to copy the file"
fi

# /scratch/ssd004/datasets/cellxgene/3d_molecule_data/15m-lib
# /home/ubuntu/utah/3d_molecule_data