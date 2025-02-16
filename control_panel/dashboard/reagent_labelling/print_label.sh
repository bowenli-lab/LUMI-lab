#!/bin/bash

# Usage: ./bluetooth_label.sh <output_file_name>

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <output_file_name>"
    exit 1
fi
#remove the extension
output_file=$(echo "$1" | cut -f 1 -d '.')

# Convert the argument to uppercase
output_file=$(echo "$1" | tr '[:lower:]' '[:upper:]')

# Add .png extension if not present
if [[ "$output_file" != *.png ]]; then
    output_file="${output_file}.png"
fi

# Check if the file exists
if [ ! -f "label_output/${output_file}" ]; then
    echo "Error: File 'label_output/${output_file}' does not exist."
    exit 1
fi

BLUETOOTH_MAC="02:13:01:6A:10:34"
python3.11 -m niimprint -m d110 -c bluetooth -a ${BLUETOOTH_MAC} -i label_output/${output_file}
