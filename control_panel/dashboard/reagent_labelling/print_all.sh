#!/bin/bash

# Loop through rows A to H
for row in {A..H}; do
    # Loop through columns 1 to 12
    for col in {1..12}; do
        # Generate the label name
        label="${row}${col}"
        echo "Printing label: ${label}"
        # Call the bluetooth_label.sh script with the generated label
        ./print_label.sh "${label}"
        #wait 10s
        sleep 10
    done
done
