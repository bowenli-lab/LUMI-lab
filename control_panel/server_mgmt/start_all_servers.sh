#!/bin/bash

# Config file path
CONFIG_FILE="servers.conf"

# Log file for errors
LOG_FILE="error_log.txt"

# Initialize log file
echo "Error Log - $(date)" > $LOG_FILE

# Read the config file and process each line
# shellcheck disable=SC2095
while IFS='|' read -r server scripts
do
  # Skip empty lines or lines starting with #
  [[ -z "$server" || "$server" =~ ^# ]] && continue

  # Split the scripts into an array
  IFS='|' read -r -a script_array <<< "$scripts"

  echo "Connecting to $server"
  for script_path in "${script_array[@]}"
  do
    echo "Executing $script_path on $server"
    ssh ${server} 'bash -s ${script_path}'

    if [ $? -eq 0 ]; then
      echo "Successfully executed $script_path on $server"
    else
      echo "Failed to execute $script_path on $server" | tee -a $LOG_FILE
    fi
  done
done < $CONFIG_FILE
