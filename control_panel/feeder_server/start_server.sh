#!/bin/bash
cd /home/pi/sdl-dev/feeder_server
export DEVICE_CODE=0
nohup uvicorn feeder_server:app --host raspberrypi2feeder.local --port 8000 --log-config logging_configs/log_config0.yaml --workers 2 &
export DEVICE_CODE=1
nohup uvicorn feeder_server:app --host raspberrypi2feeder.local --port 9001 --log-config logging_configs/log_config1.yaml --workers 2 &
export DEVICE_CODE=2
nohup uvicorn feeder_server:app --host raspberrypi2feeder.local --port 9002 --log-config logging_configs/log_config2.yaml &
export DEVICE_CODE=3
nohup uvicorn feeder_server:app --host raspberrypi2feeder.local --port 9003 --log-config logging_configs/log_config3.yaml &
