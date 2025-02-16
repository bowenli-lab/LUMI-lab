#!/bin/bash
nohup uvicorn clamp_server:app --host raspberrypi2feeder.local --port 9006 &
