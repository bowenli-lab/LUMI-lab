@echo off
start /B uvicorn plate_reader_server:app --host 192.168.1.254 --port 5000 >> output.log 2>&1