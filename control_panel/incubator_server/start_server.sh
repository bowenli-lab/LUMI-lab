# kill the server if it is already running
lsof -t -i:9007 | xargs kill -9
# start the server
cd /home/pi/sdl-dev/incubator_server
nohup uvicorn incubator_server:app --host raspberrypi2feeder.local --port 9007 --reload > incubator_server.log 2>&1 &
echo "Incubator server started"
