# auto restart by killing the old process
kill -9 $(lsof -t -i:12001)
# start the server
nohup python ./sdl_server.py 2>&1 > ./sdl_server.log &
# lsof -i :12001

# sleep for 3 seconds
sleep 7

# Check if the file exists
if [ -f "experiment_input.json" ]; then
    echo "AUTO PROPOSE experiment_input.json"

    # Use curl to send a POST request
    curl -X POST -H "Content-Type: application/json" -d @experiment_input.json http://192.168.1.11:12001/propose_job
fi
