# kill the server
lsof -ti:8000 | xargs kill

# start the server
cd /home/pi/sdl-dev/liquid_sampler_server
nohup uvicorn liquid_sampler_server:app --host raspberrypi.local --port 8000 --reload 2>&1 > ./server.log &
