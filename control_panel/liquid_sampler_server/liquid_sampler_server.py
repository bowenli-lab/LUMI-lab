from typing import List
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from liquid_sampler_api import liquid_sampler_api

app = FastAPI()


def check_i2cdetect():
    # Expected hex addresses from 60 to 77
    expected_addresses = {f'{i:02x}' for i in range(0x60, 0x78)}

    # Run the i2cdetect command
    result = subprocess.run(['i2cdetect', '-y', '1'], capture_output=True,
                            text=True)
    cmd_output = result.stdout.strip()

    # Extract detected addresses from the command output
    detected_addresses = set()
    for line in cmd_output.splitlines()[1:]:
        parts = line.split()
        detected_addresses.update(
            parts[1:])  # Skip the first element which is the row label

    # Check if all expected addresses are in the detected addresses
    return expected_addresses.issubset(detected_addresses)


class MessageResponse(BaseModel):
    """
    A class to represent the response message from the server.
    """
    message: str


class HealthMessage(BaseModel):
    """
    A class to represent the response message from the server.
    """
    message: str
    is_healthy: bool


class SingleMotor(BaseModel):
    """
    A class to represent the single motor request.
    """
    motor_number: str
    volume: str


class PumpBatch(BaseModel):
    """
    A class to represent the batch pump request.
    """
    motor_numbers: List[str]
    volumes: List[str]


@app.get("/", response_model=MessageResponse)
async def root():
    return MessageResponse(message="Hello World")


@app.get("/plate_in", response_model=MessageResponse)
async def plate_in():
    try:
        liquid_sampler_api.plate_in()
        return MessageResponse(message="Plate moved in")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthMessage)
async def health_check():
    if not check_i2cdetect():
        return HealthMessage(message="I2C bus is not healthy", is_healthy=False)
    return HealthMessage(message="Server is healthy", is_healthy=True)


@app.get("/plate_out", response_model=MessageResponse)
async def plate_out():
    try:
        liquid_sampler_api.plate_out()
        return MessageResponse(message="Plate moved out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pump", response_model=MessageResponse)
async def pump(pump_single: SingleMotor):
    try:
        liquid_sampler_api.pump(int(pump_single.motor_number),
                                float(pump_single.volume))
        return MessageResponse(
            message=f"Pumping liquid at motor {pump_single.motor_number}"
                    f" at amount {pump_single.volume}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/aspirate", response_model=MessageResponse)
async def aspirate(pump_single: SingleMotor):
    try:
        liquid_sampler_api.aspirate(int(pump_single.motor_number),
                                    float(pump_single.volume))
        return MessageResponse(
            message=f"aspirating liquid at motor {pump_single.motor_number} at "
                    f"amount {pump_single.volume}")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_pump", response_model=MessageResponse)
async def batch_pump(pump_batch: PumpBatch):
    try:
        motor_numbers = [int(motor_number) for motor_number in
                         pump_batch.motor_numbers]
        volumes = [float(volume) for volume in pump_batch.volumes]
        liquid_sampler_api.batch_pump(motor_numbers, volumes)
        return MessageResponse(
            message=f"Pumping liquid at motors {pump_batch.motor_numbers} at "
                    f"amounts {pump_batch.volumes}")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_aspirate", response_model=MessageResponse)
async def batch_aspirate(pump_batch: PumpBatch):
    try:
        motor_numbers = [int(motor_number) for motor_number in
                         pump_batch.motor_numbers]
        volumes = [float(volume) for volume in pump_batch.volumes]
        liquid_sampler_api.batch_aspirate(motor_numbers, volumes)
        return MessageResponse(
            message=f"aspirating liquid at motors {pump_batch.motor_numbers} at"
                    f"amounts {pump_batch.volumes}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/calibrate", response_model=MessageResponse)
async def calibrate():
    try:
        liquid_sampler_api.calibrate()
        return MessageResponse(message="Calibrating motors")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stop_all", response_model=MessageResponse)
async def stop_all():
    try:
        liquid_sampler_api.stop_all()
        return MessageResponse(message="Stopping all motors")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    liquid_sampler_api.shutdown()
