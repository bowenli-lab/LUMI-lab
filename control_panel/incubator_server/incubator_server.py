from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from incubator_api import incubator_api

app = FastAPI()


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


@app.get("/", response_model=MessageResponse)
async def root():
    return MessageResponse(message="Hello World")


@app.get("/health", response_model=HealthMessage)
async def health_check():
    return HealthMessage(message="Server is healthy", is_healthy=True)


@app.get("/open_incubator", response_model=MessageResponse)
async def open_incubator():
    try:
        response = incubator_api.open_incubator()
        return MessageResponse(message=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/close_incubator", response_model=MessageResponse)
async def close_incubator():
    try:
        response = incubator_api.close_incubator()
        return MessageResponse(message=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/move_plate_in/{duration}", response_model=MessageResponse)
async def move_plate_in(duration: float):
    try:
        incubator_api.move_plate_in(duration)
        return MessageResponse(
            message=f"Moving plate in for {duration} seconds")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/move_plate_out/{duration}", response_model=MessageResponse)
async def move_plate_out(duration: float):
    try:
        incubator_api.move_plate_out(duration)
        return MessageResponse(
            message=f"Moving plate out for {duration} seconds")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/incubator_plate_in", response_model=MessageResponse)
async def incubator_plate_in():
    try:
        incubator_api.incubator_plate_in()
        return MessageResponse(message="Plate moved in")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/incubator_plate_out", response_model=MessageResponse)
async def incubator_plate_out():
    try:
        incubator_api.incubator_plate_out()
        return MessageResponse(message="Plate moved out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/move_forward/{duration}", response_model=MessageResponse)
async def move_forward(duration: float):
    try:
        incubator_api.door_forward(duration)
        return MessageResponse(message=f"Moving forward for {duration} seconds")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/move_backward/{duration}", response_model=MessageResponse)
async def move_backward(duration: float):
    try:
        incubator_api.door_backward(duration)
        return MessageResponse(
            message=f"Moving backward for {duration} seconds")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/break", response_model=MessageResponse)
async def shutdown():
    try:
        incubator_api.shutdown_relay()
        return MessageResponse(message="Incubator shut down")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    incubator_api.shutdown()
