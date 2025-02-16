from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from clamp_api import clamp_api

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


@app.get("/clamp", response_model=MessageResponse)
async def clamp():
    try:
        msg = clamp_api.clamp()
        return MessageResponse(message=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/release", response_model=MessageResponse)
async def release():
    try:
        msg = clamp_api.release()
        return MessageResponse(message=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reset_clamped", response_model=MessageResponse)
async def reset_clamped():
    try:
        msg = clamp_api.reset_clamped()
        return MessageResponse(message=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/move_in/{steps}", response_model=MessageResponse)
async def move_in(steps: int):
    try:
        clamp_api.move_in(steps)
        return MessageResponse(message=f"Clamp moved in {steps} steps")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/move_out/{steps}", response_model=MessageResponse)
async def move_out(steps: int):
    try:
        clamp_api.move_out(steps)
        return MessageResponse(message=f"Clamp moved in {steps} steps")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    clamp_api.shutdown()


@app.get("/health", response_model=HealthMessage)
async def health_check():
    return HealthMessage(message="Server is healthy", is_healthy=True)
