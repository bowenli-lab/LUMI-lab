import asyncio
import os

from pydantic import BaseModel

from repeat_every_time import repeat_every
from feeder_api import FeederAPI
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from threading import Lock
import logging

DEVICE_CODE = os.getenv("DEVICE_CODE")
if DEVICE_CODE is None:
    raise ValueError("DEVICE_CODE is not set in the environment variables.")
DEVICE_CODE = int(DEVICE_CODE)
app = FastAPI()
feeder_api = FeederAPI(device_code=DEVICE_CODE)

# set up block variable
app.is_blocking = False
app.blocking_lock = Lock()
# app.auto_feed = True
app.auto_feed = False  # debug mode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthMessage(BaseModel):
    """
    A class to represent the response message from the server.
    """
    message: str
    is_healthy: bool


@app.get("/health", response_model=HealthMessage)
async def health_check():
    return HealthMessage(message="Server is healthy", is_healthy=True)

@app.get("/")
async def root():
    return {"message": "Hello World!!"}


@app.get("/plate_pos")
async def plate_pos():
    return {"plate_position": feeder_api.plate_position}


@app.get("/move_bottom")
async def move_bottom():
    """
    This method moves the plate to the bottom.
    """
    async with app.blocking_lock:
        app.is_blocking = True
        try:
            feeder_api.move_bottom()
            return {"message": "Plate moved down to the bottom."}
        except Exception as e:
            logger.error(f"Error moving plate down: {e}")
            raise HTTPException(status_code=500,
                                detail="Error moving plate down.")
        finally:
            app.is_blocking = False


@app.get("/move_down/{steps}")
async def move_down(steps: int):
    """
    This method moves the plate down by the given number of steps.

    :param steps: The number of steps to move the plate down.
    """
    app.is_blocking = True
    feeder_api.move_down(steps)
    app.is_blocking = False
    return {"message": "Plate moved down."}


@app.get("/move_up/{steps}")
async def move_up(steps: int):
    """
    This method moves the plate up by the given number of steps.

    :param steps: The number of steps to move the plate up.
    """
    app.is_blocking = True
    feeder_api.move_up(steps)
    app.is_blocking = False
    return {"message": "Plate moved up."}


@app.get("/move_up_cargo/{cargo_num}")
async def move_up_cargo(cargo_num: int):
    """
    This method moves the plate by the given number of cargos.

    :param cargo_num: The number of cargos to move the plate by.
    """
    app.is_blocking = True
    feeder_api.move_up_cargo(cargo_num)
    app.is_blocking = False
    return {"message": f"Plate moved up by cargo number {cargo_num}."}


@app.get("/move_down_cargo/{cargo_num}")
async def move_down_cargo(cargo_num: int):
    """
    This method moves the plate by the given number of cargos.

    :param cargo_num: The number of cargos to move the plate by.
    """
    app.is_blocking = True
    feeder_api.move_down_cargo(cargo_num)
    app.is_blocking = False
    return {"message": f"Plate moved down by cargo number {cargo_num}."}


@app.get("/move_up_by_type/{cargo_type}/{cargo_num}")
async def move_up_by_type(cargo_type: str, cargo_num: int):
    """
    This method moves the plate by the given number of cargos.

    :param cargo_num: The number of cargos to move the plate by.
    """
    app.is_blocking = True
    res = feeder_api.move_up_cargo_by_type(cargo_type, cargo_num)
    app.is_blocking = False

    if res:
        return {"message": f"Plate {cargo_type} moved up by cargo "
                           f"number {cargo_num}."}
    else:
        raise HTTPException(status_code=400, detail="Invalid cargo type.")


class HealthMessage(BaseModel):
    """
    A class to represent the response message from the server.
    """
    message: str
    is_healthy: bool

@app.get("/health", response_model=HealthMessage)
async def health_check():
    return HealthMessage(message="Server is healthy", is_healthy=True)


@app.get("/move_down_by_type/{cargo_type}/{cargo_num}")
async def move_down_by_type(cargo_type: str, cargo_num: int):
    """
    This method moves the plate by the given number of cargos.

    :param cargo_num: The number of cargos to move the plate by.
    """
    app.is_blocking = True
    res = feeder_api.move_down_cargo_by_type(cargo_type, cargo_num)
    app.is_blocking = False

    if res:
        return {"message": f"Plate {cargo_type} moved down by cargo "
                           f"number {cargo_num}."}
    else:
        raise HTTPException(status_code=400, detail="Invalid cargo type.")


def _feed_plate_background(num_cargos: int):
    logger.info(f"Feeding {num_cargos} cargos.")
    feeder_api.feed_plate(num_cargos)
    logger.info(f"Feeding {num_cargos} cargos completed.")


@app.get("/feed_plate")
async def feed_plate(background_tasks: BackgroundTasks,
                     num_cargos: int = Query(1, ge=0, le=15)):
    logger.info(f"Received request to feed {num_cargos} cargos.")
    background_tasks.add_task(_feed_plate_background, num_cargos)
    return {"message": f"Feeding {num_cargos} cargos in the background."}


@app.get("/release_motors")
async def release_motors():
    try:
        feeder_api.release_motors()
        return {"message": "Motors released."}
    except Exception as e:
        logger.error(f"Error releasing motors: {e}")
        raise HTTPException(status_code=500, detail="Error releasing motors.")


@app.get("/enable_auto_feed")
async def enable_auto_feed():
    """
    This method enables auto feed.
    """
    app.auto_feed = True
    return {"message": "Auto feed enabled."}


@app.get("/disable_auto_feed")
async def disable_auto_feed():
    """
    This method disables auto feed.
    """
    app.auto_feed = False
    return {"message": "Auto feed disabled."}


@app.get("/is_running")
async def is_running():
    return {"is_running": feeder_api.is_running()}


@app.on_event("startup")
@repeat_every(seconds=2)
def check_loading():
    if app.auto_feed and not app.is_blocking and not feeder_api.query_distance():
        try:
            feeder_api.feed_plate_auto()
        except Exception as e:
            logger.error(f"Error feeding plate: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    try:
        feeder_api.shutdown()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
