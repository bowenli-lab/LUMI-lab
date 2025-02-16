from datetime import datetime
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .info_api import SDL_DB_API

app = FastAPI()

sdl_db_api = SDL_DB_API()


class MessageResponse(BaseModel):
    """
    A class to represent the response message from the server.
    """

    message: str


# Pydantic model for the response
class Entry(BaseModel):
    id: str
    last_updated: datetime


@app.get("/entries", response_model=List[Entry])
def get_sample_entries():
    try:
        res = sdl_db_api.get_all_samples()
        return res
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/entry/{sample_id}")
def get_sample_entry(sample_id: str):
    try:
        res = sdl_db_api.get_reading_result_by_id(sample_id)
        return res
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/entry/{sample_id}/readings")
def get_mapped_result_by_id(sample_id: str):
    try:
        res = sdl_db_api.get_mapped_result_by_id(sample_id)
        return res
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail=str(e))


# get the gain value of the experiment
@app.get("/entry/{sample_id}/gain")
def get_gain(sample_id: str):
    try:
        res = sdl_db_api.get_gain(sample_id)
        return res
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/reagent")
def get_reagent():
    try:
        res = sdl_db_api.get_reagent()
        return res
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail=str(e))
