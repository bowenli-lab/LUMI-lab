from dataclasses import dataclass

from pydantic import BaseModel
from plate_reader_api import carrier_in, carrier_out, read_plate, parse_file
from auto_parse import ReadingInfo
from fastapi import FastAPI, HTTPException

app = FastAPI()

@dataclass
class ReadingResponse:
    reading_info: ReadingInfo
    results: dict


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
def read_root():
    return {"message": "Hellow world"}

@app.post("/carrier-in")
def post_carrier_in():
    res, msg = carrier_in()
    
    if res:
        return {"message": msg}
    else:
        raise HTTPException(status_code=400, detail=msg)

@app.post("/carrier-out")
def post_carrier_out():
    res, msg = carrier_out()
    
    if res:
        return {"message": msg}
    else:
        raise HTTPException(status_code=400, detail=msg)

@app.post("/read-plate")
def post_read_plate(protocol_path: str, experiment_path: str, csv_path: str):
    res, msg = read_plate(protocol_path, experiment_path, csv_path)
    
    if res:
        return {"message": msg}
    else:
        raise HTTPException(status_code=400, detail=msg)

@app.post("/parse-file", response_model=ReadingResponse)
def post_parse_file(csv_path: str, ):
    try:
        return ReadingResponse(reading_info=parse_file(csv_path), results=parse_file(csv_path).results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=e)

