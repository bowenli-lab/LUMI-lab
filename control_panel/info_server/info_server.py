import uvicorn
from app.main import app

if __name__ == "__main__":
    uvicorn.run("info_server:app", host="192.168.1.11",
                port=8700, reload=True)
