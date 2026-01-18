from fastapi import FastAPI
import uvicorn
from src.logger import logging
from src.exception import CustomException
import sys

app = FastAPI()

@app.get("/")
def check_health():
    try:
        logging.info('Health check OK')
        return {
            'status': "OK",
            "code" : 200
        }
    except Exception as e:
        logging.info("Division by zero")
        raise CustomException(e, sys)


if __name__ == "__main__":

    print("Breakpoint here...")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )