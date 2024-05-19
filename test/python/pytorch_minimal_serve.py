# import torch
# import math
# import signal
# import os
# import sys
# import logging
# import requests
# import json

import asyncio
import uvicorn
import numpy
import os
import time
os.environ['CURL_CA_BUNDLE'] = ''

from typing import Any
from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse, Response, StreamingResponse

from typing import Optional
from pydantic import BaseModel, Field

import uuid

KEEP_ALIVE_TIME = 10    # seconds
app = FastAPI()


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


class PromptReq(BaseModel):
    # id: str = random_uuid()
    tenant: str
    namespace: str
    func: str
    prompt: str
    

class PromptResp(BaseModel):
    request: Optional[str] = None
    result: str = Field(default = '0')
    # result: str

@app.get("/")
async def root():
    return {"message": "Hello Worldddd waahahhhhh"}

@app.get("/liveness")
async def liveness() -> Response:
    return Response("liveness1", status_code=200)

@app.get("/readiness")
async def readiness() -> Response:
    return Response("readiness1", status_code=200)

@app.post("/funccall")
async def post_func_call(body: Any = Body(None)):
    return JSONResponse(content=body, status_code=200)
 

async def main():
    config = uvicorn.Config("__main__:app",
                port=80,
                reload=True,
                log_level="debug",
                host='0.0.0.0')
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())