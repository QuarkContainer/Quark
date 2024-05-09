import torch
import math
import signal
import os
import sys
import logging
import json

import asyncio
import uvicorn

from fastapi import FastAPI, Request, APIRouter, Body
from fastapi.responses import JSONResponse, Response, StreamingResponse

from typing import Optional, Union, Any
from pydantic import BaseModel, Field, HttpUrl

import uuid

KEEP_ALIVE_TIME = 10    # seconds
app = FastAPI()


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


# init
class PromptReq(BaseModel):
    tenant: str = Field(default = 't1')
    namespace: str = Field(default = 'ns1')
    func: str = Field(default = 'f1')
    prompt: str = Field(default = 'nihao')


class PromptResp(BaseModel):
    namespace: str = Field(default = 'ns1')
    func: str = Field(default = 'test')
    prompt: str = Field(default = '¯\_(ツ)_/¯')
    result: str = Field(default = '0')


@app.get("/")
async def root() -> Response:
    return { "Hi! This is the root" }

@app.get("/liveness")
async def liveness() -> Response:
    """Health Check."""
    return Response("ALIVE", status_code=200)
    

@app.get("/readiness")
async def readiness() -> Response:
    """Health Check."""
    # return Response("/readiness\n", status_code=200)
    return Response("READY", status_code=200)

@app.post("/funccall")
async def post_func_call(req: PromptReq) -> PromptResp:   # body: Any = Body(None)
    #print("Is cuda available {}", torch.cuda.is_available())
    #print("PromptReq -> PromptResp")
    # return JSONResponse(content=body, status_code=200)
    #log_level = logging.DEBUG
    #torch._logging.set_logs(dynamo=log_level, aot=log_level, inductor=log_level)
    #torch._dynamo.config.verbose = True
    #dtype = torch.float
    # device = torch.device("cpu")
    #device = torch.device("cuda:0")
    #torch.manual_seed(0)

    # Create random input and output data
    #x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    #y = torch.sin(x)

    # Randomly initialize weights
    #a = torch.randn((), device=device, dtype=dtype)
    #b = torch.randn((), device=device, dtype=dtype)
    #c = torch.randn((), device=device, dtype=dtype)
    #d = torch.randn((), device=device, dtype=dtype)

    #learning_rate = 1e-6
    #for t in range(3):
        # Forward pass: compute predicted y
    #    y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # Compute and print loss
     #   loss = (y_pred - y).pow(2).sum().item()
     #   if t % 100 == 99:
     #       print(t, loss)

        # Backprop to compute gradients of a, b, c, d with respect to loss
     #   grad_y_pred = 2.0 * (y_pred - y)
     #   grad_a = grad_y_pred.sum()
     #   grad_b = (grad_y_pred * x).sum()
     #   grad_c = (grad_y_pred * x ** 2).sum()
     #   grad_d = (grad_y_pred * x ** 3).sum()

        # Update weights using gradient descent
     #   a -= learning_rate * grad_a
     #   b -= learning_rate * grad_b
     #   c -= learning_rate * grad_c
     #   d -= learning_rate * grad_d
        
    response = PromptResp(
        namespace = req.namespace,
        func = req.func,
        prompt = req.prompt,
        #result = f'y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3'
        result = 'ssssssssssssssssssssssssssss'
    )

    return (JSONResponse(status_code=202, content=response.model_dump()))


async def main():
    config = uvicorn.Config("__main__:app", 
                port=80, 
                reload=False, 
                log_level="debug", 
                host='0.0.0.0',
                proxy_headers=False,
                server_header=False)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
