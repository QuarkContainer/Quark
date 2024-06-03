import numpy
import os
import time
import sys
os.environ['CURL_CA_BUNDLE'] = ''

start = time.time()

import torch
torch.manual_seed(0)

start1 = time.time()

from transformers import LlamaForCausalLM, AutoTokenizer, FlaxLlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16)
#model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16, cache_dir="/model_weight")
model.to("cuda:0")
print(model)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
#tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir="/model_weight")
start2 = time.time()
print("torch time: ", start1-start)
print("model time: ", start2-start1)

import asyncio
import uvicorn

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
    prompt: str
    

class PromptResp(BaseModel):
    result: str 

    def __init__(self, result):
        self.result = result

@app.get("/")
async def root():
    print("hochan root")
    return {"message": "Hello Worldddd waahahhhhh"}

@app.get("/liveness")
async def liveness() -> Response:
    print("hochan liveness")
    return Response("liveness1", status_code=200)

@app.get("/readiness")
async def readiness() -> Response:
    print("hochan readiness")
    return Response("readiness1", status_code=200)
    
@app.post("/funccall")
async def post_func_call(req: PromptReq):
    print("req:", req.prompt)
    prompt = req.prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    generate_ids = model.generate(inputs.input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    resp = {"result": output }
    return JSONResponse(content=resp, status_code=200)


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