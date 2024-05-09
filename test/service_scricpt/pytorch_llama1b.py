import torch
import math
import logging
import os

import asyncio
import uvicorn

from fastapi import FastAPI, Request, APIRouter, Body
from fastapi.responses import JSONResponse, Response, StreamingResponse

from typing import Optional, Union, Any
from pydantic import BaseModel, Field, HttpUrl
from transformers import LlamaForCausalLM, AutoTokenizer, FlaxLlamaForCausalLM

import uuid
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import requests, os
session = requests.Session()
session.verify = False
session.trust_env = False

KEEP_ALIVE_TIME = 10    # seconds
app = FastAPI()

#os.environ['CURL_CA_BUNDLE'] = ''
# model = LlamaForCausalLM.from_pretrained("/cchen/model_weight/llama1b", local_files_only=True)
model = LlamaForCausalLM.from_pretrained("/cchen/model_weight/llama7b", torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True)
#model = LlamaForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K",torch_dtype=torch.float16, cache_dir="/home/huawei/cchen/model_weight")
model.to("cuda:0")
# tokenizer = AutoTokenizer.from_pretrained("/cchen/model_weight/llama1b", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("/cchen/model_weight/llama7b", local_files_only=True, low_cpu_mem_usage=True)

def random_uuid() -> str:
    return str(uuid.uuid4().hex)


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
    prompt = req.prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(input_ids, max_length=50, temperature=0.7)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # nputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    # generate_ids = model.generate(inputs.input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
    # output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    response = PromptResp(
        namespace = req.namespace,
        func = req.func,
        prompt = req.prompt,
        result = f'Output from server: {output_text}'
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
