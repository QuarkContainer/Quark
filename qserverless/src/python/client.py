# Copyright (c) 2021 Quark Container Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import grpc

import func_mgr

from func_pb2_grpc import *
from func_pb2 import *

funcAgentQueueTx = asyncio.Queue(100)
funcAgentQueueRx = asyncio.Queue(100)

async def generate_messages():
    while True:
        item = await funcAgentQueueTx.get()
        yield item

async def FuncAgentClientProcess(msg: FuncAgentMsg):
    msgType = msg.WhichOneof('EventBody')
    print("FuncAgentClientProcess", msg.msgId, msgType, msg)
    match msgType:
        case 'FuncAgentCallReq' :
            req = msg.FuncAgentCallReq
            res = await func_mgr.LocalCall(req.funcName, req.parameters)
            resp = FuncAgentCallResp(id=req.id, resp=res.res, error=res.error)
            funcAgentQueueTx.put_nowait(FuncAgentMsg(msgId=2, FuncAgentCallResp=resp))
        case 'FuncAgentCallResp':
            print("FuncAgentCallResp")
        case _ :
            print("FuncAgentCallRespxxxxxx")

async def StartClientProcess():
    regReq = FuncPodRegisterReq(funcPodId="xyz",namespace="ns1",packageName="package1")
    req = FuncAgentMsg(msgId=1, FuncPodRegisterReq=regReq)
    funcAgentQueueTx.put_nowait(FuncAgentMsg(msgId=1, FuncPodRegisterReq=regReq))

    async with grpc.aio.insecure_channel("unix:///var/lib/quark/nodeagent/node1/sock") as channel:
        stub = FuncAgentServiceStub(channel)
        responses = stub.StreamProcess(generate_messages())
        async for response in responses:
            await FuncAgentClientProcess(response)
            funcAgentQueueRx.put_nowait(response)

async def main():
    task = asyncio.create_task(StartClientProcess())
    while True:
        item = await funcAgentQueueRx.get()
        print(item)

asyncio.run(main())

