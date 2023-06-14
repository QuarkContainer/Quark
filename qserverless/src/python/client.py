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
import uuid
import os
import numpy as np

import func_pb2_grpc
import func_pb2
import blob_mgr
import func
import common

from func_pb2_grpc import *
from func_pb2 import *

funcAgentQueueTx = asyncio.Queue(100)
funcMgr = None

EnvVarNodeMgrPodId = "podid.core.qserverless.quarksoft.io";

def GetPodIdFromEnvVar() :
    podId = os.getenv(EnvVarNodeMgrPodId)
    if podId is None:
        podId = str(uuid.uuid4())
        return podId
    else :
        return podId
    
class FuncMgr:
    def __init__(self, clientQueue):
        self.funcPodId = GetPodIdFromEnvVar()
        self.namespace = "ns1"
        self.packageName = "package1"
        self.reqQueue = asyncio.Queue(100)
        self.clientQueue = clientQueue
        self.callerCalls = dict()
        self.blob_mgr = blob_mgr.BlobMgr(funcAgentQueueTx)
    
    async def RemoteCall(
        self,
        namespace: str, 
        packageName: str, 
        funcName: str, 
        parameters: str, 
        priority: int
        ) -> common.CallResult: 
        id = str(uuid.uuid4())
        req = func_pb2.FuncAgentCallReq (
            id = id,
            namespace = namespace,
            packageName = packageName,
            funcName = funcName,
            parameters = parameters,
            priority = 1
        )
        callQueue = asyncio.Queue(1)
        self.callerCalls[id] = callQueue
        self.clientQueue.put_nowait(func_pb2.FuncAgentMsg(msgId=0, FuncAgentCallReq=req))
        res = await callQueue.get()
        return res
    
    async def BlobCreate(self, name: str) -> common.CallResult: 
        return await self.blob_mgr.BlobCreate(name)
    
    async def BlobWrite(self, id: np.uint64, buf: bytes) -> common.CallResult:
        return await self.blob_mgr.BlobWrite(id, buf)
    
    async def BlobOpen(self, addr: common.BlobAddr) -> common.CallResult:
        return await self.blob_mgr.BlobOpen(addr)
    
    async def BlobDelete(self, svcAddr: str, name: str) -> common.CallResult:
        return await self.blob_mgr.BlobDelete(svcAddr, name)
    
    async def BlobRead(self, id: np.uint64, len: np.uint64) -> common.CallResult:
        return await self.blob_mgr.BlobRead(id, len)
    
    async def BlobSeek(self, id: np.uint64, seekType: int, pos: np.int64) -> common.CallResult:
        return await self.blob_mgr.BlobSeek(id, seekType, pos)
    
    async def BlobClose(self, id: np.uint64) -> common.CallResult:
        return await self.blob_mgr.BlobClose(id)
    
    def CallRespone(self, id: str, res: common.CallResult) :
        callQueue = self.callerCalls.get(id)
        if callQueue is None:
            print("CallRespone get unknow callid", id)
            return
        
        callQueue.put_nowait(res)
        self.callerCalls.pop(id)
        
    def LocalCall(self, req: func_pb2.FuncAgentCallReq) :
        self.reqQueue.put_nowait(req)
        
    async def FuncCall(self, name: str, parameters: str) -> common.CallResult:
        function = getattr(func, name)
        if function is None:
            return common.CallResult("", "There is no func named {}".format(name))
        result = await function(self, parameters)
        return common.CallResult(result, "")
        
    async def Process(self) :
        while True :
            req = await self.reqQueue.get();
            if req is None:
                break
            
            res = await self.FuncCall(req.funcName, req.parameters)
            resp = func_pb2.FuncAgentCallResp(id=req.id, resp=res.res, error=res.error)
            self.clientQueue.put_nowait(func_pb2.FuncAgentMsg(msgId=2, FuncAgentCallResp=resp))

funcMgr = FuncMgr(funcAgentQueueTx)       
blob_mgr.blobMgr = funcMgr.blob_mgr

async def generate_messages():
    while True:
        item = await funcAgentQueueTx.get()
        yield item

async def FuncAgentClientProcess(msg: func_pb2.FuncAgentMsg):
    msgType = msg.WhichOneof('EventBody')
    #print("FuncAgentClientProcess", msg.msgId, msgType, msg)
    match msgType:
        case 'FuncAgentCallReq' :
            req = msg.FuncAgentCallReq
            funcMgr.LocalCall(req)
        case 'FuncAgentCallResp':
            resp = msg.FuncAgentCallResp
            res = common.CallResult(res=resp.resp, error=resp.error)
            funcMgr.CallRespone(resp.id, res)
        case _ :
            funcMgr.blob_mgr.OnFuncAgentMsg(msg)
            

async def StartClientProcess():
    podId = str(uuid.uuid4())
    regReq = func_pb2.FuncPodRegisterReq(funcPodId=podId,namespace="ns1",packageName="package1")
    req = func_pb2.FuncAgentMsg(msgId=1, FuncPodRegisterReq=regReq)
    funcAgentQueueTx.put_nowait(func_pb2.FuncAgentMsg(msgId=1, FuncPodRegisterReq=regReq))

    async with grpc.aio.insecure_channel("unix:///var/lib/quark/nodeagent/node1/sock") as channel:
        stub = FuncAgentServiceStub(channel)
        responses = stub.StreamProcess(generate_messages())
        async for response in responses:
            await FuncAgentClientProcess(response)

async def RemoteCall(
    namespace: str, 
    packageName: str, 
    funcName: str, 
    parameters: str, 
    priority: int
    ) -> common.CallResult: 
    res = funcMgr.RemoteCall(namespace, packageName, funcName, parameters, priority)
    return res

async def main():
    #func.funcMgr = funcMgr
    agentClientTask = asyncio.create_task(StartClientProcess())
    await funcMgr.Process()

asyncio.run(main())

