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
import janus
import json
import sys
import io
import traceback
from contextlib import redirect_stdout
from contextlib import redirect_stderr

import qserverless.func_pb2 as func_pb2
import qserverless.blob_mgr as blob_mgr
import qserverless.func as func
import qserverless.common as common

from qserverless.func_pb2_grpc import *
from qserverless.func_pb2 import *

funcAgentQueueTx = asyncio.Queue(100)
funcMgr = None

EnvVarNodeMgrPodId      = "qserverless_podid";
EnvVarNodeMgrNamespace  = "qserverless_namespace";
EnvVarNodeMgrPackageId  = "qserverless_packageid";
EnvVarNodeAgentAddr     = "qserverless_nodeagentaddr";
DefaultNodeAgentAddr    = "unix:///var/lib/quark/nodeagent/sock";

funcPodId = None

async def generate_messages():
    while True:
        item = await funcAgentQueueTx.get()
        yield item

def GetPodIdFromEnvVar() :
    global funcPodId
    if funcPodId is not None:
        return funcPodId
    
    podId = os.getenv(EnvVarNodeMgrPodId)
    if podId is None:
        funcPodId = str(uuid.uuid4())
        return funcPodId
    else :
        return podId
    
def GetNamespaceFromEnvVar() :
    ns = os.getenv(EnvVarNodeMgrNamespace)
    if ns is None :
        return ""
    return ns
    
def GetPackageIdFromEnvVar() :
    pid = os.getenv(EnvVarNodeMgrPackageId)
    if pid is None :
        return ""
    return pid

def GetNodeAgentAddrFromEnvVar() :
    addr = os.getenv(EnvVarNodeAgentAddr)
    if addr is None :
        return DefaultNodeAgentAddr
    return addr

class FuncInstance: 
    def __init__(self, nodeId: str, podId: str, funcId: str):
        self.nodeId = nodeId
        self.podId = podId
        self.funcId = funcId
        self.msgQueue = asyncio.Queue(100)
    
    async def ReadMsg(self) :
        msg = await self.msgQueue.get()
        return msg
        

class FuncCallContext:
    def __init__(self, jobId: str, id: str, parent: FuncInstance):
        self.jobId = jobId
        self.id = id
        self.funcInstances = dict()
        self.parent = parent
        self.msgQueues = dict()
    
    async def SendMsg(self, funcInstance: FuncInstance, msgCode: int, data: str):
        await funcMgr.SendMsg(self, funcInstance, msgCode, dict(), data)
    
    async def ReadParentMsg(self):
        return await self.parent.ReadMsg()
    
    async def ReadChildMsg(self, funcInstance: FuncInstance):
        return await funcInstance.ReadMsg()
    
    async def SendParentMsg(self, msgCode: int, data: str):
        await self.SendMsg(self, self.parent, msgCode, data)
    
    async def SendChildMsg(self, funcInstance: FuncInstance, msgCode: int, data: str):
        await self.SendMsg(funcInstance, msgCode, data)
    
    def NewTaskContext(self) :
        id = str(uuid.uuid4())
        return FuncCallContext(self.jobId, id, None)
    
    def NewChildFuncInstance(self, nodeId: str, podId: str, funcId: str):
        instance = FuncInstance(nodeId, podId, funcId)
        self.funcInstances[funcId] = instance
        return instance
    
    def CallRespone(self, id: str, res: common.CallResult):
        funcInstance = self.funcInstances.get(id)
        if funcInstance is None:
            return
        if len(res.error) != 0:
            funcInstance.msgQueue.put_nowait((None, res.error))
            
        funcInstance.msgQueue.put_nowait(None)
    
    def DispatchFuncMsg(self, msg: func_pb2.FuncMsg):
        if msg.dstFuncId != self.id:
            print("get unexpect FuncMsg for func ", msg.dstFuncId, "my func is ", self.id)
            return;
        
        if self.parent is not None and msg.srcFuncId == self.parent.funcId:
            self.parent.msgQueue.put_nowait(msg.FuncMsgBody.data)
            return
        
        funcInstance = self.funcInstances.get(msg.srcFuncId)
        if funcInstance is None:
            return
        
        funcInstance.msgQueue.put_nowait((msg.FuncMsgBody.data, ""))
        
    async def RemoteCall(
        self,
        **kwargs
        ) : # -> (str, common.QErr):
        packageName = kwargs.get('packageName')
        if packageName is None :
            packageName = ""
        else: 
            del kwargs['packageName']
        funcName = kwargs['funcName']
        priority = kwargs.get('priority')
        if priority is None:
            priority = 1
        else:
            del kwargs['priority']
        
        del kwargs['funcName']
        parameters = json.dumps(kwargs)
                 
        res = await funcMgr.RemoteCall(
            self, 
            packageName, 
            funcName, 
            self.id, 
            priority, 
            parameters
        )
        
        if res.error == "":
            return (res.res, None)
        return (None, common.QErr(res.error))
    
    async def RemoteCallIterate(self, **kwargs) -> FuncInstance:
        packageName = kwargs.get('packageName')
        if packageName is None :
            packageName = ""
        else: 
            del kwargs['packageName']
        funcName = kwargs['funcName']
        priority = kwargs.get('priority')
        if priority is None:
            priority = 1
        else:
            del kwargs['priority']
        
        del kwargs['funcName']
        parameters = json.dumps(kwargs)
         
        res = await funcMgr.RemoteCallIterate(
            self, 
            packageName, 
            funcName, 
            self.id, 
            priority, 
            parameters
        )
        
        return res;
    
    def NewBlobAddr(self) -> common.BlobAddr : 
        id = self.jobId + "/" + str(uuid.uuid4())
        return common.BlobAddr(None, id)
    
    def NewBlobAddrVec(self, cols: int) -> common.BlobAddrVec:
        vec = list()
        for col in range(0, cols):
            addr = self.NewBlobAddr()
            vec.append(addr)
        return vec
    
    def NewBlobAddrMatrix(self, rows: int, cols: int) -> common.BlobAddrMatrix:
        mat = list()
        for row in range(0, rows):
            mat.append(self.NewBlobVec(cols)) 
        return mat
    
    async def BlobWriteAll(self, addr: common.BlobAddr, buf: bytes): # -> common.QErr:
        (b, err) = await self.BlobCreate(addr)
        if err is not None :
            return (None, err) 
        err = await b.Write(buf)
        if err is not None :
            return (None, err)  
        ret = await b.Close()
        if err is not None :
            return (None, err)  
        
        return (b.addr, None) 
    
    async def BlobReadAll(self, addr: common.BlobAddr): # -> (bytes, common.QErr):
        (b, err) = await self.BlobOpen(addr)
        if err is not None :
            return (None, err)
        ret = bytes()
        size = 64 * 1024
        while True:
            (data, err) = await b.Read(size)
            if err is not None :
                return (None, err)
            ret = ret + data
            if len(data) < size:
                break
            
        err = await b.Close()
        if err is not None :
            return (None, err)
        
        return (ret, None)
            
    async def BlobCreate(self, addr: common.BlobAddr): #-> (UnsealBlob, common.QErr):
        return await funcMgr.BlobCreate(addr)
    
    async def BlobWrite(self, id: np.uint64, buf: bytes) -> common.QErr:
        return await funcMgr.BlobWrite(id, buf)
    
    async def BlobOpen(self, addr: common.BlobAddr) : #-> (Blob, common.QErr):
        return await funcMgr.BlobOpen(addr)
    
    async def BlobDelete(self, svcAddr: str, name: str) -> common.QErr:
        return await funcMgr.BlobDelete(svcAddr, name)
    
    async def BlobRead(self, id: np.uint64, len: np.uint64) : #-> (bytes, common.QErr):
        return await funcMgr.BlobRead(id, len)
    
    async def BlobSeek(self, id: np.uint64, seekType: int, pos: np.int64) : #-> (int, common.QErr):
        return await funcMgr.BlobSeek(id, seekType, pos)
    
    async def BlobClose(self, id: np.uint64) -> common.QErr:
        return await funcMgr.BlobClose(id)

def NewJobContext() -> FuncCallContext :
    id = str(uuid.uuid4())
    context = FuncCallContext(id, id, None)
    funcMgr.context = context
    return context

class FuncMgr:
    def __init__(self, svcAddr: str, namespace: str, packageName: str):
        self.funcPodId = GetPodIdFromEnvVar()
        self.namespace = namespace
        self.packageName = packageName
        self.reqQueue = asyncio.Queue(100)
        self.clientQueue = funcAgentQueueTx
        self.callerCalls = dict()
        self.childrenMsg = asyncio.Queue(100)
        self.blob_mgr = blob_mgr.BlobMgr(funcAgentQueueTx)
        self.svcAddr = svcAddr
        self.context = None
        self.funcMsgs = dict()
        blob_mgr.blobMgr = self
    
    async def Call(
        self,
        context: FuncCallContext,
        packageName: str, 
        funcName: str, 
        callerFuncId: str,
        priority: int,
        parameters: str,
        callType: int
        ) : 
        
        if packageName == "" :
            packageName = self.packageName
        
        id = str(uuid.uuid4())
        req = func_pb2.FuncAgentCallReq (
            jobId = context.jobId,
            id = id,
            namespace = self.namespace,
            packageName = packageName,
            funcName = funcName,
            parameters = parameters,
            callerFuncId = callerFuncId,
            priority = priority,
            callType = callType
        )
        callQueue = janus.Queue() #asyncio.Queue(1)
        self.callerCalls[id] = callQueue
        self.clientQueue.put_nowait(func_pb2.FuncAgentMsg(msgId=0, FuncAgentCallReq=req))
        res = await callQueue.async_q.get()
        return res
    
    async def RemoteCall(
        self,
        context: FuncCallContext,
        packageName: str, 
        funcName: str, 
        callerFuncId: str,
        priority: int,
        parameters: str 
        ) -> common.CallResult: 
        return await self.Call (
            context,
            packageName,
            funcName,
            callerFuncId,
            priority,
            parameters,
            1
        )
        
    async def RemoteCallIterate(
        self,
        context: FuncCallContext,
        packageName: str, 
        funcName: str, 
        callerFuncId: str,
        priority: int,
        parameters: str 
        ) -> FuncInstance: 
        msg = await self.Call (
            context,
            packageName,
            funcName,
            callerFuncId,
            priority,
            parameters,
            2
        )
        
        childFuncInstance = context.funcInstances[msg.id]
        return childFuncInstance
        
    async def SendMsg (
        self,
        context: FuncCallContext,
        funcInstance: FuncInstance,
        msgCode: int,
        annotations: dict,
        data: str
    ) : 
        id = str(uuid.uuid4())
        msg = func_pb2.FuncMsg (
            msgId = id,
            srcNodeId = "",
            srcPodId = self.funcPodId,
            srcFuncId = context.id,
            dstNodeId = funcInstance.nodeId,
            dstPodId = funcInstance.podId,
            dstFuncId = funcInstance.funcId,
            FuncMsgBody = func_pb2.FuncMsgBody (
                msgCode = msgCode,
                annotations = annotations,
                data = data,
            )
        )
        
        msgQueue = janus.Queue() #asyncio.Queue(1)
        self.funcMsgs[id] = msgQueue
        self.clientQueue.put_nowait(func_pb2.FuncAgentMsg(msgId=0, FuncMsg=msg))
        res = await msgQueue.async_q.get()
        assert res.srcNodeId == msg.dstNodeId 
        assert res.srcPodId == msg.dstPodId
        assert res.srcFuncId == msg.dstFuncId
        #  msg.srcNodeId is not set, no need check ....assert msg.srcNodeId == res.dstNodeId 
        assert msg.srcPodId == res.dstPodId
        assert msg.srcFuncId == res.dstFuncId
        if len(res.FuncMsgAck.error):
            raise Exception("SendMsg msg annotations {} fail with error {}".format(annotations, res.FuncMsgAck.error))
    
    def MsgAck(self, ack: func_pb2.FuncMsg):
        id = ack.msgId
        msgQueue = self.funcMsgs.get(id)
        if msgQueue is None:
            return
        
        msgQueue.async_q.put_nowait(ack)
        self.funcMsgs.pop(id)
    
    async def BlobCreate(self, addr: common.BlobAddr): #-> (UnsealBlob, common.QErr):
        return await funcMgr.blob_mgr.BlobCreate(addr['name'])
    
    async def BlobWrite(self, id: np.uint64, buf: bytes) -> common.QErr:
        return await funcMgr.blob_mgr.BlobWrite(id, buf)
    
    async def BlobOpen(self, addr: common.BlobAddr) : #-> (Blob, common.QErr):
        return await funcMgr.blob_mgr.BlobOpen(addr)
    
    async def BlobDelete(self, svcAddr: str, name: str) -> common.QErr:
        return await funcMgr.blob_mgr.BlobDelete(svcAddr, name)
    
    async def BlobRead(self, id: np.uint64, len: np.uint64) : #-> (bytes, common.QErr):
        return await funcMgr.blob_mgr.BlobRead(id, len)
    
    async def BlobSeek(self, id: np.uint64, seekType: int, pos: np.int64) : #-> (int, common.QErr):
        return await funcMgr.blob_mgr.BlobSeek(id, seekType, pos)
    
    async def BlobClose(self, id: np.uint64) -> common.QErr:
        return await funcMgr.blob_mgr.BlobClose(id)
    
    def CallRespone(self, id: str, res: common.CallResult) :
        callQueue = self.callerCalls.get(id)
        self.context.CallRespone(id, res)
        if callQueue is None:
            return
        callQueue.async_q.put_nowait(res)
        self.callerCalls.pop(id)
    
    def CallIterateAck(self, id: str, msg: func_pb2.FuncAgentCallAck):
        callQueue = self.callerCalls.get(id)
        if callQueue is None:
            return
        self.context.NewChildFuncInstance(msg.calleeNodeId, msg.calleePodId, msg.id)
        callQueue.async_q.put_nowait(msg)
        self.callerCalls.pop(id)
        
    def LocalCall(self, req: func_pb2.FuncAgentCallReq) :
        self.reqQueue.put_nowait(req)
    
    async def FuncCallIterate(self, context: FuncCallContext, name: str, parameters: str) -> common.CallResult:
        logname = '/var/log/quark/{}.log'.format(context.id)
        with open(logname, 'w') as f:
            with redirect_stdout(f):
                with redirect_stderr(f):
                    try:
                        function = getattr(func, name)
                        if function is None:
                            return common.CallResult("", "There is no func named {}".format(name))
                        
                        kwargs = json.loads(parameters)
                        result = None
                        err = None

                        ack = func_pb2.FuncAgentCallAck(
                            id=context.id, error="", 
                            calleePodId=self.funcPodId
                        )
                        
                        self.clientQueue.put_nowait(func_pb2.FuncAgentMsg(FuncAgentCallAck=ack))
                
                        async for data in function(context, **kwargs):
                            await self.SendMsg(
                                context,
                                context.parent,
                                0,
                                dict(),
                                data
                            )
                            
                        return common.CallResult("", "")
                    except Exception as err:
                        err = "func xxxx {} call iterate fail with exception {} {}".format(name, err, traceback.format_exc())
                        return common.CallResult("", err)
    
    async def FuncCall(self, context: FuncCallContext, name: str, parameters: str) -> common.CallResult:
        logname = '/var/log/quark/{}.log'.format(context.id)
        with open(logname, 'w') as f:
            with redirect_stdout(f):
                with redirect_stderr(f):
                    try:
                        function = getattr(func, name)
                        if function is None:
                            return common.CallResult("", "There is no func named {}".format(name))
                        
                        kwargs = json.loads(parameters)
                        result = None
                        err = None

                        (result, err) = await function(context, **kwargs)
                        if result is None:
                            result = ""
                        if err is not None:
                            err = json.dumps(err)
                        else:
                            err = ""
                        return common.CallResult(result, err)
                    except Exception as err:
                        err = "func ttttt {} call fail with exception {} {}".format(name, err, traceback.format_exc())
                        return common.CallResult("", err)
    
    def Close(self) : 
        self.reqQueue.put_nowait(None)
    
    async def Process(self) :
        while True :
            req = await self.reqQueue.get();
            if req is None:
                break
            parent = FuncInstance(req.callerNodeId, req.callerPodId, req.callerFuncId)
            context = FuncCallContext(req.jobId, req.id, parent)
            res = None
            self.context = context
            if req.callType == 1 :
                res = await self.FuncCall(context, req.funcName, req.parameters)
            else:
               res = await self.FuncCallIterate(context, req.funcName, req.parameters)
            self.context = None
            resp = func_pb2.FuncAgentCallResp(id=req.id, resp=res.res, error=res.error)
            self.clientQueue.put_nowait(func_pb2.FuncAgentMsg(msgId=2, FuncAgentCallResp=resp))
    
    def DispatchFuncMsg(self, msg: func_pb2.FuncMsg):
        resp = func_pb2.FuncMsgAck(error="")
        if msg.dstPodId != self.funcPodId:
            resp = func_pb2.FuncMsgAck(error="can't find target func pod")
            return;
        
        if self.context is not None:
            self.context.DispatchFuncMsg(msg)
            
        ack = func_pb2.FuncMsg (
            msgId = msg.msgId,
            srcNodeId = msg.dstNodeId,
            srcPodId = msg.dstPodId,
            srcFuncId = msg.dstFuncId,
            dstNodeId = msg.srcNodeId,
            dstPodId = msg.srcPodId,
            dstFuncId = msg.srcFuncId,
            FuncMsgAck = resp
        )
        self.clientQueue.put_nowait(func_pb2.FuncAgentMsg(msgId=2, FuncMsg=ack))
            
    async def FuncAgentClientProcess(self, msg: func_pb2.FuncAgentMsg):
        msgType = msg.WhichOneof('EventBody')
        match msgType:
            case 'FuncAgentCallReq' :
                req = msg.FuncAgentCallReq
                self.LocalCall(req)
            case 'FuncAgentCallResp':
                resp = msg.FuncAgentCallResp
                res = common.CallResult(res=resp.resp, error=resp.error)
                self.CallRespone(resp.id, res)
            case 'FuncAgentCallAck':
                resp = msg.FuncAgentCallAck
                self.CallIterateAck(resp.id, resp)
            case 'FuncMsg':
                msg = msg.FuncMsg
                payloadType = msg.WhichOneof('Payload')
                match payloadType:
                    case 'FuncMsgBody':
                        self.DispatchFuncMsg(msg)
                        
                    case 'FuncMsgAck':
                        self.MsgAck(msg)
            case _ :
                self.blob_mgr.OnFuncAgentMsg(msg)
                    
    async def StartClientProcess(self):
        print("start to connect addr ", self.svcAddr)
        try: 
            async with grpc.aio.insecure_channel(self.svcAddr) as channel:
                stub = FuncAgentServiceStub(channel)
                responses = stub.StreamProcess(generate_messages())
                async for response in responses:
                    await self.FuncAgentClientProcess(response)
        except Exception as err:
            print("unexpect error", err)
            sys.exit(1)
            

def Register(svcAddr: str, namespace: str, packageName: str, clientMode: bool):
    global funcMgr
    funcMgr = FuncMgr(svcAddr, namespace, packageName)  
    podId = GetPodIdFromEnvVar()
    regReq = func_pb2.FuncPodRegisterReq(funcPodId=podId,namespace=namespace,packageName=packageName, clientMode= clientMode)
    req = func_pb2.FuncAgentMsg(msgId=1, FuncPodRegisterReq=regReq)
    funcAgentQueueTx.put_nowait(func_pb2.FuncAgentMsg(msgId=1, FuncPodRegisterReq=regReq))


async def StartSvc():
    agentClientTask = asyncio.create_task(funcMgr.StartClientProcess())
    await funcMgr.Process()


if __name__ == '__main__':
    namespace = GetNamespaceFromEnvVar()
    packageId = GetPackageIdFromEnvVar()
    svcAddr = GetNodeAgentAddrFromEnvVar()
    print("namespace = ", namespace, " packageId =", packageId, " svcAddr = ", svcAddr)
    Register(svcAddr, namespace, packageId, False)
    asyncio.run(StartSvc())

def Call(**kwargs) :# -> (str, qserverless.Err):
    id = str(uuid.uuid4())
    
    svcAddr = kwargs.get('svcAddr')
    if svcAddr is None :
        svcAddr = GetNodeAgentAddrFromEnvVar()
    else: 
        del kwargs['svcAddr']
    
    namespace = kwargs.get('namespace')
    if namespace is None :
        namespace = GetNamespaceFromEnvVar()
    else: 
        del kwargs['namespace']
    
    packageName = kwargs.get('packageName')
    if packageName is None :
        packageName = GetPackageIdFromEnvVar()
    else: 
        del kwargs['packageName']
        
    funcName = kwargs.get('funcName')
    if packageName is None :
        print("there is no funcname in parameter list")
        return (None, common.QErr("Error: there is no funcname in parameter list"))
    else: 
        del kwargs['funcName']
    
    priority = kwargs.get('priority')
    if priority is None:
        priority = 1
    else:
        del kwargs['priority']
    
    parameters = json.dumps(kwargs)
    
    # print("jobid ", id);
    # print("svcAddr ", svcAddr);
    # print("namespace ", namespace);
    # print("packageName ", packageName);
    # print("funcName ", funcName);
    # print("parameters ", parameters);
    # print("priority ", priority);
    
    req = func_pb2.FuncAgentCallReq (
        jobId = id,
        id = id,
        namespace = namespace,
        packageName = packageName,
        funcName = funcName,
        parameters = parameters,
        priority = priority
    )
    
    channel = grpc.insecure_channel(svcAddr)
    stub = FuncAgentServiceStub(channel)
    
    res = stub.FuncCall(req)
    return common.CallResult(res.resp, res.error)
    