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
import datetime

import numpy as np

import func_pb2
import common
from common import BlobAddr 

SeekFromStart = 1
SeekFromEnd = 2
SeekFromCurrent = 3

blobMgr = None

def TimestampToDateTime(ts: func_pb2.Timestamp) -> datetime.datetime:
    timedelta = datetime.timedelta(seconds=ts.seconds, microseconds=ts.nanos // 1000)
    return datetime.datetime(1970, 1, 1) + timedelta

def DatetimeToTimeStamp(time: datetime.datetime) -> func_pb2.Timestamp:
    timedelta = time - datetime.datetime(1970, 1, 1)
    return func_pb2.Timestamp(
        seconds = timedelta.total_seconds,
        nanos = timedelta.microseconds * 1000
    )

class Blob: 
    def __init__(
        self,
        id: np.uint64, 
        addr: BlobAddr, 
        size: int,
        checksum: str,
        createTime: datetime.datetime,
        lastAccessTime: datetime.datetime
        ):
        self.id = id
        self.addr = addr
        self.size = size
        self.checksum = checksum
        self.createTime = createTime
        self.lastAccessTime = lastAccessTime
        self.closed = False
    
    async def Read(self, len: np.uint64) -> common.CallResult:
        return await blobMgr.BlobRead(self.id, len)
    
    async def Seek(self, seekType: int, pos: np.int64) -> common.CallResult:
        return await blobMgr.BlobSeek(self.id, seekType, pos)
    
    async def Close(self):
        if self.closed == False:
            self.closed = True
            await blobMgr.BlobClose(self.id)
    
class UnsealBlob:
    def __init__(
        self, 
        id: np.uint64,
        addr: BlobAddr) : 
        self.id = id
        self.addr = addr
        self.closed = False 
        
    async def Write(self, buf: bytes) -> common.CallResult:
        return await blobMgr.BlobWrite(self.id, buf)

    async def Close(self):
        if self.closed == False:
            self.closed = True
            await blobMgr.BlobClose(self.id)

           
class BlobMgr:
    def __init__(self, clientQueue: asyncio.Queue) :
        self.clientQueue = clientQueue
        self.blobReqs = dict()
        self.lastMsgId = 1
        
    def BlobReq(self, msg: func_pb2.FuncAgentMsg) :
        self.clientQueue.put_nowait(msg)
    
    def OnFuncAgentMsg(self, msg: func_pb2.FuncAgentMsg) -> common.CallResult :
        msgId = msg.msgId
        reqQueue = self.blobReqs.get(msgId)
        if reqQueue is None:
            return common.CallResult("", "BlobMgr::OnFuncAgentMsg unknow msgId " + str(msgId))
        reqQueue.put_nowait(msg)
        
    def MsgId(self) -> np.uint64 :
        self.lastMsgId += 1
        return self.lastMsgId
    
    async def BlobCreate(self, name: str) -> common.CallResult:
        msgId = self.MsgId()
        req = func_pb2.BlobCreateReq(
            namespace = "",
            name = name
        )
        msg = func_pb2.FuncAgentMsg (
            msgId = msgId,
            BlobCreateReq = req
        )
        
        queue = asyncio.Queue(1)
        self.blobReqs[msgId] = queue
        self.BlobReq(msg)
        msgResp = await queue.get()
        msgType = msgResp.WhichOneof('EventBody')
        match msgType:
            case 'BlobCreateResp':
                resp = msgResp.BlobCreateResp
                if resp.error != "" :
                    return common.CallResult(None, resp.error)
                
                blob = UnsealBlob(resp.id, BlobAddr(resp.svcAddr, name))
                return common.CallResult(blob, "")
            case _ :
                return common.CallResult(None, "BlobCreate invalid resp " + msgResp)
    
    async def BlobWrite(self, id: np.uint64, buf: bytes) -> common.CallResult:
        msgId = self.MsgId()
        req = func_pb2.BlobWriteReq(
            id = id,
            data = buf
        )
        msg = func_pb2.FuncAgentMsg (
            msgId = msgId,
            BlobWriteReq = req
        )
        queue = asyncio.Queue(1)
        self.blobReqs[msgId] = queue
        self.BlobReq(msg)
        msgResp = await queue.get()
        msgType = msgResp.WhichOneof('EventBody')
        match msgType:
            case 'BlobWriteResp':
                resp = msgResp.BlobWriteResp
                if resp.error != "" :
                    return common.CallResult(None, resp.error)
                
                return common.CallResult("", "")
            case _ :
                return common.CallResult(None, "BlobCreate invalid resp " + resp)
            
    async def BlobOpen(self, addr: BlobAddr) -> common.CallResult:
        msgId = self.MsgId()
        req = func_pb2.BlobOpenReq (
            svcAddr = addr.blobSvcAddr,
            namespace = "",
            name = addr.name
        )
        msg = func_pb2.FuncAgentMsg (
            msgId = msgId,
            BlobOpenReq = req
        )
        
        queue = asyncio.Queue(1)
        self.blobReqs[msgId] = queue
        self.BlobReq(msg)
        msgResp = await queue.get()
        msgType = msgResp.WhichOneof('EventBody')
        match msgType:
            case 'BlobOpenResp':
                resp = msgResp.BlobOpenResp
                if resp.error != "" :
                    return common.CallResult(None, resp.error)
                
                blob = Blob(
                    resp.id,
                    addr,
                    resp.size, 
                    resp.checksum,
                    TimestampToDateTime(resp.createTime),
                    TimestampToDateTime(resp.lastAccessTime)
                    )
                return common.CallResult(blob, "")
            case _ :
                return common.CallResult(None, "BlobCreate invalid resp " + msgResp)
            
    async def BlobDelete(self, svcAddr: str, name: str) -> common.CallResult:
        msgId = self.MsgId()
        req = func_pb2.BlobDeleteReq (
            svcAddr = svcAddr,
            name = name
        )
        msg = func_pb2.FuncAgentMsg (
            msgId = msgId,
            BlobDeleteReq = req
        )
        
        queue = asyncio.Queue(1)
        self.blobReqs[msgId] = queue
        self.BlobReq(msg)
        msgResp = await queue.get()
        msgType = msgResp.WhichOneof('EventBody')
        match msgType:
            case 'BlobDeleteResp':
                resp = msgResp.BlobDeleteResp
                if resp.error != "" :
                    return common.CallResult(None, resp.error)
                
                return common.CallResult(None, "")
            case _ :
                return common.CallResult(None, "BlobDelete invalid resp " + msgResp)
            
    async def BlobRead(self, id: np.uint64, len: np.uint64) -> common.CallResult:
        msgId = self.MsgId()
        req = func_pb2.BlobReadReq (
            id = id,
            len = len
        )
        msg = func_pb2.FuncAgentMsg (
            msgId = msgId,
            BlobReadReq = req
        )
        
        queue = asyncio.Queue(1)
        self.blobReqs[msgId] = queue
        self.BlobReq(msg)
        msgResp = await queue.get()
        msgType = msgResp.WhichOneof('EventBody')
        match msgType:
            case 'BlobReadResp':
                resp = msgResp.BlobReadResp
                if resp.error != "" :
                    return common.CallResult(None, resp.error)
                
                return common.CallResult(resp.data, "")
            case _ :
                return common.CallResult(None, "BlobRead invalid resp " + msgResp)
        
    async def BlobSeek(self, id: np.uint64, seekType: int, pos: np.int64) -> common.CallResult:
        msgId = self.MsgId()
        req = func_pb2.BlobSeekReq (
            id = id,
            seekType = seekType,
            pos = pos
        )
        msg = func_pb2.FuncAgentMsg (
            msgId = msgId,
            BlobSeekReq = req
        )
        
        queue = asyncio.Queue(1)
        self.blobReqs[msgId] = queue
        self.BlobReq(msg)
        msgResp = await queue.get()
        msgType = msgResp.WhichOneof('EventBody')
        match msgType:
            case 'BlobReadReq':
                resp = msgResp.BlobSeekResp
                if resp.error != "" :
                    return common.CallResult(None, resp.error)
                
                return common.CallResult(resp.offset, "")
            case _ :
                return common.CallResult(None, "BlobSeek invalid resp " + msgResp)
            
    async def BlobClose(self, id: np.uint64) -> common.CallResult:
        msgId = self.MsgId()
        req = func_pb2.BlobCloseReq (
            id = id
        )
        msg = func_pb2.FuncAgentMsg (
            msgId = msgId,
            BlobCloseReq = req
        )
        
        queue = asyncio.Queue(1)
        self.blobReqs[msgId] = queue
        self.BlobReq(msg)
        msgResp = await queue.get()
        msgType = msgResp.WhichOneof('EventBody')
        match msgType:
            case 'BlobCloseResp':
                resp = msgResp.BlobCloseResp
                if resp.error != "" :
                    return common.CallResult(None, resp.error)
                
                return common.CallResult(None, "")
            case _ :
                return common.CallResult(None, "BlobClose invalid resp " + msgResp)
        
