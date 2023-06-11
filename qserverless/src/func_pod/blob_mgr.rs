// Copyright (c) 2021 Quark Container Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::time::SystemTime;
use std::io::SeekFrom;
use core::ops::Deref;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use qobjs::utility::SystemTimeProto;
use tokio::sync::oneshot;

use qobjs::func;
use qobjs::common::*;

use crate::BLOB_MGR;
use crate::FUNC_AGENT_CLIENT;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BlobAddr {
    pub blobSvcAddr: String,
    pub name: String, 
}

#[derive(Debug)]
pub struct BlobInner {
    pub id: u64,
    pub addr: BlobAddr,
    pub size: usize,
    pub checksum: String,
    pub createTime: SystemTime,
    pub lastAccessTime: SystemTime,
    pub closed: AtomicBool,
}

impl Drop for BlobInner {
    fn drop(&mut self) {
        futures::executor::block_on(self.Close()).ok();
    }
}

impl BlobInner {
    pub async fn Close(&self) -> Result<()> {
        if self.closed.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return BLOB_MGR.BlobClose(self.id).await;
        }

        return Ok(())
    }
}

#[derive(Debug)]
pub struct Blob(Arc<BlobInner>);

impl Deref for Blob {
    type Target = Arc<BlobInner>;

    fn deref(&self) -> &Arc<BlobInner> {
        &self.0
    }
}

impl Blob {
    pub async fn Read(&self, len: usize) -> Result<Vec<u8>> {
        return BLOB_MGR.BlobRead(self.id, len).await;
    }

    pub async fn Seek(&self, pos: SeekFrom) -> Result<u64> {
        return BLOB_MGR.BlobSeek(self.id, pos).await;
    }

    pub async fn Close(&self) -> Result<()> {
        return self.0.Close().await;
    }
}

pub struct UnsealBlobInner {
    pub id: u64,
    pub addr: BlobAddr, 
    pub closed: AtomicBool,
}

impl Drop for UnsealBlobInner {
    fn drop(&mut self) {
        futures::executor::block_on(self.Close()).ok();
    }
}

impl UnsealBlobInner {
    pub async fn Close(&self) -> Result<()> {
        if self.closed.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return BLOB_MGR.BlobClose(self.id).await;
        }

        return Ok(())
    }
}

pub struct UnsealBlob(Arc<UnsealBlobInner>);

impl Deref for UnsealBlob {
    type Target = Arc<UnsealBlobInner>;

    fn deref(&self) -> &Arc<UnsealBlobInner> {
        &self.0
    }
}

impl UnsealBlob {
    pub async fn Write(&self, buf: Vec<u8>) -> Result<()> {
        return BLOB_MGR.BlobWrite(self.id, buf).await;
    }

    pub async fn Close(&self) -> Result<()> {
        return self.0.Close().await;
    }
}

#[derive(Debug, Default)]
pub struct BlobMgr {
    pub blobReqs: Mutex<BTreeMap<u64, oneshot::Sender<func::FuncAgentMsg>>>,
    pub lastMsgId: AtomicU64,
}

impl BlobMgr {
    pub fn BlobReq(&self, msg: func::FuncAgentMsg) {
        FUNC_AGENT_CLIENT.get().unwrap().Send(msg);
    }

    pub fn OnFuncAgentMsg(&self, msg: func::FuncAgentMsg) -> Result<()> {
        let msgId = msg.msg_id;
        match self.blobReqs.lock().unwrap().remove(&msgId) {
            None => {
                return Err(Error::ENOENT(format!("BlobMgr:: can't find blog req {:?}", msg)));
            }
            Some(s) => {
                s.send(msg).unwrap();
                return Ok(())
            }
        }
    } 

    pub fn MsgId(&self) -> u64 {
        return self.lastMsgId.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
    }

    pub async fn BlobCreate(&self, name: &str) -> Result<UnsealBlob> {
        let msgId = self.MsgId();
        let req = func::BlobCreateReq {
            namespace: String::new(),
            name: name.to_string(),
        };

        let msg = func::FuncAgentMsg {
            msg_id: msgId,
            event_body: Some(func::func_agent_msg::EventBody::BlobCreateReq(req)),
        };

        let (tx, rx) = oneshot::channel();
        self.blobReqs.lock().unwrap().insert(msgId, tx);
        self.BlobReq(msg);
        let resp = rx.await.unwrap();
        match resp.event_body.unwrap() {
            func::func_agent_msg::EventBody::BlobCreateResp(resp) => {
                if resp.error.len() != 0 {
                    return Err(Error::EINVAL(resp.error))
                }
                return Ok(UnsealBlob(Arc::new(UnsealBlobInner { 
                    id: resp.id, 
                    addr: BlobAddr { 
                        blobSvcAddr: resp.svc_addr, 
                        name: name.to_owned(), 
                    },
                    closed: AtomicBool::new(false),
                })));
            }
            b => {
                return Err(Error::EINVAL(format!("BlobCreate invalid resp {:?}", b)));
            }
        }
    }

    pub async fn BlobWrite(&self, id: u64, buf: Vec<u8>) -> Result<()> {
        let msgId = self.MsgId();
        let req = func::BlobWriteReq {
            id: id,
            data: buf,
        };

        let msg = func::FuncAgentMsg {
            msg_id: msgId,
            event_body: Some(func::func_agent_msg::EventBody::BlobWriteReq(req)),
        };

        let (tx, rx) = oneshot::channel();
        self.blobReqs.lock().unwrap().insert(msgId, tx);
        self.BlobReq(msg);
        let resp = rx.await.unwrap();
        match resp.event_body.unwrap() {
            func::func_agent_msg::EventBody::BlobWriteResp(resp) => {
                if resp.error.len() != 0 {
                    return Err(Error::EINVAL(resp.error))
                }
                return Ok(());
            }
            b => {
                return Err(Error::EINVAL(format!("BlobWrite invalid resp {:?}", b)));
            }
        }
    } 

    pub async fn BlobOpen(&self, addr: &BlobAddr) -> Result<Blob> {
        let msgId = self.MsgId();
        let req = func::BlobOpenReq {
            svc_addr: addr.blobSvcAddr.clone(),
            namespace: String::new(),
            name: addr.name.clone(),
        };

        let msg = func::FuncAgentMsg {
            msg_id: msgId,
            event_body: Some(func::func_agent_msg::EventBody::BlobOpenReq(req)),
        };

        let (tx, rx) = oneshot::channel();
        self.blobReqs.lock().unwrap().insert(msgId, tx);
        self.BlobReq(msg);
        let resp = rx.await.unwrap();
        match resp.event_body.unwrap() {
            func::func_agent_msg::EventBody::BlobOpenResp(resp) => {
                if resp.error.len() != 0 {
                    return Err(Error::EINVAL(resp.error))
                }
                return Ok(Blob(Arc::new(BlobInner {
                    id: resp.id,
                    addr: addr.clone(),
                    size: resp.size as usize,
                    checksum: resp.checksum,
                    createTime: SystemTimeProto::FromTimestamp(resp.create_time.as_ref().unwrap()).ToSystemTime(),
                    lastAccessTime: SystemTimeProto::FromTimestamp(resp.last_access_time.as_ref().unwrap()).ToSystemTime(),
                    closed: AtomicBool::new(false),
                })));
            }
            b => {
                return Err(Error::EINVAL(format!("BlobOpen invalid resp {:?}", b)));
            }
        }
    }

    pub async fn BlobDelete(&self, svcAddr: &str, name: &str) -> Result<()> {
        let msgId = self.MsgId();
        let req = func::BlobDeleteReq {
            svc_addr: svcAddr.to_string(),
            namespace: String::new(),
            name: name.to_string(),
        };

        let msg = func::FuncAgentMsg {
            msg_id: msgId,
            event_body: Some(func::func_agent_msg::EventBody::BlobDeleteReq(req)),
        };

        let (tx, rx) = oneshot::channel();
        self.blobReqs.lock().unwrap().insert(msgId, tx);
        self.BlobReq(msg);
        let resp = rx.await.unwrap();
        match resp.event_body.unwrap() {
            func::func_agent_msg::EventBody::BlobDeleteResp(resp) => {
                if resp.error.len() != 0 {
                    return Err(Error::EINVAL(resp.error))
                }
                return Ok(());
            }
            b => {
                return Err(Error::EINVAL(format!("BlobOpen invalid resp {:?}", b)));
            }
        }
    }

    pub async fn BlobRead(&self, id: u64, len: usize) -> Result<Vec<u8>> {
        let msgId = self.MsgId();
        let req = func::BlobReadReq {
            id: id,
            len: len as u64,
        };

        let msg = func::FuncAgentMsg {
            msg_id: msgId,
            event_body: Some(func::func_agent_msg::EventBody::BlobReadReq(req)),
        };

        let (tx, rx) = oneshot::channel();
        self.blobReqs.lock().unwrap().insert(msgId, tx);
        self.BlobReq(msg);
        let resp = rx.await.unwrap();
        match resp.event_body.unwrap() {
            func::func_agent_msg::EventBody::BlobReadResp(resp) => {
                if resp.error.len() != 0 {
                    return Err(Error::EINVAL(resp.error))
                }
                return Ok(resp.data);
            }
            b => {
                return Err(Error::EINVAL(format!("BlobRead invalid resp {:?}", b)));
            }
        }
    } 

    pub async fn BlobSeek(&self, id: u64, pos: SeekFrom) -> Result<u64> {
        let msgId = self.MsgId();
        let req = match pos {
            SeekFrom::Start(pos) => {
                func::BlobSeekReq {
                    id: id,
                    seek_type: 1,
                    pos: pos as i64,
                }
            }
            SeekFrom::End(pos) => {
                func::BlobSeekReq {
                    id: id,
                    seek_type: 2,
                    pos: pos as i64,
                }
            }
            SeekFrom::Current(pos) => {
                func::BlobSeekReq {
                    id: id,
                    seek_type: 3,
                    pos: pos as i64,
                }
            }
        };

        let msg = func::FuncAgentMsg {
            msg_id: msgId,
            event_body: Some(func::func_agent_msg::EventBody::BlobSeekReq(req)),
        };

        let (tx, rx) = oneshot::channel();
        self.blobReqs.lock().unwrap().insert(msgId, tx);
        self.BlobReq(msg);
        let resp = rx.await.unwrap();
        match resp.event_body.unwrap() {
            func::func_agent_msg::EventBody::BlobSeekResp(resp) => {
                if resp.error.len() != 0 {
                    return Err(Error::EINVAL(resp.error))
                }
                return Ok(resp.offset);
            }
            b => {
                return Err(Error::EINVAL(format!("BlobSeek invalid resp {:?}", b)));
            }
        }
    } 

    pub async fn BlobClose(&self, id: u64) -> Result<()> {
        let msgId = self.MsgId();
        let req = func::BlobCloseReq {
            id: id,
        };

        let msg = func::FuncAgentMsg {
            msg_id: msgId,
            event_body: Some(func::func_agent_msg::EventBody::BlobCloseReq(req)),
        };

        let (tx, rx) = oneshot::channel();
        self.blobReqs.lock().unwrap().insert(msgId, tx);
        self.BlobReq(msg);
        let resp = rx.await.unwrap();
        match resp.event_body.unwrap() {
            func::func_agent_msg::EventBody::BlobCloseResp(resp) => {
                if resp.error.len() != 0 {
                    return Err(Error::EINVAL(resp.error))
                }
                return Ok(());
            }
            b => {
                return Err(Error::EINVAL(format!("BlobSeek invalid resp {:?}", b)));
            }
        }
    } 
}