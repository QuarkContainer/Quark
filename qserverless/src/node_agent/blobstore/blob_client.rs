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

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::ops::Deref;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use futures::channel::oneshot;
use qobjs::utility::SystemTimeProto;
use tokio::sync::Notify;
use tokio::sync::mpsc;
use qobjs::func;
use qobjs::common::*;

use crate::BLOB_SVC_CLIENT_MGR;

use super::blob::Blob;
use super::blob::BlobInner;
use super::blob::BlobState;
use super::blob::RemoteReadBlob;

#[derive(Debug, Default)]
pub struct BlobSvcClientMgr {
    pub clients: Mutex<BTreeMap<String, BlobSvcClient>>,
}

impl BlobSvcClientMgr {
    pub async fn Open(&self, addr: &str, namespace: &str, name: &str) -> Result<RemoteReadBlob> {
        let client = self.Get(addr).await?;
        let (id, blob) = client.Open(addr, namespace, name).await?;
        return Ok(RemoteReadBlob {
            id: id,
            blobSvcAddr: addr.to_string(),
            blob: blob,
            closed: false,
        });
    }

    pub async fn Delete(&self, addr: &str, namespace: &str, name: &str) -> Result<()> {
        let client = self.Get(addr).await?;
        client.Delete(addr, namespace, name).await?;
        return Ok(());
    }

    pub async fn Read(&self, addr: &str, id: u64, len: usize) -> Result<Vec<u8>> {
        let client = self.Get(addr).await?;
        let data = client.Read(id, len).await?;
        return Ok(data);
    }

    pub async fn Seek(&self, addr: &str, id: u64, seekType: u32, pos: i64) -> Result<u64> {
        let client = self.Get(addr).await?;
        let offset = client.Seek(id, pos, seekType).await?;
        return Ok(offset);
    }

    pub async fn Close(&self, addr: &str, id: u64) -> Result<()> {
        let client = self.Get(addr).await?;
        client.Close(id).await?;
        return Ok(());
    }

    pub async fn Get(&self, addr: &str) -> Result<BlobSvcClient> {
        let client = self.clients.lock().unwrap().get(addr).cloned();
        match client {
            Some(c) => {
                return Ok(c)
            }
            None => ()
        };

        let client = BlobSvcClient::Init(addr).await?;
        let mut clients = self.clients.lock().unwrap();
        match clients.get(addr).cloned() {
            None => {
                clients.insert(addr.to_string(), client.clone());
                return Ok(client);
            }
            Some(c) => {
                return Ok(c)
            }
        }
    }

    pub fn Remove(&self, addr: &str) {
        self.clients.lock().unwrap().remove(addr);
    }
}

#[derive(Debug)]
pub struct BlobSvcClientInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub svcAddr: String,
    pub lastMsgId: AtomicU64,
    pub calls: Mutex<BTreeMap<u64, oneshot::Sender<func::BlobSvcResp>>>,
    pub agentChann: mpsc::Sender<func::BlobSvcReq>,
    pub stream: Arc<Mutex<Option<tonic::Streaming<func::BlobSvcResp>>>>,
}

#[derive(Debug, Clone)]
pub struct BlobSvcClient(Arc<BlobSvcClientInner>);

impl Deref for BlobSvcClient {
    type Target = Arc<BlobSvcClientInner>;

    fn deref(&self) -> &Arc<BlobSvcClientInner> {
        &self.0
    }
}

impl BlobSvcClient {
    pub async fn Init(blobSvcAddr: &str) -> Result<Self> {
        let mut client = match func::blob_service_client::BlobServiceClient::connect(blobSvcAddr.to_string()).await {
            Ok(c) => {
                c
            }
            Err(e) => {
                return Err(Error::ECONNREFUSED(format!("BlobSvcClient connect {} fail with error {:?}", 
                    blobSvcAddr, e)));
            }
        };

        let (tx, rx) = mpsc::channel(30);
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let response = client.stream_process(stream).await?;
    
        let stream = response.into_inner();

        let inner = BlobSvcClientInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            svcAddr: blobSvcAddr.to_string(),
            lastMsgId: AtomicU64::new(0),
            calls: Mutex::new(BTreeMap::new()),
            agentChann: tx,
            stream: Arc::new(Mutex::new(Some(stream))),
        };

        let client = BlobSvcClient(Arc::new(inner));
        return Ok(client);
    }

    pub async fn Open(&self, svcAddr: &str, namespace: &str, name: &str) -> Result<(u64, Blob)> {
        let msgId = self.MsgId();
        let req = func::BlobOpenReq {
            svc_addr: svcAddr.to_string(),
            namespace: namespace.to_string(),
            name: name.to_string(),
        };

        let req = func::BlobSvcReq {
            msg_id: msgId,
            event_body: Some(func::blob_svc_req::EventBody::BlobOpenReq(req)),
        };

        let resp = self.Call(req).await?;

        match resp.event_body.unwrap() {
            func::blob_svc_resp::EventBody::BlobOpenResp(resp) => {
                if resp.error.len() > 0 {
                    return Err(Error::CommonError(format!("{}", resp.error))); 
                }
                let inner = BlobInner {
                    namespace: resp.namespace,
                    name: resp.name,
                    size: resp.size as usize,
                    checksum: resp.checksum,
                    state: BlobState::Sealed,
                    createTime: SystemTimeProto::FromTimestamp(resp.create_time.as_ref().unwrap()).ToSystemTime(),
                    lastAccessTime: SystemTimeProto::FromTimestamp(resp.last_access_time.as_ref().unwrap()).ToSystemTime(),
                };

                let blob = Blob(Arc::new(Mutex::new(inner)));
                return Ok((resp.id, blob)); 
            }
            msg => {
                return Err(Error::EINVAL(format!("Invalid Call response {:?}", msg)))
            }
        }
    }

    pub async fn Delete(&self, svcAddr: &str, namespace: &str, name: &str) -> Result<()> {
        let msgId = self.MsgId();
        let req = func::BlobDeleteReq {
            svc_addr: svcAddr.to_string(),
            namespace: namespace.to_string(),
            name: name.to_string(),
        };

        let req = func::BlobSvcReq {
            msg_id: msgId,
            event_body: Some(func::blob_svc_req::EventBody::BlobDeleteReq(req)),
        };

        let resp = self.Call(req).await?;

        match resp.event_body.unwrap() {
            func::blob_svc_resp::EventBody::BlobDeleteResp(resp) => {
                if resp.error.len() > 0 {
                    return Err(Error::CommonError(format!("{}", resp.error))); 
                }

                return Ok(()); 
            }
            msg => {
                return Err(Error::EINVAL(format!("Invalid Call response {:?}", msg)))
            }
        }
    }

    pub async fn Read(&self, id: u64, len: usize) -> Result<Vec<u8>> {
        let msgId = self.MsgId();
        let req = func::BlobReadReq {
            id: id,
            len: len as u64,
        };

        let req = func::BlobSvcReq {
            msg_id: msgId,
            event_body: Some(func::blob_svc_req::EventBody::BlobReadReq(req)),
        };

        let resp = self.Call(req).await?;

        match resp.event_body.unwrap() {
            func::blob_svc_resp::EventBody::BlobReadResp(resp) => {
                if resp.error.len() == 0 {
                    return Ok(resp.data); 
                } else {
                    return Err(Error::CommonError(format!("{}", resp.error)))
                }
                
            }
            msg => {
                return Err(Error::EINVAL(format!("Invalid Call response {:?}", msg)))
            }
        }
    }

    pub async fn Seek(&self, id: u64, pos: i64, seekType: u32) -> Result<u64> {
        let msgId = self.MsgId();
        let req = func::BlobSeekReq {
            id: id,
            pos: pos,
            seek_type: seekType,
        };

        let req = func::BlobSvcReq {
            msg_id: msgId,
            event_body: Some(func::blob_svc_req::EventBody::BlobSeekReq(req)),
        };

        let resp = self.Call(req).await?;

        match resp.event_body.unwrap() {
            func::blob_svc_resp::EventBody::BlobSeekResp(resp) => {
                if resp.error.len() == 0 {
                    return Ok(resp.offset); 
                } else {
                    return Err(Error::CommonError(format!("{}", resp.error)))
                }
                
            }
            msg => {
                return Err(Error::EINVAL(format!("Invalid Call response {:?}", msg)))
            }
        }
    }

    pub async fn Close(&self, id: u64) -> Result<()> {
        let msgId = self.MsgId();
        let req = func::BlobCloseReq {
            id: id,
        };

        let req = func::BlobSvcReq {
            msg_id: msgId,
            event_body: Some(func::blob_svc_req::EventBody::BlobCloseReq(req)),
        };

        let resp = self.Call(req).await?;

        match resp.event_body.unwrap() {
            func::blob_svc_resp::EventBody::BlobCloseResp(resp) => {
                if resp.error.len() == 0 {
                    return Ok(()); 
                } else {
                    return Err(Error::CommonError(format!("{}", resp.error)))
                }
                
            }
            msg => {
                return Err(Error::EINVAL(format!("Invalid Call response {:?}", msg)))
            }
        }
    }

    pub async fn Call(&self, req: func::BlobSvcReq) -> Result<func::BlobSvcResp> {
        let (tx, rx) = oneshot::channel();
        self.calls.lock().unwrap().insert(req.msg_id, tx);
        match self.agentChann.send(req).await {
            Ok(()) => (),
            Err(_) => {
                self.Stop();
                return Err(Error::ECONNREFUSED(format!("Call fail")))
            }
        }

        match rx.await {
            Err(_) => {
                self.Stop();
                return Err(Error::ECONNREFUSED(format!("Call fail")))
            }
            Ok(resp) => {
                return Ok(resp)
            }
        }
    }

    pub fn Stop(&self) {
        if !self.stop.swap(true, std::sync::atomic::Ordering::SeqCst) {
            BLOB_SVC_CLIENT_MGR.Remove(&self.svcAddr);
            self.closeNotify.notify_waiters();
            self.calls.lock().unwrap().clear();
        }
    }

    pub fn MsgId(&self) -> u64 {
        return self.lastMsgId.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
    }

    pub async fn OnBlobSvcResp(&self, resp: func::BlobSvcResp) -> Result<()> {
        let notify = match self.calls.lock().unwrap().remove(&resp.msg_id) {
            None => {
                error!("BlobSvcClient get non exist msg call response {}", resp.msg_id);
                return Ok(());
            }
            Some(n) => n,
        };

        notify.send(resp).unwrap();
        return Ok(())
    }

    pub async fn Process(&self) -> Result<()> {
        let mut stream = self.stream.lock().unwrap().take().unwrap();
        loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.stop.store(false, std::sync::atomic::Ordering::SeqCst);
                    break;
                }
                msg = stream.message() => {
                    let msg : func::BlobSvcResp = match msg {
                        Err(e) => {
                            error!("BlobSvcClient get error message {:?}", e);
                            self.stop.store(false, std::sync::atomic::Ordering::SeqCst);
                            break;
                        }
                        Ok(m) => {
                            match m {
                                None => {
                                    error!("BlobSvcClient get None message");
                                    self.stop.store(false, std::sync::atomic::Ordering::SeqCst);
                                    break;
                                }
                                Some(m) => m,
                            }
                        }
                    };

                    self.OnBlobSvcResp(msg).await?;
                }
            }
        }

        return Ok(())
    }
}