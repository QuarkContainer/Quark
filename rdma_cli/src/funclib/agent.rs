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

use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;
use alloc::sync::Arc;
use spin::Mutex;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use core::ops::Deref;
use lazy_static::lazy_static;

use tokio::sync::mpsc;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::mpsc::Receiver;
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot;
use tokio::sync::Mutex as TMutex;

use crate::qlib::common::*;
use crate::qlib::linux_def::*;
use crate::qlib::bytestream::*;

use crate::funclib::msg_stream::MsgStream;
use crate::funclib::message::*;

lazy_static! {
    pub static ref MSG_CENTER: MsgCenter = MsgCenter::New();
}

#[derive(Debug)]
pub struct QRequest {
    pub channelId: u64,
    pub msg: QMsg,
}

impl QRequest {
    pub fn New(channelId: u64, msg: QMsg) -> Self {
        return Self {
            channelId: channelId,
            msg: msg,
        }
    }
}

#[derive(Clone)]
pub struct MsgCenter(Arc<MsgCenterInner>);

impl Deref for MsgCenter {
    type Target = Arc<MsgCenterInner>;

    fn deref(&self) -> &Arc<MsgCenterInner> {
        &self.0
    }
}

impl MsgCenter {
    pub fn New() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let inner = MsgCenterInner {
            nextId: AtomicU64::new(1),
            channels: Mutex::new(HashMap::new()),
            reqTx: tx,
            reqRx: Mutex::new(Some(rx)),
        };

        return Self(Arc::new(inner));
    }

    pub fn GetChannel(&self, channelId: u64) -> Result<MsgChannel> {
        match self.channels.lock().get(&channelId) {
            Some(c) => return Ok(c.clone()),
            None => return Err(Error::SysError(SysErr::ENOENT)),
        }
    }

    pub fn AddChannel(&self, stream: MsgStream) -> Result<u64> {
        let id = self.nextId.fetch_add(1, Ordering::SeqCst);
        let channel = MsgChannel::New(id, stream, self.reqTx.clone());
        self.channels.lock().insert(id, channel);
        return Ok(id)
    }

    pub fn RemoveChannel(&self, id: u64) {
        self.channels.lock().remove(&id);
    }
}

pub struct MsgCenterInner {
    pub nextId: AtomicU64,
    pub channels : Mutex<HashMap<u64, MsgChannel>>,
    pub reqTx: UnboundedSender<QRequest>,
    pub reqRx: Mutex<Option<UnboundedReceiver<QRequest>>>,
}

pub struct AgentAddr {
    pub ipAddr: u32,
    pub port: u16,
}

pub struct MsgChannelInner {
    pub id: u64,
    pub nextMessageId: AtomicU64,
    //pub stream: MsgStream,
    pub pendingCalls: Mutex<HashMap<u64, oneshot::Sender<FuncResp>>>,

    pub msgTx: Sender<QMsg>,
}

#[derive(Clone)]
pub struct MsgChannel(Arc<MsgChannelInner>);

impl Deref for MsgChannel {
    type Target = Arc<MsgChannelInner>;

    fn deref(&self) -> &Arc<MsgChannelInner> {
        &self.0
    }
}

impl MsgChannel {
    pub fn New(channelId: u64, stream: MsgStream, reqTx: UnboundedSender<QRequest>) -> Self {
        let (tx, mut rx) = mpsc::channel(128);
        let inner = MsgChannelInner {
            id: channelId,
            nextMessageId: AtomicU64::new(1),
            pendingCalls: Mutex::new(HashMap::new()),
            msgTx: tx,
        };

        let channel = Self(Arc::new(inner));
        let channel2 = channel.clone();
        let stream1 = stream.clone();
        let stream2 = stream.clone();
        
        tokio::spawn(async move {
            loop {
                let req = rx.recv().await.unwrap();
                match stream1.WriteMsg(&req).await {
                    Ok(_) => (),
                    Err(_) => break,
                }
            }
            
            MSG_CENTER.RemoveChannel(channelId);
        });

        tokio::spawn(async move {
            loop {
                let msg = match stream2.ReadMsg().await {
                    Ok(msg) => msg,
                    Err(_) => break,
                };

                let messageId = msg.messageId;
                if let MsgPayload::FuncResp(resp) = msg.payload {
                    let respSender = channel2.pendingCalls.lock().remove(&messageId);
                    match respSender {
                        Some(sender) => {
                            match sender.send(resp) {
                                Ok(_) => (),
                                Err(_) => (),
                            }
                        }
                        None => {
                            info!("ResponseSwitch: get unexpect message {}", messageId);
                        }
                    }
                } else {
                    // forward all func call to central prcessing
                    reqTx.send(QRequest { channelId: channelId, msg: msg }).unwrap();
                }

                MSG_CENTER.RemoveChannel(channelId);
            }
            
        });

        return channel;
    }

    pub async fn SendRequest(&self, notify: oneshot::Sender<FuncResp>, request: MsgPayload) -> Result<()> {
        let messageId = self.nextMessageId.fetch_add(1, Ordering::SeqCst);
        let msg = QMsg {
            messageId: messageId,
            payload: request,
        };
        self.pendingCalls.lock().insert(messageId, notify);

        match self.msgTx.send(msg).await {
            Ok(_) => (),
            Err(_) => {
                // todo: handle downstream full
                panic!("MsgChannelInner::SendFuncRequest fail");
            }
        }
        return Ok(())
    }

    pub async fn SendResp(&self, requestId: u64, resp: FuncResp) {
        let msg = QMsg::NewMsg(requestId, MsgPayload::FuncResp(resp));
        match self.msgTx.send(msg).await {
            Ok(_) => (),
            Err(e) => {
                // todo: handle downstream full
                error!("MsgChannelInner::SendResp fail: {:?}", e);
            }
        }
    }

    pub async fn Drop(&self) {
        for (_reqId, sender) in self.pendingCalls.lock().drain() {
            let errorResp = FuncResp::NewErr(HTTP_INTERN_ERR);
            sender.send(errorResp).unwrap();
        }
    }
}

pub struct FuncInstance {
    pub id: u64,
    pub connection: MsgStream,
}

pub struct FuncAgentPeer {
    pub id: u64,
    pub connection: MsgStream,
}

pub struct FuncAgentId {
    pub NodeId: u32,
    pub Pod: u16
}

pub struct NamespaceId(u64);

pub struct AppId(u64);

pub fn GetFuncHandler(_namespaceId: NamespaceId, _appId: AppId, _funcName: String) -> Result<FuncAgentId> {
    todo!();
}
