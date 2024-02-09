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

use std::sync::Arc;
use std::sync::atomic::*;
use core::ops::Deref;

use tokio::io::Interest;
use tokio::net::UnixStream;
use tokio::sync::Notify;
use tokio::sync::mpsc;

use qshare::common::*;
use qshare::databuf::*;

enum AgentState {
    PayloadLen(PayloadLen),
    Payload(Payload),
}

#[derive(Debug, Default)]
pub struct PayloadLen {
    pub bytes: [u8; 8],
    pub offset: usize,
}

impl PayloadLen {
    pub fn Size(&self) -> usize {
        return unsafe {
            *(&self.bytes[0] as * const _ as u64 as * const usize)
        }
    }
}

#[derive(Debug, Default)]
pub struct Payload {
    pub bytes: Vec<u8>,
    pub offset: usize,
}

impl Payload {
    pub fn New(len: usize) -> Self {
        let mut bytes = Vec::with_capacity(len);
        bytes.resize(len, 0);
        return Self {
            bytes: bytes,
            offset: 0,
        }
    }
}

pub struct TsotAgentInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub stream: UnixStream,
    pub tsotSvc: mpsc::Sender<TsotMsg>, 
    pub agentChann: mpsc::Sender<TsotMsg>,
}

#[derive(Clone)]
pub struct TsotAgent(Arc<TsotAgentInner>);

impl Deref for TsotAgent {
    type Target = Arc<TsotAgentInner>;

    fn deref(&self) -> &Arc<TsotAgentInner> {
        &self.0
    }
}

impl TsotAgent {
    pub fn New(tsotSvc: mpsc::Sender<TsotMsg>, stream: UnixStream) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<TsotMsg>(30);

        let inner = TsotAgentInner{
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),

            stream: stream,
            tsotSvc: tsotSvc,
            agentChann: tx,
        };

        let ret = Self(Arc::new(inner));

        let clone = ret.clone();

        tokio::spawn(async move {
            clone.Process(rx).await.unwrap();
        });

        return Ok(ret)
    }

    pub async fn Process(&self, mut rx: mpsc::Receiver<TsotMsg>) -> Result<()> {
        let mut state = AgentState::PayloadLen(PayloadLen::default());
        
        loop {
            let ready = self.stream.ready(
                Interest::READABLE | 
                Interest::WRITABLE |
                Interest::ERROR
            );

            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.stop.store(false, Ordering::SeqCst);
                    break;
                }
                _msg = rx.recv() => {

                }
                _ = ready => {
                    match &mut state {
                        AgentState::PayloadLen(l) => {
                            let offset = l.offset;
                            match self.stream.try_read(&mut l.bytes[offset..]) {
                                Ok(n) => {
                                    l.offset += n;
                                    if l.offset == 8 {
                                        let len = l.Size();
                                        state = AgentState::Payload(Payload::New(len));
                                    }
                                }
                                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                                    continue;
                                }
                                Err(e) => {
                                    return Err(e.into());
                                }
                            }
                        }
                        AgentState::Payload(p) => {
                            let offset = p.offset;
                            match self.stream.try_read(&mut p.bytes[offset..]) {
                                Ok(n) => {
                                    p.offset += n;
                                    if p.offset == p.bytes.len() {
                                        
                                        state = AgentState::PayloadLen(PayloadLen::default());
                                    }
                                }
                                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                                    continue;
                                }
                                Err(e) => {
                                    return Err(e.into());
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return Ok(())
    }
}