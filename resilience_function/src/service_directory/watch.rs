// Copyright (c) 2021 Quark Container Authors / 2014 The Kubernetes Authors
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

use etcd_client::Event;
use etcd_client::GetOptions;
use etcd_client::WatchOptions;
use tokio::sync::mpsc;
use tokio::sync::Notify;
use std::sync::Mutex;
use std::sync::atomic::AtomicI64;
use prost::Message;

use crate::etcd_client::EtcdClient;
use crate::selection_predicate::SelectionPredicate;
use crate::shared::common::*;
use crate::service_directory::*;

use super::etcd_store::*;

#[derive(Debug)]
pub enum EventType {
    Added,
    Modified,
    Deleted,
    Error(String),
}

#[derive(Debug)]
pub struct WatchEvent {
    pub type_: EventType,

    // Object is:
	//  * If Type is Added or Modified: the new state of the object.
	//  * If Type is Deleted: the state of the object immediately before deletion.
	//  * If Type is Error:
    pub obj: DataObject,
}

pub struct TmpEvent {
    pub key: String,
    pub value: Vec<u8>,
    pub preValue: Option<Vec<u8>>,
    pub rev: i64,
    pub isDeleted: bool,
    pub isCreated: bool,
}

pub struct Watcher {
    pub client: EtcdClient,
    pub resultRecv: mpsc::Receiver<WatchEvent>,
    pub resultSender: Mutex<Option<mpsc::Sender<WatchEvent>>>,
    pub closeNotify: Notify,

    pub key: String,
    pub initialRev: AtomicI64,
    pub internalPred: SelectionPredicate,
}

impl Watcher {
    pub const RESULT_CHANNEL_SIZE: usize = 300;

    pub fn New(client: &EtcdClient, key: &str, rev: i64, pred: SelectionPredicate) -> Self {
        let (resultSender, resultRecv) = mpsc::channel(Self::RESULT_CHANNEL_SIZE);
        return Self {
            client: client.clone(),
            resultRecv: resultRecv,
            resultSender: Mutex::new(Some(resultSender)),
            closeNotify: Notify::new(),

            key: key.to_string(),
            initialRev: AtomicI64::new(rev),
            internalPred: pred,
        }
    }

    pub async fn Processing(&self) -> Result<()> {
        let mut sender = self.resultSender.lock().unwrap().take().unwrap();
        
        let initialRev = self.initialRev.load(std::sync::atomic::Ordering::Relaxed);
        if initialRev == 0 {
            let ret = self.Sync(&mut sender).await;
            match ret {
                Err(e) => self.SendErr(&mut sender, format!("failed to sync with latest state: {:?}", e)).await.unwrap(),
                Ok(()) => (),
            }
        }

        let options = WatchOptions::new()
                    .with_start_revision(initialRev + 1)
                    .with_prev_key();

        let key: &str = &self.key;
        let (_watcher, mut stream) = self.client.lock().await.watch(key, Some(options)).await?;

        while let Some(resp) = stream.message().await? {
            for e in resp.events() {
                match self.ParseEvent(e) {
                    Err(e) => self.SendErr(&mut sender, format!("{:?}", e)).await?,
                    Ok(event) => {
                        match event {
                            None => (),
                            Some(event) => {
                                self.SendEvent(&mut sender, event).await?;
                            }
                        }
                    }
                }
            }
        }

        return Ok(())
    }

    pub fn IsCreateEvent(e: &Event) -> bool {
        return e.event_type() == etcd_client::EventType::Put 
            && e.kv().is_some() 
            && e.kv().unwrap().create_revision() == e.kv().unwrap().mod_revision()
    }

    pub fn ParseEvent(&self, e: &Event) -> Result<Option<WatchEvent>> {
        let kv = e.kv().unwrap();
        if Self::IsCreateEvent(e) && e.prev_kv().is_none() {
            let kv = e.kv().unwrap();
            // If the previous value is nil, error. One example of how this is possible is if the previous value has been compacted already.
		    return Err(Error::CommonError(format!("etcd event received with PrevKv=nil (key={}, modRevision={}, type={:?})", 
                                String::from_utf8(kv.key().to_vec())?, kv.mod_revision(), e.event_type())))

        }

        let curObj: Option<DataObject> = if e.event_type() == etcd_client::EventType::Delete {
            None
        } else {
            let obj = Object::decode(kv.value())?;
            let mut inner : DataObjInner = obj.into();
            inner.reversion = kv.mod_revision();
            Some(inner.into())
        };

        let oldObj: Option<DataObject> = match e.prev_kv() {
            None => None,
            Some(pkv) => {
                if e.event_type() == etcd_client::EventType::Delete || !self.AcceptAll() {
                    let obj = Object::decode(pkv.value())?;
                    let mut inner : DataObjInner = obj.into();
                    inner.reversion = kv.mod_revision();
                    Some(inner.into())
                } else {
                    None
                }
            }
        };

        if e.event_type() == etcd_client::EventType::Delete {
            match oldObj {
                None => return Ok(None),
                Some(obj) => {
                    if !self.Filter(&obj) {
                        return Ok(None);
                    }

                    return Ok(Some(WatchEvent {
                        type_: EventType::Deleted,
                        obj: obj,
                    }))
                }
            }
        } else if Self::IsCreateEvent(e) {
            match curObj {
                None => return Ok(None),
                Some(obj) => {
                    if !self.Filter(&obj) {
                        return Ok(None);
                    }

                    return Ok(Some(WatchEvent {
                        type_: EventType::Added,
                        obj: obj,
                    }))
                }
            }
        } else {
            if self.AcceptAll() {
                return Ok(Some(WatchEvent {
                    type_: EventType::Modified,
                    obj: curObj.unwrap(),
                }))
            }

            let curObjPasses = self.Filter(curObj.as_ref().unwrap());
            let oldObjPasses = self.Filter(oldObj.as_ref().unwrap());
            if curObjPasses && oldObjPasses {
                return Ok(Some(WatchEvent {
                    type_: EventType::Modified,
                    obj: curObj.unwrap(),
                }))
            } else if !curObjPasses && oldObjPasses {
                return Ok(Some(WatchEvent {
                    type_: EventType::Deleted,
                    obj: oldObj.unwrap(),
                }))
            }
        }

        return Ok(None);
    }

    pub fn Filter(&self, obj: &DataObject) -> bool {
        if self.internalPred.Empty() {
            return true;
        }

        match self.internalPred.Match(obj) {
            Err(_e) => return false,
            Ok(matched) => return matched,
        }
    }

    pub fn AcceptAll(&self) -> bool {
        return self.internalPred.Empty();
    }

    pub async fn SendErr(&self, sender: &mut mpsc::Sender<WatchEvent>, err: String) -> Result<()> {
        let errEvent = WatchEvent {
            type_: EventType::Error(err),
            obj: DataObject::default(),
        };

        self.SendEvent(sender, errEvent).await?;
        return Ok(())
    }

    pub async fn SendEvent(&self, sender: &mut mpsc::Sender<WatchEvent>, event: WatchEvent) -> Result<()> {
        tokio::select! {
            _ = sender.send(event) => return Ok(()),
            _ = self.closeNotify.notified() => return Err(Error::ContextCancel),
        }
    }

    pub async fn Sync(&self, sender: &mut mpsc::Sender<WatchEvent>) -> Result<()> {
        let options = GetOptions::new().with_prefix();

        let key: &str = &self.key;
        let getResp = self.client.lock().await.get(key, Some(options)).await?;

        let initalRev = getResp.header().unwrap().revision();
        self.initialRev.store(initalRev, std::sync::atomic::Ordering::SeqCst);

        for kv in getResp.kvs() {
            let obj = Object::decode(kv.value())?;
            let mut inner : DataObjInner = obj.into();
            inner.reversion = kv.mod_revision();
            let obj = inner.into();

            if self.internalPred.Match(&obj)? {
                let event = WatchEvent {
                    type_: EventType::Added,
                    obj: obj,
                };
                self.SendEvent(sender, event).await?;
            }
        }

        return Ok(())
    }
}