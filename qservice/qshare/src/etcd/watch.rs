
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
use etcd_client::WatchResponse;
use etcd_client::WatchStream;
use std::sync::atomic::AtomicI64;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Mutex as TMutex;
use tokio::sync::Notify;

use crate::etcd::etcd_client::EtcdClient;
use crate::common::*;
use crate::metastore::selection_predicate::*;
use crate::qmeta::*;
use crate::metastore::data_obj::*;

pub struct WatchReader {
    pub resultRecv: TMutex<mpsc::Receiver<WatchEvent>>,
    pub closeNotify: Arc<tokio::sync::Notify>,
}

impl WatchReader {
    pub async fn GetNextEvent(&self) -> Option<WatchEvent> {
        return self.resultRecv.lock().await.recv().await;
    }

    pub fn Close(&self) {
        self.closeNotify.notify_one();
    }
}



pub struct Watcher {
    pub client: EtcdClient,
    pub resultSender: mpsc::Sender<WatchEvent>,
    pub closeNotify: Arc<tokio::sync::Notify>,

    pub key: String,
    pub initialRev: AtomicI64,
    pub internalPred: SelectionPredicate,

    pub watcher: Option<etcd_client::Watcher>,
    pub stream: Option<Arc<TMutex<WatchStream>>>,
}

impl Watcher {
    pub const RESULT_CHANNEL_SIZE: usize = 300;

    pub fn New(
        client: &EtcdClient,
        key: &str,
        rev: i64,
        pred: SelectionPredicate,
    ) -> (Self, WatchReader) {
        let (resultSender, resultRecv) = mpsc::channel(Self::RESULT_CHANNEL_SIZE);
        let reader = WatchReader {
            resultRecv: TMutex::new(resultRecv),
            closeNotify: Arc::new(Notify::new()),
        };

        let watcher = Self {
            client: client.clone(),
            resultSender: resultSender,
            closeNotify: reader.closeNotify.clone(),

            key: key.to_string(),
            initialRev: AtomicI64::new(rev),
            internalPred: pred,

            watcher: None,
            stream: None,
        };

        return (watcher, reader);
    }

    pub async fn Processing(&mut self) -> Result<()> {
        let ret = self.ProcessingInner().await;
        //self.resultSender.closed().await;
        return ret;
    }

    pub async fn ProcessingInner(&mut self) -> Result<()> {
        if self.watcher.is_none() {
            let initialRev = self.initialRev.load(std::sync::atomic::Ordering::Relaxed);
            if initialRev == 0 {
                let ret = self.Sync().await;
                match ret {
                    Err(e) => self
                        .SendErr(format!("failed to sync with latest state: {:?}", e))
                        .await
                        .unwrap(),
                    Ok(()) => (),
                }
            }

            let initialRev = self.initialRev.load(std::sync::atomic::Ordering::Relaxed);
            let options = WatchOptions::new()
                .with_start_revision(initialRev + 1)
                .with_prev_key()
                .with_prefix();

            let key: &str = &self.key;
            let (watcher, stream) = self.client.lock().await.watch(key, Some(options)).await?;
            self.watcher = Some(watcher);
            self.stream = Some(Arc::new(TMutex::new(stream)));
        }

        while let Some(resp) = self.WatchNext().await? {
            for e in resp.events() {
                match self.ParseEvent(e) {
                    Err(e) => self.SendErr(format!("{:?}", e)).await?,

                    Ok(event) => match event {
                        None => (),
                        Some(event) => {
                            self.SendEvent(event).await?;
                        }
                    },
                }
            }
        }

        return Ok(());
    }

    pub async fn WatchNext(&self) -> Result<Option<WatchResponse>> {
        let stream = self.stream.clone().unwrap();
        let mut stream = stream.lock().await;

        tokio::select! {
            ret = stream.message() => return Ok(ret?),
            _ = self.closeNotify.notified() => return Err(Error::ContextCancel),
        }
    }

    pub fn IsCreateEvent(e: &Event) -> bool {
        return e.event_type() == etcd_client::EventType::Put
            && e.kv().is_some()
            && e.kv().unwrap().create_revision() == e.kv().unwrap().mod_revision();
    }

    pub fn ParseEvent(&self, e: &Event) -> Result<Option<WatchEvent>> {
        let kv = e.kv().unwrap();

        if Self::IsCreateEvent(e) && e.prev_kv().is_some() {
            let kv = e.kv().unwrap();
            // If the previous value is nil, error. One example of how this is possible is if the previous value has been compacted already.
            return Err(Error::CommonError(format!(
                "etcd event received with PrevKv=nil (key={}, modRevision={}, type={:?})",
                String::from_utf8(kv.key().to_vec())?,
                kv.mod_revision(),
                e.event_type()
            )));
        }

        let curObj: Option<DataObject> = if e.event_type() == etcd_client::EventType::Delete {
            None
        } else {
            let obj = Object::Decode(kv.value())?;
            Some(DataObject::NewFromObject(&obj, 0, kv.mod_revision()))
        };

        let oldObj: Option<DataObject> = match e.prev_kv() {
            None => None,
            Some(pkv) => {
                if e.event_type() == etcd_client::EventType::Delete || !self.AcceptAll() {
                    let obj = Object::Decode(pkv.value())?;
                    Some(DataObject::NewFromObject(&obj, 0, kv.mod_revision()))
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
                    }));
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
                    }));
                }
            }
        } else {
            if self.AcceptAll() {
                return Ok(Some(WatchEvent {
                    type_: EventType::Modified,
                    obj: curObj.unwrap(),
                }));
            }

            let curObjPasses = match &curObj {
                None => false,
                Some(o) => self.Filter(o),
            };

            let oldObjPasses = match &oldObj {
                None => false,
                Some(o) => self.Filter(o),
            };

            if curObjPasses && oldObjPasses {
                return Ok(Some(WatchEvent {
                    type_: EventType::Modified,
                    obj: curObj.unwrap(),
                }));
            } else if curObjPasses && !oldObjPasses {
                return Ok(Some(WatchEvent {
                    type_: EventType::Added,
                    obj: curObj.unwrap(),
                }));
            } else if !curObjPasses && oldObjPasses {
                return Ok(Some(WatchEvent {
                    type_: EventType::Deleted,
                    obj: oldObj.unwrap(),
                }));
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

    pub async fn SendErr(&self, err: String) -> Result<()> {
        let errEvent = WatchEvent {
            type_: EventType::Error(err),
            obj: DataObject::default(),
        };

        self.SendEvent(errEvent).await?;
        return Ok(());
    }

    pub async fn SendEvent(&self, event: WatchEvent) -> Result<()> {
        tokio::select! {
            _ = self.resultSender.send(event) => return Ok(()),
            _ = self.closeNotify.notified() => return Err(Error::ContextCancel),
        }
    }

    pub async fn Sync(&self) -> Result<()> {
        let options = GetOptions::new().with_prefix();

        let key: &str = &self.key;
        let getResp = self.client.lock().await.get(key, Some(options)).await?;

        let initalRev = getResp.header().unwrap().revision();
        self.initialRev
            .store(initalRev, std::sync::atomic::Ordering::SeqCst);

        for kv in getResp.kvs() {
            let obj = Object::Decode(kv.value())?;
            let obj = DataObject::NewFromObject(&obj, kv.mod_revision(), kv.mod_revision());

            if self.internalPred.Match(&obj)? {
                let event = WatchEvent {
                    type_: EventType::Added,
                    obj: obj,
                };
                self.SendEvent(event).await?;
            }
        }

        return Ok(());
    }
}
