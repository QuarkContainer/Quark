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

use std::sync::Arc;
use etcd_client::Event;
use etcd_client::GetOptions;
use etcd_client::WatchOptions;
use etcd_client::WatchResponse;
use etcd_client::WatchStream;
use tokio::sync::mpsc;
use tokio::sync::Notify;
use std::sync::atomic::AtomicI64;
use prost::Message;
use tokio::sync::Mutex as TMutex;

use crate::etcd_client::EtcdClient;
use crate::selection_predicate::SelectionPredicate;
use crate::shared::common::*;
use crate::service_directory::*;

use super::etcd_store::*;

#[derive(Debug, PartialEq, Eq)]
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
    pub closeNotify:  Arc<tokio::sync::Notify>,

    pub key: String,
    pub initialRev: AtomicI64,
    pub internalPred: SelectionPredicate,

    pub watcher: Option<etcd_client::Watcher>,
}

impl Watcher {
    pub const RESULT_CHANNEL_SIZE: usize = 300;

    pub fn New(client: &EtcdClient, key: &str, rev: i64, pred: SelectionPredicate) -> (Self, WatchReader) {
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
        };

        return (watcher, reader)
    }


    pub async fn Processing(mut self) -> Result<()> {
        let ret = self.ProcessingInner().await;
        //self.resultSender.closed().await;
        return ret;
    }

    pub async fn ProcessingInner(&mut self) -> Result<()> {
        let initialRev = self.initialRev.load(std::sync::atomic::Ordering::Relaxed);
        if initialRev == 0 {
            let ret = self.Sync().await;
            match ret {
                Err(e) => self.SendErr(format!("failed to sync with latest state: {:?}", e)).await.unwrap(),
                Ok(()) => (),
            }
        }
           
        let initialRev = self.initialRev.load(std::sync::atomic::Ordering::Relaxed);
        let options = WatchOptions::new()
                    .with_start_revision(initialRev + 1)
                    .with_prev_key()
                    .with_prefix()
                    ;

        let key: &str = &self.key;
        let (watcher, mut stream) = self.client.lock().await.watch(key, Some(options)).await?;
        
        self.watcher = Some(watcher);

        while let Some(resp) = self.WatchNext(&mut stream).await? {
            for e in resp.events() {
                match self.ParseEvent(e) {
                    Err(e) => {
                        self.SendErr(format!("{:?}", e)).await?
                    },
                    
                    Ok(event) => {
                        match event {
                            None => (),
                            Some(event) => {
                                self.SendEvent(event).await?;
                            }
                        }
                    }
                }
            }
        }

        return Ok(())
    }

    pub async fn WatchNext(&self, stream: &mut WatchStream) -> Result<Option<WatchResponse>> {
        tokio::select! { 
            ret = stream.message() => return Ok(ret?),
            _ = self.closeNotify.notified() => return Err(Error::ContextCancel),
        }
    }

    pub fn IsCreateEvent(e: &Event) -> bool {
        return e.event_type() == etcd_client::EventType::Put 
            && e.kv().is_some() 
            && e.kv().unwrap().create_revision() == e.kv().unwrap().mod_revision()
    }

    pub fn ParseEvent(&self, e: &Event) -> Result<Option<WatchEvent>> {
        let kv = e.kv().unwrap();
                
        if Self::IsCreateEvent(e) && e.prev_kv().is_some() {
            let kv = e.kv().unwrap();
            // If the previous value is nil, error. One example of how this is possible is if the previous value has been compacted already.
		    return Err(Error::CommonError(format!("etcd event received with PrevKv=nil (key={}, modRevision={}, type={:?})", 
                                String::from_utf8(kv.key().to_vec())?, kv.mod_revision(), e.event_type())))

        }

        let curObj: Option<DataObject> = if e.event_type() == etcd_client::EventType::Delete {
            None
        } else {
            let obj = Object::decode(kv.value())?;
            let inner : DataObjInner = obj.into();
            inner.SetRevision(kv.mod_revision());
            Some(inner.into())
        };

        let oldObj: Option<DataObject> = match e.prev_kv() {
            None => None,
            Some(pkv) => {
                if e.event_type() == etcd_client::EventType::Delete || !self.AcceptAll() {
                    let obj = Object::decode(pkv.value())?;
                    let inner : DataObjInner = obj.into();
                    inner.SetRevision(kv.mod_revision());
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
                }))
            } else if curObjPasses && !oldObjPasses {
                return Ok(Some(WatchEvent {
                    type_: EventType::Added,
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

    pub async fn SendErr(&self, err: String) -> Result<()> {
        let errEvent = WatchEvent {
            type_: EventType::Error(err),
            obj: DataObject::default(),
        };

        self.SendEvent(errEvent).await?;
        return Ok(())
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
        self.initialRev.store(initalRev, std::sync::atomic::Ordering::SeqCst);

        for kv in getResp.kvs() {
            let obj = Object::decode(kv.value())?;
            let inner : DataObjInner = obj.into();
            inner.SetRevision(kv.mod_revision());
            let obj = inner.into();

            if self.internalPred.Match(&obj)? {
                let event = WatchEvent {
                    type_: EventType::Added,
                    obj: obj,
                };
                self.SendEvent(event).await?;
            }
        }

        return Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{types::DeepCopy};
    use super::*;

    pub async fn TestCheckResultFunc(
        r: &WatchReader, 
        expectEventType: EventType, 
        check: impl Fn(&DataObject) -> Result<()>
    ) {
        use tokio::time::sleep;
        use std::time::Duration;

        tokio::select! {
            event = r.GetNextEvent() => {
                match event {
                    None => assert!(false, "no event recv"),
                    Some(event) => {
                        assert!(event.type_== expectEventType, "actual is {:?} expected {:?}", event.type_, expectEventType);
                        check(&event.obj).unwrap();
                    }
                }
            }
            _ = sleep(Duration::from_secs(5)) => {
                assert!(false, "time out after waiting {} on ResultChan", 5);
            }
        }
    }

    pub async fn TestCheckResult(
        r: &WatchReader, 
        expectEventType: EventType, 
        expectObj: &DataObject) {

        TestCheckResultFunc(r, expectEventType, |obj|{
            assert!(obj == expectObj, "actual {:#?} expected {:#?}", obj, expectObj);
            return Ok(())
        }).await;
    }

    pub async fn TestCheckQTypeResult(
        r: &WatchReader, 
        expectEventType: EventType, 
        expectObj: &DataObject) {
            
            return TestCheckResult(r, expectEventType, &expectObj).await;
    }

    pub struct TestWatchStruct {
        pub obj: DataObject,
        pub expectEvent: bool,
        pub watchType: EventType,
    }

    pub async fn Idle() {}

    pub async fn TestWatch() -> Result<()> {
        let mut store = EtcdStore::New("localhost:2379", true).await?;
        let _initRv = store.Clear("pods").await?;

        let basePod = DataObject::NewPod("", "foo", "", "").unwrap();
        let basePodAssigned = DataObject::NewPod("", "foo", "bar", "").unwrap();

        struct Test <'a> {
            name: &'a str,
            namespace: &'a str,
            pred: SelectionPredicate,
            watchTests: Vec<TestWatchStruct>,
        }

        let tests = [
            /*Test {
                name: "creat a key",
                namespace: &format!("test-ns-1"),
                pred: SelectionPredicate::default(),
                watchTests: vec![
                    TestWatchStruct {
                        obj: basePod.DeepCopy(),
                        expectEvent: true,
                        watchType: EventType::Added,
                    }
                ],
            },
            Test {
                name: "key updated to match predicate",
                namespace: &format!("test-ns-2"),
                pred: SelectionPredicate {
                    field: crate::selector::Selector::Parse("spec.nodename=bar").unwrap(),
                    ..Default::default()
                },
                watchTests: vec![
                    TestWatchStruct {
                        obj: basePod.DeepCopy(),
                        expectEvent: false,
                        watchType: EventType::Added,
                    },
                    TestWatchStruct {
                        obj: basePodAssigned.DeepCopy(),
                        expectEvent: true,
                        watchType: EventType::Added,
                    },
                ],
            },
            Test {
                name: "update",
                namespace: &format!("test-ns-3"),
                pred: SelectionPredicate {
                    //field: Selector::Parse("spec.nodename=bar").unwrap(),
                    ..Default::default()
                },
                watchTests: vec![
                    TestWatchStruct {
                        obj: basePod.DeepCopy(),
                        expectEvent: true,
                        watchType: EventType::Added,
                    },
                    TestWatchStruct {
                        obj: basePodAssigned.DeepCopy(),
                        expectEvent: true,
                        watchType: EventType::Modified,
                    },
                ],
            },*/
            Test {
                name: "delete because of being filtered",
                namespace: &format!("test-ns-4"),
                pred: SelectionPredicate {
                    field: crate::selector::Selector::Parse("spec.nodename!=bar").unwrap(),
                    ..Default::default()
                },
                watchTests: vec![
                    TestWatchStruct {
                        obj: basePod.DeepCopy(),
                        expectEvent: true,
                        watchType: EventType::Added,
                    },
                    TestWatchStruct {
                        obj: basePodAssigned.DeepCopy(),
                        expectEvent: true,
                        watchType: EventType::Deleted,
                    },
                ],
            },
        ];

        for tt in tests {
            let watchKey = format!("pods/{}", tt.namespace);
            let key = watchKey.clone() + "/foo";

            let (w, r) = store.Watch(&watchKey, 0, tt.pred).unwrap();
            let t = tokio::spawn(async move {
                w.Processing().await
            });

            let mut prevObj = DataObject::NewPod("", "", "", "")?;
            for watchTest in tt.watchTests {
                let dataObj = watchTest.obj;

                let newVersion = store.Update(&key, 0, &dataObj).await?;
                if watchTest.expectEvent {
                    let expectObj = if watchTest.watchType == EventType::Deleted {
                        let expectObj = prevObj.DeepCopy();
                        expectObj.SetRevision(newVersion);
                        expectObj
                    } else {
                        dataObj.DeepCopy()
                    };

                    TestCheckQTypeResult(&r, watchTest.watchType, &expectObj).await;
                }
                prevObj = dataObj;
             }

             r.Close();
             let res = t.await.unwrap();
             println!("{:?}", &res);
             //res.unwrap();
        }

        return Ok(())
    }

    #[test]
    pub fn TestWatchSync() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build().unwrap();
        
        rt.block_on(TestWatch()).unwrap();
    }
}
