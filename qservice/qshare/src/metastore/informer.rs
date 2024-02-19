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

use std::{collections::BTreeMap, sync::Arc, fmt::Debug};
use core::ops::Deref;

use tokio::sync::{RwLock as TRwLock, Notify};

use super::data_obj::*;
use super::cacher_client::*;
use super::selection_predicate::*;
use super::store::ThreadSafeStore;
use crate::common::*;

pub trait EventHandler : Debug + Send + Sync {
    fn handle(&self, store: &ThreadSafeStore, event: &DeltaEvent);
    //fn OnAdd(&self, store: &ThreadSafeStore, obj: &DataObject, isInInitialList: bool);
    //fn OnUpdate(&self, store: &ThreadSafeStore, oldObj: &DataObject, newObj: &DataObject);
    //fn OnDelete(&self, store: &ThreadSafeStore, obj: &DataObject);
}

#[derive(Debug, Clone)]
pub struct Informer(Arc<TRwLock<InformerInner>>);

impl Deref for Informer {
    type Target = Arc<TRwLock<InformerInner>>;

    fn deref(&self) -> &Arc<TRwLock<InformerInner>> {
        &self.0
    }
}

unsafe impl Send for Informer {}

impl Informer {
    pub async fn New(client: &CacherClient, objType: &str, namespace: &str, opts: &ListOption) -> Result<Self> {
        //let client = CacherClient::New(addr.to_string()).await?;

        let inner = InformerInner {
            objType: objType.to_string(),
            namespace: namespace.to_string(),
            opts: opts.DeepCopy(),
            revision: 0,
            store: ThreadSafeStore::default(),
            lastEventHandlerId: 0,
            client: client.clone(),
            handlers: BTreeMap::new(),
            closeNotify: Arc::new(Notify::new()),
            closed: false,
            task: None,
        };

        let informer = Self(Arc::new(TRwLock::new(inner)));
        
        informer.write().await.task = None; //Some(task);

        return Ok(informer);
    }

    pub async fn Close(&self) -> Result<()> {
        let (notify, task) = {
            let mut inner = self.write().await;
            (inner.closeNotify.clone(), inner.task.take())
        };

        match task {
            None => {
                return Err(Error::CommonError(format!("informer close with none task")));
            }
            Some(t) => {
                notify.notify_waiters();
                return t.await?;
            }
        }
    }

    pub async fn AddEventHandler(&self, h: Arc<dyn EventHandler>) -> Result<u64> {
        return self.write().await.AddEventHandler(h)
    }

    pub async fn RemoveEventHandler(&self, id: u64) -> Option<Arc<dyn EventHandler>> {
        return self.write().await.RemoveEventHandler(id);
    }

    pub async fn InitList(&self) -> Result<()> {
        let mut inner = self.write().await;
        let client = inner.client.clone();
        let store = inner.store.clone();
        
        let objType = inner.objType.clone();
        let namespace = inner.namespace.clone();
        let mut opts = inner.opts.DeepCopy();
        
        let objs = client.List(&objType, &namespace, &opts).await?;
        opts.revision = objs.revision + 1;
        inner.revision = objs.revision;
        for o in objs.objs {
            store.Add(&o)?;
            inner.Distribute(&DeltaEvent {
                type_: EventType::Added,
                inInitialList: true,
                obj: o,
                oldObj: None,
            });
        }
        
        return Ok(())
    }

    pub async fn WatchUpdate(&self) -> Result<()> {
        let inner = self.write().await;
        let client = inner.client.clone();
        let objType = inner.objType.clone();
        let namespace = inner.namespace.clone();
        let mut opts = inner.opts.DeepCopy();
        opts.revision = inner.revision + 1;
        let store = inner.store.clone();
        let closeNotify = inner.closeNotify.clone();
        drop(inner);

        loop {
            let mut ws = client.Watch(&objType, &namespace, &opts).await?;
            loop {
                let event = tokio::select! { 
                    e = ws.Next() => {
                        e
                    }
                    _ = closeNotify.notified() => {
                        self.write().await.closed = true;
                        return Ok(())
                    }
                };

                let event = match event {
                    Err(e) => {
                        error!("watch get error {:?}", e);
                        break;
                    }
                    Ok(e) => {
                        match e {
                            None => break,
                            Some(e) => {
                                opts.revision = e.obj.Revision();
                                let de = match e.type_ {
                                    EventType::Added => {
                                        store.Add(&e.obj)?;
                                        DeltaEvent {
                                            type_: e.type_,
                                            inInitialList: false,
                                            obj: e.obj.clone(),
                                            oldObj: None
                                        }
                                    }
                                    EventType::Modified => {
                                        let oldObj = store.Update(&e.obj)?;
                                        DeltaEvent {
                                            type_: e.type_,
                                            inInitialList: false,
                                            obj: e.obj.clone(),
                                            oldObj: Some(oldObj)
                                        }
                                    }
                                    EventType::Deleted => {
                                        let old = store.Delete(&e.obj)?;
                                        DeltaEvent {
                                            type_: e.type_,
                                            inInitialList: false,
                                            obj: e.obj.clone(),
                                            oldObj: Some(old),
                                        }
                                    }
                                    _ => panic!("Informer::Process get unexpect type {:?}", e.type_),
                                };
                                de
                            }
                        }
                    }
                };

                self.read().await.Distribute(&event);
            }

            let objs = client.List(&objType, &namespace, &opts).await?;
            opts.revision = objs.revision + 1;
            let inner = self.read().await;
            for o in objs.objs {
                inner.Distribute(&DeltaEvent {
                    type_: EventType::Added,
                    inInitialList: false,
                    obj: o,
                    oldObj: None,
                });
            }
        }
        
    }

    pub async fn Process(&self, notify: Arc<Notify>) -> Result<()> {
        self.InitList().await?;
        
        notify.notify_waiters();
        self.WatchUpdate().await?;
        return Ok(())   
    }
}

#[derive(Debug)]
pub struct InformerInner {
    pub objType: String,
    pub namespace: String,
    pub opts: ListOption,

    pub revision: i64,

    pub store: ThreadSafeStore,

    pub lastEventHandlerId: u64,
    pub client: CacherClient,
    pub handlers: BTreeMap<u64, Arc<dyn EventHandler>>,

    pub closeNotify: Arc<Notify>,
    pub closed: bool,

    pub task: Option<tokio::task::JoinHandle<Result<()>>>,
}

impl InformerInner {
    pub fn Distribute(&self, event: &DeltaEvent) {
        for (_, h) in &self.handlers {
            self.DistributeToHandler(event, h)
        }
    }

    pub fn DistributeToHandler(&self, event: &DeltaEvent, h: &Arc<dyn EventHandler>) {
        /*match event.type_ {
            EventType::Added => {
                h.OnAdd(&self.store, &event.obj, isInInitialList);
            }
            EventType::Deleted => {
                h.OnDelete(&self.store,&event.obj);
            }
            EventType::Modified => {
                h.OnUpdate(&self.store, event.oldObj.as_ref().unwrap(), &event.obj);
            }
            _ => {
                panic!("InformerInner::Distribute get unexpected type {:?}", event.type_);
            }
        }*/

        h.handle(&self.store, event)
    }

    pub fn AddEventHandler(&mut self, h: Arc<dyn EventHandler>) -> Result<u64> {
        if self.closed {
            return Err(Error::CommonError("the informer is closed".to_owned()));
        }

        self.lastEventHandlerId += 1;
        
        let objs = self.store.List();
        for obj in objs {
            self.DistributeToHandler(&DeltaEvent { 
                type_: EventType::Added, 
                inInitialList: true,
                obj: obj, 
                oldObj: None 
            }, &h)
        }

        let id = self.lastEventHandlerId;
        self.handlers.insert(id, h.clone());

        return Ok(id);
    }

    pub fn RemoveEventHandler(&mut self, id: u64) -> Option<Arc<dyn EventHandler>> {
        return self.handlers.remove(&id);
    }
}