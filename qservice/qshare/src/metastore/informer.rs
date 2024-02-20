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

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::time::Duration;
use std::{collections::BTreeMap, sync::Arc, fmt::Debug};
use core::ops::Deref;

use tokio::sync::{RwLock as TRwLock, Notify};

use rand::Rng;

use super::data_obj::*;
use super::cacher_client::*;
use super::selection_predicate::*;
use super::store::ThreadSafeStore;
use crate::common::*;

pub trait EventHandler : Debug + Send + Sync {
    fn handle(&self, store: &ThreadSafeStore, event: &DeltaEvent);
}

#[derive(Debug, Clone)]
pub struct Informer(Arc<InformerInner>);

impl Deref for Informer {
    type Target = Arc<InformerInner>;

    fn deref(&self) -> &Arc<InformerInner> {
        &self.0
    }
}

unsafe impl Send for Informer {}

impl Informer {
    pub async fn New(addresses: Vec<String>, objType: &str, namespace: &str, opts: &ListOption) -> Result<Self> {
        let inner = InformerInner {
            objType: objType.to_string(),
            namespace: namespace.to_string(),
            opts: opts.DeepCopy(),
            revision: AtomicI64::new(0),
            store: ThreadSafeStore::default(),
            lastEventHandlerId: AtomicU64::new(0),
            serverAddresses: addresses,
            handlers: TRwLock::new(BTreeMap::new()),
            closeNotify: Arc::new(Notify::new()),
            closed: AtomicBool::new(false)
        };

        let informer = Self(Arc::new(inner));

        return Ok(informer);
    }

    pub async fn Close(&self) -> Result<()> {
        let notify = self.closeNotify.clone();

        notify.notify_waiters();
        return Ok(())
    }


    pub async fn AddEventHandler(&self, h: Arc<dyn EventHandler>) -> Result<u64> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(Error::CommonError("the informer is closed".to_owned()));
        }

        let id = self.lastEventHandlerId.fetch_add(1, Ordering::SeqCst);
        
        let objs = self.store.List();
        for obj in objs {
            let event = DeltaEvent { 
                type_: EventType::Added, 
                inInitialList: true,
                obj: obj, 
                oldObj: None 
            };

            h.handle(&self.store, &event);
        }

        self.handlers.write().await.insert(id, h.clone());

        return Ok(id);
    }

    pub async fn RemoveEventHandler(&mut self, id: u64) -> Option<Arc<dyn EventHandler>> {
        return self.handlers.write().await.remove(&id);
    }

    pub async fn GetClient(&self) -> Option<CacherClient> {
        let size = self.serverAddresses.len();
        let offset: usize = rand::thread_rng().gen_range(0..size);
        loop {
            for i in 0..size {
                let idx = (offset + i) % size;
                let addr = &self.serverAddresses[idx];

                tokio::select! { 
                    out = CacherClient::New(addr.clone()) => {
                        match out {
                            Ok(client) => return Some(client),
                            Err(e) => {
                                error!("informer::GetClient fail to connect to {} with error {:?}", addr, e);
                            }
                        }
                    }
                    _ = self.closeNotify.notified() => {
                        self.closed.store(true, Ordering::SeqCst);
                        return None
                    }
                };
            }

            // retry after one second
            tokio::select! { 
                _ = tokio::time::sleep(Duration::from_millis(1000)) => {}
                _ = self.closeNotify.notified() => {
                    self.closed.store(true, Ordering::SeqCst);
                    return None
                }
            }
        }
    }

    pub async fn InitList(&self, client: &CacherClient) -> Result<()> {
        let store = self.store.clone();
        
        let objType = self.objType.clone();
        let namespace = self.namespace.clone();
        let opts = self.opts.DeepCopy();
        
        let objs = client.List(&objType, &namespace, &opts).await?;
        self.revision.store(objs.revision, Ordering::SeqCst);
        for o in objs.objs {
            store.Add(&o)?;
            self.Distribute(&DeltaEvent {
                type_: EventType::Added,
                inInitialList: true,
                obj: o,
                oldObj: None,
            }).await;
        }
        
        return Ok(())
    }

    pub async fn WatchUpdate(&self, client: &CacherClient) -> Result<()> {
        let objType = self.objType.clone();
        let namespace = self.namespace.clone();
        let mut opts = self.opts.DeepCopy();
        opts.revision = self.revision.load(Ordering::SeqCst) + 1;
        let store = self.store.clone();
        let closeNotify = self.closeNotify.clone();
        
        loop {
            let mut ws = client.Watch(&objType, &namespace, &opts).await?;
            loop {
                let event = tokio::select! { 
                    e = ws.Next() => {
                        e
                    }
                    _ = closeNotify.notified() => {
                        self.closed.store(true, Ordering::SeqCst);
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

                self.Distribute(&event).await;
            }

            let objs = client.List(&objType, &namespace, &opts).await?;
            opts.revision = objs.revision + 1;
            for o in objs.objs {
                self.Distribute(&DeltaEvent {
                    type_: EventType::Added,
                    inInitialList: false,
                    obj: o,
                    oldObj: None,
                }).await;
            }
        }
        
    }

    pub async fn Process(&self, notify: Arc<Notify>) -> Result<()> {
        let mut client = match self.GetClient().await {
            None => return Ok(()),
            Some(c) => c,
        };

        loop {
            match self.InitList(&client).await {
                Err(e) => {
                    error!("informer initlist fail with error {:?}", e);
                }
                Ok(()) => break,
            }

            client = match self.GetClient().await {
                None => return Ok(()),
                Some(c) => c,
            };
        }
        
        
        notify.notify_waiters();

        loop {
            match self.WatchUpdate(&client).await {
                Err(e) => {
                    error!("informer WatchUpdate fail with error {:?}", e);
                }
                Ok(()) => break,
            }

            client = match self.GetClient().await {
                None => return Ok(()),
                Some(c) => c,
            };
        }

        return Ok(())   
    }

    pub async fn Distribute(&self, event: &DeltaEvent) {
        let handlers = self.handlers.read().await;
        for h in handlers.values().into_iter() {
            h.handle(&self.store, event)
        }
    }

}

#[derive(Debug)]
pub struct InformerInner {
    pub objType: String,
    pub namespace: String,
    pub opts: ListOption,

    pub revision: AtomicI64,

    pub serverAddresses: Vec<String>,
    
    pub lastEventHandlerId: AtomicU64,
    pub store: ThreadSafeStore,
    pub handlers: TRwLock<BTreeMap<u64, Arc<dyn EventHandler>>>,

    pub closeNotify: Arc<Notify>,
    pub closed: AtomicBool,
}
