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

use std::collections::BTreeSet;
use std::fmt::Debug;
use std::ops::Deref;
use std::time::{Duration, Instant};
use std::{collections::BTreeMap, sync::Arc, time::SystemTime};

use std::sync::RwLock;
use tokio::sync::Notify;

use crate::common::*;
use crate::selection_predicate::*;
use crate::types::*;
use crate::watch::*;


use async_trait::async_trait;

#[async_trait]
pub trait BackendStore : Sync + Send + Debug {
    async fn Create(&self, obj: &DataObject) -> Result<DataObject>;
    async fn Update(&self, expectedRev: i64, obj: &DataObject) -> Result<DataObject>;
    async fn Delete(&self, key: &str, expectedRev: i64) -> Result<i64>;
    async fn Get(&self, key: &str, minRevision: i64) -> Result<Option<DataObject>>;
    async fn List(&self, prefix: &str, opts: &ListOption) -> Result<DataObjList>;

    // register cacher for the prefix, the ready will be notified when the first list finish
    // the notify is used by Cacher to notice BackendStore stop update the Cacher
    fn Register(&self, cacher: Cacher, rev: i64, prefix: String, ready: Arc<Notify>, notify: Arc<Notify>) -> Result<()>;
}

#[derive(Clone, Debug)]
pub struct Cacher(Arc<RwLock<CacherInner>>);

impl Deref for Cacher {
    type Target = Arc<RwLock<CacherInner>>;

    fn deref(&self) -> &Arc<RwLock<CacherInner>> {
        &self.0
    }
}

impl Cacher {
    pub fn ProcessEvent(&self, event: &WatchEvent) -> Result<()> {
        let mut inner = self.write().unwrap();
        return inner.ProcessEvent(event);
    }

    pub async fn WaitUntilRev(&self, revision: i64) -> Result<()> {
        if revision == 0 {
            // 0 means current rev
            return Ok(());
        }

        let time = Instant::now();
        let until = time.checked_add(Duration::from_secs(3)).unwrap();

        let revNotify = self.read().unwrap().revNotify.clone();
        loop {
            if self.read().unwrap().revision >= revision {
                return Ok(());
            }
            tokio::select! {
                _ = revNotify.notified() => {
                    return Ok(())
                }
                _ = tokio::time::sleep(until.duration_since(time)) => {
                    return Err(Error::Timeout)
                }
            }
        }
    }

    pub async fn New(store: Arc<dyn BackendStore>, objType: &str, rev: i64) -> Result<Self> {
        let storeClone = store.clone();
        let inner = CacherInner::New(store, objType);
        let notify = inner.closeNotify.clone();
        let ret = Self(Arc::new(RwLock::new(inner)));

        let prefixClone = objType.to_string();

        let watch = ret.clone();
        let ready = Arc::new(Notify::new());
        let readyClone = ready.clone();

        /*let future = tokio::spawn(async move{
            storeClone.Process(watch, rev, prefixClone, readyClone, notify).await
        });*/

        storeClone.Register(watch, rev, prefixClone, readyClone, notify)?;
    
        //ret.0.write().unwrap().bgWorker = Some(future);

        ready.notified().await;
        
        return Ok(ret);
    }

    pub async fn Stop(&self) -> Result<()> {
        let notify = self.read().unwrap().closeNotify.clone();
        notify.notify_waiters();
        //let worker = self.write().unwrap().bgWorker.take();
        let objType = self.read().unwrap().objectType.clone();
        defer!(info!("cacher[{}] stop ... ", &objType));
        /*match worker {
            None => return Ok(()),
            Some(w) => {
                return w.await?;
            }
        }*/

        return Ok(())
    }

    pub async fn Create(&self, obj: &DataObject) -> Result<DataObject> {
        let store = self.Store();
        return store.Create(&obj).await;
    }

    pub async fn Delete(&self, namespace: &str, name: &str) -> Result<i64> {
        let key = namespace.to_string() + "/" + name;
        let obj = self.GetObject(&key);
        let key = self.StoreKey(&key);
        
        match obj {
            None => {
                return self.Store().Delete(&key, 0).await;
            }
            Some(obj) => {
                return self.Store().Delete(&key, obj.Revision()).await;
            }
        }
    }

    pub fn Watch(
        &self,
        namespace: &str,
        revision: i64,
        predicate: SelectionPredicate,
    ) -> Result<CacheWatchStream> {
        let mut inner = self.write().unwrap();
        return inner.Watch(namespace, revision, predicate);
    }

    pub fn RemoveWatch(&self, watcherId: u64) -> Result<()> {
        let mut inner = self.write().unwrap();
        match inner.watchers.remove(&watcherId) {
            None => {
                return Err(Error::CommonError(format!(
                    "doesn't existing watcher {}",
                    watcherId
                )))
            }
            Some(_) => return Ok(()),
        }
    }

    pub async fn Get(
        &self,
        namespace: &str,
        name: &str,
        revision: i64,
    ) -> Result<Option<DataObject>> {
        let objKey = namespace.to_string() + "/" + name;
        if revision == -1 {
            // -1 means get from etcd
            let store: Arc<dyn BackendStore> = self.Store();
            let key = &self.StoreKey(&objKey);
            return store.Get(key, revision).await;
        }

        self.WaitUntilRev(revision).await?;
        match self.read().unwrap().cacheStore.get(&objKey) {
            None => return Ok(None),
            Some(o) => return Ok(Some(o.clone())),
        }
    }

    pub async fn List(&self, namespace: &str, opts: &ListOption) -> Result<DataObjList> {
        if opts.revision == -1 {
            let store = self.Store();
            let mut opts = opts.DeepCopy();
            opts.revision = 0;
            let prefix = &self.StoreKey(namespace);

            return store.List(prefix, &opts).await;
        }

        self.WaitUntilRev(opts.revision).await?;

        let mut objs: Vec<DataObject> = Vec::new();

        let inner = self.read().unwrap();
        for (_, obj) in &inner.cacheStore {
            if obj.Key().starts_with(namespace) && opts.predicate.Match(obj)? {
                objs.push(obj.clone());
            }
        }

        return Ok(DataObjList {
            objs: objs,
            revision: inner.revision,
            ..Default::default()
        });
    }

    pub fn Store(&self) -> Arc<dyn BackendStore> {
        return self.read().unwrap().backendStore.clone();
    }

    pub fn GetObject(&self, key: &str) -> Option<DataObject> {
        return self.read().unwrap().cacheStore.get(key).cloned();
    }

    pub fn StoreKey(&self, key: &str) -> String {
        return self.read().unwrap().StoreKey(key);
    }

    pub async fn Update(&self, obj: &DataObject) -> Result<DataObject> {
        let store = self.Store();
        let key = obj.Key();
        match self.GetObject(&key) {
            None => {
                return store.Update(0, obj).await;
            }
            Some(o) => {
                let rev = o.Revision();
                return store.Update(rev, obj).await;
            }
        }
    }

    pub fn Contains(&self, key: &str) -> bool {
        return self.read().unwrap().GetByKey(key).is_some();
    }

    pub fn Refresh(&self, objs: &[DataObject]) -> Result<()> {
        return self.write().unwrap().Refresh(objs);
    }
}

pub const DEFAULT_CACHE_COUNT: usize = 2000;

#[derive(Debug)]
pub struct CacherInner {
    pub backendStore: Arc<dyn BackendStore>,

    pub objectType: String, // like "pods"

    pub cache: RingBuf,

    // ResourceVersion up to which the watchCache is propagated.
    pub revision: i64,

    // ResourceVersion of the last list result (populated via Replace() method).
    pub listRevision: i64,

    pub lastWatcherId: u64,

    pub closeNotify: Arc<Notify>,

    pub revNotify: Arc<Notify>,

    pub cacheStore: BTreeMap<String, DataObject>,

    pub watchers: BTreeMap<u64, CacheWatcher>,

    //pub bgWorker: Option<JoinHandle<Result<()>>>,
}

impl CacherInner {
    pub fn New(store: Arc<dyn BackendStore>, objectType: &str) -> Self {
        return Self {
            backendStore: store,
            objectType: objectType.to_string(),
            cache: RingBuf::New(DEFAULT_CACHE_COUNT),
            revision: 0,
            listRevision: 0,
            lastWatcherId: 0,
            closeNotify: Arc::new(Notify::new()),
            revNotify: Arc::new(Notify::new()),
            cacheStore: BTreeMap::new(),
            watchers: BTreeMap::new(),
            //bgWorker: None,
        };
    }

    pub fn StoreKey(&self, key: &str) -> String {
        return self.objectType.clone() + "/" + key;
    }

    pub fn Refresh(&mut self, objs: &[DataObject]) -> Result<()> {
        let mut set = BTreeSet::new();

        let mut events = Vec::new();
        for obj in objs {
            set.insert(obj.Key());
            if self.cacheStore.contains_key(&obj.Key()) {
                let event = WatchEvent {
                    type_: EventType::Modified,
                    obj: obj.clone(),
                };
                events.push(event);
            } else {
                let event = WatchEvent {
                    type_: EventType::Added,
                    obj: obj.clone(),
                };
                events.push(event);
            }
        }

        for (key, obj)  in &self.cacheStore {
            if !set.contains(key) {
                let event = WatchEvent {
                    type_: EventType::Deleted,
                    obj: obj.clone(),
                };
                events.push(event);
            }
        }

        for event in &events {
            self.ProcessEvent(event)?;
        }        

        return Ok(())
    }

    pub fn ProcessEvent(&mut self, event: &WatchEvent) -> Result<()> {
        let mut wcEvent = WatchCacheEvent {
            type_: event.type_.DeepCopy(),
            obj: event.obj.clone(),
            prevObj: None,
            revision: event.obj.Revision(),
            recordTime: SystemTime::now(),
        };

        let key = event.obj.Key();

        match self.cacheStore.get(&key) {
            None => (),
            Some(o) => {
                wcEvent.prevObj = Some(o.clone());
            }
        }

        let mut removeWatches = Vec::new();

        for (idx, w) in &mut self.watchers {
            match w.SendWatchCacheEvent(&wcEvent) {
                Ok(()) => (),
                Err(_) => {
                    removeWatches.push(*idx);
                }
            }
        }

        for idx in removeWatches {
            self.watchers.remove(&idx);
        }

        self.cache.Push(wcEvent);

        if event.type_ == EventType::Deleted {
            self.cacheStore.remove(&key);
        } else {
            self.cacheStore.insert(key, event.obj.clone());
        }

        self.revision = event.obj.Revision();
        return Ok(());
    }

    pub fn GetByKey(&self, key: &str) -> Option<DataObject> {
        match self.cacheStore.get(key) {
            None => return None,
            Some(o) => return Some(o.clone()),
        }
    }

    pub fn GetAllEventsFromStore(
        &self,
        revision: i64,
        pred: &SelectionPredicate,
    ) -> Result<Vec<WatchCacheEvent>> {
        let mut buf = Vec::new();

        for (_, obj) in &self.cacheStore {
            if !pred.Match(obj)? {
                continue;
            }

            buf.push(WatchCacheEvent {
                type_: EventType::Added,
                obj: obj.clone(),
                prevObj: None,
                revision: revision,
                recordTime: SystemTime::now(),
            })
        }

        return Ok(buf);
    }

    pub fn GetAllEvents(
        &self,
        revision: i64,
        pred: &SelectionPredicate,
    ) -> Result<Vec<WatchCacheEvent>> {
        let size = self.cache.Size();

        let oldest = if self.listRevision > 0 && self.cache.tail == 0 {
            // If no event was removed from the buffer since last relist, the oldest watch
            // event we can deliver is one greater than the resource version of the list.
            self.listRevision + 1
        } else if size > 0 {
            self.cache.buf[self.cache.OldestIdx()]
                .as_ref()
                .unwrap()
                .revision
        } else {
            return Err(Error::CommonError(format!(
                "watch cache isn't correctly initialized"
            )));
        };

        if revision == 0 {
            // resourceVersion = 0 means that we don't require any specific starting point
            // and we would like to start watching from ~now.
            // However, to keep backward compatibility, we additionally need to return the
            // current state and only then start watching from that point.
            return Ok(self.GetAllEventsFromStore(revision, pred)?);
        }

        if revision < oldest - 1 {
            return Err(Error::CommonError(format!(
                "too old resource version: {} ({})",
                revision,
                oldest - 1
            )));
        }

        return self.cache.GetAllEvents(revision, pred);
    }

    pub fn Watch(
        &mut self,
        namespace: &str,
        revision: i64,
        predicate: SelectionPredicate,
    ) -> Result<CacheWatchStream> {
        self.lastWatcherId += 1;
        let wid = self.lastWatcherId;

        let events = self.GetAllEvents(revision, &predicate)?;
        let channelSize = (events.len() + 200).max(DEFAULT_CACHE_COUNT);

        let (mut watcher, stream) = CacheWatcher::New(wid, channelSize, namespace, predicate, "");

        for e in events {
            watcher.SendWatchCacheEvent(&e)?;
        }

        self.watchers.insert(wid, watcher);
        return Ok(stream);
    }
}
