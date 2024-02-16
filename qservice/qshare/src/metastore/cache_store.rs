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

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use std::sync::RwLock;
use std::ops::Deref;
use std::fmt::Debug;
use uuid::Uuid;

use tokio::sync::Notify;

use crate::common::*;
use super::cacher::DEFAULT_CACHE_COUNT;
use super::data_obj::*;
use super::selection_predicate::*;
use super::watch::*;

use async_trait::async_trait;

#[async_trait]
pub trait BackendStore1 : Sync + Send + Debug {
    async fn Get1(&self, key: &str, minRevision: i64) -> Result<Option<DataObject>>;
    async fn List1(&self, prefix: &str, opts: &ListOption) -> Result<DataObjList>;

    // register cacher for the prefix, the ready will be notified when the first list finish
    // the notify is used by Cacher to notice BackendStore stop update the Cacher
    fn Register1(&self, cacher: CacheStore, rev: i64, prefix: String, ready: Arc<Notify>, notify: Arc<Notify>) -> Result<()>;
}

#[derive(Clone, Debug)]
pub struct CacheStore(Arc<RwLock<CacheStoreInner>>);

impl Deref for CacheStore {
    type Target = Arc<RwLock<CacheStoreInner>>;

    fn deref(&self) -> &Arc<RwLock<CacheStoreInner>> {
        &self.0
    }
}

impl CacheStore {
    pub async fn New(store: Arc<dyn BackendStore1>, objType: &str, rev: i64) -> Result<Self> {
        let storeClone = store.clone();
        let inner = CacheStoreInner::New(store, objType);
        let notify = inner.closeNotify.clone();
        let ret = Self(Arc::new(RwLock::new(inner)));

        let prefixClone = objType.to_string();

        let watch = ret.clone();
        let ready = Arc::new(Notify::new());
        let readyClone = ready.clone();


        storeClone.Register1(watch, rev, prefixClone, readyClone, notify)?;
    
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
            let store: Arc<dyn BackendStore1> = self.Store();
            let key = &self.StoreKey(&objKey);
            return store.Get1(key, revision).await;
        }

        self.WaitUntilRev(revision).await?;
        match self.read().unwrap().cacheStore.get(&objKey) {
            None => return Ok(None),
            Some(o) => return Ok(Some(o.clone())),
        }
    }

    pub fn Count(&self) -> usize {
        return self.read().unwrap().cacheStore.len();
    }

    pub async fn List(&self, namespace: &str, opts: &ListOption) -> Result<DataObjList> {
        if opts.revision == -1 {
            let store = self.Store();
            let mut opts = opts.DeepCopy();
            opts.revision = 0;
            let prefix = &self.StoreKey(namespace);

            return store.List1(prefix, &opts).await;
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
            revision: inner.channelRev,
            ..Default::default()
        });
    }

    pub fn Store(&self) -> Arc<dyn BackendStore1> {
        return self.read().unwrap().backendStore.clone();
    }

    pub fn GetObject(&self, key: &str) -> Option<DataObject> {
        return self.read().unwrap().cacheStore.get(key).cloned();
    }

    pub fn StoreKey(&self, key: &str) -> String {
        return self.read().unwrap().StoreKey(key);
    }

    pub fn Contains(&self, key: &str) -> bool {
        return self.read().unwrap().GetByKey(key).is_some();
    }

    pub fn Refresh(&self, objs: &[DataObject]) -> Result<()> {
        return self.write().unwrap().Refresh(objs);
    }

    pub fn ProcessEvent(&self, event: &WatchEvent) -> Result<()> {
        let mut inner = self.write().unwrap();
        return inner.ProcessEvent(event);
    }

    pub async fn WaitUntilRev(&self, channelRev: i64) -> Result<()> {
        if channelRev == 0 {
            // 0 means current rev
            return Ok(());
        }

        let time = Instant::now();
        let until = time.checked_add(Duration::from_secs(3)).unwrap();

        let revNotify = self.read().unwrap().revNotify.clone();
        loop {
            if self.read().unwrap().channelRev >= channelRev {
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
}

#[derive(Debug)]
pub struct CacheStoreInner {
    pub backendStore: Arc<dyn BackendStore1>,

    pub objectType: String, // like "pod"

    pub channleId: Uuid,

    pub cache: RingBuf,

    pub channelRev: i64,

    // ResourceVersion of the last list result (populated via Replace() method).
    pub listRevision: i64,

    pub lastWatcherId: u64,

    pub closeNotify: Arc<Notify>,

    pub revNotify: Arc<Notify>,

    pub cacheStore: BTreeMap<String, DataObject>,

    pub watchers: BTreeMap<u64, CacheWatcher>,
}

impl CacheStoreInner {
    pub fn New(store: Arc<dyn BackendStore1>, objectType: &str) -> Self {
        return Self {
            backendStore: store,
            objectType: objectType.to_string(),
            channleId: Uuid::new_v4(),
            cache: RingBuf::New(DEFAULT_CACHE_COUNT),
            channelRev: 0,
            listRevision: 0,
            lastWatcherId: 0,
            closeNotify: Arc::new(Notify::new()),
            revNotify: Arc::new(Notify::new()),
            cacheStore: BTreeMap::new(),
            watchers: BTreeMap::new(),
            //bgWorker: None,
        };
    }

    pub fn ChannelRev(&mut self) -> i64 {
        self.channelRev += 1;
        return self.channelRev;
    }

    pub fn StoreKey(&self, key: &str) -> String {
        return self.objectType.clone() + "/" + key;
    }

    pub fn Refresh(&mut self, objs: &[DataObject]) -> Result<()> {
        let mut set = BTreeSet::new();

        let mut events = Vec::new();
        for obj in objs {
            set.insert(obj.Key());
            match self.cacheStore.get(&obj.Key()) {
                Some(oldObj) => {
                    if oldObj.revision != obj.revision {
                        let event = WatchEvent {
                            type_: EventType::Modified,
                            obj: obj.clone(),
                        };
                        events.push(event);
                    }
                }
                None => {
                    let event = WatchEvent {
                        type_: EventType::Added,
                        obj: obj.clone(),
                    };
                    events.push(event);
                }
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

    pub fn Add(&mut self, obj: &DataObject) -> Result<()> {
        // the object's creator is the cachestore, so the channelRev == Revision
        let rev = self.ChannelRev();
        let event = WatchEvent {
            type_: EventType::Added,
            obj: obj.CopyWithRev(rev, rev),
        };

        return self.ProcessEvent(&event);
    }

    pub fn Remove(&mut self, obj: &DataObject) -> Result<()> {
        let rev = self.ChannelRev();
        let event = WatchEvent {
            type_: EventType::Deleted,
            obj: obj.CopyWithRev(rev, rev),
        };

        return self.ProcessEvent(&event);
    }

    pub fn Update(&mut self, obj: &DataObject) -> Result<()> {
        let rev = self.ChannelRev();
        let event = WatchEvent {
            type_: EventType::Modified,
            obj: obj.CopyWithRev(rev, rev),
        };

        return self.ProcessEvent(&event);
    }

    pub fn ProcessEvent(&mut self, event: &WatchEvent) -> Result<()> {
        let mut wcEvent = WatchCacheEvent {
            type_: event.type_.DeepCopy(),
            obj: event.obj.clone(),
            prevObj: None,
            channelRev: self.ChannelRev(),
            revision: event.obj.Revision(),
            recordTime: SystemTime::now(),
        };

        let key = event.obj.Key();

        match self.cacheStore.get(&key) {
            None => (),
            Some(o) => {
                let newRev = event.obj.revision;
                let preRev = o.revision;

                // get older update, ignore this
                if newRev <= preRev {
                    return Ok(())
                }
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
                channelRev: obj.channelRev,
                revision: obj.revision,
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
            return Ok(self.GetAllEventsFromStore(pred)?);
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