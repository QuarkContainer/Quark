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
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::atomic::AtomicI64;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

use tokio::sync::Notify;

use super::data_obj::*;
use super::selection_predicate::*;
use super::watch::*;
use crate::common::*;

use async_trait::async_trait;

pub const DEFAULT_CACHE_COUNT: usize = 2000;

#[async_trait]
pub trait BackendStore: Sync + Send + Debug {
    async fn Get(&self, key: &str, minRevision: i64) -> Result<Option<DataObject>>;
    async fn List(&self, prefix: &str, opts: &ListOption) -> Result<DataObjList>;

    // register cacher for the prefix, the ready will be notified when the first list finish
    // the notify is used by Cacher to notice BackendStore stop update the Cacher
    fn Register(
        &self,
        cacher: CacheStore,
        rev: i64,
        prefix: String,
        ready: Arc<Notify>,
        notify: Arc<Notify>,
    ) -> Result<()>;
}

#[derive(Debug, Clone, Default)]
pub struct ChannelRev {
    pub rev: Arc<AtomicI64>,
}

impl ChannelRev {
    pub fn Next(&self) -> i64 {
        return self.rev.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
    }

    pub fn Current(&self) -> i64 {
        return self.rev.load(std::sync::atomic::Ordering::SeqCst) + 1;
    }
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
    pub async fn New(
        store: Option<Arc<dyn BackendStore>>,
        objType: &str,
        rev: i64,
        channelRev: &ChannelRev,
    ) -> Result<Self> {
        let storeClone = store.clone();
        let inner = CacheStoreInner::New(store, objType, channelRev);
        let notify = inner.closeNotify.clone();
        let ret = Self(Arc::new(RwLock::new(inner)));

        let prefixClone = objType.to_string();

        let watch = ret.clone();
        let ready = Arc::new(Notify::new());
        let readyClone = ready.clone();

        match storeClone {
            None => (),
            Some(s) => {
                s.Register(watch, rev, prefixClone, readyClone, notify)?;
                ready.notified().await;
            }
        }

        return Ok(ret);
    }

    pub fn ObjType(&self) -> String {
        return self.read().unwrap().objectType.clone();
    }

    pub async fn Stop(&self) -> Result<()> {
        let notify = self.read().unwrap().closeNotify.clone();
        notify.notify_waiters();
        let objType = self.read().unwrap().objectType.clone();
        defer!(info!("cacher[{}] stop ... ", &objType));

        return Ok(());
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

    pub fn Add(&self, obj: &DataObject) -> Result<()> {
        return self.write().unwrap().Add(obj);
    }

    pub fn Update(&self, obj: &DataObject) -> Result<()> {
        return self.write().unwrap().Update(obj);
    }

    pub fn Remove(&self, obj: &DataObject) -> Result<()> {
        return self.write().unwrap().Remove(obj);
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
            let store: Arc<dyn BackendStore> = match self.Store() {
                None => return Ok(None),
                Some(s) => s,
            };
            let key = &self.StoreKey(&objKey);
            return store.Get(key, revision).await;
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
            let store = match self.Store() {
                None => return Ok(DataObjList::default()),
                Some(s) => s,
            };
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
            revision: inner.channelRev.Current(),
            ..Default::default()
        });
    }

    pub fn Store(&self) -> Option<Arc<dyn BackendStore>> {
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
            if self.read().unwrap().channelRev.Current() >= channelRev {
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
    pub backendStore: Option<Arc<dyn BackendStore>>,

    pub objectType: String, // like "pod"

    pub channelId: Uuid,

    pub cache: RingBuf,

    pub channelRev: ChannelRev,

    // ResourceVersion of the last list result (populated via Replace() method).
    pub listRevision: i64,

    pub lastWatcherId: u64,

    pub closeNotify: Arc<Notify>,

    pub revNotify: Arc<Notify>,

    pub cacheStore: BTreeMap<String, DataObject>,

    pub watchers: BTreeMap<u64, CacheWatcher>,
}

impl CacheStoreInner {
    pub fn New(
        store: Option<Arc<dyn BackendStore>>,
        objectType: &str,
        channelRev: &ChannelRev,
    ) -> Self {
        return Self {
            backendStore: store,
            objectType: objectType.to_string(),
            channelId: Uuid::new_v4(),
            cache: RingBuf::New(DEFAULT_CACHE_COUNT),
            channelRev: channelRev.clone(),
            listRevision: 0,
            lastWatcherId: 0,
            closeNotify: Arc::new(Notify::new()),
            revNotify: Arc::new(Notify::new()),
            cacheStore: BTreeMap::new(),
            watchers: BTreeMap::new(),
            //bgWorker: None,
        };
    }

    pub fn ChannelRev(&self) -> i64 {
        self.channelRev.Next()
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

        for (key, obj) in &self.cacheStore {
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

        return Ok(());
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
        let revision = event.obj.revision;
        let channelRev = self.ChannelRev();
        let mut wcEvent = WatchCacheEvent {
            type_: event.type_.DeepCopy(),
            obj: event.obj.CopyWithRev(channelRev, revision),
            prevObj: None,
            channelRev: channelRev,
            revision: revision,
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
                    return Ok(());
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

    pub fn GetAllEventsFromStore(&self, pred: &SelectionPredicate) -> Result<Vec<WatchCacheEvent>> {
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

        error!("GetAllEvents 1 {}/{}", revision, self.listRevision);
        let oldest = if self.listRevision >= 0 && self.cache.tail == 0 {
            // If no event was removed from the buffer since last relist, the oldest watch
            // event we can deliver is one greater than the resource version of the list.
            error!("GetAllEvents 2");
            self.listRevision + 1
        } else if size > 0 {
            self.cache.buf[self.cache.OldestIdx()]
                .as_ref()
                .unwrap()
                .channelRev
        } else {
            return Err(Error::CommonError(format!(
                "CacheStoreInner: watch cache isn't correctly initialized self.listRevision = {} tail = {}", 
                self.listRevision,
                self.cache.tail,
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

        error!("GetAllEvents 3");
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
