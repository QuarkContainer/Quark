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

use std::time::{Duration, Instant};
use std::{sync::Arc, collections::BTreeMap, time::SystemTime};
use tokio::sync::{mpsc::error::TrySendError};
use tokio::sync::RwLock as TRwLock;
use spin::Mutex;
use std::ops::{Deref};
use tokio::sync::{mpsc::Receiver, Notify, mpsc::Sender, mpsc::channel};
use tokio::task::JoinHandle;
use crate::etcd_store::DataObjList;
use crate::{etcd_store::{DataObject, EtcdStore}, watch::{EventType, WatchEvent}, selection_predicate::{SelectionPredicate}, types::DeepCopy};
use crate::shared::common::*;
use crate::ListOption;

pub struct ThreadSafeStoreInner {
    pub map: BTreeMap<String, DataObject>
}

pub struct ThreadSafeStore(Arc<Mutex<ThreadSafeStoreInner>>);

impl Deref for ThreadSafeStore {
    type Target = Arc<Mutex<ThreadSafeStoreInner>>;

    fn deref(&self) -> &Arc<Mutex<ThreadSafeStoreInner>> {
        &self.0
    }
}

impl ThreadSafeStore {
    pub fn Add(&self, key: &str, obj: DataObject) {
        self.Update(key, obj);
    }

    pub fn Update(&self, key: &str, obj: DataObject) {
        self.lock().map.insert(key.to_string(), obj);
    }

    pub fn Delete(&self, key: &str) {
        self.lock().map.remove(key);
    }

    pub fn List(&self) -> Vec<DataObject> {
        let list : Vec<DataObject> = self.lock().map.values().cloned().collect();
        return list;
    }

    pub fn ListKeys(&self) -> Vec<String> {
        return self.lock().map.keys().cloned().collect();
    }

    pub fn Replace(&self, items: BTreeMap<String, DataObject>) {
        self.lock().map = items;
    }

    pub fn GetByKey(&self, key: &str) -> Option<DataObject> {
        return match self.lock().map.get(key) {
            None => None,
            Some(v) => Some(v.clone())
        }
    }
}

pub struct Cache {
    pub store: ThreadSafeStore,
}

impl Cache {
    pub fn Add(&self, obj: DataObject) {
        let key = obj.Key();
        self.store.Add(&key, obj);
    }

    pub fn Update(&self, obj: DataObject) {
        let key = obj.Key();
        self.store.Update(&key, obj);
    }

    pub fn Delete(&self, obj: DataObject) {
        let key = obj.Key();
        self.store.Delete(&key);
    }

    pub fn List(&self) -> Vec<DataObject> {
        return self.store.List();
    }

    pub fn ListKeys(&self) -> Vec<String> {
        return self.store.ListKeys()
    }

    pub fn Replace(&self, list: Vec<DataObject>) {
        let mut map = BTreeMap::new();

        for obj in list {
            let key = obj.Key();
            map.insert(key, obj);
        }

        self.store.Replace(map);
    }

    pub fn GetByKey(&self, key: &str) -> Option<DataObject> {
        self.store.GetByKey(key)
    }
}

#[derive(Debug, Clone)]
pub struct WatchCacheEvent {
    pub type_: EventType,
    pub obj: DataObject,
    pub prevObj: Option<DataObject>,
    pub revision: i64,
    pub recordTime: SystemTime,
}

impl Default for WatchCacheEvent {
    fn default() -> Self {
        return Self {
            type_: Default::default(),
            obj: Default::default(),
            prevObj: Default::default(),
            revision: Default::default(),
            recordTime: SystemTime::now(),
        }
    }
}

impl WatchCacheEvent {
    pub fn Key(&self) -> String {
        return self.obj.Key();
    }
}

pub struct RingBuf {
    pub buf: Vec<Option<WatchCacheEvent>>,
    pub head: usize,
    pub tail: usize,
}

impl RingBuf {
    pub fn New(cap: usize) -> Self {
        let mut buf = Vec::with_capacity(cap);
        buf.resize(cap, None);
        return Self {
            buf: buf,
            head: 0,
            tail: 0,
        }
    }

    pub fn Reset(&mut self) {
        for i in 0..self.buf.len() {
            self.buf[i] = None;
        }

        self.head = 0;
        self.tail = 0;
    }

    pub fn GetAllEvents(&self, startRev: i64, pred: &SelectionPredicate) -> Result<Vec<WatchCacheEvent>> {
        let mut buf = Vec::new();
        for i in self.head..self.tail {
            let idx = i % self.buf.capacity();
            // todo: use binary search to find the first revision
            match &self.buf[idx] {
                None => return Err(Error::CommonError("watch cache isn't correctly initialized".to_owned())),
                Some(o) => {
                    if o.revision >= startRev && pred.Match(&o.obj)? {
                        buf.push(o.clone());
                    }
                }
            }
        }

        return Ok(buf);
    }

    pub fn OldestIdx(&self) -> usize {
        return self.head % self.buf.capacity();
    }

    pub fn Size(&self) -> usize {
        return self.tail - self.head;
    }

    pub fn Full(&self) -> bool {
        return self.tail - self.head == self.buf.capacity();
    }

    pub fn Push(&mut self, event: WatchCacheEvent) {
        if self.Full() {
            self.head += 1;
        }

        let idx = self.tail % self.buf.capacity();
        self.buf[idx] = Some(event);
        self.tail += 1;
    }

    pub fn Pop(&mut self) -> Option<WatchCacheEvent> {
        if self.head == self.tail {
            return None;
        }

        assert!(self.tail > self.head);
        let idx = self.head % self.buf.capacity();
        self.head += 1;
        return self.buf[idx].take();
    }
}

#[derive(Clone)]
pub struct Cacher(Arc<TRwLock<CacherInner>>);

impl Deref for Cacher {
    type Target = Arc<TRwLock<CacherInner>>;

    fn deref(&self) -> &Arc<TRwLock<CacherInner>> {
        &self.0
    }
}

impl Cacher {
    pub async fn ProcessEvent(&self, event: &WatchEvent) -> Result<()> {
        let mut inner = self.write().await;
        return inner.ProcessEvent(event).await;
    }


    pub async fn WaitUntilRev(&self, revision: i64) -> Result<()> {
        if revision == 0 { // 0 means current rev
            return Ok(())
        }

        let time = Instant::now();
        let until = time.checked_add(Duration::from_secs(3)).unwrap();

        let revNotify = self.read().await.revNotify.clone();
        loop {
            if self.read().await.revision >= revision {
                return Ok(())
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

    pub async fn New(store: &EtcdStore, prefix: &str, rev: i64) -> Result<Self> {
        let inner = CacherInner::New(store, prefix);
        let notify = inner.closeNotify.clone();
        let ret = Self(Arc::new(TRwLock::new(inner)));

        let storeClone = store.Copy();
        let prefixClone = prefix.to_string();

        
        let watch = ret.clone();
        let ready = Arc::new(Notify::new());
        let readyClone = ready.clone();
        let future = tokio::spawn(async move {
            let list = storeClone.List(&prefixClone, &ListOption {
                revision: rev,
                ..Default::default()
            }).await?;
            {
                let mut inner = watch.write().await;
                inner.listRevision = list.revision;
                inner.revision = list.revision;
                
                for o in list.objs {
                    let key = o.Key();
                    inner.cacheStore.insert(key, o);
                }
            }

            ready.notify_one();            
            loop  {
                let (mut w, r) = storeClone.Watch(&prefixClone, list.revision, SelectionPredicate::default())?;

                loop {
                    tokio::select! {
                        prossResult = w.Processing() => {
                            prossResult?;
                        }
                        event = r.GetNextEvent() => {
                            match event {
                                None => break,
                                Some(event) => {
                                    watch.ProcessEvent(&event).await?;
                                }
                            }
                        }
                        _ = notify.notified() => {
                            return Ok(())
                        }
                    }
                }

                let list = storeClone.List(&prefixClone, &ListOption::default()).await?;

                {
                    let mut inner = watch.write().await;
                     // close all watches
                     inner.watchers.clear();
                    // clear all cached data
                    inner.cacheStore.clear();
                    inner.cache.Reset();

                    inner.listRevision = list.revision;
                    inner.revision = list.revision;
                    
                    for o in list.objs {
                        let key = o.Key();
                        inner.cacheStore.insert(key, o);
                    }

                    inner.revNotify.notify_waiters();
                }
            }
        });

        ret.0.write().await.bgWorker  = Some(future);

        readyClone.notified().await;
        return Ok(ret);
    }

    pub async fn Stop(&self) -> Result<()> {
        let notify = self.read().await.closeNotify.clone();
        notify.notify_waiters();
        let worker = self.write().await.bgWorker.take();
        match worker {
            None => {
                return Ok(())
            },
            Some(w) => {
                return w.await?;
            }
        }
    }

    pub async fn Create(&self, obj: &DataObject) -> Result<()> {
        let store = self.read().await.etcdStore.Copy();
        let key = &self.read().await.StoreKey(&obj.Key());
        return store.Create(key, &obj).await;
    }

    pub async fn Delete(&self, namespace: &str, name: &str) -> Result<()> {
        let inner = self.write().await;
        let key = namespace.to_string() + "/" + name;
        let key = &inner.StoreKey(&key);
        match inner.cacheStore.get(key) {
            None => {
                return inner.etcdStore.Delete(key, 0).await;
            }
            Some(obj) => {
                return inner.etcdStore.Delete(key, obj.Revision()).await;
            }
        }
    }

    pub async fn Watch(&self, namespace: &str, revision: i64, predicate: SelectionPredicate) -> Result<CacheWatchStream> {
        let mut inner = self.write().await;
        return inner.Watch(namespace, revision, predicate).await;
    }

    pub async fn Get(&self, namespace: &str, name: &str, revision: i64) -> Result<Option<DataObject>> {
        let objKey = namespace.to_string() + "/" + name;
        if revision == -1 { // -1 means get from etcd
            let store = self.read().await.etcdStore.Copy();
            let key = &self.read().await.StoreKey(&objKey);
            return store.Get(key, revision).await;
        }

        self.WaitUntilRev(revision).await?;
        match self.read().await.cacheStore.get(&objKey) {
            None => return Ok(None),
            Some(o) => return Ok(Some(o.clone())),
        }
    }

    pub async fn List(&self, namespace: &str, opts: &ListOption) -> Result<DataObjList> {
        if opts.revision == -1 {
            let store = self.read().await.etcdStore.Copy();
            let mut opts = opts.DeepCopy();
            opts.revision = 0;
            let prefix = &self.read().await.StoreKey(namespace);
            

            return store.List(prefix, &opts).await;
        }

        self.WaitUntilRev(opts.revision).await?;

        let mut objs: Vec<DataObject> = Vec::new();

        let inner = self.read().await;
        for (_, obj) in &inner.cacheStore {
            if obj.Key().starts_with(namespace) && opts.predicate.Match(obj)? {
                objs.push(obj.clone());
            }
        }

        return Ok(DataObjList {
            objs: objs,
            revision: inner.revision,
            ..Default::default()
        })
    }

    pub async fn Update(&self, obj: &DataObject) -> Result<()> {
        let inner = self.read().await;
        let key = obj.Key();
        match self.read().await.cacheStore.get(&key) {
            None => {
                let key = &inner.StoreKey(&key);
                return inner.etcdStore.Update(key, 0, obj).await;
            },
            Some(o) => {    
                let rev = o.Revision();
                let key = &inner.StoreKey(&key);
                return inner.etcdStore.Update(key, rev, obj).await;
            }
        }
    }
}

pub const DEFAULT_CACHE_COUNT: usize = 2000;

pub struct CacherInner {
    pub etcdStore: EtcdStore,

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

    pub bgWorker: Option<JoinHandle<Result<()>>>,
}

impl CacherInner {
    pub fn New(store: &EtcdStore, objectType: &str) -> Self {
        return Self {
            etcdStore: store.Copy(),
            objectType: objectType.to_string(),
            cache: RingBuf::New(DEFAULT_CACHE_COUNT),
            revision: 0,
            listRevision: 0,
            lastWatcherId: 0,
            closeNotify: Arc::new(Notify::new()),
            revNotify: Arc::new(Notify::new()),
            cacheStore: BTreeMap::new(),
            watchers: BTreeMap::new(),
            bgWorker: None,
        }
    }

    pub fn StoreKey(&self, key: &str) -> String {
        return self.objectType.clone() + "/" + key;
    }

    pub async fn ProcessEvent(&mut self, event: &WatchEvent) -> Result<()> {
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
        return Ok(())
    }

    pub fn GetByKey(&self, key: &str) -> Option<DataObject> {
        match self.cacheStore.get(key) {
            None => return None,
            Some(o) => return Some(o.clone()),
        }
    }

    pub fn GetAllEventsFromStore(&self, revision: i64, pred: &SelectionPredicate) -> Result<Vec<WatchCacheEvent>> {
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

    pub fn GetAllEvents(&self, revision: i64, pred: &SelectionPredicate) -> Result<Vec<WatchCacheEvent>> {
        let size = self.cache.Size();

        let oldest = if self.listRevision > 0 && self.cache.tail == 0 {
            // If no event was removed from the buffer since last relist, the oldest watch
		    // event we can deliver is one greater than the resource version of the list.
		    self.listRevision + 1        
        } else if size > 0 {
            self.cache.buf[self.cache.OldestIdx()].as_ref().unwrap().revision
        } else {
            return Err(Error::CommonError(format!("watch cache isn't correctly initialized")))
        };

        if revision == 0 {
            // resourceVersion = 0 means that we don't require any specific starting point
            // and we would like to start watching from ~now.
            // However, to keep backward compatibility, we additionally need to return the
            // current state and only then start watching from that point.
            return Ok(self.GetAllEventsFromStore(revision, pred)?)
        } 

        if revision < oldest - 1 {
            return Err(Error::CommonError(format!("too old resource version: {} ({})", revision, oldest-1)));
        }

        return self.cache.GetAllEvents(revision, pred);
    }

    pub async fn Watch(&mut self, namespace: &str, revision: i64, predicate: SelectionPredicate) -> Result<CacheWatchStream> {
        self.lastWatcherId += 1;
        let wid = self.lastWatcherId;

        let events = self.GetAllEvents(revision, &predicate)?;
        let channelSize = (events.len() + 200).max(DEFAULT_CACHE_COUNT);

        let (mut watcher, stream) = 
            CacheWatcher::New(wid, channelSize, namespace, predicate, "");

        for e in events {
            watcher.SendWatchCacheEvent(&e)?;
        }

        self.watchers.insert(wid, watcher);
        return Ok(stream)
    }
}

pub struct CacheWatchStream {
    pub stream: Receiver<WatchEvent>,
}

pub struct CacheWatcher {
    pub id: u64,

    pub sender: Sender<WatchEvent>,

    pub namespace: String, // expect Key
    
    pub predicate: SelectionPredicate,

    pub identifier: String,
}

impl CacheWatcher {
    pub fn New(id: u64, channelSize: usize, namespace: &str, predicate: SelectionPredicate, identifier: &str) -> (Self, CacheWatchStream) {
        let (tx, rx) = channel(channelSize);

        let w = Self {
            id: id,
            sender: tx,
            namespace: namespace.to_string(),
            predicate: predicate,
            identifier: identifier.to_string(),
        };

        return (w, CacheWatchStream {
            stream: rx,
        });
    }

    pub fn Match(&self, objKey: &str, obj: &DataObject) -> Result<bool> {
        if !objKey.starts_with(&self.namespace) {
            return Ok(false);
        }

        return self.predicate.Match(obj)
    }

    pub fn ConvertToWatchEvent(&self, event: &WatchCacheEvent) -> Result<Option<WatchEvent>> {
        let curObjPasses = event.type_ != EventType::Deleted && self.Match(&event.Key(), &event.obj)?;
        let oldObjPasses = match &event.prevObj {
            None => false,
            Some(old) => {
                self.Match(&event.Key(), old)?
            }
        };

        if curObjPasses && oldObjPasses {
            return Ok(Some(WatchEvent { type_:  EventType::Modified, obj: event.obj.DeepCopy()}));
        } else if curObjPasses && !oldObjPasses {
            return Ok(Some(WatchEvent { type_:  EventType::Added, obj: event.obj.DeepCopy()}));
        } else if !curObjPasses && oldObjPasses {
            let oldObj = event.prevObj.as_ref().unwrap().DeepCopy();
            oldObj.SetRevision(event.revision);
            return Ok(Some(WatchEvent { type_:  EventType::Deleted, obj: oldObj}));
        } 

        return Ok(None)
    }

    pub fn SendWatchCacheEvent(&mut self, event: &WatchCacheEvent) -> Result<()> {
        let watchEvent = match self.ConvertToWatchEvent(event)? {
            None => return Ok(()),
            Some(event) => event,
        };

        match self.sender.try_send(watchEvent) {
            Ok(()) => return Ok(()),
            Err(e) => {
                match e {
                    TrySendError::Full(_) => return Err(Error::TokioChannFull),
                    TrySendError::Closed(_) => return Err(Error::TokioChannClose),
                }
            }
        }
    }
}

pub struct WatchCacheInterval {
    // startIndex denotes the starting point of the interval
	// being considered. The value is the index in the actual
	// source of watchCacheEvents. If the source of events is
	// the watchCache, then this must be used modulo capacity.
    pub startIdx: usize,

    pub endIdx: usize,
}

pub const BUFFER_SIZE : usize = 100;

pub struct WatchCacheBuf {
    pub buffer: Vec<Option<WatchCacheEvent>>,

    pub head: usize,
    pub tail: usize,
}

impl WatchCacheBuf {
    pub fn New() -> Self {
        let cap = BUFFER_SIZE;
        let mut buf = Vec::with_capacity(cap);
        buf.resize(cap, None);
        return Self {
            buffer: buf,
            head: 0,
            tail: 0,
        }
    }

    pub fn IsEmpty(&self) -> bool {
        return self.head == self.tail;
    }

    pub fn IsFull(&self) -> bool {
        return self.tail - self.head == self.buffer.capacity();
    }

    pub fn Reset(&mut self) {
        self.head = 0;
        self.tail = 0;
    }

    pub fn Push(&mut self, event: WatchCacheEvent) -> Option<WatchCacheEvent> {
        if self.IsFull() {
            return Some(event)
        };

        let idx = self.tail % self.buffer.capacity();
        self.buffer[idx] = Some(event);
        self.tail += 1;
        return None;
    }

    pub fn Pop(&mut self) -> Option<WatchCacheEvent> {
        if self.IsEmpty() {
            return None;
        }

        let idx = self.head % self.buffer.capacity();
        self.head += 1;
        return self.buffer[idx].take();
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    //use super::super::selector::*;

    pub fn ComputePodKey(obj: &DataObject) -> String {
        return format!("/pods/{}/{}", &obj.Namespace(), &obj.Name());
    }    

    // SeedMultiLevelData creates a set of keys with a multi-level structure, returning a resourceVersion
    // from before any were created along with the full set of objects that were persisted
    async fn SeedMultiLevelData(store: &mut EtcdStore) -> Result<(i64, Vec<DataObject>)> {
        // Setup storage with the following structure:
        //  /
        //   - first/
        //  |         - bar
        //  |
        //   - second/
        //  |         - bar
        //  |         - fooo
        //  |
        //   - third/
        //  |         - barfooo
        //  |         - fooo
        let barFirst = DataObject::NewPod("first", "bar", "", "")?;
        let barSecond = DataObject::NewPod("second", "bar", "", "")?;
        let foooSecond = DataObject::NewPod("second", "fooo", "", "")?;
        let barfoooThird = DataObject::NewPod("third", "barfooo", "", "")?;
        let foooThird = DataObject::NewPod("third", "fooo", "", "")?;

        struct Test {
            key: String,
            obj: DataObject,
        }

        let mut tests = [
        Test {
            key: ComputePodKey(&barFirst),
            obj: barFirst,
        },
        Test {
            key: ComputePodKey(&barSecond),
            obj: barSecond,
        },
        Test {
            key: ComputePodKey(&foooSecond),
            obj: foooSecond,
        },
        Test {
            key: ComputePodKey(&barfoooThird),
            obj: barfoooThird,
        },
        Test {
            key: ComputePodKey(&foooThird),
            obj: foooThird,
        },
    ];

        let initRv = store.Clear("").await?;
        for t in &mut tests {
            store.Create(&t.key, &t.obj).await?;
        }

        let mut pods = Vec::new();
        for t in tests {
            pods.push(t.obj);
        }

        return Ok((initRv, pods))
    }

    pub async fn RunTestCacher1() -> Result<()> {
        let mut store = EtcdStore::New("localhost:2379", true).await?;

        let (_, preset) = SeedMultiLevelData(&mut store).await?;
        
        let listOptions = ListOption {
            revision: 0,
            ..Default::default()
        };

        let cacher = Cacher::New(&store, "pods", 0).await?;
        
        let list = cacher.List("second", &listOptions).await?;

        assert!(list.objs.len()==2, "objs is {:#?}", list.objs.len());
        for i in 0..list.objs.len() {
            assert!(preset[i+1]==list.objs[i], 
                "expect {:#?}, actual {:#?}", preset[i+1], &list.objs[i]);
        }

        let obj = cacher.Get("second", "bar", 0).await?;
        match &obj {
            None => assert!(false),
            Some(o) => {
                assert!(o==&preset[1], "expect is {:#?}, actual is {:#?}", &preset[1], o);
            }
        }
        
        let obj = store.Get("/pods/second/bar", 0).await?;
        match &obj {
            None => assert!(false),
            Some(o) => {
                assert!(o==&preset[1]);
            }
        }

        let list = cacher.List("", &ListOption::default()).await?;
        assert!(list.objs.len() == 5);
        
        let mut w = cacher.Watch("", list.revision+1, SelectionPredicate::default()).await?;
        
        let bar1Second = DataObject::NewPod("second", "bar1", "", "")?; 
        cacher.Create(&bar1Second).await?;
        let event =
            tokio::select! {
                x = w.stream.recv() => x,
                _ = tokio::time::sleep(Duration::from_millis(2000)) => {
                    assert!(false, "can't get create event on time");
                    return Ok(())
                }
            };
        match &event {
            None => assert!(false, "event1 is {:#?}", event),
            Some(e) => {
                assert!(e.type_ == EventType::Added);
                assert!(e.obj == bar1Second, "expect is {:#?}, actual is {:#?}", &preset[1], e.obj);
            }
        }

        let bar1Second = DataObject::NewPod("second", "bar1", "abc", "")?; 
        cacher.Update(&bar1Second).await?;

        let event =
            tokio::select! {
                x = w.stream.recv() => x,
                _ = tokio::time::sleep(Duration::from_millis(2000)) => {
                    assert!(false, "can't get create event on time");
                    return Ok(())
                }
            };
        match event {
            None => assert!(false, "event2 is {:#?}", event),
            Some(e) => {
                assert!(e.type_ == EventType::Modified, "e is {:#?}", e);
                assert!(bar1Second == e.obj, "expect is {:#?}, actual is {:#?}", bar1Second, e.obj);
            }
        }
        
        
        cacher.Delete("second", "bar1").await?;

        let event = w.stream.recv().await;
        match event {
            None => assert!(false, "event is {:#?}", event),
            Some(e) => {
                assert!(e.type_ == EventType::Deleted);
                e.obj.SetRevision(bar1Second.Revision());
                assert!(e.obj == bar1Second, "expect is {:#?}, actual is {:#?}", &bar1Second, e.obj);
            }
        }

        return Ok(())
    }

    #[test]
    pub fn RunTestCacher1Sync() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build().unwrap();
        
        rt.block_on(RunTestCacher1()).unwrap();
    }
}