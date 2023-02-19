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

use std::{sync::Arc, collections::BTreeMap, time::SystemTime};
use tokio::sync::{mpsc::error::TrySendError};
use tokio::sync::Mutex as TMutex;
use spin::Mutex;
use std::ops::Deref;
use tokio::sync::{mpsc::Receiver, Notify, mpsc::Sender, mpsc::channel};
use tokio::task::JoinHandle;
use crate::{etcd_store::{DataObject, EtcdStore}, watch::{EventType, WatchEvent}, selection_predicate::{SelectionPredicate}, types::DeepCopy};
use crate::shared::common::*;

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
    pub key: String,
    pub revision: i64,
    pub recordTime: SystemTime,
}

impl Default for WatchCacheEvent {
    fn default() -> Self {
        return Self {
            type_: Default::default(),
            obj: Default::default(),
            prevObj: Default::default(),
            key: Default::default(),
            revision: Default::default(),
            recordTime: SystemTime::now(),
        }
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
        return self.head - self.tail == self.buf.capacity();
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
pub struct Cacher(Arc<TMutex<CacherInner>>);

impl Deref for Cacher {
    type Target = Arc<TMutex<CacherInner>>;

    fn deref(&self) -> &Arc<TMutex<CacherInner>> {
        &self.0
    }
}

impl Cacher {
    pub async fn ProcessEvent(&self, event: &WatchEvent) -> Result<()> {
        let key = event.obj.Key();

        let mut wcEvent = WatchCacheEvent {
            type_: event.type_.DeepCopy(),
            obj: event.obj.clone(),
            prevObj: None,
            key: key.clone(),
            revision: event.obj.Revision(),
            recordTime: SystemTime::now(),
        };

        let mut inner = self.lock().await;
        match inner.cacheStore.get(&key) {
            None => (),
            Some(o) => {
                wcEvent.prevObj = Some(o.clone());
            }
        }

        let mut removeWatches = Vec::new();
        
        for (idx, w) in &mut inner.watchs {
            match w.SendWatchCacheEvent(&wcEvent) {
                Ok(()) => (),
                Err(_) => {
                    removeWatches.push(*idx);
                }
            }
        }

        for idx in removeWatches {
            inner.watchs.remove(&idx);
        }

        inner.cache.Push(wcEvent);

        if event.type_ == EventType::Deleted {
            inner.cacheStore.remove(&key);
        } else {
            inner.cacheStore.insert(key, event.obj.clone());
        }

        
        return Ok(())
    }

    pub async fn New(store: &EtcdStore, prefix: &str, rev: i64) -> Result<Self> {
        let inner = CacherInner::New(store, prefix);
        let notify = inner.notify.clone();
        let ret = Self(Arc::new(TMutex::new(inner)));

        let storeClone = store.Copy();
        let prefixClone = prefix.to_string();

        
        let watch = ret.clone();
        let future = tokio::spawn(async move {
            
            loop  {
                let (mut w, r) = storeClone.Watch(&prefixClone, rev, SelectionPredicate::default())?;
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
            
            }
        });

        ret.0.lock().await.future  = Some(future);

        return Ok(ret);
    }

    pub async fn Close(&self) {
        let notify = self.lock().await.notify.clone();
        notify.notify_waiters();
    }

    pub async fn Watch(&self, key: &str, revision: i64, predicate: SelectionPredicate) -> Result<CacheWatchStream> {
        let mut inner = self.lock().await;
        return inner.Watch(key, revision, predicate).await;
    }

}

pub const DEFAULT_CACHE_COUNT: usize = 2000;

pub struct CacherInner {
    pub etcdStore: EtcdStore,

    pub resourcePrefix: String,

    pub cache: RingBuf,

    // ResourceVersion up to which the watchCache is propagated.
    pub revision: i64,

    // ResourceVersion of the last list result (populated via Replace() method).
    pub listRevision: i64, 

    pub lastWatcherId: u64,

    pub notify: Arc<Notify>,

    pub cacheStore: BTreeMap<String, DataObject>,

    pub watchs: BTreeMap<u64, CacheWatcher>,

    pub future: Option<JoinHandle<Result<()>>>,
}

impl CacherInner {
    pub fn New(store: &EtcdStore, prefix: &str) -> Self {
        return Self {
            etcdStore: store.Copy(),
            resourcePrefix: prefix.to_string(),
            cache: RingBuf::New(DEFAULT_CACHE_COUNT),
            revision: 0,
            listRevision: 0,
            lastWatcherId: 0,
            notify: Arc::new(Notify::new()),
            cacheStore: BTreeMap::new(),
            watchs: BTreeMap::new(),
            future: None,
        }
    }

    pub fn GetByKey(&self, key: &str) -> Option<DataObject> {
        match self.cacheStore.get(key) {
            None => return None,
            Some(o) => return Some(o.clone()),
        }
    }
    
    pub async fn Create(&mut self, key: &str, obj: &DataObject) -> Result<()> {
        self.etcdStore.Create(key, &obj.obj).await?;
        return Ok(())
    }

    pub async fn Delete(&mut self, key: &str) -> Result<()> {
        match self.cacheStore.get(key) {
            None => {
                return self.etcdStore.Delete(key, 0).await;
            }
            Some(obj) => {
                return self.etcdStore.Delete(key, obj.Revision()).await;
            }
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
                key: obj.Key(),
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

    pub async fn Watch(&mut self, key: &str, revision: i64, predicate: SelectionPredicate) -> Result<CacheWatchStream> {
        self.lastWatcherId += 1;
        let wid = self.lastWatcherId;

        let events = self.GetAllEvents(revision, &predicate)?;
        let channelSize = (events.len() + 200).max(DEFAULT_CACHE_COUNT);

        let (mut watcher, stream) = 
            CacheWatcher::New(wid, channelSize, key, predicate, "");

        for e in events {
            watcher.SendWatchCacheEvent(&e)?;
        }

        self.watchs.insert(wid, watcher);
        return Ok(stream)
    }
}

pub struct CacheWatchStream {
    pub stream: Receiver<WatchEvent>,
}

pub struct CacheWatcher {
    pub id: u64,

    pub sender: Sender<WatchEvent>,

    pub filterKey: String, // expect Key
    
    pub predicate: SelectionPredicate,

    pub identifier: String,
}

impl CacheWatcher {
    pub fn New(id: u64, channelSize: usize, filterKey: &str, predicate: SelectionPredicate, identifier: &str) -> (Self, CacheWatchStream) {
        let (tx, rx) = channel(channelSize);

        let w = Self {
            id: id,
            sender: tx,
            filterKey: filterKey.to_string(),
            predicate: predicate,
            identifier: identifier.to_string(),
        };

        return (w, CacheWatchStream {
            stream: rx,
        });
    }

    pub fn Match(&self, objKey: &str, obj: &DataObject) -> Result<bool> {
        if !objKey.starts_with(&self.filterKey) {
            return Ok(false);
        }

        return self.predicate.Match(obj)
    }

    pub fn ConvertToWatchEvent(&self, event: &WatchCacheEvent) -> Result<Option<WatchEvent>> {
        let curObjPasses = event.type_ != EventType::Deleted && self.Match(&event.key, &event.obj)?;
        let oldObjPasses = match &event.prevObj {
            None => false,
            Some(old) => {
                self.Match(&event.key, old)?
            }
        };

        if !curObjPasses && !oldObjPasses {
            return Ok(None)
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