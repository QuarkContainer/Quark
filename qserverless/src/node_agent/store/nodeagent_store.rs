// Copyright (c) 2021 Quark Container Authors
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

use std::sync::Mutex;
use std::{collections::BTreeMap, sync::Arc};
use std::ops::Deref;

use tokio::sync::{mpsc::Receiver};
use tokio::sync::mpsc::channel;
use tokio::sync::mpsc::Sender;
use tokio::sync::mpsc::error::TrySendError;

use k8s_openapi::api::core::v1 as k8s;

use qobjs::runtime_types::{QuarkPodJson, QuarkPod};
use qobjs::common::*;

use super::rocksdb::{RocksObjStore, RocksStore};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum EventType {
    None,
    Added,
    Modified,
    Deleted,
    Error(String),
}

impl EventType {
    pub fn DeepCopy(&self) -> Self {
        match self {
            Self::None => return Self::None,
            Self::Added => return Self::Added,
            Self::Modified => return Self::Modified,
            Self::Deleted => return Self::Deleted,
            Self::Error(str) => return Self::Error(str.to_string()),
        }
    }
}

impl Default for EventType {
    fn default() -> Self {
        return Self::None;
    }
}
#[derive(Debug, Clone)]
pub struct WatchEvent {
    pub type_: EventType,
    pub revision: i64,

    pub obj: NodeAgentEventObj,
}

#[derive(Debug, Clone)]
pub enum NodeAgentEventObj {
    Pod(k8s::Pod),
    Node(k8s::Node),
}

#[derive(Debug)]
pub struct RingBuf {
    pub buf: Vec<Option<WatchEvent>>,
    pub head: usize,
    pub tail: usize,
}

impl RingBuf {
    pub fn New(cap: usize) -> Self {
        let mut buf = Vec::with_capacity(cap);
        for _i in 0..cap {
            buf.push(None);
        }
        
        return Self {
            buf: buf,
            head: 0,
            tail: 0,
        };
    }

    pub fn Reset(&mut self) {
        for i in 0..self.buf.len() {
            self.buf[i] = None;
        }

        self.head = 0;
        self.tail = 0;
    }

    pub fn GetAllEvents(
        &self,
        startRev: i64,
    ) -> Result<Vec<WatchEvent>> {
        let mut buf = Vec::new();
        for i in self.head..self.tail {
            let idx = i % self.buf.capacity();
            // todo: use binary search to find the first revision
            match &self.buf[idx] {
                None => {
                    return Err(Error::CommonError(
                        "watch cache isn't correctly initialized".to_owned(),
                    ))
                }
                Some(o) => {
                    if o.revision >= startRev {
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

    pub fn Push(&mut self, event: WatchEvent) {
        if self.Full() {
            self.head += 1;
        }

        let idx = self.tail % self.buf.capacity();
        self.buf[idx] = Some(event);
        self.tail += 1;
    }

    pub fn Pop(&mut self) -> Option<WatchEvent> {
        if self.head == self.tail {
            return None;
        }

        assert!(self.tail > self.head);
        let idx = self.head % self.buf.capacity();
        self.head += 1;
        return self.buf[idx].take();
    }
}

#[derive(Debug)]
pub struct NodeAgentStoreInner {
    pub podStore: RocksObjStore<QuarkPodJson>,

    pub eventQueue: RingBuf,

    pub podCache: BTreeMap<String, k8s::Pod>,
    pub nodeCache: BTreeMap<String, k8s::Node>,

    pub revision: i64,
    pub listRevision: i64,

    pub lastWatcherId: u64,

    pub watchers: BTreeMap<u64, StoreWatcher>,
}

impl NodeAgentStoreInner {
    pub fn New() -> Result<(Self, Vec<QuarkPodJson>)> {
        let store = RocksStore::New()?;
        let podStore: RocksObjStore<QuarkPodJson> = store.NewObjStore("pods");

        // todo: why does this affect cadvisor?
        let pods = podStore.Load()?;
        let mut podCache = BTreeMap::new();
        for p in &pods {
            let key = p.id.clone();
            podCache.insert(key, p.pod.clone());
        }

        return Ok((Self {
            podStore: podStore,
            eventQueue: RingBuf::New(2000),
            podCache: podCache,
            nodeCache: BTreeMap::new(),
            revision: store.initRev,
            listRevision: store.initRev,
            lastWatcherId: 0,
            watchers: BTreeMap::new(),
        }, pods))
    }

    pub fn CreatePod(&mut self, obj: &QuarkPod) -> Result<()> {
        let key = obj.lock().unwrap().id.clone();

        self.revision += 1;
        obj.Pod().write().unwrap().metadata.resource_version = Some(self.revision.to_string());
        let jsonObj = obj.ToQuarkPodJson();
        let event = WatchEvent {
            type_: EventType::Added,
            revision: self.revision,
            obj: NodeAgentEventObj::Pod(jsonObj.pod.clone())
        };
        self.podCache.insert(key.to_string(), jsonObj.pod.clone());
        self.podStore.Save(self.revision, &key, &jsonObj)?;
        self.ProcessEvent(event)?;
        return Ok(())
    }

    pub fn UpdatePod(&mut self, obj: &QuarkPod) -> Result<()> {
        let key = obj.lock().unwrap().id.clone();
        self.revision += 1;
        obj.Pod().write().unwrap().metadata.resource_version = Some(self.revision.to_string());
        let jsonObj = obj.ToQuarkPodJson();
        assert!(self.podCache.contains_key(&key));
        self.podCache.insert(key.clone(), jsonObj.pod.clone());
        self.podStore.Save(self.revision, &key, &jsonObj)?;
        if !obj.PodInTerminating() {
            let event = WatchEvent {
                type_: EventType::Modified,
                revision: self.revision,
                obj: NodeAgentEventObj::Pod(jsonObj.pod.clone())
            };
            self.ProcessEvent(event)?;
        }
        
        return Ok(())
    }

    pub fn DeletePod(&mut self, key: &str) -> Result<()> {
        assert!(self.podCache.contains_key(key));
        self.revision += 1;
        let mut pod = self.podCache.remove(key).unwrap();
        pod.metadata.resource_version = Some(self.revision.to_string());
        let event = WatchEvent {
            type_: EventType::Deleted,
            revision: self.revision,
            obj: NodeAgentEventObj::Pod(pod)
        };

        self.podStore.Remove(self.revision, key)?;
        self.ProcessEvent(event)?;
        return Ok(())
    }

    pub fn GetAllEvents(
        &self,
        revision: i64
    ) -> Result<Vec<WatchEvent>> {
        let size = self.eventQueue.Size();
        
        let oldest = if self.listRevision > 0 && self.eventQueue.tail == 0 {
            // If no event was removed from the buffer since last relist, the oldest watch
            // event we can deliver is one greater than the resource version of the list.
            self.listRevision + 1
        } else if size > 0 {
            self.eventQueue.buf[self.eventQueue.OldestIdx()]
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
            return self.GetAllEventsFromStore(revision);
        }

        if revision < oldest - 1 {
            return Err(Error::CommonError(format!(
                "too old resource version: {} ({})",
                revision,
                oldest - 1
            )));
        }

        return self.eventQueue.GetAllEvents(revision); 
    }

    pub fn GetAllEventsFromStore(
        &self,
        revision: i64
    ) -> Result<Vec<WatchEvent>> {
        let mut buf = Vec::new();

        for (_, pod) in &self.podCache {
            buf.push(WatchEvent {
                type_: EventType::Added,
                revision: revision,
                obj: NodeAgentEventObj::Pod(pod.clone())
            });
        }

        for (_, node) in &self.nodeCache {
            buf.push(WatchEvent {
                type_: EventType::Added,
                revision: revision,
                obj: NodeAgentEventObj::Node(node.clone())
            });
        }

        return Ok(buf);
    }

    pub fn ProcessEvent(&mut self, event: WatchEvent) -> Result<()> {
        let mut removeWatches = Vec::new();
        for (idx, w) in &mut self.watchers {
            match w.SendWatchEvent(event.clone()) {
                Ok(()) => (),
                Err(_) => {
                    removeWatches.push(*idx);
                }
            }
        }

        for idx in removeWatches {
            self.watchers.remove(&idx);
        }
        self.eventQueue.Push(event);

        return Ok(())
    }

    pub const DEFAULT_CACHE_COUNT: usize = 2000;
    pub fn Watch(&mut self, revision: i64) -> Result<StoreWatchStream> {
        self.lastWatcherId += 1;
        let wid = self.lastWatcherId;

        let events = self.GetAllEvents(revision)?;
        let channelSize = (events.len() + 200).max(Self::DEFAULT_CACHE_COUNT);

        let (mut watcher, stream) = StoreWatcher::New(wid, channelSize);

        for e in events {
            watcher.SendWatchEvent(e)?;
        }

        self.watchers.insert(wid, watcher);
        return Ok(stream);
    }

    pub fn List(&self) -> PodList {
        let pods : Vec<k8s::Pod> = self.podCache.values().cloned().collect();
        return PodList { 
            revision: self.revision, 
            pods: pods 
        };
    }
}

#[derive(Debug)]
pub struct PodList {
    pub revision: i64,
    pub pods: Vec<k8s::Pod>,
}

#[derive(Debug)]
pub struct NodeAgentStore(Arc<Mutex<NodeAgentStoreInner>>);

impl Deref for NodeAgentStore {
    type Target = Arc<Mutex<NodeAgentStoreInner>>;

    fn deref(&self) -> &Arc<Mutex<NodeAgentStoreInner>> {
        &self.0
    }
}

impl NodeAgentStore {
    pub fn New() -> Result<(Self, Vec<QuarkPodJson>)> {
        let (inner, pods) = NodeAgentStoreInner::New()?;
        let store = Self(Arc::new(Mutex::new(inner)));
        return Ok((store, pods))
    }

    pub fn CreatePod(&self, obj: &QuarkPod) -> Result<()> {
        return self.lock().unwrap().CreatePod(obj);
    }

    pub fn UpdatePod(&self, obj: &QuarkPod) -> Result<()> {
        return self.lock().unwrap().UpdatePod(obj);
    }

    pub fn DeletePod(&self, key: &str) -> Result<()> {
        return self.lock().unwrap().DeletePod(key);
    }

    pub fn List(&self) -> PodList {
        return self.lock().unwrap().List();
    }

    pub fn Watch(&self, revision: i64) -> Result<StoreWatchStream> {
        return self.lock().unwrap().Watch(revision);
    }
}

#[derive(Debug)]
pub struct StoreWatcher {
    pub id: u64,
    pub sender: Sender<WatchEvent>,
}

impl StoreWatcher {
    pub fn New(id: u64, channelSize: usize) -> (Self, StoreWatchStream) {
        let (tx, rx) = channel(channelSize);
        let w = Self {
            id: id,
            sender: tx, 
        };

        return (w, StoreWatchStream {
            id: id,
            stream: rx,
        })
    }

    pub fn SendWatchEvent(&mut self, event: WatchEvent) -> Result<()> {
        match self.sender.try_send(event) {
            Ok(()) => return Ok(()),
            Err(e) => match e {
                TrySendError::Full(_) => return Err(Error::TokioChannFull),
                TrySendError::Closed(_) => return Err(Error::TokioChannClose),
            },
        }
    }
}

pub struct StoreWatchStream {
    pub id: u64,
    pub stream: Receiver<WatchEvent>,
}
