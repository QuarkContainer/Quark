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

use std::time::SystemTime;

use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{mpsc::channel, mpsc::Receiver, mpsc::Sender};


use crate::common::*;
use crate::selection_predicate::SelectionPredicate;
use crate::types::*;
use crate::types::{EventType, WatchEvent};


pub struct CacheWatchStream {
    pub id: u64,
    pub stream: Receiver<WatchEvent>,
}

#[derive(Debug)]
pub struct CacheWatcher {
    pub id: u64,

    pub sender: Sender<WatchEvent>,

    pub namespace: String, // expect Key

    pub predicate: SelectionPredicate,

    pub identifier: String,
}

impl CacheWatcher {
    pub fn New(
        id: u64,
        channelSize: usize,
        namespace: &str,
        predicate: SelectionPredicate,
        identifier: &str,
    ) -> (Self, CacheWatchStream) {
        let (tx, rx) = channel(channelSize);

        let w = Self {
            id: id,
            sender: tx,
            namespace: namespace.to_string(),
            predicate: predicate,
            identifier: identifier.to_string(),
        };

        return (w, CacheWatchStream { id: id, stream: rx });
    }

    pub fn Match(&self, objKey: &str, obj: &DataObject) -> Result<bool> {
        if !objKey.starts_with(&self.namespace) {
            return Ok(false);
        }

        return self.predicate.Match(obj);
    }

    pub fn ConvertToWatchEvent(&self, event: &WatchCacheEvent) -> Result<Option<WatchEvent>> {
        let curObjPasses =
            event.type_ != EventType::Deleted && self.Match(&event.Key(), &event.obj)?;
        let oldObjPasses = match &event.prevObj {
            None => false,
            Some(old) => self.Match(&event.Key(), old)?,
        };

        if curObjPasses && oldObjPasses {
            return Ok(Some(WatchEvent {
                type_: EventType::Modified,
                obj: event.obj.DeepCopy(),
            }));
        } else if curObjPasses && !oldObjPasses {
            return Ok(Some(WatchEvent {
                type_: EventType::Added,
                obj: event.obj.DeepCopy(),
            }));
        } else if !curObjPasses && oldObjPasses {
            let oldObj = event.prevObj.as_ref().unwrap().CopyWithRev(event.revision);
            return Ok(Some(WatchEvent {
                type_: EventType::Deleted,
                obj: oldObj,
            }));
        }

        return Ok(None);
    }

    pub fn SendWatchCacheEvent(&mut self, event: &WatchCacheEvent) -> Result<()> {
        let watchEvent = match self.ConvertToWatchEvent(event)? {
            None => return Ok(()),
            Some(event) => event,
        };

        match self.sender.try_send(watchEvent) {
            Ok(()) => return Ok(()),
            Err(e) => match e {
                TrySendError::Full(_) => return Err(Error::TokioChannFull),
                TrySendError::Closed(_) => return Err(Error::TokioChannClose),
            },
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

pub const BUFFER_SIZE: usize = 100;

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
        };
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
            return Some(event);
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

        return Ok((initRv, pods));
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

        assert!(list.objs.len() == 2, "objs is {:#?}", list.objs.len());
        for i in 0..list.objs.len() {
            assert!(
                preset[i + 1] == list.objs[i],
                "expect {:#?}, actual {:#?}",
                preset[i + 1],
                &list.objs[i]
            );
        }

        let obj = cacher.Get("second", "bar", 0).await?;
        match &obj {
            None => assert!(false),
            Some(o) => {
                assert!(
                    o == &preset[1],
                    "expect is {:#?}, actual is {:#?}",
                    &preset[1],
                    o
                );
            }
        }

        let obj = store.Get("/pods/second/bar", 0).await?;
        match &obj {
            None => assert!(false),
            Some(o) => {
                assert!(o == &preset[1]);
            }
        }

        let list = cacher.List("", &ListOption::default()).await?;
        assert!(list.objs.len() == 5);
        for i in 0..list.objs.len() {
            assert!(
                preset[i] == list.objs[i],
                "expect {:#?}, actual {:#?}",
                preset[i],
                &list.objs[i]
            );
        }

        let mut w = cacher
            .Watch("", list.revision + 1, SelectionPredicate::default())
            .await?;

        let bar1Second = DataObject::NewPod("second", "bar1", "", "")?;
        cacher.Create(&bar1Second).await?;
        let event = tokio::select! {
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
                assert!(
                    e.obj == bar1Second,
                    "expect is {:#?}, actual is {:#?}",
                    &preset[1],
                    e.obj
                );
            }
        }

        let bar1Second = DataObject::NewPod("second", "bar1", "abc", "")?;
        cacher.Update(&bar1Second).await?;

        let event = tokio::select! {
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
                assert!(
                    bar1Second == e.obj,
                    "expect is {:#?}, actual is {:#?}",
                    bar1Second,
                    e.obj
                );
            }
        }

        cacher.Delete("second", "bar1").await?;

        let event = w.stream.recv().await;
        match event {
            None => assert!(false, "event is {:#?}", event),
            Some(e) => {
                assert!(e.type_ == EventType::Deleted);
                let tmp = e.obj.CopyWithRev(bar1Second.Revision());
                assert!(
                    tmp == bar1Second,
                    "expect is {:#?}, actual is {:#?}",
                    &bar1Second,
                    tmp
                );
            }
        }

        return Ok(());
    }

    //#[test]
    pub fn RunTestCacher1Sync() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(RunTestCacher1()).unwrap();
    }

    pub async fn RunTestCacher2() -> Result<()> {
        let mut store = EtcdStore::New("localhost:2379", true).await?;

        let (_, preset) = SeedMultiLevelData(&mut store).await?;

        let listOptions = ListOption {
            revision: 0,
            ..Default::default()
        };

        let cacher = Cacher::New(&store, "pods", 0).await?;

        let list = cacher.List("second", &listOptions).await?;

        assert!(list.objs.len() == 2, "objs is {:#?}", list.objs.len());
        for i in 0..list.objs.len() {
            assert!(
                preset[i + 1] == list.objs[i],
                "expect {:#?}, actual {:#?}",
                preset[i + 1],
                &list.objs[i]
            );
        }

        let mut w = cacher
            .Watch("second", list.revision + 1, SelectionPredicate::default())
            .await?;
        let mut w2 = cacher
            .Watch("", list.revision + 1, SelectionPredicate::default())
            .await?;

        let bar1Second = DataObject::NewPod("second", "bar1", "", "")?;
        cacher.Create(&bar1Second).await?;
        let event = tokio::select! {
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
                assert!(
                    e.obj == bar1Second,
                    "expect is {:#?}, actual is {:#?}",
                    &preset[1],
                    e.obj
                );
            }
        }

        let event = tokio::select! {
            x = w2.stream.recv() => x,
            _ = tokio::time::sleep(Duration::from_millis(2000)) => {
                assert!(false, "can't get create event on time");
                return Ok(())
            }
        };
        match &event {
            None => assert!(false, "event1 is {:#?}", event),
            Some(e) => {
                assert!(e.type_ == EventType::Added);
                assert!(
                    e.obj == bar1Second,
                    "expect is {:#?}, actual is {:#?}",
                    &preset[1],
                    e.obj
                );
            }
        }

        let bar1Second = DataObject::NewPod("second", "bar1", "abc", "")?;
        cacher.Update(&bar1Second).await?;

        cacher.RemoveWatch(w2.id).await?;

        let event = tokio::select! {
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
                assert!(
                    bar1Second == e.obj,
                    "expect is {:#?}, actual is {:#?}",
                    bar1Second,
                    e.obj
                );
            }
        }

        let event = tokio::select! {
            x = w2.stream.recv() => x,
            _ = tokio::time::sleep(Duration::from_millis(2000)) => {
                assert!(false, "can't get create event on time");
                return Ok(())
            }
        };
        match event {
            None => assert!(false, "event2 is {:#?}", event),
            Some(e) => {
                assert!(e.type_ == EventType::Modified, "e is {:#?}", e);
                assert!(
                    bar1Second == e.obj,
                    "expect is {:#?}, actual is {:#?}",
                    bar1Second,
                    e.obj
                );
            }
        }

        cacher.RemoveWatch(w.id).await?;

        cacher.Delete("second", "bar1").await?;

        let event = w.stream.recv().await;
        match event {
            None => (),
            Some(e) => {
                assert!(false, "event is {:#?}", e)
            }
        }

        let event = tokio::select! {
            x = w2.stream.recv() => x,
            _ = tokio::time::sleep(Duration::from_millis(2000)) => {
                assert!(false, "can't get create event on time");
                return Ok(())
            }
        };
        match event {
            None => (),
            Some(_e) => {
                assert!(false);
            }
        }

        cacher.Stop().await?;

        return Ok(());
    }

    //#[test]
    pub fn RunTestCacher2Sync() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(RunTestCacher2()).unwrap();
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
        };
    }
}

impl WatchCacheEvent {
    pub fn Key(&self) -> String {
        return self.obj.Key();
    }
}

#[derive(Debug)]
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
        pred: &SelectionPredicate,
    ) -> Result<Vec<WatchCacheEvent>> {
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