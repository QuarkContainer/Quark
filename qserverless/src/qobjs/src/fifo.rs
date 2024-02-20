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


use std::collections::{BTreeMap, VecDeque, HashSet};
use std::sync::Arc;
use spin::RwLock;
use tokio::sync::Notify;
use core::ops::Deref;

use crate::types::DataObject;
use crate::common::*;
use crate::store::Store;

// PopProcessFunc is passed to Pop() method of Queue interface.
// It is supposed to process the accumulator popped from the queue.
pub type PopProcess = fn(_obj: &DataObject, _isInInitialList: bool) -> Result<()>;

// Queue extends Store with a collection of Store keys to "process".
// Every Add, Update, or Delete may put the object's key in that collection.
// A Queue has a way to derive the corresponding key given an accumulator.
// A Queue can be accessed concurrently from multiple goroutines.
// A Queue can be "closed", after which Pop operations return an error.
pub trait Queue : Store {
    // Pop blocks until there is at least one key to process or the
	// Queue is closed.  In the latter case Pop returns with an error.
	// In the former case Pop atomically picks one key to process,
	// removes that (key, accumulator) association from the Store, and
	// processes the accumulator.  Pop returns the accumulator that
	// was processed and the result of processing.  The PopProcessFunc
	// may return an ErrRequeue{inner} and in this case Pop will (a)
	// return that (key, accumulator) association to the Queue as part
	// of the atomic processing and (b) return the inner error from
	// Pop.
    
    // rust doesn't support async in trait now
    //async fn Pop(&self, _f: PopProcess) -> Result<()>;

    // AddIfNotPresent puts the given accumulator into the Queue (in
	// association with the accumulator's key) if and only if that key
	// is not already associated with a non-empty accumulator.
	fn AddIfNotPresent(&self, _obj: &DataObject) -> Result<()> {
        todo!();
    }

    // HasSynced returns true if the first batch of keys have all been
	// popped.  The first batch of keys are those of the first Replace
	// operation if that happened before any Add, AddIfNotPresent,
	// Update, or Delete; otherwise the first batch is empty.
	fn HasSynced(&self) -> bool {
        todo!();
    }

    // Close the queue
	fn Close(&self) {
        todo!();
    }
}

#[derive(Debug, Default)]
pub struct FifoInnner {
    pub items: BTreeMap<String, DataObject>,
    pub queue: VecDeque<String>,

    // populated is true if the first batch of items inserted by Replace() has been populated
	// or Delete/Add/Update was called first.
	pub populated: bool, 

    // initialPopulationCount is the number of items inserted by the first call of Replace()
    pub initialPopulationCount: usize,

    // Indication the queue is closed.
	// Used to indicate a queue is closed so a control loop can exit when a queue is empty.
	// Currently, not used to gate any of CRUD operations.
	pub closed: bool,

    pub notify: Arc<Notify>,
}

#[derive(Debug, Default, Clone)]
pub struct Fifo(Arc<RwLock<FifoInnner>>);

impl Deref for Fifo {
    type Target = Arc<RwLock<FifoInnner>>;

    fn deref(&self) -> &Arc<RwLock<FifoInnner>> {
        &self.0
    }
}

impl Store for Fifo {
    // Add adds the given object to the accumulator associated with the given object's key
    fn Add(&self, obj: &DataObject) -> Result<()> {
        let key = obj.Key();

        let mut inner = self.write();
        inner.populated = true;

        match inner.items.insert(key.clone(), obj.clone()) {
            None => {
                inner.queue.push_back(key);
            }
            _ => (),
        }

        inner.notify.notify_waiters();
        return Ok(())
    }

    // Update updates the given object in the accumulator associated with the given object's key
    fn Update(&self, obj: &DataObject) -> Result<()> {
        return self.Add(obj)
    }

    // Delete deletes the given object from the accumulator associated with the given object's key
    fn Delete(&self, obj: &DataObject) ->  Result<()> {
        let key = obj.Key();
        
        let mut inner = self.write();
        inner.populated = true;

        inner.items.remove(&key);
        return Ok(())
    }

    // List returns a list of all the currently non-empty accumulators
    fn List(&self) -> Vec<DataObject> {
        let inner = self.read();

        let mut ret = Vec::new();
        for (_, o) in &inner.items {
            ret.push(o.clone());
        }

        return ret
    }

    // ListKeys returns a list of all the keys currently associated with non-empty accumulators
    fn ListKeys(&self) -> Vec<String>{
        let inner = self.read();
        let mut ret = Vec::new();
        for (k, _) in &inner.items {
            ret.push(k.clone());
        }

        return ret
    }

    // Get returns the accumulator associated with the given object's key
    fn Get(&self, key: &str) -> Option<DataObject> {
        let inner = self.read();
        match inner.items.get(key) {
            None => return None,
            Some(o) => return Some(o.clone()),
        }
    }

    // Replace will delete the contents of the store, using instead the
    // given list. Store takes ownership of the list, you should not reference
    // it after calling this function.
    fn Replace(&self, objs: &[DataObject]) -> Result<()> {
        let mut items  = BTreeMap::new();
        for obj in objs {
            let key = obj.Key();
            items.insert(key, obj.clone());
        }

        let mut inner = self.write();

        if !inner.populated {
            inner.populated = true;
            inner.initialPopulationCount = objs.len();
        }

        inner.items = items;
        inner.queue.clear();

        for obj in objs {
            inner.queue.push_back(obj.Key());
        }

        if inner.queue.len() > 0 {
            inner.notify.notify_waiters();
        }

        return Ok(())
    }

    // Resync will ensure that every object in the Store has its key in the queue.
    // This should be a no-op, because that property is maintained by all operations.
    fn Resync(&self) -> Result<()> {
        let mut inner = self.write();

        let mut inQueue = HashSet::new();
        for id in &inner.queue {
            inQueue.insert(id.clone());
        }

        let mut keys = Vec::new();
        for k in inner.items.keys() {
            keys.push(k.clone());
        }
        for k in keys {
            if !inQueue.contains(&k) {
                inner.queue.push_back(k);
            }
        }

        if inner.queue.len() > 0 {
            inner.notify.notify_waiters();
        }

        return Ok(())
    }
}

impl FifoInnner {
    fn HasSyncedLocked(&self) -> bool {
        return self.populated && self.initialPopulationCount == 0;
    }

    fn AddIfNotPresentLock(&mut self, obj: &DataObject) -> Result<()> {
        let key = obj.Key();
        self.populated = true;

        if self.items.contains_key(&key) {
            return Ok(())
        }

        self.queue.push_back(key.clone());
        self.items.insert(key, obj.clone());
        self.notify.notify_waiters();

        return Ok(())
    }
}

impl Queue for Fifo {
    // AddIfNotPresent puts the given accumulator into the Queue (in
	// association with the accumulator's key) if and only if that key
	// is not already associated with a non-empty accumulator.
	fn AddIfNotPresent(&self, obj: &DataObject) -> Result<()> {
        let mut inner = self.write();

        return inner.AddIfNotPresentLock(obj);
    }

    // HasSynced returns true if the first batch of keys have all been
	// popped.  The first batch of keys are those of the first Replace
	// operation if that happened before any Add, AddIfNotPresent,
	// Update, or Delete; otherwise the first batch is empty.
	fn HasSynced(&self) -> bool {
        let inner = self.read();

        return inner.HasSyncedLocked();
    }

    // Close the queue
	fn Close(&self) {
        let mut inner = self.write();

        inner.closed = true;
        inner.notify.notify_waiters();
    }
}

impl Fifo {
    // IsClosed checks if the queue is closed
    pub fn IsClosed(&self) -> bool {
        return self.read().closed;
    }

    // Pop blocks until there is at least one key to process or the
	// Queue is closed.  In the latter case Pop returns with an error.
	// In the former case Pop atomically picks one key to process,
	// removes that (key, accumulator) association from the Store, and
	// processes the accumulator.  Pop returns the accumulator that
	// was processed and the result of processing.  The PopProcessFunc
	// may return an ErrRequeue{inner} and in this case Pop will (a)
	// return that (key, accumulator) association to the Queue as part
	// of the atomic processing and (b) return the inner error from
	// Pop.
    async fn Pop(&self, process: PopProcess) -> Result<DataObject> {
        let mut inner;

        let notify = self.read().notify.clone();
        loop {
            loop {
                inner = self.write();
                if inner.queue.len() == 0 {
                    drop(inner);
                } else {
                    break;
                }

                notify.notified().await;
            }

            let isInInitialList = !inner.HasSyncedLocked();
            let id = inner.queue.pop_front().unwrap();

            if inner.initialPopulationCount > 0 {
                inner.initialPopulationCount -= 1;
            }

            let obj = match inner.items.remove(&id) {
                None => continue,
                Some(o) => o,
            };

            match process(&obj, isInInitialList) {
                Err(Error::ErrRequeue) => {
                    inner.AddIfNotPresentLock(&obj)?;
                    return Err(Error::ErrRequeue)
                }
                Err(e) => return Err(e),
                Ok(()) => return Ok(obj),
            }
        }
    }
}