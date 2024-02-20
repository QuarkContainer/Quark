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

use spin::RwLock;
use tokio::sync::Notify;
use core::ops::Deref;
use std::{collections::{VecDeque, BTreeMap, BTreeSet}, sync::Arc};

use crate::{store::{ThreadSafeStore, Store}, types::DataObject};
use crate::common::*;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum DeltaType {
    Added,
    Updated,
    Deleted,

    // Replaced is emitted when we encountered watch errors and had to do a
	// relist. We don't know if the replaced object has changed.
	//
	// NOTE: Previous versions of DeltaFIFO would use Sync for Replace events
	// as well. Hence, Replaced is only emitted when the option
	// EmitDeltaTypeReplaced is true.
	Replaced,
    // Sync is for synthetic events during a periodic resync.
	Sync,
}

#[derive(Debug, Clone)]
pub struct Delta {
    pub type_: DeltaType,
    pub obj: DataObject,
}

impl Delta {
    pub fn IsDup(&self, b: &Self) -> bool {
        return self.IsDeletionDup(b);
    }

    pub fn IsDeletionDup(&self, b: &Self) -> bool {
        return self.type_ == DeltaType::Deleted || b.type_ == DeltaType::Deleted;
    }
}

#[derive(Debug)]
pub struct Deltas(VecDeque<Delta>);

impl Deltas {
    pub fn Newest(&self) -> Option<Delta> {
        return self.0.back().cloned();
    }

    pub fn Oldest(&self) -> Option<Delta> {
        return self.0.front().cloned();
    }

    pub fn DeepCopy(&self) -> Self {
        let mut q = VecDeque::new();
        for d in &self.0 {
            q.push_back(d.clone())
        }

        return Self(q);
    }
}

#[derive(Debug, Default)]
pub struct DeltaFifoInner {
    pub store: ThreadSafeStore,

    // `items` maps a key to a Deltas.
	// Each such Deltas has at least one Delta.
    pub items: BTreeMap<String, Deltas>,

    // `queue` maintains FIFO order of keys for consumption in Pop().
	// There are no duplicates in `queue`.
	// A key is in `queue` if and only if it is in `items`.
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

    // emitDeltaTypeReplaced is whether to emit the Replaced or Sync
	// DeltaType when Replace() is called (to preserve backwards compat).
	pub emitDeltaTypeReplaced: bool,

    pub notify: Arc<Notify>,
}


#[derive(Debug, Default, Clone)]
pub struct DeltaFifo(Arc<RwLock<DeltaFifoInner>>);

impl Deref for DeltaFifo {
    type Target = Arc<RwLock<DeltaFifoInner>>;

    fn deref(&self) -> &Arc<RwLock<DeltaFifoInner>> {
        &self.0
    }
}

impl DeltaFifoInner {
    pub fn queueActionLocked(&mut self, actionType: DeltaType, obj: &DataObject) -> Result<()> {
        let id = obj.Key();
        let exist;
        let mut deltas = match self.items.remove(&id) {
            None => {
                exist = false;
                Deltas(VecDeque::new()) 
            }
            Some(q) => {
                exist = true;
                q
            }
        };

        let delta = Delta {
            type_: actionType,
            obj: obj.clone()
        };
        match deltas.0.back() {
            None => deltas.0.push_back(delta),
            Some(d) => {
                if !d.IsDup(&delta) {
                    deltas.0.push_back(delta);
                }
            }
        }

        if deltas.0.len() > 0 {
            if !exist {
                self.queue.push_back(id.clone());
            }

            self.items.insert(id, deltas);
            self.notify.notify_waiters();
        } else {
            self.items.insert(id, deltas);
            return Err(Error::CommonError(format!("Impossible dedupDeltas")))
        }

        return Ok(())
    }

    pub fn ListLocked(&self) -> Vec<DataObject> {
        let mut list = Vec::new();

        for (_, item) in &self.items {
            let obj = match item.Newest() {
                None => continue,
                Some(i) => i.obj,
            };
            list.push(obj);
        }

        return list;
    }

    fn HasSyncedLocked(&self) -> bool {
        return self.populated && self.initialPopulationCount == 0;
    }

    fn AddIfNotPresent(&mut self, id: &str, ds: Deltas) {
        self.populated = true;
        if self.items.contains_key(id) {
            return
        }

        self.queue.push_back(id.to_string());
        self.items.insert(id.to_string(), ds);
        self.notify.notify_waiters();
    }

    fn SyncKeyLocked(&mut self, key: &str) -> Result<()> {
        let obj = match self.store.Get(key) {
            None => {
                error!("Key {} does not exist in known objects store, unable to queue object for sync", key);
                return Ok(()) 
            }
            Some(obj) => obj
        };

        // If we are doing Resync() and there is already an event queued for that object,
        // we ignore the Resync for it. This is to avoid the race, in which the resync
        // comes with the previous value of object (since queueing an event for the object
        // doesn't trigger changing the underlying store <knownObjects>.
        let id = obj.Key();

        if self.items[&id].0.len() > 0 {
            return Ok(())
        }

        return self.queueActionLocked(DeltaType::Sync, &obj)
    }
}

pub type PopProcess = fn(_obj: &Deltas, _isInInitialList: bool) -> Result<()>;

impl DeltaFifo {
    pub fn IsClosed(&self) -> bool {
        return self.read().closed;
    }
}

impl DeltaFifo {
    // Add adds the given object to the accumulator associated with the given object's key
    pub fn Add(&self, obj: &DataObject) -> Result<()> {
        let mut inner = self.write();
        inner.populated = true;
        inner.queueActionLocked(DeltaType::Added, obj)?;
        return Ok(())
    }

    // Update updates the given object in the accumulator associated with the given object's key
    pub fn Update(&self, obj: &DataObject) -> Result<()> {
        return self.Add(obj);
    }

    // Delete deletes the given object from the accumulator associated with the given object's key
    pub fn Delete(&self, obj: &DataObject) ->  Result<()> {
        let id = obj.Key();
        let mut inner = self.write();
        inner.populated = true;

        let exist = inner.store.Get(&id).is_some();
        let itemsExist = inner.items.get(&id).is_some();
        if !exist && !itemsExist {
            // Presumably, this was deleted when a relist happened.
			// Don't provide a second report of the same deletion.
            return Ok(())
        }

        return inner.queueActionLocked(DeltaType::Deleted, obj);
    }

    // List returns a list of all the currently non-empty accumulators
    pub fn List(&self) -> Vec<DataObject> {
        return self.read().ListLocked();
    }

    // ListKeys returns a list of all the keys currently associated with non-empty accumulators
    pub fn ListKeys(&self) -> Vec<String>{
        let inner = self.read();
        return inner.items.keys().cloned().collect();
    }

    // Get returns the accumulator associated with the given object's key
    pub fn Get(&self, key: &str) -> Option<Deltas> {
        let inner = self.read();
        match inner.items.get(key) {
            None => return None,
            Some(ds) => return Some(ds.DeepCopy()),
        }
    }

    // Replace atomically does two things: (1) it adds the given objects
    // using the Sync or Replace DeltaType and then (2) it does some deletions.
    // In particular: for every pre-existing key K that is not the key of
    // an object in `list` there is the effect of
    // `Delete(DeletedFinalStateUnknown{K, O})` where O is current object
    // of K.  If `f.knownObjects == nil` then the pre-existing keys are
    // those in `f.items` and the current object of K is the `.Newest()`
    // of the Deltas associated with K.  Otherwise the pre-existing keys
    // are those listed by `f.knownObjects` and the current object of K is
    // what `f.knownObjects.GetByKey(K)` returns.
    pub fn Replace(&self, list: &[DataObject]) -> Result<()> {
        let mut inner = self.write();

        let mut keys = BTreeSet::new();

        let action = if inner.emitDeltaTypeReplaced {
            DeltaType::Replaced
        } else {
            DeltaType::Sync
        };

        for item in list {
            let key = item.Key();
            keys.insert(key);
            match inner.queueActionLocked(action, item) {
                Ok(()) => (),
                Err(e) => {
                    return Err(Error::CommonError(format!("couldn't enqueue object: {:?}", e)));
                }
            }
        }

        /*if inner.store.is_none() {
            // Do deletion detection against our own list.
            let mut queuedDeletions = 0;
            let mut deleteItems = Vec::new();
            for (k, oldItem)  in &inner.items {
                if keys.contains(k) {
                    continue;
                }

                // Delete pre-existing items not in the new list.
                // This could happen if watch deletion event was missed while
                // disconnected from apiserver.
                let deletedObj = match oldItem.Newest() {
                    None => DataObject::default(),
                    Some(d) => d.obj,
                };
                queuedDeletions += 1;
                deleteItems.push(deletedObj);
            }

            for o in deleteItems {
                inner.queueActionLocked(DeltaType::Deleted, &o)?;
            }

            if !inner.populated {
                inner.populated == true;
                // While there shouldn't be any queued deletions in the initial
			    // population of the queue, it's better to be on the safe side.
			
                inner.initialPopulationCount = keys.len() + queuedDeletions;
            }
        }*/

        let store = inner.store.clone();
        let knowKeys = store.ListKeys();
        let mut queuedDeletions = 0;

        for k in knowKeys {
            if keys.contains(&k) {
                continue;
            }

            let deletedObj = match store.Get(&k) {
                None => DataObject::default(),
                Some(o) => o.clone(),
            };

            queuedDeletions += 1;

            inner.queueActionLocked(DeltaType::Deleted, &deletedObj)?;
        }
        
        if !inner.populated {
            inner.populated = true;
            inner.initialPopulationCount = keys.len() + queuedDeletions;
        }

        return Ok(())
    }

    // Resync is meaningless in the terms appearing here but has
    // meaning in some implementations that have non-trivial
    // additional behavior (e.g., DeltaFIFO).
    pub fn Resync(&self) -> Result<()> {
        let mut inner = self.write();

        let keys = inner.store.ListKeys();
        for k in &keys {
            inner.SyncKeyLocked(k)?;
        }

        return Ok(())
    }

    pub async fn Pop(&self, process: PopProcess) -> Result<Deltas> {
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
            let depth = inner.queue.len();
            if inner.initialPopulationCount > 0 {
                inner.initialPopulationCount += 1;
            }

            let item = match inner.items.remove(&id) {
                None => continue,
                Some(i) => i,
            };

            if depth > 0 {
                //todo: handle trace
                //error!("DeltaFIFO Pop Process take too long");
            }

            match process(&item, isInInitialList) {
                Err(Error::ErrRequeue) => {
                    inner.AddIfNotPresent(&id, item);
                    return Err(Error::ErrRequeue)
                }
                Err(e)  => {
                    return Err(e)
                }
                Ok(()) => {
                    return Ok(item)
                }
            }
        }
    }
}