// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
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

use alloc::collections::btree_map::BTreeMap;
use alloc::sync::Arc;
use spin::Mutex;

use super::common::*;

struct LinkEntry<T: Clone> {
    pub key: u64,
    pub val: Option<T>,
    pub prev: Option<Arc<Mutex<LinkEntry<T>>>>,
    pub next: Option<Arc<Mutex<LinkEntry<T>>>>,
}

impl<T: Clone> Default for LinkEntry<T> {
    fn default() -> Self {
        return Self {
            key: 0,
            val: None,
            prev: None,
            next: None,
        }
    }
}

impl<T: Clone> LinkEntry<T> {
    pub fn New(key: u64, val: T) -> Self {
        return Self {
            key: key,
            val: Some(val),
            ..Default::default()
        }
    }

    pub fn Remove(&mut self) -> Result<()> {
        let prev = match self.prev.take() {
            Some(v) => v,
            None => return Err(Error::Common(format!("prev is null, key is {}", self.key)))
        };

        let next = match self.next.take() {
            Some(v) => v,
            None => return Err(Error::Common(format!("prev is null, key is {}", self.key)))
        };

        (*prev).lock().next = Some(next.clone());
        (*next).lock().prev = Some(prev.clone());

        return Ok(())
    }

    pub fn GetKey(&self) -> u64 {
        return self.key
    }
}

struct LinkedList<T: Clone> {
    pub head: Arc<Mutex<LinkEntry<T>>>,
    pub tail: Arc<Mutex<LinkEntry<T>>>,
    pub count: u64,
}

impl<T: Clone> Default for LinkedList<T> {
    fn default() -> Self {
        let head = Arc::new(Mutex::new(LinkEntry::default()));
        let tail = Arc::new(Mutex::new(LinkEntry::default()));
        (*head).lock().next = Some(tail.clone());
        (*tail).lock().prev = Some(head.clone());

        return Self {
            head,
            tail,
            count: 0,
        }
    }
}

impl<T: Clone> LinkedList<T> {
    pub fn PushFront(&mut self, entry: &Arc<Mutex<LinkEntry<T>>>) {
        let next = self.head.lock().next.take().unwrap();

        (*next).lock().prev = Some(entry.clone());
        (*entry).lock().next = Some(next.clone());

        (*entry).lock().prev = Some(self.head.clone());
        self.head.lock().next = Some(entry.clone());

        self.count += 1;
    }

    pub fn PopFront(&mut self) -> Option<Arc<Mutex<LinkEntry<T>>>> {
        if self.count == 0 {
            return None;
        }

        self.count -= 1;

        let ret = self.head.lock().next.as_ref().unwrap().clone();
        match (*ret).lock().Remove() {
            Err(e) => panic!("PopBack fail, {:?}", e),
            Ok(_) => (),
        }

        return Some(ret);
    }

    pub fn PushBack(&mut self, entry: &Arc<Mutex<LinkEntry<T>>>) {
        let prev = self.tail.lock().prev.take().unwrap();

        (*prev).lock().next = Some(entry.clone());

        (*entry).lock().prev = Some(prev.clone());
        (*entry).lock().next = Some(self.tail.clone());
        self.tail.lock().prev = Some(entry.clone());

        self.count += 1;
    }

    pub fn PopBack(&mut self) -> Option<Arc<Mutex<LinkEntry<T>>>> {
        if self.count == 0 {
            return None;
        }

        let ret = self.tail.lock().prev.as_ref().unwrap().clone();
        match (*ret).lock().Remove() {
            Err(e) => panic!("PopBack fail count is {}, {:?}", self.count, e),
            Ok(_) => (),
        }

        self.count -= 1;

        return Some(ret);
    }
}

pub struct LruCache<T: Clone> {
    maxSize: u64,
    currentSize: u64,
    list: LinkedList<T>,
    map: BTreeMap<u64, Arc<Mutex<LinkEntry<T>>>>,
}

impl<T: Clone> LruCache<T> {
    pub fn New(maxSize: u64) -> Self {
        assert!(maxSize > 0, "LruCache maxsize must be larger than 0");
        return Self {
            maxSize: maxSize,
            currentSize: 0,
            list: LinkedList::default(),
            map: BTreeMap::new(),
        }
    }

    pub fn ContainsKey(&self, key: u64) -> bool {
        return self.map.contains_key(&key);
    }

    pub fn Add(&mut self, key: u64, d: T) {
        let exist = if !self.map.contains_key(&key) {
            if self.currentSize >= self.maxSize {
                //remove the last one
                //error!("LruCache pop self.currentSize is {} self.maxSize is {}",
                //    self.currentSize, self.maxSize);
                let remove = self.list.PopBack();
                let remove = match remove {
                    None => panic!("get zero size"),
                    Some(r) => r,
                };
                let removeKey = (*remove).lock().GetKey();
                self.map.remove(&removeKey);
                self.currentSize -= 1;
            }

            let entry = Arc::new(Mutex::new(LinkEntry::New(key, d)));
            self.map.insert(key, entry);
            false
        } else {
            true
        };

        let entry = match self.map.get_mut(&key) {
            Some(e) => {
                e
            }
            None => panic!("impossible"),
        };

        if exist {
            match entry.lock().Remove() {
                Err(e) => panic!("Add fail, {:?}", e),
                Ok(_) => (),
            }
            self.list.PushFront(entry);
            return;
        }

        self.list.PushFront(entry);
        self.currentSize += 1;
    }

    //ret: true- exit the item, false-not exist
    pub fn Remove(&mut self, key: u64) -> bool {
        match self.map.remove(&key) {
            Some(e) => {
                match e.lock().Remove()  {
                    Err(e) => panic!("Remove fail, {:?}", e),
                    Ok(_) => (),
                }
                return true
            }
            None => ()
        };
        return false
    }

    pub fn Get(&self, key: u64) -> Option<T> {
        match self.map.get(&key) {
            None => None,
            Some(e) => Some(e.lock().val.as_ref().unwrap().clone()),
        }
    }

    pub fn Clear(&mut self) {
        loop {
            match self.list.PopFront() {
                None => break,
                _ => ()
            }
        }

        self.map.clear();
    }

    pub fn Size(&self) -> u64 {
        return self.currentSize
    }

    pub fn MaxSize(&self) -> u64 {
        return self.maxSize;
    }

    pub fn SetMaxSize(&mut self, max: u64) {
        //todo: shrink the cache
        self.maxSize = max;
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_cache() {
        let mut cache = LruCache::New(2);
        cache.Add(1, 1);

        assert!(cache.Get(1) == Some(1));
        cache.Add(2, 2);

        assert!(cache.Get(1) == Some(1));
        assert!(cache.Get(2) == Some(2));
        cache.Add(3, 3);
        assert!(cache.Get(1) == None);
        assert!(cache.Get(2) == Some(2));
        assert!(cache.Get(3) == Some(3));

        cache.Add(2, 2);
        assert!(cache.Get(1) == None);
        assert!(cache.Get(2) == Some(2));
        assert!(cache.Get(3) == Some(3));

        cache.Add(4, 4);
        assert!(cache.Get(2) == Some(2));
        assert!(cache.Get(3) == None);
        assert!(cache.Get(4) == Some(4));
    }
}