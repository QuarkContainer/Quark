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

use alloc::string::String;
use alloc::string::ToString;
use alloc::collections::btree_map::BTreeMap;
use alloc::collections::btree_map::Range;
use core::ops::Bound::*;
use alloc::vec::Vec;
use ::qlib::mutex::*;
use alloc::sync::Arc;

use super::super::qlib::device::*;
use super::super::qlib::common::*;
use super::super::task::*;
use super::attr::*;

#[derive(Debug, Copy, Clone)]
pub struct DentAttr {
    pub Type: InodeType,
    pub InodeId: u64,
}

impl DentAttr {
    pub fn GenericDentAttr(nt: InodeType, device: &Arc<QMutex<Device>>) -> Self {
        return Self {
            Type: nt,
            InodeId: device.lock().NextIno()
        }
    }
}

pub trait DentrySerializer {
    fn CopyOut(&mut self, task: &Task, name: &str, attr: &DentAttr) -> Result<()>;
    fn Written(&self) -> usize;
}

pub struct CollectEntriesSerilizer {
    pub Entries: BTreeMap<String, DentAttr>,
    //pub Order: Vec<String>,
}

impl CollectEntriesSerilizer {
    pub fn New() -> Self {
        return Self {
            Entries: BTreeMap::new(),
        }
    }

    pub fn Order(&self) -> Vec<String> {
        let mut res = Vec::new();

        for (s, _) in &self.Entries {
            res.push(s.clone())
        }

        return res;
    }
}

impl DentrySerializer for CollectEntriesSerilizer {
    fn CopyOut(&mut self, _task: &Task, name: &str, attr: &DentAttr) -> Result<()> {
        self.Entries.insert(name.to_string(), attr.clone());
        return Ok(())
    }

    fn Written(&self) -> usize {
        return self.Entries.len();
    }
}

pub struct DirCtx<'a> {
    pub Serializer: &'a mut DentrySerializer,
    pub DirCursor: String
}

impl<'a> DirCtx<'a> {
    pub fn New(Serializer: &'a mut DentrySerializer, cursor: &str) -> Self {
        return Self {
            Serializer: Serializer,
            DirCursor: cursor.to_string(),
        }
    }

    pub fn DirEmit(&mut self, task: &Task, name: &str, attr: &DentAttr) -> Result<()> {
        self.Serializer.CopyOut(task, name, attr)
    }

    pub fn ReadDir(&mut self, task: &Task, map: &DentMap) -> Result<usize> {
        let range = if self.DirCursor == "".to_string() {
            map.GetAll()
        } else {
            let str = self.DirCursor.clone();
            map.GetNext(str)
        };

        let mut count = 0;

        for (name, attr) in range {
            if *name == "" || *name == "." || *name == ".." {
                continue
            }

            match self.DirEmit(task, name, attr) {
                Err(error) => {
                    if count > 0 {
                        return Ok(count)
                    }
                    return Err(error);
                }
                _ => ()
            }
            count += 1;

            self.DirCursor = name.clone();
        }

        return Ok(count)
    }
}

pub struct DentMap {
    pub Entries: BTreeMap<String, DentAttr>,
}

impl Default for DentMap {
    fn default() -> Self {
        return Self {
            Entries: BTreeMap::new(),
        }
    }
}

impl DentMap {
    pub fn New(entries: BTreeMap<String, DentAttr>) -> Self {
        return Self {
            Entries: entries,
        }
    }

    pub fn Add(&mut self, name: &str, attr: &DentAttr) {
        self.Entries.insert(name.to_string(), *attr);
    }

    pub fn Remove(&mut self, name: &str) {
        self.Entries.remove(name);
    }

    pub fn GetNext(&self, name: String) -> Range<String, DentAttr> {
        return self.Entries.range((Excluded(name), Unbounded))
    }

    pub fn GetAll(&self) -> Range<String, DentAttr> {
        return self.Entries.range((Included("".to_string()), Unbounded))
    }

    pub fn Containers(&self, name: &str) -> bool {
        match self.Entries.get(name) {
            None => false,
            _ => true,
        }
    }
}