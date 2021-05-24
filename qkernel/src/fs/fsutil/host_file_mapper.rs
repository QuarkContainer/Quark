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
use spin::Mutex;

use super::super::super::Kernel::HostSpace;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::range::*;

pub struct Mapping {
    pub addr: u64,
    pub writeable: bool,
}

impl Mapping {
    pub fn Unmap(&self) {
        info!("Mapping Dropping start is {:x}", self.addr);
        HostSpace::UnMapPma(self.addr);
    }
}

const CHUNK_SHIFT: u64 = MemoryDef::HUGE_PAGE_SHIFT;
const CHUNK_SIZE: u64 = 1 << CHUNK_SHIFT;
const CHUNK_MASK: u64 = CHUNK_SIZE - 1;

fn PagesInChunk(mr: &Range, chunkStart: u64) -> i32 {
    return (mr.Intersect(&Range::New(chunkStart, CHUNK_SIZE)).Len() / MemoryDef::PAGE_SIZE) as i32
}

pub struct HostFileMapper {
    pub refs: Mutex<BTreeMap<u64, i32>>,
    pub mappings: Mutex<BTreeMap<u64, Mapping>>,
}

impl HostFileMapper {
    pub fn New() -> Self {
        return Self {
            refs: Mutex::new(BTreeMap::new()),
            mappings: Mutex::new(BTreeMap::new()),
        }
    }
}

impl HostFileMapper {
    // IncRefOn increments the reference count on all offsets in mr.
    pub fn IncRefOn(&self, mr: &Range) {
        let mut refs = self.refs.lock();

        let mut chunkStart = mr.Start() & !CHUNK_MASK;
        while chunkStart < mr.End() {
            let refcnt = match refs.get(&chunkStart) {
                None => 0,
                Some(v) => *v,
            };

            let pgs = PagesInChunk(mr, chunkStart);
            if refcnt + pgs < refcnt {
                // Would overflow.
                panic!("HostFileMapper.IncRefOn({:?}): adding {} page references to chunk {:x}, which has {} page references",
                       &mr, pgs, chunkStart, refcnt)
            }

            refs.insert(chunkStart, refcnt + pgs);

            chunkStart += CHUNK_SIZE;
        }
    }

    // DecRefOn decrements the reference count on all offsets in mr.
    /*pub fn DecRefOn(&self, mr: &Range) {
        let refs = self.refs.lock();

        let mut chunkStart = mr.Start() & !CHUNK_MASK;

    }

    // MapInternal returns a mapping of offsets in fr from fd. The returned
    // safemem.BlockSeq is valid as long as at least one reference is held on all
    // offsets in fr or until the next call to UnmapAll.
    //
    // Preconditions: The caller must hold a reference on all offsets in fr.
    pub fn MapInternal(&self, mr: &Range, fd: i32, write: bool) -> Result<Vec<Range>> {
        return Err(Error::NoData)
    }*/
}