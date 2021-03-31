// Copyright (c) 2021 QuarkSoft LLC
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

use super::super::qlib::linux_def::*;
use super::super::qlib::common::*;
use super::super::qlib::pagetable::*;

pub fn ZeroPage(pageStart: u64) {
    use alloc::slice;
    unsafe {
        let arr = slice::from_raw_parts_mut(pageStart as *mut u64, 512);
        for i in 0..512 {
            arr[i] = 0
        }
    }
}

pub fn CheckZeroPage(pageStart: u64) {
    use alloc::slice;
    unsafe {
        let arr = slice::from_raw_parts_mut(pageStart as *mut u64, 512);
        for i in 0..512 {
            if arr[i] != 0 {
                panic!("alloc non zero page {:x}", pageStart);
            }
        }
    }
}

pub struct PagePool {
    pub refCount: u64,
    //refCount for whole pma

    pub refs: BTreeMap<u64, u32>,

    //the zeroed paged which will be readyonly, e.g. page for Virtual Address 0
    pub zeroPage: u64,

    pub allocator: AlignedAllocator,
}

impl PagePool {
    pub fn Ref(&mut self, addr: u64) -> Result<u64> {
        assert!(addr & (MemoryDef::PAGE_SIZE-1) == 0);
        let refcount = match self.refs.get_mut(&addr) {
            None => { // the address is not allocated from PagePool
                return Ok(1)
            }
            Some(v) => {
                *v += 1;
                *v
            }
        };

        self.refCount += 1;
        return Ok(refcount as u64)
    }

    //todo: add ability to punch hole to save memory
    pub fn Deref(&mut self, addr: u64) -> Result<u64> {
        assert!(addr & (MemoryDef::PAGE_SIZE-1) == 0);
        let refcount = match self.refs.get_mut(&addr) {
            None => { // the address is not allocated from PagePool
                return Ok(1)
            }
            Some(v) => {
                *v -= 1;
                *v
            }
        };

        self.refCount -= 1;
        if refcount == 0 {
            self.refs.remove(&addr);
            self.Free(addr)?;
        }
        return Ok(refcount as u64)
    }

    pub fn GetRef(&self, addr: u64) -> Result<u64> {
        let refcount = match self.refs.get(&addr) {
            None => { // the address is not allocated from PagePool
                return Ok(0)
            }
            Some(v) => {
                *v
            }
        };

        return Ok(refcount as u64)
    }

    pub fn AllocPage(&mut self, incrRef: bool) -> Result<u64> {
        let addr = self.Allocate()?;
        if incrRef {
            self.Ref(addr)?;
        }
        return Ok(addr)
    }

    pub fn FreePage(&mut self, addr: u64) -> Result<()> {
        return self.Free(addr)
    }

    //unitSize: how many pages for each unit
    pub fn New() -> Self {
        return Self {
            refs: BTreeMap::new(),
            zeroPage: 0,
            //the PagePool won't be free. fake a always nonzero refcount
            refCount: 1,
            allocator: AlignedAllocator::New(MemoryDef::PAGE_SIZE as usize, MemoryDef::PAGE_SIZE as usize),
        };
    }

    pub fn Init(&mut self) {
        self.zeroPage = self.Allocate().unwrap();
        self.Ref(self.zeroPage).unwrap();
    }

    pub fn GetZeroPage(&mut self) -> u64 {
        let zeroPage = self.zeroPage;
        self.Ref(zeroPage).unwrap();
        return zeroPage;
    }

    pub fn Allocate(&mut self) -> Result<u64> {
        let addr = self.allocator.Allocate()?;
        ZeroPage(addr as u64);
        return Ok(addr as u64)
    }

    pub fn Free(&mut self, addr: u64) -> Result<()> {
        return self.allocator.Free(addr);
    }
}

