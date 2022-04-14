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

use super::super::super::linux_def::*;
use super::super::super::common::*;
use super::super::super::pagetable::*;
use super::super::super::vcpu_mgr::CPULocal;

pub fn ZeroPage(pageStart: u64) {
    use alloc::slice;
    unsafe {
        let arr = slice::from_raw_parts_mut(pageStart as *mut u64, 512);
        for i in 0..512 {
            arr[i] = 0
        }
    }

    super::super::asm::sfence();
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
    //refCount for whole pma
    pub refCount: u64,
    pub refs: BTreeMap<u64, u32>,
    pub allocator: AlignedAllocator,
}

impl PagePool {
    pub fn PrintRefs(&self) {
        //error!("PagePool left is {:#x?}", self.refs);
    }

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

    pub fn Deref(&mut self, addr: u64) -> Result<u64> {
        assert!(addr & (MemoryDef::PAGE_SIZE-1) == 0);
        let refcount = match self.refs.get_mut(&addr) {
            None => { // the address is not allocated from PagePool
                return Ok(1)
            }
            Some(v) => {
                assert!(*v >= 1, "deref fail: addresss is {:x}", addr);
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
            self.refs.insert(addr, 1);
            self.refCount += 1;
        } else {
            self.refs.insert(addr, 0);
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
            //the PagePool won't be free. fake a always nonzero refcount
            refCount: 1,
            allocator: AlignedAllocator::New(MemoryDef::PAGE_SIZE as usize, MemoryDef::PAGE_SIZE as usize),
        };
    }

    pub fn Allocate(&mut self) -> Result<u64> {
        match CPULocal::Myself().pageAllocator.lock().AllocPage() {
            Some(page) => {
                ZeroPage(page);
                return Ok(page)
            },
            None => (),
        }

        let addr = self.allocator.Allocate()?;
        ZeroPage(addr as u64);
        //error!("AllocPage {:x}", addr);

        return Ok(addr as u64)
    }

    pub fn Free(&mut self, addr: u64) -> Result<()> {
        CPULocal::Myself().pageAllocator.lock().FreePage(addr);
        return Ok(());
        //return self.allocator.Free(addr);
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
}

