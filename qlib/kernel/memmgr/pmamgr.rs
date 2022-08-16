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
use alloc::vec::Vec;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use spin::Mutex;
//use hashbrown::HashMap;
//use core::hash::BuildHasherDefault;
//use cache_padded::CachePadded;

use super::super::super::common::*;
use super::super::super::linux_def::*;
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

/*#[derive(Default)]
pub struct RawHasher {
    state: u64,
}

impl core::hash::Hasher for RawHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state = self.state.rotate_left(8) ^ u64::from(byte);
        }
    }

    fn write_u64(&mut self, i: u64) {
        self.state = i;
    }

    fn write_u32(&mut self, i: u32) {
        self.state = i as u64;
    }

    fn finish(&self) -> u64 {
        self.state
    }
}*/

pub const REF_MAP_PARTITION_CNT : usize = 32;
pub struct PagePool {
    //refCount for whole pma
    pub refCount: AtomicU64,
    //pub refs: [Mutex<CachePadded<HashMap<u32, u32, BuildHasherDefault<RawHasher>>>>; REF_MAP_PARTITION_CNT],
    //pub refs: Vec<CachePadded<Mutex<BTreeMap<u32, u32>>>>,
    pub refs: Vec<Mutex<BTreeMap<u32, u32>>>,
    //pub refs: Vec<CachePadded<Mutex<HashMap<u32, u32>>>>,
    //pub refs: [CachePadded<Mutex<BTreeMap<u32, u32>>>; REF_MAP_PARTITION_CNT],
    //pub refs: [Mutex<HashMap<u32, u32>>; REF_MAP_PARTITION_CNT],
    pub allocator: AlignedAllocator,
}

impl PagePool {
    pub fn New() -> Self {
        let mut refs = Vec::with_capacity(REF_MAP_PARTITION_CNT);
        for _i in 0..REF_MAP_PARTITION_CNT {
            refs.push(Default::default());
        }
        return Self {
            refs: refs,
            //the PagePool won't be free. fake a always nonzero refcount
            refCount: AtomicU64::new(1),
            allocator: AlignedAllocator::New(
                MemoryDef::PAGE_SIZE as usize,
                MemoryDef::PAGE_SIZE as usize,
            ),
        };
    }

    pub fn PrintRefs(&self) {
        //error!("PagePool left is {:#x?}", self.refs);
    }

    pub fn PartitionId(addr: u64) -> usize {
        return (addr as usize >> 12) % REF_MAP_PARTITION_CNT;
    }

    pub fn PageId(addr: u64) -> u32 {
        return (addr as usize >> 12) as u32;
    }

    pub fn Ref(&self, addr: u64) -> Result<u64> {
        assert!(addr & (MemoryDef::PAGE_SIZE - 1) == 0);
        let pageId = Self::PageId(addr);
        let idx = pageId as usize % REF_MAP_PARTITION_CNT;
        let mut refs = self.refs[idx].lock();
        let refcount = match refs.get_mut(&pageId) {
            None => {
                // the address is not allocated from PagePool
                return Ok(1);
            }
            Some(v) => {
                *v += 1;
                *v
            }
        };

        self.refCount.fetch_add(1, Ordering::Release);
        return Ok(refcount as u64);
    }

    pub fn Deref(&self, addr: u64) -> Result<u64> {
        assert!(addr & (MemoryDef::PAGE_SIZE - 1) == 0);
        let refcount = {
            let pageId = Self::PageId(addr);
            let idx = pageId as usize % REF_MAP_PARTITION_CNT;
            let mut refs = self.refs[idx].lock();
            let refcount = match refs.get_mut(&pageId) {
                None => {
                    // the address is not allocated from PagePool
                    return Ok(1);
                }
                Some(v) => {
                    assert!(*v >= 1, "deref fail: addresss is {:x}", addr);
                    *v -= 1;
                    *v
                }
            };

            if refcount == 0 {
                refs.remove(&pageId);
            }

            refcount
        };

        self.refCount.fetch_sub(1, Ordering::Release);
        return Ok(refcount as u64);
    }

    pub fn GetRef(&self, addr: u64) -> Result<u64> {
        let pageId = Self::PageId(addr);
        let idx = pageId as usize % REF_MAP_PARTITION_CNT;
        let refs = self.refs[idx].lock();
        let refcount = match refs.get(&pageId) {
            None => {
                // the address is not allocated from PagePool
                return Ok(0);
            }
            Some(v) => *v,
        };

        return Ok(refcount as u64);
    }

    pub fn AllocPage(&self, incrRef: bool) -> Result<u64> {
        let addr = self.Allocate()?;
        let pageId = Self::PageId(addr);
        let idx = pageId as usize % REF_MAP_PARTITION_CNT;
        let mut  refs = self.refs[idx].lock();
        if incrRef {
            refs.insert(pageId, 1);
            self.refCount.fetch_add(1, Ordering::Release);
        } else {
            refs.insert(pageId, 0);
        }

        return Ok(addr);
    }

    pub fn Allocate(&self) -> Result<u64> {
        match CPULocal::Myself().pageAllocator.lock().AllocPage() {
            Some(page) => {
                ZeroPage(page);
                return Ok(page);
            }
            None => (),
        }

        let addr = self.allocator.Allocate()?;
        ZeroPage(addr as u64);
        //error!("AllocPage {:x}", addr);

        return Ok(addr as u64);
    }

    pub fn FreePage(&self, addr: u64) -> Result<()> {
        //CPULocal::Myself().pageAllocator.lock().FreePage(addr);
        //return Ok(());
        return self.allocator.Free(addr);
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
