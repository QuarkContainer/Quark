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
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use spin::Mutex;

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

pub const REF_MAP_PARTITION_CNT : usize = 16;
pub struct PagePool {
    //refCount for whole pma
    pub refCount: AtomicU64,
    pub refs: [Mutex<BTreeMap<u64, u32>>; 16],
    pub allocator: AlignedAllocator,
}

impl PagePool {
    pub fn New() -> Self {
        return Self {
            refs: Default::default(),
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

    pub fn Ref(&self, addr: u64) -> Result<u64> {
        assert!(addr & (MemoryDef::PAGE_SIZE - 1) == 0);
        let idx = Self::PartitionId(addr);
        let mut refs = self.refs[idx].lock();
        let refcount = match refs.get_mut(&addr) {
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
            let idx = Self::PartitionId(addr);
            let mut refs = self.refs[idx].lock();
            let refcount = match refs.get_mut(&addr) {
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
                refs.remove(&addr);
            }

            refcount
        };

        if refcount == 0 {
            self.Free(addr)?;
        }
        self.refCount.fetch_sub(1, Ordering::Release);
        return Ok(refcount as u64);
    }

    pub fn GetRef(&self, addr: u64) -> Result<u64> {
        let idx = Self::PartitionId(addr);
        let refs = self.refs[idx].lock();
        let refcount = match refs.get(&addr) {
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
        let idx = Self::PartitionId(addr);
        let mut  refs = self.refs[idx].lock();
        if incrRef {
            refs.insert(addr, 1);
            self.refCount.fetch_add(1, Ordering::Release);
        } else {
            refs.insert(addr, 0);
        }

        return Ok(addr);
    }

    pub fn FreePage(&self, addr: u64) -> Result<()> {
        return self.Free(addr);
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

    pub fn Free(&self, addr: u64) -> Result<()> {
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
