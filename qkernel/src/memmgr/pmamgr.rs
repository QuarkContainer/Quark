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

use alloc::vec::Vec;

use super::super::qlib::addr::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::common::*;
use super::super::qlib::range::*;
//use super::PageTable::PageTables;


fn ZeroPage(pageStart: u64) {
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
    //global pageallocator base addr, use for pageallocator page free which allocated before pagepool
    pub pageAllocatorRange: Range,

    // pagepool basea addr
    pub baseAddr: u64,
    pub next: u32,
    pub pageCount: u32,

    pub refCount: u64,
    //refCount for whole pma

    pub freePool: Vec<u32>,
    pub refArr: Vec<u16>,

    //the zeroed paged which will be readyonly, e.g. page for Virtual Address 0
    pub zeroPage: u64,
}

impl PagePool {
    pub fn Ref(&mut self, addr: u64) -> Result<u64> {
        //assert!(self.pageAllocatorRange.Contains(addr),
        //    &format!("RefMgrInternal::ref fail addr is {:x}, pageAllocatorRange is {:x?}", addr, &self.pageAllocatorRange));
        if !self.Range().Contains(addr) {
            return Ok(1)
        }

        let idx = ((addr - self.baseAddr) / MemoryDef::PAGE_SIZE) as usize;
        self.refArr[idx] += 1;
        self.refCount += 1;
        return Ok(self.refArr[idx] as u64)
    }

    //todo: add ability to punch hole to save memory
    pub fn Deref(&mut self, addr: u64) -> Result<u64> {
        //assert!(self.pageAllocatorRange.Contains(addr));

        if !self.Range().Contains(addr) {
            return Ok(1)
        }

        //it is in pageAllocator's range but not in pagepool range,
        //it is allocated at host, doesn't need free
        if !self.Range().Contains(addr) {
            //PAGE_ALLOCATOR.FreePage(addr)?;
            return Ok(1)
        }

        let idx = ((addr - self.baseAddr) / MemoryDef::PAGE_SIZE) as usize;

        assert!(self.refArr[idx] != 0);

        self.refArr[idx] -= 1;

        if self.refArr[idx] == 0 {
            self.Free(addr).unwrap();
        }

        self.refCount -= 1;
        return Ok(self.refCount)
    }

    pub fn GetRef(&self, addr: u64) -> Result<u64> {
        if !self.Range().Contains(addr) {
            return Ok(0)
        }

        let idx = ((addr - self.baseAddr) / MemoryDef::PAGE_SIZE) as usize;

        //assert!(self.refArr[idx] != 0);
        return Ok(self.refArr[idx] as u64)
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
            pageAllocatorRange: Range::default(),
            baseAddr: 0,
            next: 0,
            pageCount: 0,
            freePool: Vec::new(),
            refArr: Vec::new(),

            zeroPage: 0,
            //the PagePool won't be free. fake a always nonzero refcount
            refCount: 1,
        };
    }

    pub fn Range(&self) -> Range {
        return Range::New(self.baseAddr, (self.pageCount as u64) << 12);
    }

    pub fn Init(&mut self, pageAllocatorRange: &Range, baseAddr: u64, pageCount: u32) {
        self.pageAllocatorRange = *pageAllocatorRange;
        self.baseAddr = baseAddr;
        self.next = 0;
        self.pageCount = pageCount;
        self.refArr = vec![0; pageCount as usize];
        self.zeroPage = self.Allocate().unwrap();
        self.Ref(self.zeroPage).unwrap();
    }

    pub fn GetZeroPage(&mut self) -> u64 {
        let zeroPage = self.zeroPage;
        self.Ref(zeroPage).unwrap();
        return zeroPage;
    }

    pub fn Allocate(&mut self) -> Result<u64> {
        if self.freePool.len() > 0 {
            let idx = self.freePool[self.freePool.len() - 1] as usize;
            self.freePool.pop();
            //self.refArr[idx] += 1;
            return Ok(Addr(self.baseAddr).AddLen(idx as u64 * MemoryDef::PAGE_SIZE_4K)?.0);
        }

        if self.next == self.pageCount {
            info!("PagePool ... Out of memory");
            return Err(Error::NoEnoughMemory)
        }

        let idx = self.next as usize;
        //self.refArr[idx] += 1;
        self.next += 1;
        return Ok(Addr(self.baseAddr).AddLen(idx as u64 * MemoryDef::PAGE_SIZE_4K)?.0);
    }

    pub fn Free(&mut self, addr: u64) -> Result<()> {
        //todo:: check ???
        Addr(addr).PageAligned()?;
        ZeroPage(addr);

        let idx = Addr(self.baseAddr).PageOffsetIdx(Addr(addr))?;

        if idx >= self.pageCount {
            return Err(Error::AddressNotInRange);
        }

        self.freePool.push(idx);
        return Ok(());
    }

    pub fn GetPageIdx(&self, addr: u64) -> Result<u32> {
        Addr(self.baseAddr).PageOffsetIdx(Addr(addr))
    }

    pub fn GetPageAddr(&self, idx: u32) -> Result<u64> {
        if idx >= self.pageCount {
            return Err(Error::AddressNotInRange);
        }

        return Ok(Addr(self.baseAddr).AddPages(idx).0);
    }
}
