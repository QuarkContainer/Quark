// Copyright (c) 2021 Quark Container Authors
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

use alloc::collections::btree_set::BTreeSet;
use alloc::vec::Vec;
use core::sync::atomic::AtomicU16;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;

use super::super::common::*;
use super::super::linux_def::*;
use super::super::mutex::*;
use super::super::pagetable::*;
use crate::qlib::kernel::arch::tee::is_cc_active;
use crate::qlib::kernel::Kernel::HostSpace;
use crate::qlib::kernel::SHARESPACE;
//use super::list_allocator::*;
use crate::GLOBAL_ALLOCATOR;

#[derive(Debug, Default)]
pub struct PageBlockAllocIntern {
    pub pageBlockList: u64,
    pub pageBlocks: BTreeSet<u64>,
}

impl Drop for PageBlockAllocIntern {
    fn drop(&mut self) {
        let mut pbAddr = self.pageBlockList;
        while pbAddr != 0 {
            let pb = PageBlock::FromAddr(pbAddr);
            pbAddr = pb.allocator.lock().next;
            pb.Drop().unwrap();
        }
    }
}

impl PageBlockAllocIntern {
    pub fn PrintPages(&self) {
        for addr in &self.pageBlocks {
            let pb = PageBlock::FromAddr(*addr);
            pb.PrintPages();
        }
    }

    pub fn GetPages(&self) -> Result<Vec<u64>> {
        let mut output = Vec::new();
        for addr in &self.pageBlocks {
            let pb = PageBlock::FromAddr(*addr);
            pb.GetPages(&mut output)?;
        }

        return Ok(output);
    }

    pub fn UnlinkPageBlock(&mut self, pb: &mut PageBlock) {
        let mut pb = pb.allocator.lock();
        let prev = pb.prev;
        let next = pb.next;
        if prev == 0 {
            //head
            self.pageBlockList = next;
        } else {
            PageBlock::FromAddr(prev).allocator.lock().next = next;
        }

        if next != 0 {
            PageBlock::FromAddr(next).allocator.lock().prev = prev;
        }

        pb.next = 0;
        pb.prev = 0;
    }

    pub fn LinkPageBlock(&mut self, pb: &mut PageBlock) {
        let pbAddr = pb.ToAddr();
        let mut pb = pb.allocator.lock();

        let next = self.pageBlockList;
        if next > 0 {
            PageBlock::FromAddr(next).allocator.lock().prev = pbAddr;
        }

        pb.prev = 0;
        pb.next = self.pageBlockList;
        self.pageBlockList = pbAddr;
    }

    pub fn Insert(&mut self, pb: &mut PageBlock) {
        let pbAddr = pb.ToAddr();
        self.LinkPageBlock(pb);
        self.pageBlocks.insert(pbAddr);
    }

    pub fn Remove(&mut self, pb: &mut PageBlock) {
        let pbAddr = pb.ToAddr();
        self.UnlinkPageBlock(pb);
        self.pageBlocks.remove(&pbAddr);
    }
}

#[derive(Debug, Default)]
pub struct PageBlockAlloc {
    pub freeCount: AtomicU64,
    pub data: QMutex<PageBlockAllocIntern>,
}

impl PageBlockAlloc {
    pub fn New() -> Self {
        return Self::default();
    }

    pub fn PrintPages(&self) {
        self.data.lock().PrintPages();
    }

    pub fn Alloc(&self) -> Result<u64> {
        let mut al = self.data.lock();

        let pb = if al.pageBlockList == 0 {
            let newpb = PageBlock::AllocPageBlock()?;

            self.freeCount
                .fetch_add(BLOCK_PAGE_COUNT - 1, Ordering::Release);
            let (addr, _) = newpb.Alloc();
            al.Insert(newpb);
            return Ok(addr);
        } else {
            PageBlock::FromAddr(al.pageBlockList)
        };

        let (addr, count) = pb.Alloc();
        if count == 0 {
            al.UnlinkPageBlock(pb);
        }
        self.freeCount.fetch_sub(1, Ordering::Release);
        return Ok(addr);
    }

    pub fn FreePage(&self, addr: u64) -> Result<()> {
        let pb = PageBlock::FromPageAddr(addr);
        let action = pb.FreePage(addr)?;

        // try to swap in the page in case it is freed before swap in
        // todo: if disable this, system is not stable. root cause this.
        if SHARESPACE.hiberMgr.ContainersPage(addr) && !is_cc_active(){
            let _ret = HostSpace::SwapInPage(addr);
        }

        match action {
            // the pb was empty and get just get one freed page, so it can allocate page now
            PageBlockAction::OkForAlloc => {
                self.data.lock().LinkPageBlock(pb);
                self.freeCount.fetch_add(1, Ordering::Release);
            }
            // all pages of pb is freed. So it is possible to free whole pb to heap
            PageBlockAction::OkForFree => {
                if self.freeCount.load(Ordering::Acquire) >= 2 * BLOCK_PAGE_COUNT {
                    self.data.lock().Remove(pb);
                    self.freeCount
                        .fetch_sub(BLOCK_PAGE_COUNT - 1, Ordering::Release);
                    pb.Drop()?;
                } else {
                    self.freeCount.fetch_add(1, Ordering::Release);
                }
            }
            PageBlockAction::AddPageCount => {
                self.freeCount.fetch_add(1, Ordering::Release);
            }
        }

        return Ok(());
    }
}

impl RefMgr for PageBlockAlloc {
    // return ref count
    fn Ref(&self, addr: u64) -> Result<u64> {
        let (heapStart, heapEnd) = GLOBAL_ALLOCATOR.HeapRange();
        //let heapStart = MemoryDef::HEAP_OFFSET;
        //let heapEnd = MemoryDef::HEAP_OFFSET + MemoryDef::HEAP_SIZE;
        if addr <= heapStart || addr >= heapEnd {
            return Ok(1);
        }

        let pb = PageBlock::FromPageAddr(addr);
        if pb.magic != PAGE_BLOCK_MAGIC {
            return Ok(1);
        }
        //assert!(pb.magic == PAGE_BLOCK_MAGIC);
        return pb.Ref(addr);
    }

    // return ref count
    fn Deref(&self, addr: u64) -> Result<u64> {
        let (heapStart, heapEnd) = GLOBAL_ALLOCATOR.HeapRange();
        //let heapStart = MemoryDef::HEAP_OFFSET;
        //let heapEnd = MemoryDef::HEAP_OFFSET + MemoryDef::HEAP_SIZE;
        if addr <= heapStart || addr >= heapEnd {
            return Ok(1);
        }

        let pb = PageBlock::FromPageAddr(addr);
        if pb.magic != PAGE_BLOCK_MAGIC {
            return Ok(1);
        }

        let refcount = pb.Deref(addr)?;
        return Ok(refcount);
    }

    fn GetRef(&self, addr: u64) -> Result<u64> {
        let (heapStart, heapEnd) = GLOBAL_ALLOCATOR.HeapRange();
        //let heapStart = MemoryDef::HEAP_OFFSET;
        //let heapEnd = MemoryDef::HEAP_OFFSET + MemoryDef::HEAP_SIZE;

        if addr <= heapStart || addr >= heapEnd {
            return Ok(1);
        }

        let pb = PageBlock::FromPageAddr(addr);
        if pb.magic != PAGE_BLOCK_MAGIC {
            return Ok(0);
        }
        //assert!(pb.magic == PAGE_BLOCK_MAGIC);
        return pb.GetRef(addr);
    }
}

impl Allocator for PageBlockAlloc {
    fn AllocPage(&self, incrRef: bool) -> Result<u64> {
        let page = self.Alloc()?;
        if incrRef {
            self.Ref(page)?;
        }

        ZeroPage(page);
        return Ok(page);
    }
}

pub const BLOCK_SIZE: u64 = 2 * MemoryDef::PAGE_SIZE_2M;
pub const BLOCK_PAGE_COUNT: u64 = 1023;
pub const PAGE_BLOCK_MAGIC: u64 = 0x1234567890abc;

pub struct FreePageBitmap {
    pub l1bitmap: u64,
    pub l2bitmap: [u64; 16],
    pub totalFreeCount: u64,
}

impl FreePageBitmap {
    pub fn New() -> Self {
        return Self {
            l1bitmap: (1 << 16) - 1,
            l2bitmap: [
                u64::MAX - 1,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
                u64::MAX,
            ],
            totalFreeCount: BLOCK_PAGE_COUNT,
        };
    }

    pub fn Pop(&mut self) -> usize {
        if self.totalFreeCount == 0 {
            return 0;
        }

        self.totalFreeCount -= 1;

        let l1idx = self.l1bitmap.trailing_zeros() as usize;
        assert!(l1idx != 64);
        let l2idx = self.l2bitmap[l1idx].trailing_zeros() as usize;

        assert!(l2idx != 64);

        self.l2bitmap[l1idx] &= !(1 << l2idx);
        if self.l2bitmap[l1idx] == 0 {
            self.l1bitmap &= !(1 << l1idx);
        }

        return l1idx * 64 + l2idx;
    }

    pub fn Push(&mut self, idx: usize) {
        let l1idx = idx / 64;
        let l2idx = idx % 64;

        assert!(self.l2bitmap[l1idx] & (1 << l2idx) == 0);
        if self.l2bitmap[l1idx] == 0 {
            self.l1bitmap |= 1 << l1idx;
        }
        self.l2bitmap[l1idx] |= 1 << l2idx;

        self.totalFreeCount += 1;
    }

    pub fn IsFree(&self, idx: usize) -> bool {
        let l1idx = idx / 64;
        let l2idx = idx % 64;

        return self.l2bitmap[l1idx] & (1 << l2idx) != 0;
    }
}

pub struct PageBlockInternal {
    pub prev: u64,
    pub next: u64,

    // when allocated/deallocated, the page will be page in freePageList
    pub freePageList: FreePageBitmap,
}

impl PageBlockInternal {
    pub fn Init(&mut self) {
        // zero will be init by ZeroPage
        //self.prev = 0;
        //self.next = 0;
        self.freePageList = FreePageBitmap::New();
    }
}

#[derive(Debug)]
pub enum PageBlockAction {
    OkForFree,
    OkForAlloc,
    AddPageCount,
}

// PageBlock will be fix in one 4KB page
pub struct PageBlock {
    pub magic: u64,
    pub allocator: QMutex<PageBlockInternal>,

    // for a 2MB pages, the first 4KB page is the Pageblock, there are maximum 511 free pages
    pub refs: [AtomicU16; BLOCK_PAGE_COUNT as usize + 1],
}

impl PageBlock {
    pub fn FromAddr(addr: u64) -> &'static mut Self {
        return unsafe { &mut *(addr as *mut Self) };
    }

    pub fn GetPages(&self, output: &mut Vec<u64>) -> Result<()> {
        for i in 1..self.refs.len() {
            let cnt = self.refs[i].load(Ordering::Relaxed);
            if cnt > 0 {
                output.push(self.IdxToAddr(i as _));
            }
        }

        return Ok(());
    }

    pub fn PrintPages(&self) {
        for i in 1..self.refs.len() {
            let cnt = self.refs[i].load(Ordering::Relaxed);
            if cnt > 0 {
                error!(
                    "PageBlock {:x} pages {:x}/{}",
                    self.ToAddr(),
                    self.IdxToAddr(i as _),
                    cnt
                )
            }
        }
    }

    pub fn FromPageAddr(addr: u64) -> &'static mut Self {
        let addr = addr & !(BLOCK_SIZE - 1);
        return Self::FromAddr(addr);
    }

    pub fn ToAddr(&self) -> u64 {
        return self as *const _ as u64;
    }

    pub fn Init(&mut self) {
        ZeroPage(self.ToAddr());
        self.magic = PAGE_BLOCK_MAGIC;
        self.allocator.lock().Init();

        // init by ZeroPage
        /*for i in 0..self.refs.len() {
            self.refs.store(0, Ordering::Release);
        }*/
    }

    pub fn AllocPageBlock() -> Result<&'static mut Self> {
        let alloc = AlignedAllocator::New(BLOCK_SIZE as _, BLOCK_SIZE as _);
        let addr = alloc.Allocate()?;
        let ret = Self::FromAddr(addr);
        ret.Init();
        return Ok(ret);
    }

    pub fn Drop(&self) -> Result<()> {
        let alloc = AlignedAllocator::New(BLOCK_SIZE as _, BLOCK_SIZE as _);
        return alloc.Free(self.ToAddr());
    }

    pub fn Idx(&self, addr: u64) -> usize {
        let myAddr = self.ToAddr();
        assert!(myAddr < addr);
        let idx = (addr - myAddr) / MemoryDef::PAGE_SIZE_4K;
        assert!(idx <= BLOCK_PAGE_COUNT);
        return idx as _;
    }

    // return ref count
    pub fn Ref(&self, addr: u64) -> Result<u64> {
        let idx = self.Idx(addr);

        let refCnt = self.refs[idx].fetch_add(1, Ordering::SeqCst) as u64;
        return Ok(refCnt + 1);
    }

    pub fn FreePage(&self, addr: u64) -> Result<PageBlockAction> {
        let mut allocaor = self.allocator.lock();
        allocaor.freePageList.Push(self.Idx(addr));
        let mut action = PageBlockAction::AddPageCount;
        if allocaor.freePageList.totalFreeCount == 1 {
            action = PageBlockAction::OkForAlloc;
        } else if allocaor.freePageList.totalFreeCount == BLOCK_PAGE_COUNT {
            action = PageBlockAction::OkForFree;
        }

        return Ok(action);
    }

    // return (ref count, action need to perform)
    pub fn Deref(&self, addr: u64) -> Result<u64> {
        let idx = self.Idx(addr);

        let refCnt = self.refs[idx].fetch_sub(1, Ordering::SeqCst) as u64;
        assert!(refCnt > 0, "Deref refcnt is {}", refCnt);
        return Ok(refCnt - 1);
    }

    pub fn IdxToAddr(&self, idx: usize) -> u64 {
        return self.ToAddr() + (idx as u64) * MemoryDef::PAGE_SIZE_4K;
    }

    // return (pageaddr, left page count)
    // unAllocPageCnt must > 0
    pub fn Alloc(&self) -> (u64, u64) {
        let mut allocaor = self.allocator.lock();
        assert!(allocaor.freePageList.totalFreeCount > 0);
        let idx = allocaor.freePageList.Pop();
        return (self.IdxToAddr(idx), allocaor.freePageList.totalFreeCount);
    }

    pub fn GetRef(&self, addr: u64) -> Result<u64> {
        let idx = self.Idx(addr);

        let refCnt = self.refs[idx].load(Ordering::Acquire) as u64;
        return Ok(refCnt);
    }
}
