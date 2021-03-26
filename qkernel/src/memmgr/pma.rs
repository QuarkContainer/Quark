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

use core::ops::Deref;
use spin::Mutex;
use spin::RwLock;
use alloc::sync::Arc;
use alloc::vec::Vec;
use x86_64::structures::paging::{PageTable};

use super::super::task::*;
use super::super::PAGE_MGR;
use super::super::qlib::addr::*;
use super::super::qlib::range::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::pagetable::*;
use super::super::qlib::vcpu_mgr::*;
use super::pmamgr::*;

pub fn AddFreePageTables(pt: &PageTablesInternal) {
    CPULocal::SetPendingFreePagetable(pt.root.0);
}

#[inline]
pub fn FreePageTables() {
    let ptAddr = CPULocal::PendingFreePagetable();
    if ptAddr != 0 {
        let _pt = FreePageTables {
            root: Addr(ptAddr)
        };
        CPULocal::SetPendingFreePagetable(0);
    }
}

pub struct FreePageTables {
    pub root: Addr
}

impl FreePageTables {
    pub fn Drop(&mut self, pagePool: &Allocator) {
        let root = self.root.0;
        if root == 0 {
            return;
        }

        let pt: *mut PageTable = self.root.0 as *mut PageTable;
        let pgdEntry = unsafe {
            &(*pt)[0]
        };
        if pgdEntry.is_unused() {
            panic!("pagetable::Drop page is not mapped")
        }

        let pudTblAddr = pgdEntry.addr().as_u64();
        pagePool.Deref(pudTblAddr).expect("PageTable::Drop fail");

        pagePool.Deref(root).expect("PageTable::Drop fail");
    }
}

pub struct PageMgr(Mutex<PageMgrInternal>);

impl Deref for PageMgr {
    type Target = Mutex<PageMgrInternal>;

    fn deref(&self) -> &Mutex<PageMgrInternal> {
        &self.0
    }
}

impl RefMgr for PageMgr {
    fn Ref(&self, addr: u64) -> Result<u64> {
        return self.RefIntern(addr);
    }

    fn Deref(&self, addr: u64) -> Result<u64> {
        return self.DerefIntern(addr);
    }

    fn GetRef(&self, addr: u64) -> Result<u64> {
        let me = self.lock();
        return me.PagePool().lock().GetRef(addr)
    }
}

impl Allocator for PageMgr {
    fn AllocPage(&self, incrRef: bool) -> Result<u64> {
        let ret = self.lock().allocator.lock().AllocPage(incrRef)?;
        return Ok(ret);
    }

    fn FreePage(&self, addr: u64) -> Result<()> {
        return self.lock().allocator.lock().FreePage(addr)
    }

    fn ZeroPage(&self) -> u64 {
        return self.lock().allocator.lock().GetZeroPage();
    }
}

extern "C" {
    pub fn __vsyscall_page();
}

impl PageMgr {
    pub fn New() -> Self {
        return Self(Mutex::new(PageMgrInternal::New()))
    }

    fn RefIntern(&self, addr: u64) -> Result<u64> {
        let me = self.lock();
        return me.PagePool().lock().Ref(addr);
    }

    fn DerefIntern(&self, addr: u64) -> Result<u64> {
        let me = self.lock();
        return me.PagePool().lock().Deref(addr);
    }
}

pub struct PageMgrInternal {
    pub allocator: Arc<Mutex<PagePool>>,
    pub zeroPage: u64,
    pub vsyscallPages: Vec<u64>,
}

impl PageMgrInternal {
    pub fn New() -> Self {
        return Self {
            allocator: Arc::new(Mutex::new(PagePool::New())),
            zeroPage: 0,
            vsyscallPages: Vec::new(),
        }
    }

    pub fn Init(&mut self) {
        self.allocator.lock().Init();
    }

    fn PagePool(&self) -> Arc<Mutex<PagePool>> {
        return self.allocator.clone();
    }

    pub fn ZeroPage(&mut self) -> u64 {
        if self.zeroPage == 0 {
            self.zeroPage = self.allocator.lock().AllocPage(false).unwrap();
            self.allocator.lock().Ref(self.zeroPage).unwrap();
        }

        return self.zeroPage;
    }

    pub fn VsyscallPages(&mut self) -> &[u64] {
        if self.vsyscallPages.len() == 0 {
            for _i in 0..4 {
                let addr = self.allocator.lock().AllocPage(true).unwrap();
                self.vsyscallPages.push(addr);
            }

            CopyPage(__vsyscall_page as u64, self.vsyscallPages[0]);
        }

        for p in &mut self.vsyscallPages {
            self.allocator.lock().Ref(*p).unwrap();
        }

        return &self.vsyscallPages;
    }
}

impl Drop for PageTablesInternal {
    fn drop(&mut self) {
        //pagetables can't be free from current kernel thread, need to be free async
        AddFreePageTables(self);
    }
}

impl Drop for FreePageTables {
    fn drop(&mut self) {
        self.Drop(&*PAGE_MGR);
    }
}

impl PageTables {
    // create a new PageTable by clone the kernel pages.
    // 1. Empty page is cloned
    // 2. Kernel takes the address space from 256GB ~ 512GB
    // 3. pages for ffffffffff600000
    pub fn Fork(&self, pagePool: &PageMgr) -> Result<Self> {
        let internal = self.read();
        let ptInternal = internal.NewWithKernelPageTables(pagePool)?;
        return Ok(Self(Arc::new(RwLock::new(ptInternal))));
    }
}

impl PageTablesInternal {
    // There are following pages need allocated:
    // 1: root page
    // 2: p3 page for 0GB to 512G
    // 3&4: p2, p1 page for Empty page at 0 address

    //  zero page, p3, p2, p1, p0 for ffffffffff600000 will be reused
    pub fn NewWithKernelPageTables(&self, pagePool: &PageMgr) -> Result<Self> {
        let mut ret = Self::New(pagePool)?;

        let zeroPage = pagePool.ZeroPage();
        ret.MapPage(Addr(0), Addr(zeroPage), PageOpts::UserReadOnly().Val(), pagePool)?;

        {
            let mut lock = pagePool.lock();
            let vsyscallPages = lock.VsyscallPages();
            ret.MapVsyscall(vsyscallPages);
        }

        unsafe {
            let pt: *mut PageTable = self.root.0 as *mut PageTable;
            let pgdEntry = &(*pt)[0];
            if pgdEntry.is_unused() {
                return Err(Error::AddressNotMap(0))
            }
            let pudTbl = pgdEntry.addr().as_u64() as *const PageTable;


            let nPt: *mut PageTable = ret.root.0 as *mut PageTable;
            let nPgdEntry = &mut (*nPt)[0];
            let nPudTbl = nPgdEntry.addr().as_u64() as *mut PageTable;

            for i in MemoryDef::KERNEL_START_P2_ENTRY..MemoryDef::KERNEL_END_P2_ENTRY {
                //memspace between 256GB to 512GB
                //copy entry[i]
                //note: only copy p3 entry, reuse p2, p1 page
                *(&mut (*nPudTbl)[i] as *mut _ as *mut u64) = *(&(*pudTbl)[i] as *const _ as *const u64);
            }
        }

        return Ok(ret)
    }

    pub fn RemapAna(&mut self, _task: &Task, newAddrRange: &Range, oldStart: u64, at: &AccessType, user: bool) -> Result<()> {
        let pageOpts = if user {
            if at.Write() {
                PageOpts::UserReadWrite().Val()
            } else if at.Read() || at.Exec() {
                PageOpts::UserReadOnly().Val()
            } else {
                PageOpts::UserNonAccessable().Val()
            }
        } else {
            if at.Write() {
                PageOpts::KernelReadWrite().Val()
            } else {
                PageOpts::KernelReadOnly().Val()
            }
        };

        self.Remap(Addr(newAddrRange.Start()),
                          Addr(newAddrRange.End()),
                          Addr(oldStart),
                          pageOpts,
                          &*PAGE_MGR)?;

        return Ok(())
    }

    pub fn RemapHost(&mut self, _task: &Task, addr: u64, phyRange: &IoVec, oldar: &Range, at: &AccessType, user: bool) -> Result<()> {
        let pageOpts = if user {
            if at.Write() {
                PageOpts::UserReadWrite().Val()
            } else if at.Read() || at.Exec() {
                PageOpts::UserReadOnly().Val()
            } else {
                PageOpts::UserNonAccessable().Val()
            }
        } else {
            if at.Write() {
                PageOpts::KernelReadWrite().Val()
            } else {
                PageOpts::KernelReadOnly().Val()
            }
        };

        self.RemapForFile(Addr(addr),
                          Addr(addr + phyRange.Len() as u64),
                          Addr(phyRange.Start()),
                          Addr(oldar.Start()),
                          Addr(oldar.End()),
                          pageOpts,
                          &*PAGE_MGR)?;

        return Ok(())
    }

    pub fn MapHost(&mut self, _task: &Task, addr: u64, phyRange: &IoVec, at: &AccessType, user: bool) -> Result<()> {
        let pageOpts = if user {
            if at.Write() {
                PageOpts::UserReadWrite().Val()
            } else if at.Read() || at.Exec() {
                PageOpts::UserReadOnly().Val()
            } else {
                PageOpts::UserNonAccessable().Val()
            }
        } else {
            if at.Write() {
                PageOpts::KernelReadWrite().Val()
            } else {
                PageOpts::KernelReadOnly().Val()
            }
        };

        self.Map(Addr(addr), Addr(addr + phyRange.Len() as u64), Addr(phyRange.Start()), pageOpts, &*PAGE_MGR, !user)?;

        return Ok(())
    }

    pub fn MUnmap(&mut self, addr: u64, len: u64) -> Result<()> {
        return self.Unmap(addr, addr + len, &*PAGE_MGR);
    }
}