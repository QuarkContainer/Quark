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

use crate::qlib::mutex::*;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;
use x86_64::structures::paging::PageTable;
use x86_64::structures::paging::PageTableFlags;
use x86_64::PhysAddr;
use x86_64::VirtAddr;

use super::super::super::addr::*;
use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::super::object_ref::*;
use super::super::super::pagetable::*;
use super::super::super::range::*;
use super::super::asm::*;
use super::super::task::*;
use super::super::PAGE_MGR;
use super::pmamgr::*;
use crate::qlib::vcpu_mgr::CPULocal;

pub type PageMgrRef = ObjectRef<PageMgr>;

pub struct PageMgr(QMutex<PageMgrInternal>);

impl Deref for PageMgr {
    type Target = QMutex<PageMgrInternal>;

    fn deref(&self) -> &QMutex<PageMgrInternal> {
        &self.0
    }
}

impl RefMgr for PageMgr {
    fn Ref(&self, addr: u64) -> Result<u64> {
        let me = self.lock();
        return me.PagePool().lock().Ref(addr);
    }

    fn Deref(&self, addr: u64) -> Result<u64> {
        let me = self.lock();
        return me.PagePool().lock().Deref(addr);
    }

    fn GetRef(&self, addr: u64) -> Result<u64> {
        let me = self.lock();
        return me.PagePool().lock().GetRef(addr);
    }
}

impl Allocator for PageMgr {
    fn AllocPage(&self, incrRef: bool) -> Result<u64> {
        match CPULocal::Myself().pageAllocator.lock().AllocPage() {
            Some(page) => {
                ZeroPage(page);
                return Ok(page);
            }
            None => (),
        }
        let addr = self.lock().allocator.lock().AllocPage(incrRef)?;
        //error!("PageMgr allocpage ... incrRef is {}, addr is {:x}", incrRef, addr);
        return Ok(addr);
    }

    fn FreePage(&self, addr: u64) -> Result<()> {
        CPULocal::Myself().pageAllocator.lock().FreePage(addr);
        return Ok(());
        //return self.lock().allocator.lock().FreePage(addr);
    }
}

extern "C" {
    pub fn __vsyscall_page();
}

impl Default for PageMgr {
    fn default() -> Self {
        return Self::New();
    }
}

impl PageMgr {
    pub fn New() -> Self {
        return Self(QMutex::new(PageMgrInternal::New()));
    }

    pub fn Addr(&self) -> u64 {
        return self as *const _ as u64;
    }

    pub fn PrintRefs(&self) {
        self.lock().allocator.lock().PrintRefs();
    }

    pub fn DerefPage(&self, addr: u64) {
        self.lock().allocator.lock().Deref(addr).unwrap();
    }
}

pub struct PageMgrInternal {
    pub allocator: Arc<QMutex<PagePool>>,
    pub zeroPage: u64,
    pub vsyscallPages: Vec<u64>,
}

impl PageMgrInternal {
    pub fn New() -> Self {
        return Self {
            allocator: Arc::new(QMutex::new(PagePool::New())),
            zeroPage: 0,
            vsyscallPages: Vec::new(),
        };
    }

    fn PagePool(&self) -> Arc<QMutex<PagePool>> {
        return self.allocator.clone();
    }

    pub fn ZeroPage(&mut self) -> u64 {
        if self.zeroPage == 0 {
            self.zeroPage = self.allocator.lock().AllocPage(false).unwrap();
        }

        self.allocator.lock().Ref(self.zeroPage).unwrap();
        return self.zeroPage;
    }

    pub fn Deref(&self, addr: u64) {
        self.allocator.lock().Deref(addr).unwrap();
    }

    pub fn VsyscallPages(&mut self) -> &[u64] {
        if self.vsyscallPages.len() == 0 {
            for _i in 0..4 {
                let addr = self.allocator.lock().AllocPage(true).unwrap();
                self.vsyscallPages.push(addr);
            }

            self.CopyVsysCallPages();
        }

        for p in &mut self.vsyscallPages {
            self.allocator.lock().Ref(*p).unwrap();
        }

        return &self.vsyscallPages;
    }
}

impl Drop for PageTables {
    fn drop(&mut self) {
        // it will happen when execv
        if self.GetRoot() == 0 {
            return;
        }

        self.Drop();
        return;
    }
}

impl PageTables {
    pub fn Drop(&self) {
        self.UnmapAll()
            .expect("FreePageTables::Drop fail at UnmapAll");
        self.SetRoot(0);
    }

    // create a new PageTable by clone the kernel pages.
    // 1. Empty page is cloned
    // 2. Kernel takes the address space from 256GB ~ 512GB
    // 3. pages for ffffffffff600000
    pub fn Fork(&self, pagePool: &PageMgr) -> Result<Self> {
        let pt = self.NewWithKernelPageTables(pagePool)?;
        return Ok(pt);
    }

    pub fn InitVsyscall(&self, phyAddrs: &[u64] /*4 pages*/) {
        let vaddr = 0xffffffffff600000;
        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;
        unsafe {
            let p4Idx = VirtAddr::new(vaddr).p4_index();
            let p3Idx = VirtAddr::new(vaddr).p3_index();
            let p2Idx = VirtAddr::new(vaddr).p2_index();
            let p1Idx = VirtAddr::new(vaddr).p1_index();

            let pgdEntry = &mut (*pt)[p4Idx];
            let pudTbl: *mut PageTable;

            assert!(pgdEntry.is_unused());
            pudTbl = phyAddrs[3] as *mut PageTable;
            pgdEntry.set_addr(
                PhysAddr::new(pudTbl as u64),
                PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE,
            );

            let pudEntry = &mut (*pudTbl)[p3Idx];
            let pmdTbl: *mut PageTable;

            assert!(pudEntry.is_unused());
            pmdTbl = phyAddrs[2] as *mut PageTable;
            pudEntry.set_addr(
                PhysAddr::new(pmdTbl as u64),
                PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE,
            );

            let pmdEntry = &mut (*pmdTbl)[p2Idx];
            let pteTbl: *mut PageTable;

            assert!(pmdEntry.is_unused());
            pteTbl = phyAddrs[1] as *mut PageTable;
            pmdEntry.set_addr(
                PhysAddr::new(pteTbl as u64),
                PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE,
            );

            let pteEntry = &mut (*pteTbl)[p1Idx];
            assert!(pteEntry.is_unused());
            pteEntry.set_addr(
                PhysAddr::new(phyAddrs[0]),
                PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE,
            );

            Invlpg(vaddr);
        }
    }

    // There are following pages need allocated:
    // 1: root page
    // 2: p3 page for 0GB to 512G

    pub fn NewWithKernelPageTables(&self, pagePool: &PageMgr) -> Result<Self> {
        let ret = Self::New(pagePool)?;

        unsafe {
            let pt: *mut PageTable = self.GetRoot() as *mut PageTable;
            let pgdEntry = &(*pt)[0];
            if pgdEntry.is_unused() {
                return Err(Error::AddressNotMap(0));
            }
            let pudTbl = pgdEntry.addr().as_u64() as *const PageTable;

            let nPt: *mut PageTable = ret.GetRoot() as *mut PageTable;
            let nPgdEntry = &mut (*nPt)[0];
            let nPudTbl = pagePool.AllocPage(true)? as *mut PageTable;
            nPgdEntry.set_addr(
                PhysAddr::new(nPudTbl as u64),
                PageTableFlags::PRESENT
                    | PageTableFlags::WRITABLE
                    | PageTableFlags::USER_ACCESSIBLE,
            );

            for i in MemoryDef::KERNEL_START_P2_ENTRY..MemoryDef::KERNEL_END_P2_ENTRY {
                //memspace between 256GB to 512GB
                //copy entry[i]
                *(&mut (*nPudTbl)[i] as *mut _ as *mut u64) =
                    *(&(*pudTbl)[i] as *const _ as *const u64);
            }
        }

        ret.MapPage(
            Addr(MemoryDef::KVM_IOEVENTFD_BASEADDR),
            Addr(MemoryDef::KVM_IOEVENTFD_BASEADDR),
            PageOpts::Kernel().Val(),
            pagePool,
        )?;

        {
            let mut lock = pagePool.lock();
            let vsyscallPages = lock.VsyscallPages();
            ret.MapVsyscall(vsyscallPages);
        }

        return Ok(ret);
    }

    pub fn RemapAna(
        &self,
        _task: &Task,
        newAddrRange: &Range,
        oldStart: u64,
        at: &AccessType,
        user: bool,
    ) -> Result<()> {
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

        self.Remap(
            Addr(newAddrRange.Start()),
            Addr(newAddrRange.End()),
            Addr(oldStart),
            pageOpts,
            &*PAGE_MGR,
        )?;

        return Ok(());
    }

    pub fn RemapHost(
        &self,
        _task: &Task,
        addr: u64,
        phyRange: &IoVec,
        oldar: &Range,
        at: &AccessType,
        user: bool,
    ) -> Result<()> {
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

        self.RemapForFile(
            Addr(addr),
            Addr(addr + phyRange.Len() as u64),
            Addr(phyRange.Start()),
            Addr(oldar.Start()),
            Addr(oldar.End()),
            pageOpts,
            &*PAGE_MGR,
        )?;

        return Ok(());
    }

    pub fn MapHost(
        &self,
        _task: &Task,
        addr: u64,
        phyRange: &IoVec,
        at: &AccessType,
        user: bool,
    ) -> Result<()> {
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

        self.Map(
            Addr(addr),
            Addr(addr + phyRange.Len() as u64),
            Addr(phyRange.Start()),
            pageOpts,
            &*PAGE_MGR,
            !user,
        )?;

        return Ok(());
    }

    pub fn PrintZero(&self) {
        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;

        let pgdEntry = unsafe { &(*pt)[0] };

        assert!(!pgdEntry.is_unused(), "pagetable::Drop page is not mapped");

        let pudTblAddr = pgdEntry.addr().as_u64();
        let pudTbl = pudTblAddr as *mut PageTable;
        let pudEntry = unsafe { &(*pudTbl)[0] };
        //assert!(!pudEntry.is_unused(), "pagetable::Drop page is not mapped");

        let pmdTblAddr = pudEntry.addr().as_u64();
        let pmdTbl = pmdTblAddr as *mut PageTable;
        let pmdEntry = unsafe { &(*pmdTbl)[0] };
        //assert!(!pmdEntry.is_unused(), "pagetable::Drop page is not mapped");

        let pteTblAddr = pmdEntry.addr().as_u64();

        error!(
            "PrintZero pudTblAddr is {:x}, pmdTblAddr is {:x}, pteTblAddr is {:x}",
            pudTblAddr, pmdTblAddr, pteTblAddr
        );
    }

    pub fn UnmapAll(&self) -> Result<()> {
        self.Unmap(MemoryDef::PAGE_SIZE, MemoryDef::PHY_LOWER_ADDR, &*PAGE_MGR)?;
        self.Unmap(MemoryDef::PHY_UPPER_ADDR, MemoryDef::LOWER_TOP, &*PAGE_MGR)?;

        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;

        let pgdEntry = unsafe { &(*pt)[0] };

        assert!(!pgdEntry.is_unused(), "pagetable::Drop page is not mapped");

        let pudTblAddr = pgdEntry.addr().as_u64();

        PAGE_MGR.Deref(pudTblAddr).expect("PageTable::Drop fail");
        PAGE_MGR.Deref(self.GetRoot())?;
        return Ok(());
    }

    pub fn MUnmap(&mut self, addr: u64, len: u64) -> Result<()> {
        return self.Unmap(addr, addr + len, &*PAGE_MGR);
    }
}
