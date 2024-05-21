// Copyright (c) 2022 Quark Container Authors / 2018 The gVisor Authors.
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

use alloc::alloc::{alloc, dealloc, Layout};
use alloc::collections::BTreeSet;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::hint::spin_loop;
use core::sync::atomic::fence;
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use crate::qlib::kernel::SHARESPACE;
use crate::{GLOBAL_ALLOCATOR, IS_GUEST};

cfg_cc! {
    use crate::qlib::kernel::Kernel::ENABLE_CC;
}

cfg_x86_64! {
   pub use x86_64::structures::paging::page_table::PageTableEntry;
   pub use x86_64::structures::paging::page_table::PageTableIndex;
   pub use x86_64::structures::paging::PageTable;
   pub use x86_64::structures::paging::PageTableFlags;
   pub use x86_64::PhysAddr;
   pub use x86_64::VirtAddr;

   #[inline]
   pub fn default_table_user() -> PageTableFlags {
       return PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE | PageTableFlags::WRITABLE;
   }
}

cfg_aarch64! {
   pub use super::pagetable_aarch64::PageTableEntry;
   pub use super::pagetable_aarch64::PageTableIndex;
   pub use super::pagetable_aarch64::PageTable;
   pub use super::pagetable_aarch64::PageTableFlags;
   pub use super::pagetable_aarch64::PhysAddr;
   pub use super::pagetable_aarch64::VirtAddr;

   #[inline]
   pub fn default_table_user() -> PageTableFlags {
       return PageTableFlags::VALID | PageTableFlags::TABLE | PageTableFlags::ACCESSED | PageTableFlags::USER_ACCESSIBLE;
   }
}

use super::super::asm::*;
use super::addr::*;
use super::common::{Allocator, Error, Result};
use super::kernel::Kernel::HostSpace;
use super::linux_def::*;
use super::mutex::*;
use crate::kernel_def::Invlpg;
use crate::qlib::kernel::PAGE_MGR;

#[derive(Default)]
pub struct PageTables {
    //Root page guest physical address
    pub root: AtomicU64,
    pub tlbEpoch: AtomicU64,
    pub tlbshootdown: AtomicBool,
    pub freePages: QMutex<Vec<u64>>,
    pub hibernateLock: QMutex<()>,
}

impl PageTables {
    pub fn New(pagePool: &Allocator) -> Result<Self> {
        let root = if IS_GUEST {
            let root = pagePool.AllocPage(true)?;
            root
        } else {
            // on host
            let root = pagePool.AllocPage(true)?;
            MemoryDef::hva_to_gpa(root)
        };

        Ok(Self {
            root: AtomicU64::new(root),
            tlbEpoch: AtomicU64::new(0),
            tlbshootdown: AtomicBool::new(false),
            freePages: Default::default(),
            hibernateLock: Default::default(),
        })
    }

    pub fn TlbShootdown(&self) -> bool {
        return self.tlbshootdown.swap(false, Ordering::SeqCst);
    }

    pub fn EnableTlbShootdown(&self) {
        self.tlbEpoch.fetch_add(1, Ordering::SeqCst);
        self.tlbshootdown.store(true, Ordering::SeqCst)
    }

    pub fn TLBEpoch(&self) -> u64 {
        return self.tlbEpoch.load(Ordering::SeqCst);
    }

    pub fn Clone(&self) -> Self {
        return Self {
            root: AtomicU64::new(self.GetRoot()),
            tlbEpoch: AtomicU64::new(0),
            tlbshootdown: AtomicBool::new(false),
            freePages: Default::default(),
            hibernateLock: Default::default(),
        };
    }

    pub fn Init(root: u64) -> Self {
        return Self {
            root: AtomicU64::new(root),
            tlbEpoch: AtomicU64::new(0),
            tlbshootdown: AtomicBool::new(false),
            freePages: Default::default(),
            hibernateLock: Default::default(),
        };
    }

    pub fn SwitchTo(&self) {
        #[cfg(feature = "cc")]
        if ENABLE_CC.load(Ordering::Acquire) {
            return self.SwitchTo_cc();
        }
        let addr = self.GetRoot();
        Self::Switch(addr);
    }

    #[cfg(target_arch = "x86_64")]
    pub fn IsActivePagetable(&self) -> bool {
        #[cfg(feature = "cc")]
        if ENABLE_CC.load(Ordering::Acquire) {
            match self.IsActivePagetable_cc() {
                Err(_) => (),
                Ok(r) => return r,
            }
        }
        let root = self.GetRoot();
        return root == Self::CurrentCr3();
    }

    #[cfg(target_arch = "x86_64")]
    pub fn CurrentCr3() -> u64 {
        return CurrentCr3();
    }

    //switch pagetable for the cpu, Cr3
    #[cfg(target_arch = "x86_64")]
    pub fn Switch(cr3: u64) {
        LoadCr3(cr3)
    }

    #[cfg(target_arch = "aarch64")]
    pub fn Switch(table: u64) {
        LoadUserTable(table)
    }

    pub fn SetRoot(&self, root: u64) {
        self.root.store(root, Ordering::Release)
    }

    pub fn GetRoot(&self) -> u64 {
        return self.root.load(Ordering::Acquire);
    }

    pub fn SwapZero(&self) -> u64 {
        return self.root.swap(0, Ordering::Acquire);
    }

    #[cfg(target_arch = "x86_64")]
    pub fn Print(&self) {
        let cr3 = CurrentCr3();

        info!(
            "the page root is {:x}, the cr3 is {:x}",
            self.GetRoot(),
            cr3
        );
    }

    pub fn CopyRange(&self, to: &Self, start: u64, len: u64, pagePool: &Allocator) -> Result<()> {
        #[cfg(feature = "cc")]
        if ENABLE_CC.load(Ordering::Acquire) {
            match self.CopyRange_cc(to, start, len, pagePool) {
                Err(Error::InvalidInput) => (),
                r => return r,
            }
        }
        if start & MemoryDef::PAGE_MASK != 0 || len & MemoryDef::PAGE_MASK != 0 {
            return Err(Error::UnallignedAddress(format!("CopyRange {:x?}", len)));
        }

        let mut vAddr = start;
        while vAddr < start + len {
            match self.VirtualToEntry(vAddr) {
                Ok(entry) => {
                    let phyAddr = entry.addr().as_u64();
                    to.MapPage(Addr(vAddr), Addr(phyAddr), entry.flags(), pagePool)?;
                }
                Err(_) => (),
            }
            vAddr += MemoryDef::PAGE_SIZE;
        }

        Ok(())
    }

    // Copy the range and make the range readonly for from and to pagetable. It is used for VirtualArea private area.
    // The Copy On Write will be done when write to the page
    pub fn ForkRange(&self, to: &Self, start: u64, len: u64, pagePool: &Allocator) -> Result<()> {
        #[cfg(feature = "cc")]
        if ENABLE_CC.load(Ordering::Acquire) {
            match self.ForkRange_cc(to, start, len, pagePool) {
                Err(Error::InvalidInput) => (),
                r => return r,
            }
        }
        if start & MemoryDef::PAGE_MASK != 0 || len & MemoryDef::PAGE_MASK != 0 {
            return Err(Error::UnallignedAddress(format!(
                "ForkRange start {:x} len {:x}",
                start, len
            )));
        }

        //change to read only
        //todo: there is chance the orignal range is changed to readonly by mprotected before. Need to handle.
        let _ = self.MProtect(
            Addr(start),
            Addr(start + len),
            PageOpts::UserReadOnly().Val(),
            false,
        ); //there won't be any failure

        let mut vAddr = start;
        while vAddr < start + len {
            match self.VirtualToEntry(vAddr) {
                Ok(entry) => {
                    let phyAddr = entry.addr().as_u64();
                    to.MapPage(
                        Addr(vAddr),
                        Addr(phyAddr),
                        PageOpts::UserReadOnly().Val(),
                        pagePool,
                    )?;
                }
                Err(_) => (),
            }
            vAddr += MemoryDef::PAGE_SIZE;
        }

        Ok(())
    }

    pub fn PrintPath(&self, vaddr: u64) {
        let vaddr = VirtAddr::new(vaddr);

        let p4Idx = vaddr.p4_index();
        let p3Idx = vaddr.p3_index();
        let p2Idx = vaddr.p2_index();
        let p1Idx = vaddr.p1_index();

        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;

        unsafe {
            info!("pt1: {:x}", self.GetRoot());
            let pgdEntry = &(*pt)[p4Idx];
            if pgdEntry.is_unused() {
                return;
            }

            let pudTbl = pgdEntry.addr().as_u64() as *const PageTable;
            info!("pt2: {:x}", pgdEntry.addr().as_u64());
            let pudEntry = &(*pudTbl)[p3Idx];
            if pudEntry.is_unused() {
                return;
            }

            let pmdTbl = pudEntry.addr().as_u64() as *const PageTable;
            info!("pt3: {:x}", pudEntry.addr().as_u64());
            let pmdEntry = &(*pmdTbl)[p2Idx];
            if pmdEntry.is_unused() {
                return;
            }

            let pteTbl = pmdEntry.addr().as_u64() as *const PageTable;
            info!("pt4: {:x}", pmdEntry.addr().as_u64());
            let pteEntry = &(*pteTbl)[p1Idx];
            if pteEntry.is_unused() {
                return;
            }
        }
    }

    #[inline]
    pub fn VirtualToEntry(&self, vaddr: u64) -> Result<&PageTableEntry> {
        #[cfg(feature = "cc")]
        if ENABLE_CC.load(Ordering::Acquire) {
            match self.VirtualToEntry_cc(vaddr) {
                Err(Error::InvalidInput) => (),
                r => return r,
            }
        }

        let addr = vaddr;
        let vaddr = VirtAddr::new(vaddr);

        let p4Idx = vaddr.p4_index();
        let p3Idx = vaddr.p3_index();
        let p2Idx = vaddr.p2_index();
        let p1Idx = vaddr.p1_index();

        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;

        unsafe {
            let pgdEntry = &(*pt)[p4Idx];
            if pgdEntry.is_unused() {
                return Err(Error::AddressNotMap(addr));
            }

            let pudTbl = pgdEntry.addr().as_u64() as *const PageTable;
            let pudEntry = &(*pudTbl)[p3Idx];
            if pudEntry.is_unused() {
                return Err(Error::AddressNotMap(addr));
            }

            let pmdTbl = pudEntry.addr().as_u64() as *const PageTable;
            let pmdEntry = &(*pmdTbl)[p2Idx];
            if pmdEntry.is_unused() {
                return Err(Error::AddressNotMap(addr));
            }

            let pteTbl = pmdEntry.addr().as_u64() as *mut PageTable;
            let pteEntry = &mut (*pteTbl)[p1Idx];
            if pteEntry.is_unused() {
                return Err(Error::AddressNotMap(addr));
            }

            // try to swapin page if it is swapout
            self.HandlingSwapInPage(addr, pteEntry);

            return Ok(pteEntry);
        }
    }

    pub fn VirtualToPhy(&self, vaddr: u64) -> Result<(u64, AccessType)> {
        #[cfg(feature = "cc")]
        if ENABLE_CC.load(Ordering::Acquire) {
            match self.VirtualToPhy_cc(vaddr) {
                Err(Error::InvalidInput) => (),
                r => return r,
            }
        }
        let pteEntry = self.VirtualToEntry(vaddr)?;
        if pteEntry.is_unused() {
            return Err(Error::AddressNotMap(vaddr));
        }

        let vaddr = VirtAddr::new(vaddr);
        let pageAddr: u64 = vaddr.page_offset().into();
        let phyAddr = pteEntry.addr().as_u64() + pageAddr;
        let permission = AccessType::NewFromPageFlags(pteEntry.flags());

        return Ok((phyAddr, permission));
    }

    pub fn PrintPageFlags(&self, vaddr: u64) -> Result<()> {
        let pteEntry = self.VirtualToEntry(vaddr)?;
        if pteEntry.is_unused() {
            return Err(Error::AddressNotMap(vaddr));
        }

        info!("Flags is {:x?}", pteEntry);
        return Ok(());
    }

    pub fn MapVsyscall(&self, phyAddrs: Arc<Vec<u64>> /*4 pages*/) {
        #[cfg(feature = "cc")]
        if ENABLE_CC.load(Ordering::Acquire) {
            match self.MapVsyscall_cc(phyAddrs.clone()) {
                Err(Error::InvalidInput) => (),
                _ => return,
            }
        }
        let vaddr = 0xffffffffff600000;
        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;
        unsafe {
            let p4Idx = VirtAddr::new(vaddr).p4_index();
            let pgdEntry = &mut (*pt)[p4Idx];
            let pudTbl: *mut PageTable;

            assert!(pgdEntry.is_unused());
            pudTbl = phyAddrs[3] as *mut PageTable;

            pgdEntry.set_addr(
                PhysAddr::new(pudTbl as u64),
                PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE,
            );
            Invlpg(vaddr);
        }
    }

    pub fn MapPage(
        &self,
        vaddr: Addr,
        phyAddr: Addr,
        flags: PageTableFlags,
        pagePool: &Allocator,
    ) -> Result<bool> {
        #[cfg(feature = "cc")]
        if ENABLE_CC.load(Ordering::Acquire) {
            match self.MapPage_cc(vaddr, phyAddr, flags, pagePool) {
                Err(Error::InvalidInput) => (),
                r => return r,
            }
        }
        let mut res = false;

        let vaddr = Addr(vaddr.0 & !(PAGE_SIZE - 1));
        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;
        unsafe {
            let p4Idx = VirtAddr::new(vaddr.0).p4_index();
            let p3Idx = VirtAddr::new(vaddr.0).p3_index();
            let p2Idx = VirtAddr::new(vaddr.0).p2_index();
            let p1Idx = VirtAddr::new(vaddr.0).p1_index();

            let pgdEntry = &mut (*pt)[p4Idx];
            let pudTbl: *mut PageTable;

            if pgdEntry.is_unused() {
                pudTbl = pagePool.AllocPage(true)? as *mut PageTable;
                pgdEntry.set_addr(PhysAddr::new(pudTbl as u64), default_table_user());
            } else {
                pudTbl = pgdEntry.addr().as_u64() as *mut PageTable;
            }

            let pudEntry = &mut (*pudTbl)[p3Idx];
            let pmdTbl: *mut PageTable;

            if pudEntry.is_unused() {
                pmdTbl = pagePool.AllocPage(true)? as *mut PageTable;
                pudEntry.set_addr(PhysAddr::new(pmdTbl as u64), default_table_user());
            } else {
                pmdTbl = pudEntry.addr().as_u64() as *mut PageTable;
            }

            let pmdEntry = &mut (*pmdTbl)[p2Idx];
            let pteTbl: *mut PageTable;

            if pmdEntry.is_unused() {
                pteTbl = pagePool.AllocPage(true)? as *mut PageTable;
                pmdEntry.set_addr(PhysAddr::new(pteTbl as u64), default_table_user());
            } else {
                pteTbl = pmdEntry.addr().as_u64() as *mut PageTable;
            }

            let pteEntry = &mut (*pteTbl)[p1Idx];

            pagePool.Ref(phyAddr.0).unwrap();
            if !pteEntry.is_unused() {
                self.freeEntry(pteEntry, pagePool)?;

                /*let addr = pteEntry.addr().as_u64();
                let bit9 = pteEntry.flags() & PageTableFlags::BIT_9 == PageTableFlags::BIT_9;

                if vaddr.0 != 0 && !bit9 {
                    pagePool.Deref(addr).unwrap();
                }*/
                res = true;
            }

            pteEntry.set_addr(PhysAddr::new(phyAddr.0), flags);
            Invlpg(vaddr.0);
        }

        return Ok(res);
    }

    pub fn FreePage(&self, page: u64) {
        self.freePages.lock().push(page);
    }

    pub fn FreePages(&self) {
        let mut pages = self.freePages.lock();
        loop {
            match pages.pop() {
                None => break,
                Some(page) => {
                    PAGE_MGR.FreePage(page).unwrap();

                    /*let layout = Layout::from_size_align(MemoryDef::PAGE_SIZE_4K as _, MemoryDef::PAGE_SIZE_4K as _);
                    match layout {
                        Err(_e) => {
                            panic!("pagetable free unaligned page {:x}", page);
                        },
                        Ok(l) => unsafe {
                            dealloc(page as *mut u8, l);
                        },
                    }*/
                }
            }
        }
    }

    pub fn Remap(
        &self,
        start: Addr,
        end: Addr,
        oldStart: Addr,
        flags: PageTableFlags,
        pagePool: &Allocator,
    ) -> Result<bool> {
        #[cfg(feature = "cc")]
        if ENABLE_CC.load(Ordering::Acquire) {
            match self.Remap_cc(start, end, oldStart, flags, pagePool) {
                Err(Error::InvalidInput) => (),
                r => return r,
            }
        }
        start.PageAligned()?;
        oldStart.PageAligned()?;
        if end.0 < start.0 {
            return Err(Error::AddressNotInRange);
        }
        let mut addrs = Vec::new();

        let mut offset = 0;
        while start.0 + offset < end.0 {
            let entry = self.VirtualToEntry(oldStart.0 + offset);
            match entry {
                Ok(oldentry) => {
                    let phyAddr = oldentry.addr().as_u64();
                    addrs.push(Some(phyAddr));
                    pagePool.Ref(phyAddr).unwrap();
                    self.Unmap(
                        oldStart.0 + offset,
                        oldStart.0 + offset + MemoryDef::PAGE_SIZE,
                        pagePool,
                    )?;
                }
                Err(_) => {
                    addrs.push(None);
                }
            }
            offset += MemoryDef::PAGE_SIZE;
        }

        let mut offset = 0;
        let mut idx = 0;
        while start.0 + offset < end.0 {
            match addrs[idx] {
                Some(phyAddr) => {
                    self.MapPage(Addr(start.0 + offset), Addr(phyAddr), flags, pagePool)?;
                    let count = pagePool.Deref(phyAddr).unwrap();
                    if count == 0 {
                        self.FreePage(phyAddr);
                    }
                }
                None => (),
            }
            offset += MemoryDef::PAGE_SIZE;
            idx += 1;
        }

        return Ok(false);
    }

    pub fn RemapForFile(
        &self,
        start: Addr,
        end: Addr,
        physical: Addr,
        oldStart: Addr,
        oldEnd: Addr,
        flags: PageTableFlags,
        pagePool: &Allocator,
    ) -> Result<bool> {
        #[cfg(feature = "cc")]
        if ENABLE_CC.load(Ordering::Acquire) {
            match self.RemapForFile_cc(start, end, physical, oldStart, oldEnd, flags, pagePool) {
                Err(Error::InvalidInput) => (),
                r => return r,
            }
        }
        start.PageAligned()?;
        oldStart.PageAligned()?;
        if end.0 < start.0 {
            return Err(Error::AddressNotInRange);
        }

        /*let mut addrs = Vec::new();

        let mut offset = 0;
        while start.0 + offset < end.0 {
            let entry = self.VirtualToEntry(oldStart.0 + offset);
            match entry {
                Ok(oldentry) => {
                    let phyAddr = oldentry.addr().as_u64();
                    addrs.push(Some(phyAddr));
                    pagePool.Ref(phyAddr).unwrap();
                    self.Unmap(oldStart.0 + offset, oldStart.0 + offset + MemoryDef::PAGE_SIZE, pagePool)?;
                }
                Err(_) => {
                    addrs.push(None);
                }
            }
            offset += MemoryDef::PAGE_SIZE;
        }*/

        // todo: handle overlap...
        let mut offset = 0;
        'a: while start.0 + offset < end.0 {
            if oldStart.0 + offset < oldEnd.0 {
                let entry = self.VirtualToEntry(oldStart.0 + offset);
                match entry {
                    Ok(oldentry) => {
                        let phyAddr = oldentry.addr().as_u64();
                        self.MapPage(Addr(start.0 + offset), Addr(phyAddr), flags, pagePool)?;
                        self.Unmap(
                            oldStart.0 + offset,
                            oldStart.0 + offset + MemoryDef::PAGE_SIZE,
                            pagePool,
                        )?;
                        offset += MemoryDef::PAGE_SIZE;
                        continue 'a;
                    }
                    Err(_) => (),
                }
            }

            let targetPhyAddr = physical.0 + offset;
            self.MapPage(Addr(start.0 + offset), Addr(targetPhyAddr), flags, pagePool)?;

            offset += MemoryDef::PAGE_SIZE;
        }

        return Ok(false);
    }

    pub fn MapWith1G(
        &self,
        start: Addr,
        end: Addr,
        physical: Addr,
        flags: PageTableFlags,
        pagePool: &Allocator,
        _kernel: bool,
    ) -> Result<bool> {
        assert!(IS_GUEST == false, "MapWith1G");
        if start.0 & (MemoryDef::HUGE_PAGE_SIZE_1G - 1) != 0
            || end.0 & (MemoryDef::HUGE_PAGE_SIZE_1G - 1) != 0
        {
            info!("start/end address not 1G aligned");
            panic!("start/end address not 1G aligned")
        }

        let mut res = false;

        let mut curAddr = start;
        let pt: *mut PageTable = MemoryDef::gpa_to_hva(self.GetRoot()) as *mut PageTable;

        unsafe {
            let mut p4Idx = VirtAddr::new(curAddr.0).p4_index();
            let mut p3Idx = VirtAddr::new(curAddr.0).p3_index();

            while curAddr.0 < end.0 {
                let pgdEntry = &mut (*pt)[p4Idx];
                let pudTbl: *mut PageTable;

                if pgdEntry.is_unused() {
                    pudTbl = pagePool.AllocPage(true)? as *mut PageTable;
                    pgdEntry.set_addr(
                        PhysAddr::new(MemoryDef::hva_to_gpa(pudTbl as u64)),
                        default_table_user(),
                    );
                } else {
                    pudTbl = MemoryDef::gpa_to_hva(pgdEntry.addr().as_u64()) as *mut PageTable;
                }

                while curAddr.0 < end.0 {
                    let pudEntry = &mut (*pudTbl)[p3Idx];
                    let newphysAddr = curAddr.0 - start.0 + physical.0;

                    // Question: if we also do this for kernel, do we still need this?
                    if !pudEntry.is_unused() {
                        res = self.freeEntry(pudEntry, pagePool)?;
                    }
                    pudEntry.set_addr(
                        PhysAddr::new(newphysAddr),
                        flags | PageTableFlags::HUGE_PAGE,
                    );
                    curAddr = curAddr.AddLen(MemoryDef::HUGE_PAGE_SIZE_1G)?;

                    if p3Idx == PageTableIndex::new(MemoryDef::ENTRY_COUNT - 1) {
                        p3Idx = PageTableIndex::new(0);
                        break;
                    } else {
                        p3Idx = PageTableIndex::new(u16::from(p3Idx) + 1);
                    }
                }

                p4Idx = PageTableIndex::new(u16::from(p4Idx) + 1);
            }
        }
        return Ok(res);
    }

    //return true when there is previous mapping in the range
    pub fn Map(
        &self,
        start: Addr,
        end: Addr,
        physical: Addr,
        flags: PageTableFlags,
        pagePool: &Allocator,
        kernel: bool,
    ) -> Result<bool> {
        start.PageAligned()?;
        end.PageAligned()?;
        if end.0 < start.0 {
            return Err(Error::AddressNotInRange);
        }

        if start.0 < MemoryDef::LOWER_TOP {
            if end.0 <= MemoryDef::LOWER_TOP {
                return self.mapCanonical(start, end, physical, flags, pagePool, kernel);
            } else if end.0 > MemoryDef::LOWER_TOP && end.0 <= MemoryDef::UPPER_BOTTOM {
                return self.mapCanonical(
                    start,
                    Addr(MemoryDef::LOWER_TOP),
                    physical,
                    flags,
                    pagePool,
                    kernel,
                );
            } else {
                return self.mapCanonical(
                    start,
                    Addr(MemoryDef::LOWER_TOP),
                    physical,
                    flags,
                    pagePool,
                    kernel,
                );

                //todo: check the physical address
                //self.mapCanonical(UPPER_BOTTOM, end, physical, opts)
            }
        } else if start.0 < MemoryDef::UPPER_BOTTOM {
            if end.0 > MemoryDef::UPPER_BOTTOM {
                return self.mapCanonical(
                    Addr(MemoryDef::UPPER_BOTTOM),
                    end,
                    physical,
                    flags,
                    pagePool,
                    kernel,
                );
            }
        } else {
            return self.mapCanonical(start, end, physical, flags, pagePool, kernel);
        }

        return Ok(false);
    }

    pub fn UnmapNext(start: u64, size: u64) -> u64 {
        let mut start = start;
        start &= !(size - 1);
        start += size;
        return start;
    }

    pub fn UnusedEntryCount(tbl: *const PageTable) -> usize {
        let mut count = 0;
        unsafe {
            for idx in 0..MemoryDef::ENTRY_COUNT as usize {
                let entry: &PageTableEntry = &(*tbl)[PageTableIndex::new(idx as u16)];
                if entry.is_unused() {
                    count += 1;
                }
            }

            return count;
        }
    }

    pub fn Unmap(&self, start: u64, end: u64, pagePool: &Allocator) -> Result<()> {
        #[cfg(feature = "cc")]
        if ENABLE_CC.load(Ordering::Acquire) {
            match self.Unmap_cc(start, end, pagePool) {
                Err(Error::InvalidInput) => (),
                r => return r,
            }
        }
        Addr(start).PageAligned()?;
        Addr(end).PageAligned()?;
        let mut start = start;
        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;
        unsafe {
            let mut p4Idx: u16 = VirtAddr::new(start).p4_index().into();
            while start < end && p4Idx < MemoryDef::ENTRY_COUNT {
                let pgdEntry: &mut PageTableEntry = &mut (*pt)[PageTableIndex::new(p4Idx)];
                if pgdEntry.is_unused() {
                    start = Self::UnmapNext(start, MemoryDef::PGD_SIZE);
                    p4Idx += 1;
                    continue;
                }

                let pudTbl = pgdEntry.addr().as_u64() as *mut PageTable;
                let unusedPUDEntryCount = Self::UnusedEntryCount(pudTbl);
                let mut clearPUDEntries = 0;

                let mut p3Idx: u16 = VirtAddr::new(start).p3_index().into();
                while p3Idx < MemoryDef::ENTRY_COUNT && start < end {
                    let pudEntry: &mut PageTableEntry = &mut (*pudTbl)[PageTableIndex::new(p3Idx)];
                    if pudEntry.is_unused() {
                        start = Self::UnmapNext(start, MemoryDef::PUD_SIZE);
                        p3Idx += 1;
                        continue;
                    }

                    let pmdTbl = pudEntry.addr().as_u64() as *mut PageTable;
                    let mut clearPMDEntries = 0;
                    let mut p2Idx: u16 = VirtAddr::new(start).p2_index().into();
                    let unusedPMDEntryCount = Self::UnusedEntryCount(pmdTbl);

                    while p2Idx < MemoryDef::ENTRY_COUNT && start < end {
                        let pmdEntry: &mut PageTableEntry =
                            &mut (*pmdTbl)[PageTableIndex::new(p2Idx)];
                        if pmdEntry.is_unused() {
                            start = Self::UnmapNext(start, MemoryDef::PMD_SIZE);
                            p2Idx += 1;
                            continue;
                        }

                        let pteTbl = pmdEntry.addr().as_u64() as *mut PageTable;
                        let mut clearPTEEntries = 0;
                        let mut p1Idx: u16 = VirtAddr::new(start).p1_index().into();

                        let unusedPTEEntryCount = Self::UnusedEntryCount(pteTbl);
                        while p1Idx < MemoryDef::ENTRY_COUNT && start < end {
                            let pteEntry: &mut PageTableEntry =
                                &mut (*pteTbl)[PageTableIndex::new(p1Idx)];
                            if pteEntry.is_unused() {
                                start += MemoryDef::PAGE_SIZE;
                                p1Idx += 1;
                                continue;
                            }

                            clearPTEEntries += 1;
                            match self.freeEntry(pteEntry, pagePool) {
                                Err(_e) => {
                                    //info!("pagetable::Unmap Error: paddr {:x}, vaddr is {:x}, error is {:x?}",
                                    //    pteEntry.addr().as_u64(), start, e);
                                }
                                Ok(_) => (),
                            }

                            Invlpg(start);
                            start += MemoryDef::PAGE_SIZE;
                            p1Idx += 1;
                        }

                        if clearPTEEntries + unusedPTEEntryCount == MemoryDef::ENTRY_COUNT as usize
                        {
                            let currAddr = pmdEntry.addr().as_u64();
                            let refCnt = pagePool.Deref(currAddr)?;
                            if refCnt == 0 {
                                self.FreePage(currAddr);
                            }
                            pmdEntry.set_unused();
                            clearPMDEntries += 1;
                            //info!("unmap pmdEntry {:x}", currAddr);
                        }

                        p2Idx += 1;
                    }

                    if clearPMDEntries + unusedPMDEntryCount == MemoryDef::ENTRY_COUNT as usize {
                        let currAddr = pudEntry.addr().as_u64();
                        let refCnt = pagePool.Deref(currAddr)?;
                        if refCnt == 0 {
                            self.FreePage(currAddr);
                        }
                        pudEntry.set_unused();
                        clearPUDEntries += 1;

                        //info!("unmap pudEntry {:x}", currAddr);
                    }

                    p3Idx += 1;
                }

                if clearPUDEntries + unusedPUDEntryCount == MemoryDef::ENTRY_COUNT as usize {
                    let currAddr = pgdEntry.addr().as_u64();
                    let refCnt = pagePool.Deref(currAddr)?;
                    if refCnt == 0 {
                        self.FreePage(currAddr);
                    }
                    pgdEntry.set_unused();
                    //info!("unmap pgdEntry {:x}", currAddr);
                }

                p4Idx += 1;
            }
        }

        return Ok(());
    }

    pub fn ToVirtualAddr(
        p4Idx: PageTableIndex,
        p3Idx: PageTableIndex,
        p2Idx: PageTableIndex,
        p1Idx: PageTableIndex,
    ) -> Addr {
        let p4Idx = u64::from(p4Idx);
        let p3Idx = u64::from(p3Idx);
        let p2Idx = u64::from(p2Idx);
        let p1Idx = u64::from(p1Idx);
        let addr = (p4Idx << MemoryDef::PGD_SHIFT)
            + (p3Idx << MemoryDef::PUD_SHIFT)
            + (p2Idx << MemoryDef::PMD_SHIFT)
            + (p1Idx << MemoryDef::PTE_SHIFT);
        return Addr(addr);
    }

    pub fn GetAllPagetablePages(&self, pages: &mut BTreeSet<u64>) -> Result<()> {
        assert!(IS_GUEST == false, "GetAllPagetablePages");
        self.GetAllPagetablePagesWithRange(
            Addr(MemoryDef::PAGE_SIZE),
            Addr(MemoryDef::phy_lower_hva()),
            pages,
        )?;

        self.GetAllPagetablePagesWithRange(
            Addr(MemoryDef::phy_upper_hva()),
            Addr(MemoryDef::LOWER_TOP),
            pages,
        )?;

        return Ok(());
    }

    pub fn GetAllPagetablePagesWithRange(
        &self,
        start: Addr,
        end: Addr,
        pages: &mut BTreeSet<u64>,
    ) -> Result<()> {
        #[cfg(feature = "cc")]
        if ENABLE_CC.load(Ordering::Acquire) {
            match self.GetAllPagetablePagesWithRange_cc(start, end, pages) {
                Err(Error::InvalidInput) => (),
                r => return r,
            }
        }
        //let mut curAddr = start;
        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;
        unsafe {
            let mut p4Idx = VirtAddr::new(start.0).p4_index();
            let mut p3Idx = VirtAddr::new(start.0).p3_index();
            let mut p2Idx = VirtAddr::new(start.0).p2_index();
            let mut p1Idx = VirtAddr::new(start.0).p1_index();

            //pages.insert(self.GetRoot());
            //error!("l1 page {:x}", self.GetRoot());
            while Self::ToVirtualAddr(p4Idx, p3Idx, p2Idx, p1Idx).0 < end.0 {
                let pgdEntry = &mut (*pt)[p4Idx];
                let pudTbl: *mut PageTable;

                if pgdEntry.is_unused() {
                    p4Idx = PageTableIndex::new(u16::from(p4Idx) + 1);
                    p3Idx = PageTableIndex::new(0);
                    p2Idx = PageTableIndex::new(0);
                    p1Idx = PageTableIndex::new(0);

                    continue;
                } else {
                    //error!("l2 page {:x}", pgdEntry.addr().as_u64());
                    //pages.insert(pgdEntry.addr().as_u64());
                    pudTbl = pgdEntry.addr().as_u64() as *mut PageTable;
                }

                while Self::ToVirtualAddr(p4Idx, p3Idx, p2Idx, p1Idx).0 < end.0 {
                    let pudEntry = &mut (*pudTbl)[p3Idx];
                    let pmdTbl: *mut PageTable;
                    if pudEntry.is_unused() {
                        if p3Idx == PageTableIndex::new(MemoryDef::ENTRY_COUNT - 1) {
                            p3Idx = PageTableIndex::new(0);
                            break;
                        } else {
                            p3Idx = PageTableIndex::new(u16::from(p3Idx) + 1);
                        }

                        p2Idx = PageTableIndex::new(0);
                        p1Idx = PageTableIndex::new(0);

                        continue;
                    } else {
                        //error!("l3 page {:x}", pudEntry.addr().as_u64());
                        pages.insert(pudEntry.addr().as_u64());
                        pmdTbl = pudEntry.addr().as_u64() as *mut PageTable;
                    }

                    while Self::ToVirtualAddr(p4Idx, p3Idx, p2Idx, p1Idx).0 < end.0 {
                        let pmdEntry = &mut (*pmdTbl)[p2Idx];

                        if pmdEntry.is_unused() {
                            if p2Idx == PageTableIndex::new(MemoryDef::ENTRY_COUNT - 1) {
                                p2Idx = PageTableIndex::new(0);
                                break;
                            } else {
                                p2Idx = PageTableIndex::new(u16::from(p2Idx) + 1);
                            }

                            p1Idx = PageTableIndex::new(0);
                            continue;
                        } else {
                            //error!("l4 page {:x}", pmdEntry.addr().as_u64());
                            // add l4 pagetable page address
                            pages.insert(pmdEntry.addr().as_u64());
                        }

                        if p2Idx == PageTableIndex::new(MemoryDef::ENTRY_COUNT - 1) {
                            p2Idx = PageTableIndex::new(0);
                            break;
                        } else {
                            p2Idx = PageTableIndex::new(u16::from(p2Idx) + 1);
                        }
                    }

                    if p3Idx == PageTableIndex::new(MemoryDef::ENTRY_COUNT - 1) {
                        p3Idx = PageTableIndex::new(0);
                        break;
                    } else {
                        p3Idx = PageTableIndex::new(u16::from(p3Idx) + 1);
                    }
                }

                p4Idx = PageTableIndex::new(u16::from(p4Idx) + 1);
            }
        }

        return Ok(());
    }

    pub fn Traverse(
        &self,
        start: Addr,
        end: Addr,
        mut f: impl FnMut(&mut PageTableEntry, u64),
        failFast: bool,
    ) -> Result<()> {
        #[cfg(feature = "cc")]
        if ENABLE_CC.load(Ordering::Acquire) {
            match self.Traverse_cc(start, end, &mut f, failFast) {
                Err(Error::InvalidInput) => (),
                r => return r,
            }
        }
        start.PageAligned()?;
        end.PageAligned()?;
        //let mut curAddr = start;
        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;
        unsafe {
            let mut p4Idx = VirtAddr::new(start.0).p4_index();
            let mut p3Idx = VirtAddr::new(start.0).p3_index();
            let mut p2Idx = VirtAddr::new(start.0).p2_index();
            let mut p1Idx = VirtAddr::new(start.0).p1_index();

            while Self::ToVirtualAddr(p4Idx, p3Idx, p2Idx, p1Idx).0 < end.0 {
                let pgdEntry = &mut (*pt)[p4Idx];
                let pudTbl: *mut PageTable;

                if pgdEntry.is_unused() {
                    if failFast {
                        return Err(Error::AddressNotMap(
                            Self::ToVirtualAddr(p4Idx, p3Idx, p2Idx, p1Idx).0,
                        ));
                    }

                    p4Idx = PageTableIndex::new(u16::from(p4Idx) + 1);
                    p3Idx = PageTableIndex::new(0);
                    p2Idx = PageTableIndex::new(0);
                    p1Idx = PageTableIndex::new(0);

                    continue;
                } else {
                    pudTbl = pgdEntry.addr().as_u64() as *mut PageTable;
                }

                while Self::ToVirtualAddr(p4Idx, p3Idx, p2Idx, p1Idx).0 < end.0 {
                    let pudEntry = &mut (*pudTbl)[p3Idx];
                    let pmdTbl: *mut PageTable;
                    if pudEntry.is_unused() {
                        if failFast {
                            return Err(Error::AddressNotMap(
                                Self::ToVirtualAddr(p4Idx, p3Idx, p2Idx, p1Idx).0,
                            ));
                        }

                        if p3Idx == PageTableIndex::new(MemoryDef::ENTRY_COUNT - 1) {
                            p3Idx = PageTableIndex::new(0);
                            break;
                        } else {
                            p3Idx = PageTableIndex::new(u16::from(p3Idx) + 1);
                        }

                        p2Idx = PageTableIndex::new(0);
                        p1Idx = PageTableIndex::new(0);

                        continue;
                    } else {
                        pmdTbl = pudEntry.addr().as_u64() as *mut PageTable;
                    }

                    while Self::ToVirtualAddr(p4Idx, p3Idx, p2Idx, p1Idx).0 < end.0 {
                        let pmdEntry = &mut (*pmdTbl)[p2Idx];
                        let pteTbl: *mut PageTable;

                        if pmdEntry.is_unused() {
                            if failFast {
                                return Err(Error::AddressNotMap(
                                    Self::ToVirtualAddr(p4Idx, p3Idx, p2Idx, p1Idx).0,
                                ));
                            }

                            if p2Idx == PageTableIndex::new(MemoryDef::ENTRY_COUNT - 1) {
                                p2Idx = PageTableIndex::new(0);
                                break;
                            } else {
                                p2Idx = PageTableIndex::new(u16::from(p2Idx) + 1);
                            }

                            p1Idx = PageTableIndex::new(0);
                            continue;
                        } else {
                            pteTbl = pmdEntry.addr().as_u64() as *mut PageTable;
                        }

                        while Self::ToVirtualAddr(p4Idx, p3Idx, p2Idx, p1Idx).0 < end.0 {
                            let pteEntry = &mut (*pteTbl)[p1Idx];

                            if pteEntry.is_unused() {
                                if failFast {
                                    return Err(Error::AddressNotMap(
                                        Self::ToVirtualAddr(p4Idx, p3Idx, p2Idx, p1Idx).0,
                                    ));
                                }
                            } else {
                                f(pteEntry, Self::ToVirtualAddr(p4Idx, p3Idx, p2Idx, p1Idx).0);
                            }

                            if p1Idx == PageTableIndex::new(MemoryDef::ENTRY_COUNT - 1) {
                                p1Idx = PageTableIndex::new(0);
                                break;
                            } else {
                                p1Idx = PageTableIndex::new(u16::from(p1Idx) + 1);
                            }
                        }

                        if p2Idx == PageTableIndex::new(MemoryDef::ENTRY_COUNT - 1) {
                            p2Idx = PageTableIndex::new(0);
                            break;
                        } else {
                            p2Idx = PageTableIndex::new(u16::from(p2Idx) + 1);
                        }
                    }

                    if p3Idx == PageTableIndex::new(MemoryDef::ENTRY_COUNT - 1) {
                        p3Idx = PageTableIndex::new(0);
                        break;
                    } else {
                        p3Idx = PageTableIndex::new(u16::from(p3Idx) + 1);
                    }
                }

                p4Idx = PageTableIndex::new(u16::from(p4Idx) + 1);
            }
        }

        return Ok(());
    }

    pub fn SetPageFlags(&self, addr: Addr, flags: PageTableFlags) {
        //self.MProtect(addr, addr.AddLen(4096).unwrap(), PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE, false).unwrap();
        self.MProtect(
            addr,
            addr.AddLen(MemoryDef::PAGE_SIZE).unwrap(),
            flags,
            true,
        )
        .unwrap();
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

    #[cfg(target_arch = "x86_64")]
    pub fn SwapOutPages(
        &self,
        start: u64,
        len: u64,
        pages: &mut BTreeSet<u64>,
        updatePageEntry: bool,
    ) -> Result<()> {
        assert!(IS_GUEST == false);
        let end = start + len;
        self.Traverse(
            Addr(MemoryDef::PAGE_SIZE),
            Addr(MemoryDef::phy_lower_hva()),
            |entry: &mut PageTableEntry, _virtualAddr| {
                let phyAddr = entry.addr().as_u64();
                if start <= phyAddr && phyAddr < end {
                    let mut flags = entry.flags();
                    let needInsert = flags & PageTableFlags::BIT_9 != PageTableFlags::BIT_9;
                    if updatePageEntry && needInsert {
                        //error!("SwapOutPages 1 {:x?}/{:x}/{:x}/{:x}/{:x}", self.root, phyAddr, _virtualAddr, start, end);
                        flags &= !PageTableFlags::PRESENT;
                        // flags bit9 which indicate the page is swapped out
                        flags |= PageTableFlags::BIT_9;
                        entry.set_flags(flags);
                    }

                    if needInsert {
                        pages.insert(phyAddr);
                    }
                }
            },
            false,
        )?;

        return self.Traverse(
            Addr(MemoryDef::phy_upper_hva()),
            Addr(MemoryDef::LOWER_TOP),
            |entry, _virtualAddr| {
                let phyAddr = entry.addr().as_u64();
                if start <= phyAddr && phyAddr < end {
                    let mut flags = entry.flags();
                    let needInsert = flags & PageTableFlags::BIT_9 != PageTableFlags::BIT_9;
                    if updatePageEntry && needInsert {
                        //error!("SwapOutPages 1 {:x?}/{:x}/{:x}/{:x}/{:x}", self.root, phyAddr, _virtualAddr, start, end);
                        flags &= !PageTableFlags::PRESENT;
                        // flags bit9 which indicate the page is swapped out
                        flags |= PageTableFlags::BIT_9;
                        entry.set_flags(flags);
                    }

                    if needInsert {
                        pages.insert(phyAddr);
                    }
                }
            },
            false,
        );
    }

    #[cfg(target_arch = "aarch64")]
    pub fn SwapOutPages(
        &self,
        start: u64,
        len: u64,
        pages: &mut BTreeSet<u64>,
        updatePageEntry: bool,
    ) -> Result<()> {
        Ok(())
    }

    // ret: >0: the swapped out page addr, 0: the page is missing
    pub fn SwapInPage(&self, vaddr: Addr) -> Result<u64> {
        #[cfg(feature = "cc")]
        if ENABLE_CC.load(Ordering::Acquire) {
            match self.SwapInPage_cc(vaddr) {
                Err(Error::InvalidInput) => (),
                r => return r,
            }
        }
        let vaddr = Addr(vaddr.0 & !(PAGE_SIZE - 1));
        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;
        unsafe {
            let p4Idx = VirtAddr::new(vaddr.0).p4_index();
            let p3Idx = VirtAddr::new(vaddr.0).p3_index();
            let p2Idx = VirtAddr::new(vaddr.0).p2_index();
            let p1Idx = VirtAddr::new(vaddr.0).p1_index();

            let pgdEntry = &mut (*pt)[p4Idx];
            let pudTbl: *mut PageTable;

            if pgdEntry.is_unused() {
                return Ok(0);
            } else {
                pudTbl = pgdEntry.addr().as_u64() as *mut PageTable;
            }

            let pudEntry = &mut (*pudTbl)[p3Idx];
            let pmdTbl: *mut PageTable;

            if pudEntry.is_unused() {
                return Ok(0);
            } else {
                pmdTbl = pudEntry.addr().as_u64() as *mut PageTable;
            }

            let pmdEntry = &mut (*pmdTbl)[p2Idx];
            let pteTbl: *mut PageTable;

            if pmdEntry.is_unused() {
                return Ok(0);
            } else {
                pteTbl = pmdEntry.addr().as_u64() as *mut PageTable;
            }

            let pteEntry = &mut (*pteTbl)[p1Idx];

            if pteEntry.is_unused() {
                return Ok(0);
            }

            /*let mut flags = pteEntry.flags();
            if flags & PageTableFlags::BIT_9 == PageTableFlags::BIT_9 {
                flags |= PageTableFlags::PRESENT;
                // flags bit9 which indicate the page is swapped out
                flags &= !PageTableFlags::BIT_9;
                pteEntry.set_flags(flags);
            } */

            self.HandlingSwapInPage(vaddr.0, pteEntry);

            // the page might be swapping in by another vcpu
            let addr = pteEntry.addr().as_u64();
            return Ok(addr);
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub fn HandlingSwapInPage(&self, vaddr: u64, pteEntry: &mut PageTableEntry) {
        let cc_enabled = SHARESPACE.config.read().EnableCC;
        if cc_enabled {
            return;
        }

        let flags = pteEntry.flags();
        // bit9 : whether the page is swapout
        // bit10: whether there is thread is working on swapin the page

        if flags & PageTableFlags::BIT_9 == PageTableFlags::BIT_9 {
            let needSwapin = {
                let _l = crate::GLOBAL_LOCK.lock();

                let mut flags = pteEntry.flags();

                // the page has been swaped in
                if flags & PageTableFlags::BIT_9 != PageTableFlags::BIT_9 {
                    return;
                }

                // is there another thread doing swapin?
                if flags & PageTableFlags::BIT_10 == PageTableFlags::BIT_10 {
                    // another thread is swapping in
                    false
                } else {
                    flags |= PageTableFlags::BIT_10;
                    true
                }
            };

            if needSwapin {
                let addr = pteEntry.addr().as_u64();
                let _ret = HostSpace::SwapInPage(addr);

                let mut flags = pteEntry.flags();
                flags |= PageTableFlags::PRESENT;
                // flags bit9 which indicate the page is swapped out
                flags &= !PageTableFlags::BIT_9;
                flags &= !PageTableFlags::BIT_10;
                pteEntry.set_flags(flags);
                Invlpg(vaddr);
                fence(Ordering::SeqCst);
            } else {
                loop {
                    let flags = pteEntry.flags();

                    // wait bit9 is cleared
                    if flags & PageTableFlags::BIT_9 != PageTableFlags::BIT_9 {
                        return;
                    }

                    spin_loop();
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn HandlingSwapInPage(&self, vaddr: u64, pteEntry: &mut PageTableEntry) {}

    pub fn MProtect(
        &self,
        start: Addr,
        end: Addr,
        flags: PageTableFlags,
        failFast: bool,
    ) -> Result<()> {
        defer!(self.EnableTlbShootdown());
        return self.Traverse(
            start,
            end,
            |entry, virtualAddr| {
                self.HandlingSwapInPage(virtualAddr, entry);
                entry.set_flags(flags);
                Invlpg(virtualAddr);
            },
            failFast,
        );
    }

    pub fn freeEntry(&self, entry: &mut PageTableEntry, pagePool: &Allocator) -> Result<bool> {
        #[cfg(feature = "cc")]
        if ENABLE_CC.load(Ordering::Acquire) {
            match self.freeEntry_cc(entry, pagePool) {
                Err(Error::InvalidInput) => (),
                r => return r,
            }
        }
        let currAddr = entry.addr().as_u64();
        let refCnt = pagePool.Deref(currAddr)?;
        if refCnt == 0 {
            self.FreePage(currAddr);
        }
        entry.set_unused();
        self.EnableTlbShootdown();
        return Ok(true);
    }

    // if kernel == true, don't need to reference in the pagePool
    fn mapCanonical(
        &self,
        start: Addr,
        end: Addr,
        phyAddr: Addr,
        flags: PageTableFlags,
        pagePool: &Allocator,
        kernel: bool,
    ) -> Result<bool> {
        #[cfg(feature = "cc")]
        if ENABLE_CC.load(Ordering::Acquire) {
            match self.mapCanonical_cc(start, end, phyAddr, flags, pagePool, kernel) {
                Err(Error::InvalidInput) => (),
                r => return r,
            }
        }
        let mut res = false;

        //info!("mapCanonical virtual start is {:x}, len is {:x}, phystart is {:x}", start.0, end.0 - start.0, phyAddr.0);
        let mut curAddr = start;
        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;
        unsafe {
            let mut p4Idx = VirtAddr::new(curAddr.0).p4_index();
            let mut p3Idx = VirtAddr::new(curAddr.0).p3_index();
            let mut p2Idx = VirtAddr::new(curAddr.0).p2_index();
            let mut p1Idx = VirtAddr::new(curAddr.0).p1_index();

            while curAddr.0 < end.0 {
                let pgdEntry = &mut (*pt)[p4Idx];
                let pudTbl: *mut PageTable;

                if pgdEntry.is_unused() {
                    pudTbl = pagePool.AllocPage(true)? as *mut PageTable;
                    pgdEntry.set_addr(PhysAddr::new(pudTbl as u64), default_table_user());
                } else {
                    pudTbl = pgdEntry.addr().as_u64() as *mut PageTable;
                }

                while curAddr.0 < end.0 {
                    let pudEntry = &mut (*pudTbl)[p3Idx];
                    let pmdTbl: *mut PageTable;

                    if pudEntry.is_unused() {
                        pmdTbl = pagePool.AllocPage(true)? as *mut PageTable;
                        pudEntry.set_addr(PhysAddr::new(pmdTbl as u64), default_table_user());
                    } else {
                        pmdTbl = pudEntry.addr().as_u64() as *mut PageTable;
                    }

                    while curAddr.0 < end.0 {
                        let pmdEntry = &mut (*pmdTbl)[p2Idx];
                        let pteTbl: *mut PageTable;

                        if pmdEntry.is_unused() {
                            pteTbl = pagePool.AllocPage(true)? as *mut PageTable;
                            pmdEntry.set_addr(PhysAddr::new(pteTbl as u64), default_table_user());
                        } else {
                            pteTbl = pmdEntry.addr().as_u64() as *mut PageTable;
                        }

                        while curAddr.0 < end.0 {
                            let pteEntry = &mut (*pteTbl)[p1Idx];
                            let newAddr = curAddr.0 - start.0 + phyAddr.0;

                            if !kernel {
                                pagePool.Ref(newAddr)?;
                            }

                            if !pteEntry.is_unused() {
                                /*let bit9 = pteEntry.flags() & PageTableFlags::BIT_9 == PageTableFlags::BIT_9;

                                if !bit9 {
                                    res = true;
                                    let currAddr = pteEntry.addr().as_u64();
                                    pagePool.Deref(currAddr)?;
                                    pteEntry.set_flags(PageTableFlags::PRESENT | PageTableFlags::BIT_9);
                                }*/
                                res = self.freeEntry(pteEntry, pagePool)?;
                            }

                            //info!("set addr: vaddr is {:x}, paddr is {:x}, flags is {:b}", curAddr.0, phyAddr.0, flags.bits());
                            pteEntry.set_addr(PhysAddr::new(newAddr), flags);
                            Invlpg(curAddr.0);
                            curAddr = curAddr.AddLen(MemoryDef::PAGE_SIZE_4K)?;

                            if p1Idx == PageTableIndex::new(MemoryDef::ENTRY_COUNT - 1) {
                                p1Idx = PageTableIndex::new(0);
                                break;
                            } else {
                                p1Idx = PageTableIndex::new(u16::from(p1Idx) + 1);
                            }
                        }

                        if p2Idx == PageTableIndex::new(MemoryDef::ENTRY_COUNT - 1) {
                            p2Idx = PageTableIndex::new(0);
                            break;
                        } else {
                            p2Idx = PageTableIndex::new(u16::from(p2Idx) + 1);
                        }
                    }

                    if p3Idx == PageTableIndex::new(MemoryDef::ENTRY_COUNT - 1) {
                        p3Idx = PageTableIndex::new(0);
                        break;
                    } else {
                        p3Idx = PageTableIndex::new(u16::from(p3Idx) + 1);
                    }
                }

                p4Idx = PageTableIndex::new(u16::from(p4Idx) + 1);
            }
        }

        return Ok(res);
    }
}

pub struct PageBufAllocator {
    pub buf: Vec<u64>,
    pub allocator: AlignedAllocator,
}

impl PageBufAllocator {
    pub fn New() -> Self {
        return Self {
            buf: Vec::with_capacity(4096 * 16),
            allocator: AlignedAllocator::New(
                MemoryDef::PAGE_SIZE as usize,
                MemoryDef::PAGE_SIZE as usize,
            ),
        };
    }

    pub fn Allocate(&mut self) -> Result<u64> {
        match self.buf.pop() {
            None => return self.allocator.Allocate(),
            Some(addr) => return Ok(addr),
        }
    }

    pub fn Free(&mut self, addr: u64) -> Result<()> {
        if self.buf.len() < 4096 * 16 {
            self.buf.push(addr);
            return Ok(());
        }

        return self.allocator.Free(addr);
    }
}

#[derive(Debug)]
pub struct AlignedAllocator {
    pub size: usize,
    pub align: usize,
}

impl AlignedAllocator {
    pub fn New(size: usize, align: usize) -> Self {
        return Self {
            size: size,
            align: align,
        };
    }

    // allocate memory from guest private allocator
    pub fn Allocate(&self) -> Result<u64> {
        let layout = Layout::from_size_align(self.size, self.align);
        match layout {
            Err(_e) => Err(Error::UnallignedAddress(format!("Allocate {:?}", self))),
            Ok(l) => unsafe {
                let is_on_host = GLOBAL_ALLOCATOR.is_host_allocator.load(Ordering::SeqCst);

                let addr = if is_on_host {
                    GLOBAL_ALLOCATOR.AllocGuestPrivatMem(self.size, self.align)
                } else {
                    alloc(l)
                };
                Ok(addr as u64)
            },
        }
    }

    pub fn Free(&self, addr: u64) -> Result<()> {
        let layout = Layout::from_size_align(self.size, self.align);
        match layout {
            Err(_e) => Err(Error::UnallignedAddress(format!("Allocate {:?}", self))),
            Ok(l) => unsafe {
                dealloc(addr as *mut u8, l);
                Ok(())
            },
        }
    }
}

#[cfg(test1)]
mod tests {
    use super::super::buddyallocator::*;
    use super::*;
    use alloc::vec::Vec;

    #[repr(align(4096))]
    #[derive(Clone)]
    struct Page {
        data: [u64; 512],
    }

    impl Default for Page {
        fn default() -> Self {
            return Page { data: [0; 512] };
        }
    }

    #[test]
    fn test_MapPage() {
        let mem: Vec<Page> = vec![Default::default(); 256]; //256 Pages
        let mut allocator = MemAllocator::Init(&mem[0] as *const _ as u64, 8); //2^8
        let mut pt = PageTables::New(&Allocator).unwrap();

        let phyAddr = 6 * 4096;
        pt.MapPage(
            Addr(0),
            Addr(phyAddr),
            PageOpts::UserReadOnly().Val(),
            &Allocator,
        )
        .unwrap();

        let res = pt.VirtualToPhy(0).unwrap();
        assert_eq!(res, phyAddr);
    }

    #[test]
    fn test_Map() {
        let mem: Vec<Page> = vec![Default::default(); 256]; //256 Pages
        let mut allocator = MemAllocator::Init(&mem[0] as *const _ as u64, 8); //2^8
        let mut pt = PageTables::New(&Allocator).unwrap();

        pt.Map(
            Addr(4096 * 500),
            Addr(4096 * 600),
            Addr(4096 * 550),
            PageOpts::UserReadOnly().Val(),
            &Allocator,
        )
        .unwrap();

        for i in 500..600 {
            let vAddr = i * 4096;
            let pAddr = pt.VirtualToPhy(vAddr).unwrap();
            assert_eq!(vAddr + 50 * 4096, pAddr);
        }
    }

    #[test]
    fn test_KernelPage() {
        let mem: Vec<Page> = vec![Default::default(); 1024]; //256 Pages
        let mut allocator = MemAllocator::Init(&mem[0] as *const _ as u64, 10); //2^10
        let mut pt = PageTables::New(&Allocator).unwrap();

        pt.InitKernel(&Allocator).unwrap();
        let nPt = pt.NewWithKernelPageTables(&Allocator).unwrap();

        assert_eq!(pt.VirtualToPhy(0).unwrap(), nPt.VirtualToPhy(0).unwrap());
    }

    #[test]
    fn test_CopyRange() {
        let mem: Vec<Page> = vec![Default::default(); 256]; //256 Pages
        let mut allocator = MemAllocator::Init(&mem[0] as *const _ as u64, 8); //2^8
        let mut pt = PageTables::New(&Allocator).unwrap();

        pt.Map(
            Addr(4096 * 500),
            Addr(4096 * 600),
            Addr(4096 * 550),
            PageOpts::UserReadOnly().Val(),
            &Allocator,
        )
        .unwrap();

        let mut nPt = PageTables::New(&Allocator).unwrap();

        pt.CopyRange(&mut nPt, 4096 * 500, 4096 * 100, &Allocator)
            .unwrap();
        for i in 500..600 {
            let vAddr = i * 4096;
            let pAddr = nPt.VirtualToPhy(vAddr).unwrap();
            assert_eq!(vAddr + 50 * 4096, pAddr);
        }
    }

    #[test]
    fn test_ForkRange() {
        let mem: Vec<Page> = vec![Default::default(); 256]; //256 Pages
        let mut allocator = MemAllocator::Init(&mem[0] as *const _ as u64, 8); //2^8
        let mut pt = PageTables::New(&Allocator).unwrap();

        pt.Map(
            Addr(4096 * 400),
            Addr(4096 * 600),
            Addr(4096 * 450),
            PageOpts::UserReadWrite().Val(),
            &Allocator,
        )
        .unwrap();

        let mut nPt = PageTables::New(&Allocator).unwrap();

        pt.ForkRange(&mut nPt, 4096 * 500, 4096 * 100, &Allocator)
            .unwrap();
        //pt.MProtect(Addr(4096 * 500), Addr(4096 * 600), PageOpts::UserReadOnly().Val(), true).unwrap();
        for i in 500..600 {
            let vAddr = i * 4096;
            let pAddr = nPt.VirtualToPhy(vAddr).unwrap();
            assert_eq!(vAddr + 50 * 4096, pAddr);

            assert_eq!(
                pt.VirtualToEntry(vAddr).unwrap().flags(),
                PageOpts::UserReadOnly().Val()
            );
            assert_eq!(
                nPt.VirtualToEntry(vAddr).unwrap().flags(),
                PageOpts::UserReadOnly().Val()
            );
        }

        for i in 400..500 {
            let vAddr = i * 4096;
            assert_eq!(
                pt.VirtualToEntry(vAddr).unwrap().flags(),
                PageOpts::UserReadWrite().Val()
            );
        }
    }
}
