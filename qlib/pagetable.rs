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

use alloc::sync::Arc;
use spin::RwLock;
use core::ops::Deref;
use x86_64::structures::paging::{PageTable};
use x86_64::structures::paging::page_table::PageTableEntry;
use x86_64::structures::paging::page_table::PageTableIndex;
use x86_64::PhysAddr;
use x86_64::VirtAddr;
use x86_64::structures::paging::PageTableFlags;
use alloc::alloc::{Layout, alloc, dealloc};
use alloc::vec::Vec;

use super::common::{Error, Result, Allocator};
use super::addr::{Addr, PageOpts};
use super::linux_def::*;
use super::mem::stackvec::*;
use super::super::asm::*;

#[derive(Clone, Default, Debug)]
pub struct PageTables(pub Arc<RwLock<PageTablesInternal>>);

impl Deref for PageTables {
    type Target = Arc<RwLock<PageTablesInternal>>;

    fn deref(&self) -> &Arc<RwLock<PageTablesInternal>> {
        &self.0
    }
}

impl PageTables {
    pub fn New(pagePool: &Allocator) -> Result<Self> {
        let internal = PageTablesInternal::New(pagePool)?;
        return Ok(PageTables(Arc::new(RwLock::new(internal))))
    }

    pub fn Init(root: u64) -> Self {
        return PageTables(Arc::new(RwLock::new(PageTablesInternal::Init(root))))
    }

    pub fn SwitchTo(&self) {
        let addr = self.read().root;
        Self::Switch(addr.0);
    }

    pub fn IsActivePagetable(&self) -> bool {
        let root = self.read().root.0;
        return root == Self::CurrentCr3()
    }

    pub fn CurrentCr3() -> u64 {
        return CurrentCr3()
    }

    //switch pagetable for the cpu, Cr3
    pub fn Switch(cr3: u64) {
        //unsafe { llvm_asm!("mov $0, %cr3" : : "r" (cr3) ) };
        LoadCr3(cr3)
    }
}

#[derive(Clone, Default, Debug)]
pub struct PageTablesInternal {
    //Root page guest physical address
    pub root: Addr,
}

impl PageTablesInternal {
    pub fn New(pagePool: &Allocator) -> Result<Self> {
        let root = pagePool.AllocPage(true)?;
        pagePool.Ref(root).unwrap();
        Ok(Self {
            //pagePool : pagePool.clone(),
            root: Addr(root),
        })
    }

    pub fn Init(root: u64) -> Self {
        return Self {
            root: Addr(root),
        }
    }

    pub fn SetRoot(&mut self, root: u64) {
        self.root = Addr(root);
    }

    pub fn GetRoot(&self) -> u64 {
        return self.root.0;
    }

    pub fn Print(&self) {
        //let cr3 : u64;
        //unsafe { llvm_asm!("mov %cr3, $0" : "=r" (cr3) ) };
        let cr3 = ReadCr3();

        info!("the page root is {:x}, the cr3 is {:x}", self.root.0, cr3);
    }

    pub fn CopyRange(&mut self, to: &mut Self, start: u64, len: u64, pagePool: &Allocator) -> Result<()> {
        if start & MemoryDef::PAGE_MASK != 0 || len & MemoryDef::PAGE_MASK != 0 {
            return Err(Error::UnallignedAddress);
        }

        let mut vAddr = start;
        while vAddr < start + len {
            match self.VirtualToEntry(vAddr) {
                Ok(entry) => {
                    let phyAddr = entry.addr().as_u64();
                    to.MapPage(Addr(vAddr), Addr(phyAddr), entry.flags(), pagePool)?;
                }
                Err(_) => ()
            }
            vAddr += MemoryDef::PAGE_SIZE;
        }

        Ok(())
    }

    // Copy the range and make the range readonly for from and to pagetable. It is used for VirtualArea private area.
    // The Copy On Write will be done when write to the page
    pub fn ForkRange(&mut self, to: &mut Self, start: u64, len: u64, pagePool: &Allocator) -> Result<()> {
        if start & MemoryDef::PAGE_MASK != 0 || len & MemoryDef::PAGE_MASK != 0 {
            return Err(Error::UnallignedAddress);
        }

        let mut vAddr = start;
        while vAddr < start + len {
            match self.VirtualToEntry(vAddr) {
                Ok(entry) => {
                    let phyAddr = entry.addr().as_u64();
                    to.MapPage(Addr(vAddr), Addr(phyAddr), PageOpts::UserReadOnly().Val(), pagePool)?;
                }
                Err(_) => ()
            }
            vAddr += MemoryDef::PAGE_SIZE;
        }

        //change to read only
        //todo: there is chance the orignal range is changed to readonly by mprotected before. Need to handle.
        let _ = self.MProtect(Addr(start), Addr(start + len), PageOpts::UserReadOnly().Val(), false);//there won't be any failure

        Ok(())
    }

    pub fn PrintPath(&self, vaddr: u64) {
        let vaddr = VirtAddr::new(vaddr);

        let p4Idx = vaddr.p4_index();
        let p3Idx = vaddr.p3_index();
        let p2Idx = vaddr.p2_index();
        let p1Idx = vaddr.p1_index();

        let pt: *mut PageTable = self.root.0 as *mut PageTable;

        unsafe {
            info!("pt1: {:x}", self.root.0);
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
        let addr = vaddr;
        let vaddr = VirtAddr::new(vaddr);

        let p4Idx = vaddr.p4_index();
        let p3Idx = vaddr.p3_index();
        let p2Idx = vaddr.p2_index();
        let p1Idx = vaddr.p1_index();

        let pt: *mut PageTable = self.root.0 as *mut PageTable;

        unsafe {
            let pgdEntry = &(*pt)[p4Idx];
            if pgdEntry.is_unused() {
                return Err(Error::AddressNotMap(addr))
            }

            let pudTbl = pgdEntry.addr().as_u64() as *const PageTable;
            let pudEntry = &(*pudTbl)[p3Idx];
            if pudEntry.is_unused() {
                return Err(Error::AddressNotMap(addr))
            }

            let pmdTbl = pudEntry.addr().as_u64() as *const PageTable;
            let pmdEntry = &(*pmdTbl)[p2Idx];
            if pmdEntry.is_unused() {
                return Err(Error::AddressNotMap(addr))
            }

            let pteTbl = pmdEntry.addr().as_u64() as *const PageTable;
            let pteEntry = &(*pteTbl)[p1Idx];
            if pteEntry.is_unused() {
                return Err(Error::AddressNotMap(addr))
            }
            return Ok(pteEntry);
        }
    }

    //return (phyaddress, writeable)
    pub fn VirtualToPhy(&self, vaddr: u64) -> Result<(u64, bool)> {
        let pteEntry = self.VirtualToEntry(vaddr)?;
        if pteEntry.is_unused() {
            return Err(Error::AddressNotMap(vaddr))
        }

        let vaddr = VirtAddr::new(vaddr);
        let pageAddr: u64 = vaddr.page_offset().into();
        let phyAddr = pteEntry.addr().as_u64() + pageAddr;
        let writeable = pteEntry.flags() & PageTableFlags::WRITABLE == PageTableFlags::WRITABLE;

        return Ok((phyAddr, writeable))
    }

    pub fn PrintPageFlags(&self, vaddr: u64) -> Result<()> {
        let pteEntry = self.VirtualToEntry(vaddr)?;
        if pteEntry.is_unused() {
            return Err(Error::AddressNotMap(vaddr))
        }

        info!("Flags is {:x?}", pteEntry);
        return Ok(())
    }

    pub fn MapVsyscall(&mut self, phyAddrs: &[u64]/*4 pages*/) {
        let vaddr = 0xffffffffff600000;
        let pt: *mut PageTable = self.root.0 as *mut PageTable;
        unsafe {
            let p4Idx = VirtAddr::new(vaddr).p4_index();
            let pgdEntry = &mut (*pt)[p4Idx];
            let pudTbl: *mut PageTable;

            assert!(pgdEntry.is_unused());
            pudTbl = phyAddrs[3] as *mut PageTable;
            pgdEntry.set_addr(PhysAddr::new(pudTbl as u64), PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE);
        }
    }

    pub fn MapPage(&mut self, vaddr: Addr, phyAddr: Addr, flags: PageTableFlags, pagePool: &Allocator) -> Result<bool> {
        let mut res = false;

        let pt: *mut PageTable = self.root.0 as *mut PageTable;
        unsafe {
            let p4Idx = VirtAddr::new(vaddr.0).p4_index();
            let p3Idx = VirtAddr::new(vaddr.0).p3_index();
            let p2Idx = VirtAddr::new(vaddr.0).p2_index();
            let p1Idx = VirtAddr::new(vaddr.0).p1_index();

            let pgdEntry = &mut (*pt)[p4Idx];
            let pudTbl: *mut PageTable;

            if pgdEntry.is_unused() {
                pudTbl = pagePool.AllocPage(true)? as *mut PageTable;
                pagePool.Ref(pudTbl as u64).unwrap();
                (*pudTbl).zero();
                pgdEntry.set_addr(PhysAddr::new(pudTbl as u64), PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE);
            } else {
                pudTbl = pgdEntry.addr().as_u64() as *mut PageTable;
            }

            let pudEntry = &mut (*pudTbl)[p3Idx];
            let pmdTbl: *mut PageTable;

            if pudEntry.is_unused() {
                pmdTbl = pagePool.AllocPage(true)? as *mut PageTable;
                pagePool.Ref(pmdTbl as u64).unwrap();
                (*pmdTbl).zero();
                pudEntry.set_addr(PhysAddr::new(pmdTbl as u64), PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE);
            } else {
                pmdTbl = pudEntry.addr().as_u64() as *mut PageTable;
            }

            let pmdEntry = &mut (*pmdTbl)[p2Idx];
            let pteTbl: *mut PageTable;

            if pmdEntry.is_unused() {
                pteTbl = pagePool.AllocPage(true)? as *mut PageTable;
                pagePool.Ref(pteTbl as u64).unwrap();
                (*pteTbl).zero();
                pmdEntry.set_addr(PhysAddr::new(pteTbl as u64), PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE);
            } else {
                pteTbl = pmdEntry.addr().as_u64() as *mut PageTable;
            }

            let pteEntry = &mut (*pteTbl)[p1Idx];

            pagePool.Ref(phyAddr.0).unwrap();
            if !pteEntry.is_unused() {
                Self::freeEntry(pteEntry, pagePool)?;

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

    pub fn Remap(&mut self, start: Addr, end: Addr, oldStart: Addr, flags: PageTableFlags, pagePool: &Allocator) -> Result<bool> {
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
                    self.Unmap(oldStart.0 + offset, oldStart.0 + offset + MemoryDef::PAGE_SIZE, pagePool)?;
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
                    self.MapPage(Addr(start.0+offset), Addr(phyAddr), flags, pagePool)?;
                    pagePool.Deref(phyAddr).unwrap();
                }
                None =>()
            }
            offset += MemoryDef::PAGE_SIZE;
            idx += 1;
        }

        return Ok(false)
    }

    pub fn RemapForFile(&mut self, start: Addr, end: Addr, physical: Addr, oldStart: Addr, oldEnd: Addr, flags: PageTableFlags, pagePool: &Allocator) -> Result<bool> {
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
                        self.MapPage(Addr(start.0+offset), Addr(phyAddr), flags, pagePool)?;
                        self.Unmap(oldStart.0 + offset, oldStart.0 + offset + MemoryDef::PAGE_SIZE, pagePool)?;
                        offset += MemoryDef::PAGE_SIZE;
                        continue 'a;
                    }
                    Err(_) =>()
                }
            }

            let targetPhyAddr = physical.0 + offset;
            self.MapPage(Addr(start.0 + offset), Addr(targetPhyAddr), flags, pagePool)?;

            offset += MemoryDef::PAGE_SIZE;
        }

        return Ok(false)
    }

    //return true when there is previous mapping in the range
    pub fn Map(&mut self, start: Addr, end: Addr, physical: Addr, flags: PageTableFlags, pagePool: &Allocator, kernel: bool) -> Result<bool> {
        start.PageAligned()?;
        if end.0 < start.0 {
            return Err(Error::AddressNotInRange);
        }

        if start.0 < MemoryDef::LOWER_TOP {
            if end.0 <= MemoryDef::LOWER_TOP {
                return self.mapCanonical(start, end, physical, flags, pagePool, kernel)
            } else if end.0 > MemoryDef::LOWER_TOP && end.0 <= MemoryDef::UPPER_BOTTOM {
                return self.mapCanonical(start, Addr(MemoryDef::LOWER_TOP), physical, flags, pagePool, kernel)
            } else {
                return self.mapCanonical(start, Addr(MemoryDef::LOWER_TOP), physical, flags, pagePool, kernel)

                //todo: check the physical address
                //self.mapCanonical(UPPER_BOTTOM, end, physical, opts)
            }
        } else if start.0 < MemoryDef::UPPER_BOTTOM {
            if end.0 > MemoryDef::UPPER_BOTTOM {
                return self.mapCanonical(Addr(MemoryDef::UPPER_BOTTOM), end, physical, flags, pagePool, kernel)
            }
        } else {
            return self.mapCanonical(start, end, physical, flags, pagePool, kernel)
        }

        return Ok(false);
    }

    pub fn UnmapNext(start: u64, size: u64) -> u64 {
        let mut start = start;
        start &= !(size - 1);
        start += size;
        return start;
    }

    pub fn Unmap(&mut self, start: u64, end: u64, pagePool: &Allocator) -> Result<()> {
        let mut start = start;
        let pt: *mut PageTable = self.root.0 as *mut PageTable;
        unsafe {
            let mut p4Idx : u16 = VirtAddr::new(start).p4_index().into();
            while start < end && p4Idx < MemoryDef::ENTRY_COUNT {
                let pgdEntry : &mut PageTableEntry = &mut (*pt)[PageTableIndex::new(p4Idx)];
                if pgdEntry.is_unused() {
                    start = Self::UnmapNext(start, MemoryDef::PGD_SIZE);
                    p4Idx += 1;
                    continue;
                }

                let pudTbl = pgdEntry.addr().as_u64() as *mut PageTable;
                let mut clearPUDEntries = 0;

                let mut p3Idx : u16 = VirtAddr::new(start).p3_index().into();
                while p3Idx < MemoryDef::ENTRY_COUNT && start < end {
                    let pudEntry : &mut PageTableEntry = &mut (*pudTbl)[PageTableIndex::new(p3Idx)];
                    if pudEntry.is_unused() {
                        clearPUDEntries += 1;
                        start = Self::UnmapNext(start, MemoryDef::PUD_SIZE);
                        p3Idx += 1;
                        continue;
                    }

                    let pmdTbl = pudEntry.addr().as_u64() as *mut PageTable;
                    let mut clearPMDEntries = 0;
                    let mut p2Idx : u16 = VirtAddr::new(start).p2_index().into();

                    while p2Idx < MemoryDef::ENTRY_COUNT && start < end {
                        let pmdEntry : &mut PageTableEntry = &mut (*pmdTbl)[PageTableIndex::new(p2Idx)];
                        if pmdEntry.is_unused() {
                            clearPMDEntries += 1;
                            start = Self::UnmapNext(start, MemoryDef::PMD_SIZE);
                            p2Idx += 1;
                            continue;
                        }

                        let pteTbl = pmdEntry.addr().as_u64() as *mut PageTable;
                        let mut clearPTEEntries = 0;
                        let mut p1Idx : u16 = VirtAddr::new(start).p1_index().into();

                        while p1Idx < MemoryDef::ENTRY_COUNT && start < end {
                            let pteEntry : &mut PageTableEntry = &mut (*pteTbl)[PageTableIndex::new(p1Idx)];
                            if pteEntry.is_unused() {
                                clearPTEEntries += 1;
                                start += MemoryDef::PAGE_SIZE;
                                p1Idx += 1;
                                continue;
                            }

                            match Self::freeEntry(pteEntry, pagePool) {
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

                        if clearPTEEntries == MemoryDef::ENTRY_COUNT {
                            let currAddr = pmdEntry.addr().as_u64();
                            pagePool.Deref(currAddr)?;
                            pmdEntry.set_unused();
                            clearPMDEntries += 1;

                            //info!("unmap pmdEntry {:x}", currAddr);
                        }

                        p2Idx += 1;
                    }

                    if clearPMDEntries == MemoryDef::ENTRY_COUNT {
                        let currAddr = pudEntry.addr().as_u64();
                        pagePool.Deref(currAddr)?;
                        pudEntry.set_unused();
                        clearPUDEntries += 1;

                        //info!("unmap pudEntry {:x}", currAddr);
                    }

                    p3Idx += 1;
                }

                if clearPUDEntries == MemoryDef::ENTRY_COUNT {
                    let currAddr = pgdEntry.addr().as_u64();
                    pagePool.Deref(currAddr)?;
                    pgdEntry.set_unused();
                    //info!("unmap pgdEntry {:x}", currAddr);
                }

                p4Idx += 1;
            }
        }

        return Ok(())
    }

    pub fn ToVirtualAddr(p4Idx: PageTableIndex, p3Idx: PageTableIndex, p2Idx: PageTableIndex, p1Idx: PageTableIndex) -> Addr {
        let p4Idx = u64::from(p4Idx);
        let p3Idx = u64::from(p3Idx);
        let p2Idx = u64::from(p2Idx);
        let p1Idx = u64::from(p1Idx);
        let addr = (p4Idx << MemoryDef::PGD_SHIFT) + (p3Idx << MemoryDef::PUD_SHIFT) + (p2Idx << MemoryDef::PMD_SHIFT) + (p1Idx << MemoryDef::PTE_SHIFT);
        return Addr(addr)
    }

    pub fn Traverse(&mut self, start: Addr, end: Addr, mut f: impl FnMut(&mut PageTableEntry, u64), failFast: bool) -> Result<()> {
        //let mut curAddr = start;
        let pt: *mut PageTable = self.root.0 as *mut PageTable;
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
                        return Err(Error::AddressNotMap(Self::ToVirtualAddr(p4Idx, p3Idx, p2Idx, p1Idx).0));
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
                            return Err(Error::AddressNotMap(Self::ToVirtualAddr(p4Idx, p3Idx, p2Idx, p1Idx).0));
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
                                return Err(Error::AddressNotMap(Self::ToVirtualAddr(p4Idx, p3Idx, p2Idx, p1Idx).0));
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
                                    return Err(Error::AddressNotMap(Self::ToVirtualAddr(p4Idx, p3Idx, p2Idx, p1Idx).0));
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

    pub fn SetPageFlags(&mut self, addr: Addr, flags: PageTableFlags) {
        //self.MProtect(addr, addr.AddLen(4096).unwrap(), PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE, false).unwrap();
        self.MProtect(addr, addr.AddLen(MemoryDef::PAGE_SIZE).unwrap(), flags, true).unwrap();
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

    pub fn MProtect(&mut self, start: Addr, end: Addr, flags: PageTableFlags, failFast: bool) -> Result<()> {
        //info!("MProtoc: start={:x}, end={:x}, flag = {:?}", start.0, end.0, flags);
        return self.Traverse(start, end, |entry, virtualAddr| {
            entry.set_flags(flags);
            Invlpg(virtualAddr);
        }, failFast)
    }

    //get the list for page phyaddress for a virtual address range
    pub fn GetAddresses(&mut self, start: Addr, end: Addr, vec: &mut StackVec<u64>) -> Result<()> {
        self.Traverse(start, end, |entry, _virtualAddr| {
            let addr = entry.addr().as_u64();
            vec.Push(addr);
        }, true)?;

        return Ok(())
    }

    fn freeEntry(entry: &mut PageTableEntry,  pagePool: &Allocator) -> Result<bool> {
        let currAddr = entry.addr().as_u64();
        pagePool.Deref(currAddr)?;
        entry.set_unused();
        return Ok(true)
    }

    // if kernel == true, don't need to reference in the pagePool
    fn mapCanonical(&self, start: Addr, end: Addr, phyAddr: Addr, flags: PageTableFlags, pagePool: &Allocator, kernel: bool) -> Result<bool> {
        let mut res = false;

        //info!("mapCanonical virtual start is {:x}, len is {:x}, phystart is {:x}", start.0, end.0 - start.0, phyAddr.0);
        let mut curAddr = start;
        let pt: *mut PageTable = self.root.0 as *mut PageTable;
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
                    (*pudTbl).zero();
                    pgdEntry.set_addr(PhysAddr::new(pudTbl as u64), PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE);
                } else {
                    pudTbl = pgdEntry.addr().as_u64() as *mut PageTable;
                }

                while curAddr.0 < end.0 {
                    let pudEntry = &mut (*pudTbl)[p3Idx];
                    let pmdTbl: *mut PageTable;

                    if pudEntry.is_unused() {
                        pmdTbl = pagePool.AllocPage(true)? as *mut PageTable;
                        (*pmdTbl).zero();
                        pudEntry.set_addr(PhysAddr::new(pmdTbl as u64), PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE);
                    } else {
                        pmdTbl = pudEntry.addr().as_u64() as *mut PageTable;
                    }

                    while curAddr.0 < end.0 {
                        let pmdEntry = &mut (*pmdTbl)[p2Idx];
                        let pteTbl: *mut PageTable;

                        if pmdEntry.is_unused() {
                            pteTbl = pagePool.AllocPage(true)? as *mut PageTable;
                            (*pteTbl).zero();
                            pmdEntry.set_addr(PhysAddr::new(pteTbl as u64), PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE);
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
                                res = Self::freeEntry(pteEntry, pagePool)?;
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

pub struct AlignedAllocator {
    pub size: usize,
    pub align: usize,
}

impl AlignedAllocator {
    pub fn New(size: usize, align: usize) -> Self {
        return Self {
            size: size,
            align: align,
        }
    }

    pub fn Allocate(&self) -> Result<u64> {
        let layout = Layout::from_size_align(self.size, self.align);
        match layout {
            Err(_e) => Err(Error::UnallignedAddress),
            Ok(l) => unsafe {
                let addr = alloc(l);
                Ok(addr as u64)
            }
        }
    }

    pub fn Free(&self, addr: u64) -> Result<()> {
        let layout = Layout::from_size_align(self.size, self.align);
        match layout {
            Err(_e) => Err(Error::UnallignedAddress),
            Ok(l) => unsafe {
                dealloc(addr as *mut u8, l);
                Ok(())
            }
        }
    }
}

#[cfg(test1)]
mod tests {
    use super::*;
    use super::super::buddyallocator::*;
    use alloc::vec::Vec;

    #[repr(align(4096))]
    #[derive(Clone)]
    struct Page {
        data: [u64; 512]
    }

    impl Default for Page {
        fn default() -> Self {
            return Page {
                data: [0; 512]
            };
        }
    }

    #[test]
    fn test_MapPage() {
        let mem: Vec<Page> = vec![Default::default(); 256]; //256 Pages
        let mut allocator = MemAllocator::Init(&mem[0] as *const _ as u64, 8); //2^8
        let mut pt = PageTables::New(&Allocator).unwrap();

        let phyAddr = 6 * 4096;
        pt.MapPage(Addr(0), Addr(phyAddr), PageOpts::UserReadOnly().Val(), &Allocator).unwrap();

        let res = pt.VirtualToPhy(0).unwrap();
        assert_eq!(res, phyAddr);
    }

    #[test]
    fn test_Map() {
        let mem: Vec<Page> = vec![Default::default(); 256]; //256 Pages
        let mut allocator = MemAllocator::Init(&mem[0] as *const _ as u64, 8); //2^8
        let mut pt = PageTables::New(&Allocator).unwrap();

        pt.Map(Addr(4096 * 500), Addr(4096 * 600), Addr(4096 * 550), PageOpts::UserReadOnly().Val(), &Allocator).unwrap();

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

        pt.Map(Addr(4096 * 500), Addr(4096 * 600), Addr(4096 * 550), PageOpts::UserReadOnly().Val(), &Allocator).unwrap();

        let mut nPt = PageTables::New(&Allocator).unwrap();

        pt.CopyRange(&mut nPt, 4096 * 500, 4096 * 100, &Allocator).unwrap();
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

        pt.Map(Addr(4096 * 400), Addr(4096 * 600), Addr(4096 * 450), PageOpts::UserReadWrite().Val(), &Allocator).unwrap();

        let mut nPt = PageTables::New(&Allocator).unwrap();

        pt.ForkRange(&mut nPt, 4096 * 500, 4096 * 100, &Allocator).unwrap();
        //pt.MProtect(Addr(4096 * 500), Addr(4096 * 600), PageOpts::UserReadOnly().Val(), true).unwrap();
        for i in 500..600 {
            let vAddr = i * 4096;
            let pAddr = nPt.VirtualToPhy(vAddr).unwrap();
            assert_eq!(vAddr + 50 * 4096, pAddr);

            assert_eq!(pt.VirtualToEntry(vAddr).unwrap().flags(), PageOpts::UserReadOnly().Val());
            assert_eq!(nPt.VirtualToEntry(vAddr).unwrap().flags(), PageOpts::UserReadOnly().Val());
        }

        for i in 400..500 {
            let vAddr = i * 4096;
            assert_eq!(pt.VirtualToEntry(vAddr).unwrap().flags(), PageOpts::UserReadWrite().Val());
        }
    }
}
