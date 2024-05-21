pub use super::linux_def::*;
use super::mem::list_allocator::HostAllocator;
use crate::qlib::cc::sev_snp::{get_cbit_mask, snp_active, C_BIT_MASK};
use crate::qlib::kernel::memmgr::pma::PageMgr;
use alloc::collections::BTreeSet;
use alloc::string::ToString;
pub use alloc::sync::Arc;
pub use alloc::vec::Vec;
pub use x86_64::instructions::tlb::flush;
pub use x86_64::structures::paging::{Page, Size1GiB, Size2MiB, Size4KiB};
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

use super::addr::*;
use super::common::{Allocator, Error, Result};
use super::pagetable::*;
use crate::kernel_def::Invlpg;
use core::sync::atomic::Ordering;

impl PageTables {
    pub fn IsActivePagetable_cc(&self) -> Result<bool> {
        if snp_active() {
            return Ok(self.IsActivePagetable_snp());
        }
        return Err(Error::InvalidInput);
    }

    pub fn SwitchTo_cc(&self) {
        if snp_active() {
            self.SwitchTo_snp();
        } else {
            let addr = self.GetRoot();
            Self::Switch(addr);
        }
    }
    pub fn ForkRange_cc(
        &self,
        to: &Self,
        start: u64,
        len: u64,
        pagePool: &Allocator,
    ) -> Result<()> {
        if snp_active() {
            return self.ForkRange_snp(to, start, len, pagePool);
        }
        return Err(Error::InvalidInput);
    }

    pub fn CopyRange_cc(
        &self,
        to: &Self,
        start: u64,
        len: u64,
        pagePool: &Allocator,
    ) -> Result<()> {
        if snp_active() {
            return self.CopyRange_snp(to, start, len, pagePool);
        }
        return Err(Error::InvalidInput);
    }

    pub fn InitVsyscall_cc(&self, phyAddrs: Arc<Vec<u64>> /*4 pages*/) -> Result<()> {
        if snp_active() {
            self.InitVsyscall_snp(phyAddrs);
            return Ok(());
        }
        return Err(Error::InvalidInput);
    }
    pub fn NewWithKernelPageTables_cc(&self, pagePool: &PageMgr) -> Result<Self> {
        if snp_active() {
            return self.NewWithKernelPageTables_snp(pagePool);
        }
        return Err(Error::InvalidInput);
    }

    pub fn MapVsyscall_cc(&self, phyAddrs: Arc<Vec<u64>> /*4 pages*/) -> Result<()> {
        if snp_active() {
            self.MapVsyscall_snp(phyAddrs);
            return Ok(());
        }
        return Err(Error::InvalidInput);
    }

    pub fn Remap_cc(
        &self,
        start: Addr,
        end: Addr,
        oldStart: Addr,
        flags: PageTableFlags,
        pagePool: &Allocator,
    ) -> Result<bool> {
        if snp_active() {
            return self.Remap_snp(start, end, oldStart, flags, pagePool);
        }
        return Err(Error::InvalidInput);
    }

    pub fn RemapForFile_cc(
        &self,
        start: Addr,
        end: Addr,
        physical: Addr,
        oldStart: Addr,
        oldEnd: Addr,
        flags: PageTableFlags,
        pagePool: &Allocator,
    ) -> Result<bool> {
        if snp_active() {
            return self.RemapForFile_snp(start, end, physical, oldStart, oldEnd, flags, pagePool);
        }
        return Err(Error::InvalidInput);
    }

    #[inline]
    pub fn VirtualToEntry_cc(&self, vaddr: u64) -> Result<&PageTableEntry> {
        if snp_active() {
            return self.VirtualToEntry_snp(vaddr);
        }
        return Err(Error::InvalidInput);
    }

    pub fn VirtualToPhy_cc(&self, vaddr: u64) -> Result<(u64, AccessType)> {
        if snp_active() {
            return self.VirtualToPhy_snp(vaddr);
        }
        return Err(Error::InvalidInput);
    }

    pub fn MapPage_cc(
        &self,
        vaddr: Addr,
        phyAddr: Addr,
        flags: PageTableFlags,
        pagePool: &Allocator,
    ) -> Result<bool> {
        if snp_active() {
            return self.MapPage_snp(vaddr, phyAddr, flags, pagePool);
        }
        return Err(Error::InvalidInput);
    }

    pub fn Unmap_cc(&self, start: u64, end: u64, pagePool: &Allocator) -> Result<()> {
        if snp_active() {
            return self.Unmap_snp(start, end, pagePool);
        }
        return Err(Error::InvalidInput);
    }

    pub fn mapCanonical_cc(
        &self,
        start: Addr,
        end: Addr,
        phyAddr: Addr,
        flags: PageTableFlags,
        pagePool: &Allocator,
        kernel: bool,
    ) -> Result<bool> {
        if snp_active() {
            return self.mapCanonical_snp(start, end, phyAddr, flags, pagePool, kernel);
        }
        return Err(Error::InvalidInput);
    }

    pub fn freeEntry_cc(&self, entry: &mut PageTableEntry, pagePool: &Allocator) -> Result<bool> {
        if snp_active() {
            return self.freeEntry_snp(entry, pagePool);
        }
        return Err(Error::InvalidInput);
    }

    pub fn GetAllPagetablePagesWithRange_cc(
        &self,
        start: Addr,
        end: Addr,
        pages: &mut BTreeSet<u64>,
    ) -> Result<()> {
        if snp_active() {
            return self.GetAllPagetablePagesWithRange_snp(start, end, pages);
        }
        return Err(Error::InvalidInput);
    }

    pub fn Traverse_cc(
        &self,
        start: Addr,
        end: Addr,
        f: impl FnMut(&mut PageTableEntry, u64),
        failFast: bool,
    ) -> Result<()> {
        if snp_active() {
            return self.Traverse_snp(start, end, f, failFast);
        }
        return Err(Error::InvalidInput);
    }

    pub fn SwapInPage_cc(&self, vaddr: Addr) -> Result<u64> {
        if snp_active() {
            return self.SwapInPage_snp(vaddr);
        }
        return Err(Error::InvalidInput);
    }

    pub fn IsActivePagetable_snp(&self) -> bool {
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
        let root = self.GetRoot();
        return root == (Self::CurrentCr3() | c_bit_mask);
    }

    pub fn SwitchTo_snp(&self) {
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
        let addr = self.GetRoot() | c_bit_mask;
        Self::Switch(addr);
    }

    pub fn ForkRange_snp(
        &self,
        to: &Self,
        start: u64,
        len: u64,
        pagePool: &Allocator,
    ) -> Result<()> {
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
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
                    let phyAddr = entry.addr().as_u64() & !c_bit_mask;
                    to.MapPage_snp(
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

    pub fn CopyRange_snp(
        &self,
        to: &Self,
        start: u64,
        len: u64,
        pagePool: &Allocator,
    ) -> Result<()> {
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
        if start & MemoryDef::PAGE_MASK != 0 || len & MemoryDef::PAGE_MASK != 0 {
            return Err(Error::UnallignedAddress(format!("CopyRange {:x?}", len)));
        }

        let mut vAddr = start;
        while vAddr < start + len {
            match self.VirtualToEntry(vAddr) {
                Ok(entry) => {
                    let phyAddr = entry.addr().as_u64() & !c_bit_mask;
                    to.MapPage_snp(Addr(vAddr), Addr(phyAddr), entry.flags(), pagePool)?;
                }
                Err(_) => (),
            }
            vAddr += MemoryDef::PAGE_SIZE;
        }

        Ok(())
    }

    pub fn InitVsyscall_snp(&self, phyAddrs: Arc<Vec<u64>> /*4 pages*/) {
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
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
                PhysAddr::new(pudTbl as u64 | c_bit_mask),
                PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE,
            );

            let pudEntry = &mut (*pudTbl)[p3Idx];
            let pmdTbl: *mut PageTable;

            assert!(pudEntry.is_unused());
            pmdTbl = phyAddrs[2] as *mut PageTable;
            pudEntry.set_addr(
                PhysAddr::new(pmdTbl as u64 | c_bit_mask),
                PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE,
            );

            let pmdEntry = &mut (*pmdTbl)[p2Idx];
            let pteTbl: *mut PageTable;

            assert!(pmdEntry.is_unused());
            pteTbl = phyAddrs[1] as *mut PageTable;
            pmdEntry.set_addr(
                PhysAddr::new(pteTbl as u64 | c_bit_mask),
                PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE,
            );

            let pteEntry = &mut (*pteTbl)[p1Idx];
            assert!(pteEntry.is_unused());
            pteEntry.set_addr(
                PhysAddr::new(phyAddrs[0] | c_bit_mask),
                PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE,
            );

            Invlpg(vaddr);
        }
    }

    pub fn NewWithKernelPageTables_snp(&self, pagePool: &PageMgr) -> Result<Self> {
        let ret = Self::New(pagePool)?;
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
        unsafe {
            let pt: *mut PageTable = self.GetRoot() as *mut PageTable;
            let pgdEntry = &(*pt)[0];
            if pgdEntry.is_unused() {
                return Err(Error::AddressNotMap(0));
            }

            let pudTbl = (pgdEntry.addr().as_u64() & !c_bit_mask) as *const PageTable;
            let nPt: *mut PageTable = ret.GetRoot() as *mut PageTable;
            let nPgdEntry = &mut (*nPt)[0];
            let nPudTbl = pagePool.AllocPage(true)? as *mut PageTable;
            nPgdEntry.set_addr(
                PhysAddr::new(nPudTbl as u64 | c_bit_mask),
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

        {
            let vsyscallPages = pagePool.VsyscallPages();
            ret.MapVsyscall_snp(vsyscallPages);
        }

        return Ok(ret);
    }

    pub fn MapVsyscall_snp(&self, phyAddrs: Arc<Vec<u64>> /*4 pages*/) {
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
        let vaddr = 0xffffffffff600000;
        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;
        unsafe {
            let p4Idx = VirtAddr::new(vaddr).p4_index();
            let pgdEntry = &mut (*pt)[p4Idx];
            let pudTbl: *mut PageTable;

            assert!(pgdEntry.is_unused());
            pudTbl = phyAddrs[3] as *mut PageTable;

            pgdEntry.set_addr(
                PhysAddr::new(pudTbl as u64 | c_bit_mask),
                PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE,
            );
            Invlpg(vaddr);
        }
    }

    pub fn Remap_snp(
        &self,
        start: Addr,
        end: Addr,
        oldStart: Addr,
        flags: PageTableFlags,
        pagePool: &Allocator,
    ) -> Result<bool> {
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
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
                    let phyAddr = oldentry.addr().as_u64() & !c_bit_mask;
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

    pub fn RemapForFile_snp(
        &self,
        start: Addr,
        end: Addr,
        physical: Addr,
        oldStart: Addr,
        oldEnd: Addr,
        flags: PageTableFlags,
        pagePool: &Allocator,
    ) -> Result<bool> {
        start.PageAligned()?;
        oldStart.PageAligned()?;
        if end.0 < start.0 {
            return Err(Error::AddressNotInRange);
        }
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
        let mut offset = 0;
        'a: while start.0 + offset < end.0 {
            if oldStart.0 + offset < oldEnd.0 {
                let entry = self.VirtualToEntry(oldStart.0 + offset);
                match entry {
                    Ok(oldentry) => {
                        let phyAddr = oldentry.addr().as_u64() & !c_bit_mask;
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

    pub fn VirtualToPhy_snp(&self, vaddr: u64) -> Result<(u64, AccessType)> {
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
        let pteEntry = self.VirtualToEntry_snp(vaddr)?;
        if pteEntry.is_unused() {
            return Err(Error::AddressNotMap(vaddr));
        }
        let vaddr = VirtAddr::new(vaddr);
        let pageAddr: u64 = vaddr.page_offset().into();
        let phyAddr = (pteEntry.addr().as_u64() & !c_bit_mask) + pageAddr;
        //info!("VirtualToPhy_snp vaddr:{:x}, pageAddr:{:x}, phyAddr:{:x}",vaddr.as_u64(),pageAddr,phyAddr);
        let permission = AccessType::NewFromPageFlags(pteEntry.flags());

        return Ok((phyAddr, permission));
    }

    // ret: >0: the swapped out page addr, 0: the page is missing
    pub fn SwapInPage_snp(&self, vaddr: Addr) -> Result<u64> {
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
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
                pudTbl = (pgdEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
            }

            let pudEntry = &mut (*pudTbl)[p3Idx];
            let pmdTbl: *mut PageTable;

            if pudEntry.is_unused() {
                return Ok(0);
            } else {
                pmdTbl = (pudEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
            }

            let pmdEntry = &mut (*pmdTbl)[p2Idx];
            let pteTbl: *mut PageTable;

            if pmdEntry.is_unused() {
                return Ok(0);
            } else {
                pteTbl = (pmdEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
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
            let addr = pteEntry.addr().as_u64() & !c_bit_mask;
            return Ok(addr);
        }
    }

    pub fn Traverse_snp(
        &self,
        start: Addr,
        end: Addr,
        mut f: impl FnMut(&mut PageTableEntry, u64),
        failFast: bool,
    ) -> Result<()> {
        start.PageAligned()?;
        end.PageAligned()?;
        //let mut curAddr = start;
        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
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
                    pudTbl = (pgdEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
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
                        pmdTbl = (pudEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
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
                            pteTbl = (pmdEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
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

    pub fn GetAllPagetablePagesWithRange_snp(
        &self,
        start: Addr,
        end: Addr,
        pages: &mut BTreeSet<u64>,
    ) -> Result<()> {
        //let mut curAddr = start;
        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
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
                    pudTbl = (pgdEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
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
                        pages.insert(pudEntry.addr().as_u64() & !c_bit_mask);
                        pmdTbl = (pudEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
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
                            pages.insert(pmdEntry.addr().as_u64() & !c_bit_mask);
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

    pub fn freeEntry_snp(&self, entry: &mut PageTableEntry, pagePool: &Allocator) -> Result<bool> {
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
        let currAddr = entry.addr().as_u64() & !c_bit_mask;
        let refCnt = pagePool.Deref(currAddr)?;
        if refCnt == 0 {
            self.FreePage(currAddr);
        }
        entry.set_unused();
        self.EnableTlbShootdown();
        return Ok(true);
    }

    fn mapCanonical_snp(
        &self,
        start: Addr,
        end: Addr,
        phyAddr: Addr,
        flags: PageTableFlags,
        pagePool: &Allocator,
        kernel: bool,
    ) -> Result<bool> {
        let mut res = false;
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
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
                    pgdEntry.set_addr(
                        PhysAddr::new(pudTbl as u64 | c_bit_mask),
                        default_table_user(),
                    );
                } else {
                    pudTbl = (pgdEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
                }

                while curAddr.0 < end.0 {
                    let pudEntry = &mut (*pudTbl)[p3Idx];
                    let pmdTbl: *mut PageTable;

                    if pudEntry.is_unused() {
                        pmdTbl = pagePool.AllocPage(true)? as *mut PageTable;
                        pudEntry.set_addr(
                            PhysAddr::new(pmdTbl as u64 | c_bit_mask),
                            default_table_user(),
                        );
                    } else {
                        pmdTbl = (pudEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
                    }

                    while curAddr.0 < end.0 {
                        let pmdEntry = &mut (*pmdTbl)[p2Idx];
                        let pteTbl: *mut PageTable;

                        if pmdEntry.is_unused() {
                            pteTbl = pagePool.AllocPage(true)? as *mut PageTable;
                            pmdEntry.set_addr(
                                PhysAddr::new(pteTbl as u64 | c_bit_mask),
                                default_table_user(),
                            );
                        } else {
                            pteTbl = (pmdEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
                        }

                        while curAddr.0 < end.0 {
                            let pteEntry = &mut (*pteTbl)[p1Idx];
                            let newAddr = curAddr.0 - start.0 + phyAddr.0;
                            let is_shared_page = HostAllocator::IsHostGuestSharedHeapAddr(newAddr);
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
                                res = self.freeEntry_snp(pteEntry, pagePool)?;
                            }

                            //info!("set addr: vaddr is {:x}, paddr is {:x}, flags is {:b}", curAddr.0, phyAddr.0, flags.bits());
                            if is_shared_page {
                                pteEntry.set_addr(PhysAddr::new(newAddr), flags);
                            } else {
                                pteEntry.set_addr(PhysAddr::new(newAddr | c_bit_mask), flags);
                            }

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

    #[inline]
    pub fn VirtualToEntry_snp(&self, vaddr: u64) -> Result<&PageTableEntry> {
        let addr = vaddr;
        let vaddr = VirtAddr::new(vaddr);
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);

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

            let pudTbl = (pgdEntry.addr().as_u64() & !c_bit_mask) as *const PageTable;
            let pudEntry = &(*pudTbl)[p3Idx];
            if pudEntry.is_unused() {
                return Err(Error::AddressNotMap(addr));
            }

            let pmdTbl = (pudEntry.addr().as_u64() & !c_bit_mask) as *const PageTable;
            let pmdEntry = &(*pmdTbl)[p2Idx];
            if pmdEntry.is_unused() {
                return Err(Error::AddressNotMap(addr));
            }

            let pteTbl = (pmdEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
            let pteEntry = &mut (*pteTbl)[p1Idx];
            if pteEntry.is_unused() {
                return Err(Error::AddressNotMap(addr));
            }

            // try to swapin page if it is swapout
            self.HandlingSwapInPage(addr, pteEntry);

            return Ok(pteEntry);
        }
    }

    pub fn Unmap_snp(&self, start: u64, end: u64, pagePool: &Allocator) -> Result<()> {
        info!("Unmap_snp start:{:x},end:{:x}", start, end);
        Addr(start).PageAligned()?;
        Addr(end).PageAligned()?;
        let mut start = start;
        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
        unsafe {
            let mut p4Idx: u16 = VirtAddr::new(start).p4_index().into();
            while start < end && p4Idx < MemoryDef::ENTRY_COUNT {
                let pgdEntry: &mut PageTableEntry = &mut (*pt)[PageTableIndex::new(p4Idx)];
                if pgdEntry.is_unused() {
                    start = Self::UnmapNext(start, MemoryDef::PGD_SIZE);
                    p4Idx += 1;
                    continue;
                }

                let pudTbl = (pgdEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
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

                    let pmdTbl = (pudEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
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

                        let pteTbl = (pmdEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
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
                            match self.freeEntry_snp(pteEntry, pagePool) {
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
                            let currAddr = pmdEntry.addr().as_u64() & !c_bit_mask;
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
                        let currAddr = pudEntry.addr().as_u64() & !c_bit_mask;
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
                    let currAddr = pgdEntry.addr().as_u64() & !c_bit_mask;
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
    pub fn MapPage_snp(
        &self,
        vaddr: Addr,
        phyAddr: Addr,
        flags: PageTableFlags,
        pagePool: &Allocator,
    ) -> Result<bool> {
        let mut res = false;
        let c_bit_mask = C_BIT_MASK.load(Ordering::Relaxed);
        let is_shared_page = HostAllocator::IsHostGuestSharedHeapAddr(phyAddr.0);
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
                pgdEntry.set_addr(
                    PhysAddr::new(pudTbl as u64 | c_bit_mask),
                    default_table_user(),
                );
            } else {
                pudTbl = (pgdEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
            }

            let pudEntry = &mut (*pudTbl)[p3Idx];
            let pmdTbl: *mut PageTable;

            if pudEntry.is_unused() {
                pmdTbl = pagePool.AllocPage(true)? as *mut PageTable;
                pudEntry.set_addr(
                    PhysAddr::new(pmdTbl as u64 | c_bit_mask),
                    default_table_user(),
                );
            } else {
                pmdTbl = (pudEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
            }

            let pmdEntry = &mut (*pmdTbl)[p2Idx];
            let pteTbl: *mut PageTable;

            if pmdEntry.is_unused() {
                pteTbl = pagePool.AllocPage(true)? as *mut PageTable;
                pmdEntry.set_addr(
                    PhysAddr::new(pteTbl as u64 | c_bit_mask),
                    default_table_user(),
                );
            } else {
                pteTbl = (pmdEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
            }

            let pteEntry = &mut (*pteTbl)[p1Idx];

            pagePool.Ref(phyAddr.0).unwrap();
            if !pteEntry.is_unused() {
                info!(
                    "Free entry for used page snp vaddr:{:x},phyaddr:{:x}, is shared:{}",
                    vaddr.0, phyAddr.0, is_shared_page
                );
                self.freeEntry_snp(pteEntry, pagePool)?;

                /*let addr = pteEntry.addr().as_u64();
                let bit9 = pteEntry.flags() & PageTableFlags::BIT_9 == PageTableFlags::BIT_9;

                if vaddr.0 != 0 && !bit9 {
                    pagePool.Deref(addr).unwrap();
                }*/
                res = true;
            }
            if is_shared_page {
                pteEntry.set_addr(PhysAddr::new(phyAddr.0), flags);
            } else {
                pteEntry.set_addr(PhysAddr::new(phyAddr.0 | c_bit_mask), flags);
            };

            Invlpg(vaddr.0);
        }
        return Ok(res);
    }

    ///smash pages, 1gb->2mb if to2mb is true, 2mb->4kb if to2mb is false
    ///
    ///return Ok(true) if smash is done, Ok(false) if smash is not done, but already has 2mb/4kb page
    pub fn smash(&self, vaddr: VirtAddr, pagePool: &Allocator, to2mb: bool) -> Result<bool> {
        if !vaddr.is_aligned(MemoryDef::PAGE_SIZE_4K) {
            return Err(Error::UnallignedAddress(vaddr.as_u64().to_string()));
        }

        let addr = vaddr.as_u64();
        let c_bit_mask = get_cbit_mask();
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

            let pudTbl = (pgdEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
            let pudEntry = &mut (*pudTbl)[p3Idx];
            if pudEntry.is_unused() {
                return Err(Error::AddressNotMap(addr));
            } else if pudEntry.flags().contains(PageTableFlags::HUGE_PAGE) && to2mb {
                //smash 1G frames here
                let page = Page::<Size1GiB>::containing_address(vaddr);
                let new_pagetable = &mut *(pagePool.AllocPage(true)? as *mut PageTable);
                let old_addr = pudEntry.addr();

                let old_flags = pudEntry.flags();
                new_pagetable.iter_mut().enumerate().for_each(|(i, e)| {
                    e.set_addr(
                        old_addr + i.checked_mul(Page::<Size2MiB>::SIZE as usize).unwrap(),
                        old_flags,
                    );
                });
                pudEntry.set_addr(
                    PhysAddr::new(
                        VirtAddr::from_ptr(new_pagetable).as_u64()
                            | (old_addr.as_u64() & c_bit_mask),
                    ),
                    old_flags & (!PageTableFlags::HUGE_PAGE),
                );
                flush(page.start_address());
                return Ok(true);
            }

            let pmdTbl = (pudEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
            let pmdEntry = &mut (*pmdTbl)[p2Idx];
            if pmdEntry.is_unused() {
                return Err(Error::AddressNotMap(addr));
            } else if to2mb {
                return Ok(false);
            } else if pmdEntry.flags().contains(PageTableFlags::HUGE_PAGE) && !to2mb {
                //smash 2M frames here
                let page = Page::<Size2MiB>::containing_address(vaddr);
                let new_pagetable = &mut *(pagePool.AllocPage(true)? as *mut PageTable);
                let old_addr = pmdEntry.addr();
                let old_flags = pmdEntry.flags() & (!PageTableFlags::HUGE_PAGE);
                new_pagetable.iter_mut().enumerate().for_each(|(i, e)| {
                    e.set_addr(
                        old_addr + i.checked_mul(Page::<Size4KiB>::SIZE as usize).unwrap(),
                        old_flags,
                    );
                });
                pmdEntry.set_addr(
                    PhysAddr::new(
                        VirtAddr::from_ptr(new_pagetable).as_u64()
                            | (old_addr.as_u64() & c_bit_mask),
                    ),
                    old_flags,
                );
                flush(page.start_address());
                return Ok(true);
            }

            let pteTbl = (pmdEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
            let pteEntry = &mut (*pteTbl)[p1Idx];
            if pteEntry.is_unused() {
                return Err(Error::AddressNotMap(addr));
            }
            return Ok(false);
        }
    }

    pub fn clear_c_bit_address_range(
        &self,
        start: VirtAddr,
        end: VirtAddr,
        pagePool: &Allocator,
    ) -> Result<()> {
        if !start.is_aligned(MemoryDef::PAGE_SIZE_4K) {
            return Err(Error::UnallignedAddress(start.as_u64().to_string()));
        }

        if !end.is_aligned(MemoryDef::PAGE_SIZE_4K) {
            return Err(Error::UnallignedAddress(end.as_u64().to_string()));
        }
        let c_bit_mask = get_cbit_mask();
        let mut current = start;
        loop {
            if current >= end {
                return Ok(());
            }

            let addr = current.as_u64();
            let p4Idx = current.p4_index();
            let p3Idx = current.p3_index();
            let p2Idx = current.p2_index();
            let p1Idx = current.p1_index();
            let pt: *mut PageTable = self.GetRoot() as *mut PageTable;
            unsafe {
                let pgdEntry = &(*pt)[p4Idx];
                if pgdEntry.is_unused() {
                    return Err(Error::AddressNotMap(addr));
                }

                let pudTbl = (pgdEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
                let pudEntry = &mut (*pudTbl)[p3Idx];
                if pudEntry.is_unused() {
                    return Err(Error::AddressNotMap(addr));
                } else if pudEntry.flags().contains(PageTableFlags::HUGE_PAGE) {
                    if ((pudEntry.addr().as_u64())&!c_bit_mask )!= current.as_u64() //frame offset is not 0
                        || current + Page::<Size1GiB>::SIZE as usize > end
                    {
                        self.smash(current, pagePool, true)?;
                        return self.clear_c_bit_address_range(current, end, pagePool);
                    }

                    //clear share bit of 1gb page
                    let page = Page::<Size1GiB>::containing_address(current);
                    let old_addr = pudEntry.addr();
                    let old_flags = pudEntry.flags();
                    pudEntry.set_addr(PhysAddr::new(old_addr.as_u64() & !c_bit_mask), old_flags);
                    flush(page.start_address());
                    current += Page::<Size1GiB>::SIZE;
                    continue;
                }

                let pmdTbl = (pudEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
                let pmdEntry = &mut (*pmdTbl)[p2Idx];
                if pmdEntry.is_unused() {
                    return Err(Error::AddressNotMap(addr));
                } else if pmdEntry.flags().contains(PageTableFlags::HUGE_PAGE) {
                    if ((pmdEntry.addr().as_u64())&!c_bit_mask )!= current.as_u64() //frame offset is not 0
                        || current + Page::<Size2MiB>::SIZE as usize > end
                    {
                        self.smash(current, pagePool, false)?;
                        return self.clear_c_bit_address_range(current, end, pagePool);
                    }

                    //clear share bit of 2mb page
                    let page = Page::<Size2MiB>::containing_address(current);
                    let old_addr = pmdEntry.addr();
                    let old_flags = pmdEntry.flags();
                    pmdEntry.set_addr(PhysAddr::new(old_addr.as_u64() & !c_bit_mask), old_flags);
                    flush(page.start_address());
                    current += Page::<Size2MiB>::SIZE;
                    continue;
                }

                let pteTbl = (pmdEntry.addr().as_u64() & !c_bit_mask) as *mut PageTable;
                let pteEntry = &mut (*pteTbl)[p1Idx];
                if pteEntry.is_unused() {
                    return Err(Error::AddressNotMap(addr));
                }
                let page = Page::<Size4KiB>::containing_address(current);
                let old_addr = pteEntry.addr();
                let old_flags = pteEntry.flags();
                pteEntry.set_addr(PhysAddr::new(old_addr.as_u64() & !c_bit_mask), old_flags);
                flush(page.start_address());
                current += Page::<Size4KiB>::SIZE;
            }
        }
    }

    pub fn MapWith1GSevSnp(
        &self,
        start: Addr,
        end: Addr,
        physical: Addr,
        flags: PageTableFlags,
        pagePool: &Allocator,
        c_bit: u64,
        _kernel: bool,
    ) -> Result<bool> {
        if start.0 & (MemoryDef::HUGE_PAGE_SIZE_1G - 1) != 0
            || end.0 & (MemoryDef::HUGE_PAGE_SIZE_1G - 1) != 0
        {
            panic!("start/end address not 1G aligned")
        }

        let mut res = false;
        let c_bit_flag = unsafe { PageTableFlags::from_bits_unchecked(1u64 << c_bit) };
        let mut curAddr = start;
        let pt: *mut PageTable = self.GetRoot() as *mut PageTable;
        unsafe {
            let mut p4Idx = VirtAddr::new(curAddr.0).p4_index();
            let mut p3Idx = VirtAddr::new(curAddr.0).p3_index();

            while curAddr.0 < end.0 {
                let pgdEntry = &mut (*pt)[p4Idx];
                let pudTbl: *mut PageTable;

                if pgdEntry.is_unused() {
                    let ret = pagePool.AllocPage(true)?;
                    pudTbl = ret as *mut PageTable;
                    pgdEntry.set_addr(
                        PhysAddr::new(pudTbl as u64),
                        default_table_user() | c_bit_flag,
                    );
                } else {
                    pudTbl = (pgdEntry.addr().as_u64() & (1u64 << c_bit - 1)) as *mut PageTable;
                }

                while curAddr.0 < end.0 {
                    let pudEntry = &mut (*pudTbl)[p3Idx];
                    let newphysAddr = curAddr.0 - start.0 + physical.0;

                    // Question: if we also do this for kernel, do we still need this?
                    if !pudEntry.is_unused() {
                        res = self.freeEntry_cc(pudEntry, pagePool)?;
                    }

                    pudEntry.set_addr(
                        PhysAddr::new(newphysAddr),
                        flags | PageTableFlags::HUGE_PAGE | c_bit_flag,
                    );

                    info!(
                        "Add pagetable entry: pgdEntry addr:{:x}, pudEntry addr:{:x}",
                        pgdEntry.addr(),
                        pudEntry.addr()
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
}
