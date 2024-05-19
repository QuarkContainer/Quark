pub use super::linux_def::*;
use crate::qlib::cc::sev_snp::{snp_active, C_BIT_MASK};
pub use alloc::sync::Arc;
pub use alloc::vec::Vec;
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
use core::sync::atomic::Ordering;

impl PageTables {
    pub fn freeEntry_cc(&self, entry: &mut PageTableEntry, pagePool: &Allocator) -> Result<bool> {
        if snp_active() {
            return self.freeEntry_snp(entry, pagePool);
        }
        return Err(Error::InvalidInput);
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
