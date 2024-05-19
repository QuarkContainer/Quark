use core::ptr;
use spin::mutex::Mutex;
use x86_64::registers::model_specific::Msr;
use x86_64::structures::paging::{Page, Size2MiB, Size4KiB};
use x86_64::{PhysAddr, VirtAddr};

use crate::qlib::addr::PAGE_SIZE;

use super::super::super::kernel::{KERNEL_PAGETABLE, PAGE_MGR};
use super::super::super::linux_def::*;
use super::*;
pub struct GhcbMsr;
const GHCB_HANDLE: spin::mutex::Mutex<core::option::Option<ghcb::GhcbHandle<'static>>> = Mutex::new(None);
pub static GHCB: [Mutex<Option<GhcbHandle<'static>>>; 0x200] = [GHCB_HANDLE; 0x200];
/// GHCB page sizes
#[derive(Copy, Clone)]
#[repr(C)]
#[non_exhaustive]
enum RmpPgSize {
    Size4k = 0,
    Size2m = 1,
}

/// GHCB page operation
#[derive(Copy, Clone)]
#[repr(C)]
#[non_exhaustive]
enum RmpPgOp {
    // Private = 1,
    Shared = 2,
    // PSmash,
    // UnSmash,
}

/// GHCB page state entry
#[derive(Debug, Copy, Clone, Default)]
#[repr(C)]
pub struct PscEntry {
    entry: u64,
}

impl PscEntry {
    #[inline(always)]
    #[allow(clippy::integer_arithmetic)]
    fn set_entry(&mut self, cur_page: u64, operation: RmpPgOp, pagesize: RmpPgSize) {
        self.entry = cur_page | ((operation as u64) << 52) | ((pagesize as u64) << 56)
    }
}

const PSC_ENTRY_LEN: u64 = 253;
/// GHCB page state description
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct SnpPscDesc {
    pub cur_entry: u16,
    pub end_entry: u16,
    pub reserved: u32,
    pub entries: [PscEntry; 253],
}

impl Default for SnpPscDesc {
    fn default() -> Self {
        return SnpPscDesc {
            cur_entry: 0u16,
            end_entry: 0u16,
            reserved: 0u32,
            entries: [PscEntry::default(); 253],
        };
    }
}

impl GhcbMsr {
    /// The underlying model specific register.
    pub const MSR: Msr = Msr::new(0xC001_0130);

    const GPA_REQ: u64 = 0x12;
    const GPA_RESP: u64 = 0x13;
    const PSC_REQ: u64 = 0x14;
    const PSC_RESP: u64 = 0x15;

    /// Request an VM exit via the GHCB MSR protocol
    pub const EXIT_REQ: u64 = 0x100;

    const PSC_OP_POS: u64 = 52;
    const PSC_ERROR_POS: u64 = 32;
    const PSC_ERROR_MASK: u64 = u64::MAX >> Self::PSC_ERROR_POS;
}

pub const GHCB_SHARED_BUFFER_SIZE: u64 = 2032;
/// GHCB
#[derive(Debug, Copy, Clone)]
#[repr(C, align(4096))]
pub struct Ghcb {
    save_area: GhcbSaveArea,
    shared_buffer: [u8; 2032],
    reserved1: [u8; 10],
    protocol_version: u16,
    ghcb_usage: u32,
}
impl Default for Ghcb {
    fn default() -> Self {
        return Ghcb {
            save_area: GhcbSaveArea::default(),
            shared_buffer: [0u8; 2032],
            reserved1: [0u8; 10],
            protocol_version: 0u16,
            ghcb_usage: 0u32,
        };
    }
}

impl Ghcb {
    pub fn get_shared_buffer_clone(&self) -> [u8; 2032] {
        return self.shared_buffer.clone();
    }

    pub fn set_shared_buffer(&mut self, shared_buffer: [u8; 2032]) {
        self.shared_buffer = shared_buffer;
    }
}
#[derive(Copy, Clone)]
#[non_exhaustive]
pub enum GhcbError {
    /// Unexpected state from the VMM
    VmmError,
    /// Instruction caused exception
    Exception(u64),
}

/// GHCB Save Area
#[derive(Debug, Copy, Clone)]
#[repr(C, packed)]
pub struct GhcbSaveArea {
    reserved1: [u8; 203],
    cpl: u8,
    reserved8: [u8; 300],
    rax: u64,
    reserved4: [u8; 264],
    rcx: u64,
    rdx: u64,
    rbx: u64,
    reserved5: [u8; 112],
    sw_exit_code: u64,
    sw_exit_info1: u64,
    sw_exit_info2: u64,
    sw_scratch: u64,
    reserved6: [u8; 56],
    xcr0: u64,
    valid_bitmap: [u8; 16],
    x87state_gpa: u64,
    reserved7: [u8; 1016],
}

impl Default for GhcbSaveArea {
    fn default() -> Self {
        return GhcbSaveArea {
            reserved1: [0u8; 203],
            cpl: 0,
            reserved8: [0u8; 300],
            rax: 0,
            reserved4: [0u8; 264],
            rcx: 0,
            rdx: 0,
            rbx: 0,
            reserved5: [0u8; 112],
            sw_exit_code: 0,
            sw_exit_info1: 0,
            sw_exit_info2: 0,
            sw_scratch: 0,
            reserved6: [0u8; 56],
            xcr0: 0,
            valid_bitmap: [0u8; 16],
            x87state_gpa: 0,
            reserved7: [0u8; 1016],
        };
    }
}

/// A handle to the GHCB block
pub struct GhcbHandle<'a> {
    ghcb: &'a mut Ghcb,
}

impl<'a> Default for GhcbHandle<'a> {
    fn default() -> Self {
        return GhcbHandle {
            ghcb: unsafe { &mut *(MemoryDef::GHCB_OFFSET as *mut Ghcb) },
        };
    }
}

pub fn next_contig_gpa_range(
    desc: &mut SnpPscDesc,
    entries_processed: &mut u16,
    gfn_base: &mut u64,
    gfn_count: &mut i32,
    range_to_private: &mut bool,
) -> bool {
    *entries_processed = 0;
    *gfn_base = 0;
    *gfn_count = 0;
    *range_to_private = false;

    for i in desc.cur_entry..=desc.end_entry {
        let entry = &mut desc.entries[i as usize];
        let operation = (entry.entry >> 52) & 0xF;
        let pagesize = (entry.entry >> 56) & 0x1;
        let gfn = (entry.entry >> 12) & ((1 << 40) - 1);
        let to_private = operation == 1;
        let page_count = if pagesize == 1 { 512 } else { 1 };
        if *gfn_count == 0 {
            *range_to_private = to_private;
            *gfn_base = gfn;
        }

        if (gfn != *gfn_base + (*gfn_count) as u64) || (to_private != *range_to_private) {
            return true;
        }
        *gfn_count += page_count;
        entry.entry &= !((1 << 12) - 1);
        entry.entry |= (page_count as u64) & ((1 << 12) - 1);
        *entries_processed += 1;
    }
    if *gfn_count != 0 {
        return true;
    } else {
        return false;
    }
}
