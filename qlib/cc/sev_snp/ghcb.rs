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

pub unsafe fn vmgexit_msr(request_code: u64, value: u64, expected_response: u64) -> u64 {
    let val = request_code | value;

    let mut msr = GhcbMsr::MSR;

    msr.write(val);

    asm!("rep vmmcall", options(nostack));

    let retcode = msr.read();

    if expected_response != retcode & 0xFFF {
        early_panic(1, 2);
    }

    retcode & (!0xFFF)
}

pub unsafe fn early_panic(reason: u64, value: u64) {
    vmgexit_msr(
        GhcbMsr::EXIT_REQ,
        value.wrapping_shl(16) | (reason & 0x7).wrapping_shl(12),
        0,
    );
}

pub fn ghcb_msr_make_page_shared(page_virt: VirtAddr) {
    let pt = &KERNEL_PAGETABLE;
    // const SNP_PAGE_STATE_PRIVATE: u64 = 1;
    const SNP_PAGE_STATE_SHARED: u64 = 2;
    // Since the initial kernel pt is set up in 1gb page frame,
    // it should be smash twice to 4kb page frame to initial ghcb
    pt.smash(page_virt, &*PAGE_MGR, true).unwrap();
    pt.smash(page_virt, &*PAGE_MGR, false).unwrap();
    pvalidate(page_virt, PvalidateSize::Size4K, false).unwrap();

    if pt
        .clear_c_bit_address_range(page_virt, page_virt + Page::<Size4KiB>::SIZE, &*PAGE_MGR)
        .is_err()
    {
        unsafe {
            early_panic(4, 0x30);
        }
    }

    let gpa = page_virt;

    const SHARED_BIT: u64 = SNP_PAGE_STATE_SHARED << GhcbMsr::PSC_OP_POS;

    let val = gpa.as_u64() | SHARED_BIT;

    unsafe {
        let ret = vmgexit_msr(GhcbMsr::PSC_REQ, val, GhcbMsr::PSC_RESP);

        if (ret & GhcbMsr::PSC_ERROR_MASK) != 0 {
            early_panic(4, 0x33);
        }
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

impl<'a> GhcbHandle<'a> {
    pub fn new(vcpuid: u64) -> Self {
        return GhcbHandle {
            ghcb: unsafe { &mut *((MemoryDef::GHCB_OFFSET + vcpuid * PAGE_SIZE) as *mut Ghcb) },
        };
    }
    pub fn init(&mut self) {
        let ghcb_virt = VirtAddr::from_ptr(self.ghcb);
        ghcb_msr_make_page_shared(ghcb_virt);

        unsafe {
            let gpa = ghcb_virt.as_u64();

            let ret = vmgexit_msr(GhcbMsr::GPA_REQ, gpa, GhcbMsr::GPA_RESP);

            if ret != gpa {
                early_panic(4, 0x34);
            }
        }
        *self.ghcb = Ghcb::default();
    }

    /// do a vmgexit with the ghcb block
    ///
    /// # Safety
    /// undefined behaviour if not everything is setup according to the GHCB protocol
    pub unsafe fn vmgexit(
        &mut self,
        exit_code: u64,
        exit_info_1: u64,
        exit_info_2: u64,
    ) -> Result<(), GhcbError> {
        // const GHCB_PROTOCOL_MIN: u16 = 1;
        const GHCB_PROTOCOL_MAX: u16 = 2;
        const GHCB_DEFAULT_USAGE: u32 = 0;

        self.ghcb.save_area.sw_exit_code = exit_code;
        self.set_offset_valid(ptr::addr_of!(self.ghcb.save_area.sw_exit_code) as _);

        self.ghcb.save_area.sw_exit_info1 = exit_info_1;
        self.set_offset_valid(ptr::addr_of!(self.ghcb.save_area.sw_exit_info1) as _);

        self.ghcb.save_area.sw_exit_info2 = exit_info_2;
        self.set_offset_valid(ptr::addr_of!(self.ghcb.save_area.sw_exit_info2) as _);

        self.ghcb.ghcb_usage = GHCB_DEFAULT_USAGE;
        // FIXME: do protocol negotiation
        self.ghcb.protocol_version = GHCB_PROTOCOL_MAX;

        // prevent earlier writes from being moved beyond this point
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::Release);

        let gpa = VirtAddr::from_ptr(self.ghcb).as_u64();
        let mut msr = GhcbMsr::MSR;

        msr.write(gpa);

        asm!("rep vmmcall", options(nostack));

        // prevent later reads from being moved before this point
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::Acquire);

        if (self.ghcb.save_area.sw_exit_info1 & 0xffff_ffff) == 1 {
            const SVM_EVTINJ_VALID: u64 = 1 << 31;
            const SVM_EVTINJ_TYPE_SHIFT: u64 = 8;
            const SVM_EVTINJ_TYPE_MASK: u64 = 7 << SVM_EVTINJ_TYPE_SHIFT;
            const SVM_EVTINJ_TYPE_EXEPT: u64 = 3 << SVM_EVTINJ_TYPE_SHIFT;
            const SVM_EVTINJ_VEC_MASK: u64 = 0xff;
            const UD: u64 = 6;
            const GP: u64 = 13;

            // VmgExitErrorCheck, see
            // https://github.com/AMDESE/ovmf/blob/sev-snp-v6/OvmfPkg/Library/VmgExitLib/VmgExitLib.c
            // or linux kernel arch/x86/kernel/sev-shared.c
            let exit_info2 = self.ghcb.save_area.sw_exit_info2;
            let vector = exit_info2 & SVM_EVTINJ_VEC_MASK;

            if (exit_info2 & SVM_EVTINJ_VALID != 0)
                && (exit_info2 & SVM_EVTINJ_TYPE_MASK == SVM_EVTINJ_TYPE_EXEPT)
                && (vector == GP || vector == UD)
            {
                return Err(GhcbError::Exception(vector));
            }

            Err(GhcbError::VmmError)
        } else {
            Ok(())
        }
    }

    /// clear all bits in the valid offset bitfield
    pub fn invalidate(&mut self) {
        self.ghcb.save_area.sw_exit_code = 0;
        self.ghcb
            .save_area
            .valid_bitmap
            .iter_mut()
            .for_each(|e| *e = 0);
    }

    /// check if a bit is set in the valid offset bitfield
    pub fn check_offset_valid(&mut self, offset: usize) -> bool {
        let offset = offset.checked_sub(self.ghcb as *const _ as usize).unwrap();
        let offset = offset / 8;
        return (self.ghcb.save_area.valid_bitmap[offset / 8]
            & 1u8.checked_shl((offset & 0x7) as u32).unwrap())
            != 0;
    }

    pub fn check_rax_valid(&mut self) -> bool {
        return self.check_offset_valid(ptr::addr_of!(self.ghcb.save_area.rax) as _);
    }

    pub fn check_rdx_valid(&mut self) -> bool {
        return self.check_offset_valid(ptr::addr_of!(self.ghcb.save_area.rdx) as _);
    }

    pub fn check_rcx_valid(&mut self) -> bool {
        return self.check_offset_valid(ptr::addr_of!(self.ghcb.save_area.rcx) as _);
    }

    pub fn set_rax(&mut self, rax: u64) {
        self.ghcb.save_area.rax = rax;
        self.set_offset_valid(ptr::addr_of!(self.ghcb.save_area.rax) as _);
    }

    pub fn set_rcx(&mut self, rcx: u64) {
        self.ghcb.save_area.rcx = rcx;
        self.set_offset_valid(ptr::addr_of!(self.ghcb.save_area.rcx) as _);
    }

    pub fn set_rdx(&mut self, rdx: u64) {
        self.ghcb.save_area.rdx = rdx;
        self.set_offset_valid(ptr::addr_of!(self.ghcb.save_area.rdx) as _);
    }

    pub fn set_sw_scratch(&mut self, sw_scratch: u64) {
        self.ghcb.save_area.sw_scratch = sw_scratch;
        self.set_offset_valid(ptr::addr_of!(self.ghcb.save_area.sw_scratch) as _);
    }

    pub fn Rax(&self) -> u64 {
        return self.ghcb.save_area.rax;
    }

    pub fn Rdx(&self) -> u64 {
        return self.ghcb.save_area.rdx;
    }

    pub fn Rcx(&self) -> u64 {
        return self.ghcb.save_area.rcx;
    }

    pub fn shared_buffer_addr(&self) -> u64 {
        return ptr::addr_of!(self.ghcb.shared_buffer) as _;
    }

    /// set a bit in the valid offset bitfield
    pub fn set_offset_valid(&mut self, offset: usize) {
        let offset = offset.checked_sub(self.ghcb as *const _ as usize).unwrap();
        let offset = offset / 8;
        self.ghcb.save_area.valid_bitmap[offset / 8] |=
            1u8.checked_shl((offset & 0x7) as u32).unwrap();
    }

    pub fn do_io_out(&mut self, portnumber: u16, value: u16) {
        const IOIO_TYPE_OUT: u64 = 0;
        const IOIO_DATA_16: u64 = 1 << 5;
        const SVM_EXIT_IOIO_PROT: u64 = 0x7B;

        self.invalidate();

        self.ghcb.save_area.rax = value as _;
        let offset: usize = ptr::addr_of!(self.ghcb.save_area.rax) as _;
        self.set_offset_valid(offset);

        unsafe {
            if self
                .vmgexit(
                    SVM_EXIT_IOIO_PROT,
                    IOIO_DATA_16 | IOIO_TYPE_OUT | ((portnumber as u64).checked_shl(16).unwrap()),
                    0,
                )
                .is_err()
            {
                early_panic(4, 0x10);
            }
        }
    }

    pub fn set_memory_shared_2mb(&mut self, virt_addr: VirtAddr, npages: u64) {
        let mut time = 0u64;
        let mut npages_current = npages;
        assert!(npages >= 1);
        let pt = &KERNEL_PAGETABLE;
        (virt_addr.as_u64()
            ..(virt_addr + MemoryDef::PAGE_SIZE_2M.checked_mul(npages as u64).unwrap()).as_u64())
            .step_by(MemoryDef::PAGE_SIZE_2M as usize)
            .for_each(|a| {
                let virt = VirtAddr::new(a);
                match pt.smash(virt, &*PAGE_MGR, true) {
                    Ok(_) => (),
                    Err(_) => unsafe { early_panic(0x4, 0x14) },
                };
                for i in 0..512 {
                    let new_virt = virt + i * MemoryDef::PAGE_SIZE;
                    match pvalidate(new_virt, PvalidateSize::Size4K, false) {
                        Ok(_) => (),
                        Err(_) => unsafe { early_panic(0x4, 0x24) },
                    }
                }
            });

        match pt.clear_c_bit_address_range(
            virt_addr,
            virt_addr + MemoryDef::PAGE_SIZE_2M.checked_mul(npages as u64).unwrap(),
            &*PAGE_MGR,
        ) {
            Ok(_) => (),
            Err(_) => {
                self.do_io_out(0x3f, (virt_addr.as_u64() >> 12) as u16);
                unsafe {
                    early_panic(0x3, 0x3);
                }
            }
        }
        loop {
            let mut stop = false;
            let pages = if npages_current >= PSC_ENTRY_LEN {
                PSC_ENTRY_LEN
            } else {
                stop = true;
                npages_current
            };
            self.set_memory_shared_ghcb_2mb(
                virt_addr + MemoryDef::PAGE_SIZE_2M * PSC_ENTRY_LEN * time,
                pages as usize,
            );
            if stop {
                break;
            }
            npages_current -= PSC_ENTRY_LEN;
            time += 1;
        }
    }

    pub fn set_memory_shared_ghcb_2mb(&mut self, virt_addr: VirtAddr, npages: usize) {
        const SVM_VMGEXIT_PSC: u64 = 0x80000010;
        // Fill in shared_buffer
        // SnpPscDesc has the exact same size.
        let psc_desc: &mut SnpPscDesc =
            unsafe { &mut *(self.ghcb.shared_buffer.as_mut_ptr() as *mut SnpPscDesc) };

        *psc_desc = SnpPscDesc::default();

        // FIXME
        assert!(psc_desc.entries.len() >= npages);

        psc_desc.cur_entry = 0;
        psc_desc.end_entry = (npages as u16).checked_sub(1).unwrap();

        let mut pa_addr = PhysAddr::new(virt_addr.as_u64());

        for i in 0..npages {
            psc_desc.entries[i].set_entry(pa_addr.as_u64(), RmpPgOp::Shared, RmpPgSize::Size2m);
            pa_addr += Page::<Size2MiB>::SIZE;
        }

        loop {
            // Use `read_volatile` to be safe
            let cur_entry = unsafe { ptr::addr_of!(psc_desc.cur_entry).read_volatile() };
            let end_entry = unsafe { ptr::addr_of!(psc_desc.end_entry).read_volatile() };

            if cur_entry > end_entry {
                break;
            }

            self.invalidate();

            let addr = ptr::addr_of!(self.ghcb.shared_buffer);
            self.ghcb.save_area.sw_scratch = (VirtAddr::from_ptr(addr)).as_u64();
            let offset: usize = ptr::addr_of!(self.ghcb.save_area.sw_scratch) as _;
            self.set_offset_valid(offset);

            unsafe {
                if self.vmgexit(SVM_VMGEXIT_PSC, 0, 0).is_err() {
                    early_panic(4, 0x33);
                }
            }

            if psc_desc.reserved != 0 {
                unsafe {
                    early_panic(4, 0x35);
                }
            }
            if (psc_desc.end_entry > end_entry) || (cur_entry > psc_desc.cur_entry) {
                unsafe {
                    early_panic(4, 0x36);
                }
            }
        }
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
