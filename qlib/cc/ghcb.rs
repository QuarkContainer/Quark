use x86_64::registers::model_specific::Msr;
use x86_64::{PhysAddr, VirtAddr};
use x86_64::structures::paging::{Page, Size4KiB};
use super::*;
use super::super::pagetable::PageTables;
use crate::PAGE_MGR;
pub struct GhcbMsr;

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

impl Default for GhcbSaveArea{
    fn default() -> Self {
        return GhcbSaveArea{
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
        }
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

pub unsafe fn early_panic(reason: u64, value: u64) -> !{
    vmgexit_msr(
        GhcbMsr::EXIT_REQ,
        value.wrapping_shl(16) | (reason & 0x7).wrapping_shl(12),
        0,
    );
    unreachable!();
}

pub fn ghcb_msr_make_page_shared(pt: PageTables, page_virt: VirtAddr) {
    // const SNP_PAGE_STATE_PRIVATE: u64 = 1;
    const SNP_PAGE_STATE_SHARED: u64 = 2;

    // Since the initial kernel pt is set up in 1gb page frame,
    // it should be smash twice to 4kb page frame to initial ghcb
    pt.smash(page_virt,&*PAGE_MGR).unwrap();
    match pt.VirtualToEntry(page_virt.as_u64()){
        Err(_) => {
            pt.smash(page_virt, &*PAGE_MGR)
        }
        Ok(_) => Ok(())
    }.unwrap();

    pvalidate(page_virt, PvalidateSize::Size4K, false).unwrap();

    if pt.clear_c_bit_address_range(page_virt, page_virt + Page::<Size4KiB>::SIZE, &*PAGE_MGR).is_err() {
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