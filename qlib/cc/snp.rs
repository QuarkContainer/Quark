//use x86_64::registers::model_specific::Msr;

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