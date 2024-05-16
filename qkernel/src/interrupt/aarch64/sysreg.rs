// https://docs.kernel.org/arch/arm64/cpu-feature-registers.html#implementation
use crate::kernel_def::read_user_opcode;
use core::arch::asm;

pub struct OpCodeDefs;
impl OpCodeDefs {
    // [4:0]
    pub const XT_MASK: u32 = 0x1f;
    // [31:21]
    pub const MRS_MASK: u32 = 0x7FF << 21;
    pub const MRS: u32 = 0x6A9 << 21;
    pub const MSR: u32 = 0x6A8 << 21;
    // [20:5] : op0 + op1 + CRn + CRm + op2
    // encoding of sysregs for MRS/MSR.
    pub const REG_MASK: u32 = 0xFFFF << 5;
    pub const REG_MIDR_EL1: u32 = 0xC000 << 5;
    // EL0 masking for the sys registers: not all the fields shoud be exposed to EL0
    // TODO add other sys regs in the above document if needed
    pub const VIS_MASK_MIDR_EL1: u64 = 0xf0fffffff0fffff0;
}

pub enum SysmovResult {
    ReadSuccess(u64, u8),
    WriteSuccess,
    Invalid,
}

pub fn try_emulate_mrs(pc: u64) -> SysmovResult {
    match unsafe { read_user_opcode(pc) } {
        Some(opcode) => sysreg_mov_el0(opcode),
        _ => SysmovResult::Invalid,
    }
}

/// decode opcode from EL0, if allowed, return the sysreg value and Xt number.
/// otherwise return None.
pub fn sysreg_mov_el0(opcode: u32) -> SysmovResult {
    // it seems we don't need a path for sysreg write
    if opcode & OpCodeDefs::MRS_MASK != OpCodeDefs::MRS {
        return SysmovResult::Invalid;
    }
    let xt = opcode & OpCodeDefs::XT_MASK;
    if xt > 30 {
        return SysmovResult::Invalid;
    }
    // read sysreg
    let val: u64;
    let result = match opcode & OpCodeDefs::REG_MASK {
        OpCodeDefs::REG_MIDR_EL1 => {
            unsafe {
                asm!("MRS {0}, MIDR_EL1", out(reg) val);
            }
            SysmovResult::ReadSuccess(val & OpCodeDefs::VIS_MASK_MIDR_EL1, xt as u8)
        }
        _ => SysmovResult::Invalid,
    };
    return result;
}
