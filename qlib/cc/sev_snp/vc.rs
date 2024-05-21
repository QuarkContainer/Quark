use yaxpeax_x86::amd64::{Opcode, Operand, RegSpec};
use yaxpeax_x86::long_mode::Instruction;

use super::super::super::common::*;
use super::super::super::kernel::SignalDef::PtRegs;
use super::super::super::linux_def::*;
use super::cpuid_page::*;
use super::ghcb::*;
use super::*;
use core::ptr;

#[no_mangle]
pub fn vc_check_opcode_bytes(code: Opcode, errorCode: u64) -> VCResult {
    match errorCode {
        SVMExitDef::SVM_EXIT_IOIO => return VCResult::EsOk,
        SVMExitDef::SVM_EXIT_NPF => return VCResult::EsOk,
        SVMExitDef::SVM_EXIT_CPUID => {
            if code == Opcode::CPUID {
                return VCResult::EsOk;
            }
        }
        SVMExitDef::SVM_EXIT_INVD => {
            if code == Opcode::INVD {
                return VCResult::EsOk;
            }
        }
        SVMExitDef::SVM_EXIT_MONITOR => {
            if code == Opcode::MONITOR {
                return VCResult::EsOk;
            }
        }
        SVMExitDef::SVM_EXIT_MWAIT => {
            if code == Opcode::MWAIT {
                return VCResult::EsOk;
            }
        }
        SVMExitDef::SVM_EXIT_MSR => {
            if code == Opcode::RDMSR || code == Opcode::WRMSR {
                return VCResult::EsOk;
            }
        }
        SVMExitDef::SVM_EXIT_RDPMC => {
            if code == Opcode::RDPMC {
                return VCResult::EsOk;
            }
        }
        SVMExitDef::SVM_EXIT_RDTSC => {
            if code == Opcode::RDTSC {
                return VCResult::EsOk;
            }
        }
        SVMExitDef::SVM_EXIT_RDTSCP => {
            if code == Opcode::RDTSCP {
                return VCResult::EsOk;
            }
        }
        SVMExitDef::SVM_EXIT_VMMCALL => {
            if code == Opcode::VMMCALL {
                return VCResult::EsOk;
            }
        }
        SVMExitDef::SVM_EXIT_WBINVD => {
            if code == Opcode::WBINVD {
                return VCResult::EsOk;
            }
        }
        _ => (),
    }
    return VCResult::EsUnsupported;
}

#[no_mangle]
pub fn vc_handle_exitcode_ghcb(
    insn: &Instruction,
    ghcb: &mut GhcbHandle,
    errorCode: u64,
    sf: &mut PtRegs,
) -> VCResult {
    let code = insn.opcode();
    let mut ret = vc_check_opcode_bytes(code, errorCode);
    if ret != VCResult::EsOk {
        return ret;
    }
    const X86_TRAP_AC: u64 = 17;
    const SVM_TRAP_AC: u64 = SVMExitDef::SVM_EXIT_EXCP_BASE + X86_TRAP_AC;
    match errorCode {
        SVM_TRAP_AC => ret = VCResult::EsException(X86_TRAP_AC),
        SVMExitDef::SVM_EXIT_RDTSC | SVMExitDef::SVM_EXIT_RDTSCP => {
            ret = vc_handle_rdtsc(ghcb, errorCode, sf)
        }
        SVMExitDef::SVM_EXIT_RDPMC => ret = vc_handle_rdpmc(ghcb, sf),
        SVMExitDef::SVM_EXIT_IOIO => ret = vc_handle_ioio(ghcb, insn, sf),
        SVMExitDef::SVM_EXIT_MSR => ret = vc_handle_msr(ghcb, code, sf),
        SVMExitDef::SVM_EXIT_WBINVD => ret = vc_handle_wbinvd(ghcb),
        _ => ret = VCResult::EsUnsupported,
    }
    return ret;
}

pub fn vc_handle_wbinvd(ghcb: &mut GhcbHandle) -> VCResult {
    let ret;
    unsafe {
        match ghcb.vmgexit(SVMExitDef::SVM_EXIT_WBINVD, 0, 0) {
            Ok(_) => ret = VCResult::EsOk,
            Err(GhcbError::Exception(e)) => ret = VCResult::EsException(e),
            Err(GhcbError::VmmError) => ret = VCResult::EsVmmError,
        }
    }
    return ret;
}

pub fn vc_handle_msr(ghcb: &mut GhcbHandle, code: Opcode, sf: &mut PtRegs) -> VCResult {
    let ret;
    /* Is it a WRMSR? */
    let exit_info_1 = if code == Opcode::WRMSR { 1 } else { 0 };

    ghcb.set_rcx(sf.rcx);
    if exit_info_1 != 0 {
        ghcb.set_rax(sf.rax);
        ghcb.set_rdx(sf.rdx);
    }
    unsafe {
        match ghcb.vmgexit(SVMExitDef::SVM_EXIT_MSR, exit_info_1, 0) {
            Ok(_) => ret = VCResult::EsOk,
            Err(GhcbError::Exception(e)) => ret = VCResult::EsException(e),
            Err(GhcbError::VmmError) => ret = VCResult::EsVmmError,
        }
    }

    if (ret == VCResult::EsOk) && (exit_info_1 == 0) {
        sf.rax = ghcb.Rax();
        sf.rdx = ghcb.Rdx();
    }
    return ret;
}

const IOIO_TYPE_STR: u32 = 1 << 2;
const IOIO_TYPE_IN: u32 = 1;
const IOIO_TYPE_INS: u32 = IOIO_TYPE_IN | IOIO_TYPE_STR;
const IOIO_TYPE_OUT: u32 = 0;
const IOIO_TYPE_OUTS: u32 = IOIO_TYPE_OUT | IOIO_TYPE_STR;

const IOIO_REP: u32 = 1 << 3;

const IOIO_ADDR_64: u32 = 1 << 9;
const IOIO_ADDR_32: u32 = 1 << 8;
const IOIO_ADDR_16: u32 = 1 << 7;

const IOIO_DATA_32: u32 = 1 << 6;
const IOIO_DATA_16: u32 = 1 << 5;
const IOIO_DATA_8: u32 = 1 << 4;

const IOIO_SEG_ES: u32 = 0 << 10;
const IOIO_SEG_DS: u32 = 3 << 10;

pub fn vc_ioio_exitinfo(exit_info_1: &mut u64, insn: &Instruction, sf: &mut PtRegs) -> VCResult {
    let code = insn.opcode();
    let port: u64;
    let mut addr_size = 64;
    let mut data_size = 0;
    let mut addr_override = false;
    let mut operand_overide = false;
    let mut has_rep = false;
    *exit_info_1 = 0;
    for i in 0..4 {
        let prefix = (sf.rip + i) as *const u8;
        unsafe {
            match *prefix {
                0x66 => operand_overide = true,
                0x67 => addr_override = true,
                0xf0 | 0xf2 | 0xf3 => has_rep = true,
                _ => break,
            }
        }
    }
    match code {
        /* INS opcodes */
        Opcode::INS => {
            *exit_info_1 |= IOIO_TYPE_INS as u64;
            port = sf.rdx & 0xffff;
            match insn.operand(1) {
                Operand::Register(r) => {
                    let class = r.class();
                    if class == RegSpec::al().class() {
                        data_size = 8;
                    } else if class == RegSpec::ax().class() {
                        if operand_overide {
                            data_size = 16;
                        } else {
                            data_size = 32;
                        }
                    }
                }
                _ => (),
            }
        }
        /* OUTS opcodes */
        Opcode::OUTS => {
            *exit_info_1 |= IOIO_TYPE_OUTS as u64;
            *exit_info_1 |= IOIO_SEG_DS as u64;
            port = sf.rdx & 0xffff;
            match insn.operand(0) {
                Operand::Register(r) => {
                    let class = r.class();
                    if class == RegSpec::al().class() {
                        data_size = 8;
                    } else if class == RegSpec::ax().class() {
                        if operand_overide {
                            data_size = 16;
                        } else {
                            data_size = 32;
                        }
                    }
                }
                _ => (),
            }
        }
        Opcode::IN => {
            *exit_info_1 |= IOIO_TYPE_IN as u64;
            match insn.operand(1) {
                /* IN immediate opcodes */
                Operand::ImmediateU8(imm) => port = imm as u64 & 0xffff,
                /* IN register opcodes */
                Operand::Register(_) => port = sf.rdx & 0xffff,
                _ => return VCResult::EsDecodeFailed,
            }
            match insn.operand(0) {
                Operand::Register(r) => {
                    let class = r.class();
                    if class == RegSpec::al().class() {
                        data_size = 8;
                    } else if class == RegSpec::ax().class() {
                        data_size = 16;
                    } else if class == RegSpec::eax().class() {
                        data_size = 32;
                    }
                }
                _ => return VCResult::EsDecodeFailed,
            }
        }
        Opcode::OUT => {
            *exit_info_1 |= IOIO_TYPE_OUT as u64;
            match insn.operand(0) {
                /* OUT immediate opcodes */
                Operand::ImmediateU8(imm) => port = imm as u64 & 0xffff,
                /* OUT register opcodes */
                Operand::Register(_) => port = sf.rdx & 0xffff,
                _ => return VCResult::EsDecodeFailed,
            }
            match insn.operand(1) {
                Operand::Register(r) => {
                    let class = r.class();
                    if class == RegSpec::al().class() {
                        data_size = 8;
                    } else if class == RegSpec::ax().class() {
                        data_size = 16;
                    } else if class == RegSpec::eax().class() {
                        data_size = 32;
                    }
                }
                _ => return VCResult::EsDecodeFailed,
            }
        }
        _ => return VCResult::EsDecodeFailed,
    }

    *exit_info_1 |= port << 16;

    match data_size {
        8 => {
            *exit_info_1 |= IOIO_DATA_8 as u64;
        }
        16 => {
            *exit_info_1 |= IOIO_DATA_16 as u64;
        }
        32 => {
            *exit_info_1 |= IOIO_DATA_32 as u64;
        }

        _ => return VCResult::EsDecodeFailed,
    }

    if addr_override {
        addr_size = 32;
    }
    match addr_size {
        //16 => *exit_info_1 |= IOIO_ADDR_16 as u64,
        32 => *exit_info_1 |= IOIO_ADDR_32 as u64,
        64 => *exit_info_1 |= IOIO_ADDR_64 as u64,
        _ => return VCResult::EsDecodeFailed,
    }

    if has_rep {
        *exit_info_1 |= IOIO_REP as u64;
    }

    return VCResult::EsOk;
}

pub fn vc_handle_ioio(ghcb: &mut GhcbHandle, insn: &Instruction, sf: &mut PtRegs) -> VCResult {
    let mut exit_info_1: u64 = 0;
    let exit_info_2;
    let mut ret;
    ret = vc_ioio_exitinfo(&mut exit_info_1, insn, sf);
    if ret != VCResult::EsOk {
        return ret;
    }

    if (exit_info_1 & IOIO_TYPE_STR as u64) != 0 {
        /* (REP) INS/OUTS */

        let df = (sf.eflags & RFLAGS_DF) == RFLAGS_DF;
        let io_bytes = (exit_info_1 >> 4) & 0x7;
        let ghcb_count = GHCB_SHARED_BUFFER_SIZE / io_bytes;
        let op_count = if (exit_info_1 & IOIO_REP as u64) != 0 {
            sf.rcx
        } else {
            1
        };
        exit_info_2 = op_count.min(ghcb_count);
        let exit_bytes = exit_info_2 * io_bytes;

        if !((exit_info_1 & IOIO_TYPE_IN as u64) != 0) {
            for i in 0..exit_info_2 {
                unsafe {
                    if df {
                        ptr::copy_nonoverlapping(
                            (sf.rsi - (i * io_bytes)) as *const u8,
                            (ghcb.shared_buffer_addr() + (i * io_bytes)) as *mut u8,
                            io_bytes as usize,
                        );
                    } else {
                        ptr::copy_nonoverlapping(
                            (sf.rsi + (i * io_bytes)) as *const u8,
                            (ghcb.shared_buffer_addr() + (i * io_bytes)) as *mut u8,
                            io_bytes as usize,
                        );
                    }
                }
            }
        }
        /*
         * Issue an VMGEXIT to the HV to consume the bytes from the
         * shared buffer or to have it write them into the shared buffer
         * depending on the instruction: OUTS or INS.
         */
        ghcb.set_sw_scratch(ghcb.shared_buffer_addr());
        unsafe {
            match ghcb.vmgexit(SVMExitDef::SVM_EXIT_IOIO, exit_info_1, exit_info_2) {
                Ok(_) => ret = VCResult::EsOk,
                Err(GhcbError::Exception(e)) => ret = VCResult::EsException(e),
                Err(GhcbError::VmmError) => ret = VCResult::EsVmmError,
            }
        }
        if ret != VCResult::EsOk {
            return ret;
        }

        /* Read bytes from shared buffer into the guest's destination. */
        if (exit_info_1 & IOIO_TYPE_IN as u64) != 0 {
            for i in 0..exit_info_2 {
                if df {
                    unsafe {
                        ptr::copy_nonoverlapping(
                            (ghcb.shared_buffer_addr() + (i * io_bytes)) as *const u8,
                            (sf.rsi - (i * io_bytes)) as *mut u8,
                            io_bytes as usize,
                        );
                    }
                } else {
                    unsafe {
                        ptr::copy_nonoverlapping(
                            (ghcb.shared_buffer_addr() + (i * io_bytes)) as *const u8,
                            (sf.rsi + (i * io_bytes)) as *mut u8,
                            io_bytes as usize,
                        );
                    }
                }
            }

            if df {
                sf.rdi -= exit_bytes;
            } else {
                sf.rdi += exit_bytes;
            }
        } else {
            if df {
                sf.rsi -= exit_bytes;
            } else {
                sf.rsi += exit_bytes;
            }
        }

        if (exit_info_1 & IOIO_REP as u64) != 0 {
            sf.rcx -= exit_info_2;
        }
        if sf.rcx != 0 {
            ret = VCResult::EsRetry;
        }
    } else {
        /* IN/OUT into/from rAX */
        let bits = ((exit_info_1 & 0x70) >> 1) as i32;
        let mut rax = 0;
        if (!(exit_info_1 & IOIO_TYPE_IN as u64)) != 0 {
            rax = lower_bits(sf.rax, bits);
        }
        ghcb.set_rax(rax);
        unsafe {
            match ghcb.vmgexit(SVMExitDef::SVM_EXIT_IOIO, exit_info_1, 0) {
                Ok(_) => ret = VCResult::EsOk,
                Err(GhcbError::Exception(e)) => ret = VCResult::EsException(e),
                Err(GhcbError::VmmError) => ret = VCResult::EsVmmError,
            }
        }
        if ret != VCResult::EsOk {
            return ret;
        }

        if (exit_info_1 & IOIO_TYPE_IN as u64) != 0 {
            if !ghcb.check_rax_valid() {
                return VCResult::EsVmmError;
            }
            sf.rax = lower_bits(ghcb.Rax(), bits);
        }
    }
    return ret;
}

pub fn vc_handle_cpuid(sf: &mut PtRegs) -> VCResult {
    let cpuid = &CpuidPage::from(MemoryDef::CPUID_PAGE);
    let cpuid_result = cpuid.check_cpuid(sf.rax as u32, sf.rcx as u32);
    sf.rax = cpuid_result.eax as u64;
    sf.rbx = cpuid_result.ebx as u64;
    sf.rcx = cpuid_result.ecx as u64;
    sf.rdx = cpuid_result.edx as u64;
    return VCResult::EsOk;
}

pub fn vc_handle_rdtsc(ghcb: &mut GhcbHandle, errorCode: u64, sf: &mut PtRegs) -> VCResult {
    let rdtscp = errorCode == SVMExitDef::SVM_EXIT_RDTSCP;
    let ret;
    unsafe {
        match ghcb.vmgexit(errorCode, 0, 0) {
            Ok(_) => ret = VCResult::EsOk,
            Err(GhcbError::Exception(e)) => ret = VCResult::EsException(e),
            Err(GhcbError::VmmError) => ret = VCResult::EsVmmError,
        }
    }
    if ret != VCResult::EsOk {
        return ret;
    }

    let rax_valid = ghcb.check_rax_valid();
    let rdx_valid = ghcb.check_rdx_valid();
    let rcx_valid = ghcb.check_rcx_valid();

    if !(rax_valid && rdx_valid && (!rdtscp || rcx_valid)) {
        return VCResult::EsVmmError;
    }

    sf.rax = ghcb.Rax();
    sf.rdx = ghcb.Rdx();
    if rdtscp {
        sf.rcx = ghcb.Rcx();
    }

    return VCResult::EsOk;
}

pub fn vc_handle_rdpmc(ghcb: &mut GhcbHandle, sf: &mut PtRegs) -> VCResult {
    ghcb.set_rcx(sf.rcx);
    let ret;
    unsafe {
        match ghcb.vmgexit(SVMExitDef::SVM_EXIT_RDPMC, 0, 0) {
            Ok(_) => ret = VCResult::EsOk,
            Err(GhcbError::Exception(e)) => ret = VCResult::EsException(e),
            Err(GhcbError::VmmError) => ret = VCResult::EsVmmError,
        }
    }
    if ret != VCResult::EsOk {
        return ret;
    }

    let rax_valid = ghcb.check_rax_valid();
    let rdx_valid = ghcb.check_rdx_valid();

    if !(rax_valid && rdx_valid) {
        return VCResult::EsVmmError;
    }

    sf.rax = ghcb.Rax();
    sf.rdx = ghcb.Rdx();
    return VCResult::EsOk;
}

fn lower_bits(val: u64, bits: i32) -> u64 {
    let mask = (1u64 << bits) - 1;
    val & mask
}
