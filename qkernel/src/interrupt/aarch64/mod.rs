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

pub mod fault;
pub mod sysreg;
use core::arch::asm;
use super::super::syscall_dispatch_aarch64;
use crate::qlib::linux_def::MmapProt;
use crate::qlib::addr::AccessType;
use crate::qlib::kernel::task;
use crate::qlib::kernel::threadmgr::task_sched::SchedState;
use crate::qlib::kernel::SignalDef::PtRegs;
use crate::qlib::vcpu_mgr::*;
use crate::kernel_def;
use self::fault::{PageFaultHandler, PageFaultErrorCode};

pub unsafe fn InitSingleton() {
}

pub fn init() {
    // TODO set up GIC
}

pub struct EsrDefs{}
pub struct ISSDataAbort {}
pub struct ISSInstrAbort {}

// TODO move the defs to another module if it's too big.
impl EsrDefs{
    // ESR defs for Exception Class
    pub const ESR_IL:u64            = 0x01 << 25;
    pub const ESR_EC_SHIFT:u64      = 26;
    pub const ESR_EC_MASK:u64       = 0x3f << 26;
    pub const EC_UNKNOWN:u64        = 0x00;    /* Unkwn exception */
    pub const EC_FP_SIMD:u64        = 0x07;    /* FP/SIMD trap */
    pub const EC_BRANCH_TGT:u64     = 0x0d;    /* Branch target exception */
    pub const EC_ILL_STATE:u64      = 0x0e;    /* Illegal execution state */
    pub const EC_SVC:u64            = 0x15;    /* SVC trap */
    pub const EC_MSR:u64            = 0x18;    /* MSR/MRS trap */
    pub const EC_FPAC:u64           = 0x1c;    /* Faulting PAC trap */
    pub const EC_INSN_ABORT_L:u64   = 0x20;    /* Instruction abort, from lower EL */
    pub const EC_INSN_ABORT:u64     = 0x21;    /* Instruction abort, from same EL */ 
    pub const EC_PC_ALIGN:u64       = 0x22;    /* PC alignment fault */
    pub const EC_DATA_ABORT_L:u64   = 0x24;    /* Data abort, from lower EL */
    pub const EC_DATA_ABORT:u64     = 0x25;    /* Data abort, from same EL */ 
    pub const EC_SP_ALIGN:u64       = 0x26;    /* SP alignment fault */
    pub const EC_TRAP_FP:u64        = 0x2c;    /* Trapped FP exception */
    pub const EC_SERROR:u64         = 0x2f;    /* SError interrupt */
    pub const EC_SOFTSTP_EL0:u64    = 0x32;    /* Software Step, from lower EL */
    pub const EC_SOFTSTP_EL1:u64    = 0x33;    /* Software Step, from same EL */
    pub const EC_WATCHPT_EL1:u64    = 0x35;    /* Watchpoint, from same EL */
    pub const EC_BRK:u64            = 0x3c;    /* Breakpoint */

    pub const ESR_CM:u64            = 0x01 << 8;     /* Cache Maintenance */
    pub const ESR_WNR:u64           = 0x01 << 6;     /* Write not read */

    // ESR defs for ISS
    pub const ISS_FNV:u64       = 0x01 << 10;      /* FAR not valid */
    pub const ISS_DFSC_MASK:u64 = 0x3f << 0;    /* DFSC and IFSC masks are the same */
    // Access Size Fault L0~3
    pub const DFSC_ASF_L0:u64  = 0x0;
    pub const DFSC_ASF_L1:u64  = 0x1;
    pub const DFSC_ASF_L2:u64  = 0x2;
    pub const DFSC_ASF_L3:u64  = 0x3;

    // Translation Fault L0~3
    pub const DFSC_TF_L0:u64   = 0x4;
    pub const DFSC_TF_L1:u64   = 0x5;
    pub const DFSC_TF_L2:u64   = 0x6;
    pub const DFSC_TF_L3:u64   = 0x7;

    // Access Flag Fault L0~3
    pub const DFSC_AF_L0:u64   = 0x8;      /* if FEAT_LPA2 is implmented */
    pub const DFSC_AF_L1:u64   = 0x9;
    pub const DFSC_AF_L2:u64   = 0xa;
    pub const DFSC_AF_L3:u64   = 0xb;

    // Permission Fault L0 ~ 3
    pub const DFSC_PF_L0:u64   = 0xc;      /* if FEAT_LPA2 is implemented */
    pub const DFSC_PF_L1:u64   = 0xd;
    pub const DFSC_PF_L2:u64   = 0xe;
    pub const DFSC_PF_L3:u64   = 0xf;

    // Synchronous External Abort (SEA)
    // not on translation table walk
    pub const DFSC_SEA:u64      = 0x10;
    // on translation table walk, L -1~3
    pub const DFSC_SEA_M1:u64   = 0x13;     /* if FEAT_LPA2 is implemented*/
    pub const DFSC_SEA_L0:u64  = 0x14;
    pub const DFSC_SEA_L1:u64  = 0x15;
    pub const DFSC_SEA_L2:u64  = 0x16;
    pub const DFSC_SEA_L3:u64  = 0x17;

    // Synchronous parity or ECC error when FEAT_RAS NOT implemented.
    // not on table walk
    pub const DFSC_ECC:u64      = 0x18;
    // on table walk: L-1~3
    pub const DFSC_ECC_M1:u64   = 0x1b;
    pub const DFSC_ECC_L0:u64  = 0x1c;
    pub const DFSC_ECC_L1:u64  = 0x1d;
    pub const DFSC_ECC_L2:u64  = 0x1e;
    pub const DFSC_ECC_L3:u64  = 0x1f;

    pub const DFSC_ALIGN:u64    = 0x21;

    // the Granule Protection Faults
    // GPF on table walk, L-1~3
    pub const DFSC_GPF_M1:u64   = 0x23;
    pub const DFSC_GPF_L0:u64  = 0x23;
    pub const DFSC_GPF_L1:u64  = 0x25;
    pub const DFSC_GPF_L2:u64  = 0x26;
    pub const DFSC_GPF_L3:u64  = 0x27;
    // GPF not on table walk
    pub const DFSC_GPF:u64      = 0x28;

    pub const DFSC_ASF_M1:u64   = 0x29;
    pub const DFSC_TF_M1:u64    = 0x2b;

    pub const DFSC_TLB_CONFLICT:u64 = 0x30;
    // unsupported atomic hardware update
    pub const DFSC_UAHU:u64     = 0x31;

    #[inline]
    pub fn GetExceptionFromESR(esr:u64) -> u64{
        return (esr & EsrDefs::ESR_EC_MASK) >> EsrDefs::ESR_EC_SHIFT;
    }

    #[inline]
    pub fn IsCM(esr:u64) -> bool {
        return (esr & EsrDefs::ESR_CM) != 0;
    }

    #[inline]
    pub fn IsWnR(esr:u64) -> bool {
        return (esr & EsrDefs::ESR_WNR) != 0;
    }
}

pub fn GetEsrEL1() -> u64 {
    unsafe {
        let value:u64;
        asm!(
            "mrs  {}, esr_el1",
            out(reg) value,
           );
        return value;
    }
}

// TODO, the FAR is only valid when ESR.ISS.FnV==0
// For faults other than data/instr aborts, this flag
// should be checked before using FAR.
pub fn GetFarEL1() -> u64 {
    unsafe {
        let value:u64;
        asm!(
            "mrs  {}, far_el1",
            out(reg) value,
           );
        return value;
    }
}

pub fn GetException() -> u64{
    let esr:u64 = GetEsrEL1();
    return EsrDefs::GetExceptionFromESR(esr);
}

#[no_mangle]
pub extern "C" fn exception_handler_unhandled(_ptregs_addr:usize, exception_type:usize){
    // MUST CHECK ptr!=0 before dereferencing
    // ptr == 0 indicates an empty entry in the exception table,
    // in which case the context won't be saved/restored by the wrapper
    // and this function MUST NOT return
    panic!("unhandled exception - {}",
           match exception_type {
               0 => "EL1T_SYN",
               1 => "EL1T_IRQ",
               2 => "EL1T_FIQ",
               3 => "EL1T_SERROE",
               4 => "EL1H_SYN",
               5 => "EL1H_IRQ",
               6 => "EL1H_FIQ",
               7 => "EL1H_SERROE",
               _ => "NON-DEFINED",
          }
    );
}

#[no_mangle]
pub extern "C" fn exception_handler_el1h_sync(ptregs_addr:usize){
    let esr = GetEsrEL1();
    let ec = EsrDefs::GetExceptionFromESR(esr);

    match ec {
        EsrDefs::EC_DATA_ABORT => {
            let far = GetFarEL1();
            MemAbortKernel(ptregs_addr, esr, far, false);
        },
        EsrDefs::EC_INSN_ABORT => {
            let far = GetFarEL1();
            MemAbortKernel(ptregs_addr, esr, far, true);
        },
        _ => {
            panic!("unhandled sync exception from el1: {}\n", ec);
        }
    }
}

#[no_mangle]
pub extern "C" fn exception_handler_el1h_irq(ptregs_addr:usize){
    return exception_handler_unhandled(ptregs_addr, 5);
}
#[no_mangle]
pub extern "C" fn exception_handler_el1h_fiq(ptregs_addr:usize){
    return exception_handler_unhandled(ptregs_addr, 6);
}
#[no_mangle]
pub extern "C" fn exception_handler_el1h_serror(ptregs_addr:usize){
    return exception_handler_unhandled(ptregs_addr, 7);
}

#[no_mangle]
pub extern "C" fn exception_handler_el0_sync(ptregs_addr: usize) {
    let currTask = task::Task::Current();
    currTask.AccountTaskLeave(SchedState::RunningApp);
    currTask.SaveTLS();
    let esr = GetEsrEL1();
    let ec = EsrDefs::GetExceptionFromESR(esr);
    if ptregs_addr == 0 {
        panic!("exception frame is null pointer\n")
    }
    // arm64 linux syscall calling convention
    // TODO maybe there is a better/safer way of this pointer cast
    let ctx_p = ptregs_addr as *mut PtRegs;
    let ctx_p = ctx_p.cast::<PtRegs>();
    let ctx = unsafe { &mut *ctx_p };
    match ec {
        EsrDefs::EC_SVC => {
            // syscall number from w8
            let call_no = ctx.regs[8] as u32;
            let arg0 = ctx.regs[0];
            let arg1 = ctx.regs[1];
            let arg2 = ctx.regs[2];
            let arg3 = ctx.regs[3];
            let arg4 = ctx.regs[4];
            let arg5 = ctx.regs[5];
            // write syscall ret to x0
            ctx.regs[0] = syscall_dispatch_aarch64(call_no, arg0, arg1, arg2, arg3, arg4, arg5);
            // TODO do we need to write the "second ret val" back to x1?
        },
        EsrDefs::EC_DATA_ABORT_L => {
            let far = GetFarEL1();
            MemAbortUser(ptregs_addr, esr, far, false);
        },
        EsrDefs::EC_INSN_ABORT_L => {
            let far = GetFarEL1();
            MemAbortUser(ptregs_addr, esr, far, true);
        },
        EsrDefs::EC_UNKNOWN => match sysreg::try_emulate_mrs(ctx.pc) {
            sysreg::SysmovResult::ReadSuccess(val, xt) => {
                debug!("el0 mrs emulated: set X{} to 0x{:x}", xt, val);
                ctx.regs[xt as usize] = val;
                ctx.pc += 4;
            }
            _ => {
                unsafe {
                     if let Some(opcode) = kernel_def::read_user_opcode(ctx.pc) {
                         debug!("VM: UNKNOWN_EXCEPTION from EL0, current-PC: {:#x}, retrieved PC[opcode]:{:#x}.", ctx.pc, opcode);
                     } else {
                         debug!("VM: UNKNOWN_EXCEPTION from EL0, current-PC: {:#x}, can not retrieve PC[opcode].", ctx.pc);
                     }
                }
                panic!("VM: panic on UNKNOWN_EXCEPTION from EL0 - current-context: {:?}", ctx);
            }
        },
        _ => {
            panic!(
                "unhandled sync exception from el0: {},\n current-context: {:?}",
                ec, ctx
            );
        } // TODO (default case) for a unhandled exception from user,
          // the kill the user process instead of panicing
    }
    CPULocal::Myself().SetMode(VcpuMode::User);
}

#[no_mangle]
pub extern "C" fn exception_handler_el0_irq(){return;}
#[no_mangle]
pub extern "C" fn exception_handler_el0_fiq(){return;}
#[no_mangle]
pub extern "C" fn exception_handler_el0_serror(){return;}

#[inline]
pub fn GetFaultAccessType(esr:u64, is_exe:bool) -> AccessType{
    if is_exe {
        return AccessType(MmapProt::PROT_EXEC);
    }
    if !EsrDefs::IsCM(esr) && EsrDefs::IsWnR(esr){
        return AccessType(MmapProt::PROT_WRITE)
    }
    return AccessType(MmapProt::PROT_READ);

}

pub fn MemAbortUser(ptregs_addr:usize, esr:u64, far:u64, is_instr:bool){
    debug!("get {} abort fault from el0",
           match is_instr {
               true  => "instruction",
               false => "data",
           });
    let dfsc = esr & EsrDefs::ISS_DFSC_MASK;
    let access_type = GetFaultAccessType(esr, is_instr);
    let dfsc_root = dfsc & PageFaultErrorCode::GEN_xxSC_MASK;
    match dfsc_root {
        PageFaultErrorCode::GEN_PERMISSION_FAULT |
        PageFaultErrorCode::GEN_TRANSLATION_FAULT|
        PageFaultErrorCode::GEN_ACCESS_FLAG_FAULT => {
            info!("DFSC/IFSC == {:#X}, FAR == {:#X}, acces-type fault == {}, \
                  during address translation == {}.", dfsc, far,
                  access_type.String(), EsrDefs::IsCM(esr));
            let ptregs_ptr = ptregs_addr as *mut PtRegs;
            let ptregs = unsafe {
                &mut *ptregs_ptr
            };
            //
            //TODO: on work
            //
            let error_code = PageFaultErrorCode::new(true, esr);
            PageFaultHandler(ptregs , far, error_code);
            return;
        },
        _ => {
            // TODO insert proper handler
            panic!("DFSC/IFSC == 0x{:02x}, FAR == {:#x}, acces-type fault = {},\
                   during address translation: {} ",  dfsc, far, access_type.String(),
                   match EsrDefs::IsCM(esr) {
                       true => "Yes",
                       false => "No",
                   });
        },
    }
}

pub fn MemAbortKernel(ptregs_addr:usize, esr:u64, far:u64, is_instr:bool){
    debug!("get {} abort fault from el1",
           match is_instr {
               true  => "instruction",
               false => "data",
           });
    let dfsc = esr & EsrDefs::ISS_DFSC_MASK;
    let access_type = GetFaultAccessType(esr, is_instr);
    let dfsc_root = dfsc & PageFaultErrorCode::GEN_xxSC_MASK;
    match dfsc_root {
        PageFaultErrorCode::GEN_PERMISSION_FAULT |
        PageFaultErrorCode::GEN_TRANSLATION_FAULT|
        PageFaultErrorCode::GEN_ACCESS_FLAG_FAULT => {
            info!("DFSC/IFSC == {:#X}, FAR == {:#X}, acces-type fault == {}, \
                  during address translation == {}.", dfsc, far,
                  access_type.String(), EsrDefs::IsCM(esr));
            let ptregs_ptr = ptregs_addr as *mut PtRegs;
            let ptregs = unsafe {
                &mut *ptregs_ptr
            };
            let error_code = PageFaultErrorCode::new(false, esr);
            PageFaultHandler(ptregs , far, error_code);
            return;
        },
        _ => {
            // TODO insert proper handler
            panic!("DFSC/IFSC == 0x{:02x}, FAR == {:#x}, acces-type fault = {},\
                   during address translation: {} ",  dfsc, far, access_type.String(),
                   match EsrDefs::IsCM(esr) {
                       true => "Yes",
                       false => "No",
                   });
        },
    }
}
