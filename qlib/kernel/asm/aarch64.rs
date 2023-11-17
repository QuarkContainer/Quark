// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
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

//use super::super::perf_tunning::*;

use core::arch::asm;
use core::sync::atomic::Ordering;

use bitflags::bitflags;

use crate::qlib::kernel::task;
use crate::qlib::vcpu_mgr::CPULocal;

pub const _PSR_D_BIT:u64      = 0x00000200;
pub const _PSR_A_BIT:u64      = 0x00000100;
pub const _PSR_I_BIT:u64      = 0x00000080;
pub const _PSR_F_BIT:u64      = 0x00000040;
pub const _PSR_DAIF_SHIFT:u64 = 6;
pub const _PSR_DAIF_MASK:u64  = 0xf << _PSR_DAIF_SHIFT;

// PSR bits.
pub const _PSR_MODE_EL0t:u64 = 0x00000000;
pub const _PSR_MODE_EL1t:u64 = 0x00000004;
pub const _PSR_MODE_EL1h:u64 = 0x00000005;
pub const _PSR_MODE_MASK:u64 = 0x0000000f;

pub const PsrFlagsClear:u64 = _PSR_MODE_MASK | _PSR_DAIF_MASK;
pub const PsrModeMask:u64   = _PSR_MODE_MASK;

// KernelFlagsSet should always be set in the kernel.
pub const KernelFlagsSet:u64 = _PSR_MODE_EL1h | _PSR_D_BIT | _PSR_A_BIT | _PSR_I_BIT | _PSR_F_BIT;

// UserFlagsSet are always set in userspace.
pub const UserFlagsSet:u64 = _PSR_MODE_EL0t;

#[inline]
pub fn WriteMsr(msr: u32, value: u64) {

}

#[inline]
pub fn ReadMsr(msr: u32) -> u64 {
    0
}

#[inline]
pub fn SwapGs() {
}

pub fn GetVcpuId() -> usize {
    let ret: u64;
    unsafe {
        asm!("\
        mrs {0}, tpidr_el1
        ldr {1}, [{0}, #16]",
        out(reg) _,
        out(reg) ret);
    };
    return ret as usize;
}

#[inline]
pub fn Hlt() {
}

#[inline]
pub fn LoadUserTable(table: u64) {
    unsafe {
        asm!("msr ttbr0_el1, {0}
        ic iallu
        dsb nsh
        isb",
        in(reg) table);
    };
}

#[inline]
pub fn CurrentUserTable() -> u64 {
    let table: u64;
    unsafe {
        asm!("mrs {0}, ttbr0_el1", out(reg) table);
    };
    return table;
}

#[inline]
pub fn CurrentKernelTable() -> u64 {
    let table: u64;
    unsafe {
        asm!("mrs {0}, ttbr1_el1", out(reg) table);
    };
    return table;
}

#[inline]
pub fn EnterUser(entry: u64, userStackAddr: u64, kernelStackAddr: u64) -> ! {
    let currTask = task::Task::Current();
    let pt = currTask.GetPtRegs();
    CPULocal::SetKernelStack(kernelStackAddr);
    CPULocal::SetUserStack(userStackAddr);
    *pt = Default::default();

    pt.pc = entry;
    pt.sp = userStackAddr;
    pt.pstate &= !PsrFlagsClear;
    pt.pstate |= UserFlagsSet;
    IRet(pt as *const _ as u64);
}

#[inline]
pub fn SyscallRet(kernelRsp: u64) -> ! {
    unsafe {
        asm!(
            "nop"
        );
        panic!("won't reach");
    }
}

#[inline]
pub fn IRet(kernelRsp: u64) -> ! {
    unsafe {
        asm!("\
              mov sp, x0
              ldp x30, x23, [sp, #16*15]
              ldp x21, x22, [sp, #16*16]
              msr sp_el0, x23
              msr elr_el1, x21
              msr spsr_el1, x22
              ldp  x0,  x1,  [sp, #16*0]
              ldp  x2,  x3,  [sp, #16*1]
              ldp  x4,  x5,  [sp, #16*2]
              ldp  x6,  x7,  [sp, #16*3]
              ldp  x8,  x9,  [sp, #16*4]
              ldp x10, x11,  [sp, #16*5]
              ldp x12, x13,  [sp, #16*6]
              ldp x14, x15,  [sp, #16*7]
              ldp x16, x17,  [sp, #16*8]
              ldp x18, x19,  [sp, #16*9]
              ldp x20, x21, [sp, #16*10]
              ldp x22, x23, [sp, #16*11]
              ldp x24, x25, [sp, #16*12]
              ldp x26, x27, [sp, #16*13]
              ldp x28, x29, [sp, #16*14]
              add sp, sp, #16*17
              eret
            ",
            in("x0") kernelRsp);
        panic!("won't reach");
    }
}

#[inline]
pub fn GetCurrentKernelSp() -> u64 {
    // we can only mrs sp_el1 in EL2 and EL3
    // so we can only get sp_el1 by move
    let ret: u64;
    unsafe {
        asm!("\
        mrs {1}, spsel
        msr spsel, #1
        mov {0}, sp
        msr spsel, {1}
        ",
        out(reg) ret,
        out(reg) _);
    }
    ret
}

#[inline]
pub fn GetCurrentUserSp() -> u64 {
    return sp_el0();
}

#[inline]
pub fn Clflush(addr: u64) {
}

// HostID executes a native CPUID instruction.
// return (ax, bx, cx, dx)
pub fn AsmHostID(axArg: u32, cxArg: u32) -> (u32, u32, u32, u32) {
    return (0, 0, 0, 0);
}

#[inline(always)]
pub fn ReadBarrier() {
    unsafe {
        asm!("dsb ishld")
    };
}

#[inline(always)]
pub fn WriteBarrier() {
    unsafe {
        asm!("dsb ishst")
    };
}

#[inline(always)]
pub fn GetCpu() -> u32 {
    let rcx: u64 = 0;
//    unsafe {
//        asm!("\
//            rdtscp
//            ",
//            out("rcx") rcx
//        )
//    };

    return (rcx & 0xfff) as u32;
}

#[inline(always)]
pub fn GetRflags() -> u64 {
    let rax: u64 = 0;
//    unsafe {
//        asm!("\
//                pushfq                  # push eflags into stack
//                pop rax                 # pop it into rax
//            ",
//            out("rax") rax
//        )
//    };

    return rax;
}

#[inline(always)]
pub fn SetRflags(val: u64) {
//    unsafe {
//        asm!("\
//                push rax
//                popfq
//            ",
//            in("rax") val)
//    };
}

pub fn SaveFloatingPoint(addr: u64) {
//    if SUPPORT_XSAVEOPT.load(Ordering::Acquire) {
//        xsaveopt(addr);
//    } else if SUPPORT_XSAVE.load(Ordering::Acquire) {
//        xsave(addr);
//    } else {
//        fxsave(addr);
//    }
//    NOTE Arm does not seem to have direct aquivalents for the above instructions: xsaveopt/xsave/fxsave.
//          As both SUPPORT_XSAVEOPT/XSAVE are per default not taken,
//          only an fxsave is emulated.

            fxsave(addr);
}

//    NOTE Arm does not seem to have direct aquivalents for the above instructions: xrstor/fxrstor.
//          As SUPPORT_XSAVE is per default not taken,
//          only an fxrstor is emulated.
pub fn LoadFloatingPoint(addr: u64) {
        fxrstor(addr);
}

pub fn xsave(addr: u64) {
//    unsafe {
//        asm!("\
//                xsave64 [rdi + 0]
//            ",
//            in("rdi") addr, )
//    };
}

pub fn xsaveopt(addr: u64) {
//    let negtive1: u64 = 0xffffffff;
//    unsafe {
//        asm!("\
//                xsaveopt64 [rdi + 0]
//            ",
//            in("rdi") addr, 
//            in("eax") negtive1,
//            in("edx") negtive1)
//    };
}

pub fn xrstor(addr: u64) {
//    let negtive1: u64 = 0xffffffff;
//    unsafe {
//        asm!("\
//                xrstor64 [rdi + 0]
//            ",
//            in("rdi") addr,
//            in("eax") negtive1,
//            in("edx") negtive1)
//    };
}

//  FPCR;FPSR Registers are saved placed after 'q15'.
pub fn fxsave(addr: u64) {
    unsafe {
        asm!("\
              mrs x1, FPCR
              mrs x2, FPSR
              stp  q0,  q1,  [x0, #32*0]
              stp  q2,  q3,  [x0, #32*1]
              stp  q4,  q5,  [x0, #32*2]
              stp  q6,  q7,  [x0, #32*3]
              stp  q8,  q9,  [x0, #32*4]
              stp q10, q11,  [x0, #32*5]
              stp q12, q13,  [x0, #32*6]
              stp q14, q15,  [x0, #32*7]
              stp q16, q17,  [x0, #32*8]
              stp q18, q19,  [x0, #32*9]
              stp q20, q21, [x0, #32*10]
              stp q22, q23, [x0, #32*11]
              stp q24, q25, [x0, #32*12]
              stp q26, q27, [x0, #32*13]
              stp q28, q29, [x0, #32*14]
              stp q30, q31, [x0, #32*15]!
              stp  x1,  x2, [x0, #32*1]
            ",
            in("x0") addr,
            // Let compiler know about clobbered registers
            out("x1") _,
            out("x2") _,)
    };
}

pub fn fxrstor(addr: u64) {
    unsafe {
        asm!("\
              ldp  q0,  q1,  [x0, #32*0]
              ldp  q2,  q3,  [x0, #32*1]
              ldp  q4,  q5,  [x0, #32*2]
              ldp  q6,  q7,  [x0, #32*3]
              ldp  q8,  q9,  [x0, #32*4]
              ldp q10, q11,  [x0, #32*5]
              ldp q12, q13,  [x0, #32*6]
              ldp q14, q15,  [x0, #32*7]
              ldp q16, q17,  [x0, #32*8]
              ldp q18, q19,  [x0, #32*9]
              ldp q20, q21, [x0, #32*10]
              ldp q22, q23, [x0, #32*11]
              ldp q24, q25, [x0, #32*12]
              ldp q26, q27, [x0, #32*13]
              ldp q28, q29, [x0, #32*14]
              ldp q30, q31, [x0, #32*15]!
              ldp  x1,  x2, [x0, #32*1]
              msr FPCR, x1
              msr FPSR, x2
            ",
            in("x0") addr,
            // Let compiler know about clobbered registers
            out("x1") _,
            out("x2") _,)
    };
}

#[inline(always)]
pub fn mfence() {
    unsafe {
        asm!(
            "
            dsb ish
        "
        )
    }
}

#[inline(always)]
pub fn sfence() {
    unsafe {
        asm!(
            "
            dsb ishst
        "
        )
    }
}

#[inline(always)]
pub fn lfence() {
    unsafe {
        asm!(
            "
            dmb ishld
        "
        )
    }
}

pub fn stmxcsr(addr: u64) {
//    unsafe {
//        asm!("\
//                STMXCSR [rax]
//            ",
//            in("rax") addr)
//    };
}

pub fn ldmxcsr(addr: u64) {
//    unsafe {
//        asm!("\
//                LDMXCSR [rax]
//            ",
//            in("rax") addr)
//    };
}

pub fn FSTCW(addr: u64) {
//    unsafe {
//        asm!("\
//                FSTCW [rax]
//            ",
//            in("rax") addr
//        )
//    };
}

pub fn FLDCW(addr: u64) {
//    unsafe {
//        asm!("\
//                FLDCW [rax]
//            ",
//            in("rax") addr)
//    };
}

pub fn FNCLEX() {
//    unsafe {
//        asm!(
//            "\
//            FNCLEX
//        "
//        )
//    };
}

pub fn fninit() {
//    unsafe {
//        asm!(
//            "\
//            fninit
//            "
//        )
//    };
}

pub fn xsetbv(val: u64) {
//    let reg = 0u64;
//    let val_l = val & 0xffff;
//    let val_h = val >> 32;
//    unsafe {
//        asm!("\
//                xsetbv
//            ",
//            in("rcx") reg,
//            in("edx") val_h,
//            in("eax") val_l,
//        )
//    };
}

pub fn xgetbv() -> u64 {
    let reg: u64 = 0;
    let val_l: u32 = 0;
    let val_h: u32 = 0;
//    unsafe {
//        asm!("\
//                xgetbv
//            ",
//            out("edx") val_h,
//            out("eax") val_l,
//            in("rcx") reg
//        )
//    };
    let val = ((val_h as u64) << 32) | ((val_l as u64) & 0xffff);
    return val;
}

bitflags! {
    pub struct MairEl1: u64 {
        const DEVICE_MEMORY = 0x00 << 16;
        const NORMAL_UNCACHED_MEMORY = 0x44 << 8;
        const NORMAL_WRITEBACK_MEMORY = 0xff;
    }
}

#[inline]
pub fn ttbr0_el1() -> u64 {
    let ret: u64;
    unsafe {
        asm!("mrs {0}, ttbr0_el1", out(reg) ret);
    }
    ret
}

#[inline]
pub fn ttbr0_el1_write(val: u64) {
    unsafe {
        asm!("msr ttbr0_el1, {0}", in(reg) val);
    }
}

#[inline]
pub fn ttbr1_el1() -> u64 {
    let ret: u64;
    unsafe {
        asm!("mrs {0}, ttbr1_el1", out(reg) ret);
    }
    ret
}

#[inline]
pub fn ttbr1_el1_write(val: u64) {
    unsafe {
        asm!("msr ttbr1_el1, {0}", in(reg) val);
    }
}

#[inline]
pub fn mair_el1() -> MairEl1 {
    let ret: u64;
    unsafe {
        asm!("mrs {0}, mair_el1", out(reg) ret);
    }
    MairEl1::from_bits_truncate(ret)
}

#[inline]
pub fn mair_el1_write(val: MairEl1) {
    unsafe {
        asm!("msr mair_el1, {0}", in(reg) val.bits());
    }
}

#[inline]
pub fn tpidr_el0() -> u64 {
    let ret: u64;
    unsafe {
        asm!("mrs {0}, tpidr_el0", out(reg) ret);
    }
    ret
}

#[inline]
pub fn tpidr_el0_write(val: u64) {
    unsafe {
        asm!("msr tpidr_el0, {0}", in(reg) val);
    }
}

#[inline]
pub fn tpidr_el1() -> u64 {
    let ret: u64;
    unsafe {
        asm!("mrs {0}, tpidr_el1", out(reg) ret);
    }
    ret
}

#[inline]
pub fn tpidr_el1_write(val: u64) {
    unsafe {
        asm!("msr tpidr_el1, {0}", in(reg) val);
    }
}

#[inline]
pub fn tpidrro_el0() -> u64 {
    let ret: u64;
    unsafe {
        asm!("mrs {0}, tpidrro_el0", out(reg) ret);
    }
    ret
}

#[inline]
pub fn tpidrro_el0_write(val: u64) {
    unsafe {
        asm!("msr tpidrro_el0, {0}", in(reg) val);
    }
}

#[inline]
pub fn esr_el1() -> u64 {
    let ret: u64;
    unsafe {
        asm!("mrs {0}, esr_el1", out(reg) ret);
    }
    ret
}

#[inline]
pub fn cntfreq_el0() -> u64 {
    let ret: u64;
    unsafe {
        asm!("mrs {0}, cntfrq_el0", out(reg) ret);
    }
    ret
}

#[inline]
pub fn tmr_ctrl() -> u64 {
    let ret: u64;
    unsafe {
        asm!("mrs {0}, cntp_ctl_el0", out(reg) ret);
    }
    ret
}

#[inline]
pub fn tmr_ctrl_write(val: u64) {
    unsafe {
        asm!("msr cntp_ctl_el0, {0}", in(reg) val);
    }
}

#[inline]
pub fn tmr_tval() -> u64 {
    let ret: u64;
    unsafe {
        asm!("mrs {0}, cntp_tval_el0", out(reg) ret);
    }
    ret
}

#[inline]
pub fn tmr_tval_write(val: u64) {
    unsafe {
        asm!("msr cntp_tval_el0, {0}", in(reg) val);
    }
}

#[inline]
pub fn midr() -> u64 {
    let ret: u64;
    unsafe {
        asm!("mrs {0}, midr_el1", out(reg) ret);
    }
    ret
}

#[inline]
pub fn sp_el0() -> u64 {
    let ret: u64;
    unsafe {
        asm!("mrs {0}, sp_el0", out(reg) ret);
    }
    ret
}

#[inline]
pub fn sp_el0_write(val: u64) {
    unsafe {
        asm!("msr sp_el0, {0}", in(reg) val);
    }
}

#[inline]
pub fn sp_el1() -> u64 {
    let ret: u64;
    unsafe {
        asm!("mrs {0}, sp_el1", out(reg) ret);
    }
    ret
}

#[inline]
pub fn sp_el1_write(val: u64) {
    unsafe {
        asm!("msr sp_el1, {0}", in(reg) val);
    }
}

#[inline]
pub fn spsel() -> u64 {
    let ret: u64;
    unsafe {
        asm!("mrs {0}, spsel", out(reg) ret);
    }
    ret
}
