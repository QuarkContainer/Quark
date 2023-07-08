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

use crate::qlib::kernel::task;
use crate::qlib::vcpu_mgr::CPULocal;

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
    return 0;
}

#[inline]
pub fn Hlt() {
}

#[inline]
pub fn LoadCr3(cr3: u64) {
    unsafe {
        asm!(
            "mov cr3, {0}",
            in(reg) cr3
        )
    };
}

#[inline]
pub fn CurrentCr3() -> u64 {
    let cr3: u64;
    unsafe {
        asm!(
            "mov {0}, cr3",
            out(reg) cr3
        )
    };
    return cr3;
}

#[inline]
pub fn EnterUser(entry: u64, userStackAddr: u64, kernelStackAddr: u64) -> ! {
    unsafe {
        asm!(
            "nop"
        );
        panic!("won't reach");
    }
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
        asm!(
            "nop"
        );
        panic!("won't reach");
    }
}

#[inline]
pub fn GetRsp() -> u64 {
    return 0;
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
        asm!("dmb ish")
    };
}

#[inline(always)]
pub fn WriteBarrier() {
    unsafe {
        asm!("dsb ish")
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
}

pub fn LoadFloatingPoint(addr: u64) {
//    if SUPPORT_XSAVE.load(Ordering::Acquire) {
//        xrstor(addr);
//    } else {
//        fxrstor(addr);
//    }
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

pub fn fxsave(addr: u64) {
//    unsafe {
//        asm!("\
//                fxsave64 [rax + 0]
//            ",
//            in("rax") addr)
//    };
}

pub fn fxrstor(addr: u64) {
//    unsafe {
//        asm!("\
//                fxrstor64 [rax + 0]
//            ",
//            in("rax") addr)
//    };
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
            dsb ish
        "
        )
    }
}

#[inline(always)]
pub fn lfence() {
    unsafe {
        asm!(
            "
            dmb ish
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
