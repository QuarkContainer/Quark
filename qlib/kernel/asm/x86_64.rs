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

use super::super::{SUPPORT_XSAVE, SUPPORT_XSAVEOPT};
use core::arch::asm;
use core::sync::atomic::Ordering;

use crate::qlib::kernel::task;
use crate::qlib::vcpu_mgr::CPULocal;

#[inline]
pub fn WriteMsr(msr: u32, value: u64) {
    unsafe {
        let low = value as u32;
        let high = (value >> 32) as u32;
        asm!(
            "wrmsr",
            in("ecx") msr,
            in("eax") low,
            in("edx") high
        );
    }
}

#[inline]
pub fn ReadMsr(msr: u32) -> u64 {
    let (high, low): (u32, u32);
    unsafe {
        asm!(
            "rdmsr",
            out("eax") low,
            out("edx") high,
            in("ecx") msr
        );
    }
    ((high as u64) << 32) | (low as u64)
}

#[inline]
pub fn SwapGs() {
    unsafe {
        asm!("swapgs");
    }
}

pub fn GetVcpuId() -> usize {
    let result: usize;
    unsafe {
        asm!(
            "mov rax, gs:16",
            out("rax") result
        );
    }

    return result;
}

#[inline]
pub fn Hlt() {
    unsafe { asm!("hlt") }
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
pub fn CurrentUserTable() -> u64 {
    return CurrentCr3();
}

#[inline]
pub fn CurrentKernelTable() -> u64 {
    return CurrentCr3();
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
    let currTask = task::Task::Current();
    let pt = currTask.GetPtRegs();
    CPULocal::SetKernelStack(kernelStackAddr);
    CPULocal::SetUserStack(userStackAddr);
    *pt = Default::default();

    pt.rip = entry;
    pt.cs = 0x23;
    pt.eflags = 0x2 | 1 << 9 | 1 << 12 | 1 << 13; //USER_FLAGS_SET;
    pt.rsp = userStackAddr;
    pt.ss = 0x1b;

    unsafe {
        asm!(
            "
            fninit
            "
        );
    }

    IRet(pt as *const _ as u64);
}

#[inline]
pub fn EnterUser1(entry: u64, userStackAddr: u64, kernelStackAddr: u64) -> ! {
    //PerfGoto(PerfType::User);
    unsafe {
        asm!("
            fninit
            mov gs:0, rdx

            mov rcx, rdi
            mov r11, 0x2 | 1<<9 | 1<<12 | 1<<13

            mov rsp, rsi

            /* clean up registers */
            xor rax, rax
            xor rbx, rbx
            xor rdx, rdx
            xor rdi, rdi
            xor rsi, rsi
            xor rbp, rbp
            xor r8, r8
            xor r9, r9
            xor r10, r10
            xor r12, r12
            xor r13, r13
            xor r14, r14
            xor r15, r15

            swapgs

            .byte 0x48
            sysret
              ",
              in("rdi") entry,
              in("rsi") userStackAddr,
              in("rdx") kernelStackAddr);
        panic!("won't reach");
    }
}

#[inline]
pub fn SyscallRet(kernelRsp: u64) -> ! {
    unsafe {
        asm!("
                mov rsp, {0}
                //we have to store callee save registers for signal handling
                pop r15
                pop r14
                pop r13
                pop r12
                pop rbp
                pop rbx

                pop r11
                pop r10
                pop r9
                pop r8

                pop rax
                pop rcx
                pop rdx
                pop rsi
                pop rdi

                //the return frame for orig_rax, iretq
                add rsp, 4 * 8
                pop rsp
                //pop gs:8
                //add rsp, 1 * 8

                //mov rsp, gs:8
                swapgs
                .byte 0x48
                sysret
              ",
              in(reg) kernelRsp);
        panic!("won't reach");
    }
}

#[inline]
pub fn IRet(kernelRsp: u64) -> ! {
    unsafe {
        asm!("
                mov rsp, rax
                //we have to store callee save registers for signal handling
                pop r15
                pop r14
                pop r13
                pop r12
                pop rbp
                pop rbx

                pop r11
                pop r10
                pop r9
                pop r8

                pop rax
                pop rcx
                pop rdx
                pop rsi
                pop rdi

                add rsp, 8
                swapgs
                iretq
              ",
        in("rax") kernelRsp
        );
        panic!("won't reach");
    }
}

#[inline]
pub fn GetCurrentKernelSp() -> u64 {
    let rsp: u64;
    unsafe {
        asm!(
            "mov rax, rsp",
            out("rax") rsp
        )
    };
    return rsp;
}

#[inline]
pub fn Clflush(addr: u64) {
    unsafe {
        asm!(
            "clflush (rax)",
            in("rax") addr
        )
    }
}

// HostID executes a native CPUID instruction.
// return (ax, bx, cx, dx)
pub fn AsmHostID(axArg: u32, cxArg: u32) -> (u32, u32, u32, u32) {
    let mut ax: u32 = axArg;
    let bx: u32;
    let mut cx: u32 = cxArg;
    let dx: u32;
    unsafe {
        asm!("
              mov {0:r}, rbx 
              CPUID
              xchg {0:r}, rbx 
            ",
        lateout(reg) bx,
        inout("eax") ax,
        inout("ecx") cx,
        out("edx") dx,
        );
    }

    return (ax, bx, cx, dx);
}

#[inline(always)]
fn Barrier() {
    unsafe {
        asm!(
            "
                mfence
            "
        );
    }
}

#[inline(always)]
pub fn ReadBarrier() {
    Barrier();
}

#[inline(always)]
pub fn WriteBarrier() {
    Barrier();
}

#[inline(always)]
pub fn GetCpu() -> u32 {
    let rcx: u64;
    unsafe {
        asm!("\
            rdtscp
            ",
            out("rcx") rcx
        )
    };

    return (rcx & 0xfff) as u32;
}

#[inline(always)]
pub fn GetRflags() -> u64 {
    let rax: u64;
    unsafe {
        asm!("\
                pushfq                  # push eflags into stack
                pop rax                 # pop it into rax
            ",
            out("rax") rax
        )
    };

    return rax;
}

#[inline(always)]
pub fn SetRflags(val: u64) {
    unsafe {
        asm!("\
                push rax
                popfq
            ",
            in("rax") val)
    };
}

pub fn SaveFloatingPoint(addr: u64) {
    if SUPPORT_XSAVEOPT.load(Ordering::Acquire) {
        xsaveopt(addr);
    } else if SUPPORT_XSAVE.load(Ordering::Acquire) {
        xsave(addr);
    } else {
        fxsave(addr);
    }
}

pub fn LoadFloatingPoint(addr: u64) {
    if SUPPORT_XSAVE.load(Ordering::Acquire) {
        xrstor(addr);
    } else {
        fxrstor(addr);
    }
}

pub fn xsave(addr: u64) {
    unsafe {
        asm!("\
                xsave64 [rdi + 0]
            ",
            in("rdi") addr, )
    };
}

pub fn xsaveopt(addr: u64) {
    let negtive1: u64 = 0xffffffff;
    unsafe {
        asm!("\
                xsaveopt64 [rdi + 0]
            ",
            in("rdi") addr, 
            in("eax") negtive1,
            in("edx") negtive1)
    };
}

pub fn xrstor(addr: u64) {
    let negtive1: u64 = 0xffffffff;
    unsafe {
        asm!("\
                xrstor64 [rdi + 0]
            ",
            in("rdi") addr,
            in("eax") negtive1,
            in("edx") negtive1)
    };
}

pub fn fxsave(addr: u64) {
    unsafe {
        asm!("\
                fxsave64 [rax + 0]
            ",
            in("rax") addr)
    };
}

pub fn fxrstor(addr: u64) {
    unsafe {
        asm!("\
                fxrstor64 [rax + 0]
            ",
            in("rax") addr)
    };
}

#[inline(always)]
pub fn mfence() {
    unsafe {
        asm!(
            "
            sfence
            lfence
        "
        )
    }
}

#[inline(always)]
pub fn sfence() {
    unsafe {
        asm!(
            "
            sfence
        "
        )
    }
}

#[inline(always)]
pub fn lfence() {
    unsafe {
        asm!(
            "
            lfence
        "
        )
    }
}

pub fn stmxcsr(addr: u64) {
    unsafe {
        asm!("\
                STMXCSR [rax]
            ",
            in("rax") addr)
    };
}

pub fn ldmxcsr(addr: u64) {
    unsafe {
        asm!("\
                LDMXCSR [rax]
            ",
            in("rax") addr)
    };
}

pub fn FSTCW(addr: u64) {
    unsafe {
        asm!("\
                FSTCW [rax]
            ",
            in("rax") addr
        )
    };
}

pub fn FLDCW(addr: u64) {
    unsafe {
        asm!("\
                FLDCW [rax]
            ",
            in("rax") addr)
    };
}

pub fn FNCLEX() {
    unsafe {
        asm!(
            "\
            FNCLEX
        "
        )
    };
}

pub fn fninit() {
    unsafe {
        asm!(
            "\
            fninit
            "
        )
    };
}

pub fn xsetbv(val: u64) {
    let reg = 0u64;
    let val_l = val & 0xffff;
    let val_h = val >> 32;
    unsafe {
        asm!("\
                xsetbv
            ",
            in("rcx") reg,
            in("edx") val_h,
            in("eax") val_l,
        )
    };
}

pub fn xgetbv() -> u64 {
    let reg: u64 = 0;
    let val_l: u32;
    let val_h: u32;
    unsafe {
        asm!("\
                xgetbv
            ",
            out("edx") val_h,
            out("eax") val_l,
            in("rcx") reg
        )
    };
    let val = ((val_h as u64) << 32) | ((val_l as u64) & 0xffff);
    return val;
}
