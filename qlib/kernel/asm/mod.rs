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

#[inline]
pub fn WriteMsr(msr: u32, value: u64) {
    unsafe {
        let low = value as u32;
        let high = (value >> 32) as u32;
        llvm_asm!("wrmsr" :: "{ecx}" (msr), "{eax}" (low), "{edx}" (high) : "memory" : "volatile" );
    }
}

#[inline]
pub fn ReadMsr(msr: u32) -> u64 {
    let (high, low): (u32, u32);
    unsafe {
        llvm_asm!("rdmsr" : "={eax}" (low), "={edx}" (high) : "{ecx}" (msr) : "memory" : "volatile");
    }
    ((high as u64) << 32) | (low as u64)
}

#[inline]
pub fn SwapGs() {
    unsafe {
        llvm_asm!("swapgs":: : "memory" : "volatile");
    }
}

pub fn GetVcpuId() -> usize {
    let result: usize;
    unsafe {
        llvm_asm!("mov %gs:16, %rax" : "={rax}" (result): : "memory" : "volatile");
    }

    return result;
}

#[inline(always)]
pub fn HyperCall64(type_: u16, para1: u64, para2: u64, para3: u64) {
    unsafe {
        let data: u8 = 0;
        llvm_asm!("out $1, $0":: "{dx}"(type_), "{al}"(data), "{rbx}"(para1), "{rcx}"(para2), "{rdi}"(para3): "memory" : "volatile" )
    }
}

#[inline]
pub fn Hlt() {
    unsafe {
        llvm_asm!(
                "hlt"
        )
    }
}

#[inline]
pub fn LoadCr3(cr3: u64) {
    unsafe { llvm_asm!("mov $0, %cr3" : : "r" (cr3) : "memory" : "volatile" ) };
}

#[inline]
pub fn CurrentCr3() -> u64 {
    let cr3: u64;
    unsafe { llvm_asm!("mov %cr3, $0" : "=r" (cr3) : : "memory" : "volatile" ) };
    return cr3;
}

#[inline]
pub fn EnterUser(entry: u64, userStackAddr: u64, kernelStackAddr: u64) {
    //PerfGoto(PerfType::User);
    unsafe {
        llvm_asm!("
            //mov gs:0, rsp
            mov gs:0, rdx

            mov rcx, rdi
            mov r11, 0x2 | 1<<9

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
              "
              :
              : "{rdi}"(entry), "{rsi}"(userStackAddr), "{rdx}"(kernelStackAddr)
              : "memory"
              : "intel", "volatile");
        ::core::intrinsics::unreachable();
    }
}

#[inline]
pub fn SyscallRet(kernelRsp: u64) -> ! {
    unsafe {
        llvm_asm!("
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
              "
              :
              : "{rsp}"(kernelRsp)
              : "memory" : "intel", "volatile");
        ::core::intrinsics::unreachable();
    }
}

#[inline]
pub fn IRet(kernelRsp: u64) -> ! {
    unsafe {
        llvm_asm!("
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
              "
              :
              : "{rsp}"(kernelRsp)
              : "memory" : "intel", "volatile");
        ::core::intrinsics::unreachable();
    }
}

#[inline]
pub fn child_clone(userSp: u64) {
    unsafe {
        llvm_asm!("
            mov rbx, [rsp - 16]
            //fxrstor64 [rbx + 0]

            mov gs:8, rdi

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

            add rsp, 6 * 8

            //kernel stack
            mov gs:0, rsp

            //user stack
            mov rsp, gs:8

            swapgs
            .byte 0x48
            sysret
        ":
             : "{rdi}"(userSp)
             : "memory" : "intel", "volatile");
    }
}


#[inline]
pub fn GetRsp() -> u64 {
    let rsp: u64;
    unsafe { llvm_asm!("mov %rsp, $0" : "=r" (rsp) : : "memory" : "volatile" ) };
    return rsp;
}

#[inline]
pub fn Invlpg(addr: u64) {
    if !super::SHARESPACE.config.read().KernelPagetable {
        unsafe { llvm_asm!("
            invlpg ($0)
            " :: "r" (addr): "memory" : "volatile" ) };
    }
}

#[inline]
pub fn Clflush(addr: u64) {
    unsafe { llvm_asm!("clflush ($0)" :: "r" (addr): "memory" : "volatile" ) }
}

// muldiv64 multiplies two 64-bit numbers, then divides the result by another
// 64-bit number.
//
// It requires that the result fit in 64 bits, but doesn't require that
// intermediate values do; in particular, the result of the multiplication may
// require 128 bits.
//
// It returns !ok if divisor is zero or the result does not fit in 64 bits.
#[inline(always)]
pub fn muldiv64(value: u64, multiplier: u64, divisor: u64) -> (u64, bool) {
    let val = value as u128 * multiplier as u128;
    let res = val / divisor as u128;
    if res > core::u64::MAX as u128 {
        return (0, false)
    }

    return (res as u64, true)
}

// HostID executes a native CPUID instruction.
// return (ax, bx, cx, dx)
pub fn AsmHostID(axArg: u32, cxArg: u32) -> (u32, u32, u32, u32) {
    let ax: u32;
    let bx: u32;
    let cx: u32;
    let dx: u32;
    unsafe {
        llvm_asm!("
              CPUID
            "
            : "={eax}"(ax), "={ebx}"(bx), "={ecx}"(cx), "={edx}"(dx)
            : "{eax}"(axArg), "{ecx}"(cxArg)
            :
            : );
    }

    return (ax, bx, cx, dx)
}

#[inline(always)]
fn Barrier() {
    unsafe {
        llvm_asm!("
                mfence
            "
            :
            :
            : "memory"
            : );
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

// Rdtsc reads the TSC.
//
// Intel SDM, Vol 3, Ch 17.15:
// "The RDTSC instruction reads the time-stamp counter and is guaranteed to
// return a monotonically increasing unique value whenever executed, except for
// a 64-bit counter wraparound. Intel guarantees that the time-stamp counter
// will not wraparound within 10 years after being reset."
//
// We use int64, so we have 5 years before wrap-around.
#[inline(always)]
pub fn Rdtsc() -> i64 {
    let rax: u64;
    let rdx: u64;
    unsafe {
        llvm_asm!("\
        lfence
        rdtsc
        " : "={rax}"(rax), "={rdx}"(rdx)
        : : "memory" : "volatile")
    };

    return rax as i64 | ((rdx as i64) << 32);
}

#[inline(always)]
pub fn GetCpu() -> u32 {
    let rcx: u64;
    unsafe {
        llvm_asm!("\
        rdtscp
        " : "={rcx}"(rcx)
        : :  "memory" : "volatile")
    };

    return (rcx & 0xfff) as u32;
}

#[inline(always)]
pub fn GetRflags() -> u64 {
    let rax: u64;
    unsafe {
        llvm_asm!("\
            pushfq                  # push eflags into stack
            pop rax                 # pop it into rax
        " : "={rax}"(rax)
        : : "memory" : "intel", "volatile")
    };

    return rax;
}

#[inline(always)]
pub fn SetRflags(val: u64) {
    unsafe {
        llvm_asm!("\
            push rax
            popfq
        " : : "{rax}"(val)
        : "memory" : "intel", "volatile")
    };
}

pub fn SaveFloatingPoint(addr: u64) {
    fxsave(addr);
    //xsaveopt(addr);
}

pub fn xsaveopt(addr: u64) {
    let negtive1 : u64 = 0xffffffff;
    unsafe {
        llvm_asm!("\
            xsaveopt [rdi + 0]
        " : : "{rdi}"(addr), "{rax}"(negtive1), "{rdx}"(negtive1)
        : "memory" : "intel", "volatile")
    };
}

pub fn fxsave(addr: u64) {
    unsafe {
        llvm_asm!("\
            fxsave64 [rbx + 0]
        " : : "{rbx}"(addr)
        : "memory" : "intel", "volatile")
    };
}

pub fn LoadFloatingPoint(addr: u64) {
    fxrstor(addr);
}

pub fn fxrstor(addr: u64) {
    unsafe {
        llvm_asm!("\
            fxrstor64 [rbx + 0]
        " : : "{rbx}"(addr)
        : "memory" : "intel", "volatile")
    };
}

#[inline(always)]
pub fn mfence() {
    unsafe { llvm_asm!("
        sfence
        lfence
    " : : : "memory" : "volatile" ) }
}

#[inline(always)]
pub fn sfence() {
    unsafe { llvm_asm!("
        sfence
    " : : : "memory" : "volatile" ) }
}

#[inline(always)]
pub fn lfence() {
    unsafe { llvm_asm!("
        lfence
    " : : : "memory" : "volatile" ) }
}