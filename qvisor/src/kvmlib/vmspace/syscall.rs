// Copyright (c) 2021 QuarkSoft LLC
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

#[inline(always)]
pub unsafe fn syscall0(n: usize) -> usize {
    let ret: usize;
    llvm_asm!("syscall" : "={rax}"(ret)
                   : "{rax}"(n)
                   : "rcx", "r11", "memory"
                   : "volatile");
    ret
}

#[inline(always)]
pub unsafe fn syscall1(n: usize, a1: usize) -> usize {
    let ret: usize;
    llvm_asm!("syscall" : "={rax}"(ret)
                   : "{rax}"(n), "{rdi}"(a1)
                   : "rcx", "r11", "memory"
                   : "volatile");
    ret
}

#[inline(always)]
pub unsafe fn syscall2(n: usize, a1: usize, a2: usize) -> usize {
    let ret: usize;
    llvm_asm!("syscall" : "={rax}"(ret)
                   : "{rax}"(n), "{rdi}"(a1), "{rsi}"(a2)
                   : "rcx", "r11", "memory"
                   : "volatile");
    ret
}

#[inline(always)]
pub unsafe fn syscall3(n: usize, a1: usize, a2: usize, a3: usize) -> usize {
    let ret: usize;
    llvm_asm!("syscall" : "={rax}"(ret)
                   : "{rax}"(n), "{rdi}"(a1), "{rsi}"(a2), "{rdx}"(a3)
                   : "rcx", "r11", "memory"
                   : "volatile");
    ret
}

#[inline(always)]
pub unsafe fn syscall4(n: usize, a1: usize, a2: usize, a3: usize,
                       a4: usize) -> usize {
    let ret: usize;
    llvm_asm!("syscall" : "={rax}"(ret)
                   : "{rax}"(n), "{rdi}"(a1), "{rsi}"(a2), "{rdx}"(a3),
                     "{r10}"(a4)
                   : "rcx", "r11", "memory"
                   : "volatile");
    ret
}

#[inline(always)]
pub unsafe fn syscall5(n: usize, a1: usize, a2: usize, a3: usize,
                       a4: usize, a5: usize) -> usize {
    let ret: usize;
    llvm_asm!("syscall" : "={rax}"(ret)
                   : "{rax}"(n), "{rdi}"(a1), "{rsi}"(a2), "{rdx}"(a3),
                     "{r10}"(a4), "{r8}"(a5)
                   : "rcx", "r11", "memory"
                   : "volatile");
    ret
}

#[inline(always)]
pub unsafe fn syscall6(n: usize, a1: usize, a2: usize, a3: usize,
                       a4: usize, a5: usize, a6: usize) -> usize {
    let ret: usize;
    llvm_asm!("syscall" : "={rax}"(ret)
                   : "{rax}"(n), "{rdi}"(a1), "{rsi}"(a2), "{rdx}"(a3),
                     "{r10}"(a4), "{r8}"(a5), "{r9}"(a6)
                   : "rcx", "r11", "memory"
                   : "volatile");
    ret
}

pub fn KLoadBinary(_fileName: &String, _envs: &Vec<String>, _args: &Vec<String>) {}