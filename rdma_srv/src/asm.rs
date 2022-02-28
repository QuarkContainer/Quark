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

pub fn LoadCr3(_cr3: u64) {}
pub fn ReadCr3() -> u64 { 0 }
pub fn HyperCall(_type_: u16, _para1: u64) {}
pub fn Invlpg(_addr: u64) {}
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

#[inline]
pub fn CurrentCr3() -> u64 {
    let cr3: u64;
    unsafe { llvm_asm!("mov %cr3, $0" : "=r" (cr3) ) };
    return cr3;
}

#[inline(always)]
pub fn mfence() {
    unsafe { llvm_asm!("mfence" : : : "memory" : "volatile" ) }
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