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

use super::super::super::super::{limits::LimitSet,
                                 kernel::memmgr::arch::MmapLayout,
                                 MemoryDef};
use super::super::super::super::common::*;

// MAX_ADDR64 is the maximum userspace address. It is TASK_SIZE in Linux
// for a 64-bit process.
// ref: linux/latest/source/arch/arm64/include/asm/processor.h
const VA_BITS_4KB_4L_PT: u16 = 48; 
pub const MAX_ADDR64: u64 = 1 << VA_BITS_4KB_4L_PT;
// MAX_STACK_RAND64 is the maximum randomization to apply to the stack.
// ref: linux/latest/source/mm/util.c: mmap_base
const STACK_RND_MASK: u64 = 0x3ffff;
pub const MAX_STACK_RAND64: u64 = STACK_RND_MASK << MemoryDef::PAGE_SHIFT;
// MAX_MMAP_RAND64 is the maximum randomization to apply to the mmap layout.
const MMAP_RND_BITS: u64 = 33;
pub const MAX_MMAP_RAND64: u64 = (1u64 << MMAP_RND_BITS) << MemoryDef::PAGE_SHIFT;
// MIN_GAP64 is the minimum gap to leave at the top of the address space for the stack.
const SIZE_128M: u64 = 0x8000000;
pub const MIN_GAP64: u64 = SIZE_128M + MAX_STACK_RAND64;
// MIN_MMAP_RAND64 is the smallest we are willing to make the
// randomization to stay above PREFERRED_TOP_DOWN_BASE_MIN.
// ref: gVisor
pub const MIN_MMAP_RAND64: u64 = (1 << 18) * MemoryDef::PAGE_SIZE;

// PREFERRED_PIELOAD_ADDR is the standard Linux position-independVent executable base load address.
// ref: linux/latest/arch/arm64/include/asm/elf.h
pub const PREFERRED_PIELOAD_ADDR: u64 = (2 * MAX_ADDR64) / 3;

//
// Select a preferred minimum TopDownBase address.
// ref: ../x86_64/context.rs
//
pub const PREFERRED_TOP_DOWN_ALLOC_MIN: u64 = 0x7E8000000000;
pub const PREFERRED_ALLOCATION_GAP: u64 = 128 << 30;
pub const PREFERRED_TOP_DOWN_BASE_MIN: u64 =
          PREFERRED_TOP_DOWN_ALLOC_MIN +
          PREFERRED_ALLOCATION_GAP;
//
//NOTE: Every thing here is placeholder.
//

pub struct Context64 {
    pub data: [u64; 4069],
}

impl Context64 {
    pub fn PIELoadAddress(l: &MmapLayout) -> Result<u64> {
        Ok(0)
    }

    pub fn MMapRand(max: u64) -> Result<u64> {
        Ok(0)
    }

    pub fn NewMmapLayout(min: u64, max: u64, r: &LimitSet) -> Result<MmapLayout> {
       Ok(
           MmapLayout{
            ..Default::default()
           }) 
    }

}
