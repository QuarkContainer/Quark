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

use crate::qlib::{kernel::{arch::__arch::arch_def::*, kernel_util::RandU64,
    memmgr::arch::{MmapLayout, MMAP_BOTTOM_UP, MMAP_TOP_DOWN}}, limits::{LimitSet,
    LimitType, INFINITY}, common::*, addr::Addr, linux_def::SysErr, MemoryDef};

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
const MMAP_RND_BITS: u64 = 28;
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
    PREFERRED_TOP_DOWN_ALLOC_MIN + PREFERRED_ALLOCATION_GAP;

pub struct Context64 {
    pub state: State,
}

impl Context64 {

    pub fn PIELoadAddress(l: &MmapLayout) -> Result<u64> {
        let mut pref_base: u64 = PREFERRED_PIELOAD_ADDR;
        let top_addr = match Addr(pref_base).AddLen(MAX_MMAP_RAND64) {
            Ok(a) => a.0,
            Err(_) => panic!("Preferred PIELoadAddress {} is too large.", pref_base),
        };

        if top_addr > l.MaxAddr {
            // Linux resets base to 2/3rd;
            pref_base = (l.TopDownBase * 2) / 3;
        }

        Ok(Addr(pref_base + MMapRand(MAX_MMAP_RAND64).unwrap())
            .RoundDown()
            .unwrap()
            .0)
    }


    pub fn NewMmapLayout(min: u64, max: u64, r: &LimitSet) -> Result<MmapLayout> {
        //
        // Sanity check.
        //
        if min > max {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let mut max_addr = if max > MAX_ADDR64 { MAX_ADDR64 } else { max };
        max_addr = Addr(max_addr).RoundUp()?.0;
        let min_addr = Addr(min).RoundUp()?.0;

        let stack_size: u64 = r.Get(LimitType::Stack).Cur;
        let default_direction = if stack_size == INFINITY {
            MMAP_BOTTOM_UP
        } else {
            MMAP_TOP_DOWN
        };

        let max_gap = (max_addr / 6) * 5; // Linux defiened
        let mut gap = if stack_size < MIN_GAP64 {
            MIN_GAP64
        } else {
            stack_size
        };
        if gap > max_gap {
            gap = max_gap;
        }

        let mut max_rand: u64 = MAX_MMAP_RAND64;
        if (max_addr - gap - max_rand) < PREFERRED_TOP_DOWN_BASE_MIN {
            let bottom_base_adjust = PREFERRED_TOP_DOWN_BASE_MIN - (max_addr - gap - max_rand);
            let max_adjust = MAX_MMAP_RAND64 - MIN_MMAP_RAND64;
            if bottom_base_adjust <= max_adjust {
                max_rand = MAX_MMAP_RAND64 - bottom_base_adjust;
            }
        }

        let rand: u64 = MMapRand(max_rand)?;

        let layout = MmapLayout {
            MinAddr: min_addr,
            MaxAddr: max_addr,
            //
            // TASK_UNMAPPED_BASE
            //
            BottomUpBase: Addr(max / 3 + rand).RoundDown()?.0,
            TopDownBase: Addr(max - gap - rand).RoundDown()?.0,
            DefaultDirection: default_direction,
            //
            // Stack allocation must use the max randomization to avoid
            // eating into the gap.
            //
            MaxStackRand: max_rand,
            sharedLoadsOffset: 0,
        };

        //
        // Layout sanity check.
        //
        if !layout.Valid() {
            panic!("QKernel: Context64 - Invalid MmapLayout: {:?}", layout);
        }

        Ok(layout)
    }
}

// mmapRand returns a random adjustment for randomizing an mmap layout.
pub fn MMapRand(max: u64) -> Result<u64> {
	let _new_addr: u64 = RandU64().unwrap() % max;
	Ok(Addr(_new_addr).RoundDown().unwrap().0)
}
