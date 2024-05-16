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

use super::super::super::addr::*;
use super::super::arch::__arch::context::*;

pub type MmapDirection = i32;

pub const MMAP_BOTTOM_UP: MmapDirection = 0;
pub const MMAP_TOP_DOWN: MmapDirection = 1;

// MmapLayout defines the layout of the user address space for a particular
// MemoryManager.
//
// Note that "highest address" below is always exclusive.

#[derive(Debug, Copy, Clone, Default)]
pub struct MmapLayout {
    // MinAddr is the lowest mappable address.
    pub MinAddr: u64,

    // MaxAddr is the highest mappable address.
    pub MaxAddr: u64,

    // BottomUpBase is the lowest address that may be returned for a
    // MmapBottomUp mmap.
    pub BottomUpBase: u64,

    // TopDownBase is the highest address that may be returned for a
    // MmapTopDown mmap.
    pub TopDownBase: u64,

    // DefaultDirection is the direction for most non-fixed mmaps in this
    // layout.
    pub DefaultDirection: MmapDirection,

    // MaxStackRand is the maximum randomization to apply to stack
    // allocations to maintain a proper gap between the stack and
    // TopDownBase.
    pub MaxStackRand: u64,

    pub sharedLoadsOffset: u64,
}

impl MmapLayout {
    // Valid returns true if this layout is valid.
    pub fn Valid(&self) -> bool {
        if self.MinAddr > self.MaxAddr {
            return false;
        }

        if self.BottomUpBase < self.MinAddr {
            return false;
        }

        if self.BottomUpBase > self.MaxAddr {
            return false;
        }

        if self.TopDownBase < self.MinAddr {
            return false;
        }

        if self.TopDownBase > self.MaxAddr {
            return false;
        }

        return true;
    }

    pub fn MapStackAddr(&self) -> u64 {
        return Addr(self.MaxAddr - MMapRand(self.MaxStackRand).expect("MapStackAddr fail"))
            .RoundDown()
            .unwrap()
            .0;
    }
}
