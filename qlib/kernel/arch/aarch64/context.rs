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

use super::super::super::super::{limits::LimitSet, kernel::memmgr::arch::MmapLayout};
use super::super::super::super::common::*;


// MAX_ADDR64 is the maximum userspace address. It is TASK_SIZE in Linux
// for a 64-bit process.
// ref: linux/latest/source/arch/arm64/include/asm/processor.h
const VA_BITS_4KB_4L_PT: u16 = 48; 
pub const MAX_ADDR64: u64 = 1 << VA_BITS_4KB_4L_PT;

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
