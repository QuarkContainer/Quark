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

use super::super::linux_def::*;

// VirtualAddressBits returns the number bits available for virtual addresses.
//
// Note that sign-extension semantics apply to the highest order bit.
pub const fn VirtualAddressBits() -> u32 {
    /*let (ax, _, _, _) = HostID(0x80000008, 0);
    return (ax >> 8) & 0xff;*/
    return 48;
}

// UserspaceSize is the total size of userspace.
pub const USERSPACE_SIZE : u64 = 1 << (VirtualAddressBits() as usize - 1);

// UserspaceSize is the total size of userspace.
pub const KERNEL_START_ADDRESS : u64 = !0 - (USERSPACE_SIZE - 1);


// MaximumUserAddress is the largest possible user address.
pub const  MAXIMUM_USER_ADDRESS: u64 = (USERSPACE_SIZE - 1) & !(MemoryDef::PAGE_SIZE - 1);

pub const MIN_USER_ADDR : u64 = MemoryDef::PAGE_SIZE;
pub const MAX_USER_ADDR : u64 = MAXIMUM_USER_ADDRESS;