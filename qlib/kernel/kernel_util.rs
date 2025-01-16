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

use alloc::boxed::Box;
use crate::GUEST_HOST_SHARED_ALLOCATOR;
use super::super::common::*;
use super::Kernel::*;

pub const GRND_NONBLOCK: u32 = 0x01;
pub const GRND_RANDOM: u32 = 0x02;

pub fn Random(buf: u64, len: u64, flags: u32) -> Result<()> {
    let res = HostSpace::GetRandom(buf, len, flags);
    if res >= 0 {
        return Ok(());
    }

    return Err(Error::SysError(-res as i32));
}

pub fn RandU64() -> Result<u64> {
    let res = Box::new_in(0u64, GUEST_HOST_SHARED_ALLOCATOR);
    Random(&*res as *const _ as u64, 8, GRND_RANDOM)?;
    return Ok(*res);
}

pub fn RandU128() -> Result<(u64, u64)> {
    let res = Box::new_in([0u64; 2], GUEST_HOST_SHARED_ALLOCATOR);
    Random(&res[0] as *const _ as u64, 16, GRND_RANDOM)?;
    return Ok((res[0], res[1]));
}
