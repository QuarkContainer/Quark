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

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::task::*;

pub const SECCOMP_MODE_NONE   : i32 = 0;
pub const SECCOMP_MODE_FILTER : i32 = 2;

pub const SECCOMP_RET_ACTION_FULL : u64 = 0xffff0000;
pub const SECCOMP_RET_ACTION      : u64 = 0x7fff0000;
pub const SECCOMP_RET_DATA        : u64 = 0x0000ffff;

pub const SECCOMP_SET_MODE_FILTER   : u64 = 1;
pub const SECCOMP_FILTER_FLAG_TSYNC : u64 = 1;
pub const SECCOMP_GET_ACTION_AVAIL  : u64 = 2;

pub fn seccomp(_task: &mut Task, _mode: u64, _flags: u64, _addr: u64) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOSYS))
}