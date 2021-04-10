// Copyright (c) 2021 Quark Container Authors
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

use super::super::task::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::super::Kernel;

pub fn SysInfo(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;

    let mut info : LibcSysinfo = LibcSysinfo::default();

    let ret = Kernel::HostSpace::Sysinfo(&mut info as * mut _ as u64);
    if ret < 0 {
        return Err(Error::SysError(-ret as i32))
    }

    let sysInfo: &mut LibcSysinfo = task.GetTypeMut(addr)?;
    *sysInfo = info;

    return Ok(ret)
}