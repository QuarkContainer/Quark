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

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::qmsg::qcall::StatmInfo;
use super::super::qlib::usage::memory::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;

pub fn SysInfo(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let mut info = Box::new_in(LibcSysinfo::default(), GUEST_HOST_SHARED_ALLOCATOR);

    /*let ret = Kernel::HostSpace::Sysinfo(&mut *info as * mut _ as u64);
    if ret < 0 {
        return Err(Error::SysError(-ret as i32))
    }*/

    let statm = Box::new_in(StatmInfo::default(), GUEST_HOST_SHARED_ALLOCATOR);
    // TODO(Cong): bypassing this issue for now, fix this...
    //Kernel::HostSpace::Statm(&mut *statm);
    info!("pass to here, rss, {}", statm.rss);

    let totalUsage = statm.rss;
    let totalSize = TotalMemory(0, totalUsage);

    //let sysInfo: &mut LibcSysinfo = task.GetTypeMut(addr)?;
    info.procs = task.Thread().PIDNamespace().Tasks().len() as u16;
    info.uptime = Task::MonoTimeNow().Seconds() as i64;
    info.totalram = totalSize; //super::super::ALLOCATOR.Total() as u64;
    info.freeram = totalSize - totalUsage; // super::super::ALLOCATOR.Free() as u64;
    info.mem_unit = 1;

    //*sysInfo = info;
    task.CopyOutObj(&*info, addr)?;
    //error!("SysInfo output is {:?}", &info);

    //return Ok(ret)

    return Ok(0);
}
