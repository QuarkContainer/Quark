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
use super::super::syscalls::syscalls::*;
use super::super::task::*;

pub const ONLY_SCHEDULER: i32 = Sched::SCHED_NORMAL;
pub const ONLY_PRIORITY: i32 = 0;

// SchedParam replicates struct sched_param in sched.h.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SchedParam {
    pub schedPriority: i32,
}

// SchedGetparam implements linux syscall sched_getparam(2).
pub fn SysSchedGetparam(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let pid = args.arg0 as i32;
    let param = args.arg1 as u64;

    if param == 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if pid < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let thread = task.Thread();
    let tg = thread.ThreadGroup();
    let pidns = tg.PIDNamespace();
    if pid != 0 && pidns.TaskWithID(pid).is_none() {
        return Err(Error::SysError(SysErr::ESRCH));
    }

    let r = SchedParam {
        schedPriority: ONLY_PRIORITY,
    };

    task.CopyOutObj(&r, param)?;
    return Ok(0);
}

// SchedGetscheduler implements linux syscall sched_getscheduler(2).
pub fn SysSchedGetscheduler(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let pid = args.arg0 as i32;

    if pid < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let thread = task.Thread();
    let tg = thread.ThreadGroup();
    let pidns = tg.PIDNamespace();
    if pid != 0 && pidns.TaskWithID(pid).is_none() {
        return Err(Error::SysError(SysErr::ESRCH));
    }

    return Ok(ONLY_SCHEDULER as i64);
}

// SchedSetscheduler implements linux syscall sched_setscheduler(2).
pub fn SysSchedSetscheduler(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let pid = args.arg0 as i32;
    let policy = args.arg1 as i32;
    let param = args.arg2 as u64;

    if pid < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if policy != ONLY_SCHEDULER {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let thread = task.Thread();
    let tg = thread.ThreadGroup();
    let pidns = tg.PIDNamespace();
    if pid != 0 && pidns.TaskWithID(pid).is_none() {
        return Err(Error::SysError(SysErr::ESRCH));
    }

    let r: SchedParam = task.CopyInObj(param)?;
    if r.schedPriority != ONLY_PRIORITY {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    return Ok(0);
}

// SchedGetPriorityMax implements linux syscall sched_get_priority_max(2).
pub fn SysSchedGetPriorityMax(_task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    return Ok(ONLY_PRIORITY as i64);
}

// SchedGetPriorityMin implements linux syscall sched_get_priority_min(2).
pub fn SysSchedGetPriorityMin(_task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    return Ok(ONLY_PRIORITY as i64);
}
