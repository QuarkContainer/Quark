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
use super::super::qlib::linux::rusage::*;
use super::super::qlib::linux::time::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::super::task::Task;

fn GetUsage(task: &Task, which: i32) -> Result<Rusage> {
    let cs = match which {
        RUSAGE_SELF => task.Thread().ThreadGroup().CPUStats(),
        RUSAGE_CHILDREN => task.Thread().ThreadGroup().JoinedChildCPUStats(),
        RUSAGE_THREAD => task.Thread().CPUStats(),
        RUSAGE_BOTH => {
            let tg = task.Thread().ThreadGroup();
            let mut cs = tg.CPUStats();
            cs.Accumulate(&tg.JoinedChildCPUStats());
            cs
        }
        _ => return Err(Error::SysError(SysErr::EINVAL)),
    };

    return Ok(Rusage {
        UTime: Timeval::FromNs(cs.UserTime),
        STime: Timeval::FromNs(cs.SysTime),
        NVCSw: cs.VoluntarySwitches as i64,
        MaxRSS: (task.Thread().MaxRSS(which) / 1024) as i64,
        ..Default::default()
    });
}

// Getrusage implements linux syscall getrusage(2).
//	marked "y" are supported now
//	marked "*" are not used on Linux
//	marked "p" are pending for support
//
//	y    struct timeval ru_utime; /* user CPU time used */
//	y    struct timeval ru_stime; /* system CPU time used */
//	p    long   ru_maxrss;        /* maximum resident set size */
//	*    long   ru_ixrss;         /* integral shared memory size */
//	*    long   ru_idrss;         /* integral unshared data size */
//	*    long   ru_isrss;         /* integral unshared stack size */
//	p    long   ru_minflt;        /* page reclaims (soft page faults) */
//	p    long   ru_majflt;        /* page faults (hard page faults) */
//	*    long   ru_nswap;         /* swaps */
//	p    long   ru_inblock;       /* block input operations */
//	p    long   ru_oublock;       /* block output operations */
//	*    long   ru_msgsnd;        /* IPC messages sent */
//	*    long   ru_msgrcv;        /* IPC messages received */
//	*    long   ru_nsignals;      /* signals received */
//	y    long   ru_nvcsw;         /* voluntary context switches */
//	y    long   ru_nivcsw;        /* involuntary context switches */
pub fn SysGetrusage(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let which = args.arg0 as i32;
    let addr = args.arg1 as u64;

    if which != RUSAGE_SELF && which != RUSAGE_CHILDREN && which != RUSAGE_THREAD {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let ru = GetUsage(task, which)?;
    //*task.GetTypeMut(addr)? = ru;
    task.CopyOutObj(&ru, addr)?;
    return Ok(0);
}

pub fn SysTimes(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;

    if addr == 0 {
        return Ok(0);
    }

    let tg = task.Thread().ThreadGroup();
    let cs1 = tg.CPUStats();
    let cs2 = tg.JoinedChildCPUStats();

    let r = Tms {
        UTime: ClockTFromDuration(cs1.UserTime),
        STime: ClockTFromDuration(cs1.SysTime),
        CUTime: ClockTFromDuration(cs2.UserTime),
        CSTime: ClockTFromDuration(cs2.SysTime),
    };

    //*task.GetTypeMut(addr)? = r;
    task.CopyOutObj(&r, addr)?;
    return Ok(0);
}
