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

use alloc::collections::btree_set::BTreeSet;

//use super::super::Kernel;
use super::super::qlib::common::*;
use super::super::qlib::limits::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::singleton::*;
use super::super::syscalls::syscalls::*;
use super::super::task::Task;
use super::super::threadmgr::thread::*;

pub static SETABLE_LIMITS: Singleton<BTreeSet<LimitType>> = Singleton::<BTreeSet<LimitType>>::New();

pub unsafe fn InitSingleton() {
    SETABLE_LIMITS.Init(
        [
            LimitType::NumberOfFiles,
            LimitType::AS,
            LimitType::CPU,
            LimitType::Data,
            LimitType::FileSize,
            LimitType::MemoryLocked,
            LimitType::Stack,
            // These are not enforced, but we include them here to avoid returning
            // EPERM, since some apps expect them to succeed.
            LimitType::Core,
            LimitType::ProcessCount,
        ]
        .iter()
        .cloned()
        .collect(),
    );
}

pub fn PrLimit64(thread: &Thread, resource: LimitType, newLimit: Option<Limit>) -> Result<Limit> {
    let tg = thread.ThreadGroup();
    let limits = tg.Limits();

    if newLimit.is_none() {
        return Ok(limits.Get(resource));
    }

    if !SETABLE_LIMITS.contains(&resource) {
        return Err(Error::SysError(SysErr::EPERM));
    }

    // "A privileged process (under Linux: one with the CAP_SYS_RESOURCE
    // capability in the initial user namespace) may make arbitrary changes
    // to either limit value."

    let kernel = thread.Kernel();
    let root = kernel.RootUserNamespace();
    let privileged = thread.HasCapabilityIn(Capability::CAP_SYS_RESOURCE, &root);

    let oldLim = limits.Set(resource, newLimit.unwrap(), privileged)?;

    if resource == LimitType::CPU {
        thread.NotifyRlimitCPUUpdated()
    }

    return Ok(oldLim);
}

#[derive(Default, Clone, Copy)]
pub struct RLimit64 {
    pub Cur: u64,
    pub Max: u64,
}

impl RLimit64 {
    pub fn ToLimit(&self) -> Limit {
        return Limit {
            Cur: FromLinux(self.Cur),
            Max: FromLinux(self.Max),
        };
    }

    pub fn FromLimit(lim: &Limit) -> Self {
        return Self {
            Cur: ToLinux(lim.Cur),
            Max: ToLinux(lim.Max),
        };
    }
}

// Getrlimit implements linux syscall getrlimit(2).
pub fn SysGetrlimit(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let lr = args.arg0 as i32;
    let addr = args.arg1 as u64;

    let resource = match FROM_LINUX_RESOURCE.Get(lr) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(r) => r,
    };

    let lim = PrLimit64(&task.Thread(), resource, None)?;
    let rlim = RLimit64::FromLimit(&lim);

    //*task.GetTypeMut(addr)? = rlim;
    task.CopyOutObj(&rlim, addr)?;
    return Ok(0);
}

pub fn SysSetrlimit(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let lr = args.arg0 as i32;
    let addr = args.arg1 as u64;

    let resource = match FROM_LINUX_RESOURCE.Get(lr) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(r) => r,
    };

    let rlim: RLimit64 = task.CopyInObj(addr)?;

    PrLimit64(&task.Thread(), resource, Some(rlim.ToLimit()))?;

    return Ok(0);
}

pub fn SysPrlimit64(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let tid = args.arg0 as i32;
    let lr = args.arg1 as i32;
    let newRlimAddr = args.arg2 as u64;
    let oldRlimAddr = args.arg3 as u64;

    let resource = match FROM_LINUX_RESOURCE.Get(lr) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(r) => r,
    };

    let newlim = if newRlimAddr != 0 {
        let nrl: RLimit64 = task.CopyInObj(newRlimAddr)?;
        Some(nrl.ToLimit())
    } else {
        None
    };

    if tid < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let thread = task.Thread();
    let pidns = thread.PIDNamespace();
    let ot = if tid > 0 {
        let ot = match pidns.TaskWithID(tid) {
            None => return Err(Error::SysError(SysErr::ESRCH)),
            Some(t) => t,
        };
        ot
    } else {
        thread.clone()
    };

    // "To set or get the resources of a process other than itself, the caller
    // must have the CAP_SYS_RESOURCE capability, or the real, effective, and
    // saved set user IDs of the target process must match the real user ID of
    // the caller and the real, effective, and saved set group IDs of the
    // target process must match the real group ID of the caller."
    if !thread.HasCapabilityIn(Capability::CAP_SYS_RESOURCE, &pidns.UserNamespace()) {
        let cred = thread.Credentials();
        let tcred = ot.Credentials();

        if cred != tcred {
            let credlock = cred.lock();
            let tcredlock = tcred.lock();

            if credlock.RealKUID != tcredlock.RealKUID
                || credlock.RealKUID != tcredlock.EffectiveKUID
                || credlock.RealKUID != tcredlock.SavedKUID
                || credlock.RealKGID != tcredlock.RealKGID
                || credlock.RealKGID != tcredlock.EffectiveKGID
                || credlock.RealKGID != tcredlock.SavedKGID
            {
                return Err(Error::SysError(SysErr::EPERM));
            }
        }
    }

    let oldLim = PrLimit64(&ot, resource, newlim)?;

    if oldRlimAddr != 0 {
        let rlim = RLimit64::FromLimit(&oldLim);
        //*task.GetTypeMut(oldRlimAddr)? = rlim;
        task.CopyOutObj(&rlim, oldRlimAddr)?;
    }

    return Ok(0);
}
