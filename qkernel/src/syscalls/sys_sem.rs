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

use alloc::vec::Vec;

use super::super::qlib::common::*;
use super::super::kernel::timer::*;
use super::super::qlib::linux::ipc::*;
use super::super::qlib::linux::sem::*;
use super::super::qlib::linux::time::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::auth::id::*;
use super::super::qlib::auth::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;
use kernel::time::Time;

pub const OPS_MAX : i32 = 500; // SEMOPM

// Semget handles: semget(key_t key, int nsems, int semflg)
pub fn SysSemgetl(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let key = args.arg0 as i32;
    let nsems = args.arg1 as i32;
    let flag = args.arg2 as i32;

    let private = key == IPC_PRIVATE;
    let create = (flag & IPC_CREAT as i32) == IPC_CREAT as i32;
    let exclusive = flag & IPC_EXCL as i32 == IPC_EXCL as i32;
    let mode = FileMode((flag as u16) & 0o777);

    let r = task.IPCNamespace().SemaphoreRegistry();
    let set = r.FindOrCreate(task, key, nsems, mode, private, create, exclusive)?;
    return Ok(set.Id() as i64)
}


// Semtimedop handles: semop(int semid, struct sembuf *sops, size_t nsops, const struct timespec *timeout)
pub fn SysSemtimedop(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let id = args.arg0 as i32;
    let sembufAddr = args.arg1 as u64;
    let nsops = args.arg2 as u32;
    let timespecAddr = args.arg3 as u64;

    if timespecAddr == 0 {
        return SysSemop(task, args);
    }

    if nsops <= 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    if nsops > OPS_MAX as _ {
        return Err(Error::SysError(SysErr::E2BIG))
    }

    let ops : Vec<Sembuf> = task.CopyInVec(sembufAddr, nsops as usize)?;
    let ts : Timespec = task.CopyInObj::<Timespec>(timespecAddr)?;
    SemTimedOp(task, id, &ops, Some(ts.ToDuration()?))?;
    return Ok(0)
}

// Semop handles: semop(int semid, struct sembuf *sops, size_t nsops)
pub fn SysSemop(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let id = args.arg0 as i32;
    let sembufAddr = args.arg1 as u64;
    let nsops = args.arg2 as u32;

    if nsops <= 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    if nsops > OPS_MAX as _ {
        return Err(Error::SysError(SysErr::E2BIG))
    }

    let ops : Vec<Sembuf> = task.CopyInVec(sembufAddr, nsops as usize)?;
    SemTimedOp(task, id, &ops, None)?;
    return Ok(0)
}

pub fn SemTimedOp(task: &Task, id: i32, ops: &[Sembuf], timeout: Option<Duration>) -> Result<()> {
    let set = match task.IPCNamespace().SemaphoreRegistry().lock().findByID(id) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(s) => s
    };

    let creds = task.creds.clone();
    let thread = task.Thread();
    let tg = thread.ThreadGroup();

    let pidns = thread.PIDNamespace();
    let pid = pidns.IDOfThreadGroup(&tg);

    let deadline = match timeout  {
        None => None,
        Some(dur) => {
            let now = MonotonicNow();
            Some(Time(now + dur))
        }
    };

    let general = task.blocker.generalEntry.clone();
    loop {
        let (waiterid, id) = set.ExecuteOps(task, ops, &creds, &general, pid)?;
        if id == -1 {
            return Ok(())
        }

        defer!(set.AbortWait(task, waiterid, id, &general));
        let ret = task.blocker.BlockWithMonoTimer(true, deadline);
        match ret {
            Err(Error::ErrInterrupted) => {
                return Err(Error::SysError(SysErr::ERESTARTNOHAND));
            }
            Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                return Err(Error::SysError(SysErr::EAGAIN));
            }
            Err(e) => {
                return Err(e);
            }
            _ => (),
        }
    }
}

// Semctl handles: semctl(int semid, int semnum, int cmd, ...)
pub fn SysSemctl(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let id = args.arg0 as i32;
    let num = args.arg1 as i32;
    let cmd = args.arg2 as i32;

    match cmd {
        SETVAL => {
            let val = args.arg3 as i32;
            if val > i16::MAX as i32 {
                return Err(Error::SysError(SysErr::ERANGE))
            }

            setVal(task, id, num, val as _)?;
            return Ok(0)
        }
        SETALL => {
            let array = args.arg3 as u64;
            setValAll(task, id, array)?;
            return Ok(0)
        }
        GETVAL => {
            let v = getVal(task, id, num)?;
            return Ok(v as _)
        }
        GETALL => {
            let array = args.arg3 as u64;
            getValAll(task, id, array)?;
            return Ok(0)
        }
        IPC_RMID => {
            remove(task, id)?;
            return Ok(0)
        }
        IPC_SET => {
            let arg = args.arg3 as u64;
            let s : SemidDS = task.CopyInObj::<SemidDS>(arg)?;
            let mode = FileMode(s.SemPerm.Mode & 0o777);
            let perm = FilePermissions::FromMode(mode);
            ipcSet(task, id, UID(s.SemPerm.UID), GID(s.SemPerm.GID), &perm)?;
            return Ok(0)
        }
        GETPID => {
            let v = getPid(task, id, num)?;
            return Ok(v as _)
        }
        IPC_STAT => {
            let arg = args.arg3 as u64;
            let ds = ipcStat(task, id)?;
            task.CopyOutObj(&ds, arg)?;
            return Ok(0)
        }
        GETZCNT => {
            let v = getSemzcnt(task, id, num)?;
            return Ok(v as _)
        }
        GETNCNT => {
            let v = getSemncnt(task, id, num)?;
            return Ok(v as _)
        }
        IPC_INFO => {
            let buf = args.arg3 as u64;
            let r = task.IPCNamespace().SemaphoreRegistry();
            let info = r.IPCInfo();
            task.CopyOutObj(&info, buf)?;
            let lastIdx = r.lock().HighestIndex();
            return Ok(lastIdx as _)
        }
        SEM_INFO => {
            let buf = args.arg3 as u64;
            let r = task.IPCNamespace().SemaphoreRegistry();
            let info = r.SemInfo();
            task.CopyOutObj(&info, buf)?;
            let lastIdx = r.lock().HighestIndex();
            return Ok(lastIdx as _)
        }
        SEM_STAT => {
            let arg = args.arg3 as u64;
            let (ds, semid) = semStat(task, id)?;
            task.CopyOutObj(&ds, arg)?;
            return Ok(semid as _)
        }
        SEM_STAT_ANY => {
            let arg = args.arg3 as u64;
            let (ds, semid) = semStatAny(task, id)?;
            task.CopyOutObj(&ds, arg)?;
            return Ok(semid as _)
        }
        _ => {
            return Err(Error::SysError(SysErr::EINVAL));
        }
    }
}

fn remove(task: &Task, id: i32) -> Result<()> {
    let r = task.IPCNamespace().SemaphoreRegistry();
    let creds = task.creds.clone();
    r.RemoveId(id, &creds)
}

fn ipcSet(task: &Task, id: i32, uid: UID, gid: GID, perms: &FilePermissions) -> Result<()> {
    let set = match task.IPCNamespace().SemaphoreRegistry().lock().findByID(id) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(s) => s
    };

    let creds = task.creds.clone();
    let kuid = creds.lock().UserNamespace.MapToKUID(uid);
    if !kuid.Ok() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let kgid = creds.lock().UserNamespace.MapToKGID(gid);
    if !kgid.Ok() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let owner= FileOwner {
        UID: kuid,
        GID: kgid,
    };

    return set.Change(task, &creds, &owner, perms)
}

fn ipcStat(task: &Task, id: i32) -> Result<SemidDS>{
    let set = match task.IPCNamespace().SemaphoreRegistry().lock().findByID(id) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(s) => s
    };
    let creds = task.creds.clone();
    return set.GetStat(&creds);
}

fn semStat(task: &Task, id: i32) -> Result<(SemidDS, i32)>{
    let set = match task.IPCNamespace().SemaphoreRegistry().lock().findByID(id) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(s) => s
    };
    let creds = task.creds.clone();
    let ds = set.GetStat(&creds)?;
    return Ok((ds, set.Id()))
}

fn semStatAny(task: &Task, id: i32) -> Result<(SemidDS, i32)>{
    let set = match task.IPCNamespace().SemaphoreRegistry().lock().findByID(id) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(s) => s
    };
    let creds = task.creds.clone();
    let ds = set.GetStatAny(&creds)?;
    return Ok((ds, set.Id()))
}

fn setVal(task: &Task, id: i32, num: i32, val: i16) -> Result<()> {
    let set = match task.IPCNamespace().SemaphoreRegistry().lock().findByID(id) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(s) => s
    };
    let creds = task.creds.clone();
    let thread = task.Thread();
    let tg = thread.ThreadGroup();

    let pidns = thread.PIDNamespace();
    let pid = pidns.IDOfThreadGroup(&tg);
    return set.SetVal(task, num, val, &creds, pid)
}

fn setValAll(task: &Task, id: i32, array: u64) -> Result<()> {
    let set = match task.IPCNamespace().SemaphoreRegistry().lock().findByID(id) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(s) => s
    };

    let vals: Vec<u16> = task.CopyInVec(array, set.Size() as usize)?;

    let creds = task.creds.clone();
    let thread = task.Thread();
    let tg = thread.ThreadGroup();

    let pidns = thread.PIDNamespace();
    let pid = pidns.IDOfThreadGroup(&tg);
    return set.SetValAll(task, &vals, &creds, pid)
}


fn getVal(task: &Task, id: i32, num: i32) -> Result<i16> {
    let set = match task.IPCNamespace().SemaphoreRegistry().lock().findByID(id) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(s) => s
    };

    let creds = task.creds.clone();
    return set.GetVal(num, &creds);
}

fn getValAll(task: &Task, id: i32, array: u64) -> Result<()> {
    let set = match task.IPCNamespace().SemaphoreRegistry().lock().findByID(id) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(s) => s
    };

    let creds = task.creds.clone();
    let values = set.GetValAll(&creds)?;
    let cnt = values.len();
    task.CopyOutSlice(&values[0..cnt], array, cnt)?;
    return Ok(())
}

fn getPid(task: &Task, id: i32, num: i32) -> Result<i32> {
    let set = match task.IPCNamespace().SemaphoreRegistry().lock().findByID(id) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(s) => s
    };

    let creds = task.creds.clone();
    let gpid = set.GetPID(num, &creds)?;

    let pidns = task.Thread().PIDNamespace();
    match pidns.ThreadGroupWithID(gpid) {
        None => return Ok(0),
        Some(tg) => return Ok(tg.ID())
    }
}

fn getSemzcnt(task: &Task, id: i32, num: i32) -> Result<u16> {
    let set = match task.IPCNamespace().SemaphoreRegistry().lock().findByID(id) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(s) => s
    };

    let creds = task.creds.clone();
    return set.GetZeroWaiters(num, &creds)
}

fn getSemncnt(task: &Task, id: i32, num: i32) -> Result<u16> {
    let set = match task.IPCNamespace().SemaphoreRegistry().lock().findByID(id) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(s) => s
    };

    let creds = task.creds.clone();
    return set.GetNegativeWaiters(num, &creds)
}