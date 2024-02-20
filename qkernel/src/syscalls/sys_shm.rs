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
use super::super::qlib::kernel::kernel::shm::*;
use super::super::qlib::linux::ipc::*;
use super::super::qlib::linux::shm::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;

// Shmget implements shmget(2).
pub fn SysShmget(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let key = args.arg0 as i32;
    let size = args.arg1 as i64;
    let flag = args.arg2 as i32;

    let private = key == IPC_PRIVATE;
    let create = (flag & IPC_CREAT as i32) == IPC_CREAT as i32;
    let exclusive = flag & IPC_EXCL as i32 == IPC_EXCL as i32;
    let mode = FileMode((flag as u16) & 0o777);

    let pid = task.Thread().ThreadGroup().ID();
    let r = task.IPCNamespace().ShmRegistry();
    let segment = r.FindOrCreate(
        task,
        pid,
        key,
        size as u64,
        &mode,
        private,
        create,
        exclusive,
    )?;
    return Ok(segment.Id() as _);
}

// findSegment retrives a shm segment by the given id.
//
// findSegment returns a reference on Shm.
fn FindSegment(task: &Task, id: i32) -> Result<Shm> {
    let r = task.IPCNamespace().ShmRegistry();
    match r.FindByID(id) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(segment) => return Ok(segment),
    }
}

// Shmat implements shmat(2).
pub fn SysShmat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let id = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let flag = args.arg2 as u16;

    let segment = FindSegment(task, id)?;
    let mut opts = segment.ConfigureAttach(
        task,
        addr,
        &AttachOpts {
            Execute: flag & SHM_EXEC == SHM_EXEC,
            ReadOnly: flag & SHM_RDONLY == SHM_RDONLY,
            Remap: flag & SHM_REMAP == SHM_REMAP,
        },
    )?;

    let addr = task.mm.MMap(task, &mut opts)?;
    let pid = task.Thread().ThreadGroup().ID();
    segment.lock().lastAttachDetachPID = pid;
    segment.lock().attachTime = task.Now();
    return Ok(addr as _);
}

// Shmdt implements shmdt(2).
pub fn SysShmdt(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg1 as u64;

    task.mm.DetachShm(task, addr)?;
    return Ok(0);
}

// Shmctl implements shmctl(2).
pub fn SysShmctl(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let id = args.arg0 as i32;
    let cmd = args.arg1 as i32;
    let buf = args.arg2 as u64;

    let r = task.IPCNamespace().ShmRegistry();

    match cmd {
        // Technically, we should be treating id as "an index into the kernel's
        // internal array that maintains information about all shared memory
        // segments on the system". Since we don't track segments in an array,
        // we'll just pretend the shmid is the index and do the same thing as
        // IPC_STAT. Linux also uses the index as the shmid.
        SHM_STAT | IPC_STAT => {
            let segment = FindSegment(task, id)?;
            let stat = segment.IPCStat(task)?;
            task.CopyOutObj(&stat, buf)?;
            return Ok(0);
        }
        IPC_INFO => {
            let params = r.IPCInfo();
            task.CopyOutObj(&params, buf)?;
            return Ok(0);
        }
        SHM_INFO => {
            let info = r.ShmInfo();
            task.CopyOutObj(&info, buf)?;
            return Ok(0);
        }
        _ => (),
    }

    // Remaining commands refer to a specific segment.
    let segment = FindSegment(task, id)?;
    match cmd {
        IPC_SET => {
            let ds: ShmidDS = task.CopyInObj(buf)?;
            segment.Set(task, &ds)?;
            return Ok(0);
        }
        IPC_RMID => {
            segment.MarkDestroyed();
            return Ok(0);
        }
        SHM_LOCK | SHM_UNLOCK => {
            // We currently do not support memory locking anywhere.
            // mlock(2)/munlock(2) are currently stubbed out as no-ops so do the
            // same here.
            error!("SysShmctl unsupport cmd {}", cmd);
        }
        _ => (),
    }

    return Err(Error::SysError(SysErr::EINVAL));
}
