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
use super::super::qlib::kernel::kernel::ipc_namespace::*;
use super::super::qlib::kernel::kernel::msgqueue::*;
use super::super::qlib::linux::ipc::*;
use super::super::qlib::linux::msgqueue::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;

// Msgget implements msgget(2).
pub fn SysMsgget(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let key = args.arg0 as i32;
    let flag = args.arg1 as i32;

    let private = key == IPC_PRIVATE;
    let create = (flag & IPC_CREAT as i32) == IPC_CREAT as i32;
    let exclusive = flag & IPC_EXCL as i32 == IPC_EXCL as i32;
    let mode = FileMode((flag as u16) & 0o777);

    let r = task.IPCNamespace().MsgqueueRegistry();
    let queue = r.FindOrCreate(task, key, &mode, private, create, exclusive)?;
    return Ok(queue.Id() as _);
}

pub fn CopyInMsg(task: &Task, addr: u64, size: usize) -> Result<Message> {
    let type_: i64 = task.CopyInObj(addr)?;
    let text: Vec<u8> = task.CopyInVec(addr + 8, size)?;
    return Ok(Message::New(type_, text));
}

// Msgsnd implements msgsnd(2).
pub fn SysMsgsnd(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let id = args.arg0 as i32;
    let msgAddr = args.arg1 as u64;
    let size = args.arg2 as i64;
    let flags = args.arg3 as i16;

    if size < 0 || size > MSGMAX as _ {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let wait = flags & IPC_NOWAIT != IPC_NOWAIT;
    let pid = task.Thread().ThreadGroup().ID();

    let msg = CopyInMsg(task, msgAddr, size as usize)?;
    let r = task.IPCNamespace().MsgqueueRegistry();
    let queue = r.FindById(id)?;
    queue.Send(task, &msg, wait, pid)?;
    return Ok(0);
}

// Msgrcv implements msgrcv(2).
pub fn SysMsgrcv(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let id = args.arg0 as i32;
    let msgAddr = args.arg1 as u64;
    let size = args.arg2 as i64;
    let mType = args.arg3 as i64;
    let flags = args.arg4 as i32;

    let wait = (flags as i16) & IPC_NOWAIT != IPC_NOWAIT;
    let except = flags & MSG_EXCEPT == MSG_EXCEPT;
    let truncate = flags & MSG_NOERROR == MSG_NOERROR;
    let msgCopy = flags & MSG_COPY == MSG_COPY;

    let msg = Receive(task, id, mType, size, msgCopy, wait, truncate, except)?;
    let size = msg.Size();
    CopyOutMsg(task, msgAddr, msg)?;
    return Ok(size as _);
}

pub fn CopyOutMsg(task: &Task, addr: u64, msg: Message) -> Result<()> {
    let msg = msg.lock();
    task.CopyOutObj(&msg.Type, addr)?;
    task.CopyOutSlice(&msg.Text, addr + 8, msg.Text.len())?;
    return Ok(());
}

// receive returns a message from the queue with the given ID. If msgCopy is
// true, a message is copied from the queue without being removed. Otherwise,
// a message is removed from the queue and returned.
pub fn Receive(
    task: &Task,
    id: ID,
    mType: i64,
    maxSize: i64,
    msgCopy: bool,
    wait: bool,
    truncate: bool,
    except: bool,
) -> Result<Message> {
    let pid = task.Thread().ThreadGroup().ID();

    let r = task.IPCNamespace().MsgqueueRegistry();
    let queue = r.FindById(id)?;
    if msgCopy {
        if wait || except {
            return Err(Error::SysError(SysErr::EINVAL));
        }
        return queue.Copy(mType);
    }

    return queue.Receive(task, mType, maxSize, wait, truncate, except, pid);
}

// Msgctl implements msgctl(2).
pub fn SysMsgctl(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let id = args.arg0 as i32;
    let cmd = args.arg1 as i32;
    let buf = args.arg2 as u64;

    let creds = task.creds.clone();
    let r = task.IPCNamespace().MsgqueueRegistry();

    match cmd {
        IPC_INFO => {
            let info = r.IPCInfo(task);
            task.CopyOutObj(&info, buf)?;
            return Ok(0);
        }
        MSG_INFO => {
            let info = r.MsgInfo(task);
            task.CopyOutObj(&info, buf)?;
            return Ok(0);
        }
        IPC_RMID => {
            r.Remove(id, &creds)?;
            return Ok(0);
        }
        _ => (),
    }

    // Remaining commands use a queue.
    let queue = r.FindById(id)?;

    match cmd {
        // Technically, we should be treating id as "an index into the kernel's
        // internal array that maintains information about all shared memory
        // segments on the system". Since we don't track segments in an array,
        // we'll just pretend the msqid is the index and do the same thing as
        // IPC_STAT. Linux also uses the index as the msqid.
        MSG_STAT | IPC_STAT => {
            let stat = queue.Stat(task)?;
            task.CopyOutObj(&stat, buf)?;
            return Ok(0);
        }
        MSG_STAT_ANY => {
            let stat = queue.StatAny(task)?;
            task.CopyOutObj(&stat, buf)?;
            return Ok(0);
        }
        IPC_SET => {
            let ds: MsqidDS = match task.CopyInObj(buf) {
                Err(_) => return Err(Error::SysError(SysErr::EINVAL)),
                Ok(ds) => ds,
            };
            queue.Set(task, &ds)?;
            return Ok(0);
        }
        _ => return Err(Error::SysError(SysErr::EINVAL)),
    }
}
