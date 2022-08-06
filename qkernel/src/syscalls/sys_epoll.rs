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

use super::super::kernel::epoll::epoll::*;
use super::super::kernel::epoll::epoll_entry::*;
use super::super::kernel::fd_table::*;
use super::super::kernel::time::*;
use super::super::kernel::timer::*;
use super::super::kernel::waiter::*;
use super::super::qlib::addr::*;
use super::super::qlib::common::*;
use super::super::qlib::linux::time::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;
use super::super::SignalDef::*;
use super::sys_poll::CopyTimespecIntoDuration;

// CreateEpoll implements the epoll_create(2) linux syscall.
pub fn CreateEpoll(task: &Task, closeOnExec: bool) -> Result<i64> {
    let file = NewEventPoll(task);

    let flags = FDFlags {
        CloseOnExec: closeOnExec,
    };

    let fd = task.NewFDFrom(0, &file, &flags)?;

    return Ok(fd as i64);
}

// AddEpoll implements the epoll_ctl(2) linux syscall when op is EPOLL_CTL_ADD.
pub fn AddEpoll(
    task: &Task,
    epfd: i32,
    fd: i32,
    flags: EntryFlags,
    mask: EventMask,
    userData: [i32; 2],
) -> Result<()> {
    // Get epoll from the file descriptor.
    let epollfile = task.GetFile(epfd)?;

    // Get the target file id.
    let file = task.GetFile(fd)?;

    let inode = file.Dirent.Inode();

    //the fd doesn't support epoll
    let inodeOp = inode.lock().InodeOp.clone();

    if !inodeOp.WouldBlock() {
        //error!("AddEpoll 1.1 inodetype is {:?}, fopstype is {:?}", inode.InodeType(), fops.FopsType());
        return Err(Error::SysError(SysErr::EPERM));
    }

    let fops = epollfile.FileOp.clone();
    let ep = match fops.as_any().downcast_ref::<EventPoll>() {
        None => return Err(Error::SysError(SysErr::EBADF)),
        Some(ep) => ep,
    };

    return ep.AddEntry(
        task,
        fd,
        file,
        flags,
        mask,
        userData,
    );
}

// UpdateEpoll implements the epoll_ctl(2) linux syscall when op is EPOLL_CTL_MOD.
pub fn UpdateEpoll(
    task: &Task,
    epfd: i32,
    fd: i32,
    flags: EntryFlags,
    mask: EventMask,
    userData: [i32; 2],
) -> Result<()> {
    // Get epoll from the file descriptor.
    let epollfile = task.GetFile(epfd)?;

    // Get the target file id.
    let file = task.GetFile(fd)?;

    let fops = epollfile.FileOp.clone();
    let ep = match fops.as_any().downcast_ref::<EventPoll>() {
        None => return Err(Error::SysError(SysErr::EBADF)),
        Some(ep) => ep,
    };

    return ep.UpdateEntry(
        task,
        file,
        flags,
        mask,
        userData,
    );
}

pub fn RemoveEpoll(task: &Task, epfd: i32, fd: i32) -> Result<()> {
    // Get epoll from the file descriptor.
    let epollfile = task.GetFile(epfd)?;

    // Get the target file id.
    let file = task.GetFile(fd)?;

    let fops = epollfile.FileOp.clone();
    let ep = match fops.as_any().downcast_ref::<EventPoll>() {
        None => return Err(Error::SysError(SysErr::EBADF)),
        Some(ep) => ep,
    };

    // Try to remove the entry.
    return ep.RemoveEntry(
        task,
        file,
    );
}

// WaitEpoll implements the epoll_wait(2) linux syscall.
pub fn WaitEpoll(task: &Task, epfd: i32, max: i32, timeout: i64) -> Result<Vec<Event>> {
    // Get epoll from the file descriptor.
    let epollfile = task.GetFile(epfd)?;

    let fops = epollfile.FileOp.clone();
    let ep = match fops.as_any().downcast_ref::<EventPoll>() {
        None => return Err(Error::SysError(SysErr::EBADF)),
        Some(ep) => ep,
    };

    // Try to read events and return right away if we got them or if the
    // caller requested a non-blocking "wait".
    let r = ep.ReadEvents(task, max);
    if r.len() != 0 {
        return Ok(r);
    }

    if timeout == 0 {
        super::super::taskMgr::Yield(); // yield vcpu to avoid live lock
        return Ok(r);
    }

    if timeout == 0 {
        super::super::taskMgr::Yield(); // yield vcpu to avoid live lock
        return Ok(r)
    }

    // We'll have to wait. Set up the timer if a timeout was specified and
    // and register with the epoll object for readability events.
    let mut deadline = None;

    if timeout > 0 {
        let now = MonotonicNow();
        deadline = Some(Time(now + timeout));
    }

    let general = task.blocker.generalEntry.clone();
    ep.EventRegister(task, &general, EVENT_READ);
    defer!(ep.EventUnregister(task, &general));

    // Try to read the events again until we succeed, timeout or get
    // interrupted.
    loop {
        let r = ep.ReadEvents(task, max);
        if r.len() != 0 {
            return Ok(r);
        }

        //let start = super::super::asm::Rdtsc();
        match task.blocker.BlockWithMonoTimer(true, deadline) {
            Err(Error::ErrInterrupted) => {
                return Err(Error::SysError(SysErr::EINTR));
            }
            Err(e) => {
                return Err(e);
            }
            _ => (),
        }

        //error!("WaitEpoll after block timeout is {}", timeout);
    }
}

// EpollCreate1 implements the epoll_create1(2) linux syscall.
pub fn SysEpollCreate1(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let flags = args.arg0 as i32;

    if flags & !LibcConst::EPOLL_CLOEXEC as i32 != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let closeOnExec = flags & LibcConst::EPOLL_CLOEXEC as i32 != 0;
    let fd = CreateEpoll(task, closeOnExec)?;

    return Ok(fd);
}

// EpollCreate implements the epoll_create(2) linux syscall.
pub fn SysEpollCreate(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let size = args.arg0 as i32;

    if size <= 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let fd = CreateEpoll(task, false)?;

    return Ok(fd);
}

// EpollCtl implements the epoll_ctl(2) linux syscall.
pub fn SysEpollCtl(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let epfd = args.arg0 as i32;
    let op = args.arg1 as i32;
    let fd = args.arg2 as i32;
    let eventAddr = args.arg3 as u64;

    // Capture the event state if needed.
    let mut flags = 0;
    let mut mask = 0;
    let mut data: [i32; 2] = [0, 0];

    if op != LibcConst::EPOLL_CTL_DEL as i32 {
        let e: EpollEvent = task.CopyInObj(eventAddr)?;

        if e.Events & LibcConst::EPOLLONESHOT as u32 != 0 {
            flags |= ONE_SHOT;
        }

        if e.Events & (-LibcConst::EPOLLET) as u32 != 0 {
            flags |= EDGE_TRIGGERED;
        }

        mask = EventMaskFromLinux(e.Events);
        data[0] = e.FD;
        data[1] = e.Pad;
    }

    // Perform the requested operations.
    match op as u64 {
        LibcConst::EPOLL_CTL_ADD => {
            // See fs/eventpoll.c.
            mask |= EVENT_HUP | EVENT_ERR;
            AddEpoll(task, epfd, fd, flags, mask, data)?;
            return Ok(0);
        }
        LibcConst::EPOLL_CTL_DEL => {
            RemoveEpoll(task, epfd, fd)?;
            return Ok(0);
        }
        LibcConst::EPOLL_CTL_MOD => {
            // Same as EPOLL_CTL_ADD.
            UpdateEpoll(task, epfd, fd, flags, mask, data)?;
            return Ok(0);
        }
        _ => return Err(Error::SysError(SysErr::EINVAL)),
    }
}

// copyOutEvents copies epoll events from the kernel to user memory.
pub fn CopyOutEvents(task: &Task, addr: u64, e: &[Event]) -> Result<()> {
    let itemLen: usize = 12;

    Addr(addr).AddLen((itemLen * e.len()) as u64)?;

    //error!("epool CopyOutEvents events is {:x?}", e);
    for i in 0..e.len() {
        /*let output : &mut Event = task.GetTypeMut(addr + (i * itemLen) as u64)?;
        output.Events = e[i].Events;
        output.Data[0] = e[i].Data[0];
        output.Data[1] = e[i].Data[1];*/
        task.CopyOutObj(&e[i], addr + (i * itemLen) as u64)?;
    }

    return Ok(());
}

// EpollWait implements the epoll_wait(2) linux syscall.
pub fn SysEpollWait(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let epfd = args.arg0 as i32;
    let eventAddr = args.arg1 as u64;
    let maxEvents = args.arg2 as i32;
    let timeout = args.arg3 as i64 * MILLISECOND;

    let r = match WaitEpoll(task, epfd, maxEvents, timeout) {
        Err(Error::SysError(SysErr::ETIMEDOUT)) => return Ok(0),
        Err(e) => return Err(e),
        Ok(r) => r,
    };

    if r.len() != 0 {
        CopyOutEvents(task, eventAddr, &r)?;
    }

    return Ok(r.len() as i64);
}

// EpollPwait implements the epoll_pwait(2) linux syscall.
pub fn SysPwait(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let maskAddr = args.arg4 as u64;
    let maskSize = args.arg5 as u32;

    if maskAddr != 0 {
        let mask = CopyInSigSet(task, maskAddr, maskSize as usize)?;

        let thread = task.Thread();
        let oldmask = thread.SignalMask();
        thread.SetSignalMask(mask);
        thread.SetSavedSignalMask(oldmask);
    }

    return SysEpollWait(task, args);
}

// EpollPwait2 implements the epoll_pwait(2) linux syscall.
pub fn SysPwait2(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let epfd = args.arg0 as i32;
    let eventAddr = args.arg1 as u64;
    let maxEvents = args.arg2 as i32;
    let timeoutPtr = args.arg3 as u64;
    let maskAddr = args.arg4 as u64;
    let maskSize = args.arg5 as u32;

    let haveTimeout = timeoutPtr != 0;

    let mut timeout = -1;
    if haveTimeout {
        timeout = CopyTimespecIntoDuration(task, timeoutPtr)?;
    }

    if maskAddr != 0 {
        let mask = CopyInSigSet(task, maskAddr, maskSize as usize)?;

        let thread = task.Thread();
        let oldmask = thread.SignalMask();
        thread.SetSignalMask(mask);
        thread.SetSavedSignalMask(oldmask);
    }

    let r = match WaitEpoll(task, epfd, maxEvents, timeout) {
        Err(Error::SysError(SysErr::ETIMEDOUT)) => return Ok(0),
        Err(e) => return Err(e),
        Ok(r) => r,
    };

    if r.len() != 0 {
        CopyOutEvents(task, eventAddr, &r)?;
    }

    return Ok(r.len() as i64);
}