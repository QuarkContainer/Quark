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
use alloc::boxed::Box;
use alloc::collections::btree_map::BTreeMap;

use super::super::task::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::linux::time::*;
use super::super::SignalDef::*;
use super::super::syscalls::syscalls::*;
use super::super::kernel::waiter::*;
use super::super::kernel::timer::*;
use super::super::fs::file::*;
use super::super::threadmgr::task_syscall::*;

// fileCap is the maximum allowable files for poll & select.
pub const FILE_CAP : i32 = 1024 * 1024;

// SELECT_READ_EVENTS is analogous to the Linux kernel's
// fs/select.c:POLLIN_SET.
pub const SELECT_READ_EVENTS : i16 = (LibcConst::EPOLLIN | LibcConst::EPOLLHUP | LibcConst::EPOLLERR) as i16;

// SELECT_WRITE_EVENTS is analogous to the Linux kernel's
// fs/select.c:POLLOUT_SET.
pub const SELECT_WRITE_EVENTS : i16 = (LibcConst::EPOLLOUT | LibcConst::EPOLLERR) as i16;

// SELECT_EXCEPT_EVENTS is analogous to the Linux kernel's
// fs/select.c:POLLEX_SET.
pub const SELECT_EXCEPT_EVENTS : i16 = (LibcConst::EPOLLPRI) as i16;

pub const TIMEOUT_PROCESS_TIME : i64 = 30_000;

pub fn DoSelect(task: &Task, nfds: i32, readfds: u64, writefds: u64, exceptfds: u64, timeout: i64) -> Result<i64> {
    if nfds == 0 {
        if timeout == 0 {
            return Ok(0)
        }

        let (_remain, res) = task.blocker.BlockWithMonoTimeout(false, Some(timeout));
        match res {
            Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                return Ok(0)
            }
            Err(e) => {
                return Err(e)
            }
            Ok(()) => return Ok(0)
        };
    }

    if nfds < 0 || nfds > FILE_CAP {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    // Capture all the provided input vectors.
    let byteCount = ((nfds + 7) / 8) as usize;

    let bitsPartial = (nfds % 8) as usize;
    let mut r = Vec::with_capacity(byteCount);
    let mut w = Vec::with_capacity(byteCount);
    let mut e = Vec::with_capacity(byteCount);

    if readfds != 0 {
        let arr = task.GetSliceMut::<u8>(readfds, byteCount)?;

        for b in arr {
            r.push(*b)
        }

        if bitsPartial != 0 {
            r[byteCount-1] &= !(0xff << bitsPartial)
        }
    } else {
        for _i in 0..byteCount {
            r.push(0)
        }
    }

    if writefds != 0 {
        let arr = task.GetSliceMut::<u8>(writefds, byteCount)?;

        for b in arr {
            w.push(*b)
        }

        if bitsPartial != 0 {
            w[byteCount-1] &= !(0xff << bitsPartial)
        }
    } else {
        for _i in 0..byteCount {
            w.push(0)
        }
    }

    if exceptfds != 0 {
        let arr = task.GetSliceMut::<u8>(exceptfds, byteCount)?;

        for b in arr {
            e.push(*b)
        }

        if bitsPartial != 0 {
            e[byteCount-1] &= !(0xff << bitsPartial)
        }
    } else {
        for _i in 0..byteCount {
            e.push(0)
        }
    }

    let mut fdcnt = 0;
    for i in 0..byteCount {
        let mut v = r[i] | w[i] | e[i];
        while v != 0 {
            v &= v-1;
            fdcnt += 1;
        }
    }

    // Build the PollFD array.
    let mut pfd = Vec::with_capacity(fdcnt);
    let mut fd : i32 = 0;
    for i in 0..byteCount {
        let rv = r[i];
        let wv = w[i];
        let ev = e[i];
        let v = rv | wv | ev;

        let mut m : u8 = 1;
        for _j in 0..8 {
            if (v & m) != 0 {
                // Make sure the fd is valid and decrement the reference
                // immediately to ensure we don't leak. Note, another thread
                // might be about to close fd. This is racy, but that's
                // OK. Linux is racy in the same way.
                let _file = task.GetFile(fd)?;

                let mut mask : i16 = 0;
                if rv & m !=0 {
                    mask |= SELECT_READ_EVENTS;
                }

                if wv & m != 0 {
                    mask |= SELECT_WRITE_EVENTS;
                }

                if ev & m != 0 {
                    mask |= SELECT_EXCEPT_EVENTS;
                }

                pfd.push(PollFd {
                    fd: fd,
                    events: mask,
                    revents: 0,
                })
            }

            fd += 1;
            m <<= 1;
        }
    }

    // Do the syscall, then count the number of bits set.
    let (_, res) = PollBlock(task, &mut pfd, timeout);
    match res {
        Err(_) => return Err(Error::SysError(SysErr::EINTR)),
        Ok(_) => (),
    }

    // r, w, and e are currently event mask bitsets; unset bits corresponding
    // to events that *didn't* occur.
    let mut bitSetCount = 0;
    for idx in 0..pfd.len() {
        let events = pfd[idx].revents;
        let i = (pfd[idx].fd / 8) as usize;
        let j = (pfd[idx].fd % 8) as usize;
        let m : u8 = 1 << j;
        if r[i] & m != 0 {
            if (events & SELECT_READ_EVENTS) != 0 {
                bitSetCount += 1;
            } else {
                r[i] &= !m;
            }
        }
        if w[i] & m != 0 {
            if (events & SELECT_WRITE_EVENTS) != 0 {
                bitSetCount += 1;
            } else {
                w[i] &= !m;
            }
        }
        if e[i] & m != 0 {
            if (events & SELECT_EXCEPT_EVENTS) != 0 {
                bitSetCount += 1;
            } else {
                w[i] &= !m;
            }
        }
    }

    // Copy updated vectors back.
    if readfds != 0 {
        let arr = task.GetSliceMut::<u8>(readfds, byteCount)?;

        for i in 0..arr.len() {
            arr[i] = r[i];
        }
    }

    if writefds != 0 {
        let arr = task.GetSliceMut::<u8>(writefds, byteCount)?;

        for i in 0..arr.len() {
            arr[i] = w[i];
        }
    }

    if exceptfds != 0 {
        let arr = task.GetSliceMut::<u8>(exceptfds, byteCount)?;

        for i in 0..arr.len() {
            arr[i] = e[i];
        }
    }

    return Ok(bitSetCount)
}


pub fn PollBlock(task: &Task, pfd: &mut [PollFd], timeout: i64) -> (Duration, Result<usize>) {
    // no fd to wait, just a nansleep
    if pfd.len() == 0 {
        if timeout == 0 {
            return (0, Ok(0))
        }

        let timeout = if timeout >= 0 {
            timeout
        } else {
            core::i64::MAX //if pfd is empty, timeout < 0, needs to wait forever
        };

        let (remain, res) = task.blocker.BlockWithMonoTimeout(false, Some(timeout));
        match res {
            Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                return (0, Ok(0))
            }
            Err(e) => {
                return (remain, Err(e))
            }
            Ok(()) => return (remain, Ok(0))
        };
    }

    let mut timeout = timeout;
    let general = task.blocker.generalEntry.clone();

    let mut n = 0;

    //info!("PollBlock 1, pfd is {:?}", pfd);
    // map <File -> (Mask, Readiness)>
    let mut waits = BTreeMap::new();

    for i in 0..pfd.len() {
        match task.GetFile(pfd[i].fd) {
            Err(_) => {
                pfd[i].revents = PollConst::POLLNVAL as i16;
            },
            Ok(f) => {
                match waits.get_mut(&f) {
                    None => {
                        let r = f.Readiness(task, EventMaskFromLinux(pfd[i].events as u32));
                        pfd[i].revents = ToLinux(r) as i16 & pfd[i].events;
                        waits.insert(f, (pfd[i].events, r));
                    }
                    Some(t) => {
                        (*t).0 |= pfd[i].events;
                        pfd[i].revents = (*t).1 as i16 & pfd[i].events;
                    }
                }
            }
        };

        if pfd[i].revents != 0 {
            n += 1;
        }
    }

    if n > 0 {
        return (timeout, Ok(n))
    }

    for (f, (mask, _)) in waits.iter() {
        f.EventRegister(task, &general, EventMaskFromLinux(*mask as u32));
    }

    defer!(
        for f in waits.keys() {
            f.EventUnregister(task, &general);
        }
    );

    if timeout == 0 {
        return (timeout, Ok(n));
    }

    while n == 0 {
        // before we got notified, we have to count how many files are ready in case
        // there is ready when we register the event. If none,
        // then this was a spurious notification, and we just go back
        // to sleep with the remaining timeout.

        for i in 0..pfd.len() {
            match task.GetFile(pfd[i].fd) {
                Err(_) => (),
                Ok(f) => {
                    let r = f.Readiness(task, EventMaskFromLinux(pfd[i].events as u32));
                    let rl = ToLinux(r) as i16 & pfd[i].events;
                    if rl != 0 {
                        pfd[i].revents = rl;
                        n += 1;
                    }
                }
            };
        }

        if n > 0 {
            break;
        }

        let (timeoutTmp, res) = if timeout > 0 {
            task.blocker.BlockWithMonoTimeout(true, Some(timeout))
        } else {
            task.blocker.BlockWithMonoTimeout(true, None)
        };

        timeout = timeoutTmp;
        match res {
            Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                return (0, Ok(0))
            }
            Err(e) => {
                return (timeout, Err(e))
            }
            Ok(()) => (),
        };

    }

    return (timeout, Ok(n))
}

pub fn InitReadiness(task: &Task, pfd: &mut PollFd, files: &mut Vec<Option<File>>, entry: &WaitEntry) {
    if pfd.fd < 0 {
        pfd.revents = 0;
        files.push(None);
        return
    }

    let file = match task.GetFile(pfd.fd) {
        Err(_) => {
            pfd.revents = PollConst::POLLNVAL as i16;
            files.push(None);
            return
        }
        Ok(f) => f,
    };

    file.EventRegister(task, &entry, EventMaskFromLinux(pfd.events as u32));
    files.push(Some(file.clone()));

    let r = file.Readiness(task, EventMaskFromLinux(pfd.events as u32));
    pfd.revents = ToLinux(r) as i16 & pfd.events;
}

pub fn SysSelect(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let nfds = args.arg0 as i32;
    let readfds = args.arg1 as u64;
    let writefds = args.arg2 as u64;
    let exceptfds = args.arg3 as u64;
    let timeValAddr = args.arg4 as u64;

    // Use a negative Duration to indicate "no timeout".
    let mut timeout = -1 as Duration;
    if timeValAddr != 0 {
        let timeval : Timeval = task.CopyInObj(timeValAddr)?;
        if timeval.Sec < 0 || timeval.Usec < 0 {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        timeout = timeval.ToDuration();
        if timeout <= TIMEOUT_PROCESS_TIME {
            timeout = 0;
        }
    }

    let startNs = MonotonicNow();

    let res = DoSelect(task, nfds, readfds, writefds, exceptfds, timeout);
    CopyOutTimevalRemaining(task, startNs, timeout, timeValAddr)?;

    match res {
        Err(Error::SysError(SysErr::EINTR)) => return Err(Error::SysError(SysErr::ERESTARTNOHAND)),
        Err(e) => return Err(e),
        Ok(n) => return Ok(n),
    }
}

pub fn SysPSelect(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let nfds = args.arg0 as i32;
    let readfds = args.arg1 as u64;
    let writefds = args.arg2 as u64;
    let exceptfds = args.arg3 as u64;
    let timespecAddr = args.arg4 as u64;
    let maskWithSizeAddr = args.arg5 as u64;

    // Use a negative Duration to indicate "no timeout".
    let timeout = CopyTimespecIntoDuration(task, timespecAddr)?;

    let startNs = MonotonicNow();

    if maskWithSizeAddr != 0 {
        let (maskAddr, size) = CopyInSigSetWithSize(task, maskWithSizeAddr)?;
        if maskAddr != 0 {
            let mask = CopyInSigSet(task, maskAddr, size)?;

            let thread = task.Thread();
            let oldmask = thread.SignalMask();
            thread.SetSignalMask(mask);
            thread.SetSavedSignalMask(oldmask);
        }
    }

    let res = DoSelect(task, nfds, readfds, writefds, exceptfds, timeout);
    CopyOutTimespecRemaining(task, startNs, timeout, timespecAddr)?;

    match res {
        Err(Error::SysError(SysErr::EINTR)) => return Err(Error::SysError(SysErr::ERESTARTSYS)),
        Err(e) => return Err(e),
        Ok(n) => return Ok(n),
    }
}

// Poll implements linux syscall poll(2).
pub fn SysPoll(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let pfdAddr = args.arg0 as u64;
    let nfds = args.arg1 as u32; // poll(2) uses unsigned long.
    let timeout = args.arg2 as i32 as i64 * MILLISECOND;

    let n = Poll(task, pfdAddr, nfds, timeout)?;
    return Ok(n)
}

// Ppoll implements linux syscall ppoll(2).
pub fn SysPpoll(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let pfdAddr = args.arg0 as u64;
    let nfds = args.arg1 as u32; // poll(2) uses unsigned long.
    let timespecAddr = args.arg2 as u64;
    let maskAddr = args.arg3 as u64;
    let maskSize = args.arg4 as u32;

    let timeout = CopyTimespecIntoDuration(task, timespecAddr)?;
    let startNs = MonotonicNow();

    if maskAddr != 0 {
        let mask = CopyInSigSet(task, maskAddr, maskSize as usize)?;
        let thread = task.Thread();
        let oldmask = thread.SignalMask();
        thread.SetSignalMask(mask);
        thread.SetSavedSignalMask(oldmask);
    }

    let (_remain, res) = DoPoll(task, pfdAddr, nfds, timeout);
    CopyOutTimespecRemaining(task, startNs, timeout, timespecAddr)?;

    // doPoll returns EINTR if interrupted, but ppoll is normally restartable
    // if interrupted by something other than a signal handled by the
    // application (i.e. returns ERESTARTNOHAND). However, if
    // copyOutTimespecRemaining failed, then the restarted ppoll would use the
    // wrong timeout, so the error should be left as EINTR.
    //
    // Note that this means that if err is nil but copyErr is not, copyErr is
    // ignored. This is consistent with Linux.
    match res {
        Err(Error::SysError(SysErr::EINTR)) => return Err(Error::SysError(SysErr::ERESTARTNOHAND)),
        Err(e) => return Err(e),
        Ok(n) => return Ok(n as i64),
    }
}

pub struct PollRestartBlock {
    pub pfdAddr: u64,
    pub nfds: u32,
    pub timeout: Duration,
}

impl SyscallRestartBlock for PollRestartBlock {
    fn Restart(&self, task: &mut Task) -> Result<i64> {
        return Poll(task, self.pfdAddr, self.nfds, self.timeout)
    }
}

pub fn Poll(task: &mut Task, pfdAddr: u64, nfds: u32, timeout: Duration) -> Result<i64> {
    if nfds > 4096 {
        // linux support poll max 4096 fds
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let (remain, res) = DoPoll(task, pfdAddr, nfds, timeout);
    match res {
        Err(Error::SysError(SysErr::EINTR)) => {
            let b = Box::new(PollRestartBlock {
                pfdAddr: pfdAddr,
                nfds: nfds,
                timeout: remain,
            });
            task.SetSyscallRestartBlock(b);
            return Err(Error::SysError(SysErr::ERESTART_RESTARTBLOCK));
        }
        Err(e) => {
            return Err(e)
        }
        Ok(n) => return Ok(n as i64)
    }
}

pub fn DoPoll(task: &Task, addr: u64, nfds: u32, timeout: Duration) -> (Duration, Result<usize>) {
    //todo: handle fileCap

    if (nfds as i32) < 0 {
        return (0, Err(Error::SysError(SysErr::EINVAL)))
    }

    let mut pfd : Vec<PollFd> = if addr != 0 {
        match task.CopyInVec(addr, nfds as usize) {
            Err(e) => {
                return (timeout, Err(e))
            },
            Ok(pfd) => pfd,
        }
    } else {
        if nfds > 0 {
            return (timeout, Err(Error::SysError(SysErr::EFAULT)))
        }
        Vec::new()
    };

    //info!("DoPoll pfd is {:?}", pfd);

    // Compatibility warning: Linux adds POLLHUP and POLLERR just before
    // polling, in fs/select.c:do_pollfd(). Since pfd is copied out after
    // polling, changing event masks here is an application-visible difference.
    // (Linux also doesn't copy out event masks at all, only revents.)
    for i in 0..pfd.len() {
        pfd[i].events |= (LibcConst::EPOLLHUP | LibcConst::EPOLLERR) as i16;
    }

    // Do the syscall, then count the number of bits set.
    let (remainingTimeout, res) = PollBlock(task, &mut pfd, timeout);
    let n = match res {
        Err(_) => return (remainingTimeout, Err(Error::SysError(SysErr::EINTR))),
        Ok(n) => n,
    };

    // The poll entries are copied out regardless of whether
    // any are set or not. This aligns with the Linux behavior.
    if nfds > 0 {
        match task.CopyOutSlice(&pfd, addr, pfd.len()) {
            Err(e) => return (remainingTimeout, Err(e)),
            Ok(()) => (),
        }
    }

    return (remainingTimeout, Ok(n))
}

pub fn CopyOutTimevalRemaining(task: &Task, startNs: i64, timeout: Duration, timeValAddr: u64) -> Result<()> {
    if timeout < 0 {
        return Ok(())
    }

    let remaining = TimeoutRemain(task, startNs, timeout);
    let tvRemaining = Timeval::FromNs(remaining);
    let timeval : &mut Timeval = task.GetTypeMut(timeValAddr)?;
    *timeval = tvRemaining;

    return Ok(())
}

pub fn CopyOutTimespecRemaining(task: &Task, startNs: i64, timeout: Duration, timespecAddr: u64) -> Result<()> {
    if timeout < 0 {
        return Ok(())
    }

    let remaining = TimeoutRemain(task, startNs, timeout);
    let tsRemaining = Timespec::FromNs(remaining);
    let ts : &mut Timespec = task.GetTypeMut(timespecAddr)?;
    *ts = tsRemaining;

    return Ok(())
}

pub fn TimeoutRemain(_task: &Task, startNs: i64, timeout: Duration) -> Duration {
    let now = MonotonicNow();
    let remaining = timeout - (now - startNs);
    if remaining < 0 {
        return 0
    }

    return remaining;
}

pub fn CopyTimespecIntoDuration(task: &Task, timespecAddr: u64) -> Result<Duration> {
    let mut timeout = -1 as Duration;
    if timespecAddr != 0 {
        let timespec : Timespec = task.CopyInObj(timespecAddr)?;
        if !timespec.IsValid() {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        timeout = timespec.ToDuration()?;
        if timeout <= TIMEOUT_PROCESS_TIME {
            timeout = 0;
        }
    }

    return Ok(timeout);
}