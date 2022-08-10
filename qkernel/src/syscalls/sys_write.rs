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

use super::super::fs::file::*;
use super::super::fs::inotify::*;
use super::super::kernel::time::*;
use super::super::kernel::timer::*;
use super::super::kernel::waiter::*;
//use super::super::kernel_def::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::mem::block::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;

pub fn SysWrite(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let size = args.arg2 as i64;

    let n = Write(task, fd, addr, size)?;
    task.ioUsage.AccountWriteSyscall(n);
    return Ok(n);
}

pub fn Write(task: &Task, fd: i32, addr: u64, size: i64) -> Result<i64> {
    //task.PerfGoto(PerfType::Write);
    //defer!(task.PerfGofrom(PerfType::Write));

    let file = task.GetFile(fd)?;

    /*let fopsType = file.FileOp.FopsType();
    if fd <= 2 || fopsType == FileOpsType::TTYFileOps {
         use super::super::util::cstring::*;
        let (str, err) = CString::CopyInString(task, addr, size as usize);
         match err {
             Ok(_) => {
                 error!("(Data) Write: {}", str);
             }
             Err(_) => {
                 error!("(Data) Write fail: {}", str);
             }
         }
    }*/

    if !file.Flags().Write || file.Flags().Path {
        return Err(Error::SysError(SysErr::EBADF));
    }

    if size < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    /*
    unix socket allows sending 0-sized empty packet.
    if size == 0 {
        return Ok(0)
    }
    */

    let iov = IoVec::NewFromAddr(addr, size as usize);
    let iovs: [IoVec; 1] = [iov];

    return writev(task, &file, &iovs);
}

pub fn SysPwrite64(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let size = args.arg2 as i64;
    let offset = args.arg3 as i64;

    let n = Pwrite64(task, fd, addr, size, offset)?;
    task.ioUsage.AccountWriteSyscall(n);
    return Ok(n);
}

pub fn Pwrite64(task: &Task, fd: i32, addr: u64, size: i64, offset: i64) -> Result<i64> {
    //task.PerfGoto(PerfType::Write);
    //defer!(task.PerfGofrom(PerfType::Write));

    let file = task.GetFile(fd)?;

    if offset < 0 || core::i64::MAX - offset < size {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if !file.Flags().PWrite {
        return Err(Error::SysError(SysErr::ESPIPE));
    }

    if !file.Flags().Write || file.Flags().Path {
        return Err(Error::SysError(SysErr::EBADF));
    }

    if size < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if size == 0 {
        return Ok(0);
    }

    let iov = IoVec::NewFromAddr(addr, size as usize);
    let iovs: [IoVec; 1] = [iov];

    return pwritev(task, &file, &iovs, offset);
}

pub fn SysPWritev2(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    // While the syscall is
    // pwritev2(int fd, struct iovec* iov, int iov_cnt, off_t offset, int flags)
    // the linux internal call
    // (https://elixir.bootlin.com/linux/v4.18/source/fs/read_write.c#L1354)
    // splits the offset argument into a high/low value for compatibility with
    // 32-bit architectures. The flags argument is the 5th argument.

    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let iovcnt = args.arg2 as i32;
    let offset = args.arg3 as i64;
    let flags = args.arg5 as i32;

    if args.arg4 as i32 & 0x4 == 1 {
        return Err(Error::SysError(SysErr::EACCES));
    }

    if offset < -1 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    // Check that flags are supported. RWF_DSYNC/RWF_SYNC can be ignored since
    // all state is in-memory.

    // doens't support Flags::RWF_APPEND
    if flags & !(Flags::RWF_HIPRI | Flags::RWF_DSYNC | Flags::RWF_SYNC) != 0 {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    if offset == -1 {
        let n = Writev(task, fd, addr, iovcnt)?;
        task.ioUsage.AccountWriteSyscall(n);
        return Ok(n);
    }

    let n = Pwritev(task, fd, addr, iovcnt, offset)?;
    task.ioUsage.AccountWriteSyscall(n);
    return Ok(n);
}

pub fn SysWritev(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let iovcnt = args.arg2 as i32;

    if fd < 3 {
        use super::super::util::cstring::*;

        let srcs = task.IovsFromAddr(addr, iovcnt as usize)?;

        for i in 0..srcs.len() {
            let str = CString::ToStringWithLen(task, srcs[i].start, srcs[i].len as usize)?;
            info!("Write: {}", str);
        }
    }

    let n = Writev(task, fd, addr, iovcnt)?;
    task.ioUsage.AccountWriteSyscall(n);
    return Ok(n);
}

pub fn Writev(task: &Task, fd: i32, addr: u64, iovcnt: i32) -> Result<i64> {
    let file = task.GetFile(fd)?;

    if !file.Flags().Write || file.Flags().Path {
        return Err(Error::SysError(SysErr::EBADF));
    }

    if iovcnt < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if iovcnt == 0 {
        return Ok(0);
    }

    let srcs = task.IovsFromAddr(addr, iovcnt as usize)?;
    return writev(task, &file, &srcs);
}

pub fn SysPwritev(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let iovcnt = args.arg2 as i32;
    let offset = args.arg3 as i64;

    let n = Pwritev(task, fd, addr, iovcnt, offset)?;
    task.ioUsage.AccountWriteSyscall(n);
    return Ok(n);
}

pub fn Pwritev(task: &Task, fd: i32, addr: u64, iovcnt: i32, offset: i64) -> Result<i64> {
    let file = task.GetFile(fd)?;

    if offset < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if !file.Flags().PWrite {
        return Err(Error::SysError(SysErr::ESPIPE));
    }

    if !file.Flags().Write || file.Flags().Path {
        return Err(Error::SysError(SysErr::EBADF));
    }

    if iovcnt < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if iovcnt == 0 {
        return Ok(0);
    }

    let srcs = task.IovsFromAddr(addr, iovcnt as usize)?;
    return pwritev(task, &file, &srcs, offset);
}

fn RepWritev(task: &Task, f: &File, srcs: &[IoVec]) -> Result<i64> {
    let len = Iovs(srcs).Count();
    let mut count = 0;
    let mut srcs = srcs;
    let mut tmp;

    loop {
        match f.Writev(task, srcs) {
            Err(e) => {
                if count > 0 {
                    break;
                }

                return Err(e);
            }
            Ok(n) => {
                count += n;
                if count == len as i64 {
                    break;
                }

                tmp = Iovs(srcs).DropFirst(n as usize);
                srcs = &tmp;
            }
        }
    }

    if count > 0 {
        f.Dirent.InotifyEvent(InotifyEvent::IN_MODIFY, 0, EventType::PathEvent)
    }
    return Ok(count);
}

pub fn writev(task: &Task, f: &File, srcs: &[IoVec]) -> Result<i64> {
    let iovs = task.AdjustIOVecPermission(srcs, false, true)?;
    let srcs = &iovs;

    let len = Iovs(srcs).Count();

    let wouldBlock = f.WouldBlock();
    if !wouldBlock {
        return RepWritev(task, f, srcs);
    }

    if f.Flags().NonBlocking {
        match f.Writev(task, srcs) {
            Err(Error::ErrInterrupted) => return Err(Error::SysError(SysErr::ERESTARTSYS)),
            Err(e) => {
                return Err(e);
            }
            Ok(n) => {
                if n > 0 {
                    f.Dirent.InotifyEvent(InotifyEvent::IN_MODIFY, 0, EventType::PathEvent)
                }
                return Ok(n);
            }
        };
    }

    let mut deadline = None;

    let dl = f.FileOp.SendTimeout();
    if dl < 0 {
        return Err(Error::SysError(SysErr::EWOULDBLOCK));
    }

    if dl > 0 {
        let now = MonotonicNow();
        deadline = Some(Time(now + dl));
    }

    let general = task.blocker.generalEntry.clone();

    f.EventRegister(task, &general, EVENT_WRITE);
    defer!(f.EventUnregister(task, &general));

    let mut count = 0;
    let mut srcs = srcs;
    let mut tmp;
    loop {
        match f.Writev(task, srcs) {
            Err(Error::SysError(SysErr::EWOULDBLOCK)) => {
                if f.Flags().NonBlocking {
                    if count > 0 {
                        break;
                    }
                    return Err(Error::SysError(SysErr::EWOULDBLOCK))
                }
            },
            Err(e) => {
                if count > 0 {
                    break;
                }

                return Err(e);
            }
            Ok(n) => {
                count += n;
                if count == len as i64 || f.Flags().NonBlocking {
                    break;
                }

                tmp = Iovs(srcs).DropFirst(n as usize);
                srcs = &tmp;
            }
        }

        match task.blocker.BlockWithMonoTimer(true, deadline) {
            Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                if count > 0 {
                    break;
                }
                return Err(Error::SysError(SysErr::EWOULDBLOCK));
            }
            Err(Error::ErrInterrupted) => {
                if count > 0 {
                    break;
                }
                return Err(Error::SysError(SysErr::ERESTARTSYS))
            },
            Err(e) => {
                if count > 0 {
                    break;
                }
                return Err(e);
            }
            _ => (),
        }
    }

    if count > 0 {
        f.Dirent.InotifyEvent(InotifyEvent::IN_MODIFY, 0, EventType::PathEvent)
    }
    return Ok(count);
}

fn RepPwritev(task: &Task, f: &File, srcs: &[IoVec], offset: i64) -> Result<i64> {
    let len = Iovs(srcs).Count();
    let mut count = 0;
    let mut srcs = srcs;
    let mut tmp;

    loop {
        match f.Pwritev(task, srcs, offset + count) {
            Err(e) => {
                if count > 0 {
                    break;
                }

                return Err(e);
            }
            Ok(n) => {
                count += n;
                if count == len as i64 {
                    break;
                }

                tmp = Iovs(srcs).DropFirst(n as usize);
                srcs = &tmp;
            }
        }
    }

    if count > 0 {
        f.Dirent.InotifyEvent(InotifyEvent::IN_MODIFY, 0, EventType::PathEvent)
    }

    return Ok(count);
}

fn pwritev(task: &Task, f: &File, srcs: &[IoVec], offset: i64) -> Result<i64> {
    let mut iovs = task.AdjustIOVecPermission(srcs, false, true)?;
    let srcs = &mut iovs;

    let wouldBlock = f.WouldBlock();
    if !wouldBlock {
        return RepPwritev(task, f, srcs, offset);
    }

    match f.Pwritev(task, srcs, offset) {
        Err(e) => {
            if e != Error::SysError(SysErr::EWOULDBLOCK) || f.Flags().NonBlocking {
                return Err(e);
            }
        }
        Ok(n) => {
            if n > 0 {
                f.Dirent.InotifyEvent(InotifyEvent::IN_MODIFY, 0, EventType::PathEvent)
            }
            return Ok(n)
        },
    };

    let general = task.blocker.generalEntry.clone();

    f.EventRegister(task, &general, EVENT_WRITE);
    defer!(f.EventUnregister(task, &general));

    loop {
        match f.Pwritev(task, srcs, offset) {
            Err(Error::SysError(SysErr::EWOULDBLOCK)) => (),
            Err(e) => {
                return Err(e);
            }
            Ok(n) => {
                if n > 0 {
                    f.Dirent.InotifyEvent(InotifyEvent::IN_MODIFY, 0, EventType::PathEvent)
                }
                return Ok(n);
            }
        }

        match task.blocker.BlockWithMonoTimer(true, None) {
            Err(e) => {
                return Err(e);
            }
            _ => (),
        }
    }
}
