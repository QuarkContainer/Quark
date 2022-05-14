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
use super::super::kernel::time::*;
use super::super::kernel::timer::*;
use super::super::kernel::waiter::*;
use super::super::kernel_def::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::mem::block::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;

pub fn SysRead(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let size = args.arg2 as i64;

    let n = Read(task, fd, addr, size)?;
    task.ioUsage.AccountReadSyscall(n);
    return Ok(n);
}

pub fn Read(task: &Task, fd: i32, addr: u64, size: i64) -> Result<i64> {
    task.PerfGoto(PerfType::Read);
    defer!(task.PerfGofrom(PerfType::Read));

    let file = task.GetFile(fd)?;

    if !file.Flags().Read {
        return Err(Error::SysError(SysErr::EBADF));
    }

    if size < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if size == 0 {
        return Ok(0);
    }

    let iov = IoVec::NewFromAddr(addr, size as usize);

    let mut iovs: [IoVec; 1] = [iov];

    let n = readv(task, &file, &mut iovs)?;
    /*if fd == 0 {
        use alloc::string::ToString;
        use super::super::qlib::util::*;

        let str = CString::ToStringWithLen(addr, n as usize).to_string();
        info!("(Data)Read({}): {}", n, str);
    }*/

    return Ok(n);
}

pub fn SysPread64(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let size = args.arg2 as i64;
    let offset = args.arg3 as i64;

    let n = Pread64(task, fd, addr, size, offset)?;
    task.ioUsage.AccountReadSyscall(n);
    return Ok(n);
}

pub fn Pread64(task: &Task, fd: i32, addr: u64, size: i64, offset: i64) -> Result<i64> {
    task.PerfGoto(PerfType::Read);
    defer!(task.PerfGofrom(PerfType::Read));

    let file = task.GetFile(fd)?;

    if offset < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if !file.Flags().Pread {
        return Err(Error::SysError(SysErr::ESPIPE));
    }

    if !file.Flags().Read {
        return Err(Error::SysError(SysErr::EBADF));
    }

    if size < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if size == 0 {
        return Ok(0);
    }

    let iov = IoVec::NewFromAddr(addr, size as usize);
    let mut iovs: [IoVec; 1] = [iov];
    return preadv(task, &file, &mut iovs, offset);
}

pub fn SysReadv(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let iovcnt = args.arg2 as i32;

    let n = Readv(task, fd, addr, iovcnt)?;
    task.ioUsage.AccountReadSyscall(n);
    return Ok(n);
}

pub fn Readv(task: &Task, fd: i32, addr: u64, iovcnt: i32) -> Result<i64> {
    let file = task.GetFile(fd)?;

    if !file.Flags().Read {
        return Err(Error::SysError(SysErr::EBADF));
    }

    if iovcnt < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let mut dsts = task.IovsFromAddr(addr, iovcnt as usize)?;

    return readv(task, &file, &mut dsts);
}

pub fn SysPreadv(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let iovcnt = args.arg2 as i32;
    let offset = args.arg3 as i64;

    let n = Preadv(task, fd, addr, iovcnt, offset)?;
    task.ioUsage.AccountReadSyscall(n);
    return Ok(n);
}

pub fn SysPreadv2(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    // While the syscall is
    // preadv2(int fd, struct iovec* iov, int iov_cnt, off_t offset, int flags)
    // the linux internal call
    // (https://elixir.bootlin.com/linux/v4.18/source/fs/read_write.c#L1248)
    // splits the offset argument into a high/low value for compatibility with
    // 32-bit architectures. The flags argument is the 5th argument.

    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let iovcnt = args.arg2 as i32;
    let offset = args.arg3 as i64;
    let flags = args.arg5 as i32;

    if offset < -1 {
        return Err(Error::SysError(SysErr::EINVAL));
    }
    // Check flags field.
    // Note: qkernel does not implement the RWF_HIPRI feature, but the flag is
    // accepted as a valid flag argument for preadv2.
    if flags & !(Flags::RWF_VALID) != 0 {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    if offset == -1 {
        let n = Readv(task, fd, addr, iovcnt)?;
        task.ioUsage.AccountWriteSyscall(n);
        return Ok(n);
    }

    let n = Preadv(task, fd, addr, iovcnt, offset)?;
    task.ioUsage.AccountReadSyscall(n);
    return Ok(n);
}

pub fn Preadv(task: &Task, fd: i32, addr: u64, iovcnt: i32, offset: i64) -> Result<i64> {
    let file = task.GetFile(fd)?;

    if offset < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if !file.Flags().Pread {
        return Err(Error::SysError(SysErr::ESPIPE));
    }

    if !file.Flags().Read {
        return Err(Error::SysError(SysErr::EBADF));
    }

    if iovcnt < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if iovcnt == 0 {
        return Ok(0);
    }

    let mut dsts = task.IovsFromAddr(addr, iovcnt as usize)?;
    return preadv(task, &file, &mut dsts, offset);
}

fn RepReadv(task: &Task, f: &File, dsts: &mut [IoVec]) -> Result<i64> {
    let len = Iovs(dsts).Count();
    let mut count = 0;
    let mut dsts = dsts;
    let mut tmp;

    loop {
        match f.Readv(task, dsts) {
            Err(e) => {
                if count > 0 {
                    return Ok(count);
                }

                return Err(e);
            }
            Ok(n) => {
                if n == 0 {
                    return Ok(count);
                }

                count += n;
                if count == len as i64 {
                    return Ok(count);
                }

                tmp = Iovs(dsts).DropFirst(n as usize);

                if tmp.len() == 0 {
                    return Ok(count);
                }
                dsts = &mut tmp;
            }
        }
    }
}

fn readv(task: &Task, f: &File, dsts: &mut [IoVec]) -> Result<i64> {
    //let mut iovs = task.GetIOVecPermission(dsts, true)?;
    //let dsts = &mut iovs;
    task.CheckIOVecPermission(dsts, true)?;

    let wouldBlock = f.WouldBlock();
    if !wouldBlock {
        return RepReadv(task, f, dsts);
    }

    match f.Readv(task, dsts) {
        Err(Error::ErrInterrupted) => return Err(Error::SysError(SysErr::ERESTARTSYS)),
        Err(e) => {
            if e != Error::SysError(SysErr::EWOULDBLOCK) || f.Flags().NonBlocking {
                return Err(e);
            }
        }
        Ok(n) => return Ok(n),
    };

    let mut deadline = None;

    let dl = f.FileOp.RecvTimeout();
    if dl < 0 {
        return Err(Error::SysError(SysErr::EWOULDBLOCK));
    }

    if dl > 0 {
        let now = MonotonicNow();
        deadline = Some(Time(now + dl));
    }

    let general = task.blocker.generalEntry.clone();

    f.EventRegister(task, &general, EVENT_READ);
    defer!(f.EventUnregister(task, &general));

    let len = Iovs(dsts).Count();
    let mut count = 0;
    let mut dsts = dsts;
    let mut tmp;
    loop {
        loop {
            match f.Readv(task, dsts) {
                Err(Error::SysError(SysErr::EWOULDBLOCK)) => {
                    if count > 0 {
                        return Ok(count);
                    }
                    break;
                }
                Err(e) => {
                    if count > 0 {
                        return Ok(count);
                    }
                    return Err(e);
                }
                Ok(n) => {
                    if n == 0 {
                        return Ok(count);
                    }

                    count += n;
                    if count == len as i64 {
                        return Ok(count);
                    }

                    tmp = Iovs(dsts).DropFirst(n as usize);
                    dsts = &mut tmp;
                }
            }
        }

        match task.blocker.BlockWithMonoTimer(true, deadline) {
            Err(Error::ErrInterrupted) => {
                return Err(Error::SysError(SysErr::ERESTARTSYS));
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

fn preadv(task: &Task, f: &File, dsts: &mut [IoVec], offset: i64) -> Result<i64> {
    //let mut iovs = task.GetIOVecPermission(dsts, true)?;
    //let dsts = &mut iovs;

    task.CheckIOVecPermission(dsts, true)?;
    match f.Preadv(task, dsts, offset) {
        Err(e) => {
            if e != Error::SysError(SysErr::EWOULDBLOCK) || f.Flags().NonBlocking {
                return Err(e);
            }
        }
        Ok(n) => return Ok(n),
    };

    let general = task.blocker.generalEntry.clone();

    f.EventRegister(task, &general, EVENT_READ);
    defer!(f.EventUnregister(task, &general));

    loop {
        match f.Preadv(task, dsts, offset) {
            Err(Error::SysError(SysErr::EWOULDBLOCK)) => (),
            Err(e) => {
                return Err(e);
            }
            Ok(n) => {
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
