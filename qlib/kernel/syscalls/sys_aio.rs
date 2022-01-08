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


use super::super::task::*;
use super::super::kernel::aio::aio_context::*;
use super::super::kernel::eventfd::*;
use super::super::kernel::time::*;
use super::super::kernel::waiter::*;
use super::super::fs::file::*;
use super::super::fs::host::hostinodeop::*;
use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::super::quring::async::*;
use super::super::IOURING;
use super::super::SHARESPACE;
use super::sys_poll::*;

// IoSetup implements linux syscall io_setup(2).
pub fn SysIoSetup(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let enableAIO = SHARESPACE.config.read().EnableAIO;

    if !enableAIO {
        return Err(Error::SysError(SysErr::ENOSYS))
    }

    let nrEvents = args.arg0 as i32;
    let idAddr = args.arg1 as u64;

    // Linux uses the native long as the aio ID.
    //
    // The context pointer _must_ be zero initially.
    //let idPtr = task.GetTypeMut(idAddr)?;
    let idIn : u64 = task.CopyInObj(idAddr)?;
    if idIn != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let id = task.mm.NewAIOContext(task, nrEvents as usize)?;
    task.CopyOutObj(&id, idAddr)?;
    return Ok(0)
}

// IoDestroy implements linux syscall io_destroy(2).
pub fn SysIoDestroy(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let id = args.arg0 as u64;

    if !task.mm.DestroyAIOContext(task, id) {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    // Fixme: Linux blocks until all AIO to the destroyed context is done.
    return Ok(0)
}

// IoGetevents implements linux syscall io_getevents(2).
pub fn SysIoGetevents(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let id = args.arg0 as u64;
    let minEvents = args.arg1 as i32;
    let events = args.arg2 as i32 as usize;
    let mut eventsAddr = args.arg3 as u64;
    let timespecAddr = args.arg4 as u64;

    // Sanity check arguments.
    if minEvents < 0 || minEvents > events as i32 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let ctx = match task.mm.LookupAIOContext(task, id) {
        None => return Err(Error::SysError(SysErr::EINVAL)),
        Some(c) => c
    };

    let timeout = CopyTimespecIntoDuration(task, timespecAddr)?;

    let deadline = if timeout == -1 {
        None
    } else {
        Some(Task::MonoTimeNow().Add(timeout))
    };

    for count in 0..events {
        let event;
        if count >= minEvents as usize {
            match ctx.PopRequest() {
                None => return Ok(count as i64),
                Some(v) => event = v,
            }
         } else {
            match WaitForRequest(&ctx, task, deadline) {
                Err(e) => {
                    if count > 0 || e == Error::SysError(SysErr::ETIMEDOUT){
                        return Ok(count as i64)
                    }

                    return Err(e)
                }
                Ok(v) => event = v,
            }
        }

        match task.CopyOutObj(&event, eventsAddr) {
            Err(e) => {
                if count > 0 {
                    return Ok(count as i64)
                }

                return Err(e)
            }
            Ok(()) => (),
        };

        //*eventPtr = event;
        eventsAddr += IOEVENT_SIZE;
    }

    return Ok(events as i64)
}

pub fn WaitForRequest(ctx: &AIOContext, task: &Task, dealine: Option<Time>) -> Result<IOEvent> {
    match ctx.PopRequest() {
        None => (),
        Some(v) => return Ok(v)
    }

    let general = task.blocker.generalEntry.clone();
    ctx.EventRegister(task, &general, EVENT_IN | EVENT_HUP);
    defer!(ctx.EventUnregister(task, &general));

    loop {
        match ctx.PopRequest() {
            None => {
                if ctx.lock().dead {
                    return Err(Error::SysError(SysErr::EINVAL))
                }
            },
            Some(v) => return Ok(v)
        }

        let err = task.blocker.BlockWithMonoTimer(true, dealine);
        match err {
            Ok(()) => (),
            Err(e) => {
                return Err(e)
            }
        }
    }
}

pub fn SysIOSubmit(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let id = args.arg0 as u64;
    let nrEvents = args.arg1 as i32;
    let mut addr = args.arg2 as u64;

    // Sanity check arguments.
    if nrEvents < 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    for i in 0..nrEvents as usize {
        let cbAddr : u64 = match task.CopyInObj(addr) {
            Err(e) => {
                if i > 0 {
                    // Some successful.
                    return Ok(i as i64)
                }

                return Err(e)
            }
            Ok(ptr) => ptr,
        };

        // Copy in this callback.
        let cb : IOCallback = match task.CopyInObj(cbAddr) {
            Err(e) => {
                if i > 0 {
                    // Some successful.
                    return Ok(i as i64)
                }
                return Err(e)
            }
            Ok(c) => c,
        };

        match SubmitCallback(task, id, &cb, cbAddr) {
            Err(e) => {
                if i > 0 {
                    // Partial success.
                    return Ok(i as i64)
                }

                return Err(e)
            }
            Ok(()) => ()
        }

        addr += 8;
    }

    return Ok(nrEvents as i64)
}

pub fn SubmitCallback(task: &Task, id: u64, cb: &IOCallback, cbAddr: u64) -> Result<()> {
    let file = task.GetFile(cb.fd as i32)?;

    let eventfops = if cb.flags & IOCB_FLAG_RESFD as u32 != 0 {
        let eventFile = task.GetFile(cb.resfd as i32)?;

        let eventfops = match eventFile.FileOp.as_any().downcast_ref::<EventOperations>() {
            None => {
                return Err(Error::SysError(SysErr::EINVAL))
            }
            Some(e) => {
                e.clone()
            }
        };

        Some(eventfops)
    } else {
        None
    };

    match cb.opcode {
        IOCB_CMD_PREAD |
        IOCB_CMD_PWRITE |
        IOCB_CMD_PREADV |
        IOCB_CMD_PWRITEV => {
            if cb.offset < 0 {
                return Err(Error::SysError(SysErr::EINVAL))
            }
        }
        _ => ()
    }

    let ctx = match task.mm.LookupAIOContext(task, id) {
        Some(ctx) => ctx,
        None => {
            return Err(Error::SysError(SysErr::EINVAL))
        }
    };

    if !ctx.Prepare() {
        // Context is busy.
        return Err(Error::SysError(SysErr::EAGAIN))
    }

    return PerformanceCallback(task, &file, cbAddr, cb, &ctx, eventfops)
}

pub fn PerformanceCallback(task: &Task, file: &File, cbAddr: u64, cb: &IOCallback, ctx: &AIOContext, eventfops: Option<EventOperations>) -> Result<()> {
    let inode = file.Dirent.Inode();
    let iops = inode.lock().InodeOp.clone();
    let iops = match iops.as_any().downcast_ref::<HostInodeOp>() {
        None => {
            error!("can't do aio on file type {:?}", file.FileType());
            return Err(Error::SysError(SysErr::EINVAL))
        }
        Some(e) => {
            e.clone()
        }
    };

    let fd = iops.HostFd();
    let mut cb = *cb;
    cb.fd = fd as u32;

    match cb.opcode {
        IOCB_CMD_PREAD => {
            let ops = AIORead::NewRead(task, ctx.clone(), &cb, cbAddr, eventfops)?;
            IOURING.AUCall(AsyncOps::AIORead(ops));
        }
        IOCB_CMD_PREADV => {
            let ops = AIORead::NewReadv(task, ctx.clone(), &cb, cbAddr, eventfops)?;
            IOURING.AUCall(AsyncOps::AIORead(ops));
        }
        IOCB_CMD_PWRITE => {
            let ops = AIOWrite::NewWrite(task, ctx.clone(), &cb, cbAddr, eventfops)?;
            IOURING.AUCall(AsyncOps::AIOWrite(ops));
        }
        IOCB_CMD_PWRITEV => {
            let ops = AIOWrite::NewWritev(task, ctx.clone(), &cb, cbAddr, eventfops)?;
            IOURING.AUCall(AsyncOps::AIOWrite(ops));
        }
        IOCB_CMD_FSYNC => {
            let ops = AIOFsync::New(task, ctx.clone(), &cb, cbAddr, eventfops, false)?;
            IOURING.AUCall(AsyncOps::AIOFsync(ops));
        }
        IOCB_CMD_FDSYNC => {
            let ops = AIOFsync::New(task, ctx.clone(), &cb, cbAddr, eventfops, true)?;
            IOURING.AUCall(AsyncOps::AIOFsync(ops));
        }
        _ => {
            panic!("PerformanceCallback get unsupported aio {}", cb.opcode)
            //return Err(Error::SysError(SysErr::EINVAL))
        }
    }

    return Ok(())
}

// IoCancel implements linux syscall io_cancel(2).
//
// It is not presently supported (ENOSYS indicates no support on this
// architecture).
pub fn SysIOCancel(_task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOSYS))
}