// Copyright (c) 2021 QuarkSoft LLC
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
use super::super::kernel::time::*;
use super::super::kernel::waiter::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::sys_poll::*;

// IoSetup implements linux syscall io_setup(2).
pub fn SysIoSetup(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let nrEvents = args.arg0 as i32;
    let idAddr = args.arg1 as u64;

    // Linux uses the native long as the aio ID.
    //
    // The context pointer _must_ be zero initially.
    let idPtr = task.GetTypeMut(idAddr)?;
    let idIn : u64 = *idPtr;
    if idIn != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let id = task.mm.NewAIOContext(task, nrEvents as usize)?;
    *idPtr = id;
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
        Some(task.Now().Add(timeout))
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

        let eventPtr = match task.GetTypeMut(eventsAddr) {
            Err(e) => {
                if count > 0 {
                    return Ok(count as i64)
                }

                return Err(e)
            }
            Ok(e) => e,
        };

        *eventPtr = event;
        eventsAddr += IOEVENT_SIZE;
    }

    return Ok(events as i64)
}

pub fn WaitForRequest(ctx: &AIOContext, task: &Task, dealine: Option<Time>) -> Result<IOEvent> {
    loop {
        match ctx.PopRequest() {
            None => (),
            Some(v) => return Ok(v)
        }

        let general = task.blocker.generalEntry.clone();
        ctx.EventRegister(task, &general, EVENT_IN | EVENT_HUP);
        defer!(ctx.EventUnregister(task, &general));


        task.blocker.BlockWithMonoTimer(true, dealine)?
    }
}