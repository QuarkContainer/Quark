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

use super::super::kernel::waiter::*;
use crate::qlib::kernel::socket::hostinet::uring_socket::*;
//use crate::qlib::kernel::socket::hostinet::socket::*;

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::task::*;
use super::super::syscalls::syscalls::*;


pub const BLOCK_WRITE : i32 = 1;
pub const BLOCK_READ : i32 = 2;

pub const BUF_SIZE : usize = 1 << 16; // 64K

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RingBufIovs {
    pub iovs: [IoVec; 2],
    pub iovcnt: i32,
}

impl RingBufIovs {
    pub fn New(iovs: &[IoVec]) -> Result<Self> {
        if iovs.len() > 2 {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        let mut ret = Self::default();

        for i in 0..iovs.len() {
            ret.iovs[i] = iovs[i];
        }

        ret.iovcnt = iovs.len() as _;

        return Ok(ret);
    }
}

// arg0: fd
// arg1: produce byte count
// arg2: address of write RingbufIovs, if 0, then there is no RingbufIovs
// arg3: flags, the function flags,only support SOCK_NONBLOCK

pub fn SysSocketProduce(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let count: i32 = args.arg1 as i32;
    let iovsAddr = args.arg2 as u64;
    let flags = args.arg3 as i32;

    if flags & !SocketFlags::SOCK_NONBLOCK != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let block = flags & SocketFlags::SOCK_NONBLOCK == 0;

    let file = task.GetFile(fd)?;
    let sockops = file.FileOp.clone();

    match sockops.as_any().downcast_ref::<UringSocketOperations>() {
        Some(uringSocket) => {
            let iovs = uringSocket.Produce(task, count as usize)?;
            if iovs.len() != 0 || !block || iovsAddr == 0 {
                if iovsAddr !=  0 {
                    task.CopyOutObj(&RingBufIovs::New(&iovs)?, iovsAddr)?;
                }
                
                return Ok(0)
            }
            
            let general = task.blocker.generalEntry.clone();
            uringSocket.EventRegister(task, &general, EVENT_WRITE);
            defer!(uringSocket.EventUnregister(task, &general));

            loop {
                let iovs = uringSocket.Produce(task, 0)?;
                if iovs.len() != 0 {
                    if iovsAddr !=  0 {
                        task.CopyOutObj(&RingBufIovs::New(&iovs)?, iovsAddr)?;
                    }
                    return Ok(0)
                }
            }
        }
        None => (),
    }

    return Ok(0)
}

// arg0: fd
// arg1: consume byte count
// arg2: address of write RingbufIovs, if 0, then there is no RingbufIovs
// arg3: flags, the function flags,only support SOCK_NONBLOCK
pub fn SysSocketConsume(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let count: i32 = args.arg1 as i32;
    let iovsAddr = args.arg2 as u64;
    let flags = args.arg3 as i32;

    if flags & !SocketFlags::SOCK_NONBLOCK != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let block = flags & SocketFlags::SOCK_NONBLOCK == 0;

    let file = task.GetFile(fd)?;
    let sockops = file.FileOp.clone();

    match sockops.as_any().downcast_ref::<UringSocketOperations>() {
        Some(uringSocket) => {
            let iovs = uringSocket.Consume(task, count as usize)?;
            if iovs.len() != 0 || !block || iovsAddr == 0 {
                if iovsAddr != 0 {
                    task.CopyOutObj(&RingBufIovs::New(&iovs)?, iovsAddr)?;
                }               
                return Ok(0)
            }
            
            let general = task.blocker.generalEntry.clone();
            uringSocket.EventRegister(task, &general, EVENT_WRITE);
            defer!(uringSocket.EventUnregister(task, &general));

            loop {
                let iovs = uringSocket.Produce(task, 0)?;
                if iovs.len() != 0 {
                    if iovsAddr != 0 {
                        task.CopyOutObj(&RingBufIovs::New(&iovs)?, iovsAddr)?;
                    } 
                    return Ok(0)
                }
            }
        }
        None => (),
    }

    return Ok(0)
}
