// Copyright (c) 2021 Quark Container Authors 
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

use alloc::sync::Arc;
use core::ops::Deref;

use super::resilience_conn::*;
use crate::qlib::common::*;
use crate::qlib::linux_def::*;
use crate::qlib::kernel::fs::file::*;
use crate::qlib::kernel::kernel::waiter::*;
use crate::qlib::kernel::task::*;

pub struct ResilienceSocketOpsInner {
    pub sessionId: u64,
    pub session: ResilienceSession,
}

#[derive(Clone)]
pub struct ResilienceSocketOps(pub Arc<ResilienceSocketOpsInner>);

impl Deref for ResilienceSocketOps {
    type Target = Arc<ResilienceSocketOpsInner>;

    fn deref(&self) -> &Arc<ResilienceSocketOpsInner> {
        &self.0
    }
}

impl Waitable for ResilienceSocketOps {
    fn Readiness(&self, _task: &Task, mask: EventMask) -> EventMask {
        let mut event = EVENT_OUT;
        if self.session.lock().messages.len() > 0 {
            event |= EVENT_IN;
        }

        return event & mask; 
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        let queue = self.session.lock().queue.clone();
        queue.EventRegister(task, e, mask);
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        let queue = self.session.lock().queue.clone();
        queue.EventUnregister(task, e);
    }
}

impl SpliceOperations for ResilienceSocketOps {}

//impl FileOperations for ResilienceSocketOps {}

impl SockOperations for ResilienceSocketOps {
    fn Connect(&self, _task: &Task, _socketaddr: &[u8], _blocking: bool) -> Result<i64> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn Accept(
        &self,
        _task: &Task,
        _addr: &mut [u8],
        _addrlen: &mut u32,
        _flags: i32,
        _blocking: bool,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn Bind(&self, _task: &Task, _sockaddr: &[u8]) -> Result<i64> {
        let sessionId = self.session.lock().sessionId;
        RESILIENCE_CONN.SetServiceSessionId(sessionId);
        return Ok(0);
    }

    fn Listen(&self, _task: &Task, _backlog: i32) -> Result<i64> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn Shutdown(&self, _task: &Task, _how: i32) -> Result<i64> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn GetSockOpt(&self, _task: &Task, _level: i32, _name: i32, _addr: &mut [u8]) -> Result<i64> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn SetSockOpt(&self, _task: &Task, _level: i32, _name: i32, _opt: &[u8]) -> Result<i64> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn GetSockName(&self, _task: &Task, _socketaddr: &mut [u8]) -> Result<i64> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn GetPeerName(&self, _task: &Task, _socketaddr: &mut [u8]) -> Result<i64> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn RecvMsg(
        &self,
        _task: &Task,
        _dst: &mut [IoVec],
        _flags: i32,
        _deadline: Option<Time>,
        _senderRequested: bool,
        _controlDataLen: usize,
    ) -> Result<(i64, i32, Option<(SockAddr, usize)>, Vec<u8>)> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn SendMsg(
        &self,
        task: &Task,
        src: &[IoVec],
        _flags: i32,
        _msgHdr: &mut MsgHdr,
        _deadline: Option<Time>,
    ) -> Result<i64> {
        if msgHdr.msgName == 0 || msgHdr.nameLen == 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let ptr = msgHdr.msgName as *const u8;
        let slice = unsafe { slice::from_raw_parts(ptr, msgHdr.nameLen as usize) }; 

        let funcName = match String::from_utf8(slice) {
            Ok(v) => v,
            Err(e) => return Err(Error::SysError(SysErr::EINVAL)),
        };

        let size = IoVec::NumBytes(srcs);
        if size == 0 {
            return Ok(0)
        }

        if size <= 8 {
            // at least needs 8 byes user data
            eturn Err(Error::SysError(SysErr::EINVAL)); 
        }

        let size = IoVec::NumBytes(srcs);
        let mut buf = DataBuff::New(size);
        let len = task.CopyDataInFromIovs(&mut buf.buf, srcs, true)?;

        RESILIENCE_CONN.
        return Err(Error::SysError(SysErr::ENOTSOCK));
    }
}