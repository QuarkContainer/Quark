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

use super::super::super::super::linux_def::*;
use super::super::super::super::common::*;
use super::super::super::super::socket_buf::*;
use super::super::super::task::Task;

impl SocketBuff {
    pub fn Readv(&self, task: &Task, iovs: &mut [IoVec]) -> Result<(bool, usize)> {
        let mut trigger = false;
        let mut cnt = 0;

        let mut buf = self.readBuf.lock();
        let srcIovs = buf.GetDataIovsVec();
        if srcIovs.len() > 0 {
            cnt = task.mm.CopyIovsOutFromIovs(task, &srcIovs, iovs)?;
            trigger = buf.Consume(cnt);
        }

        if cnt > 0 {
            return Ok((trigger, cnt))
        } else if self.Error() != 0 {
            return Err(Error::SysError(self.Error()));
        } else if self.RClosed() {
            return Ok((false, 0))
        } else {
            return Err(Error::SysError(SysErr::EAGAIN))
        }
    }

    pub fn Writev(&self, task: &Task, iovs: &[IoVec]) -> Result<(usize, Option<(u64, usize)>)> {
        //debug!("Writev::1");
        if self.Error() != 0 {
            //debug!("Writev::1.1, error: {}", self.Error());
            return Err(Error::SysError(self.Error()));
        }

        if self.WClosed() {
            error!("writev it is closed");
            //return Ok((0, None))
            return Err(Error::SysError(SysErr::EPIPE))
        }

        //debug!("Writev::2");

        let mut buf = self.writeBuf.lock();
        let dstIovs = buf.GetSpaceIovsVec();
        //debug!("Writev::3, buffer.len: {}", dstIovs.len());
        if dstIovs.len() == 0 {
            return Err(Error::SysError(SysErr::EAGAIN));
        }

        let cnt = task.mm.CopyIovsOutToIovs(task, iovs, &dstIovs)?;

        //debug!("Writev::4, cnt: {}", dstIovs.len());
        if cnt == 0 {
            error!("writev cnt is zero....");
            return Err(Error::SysError(SysErr::EAGAIN));
        }

        let trigger = buf.Produce(cnt);
        if !trigger {
            //debug!("Writev::5, cnt: {}", dstIovs.len());
            return Ok((cnt, None))
        } else {
            let (addr, len) = buf.GetDataBuf();
            //debug!("Writev::6, cnt: {}, addr: {}, len: {}", dstIovs.len(), cnt, len);
            return Ok((cnt, Some((addr, len))))
        }
    }
}