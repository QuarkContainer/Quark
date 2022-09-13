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

use alloc::vec::Vec;

use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::socket_buf::*;
use super::super::super::task::Task;

impl SocketBuffIntern {
    pub fn Consume(&self, _task: &Task, cnt: usize) -> Result<(Vec<IoVec>, bool)> {
        let mut buf = self.readBuf.lock();
        let trigger = if cnt > 0 {
            buf.ConsumeWithCheck(cnt)?
        } else {
            false
        };

        let iovs = buf.GetDataIovsVec();
        if iovs.len() == 0 {            
            if self.Error() != 0 {
                return Err(Error::SysError(self.Error()));
            } else if self.RClosed() {
                return Ok((iovs, false));
            }
        }

        return Ok((iovs, trigger));
    }

    pub fn Readv(&self, task: &Task, iovs: &mut [IoVec], peek: bool) -> Result<(bool, usize)> {
        let mut trigger = false;
        let mut cnt = 0;

        let mut buf = self.readBuf.lock();
        let srcIovs = buf.GetDataIovsVec();
        if srcIovs.len() > 0 {
            cnt = task.mm.CopyIovsOutFromIovs(task, &srcIovs, iovs, true)?;
            trigger = if !peek {
                buf.Consume(cnt)
            } else {
                false
            }
        }

        // if srcIovs.len() > 0 {
        //     let mut srcLen = 0;
        //     for iovs1 in srcIovs {
        //         srcLen += iovs1.len;
        //     }

        //     let mut dstLen = 0;
        //     for iovs2 in iovs {
        //         dstLen += iovs2.len;
        //     }
        //     cnt = if srcLen <= dstLen { srcLen } else { dstLen };
        //     trigger = buf.Consume(cnt);
        // }

        if cnt > 0 {
            return Ok((trigger, cnt));
        } else if self.Error() != 0 {
            return Err(Error::SysError(self.Error()));
        } else if self.RClosed() {
            return Ok((false, 0));
        } else {
            return Err(Error::SysError(SysErr::EAGAIN));
        }
    }

    pub fn Produce(&self, _task: &Task, cnt: usize) -> Result<(Vec<IoVec>, Option<(u64, usize)>)> {
        if self.Error() != 0 {
            return Err(Error::SysError(self.Error()));
        }

        if self.WClosed() {
            return Err(Error::SysError(SysErr::EPIPE));
        }

        let mut buf = self.writeBuf.lock();
        let trigger = if cnt > 0 {
            buf.ProduceWithCheck(cnt)?
        } else {
            false
        };
        let iovs = buf.GetSpaceIovsVec();
        if !trigger {
            return Ok((iovs, None));
        } else {
            let (addr, len) = buf.GetDataBuf();
            return Ok((iovs, Some((addr, len))));
        }
    }

    pub fn Writev(&self, task: &Task, iovs: &[IoVec]) -> Result<(usize, Option<(u64, usize)>)> {
        if self.Error() != 0 {
            return Err(Error::SysError(self.Error()));
        }

        if self.WClosed() {
            error!("writev it is closed");
            //return Ok((0, None))
            return Err(Error::SysError(SysErr::EPIPE));
        }

        let mut buf = self.writeBuf.lock();
        let dstIovs = buf.GetSpaceIovsVec();
        if dstIovs.len() == 0 {
            return Err(Error::SysError(SysErr::EAGAIN));
        }

        let cnt = task.mm.CopyIovsInFromIovs(task, iovs, &dstIovs, true)?;

        // let mut srcLen = 0;
        // for iovs1 in dstIovs {
        //     srcLen += iovs1.len;
        // }

        // let mut dstLen = 0;
        // for iovs2 in iovs {
        //     dstLen += iovs2.len;
        // }
        // let cnt = if srcLen <= dstLen { srcLen } else { dstLen };

        if cnt == 0 {
            error!("writev cnt is zero....");
            return Err(Error::SysError(SysErr::EAGAIN));
        }

        let trigger = buf.Produce(cnt);
        if !trigger {
            return Ok((cnt, None));
        } else {
            let (addr, len) = buf.GetDataBuf();
            return Ok((cnt, Some((addr, len))));
        }
    }
    pub fn RDMAReadv(&self, _task: &Task, iovs: &mut [IoVec]) -> Result<(bool, usize)> {
        let mut trigger = false;
        let mut cnt = 0;

        let mut buf = self.readBuf.lock();
        let srcIovs = buf.GetDataIovsVec();

        if srcIovs.len() > 0 {
            let mut srcLen = 0;
            for iovs1 in srcIovs {
                srcLen += iovs1.len;
            }

            let mut dstLen = 0;
            for iovs2 in iovs {
                dstLen += iovs2.len;
            }
            cnt = if srcLen <= dstLen { srcLen } else { dstLen };
            trigger = buf.Consume(cnt);
        }

        if cnt > 0 {
            return Ok((trigger, cnt));
        } else if self.Error() != 0 {
            return Err(Error::SysError(self.Error()));
        } else if self.RClosed() {
            return Ok((false, 0));
        } else {
            return Err(Error::SysError(SysErr::EAGAIN));
        }
    }

    pub fn RDMAWritev(&self, _task: &Task, iovs: &[IoVec]) -> Result<(usize, Option<(u64, usize)>)> {
        if self.Error() != 0 {
            return Err(Error::SysError(self.Error()));
        }

        if self.WClosed() {
            error!("writev it is closed");
            //return Ok((0, None))
            return Err(Error::SysError(SysErr::EPIPE));
        }

        let mut buf = self.writeBuf.lock();
        let dstIovs = buf.GetSpaceIovsVec();
        if dstIovs.len() == 0 {
            return Err(Error::SysError(SysErr::EAGAIN));
        }

        let mut srcLen = 0;
        for iovs1 in dstIovs {
            srcLen += iovs1.len;
        }

        let mut dstLen = 0;
        for iovs2 in iovs {
            dstLen += iovs2.len;
        }
        let cnt = if srcLen <= dstLen { srcLen } else { dstLen };

        if cnt == 0 {
            error!("writev cnt is zero....");
            return Err(Error::SysError(SysErr::EAGAIN));
        }

        let trigger = buf.Produce(cnt);
        if !trigger {
            return Ok((cnt, None));
        } else {
            let (addr, len) = buf.GetDataBuf();
            return Ok((cnt, Some((addr, len))));
        }
    }
}
