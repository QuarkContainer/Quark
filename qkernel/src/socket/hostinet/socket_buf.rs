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

use ::qlib::mutex::*;
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicI32;
use core::sync::atomic::Ordering;

use super::super::super::qlib::bytestream::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::common::*;
use super::super::super::task::Task;

pub struct SocketBuff {
    pub wClosed: AtomicBool,
    pub rClosed: AtomicBool,
    pub pendingWShutdown: AtomicBool,
    pub error: AtomicI32,

    pub readBuf: QMutex<ByteStream>,
    pub writeBuf: QMutex<ByteStream>,
}

impl SocketBuff {
    pub fn Init(pageCount: u64) -> Self {
        return Self {
            wClosed: AtomicBool::new(false),
            rClosed: AtomicBool::new(false),
            pendingWShutdown: AtomicBool::new(false),
            error: AtomicI32::new(0),
            readBuf: QMutex::new(ByteStream::Init(pageCount)),
            writeBuf: QMutex::new(ByteStream::Init(pageCount)),
        }
    }

    pub fn PendingWriteShutdown(&self) -> bool {
        self.pendingWShutdown.load(Ordering::SeqCst)
    }

    pub fn SetPendingWriteShutdown(&self) {
        self.pendingWShutdown.store(true, Ordering::SeqCst)
    }

    pub fn HasWritingData(&self) -> bool {
        return self.writeBuf.lock().AvailableDataSize() > 0;
    }

    pub fn WriteBufAvailableDataSize(&self) -> usize {
        return self.writeBuf.lock().available;
    }

    pub fn Events(&self) -> EventMask {
        let mut event = EventMask::default();
        if self.readBuf.lock().AvailableDataSize() > 0 {
            event |= EVENT_IN;
        } else if self.RClosed() || self.WClosed() {
            event |= EVENT_IN
        }

        if self.writeBuf.lock().AvailableSpace() > 0 {
            event |= EVENT_OUT;
        }

        if self.Error() != 0 {
            event |= EVENT_ERR;
        }

        return event
    }

    pub fn WClosed(&self) -> bool {
        self.wClosed.load(Ordering::SeqCst)
    }

    pub fn RClosed(&self) -> bool {
        self.rClosed.load(Ordering::SeqCst)
    }

    pub fn SetWClosed(&self) {
        self.wClosed.store(true, Ordering::SeqCst)
    }

    pub fn SetRClosed(&self) {
        self.rClosed.store(true, Ordering::SeqCst)
    }

    pub fn Error(&self) -> i32 {
        self.error.load(Ordering::SeqCst)
    }

    pub fn SetErr(&self, err: i32) {
        self.error.store(err, Ordering::SeqCst)
    }

    // get iovs(max 2 iovs) for free read buf space
    // ret: 0: no more space, 1: 1 iov, 2: 2 iovs
    pub fn GetFreeReadIovs(&self) -> (u64, usize) {
        return self.readBuf.lock().GetSpaceIovs();
    }

    pub fn GetFreeReadBuf(&self) -> (u64, usize) {
        return self.readBuf.lock().GetSpaceBuf();
    }

    pub fn ProduceReadBuf(&self, size: usize) -> bool {
        return self.readBuf.lock().Produce(size);
    }

    pub fn ProduceAndGetFreeReadBuf(&self, size: usize) -> (bool, u64, usize) {
        let mut r = self.readBuf.lock();
        let trigger = r.Produce(size);
        let (addr, size) = r.GetSpaceBuf();
        return (trigger, addr, size)
    }

    pub fn GetAvailableWriteIovs(&self) -> (u64, usize) {
        return self.writeBuf.lock().GetDataIovs();
    }

    pub fn ConsumeWriteBuf(&self, size: usize) -> bool {
        return self.writeBuf.lock().Consume(size);
    }

    pub fn ConsumeAndGetAvailableWriteBuf(&self, size: usize) -> (bool, u64, usize) {
        let mut w = self.writeBuf.lock();
        let trigger = w.Consume(size);
        let (addr, size) = w.GetDataBuf();
        return (trigger, addr, size)
    }

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

    pub fn GetAvailableWriteBuf(&self) -> (u64, usize) {
        return self.writeBuf.lock().GetDataBuf();
    }

    pub fn Writev(&self, task: &Task, iovs: &[IoVec]) -> Result<(usize, Option<(u64, usize)>)> {
        if self.Error() != 0 {
            return Err(Error::SysError(self.Error()));
        }

        if self.WClosed() {
            error!("writev it is closed");
            //return Ok((0, None))
            return Err(Error::SysError(SysErr::EPIPE))
        }

        let mut buf = self.writeBuf.lock();
        let dstIovs = buf.GetSpaceIovsVec();
        if dstIovs.len() == 0 {
            return Err(Error::SysError(SysErr::EAGAIN));
        }

        let cnt = task.mm.CopyIovsOutToIovs(task, iovs, &dstIovs)?;

        if cnt == 0 {
            error!("writev cnt is zero....");
            return Err(Error::SysError(SysErr::EAGAIN));
        }

        let trigger = buf.Produce(cnt);
        if !trigger {
            return Ok((cnt, None))
        } else {
            let (addr, len) = buf.GetDataBuf();
            return Ok((cnt, Some((addr, len))))
        }
    }
}