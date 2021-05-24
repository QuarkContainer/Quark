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

use spin::Mutex;
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicI32;
use core::sync::atomic::Ordering;

use super::super::super::qlib::bytestream::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::common::*;

pub struct SocketBuff {
    pub closed: AtomicBool,
    pub error: AtomicI32,

    pub readBuf: Mutex<ByteStream>,
    pub writeBuf: Mutex<ByteStream>,
}

impl SocketBuff {
    pub fn Init(pageCount: u64) -> Self {
        return Self {
            closed: AtomicBool::new(false),
            error: AtomicI32::new(0),
            readBuf: Mutex::new(ByteStream::Init(pageCount)),
            writeBuf: Mutex::new(ByteStream::Init(pageCount)),
        }
    }

    pub fn WriteBufAvailableDataSize(&self) -> usize {
        return self.writeBuf.lock().available;
    }

    pub fn Events(&self) -> EventMask {
        let mut event = EventMask::default();
        if self.readBuf.lock().AvailableDataSize() > 0 {
            event |= EVENT_IN;
        } else if self.Closed() {
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

    pub fn Closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    pub fn SetClosed(&self) {
        self.closed.store(true, Ordering::SeqCst)
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

    pub fn Readv(&self, iovs: &mut [IoVec]) -> Result<(bool, usize)> {
        let mut trigger = false;
        let mut cnt = 0;

        let mut readbuf = self.readBuf.lock();
        for iov in iovs {
            let (toTrigger, size) = readbuf.read(iov.ToSliceMut())?;
            if toTrigger {
                trigger = true;
            }

            cnt += size;
            if size < iov.len && cnt != 0 {
                return Ok((trigger, cnt))
            }
        }

        if cnt > 0 {
            return Ok((trigger, cnt))
        } else if self.Error() != 0 {
            return Err(Error::SysError(self.Error()));
        } else if self.Closed() {
            return Ok((false, 0))
        } else {
            return Err(Error::SysError(SysErr::EAGAIN))
        }
    }

    pub fn GetAvailableWriteBuf(&self) -> (u64, usize) {
        return self.writeBuf.lock().GetDataBuf();
    }

    pub fn Writev(&self, iovs: &[IoVec]) -> Result<(usize, Option<(u64, usize)>)> {
        if self.Error() != 0 {
            return Err(Error::SysError(self.Error()));
        }

        if self.Closed() {
            error!("writev it is closed");
            //return Ok((0, None))
            return Err(Error::SysError(SysErr::EPIPE))
        }

        let mut trigger = false;
        let mut cnt = 0;
        let mut writebuf = self.writeBuf.lock();
        for iov in iovs {
            let (toTrigger, size) = writebuf.write(iov.ToSlice())?;
            if toTrigger {
                trigger = true;
            }

            cnt += size;
            if size < iov.len {
                break
            }
        }

        if cnt == 0 {
            error!("writev cnt is zero....");
            return Err(Error::SysError(SysErr::EAGAIN));
        }

        if !trigger {
            return Ok((cnt, None))
        } else {
            let (addr, len) = writebuf.GetDataBuf();
            return Ok((cnt, Some((addr, len))))
        }
    }
}