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

use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicI32;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use alloc::collections::vec_deque::VecDeque;
use alloc::sync::Arc;
use core::ops::Deref;
use core::fmt;

use super::mutex::*;
use super::bytestream::*;
use super::linux_def::*;
use super::common::*;

pub struct SocketBuff {
    pub wClosed: AtomicBool,
    pub rClosed: AtomicBool,
    pub pendingWShutdown: AtomicBool,
    pub error: AtomicI32,

    // used by RDMA data socket, used to sync with rdma remote peer for the local read buff free space size
    // when socket application consume data and free read buf space, it will fetch_add the value
    // if the value >= 0.5 of read buf, we will send the information to the remote peer immediately otherwise,
    // when rdmadata socket send data to peer, it will read and clear the consumeReadData and send the information
    // to the peer in the rdmawrite packet to save rdmawrite call
    pub consumeReadData: AtomicU64,

    pub readBuf: QMutex<ByteStream>,
    pub writeBuf: QMutex<ByteStream>,
}

impl fmt::Debug for SocketBuff {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "wClosed {:?}, rClosed {:?}, pendingWShutdown {:?}, error {:?}",
               self.wClosed, self.wClosed, self.pendingWShutdown, self.error)
    }
}

impl Default for SocketBuff {
    fn default() -> Self {
        return SocketBuff::Init(MemoryDef::DEFAULT_BUF_PAGE_COUNT)
    }
}

impl SocketBuff {
    pub fn Init(pageCount: u64) -> Self {
        return Self {
            wClosed: AtomicBool::new(false),
            rClosed: AtomicBool::new(false),
            pendingWShutdown: AtomicBool::new(false),
            error: AtomicI32::new(0),
            consumeReadData: AtomicU64::new(0),
            readBuf: QMutex::new(ByteStream::Init(pageCount)),
            writeBuf: QMutex::new(ByteStream::Init(pageCount)),
        }
    }

    pub fn AddConsumeReadData(&self, count: u64) -> u64 {
        return self.consumeReadData.fetch_add(count, Ordering::Relaxed) + count
    }

    pub fn GetAndClearConsumeReadData(&self) -> u64 {
        return self.consumeReadData.swap(0, Ordering::Relaxed)
    }

    pub fn ReadBuf(&self) -> (u64, usize) {
        return self.readBuf.lock().GetRawBuf();
    }

    pub fn PendingWriteShutdown(&self) -> bool {
        self.pendingWShutdown.load(Ordering::SeqCst)
    }

    pub fn SetPendingWriteShutdown(&self) {
        self.pendingWShutdown.store(true, Ordering::SeqCst)
    }

    pub fn HasWriteData(&self) -> bool {
        return self.writeBuf.lock().AvailableDataSize() > 0;
    }

    pub fn HasReadData(&self) -> bool {
        return self.readBuf.lock().AvailableDataSize() > 0;
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

    pub fn GetAvailableWriteBuf(&self) -> (u64, usize) {
        return self.writeBuf.lock().GetDataBuf();
    }
}

pub const TCP_ADDR_LEN : usize = 128;

#[derive(Default, Debug)]
pub struct AcceptItem {
    pub fd: i32,
    pub addr: TcpSockAddr,
    pub len: u32,
    pub sockBuf: Arc<SocketBuff>,
}

#[derive(Default, Clone,  Debug)]
pub struct AcceptQueue(Arc<QMutex<AcceptQueueIntern>>);

impl Deref for AcceptQueue {
    type Target = Arc<QMutex<AcceptQueueIntern>>;

    fn deref(&self) -> &Arc<QMutex<AcceptQueueIntern>> {
        &self.0
    }
}

#[derive(Default, Debug)]
pub struct AcceptQueueIntern {
    pub queue: VecDeque<AcceptItem>,
    pub queueLen: usize,
    pub error: i32,
    pub total: u64,
}

impl AcceptQueueIntern {
    pub fn SetErr(&mut self, error: i32) {
        self.error = error
    }

    pub fn Err(&self) -> i32 {
        return self.error
    }

    pub fn SetQueueLen(&mut self, len: usize) {
        self.queueLen = len;
    }

    pub fn HasSpace(&self) -> bool {
        return self.queue.len() < self.queueLen
    }

    //return: (trigger, hasSpace)
    pub fn EnqSocket(&mut self, fd: i32, addr: TcpSockAddr, len: u32, sockBuf: Arc<SocketBuff>) -> (bool, bool) {
        let item = AcceptItem {
            fd: fd,
            addr: addr,
            len: len,
            sockBuf: sockBuf,
        };

        self.queue.push_back(item);
        self.total += 1;
        let trigger = self.queue.len() == 1;
        return (trigger, self.queue.len() < self.queueLen);
    }


    pub fn DeqSocket(&mut self) -> (bool, Result<AcceptItem>) {
        let trigger = self.queue.len() == self.queueLen;

        match self.queue.pop_front() {
            None => {
                if self.error != 0 {
                    return (false, Err(Error::SysError(self.error)))
                }
                return (trigger, Err(Error::SysError(SysErr::EAGAIN)))
            }
            Some(item) => {
                return (trigger, Ok(item))
            }
        }
    }

    pub fn Events(&self) -> EventMask {
        let mut event = EventMask::default();
        if self.queue.len() > 0 {
            event |= EVENT_IN;
        }

        if self.error != 0 {
            event |= EVENT_ERR;
        }

        return event
    }
}

