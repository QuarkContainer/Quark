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

use alloc::collections::vec_deque::VecDeque;
use alloc::sync::Arc;
use alloc::sync::Weak;
use core::fmt;
use core::ops::Deref;
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicI32;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;

use super::bytestream::*;
use super::common::*;
use super::linux_def::*;
use super::mutex::*;
use crate::qlib::kernel::kernel::waiter::Queue;
use crate::qlib::kernel::socket::hostinet::loopbacksocket::LoopbackSocket;
use crate::qlib::kernel::Kernel::HostSpace;
use crate::GuestHostSharedAllocator;
use crate::GUEST_HOST_SHARED_ALLOCATOR;

#[derive(Clone)]
pub struct SocketBuffWeak(pub Weak<SocketBuffIntern, GuestHostSharedAllocator>);

impl Default for SocketBuffWeak {
    fn default() -> Self {
        SocketBuff::default().Downgrade()
    }
}

impl SocketBuffWeak {
    pub fn Upgrade(&self) -> Option<SocketBuff> {
        let f = match self.0.upgrade() {
            None => return None,
            Some(f) => f,
        };

        return Some(SocketBuff(f));
    }
}

#[derive(Clone)]
pub struct SocketBuff(pub Arc<SocketBuffIntern, GuestHostSharedAllocator>);

impl Deref for SocketBuff {
    type Target = Arc<SocketBuffIntern, GuestHostSharedAllocator>;

    fn deref(&self) -> &Arc<SocketBuffIntern, GuestHostSharedAllocator> {
        &self.0
    }
}

impl Default for SocketBuff {
    fn default() -> Self {
        Self(Arc::new_in(
            SocketBuffIntern::default(),
            GUEST_HOST_SHARED_ALLOCATOR,
        ))
    }
}

impl PartialEq for SocketBuff {
    fn eq(&self, other: &Self) -> bool {
        return Arc::ptr_eq(&self.0, &other.0);
    }
}

impl fmt::Debug for SocketBuff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return f.debug_struct("SocketBuff").finish();
    }
}

impl SocketBuff {
    pub fn New(readbuf: ByteStream, writebuf: ByteStream) -> Self {
        let inner = SocketBuffIntern::New(readbuf, writebuf);
        return Self(Arc::new_in(inner, GUEST_HOST_SHARED_ALLOCATOR));
    }

    pub fn Downgrade(&self) -> SocketBuffWeak {
        return SocketBuffWeak(Arc::downgrade(&self.0));
    }
}

pub struct SocketBuffIntern {
    pub wClosed: AtomicBool,
    pub rClosed: AtomicBool,
    pub pendingWShutdown: AtomicBool,
    pub error: AtomicI32,

    // used by RDMA data socket, used to sync with rdma remote peer for the local read buff free space size
    // when socket application consume data and free read buf space, it will fetch_add the value
    // if the value >= 0.5 of read buf, we will send the information to the remote peer immediately otherwise,
    // when rdmadata socket send data to peer, it will read and clear the consumeReadData and send the information
    // to the peer in the rdmawrite packet to save rdmawrite call
    pub consumeReadData: &'static AtomicU64,

    pub readBuf: ByteStream,
    pub writeBuf: ByteStream,
}

impl fmt::Debug for SocketBuffIntern {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "wClosed {:?}, rClosed {:?}, pendingWShutdown {:?}, error {:?} readbuff {:x?}, writebuff {:x?}",
            self.wClosed, self.wClosed, self.pendingWShutdown, self.error, self.readBuf, self.writeBuf
        )
    }
}

impl Default for SocketBuffIntern {
    fn default() -> Self {
        return SocketBuffIntern::Init(MemoryDef::DEFAULT_BUF_PAGE_COUNT);
    }
}

impl SocketBuffIntern {
    pub fn New(readbuf: ByteStream, writebuf: ByteStream) -> Self {
        return Self {
            wClosed: AtomicBool::new(false),
            rClosed: AtomicBool::new(false),
            pendingWShutdown: AtomicBool::new(false),
            error: AtomicI32::new(0),
            consumeReadData: unsafe {
                let addr = 0 as *mut AtomicU64;
                &mut (*addr)
            },
            readBuf: readbuf,
            writeBuf: writebuf,
        };
    }

    pub fn Init(pageCount: u64) -> Self {
        return Self {
            wClosed: AtomicBool::new(false),
            rClosed: AtomicBool::new(false),
            pendingWShutdown: AtomicBool::new(false),
            error: AtomicI32::new(0),
            consumeReadData: unsafe {
                let addr = 0 as *mut AtomicU64;
                &mut (*addr)
            },
            readBuf: ByteStream(Arc::new_in(
                QMutex::new(ByteStreamIntern::Init(pageCount)),
                GUEST_HOST_SHARED_ALLOCATOR,
            )),
            writeBuf: ByteStream(Arc::new_in(
                QMutex::new(ByteStreamIntern::Init(pageCount)),
                GUEST_HOST_SHARED_ALLOCATOR,
            )),
        };
    }

    pub fn InitWithShareMemory(
        pageCount: u64,
        readBufHeadTailAddr: u64,
        readBufWaitingRWAddr: u64,
        writeBufHeadTailAddr: u64,
        writeBufWaitingRWAddr: u64,
        consumeReadDataAddr: u64,
        readBufAddr: u64,
        writeBufAddr: u64,
        init: bool,
    ) -> Self {
        let consumeReadData = unsafe {
            let addr = consumeReadDataAddr as *mut AtomicU64;
            &mut (*addr)
        };
        if init {
            consumeReadData.store(0, Ordering::Release);
        }

        return Self {
            wClosed: AtomicBool::new(false),
            rClosed: AtomicBool::new(false),
            pendingWShutdown: AtomicBool::new(false),
            error: AtomicI32::new(0),
            consumeReadData,
            readBuf: ByteStream(Arc::new_in(
                QMutex::new(ByteStreamIntern::InitWithShareMemory(
                    pageCount,
                    readBufHeadTailAddr,
                    readBufWaitingRWAddr,
                    readBufAddr,
                    init,
                )),
                GUEST_HOST_SHARED_ALLOCATOR,
            )),
            writeBuf: ByteStream(Arc::new_in(
                QMutex::new(ByteStreamIntern::InitWithShareMemory(
                    pageCount,
                    writeBufHeadTailAddr,
                    writeBufWaitingRWAddr,
                    writeBufAddr,
                    init,
                )),
                GUEST_HOST_SHARED_ALLOCATOR,
            )),
        };
    }

    pub fn NewDummySockBuf() -> Self {
        SocketBuffIntern::Init(2)
    }

    pub fn AddConsumeReadData(&self, count: u64) -> u64 {
        return self.consumeReadData.fetch_add(count, Ordering::Relaxed) + count;
    }

    pub fn GetAndClearConsumeReadData(&self) -> u64 {
        return self.consumeReadData.swap(0, Ordering::Relaxed);
    }

    pub fn ReadBuf(&self) -> (u64, usize) {
        return self.readBuf.lock().GetRawBuf();
    }

    pub fn WriteBuf(&self) -> (u64, usize) {
        return self.writeBuf.lock().GetRawBuf();
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
        return self.writeBuf.lock().AvailableDataSize();
    }

    pub fn Events(&self) -> EventMask {
        let mut event = EventMask::default();
        if self.readBuf.lock().AvailableDataSize() > 0 {
            event |= READABLE_EVENT;
        } else if self.RClosed() || self.WClosed() {
            event |= READABLE_EVENT | EVENT_HUP;
        }

        if self.writeBuf.lock().AvailableSpace() > 0 {
            event |= WRITEABLE_EVENT;
        }

        if self.Error() != 0 {
            event |= EVENT_ERR;
        }

        return event;
    }

    pub fn WClosed(&self) -> bool {
        self.wClosed.load(Ordering::Acquire)
    }

    pub fn RClosed(&self) -> bool {
        self.rClosed.load(Ordering::Acquire)
    }

    pub fn SetWClosed(&self) {
        self.wClosed.store(true, Ordering::Release)
    }

    pub fn SetRClosed(&self) {
        self.rClosed.store(true, Ordering::Release)
    }

    pub fn Error(&self) -> i32 {
        self.error.load(Ordering::Acquire)
    }

    pub fn SetErr(&self, err: i32) {
        self.error.store(err, Ordering::Release)
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
        return (trigger, addr, size);
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
        return (trigger, addr, size);
    }

    pub fn GetAvailableWriteBuf(&self) -> (u64, usize) {
        return self.writeBuf.lock().GetDataBuf();
    }
}

pub const TCP_ADDR_LEN: usize = 128;

#[derive(Clone, Debug)]
pub enum AcceptSocket {
    SocketBuff(SocketBuff),
    LoopbackSocket(LoopbackSocket),
    None,
}

impl From<LoopbackSocket> for AcceptSocket {
    fn from(item: LoopbackSocket) -> Self {
        return Self::LoopbackSocket(item);
    }
}

impl From<SocketBuff> for AcceptSocket {
    fn from(item: SocketBuff) -> Self {
        return Self::SocketBuff(item);
    }
}

impl Default for AcceptSocket {
    fn default() -> Self {
        return Self::None;
    }
}

#[derive(Default, Debug)]
pub struct AcceptItem {
    pub fd: i32,
    pub addr: TcpSockAddr,
    pub len: u32,
    pub sockBuf: AcceptSocket,
    pub queue: Queue,
}

#[derive(Clone, Debug)]
pub struct AcceptQueue(Arc<QMutex<AcceptQueueIntern>, GuestHostSharedAllocator>);

impl Deref for AcceptQueue {
    type Target = Arc<QMutex<AcceptQueueIntern>, GuestHostSharedAllocator>;

    fn deref(&self) -> &Arc<QMutex<AcceptQueueIntern>, GuestHostSharedAllocator> {
        &self.0
    }
}

impl AcceptQueue {
    pub fn New(len: usize, queue: Queue) -> Self {
        let inner = AcceptQueueIntern {
            aiQueue: VecDeque::new_in(GUEST_HOST_SHARED_ALLOCATOR),
            queueLen: len,
            error: 0,
            total: 0,
            queue: queue,
        };

        return Self(Arc::new_in(QMutex::new(inner), GUEST_HOST_SHARED_ALLOCATOR));
    }

    pub fn EnqSocket(
        &self,
        fd: i32,
        addr: TcpSockAddr,
        len: u32,
        sockBuf: AcceptSocket,
        queue: Queue,
    ) -> bool {
        let (trigger, hasSpace) = {
            let mut inner = self.lock();
            let item = AcceptItem {
                fd: fd,
                addr: addr,
                len: len,
                sockBuf: sockBuf,
                queue: queue,
            };

            inner.aiQueue.push_back(item);
            inner.total += 1;
            let trigger = inner.aiQueue.len() == 1;

            (trigger, inner.aiQueue.len() < inner.queueLen)
        };

        if trigger {
            let queue = self.lock().queue.clone();
            queue.Notify(READABLE_EVENT)
        }

        return hasSpace;
    }
}

pub struct AcceptQueueIntern {
    pub aiQueue: VecDeque<AcceptItem, GuestHostSharedAllocator>,
    pub queueLen: usize,
    pub error: i32,
    pub total: u64,
    pub queue: Queue,
}

impl fmt::Debug for AcceptQueueIntern {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "AcceptQueueIntern aiQueue {:x?}", self.aiQueue)
    }
}

impl Drop for AcceptQueueIntern {
    fn drop(&mut self) {
        for ai in &mut self.aiQueue {
            HostSpace::Close(ai.fd);
        }
    }
}

impl AcceptQueueIntern {
    pub fn SetErr(&mut self, error: i32) {
        self.error = error
    }

    pub fn Err(&self) -> i32 {
        return self.error;
    }

    pub fn SetQueueLen(&mut self, len: usize) {
        self.queueLen = len;
    }

    pub fn HasSpace(&self) -> bool {
        return self.aiQueue.len() < self.queueLen;
    }

    pub fn DeqSocket(&mut self) -> (bool, Result<AcceptItem>) {
        let trigger = self.aiQueue.len() == self.queueLen;

        match self.aiQueue.pop_front() {
            None => {
                if self.error != 0 {
                    return (false, Err(Error::SysError(self.error)));
                }
                return (trigger, Err(Error::SysError(SysErr::EAGAIN)));
            }
            Some(item) => return (trigger, Ok(item)),
        }
    }

    pub fn Events(&self) -> EventMask {
        let mut event = EventMask::default();
        if self.aiQueue.len() > 0 {
            event |= READABLE_EVENT;
        }

        if self.error != 0 {
            event |= EVENT_ERR;
        }

        return event;
    }
}
