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

use crate::qlib::mutex::*;
use alloc::collections::vec_deque::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::marker::Send;
use core::ops::Deref;
use core::sync::atomic::AtomicU32;
use core::sync::atomic::Ordering;
use enum_dispatch::enum_dispatch;
use spin::Mutex;

use super::super::super::super::kernel_def::*;
use super::super::super::common::*;
use super::super::super::linux_def;
use super::super::super::linux_def::*;
use super::super::super::socket_buf::*;
use crate::qlib::kernel::socket::hostinet::tsotsocket::TsotSocketOperations;
//use super::super::super::uring::squeue;
use super::super::fs::file::*;
use super::super::kernel::aio::aio_context::*;
use super::super::kernel::async_wait::*;
use super::super::kernel::eventfd::*;
use super::super::kernel::timer;
use super::super::kernel::waiter::qlock::*;
use super::super::kernel::waiter::*;
use super::super::socket::hostinet::socket::*;
use super::super::socket::hostinet::uring_socket::*;
use super::super::task::*;
use super::super::IOURING;
use super::super::SHARESPACE;
use super::uring_op::UringCall;
use crate::qlib::kernel::kernel::kernel::GetKernel;
use crate::qlib::kernel::tcpip::tcpip::SockAddrInet;
use crate::GUEST_HOST_SHARED_ALLOCATOR;
use crate::GuestHostSharedAllocator;
use crate::qlib::kernel::arch::tee::is_cc_active;

pub enum UringOps {
    UringCall(UringCall),
    AsyncOps(AsyncOps),
}

pub struct UringEntry {
    pub ops: UringOps,
    pub linked: bool,
    pub userdata: u64,
}

#[enum_dispatch(AsyncOps)]
pub trait AsyncOpsTrait {
    fn Process(&mut self, _result: i32) -> bool {
        panic!("doesn't support AsyncOpsTrait::Process")
    }
}

#[enum_dispatch]
#[repr(align(128))]
#[derive(Clone)]
pub enum AsyncOps {
    AsyncTimeout(AsyncTimeout),
    AsyncTimerRemove(AsyncTimerRemove),
    AsyncTTYWrite(AsyncTTYWrite),
    AsyncWrite(AsyncWritev),
    AsyncEventfdWrite(AsyncEventfdWrite),
    AsycnSendMsg(AsycnSendMsg),
    AsycnRecvMsg(AsycnRecvMsg),
    AsyncFiletWrite(AsyncFiletWrite),
    AsyncFileRead(AsyncFileRead),
    AIOWrite(AIOWrite),
    AIORead(AIORead),
    AIOFsync(AIOFsync),
    AsyncRawTimeout(AsyncRawTimeout),
    AsyncLogFlush(AsyncLogFlush),
    AsyncStatx(AsyncStatx),
    AsyncLinkTimeout(AsyncLinkTimeout),
    UnblockBlockPollAdd(UnblockBlockPollAdd),
    AsyncBufWrite(AsyncBufWrite),
    AsyncAccept(AsyncAccept),
    AsyncEpollCtl(AsyncEpollCtl),
    AsyncSend(AsyncSend),
    TsotAsyncSend(TsotAsyncSend),
    PollHostEpollWait(PollHostEpollWait),
    AsyncConnect(AsyncConnect),
    TsotPoll(TsotPoll),
    DNSRecv(DNSRecv),
    DNSSend(DNSSend),
    None(AsyncNone),
}

impl AsyncOps {
    pub fn ProcessResult(&mut self, result: i32, id: usize) -> bool {
        let ret = self.Process(result);

        if ret {
            IOURING.AUCallDirect(self, id);
        }

        return ret;
    }

    pub fn Type(&self) -> usize {
        match self {
            AsyncOps::AsyncTimeout(_) => return 1,
            AsyncOps::AsyncTimerRemove(_) => return 2,
            AsyncOps::AsyncTTYWrite(_) => return 3,
            AsyncOps::AsyncWrite(_) => return 4,
            AsyncOps::AsyncEventfdWrite(_) => return 5,
            AsyncOps::AsycnSendMsg(_) => return 6,
            AsyncOps::AsycnRecvMsg(_) => return 7,
            AsyncOps::AsyncFiletWrite(_) => return 8,
            AsyncOps::AsyncFileRead(_) => return 9,
            AsyncOps::AIOWrite(_) => return 10,
            AsyncOps::AIORead(_) => return 11,
            AsyncOps::AIOFsync(_) => return 12,
            AsyncOps::AsyncRawTimeout(_) => return 13,
            AsyncOps::AsyncLogFlush(_) => return 14,
            AsyncOps::AsyncStatx(_) => return 15,
            AsyncOps::AsyncLinkTimeout(_) => return 16,
            AsyncOps::UnblockBlockPollAdd(_) => return 17,
            AsyncOps::AsyncBufWrite(_) => return 18,
            AsyncOps::AsyncAccept(_) => return 19,
            AsyncOps::AsyncEpollCtl(_) => return 20,
            AsyncOps::AsyncSend(_) => return 21,
            AsyncOps::TsotAsyncSend(_) => return 22,
            AsyncOps::PollHostEpollWait(_) => return 23,
            AsyncOps::AsyncConnect(_) => return 24,
            AsyncOps::TsotPoll(_) => return 25,
            AsyncOps::DNSRecv(_) => return 26,
            AsyncOps::DNSSend(_) => return 27,
            AsyncOps::None(_) => (),
        };

        return 0;
    }
}

pub struct UringAsyncMgr {
    pub ops: Vec<QMutex<AsyncOps>, GuestHostSharedAllocator>,
    pub ids: QMutex<VecDeque<u16, GuestHostSharedAllocator>>,

    // It might not be ok to free AsyncOps in Qvisor (Some drop function will use qkernel's version).
    // That's weird rust compiler behavior. So we have to store the idx here
    // and wait for qkernel to clear it.
    pub freeids: QMutex<VecDeque<u16, GuestHostSharedAllocator>>,
}

impl Default for UringAsyncMgr {
    fn default() -> Self {
        return Self::New(MemoryDef::QURING_SIZE);
    }
}
unsafe impl Sync for UringAsyncMgr {}
unsafe impl Send for UringAsyncMgr {}

impl UringAsyncMgr {
    pub fn New(size: usize) -> Self {
        let mut ids = VecDeque::with_capacity_in(size, GUEST_HOST_SHARED_ALLOCATOR);
        let mut ops = Vec::with_capacity_in(size, GUEST_HOST_SHARED_ALLOCATOR);
        for i in 0..size {
            ids.push_back(i as u16);
            ops.push(QMutex::new(AsyncOps::None(AsyncNone {})));
        }
        return Self {
            ops: ops,
            ids: QMutex::new(ids),
            freeids: QMutex::new(VecDeque::new_in(GUEST_HOST_SHARED_ALLOCATOR)),
        };
    }

    pub fn Print(&self) {
        let mut vec = Vec::new();
        for op in &self.ops {
            vec.push(op.lock().Type());
        }
        print!("UringAsyncMgr Print {:?}", vec);
        //error!("UringAsyncMgr Print {:?}", vec);
    }

    pub fn AllocSlot(&self) -> Option<usize> {
        self.Clear();
        match self.ids.lock().pop_front() {
            None => None,
            Some(id) => Some(id as usize),
        }
    }

    pub fn freeSlot(&self, id: usize) {
        *self.ops[id].lock() = AsyncOps::None(AsyncNone {});
        self.ids.lock().push_back(id as u16);
    }

    pub fn SetOps(&self, id: usize, ops: AsyncOps) -> UringEntry {
        // squeue::Entry {
            *self.ops[id].lock() = ops.clone();
        
        let uringEntry = UringEntry {
            ops: UringOps::AsyncOps(ops),
            userdata: id as u64,
            linked: false,
        };

        return uringEntry;
    }
}

#[derive(Debug, Clone)]
pub struct AsyncEventfdWrite {
    pub fd: i32,
    pub addr: u64,
}

impl AsyncEventfdWrite {
    pub fn New(fd: i32) -> Self {
        return Self { fd: fd, addr: 1 };
    }
}

impl AsyncOpsTrait for AsyncEventfdWrite {
    fn Process(&mut self, result: i32) -> bool {
        if result < 0 {
            panic!("AsyncEventfdWrite result {}, fd {}", result, self.fd);
        }
        return false;
    }
}

#[derive(Debug, Clone)]
pub struct AsyncTimeout {
    pub timeout: u64,
}

impl AsyncTimeout {
    pub fn New(_expire: i64, timeout: i64) -> Self {
        return Self {
            timeout: timeout as u64,
        };
    }
}

impl AsyncOpsTrait for AsyncTimeout {
    fn Process(&mut self, result: i32) -> bool {
        if result == -SysErr::ETIME {
            timer::Timeout();
        }

        return false;
    }
}

#[derive(Debug, Clone)]
pub struct AsyncRawTimeout {
    pub timerId: u64,
    pub seqNo: u64,
    pub timeout: u64,
}

impl AsyncRawTimeout {
    pub fn New(timerId: u64, seqNo: u64, timeout: i64) -> Self {
        return Self {
            timerId: timerId,
            seqNo: seqNo,
            timeout: timeout as u64,
        };
    }
}

impl AsyncOpsTrait for AsyncRawTimeout {
    fn Process(&mut self, result: i32) -> bool {
        if result == -SysErr::ETIME {
            // todo: deprecate this
            //timer::FireTimer(self.timerId, self.seqNo);
        }

        return false;
    }
}

#[derive(Debug, Clone)]
pub struct AsyncTimerRemove {
    pub userData: u64,
}

impl AsyncTimerRemove {
    pub fn New(userData: u64) -> Self {
        return Self { userData: userData };
    }
}

impl AsyncOpsTrait for AsyncTimerRemove {
    fn Process(&mut self, _result: i32) -> bool {
        return false;
    }
}

#[derive(Clone)]
pub struct AsyncStatx {
    pub dirfd: i32,
    pub pathname: u64,
    pub future: Future<linux_def::Statx>,
    pub flags: i32,
    pub mask: u32,
    pub statx: linux_def::Statx,
    pub mw: MultiWait,
}

impl AsyncStatx {
    pub fn New(
        dirfd: i32,
        pathname: u64,
        flags: i32,
        mask: u32,
        future: Future<linux_def::Statx>,
        mw: &MultiWait,
    ) -> Self {
        mw.AddWait();
        return Self {
            dirfd,
            pathname,
            future,
            flags,
            mask,
            statx: linux_def::Statx::default(),
            mw: mw.clone(),
        };
    }
}

impl AsyncOpsTrait for AsyncStatx {
    fn Process(&mut self, result: i32) -> bool {
        if result < 0 {
            self.future.Set(Err(Error::SysError(-result)))
        } else {
            self.future.Set(Ok(self.statx))
        }

        self.mw.Done();

        return false;
    }
}

#[derive(Clone)]
pub struct AsyncTTYWrite {
    pub file: File,
    pub fd: i32,
    pub addr: u64,
    pub len: usize,
}

impl AsyncTTYWrite {
    pub fn New(file: &File, fd: i32, addr: u64, len: usize) -> Self {
        return Self {
            file: file.clone(),
            fd: fd,
            addr: addr,
            len: len,
        };
    }
}

impl AsyncOpsTrait for AsyncTTYWrite {
    fn Process(&mut self, _result: i32) -> bool {
        //error!("AsyncWrite::Process result is {}", result);
        return false;
    }
}

#[derive(Clone)]
pub struct AsyncWritev {
    pub file: File,
    pub fd: i32,
    pub addr: u64,
    pub len: u32,
    pub offset: i64,
}

impl AsyncOpsTrait for AsyncWritev {
    fn Process(&mut self, _result: i32) -> bool {
        // add back when need
        //BUF_MGR.Free(self.addr, self.len as u64);
        return false;
    }
}

impl AsyncWritev {
    pub fn New(file: &File, fd: i32, addr: u64, len: usize, offset: i64) -> Self {
        return Self {
            file: file.clone(),
            fd: fd,
            addr: addr,
            len: len as u32,
            offset: offset,
        };
    }
}

pub struct AsyncBufWriteInner {
    pub fd: i32,
    pub buf: DataBuff,
    pub offset: i64,
    pub lockGuard: QMutex<Option<QAsyncLockGuard>>,
}

#[derive(Clone)]
pub struct AsyncBufWrite(Arc<AsyncBufWriteInner, GuestHostSharedAllocator>);

impl Deref for AsyncBufWrite {
    type Target = Arc<AsyncBufWriteInner, GuestHostSharedAllocator>;

    fn deref(&self) -> &Arc<AsyncBufWriteInner, GuestHostSharedAllocator> {
        &self.0
    }
}

impl AsyncOpsTrait for AsyncBufWrite {
    fn Process(&mut self, result: i32) -> bool {
        assert!(
            result as usize == self.buf.Len(),
            "result is {}, self.buf.len() is {}, fd is {}",
            result,
            self.buf.Len(),
            self.fd
        );
        *self.lockGuard.lock() = None;
        return false;
    }
}

impl AsyncBufWrite {
    pub fn New(fd: i32, buf: DataBuff, offset: i64, lockGuard: QAsyncLockGuard) -> Self {
        let inner = AsyncBufWriteInner {
            fd,
            buf,
            offset,
            lockGuard: QMutex::new(Some(lockGuard)),
        };

        return Self(Arc::new_in(inner, GUEST_HOST_SHARED_ALLOCATOR));
    }
}

#[derive(Clone)]
pub struct AsyncLogFlush {
    pub fd: i32,
    pub addr: u64,
    pub len: usize,
}

impl AsyncOpsTrait for AsyncLogFlush {
    fn Process(&mut self, result: i32) -> bool {
        if result <= 0 {
            panic!("AsyncLogFlush fail {}/{}", result, self.fd)
        }

        let (addr, len) = SHARESPACE.ConsumeAndGetAvailableWriteBuf(result as usize);

        if addr == 0 {
            return false;
        }

        self.addr = addr;
        self.len = len;

        return true;
    }
}

impl AsyncLogFlush {
    pub fn New(fd: i32, addr: u64, len: usize) -> Self {
        return Self { fd, addr, len };
    }
}

#[derive(Clone)]
pub struct AsyncSend {
    pub fd: i32,
    pub queue: Queue,
    pub buf: SocketBuff,
    pub addr: u64,
    pub len: usize,

    // keep the socket in the async ops to avoid socket before send finish
    pub ops: UringSocketOperations,
}

impl AsyncOpsTrait for AsyncSend {
    fn Process(&mut self, result: i32) -> bool {
        if result < 0 {
            self.buf.SetErr(-result);
            self.queue
                .Notify(EventMaskFromLinux((EVENT_ERR | READABLE_EVENT) as u32));
            return false;
            //return true;
        }

        // EOF
        // to debug
        if result == 0 {
            self.buf.SetWClosed();
            if self.buf.ProduceReadBuf(0) {
                self.queue
                    .Notify(EventMaskFromLinux(WRITEABLE_EVENT as u32));
            } else {
                self.queue
                    .Notify(EventMaskFromLinux(WRITEABLE_EVENT as u32));
            }
            return false;
        }

        let (trigger, addr, len) = self.buf.ConsumeAndGetAvailableWriteBuf(result as usize);
        if trigger {
            self.queue
                .Notify(EventMaskFromLinux(WRITEABLE_EVENT as u32));
        }

        if addr == 0 {
            if self.buf.PendingWriteShutdown() {
                self.queue.Notify(EVENT_PENDING_SHUTDOWN);
            }

            return false;
        }

        self.addr = addr;
        self.len = len;

        return true;
    }
}

impl AsyncSend {
    pub fn New(
        fd: i32,
        queue: Queue,
        buf: SocketBuff,
        addr: u64,
        len: usize,
        ops: &UringSocketOperations,
    ) -> Self {
        return Self {
            fd,
            queue,
            buf,
            addr,
            len,
            ops: ops.clone(),
        };
    }
}

#[derive(Clone)]
pub struct TsotAsyncSend {
    pub fd: i32,
    pub queue: Queue,
    pub buf: SocketBuff,
    pub addr: u64,
    pub len: usize,

    // keep the socket in the async ops to avoid socket before send finish
    pub ops: TsotSocketOperations,
}

impl AsyncOpsTrait for TsotAsyncSend {
    fn Process(&mut self, result: i32) -> bool {
        if result < 0 {
            self.buf.SetErr(-result);
            self.queue
                .Notify(EventMaskFromLinux((EVENT_ERR | READABLE_EVENT) as u32));
            return false;
            //return true;
        }

        // EOF
        // to debug
        if result == 0 {
            self.buf.SetWClosed();
            if self.buf.ProduceReadBuf(0) {
                self.queue
                    .Notify(EventMaskFromLinux(WRITEABLE_EVENT as u32));
            } else {
                self.queue
                    .Notify(EventMaskFromLinux(WRITEABLE_EVENT as u32));
            }
            return false;
        }

        let (trigger, addr, len) = self.buf.ConsumeAndGetAvailableWriteBuf(result as usize);
        if trigger {
            self.queue
                .Notify(EventMaskFromLinux(WRITEABLE_EVENT as u32));
        }

        if addr == 0 {
            if self.buf.PendingWriteShutdown() {
                self.queue.Notify(EVENT_PENDING_SHUTDOWN);
            }

            return false;
        }

        self.addr = addr;
        self.len = len;

        return true;
    }
}

impl TsotAsyncSend {
    pub fn New(
        fd: i32,
        queue: Queue,
        buf: SocketBuff,
        addr: u64,
        len: usize,
        ops: &TsotSocketOperations,
    ) -> Self {
        return Self {
            fd,
            queue,
            buf,
            addr,
            len,
            ops: ops.clone(),
        };
    }
}

#[derive(Clone)]
pub struct AsyncFiletWrite {
    pub fd: i32,
    pub queue: Queue,
    pub buf: SocketBuff,
    pub addr: u64,
    pub len: usize,
}

impl AsyncOpsTrait for AsyncFiletWrite {
    fn Process(&mut self, result: i32) -> bool {
        if result < 0 {
            self.buf.SetErr(-result);
            self.queue
                .Notify(EventMaskFromLinux((EVENT_ERR | READABLE_EVENT) as u32));
            SHARESPACE.DecrPendingWrite();
            return false;
            //return true;
        }

        // EOF
        // to debug
        if result == 0 {
            self.buf.SetWClosed();
            if self.buf.HasWriteData() {
                self.queue
                    .Notify(EventMaskFromLinux(WRITEABLE_EVENT as u32));
            } else {
                self.queue
                    .Notify(EventMaskFromLinux(WRITEABLE_EVENT as u32));
            }
            SHARESPACE.DecrPendingWrite();
            return false;
        }

        let (trigger, addr, len) = self.buf.ConsumeAndGetAvailableWriteBuf(result as usize);
        if trigger {
            self.queue
                .Notify(EventMaskFromLinux(WRITEABLE_EVENT as u32));
        }

        if addr == 0 {
            if self.buf.PendingWriteShutdown() {
                self.queue.Notify(EVENT_PENDING_SHUTDOWN);
            }

            SHARESPACE.DecrPendingWrite();
            return false;
        }

        self.addr = addr;
        self.len = len;

        return true;
    }
}

impl AsyncFiletWrite {
    pub fn New(
        fd: i32,
        queue: Queue,
        buf: SocketBuff,
        addr: u64,
        len: usize,
        _fops: Arc<FileOperations>,
    ) -> Self {
        SHARESPACE.IncrPendingWrite();
        return Self {
            fd,
            queue,
            buf,
            addr,
            len,
        };
    }
}

#[derive(Debug)]
pub struct AcceptAddr {
    pub addr: TcpSockAddr,
    pub len: AtomicU32,
}

impl AcceptAddr {
    pub fn New() -> Self {
        return Self {
            addr: TcpSockAddr::default(),
            len: AtomicU32::new(16),
        };
    }
}

#[derive(Clone)]
pub struct AsyncAccept {
    pub fd: i32,
    pub queue: Queue,
    pub acceptQueue: AcceptQueue,
    pub addr: Arc<AcceptAddr, GuestHostSharedAllocator>,
}

impl AsyncOpsTrait for AsyncAccept {
    fn Process(&mut self, result: i32) -> bool {
        if result < 0 {
            self.acceptQueue.lock().SetErr(-result);
            self.queue
                .Notify(EventMaskFromLinux((EVENT_ERR | READABLE_EVENT) as u32));
            return false;
        }

        /**************************hibernate wakeu **************************/
        // so far the quark hibernate is wakeup by accept.
        // todo: find better to handle this
        if !is_cc_active() {
            if SHARESPACE.reapFileAvaiable.load(Ordering::Relaxed) {
                ReapSwapIn();
            }

            if SHARESPACE.hibernatePause.load(Ordering::Relaxed) {
                GetKernel().Unpause();
                SHARESPACE.hibernatePause.store(false, Ordering::SeqCst);
            }

        }
        
        /**************************hibernate wakeu end **************************/

        NewSocket(result);
        let sockBuf = SocketBuff(Arc::new_in(
            SocketBuffIntern::default(),
            crate::GUEST_HOST_SHARED_ALLOCATOR,
        ));
        let hasSpace = self.acceptQueue.EnqSocket(
            result,
            self.addr.addr.Dup(),
            self.addr.len.load(Ordering::SeqCst),
            sockBuf.into(),
            Queue::default(),
        );

        self.addr.len.store(16, Ordering::SeqCst);

        return hasSpace;
    }
}

impl AsyncAccept {
    pub fn New(fd: i32, queue: Queue, acceptQueue: AcceptQueue) -> Self {
        return Self {
            fd,
            queue,
            acceptQueue,
            addr: Arc::new_in(AcceptAddr::New(), GUEST_HOST_SHARED_ALLOCATOR), //size of TcpSockAddr
        };
    }
}

#[derive(Clone)]
pub struct AsyncFileRead {
    pub fd: i32,
    pub queue: Queue,
    pub buf: SocketBuff,
    pub addr: u64,
    pub len: usize,
    pub isSocket: bool,
}

impl AsyncOpsTrait for AsyncFileRead {
    fn Process(&mut self, result: i32) -> bool {
        if result < 0 {
            self.buf.SetErr(-result);
            self.queue
                .Notify(EventMaskFromLinux((EVENT_ERR | READABLE_EVENT) as u32));
            return false;
        }

        // EOF
        if result == 0 {
            self.buf.SetRClosed();
            if self.buf.HasReadData() {
                self.queue.Notify(EventMaskFromLinux(READABLE_EVENT as u32));
            } else {
                self.queue.Notify(EventMaskFromLinux(EVENT_HUP as u32));
            }
            return false;
        }

        let (trigger, addr, len) = self.buf.ProduceAndGetFreeReadBuf(result as usize);
        if trigger {
            self.queue.Notify(EventMaskFromLinux(READABLE_EVENT as u32));
        }

        if len == 0 {
            return false;
        }

        self.addr = addr;
        self.len = len;
        return true;
    }
}

impl AsyncFileRead {
    pub fn New(
        fd: i32,
        queue: Queue,
        buf: SocketBuff,
        addr: u64,
        len: usize,
        isSocket: bool,
    ) -> Self {
        return Self {
            fd,
            queue,
            buf,
            addr,
            len,
            isSocket,
        };
    }
}

pub struct AsycnSendMsgIntern {
    pub fd: i32,
    pub ops: SocketOperations,
    pub remoteAddr: Vec<u8>,
    pub msg: MsgHdr,
}

#[derive(Clone)]
pub struct AsycnSendMsg(Arc<QMutex<AsycnSendMsgIntern>>);

impl Deref for AsycnSendMsg {
    type Target = Arc<QMutex<AsycnSendMsgIntern>>;

    fn deref(&self) -> &Arc<QMutex<AsycnSendMsgIntern>> {
        &self.0
    }
}

impl AsyncOpsTrait for AsycnSendMsg {
    fn Process(&mut self, result: i32) -> bool {
        let intern = self.lock();
        let buf = intern.ops.SocketBuf();
        if result < 0 {
            buf.SetErr(-result);
            intern.ops.Notify(EVENT_ERR | READABLE_EVENT);
            return false;
        }

        // EOF
        /*if result == 0 {
            buf.SetClosed();

            if buf.ConsumeWriteBuf(0) {
                intern.ops.Notify(EVENT_HUP);
            }
            return
        }*/

        if buf.ConsumeWriteBuf(result as usize) {
            intern.ops.Notify(WRITEABLE_EVENT);
        }

        let (addr, cnt) = intern.ops.SocketBuf().GetAvailableWriteIovs();
        if cnt == 0 {
            return false;
        }

        //let sendMsgOp = AsycnSendMsg::New(intern.fd, &intern.ops);
        self.lock().SetIovs(addr, cnt);

        return true;
    }
}

impl AsycnSendMsg {
    pub fn New(fd: i32, ops: &SocketOperations) -> Self {
        let intern = AsycnSendMsgIntern::New(fd, ops);
        return Self(Arc::new(QMutex::new(intern)));
    }
}

impl AsycnSendMsgIntern {
    pub fn New(fd: i32, ops: &SocketOperations) -> Self {
        return Self {
            fd: fd,
            ops: ops.clone(),
            remoteAddr: ops.GetRemoteAddr().unwrap(),
            msg: MsgHdr::default(),
        };
    }

    pub fn SetIovs(&mut self, addr: u64, cnt: usize) {
        self.msg.iov = addr;
        self.msg.iovLen = cnt;
        self.msg.msgName = &self.remoteAddr[0] as *const _ as u64;
        self.msg.nameLen = self.remoteAddr.len() as u32;
    }
}

pub struct AsycnRecvMsgIntern {
    pub fd: i32,
    pub ops: SocketOperations,
    pub remoteAddr: Vec<u8>,
    pub msg: MsgHdr,
}

#[derive(Clone)]
pub struct AsycnRecvMsg(Arc<QMutex<AsycnRecvMsgIntern>>);

impl Deref for AsycnRecvMsg {
    type Target = Arc<QMutex<AsycnRecvMsgIntern>>;

    fn deref(&self) -> &Arc<QMutex<AsycnRecvMsgIntern>> {
        &self.0
    }
}

impl AsyncOpsTrait for AsycnRecvMsg {
    fn Process(&mut self, result: i32) -> bool {
        let intern = self.lock();
        let buf = intern.ops.SocketBuf();
        if result < 0 {
            buf.SetErr(-result);
            intern.ops.Notify(EVENT_ERR | READABLE_EVENT);
            return false;
        }

        // EOF
        if result == 0 {
            buf.SetRClosed();
            if buf.ProduceReadBuf(0) {
                intern.ops.Notify(READABLE_EVENT);
            }
            return false;
        }

        if buf.ProduceReadBuf(result as usize) {
            intern.ops.Notify(READABLE_EVENT);
        }

        //let recvMsgOp = AsycnRecvMsg::New(intern.fd, &intern.ops);
        let (addr, cnt) = intern.ops.SocketBuf().GetFreeReadIovs();
        self.lock().SetIovs(addr, cnt);

        return true;
    }
}

impl AsycnRecvMsg {
    pub fn New(fd: i32, ops: &SocketOperations) -> Self {
        let intern = AsycnRecvMsgIntern::New(fd, ops);
        return Self(Arc::new(QMutex::new(intern)));
    }
}

impl AsycnRecvMsgIntern {
    pub fn New(fd: i32, ops: &SocketOperations) -> Self {
        let ret = Self {
            fd: fd,
            remoteAddr: ops.GetRemoteAddr().unwrap(),
            ops: ops.clone(),
            msg: MsgHdr::default(),
        };

        return ret;
    }

    pub fn SetIovs(&mut self, addr: u64, cnt: usize) {
        self.msg.iov = addr;
        self.msg.iovLen = cnt;
        self.msg.msgName = &self.remoteAddr[0] as *const _ as u64;
        self.msg.nameLen = self.remoteAddr.len() as u32;
    }
}

pub struct AIOWriteInner {
    pub fd: i32,
    pub buf: DataBuff,
    pub offset: i64,

    pub cbAddr: u64,
    pub cbData: u64,
    pub ctx: AIOContext,
    pub eventfops: Option<EventOperations>,
}

#[derive(Clone)]
pub struct AIOWrite(Arc<AIOWriteInner, GuestHostSharedAllocator>);

impl Deref for AIOWrite {
    type Target = Arc<AIOWriteInner, GuestHostSharedAllocator>;

    fn deref(&self) -> &Arc<AIOWriteInner, GuestHostSharedAllocator> {
        &self.0
    }
}

impl AsyncOpsTrait for AIOWrite {
    fn Process(&mut self, result: i32) -> bool {
        let ev = IOEvent {
            data: self.cbData,
            obj: self.cbAddr,
            result: result as i64,
            result2: 0,
        };

        // Queue the result for delivery.
        self.ctx.FinishRequest(ev);

        // Notify the event file if one was specified. This needs to happen
        // *after* queueing the result to avoid racing with the thread we may
        // wake up.
        match &self.eventfops {
            None => (),
            Some(ref eventfops) => {
                eventfops.Signal(1).expect("AIOWrite eventfops signal fail");
            }
        }

        return false;
    }
}

impl AIOWrite {
    pub fn NewWrite(
        task: &Task,
        ctx: AIOContext,
        cb: &IOCallback,
        cbAddr: u64,
        eventfops: Option<EventOperations>,
    ) -> Result<Self> {
        let vec = task.CopyInVec(cb.buf, cb.bytes as usize)?;
        let mut shared_vec = Vec::new_in(GUEST_HOST_SHARED_ALLOCATOR);
        for item in vec {
            shared_vec.push(item);
        };
        let buf = DataBuff { buf: shared_vec };
        let inner = AIOWriteInner {
            fd: cb.fd as i32,
            buf: buf,
            offset: cb.offset,
            cbAddr: cbAddr,
            cbData: cb.data,
            ctx: ctx,
            eventfops: eventfops,
        };

        return Ok(Self(Arc::new_in(inner, GUEST_HOST_SHARED_ALLOCATOR)));
    }

    pub fn NewWritev(
        task: &Task,
        ctx: AIOContext,
        cb: &IOCallback,
        cbAddr: u64,
        eventfops: Option<EventOperations>,
    ) -> Result<Self> {
        let srcs = task.IovsFromAddr(cb.buf, cb.bytes as usize)?;
        let size = IoVec::NumBytes(&srcs);
        let mut buf = DataBuff::New(size);
        task.CopyDataInFromIovs(&mut buf.buf, &srcs, false)?;

        let inner = AIOWriteInner {
            fd: cb.fd as i32,
            buf: buf,
            offset: cb.offset,
            cbAddr: cbAddr,
            cbData: cb.data,
            ctx: ctx,
            eventfops: eventfops,
        };

        return Ok(Self(Arc::new_in(inner, GUEST_HOST_SHARED_ALLOCATOR)));
    }
}

pub struct AIOReadInner {
    pub fd: i32,
    pub buf: DataBuff,
    pub iovs: Vec<IoVec, GuestHostSharedAllocator>,
    pub offset: i64,
    pub taskId: u64,

    pub cbAddr: u64,
    pub cbData: u64,
    pub ctx: AIOContext,
    pub eventfops: Option<EventOperations>,
}

#[derive(Clone)]
pub struct AIORead(Arc<AIOReadInner, GuestHostSharedAllocator>);

impl Deref for AIORead {
    type Target = Arc<AIOReadInner, GuestHostSharedAllocator>;

    fn deref(&self) -> &Arc<AIOReadInner, GuestHostSharedAllocator> {
        &self.0
    }
}

impl AsyncOpsTrait for AIORead {
    fn Process(&mut self, result: i32) -> bool {
        if result > 0 {
            let task = Task::GetTask(self.taskId);
            let len = task
                .CopyDataOutToIovsManual(&self.buf.buf[0..result as usize], &self.iovs, false)
                .expect("AIORead Process fail ...");
            assert!(len == result as usize);
        }

        let ev = IOEvent {
            data: self.cbData,
            obj: self.cbAddr,
            result: result as i64,
            result2: 0,
        };

        // Queue the result for delivery.
        self.ctx.FinishRequest(ev);

        // Notify the event file if one was specified. This needs to happen
        // *after* queueing the result to avoid racing with the thread we may
        // wake up.
        match &self.eventfops {
            None => (),
            Some(ref eventfops) => {
                eventfops.Signal(1).expect("AIOWrite eventfops signal fail");
            }
        }

        return false;
    }
}

impl AIORead {
    pub fn NewRead(
        task: &Task,
        ctx: AIOContext,
        cb: &IOCallback,
        cbAddr: u64,
        eventfops: Option<EventOperations>,
    ) -> Result<Self> {
        let iov = IoVec::NewFromAddr(cb.buf, cb.bytes as usize);
        let mut iovs = Vec::new_in(GUEST_HOST_SHARED_ALLOCATOR);
        iovs.push(iov);

        task.FixPermissionForIovs(&iovs, true)?;
        let buf = DataBuff::New(cb.bytes as usize);

        let inner = AIOReadInner {
            fd: cb.fd as i32,
            buf: buf,
            iovs: iovs,
            offset: cb.offset,
            taskId: task.taskId,
            cbAddr: cbAddr,
            cbData: cb.data,
            ctx: ctx,
            eventfops: eventfops,
        };

        return Ok(Self(Arc::new_in(inner, GUEST_HOST_SHARED_ALLOCATOR)));
    }

    pub fn NewReadv(
        task: &Task,
        ctx: AIOContext,
        cb: &IOCallback,
        cbAddr: u64,
        eventfops: Option<EventOperations>,
    ) -> Result<Self> {
        let iovs = task.IovsFromAddr(cb.buf, cb.bytes as usize)?;
        task.FixPermissionForIovs(&iovs, true)?;
        let size = IoVec::NumBytes(&iovs);
        let buf = DataBuff::New(size as usize);

        let mut iovs_s = Vec::new_in(GUEST_HOST_SHARED_ALLOCATOR);
        for a in iovs{
            iovs_s.push(a)
        }

        let inner = AIOReadInner {
            fd: cb.fd as i32,
            buf: buf,
            iovs: iovs_s,
            offset: cb.offset,
            taskId: task.taskId,
            cbAddr: cbAddr,
            cbData: cb.data,
            ctx: ctx,
            eventfops: eventfops,
        };

        return Ok(Self(Arc::new_in(inner, GUEST_HOST_SHARED_ALLOCATOR)));
    }
}

pub struct AIOFsyncInner {
    pub fd: i32,
    pub dataSyncOnly: bool,

    pub cbAddr: u64,
    pub cbData: u64,
    pub ctx: AIOContext,
    pub eventfops: Option<EventOperations>,
}

#[derive(Clone)]
pub struct AIOFsync(Arc<AIOFsyncInner, GuestHostSharedAllocator>);

impl Deref for AIOFsync {
    type Target = Arc<AIOFsyncInner, GuestHostSharedAllocator>;

    fn deref(&self) -> &Arc<AIOFsyncInner, GuestHostSharedAllocator> {
        &self.0
    }
}

impl AsyncOpsTrait for AIOFsync {
    fn Process(&mut self, result: i32) -> bool {
        let ev = IOEvent {
            data: self.cbData,
            obj: self.cbAddr,
            result: result as i64,
            result2: 0,
        };

        // Queue the result for delivery.
        self.ctx.FinishRequest(ev);

        // Notify the event file if one was specified. This needs to happen
        // *after* queueing the result to avoid racing with the thread we may
        // wake up.
        match &self.eventfops {
            None => (),
            Some(ref eventfops) => {
                eventfops.Signal(1).expect("AIOWrite eventfops signal fail");
            }
        }

        return false;
    }
}

impl AIOFsync {
    pub fn New(
        _task: &Task,
        ctx: AIOContext,
        cb: &IOCallback,
        cbAddr: u64,
        eventfops: Option<EventOperations>,
        dataSyncOnly: bool,
    ) -> Result<Self> {
        let inner = AIOFsyncInner {
            fd: cb.fd as i32,
            dataSyncOnly: dataSyncOnly,
            cbAddr: cbAddr,
            cbData: cb.data,
            ctx: ctx,
            eventfops: eventfops,
        };

        return Ok(Self(Arc::new_in(inner, GUEST_HOST_SHARED_ALLOCATOR)));
    }
}

#[derive(Clone)]
pub struct AsyncLinkTimeout {
    pub timeout: i64,
}

impl AsyncOpsTrait for AsyncLinkTimeout {
    fn Process(&mut self, _result: i32) -> bool {
        return false;
    }
}

impl AsyncLinkTimeout {
    pub fn New(timeout: i64) -> Self {
        return Self { timeout: timeout };
    }
}

#[derive(Clone)]
pub struct UnblockBlockPollAdd {
    pub fd: i32,
    pub flags: u32,
    pub wait: MultiWait,
    pub data: Future<EventMask>,
}

impl AsyncOpsTrait for UnblockBlockPollAdd {
    fn Process(&mut self, result: i32) -> bool {
        if result >= 0 {
            self.data.Set(Ok(result as EventMask));
        } else {
            self.data.Set(Err(Error::SysError(result)))
        }

        self.wait.Done();
        return false;
    }
}

impl UnblockBlockPollAdd {
    pub fn New(fd: i32, flags: u32, wait: &MultiWait, data: &Future<EventMask>) -> Self {
        wait.AddWait();

        return Self {
            fd,
            flags,
            wait: wait.clone(),
            data: data.clone(),
        };
    }
}

#[derive(Clone)]
pub struct AsyncConnect {
    pub fd: i32,
    pub addr: Arc<TcpSockAddr, GuestHostSharedAllocator>,
    pub len: u32,
    pub socket: UringSocketOperationsWeak,
}

impl AsyncOpsTrait for AsyncConnect {
    fn Process(&mut self, result: i32) -> bool {
        let socket = match self.socket.Upgrade() {
            None => return false,
            Some(s) => s,
        };

        socket.SetConnErrno(result);

        if result == 0 {
            socket
                .SetRemoteAddr(self.addr.data[0..self.len as _].to_vec())
                .expect(&format!(
                    "AsyncConnect fail {:?}",
                    &self.addr.data[0..self.len as _]
                ));
            socket.PostConnect();
        } else {
            let socktype = UringSocketType::TCPInit;
            *socket.socketType.lock() = socktype;
        }

        socket.queue.Notify(EventMaskFromLinux((EVENT_OUT) as u32));

        drop(socket);
        return false;
    }
}

impl AsyncConnect {
    pub fn New(fd: i32, socket: &UringSocketOperations, sockAddr: &[u8]) -> Self {
        let mut addr = TcpSockAddr::default();
        let len = if sockAddr.len() < addr.data.len() {
            sockAddr.len()
        } else {
            addr.data.len()
        };

        for i in 0..len {
            addr.data[i] = sockAddr[i];
        }

        let socktype = UringSocketType::TCPConnecting;
        *socket.socketType.lock() = socktype;
        socket.SetConnErrno(-SysErr::EINPROGRESS);
        return Self {
            fd,
            addr: Arc::new_in(addr, GUEST_HOST_SHARED_ALLOCATOR),
            len: len as _,
            socket: socket.Downgrade(),
        };
    }
}

#[derive(Clone)]
pub struct TsotPoll {
    pub fd: i32,
}

impl AsyncOpsTrait for TsotPoll {
    fn Process(&mut self, result: i32) -> bool {
        if result < 0 {
            error!("TsotPoll::Process result {}", result);
        }

        SHARESPACE.tsotSocketMgr.Process().unwrap();

        return true;
    }
}

impl TsotPoll {
    pub fn New(fd: i32) -> Self {
        return Self { fd };
    }
}

#[derive(Clone)]
pub struct DNSRecv {
    pub fd: i32,
    pub msgAddr: u64,
}

impl AsyncOpsTrait for DNSRecv {
    fn Process(&mut self, result: i32) -> bool {
        if result < 0 {
            error!("DNSRecv::Process result {}", result);
        }

        SHARESPACE.dnsSvc.ProcessDnsReq(result).unwrap();
        return true;
    }
}

impl DNSRecv {
    pub fn New(fd: i32, msgAddr: u64) -> Self {
        return Self {
            fd: fd,
            msgAddr: msgAddr,
        };
    }
}

#[derive(Debug)]
pub struct DNSSendInner {
    pub fd: i32,
    pub buf: DataBuff,
    pub iov: IoVec,
    pub peerAddr: SockAddrInet,
    pub msg: MsgHdr,
}

#[derive(Debug, Clone)]
pub struct DNSSend(Arc<Mutex<DNSSendInner>>);

impl Deref for DNSSend {
    type Target = Arc<Mutex<DNSSendInner>>;

    fn deref(&self) -> &Arc<Mutex<DNSSendInner>> {
        &self.0
    }
}

impl AsyncOpsTrait for DNSSend {
    fn Process(&mut self, result: i32) -> bool {
        if result < 0 {
            error!("DNSSend::Process result {}", result);
        }

        return false;
    }
}

impl DNSSend {
    pub fn New(fd: i32, buf: DataBuff, peerAddr: SockAddrInet) -> Self {
        let iov = IoVec {
            start: &buf.buf[0] as *const _ as u64,
            len: buf.buf.len(),
        };
        let inner = DNSSendInner {
            fd: fd,
            buf: buf,
            iov: iov,
            peerAddr: peerAddr,
            msg: MsgHdr::default(),
        };

        let send = Self(Arc::new(Mutex::new(inner)));

        {
            let mut lock = send.lock();
            let mut msg = MsgHdr::default();
            msg.msgName = &lock.peerAddr as *const _ as u64;
            msg.nameLen = lock.peerAddr.Len() as u32;
            msg.iov = &lock.iov as *const _ as u64;
            msg.iovLen = 1;
            lock.msg = msg;
        }

        return send;
    }
}

#[derive(Clone)]
pub struct PollHostEpollWait {
    pub fd: i32,
}

impl AsyncOpsTrait for PollHostEpollWait {
    fn Process(&mut self, result: i32) -> bool {
        if result < 0 {
            error!("PollHostEpollWait::Process result {}", result);
        }

        // we don't handle the host epollwait in kernel.
        // todo: fix this when merge kernel IO CPU and host IO thread
        // check whether there is vcpu waiting in the host can process this
        /*match SHARESPACE.TryLockEpollProcess() {
            None => (),
            Some(_) => {
                GUEST_NOTIFIER.ProcessHostEpollWait();
            }
        }

        return false;*/

        return false;
    }
}

impl PollHostEpollWait {
    pub fn New(fd: i32) -> Self {
        return Self { fd };
    }
}

#[derive(Clone, Debug, Copy)]
pub struct AsyncEpollCtl {
    pub epollfd: i32,
    pub fd: i32,
    pub op: i32,
    pub ev: EpollEvent,
}

impl AsyncOpsTrait for AsyncEpollCtl {
    fn Process(&mut self, _result: i32) -> bool {
        //assert!(result >= 0, "AsyncEpollCtl process fail fd is {} {}, {:?}", self.fd, result, self);

        return false;
    }
}

impl AsyncEpollCtl {
    pub fn New(epollfd: i32, fd: i32, op: i32, mask: u32) -> Self {
        return Self {
            epollfd: epollfd,
            fd: fd,
            op: op,
            ev: EpollEvent::new(mask, fd as u64)
        };
    }
}

#[derive(Clone, Debug, Copy)]
pub struct AsyncNone {}

impl AsyncOpsTrait for AsyncNone {}
