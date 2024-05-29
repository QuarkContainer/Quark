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
#[cfg(feature = "cc")]
use core::borrow::Borrow;

use alloc::sync::Arc;

use super::super::super::bytestream::*;
use super::super::super::common::*;
use super::super::super::object_ref::*;
use super::super::super::CompleteEntry;
//pub use super::super::super::uring::*;
use super::super::fs::file::*;
use super::super::task::*;
use super::super::taskMgr::*;

use super::super::super::linux_def::*;
use super::super::super::socket_buf::*;
use super::super::super::vcpu_mgr::*;
use super::super::kernel::async_wait::*;
use super::super::kernel::waiter::qlock::*;
use super::super::kernel::waiter::*;
use super::super::socket::hostinet::tsotsocket::*;
use super::super::socket::hostinet::uring_socket::*;
use super::super::IOURING;
use super::super::SHARESPACE;
use super::uring_async::*;
use super::uring_op::*;

pub fn QUringTrigger() -> usize {
    return IOURING.DrainCompletionQueue();
}

pub fn QUringProcessOne() -> bool {
    return IOURING.ProcessOne();
}

// unsafe impl Send for Submission {}
// unsafe impl Sync for Submission {}

pub type IOUringRef = ObjectRef<QUring>;

pub struct QUring {
    pub asyncMgr: UringAsyncMgr,
}

impl Default for QUring {
    fn default() -> Self {
        return Self::New(MemoryDef::QURING_SIZE);
    }
}

impl QUring {
    pub const MAX_URING_COUNT: usize = 8;

    pub fn New(size: usize) -> Self {
        let ret = QUring {
            asyncMgr: UringAsyncMgr::New(size),
        };

        return ret;
    }

    pub fn Addr(&self) -> u64 {
        return self as *const _ as u64;
    }

    pub fn TimerRemove(&self, task: &Task, userData: u64) -> i64 {
        let msg = UringOp::TimerRemove(TimerRemoveOp { userData: userData });

        return self.UCall(task, msg);
    }

    pub fn AsyncTimerRemove(&self, userData: u64) -> usize {
        let ops = AsyncTimerRemove::New(userData);
        let idx = self.AUCall(AsyncOps::AsyncTimerRemove(ops));
        return idx;
    }

    pub fn Timeout(&self, expire: i64, timeout: i64) -> usize {
        let ops = AsyncTimeout::New(expire, timeout);
        let idx = self.AUCall(AsyncOps::AsyncTimeout(ops));

        return idx;
    }

    pub fn UnblockPollAdd(&self, fd: i32, flags: u32, wait: &MultiWait) -> Future<EventMask> {
        let future = Future::New(0 as EventMask);
        let ops = UnblockBlockPollAdd::New(fd, flags, wait, &future);
        let timeout = AsyncLinkTimeout::New(0);
        self.AUCallLinked(
            AsyncOps::UnblockBlockPollAdd(ops),
            AsyncOps::AsyncLinkTimeout(timeout),
        );
        return future;
    }

    pub fn RawTimeout(&self, _task: &Task, timerId: u64, seqNo: u64, ns: i64) -> usize {
        let ops = AsyncRawTimeout::New(timerId, seqNo, ns);
        let idx = self.AUCall(AsyncOps::AsyncRawTimeout(ops));

        return idx;
    }

    pub fn EpollCtl(&self, epollfd: i32, fd: i32, op: i32, mask: u32) -> usize {
        let ops = AsyncEpollCtl::New(epollfd, fd, op, mask);
        let idx = self.AUCall(AsyncOps::AsyncEpollCtl(ops));

        return idx;
    }

    pub fn Read(&self, task: &Task, fd: i32, addr: u64, len: u32, offset: i64) -> i64 {
        let msg = UringOp::Read(ReadOp {
            fd: fd,
            addr: addr,
            len: len,
            offset: offset,
        });

        return self.UCall(task, msg);
    }

    pub fn SyncAccept(&self, task: &Task, fd: i32) -> i64 {
        let msg = UringOp::Accept(AcceptOp { fd: fd });

        return self.UCall(task, msg);
    }

    pub fn Write(&self, task: &Task, fd: i32, addr: u64, len: u32, offset: i64) -> i64 {
        let msg = UringOp::Write(WriteOp {
            fd: fd,
            addr: addr,
            len: len,
            offset: offset,
        });

        return self.UCall(task, msg);
    }

    pub fn Splice(
        &self,
        task: &Task,
        fdIn: i32,
        offsetIn: i64,
        fdOut: i32,
        offsetOut: i64,
        len: u32,
        flags: u32,
    ) -> i64 {
        let msg = UringOp::Splice(SpliceOp {
            fdIn: fdIn,
            offsetIn: offsetIn,
            fdOut: fdOut,
            offsetOut: offsetOut,
            len: len,
            flags: flags,
        });

        return self.UCall(task, msg);
    }

    pub fn LogFlush(&self) {
        let uringPrint = super::super::SHARESPACE.config.read().Async();
        if !uringPrint {
            return;
        }

        let fd = super::super::SHARESPACE.Logfd();
        let (addr, len) = super::super::SHARESPACE.GetDataBuf();
        let ops = AsyncLogFlush::New(fd, addr, len);
        self.AUCall(AsyncOps::AsyncLogFlush(ops));
    }

    pub fn EventfdWrite(&self, fd: i32) {
        let ops = AsyncEventfdWrite::New(fd);
        self.AUCall(AsyncOps::AsyncEventfdWrite(ops));
    }

    pub fn AsyncStatx(
        &self,
        dirfd: i32,
        pathname: u64,
        flags: i32,
        mask: u32,
        mw: &MultiWait,
    ) -> Future<Statx> {
        let future = Future::New(Statx::default());
        let ops = AsyncStatx::New(dirfd, pathname, flags, mask, future.clone(), mw);

        self.AUCall(AsyncOps::AsyncStatx(ops));
        return future;
    }

    pub fn Fsync(&self, task: &Task, fd: i32, dataSyncOnly: bool) -> i64 {
        let msg = UringOp::Fsync(FsyncOp {
            fd: fd,
            dataSyncOnly: dataSyncOnly,
        });

        return self.UCall(task, msg);
    }

    pub fn Statx(
        &self,
        task: &Task,
        dirfd: i32,
        pathname: u64,
        statxBuf: u64,
        flags: i32,
        mask: u32,
    ) -> i64 {
        let msg = UringOp::Statx(StatxOp {
            dirfd: dirfd,
            pathname: pathname,
            statxBuf: statxBuf,
            flags: flags,
            mask: mask,
        });

        return self.UCall(task, msg);
    }

    pub fn AsyncConnect(fd: i32, socket: &UringSocketOperations, sockAddr: &[u8]) -> Result<()> {
        let connectop = AsyncConnect::New(fd, socket, sockAddr);
        IOURING.AUCall(AsyncOps::AsyncConnect(connectop));

        return Ok(());
    }

    pub fn AcceptInit(&self, fd: i32, queue: &Queue, acceptQueue: &AcceptQueue) -> Result<()> {
        let acceptOp = AsyncAccept::New(fd, queue.clone(), acceptQueue.clone());
        IOURING.AUCall(AsyncOps::AsyncAccept(acceptOp));

        return Ok(());
    }

    pub fn Accept(&self, fd: i32, queue: &Queue, acceptQueue: &AcceptQueue) -> Result<AcceptItem> {
        let (trigger, ai) = acceptQueue.lock().DeqSocket();
        if trigger {
            let acceptOp = AsyncAccept::New(fd, queue.clone(), acceptQueue.clone());
            IOURING.AUCall(AsyncOps::AsyncAccept(acceptOp));
        }

        return ai;
    }

    pub fn PollHostEpollWaitInit(&self, hostEpollWaitfd: i32) {
        let op = PollHostEpollWait::New(hostEpollWaitfd);
        IOURING.AUCall(AsyncOps::PollHostEpollWait(op));
    }

    #[allow(unused_variables)]
    pub fn TsotPollInit(&self, tsotSocket: i32) {
        #[cfg(not(feature = "cc"))]{
            let op = TsotPoll::New(tsotSocket);
            IOURING.AUCall(AsyncOps::TsotPoll(op));
        }
        #[cfg(feature = "cc")]
        todo!();
    }

    #[allow(unused_variables)]
    pub fn DNSRecvInit(&self, fd: i32, msgAddr: u64) {
        #[cfg(not(feature = "cc"))]{
            let op = DNSRecv::New(fd, msgAddr);
            IOURING.AUCall(AsyncOps::DNSRecv(op));
        }
        #[cfg(feature = "cc")]
        todo!();
    }

    #[allow(unused_variables)]
    pub fn SendDns(&self, op: DNSSend) {
        #[cfg(not(feature = "cc"))]
        IOURING.AUCall(AsyncOps::DNSSend(op));
        #[cfg(feature = "cc")]
        todo!();
    }

    pub fn BufSockInit(fd: i32, queue: Queue, buf: SocketBuff, isSocket: bool) -> Result<()> {
        let (addr, len) = buf.GetFreeReadBuf();
        let readop = AsyncFileRead::New(fd, queue, buf, addr, len, isSocket);

        IOURING.AUCall(AsyncOps::AsyncFileRead(readop));

        return Ok(());
    }

    pub fn RingFileWrite(
        task: &Task,
        fd: i32,
        queue: Queue,
        buf: SocketBuff,
        srcs: &[IoVec],
        fops: Arc<FileOperations>,
    ) -> Result<i64> {
        let (count, writeBuf) = buf.Writev(task, srcs)?;

        if let Some((addr, len)) = writeBuf {
            let writeop = AsyncFiletWrite::New(fd, queue, buf, addr, len, fops);

            IOURING.AUCall(AsyncOps::AsyncFiletWrite(writeop));
        }

        return Ok(count as i64);
    }

    pub fn SocketProduce(
        task: &Task,
        fd: i32,
        queue: Queue,
        buf: SocketBuff,
        count: usize,
        ops: &UringSocketOperations,
        iovs: &mut SocketBufIovs,
    ) -> Result<()> {
        let writeBuf = buf.Produce(task, count, iovs)?;
        if let Some((addr, len)) = writeBuf {
            let writeop = AsyncSend::New(fd, queue, buf, addr, len, ops);

            IOURING.AUCall(AsyncOps::AsyncSend(writeop));
        }

        return Ok(());
    }

    #[allow(unused_variables)]
    pub fn TsotSocketProduce(
        task: &Task,
        fd: i32,
        queue: Queue,
        buf: SocketBuff,
        count: usize,
        ops: &TsotSocketOperations,
        iovs: &mut SocketBufIovs,
    ) -> Result<()> {
        #[cfg(feature = "cc")]{
            todo!();
        }
        #[cfg(not(feature = "cc"))]
        {
            let writeBuf = buf.Produce(task, count, iovs)?;
            if let Some((addr, len)) = writeBuf {
            let writeop = TsotAsyncSend::New(fd, queue, buf, addr, len, ops);
                IOURING.AUCall(AsyncOps::TsotAsyncSend(writeop));
            }
            return Ok(());
        }
    }

    pub fn SocketSend(
        task: &Task,
        fd: i32,
        queue: Queue,
        buf: SocketBuff,
        srcs: &[IoVec],
        ops: &UringSocketOperations,
    ) -> Result<i64> {
        let (count, writeBuf) = buf.Writev(task, srcs)?;

        if let Some((addr, len)) = writeBuf {
            let writeop = AsyncSend::New(fd, queue, buf, addr, len, ops);

            IOURING.AUCall(AsyncOps::AsyncSend(writeop));
        }

        return Ok(count as i64);
    }

    #[allow(unused_variables)]
    pub fn TsotSocketSend(
        task: &Task,
        fd: i32,
        queue: Queue,
        buf: SocketBuff,
        srcs: &[IoVec],
        ops: &TsotSocketOperations,
    ) -> Result<i64> {
        #[cfg(feature = "cc")]
        {
            todo!();
        }
        #[cfg(not(feature = "cc"))]
        {
            let (count, writeBuf) = buf.Writev(task, srcs)?;

            if let Some((addr, len)) = writeBuf {
                let writeop = TsotAsyncSend::New(fd, queue, buf, addr, len, ops);

                IOURING.AUCall(AsyncOps::TsotAsyncSend(writeop));
            }

            return Ok(count as i64);
        }
    }

    pub fn SocketConsume(
        task: &Task,
        fd: i32,
        queue: Queue,
        buf: SocketBuff,
        count: usize,
        iovs: &mut SocketBufIovs,
    ) -> Result<()> {
        let trigger = buf.Consume(task, count, iovs)?;

        if trigger {
            let (addr, len) = buf.GetFreeReadBuf();
            let readop = AsyncFileRead::New(fd, queue, buf, addr, len, true);

            IOURING.AUCall(AsyncOps::AsyncFileRead(readop));
        }

        return Ok(());
    }

    pub fn RingFileRead(
        task: &Task,
        fd: i32,
        queue: Queue,
        buf: SocketBuff,
        dsts: &mut [IoVec],
        isSocket: bool,
        peek: bool,
    ) -> Result<i64> {
        let (trigger, cnt) = buf.Readv(task, dsts, peek)?;

        if trigger {
            let (addr, len) = buf.GetFreeReadBuf();
            let readop = AsyncFileRead::New(fd, queue, buf, addr, len, isSocket);

            IOURING.AUCall(AsyncOps::AsyncFileRead(readop));
        }

        return Ok(cnt as i64);
    }

    pub fn BufFileWrite(
        &self,
        fd: i32,
        buf: DataBuff,
        offset: i64,
        lockGuard: QAsyncLockGuard,
    ) -> i64 {
        let len = buf.Len() as i64;
        let writeop = AsyncBufWrite::New(fd, buf, offset, lockGuard);

        IOURING.AUCall(AsyncOps::AsyncBufWrite(writeop));
        return len;
    }

    pub fn Process(&self, cqe: &CompleteEntry) {
        if super::super::Shutdown() {
            return;
        }

        let data = cqe.user_data();
        let ret = cqe.result();

        // the taskid should be larger than 0x1000 (4K)
        if data > 0x10000 {
            let call = unsafe { &mut *(data as *mut UringCall) };

            call.ret = ret;
            //error!("uring process: call is {:x?}", &call);
            ScheduleQ(call.taskId, true);
        } else {
            let idx = data as usize;
            let rerun = {
                let mut ops = self.asyncMgr.ops[idx].lock();
                //error!("uring process2: call is {:?}, idx {}", ops.Type(), idx);
                ops.ProcessResult(ret, idx)
            };

            if super::super::Shutdown() {
                return;
            }
            if !rerun {
                self.asyncMgr.FreeSlot(idx);
            }
        }

        //error!("uring process:xxx");
    }

    pub fn UCall(&self, task: &Task, msg: UringOp) -> i64 {
        let call = UringCall {
            taskId: task.GetTaskId(),
            ret: 0,
            msg: msg,
        };

        {
            self.UringCall(&call);
        }

        Wait();

        return call.ret as i64;
    }

    pub fn AUCallDirect(&self, ops: &AsyncOps, id: usize) {
        let uringEntry = UringEntry {
            ops: UringOps::AsyncOps(ops.clone()),
            userdata: id as u64,
            linked: false,
        };

        self.AUringCall(uringEntry)
    }

    pub fn AUCall(&self, ops: AsyncOps) -> usize {
        let index;

        loop {
            match self.asyncMgr.AllocSlot() {
                None => {
                    self.asyncMgr.Print();
                    //error!("AUCall async slots usage up...");
                    print!("AUCall async slots usage up...");
                }
                Some(idx) => {
                    index = idx;
                    break;
                }
            }
        }

        let entry = self.asyncMgr.SetOps(index, ops);
        self.AUringCall(entry);
        return index as usize;
    }

    pub fn AUCallLinked(&self, ops1: AsyncOps, ops2: AsyncOps) {
        let index1;

        loop {
            match self.asyncMgr.AllocSlot() {
                None => {
                    self.asyncMgr.Print();
                    //error!("AUCall async slots usage up...");
                    print!("AUCall async slots usage up...");
                }
                Some(idx) => {
                    index1 = idx;
                    break;
                }
            }
        }

        let index2;
        loop {
            match self.asyncMgr.AllocSlot() {
                None => {
                    self.asyncMgr.Print();
                    //error!("AUCall async slots usage up...");
                    print!("AUCall async slots usage up...");
                }
                Some(idx) => {
                    index2 = idx;
                    break;
                }
            }
        }

        let mut entry1 = self.asyncMgr.SetOps(index1, ops1);
        entry1.linked = true;
        let entry2 = self.asyncMgr.SetOps(index2, ops2);

        self.AUringCallLinked(entry1, entry2);
    }

    pub fn NextCompleteEntry(&self) -> Option<CompleteEntry> {
        return SHARESPACE.uringQueue.completeq.pop();
    }

    pub fn ProcessOne(&self) -> bool {
        if super::super::Shutdown() {
            return false;
        }

        let cqe = self.NextCompleteEntry();

        match cqe {
            None => return false,
            Some(cqe) => {
                self.Process(&cqe);
                return true;
            }
        }
    }

    pub fn DrainCompletionQueue(&self) -> usize {
        let mut count = 0;
        loop {
            if super::super::Shutdown() {
                return 0;
            }

            let cqe = self.NextCompleteEntry();

            match cqe {
                None => break,
                Some(cqe) => {
                    count += 1;
                    self.Process(&cqe);
                }
            }
        }

        return count;
    }

    // we will leave some queue idle to make uring more stable
    // todo: fx this, do we need throttling?
    pub const SUBMISSION_QUEUE_FREE_COUNT: usize = 10;
    pub fn NextUringIdx(cnt: u64) -> usize {
        return CPULocal::NextUringIdx(cnt);
    }

    pub fn UringCall(&self, call: &UringCall) {
        let uringEntry = UringEntry {
            ops: UringOps::UringCall(*call),
            userdata: call.Ptr(),
            linked: false,
        };
        self.UringPush(uringEntry);
    }

    pub fn UringPush(&self, entry: UringEntry) {
        #[cfg(not(feature = "cc"))]
        {
            let mut s = SHARESPACE.uringQueue.submitq.lock();
            s.push_back(entry);
        }

        #[cfg(feature = "cc")]
        {
            let mut entry = entry;
            loop {
                let r = SHARESPACE.uringQueue.submitq.push(entry);
                if r.is_ok() {
                    break;
                } else {
                    entry = r.err().unwrap();
                }
            }
        }

        SHARESPACE.Submit().expect("QUringIntern::submit fail");
        return;
    }

    pub fn AUringCall(&self, entry: UringEntry) {
        self.UringPush(entry);
    }

    pub fn AUringCallLinked(&self, entry1: UringEntry, entry2: UringEntry) {
        #[cfg(not(feature = "cc"))]
        {
            let mut s = SHARESPACE.uringQueue.submitq.lock();
            s.push_back(entry1);
            s.push_back(entry2);
        }

        #[cfg(feature = "cc")]
        {
            let s = SHARESPACE.uringQueue.submitq.borrow();
            if s.push(entry1).is_err() || s.push(entry2).is_err(){
                panic!("submitq is full");
            }
        }

        SHARESPACE.Submit().expect("QUringIntern::submit fail");
        return;
    }
}
