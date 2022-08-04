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

use alloc::sync::Arc;
use core::sync::atomic;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;

use super::super::super::common::*;
use super::super::super::object_ref::*;
pub use super::super::super::uring::cqueue;
pub use super::super::super::uring::cqueue::CompletionQueue;
pub use super::super::super::uring::squeue::SubmissionQueue;
pub use super::super::super::uring::*;
use super::super::fs::file::*;
use super::super::task::*;
use super::super::taskMgr::*;

use super::super::super::linux_def::*;
use super::super::super::socket_buf::*;
use super::super::super::uring::util::*;
use super::super::super::vcpu_mgr::*;
use super::super::kernel::async_wait::*;
use super::super::kernel::waiter::qlock::*;
use super::super::kernel::waiter::*;
use super::super::socket::hostinet::uring_socket::*;
use super::super::Kernel::HostSpace;
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

unsafe impl Send for Submission {}
unsafe impl Sync for Submission {}

impl IoUring {
    pub fn SqLen(&self) -> usize {
        let sq = self.sq.lock();
        unsafe {
            let head = (*sq.head).load(atomic::Ordering::Acquire);
            let tail = unsync_load(sq.tail);

            tail.wrapping_sub(head) as usize
        }
    }

    pub fn IsFull(&self) -> bool {
        return self.sq.lock().is_full();
    }

    pub fn FreeSlots(&self) -> usize {
        return self.sq.lock().freeSlot();
    }

    pub fn NeedWakeup(&self) -> bool {
        unsafe {
            (*self.sq.lock().flags).load(atomic::Ordering::Acquire) & sys::IORING_SQ_NEED_WAKEUP
                != 0
        }
    }

    pub fn SubmitAndWait(&self, _want: usize) -> Result<usize> {
        self.pendingCnt.fetch_add(1, Ordering::Release);

        if SHARESPACE.HostProcessor() == 0 {
            SHARESPACE.scheduler.VcpuArr[0].Wakeup();
        }

        return Ok(0);
    }

    pub fn Submit(&self) -> Result<usize> {
        return self.SubmitAndWait(0);
    }

    pub fn Enter(
        &self,
        to_submit: u32,
        min_complete: u32,
        flags: u32,
    ) -> Result<usize> {
        let ret = HostSpace::IoUringEnter(to_submit, min_complete, flags);
        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        return Ok(ret as usize);
    }

    pub fn Next(&mut self) -> Option<cqueue::Entry> {
        //return self.cq.available().next()
        return self.cq.lock().next();
    }

    pub fn CqLen(&mut self) -> usize {
        return self.cq.lock().len();
    }

    pub fn Overflow(&self) -> u32 {
        return self.cq.lock().overflow();
    }
}

pub type IOUringRef = ObjectRef<QUring>;

#[derive(Default)]
pub struct QUring {
    pub uringsAddr: AtomicU64,
    pub asyncMgr: UringAsyncMgr,
}

impl QUring {
    pub const MAX_URING_COUNT: usize = 8;

    pub fn New(size: usize) -> Self {
        let ret = QUring {
            asyncMgr: UringAsyncMgr::New(size),
            uringsAddr: AtomicU64::new(0),
        };

        return ret;
    }

    pub fn Addr(&self) -> u64 {
        return self as *const _ as u64;
    }

    pub fn SetIOUringsAddr(&self, addr: u64) {
        self.uringsAddr.store(addr, atomic::Ordering::SeqCst);
    }

    #[inline(always)]
    pub fn IOUring(&self) -> &'static IoUring {
        let addr = self.uringsAddr.load(atomic::Ordering::Relaxed);
        let uring = unsafe { &*(addr as *const IoUring) };

        return uring;
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

    pub fn BufSockInit(fd: i32, queue: Queue, buf: Arc<SocketBuff>, isSocket: bool) -> Result<()> {
        let (addr, len) = buf.GetFreeReadBuf();
        let readop = AsyncFileRead::New(fd, queue, buf, addr, len, isSocket);

        IOURING.AUCall(AsyncOps::AsyncFileRead(readop));

        return Ok(());
    }

    pub fn RingFileWrite(
        task: &Task,
        fd: i32,
        queue: Queue,
        buf: Arc<SocketBuff>,
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

    pub fn SocketSend(
        task: &Task,
        fd: i32,
        queue: Queue,
        buf: Arc<SocketBuff>,
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

    pub fn RingFileRead(
        task: &Task,
        fd: i32,
        queue: Queue,
        buf: Arc<SocketBuff>,
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

    pub fn Process(&self, cqe: &cqueue::Entry) {
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
                ops.Process(ret, idx)
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
        let entry = ops.SEntry().user_data(id as u64);
        self.AUringCall(entry)
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

        let entry1 = self.asyncMgr.SetOps(index1, ops1);
        let entry2 = self.asyncMgr.SetOps(index2, ops2);

        self.AUringCallLinked(entry1, entry2);
    }

    pub fn NextCompleteEntry(&self) -> Option<cqueue::Entry> {
        if super::super::SHARESPACE.config.read().UringBuf {
            return self.IOUring().completeq.lock().pop_front();
        } else {
            return self.IOUring().cq.lock().next();
        }
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
        let entry = call.SEntry();
        let entry = entry.user_data(call.Ptr());


        self.UringPush(entry);
    }

    pub fn UringPush(&self, entry: squeue::Entry) {
        if super::super::SHARESPACE.config.read().UringBuf {
            let mut s = self.IOUring().submitq.lock();
            s.push_back(entry);
        } else {
            loop {
                let mut s = self.IOUring().sq.lock();
                if s.freeSlot() < Self::SUBMISSION_QUEUE_FREE_COUNT {
                    drop(s);
                    super::super::super::ShareSpace::Yield();
                    error!("AUringCall1: submission full... ");
                    continue;
                }

                unsafe {
                    match s.push(entry) {
                        Ok(_) => (),
                        Err(_) => panic!("AUringCall submission queue is full"),
                    }
                }

                break;
            }
        }

        self.IOUring()
            .Submit()
            .expect("QUringIntern::submit fail");
        return;
    }

    pub fn AUringCall(&self, entry: squeue::Entry) {
        self.UringPush(entry);
    }

    pub fn AUringCallLinked(&self, entry1: squeue::Entry, entry2: squeue::Entry) {
        if super::super::SHARESPACE.config.read().UringBuf {
            let mut s = self.IOUring().submitq.lock();
            s.push_back(entry1.flags(squeue::Flags::IO_LINK));
            s.push_back(entry2);
        } else {
            loop {
                let mut s = self.IOUring().sq.lock();
                if s.freeSlot() < Self::SUBMISSION_QUEUE_FREE_COUNT + 1 {
                    drop(s);
                    super::super::super::ShareSpace::Yield();
                    error!("AUringCallLinked: submission full... idx");
                    continue;
                }

                unsafe {
                    match s.push(entry1.flags(squeue::Flags::IO_LINK)) {
                        Ok(_) => (),
                        Err(_e) => {
                            panic!("AUringCallLinked push fail 1 ...");
                        }
                    }

                    match s.push(entry2) {
                        Ok(_) => (),
                        Err(_e) => {
                            panic!("AUringCallLinked push fail 2 ...");
                        }
                    }
                }

                break;
            }
        }

        self.IOUring()
            .Submit()
            .expect("QUringIntern::submit fail");
        return;
    }
}
