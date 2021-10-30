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
use core::sync::atomic;

use super::super::task::*;
use super::super::qlib::common::*;
use super::super::taskMgr::*;
pub use super::super::qlib::uring::cqueue::CompletionQueue;
pub use super::super::qlib::uring::cqueue;
pub use super::super::qlib::uring::squeue::SubmissionQueue;
pub use super::super::qlib::uring::*;
use super::super::qlib::uring::util::*;
use super::super::qlib::uring::porting::*;
use super::super::qlib::linux_def::*;
use super::super::socket::hostinet::socket::*;
use super::super::socket::unix::transport::unix::*;
use super::super::Kernel::HostSpace;
use super::super::kernel::async_wait::*;
use super::super::IOURING;
use super::uring_op::*;
use super::async::*;
use super::super::kernel::waiter::qlock::*;

pub fn QUringTrigger() -> usize {
    return IOURING.DrainCompletionQueue();
}

pub fn QUringProcessOne() -> bool {
    return IOURING.ProcessOne();
}

unsafe impl Send for Submission {}
unsafe impl Sync for Submission {}

impl Submission {
    pub fn SqLen(&self) -> usize {
        unsafe {
            let head = (*self.sq.head).load(atomic::Ordering::Acquire);
            let tail = unsync_load(self.sq.tail);

            tail.wrapping_sub(head) as usize
        }
    }

    pub fn IsFull(&self) -> bool {
        return self.sq.is_full();
    }

    pub fn FreeSlots(&self) -> usize {
        return self.sq.freeSlot();
    }

    pub fn Available(&mut self) -> squeue::AvailableQueue<'_> {
        return self.sq.available()
    }

    pub fn NeedWakeup(&self) -> bool {
        unsafe {
            (*self.sq.flags).load(atomic::Ordering::Acquire) & sys::IORING_SQ_NEED_WAKEUP != 0
        }
    }

    pub fn SubmitAndWait(&self, want: usize) -> Result<usize> {
        let len = self.SqLen();

        let mut flags = 0;

        if want > 0 {
            flags |= sys::IORING_ENTER_GETEVENTS;
        }

        if self.params.0.flags & sys::IORING_SETUP_SQPOLL != 0 {
            if self.NeedWakeup() {
                if want > 0 {
                    flags |= sys::IORING_ENTER_SQ_WAKEUP;
                } else {
                    super::super::Kernel::HostSpace::UringWake();
                    return Ok(0)
                }
            } else if want == 0 {
                // fast poll
                return Ok(len);
            }
        }

        return self.Enter(len as _, want as _, flags)
    }

    pub fn Submit(&self) -> Result<usize> {
        return self.SubmitAndWait(0)
    }

    pub fn Enter(&self,
                 to_submit: u32,
                 min_complete: u32,
                 flags: u32
    ) -> Result<usize> {
        let ret = HostSpace::IoUringEnter(self.fd.as_raw_fd(),
                                          to_submit,
                                          min_complete,
                                          flags);
        if ret < 0 {
            return Err(Error::SysError(-ret as i32))
        }

        return Ok(ret as usize)
    }

}

unsafe impl Send for Completion {}
unsafe impl Sync for Completion {}

impl Completion {
    pub fn Next(&mut self) -> Option<cqueue::Entry> {
        //return self.cq.available().next()
        return self.cq.next();
    }

    pub fn CqLen(&mut self) -> usize {
        return self.cq.len()
    }

    pub fn Overflow(&self) -> u32 {
        return self.cq.overflow();
    }
}

#[derive(Default)]
pub struct QUring {
    pub submission: QMutex<Submission>,
    pub completion: QMutex<Completion>,
    pub asyncMgr: UringAsyncMgr
}

impl QUring {
    pub fn New(size: usize) -> Self {
        let mut ret = QUring {
            submission: Default::default(),
            completion: Default::default(),
            asyncMgr: UringAsyncMgr::New(size)
        };

        super::super::Kernel::HostSpace::IoUringSetup(
            &mut ret.submission as * mut _ as u64,
            &mut ret.completion as * mut _ as u64,
        );
        return ret;
    }

    pub fn TimerRemove(&self, task: &Task, userData: u64) -> i64 {
        let msg = UringOp::TimerRemove(TimerRemoveOp{
            userData: userData,
        });

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
        self.AUCallLinked(AsyncOps::UnblockBlockPollAdd(ops), AsyncOps::AsyncLinkTimeout(timeout));
        return future;

    }

    pub fn RawTimeout(&self, _task: &Task, timerId: u64, seqNo: u64, ns: i64) -> usize {
        let ops = AsyncRawTimeout::New(timerId, seqNo, ns);
        let idx = self.AUCall(AsyncOps::AsyncRawTimeout(ops));

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

    pub fn Write(&self, task: &Task, fd: i32, addr: u64, len: u32, offset: i64) -> i64 {
        let msg = UringOp::Write(WriteOp {
            fd: fd,
            addr: addr,
            len: len,
            offset: offset,
        });

        return self.UCall(task, msg);
    }

    pub fn LogFlush(&self) {
        let fd = super::super::SHARESPACE.Logfd();
        let (addr, len) = super::super::SHARESPACE.GetDataBuf();
        let ops = AsyncLogFlush::New(fd, addr, len);
        self.AUCall(AsyncOps::AsyncLogFlush(ops));
    }

    pub fn EventfdWrite(&self, fd: i32) {
        let ops = AsyncEventfdWrite::New(fd);
        self.AUCall(AsyncOps::AsyncEventfdWrite(ops));
    }

    pub fn AsyncStatx(&self, dirfd: i32, pathname: u64, flags: i32, mask: u32, mw: &MultiWait) -> Future<Statx> {
        let future = Future::New(Statx::default());
        let ops = AsyncStatx::New(
            dirfd,
            pathname,
            flags,
            mask,
            future.clone(),
            mw
        );

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

    pub fn Statx(&self, task: &Task, dirfd: i32, pathname: u64, statxBuf: u64, flags: i32, mask: u32) -> i64 {
        let msg = UringOp::Statx(StatxOp {
            dirfd: dirfd,
            pathname: pathname,
            statxBuf: statxBuf,
            flags: flags,
            mask: mask,
        });

        return self.UCall(task, msg);
    }

    pub fn BufSockInit(&self, sockops: &SocketOperations) -> Result<()> {
        let buf = sockops.SocketBuf();

        let (addr, len) = buf.GetFreeReadBuf();
        let readop = AsyncSocketRecv::New(sockops.fd, sockops.clone(), addr, len);

        IOURING.AUCall(AsyncOps::AsyncSocketRecv(readop));

        return Ok(())
    }

    pub fn BufSockWrite(&self, _task: &Task, sockops: &SocketOperations, srcs: &[IoVec]) -> Result<i64> {
        assert!((sockops.family == AFType::AF_INET || sockops.family == AFType::AF_INET6)
            && sockops.stype == SockType::SOCK_STREAM, "family is {}, stype is {}", sockops.family, sockops.stype);

        let buf = sockops.SocketBuf();
        let (count, writeBuf) = buf.Writev(srcs)?;

        if let Some((addr, len)) = writeBuf {
            let writeop = AsyncSocketSend::New(sockops.fd, sockops.clone(), addr, len);

            IOURING.AUCall(AsyncOps::AsyncSocketSend(writeop));
            /*let sendMsgOp = AsycnSendMsg::New(sockops.fd, sockops);
            sendMsgOp.lock().SetIovs(addr, cnt);

            self.AUCall(AsyncOps::AsycnSendMsg(sendMsgOp));*/
        }

        return Ok(count as i64)
    }

    pub fn BufSockRead(&self, _task: &Task, sockops: &SocketOperations, dsts: &mut [IoVec]) -> Result<i64> {
        assert!((sockops.family == AFType::AF_INET || sockops.family == AFType::AF_INET6)
            && sockops.stype == SockType::SOCK_STREAM);

        let buf = sockops.SocketBuf();
        let (trigger, cnt) = buf.Readv(dsts)?;

        if trigger {
            let (addr, len) = buf.GetFreeReadBuf();
            let readop = AsyncSocketRecv::New(sockops.fd, sockops.clone(), addr, len);

            IOURING.AUCall(AsyncOps::AsyncSocketRecv(readop));
            //let recvMsgOp = AsycnRecvMsg::New(sockops.fd, sockops);
            //recvMsgOp.lock().SetIovs(addr, cnt);

            //self.AUCall(AsyncOps::AsycnRecvMsg(recvMsgOp));
        }

        return Ok(cnt as i64)
    }

    pub fn BufFileWrite(&self, fd: i32, buf: DataBuff, offset: i64, lockGuard: QAsyncLockGuard) -> i64 {
        let len = buf.Len() as i64;
        let writeop = AsyncBufWrite::New(fd, buf, offset, lockGuard);

        IOURING.AUCall(AsyncOps::AsyncBufWrite(writeop));
        return len
    }

    pub fn Process(&self, cqe: &cqueue::Entry) {
        let data = cqe.user_data();
        let ret = cqe.result();

        // the taskid should be larger than 0x1000 (4K)
        if data > 0x10000 {
            let call = unsafe {
                &mut *(data as * mut UringCall)
            };

            call.ret = ret;
            //error!("uring process: call is {:x?}", &call);
            ScheduleQ(call.taskId);
        } else {
            let idx = data as usize;
            let mut ops = self.asyncMgr.ops[idx].lock();
            //error!("uring process2: call is {:?}", ops.Type());

            let rerun = ops.Process(ret, idx);
            if !rerun {
                *ops = AsyncOps::None;
                self.asyncMgr.FreeSlot(idx);
            }
        }

        //error!("uring process:xxx");
    }

    pub fn UCall(&self, task: &Task, msg: UringOp) -> i64 {
        let call = UringCall {
            taskId: task.GetTaskIdQ(),
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
                },
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
                },
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
                },
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

    pub fn ProcessOne(&self) -> bool {
        let cqe = {
            let mut clock = match self.completion.try_lock() {
                None => return false,
                Some(lock) => lock
            };

            match clock.Next() {
                None => return false,
                Some(cqe) => {
                    cqe
                }
            }
        };

        self.Process(&cqe);
        return true;
    }

    pub fn DrainCompletionQueue(&self) -> usize {
        let mut count = 0;
        loop {
            let cqe = {
                let mut c = self.completion.lock();
                c.Next()
            };

            match cqe {
                None => return count,
                Some(cqe) => {
                    count += 1;
                    self.Process(&cqe);
                }
            }
        }
    }

    pub fn UringCall(&self, call: &UringCall) {
        let entry = call.SEntry();
        let entry = entry
            .user_data(call.Ptr());

        loop {
            let mut s = self.submission.lock();
            if s.FreeSlots() < 1 {
                print!("UringCall: submission full...");
                drop(s);
                super::super::qlib::ShareSpace::Yield();
                continue
            }

            unsafe {
                let mut queue = s.Available();
                queue.push(entry).ok().expect("UringCall push fail");
            }

            s.Submit().expect("QUringIntern::submit fail");
            break;
        }

    }

    pub fn AUringCall(&self, entry: squeue::Entry) {
        loop {
            let mut s = self.submission.lock();
            if s.FreeSlots() == 0 {
                print!("AUringCall1: submission full...");
                drop(s);
                super::super::qlib::ShareSpace::Yield();
                continue;
            }

            unsafe {
                let mut queue = s.Available();
                match queue.push(entry) {
                    Ok(_) => (),
                    Err(_) => panic!("AUringCall submission queue is full"),
                }
            }

            let _n = s.Submit().expect("QUringIntern::submit fail");
            break;
        }
    }

    pub fn AUringCallLinked(&self, entry1: squeue::Entry, entry2: squeue::Entry) {
        loop {
            let mut s = self.submission.lock();
            if s.FreeSlots() < 2 {
                print!("AUringCallLinked: submission full...");
                continue;
            }

            error!("AUringCallLinked xxx ");
            unsafe {
                let mut queue = s.Available();
                match queue.push(entry1.flags(squeue::Flags::IO_LINK)) {
                    Ok(_) => (),
                    Err(_e) => {
                        panic!("AUringCallLinked push fail 1 ...");
                    }
                }

                match queue.push(entry2) {
                    Ok(_) => (),
                    Err(_e) => {
                        panic!("AUringCallLinked push fail 2 ...");
                    }
                }
            }

            let _n = s.Submit().expect("QUringIntern::submit fail");
            break;
        }
    }
}
