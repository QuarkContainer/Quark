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

use spin::Mutex;
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
use super::super::fs::file::*;
use super::super::socket::hostinet::socket::*;
use super::super::socket::unix::transport::unix::*;
use super::super::Kernel::HostSpace;
use super::super::IOURING;
use super::uring_op::*;
use super::async::*;

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
    pub submission: Mutex<Submission>,
    pub completion: Mutex<Completion>,
    pub asyncMgr: Mutex<UringAsyncMgr>
}

impl QUring {
    pub fn New(size: usize) -> Self {
        let mut ret = QUring {
            submission: Default::default(),
            completion: Default::default(),
            asyncMgr: Mutex::new(UringAsyncMgr::New(size))
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

    pub fn Timeout(&self, _task: &Task, timerId: u64, seqNo: u64, ns: i64) -> usize {
        let ops = AsyncTimeout::New(timerId, seqNo, ns);
        let idx = self.AUCall(AsyncOps::AsyncTimeout(ops));

        return idx;
    }

    pub fn Read(&self, task: &Task, fd: i32, addr: u64, cnt: u32, offset: i64) -> i64 {
        let msg = UringOp::Read(ReadOp {
            fd: fd,
            addr: addr,
            cnt: cnt,
            offset: offset,
        });

        return self.UCall(task, msg);
    }

    pub fn Write(&self, task: &Task, fd: i32, addr: u64, cnt: u32, offset: i64) -> i64 {
        let msg = UringOp::Write(WriteOp {
            fd: fd,
            addr: addr,
            cnt: cnt,
            offset: offset,
        });

        return self.UCall(task, msg);
    }

    pub fn BufWrite(&self, _task: &Task, file: &File, fd: i32, addr: u64, len: usize, offset: i64) -> i64 {
        let ops = AsyncWritev::New(file, fd, addr, len, offset);
        self.AUCall(AsyncOps::AsyncWrite(ops));
        return len as i64
    }

    pub fn EventfdWrite(&self, fd: i32, addr: u64) {
        let ops = AsyncEventfdWrite::New(fd, addr);
        self.AUCall(AsyncOps::AsyncEventfdWrite(ops));
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
            && sockops.stype == SockType::SOCK_STREAM);

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
            let ops = self.asyncMgr.lock().GetOps(idx);
            //error!("uring process: async is {:?}", &ops.Type());
            ops.Process(ret);
        }
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

    pub fn AUCall(&self, ops: AsyncOps) -> usize {
        let id;

        loop {
            match self.asyncMgr.lock().AllocSlot() {
                None => {
                    error!("AUCall async slots usage up...");
                },
                Some(idx) => {
                    id = idx;
                    break;
                }
            }
        }

        let mut entry = self.asyncMgr.lock().SetOps(id, ops);
        loop {
            loop {
                if !self.submission.lock().IsFull() {
                    break;
                }

                error!("AUCall submission full...");
            }

            entry = match self.AUringCall(entry) {
                None => return id,
                Some(e) => e
            }
        }
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

        /*let mut processed = false;
        const BATCH_SIZE : usize = 16;
        let mut cqes : [cqueue::Entry; BATCH_SIZE] = Default::default();
        let mut cnt = 0;

        {
            let mut clock = match self.completion.try_lock() {
                None => return false,
                Some(lock) => lock
            };

            //let mut clock = self.completion.lock();

            while cnt < BATCH_SIZE {
                let cqe = {
                    clock.Next()
                };

                match cqe {
                    None => {
                        if cnt == 0 {
                            return false
                        }

                        break
                    },
                    Some(cqe) => {
                        cqes[cnt] = cqe;
                        cnt += 1;
                    }
                }
            }
        }

        let mut idx = 0;
        for cqe in &cqes[0..cnt] {
            processed = true;
            self.Process(idx, cqe);
            idx += 1;
        }

        return processed;*/
    }

    pub fn UringCall(&self, call: &UringCall) {
        let entry = call.SEntry();
        let entry = entry
            .user_data(call.Ptr());

        let mut s = self.submission.lock();
        unsafe {
            let mut queue = s.Available();
            queue.push(entry).ok().expect("submission queue is full");
        }

        s.Submit().expect("QUringIntern::submit fail");
    }

    pub fn AUringCall(&self, entry: squeue::Entry) -> Option<squeue::Entry> {
        //let (fd, user_data, opcode) = (entry.0.fd, entry.0.user_data, entry.0.opcode);
        let mut s = self.submission.lock();
        unsafe {
            let mut queue = s.Available();
            match queue.push(entry) {
                Ok(_) => (),
                Err(e) => return Some(e),
            }
        }

        let _n = s.Submit().expect("QUringIntern::submit fail");
        //error!("AUCall after sumbit fd is {}, user_data is {}, opcode is {}", fd, user_data, opcode);
        return None;
    }
}
