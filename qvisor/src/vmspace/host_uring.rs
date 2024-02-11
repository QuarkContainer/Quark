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

use alloc::collections::VecDeque;
use core::mem;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use std::time::Duration;
use crossbeam_queue::ArrayQueue;
use std::ptr;
use enum_dispatch::enum_dispatch;

use io_uring::opcode;
use io_uring::squeue;
use io_uring::types;

use crate::qlib::kernel::quring::uring_op::*;
use crate::qlib::kernel::quring::uring_async::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::mutex::QMutex;
use super::super::qlib::uring::porting::*;
use super::super::qlib::uring::util::*;
use super::super::qlib::uring::*;
use super::super::util::*;
use super::super::*;
use super::kernel::SHARESPACE;
use super::syscall::*;
use crate::vmspace::kernel::GlobalIOMgr;
use crate::URING;

impl Mmap {
    pub fn new(fd: i32, offset: u64, len: usize) -> Result<Mmap> {
        let prot = (MmapProt::PROT_WRITE | MmapProt::PROT_READ) as i32;

        let addr = PMA_KEEPER.MapFile(len as u64, prot, fd, offset)?;

        return Ok(Mmap {
            addr: addr,
            len: len,
        });
    }
}

impl IoUring {
    /// Create a IoUring instance
    ///
    /// The `entries` sets the size of queue,
    /// and it value should be the power of two.
    #[inline]
    pub fn new(entries: u32) -> Result<IoUring> {
        IoUring::with_params(entries, Default::default())
    }

    pub fn SubmitEntry(&self, count: usize) -> Result<usize> {
        let ret = IOUringEnter(self.fd.as_raw_fd(), count as _, 0, 0);

        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        return Ok(ret as usize);
    }

    fn SetupQueue(
        fd: i32,
        p: &sys::io_uring_params,
    ) -> Result<(MemoryMap, SubmissionQueue, CompletionQueue)> {
        let sq_len = p.sq_off.array as usize + p.sq_entries as usize * mem::size_of::<u32>();
        let cq_len =
            p.cq_off.cqes as usize + p.cq_entries as usize * mem::size_of::<sys::io_uring_cqe>();
        let sqe_len = p.sq_entries as usize * mem::size_of::<sys::io_uring_sqe>();
        let sqe_mmap = Mmap::new(fd, sys::IORING_OFF_SQES as _, sqe_len)?;

        //error!("is_feature_nodrop is {}", p.features & sys::IORING_FEAT_NODROP != 0);
        //error!("is_feature_submit_stable is {}", p.features & sys::IORING_FEAT_SUBMIT_STABLE != 0);

        if p.features & sys::IORING_FEAT_SINGLE_MMAP != 0 {
            let scq_mmap = Mmap::new(
                fd,
                sys::IORING_OFF_SQ_RING as _,
                core::cmp::max(sq_len, cq_len),
            )?;

            let sq = unsafe { SubmissionQueue::new(&scq_mmap, &sqe_mmap, p) };
            let cq = unsafe { CompletionQueue::new(&scq_mmap, p) };

            let mm = MemoryMap {
                sq_mmap: scq_mmap,
                cq_mmap: None,
                sqe_mmap,
            };

            Ok((mm, sq, cq))
        } else {
            let sq_mmap = Mmap::new(fd, sys::IORING_OFF_SQ_RING as _, sq_len)?;
            let cq_mmap = Mmap::new(fd, sys::IORING_OFF_CQ_RING as _, cq_len)?;

            let sq = unsafe { SubmissionQueue::new(&sq_mmap, &sqe_mmap, p) };
            let cq = unsafe { CompletionQueue::new(&cq_mmap, p) };
            let mm = MemoryMap {
                cq_mmap: Some(cq_mmap),
                sq_mmap,
                sqe_mmap,
            };

            Ok((mm, sq, cq))
        }
    }

    fn with_params(entries: u32, mut p: sys::io_uring_params) -> Result<IoUring> {
        let fd: i32 = IOUringSetup(entries, &mut p as *mut _ as u64) as i32;

        if fd < 0 {
            return Err(Error::SysError(-fd));
        }

        let _hostfd = GlobalIOMgr().AddFile(fd);
        let (mm, sq, cq) = Self::SetupQueue(fd, &p)?;

        Ok(IoUring {
            fd: Fd(fd),
            lock: QMutex::new(()),
            pendingCnt: AtomicU64::new(0),
            sq: QMutex::new(sq),
            cq: QMutex::new(cq),
            submitq: QMutex::new(VecDeque::with_capacity(16)),
            completeq: ArrayQueue::new(MemoryDef::QURING_SIZE),
            params: Parameters(p),
            memory: mm,
        })
    }

    /// Initiate asynchronous I/O.
    #[inline]
    pub fn submit(&self) -> Result<usize> {
        self.submitter().submit()
    }

    pub fn CopyCompleteEntry(&self) -> usize {
        let mut count = 0;

        let mut uring = URING.lock();
        {
            let mut cq = uring.completion();
            loop {
                let cqe = cq.next();
    
                match cqe {
                    None => break,
                    Some(cqe) => {
                        count += 1;
                        let entry = unsafe {
                            *(&cqe as * const _ as u64 as * const _)
                        };
                        match self.completeq.push(entry) {
                            Err(_) => {
                                panic!("CopyCompleteEntry fail ...");
                            }
                            _ => (),
                        }
                    }
                }
            }
        }

        return count;
    }

    #[inline]
    pub fn HostSubmit(&self) -> Result<usize> {
        if QUARK_CONFIG.lock().UringBuf {
            let _count = self.CopyCompleteEntry();
            
            let mut count = 0;

            {
                let mut uring = URING.lock();
                let mut sq = uring.submission();
                let mut submitq = self.submitq.lock();

                if sq.dropped() != 0 {
                    error!("uring fail dropped {}", sq.dropped());
                }

                if sq.cq_overflow() {
                    error!("uring fail overflow")
                }
                assert!(sq.dropped() == 0, "dropped {}", sq.dropped());
                assert!(!sq.cq_overflow());

                while !sq.is_full() {
                    let uringEntry = match submitq.pop_front() {
                        None => break,
                        Some(e) => e,
                    };

                    let entry = match &uringEntry.ops {
                        UringOps::UringCall(call) => {
                            call.Entry()
                        }
                        UringOps::AsyncOps(ops) => {
                            ops.Entry()
                        }
                    };
            
                    let entry = entry.user_data(uringEntry.userdata);
                    let entry = if uringEntry.linked {
                        entry.flags(squeue::Flags::IO_LINK)
                    } else {
                        entry
                    };

                    unsafe {
                        match sq.push(&entry) {
                            Ok(_) => (),
                            Err(_) => panic!("AUringCall submission queue is full"),
                        }
                    }

                    count += 1;
                }
            }
        
            if count > 0 {
                let ret = URING.lock().submit_and_wait(0)?;
                return Ok(ret);
            }

            return Ok(0);
        } else {
            let count = self.pendingCnt.swap(0, Ordering::Acquire);
            if count == 0 {
                return Ok(0);
            }

            let ret = self.SubmitEntry(count as _)?;
            return Ok(ret);
        }
    }

    /// Initiate and/or complete asynchronous I/O
    ///
    /// # Safety
    ///
    /// This provides a raw interface so developer must ensure that parameters are correct.
    #[inline]
    pub unsafe fn enter(
        &self,
        to_submit: u32,
        min_complete: u32,
        flag: u32,
        //sig: Option<&libc::sigset_t>,
    ) -> Result<usize> {
        self.submitter().enter(to_submit, min_complete, flag)
    }

    /// Initiate and/or complete asynchronous I/O
    #[inline]
    pub fn submit_and_wait(&self, want: usize) -> Result<usize> {
        self.submitter().submit_and_wait(want)
    }
}

impl Builder {
    // Build a [IoUring].
    #[inline]
    pub fn build(&self, entries: u32) -> Result<IoUring> {
        let ring = IoUring::with_params(entries, self.params)?;
        Ok(ring)
    }
}

impl<'a> Submitter<'a> {
    /// Initiate asynchronous I/O.
    #[inline]
    pub fn submit(&self) -> Result<usize> {
        self.submit_and_wait(0)
    }

    /// Initiate and/or complete asynchronous I/O
    pub fn submit_and_wait(&self, want: usize) -> Result<usize> {
        let len = self.sq_len();

        let mut flags = 0;

        if want > 0 {
            flags |= sys::IORING_ENTER_GETEVENTS;
        }

        if self.flags & sys::IORING_SETUP_SQPOLL != 0 {
            if self.sq_need_wakeup() {
                if want > 0 {
                    flags |= sys::IORING_ENTER_SQ_WAKEUP;
                } else {
                    return Ok(0);
                }
            } else if want == 0 {
                // fast poll
                return Ok(len);
            }
        }

        unsafe { self.enter(len as _, want as _, flags) }
    }

    /// Initiate and/or complete asynchronous I/O
    ///
    /// # Safety
    ///
    /// This provides a raw interface so developer must ensure that parameters are correct.
    pub unsafe fn enter(&self, to_submit: u32, min_complete: u32, flag: u32) -> Result<usize> {
        let ret = IOUringEnter(self.fd.as_raw_fd(), to_submit, min_complete, flag);

        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        return Ok(ret as usize);
    }
}

//pub const IORING_REGISTER_FILES: u32 = 2;
//pub const IORING_UNREGISTER_FILES: u32 = 3;
//pub const IORING_REGISTER_EVENTFD: u32 = 4;
pub const NR_IO_URING_SETUP: usize = 425;
pub const NR_IO_URING_ENTER: usize = 426;
pub const NR_IO_URING_REGISTER: usize = 427;

pub fn IOUringSetup(entries: u32, params: u64) -> i64 {
    let res = unsafe { syscall2(NR_IO_URING_SETUP, entries as usize, params as usize) as i64 };
    if res < 0 {
        return SysRet(res);
    }

    return res;
}

pub fn IOUringRegister(fd: i32, Opcode: u32, arg: u64, nrArgs: u32) -> i64 {
    let res = unsafe {
        syscall4(
            NR_IO_URING_REGISTER,
            fd as usize,
            Opcode as usize,
            arg as usize,
            nrArgs as usize,
        ) as i64
    };

    return res;
}

pub fn IOUringEnter(fd: i32, toSubmit: u32, minComplete: u32, flags: u32) -> i64 {
    let res = unsafe {
        syscall6(
            NR_IO_URING_ENTER,
            fd as usize,
            toSubmit as usize,
            minComplete as usize,
            flags as usize,
            0,
            core::mem::size_of::<libc::sigset_t>() as usize,
        ) as i64
    };

    return res;
}

impl UringEntry {
    pub fn Entry(&self) -> squeue::Entry {
        let entry = match &self.ops {
            UringOps::UringCall(call) => {
                error!("request is {:x?}/{:x}", call, self.userdata);
                call.Entry()
            }
            UringOps::AsyncOps(ops) => ops.Entry(),
        };

        let entry = entry.user_data(self.userdata);
        let entry = if self.linked {
            entry.flags(squeue::Flags::IO_LINK)
        } else {
            entry
        };

        return entry;
    }
}

impl TimerRemoveOp {
    pub fn Entry(&self) -> squeue::Entry {
        let op = opcode::TimeoutRemove::new(self.userData);

        return op.build();
    }
}

impl ReadOp {
    pub fn Entry(&self) -> squeue::Entry {
        let op = opcode::Read::new(types::Fd(self.fd), self.addr as *mut _, self.len).offset(self.offset as u64);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl WriteOp {
    pub fn Entry(&self) -> squeue::Entry {
        let op = opcode::Write::new(
            types::Fd(self.fd), 
            self.addr as *const _, 
            self.len
        )
        .offset(
            self.offset as u64
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl StatxOp {
    pub fn Entry(&self) -> squeue::Entry {
        let op = opcode::Statx::new(
            types::Fd(self.dirfd),
            self.pathname as *const _,
            self.statxBuf as *mut types::statx,
        )
        .flags(self.flags)
        .mask(self.mask);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl FsyncOp {
    pub fn Entry(&self) -> squeue::Entry {
        let op = if self.dataSyncOnly {
            opcode::Fsync::new(types::Fd(self.fd)).flags(types::FsyncFlags::DATASYNC)
        } else {
            opcode::Fsync::new(types::Fd(self.fd))
        };

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl SpliceOp {
    pub fn Entry(&self) -> squeue::Entry {
        let op = opcode::Splice::new(
            types::Fd(self.fdIn),
            self.offsetIn,
            types::Fd(self.fdOut),
            self.offsetOut,
            self.len,
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl EpollCtlOp {
    pub fn Entry(&self) -> squeue::Entry {
        let op = opcode::EpollCtl::new(
            types::Fd(self.epollfd),
            types::Fd(self.fd),
            self.op,
            &self.ev as *const _ as u64 as _, //*const types::epoll_event,
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl AcceptOp {
    pub fn Entry(&self) -> squeue::Entry {
        let op = opcode::Accept::new(
            types::Fd(self.fd),
            ptr::null_mut(),
            ptr::null_mut(),
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringCall {
    pub fn Entry(&self) -> squeue::Entry {
        match self.msg {
            UringOp::None => (),
            UringOp::TimerRemove(ref msg) => return msg.Entry(),
            UringOp::Read(ref msg) => return msg.Entry(),
            UringOp::Write(ref msg) => return msg.Entry(),
            UringOp::Statx(ref msg) => return msg.Entry(),
            UringOp::Fsync(ref msg) => return msg.Entry(),
            UringOp::Splice(ref msg) => return msg.Entry(),
            UringOp::Accept(ref msg) => return msg.Entry(),
        };

        panic!("UringCall SEntry UringOp::None")
    }
}

impl UringAsyncOpsTrait for AsyncEventfdWrite {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Write::new(
            types::Fd(self.fd),
            &self.addr as *const _ as u64 as *const u8,
            8,
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncTimeout {
    fn Entry(&self) -> squeue::Entry {
        let ts = types::Timespec::from(
            Duration::new(self.timeout / 1000_000_000, (self.timeout % 1000_000_000) as u32)
        );
        let op = opcode::Timeout::new(&ts);
        return op.build();
    }
}

impl UringAsyncOpsTrait for AsyncRawTimeout {
    fn Entry(&self) -> squeue::Entry {
        let ts = types::Timespec::from(
            Duration::new(self.timeout / 1000_000_000, (self.timeout % 1000_000_000) as u32)
        );
        let op = opcode::Timeout::new(&ts);
        return op.build();
    }
}

impl UringAsyncOpsTrait for AsyncTimerRemove {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::TimeoutRemove::new(self.userData);

        return op.build();
    }
}

impl UringAsyncOpsTrait for AsyncStatx {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Statx::new(
            types::Fd(self.dirfd),
            self.pathname as *const _,
            &self.statx as *const _ as u64 as *mut types::statx,
        )
        .flags(self.flags)
        .mask(self.mask);

        return op.build();
    }
}

impl UringAsyncOpsTrait for AsyncTTYWrite {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Write::new(types::Fd(self.fd), self.addr as *const _, self.len as u32);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}


impl UringAsyncOpsTrait for AsyncWritev {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Write::new(
            types::Fd(self.fd), 
            self.addr as *const u8, 
            self.len
        )
        .offset(
            self.offset as u64
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncBufWrite {
    fn Entry(&self) -> squeue::Entry {
        //let op = Write::new(types::Fd(self.fd), self.addr as * const u8, self.len as u32);
        let op = opcode::Write::new(
            types::Fd(self.fd),
            self.buf.Ptr() as *const u8,
            self.buf.Len() as u32,
        )
        .offset(self.offset as _);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncLogFlush {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Write::new(types::Fd(self.fd), self.addr as *const u8, self.len as u32); //.flags(MsgType::MSG_DONTWAIT);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncSend {
    fn Entry(&self) -> squeue::Entry {
        //let op = Write::new(types::Fd(self.fd), self.addr as * const u8, self.len as u32);
        let op = opcode::Send::new(types::Fd(self.fd), self.addr as *const u8, self.len as u32);
        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for TsotAsyncSend {
    fn Entry(&self) -> squeue::Entry {
        //let op = Write::new(types::Fd(self.fd), self.addr as * const u8, self.len as u32);
        let op = opcode::Send::new(types::Fd(self.fd), self.addr as *const u8, self.len as u32);
        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncFiletWrite {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Write::new(types::Fd(self.fd), self.addr as *const u8, self.len as u32);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncAccept {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Accept::new(
            types::Fd(self.fd),
            &self.addr as *const _ as u64 as *mut _,
            &self.len as *const _ as u64 as *mut _,
        );
        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncFileRead {
    fn Entry(&self) -> squeue::Entry {
        if self.isSocket {
            let op = opcode::Recv::new(types::Fd(self.fd), self.addr as *mut u8, self.len as u32);
            if SHARESPACE.config.read().UringFixedFile {
                return op.build().flags(squeue::Flags::FIXED_FILE);
            } else {
                return op.build();
            }
        }

        let op = opcode::Read::new(types::Fd(self.fd), self.addr as *mut u8, self.len as u32);
        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsycnSendMsg {
    fn Entry(&self) -> squeue::Entry {
        let intern = self.lock();
        let op = opcode::SendMsg::new(
            types::Fd(intern.fd), 
            &intern.msg as *const _ as *const _
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsycnRecvMsg {
    fn Entry(&self) -> squeue::Entry {
        let intern = self.lock();
        let op = opcode::RecvMsg::new(
            types::Fd(intern.fd), 
            &intern.msg as *const _ as u64 as *mut _
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AIOWrite {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Write::new(
            types::Fd(self.fd),
            self.buf.Ptr() as *const u8,
            self.buf.Len() as u32,
        )
        .offset(self.offset as u64);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AIORead {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Read::new(
            types::Fd(self.fd),
            self.buf.Ptr() as *mut u8,
            self.buf.Len() as u32,
        )
        .offset(self.offset as u64);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AIOFsync {
    fn Entry(&self) -> squeue::Entry {
        let op = if self.dataSyncOnly {
            opcode::Fsync::new(types::Fd(self.fd)).flags(types::FsyncFlags::DATASYNC)
        } else {
            opcode::Fsync::new(types::Fd(self.fd))
        };

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncLinkTimeout {
    fn Entry(&self) -> squeue::Entry {
        let ts = types::Timespec::from(
            Duration::new(self.ts.tv_sec as u64, self.ts.tv_nsec as _)
        );
        let op = opcode::LinkTimeout::new(&ts);

        return op.build();
    }
}

impl UringAsyncOpsTrait for UnblockBlockPollAdd {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::PollAdd::new(types::Fd(self.fd), self.flags);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncConnect {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::Connect::new(
            types::Fd(self.fd), 
            &self.addr as * const _ as u64 as *const _, 
            self.len
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for TsotPoll {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::PollAdd::new(types::Fd(self.fd), EVENT_READ as u32);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for PollHostEpollWait {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::PollAdd::new(types::Fd(self.fd), EVENT_READ as u32);

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncEpollCtl {
    fn Entry(&self) -> squeue::Entry {
        let op = opcode::EpollCtl::new(
            types::Fd(self.epollfd),
            types::Fd(self.fd),
            self.op,
            &self.ev as *const _ as u64 as *const types::epoll_event,
        );

        if SHARESPACE.config.read().UringFixedFile {
            return op.build().flags(squeue::Flags::FIXED_FILE);
        } else {
            return op.build();
        }
    }
}

impl UringAsyncOpsTrait for AsyncNone {}

#[enum_dispatch(AsyncOps)]
pub trait UringAsyncOpsTrait {
    fn Entry(&self) -> squeue::Entry {
        panic!("doesn't support AsyncOpsTrait::SEntry")
    }
}