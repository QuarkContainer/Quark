// Copyright (c) 2021 QuarkSoft LLC
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

use core::mem;

use super::super::qlib::uring::util::*;
use super::super::qlib::uring::porting::*;
use super::super::qlib::uring::*;
use super::super::qlib::addr::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::util::*;
use super::super::*;
use super::syscall::*;

impl Mmap {
    pub fn new(fd: i32, offset: u64, len: usize) -> Result<Mmap> {
        let prot = (MmapProt::PROT_WRITE | MmapProt::PROT_READ ) as i32;

        let addr = PMA_KEEPER.lock().MapFile(len as u64, prot, fd, offset)?;
        VMS.lock().KernelMap(Addr(addr),
                             Addr(addr + len as u64).RoundUp().unwrap(),
                             Addr(addr),
                             PageOpts::Zero().SetPresent().SetWrite().Val())?;

        return Ok(Mmap {
            addr: addr,
            len: len,
        })
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

    fn SetupQueue(
        fd: i32,
        p: &sys::io_uring_params,
    ) -> Result<(MemoryMap, SubmissionQueue, CompletionQueue)> {
        let sq_len = p.sq_off.array as usize + p.sq_entries as usize * mem::size_of::<u32>();
        let cq_len = p.cq_off.cqes as usize
            + p.cq_entries as usize * mem::size_of::<sys::io_uring_cqe>();
        let sqe_len = p.sq_entries as usize * mem::size_of::<sys::io_uring_sqe>();
        let sqe_mmap = Mmap::new(fd, sys::IORING_OFF_SQES as _, sqe_len)?;

        let sq_mmap = Mmap::new(fd, sys::IORING_OFF_SQ_RING as _, sq_len)?;
        let cq_mmap = Mmap::new(fd, sys::IORING_OFF_CQ_RING as _, cq_len)?;

        let sq = unsafe {SubmissionQueue::new(&sq_mmap, &sqe_mmap, p)};
        let cq = unsafe {CompletionQueue::new(&cq_mmap, p)};
        let mm = MemoryMap {
            cq_mmap: Some(cq_mmap),
            sq_mmap,
            sqe_mmap,
        };

        Ok((mm, sq, cq))
    }

    fn with_params(entries: u32, mut p: sys::io_uring_params) -> Result<IoUring> {
        let fd: i32 = IOUringSetup(entries, &mut p as * mut _ as u64) as i32;

        if fd < 0 {
            return Err(Error::SysError(-fd))
        }

        let _hostfd = IO_MGR.lock().AddFd(fd, false);
        let (mm, sq, cq) = Self::SetupQueue(fd, &p)?;

        Ok(IoUring {
            fd: Fd(fd),
            sq,
            cq,
            params: Parameters(p),
            memory: mm,
        })
    }

    /// Initiate asynchronous I/O.
    #[inline]
    pub fn submit(&self) -> Result<usize> {
        self.submitter().submit()
    }

    /// Get submitter and submission queue and completion queue
    pub fn split(&mut self) -> (Submitter<'_>, &mut SubmissionQueue, &mut CompletionQueue) {
        let submit = Submitter::new(&self.fd, self.params.0.flags, &self.sq);
        (submit, &mut self.sq, &mut self.cq)
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
                    return Ok(0)
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
    pub unsafe fn enter(
        &self,
        to_submit: u32,
        min_complete: u32,
        flag: u32
    ) -> Result<usize> {
        let ret = IOUringEnter(self.fd.as_raw_fd(), to_submit, min_complete, flag);

        if ret < 0 {
            return Err(Error::SysError(-ret as i32))
        }

        return Ok(ret as usize)
    }
}

//pub const IORING_REGISTER_FILES: u32 = 2;
//pub const IORING_UNREGISTER_FILES: u32 = 3;
//pub const IORING_REGISTER_EVENTFD: u32 = 4;
pub const NR_IO_URING_SETUP: usize = 425;
pub const NR_IO_URING_ENTER: usize = 426;
pub const NR_IO_URING_REGISTER: usize = 427;

pub fn IOUringSetup(entries: u32, params: u64) -> i64 {
    let res = unsafe {
        syscall2(NR_IO_URING_SETUP, entries as usize, params as usize) as i64
    };
    if res < 0 {
        return SysRet(res)
    }

    return res
}

pub fn IOUringRegister(fd: i32, Opcode: u32, arg: u64, nrArgs: u32) -> i64 {
    let res = unsafe {
        syscall4(NR_IO_URING_REGISTER, fd as usize, Opcode as usize, arg as usize, nrArgs as usize) as i64
    };
    if res < 0 {
        return SysRet(res)
    }

    return res
}

pub fn IOUringEnter(fd: i32, toSubmit: u32, minComplete: u32, flags: u32) -> i64 {
    let res = unsafe {
        syscall6(NR_IO_URING_ENTER,
                 fd as usize,
                 toSubmit as usize,
                 minComplete as usize,
                 flags as usize,
                 0,
                 core::mem::size_of::<libc::sigset_t>() as usize
        ) as i64
    };
    if res < 0 {
        return SysRet(res)
    }

    return res
}