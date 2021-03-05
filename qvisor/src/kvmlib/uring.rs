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

use alloc::sync::Arc;
use spin::Mutex;
use core::ops::Deref;
use std::io::{IoSliceMut, IoSlice};
//use std::os::unix::io::AsRawFd;
use std::os::unix::io::RawFd;
use iou::*;
use std::slice;
use std::collections::HashMap;
use lazy_static::lazy_static;

use super::qlib::common::*;
use super::qlib::range::*;
use super::qlib::qmsg::*;
use super::qlib::linux_def::*;
use super::qlib::ShareSpace;
use super::util::*;
use super::vmspace::hostfdnotifier::*;
use super::vmspace::HostFileMap::fdinfo::*;

/*lazy_static! {
    pub static ref URING : Uring<'static> = Uring::New().unwrap();
}*/

pub const IOVS_COUNT : usize = 1024;
pub const URING_REQ_COUNT : usize = 1024;

#[derive(Clone)]
pub struct Uring <'a> (Arc<Mutex<UringIntern<'a>>>);

impl <'a> Deref for Uring <'a> {
    type Target = Arc<Mutex<UringIntern<'a>>>;

    fn deref(&self) -> &Arc<Mutex<UringIntern<'a>>> {
        &self.0
    }
}

impl <'a> Uring <'a> {
    pub fn New() -> Result<Self> {
        return Ok(Self(Arc::new(Mutex::new(UringIntern::New()?))))
    }

    pub fn Eventfd(&self) -> i32 {
        return self.lock().eventfd;
    }
}

impl <'a> HostFdHandler for Uring <'a> {
    fn Process(&self, shareSpace: &'static ShareSpace, _event: EventMask) {
        self.lock().Trigger(shareSpace).ok();
    }
}

pub struct UringIntern<'a> {
    pub ring: IoUring,
    pub eventfd: RawFd,
    pub freeReqCnt: usize,

    // an array of pre-allocated iovs, workaround for read/write operation.
    // when the Uring::Read/Uring::Write is ready, it could be deprecated.
    pub iovsMgrMut: GapMgr,
    pub iovsMut: Vec<IoSliceMut<'a>>,
    pub iovsMgr: GapMgr,
    pub iovs: Vec<IoSlice<'a>>,

    // to (callback, iovsIdx, Mutable)
    pub reqs: HashMap<u64, (Arc<UringCallback>, usize, bool)>,

    pub lastReqId: u64,
}

impl <'a> UringIntern<'a> {
    pub fn New() -> Result<Self> {
        let ret = unsafe {
            libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK)
        };

        let efd = GetRet(ret)?;

        // let ring = IoUring::new(URING_REQ_COUNT as u32).map_err(|e| Error::FromIOErr(e))?;

        let ring = IoUring::new_with_flags(32, SetupFlags::SQPOLL, SetupFeatures::empty())
            .map_err(|e| Error::FromIOErr(e))?;

        let registrar: Registrar = ring.registrar();
        registrar.register_eventfd(efd).map_err(|e| Error::FromIOErr(e))?;

        let mut iovs = Vec::with_capacity(IOVS_COUNT);
        for _ in 0..IOVS_COUNT {
            iovs.push(Self::DummyIoSlice())
        }

        let mut iovsMut = Vec::with_capacity(IOVS_COUNT);
        for _ in 0..IOVS_COUNT {
            iovsMut.push(Self::DummyIoSliceMut())
        }

        return Ok(Self {
            ring: ring,
            eventfd: efd,
            freeReqCnt: URING_REQ_COUNT,
            iovsMgrMut: GapMgr::New(0, IOVS_COUNT as u64),
            iovsMut: iovsMut,
            iovsMgr: GapMgr::New(0, IOVS_COUNT as u64),
            iovs: iovs,
            reqs: HashMap::new(),
            lastReqId: 0,
        })
    }

    pub fn DummyIoSliceMut() -> IoSliceMut <'a> {
        let mut x : u8 = 0;
        let ptr = &mut x as * mut _;
        let slice = unsafe { slice::from_raw_parts_mut(ptr, 1) };
        return IoSliceMut::new(slice);
    }

    pub fn DummyIoSlice() -> IoSlice <'a> {
        let mut x : u8 = 0;
        let ptr = &mut x as * mut _;
        let slice = unsafe { slice::from_raw_parts(ptr, 1) };
        return IoSlice::new(slice);
    }

    pub fn GetIovs(&mut self) -> Result<usize> {
        let ret = self.iovsMgr.Alloc(0, 0)?;
        return Ok(ret as usize)
    }

    pub fn FreeIovs(&mut self, idx: usize) {
        self.iovsMgr.Free(idx as u64, 1);
    }

    pub fn GetIovsMut(&mut self) -> Result<usize> {
        let ret = self.iovsMgrMut.Alloc(0, 0)?;
        return Ok(ret as usize)
    }

    pub fn FreeIovsMut(&mut self, idx: usize) {
        self.iovsMgrMut.Free(idx as u64, 1);
    }

    #[inline]
    pub fn AllocReqId(&mut self) -> Result<u64> {
        if self.freeReqCnt == 0 {
            return Err(Error::NoUringReq)
        }

        self.lastReqId += 1;
        return Ok(self.lastReqId);
    }

    pub fn AddReq(&mut self, reqIdx: u64, callback: Arc<UringCallback>, iovIdx: usize, mutable: bool) {
        self.reqs.insert(reqIdx, (callback, iovIdx, mutable));
    }

    pub fn GetReq(&mut self, reqIdx: u64) -> (Arc<UringCallback>, usize, bool) {
        return self.reqs.remove(&reqIdx).unwrap();
    }

    pub fn Trigger(&mut self, sp: &'static ShareSpace) -> Result<()> {
        loop {
            let mut v : u64 = 0;
            let ret = unsafe {
                libc::read(self.eventfd, &mut v as * mut _ as *mut libc::c_void, 8)
            };

            GetRet(ret as i32)?;

            loop {
                let mut cq = self.ring.cq();
                let cqe = match cq.peek_for_cqe() {
                    None => break,
                    Some(e) => e,
                };

                self.Process(sp, &cqe)?;
            }
        }
    }

    pub fn Process(&mut self, sp: &'static ShareSpace, cqe: &CQE) -> Result<()> {
        let reqId = cqe.user_data();
        let (req, iovIdx, mutable) = self.GetReq(reqId);

        if mutable {
            self.FreeIovsMut(iovIdx);
        } else {
            self.FreeIovs(iovIdx)
        }

        self.freeReqCnt += 1;

        return req.Callback(sp, cqe);
    }

    pub fn Read(&mut self, fd: RawFd, buf: &'a mut [u8], offset: u64) -> Result<()> {
        let reqId = self.AllocReqId()?;
        let iovsIdx = self.GetIovsMut()?;

        self.freeReqCnt -= 1;
        unsafe {
            let mut sq = self.ring.sq();
            let mut sqe = sq.prepare_sqe().unwrap();

            self.iovsMut[iovsIdx] = IoSliceMut::new(buf);

            sqe.prep_read_vectored(fd, &mut self.iovsMut[iovsIdx..iovsIdx+1], offset);
            sqe.set_user_data(reqId);
            sq.submit().map_err(|e| Error::FromIOErr(e))?;
        }

        return Ok(())
    }

    pub fn BufWrite(&mut self, msg: UringBufWrite) -> Result<()> {
        let reqId = self.AllocReqId()?;
        let iovsIdx = self.GetIovs()?;
        self.freeReqCnt -= 1;

        let fd = msg.fdInfo.lock().osfd;
        msg.fdInfo.lock().pendingWriteCnt += 1;

        let ptr = msg.addr as * mut u8;
        let buf = unsafe { slice::from_raw_parts(ptr, msg.len) };
        let offset = msg.offset;
        unsafe {
            let mut sq = self.ring.sq();
            let mut sqe = sq.prepare_sqe().unwrap();

            self.iovs[iovsIdx] = IoSlice::new(buf);

            sqe.prep_write_vectored(fd, &self.iovs[iovsIdx..iovsIdx+1], offset);
            sqe.set_user_data(reqId);
            sq.submit().map_err(|e| Error::FromIOErr(e))?;
        }

        self.AddReq(reqId, Arc::new(msg), iovsIdx, true);

        return Ok(())
    }
}

pub trait UringCallback : Send + Sync {
    fn Callback(&self, sp: &'static ShareSpace, cqe: &CQE) -> Result<()> ;
}

pub struct UringReadIntern {
    pub fdInfo: FdInfo,
    pub addr: u64,
    pub len: usize,
    pub offset: u64,
}

pub struct UringRead(Arc<UringReadIntern>);

impl Deref for UringRead {
    type Target = Arc<UringReadIntern>;

    fn deref(&self) -> &Arc<UringReadIntern> {
        &self.0
    }
}

pub struct UringBufWriteIntern {
    pub fdInfo: FdInfo,
    pub addr: u64,
    pub len: usize,
    pub offset: u64,
}

pub struct UringBufWrite(Arc<UringBufWriteIntern>);

impl Deref for UringBufWrite {
    type Target = Arc<UringBufWriteIntern>;

    fn deref(&self) -> &Arc<UringBufWriteIntern> {
        &self.0
    }
}

impl UringBufWrite {
    pub fn New(fdInfo: FdInfo, addr: u64, len: usize, offset: isize) -> Self {
        return Self(Arc::new(UringBufWriteIntern {
            fdInfo: fdInfo,
            addr: addr,
            len: len,
            offset: offset as u64,
        }))
    }
}

impl UringCallback for UringBufWrite {
    fn Callback(&self, shareSpace: &'static ShareSpace, cqe: &CQE) -> Result<()> {
        self.fdInfo.lock().pendingWriteCnt -= 1;

        let fd = self.fdInfo.lock().osfd;
        match cqe.result() {
            Ok(size) => {
                // assert!(size as usize == self.len, format!("size is {}, self.len is {}", size, self.len));
                if size as usize == self.len {
                    shareSpace.AQHostInputCall(HostInputMsg::IOBufWriteResp(IOBufWriteResp{
                        fd: fd,
                        addr: self.addr,
                        len: self.len,
                        ret: 0,
                    }));
                } else {
                    let msg = UringBufWrite::New(self.fdInfo.clone(),
                                                 self.addr + size as u64,
                                                 self.len - size as usize,
                                                 (self.offset + size as u64) as isize);
                    //todo: add back this
                    //URING.lock().BufWrite(msg)?;
                }
            }
            Err(e) => {
                shareSpace.AQHostInputCall(HostInputMsg::IOBufWriteResp(IOBufWriteResp{
                    fd: fd,
                    addr: self.addr,
                    len: self.len,
                    ret: e.raw_os_error().unwrap() as i64,
                }));
            }
        }

        return Ok(())
    }
}

