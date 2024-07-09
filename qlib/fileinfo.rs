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

use alloc::sync::Arc;
use core::fmt;
use core::ops::Deref;
use core::sync::atomic::AtomicI32;
use core::sync::atomic::Ordering;
use spin::Mutex;

use crate::qlib::common::*;
use crate::qlib::kernel::kernel::waiter::*;
use crate::qlib::kernel::IOURING;
use crate::qlib::rdmasocket::*;
use crate::qlib::*;

#[cfg(feature = "cc")]
use super::mem::cc_allocator::GuestHostSharedAllocator;

#[derive(Clone)]
pub enum SockInfo {
    File,                             // it is not socket
    Socket(SocketInfo),               // normal socket
    RDMAServerSocket(RDMAServerSock), //
    RDMADataSocket(RDMADataSock),     //
    RDMAContext,
    RDMAUDPSocket(RDMAUDPSock),
}

#[derive(Clone, Default)]
pub struct SocketInfo {
    pub ipAddr: u32,
    pub port: u16,
}

impl fmt::Debug for SockInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::File => write!(f, "SockInfo::File"),
            Self::Socket(_) => write!(f, "SockInfo::Socket"),
            Self::RDMAServerSocket(_) => write!(f, "SockInfo::RDMAServerSocket"),
            Self::RDMADataSocket(_) => write!(f, "SockInfo::RDMADataSocket"),
            Self::RDMAContext => write!(f, "SockInfo::RDMAContext"),
            Self::RDMAUDPSocket(_) => write!(f, "SockInfo::RDMAUDPSocket"),
        }
    }
}

impl SockInfo {
    pub fn Notify(&self, eventmask: EventMask, waitinfo: FdWaitInfo) {
                match self {
            Self::File => {
                waitinfo.Notify(eventmask);
            }
            Self::Socket(_) => {
                waitinfo.Notify(eventmask);
            }
            Self::RDMAServerSocket(ref sock) => sock.Notify(eventmask, waitinfo),
            Self::RDMADataSocket(ref sock) => sock.Notify(eventmask, waitinfo),
            Self::RDMAContext => {
                //RDMA.PollCompletion().expect("RDMA.PollCompletion fail");
            }
            Self::RDMAUDPSocket(ref sock) => sock.Notify(eventmask, waitinfo),
        }
    }
}

#[derive(Debug)]
pub struct FdInfoIntern {
    pub fd: i32,
    pub waitInfo: FdWaitInfo,

    pub flags: Flags,
    pub sockInfo: Mutex<SockInfo>,
}

impl FdInfoIntern {}

#[derive(Clone, Debug)]
pub struct FdInfo(pub Arc<Mutex<FdInfoIntern>>);

impl Deref for FdInfo {
    type Target = Arc<Mutex<FdInfoIntern>>;

    fn deref(&self) -> &Arc<Mutex<FdInfoIntern>> {
        &self.0
    }
}

impl FdInfo {
    pub fn UpdateWaitInfo(&self, waitInfo: FdWaitInfo) {
        self.lock().waitInfo = waitInfo
    }
}

#[derive(Debug, Clone)]
pub struct FdTbl {
    map: Vec<Option<FdInfo>>,
}

impl Default for FdTbl {
    fn default() -> Self {
        let mut map = Vec::with_capacity(32);
        map.resize(32, None);
        return Self { map: map };
    }
}

impl FdTbl {
    pub fn Get(&self, fd: i32) -> Option<FdInfo> {
        let fd = fd as usize;
        if fd > self.map.len() {
            return None;
        }
        return self.map[fd].clone();
    }

    pub fn Remove(&mut self, fd: i32) -> Option<FdInfo> {
        let fd = fd as usize;
        if fd > self.map.len() {
            return None;
        }
        return self.map[fd].take();
    }

    pub fn Contains(&self, fd: i32) -> bool {
        let fd = fd as usize;
        if fd > self.map.len() {
            return false;
        }
        return self.map[fd].is_some();
    }

    pub fn Insert(&mut self, fd: i32, fdInfo: FdInfo) {
        let fd = fd as usize;
        if fd >= self.map.len() {
            let count = (fd + 1).next_power_of_two();
            self.map.resize(count, None);
        }
        self.map[fd] = Some(fdInfo)
    }
}

#[derive(Default)]
pub struct IOMgr {
    pub fdTbl: Mutex<FdTbl>,
    pub eventfd: i32,
    pub epollfd: AtomicI32,
}

impl IOMgr {
    pub fn InitPollHostEpoll(&self, hostEpollWaitfd: i32) {
        self.epollfd.store(hostEpollWaitfd, Ordering::Relaxed);
        IOURING.PollHostEpollWaitInit(hostEpollWaitfd);
    }

    pub fn Epollfd(&self) -> i32 {
        return self.epollfd.load(Ordering::Relaxed);
    }

    pub fn GetByHost(&self, fd: i32) -> Option<FdInfo> {
        match self.fdTbl.lock().Get(fd) {
            None => None,
            Some(fdInfo) => Some(fdInfo.clone()),
        }
    }
}

#[derive(Default)]
pub struct FdWaitIntern {
    pub queue: Queue,
    pub mask: EventMask,
}

impl fmt::Debug for FdWaitIntern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FdWaitInfo")
            .field("mask", &self.mask)
            .finish()
    }
}

#[cfg(not(feature = "cc"))]
#[derive(Default, Clone, Debug)]
pub struct FdWaitInfo(Arc<QMutex<FdWaitIntern>>);

#[cfg(feature = "cc")]
#[derive(Clone, Debug)]
pub struct FdWaitInfo(Arc<QMutex<FdWaitIntern>, GuestHostSharedAllocator>);

#[cfg(feature = "cc")]
impl Default for FdWaitInfo {
    fn default() -> Self {
        return FdWaitInfo(Arc::new_in(
            QMutex::new(FdWaitIntern::default()),
            crate::GUEST_HOST_SHARED_ALLOCATOR,
        ));
    }
}


#[cfg(not(feature = "cc"))]
impl Deref for FdWaitInfo {
    type Target = Arc<QMutex<FdWaitIntern>>;

    fn deref(&self) -> &Arc<QMutex<FdWaitIntern>> {
        &self.0
    }
}

#[cfg(feature = "cc")]
impl Deref for FdWaitInfo {
    type Target = Arc<QMutex<FdWaitIntern>, GuestHostSharedAllocator>;

    fn deref(&self) -> &Arc<QMutex<FdWaitIntern>, GuestHostSharedAllocator> {
        &self.0
    }
}

impl FdWaitInfo {
    #[cfg(not(feature = "cc"))]
    pub fn New(queue: Queue, mask: EventMask) -> Self {
        let intern = FdWaitIntern { queue, mask };

        return Self(Arc::new(QMutex::new(intern)));
    }

    #[cfg(feature = "cc")]
    pub fn New(queue: Queue, mask: EventMask) -> Self {
        let intern = FdWaitIntern { queue, mask };

        return Self(Arc::new_in(
            QMutex::new(intern),
            crate::GUEST_HOST_SHARED_ALLOCATOR,
        ));
    }

    pub fn UpdateFDAsync(&self, fd: i32, epollfd: i32) -> Result<()> {
                let op;
        let mask = {
            let mut fi = self.lock();

            let mask = fi.queue.Events();

            if fi.mask == 0 {
                if mask != 0 {
                    op = LibcConst::EPOLL_CTL_ADD;
                } else {
                    return Ok(());
                }
            } else {
                if mask == 0 {
                    op = LibcConst::EPOLL_CTL_DEL;
                } else {
                    if mask | fi.mask == fi.mask {
                        return Ok(());
                    }
                    op = LibcConst::EPOLL_CTL_MOD;
                }
            }

            fi.mask = mask;

            mask
        };

        let mask = mask | LibcConst::EPOLLET as u64;

        IOURING.EpollCtl(epollfd, fd, op as i32, mask as u32);
        return Ok(());
    }

    pub fn Notify(&self, mask: EventMask) {
        let queue = self.lock().queue.clone();
        queue.Notify(EventMaskFromLinux(mask as u32));
    }
}
