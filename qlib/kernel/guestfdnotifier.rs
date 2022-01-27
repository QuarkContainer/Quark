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
use alloc::collections::btree_map::BTreeMap;
use crate::qlib::mutex::*;
use core::ops::Deref;
use core::fmt;

use super::Kernel::HostSpace;
use super::kernel::waiter::*;
use super::fs::host::hostinodeop::*;
use super::super::object_ref::*;
use super::super::common::*;
use super::super::linux_def::*;
use super::SHARESPACE;
use super::IOURING;

pub static GUEST_NOTIFIER : GuestNotifierRef = GuestNotifierRef::New();

pub fn AddFD(fd: i32, iops: &HostInodeOp) {
    GUEST_NOTIFIER.AddFD(fd, iops);
}

pub fn RemoveFD(fd: i32) {
    GUEST_NOTIFIER.RemoveFD(fd);
}

pub fn UpdateFD(fd: i32) -> Result<()> {
    return GUEST_NOTIFIER.UpdateFD(fd);
}

pub fn NonBlockingPoll(fd: i32, mask: EventMask) -> EventMask {
    return HostSpace::NonBlockingPoll(fd, mask) as EventMask
}

pub fn Notify(fd: i32, mask: EventMask) {
    GUEST_NOTIFIER.Notify(fd, mask);
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

#[derive(Default, Clone, Debug)]
pub struct FdWaitInfo(Arc<QMutex<FdWaitIntern>>);

impl Deref for FdWaitInfo {
    type Target = Arc<QMutex<FdWaitIntern>>;

    fn deref(&self) -> &Arc<QMutex<FdWaitIntern>> {
        &self.0
    }
}

impl FdWaitInfo {
    pub fn New(queue: Queue, mask: EventMask) -> Self {
        let intern = FdWaitIntern {
            queue,
            mask
        };

        return Self(Arc::new(QMutex::new(intern)))
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
                    return Ok(())
                }
            } else {
                if mask == 0 {
                    op = LibcConst::EPOLL_CTL_DEL;
                } else {
                    if mask | fi.mask == fi.mask {
                        return Ok(())
                    }
                    op = LibcConst::EPOLL_CTL_MOD;
                }
            }

            fi.mask = mask;

            mask
        };

        let mask = mask | LibcConst::EPOLLET as u64;

        IOURING.EpollCtl(epollfd, fd, op as i32, mask as u32);
        return Ok(())
    }

    pub fn UpdateFDSync(&self, fd: i32) -> Result<()> {
        // let op;
        // let mask = {
        //     let mut fi = self.lock();

        //     let mask = fi.queue.Events();

        //     if fi.mask == 0 {
        //         if mask != 0 {
        //             op = LibcConst::EPOLL_CTL_ADD;
        //         } else {
        //             return Ok(())
        //         }
        //     } else {
        //         if mask == 0 {
        //             op = LibcConst::EPOLL_CTL_DEL;
        //         } else {
        //             if mask | fi.mask == fi.mask {
        //                 return Ok(())
        //             }
        //             op = LibcConst::EPOLL_CTL_MOD;
        //         }
        //     }

        //     fi.mask = mask;

        //     mask
        // };
        // error!("WaitFd1: fd: {}, op: {}, mask: {:x} ", fd, op, mask);
        // return Self::waitfd(fd, op as u32, mask);
        let mask = self.lock().queue.Events();
        return Self::waitfd(fd, 0 as u32, mask);
    }

    pub fn Notify(&self, mask: EventMask) {
        let queue = self.lock().queue.clone();
        queue.Notify(EventMaskFromLinux(mask as u32));
    }

    fn waitfd(fd: i32, op: u32, mask: EventMask) -> Result<()> {
        HostSpace::WaitFDAsync(fd, op, mask);

        return Ok(())
    }
}

// notifier holds all the state necessary to issue notifications when IO events
// occur in the observed FDs.
pub struct GuestNotifierInternal {
    // fdMap maps file descriptors to their notification queues and waiting
    // status.
    fdMap: BTreeMap<i32, FdWaitInfo>,
    pub epollfd: i32,
}

#[repr(C)]
#[repr(packed)]
#[derive(Default, Copy, Clone, Debug)]
pub struct EpollEvent {
    pub Event: u32,
    pub U64: u64
}

pub type GuestNotifierRef = ObjectRef<GuestNotifier>;
pub struct GuestNotifier(QMutex<GuestNotifierInternal>);

impl Deref for GuestNotifier {
    type Target = QMutex<GuestNotifierInternal>;

    fn deref(&self) -> &QMutex<GuestNotifierInternal> {
        &self.0
    }
}

impl Default for GuestNotifier {
    fn default() -> Self {
        return Self::New()
    }
}

impl GuestNotifier {
    pub fn New() -> Self {
        let internal = GuestNotifierInternal {
            fdMap: BTreeMap::new(),
            epollfd: 0,
        };

        return Self(QMutex::new(internal))
    }

    pub fn Addr(&self) -> u64 {
        return self as * const _ as u64
    }

    pub fn VcpuWait(&self) -> u64 {
        let ret = HostSpace::VcpuWait();
        if ret < 0 {
            panic!("ProcessHostEpollWait fail with error {}", ret)
        };

        return ret as u64
    }

    pub fn ProcessHostEpollWait(&self) {
        let ret = HostSpace::HostEpollWaitProcess();
        if ret < 0 {
            panic!("ProcessHostEpollWait fail with error {}", ret)
        };
    }

    pub fn ProcessEvents(&self, events: &[EpollEvent]) {
        for e in events {
            let fd = e.U64 as i32;
            let event = e.Event as EventMask;
            self.Notify(fd, event)
        }
    }

    pub fn InitPollHostEpoll(&self, hostEpollWaitfd: i32) {
        self.lock().epollfd = hostEpollWaitfd;
        IOURING.PollHostEpollWaitInit(hostEpollWaitfd);
    }

    fn waitfd(fd: i32, op: u32, mask: EventMask) -> Result<()> {
        HostSpace::WaitFDAsync(fd, op, mask);

        return Ok(())
    }

    pub fn UpdateFD(&self, fd: i32) -> Result<()> {
        if SHARESPACE.config.read().UringEpollCtl {
            return self.UpdateFDAsync(fd)
        } else {
            return self.UpdateFDSync(fd)
        }
    }

    pub fn FdWaitInfo(&self, fd: i32) -> Option<FdWaitInfo> {
        let fi = match self.lock().fdMap.get(&fd) {
            None => {
                return None
            }
            Some(fi) => fi.clone(),
        };

        return Some(fi)
    }

    pub fn UpdateFDAsync(&self, fd: i32) -> Result<()> {
        let fi = match self.FdWaitInfo(fd) {
            None => return Ok(()),
            Some(fi) => fi
        };

        let epollfd = self.lock().epollfd;

        return fi.UpdateFDAsync(fd, epollfd);
    }

    pub fn UpdateFDSync(&self, fd: i32) -> Result<()> {
        let fi = match self.FdWaitInfo(fd) {
            None => return Ok(()),
            Some(fi) => fi
        };

        return fi.UpdateFDSync(fd);
    }

    pub fn AddFD(&self, fd: i32, iops: &HostInodeOp) {
        let mut n = self.lock();

        let queue = iops.lock().queue.clone();

        if n.fdMap.contains_key(&fd) {
            panic!("GUEST_NOTIFIER::AddFD fd {} added twice", fd);
        }

        let waitinfo = FdWaitInfo::New(queue.clone(), 0);
        n.fdMap.insert(fd, waitinfo.clone());
        HostSpace::UpdateWaitInfo(fd, waitinfo);
    }

    pub fn RemoveFD(&self, fd: i32) {
        let mut n = self.lock();
        n.fdMap.remove(&fd);
    }

    pub fn Notify(&self, fd: i32, mask: EventMask) {
        let fi = match self.FdWaitInfo(fd) {
            None => return,
            Some(fi) => fi
        };

        fi.Notify(mask);
    }
}