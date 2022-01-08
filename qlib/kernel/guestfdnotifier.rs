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

use alloc::collections::btree_map::BTreeMap;
use ::qlib::mutex::*;
use core::ops::Deref;

use super::Kernel::HostSpace;
use super::kernel::waiter::*;
use super::fs::host::hostinodeop::*;
use super::qlib::common::*;
use super::qlib::linux_def::*;
use super::qlib::singleton::*;
use super::SHARESPACE;
use super::IOURING;

pub static GUEST_NOTIFIER : Singleton<Notifier> = Singleton::<Notifier>::New();

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

pub fn HostLogFlush() {
    //GUEST_NOTIFIER.PrintStrRespHandler(addr, len)
    super::IOURING.LogFlush();
}

pub struct GuestFdInfo {
    pub queue: Queue,
    pub mask: EventMask,
    pub waiting: bool,
}

// notifier holds all the state necessary to issue notifications when IO events
// occur in the observed FDs.
pub struct NotifierInternal {
    // fdMap maps file descriptors to their notification queues and waiting
    // status.
    fdMap: BTreeMap<i32, GuestFdInfo>,
    pub epollfd: i32,
}

#[repr(C)]
#[repr(packed)]
#[derive(Default, Copy, Clone, Debug)]
pub struct EpollEvent {
    pub Event: u32,
    pub U64: u64
}

pub struct Notifier(QMutex<NotifierInternal>);

impl Deref for Notifier {
    type Target = QMutex<NotifierInternal>;

    fn deref(&self) -> &QMutex<NotifierInternal> {
        &self.0
    }
}

impl Notifier {
    pub fn New() -> Self {
        let internal = NotifierInternal {
            fdMap: BTreeMap::new(),
            epollfd: 0,
        };

        return Self(QMutex::new(internal))
    }

    pub fn VcpuWait(&self) {
        let ret = HostSpace::VcpuWait();
        if ret < 0 {
            panic!("ProcessHostEpollWait fail with error {}", ret)
        };
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

    pub fn UpdateFDAsync(&self, fd: i32) -> Result<()> {
        let op;
        let epollfd;
        let mask = {
            let mut n = self.lock();
            epollfd = n.epollfd;
            let fi = match n.fdMap.get_mut(&fd) {
                None => {
                    return Ok(())
                }
                Some(fi) => fi,
            };

            let mask = fi.queue.Events() | LibcConst::EPOLLET as u64;

            if !fi.waiting {
                if mask != 0 {
                    op = LibcConst::EPOLL_CTL_ADD;
                    fi.waiting = true;
                } else {
                    return Ok(())
                }
            } else {
                if mask == 0 {
                    op = LibcConst::EPOLL_CTL_DEL;
                    fi.waiting = false;
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

        IOURING.EpollCtl(epollfd, fd, op as i32, mask as u32);
        return Ok(())
    }

    pub fn UpdateFDSync(&self, fd: i32) -> Result<()> {
        let op = LibcConst::EPOLL_CTL_ADD as u32/*dummy value*/;
        let mask = {
            let mut n = self.lock();
            let fi = match n.fdMap.get_mut(&fd) {
                None => {
                    return Ok(())
                }
                Some(fi) => fi,
            };

            let mask = fi.queue.Events();
            mask
        };

        return Self::waitfd(fd, op, mask);
    }

    pub fn UpdateFDSync1(&self, fd: i32) -> Result<()> {
        let op;
        let mask = {
            let mut n = self.lock();
            let fi = match n.fdMap.get_mut(&fd) {
                None => {
                    return Ok(())
                }
                Some(fi) => fi,
            };

            let mask = fi.queue.Events() | LibcConst::EPOLLET as u64;

            if !fi.waiting {
                if mask != 0 {
                    op = LibcConst::EPOLL_CTL_ADD;
                    fi.waiting = true;
                } else {
                    return Ok(())
                }
            } else {
                if mask == 0 {
                    op = LibcConst::EPOLL_CTL_DEL;
                    fi.waiting = false;
                } else {
                    if mask | fi.mask == fi.mask {
                        return Ok(())
                    }
                    op = LibcConst::EPOLL_CTL_MOD;
                }
            }

            mask
        };

        return Self::waitfd(fd, op as u32, mask);
    }

    pub fn AddFD(&self, fd: i32, iops: &HostInodeOp) {
        let mut n = self.lock();

        let queue = iops.lock().queue.clone();

        if n.fdMap.contains_key(&fd) {
            panic!("GUEST_NOTIFIER::AddFD fd {} added twice", fd);
        }

        n.fdMap.insert(fd, GuestFdInfo {
            queue: queue.clone(),
            mask: 0,
            waiting: false,
        });
    }

    pub fn RemoveFD(&self, fd: i32) {
        let mut n = self.lock();
        n.fdMap.remove(&fd);
    }

    pub fn Notify(&self, fd: i32, mask: EventMask) {
        let queue = {
            let n = self.lock();
            match n.fdMap.get(&fd) {
                None => {
                    return
                },
                Some(fi) => {
                    fi.queue.clone()
                }
            }
        };

        queue.Notify(EventMaskFromLinux(mask as u32));
    }
}