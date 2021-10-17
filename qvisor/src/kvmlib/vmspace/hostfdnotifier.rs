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

use std::collections::HashMap;
use alloc::boxed::Box;
use spin::RwLock;
use libc::*;
use core::ops::Deref;

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::ShareSpace;
use super::super::VMS;

pub trait HostFdHandler: Send + Sync {
    fn Process(&self, shareSpace: &'static ShareSpace, event: EventMask);
}

pub struct EventHandler {}

impl HostFdHandler for EventHandler {
    fn Process(&self, _shareSpace: &'static ShareSpace, _event: EventMask) {}
}

pub struct GuestFd {
    pub hostfd: i32
}

impl HostFdHandler for GuestFd {
    fn Process(&self, _shareSpace: &'static ShareSpace, event: EventMask) {
        VMS.lock().FdNotify(self.hostfd, event);
    }
}

//#[derive(Clone)]
pub struct HostFdInfo {
    pub waiting: bool,
    pub handler: Box<HostFdHandler>,
}

pub struct FdNotifierInternal {
    //main epoll fd
    pub epollfd: i32,

    //eventfd which guest notify host for message
    pub eventfd: i32,

    pub fdMap: HashMap<i32, HostFdInfo>,
}

impl FdNotifierInternal {
    pub fn WaitFd(&mut self, fd: i32, mask: EventMask) -> Result<()> {
        let n = self;

        let epollfd = n.epollfd;
        let mut fi = match n.fdMap.get_mut(&fd) {
            None => {
                // panic!("HostFdNotifier::WaitFd can't find fd {}", fd)
                return Ok(())
            }
            Some(fi) => fi,
        };

        if !fi.waiting && mask == 0 {
            return Ok(())
        }

        let mut ev = epoll_event {
            events: mask as u32 | EPOLLET as u32,
            u64: fd as u64
        };

        if !fi.waiting && mask != 0 {
            let ret = unsafe {
                epoll_ctl(epollfd, EPOLL_CTL_ADD, fd, &mut ev as *mut epoll_event)
            };

            if ret == -1 {
                return Err(Error::SysError(errno::errno().0))
            }

            fi.waiting = true;
        } else if fi.waiting {
            if mask == 0 {
                let ret = unsafe {
                    epoll_ctl(epollfd, EPOLL_CTL_DEL, fd, &mut ev as *mut epoll_event)
                };

                if ret == -1 {
                    return Err(Error::SysError(errno::errno().0))
                }

                fi.waiting = false;
            } else {
                let ret = unsafe {
                    epoll_ctl(epollfd, EPOLL_CTL_MOD, fd, &mut ev as *mut epoll_event)
                };

                if ret == -1 {
                    return Err(Error::SysError(errno::errno().0))
                }
            }
        }

        return Ok(())
    }

    pub fn AddFd<T: HostFdHandler + 'static>(&mut self, fd: i32, handler: Box<T>) {
        let n = self;

        if n.fdMap.contains_key(&fd) {
            panic!("HostFdNotifier::AddFd file descriptor {} added twice", fd);
        }

        n.fdMap.insert(fd, HostFdInfo {
            waiting: false,
            handler: handler,
        });
    }
}

pub struct HostFdNotifier(RwLock<FdNotifierInternal>);

impl Deref for HostFdNotifier {
    type Target = RwLock<FdNotifierInternal>;

    fn deref(&self) -> &RwLock<FdNotifierInternal> {
        &self.0
    }
}

impl HostFdNotifier {
    pub fn New() -> Self {
        let epfd = unsafe {
            epoll_create1(0)
        };

        if epfd == -1 {
            panic!("FdNotifier::New create epollfd fail, error is {}", errno::errno().0);
        }

        let eventfd = unsafe {
            //eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)
            eventfd(0, EFD_CLOEXEC)
        };

        if eventfd == -1 {
            panic!("FdNotifier::New create eventfd fail, error is {}", errno::errno().0);
        }

        let mut internal = FdNotifierInternal {
            epollfd: epfd,
            eventfd: eventfd,
            fdMap: HashMap::new(),
        };

        internal.AddFd(eventfd, Box::new(EventHandler {}));
        internal.WaitFd(eventfd, (EPOLLIN | EPOLLOUT | EPOLLHUP | EPOLLPRI | EPOLLERR | EPOLLHUP | EPOLLET) as EventMask).unwrap();

        return Self(RwLock::new(internal))
    }

    pub fn Eventfd(&self) -> i32 {
        return self.read().eventfd;
    }

    pub fn Notify(&self)  {
        let data: u64 = 1;
        let ret = unsafe {
            write(self.read().eventfd, &data as *const _ as *const c_void, 8)
        };

        if ret == -1 {
            let errno = errno::errno().0;
            error!("hostfdnotifier Trigger fail to write data to the eventfd, errno is {}", errno);
        }
    }

    pub fn WaitFd(&self, fd: i32, mask: EventMask) -> Result<()> {
        let mut n = self.write();

        return n.WaitFd(fd, mask);
    }

    pub fn AddFd<T: HostFdHandler + 'static>(&self, fd: i32, handler: Box<T>) {
        let mut n = self.0.write();

        n.AddFd(fd, handler);
    }

    pub fn RemoveFd(&self, fd: i32) -> Result<()> {
        let mut n = self.0.write();

        n.WaitFd(fd, 0)?;
        n.fdMap.remove(&fd);

        return Ok(())
    }

    pub const MAX_EVENTS: usize = 128;
    pub fn WaitAndNotify(&self, shareSpace: &'static ShareSpace, timeout: i32) -> Result<i32> {
        /*let mut data : u64 = 0;
        let eventfd = self.read().eventfd;
        let ret = unsafe {
            libc::read(eventfd, &mut data as * mut _ as *mut libc::c_void, 8)
        };

        if ret < 0 {
            panic!("KIOThread::Wakeup fail... eventfd is {}, errno is {}",
                   eventfd, errno::errno().0);
        }

        return Ok(0)*/


        let mut events = [epoll_event { events: 0, u64: 0 }; Self::MAX_EVENTS];

        let epollfd = self.read().epollfd;

        let waitTime = if timeout == -1 { // blocked
            //shareSpace.WaitInHost();
            if shareSpace.ReadyAsyncMsgCnt() > 0 {
                0
            } else {
                -1
            }
            //10
        } else {
            timeout
        };

        //todo: when there is multiple iothread, handle that
        let nfds = unsafe {
            epoll_wait(epollfd, &mut events[0], (events.len() - 1) as i32, waitTime)
        };

        if timeout == -1 {
            //shareSpace.WakeInHost();
        }

        if nfds == -1 {
            let err = errno::errno().0;
            if err == SysErr::EINTR {
                return Ok(0);
            }

            return Err(Error::SysError(err))
        }

        for i in 0..nfds as usize {
            let n = self.read();
            let fd = events[i].u64 as i32;
            let fi = n.fdMap.get(&fd).expect("WaitAndNotify get none");
            fi.handler.Process(shareSpace, events[i].events as EventMask);
        }

        return Ok(nfds)
    }
}