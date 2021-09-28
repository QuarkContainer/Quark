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
use spin::Mutex;
use core::ops::Deref;

use super::Kernel::HostSpace;
use super::kernel::waiter::*;
use super::fs::host::hostinodeop::*;
use super::qlib::common::*;
use super::qlib::linux_def::*;
use super::qlib::singleton::*;

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

pub fn IOBufWriteRespHandle(fd: i32, addr: u64, len: usize, ret: i64) {
    GUEST_NOTIFIER.IOBufWriteRespHandle(fd, addr, len, ret)
}

pub fn HostLogFlush() {
    //GUEST_NOTIFIER.PrintStrRespHandler(addr, len)
    super::IOURING.LogFlush();
}

pub struct GuestFdInfo {
    pub queue: Queue,
    pub mask: EventMask,
    pub waiting: bool,
    pub iops: HostInodeOpWeak,
}

// notifier holds all the state necessary to issue notifications when IO events
// occur in the observed FDs.
pub struct NotifierInternal {
    // fdMap maps file descriptors to their notification queues and waiting
    // status.
    fdMap: BTreeMap<i32, GuestFdInfo>,
}

pub struct Notifier(Mutex<NotifierInternal>);

impl Deref for Notifier {
    type Target = Mutex<NotifierInternal>;

    fn deref(&self) -> &Mutex<NotifierInternal> {
        &self.0
    }
}

impl Notifier {
    pub fn New() -> Self {
        let internal = NotifierInternal {
            fdMap: BTreeMap::new()
        };

        return Self(Mutex::new(internal))
    }

    fn waitfd(fd: i32, mask: EventMask) -> Result<()> {
        HostSpace::WaitFD(fd, mask);

        return Ok(())
    }

    pub fn UpdateFD(&self, fd: i32) -> Result<()> {
        let mask = {
            let mut n = self.lock();
            let fi = match n.fdMap.get_mut(&fd) {
                None => {
                    return Ok(())
                }
                Some(fi) => fi,
            };

            let mask = fi.queue.Events();

            if !fi.waiting && mask == 0 {
                return Ok(())
            }

            if !fi.waiting && mask != 0 {
                fi.waiting = true;
            } else if fi.waiting && mask == 0 {
                fi.waiting = false;
            } else if fi.waiting && mask != 0 {
                if mask | fi.mask == fi.mask {
                    return Ok(())
                }
            }

            mask
        };

        return Self::waitfd(fd, mask);
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
            iops: iops.Downgrade(),
        });
    }

    pub fn RemoveFD(&self, fd: i32) {
        let mut n = self.lock();
        n.fdMap.remove(&fd);
    }

    pub fn Notify(&self, fd: i32, mask: EventMask) {
        let n = self.lock();
        match n.fdMap.get(&fd) {
            None => (),
            Some(fi) => {
                fi.queue.Notify(EventMaskFromLinux(mask as u32));
            }
        }
    }

    pub fn IOBufWriteRespHandle(&self, _fd: i32, _addr: u64, _len: usize, _ret: i64) {
        /*BUF_MGR.Free(addr, len as u64);
        if ret < 0 {
            let n = self.lock();
            match n.fdMap.get(&fd) {
                None => (),
                Some(fi) => {
                    let iops = match fi.iops.Upgrade() {
                        None => {
                            return;
                        }
                        Some(iops) => iops
                    };

                    iops.lock().errorcode = ret;
                }
            }
        }*/
    }
}