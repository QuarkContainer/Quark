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
use lazy_static::lazy_static;
use core::ops::Deref;
use std::time::SystemTime;
use spin::Mutex;

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;

lazy_static! {
    pub static ref VCPU_WAIT: VcpuBitmap = VcpuBitmap::default();
}

pub struct VcpuBitmapIntern {
    pub bitmap: u64,
    pub waiters: [Option<VcpuWaiter>; 64],
}

pub struct VcpuBitmap(Mutex<VcpuBitmapIntern>);

impl Deref for VcpuBitmap {
    type Target = Mutex<VcpuBitmapIntern>;

    fn deref(&self) -> &Mutex<VcpuBitmapIntern> {
        &self.0
    }
}

impl Default for VcpuBitmap {
    fn default() -> Self {
        const INIT: Option<VcpuWaiter> = None;
        let intern = VcpuBitmapIntern {
            bitmap: 0,
            waiters: [INIT; 64]
        };

        return Self(Mutex::new(intern))
    }
}

impl VcpuBitmap {
    pub fn NewWaiter(&self, bitmap: u64) -> VcpuWaiter {
        let mut intern = self.lock();
        assert!(intern.bitmap & bitmap == 0);
        intern.bitmap |= bitmap;
        let waiter = VcpuWaiter::New(bitmap);

        for i in 0..64 {
            if bitmap & 1<<i != 0 {
                intern.waiters[i] = Some(waiter.clone());
            }
        }

        return waiter
    }

    //ret: whether the target is waked
    pub fn Wakeup(&self, vcpId: usize) -> bool {
        assert!(vcpId < 64);
        let mut intern = self.lock();

        intern.bitmap &= !(1<<vcpId);
        match intern.waiters[vcpId].take() {
            None => return false,
            Some(w) => {
                // there is more vcpu needs to wait
                if intern.bitmap & w.bitmap != 0 {
                    return false
                }

                w.Wakeup();
                return true;
            }
        }
    }

    pub fn Clear(&self, waiter: &VcpuWaiter) {
        let mut intern = self.lock();
        let bitmap = intern.bitmap & waiter.bitmap;
        intern.bitmap &= !waiter.bitmap;

        if bitmap == 0 {
            return
        }

        for i in 0..64 {
            if bitmap & 1<<i != 0 {
                intern.waiters[i] = None;
            }
        }
    }
}

#[derive(Clone)]
pub struct VcpuWaiter(Arc<VcpuWaiterIntern>);

pub struct VcpuWaiterIntern {
    pub bitmap: u64,
    pub eventfd: i32,
}

impl Deref for VcpuWaiter {
    type Target = Arc<VcpuWaiterIntern>;

    fn deref(&self) -> &Arc<VcpuWaiterIntern> {
        &self.0
    }
}

impl Drop for VcpuWaiterIntern {
    fn drop(&mut self) {
        error!("VcpuWaiter drop {}", self.eventfd);

        unsafe {
            libc::close(self.eventfd);
        }
    }
}

impl VcpuWaiter {
    pub fn New(bitmap: u64) -> Self {
        let eventfd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) };

        if eventfd < 0 {
            panic!("Vcpu::Init fail...");
        }

        let intern = VcpuWaiterIntern {
            bitmap: bitmap,
            eventfd: eventfd,
        };

        return Self(Arc::new(intern))
    }

    pub fn Clear(&self) {
        VCPU_WAIT.Clear(&self);
    }

    pub fn Wakeup(&self) {
        let val: u64 = 8;
        let ret = unsafe { libc::write(self.eventfd, &val as *const _ as *const libc::c_void, 8) };
        if ret < 0 {
            panic!("VcpuWaiter::Wakeup fail...");
        }
    }

    pub fn Wait(&self, timeout: i32) -> Result<()> {
        let mut pollfd = libc::pollfd {
            fd: self.eventfd,
            events: libc::POLLIN,
            revents: 0,
        };

        let start = SystemTime::now();

        loop {
            if VCPU_WAIT.lock().bitmap & self.bitmap == 0 {
                return Ok(())
            }

            let delta = timeout - start.elapsed().unwrap().as_millis() as i32;

            if delta <= 0 {
                return Err(Error::SysError(SysErr::ETIME))
            }

            unsafe {
                libc::poll(&mut pollfd, 1, delta);
            }

            {
                let mut data: u64 = 0;
                let ret = unsafe {
                    libc::read(self.eventfd, &mut data as *mut _ as *mut libc::c_void, 8)
                };

                if ret < 0 && errno::errno().0 != SysErr::EINTR {
                    panic!(
                        "VcpuWaiter::wait fail... eventfd is {}, errno is {}",
                        self.eventfd,
                        errno::errno().0
                    );
                }
            }
        }
    }
}