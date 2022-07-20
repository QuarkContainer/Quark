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

use libc::*;
use core::sync::atomic::Ordering;

use super::super::kvm_vcpu::*;
use super::super::qlib::common::*;
use super::super::qlib::kernel::kernel::timer::TIMER_STORE;
use super::super::qlib::kernel::GlobalRDMASvcCli;
use super::super::qlib::kernel::ASYNC_PROCESS;
use super::super::qlib::kernel::IOURING;
use super::super::qlib::kernel::TSC;
use super::super::qlib::linux_def::*;
use super::super::qlib::ShareSpace;
use super::super::runc::runtime::vm::*;
use super::super::*;

pub struct KIOThread {
    pub eventfd: i32,
}

pub const IO_WAIT_CYCLES: i64 = 100_000_000; // 1ms

impl KIOThread {
    pub fn New() -> Self {
        return Self { eventfd: 0 };
    }

    pub fn Init(&self, eventfd: i32) {
        unsafe {
            *(&self.eventfd as *const _ as u64 as *mut i32) = eventfd;
        }
    }

    pub fn ProcessOnce(sharespace: &ShareSpace) -> usize {
        let mut count = 0;

        if QUARK_CONFIG.lock().EnableRDMA {
            count += GlobalRDMASvcCli().ProcessRDMASvcMessage();
        }
        count += IOURING.IOUring().HostSubmit().unwrap();
        TIMER_STORE.Trigger();
        count += IOURING.IOUring().HostSubmit().unwrap();
        count += IOURING.DrainCompletionQueue();
        count += IOURING.IOUring().HostSubmit().unwrap();
        count += KVMVcpu::GuestMsgProcess(sharespace);
        count += IOURING.IOUring().HostSubmit().unwrap();
        count += FD_NOTIFIER.HostEpollWait() as usize;
        count += IOURING.IOUring().HostSubmit().unwrap();

        sharespace.CheckVcpuTimeout();

        return count;
    }

    pub fn Process(sharespace: &ShareSpace) {
        let mut start = TSC.Rdtsc();

        while IsRunning() {
            let count = Self::ProcessOnce(sharespace);
            if count > 0 {
                start = TSC.Rdtsc()
            }

            if TSC.Rdtsc() - start >= IO_WAIT_CYCLES {
                break;
            }
        }
    }

    pub fn Wait(&self, sharespace: &ShareSpace) -> Result<()> {
        let epfd = unsafe { epoll_create1(0) };

        if epfd == -1 {
            panic!(
                "CPULocal::Init {} create epollfd fail, error is {}",
                0,
                errno::errno().0
            );
        }

        let mut ev = epoll_event {
            events: EVENT_READ as u32 | EPOLLET as u32,
            u64: self.eventfd as u64,
        };

        super::VMSpace::UnblockFd(self.eventfd);

        let ret = unsafe {
            epoll_ctl(
                epfd,
                EPOLL_CTL_ADD,
                self.eventfd,
                &mut ev as *mut epoll_event,
            )
        };

        if ret == -1 {
            panic!(
                "CPULocal::Init {} add eventfd fail, error is {}",
                0,
                errno::errno().0
            );
        }

        let mut ev = epoll_event {
            events: EVENT_READ as u32 | EPOLLET as u32,
            u64: FD_NOTIFIER.Epollfd() as u64,
        };

        let ret = unsafe {
            epoll_ctl(
                epfd,
                EPOLL_CTL_ADD,
                FD_NOTIFIER.Epollfd(),
                &mut ev as *mut epoll_event,
            )
        };

        if ret == -1 {
            panic!(
                "CPULocal::Init {} add host epollfd fail, error is {}",
                0,
                errno::errno().0
            );
        }

        if QUARK_CONFIG.lock().EnableRDMA {
            let mut ev = epoll_event {
                events: EVENT_READ as u32 | EPOLLET as u32,
                u64: GlobalRDMASvcCli().cliEventFd as u64,
            };

            let ret = unsafe {
                epoll_ctl(
                    epfd,
                    EPOLL_CTL_ADD,
                    GlobalRDMASvcCli().cliEventFd,
                    &mut ev as *mut epoll_event,
                )
            };

            if ret == -1 {
                panic!(
                    "CPULocal::Init {} add host epollfd fail, error is {}",
                    0,
                    errno::errno().0
                );
            }
        }

        let mut events = [epoll_event { events: 0, u64: 0 }; 2];

        let mut data: u64 = 0;
        loop {
            sharespace.IncrHostProcessor();
            if !super::super::runc::runtime::vm::IsRunning() {
                return Err(Error::Exit);
            }
            if QUARK_CONFIG.lock().EnableRDMA {
                GlobalRDMASvcCli().cliShareRegion.lock().clientBitmap.store(0, Ordering::Release);
            }

            Self::Process(sharespace);

            let ret =
                unsafe { libc::read(self.eventfd, &mut data as *mut _ as *mut libc::c_void, 8) };

            if ret < 0 && errno::errno().0 != SysErr::EAGAIN {
                panic!(
                    "KIOThread::Wakeup fail... eventfd is {}, errno is {}",
                    self.eventfd,
                    errno::errno().0
                );
            }

            if QUARK_CONFIG.lock().EnableRDMA {
                let ret = unsafe {
                    libc::read(
                        GlobalRDMASvcCli().cliEventFd,
                        &mut data as *mut _ as *mut libc::c_void,
                        8,
                    )
                };

                if ret < 0 && errno::errno().0 != SysErr::EAGAIN {
                    panic!(
                        "KIOThread::Wakeup fail... cliEventFd is {}, errno is {}",
                        self.eventfd,
                        errno::errno().0
                    );
                }
            }

            if QUARK_CONFIG.lock().EnableRDMA {
                GlobalRDMASvcCli().cliShareRegion.lock().clientBitmap.store(1, Ordering::Release);
            }

            if sharespace.DecrHostProcessor() == 0 {
                Self::ProcessOnce(sharespace);
            }

            ASYNC_PROCESS.Process();
            let timeout = TIMER_STORE.Trigger() / 1000 / 1000;

            // when there is ready task, wake up for preemptive schedule
            let waitTime = if sharespace.scheduler.GlobalReadyTaskCnt() > 0 {
                if timeout == -1 || timeout > 10 {
                    10
                } else {
                    timeout
                }
            } else {
                timeout
            };

            /*if QUARK_CONFIG.lock().EnableRDMA {
                RDMA.HandleCQEvent()?;
            }*/
            let _nfds = unsafe { epoll_wait(epfd, &mut events[0], 3, waitTime as i32) };
        }
    }

    pub fn Wakeup(&self, _sharespace: &ShareSpace) {
        let val: u64 = 1;
        let ret = unsafe { libc::write(self.eventfd, &val as *const _ as *const libc::c_void, 8) };
        if ret < 0 {
            panic!("KIOThread::Wakeup fail...");
        }
    }
}
