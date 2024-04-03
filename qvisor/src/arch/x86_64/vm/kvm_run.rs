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

use libc::{clock_gettime, clockid_t, timespec};
use kvm_ioctls::VcpuExit;
use core::sync::atomic::{Ordering, fence};
use std::convert::TryInto;

use crate::arch::vCPU;
use crate::{SHARE_SPACE, KERNEL_IO_THREAD, Print, qlib, QMsg, GLOCK};
use crate::syncmgr::SyncMgr;
use crate::host_uring::HostSubmit;
use crate::kvm_vcpu::KVMVcpu;
use crate::qlib::backtracer;
use crate::qlib::GetTimeCall;
use crate::qlib::VcpuFeq;
use crate::qlib::common::Error;
use crate::qlib::linux_def::SysErr;
use crate::qlib::linux::time::Timespec;
use crate::runc::runtime::vm;
use crate::arch::__cpu_arch::x86_64vCPU;
use crate::qlib::perf_tunning::PerfPrint;

impl x86_64vCPU {
    pub(in super::super::super::__cpu_arch) fn vcpu_run(&self, id: u64)
    -> Result<(), Error> {
        let mut lastVal: u32 = 0;
        let mut first = true;

        loop {
            if !vm::IsRunning() {
                return Ok(());
            }

           // Move state internal
           // self.state
           //     .store(KVMVcpuState::GUEST as u64, Ordering::Release);
           // fence(Ordering::Acquire);
            let kvmRet = match self.vcpu_fd()
                                   .run() {
                Ok(ret) => {
                    debug!("VMM: returned - no error.");
                    ret
                },
                Err(e) => {
                    debug!("VMM: returned - error.");
                    if e.errno() == SysErr::EINTR {
                        self.vcpu_fd()
                            .set_kvm_immediate_exit(0);
                        self.dump(id)?;
                        if self.vcpu_fd()
                               .get_ready_for_interrupt_injection() > 0 {
                               VcpuExit::IrqWindowOpen
                        } else {
                            VcpuExit::Intr
                        }
                    } else {
                        let regs = self.vcpu_fd()
                                .get_regs()
                                .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

                        //
                        // Similiar to drop() ???
                        //
                        error!("vcpu error regs is {:#x?}, ioerror: {:#?}", regs, e);

                        let sregs = self.vcpu_fd()
                                .get_sregs()
                                .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

                        error!("vcpu error sregs is {:#x?}, ioerror: {:#?}", sregs, e);

                        backtracer::trace(regs.rip, regs.rsp, regs.rbp, &mut |frame| {
                            error!("host frame is {:#x?}", frame);
                            true
                        });

                        panic!("kvm virtual cpu[{}] run failed: Error {:?}", id, e)
                    }
                }
            };

            //self.state
            //    .store(KVMVcpuState::HOST as u64, Ordering::Release);

            match kvmRet {
                VcpuExit::IoIn(addr, data) => {
                    info!(
                        "[{}]Received an I/O in exit. Address: {:#x}. Data: {:#x}",
                        id, addr, data[0],
                    );

                    let vcpu_sregs = self.vcpu_fd()
                        .get_sregs()
                        .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                    if vcpu_sregs.cs.dpl != 0x0 {
                        // call from user space
                        panic!(
                            "Get VcpuExit::IoIn from guest user space, Abort, vcpu_sregs is {:#x?}",
                            vcpu_sregs
                        )
                    }
                }
                VcpuExit::IoOut(addr, data) => {
                    {
                        let mut interrupting = self.interrupting.lock();
                        interrupting.0 = false;
                        interrupting.1.clear();
                    }
            let vcpu_sregs = self
                        .vcpu_fd()
                        .get_sregs()
                        .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                    if vcpu_sregs.cs.dpl != 0x0 {
                        // call from user space
                        panic!("Get VcpuExit::IoOut from guest user space, Abort, vcpu_sregs is {:#x?}", vcpu_sregs)
                    }

                    let regs = self
                        .vcpu_fd()
                        .get_regs()
                        .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                    let para1 = regs.rsi;
                    let para2 = regs.rcx;
                    let para3 = regs.rdi;
                    let para4 = regs.r10;

                    match addr {
                        qlib::HYPERCALL_IOWAIT => {
                            if !vm::IsRunning() {
                                return Ok(());
                            }

                            defer!(SHARE_SPACE.scheduler.WakeAll());
                            match KERNEL_IO_THREAD.Wait(&SHARE_SPACE) {
                                Ok(()) => (),
                                Err(Error::Exit) => {
                                    if !vm::IsRunning() {
                                        return Ok(());
                                    }

                                    return Ok(());
                                }
                                Err(e) => {
                                    panic!("KERNEL_IO_THREAD get error {:?}", e);
                                }
                            }
                        }
                        qlib::HYPERCALL_RELEASE_VCPU => {
                            SyncMgr::WakeShareSpaceReady();
                        }
                        qlib::HYPERCALL_EXIT_VM => {
                            let exitCode = para1 as i32;

                            crate::print::LOG.Clear();
                            PerfPrint();

                            vm::SetExitStatus(exitCode);

                            //wake up Kernel io thread
                            KERNEL_IO_THREAD.Wakeup(&SHARE_SPACE);

                            //wake up workthread
                            vm::VirtualMachine::WakeAll(&SHARE_SPACE);
                        }

                        qlib::HYPERCALL_PANIC => {
                            let addr = para1;
                            let msg = unsafe { &*(addr as *const Print) };

                            eprintln!("Application error: {}", msg.str);
                            ::std::process::exit(1);
                        }

                        qlib::HYPERCALL_WAKEUP_VCPU => {
                            let vcpuId = para1 as usize;
                            SyncMgr::WakeVcpu(vcpuId);
                        }

                        qlib::HYPERCALL_PRINT => {
                            let addr = para1;
                            let msg = unsafe { &*(addr as *const Print) };
                            log!("{}", msg.str);
                        }

                        qlib::HYPERCALL_MSG => {
                            let data1 = para1;
                            let data2 = para2;
                            let data3 = para3;
                            let data4 = para4;
                            raw!(data1, data2, data3, data4);
                        }

                        qlib::HYPERCALL_OOM => {
                            let data1 = para1;
                            let data2 = para2;
                            error!(
                                "OOM!!! cpu [{}], size is {:x}, alignment is {:x}",
                                id, data1, data2
                            );
                            eprintln!(
                                "OOM!!! cpu [{}], size is {:x}, alignment is {:x}",
                                id, data1, data2
                            );
                            ::std::process::exit(1);
                        }

                        qlib::HYPERCALL_EXIT => {
                            info!("call in HYPERCALL_EXIT");
                            unsafe { libc::_exit(0) }
                        }

                        qlib::HYPERCALL_U64 => unsafe {
                            let val = *((data as *const _) as *const u32);
                            if first {
                                first = false;
                                lastVal = val
                            } else {
                                info!("get kernel u64 : 0x{:x}{:x}", lastVal, val);
                                first = true;
                            }
                        },

                        qlib::HYPERCALL_GETTIME => {
                            let data = para1;

                            unsafe {
                                let call = &mut *(data as *mut GetTimeCall);

                                let clockId = call.clockId;
                                let ts = Timespec::default();

                                let res = clock_gettime(
                                    clockId as clockid_t,
                                    &ts as *const _ as u64 as *mut timespec,
                                ) as i64;

                                if res == -1 {
                                    call.res = errno::errno().0 as i64;
                                } else {
                                    call.res = ts.ToNs()?;
                                }
                            }
                        }

                        qlib::HYPERCALL_VCPU_FREQ => {
                            let data = para1;

                            let freq = self.vcpu_fd()
                                           .get_tsc_khz()
                                           .unwrap() * 1000;
                            unsafe {
                                let call = &mut *(data as *mut VcpuFeq);
                                call.res = freq as i64;
                            }
                        }

                        qlib::HYPERCALL_VCPU_YIELD => {
                            let _ret = HostSubmit().unwrap();
                        }

                        qlib::HYPERCALL_VCPU_DEBUG => {
                            let regs = self
                                .vcpu_fd()
                                .get_regs()
                                .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                            let vcpu_sregs = self
                                .vcpu_fd()
                                .get_sregs()
                                .map_err(|e| Error::IOError(format!("vcpu::error is {:?}", e)))?;
                            error!("sregs {:x} is {:x?}", regs.rsp, vcpu_sregs);
                        }

                        qlib::HYPERCALL_VCPU_PRINT => {
                            let regs = self
                                .vcpu_fd()
                                .get_regs()
                                .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                            error!("[{}] HYPERCALL_VCPU_PRINT regs is {:#x?}", id, regs);
                        }

                        qlib::HYPERCALL_QCALL => {
                            KVMVcpu::GuestMsgProcess(&SHARE_SPACE);
                            // last processor in host
                            if SHARE_SPACE.DecrHostProcessor() == 0 {
                                KVMVcpu::GuestMsgProcess(&SHARE_SPACE);
                            }
                        }

                        qlib::HYPERCALL_HCALL => {
                            let addr = para1;
                            let eventAddr = addr as *mut QMsg; // as &mut qlib::Event;
                            let qmsg = unsafe { &mut (*eventAddr) };

                            {
                                let _l = if qmsg.globalLock {
                                    Some(GLOCK.lock())
                                } else {
                                    None
                                };

                                qmsg.ret = KVMVcpu::qCall(qmsg.msg);
                            }

                            SHARE_SPACE.IncrHostProcessor();

                            KVMVcpu::GuestMsgProcess(&SHARE_SPACE);
                            // last processor in host
                            if SHARE_SPACE.DecrHostProcessor() == 0 {
                                KVMVcpu::GuestMsgProcess(&SHARE_SPACE);
                            }
                        }

                        qlib::HYPERCALL_VCPU_WAIT => {
                            let retAddr = para3;
                            let ret = SHARE_SPACE.scheduler
                                                 .WaitVcpu(&SHARE_SPACE,
                                                           id.try_into().unwrap(),
                                                           true);
                            match ret {
                                Ok(taskId) => unsafe {
                                    *(retAddr as *mut u64) = taskId as u64;
                                },
                                Err(Error::Exit) => return Ok(()),
                                Err(e) => {
                                    panic!("HYPERCALL_HLT wait fail with error {:?}", e);
                                }
                            }
                        }

                        _ => info!("Unknow hyper call!!!!! address is {}", addr),
                    }
                }
                VcpuExit::MmioRead(addr, _data) => {
                    panic!(
                        "CPU[{}] Received an MMIO Read Request for the address {:#x}.",
                        id, addr,
                    );
                }
                VcpuExit::MmioWrite(addr, _data) => {
                    panic!(
                        "[{}] Received an MMIO Write Request to the address {:#x}.",
                        id, addr,
                    );
                }
                VcpuExit::Hlt => {
                    error!("in hlt....");
                }
                VcpuExit::FailEntry => {
                    info!("get fail entry***********************************");
                    break;
                }
                VcpuExit::Exception => {
                    info!("get exception");
                }
                VcpuExit::IrqWindowOpen => {
                    self.interrupt_guest();
                    self.vcpu_fd()
                        .set_kvm_request_interrupt_window(0);
                    fence(Ordering::SeqCst);
                    {
                        let mut interrupting = self.interrupting.lock();
                        interrupting.0 = false;
                        interrupting.1.clear();
                    }
                }
                VcpuExit::Intr => {
                    self.vcpu_fd()
                        .set_kvm_request_interrupt_window(1);
                    fence(Ordering::SeqCst);
                    {
                        let mut interrupting = self.interrupting.lock();
                        interrupting.0 = false;
                        interrupting.1.clear();
                    }

                }
                r => {
                    let vcpu_sregs = self
                        .vcpu_fd()
                        .get_sregs()
                        .map_err(|e| Error::IOError(format!("vcpu::error is {:?}", e)))?;
                    let regs = self
                        .vcpu_fd()
                        .get_regs()
                        .map_err(|e| Error::IOError(format!("vcpu::error is {:?}", e)))?;

                    error!("Panic: CPU[{}] Unexpected exit reason: {:?}, regs is {:#x?}, sregs is {:#x?}",
                        id, r, regs, vcpu_sregs);

                    backtracer::trace(regs.rip, regs.rsp, regs.rbp, &mut |frame| {
                        print!("Unexpected exit frame is {:#x?}", frame);
                        true
                    });
                    unsafe {
                        libc::exit(0);
                    }
                }
            }
        }
        Ok(())
    }
}
