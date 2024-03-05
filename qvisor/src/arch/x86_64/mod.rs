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

use crate::arch::vCPU;
use crate::qlib::common::Error;
use crate::qlib::linux_def::MemoryDef;

pub mod context;
pub mod vm;

#[derive(Default, Debug)]
pub struct x86_64vCPU {
    gdtAddr: u64,
    idtAddr: u64,
    tssIntStackStart: u64,
    tssAddr: u64,
    vcpu_fd: Option<kvm_ioctls::VcpuFd>,
}

pub type ArchvCPU = x86_64vCPU;

impl vCPU for x86_64vCPU {
    fn new(kvm_vm_fd: &kvm_ioctls::VmFd, vCPU_id: usize) -> Self {
        let kvm_vcpu_fd = kvm_vm_fd
            .create_vcpu(vCPU_id as u64)
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))
            .expect("create vcpu fail");

        Self {
            vcpu_fd: Some(kvm_vcpu_fd),
            ..Default::default()
        }
    }

    fn init(&mut self, id: usize) -> Result<(), Error> {
        self.vcpu_reg_init(id)?;
        info!("The tssIntStackStart is {:x}, tssAddr address is {:x}, idt addr is {:x}, gdt addr is {:x}",
             self.tssIntStackStart, self.tssAddr, self.idtAddr, self.gdtAddr);
        info!(
            "[{}] - The tssSegment stack is {:x}",
            id,
            self.tssIntStackStart + MemoryDef::INTERRUPT_STACK_PAGES * MemoryDef::PAGE_SIZE
        );
        Ok(())
    }

    fn run(&self) -> Result<(), Error> {
     //   /// Arch
     //   let regs: kvm_regs = kvm_regs {
     //       rflags: KERNEL_FLAGS_SET,
     //       rip: self.entry,
     //       rsp: self.topStackAddr,
     //       rax: 0x11,
     //       rbx: 0xdd,
     //       //arg0
     //       rdi: self.heapStartAddr, // self.pageAllocatorBaseAddr + self.,
     //       //arg1
     //       rsi: self.shareSpaceAddr,
     //       //arg2
     //       rdx: self.id as u64,
     //       //arg3
     //       rcx: VMS.lock().vdsoAddr,
     //       //arg4
     //       r8: self.vcpuCnt as u64,
     //       //arg5
     //       r9: self.autoStart as u64,
     //       ..Default::default()
     //   };

     //   self.vcpu
     //       .set_regs(&regs)
     //       .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;


     //   /// Where ? -> run loop
     //   let mut lastVal: u32 = 0;
     //   let mut first = true;
     //   ///
     //   loop {
     //       if !super::runc::runtime::vm::IsRunning() {
     //           return Ok(());
     //       }

     //       self.state
     //           .store(KVMVcpuState::GUEST as u64, Ordering::Release);
     //       fence(Ordering::Acquire);
     //       let kvmRet = match self.vcpu.run() {
     //           Ok(ret) => ret,
     //           Err(e) => {
     //               if e.errno() == SysErr::EINTR {
     //                   self.vcpu.set_kvm_immediate_exit(0);
     //                   self.dump()?;
     //                   if self.vcpu.get_ready_for_interrupt_injection() > 0 {
     //                       VcpuExit::IrqWindowOpen
     //                   } else {
     //                       VcpuExit::Intr
     //                   }
     //               } else {
     //                   let regs = self
     //                       .vcpu
     //                       .get_regs()
     //                       .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

     //                   error!("vcpu error regs is {:#x?}, ioerror: {:#?}", regs, e);

     //                   let sregs = self
     //                   .vcpu
     //                   .get_sregs()
     //                   .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

     //                   error!("vcpu error sregs is {:#x?}, ioerror: {:#?}", sregs, e);

     //                   backtracer::trace(regs.rip, regs.rsp, regs.rbp, &mut |frame| {
     //                       error!("host frame is {:#x?}", frame);
     //                       true
     //                   });

     //                   panic!("kvm virtual cpu[{}] run failed: Error {:?}", self.id, e)
     //               }
     //           }
     //       };
     //       self.state
     //           .store(KVMVcpuState::HOST as u64, Ordering::Release);

     //       match kvmRet {
     //           VcpuExit::IoIn(addr, data) => {
     //               info!(
     //                   "[{}]Received an I/O in exit. Address: {:#x}. Data: {:#x}",
     //                   self.id, addr, data[0],
     //               );

     //               let vcpu_sregs = self
     //                   .vcpu
     //                   .get_sregs()
     //                   .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
     //               if vcpu_sregs.cs.dpl != 0x0 {
     //                   // call from user space
     //                   panic!(
     //                       "Get VcpuExit::IoIn from guest user space, Abort, vcpu_sregs is {:#x?}",
     //                       vcpu_sregs
     //                   )
     //               }
     //           }
     //           VcpuExit::IoOut(addr, data) => {
     //               {
     //                   let mut interrupting = self.interrupting.lock();
     //                   interrupting.0 = false;
     //                   interrupting.1.clear();
     //               }

//let vcpu_sregs = self
//                        .vcpu
//                        .get_sregs()
//                        .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
//                    if vcpu_sregs.cs.dpl != 0x0 {
//                        // call from user space
//                        panic!("Get VcpuExit::IoOut from guest user space, Abort, vcpu_sregs is {:#x?}", vcpu_sregs)
//                    }
//
//                    let regs = self
//                        .vcpu
//                        .get_regs()
//                        .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
//                    let para1 = regs.rsi;
//                    let para2 = regs.rcx;
//                    let para3 = regs.rdi;
//                    let para4 = regs.r10;
//
//                    match addr {
//                        qlib::HYPERCALL_IOWAIT => {
//                            if !super::runc::runtime::vm::IsRunning() {
//                                /*{
//                                    for i in 0..8 {
//                                        error!("vcpu[{}] state is {}/{}", i, SHARE_SPACE.GetValue(i, 0), SHARE_SPACE.GetValue(i, 1))
//                                    }
//                                }*/
//
//                                return Ok(());
//                            }
//
//                            defer!(SHARE_SPACE.scheduler.WakeAll());
//                            //error!("HYPERCALL_IOWAIT sleeping ...");
//                            match KERNEL_IO_THREAD.Wait(&SHARE_SPACE) {
//                                Ok(()) => (),
//                                Err(Error::Exit) => {
//                                    if !super::runc::runtime::vm::IsRunning() {
//                                        /*{
//                                            error!("signal debug");
//                                            for i in 0..8 {
//                                                error!("vcpu[{}] state is {}/{}", i, SHARE_SPACE.GetValue(i, 0), SHARE_SPACE.GetValue(i, 1))
//                                            }
//                                        }*/
//
//                                        return Ok(());
//                                    }
//
//                                    return Ok(());
//                                }
//                                Err(e) => {
//                                    panic!("KERNEL_IO_THREAD get error {:?}", e);
//                                }
//                            }
//                            //error!("HYPERCALL_IOWAIT waking ...");
//                        }
//                        qlib::HYPERCALL_RELEASE_VCPU => {
//                            SyncMgr::WakeShareSpaceReady();
//                        }
//                        qlib::HYPERCALL_EXIT_VM => {
//                            let exitCode = para1 as i32;
//
//                            super::print::LOG.Clear();
//                            PerfPrint();
//
//                            SetExitStatus(exitCode);
//
//                            //wake up Kernel io thread
//                            KERNEL_IO_THREAD.Wakeup(&SHARE_SPACE);
//
//                            //wake up workthread
//                            VirtualMachine::WakeAll(&SHARE_SPACE);
//                        }
//
//                        qlib::HYPERCALL_PANIC => {
//                            let addr = para1;
//                            let msg = unsafe { &*(addr as *const Print) };
//
//                            eprintln!("Application error: {}", msg.str);
//                            ::std::process::exit(1);
//                        }
//
//                        qlib::HYPERCALL_WAKEUP_VCPU => {
//                            let vcpuId = para1 as usize;
//
//                            //error!("HYPERCALL_WAKEUP_VCPU vcpu id is {:x}", vcpuId);
//                            SyncMgr::WakeVcpu(vcpuId);
//                        }
//
//                        qlib::HYPERCALL_PRINT => {
//                            let addr = para1;
//                            let msg = unsafe { &*(addr as *const Print) };
//
//                            log!("{}", msg.str);
//                        }
//
//                        qlib::HYPERCALL_MSG => {
//                            let data1 = para1;
//                            let data2 = para2;
//                            let data3 = para3;
//                            let data4 = para4;
//                            raw!(data1, data2, data3, data4);
//                            /*info!(
//                                "[{}] get kernel msg [rsp {:x}/rip {:x}]: {:x}, {:x}, {:x}",
//                                self.id, regs.rsp, regs.rip, data1, data2, data3
//                            );*/
//                        }
//
//                        qlib::HYPERCALL_OOM => {
//                            let data1 = para1;
//                            let data2 = para2;
//                            error!(
//                                "OOM!!! cpu [{}], size is {:x}, alignment is {:x}",
//                                self.id, data1, data2
//                            );
//                            eprintln!(
//                                "OOM!!! cpu [{}], size is {:x}, alignment is {:x}",
//                                self.id, data1, data2
//                            );
//                            ::std::process::exit(1);
//                        }
//
//                        qlib::HYPERCALL_EXIT => {
//                            info!("call in HYPERCALL_EXIT");
//                            unsafe { libc::_exit(0) }
//                        }
//
//                        qlib::HYPERCALL_U64 => unsafe {
//                            let val = *((data as *const _) as *const u32);
//                            if first {
//                                first = false;
//                                lastVal = val
//                            } else {
//                                info!("get kernel u64 : 0x{:x}{:x}", lastVal, val);
//                                first = true;
//                            }
//                        },
//
//                        qlib::HYPERCALL_GETTIME => {
//                            let data = para1;
//
//                            unsafe {
//                                let call = &mut *(data as *mut GetTimeCall);
//
//                                let clockId = call.clockId;
//                                let ts = Timespec::default();
//
//                                let res = clock_gettime(
//                                    clockId as clockid_t,
//                                    &ts as *const _ as u64 as *mut timespec,
//                                ) as i64;
//
//                                if res == -1 {
//                                    call.res = errno::errno().0 as i64;
//                                } else {
//                                    call.res = ts.ToNs()?;
//                                }
//                            }
//                        }
//
//                        qlib::HYPERCALL_VCPU_FREQ => {
//                            let data = para1;
//
//                            let freq = self.vcpu.get_tsc_khz().unwrap() * 1000;
//                            unsafe {
//                                let call = &mut *(data as *mut VcpuFeq);
//                                call.res = freq as i64;
//                            }
//                        }
//
//                        qlib::HYPERCALL_VCPU_YIELD => {
//                            let _ret = HostSubmit().unwrap();
//                            //error!("HYPERCALL_VCPU_YIELD2 {:?}", ret);
//                            //use std::{thread, time};
//
//                            //let millis10 = time::Duration::from_millis(100);
//                            //thread::sleep(millis10);
//                        }
//
//                        qlib::HYPERCALL_VCPU_DEBUG => {
//                            let regs = self
//                                .vcpu
//                                .get_regs()
//                                .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
//                            let vcpu_sregs = self
//                                .vcpu
//                                .get_sregs()
//                                .map_err(|e| Error::IOError(format!("vcpu::error is {:?}", e)))?;
//                            //error!("[{}] HYPERCALL_VCPU_DEBUG regs is {:#x?}", self.id, regs);
//                            error!("sregs {:x} is {:x?}", regs.rsp, vcpu_sregs);
//                            //error!("vcpus is {:#x?}", &SHARE_SPACE.scheduler.VcpuArr);
//                            //unsafe { libc::_exit(0) }
//                        }
//
//                        qlib::HYPERCALL_VCPU_PRINT => {
//                            let regs = self
//                                .vcpu
//                                .get_regs()
//                                .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
//                            error!("[{}] HYPERCALL_VCPU_PRINT regs is {:#x?}", self.id, regs);
//                        }
//
//                        qlib::HYPERCALL_QCALL => {
//                            Self::GuestMsgProcess(&SHARE_SPACE);
//                            // last processor in host
//                            if SHARE_SPACE.DecrHostProcessor() == 0 {
//                                Self::GuestMsgProcess(&SHARE_SPACE);
//                            }
//                        }
//
//                        qlib::HYPERCALL_HCALL => {
//                            let addr = para1;
//
//                            let eventAddr = addr as *mut QMsg; // as &mut qlib::Event;
//                            let qmsg = unsafe { &mut (*eventAddr) };
//
//                            {
//                                let _l = if qmsg.globalLock {
//                                    Some(super::GLOCK.lock())
//                                } else {
//                                    None
//                                };
//
//                                qmsg.ret = Self::qCall(qmsg.msg);
//                            }
//
//                            SHARE_SPACE.IncrHostProcessor();
//
//                            Self::GuestMsgProcess(&SHARE_SPACE);
//                            // last processor in host
//                            if SHARE_SPACE.DecrHostProcessor() == 0 {
//                                Self::GuestMsgProcess(&SHARE_SPACE);
//                            }
//                        }
//
//                        qlib::HYPERCALL_VCPU_WAIT => {
//                            let retAddr = para3;
//
//                            let ret = SHARE_SPACE.scheduler.WaitVcpu(&SHARE_SPACE, self.id, true);
//                            match ret {
//                                Ok(taskId) => unsafe {
//                                    *(retAddr as *mut u64) = taskId as u64;
//                                },
//                                Err(Error::Exit) => return Ok(()),
//                                Err(e) => {
//                                    panic!("HYPERCALL_HLT wait fail with error {:?}", e);
//                                }
//                            }
//                        }
//
//                        _ => info!("Unknow hyper call!!!!! address is {}", addr),
//                    }
//                }
//                VcpuExit::MmioRead(addr, _data) => {
//                    panic!(
//                        "CPU[{}] Received an MMIO Read Request for the address {:#x}.",
//                        self.id, addr,
//                    );
//                }
//                VcpuExit::MmioWrite(addr, _data) => {
//                    panic!(
//                        "[{}] Received an MMIO Write Request to the address {:#x}.",
//                        self.id, addr,
//                    );
//                }
//                VcpuExit::Hlt => {
//                    error!("in hlt....");
//                }
//                VcpuExit::FailEntry => {
//                    info!("get fail entry***********************************");
//                    break;
//                }
//                VcpuExit::Exception => {
//                    info!("get exception");
//                }
//                VcpuExit::IrqWindowOpen => {
//                    self.InterruptGuest();
//                    self.vcpu.set_kvm_request_interrupt_window(0);
//                    fence(Ordering::SeqCst);
//                    {
//                        let mut interrupting = self.interrupting.lock();
//                        interrupting.0 = false;
//                        interrupting.1.clear();
//                    }
//                }
//                VcpuExit::Intr => {
//                    self.vcpu.set_kvm_request_interrupt_window(1);
//                    fence(Ordering::SeqCst);
//                    {
//                        let mut interrupting = self.interrupting.lock();
//                        interrupting.0 = false;
//                        interrupting.1.clear();
//                    }
//
                    //     SHARE_SPACE.MaskTlbShootdown(self.id as _);
                    //
                    //     let mut regs = self
                    //         .vcpu
                    //         .get_regs()
                    //         .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                    //     let mut sregs = self
                    //         .vcpu
                    //         .get_sregs()
                    //         .map_err(|e| Error::IOError(format!("vcpu::error is {:?}", e)))?;
                    //
                    //     let ss = sregs.ss.selector as u64;
                    //     let rsp = regs.rsp;
                    //     let rflags = regs.rflags;
                    //     let cs = sregs.cs.selector as u64;
                    //     let rip = regs.rip;
                    //     let isUser = (ss & 0x3) != 0;
                    //
                    //     let stackTop = if isUser {
                    //         self.tssIntStackStart + MemoryDef::PAGE_SIZE - 16
                    //     } else {
                    //         continue;
                    //     };
                    //
                    //     let mut stack = KernelStack::New(stackTop);
                    //     stack.PushU64(ss);
                    //     stack.PushU64(rsp);
                    //     stack.PushU64(rflags);
                    //     stack.PushU64(cs);
                    //     stack.PushU64(rip);
                    //
                    //     regs.rsp = stack.sp;
                    //     regs.rip = SHARE_SPACE.VirtualizationHandlerAddr();
                    //     regs.rflags = 0x2;
                    //
                    //     sregs.ss.selector = 0x10;
                    //     sregs.ss.dpl = 0;
                    //     sregs.cs.selector = 0x8;
                    //     sregs.cs.dpl = 0;
                    //
                    //     /*error!("VcpuExit::Intr ss is {:x}/{:x}/{:x}/{:x}/{:x}/{}/{:x}/{:#x?}/{:#x?}",
                    //         //self.vcpu.get_ready_for_interrupt_injection(),
                    //         ss,
                    //         rsp,
                    //         rflags,
                    //         cs,
                    //         rip,
                    //         isUser,
                    //         stackTop,
                    //         &sregs.ss,
                    //         &sregs.cs,
                    //     );*/
                    //
                    //     self.vcpu
                    //         .set_regs(&regs)
                    //         .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                    //     self.vcpu
                    //         .set_sregs(&sregs)
                    //         .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
//                }
//                r => {
//                    let vcpu_sregs = self
//                        .vcpu
//                        .get_sregs()
//                        .map_err(|e| Error::IOError(format!("vcpu::error is {:?}", e)))?;
//                    let regs = self
//                        .vcpu
//                        .get_regs()
//                        .map_err(|e| Error::IOError(format!("vcpu::error is {:?}", e)))?;
//
//                    error!("Panic: CPU[{}] Unexpected exit reason: {:?}, regs is {:#x?}, sregs is {:#x?}",
//                        self.id, r, regs, vcpu_sregs);
//
//                    backtracer::trace(regs.rip, regs.rsp, regs.rbp, &mut |frame| {
//                        print!("Unexpected exit frame is {:#x?}", frame);
//                        true
//                    });
//                    unsafe {
//                        libc::exit(0);
//                    }
//                }
//            }
//        }
        Ok(())
    }

}
