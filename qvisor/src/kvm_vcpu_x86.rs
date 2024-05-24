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

//use alloc::alloc::{alloc, Layout};
//use alloc::slice;
use core::mem::size_of;
//use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use std::os::unix::io::AsRawFd;
use std::sync::atomic::fence;
//use std::sync::mpsc::Sender;

use kvm_bindings::*;
use kvm_ioctls::VcpuExit;
use libc::*;
//use nix::sys::signal;

use crate::host_uring::HostSubmit;
use crate::qlib::cpuid::XSAVEFeature::{XSAVEFeatureBNDCSR, XSAVEFeatureBNDREGS};
use crate::qlib::kernel::asm::xgetbv;

use super::*;
//use super::qlib::kernel::TSC;
use super::amd64_def::*;
use super::qlib::buddyallocator::ZeroPage;
use super::qlib::*;
//use super::kvm_ctl::*;
use super::qlib::common::*;
use super::qlib::GetTimeCall;
//use super::qlib::kernel::stack::*;
use super::kvm_vcpu::KVMVcpuState;
use super::kvm_vcpu::SetExitSignal;
use super::qlib::linux::time::Timespec;
use super::qlib::linux_def::*;
use super::qlib::pagetable::*;
use super::qlib::perf_tunning::*;
//use super::qlib::task_mgr::*;
//use super::qlib::vcpu_mgr::*;
use super::runc::runtime::vm::*;
use super::syncmgr::*;

#[repr(C)]
pub struct SignalMaskStruct {
    length: u32,
    mask1: u32,
    mask2: u32,
    _pad: u32,
}

pub struct HostPageAllocator {
    pub allocator: AlignedAllocator,
}

impl HostPageAllocator {
    pub fn New() -> Self {
        return Self {
            allocator: AlignedAllocator::New(0x1000, 0x10000),
        };
    }
}

impl Allocator for HostPageAllocator {
    fn AllocPage(&self, _incrRef: bool) -> Result<u64> {
        let ret = self.allocator.Allocate()?;
        ZeroPage(ret);
        return Ok(ret);
    }
}

impl RefMgr for HostPageAllocator {
    fn Ref(&self, _addr: u64) -> Result<u64> {
        //panic!("HostPageAllocator doesn't support Ref");
        return Ok(1);
    }

    fn Deref(&self, _addr: u64) -> Result<u64> {
        panic!("HostPageAllocator doesn't support Deref");
    }

    fn GetRef(&self, _addr: u64) -> Result<u64> {
        panic!("HostPageAllocator doesn't support GetRef");
    }
}

impl KVMVcpu {
    fn SetupGDT(&self, sregs: &mut kvm_sregs) {
        let gdtTbl = unsafe { std::slice::from_raw_parts_mut(self.gdtAddr as *mut u64, 4096 / 8) };

        let KernelCodeSegment = SegmentDescriptor::default().SetCode64(0, 0, 0);
        let KernelDataSegment = SegmentDescriptor::default().SetData(0, 0xffffffff, 0);
        let _UserCodeSegment32 = SegmentDescriptor::default().SetCode64(0, 0, 3);
        let UserDataSegment = SegmentDescriptor::default().SetData(0, 0xffffffff, 3);
        let UserCodeSegment64 = SegmentDescriptor::default().SetCode64(0, 0, 3);

        sregs.cs = KernelCodeSegment.GenKvmSegment(KCODE);
        sregs.ds = UserDataSegment.GenKvmSegment(UDATA);
        sregs.es = UserDataSegment.GenKvmSegment(UDATA);
        sregs.ss = KernelDataSegment.GenKvmSegment(KDATA);
        sregs.fs = UserDataSegment.GenKvmSegment(UDATA);
        sregs.gs = UserDataSegment.GenKvmSegment(UDATA);

        // error!("cs is {:x?}", sregs.cs);
        // error!("ds is {:x?}", sregs.ds);
        // error!("es is {:x?}", sregs.es);
        // error!("ss is {:x?}", sregs.ss);
        // error!("fs is {:x?}", sregs.fs);
        // error!("gs is {:x?}", sregs.gs);

        gdtTbl[1] = KernelCodeSegment.AsU64();
        gdtTbl[2] = KernelDataSegment.AsU64();
        gdtTbl[3] = UserDataSegment.AsU64();
        gdtTbl[4] = UserCodeSegment64.AsU64();

        let stack_end = x86_64::VirtAddr::from_ptr(
            (self.tssIntStackStart + MemoryDef::INTERRUPT_STACK_PAGES * MemoryDef::PAGE_SIZE)
                as *const u64,
        );

        let tssSegment = self.tssAddr as *mut x86_64::structures::tss::TaskStateSegment;
        unsafe {
            (*tssSegment).interrupt_stack_table[0] = stack_end;
            (*tssSegment).iomap_base = -1 as i16 as u16;
            info!(
                "[{}] the tssSegment stack is {:x}",
                self.id,
                self.tssIntStackStart + MemoryDef::INTERRUPT_STACK_PAGES * MemoryDef::PAGE_SIZE
            );
            let (tssLow, tssHigh, limit) = Self::TSStoDescriptor(&(*tssSegment));

            gdtTbl[5] = tssLow;
            gdtTbl[6] = tssHigh;

            sregs.tr = SegmentDescriptor::New(tssLow).GenKvmSegment(TSS);
            sregs.tr.base = self.tssAddr;
            sregs.tr.limit = limit as u32;
        }
    }

    fn TSS(tss: &x86_64::structures::tss::TaskStateSegment) -> (u64, u16) {
        let addr = tss as *const _ as u64;
        let size = (size_of::<x86_64::structures::tss::TaskStateSegment>() - 1) as u64;
        return (addr, size as u16);
    }

    fn TSStoDescriptor(tss: &x86_64::structures::tss::TaskStateSegment) -> (u64, u64, u16) {
        let (tssBase, tssLimit) = Self::TSS(tss);
        let low = SegmentDescriptor::default().Set(
            tssBase as u32,
            tssLimit as u32,
            0,
            SEGMENT_DESCRIPTOR_PRESENT
                | SEGMENT_DESCRIPTOR_ACCESS
                | SEGMENT_DESCRIPTOR_WRITE
                | SEGMENT_DESCRIPTOR_EXECUTE,
        );

        let hi = SegmentDescriptor::default().SetHi((tssBase >> 32) as u32);

        return (low.AsU64(), hi.AsU64(), tssLimit);
    }

    fn setup_long_mode(&self) -> Result<()> {
        let mut vcpu_sregs = self
            .vcpu
            .get_sregs()
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

        //vcpu_sregs.cr0 = CR0_PE | CR0_MP | CR0_AM | CR0_ET | CR0_NE | CR0_WP | CR0_PG;
        vcpu_sregs.cr0 = CR0_PE | CR0_AM | CR0_ET | CR0_PG | CR0_NE; // | CR0_WP; // | CR0_MP | CR0_NE;
        vcpu_sregs.cr3 = VMS.lock().pageTables.GetRoot();
        //vcpu_sregs.cr4 = CR4_PAE | CR4_OSFXSR | CR4_OSXMMEXCPT;
        vcpu_sregs.cr4 =
            CR4_PSE | CR4_PAE | CR4_PGE | CR4_OSFXSR | CR4_OSXMMEXCPT | CR4_FSGSBASE | CR4_OSXSAVE; // | CR4_UMIP ;// CR4_PSE | | CR4_SMEP | CR4_SMAP;

        vcpu_sregs.efer = EFER_LME | EFER_LMA | EFER_SCE | EFER_NX;

        vcpu_sregs.idt = kvm_bindings::kvm_dtable {
            base: 0,
            limit: 4095,
            ..Default::default()
        };

        vcpu_sregs.gdt = kvm_bindings::kvm_dtable {
            base: self.gdtAddr,
            limit: 4095,
            ..Default::default()
        };

        self.SetupGDT(&mut vcpu_sregs);
        self.vcpu
            .set_sregs(&vcpu_sregs)
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        Ok(())
    }

    pub const KVM_INTERRUPT: u64 = 0x4004ae86;
    pub fn InterruptGuest(&self) {
        let bounce: u32 = 20; //VirtualizationException
        let ret = unsafe {
            ioctl(
                self.vcpu.as_raw_fd(),
                Self::KVM_INTERRUPT,
                &bounce as *const _ as u64,
            )
        };

        assert!(
            ret == 0,
            "InterruptGuest ret is {}/{}/{}",
            ret,
            errno::errno().0,
            self.vcpu.as_raw_fd()
        );
    }

    pub fn run(&self, tgid: i32) -> Result<()> {
        SetExitSignal();
        self.setup_long_mode()?;
        let tid = unsafe { gettid() };
        self.threadid.store(tid as u64, Ordering::SeqCst);
        self.tgid.store(tgid as u64, Ordering::SeqCst);

        let regs: kvm_regs = kvm_regs {
            rflags: KERNEL_FLAGS_SET,
            rip: self.entry,
            rsp: self.topStackAddr,
            rax: 0x11,
            rbx: 0xdd,
            //arg0
            rdi: self.heapStartAddr, // self.pageAllocatorBaseAddr + self.,
            //arg1
            rsi: self.shareSpaceAddr,
            //arg2
            rdx: self.id as u64,
            //arg3
            rcx: VMS.lock().vdsoAddr,
            //arg4
            r8: self.vcpuCnt as u64,
            //arg5
            r9: self.autoStart as u64,
            //rdx:
            //rcx:
            ..Default::default()
        };

        self.vcpu
            .set_regs(&regs)
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

        self.SetXCR0()?;

        let mut lastVal: u32 = 0;
        let mut first = true;

        if self.coreId > 0 {
            let coreid = core_affinity::CoreId {
                id: self.coreId as usize,
            };
            // print cpu id
            core_affinity::set_for_current(coreid);
        }

        self.SignalMask();

        info!(
            "start enter guest[{}]: entry is {:x}, stack is {:x}",
            self.id, self.entry, self.topStackAddr
        );
        loop {
            if !super::runc::runtime::vm::IsRunning() {
                return Ok(());
            }

            self.state
                .store(KVMVcpuState::GUEST as u64, Ordering::Release);
            fence(Ordering::Acquire);
            let kvmRet = match self.vcpu.run() {
                Ok(ret) => ret,
                Err(e) => {
                    if e.errno() == SysErr::EINTR {
                        self.vcpu.set_kvm_immediate_exit(0);
                        self.dump()?;
                        if self.vcpu.get_ready_for_interrupt_injection() > 0 {
                            VcpuExit::IrqWindowOpen
                        } else {
                            VcpuExit::Intr
                        }
                    } else {
                        let regs = self
                            .vcpu
                            .get_regs()
                            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

                        error!("vcpu error regs is {:#x?}, ioerror: {:#?}", regs, e);

                        let sregs = self
                        .vcpu
                        .get_sregs()
                        .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

                        error!("vcpu error sregs is {:#x?}, ioerror: {:#?}", sregs, e);

                        backtracer::trace(regs.rip, regs.rsp, regs.rbp, &mut |frame| {
                            error!("host frame is {:#x?}", frame);
                            true
                        });

                        panic!("kvm virtual cpu[{}] run failed: Error {:?}", self.id, e)
                    }
                }
            };
            self.state
                .store(KVMVcpuState::HOST as u64, Ordering::Release);

            match kvmRet {
                VcpuExit::IoIn(addr, data) => {
                    info!(
                        "[{}]Received an I/O in exit. Address: {:#x}. Data: {:#x}",
                        self.id, addr, data[0],
                    );

                    let vcpu_sregs = self
                        .vcpu
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
                        .vcpu
                        .get_sregs()
                        .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                    if vcpu_sregs.cs.dpl != 0x0 {
                        // call from user space
                        panic!("Get VcpuExit::IoOut from guest user space, Abort, vcpu_sregs is {:#x?}", vcpu_sregs)
                    }

                    let regs = self
                        .vcpu
                        .get_regs()
                        .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                    let para1 = regs.rsi;
                    let para2 = regs.rcx;
                    let para3 = regs.rdi;
                    let para4 = regs.r10;

                    match addr {
                        qlib::HYPERCALL_IOWAIT => {
                            if !super::runc::runtime::vm::IsRunning() {
                                /*{
                                    for i in 0..8 {
                                        error!("vcpu[{}] state is {}/{}", i, SHARE_SPACE.GetValue(i, 0), SHARE_SPACE.GetValue(i, 1))
                                    }
                                }*/

                                return Ok(());
                            }

                            defer!(SHARE_SPACE.scheduler.WakeAll());
                            //error!("HYPERCALL_IOWAIT sleeping ...");
                            match KERNEL_IO_THREAD.Wait(&SHARE_SPACE) {
                                Ok(()) => (),
                                Err(Error::Exit) => {
                                    if !super::runc::runtime::vm::IsRunning() {
                                        /*{
                                            error!("signal debug");
                                            for i in 0..8 {
                                                error!("vcpu[{}] state is {}/{}", i, SHARE_SPACE.GetValue(i, 0), SHARE_SPACE.GetValue(i, 1))
                                            }
                                        }*/

                                        return Ok(());
                                    }

                                    return Ok(());
                                }
                                Err(e) => {
                                    panic!("KERNEL_IO_THREAD get error {:?}", e);
                                }
                            }
                            //error!("HYPERCALL_IOWAIT waking ...");
                        }
                        qlib::HYPERCALL_RELEASE_VCPU => {
                            SyncMgr::WakeShareSpaceReady();
                        }
                        qlib::HYPERCALL_EXIT_VM => {
                            let exitCode = para1 as i32;

                            super::print::LOG.Clear();
                            PerfPrint();

                            SetExitStatus(exitCode);

                            //wake up Kernel io thread
                            KERNEL_IO_THREAD.Wakeup(&SHARE_SPACE);

                            //wake up workthread
                            VirtualMachine::WakeAll(&SHARE_SPACE);
                        }

                        qlib::HYPERCALL_PANIC => {
                            let addr = para1;
                            let msg = unsafe { &*(addr as *const Print) };

                            eprintln!("Application error: {}", msg.str);
                            ::std::process::exit(1);
                        }

                        qlib::HYPERCALL_WAKEUP_VCPU => {
                            let vcpuId = para1 as usize;

                            //error!("HYPERCALL_WAKEUP_VCPU vcpu id is {:x}", vcpuId);
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
                            /*info!(
                                "[{}] get kernel msg [rsp {:x}/rip {:x}]: {:x}, {:x}, {:x}",
                                self.id, regs.rsp, regs.rip, data1, data2, data3
                            );*/
                        }

                        qlib::HYPERCALL_OOM => {
                            let data1 = para1;
                            let data2 = para2;
                            error!(
                                "OOM!!! cpu [{}], size is {:x}, alignment is {:x}",
                                self.id, data1, data2
                            );
                            eprintln!(
                                "OOM!!! cpu [{}], size is {:x}, alignment is {:x}",
                                self.id, data1, data2
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

                            let freq = self.get_frequency()?;
                            unsafe {
                                let call = &mut *(data as *mut VcpuFeq);
                                call.res = freq as i64;
                            }
                        }

                        qlib::HYPERCALL_VCPU_YIELD => {
                            let _ret = HostSubmit().unwrap();
                            //error!("HYPERCALL_VCPU_YIELD2 {:?}", ret);
                            //use std::{thread, time};

                            //let millis10 = time::Duration::from_millis(100);
                            //thread::sleep(millis10);
                        }

                        qlib::HYPERCALL_VCPU_DEBUG => {
                            let regs = self
                                .vcpu
                                .get_regs()
                                .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                            let vcpu_sregs = self
                                .vcpu
                                .get_sregs()
                                .map_err(|e| Error::IOError(format!("vcpu::error is {:?}", e)))?;
                            //error!("[{}] HYPERCALL_VCPU_DEBUG regs is {:#x?}", self.id, regs);
                            error!("sregs {:x} is {:x?}", regs.rsp, vcpu_sregs);
                            //error!("vcpus is {:#x?}", &SHARE_SPACE.scheduler.VcpuArr);
                            //unsafe { libc::_exit(0) }
                        }

                        qlib::HYPERCALL_VCPU_PRINT => {
                            let regs = self
                                .vcpu
                                .get_regs()
                                .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                            error!("[{}] HYPERCALL_VCPU_PRINT regs is {:#x?}", self.id, regs);
                        }

                        qlib::HYPERCALL_QCALL => {
                            Self::GuestMsgProcess(&SHARE_SPACE);
                            // last processor in host
                            if SHARE_SPACE.DecrHostProcessor() == 0 {
                                Self::GuestMsgProcess(&SHARE_SPACE);
                            }
                        }

                        qlib::HYPERCALL_HCALL => {
                            let addr = para1;

                            let eventAddr = addr as *mut QMsg; // as &mut qlib::Event;
                            let qmsg = unsafe { &mut (*eventAddr) };

                            {
                                let _l = if qmsg.globalLock {
                                    Some(super::GLOCK.lock())
                                } else {
                                    None
                                };

                                qmsg.ret = Self::qCall(qmsg.msg);
                            }

                            SHARE_SPACE.IncrHostProcessor();

                            Self::GuestMsgProcess(&SHARE_SPACE);
                            // last processor in host
                            if SHARE_SPACE.DecrHostProcessor() == 0 {
                                Self::GuestMsgProcess(&SHARE_SPACE);
                            }
                        }

                        qlib::HYPERCALL_VCPU_WAIT => {
                            let retAddr = para3;

                            let ret = SHARE_SPACE.scheduler.WaitVcpu(&SHARE_SPACE, self.id, true);
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
                        self.id, addr,
                    );
                }
                VcpuExit::MmioWrite(addr, _data) => {
                    panic!(
                        "[{}] Received an MMIO Write Request to the address {:#x}.",
                        self.id, addr,
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
                    self.InterruptGuest();
                    self.vcpu.set_kvm_request_interrupt_window(0);
                    fence(Ordering::SeqCst);
                    {
                        let mut interrupting = self.interrupting.lock();
                        interrupting.0 = false;
                        interrupting.1.clear();
                    }
                }
                VcpuExit::Intr => {
                    self.vcpu.set_kvm_request_interrupt_window(1);
                    fence(Ordering::SeqCst);
                    {
                        let mut interrupting = self.interrupting.lock();
                        interrupting.0 = false;
                        interrupting.1.clear();
                    }

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
                }
                r => {
                    let vcpu_sregs = self
                        .vcpu
                        .get_sregs()
                        .map_err(|e| Error::IOError(format!("vcpu::error is {:?}", e)))?;
                    let regs = self
                        .vcpu
                        .get_regs()
                        .map_err(|e| Error::IOError(format!("vcpu::error is {:?}", e)))?;

                    error!("Panic: CPU[{}] Unexpected exit reason: {:?}, regs is {:#x?}, sregs is {:#x?}",
                        self.id, r, regs, vcpu_sregs);

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

        //let mut vcpu_regs = self.vcpu_fd.get_regs()?;
        Ok(())
    }

    pub fn VcpuWait(&self) -> i64 {
        let sharespace = &SHARE_SPACE;
        loop {
            if !super::runc::runtime::vm::IsRunning() {
                return -1;
            }

            {
                sharespace.IncrHostProcessor();
                Self::GuestMsgProcess(sharespace);

                defer!({
                    // last processor in host
                    if sharespace.DecrHostProcessor() == 0 {
                        Self::GuestMsgProcess(sharespace);
                    }
                });
            }

            let ret = sharespace.scheduler.WaitVcpu(sharespace, self.id, true);
            match ret {
                Ok(taskId) => return taskId as i64,
                Err(Error::Exit) => return -1,
                Err(e) => panic!("HYPERCALL_HLT wait fail with error {:?}", e),
            }
        }
    }

    pub fn SetXCR0(&self) -> Result<()> {
        let xcr0 = xgetbv();
        // mask MPX feature as it is not fully supported in VM yet
        let maskedXCR0 = xcr0 & !(XSAVEFeatureBNDREGS as u64 | XSAVEFeatureBNDCSR as u64);
        let mut xcrs_args = kvm_xcrs::default();
        xcrs_args.nr_xcrs = 1;
        xcrs_args.xcrs[0].value = maskedXCR0;
        self.vcpu
            .set_xcrs(&xcrs_args)
            .map_err(|e| Error::IOError(format!("failed to set kvm xcr0, {}", e)))?;
        Ok(())
    }

    pub fn dump(&self) -> Result<()> {
        if !Dump(self.id) {
            return Ok(());
        }
        defer!(ClearDump(self.id));
        let regs = self
            .vcpu
            .get_regs()
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        let sregs = self
            .vcpu
            .get_sregs()
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        error!("vcpu regs: {:#x?}", regs);
        let ss = sregs.ss.selector as u64;
        let isUser = (ss & 0x3) != 0;
        if isUser {
            error!("vcpu {} is in user mode, skip", self.id);
            return Ok(());
        }
        let kernelMemRegionSize = QUARK_CONFIG.lock().KernelMemSize;
        let mut frames = String::new();
        crate::qlib::backtracer::trace(regs.rip, regs.rsp, regs.rbp, &mut |frame| {
            frames.push_str(&format!("{:#x?}\n", frame));
            if frame.rbp < MemoryDef::PHY_LOWER_ADDR
                || frame.rbp >= MemoryDef::PHY_LOWER_ADDR + kernelMemRegionSize * MemoryDef::ONE_GB
            {
                false
            } else {
                true
            }
        });
        error!("vcpu {} stack: {}", self.id, frames);
        Ok(())
    }

    pub fn get_frequency(&self) -> Result<u64> {
        Ok((self.vcpu.get_tsc_khz().map_err(|e| Error::SysError(e.errno()))? as u64) * 1000)
    }
}
