// Copyright (c) 2021 QuarkSoft LLC
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

use kvm_bindings::kvm_sregs;
use kvm_bindings::kvm_regs;
use kvm_bindings::kvm_dtable;
use kvm_ioctls::VcpuExit;
use core::mem::size_of;
use libc::*;
use core::sync::atomic::Ordering;

use super::*;
use super::syncmgr::*;
//use super::kvm_ctl::*;
use super::qlib::GetTimeCall;
use super::qlib::linux::time::Timespec;
use super::qlib::common::*;
use super::qlib::task_mgr::*;
use super::qlib::linux_def::*;
use super::qlib::perf_tunning::*;
use super::qlib::*;
use super::qlib::vcpu_mgr::*;
use super::qlib::config::*;
use super::amd64_def::*;
use super::URING_MGR;
use super::runc::runtime::vm::*;

pub struct KVMVcpu {
    pub id: usize,
    pub vcpuCnt: usize,
    //index in the cpu arrary
    pub vcpu: kvm_ioctls::VcpuFd,

    pub topStackAddr: u64,
    pub entry: u64,

    pub gdtAddr: u64,
    pub idtAddr: u64,
    pub tssIntStackStart: u64,
    pub tssAddr: u64,

    pub pageAllocatorBaseAddr: u64,
    pub pageAllocatorOrd: u64,

    pub shareSpace: &'static ShareSpace,

    pub eventfd: i32,
    pub autoStart: bool,
    //the pipe id to notify io_mgr
}

//for pub shareSpace: * mut Mutex<ShareSpace>
unsafe impl Send for KVMVcpu {}

impl KVMVcpu {
    pub fn Init(id: usize,
                vcpuCnt: usize,
                vm_fd: &kvm_ioctls::VmFd,
                pageAlloc: &MemAllocator,
                entry: u64,
                pageAllocatorBaseAddr: u64,
                pageAllocatorOrd: u64,
                eventfd: i32,
                autoStart: bool) -> Result<Self> {
        const DEFAULT_STACK_PAGES: u64 = qlib::linux_def::MemoryDef::DEFAULT_STACK_PAGES; //64KB
        let stackAddr = pageAlloc.Alloc(DEFAULT_STACK_PAGES)?;
        let topStackAddr = stackAddr + (DEFAULT_STACK_PAGES << 12);

        info!("the stack addr is {:x}, topstack address is {:x}", stackAddr, topStackAddr);

        let gdtAddr = pageAlloc.Alloc(1)?;
        let idtAddr = pageAlloc.Alloc(1)?;

        let tssIntStackStart = pageAlloc.Alloc(MemoryDef::INTERRUPT_STACK_PAGES)?;
        let tssAddr = pageAlloc.Alloc(1)?;

        info!("the tssIntStackStart is {:x}, tssAddr address is {:x}", tssIntStackStart, tssAddr);

        let vcpu = vm_fd.create_vcpu(id as u8).map_err(|e| Error::IOError(format!("io::error is {:?}", e))).expect("create vcpu fail");

        return Ok(Self {
            id: id,
            vcpuCnt,
            vcpu,
            topStackAddr: topStackAddr,
            entry: entry,
            gdtAddr: gdtAddr,
            idtAddr: idtAddr,
            tssIntStackStart: tssIntStackStart,
            tssAddr: tssAddr,
            pageAllocatorBaseAddr: pageAllocatorBaseAddr,
            pageAllocatorOrd: pageAllocatorOrd,
            shareSpace: unsafe {
                &*(0 as *const ShareSpace)
            },
            eventfd: eventfd,
            autoStart: autoStart,
        })
    }

    //Notify hostfdnotifier there is message from guest kernel
    pub fn Notify(&mut self) -> Result<()> {
        let data: u64 = 1;
        let ret = unsafe {
            write(self.eventfd, &data as *const _ as *const c_void, 8)
        };

        if ret == -1 {
            let errno = errno::errno().0;
            return Err(Error::SysError(errno))
        }

        assert!(ret == 8, "hostfdnotifier Trigger fail to write data to the eventfd");
        Ok(())
    }

    fn SetupGDT(&mut self, sregs: &mut kvm_sregs) {
        let gdtTbl = unsafe {
            std::slice::from_raw_parts_mut(self.gdtAddr as *mut u64, 4096 / 8)
        };

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

        let stack_end = x86_64::VirtAddr::from_ptr((self.tssIntStackStart + MemoryDef::INTERRUPT_STACK_PAGES * MemoryDef::PAGE_SIZE) as *const u64);

        let tssSegment = self.tssAddr as *mut x86_64::structures::tss::TaskStateSegment;
        unsafe {
            (*tssSegment).interrupt_stack_table[0] = stack_end;
            (*tssSegment).iomap_base = -1 as i16 as u16;
            info!("[{}] the tssSegment stack is {:x}", self.id, self.tssIntStackStart + MemoryDef::INTERRUPT_STACK_PAGES * MemoryDef::PAGE_SIZE);
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
        return (addr, size as u16)
    }


    fn TSStoDescriptor(tss: &x86_64::structures::tss::TaskStateSegment) -> (u64, u64, u16) {
        let (tssBase, tssLimit) = Self::TSS(tss);
        let low = SegmentDescriptor::default().Set(
            tssBase as u32, tssLimit as u32, 0, SEGMENT_DESCRIPTOR_PRESENT |
            SEGMENT_DESCRIPTOR_ACCESS |
            SEGMENT_DESCRIPTOR_WRITE |
            SEGMENT_DESCRIPTOR_EXECUTE);

        let hi = SegmentDescriptor::default().SetHi((tssBase >> 32) as u32);

        return (low.AsU64(), hi.AsU64(), tssLimit);
    }

    fn setup_long_mode(&mut self) -> Result<()> {
        let mut vcpu_sregs = self.vcpu.get_sregs().map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

        //vcpu_sregs.cr0 = CR0_PE | CR0_MP | CR0_AM | CR0_ET | CR0_NE | CR0_WP | CR0_PG;
        vcpu_sregs.cr0 = CR0_PE | CR0_AM | CR0_ET | CR0_PG | CR0_WP; // | CR0_MP | CR0_NE;
        vcpu_sregs.cr3 = VMS.lock().pageTables.read().root.0;
        //vcpu_sregs.cr4 = CR4_PAE | CR4_OSFXSR | CR4_OSXMMEXCPT;
        vcpu_sregs.cr4 = CR4_PAE | CR4_OSFXSR | CR4_OSXMMEXCPT | CR4_FSGSBASE;// | CR4_UMIP ;// CR4_PSE | | CR4_SMEP | CR4_SMAP;

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
        self.vcpu.set_sregs(&vcpu_sregs).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        Ok(())
    }

    pub fn Schedule(&self, taskId: TaskIdQ) {
        self.shareSpace.scheduler.ScheduleQ(taskId.TaskId(), taskId.Queue());
    }

    pub fn run(&mut self) -> Result<()> {
        self.setup_long_mode()?;

        let regs: kvm_regs = kvm_regs {
            rflags: KERNEL_FLAGS_SET,
            rip: self.entry,
            rsp: self.topStackAddr,
            rax: 0x11,
            rbx: 0xdd,
            //arg0
            rdi: self.pageAllocatorBaseAddr,
            //arg1
            rsi: self.pageAllocatorOrd,
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

        self.vcpu.set_regs(&regs).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

        //kvmCtl::QueueTimer(&self.vcpu_fds[0]);

        //let mut shareSpace : * mut Mutex<ShareSpace> = 0 as * mut Mutex<ShareSpace>; //give a default to work around compile uninitialized error

        //self.vcpu_fds[0].EnableInterruptWindow();

        let mut lastVal: u32 = 0;
        let mut first = true;

        //let mut localStr: Vec<u8> = Vec::new();

        info!("start enter guest[{}]: entry is {:x}, stack is {:x}", self.id, self.entry, self.topStackAddr);
        loop {
            match self.vcpu.run().expect("kvm virtual cpu run failed") {
                VcpuExit::IoIn(addr, data) => {
                    info!(
                    "[{}]Received an I/O in exit. Address: {:#x}. Data: {:#x}",
                    self.id,
                    addr,
                    data[0],
                    );

                    let vcpu_sregs = self.vcpu.get_sregs().map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                    if vcpu_sregs.cs.dpl != 0x0 { // call from user space
                        panic!("Get VcpuExit::IoIn from guest user space, Abort, vcpu_sregs is {:#x?}", vcpu_sregs)
                    }
                }
                VcpuExit::IoOut(addr, data) => {
                    let mut vcpu_sregs = self.vcpu.get_sregs().map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                    if vcpu_sregs.cs.dpl != 0x0 { // call from user space
                        panic!("Get VcpuExit::IoOut from guest user space, Abort, vcpu_sregs is {:#x?}", vcpu_sregs)
                    }

                    match addr {
                        qlib::HYPERCALL_HLT => {
                            loop {
                                if !super::runc::runtime::vm::IsRunning() {
                                    return Ok(())
                                }

                                //short term workaround, need to change back to unblock my sql scenario.
                                if self.shareSpace.scheduler.GlobalReadyTaskCnt() > 0 {
                                    break;
                                }

                                /*if self.shareSpace.ReadyTaskCnt(self.id) > 0 ||
                                    self.shareSpace.ReadyTaskCnt(0) > 0 {
                                    break;
                                }*/

                                self.shareSpace.scheduler.VcpuSetWaiting(self.id);

                                if !(self.shareSpace.ReadyTaskCnt(self.id) > 0 ||
                                    self.shareSpace.ReadyTaskCnt(0) > 0) {
                                    match self.shareSpace.scheduler.WaitVcpu(self.id) {
                                        Ok(()) => (),
                                        Err(Error::Exit) => (),
                                        Err(e) => panic!("HYPERCALL_HLT wait fail with error {:?}", e),
                                    }
                                }

                                self.shareSpace.scheduler.VcpuSetSearching(self.id);
                            }
                        }
                        qlib::HYPERCALL_IOWAIT => {
                            if !super::runc::runtime::vm::IsRunning() {
                                {
                                    error!("signal debug");
                                    for i in 0..8 {
                                        error!("vcpu[{}] state is {}/{}", i, self.shareSpace.GetValue(i, 0), self.shareSpace.GetValue(i, 1))
                                    }
                                }

                                return Ok(())
                            }

                            //error!("HYPERCALL_IOWAIT sleeping ...");
                            match KERNEL_IO_THREAD.Wait(&self.shareSpace) {
                                Ok(()) => (),
                                Err(Error::Exit) => {
                                    if !super::runc::runtime::vm::IsRunning() {
                                        {
                                            error!("signal debug");
                                            for i in 0..8 {
                                                error!("vcpu[{}] state is {}/{}", i, self.shareSpace.GetValue(i, 0), self.shareSpace.GetValue(i, 1))
                                            }
                                        }

                                        return Ok(())
                                    }

                                    return Ok(())
                                },
                                Err(e) => {
                                    panic!("KERNEL_IO_THREAD get error {:?}", e);
                                }
                            }
                            //error!("HYPERCALL_IOWAIT waking ...");

                        }
                        qlib::HYPERCALL_URING_WAKE => {
                            URING_MGR.lock().Wake().expect("qlib::HYPER CALL_URING_WAKE fail");
                        }
                        qlib::HYPERCALL_INIT => {
                            info!("get io out: HYPERCALL_INIT");

                            let regs = self.vcpu.get_regs().map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                            let mut vms = VMS.lock();
                            let sharespace = unsafe {
                                &mut *(regs.rcx as * mut ShareSpace)
                            };

                            sharespace.Init();
                            KERNEL_IO_THREAD.Init(sharespace.scheduler.VcpuArr[0].eventfd);
                            URING_MGR.lock().SetupEventfd(sharespace.scheduler.VcpuArr[0].eventfd);
                            vms.shareSpace = sharespace;

                            self.shareSpace = vms.GetShareSpace();

                            SyncMgr::WakeShareSpaceReady();
                        }
                        qlib::HYPERCALL_PRINTSTATE => {
                            info!("get io out: HYPERCALL_PRINTSTATE");
                            error!("vcpu_sregs is {:#x?}", vcpu_sregs);
                        }

                        qlib::HYPERCALL_WAKEUP => {
                            //error!("qlib::HYPERCALL_WAKEUP***************");
                            self.Notify().expect("IO_MGR.lock().Notify() fail");
                        }

                        qlib::HYPERCALL_EXIT_VM => {
                            let regs = self.vcpu.get_regs().map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                            let exitCode = regs.rbx as i32;

                            PerfPrint();

                            SetExitStatus(exitCode);
                            super::ucall::ucall_server::Stop().unwrap();

                            //wake up host iothread
                            self.Notify().expect("IO_MGR.lock().Notify() fail");

                            //wake up Kernel io thread
                            KERNEL_IO_THREAD.Wakeup(VMS.lock().GetShareSpace());

                            //wake up workthread
                            VirtualMachine::WakeAll(VMS.lock().GetShareSpace());
                        }

                        qlib::HYPERCALL_PANIC => {
                            let vcpu_regs = self.vcpu.get_regs().unwrap();
                            let addr = vcpu_regs.rbx;
                            let msg = unsafe {
                                &*(addr as *const Print)
                            };

                            eprintln!("Application error: {}", msg.str);
                            ::std::process::exit(1);
                        }

                        qlib::HYPERCALL_WAKEUP_VCPU => {
                            let regs = self.vcpu.get_regs().map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                            let vcpuId = regs.rbx as usize;

                            //error!("HYPERCALL_WAKEUP_VCPU vcpu id is {:x}", vcpuId);
                            SyncMgr::WakeVcpu(vcpuId);
                        }

                        qlib::HYPERCALL_LOADIDT => {
                            let regs = self.vcpu.get_regs().map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                            let idt = regs.rcx;

                            vcpu_sregs.idt = kvm_dtable {
                                base: idt,
                                limit: 4095,
                                padding: [0; 3]
                            };
                            info!("start to set idt {:x}", idt);
                            self.vcpu.set_sregs(&vcpu_sregs).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                        }

                        qlib::HYPERCALL_PRINT => {
                            let vcpu_regs = self.vcpu.get_regs().unwrap();
                            let addr = vcpu_regs.rbx;
                            let msg = unsafe {
                                &*(addr as *const Print)
                            };

                            if msg.level == DebugLevel::Info {
                                info!("{}", msg.str);
                            } else if msg.level == DebugLevel::Error {
                                error!("{}", msg.str);
                            } else {
                                debug!("{}", msg.str);
                            }

                        }

                        qlib::HYPERCALL_MSG => {
                            let vcpu_regs = self.vcpu.get_regs().unwrap();
                            let data = vcpu_regs.rbx;
                            info!("get kernel msg: {:?}", data);
                        }

                        qlib::HYPERCALL_EXIT => {
                            info!("call in HYPERCALL_EXIT");
                            unsafe { libc::_exit(0) }
                        }

                        qlib::HYPERCALL_U64 => {
                            unsafe {
                                let val = *((data as *const _) as *const u32);
                                if first {
                                    first = false;
                                    lastVal = val
                                } else {
                                    info!("get kernel u64 : 0x{:x}{:x}", lastVal, val);
                                    first = true;
                                }
                            }
                        }

                        qlib::HYPERCALL_GETTIME => {
                            let regs = self.vcpu.get_regs().map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                            let data = regs.rcx;

                            unsafe {
                                let call = &mut *(data as *mut GetTimeCall);

                                let clockId = call.clockId;
                                let ts = Timespec::default();

                                let res = clock_gettime(clockId as clockid_t, &ts as *const _ as u64 as *mut timespec) as i64;

                                if res == -1 {
                                    call.res = errno::errno().0 as i64;
                                } else {
                                    call.res = ts.ToNs()?;
                                }
                            }
                        }

                        qlib::HYPERCALL_HCALL => {
                            let regs = self.vcpu.get_regs().map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                            let addr = regs.rcx;

                            let eventAddr = addr as *mut Event; // as &mut qlib::Event;
                            let event = unsafe {
                                &mut (*eventAddr)
                            };

                            match qcall::qCall(addr, event) {
                                qcall::QcallRet::Normal => {
                                    /*info!("HYPERCALL_HCALL finish call {:x?}", unsafe {
                                        &mut (*eventAddr)
                                    });*/
                                }
                                qcall::QcallRet::Block => {
                                    //info!("start blocked wait ...........");
                                }
                            }
                        }

                        _ => info!("Unknow hyper call!!!!! address is {}", addr)
                    }
                }
                VcpuExit::MmioRead(addr, _data) => {
                    info!(
                    "Received an MMIO Read Request for the address {:#x}.",
                    addr,
                    );
                }
                VcpuExit::MmioWrite(addr, _data) => {
                    info!(
                    "Received an MMIO Write Request to the address {:#x}.",
                    addr,
                    );
                }
                VcpuExit::Hlt => {
                    error!("in hlt....");
                    /*loop {
                        if !super::runc::runtime::vm::IsRunning() {
                            return Ok(())
                        }

                        let readyTaskCnt = self.shareSpace.lock().ReadyTaskCnt();

                        if readyTaskCnt > 0 {
                            break;
                        }

                        let asyncMsgCnt = self.shareSpace.lock().ReadyAsyncMsgCnt();
                        if asyncMsgCnt > 0 {
                            break;
                        }

                        SyncMgr::WaitVcpuWait(self.shareSpace);
                    }*/
                }
                VcpuExit::FailEntry => {
                    info!("get fail entry***********************************");
                    break
                }
                VcpuExit::Exception => {
                    info!("get exception");
                }
                VcpuExit::IrqWindowOpen => {
                    info!("get VcpuExit::IrqWindowOpen");
                    //QueueTimer(&self.vcpu);
                    //&self.vcpu.DisableInterruptWindow();
                }

                r => {
                    let vcpu_sregs = self.vcpu.get_sregs().map_err(|e| Error::IOError(format!("vcpu::error is {:?}", e)))?;
                    let regs = self.vcpu.get_regs().map_err(|e| Error::IOError(format!("vcpu::error is {:?}", e)))?;

                    error!("Panic: CPU[{}] Unexpected exit reason: {:?}, regs is {:#x?}, sregs is {:#x?}",
                        self.id, r, regs, vcpu_sregs);
                    unsafe {
                        libc::exit(0);
                    }
                },
            }
        }

        //let mut vcpu_regs = self.vcpu_fd.get_regs()?;
        Ok(())
    }
}

impl Scheduler {
   pub fn Init(&mut self) {
        for i in 0..self.vcpuCnt.load(Ordering::Relaxed) {
            self.VcpuArr[i].Init(i);
        }
    }

    pub fn WaitVcpu(&self, vcpuId: usize) -> Result<()> {
        self.vcpuWaitMask.fetch_or(1<<vcpuId, Ordering::SeqCst);
        defer!(self.vcpuWaitMask.fetch_and(!(1<<vcpuId), Ordering::SeqCst););

        return self.VcpuArr[vcpuId].Wait();
    }
}

impl CPULocal {
    pub fn Init(&mut self, vcpuId: usize) {
        let eventfd = unsafe {
            libc::eventfd(0, libc::EFD_CLOEXEC)
        };

        if eventfd < 0 {
            panic!("Vcpu::Init fail...");
        }

        let mut uring = URING_MGR.lock();

        uring.Addfd(eventfd).expect("fail to add vcpu eventfd");

        self.eventfd = eventfd;
        self.vcpuId = vcpuId;
        self.data = 1;
    }

    pub fn Wait(&self) -> Result<()> {
        self.SetWaiting();
        defer!(self.SetRunning(););

        let mut e = libc::pollfd {
            fd: self.eventfd,
            events: EVENT_READ as i16,
            revents: 0,
        };

        let ret = unsafe {
            // wait 10 ms
            libc::poll(&mut e, 1, 10)
        };

        if ret == 0 {
            return Ok(())
        }

        let mut data : u64 = 0;
        let ret = unsafe {
            libc::read(self.eventfd, &mut data as * mut _ as *mut libc::c_void, 8)
        };

        if ret < 0 {
            panic!("KIOThread::Wakeup fail... eventfd is {}, errno is {}",
                   self.eventfd, errno::errno().0);
        }

        if !super::runc::runtime::vm::IsRunning() {
            return Err(Error::Exit)
        }

        return Ok(())
    }

    pub fn Wakeup(&self) {
        let val : u64 = 1;
        let ret = unsafe {
            libc::write(self.eventfd, &val as * const _ as *const libc::c_void, 8)
        };
        if ret < 0 {
            panic!("KIOThread::Wakeup fail...");
        }
    }
}

impl ShareSpace {
    pub fn Init(&mut self) {
        self.scheduler.Init();
        self.hostIOThreadEventfd = FD_NOTIFIER.Eventfd();
        URING_MGR.lock().Addfd(self.hostIOThreadEventfd).unwrap();
        self.config.Load();
    }
}