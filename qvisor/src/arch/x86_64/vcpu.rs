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

pub mod kvm_vcpu;

use kvm_ioctls::{Kvm, VcpuExit, VmFd};
use kvm_bindings::{kvm_regs, kvm_sregs, kvm_xcrs, KVM_MAX_CPUID_ENTRIES};
use libc::gettid;
use std::{os::fd::RawFd, convert::TryInto, mem::size_of, sync::atomic::{fence, Ordering}};

use crate::{amd64_def::{SegmentDescriptor, SEGMENT_DESCRIPTOR_ACCESS, SEGMENT_DESCRIPTOR_EXECUTE,
            SEGMENT_DESCRIPTOR_PRESENT, SEGMENT_DESCRIPTOR_WRITE}, arch::{tee::{emulcc::EmulCc,
            util::{adjust_addr_to_guest, adjust_addr_to_host, confidentiality_type}, NonConf},
            VirtCpu}, kvm_vcpu::{AlignedAllocate, KVMVcpu, KVMVcpuState, SetExitSignal},
            qlib::{self, backtracer, common::{CR0_AM, CR0_ET, CR0_NE, CR0_PE, CR0_PG, CR4_FSGSBASE,
            CR4_OSFXSR, CR4_OSXMMEXCPT, CR4_OSXSAVE, CR4_PAE, CR4_PGE, CR4_PSE, EFER_LMA, EFER_LME,
            EFER_NX, EFER_SCE, KCODE, KDATA, KERNEL_FLAGS_SET, TSS, UDATA}, kernel::asm::xgetbv,
            linux_def::SysErr, task_mgr::TaskId}, CCMode, VMS};
use crate::{SHARE_SPACE, KERNEL_IO_THREAD, syncmgr::SyncMgr};
use crate::runc::runtime::vm;
use crate::arch::ConfCompExtension;
use crate::GLOCK;
use qlib::{linux_def::MemoryDef, common::Error, qmsg::qcall::{Print, QMsg},
    GetTimeCall, linux::time::Timespec, VcpuFeq,
    cpuid::XSAVEFeature::{XSAVEFeatureBNDREGS, XSAVEFeatureBNDCSR}};

pub struct X86_64VirtCpu {
    pub gtd_addr: u64,
    pub idt_addr: u64,
    pub tss_intr_stack_start: u64,
    pub tss_addr:u64,
    pub vcpu_base: KVMVcpu,
    pub conf_comp_extension: Box<dyn ConfCompExtension>,
}

pub type ArchVirtCpu = X86_64VirtCpu;

impl VirtCpu for X86_64VirtCpu {
    fn new_vcpu(vcpu_id: usize, total_vcpus: usize, vm_fd: &VmFd, entry_addr: u64,
        page_allocator_base_addr: Option<u64>, share_space_table_addr: Option<u64>,
        auto_start: bool, stack_size: usize, kvm: Option<&Kvm>, conf_extension: CCMode)
        -> Result<Self, Error> {
        if kvm.is_none() {
            return Err(Error::InvalidArgument("Expected &Kvm, found None.".to_string()));
        }

        let _vcpu_fd = vm_fd.create_vcpu(vcpu_id as u64)
            .expect("Failed to create kvm-vcpu with ID:{vcpu_id}");

        let mut error_msg = String::new();
        if let Ok(supported_cpuid) = kvm.unwrap().get_supported_cpuid(KVM_MAX_CPUID_ENTRIES)
            .map_err(|e| error_msg = format!("Failed to get supported cpuid with err:{:?}", e)) {
            _vcpu_fd.set_cpuid2(&supported_cpuid).unwrap();
        } else {
            return Err(Error::IOError(error_msg));
        }

        let _gtd_addr = AlignedAllocate(MemoryDef::PAGE_SIZE as usize,
            MemoryDef::PAGE_SIZE as usize, true)?;
        let _idt_addr = AlignedAllocate(MemoryDef::PAGE_SIZE as usize,
            MemoryDef::PAGE_SIZE as usize, true)?;
        let _tisa = AlignedAllocate(MemoryDef::PAGE_SIZE as usize,
            MemoryDef::PAGE_SIZE as usize, true)?;
        let _tss_addr = AlignedAllocate(MemoryDef::PAGE_SIZE as usize,
            MemoryDef::PAGE_SIZE as usize, true)?;
        let _vcpu_base = KVMVcpu::Init(vcpu_id, total_vcpus, entry_addr,
            stack_size, _vcpu_fd, auto_start)?;

        info!("The tssIntStackStart:{:#x}, tss address:{:#x}, idt addr:{:#x}, gdt addr:{:#x}",
            _tisa, _tss_addr, _idt_addr, _gtd_addr);
        let _conf_comp_ext = match conf_extension {
            CCMode::None =>
                NonConf::initialize_conf_extension(share_space_table_addr,
                page_allocator_base_addr)?,
            #[cfg(feature = "cc")]
            CCMode::Normal | CCMode::NormalEmu =>
                EmulCc::initialize_conf_extension(share_space_table_addr,
                page_allocator_base_addr)?,
            _ => {
                return Err(Error::InvalidArgument("Create vcpu failed - bad CCMode type"
                    .to_string()));
            }
        };

        let _self = Self {
            gtd_addr: _gtd_addr,
            idt_addr: _idt_addr,
            tss_intr_stack_start: _tisa,
            tss_addr: _tss_addr,
            vcpu_base: _vcpu_base,
            conf_comp_extension: _conf_comp_ext,
        };

        Ok(_self)
    }

    fn initialize_sys_registers(&self) -> Result<(), Error> {
        let _ = self.setup_long_mode();
        let _ = self.set_xcr0();
        self.conf_comp_extension.set_sys_registers(&self.vcpu_base.vcpu_fd, None)?;

        Ok(())
    }

    fn initialize_cpu_registers(&self) -> Result<(), Error> {
        let regs: kvm_regs = kvm_regs {
            rflags: KERNEL_FLAGS_SET,
            rip: self.vcpu_base.entry,
            rsp: self.vcpu_base.topStackAddr,
            rax: 0x11,
            rbx: 0xdd,
            //arg2
            rdx: self.vcpu_base.id as u64,
            //arg3
            rcx: VMS.lock().vdsoAddr,
            //arg4
            r8: self.vcpu_base.vcpuCnt as u64,
            //arg5
            r9: self.vcpu_base.autoStart as u64,
            ..Default::default()
        };

        self.vcpu_base.vcpu_fd
            .set_regs(&regs)
            .map_err(|e| Error::IOError(format!("Failed to set cpu-regs - error:{:?}", e)))?;
        self.conf_comp_extension.set_cpu_registers(&self.vcpu_base.vcpu_fd, None)?;
        Ok(())
    }

    fn vcpu_run(&self, tgid: i32, _kvm_fd: Option<RawFd>, _vm_fd: Option<RawFd>)
        -> Result<(), Error> {
        SetExitSignal();
        self.vcpu_base.SignalMask();
        if self.vcpu_base.coreId > 0 {
            let core_id = core_affinity::CoreId {
                id: self.vcpu_base.coreId as usize,
            };
            core_affinity::set_for_current(core_id);
        }

        info!(
            "vCPU-Run - id:[{}], entry:{:#x}, stack base:{:#x}",
            self.vcpu_base.id, self.vcpu_base.entry, self.vcpu_base.topStackAddr
        );
        let tid = unsafe { gettid() };
        self.vcpu_base.threadid.store(tid as u64, Ordering::SeqCst);
        self.vcpu_base.tgid.store(tgid as u64, Ordering::SeqCst);

        self._run(None)
    }

    fn default_hypercall_handler(&self, hypercall: u16, arg0: u64, arg1: u64,
        arg2: u64, arg3: u64) -> Result<bool, Error> {
        let id = self.vcpu_base.id;
        match hypercall {
            qlib::HYPERCALL_IOWAIT => {
                if !vm::IsRunning() {
                    return Ok(true);
                }
                defer!(SHARE_SPACE.scheduler.WakeAll());
                match KERNEL_IO_THREAD.Wait(&SHARE_SPACE) {
                    Ok(()) => (),
                    Err(Error::Exit) => {
                        return Ok(true);
                    }
                    Err(e) => {
                        panic!("KERNEL_IO_THREAD get error {:?}", e);
                    }
                }
            },
            qlib::HYPERCALL_RELEASE_VCPU => {
                SyncMgr::WakeShareSpaceReady();
            },
            qlib::HYPERCALL_EXIT_VM => {
                let exit_code = arg0 as i32;
                info!("Exit-VM called - vcpu:{}", self.vcpu_base.id);
                crate::print::LOG.Clear();
                crate::qlib::perf_tunning::PerfPrint();
                vm::SetExitStatus(exit_code);
                //wake up Kernel io thread
                KERNEL_IO_THREAD.Wakeup(&SHARE_SPACE);
                //wake up workthread
                vm::VirtualMachine::WakeAll(&SHARE_SPACE);
            },
            qlib::HYPERCALL_PANIC => {
                let addr = arg0;
                let msg = unsafe { &*(addr as *const Print) };

                eprintln!("Application error: {}", msg.str);
                ::std::process::exit(1);
            },
            qlib::HYPERCALL_WAKEUP_VCPU => {
                let vcpuId = arg0 as usize;
                SyncMgr::WakeVcpu(vcpuId);
            },
            qlib::HYPERCALL_PRINT => {
                let addr = arg0;
                let msg = unsafe { &*(addr as *const Print) };

                log!("{}", msg.str);
            },
            qlib::HYPERCALL_MSG => {
                let data1 = arg0;
                let data2 = arg1;
                let data3 = arg2;
                let data4 = arg3;
                raw!(data1, data2, data3, data4);
            },
            qlib::HYPERCALL_OOM => {
                let data1 = arg0;
                let data2 = arg1;
                error!(
                    "OOM!!! cpu [{}], size is {:#x}, alignment is {:#x}",
                    id, data1, data2
                );
                eprintln!(
                    "OOM!!! cpu [{}], size is {:#x}, alignment is {:#x}",
                    id, data1, data2
                );
                ::std::process::exit(1);
            },
            qlib::HYPERCALL_EXIT => {
                info!("HYPERCALL_EXIT called");
                unsafe { libc::_exit(0) }
            },
            qlib::HYPERCALL_GETTIME => {
                let data = arg0;
                unsafe {
                    let call = &mut *(data as *mut GetTimeCall);
                    let clockId = call.clockId;
                    let ts = Timespec::default();
                    let res = libc::clock_gettime(
                        clockId as libc::clockid_t,
                        &ts as *const _ as u64 as *mut libc::timespec,
                    ) as i64;

                    if res == -1 {
                        call.res = errno::errno().0 as i64;
                    } else {
                        call.res = ts.ToNs()?;
                    }
                }
            },
            qlib::HYPERCALL_VCPU_FREQ => {
                let data = arg0;
                let freq = self.vcpu_base.get_frequency()?;
                unsafe {
                    let call = &mut *(data as *mut VcpuFeq);
                    call.res = freq as i64;
                }
            },
            qlib::HYPERCALL_VCPU_YIELD => {
                let _ret = crate::vmspace::host_uring::HostSubmit().unwrap();
            },
            qlib::HYPERCALL_VCPU_DEBUG => {
                let regs = self.vcpu_base.vcpu_fd.get_regs()
                    .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                let vcpu_sregs = self.vcpu_base.vcpu_fd.get_sregs()
                    .map_err(|e| Error::IOError(format!("vcpu::error is {:?}", e)))?;
                error!("sregs {:x} is {:x?}", regs.rsp, vcpu_sregs);
            },
            qlib::HYPERCALL_VCPU_PRINT => {
                let regs = self.vcpu_base.vcpu_fd.get_regs()
                    .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
                error!("[{}] HYPERCALL_VCPU_PRINT regs is {:#x?}", id, regs);
            },
            qlib::HYPERCALL_QCALL => {
                KVMVcpu::GuestMsgProcess(&SHARE_SPACE);
                // last processor in host
                if SHARE_SPACE.DecrHostProcessor() == 0 {
                    KVMVcpu::GuestMsgProcess(&SHARE_SPACE);
                }
            },
            qlib::HYPERCALL_HCALL => {
                let addr = arg0;

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
            },
            qlib::HYPERCALL_VCPU_WAIT => {
                let retAddr = arg2;
                let ret = SHARE_SPACE.scheduler.WaitVcpu(&SHARE_SPACE, id, true);
                match ret {
                    //NOTE: _fn WaitVcpu()_ dependency
                    #[cfg(not(feature = "cc"))]
                    Ok(taskId) => unsafe {
                        *(retAddr as *mut u64) = taskId as u64;
                    },
                    #[cfg(feature = "cc")]
                    Ok(taskId) => unsafe {
                        *(retAddr as *mut TaskId) = taskId;
                    },
                    Err(Error::Exit) => {
                        return Ok(true)
                    },
                    Err(e) => {
                        panic!("HYPERCALL_HLT wait fail with error {:?}", e);
                    }
                }
            }
            _ => error!("Unknown hypercall - number:{}", hypercall),
        }
        Ok(false)
    }

    fn default_kvm_exit_handler(&self, kvm_exit: VcpuExit) -> Result<bool, Error> {
        let id = self.vcpu_base.id;
        match kvm_exit {
            VcpuExit::IoIn(addr, data) => {
                info!("vCPU:[{}] - I/O-exit - hypercall:{:#x}, data:{:#x}", id, addr, data[0]);

                let vcpu_sregs = self.vcpu_base.vcpu_fd.get_sregs()
                    .map_err(|e| Error::IOError(format!("Failed to get sregs - error:{:?}", e)))?;
                if vcpu_sregs.cs.dpl != 0x0 {
                    // call from user space
                    panic!("VcpuExit::IoIn abort - sregs:{:#x?}", vcpu_sregs)
                }
            },
            VcpuExit::MmioRead(addr, _data) => {
                panic!(
                    "CPU[{}] Received an MMIO Read Request for the address {:#x}.",
                    id, addr,
                );
            },
            VcpuExit::MmioWrite(addr, _data) => {
                panic!(
                    "[{}] Received an MMIO Write Request to the address {:#x}.",
                    id, addr,
                );
            },
            VcpuExit::Hlt => {
                error!("vCPU:{} - Halt-Exit", id);
            },
            VcpuExit::FailEntry => {
                error!("vCPU:{} - FailedEntry-Exit", id);
                return Ok(true);
            },
            VcpuExit::Exception => {
                info!("vCPU:{} - Exception-Exit", id);
            },
            VcpuExit::IrqWindowOpen => {
                self.vcpu_base.interrupt_guest();
                self.vcpu_base.vcpu_fd.set_kvm_request_interrupt_window(0);
                fence(Ordering::SeqCst);
                {
                    let mut interrupting = self.vcpu_base.interrupting.lock();
                    interrupting.0 = false;
                    interrupting.1.clear();
                }
            },
            VcpuExit::Intr => {
                self.vcpu_base.vcpu_fd.set_kvm_request_interrupt_window(1);
                fence(Ordering::SeqCst);
                {
                    let mut interrupting = self.vcpu_base.interrupting.lock();
                    interrupting.0 = false;
                    interrupting.1.clear();
                }
            },
            r => {
                let vcpu_sregs = self.vcpu_base.vcpu_fd.get_sregs()
                    .map_err(|e| Error::IOError(format!("Failed to get sregs - error:{:?}", e)))?;
                let regs = self.vcpu_base.vcpu_fd.get_regs()
                    .map_err(|e| Error::IOError(format!("Failed to get regs - error:{:?}", e)))?;

                error!("vCPU[{}] - Unknown-Exit: {:?}, regs:{:#x?}, sregs:{:#x?}",
                    id, r, regs, vcpu_sregs);

                backtracer::trace(regs.rip, regs.rsp, regs.rbp, &mut |frame| {
                    print!("Unexpected exit frame:{:#x?}", frame);
                    true
                });
                unsafe {
                    libc::exit(0);
                }
            }
        }
        Ok(false)
    }
}

impl X86_64VirtCpu {
    fn setup_long_mode(&self) -> Result<(), Error> {
        let mut vcpu_sregs = self
            .vcpu_base
            .vcpu_fd
            .get_sregs()
            .map_err(|e| Error::IOError(format!("Get sregs failed - error:{:?}", e)))?;

        vcpu_sregs.cr0 = CR0_PE | CR0_AM | CR0_ET | CR0_PG | CR0_NE;
        vcpu_sregs.cr3 = VMS.lock().pageTables.GetRoot();
        vcpu_sregs.cr4 = CR4_PSE | CR4_PAE | CR4_PGE | CR4_OSFXSR
            | CR4_OSXMMEXCPT | CR4_FSGSBASE | CR4_OSXSAVE;

        vcpu_sregs.efer = EFER_LME | EFER_LMA | EFER_SCE | EFER_NX;

        vcpu_sregs.idt = kvm_bindings::kvm_dtable {
            base: 0,
            limit: 4095,
            ..Default::default()
        };

        vcpu_sregs.gdt = kvm_bindings::kvm_dtable {
            base: self.gtd_addr,
            limit: 4095,
            ..Default::default()
        };

        let _ = self.setup_gdt(&mut vcpu_sregs);
        self.vcpu_base.vcpu_fd
            .set_sregs(&vcpu_sregs)
            .map_err(|e| Error::IOError(format!("Set sregs failed - error:{:?}", e)))?;
        Ok(())
    }

    fn setup_gdt(&self, sregs: &mut kvm_sregs) -> Result<(), Error> {
        let gdtTbl = unsafe {
            std::slice::from_raw_parts_mut(
                adjust_addr_to_host(self.gtd_addr, confidentiality_type(self)) as *mut u64,
                (MemoryDef::PAGE_SIZE / 8).try_into().expect("Failed to convert to usize"))
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

        gdtTbl[1] = KernelCodeSegment.AsU64();
        gdtTbl[2] = KernelDataSegment.AsU64();
        gdtTbl[3] = UserDataSegment.AsU64();
        gdtTbl[4] = UserCodeSegment64.AsU64();

        let stack_end = x86_64::VirtAddr::from_ptr(
            (self.tss_intr_stack_start + MemoryDef::INTERRUPT_STACK_PAGES * MemoryDef::PAGE_SIZE)
                as *const u64,
        );

        let tssSegment = adjust_addr_to_host(self.tss_addr, confidentiality_type(self))
            as *mut x86_64::structures::tss::TaskStateSegment;
        unsafe {
            (*tssSegment).interrupt_stack_table[0] = stack_end;
            (*tssSegment).iomap_base = -1 as i16 as u16;
            info!("vCPU:[{}] - tss segment stack:{:#x}", self.vcpu_base.id,
                self.tss_intr_stack_start + MemoryDef::INTERRUPT_STACK_PAGES * MemoryDef::PAGE_SIZE
            );
            let (tssLow, tssHigh, limit) = Self::tss_descriptor(&(*tssSegment),
                confidentiality_type(self));

            gdtTbl[5] = tssLow;
            gdtTbl[6] = tssHigh;

            sregs.tr = SegmentDescriptor::New(tssLow).GenKvmSegment(TSS);
            sregs.tr.base = self.tss_addr;
            sregs.tr.limit = limit as u32;
        }

        Ok(())
    }

    fn set_tss(tss: &x86_64::structures::tss::TaskStateSegment) -> (u64, u16) {
        let addr = tss as *const _ as u64;
        let size = (size_of::<x86_64::structures::tss::TaskStateSegment>() - 1) as u64;
        return (addr, size as u16);
    }

    fn tss_descriptor(tss: &x86_64::structures::tss::TaskStateSegment,
        confidentiality_type: CCMode) -> (u64, u64, u16) {
        let (mut tssBase, tssLimit) = Self::set_tss(tss);
        tssBase = adjust_addr_to_guest(tssBase, confidentiality_type);
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

    fn set_xcr0(&self) -> Result<(), Error> {
        let xcr0 = xgetbv();
        // mask MPX feature as it is not fully supported in VM yet
        let maskedXCR0 = xcr0 & !(XSAVEFeatureBNDREGS as u64 | XSAVEFeatureBNDCSR as u64);
        let mut xcrs_args = kvm_xcrs::default();
        xcrs_args.nr_xcrs = 1;
        xcrs_args.xcrs[0].value = maskedXCR0;
        self.vcpu_base.vcpu_fd
            .set_xcrs(&xcrs_args)
            .map_err(|e| Error::IOError(format!("Failed to set xcr0 - error:{}", e)))?;

        Ok(())
    }

    fn _run(&self, _vm_fd: Option<&VmFd>) -> Result<(), Error> {
        let mut exit_loop: bool;
        loop {
            if !vm::IsRunning() {
                break;
            }

            self.vcpu_base.state.store(KVMVcpuState::GUEST as u64, Ordering::Release);
            fence(Ordering::Acquire);

            let mut kvm_ret = match self.vcpu_base.vcpu_fd.run() {
                Ok(ret) => ret,
                Err(e) => {
                    if e.errno() == SysErr::EINTR {
                        self.vcpu_base.vcpu_fd.set_kvm_immediate_exit(0);
                        self.vcpu_base.dump()?;
                        if self.vcpu_base.vcpu_fd.get_ready_for_interrupt_injection() > 0 {
                            VcpuExit::IrqWindowOpen
                        } else {
                            VcpuExit::Intr
                        }
                    } else {
                        let regs = self.vcpu_base.vcpu_fd.get_regs()
                            .map_err(|e| Error::IOError(
                                format!("Get cpu-regs failed - error:{:?}", e)))?;

                        error!("vCPU[{}] - regs:{:#x?}, error:{:#?}", self.vcpu_base.id, regs, e);

                        let sregs = self.vcpu_base.vcpu_fd.get_sregs()
                        .map_err(|e| Error::IOError(format!("Get sregs failed - error:{:?}", e)))?;

                        error!("vCPU[{}] - sregs:{:#x?}, error:{:#?}",
                            self.vcpu_base.id, sregs, e);

                        backtracer::trace(regs.rip, regs.rsp, regs.rbp, &mut |frame| {
                            error!("host frame is {:#x?}", frame);
                            true
                        });
                        panic!("vCPU-Run failed - id:{}, error:{:?}", self.vcpu_base.id, e)
                    }
                }
            };

            self.vcpu_base.state.store(KVMVcpuState::HOST as u64, Ordering::Release);

            if let VcpuExit::IoOut(addr, _) = kvm_ret {
                {
                    let mut interrupting = self.vcpu_base.interrupting.lock();
                    interrupting.0 = false;
                    interrupting.1.clear();
                }

                let vcpu_sregs = self.vcpu_base.vcpu_fd.get_sregs()
                    .map_err(|e| Error::IOError(format!("Get sregs failed - error:{:?}", e)))?;
                if vcpu_sregs.cs.dpl != 0x0 {
                    // call from user space
                    panic!("VcpuExit::IoOut - Abort, vcpu_sregs:{:#x?}", vcpu_sregs)
                }
                let (arg0, arg1, arg2, arg3) = self.conf_comp_extension
                    .get_hypercall_arguments(&self.vcpu_base.vcpu_fd, self.vcpu_base.id)?;
                if self.conf_comp_extension.should_handle_hypercall(addr) {
                    exit_loop = self.conf_comp_extension.handle_hypercall(addr, arg0, arg1,
                        arg2, arg3, self.vcpu_base.id)
                        .expect("VM run failed - cannot handle hypercall correctly.");
                } else {
                    exit_loop = self.default_hypercall_handler(addr, arg0, arg1, arg2, arg3)
                        .expect("VM run failed - cannot handle hypercall correctly.");
                }
            } else if self.conf_comp_extension.should_handle_kvm_exit(&kvm_ret) {
                exit_loop = self.conf_comp_extension.handle_kvm_exit(&mut kvm_ret, self.vcpu_base.id, _vm_fd)?;
            } else {
                exit_loop = self.default_kvm_exit_handler(kvm_ret)?;
            }
            if exit_loop {
                return Ok(());
            }
        }
        info!("VM-Run stopped for id:{}", self.vcpu_base.id);
        Ok(())
    }
}
