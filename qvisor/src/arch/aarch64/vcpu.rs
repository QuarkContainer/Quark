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

use kvm_bindings::{kvm_vcpu_init, KVM_ARM_VCPU_PSCI_0_2};
use kvm_ioctls::{Kvm, VcpuExit, VmFd};
use kvm_vcpu::{Register, KvmAarch64Reg::*};
use libc::gettid;
use crate::{arch::{tee::NonConf, ConfCompExtension, ConfCompType, VirtCpu},
            kvm_vcpu::{KVMVcpuState, SetExitSignal}, qlib::{self, common::Error,
            linux::time::Timespec, linux_def::{MemoryDef, SysErr}, qmsg::qcall::{Print, QMsg},
            GetTimeCall, VcpuFeq}, runc::runtime::vm, syncmgr::SyncMgr, KVMVcpu, GLOCK,
            KERNEL_IO_THREAD, SHARE_SPACE, VMS};
use super::vcpu::kvm_vcpu::*;
use std::{sync::atomic::Ordering, vec::Vec};

pub struct Aarch64VirtCpu {
    tcr_el1: u64,
    mair_el1: u64,
    ttbr0_el1: u64,
    cpacr_el1: u64,
    sctlr_el1: u64,
    cntkctl_el1: u64,
    kvi: kvm_vcpu_init,
    pub vcpu_base: KVMVcpu,
    pub conf_comp_extension: Box<dyn ConfCompExtension>,
}

pub type ArchVirtCpu = Aarch64VirtCpu;

impl VirtCpu for Aarch64VirtCpu {

    fn new_vcpu(vcpu_id: usize, total_vcpus: usize, vm_fd: &VmFd, entry_addr: u64,
        page_allocator_base_addr: Option<u64>, share_space_table_addr: Option<u64>,
        auto_start: bool, stack_size: usize, _kvm: Option<&Kvm>, conf_extension: ConfCompType)
        -> Result<Self, Error> {
        let _vcpu_fd = vm_fd.create_vcpu(vcpu_id as u64)
            .expect("Failed to create kvm-vcpu with ID:{vcpu_id}");
        let _trc_el1 = _TCR_TXSZ_VA48 |  _TCR_CACHE_FLAGS | _TCR_SHARED |
                       _TCR_TG_FLAGS |  _TCR_ASID16 |  _TCR_IPS_40BITS;
        let _mair_el1 = _MT_EL1_INIT;
        let _ttbr0_el1 = VMS.lock().pageTables.GetRoot();
        let _cntkctl_el1 = _CNTKCTL_EL1_DEFAULT;
        // NOTE: FPEN[21:20] - Do not cause instructions related
        //                     to FP registers to be trapped.
        let _cpacr_el1 = 3 << 20;
        // NOTE: before PAN is fully supported, we disable this feature by setting SCTLR_EL1.SPAN
        // to 1, preventing PSTATE.PAN from being set to 1 upon exception to EL1
        let _sctlr_el1 = _SCTLR_EL1_DEFAULT;
        let _vcpu_base = KVMVcpu::Init(vcpu_id, total_vcpus, entry_addr, stack_size,
            _vcpu_fd, auto_start)?;

        let _conf_comp_ext = match conf_extension {
            ConfCompType::NoConf =>
                NonConf::initialize_conf_extension(share_space_table_addr,
                page_allocator_base_addr)?,
            _ => {
                return Err(
                    Error::InvalidArgument("Create vcpu failed - bad ConfCompType".to_string()));
            }
        };

        let mut _kvi = kvm_vcpu_init::default();
        vm_fd.get_preferred_target(&mut _kvi)
            .map_err(|e| format!("Failed to find kvm target for vcpu - error:{:?}", e))?;
        _kvi.features[0] |= 1 << KVM_ARM_VCPU_PSCI_0_2;

        let _self = Self {
            tcr_el1: _trc_el1,
            mair_el1: _mair_el1,
            ttbr0_el1: _ttbr0_el1,
            cpacr_el1: _cpacr_el1,
            sctlr_el1: _sctlr_el1,
            cntkctl_el1: _cntkctl_el1,
            kvi: _kvi,
            vcpu_base: _vcpu_base,
            conf_comp_extension: _conf_comp_ext
        };

        Ok(_self)
    }

    fn initialize_sys_registers(&self) -> Result<(), Error> {
        let tcr_el1 = Register::Reg(TcrEl1, self.tcr_el1);
        let mair_el1 = Register::Reg(MairEl1, self.mair_el1);
        let ttbr0_el1 = Register::Reg(Ttbr0El1, self.ttbr0_el1);
        let cntkctl_el1 = Register::Reg(CntkctlEl1, self.cntkctl_el1);
        let cpacr_el1 = Register::Reg(CpacrEl1, self.cpacr_el1);
        let sctlr_el1 = Register::Reg(SctlrEl1, self.sctlr_el1);
        let reg_list: Vec<Register> = vec![tcr_el1, mair_el1, ttbr0_el1, cntkctl_el1,
                                        cpacr_el1, sctlr_el1];
        self.vcpu_base.set_regs(reg_list)
    }

    fn initialize_cpu_registers(&self) -> Result<(), Error> {
        let sp_el1 = Register::Reg(SpEl1, self.vcpu_base.topStackAddr);
        let pc = Register::Reg(PC, self.vcpu_base.entry);
        let r2 = Register::Reg(R2, self.vcpu_base.id as u64);
        let vdso_entry = VMS.lock().vdsoAddr;
        let r3 = Register::Reg(R3, vdso_entry);
        let r4 = Register::Reg(R4, self.vcpu_base.vcpuCnt as u64);
        let r5 = Register::Reg(R5, self.vcpu_base.autoStart as u64);
        let reg_list = vec![sp_el1, pc, r2, r3, r4, r5];
        self.vcpu_base.set_regs(reg_list)
    }

    fn vcpu_run(&self, tgid: i32) -> Result<(), Error> {
        self.vcpu_base.vcpu_fd.vcpu_init(&self.kvi)
            .map_err(|e| Error::SystemErr(e.errno()))?;
        self.initialize_sys_registers().expect("Can not run vcpu - failed to init sysregs");
        self.initialize_cpu_registers().expect("Can not run vcpu - failed to init cpu-regs");
        self.conf_comp_extension.set_sys_registers(&self.vcpu_base.vcpu_fd)?;
        self.conf_comp_extension.set_cpu_registers(&self.vcpu_base.vcpu_fd)?;
        SetExitSignal();
        self.vcpu_base.SignalMask();
        if self.vcpu_base.cordId > 0 {
            let core_id = core_affinity::CoreId {
                id: self.vcpu_base.cordId as usize,
            };
            core_affinity::set_for_current(core_id);
        }

        info!("vCPU-Run - id:[{}], entry:{:#x}, stack base:{:#x}",
            self.vcpu_base.id, self.vcpu_base.entry, self.vcpu_base.topStackAddr);
        let tid = unsafe { gettid() };
        self.vcpu_base.threadid.store(tid as u64, Ordering::SeqCst);
        self.vcpu_base.tgid.store(tgid as u64, Ordering::SeqCst);
        self._run()
    }

    fn default_hypercall_handler(&self, hypercall: u16, _data: &[u8], arg0: u64, arg1: u64,
        arg2: u64, arg3: u64) -> Result<bool, Error> {
        let id = self.vcpu_base.id;
        match hypercall {
            qlib::HYPERCALL_IOWAIT => {
                if !vm::IsRunning() {
                    return Ok(true);
                }
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
            qlib::HYPERCALL_U64 => {
                info!("HYPERCALL_U64 is not handled");
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
                // TODO: the cntfreq_el0 register may not be properly programmed
                // to represent the system counter frequency in many platforms
                // (careless firmware implementations). There should be a sanity
                // check here, if the cntfreq reads 0, work around it and get
                // the actual frequency.
                let freq = self.vcpu_base.get_frequency()?;
                if freq == 0 {
                    panic!("system counter frequency (cntfrq_el0) reads 0. It\
                           may not be properly programmed by the firmware");
                }
                unsafe {
                    let call = &mut *(data as *mut VcpuFeq);
                    call.res = freq as i64;
                }
            },
            qlib::HYPERCALL_VCPU_YIELD => {
                let _ret = crate::vmspace::host_uring::HostSubmit().unwrap();
            },
            qlib::HYPERCALL_VCPU_DEBUG => {
                error!("DEBUG not implemented");
            },
            qlib::HYPERCALL_VCPU_PRINT => {
                error!("[{}] HYPERCALL_VCPU_PRINT", id);
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
                    Ok(taskId) => unsafe {
                        *(retAddr as *mut u64) = taskId as u64;
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
            VcpuExit::MmioRead(addr, _data) => {
                self.vcpu_base.backtrace()?;
                panic!("CPU[{}] Received an MMIO Read Request for the address {:#x}.",
                    self.vcpu_base.id, addr,);
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
                self.vcpu_base.InterruptGuest();
                self.vcpu_base.vcpu_fd.set_kvm_request_interrupt_window(0);
                {
                    let mut interrupting = self.vcpu_base.interrupting.lock();
                    interrupting.0 = false;
                    interrupting.1.clear();
                }
            },
            VcpuExit::Intr => {
                self.vcpu_base.vcpu_fd.set_kvm_request_interrupt_window(1);
                {
                    let mut interrupting = self.vcpu_base.interrupting.lock();
                    interrupting.0 = false;
                    interrupting.1.clear();
                }
            },
            r => {
                error!("Panic: CPU[{}] Unexpected exit reason: {:?}", self.vcpu_base.id, r);
                unsafe {
                    libc::exit(0);
                }
            }
        }
        Ok(false)
    }
}

impl Aarch64VirtCpu {
    fn _run(&self) -> Result<(), Error> {
        let mut exit_loop: bool = false;
        loop {
            if !vm::IsRunning() {
                break;
            }
            self.vcpu_base.state.store(KVMVcpuState::GUEST as u64, Ordering::Release);
            let kvm_ret = match self.vcpu_base.vcpu_fd.run() {
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
                        self.vcpu_base.backtrace()?;
                        panic!("vCPU-Run failed - id:{}, error:{:?}", self.vcpu_base.id, e)
                    }
                }
            };
            self.vcpu_base.state.store(KVMVcpuState::HOST as u64, Ordering::Release);
            if let VcpuExit::MmioWrite(addr, data) = kvm_ret {
                {
                    let mut interrupting = self.vcpu_base.interrupting.lock();
                    interrupting.0 = false;
                    interrupting.1.clear();
                }
                let hypercall = (addr - MemoryDef::HYPERCALL_MMIO_BASE) as u16;
                if hypercall > u16::MAX {
                    panic!("cpu[{}] Received hypercall id max than 255", self.vcpu_base.id);
                }
                let (arg0, arg1, arg2, arg3) = self.conf_comp_extension
                    .get_hypercall_arguments(&self.vcpu_base.vcpu_fd, self.vcpu_base.id)?;
                if self.conf_comp_extension.should_handle_hypercall(hypercall) {
                    exit_loop = self.conf_comp_extension.handle_hypercall(hypercall, data, arg0,
                        arg1, arg2, arg3, self.vcpu_base.id)
                        .expect("VM run failed - cannot handle hypercall correctly.");
                } else {
                    exit_loop = self.default_hypercall_handler(hypercall, data, arg0, arg1,
                        arg2, arg3)
                        .expect("VM run failed - cannot handle hypercall correctly.");
                }
            } else if self.conf_comp_extension.should_handle_kvm_exit(&kvm_ret) {
                exit_loop = self.conf_comp_extension.handle_kvm_exit(&kvm_ret, self.vcpu_base.id)?;
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
