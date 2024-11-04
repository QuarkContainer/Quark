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

use alloc::sync::Arc;
use core::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::os::unix::io::AsRawFd;
use std::thread;
use std::thread::JoinHandle;

use kvm_bindings::*;
use kvm_ioctls::{Kvm, VmFd};
use lazy_static::lazy_static;
use nix::sys::signal;

use crate::arch::VirtCpu;
use crate::arch::vm::vcpu::ArchVirtCpu;
#[cfg (feature = "cc")]
use crate::qlib::MAX_VCPU_COUNT;
use crate::runc::runtime::vm_type::emulcc::VmCcEmul;

use super::super::super::elf_loader::*;
use super::super::super::kvm_vcpu::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::config::CCMode;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::perf_tunning::*;
use super::super::super::qlib::task_mgr::*;
use super::super::super::qlib::ShareSpace;
use super::super::super::runc::runtime::loader::*;
use super::super::super::syncmgr;
use super::super::super::{ThreadId, THREAD_ID, URING_MGR,
    VCPU, VMS,};
use super::vm_type::noncc::VmNormal;
use super::vm_type::VmType;

pub const SANDBOX_UID_NAME: &str = "io.kubernetes.cri.sandbox-uid";

lazy_static! {
    static ref EXIT_STATUS: AtomicI32 = AtomicI32::new(-1);
    static ref DUMP: AtomicU64 = AtomicU64::new(0);
}

#[inline]
pub fn IsRunning() -> bool {
    return EXIT_STATUS.load(Ordering::Relaxed) == -1;
}

pub fn SetExitStatus(status: i32) {
    EXIT_STATUS.store(status, Ordering::Release);
}

pub fn GetExitStatus() -> i32 {
    return EXIT_STATUS.load(Ordering::Acquire);
}

pub fn Dump(id: usize) -> bool {
    assert!(id < MAX_VCPU_COUNT);
    return DUMP.load(Ordering::Acquire) & (0x1 << id) > 0;
}

pub fn SetDumpAll() {
    DUMP.store(u64::MAX, Ordering::Release);
}

pub fn ClearDump(id: usize) {
    assert!(id < MAX_VCPU_COUNT);
    DUMP.fetch_and(!(0x1 << id), Ordering::Release);
}

pub struct VirtualMachine {
    pub kvm: Kvm,
    pub vmfd: VmFd,
    pub vm_type: Box<dyn VmType>,
    pub vcpus: Vec<Arc<ArchVirtCpu>>,
    pub elf: KernelELF,
}

impl VirtualMachine {
    pub fn Umask() -> u32 {
        let umask = unsafe { libc::umask(0) };
        return umask;
    }

    pub const KVM_IOEVENTFD_FLAG_DATAMATCH: u32 = (1 << kvm_ioeventfd_flag_nr_datamatch);
    pub const KVM_IOEVENTFD_FLAG_PIO: u32 = (1 << kvm_ioeventfd_flag_nr_pio);
    pub const KVM_IOEVENTFD_FLAG_DEASSIGN: u32 = (1 << kvm_ioeventfd_flag_nr_deassign);
    pub const KVM_IOEVENTFD_FLAG_VIRTIO_CCW_NOTIFY: u32 =
        (1 << kvm_ioeventfd_flag_nr_virtio_ccw_notify);

    pub const KVM_IOEVENTFD: u64 = 0x4040ae79;

    pub fn IoEventfdAddEvent(vmfd: i32, addr: u64, eventfd: i32) {
        let kvmIoEvent = kvm_ioeventfd {
            addr: addr,
            len: 8,
            datamatch: 1,
            fd: eventfd,
            flags: Self::KVM_IOEVENTFD_FLAG_DATAMATCH,
            ..Default::default()
        };

        let ret = unsafe { libc::ioctl(vmfd, Self::KVM_IOEVENTFD, &kvmIoEvent as *const _ as u64) };

        assert!(
            ret == 0,
            "IoEventfdAddEvent ret is {}/{}/{}",
            ret,
            errno::errno().0,
            vmfd.as_raw_fd()
        );
    }

    #[cfg(debug_assertions)]
    pub const KERNEL_IMAGE: &'static str = "/usr/local/bin/qkernel_d.bin";

    #[cfg(not(debug_assertions))]
    pub const KERNEL_IMAGE: &'static str = "/usr/local/bin/qkernel.bin";

    pub const VDSO_PATH: &'static str = "/usr/local/bin/vdso.so";

    pub fn Init(args: Args, cc_mode: CCMode) -> Result<Self> {
        PerfGoto(PerfType::Other);
        let (vm_type, kernel_elf) = match cc_mode {
            CCMode::None => VmNormal::init(Some(&args))?,
            CCMode::Normal | CCMode::NormalEmu =>
                VmCcEmul::init(Some(&args))?,
            _ => panic!("Unhandled type."),
        };
        let umask = Self::Umask();
        info!("VMM: Reset umask from {:o} to {}", umask, 0);
        info!("VMM: VM will be created with parameters:{:?}", vm_type);
        let vm = vm_type.create_vm(kernel_elf, args).expect("VM: faield to create.");
        info!("VMM: Vm creation done.");
        PerfGofrom(PerfType::Other);
        Ok(vm)
    }

    fn spawn_vm_vcpus(&mut self, threads: &mut Vec<JoinHandle<()>>, from: usize,
        count: usize, tgid: i32) {
        let _vm_fd_raw = self.vmfd.as_raw_fd();
        let _kvm_fd_raw = self.kvm.as_raw_fd();
        let vm_type = self.vm_type.get_type();
        for i in from..count {
            let cpu_name = i;
            let cpu_obj = self.vcpus[cpu_name].clone();
            threads.push(
                thread::Builder::new()
                .name(cpu_name.to_string())
                .spawn(move || {
                    THREAD_ID.with(|f| {
                        *f.borrow_mut() = cpu_name as i32;
                    });
                    VCPU.with(|f| {
                        *f.borrow_mut() = Some(cpu_obj.clone());
                    });
                    info!("VMM: vCPU#{} - ThreadID:{} started", cpu_name, ThreadId());
                    match vm_type {
                        _ => { cpu_obj.vcpu_run(tgid, None, None).expect("VMM: vCPU failed to run."); }
                    };

                    info!("VMM: vCPU#{} - ThreadID:{} finished", cpu_name, ThreadId());
                }).expect("VMM: Failed to spawn thread for vCPU")
            );
        }
    }

    ///  vCPU0 - Boot vCPU, prepares the enviroment for all the other vcpus.
    ///  Running order: Based on VmType::CCMode
    ///     RealmVM - All expect boot vCPU enter KVM in powered-off state.
    ///         Boot vCPU boots after a certain delay to unsure others have called kvm_run.
    ///     SevSnp -
    ///     TDX -
    ///     Others - Boot vCPU runs first, prepares shared space infrastructure, then allows
    ///         the remaining cpus to run.
    pub fn run(&mut self) -> Result<i32> {
        SetSigusr1Handler();
        let mut threads: Vec<JoinHandle<()>> = Vec::new();
        let tgid = unsafe { libc::gettid() };

        match self.vm_type.get_type() {
            _ =>  {
                self.spawn_vm_vcpus(&mut threads, 0, 1, tgid);
                syncmgr::SyncMgr::WaitShareSpaceReady();
                info!("VMM: Shared-space ready...");
                self.spawn_vm_vcpus(&mut threads, 1, self.vcpus.len(), tgid);
            }
        }

        for t in threads {
            t.join().expect("the working threads has panicked");
        }

        URING_MGR.lock().Close();
        Ok(GetExitStatus())
    }

    pub fn WakeAll(shareSpace: &ShareSpace) {
        shareSpace.scheduler.WakeAll();
    }

    pub fn Schedule(shareSpace: &ShareSpace, taskId: TaskId, cpuAff: bool) {
        shareSpace
            .scheduler
            .ScheduleQ(taskId, taskId.Queue(), cpuAff);
    }

    pub fn PrintQ(shareSpace: &ShareSpace, vcpuId: u64) -> String {
        return shareSpace.scheduler.PrintQ(vcpuId);
    }
}

fn SetSigusr1Handler() {
    let sig_action = signal::SigAction::new(
        signal::SigHandler::Handler(handleSigusr1),
        signal::SaFlags::empty(),
        signal::SigSet::empty(),
    );
    unsafe {
        signal::sigaction(signal::Signal::SIGUSR1, &sig_action)
            .expect("sigusr1 sigaction set fail");
    }
}

extern "C" fn handleSigusr1(_signal: i32) {
    SetDumpAll();
    let vms = VMS.lock();
    for vcpu in &vms.vcpus {
        if vcpu.vcpu_base.state.load(Ordering::Acquire) == KVMVcpuState::HOST as u64 {
            vcpu.vcpu_base.dump().unwrap_or_default();
        }
        vcpu.vcpu_base.Signal(Signal::SIGCHLD);
    }
    drop(vms);
}
