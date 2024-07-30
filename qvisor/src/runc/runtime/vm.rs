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
use std::os::unix::io::FromRawFd;
use std::thread;

use kvm_bindings::*;
use kvm_ioctls::{Cap, Kvm, VmFd};
use lazy_static::lazy_static;
use nix::sys::signal;

use crate::arch::VirtCpu;
use crate::arch::vm::vcpu::ArchVirtCpu;
#[cfg (feature = "cc")]
use crate::qlib::kernel::Kernel::{ENABLE_CC, IDENTICAL_MAPPING};
use crate::qlib::MAX_VCPU_COUNT;
use crate::runc::runtime::vm_type::emulcc::VmCcEmul;
use crate::tsot_agent::TSOT_AGENT;
use crate::FD_NOTIFIER;
//use crate::vmspace::hibernate::HiberMgr;

use super::super::super::elf_loader::*;
use super::super::super::kvm_vcpu::*;
use super::super::super::print::LOG;
use super::super::super::qlib::addr;
use super::super::super::qlib::common::*;
use super::super::super::qlib::config::CCMode;
use super::super::super::qlib::kernel::kernel::futex;
use super::super::super::qlib::kernel::kernel::timer;
use super::super::super::qlib::kernel::task;
use super::super::super::qlib::kernel::vcpu;
use super::super::super::qlib::kernel::IOURING;
use super::super::super::qlib::kernel::KERNEL_PAGETABLE;
use super::super::super::qlib::kernel::KERNEL_STACK_ALLOCATOR;
use super::super::super::qlib::kernel::PAGE_MGR;
use super::super::super::qlib::kernel::SHARESPACE;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::pagetable::AlignedAllocator;
use super::super::super::qlib::pagetable::PageTables;
use super::super::super::qlib::perf_tunning::*;
use super::super::super::qlib::task_mgr::*;
use super::super::super::qlib::ShareSpace;
use super::super::super::runc::runtime::loader::*;
use super::super::super::syncmgr;
use super::super::super::vmspace::*;
use super::super::super::SHARE_SPACE;
use super::super::super::SHARE_SPACE_STRUCT;
use super::super::super::{
    ThreadId, KERNEL_IO_THREAD, PMA_KEEPER, QUARK_CONFIG, ROOT_CONTAINER_ID, THREAD_ID, URING_MGR,
    VCPU, VMS,
};
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
    pub fn SetMemRegion(
        slotId: u32,
        vm_fd: &VmFd,
        phyAddr: u64,
        hostAddr: u64,
        pageMmapsize: u64,
    ) -> Result<()> {
        info!(
            "SetMemRegion phyAddr = {:x}, hostAddr={:x}; pageMmapsize = {:x} MB",
            phyAddr,
            hostAddr,
            (pageMmapsize >> 20)
        );

        // guest_phys_addr must be <512G
        let mem_region = kvm_userspace_memory_region {
            slot: slotId,
            guest_phys_addr: phyAddr,
            memory_size: pageMmapsize,
            userspace_addr: hostAddr,
            flags: 0, //kvm_bindings::KVM_MEM_LOG_DIRTY_PAGES,
        };

        unsafe {
            vm_fd
                .set_user_memory_region(mem_region)
                .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        }

        return Ok(());
    }

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
        info!(
            "Reset umask from {:o} to {}", umask, 0);
        info!("VM will be created with parameters:{:?}", vm_type);
        let vm = vm_type.create_vm(kernel_elf, args)
                        .expect("VM: faield to create.");
        info!("Vm creation done.");
        PerfGofrom(PerfType::Other);
        Ok(vm)
    }

    pub fn run(&mut self) -> Result<i32> {
        // start the io thread
        let cpu = self.vcpus[0].clone();
        SetSigusr1Handler();
        let mut threads = Vec::new();
        let tgid = unsafe { libc::gettid() };
        threads.push(
            thread::Builder::new()
                .name("0".to_string())
                .spawn(move || {
                    THREAD_ID.with(|f| {
                        *f.borrow_mut() = 0;
                    });
                    VCPU.with(|f| {
                        *f.borrow_mut() = Some(cpu.clone());
                    });
                    cpu.vcpu_run(tgid, None).expect("vcpu run fail");
                    info!("cpu0 finish");
                })
                .unwrap(),
        );

        syncmgr::SyncMgr::WaitShareSpaceReady();
        info!("shareSpace ready...");
        // start the vcpu threads
        for i in 1..self.vcpus.len() {
            let cpu = self.vcpus[i].clone();

            threads.push(
                thread::Builder::new()
                    .name(format!("{}", i))
                    .spawn(move || {
                        THREAD_ID.with(|f| {
                            *f.borrow_mut() = i as i32;
                        });
                        VCPU.with(|f| {
                            *f.borrow_mut() = Some(cpu.clone());
                        });
                        info!("cpu#{} start", ThreadId());
                        cpu.vcpu_run(tgid, None).expect("vcpu run fail");
                        info!("cpu#{} finish", ThreadId());
                    })
                    .unwrap(),
            );
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

#[cfg(target_arch = "aarch64")]
fn get_kvm_vcpu_init(vmfd: &VmFd) -> Result<kvm_vcpu_init> {
    let mut kvi = kvm_vcpu_init::default();
    vmfd.get_preferred_target(&mut kvi)
        .map_err(|e| Error::SysError(e.errno()))?;
    kvi.features[0] |= 1 << kvm_bindings::KVM_ARM_VCPU_PSCI_0_2;
    Ok(kvi)
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
