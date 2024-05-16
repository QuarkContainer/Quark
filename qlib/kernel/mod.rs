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

use core::arch::asm;
use core::sync::atomic::Ordering;
use core::sync::atomic::{AtomicBool, AtomicI32, AtomicI64};

use crate::qlib::fileinfo::*;
use crate::qlib::rdma_svc_cli::RDMASvcClient;

use super::super::kernel_def::VcpuFreq;
use super::super::ShareSpaceRef;
use super::control_msg::*;
use super::pagetable::*;
use super::singleton::*;

use self::arch::__arch::arch_def::ArchFPState;
use self::boot::loader::*;
use self::kernel::async_process::*;
use self::memmgr::pma::*;
use self::quring::*;
use self::taskMgr::*;

pub mod Kernel;
pub mod SignalDef;
pub mod arch;
pub mod asm;
pub mod boot;
pub mod fd;
pub mod fs;
pub mod guestfdnotifier;
pub mod kernel;
pub mod kernel_util;
pub mod loader;
pub mod memmgr;
pub mod mm;
pub mod perflog;
pub mod quring;
pub mod seqcount;
pub mod socket;
pub mod stack;
pub mod task;
pub mod taskMgr;
pub mod tcpip;
pub mod threadmgr;
pub mod uid;
pub mod util;
pub mod vcpu;
pub mod version;
pub mod dns;

pub static TSC: Tsc = Tsc::New();
pub static SHARESPACE: ShareSpaceRef = ShareSpaceRef::New();
pub static IOURING: IOUringRef = IOUringRef::New();
pub static KERNEL_PAGETABLE: Singleton<PageTables> = Singleton::<PageTables>::New();
// used for allocate new page table see pub fn Fork(&self, pagePool: &PageMgr) -> Result<Self> {
pub static PAGE_MGR: PageMgrRef = PageMgrRef::New();    
pub static LOADER: Singleton<Loader> = Singleton::<Loader>::New();
pub static WAIT_CONTAINER_FD: AtomicI32 = AtomicI32::new(-1);
pub static KERNEL_STACK_ALLOCATOR: Singleton<AlignedAllocator> =
    Singleton::<AlignedAllocator>::New();

pub static EXIT_CODE: Singleton<AtomicI32> = Singleton::<AtomicI32>::New();

#[cfg(target_arch = "x86_64")]
pub static VCPU_FREQ: AtomicI64 = AtomicI64::new(2_000_000_000); // default 2GHZ

#[cfg(target_arch = "aarch64")]
// CAUTIONS!!: for arm, the rdtsc counter is replaced with the system counter
// (cntvct_el0), and vcpu frequency" with the system counter frequency (cntfreq)
// Counter frequency is implementation defined, the cntfreq value is provided by
// the EL2+ firmware, and is not necessarily related to the actual cpu or vcpu
// frequency. Nevertheless, we are keeping the namings for convenience atm. Do
// not mistake VCPU_FREQ for the actuall vcpu frequency.
pub static VCPU_FREQ: AtomicI64 = AtomicI64::new(25_000_000); // default 25MHZ

pub static ASYNC_PROCESS: AsyncProcess = AsyncProcess::New();
pub static FP_STATE: ArchFPState = ArchFPState::Init();
pub static SUPPORT_XSAVE: AtomicBool = AtomicBool::new(false);
pub static SUPPORT_XSAVEOPT: AtomicBool = AtomicBool::new(false);

pub fn SetWaitContainerfd(fd: i32) {
    WAIT_CONTAINER_FD.store(fd, Ordering::SeqCst)
}

pub fn WaitContainerfd() -> i32 {
    WAIT_CONTAINER_FD.load(Ordering::SeqCst)
}

pub fn GlobalIOMgr<'a>() -> &'a IOMgr {
    return &SHARESPACE.ioMgr;
}

pub fn GlobalRDMASvcCli<'a>() -> &'a RDMASvcClient {
    return &SHARESPACE.rdmaSvcCli;
}

#[inline]
pub fn Timestamp() -> i64 {
    Scale(TSC.Rdtsc())
}

#[inline]
pub fn Scale(tsc: i64) -> i64 {
    (tsc as i128 * 1000_000 / LoadVcpuFreq() as i128) as i64
}

pub fn Ns2tsc(ns: i64) -> i64 {
    return (ns as i128 * LoadVcpuFreq() as i128 / 1000_000_000) as i64;
}

pub fn VcpuFreqInit() {
    VCPU_FREQ.store(VcpuFreq(), Ordering::SeqCst);
}

#[inline]
pub fn LoadVcpuFreq() -> i64 {
    return VCPU_FREQ.load(Ordering::Relaxed);
}

#[inline]
pub fn Shutdown() -> bool {
    let ret = SHARESPACE.IsShutdown();
    return ret;
}

#[derive(Default)]
pub struct Tsc {
    pub offset: AtomicI64,
}

impl Tsc {
    pub const fn New() -> Self {
        return Self {
            offset: AtomicI64::new(0),
        };
    }

    // return : count of us
    pub fn Scale(tsc: i64) -> i64 {
        return Scale(tsc);
    }

    #[inline(always)]
    #[cfg(target_arch = "x86_64")]
    pub fn RawRdtsc() -> i64 {
        let rax: u64;
        let rdx: u64;
        unsafe {
            asm!("
                lfence
                rdtsc
                ",
                out("rax") rax,
                out("rdx") rdx
            )
        };

        return rax as i64 | ((rdx as i64) << 32);
    }

    #[inline(always)]
    #[cfg(target_arch = "aarch64")]
    pub fn RawRdtsc() -> i64 {
        let val: u64;
        unsafe {
            asm!("mrs {0}, cntvct_el0",
            out(reg) val
            )
        };

        return val as i64;
    }

    pub fn SetOffset(&self, offset: i64) {
        self.offset.store(offset, Ordering::SeqCst);
    }

    pub fn Rdtsc(&self) -> i64 {
        return Self::RawRdtsc() - self.offset.load(Ordering::SeqCst);
    }
}

pub fn SignalProcess(signalArgs: &SignalArgs) {
    *SHARESPACE.signalArgs.lock() = Some(signalArgs.clone());
    CreateTask(SHARESPACE.SignalHandlerAddr(), 0 as *const u8, true);
}
