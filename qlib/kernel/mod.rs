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

pub mod asm;
pub mod arch;
pub mod boot;
pub mod fs;
pub mod Kernel;
pub mod kernel;
pub mod memmgr;
pub mod quring;
pub mod socket;
pub mod tcpip;
pub mod threadmgr;
pub mod util;
pub mod aqcall;
pub mod fd;
pub mod heap;
pub mod kernel_util;
pub mod mm;
pub mod perflog;
pub mod seqcount;
pub mod SignalDef;
pub mod stack;
pub mod task;
pub mod taskMgr;
pub mod uid;
pub mod vcpu;
pub mod version;
pub mod loader;
pub mod guestfdnotifier;

use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicI32;
use core::sync::atomic::AtomicI64;
use core::sync::atomic::Ordering;

use super::super::ShareSpaceRef;
use super::singleton::*;
use super::pagetable::*;
use self::quring::*;
use self::boot::loader::*;
use self::memmgr::pma::*;

pub static TSC: Tsc = Tsc::New();
pub static SHARESPACE: ShareSpaceRef = ShareSpaceRef::New();
pub static IOURING: IOUringRef = IOUringRef::New();
pub static KERNEL_PAGETABLE: Singleton<PageTables> = Singleton::<PageTables>::New();
pub static PAGE_MGR: Singleton<PageMgr> = Singleton::<PageMgr>::New();
pub static LOADER: Singleton<Loader> = Singleton::<Loader>::New();
pub static KERNEL_STACK_ALLOCATOR: Singleton<AlignedAllocator> =
    Singleton::<AlignedAllocator>::New();

pub static SHUTDOWN: Singleton<AtomicBool> = Singleton::<AtomicBool>::New();
pub static EXIT_CODE: Singleton<AtomicI32> = Singleton::<AtomicI32>::New();

pub fn Shutdown() -> bool {
    return SHUTDOWN.load(self::super::linux_def::QOrdering::RELAXED);
}

#[derive(Default)]
pub struct Tsc {
    pub offset: AtomicI64,
}

impl Tsc {
    pub const fn New() -> Self {
        return  Self {
            offset: AtomicI64::new(0)
        }
    }

    #[inline(always)]
    pub fn RawRdtsc() -> i64 {
        let rax: u64;
        let rdx: u64;
        unsafe {
            llvm_asm!("\
        lfence
        rdtsc
        " : "={rax}"(rax), "={rdx}"(rdx)
        : : )
        };

        return rax as i64 | ((rdx as i64) << 32);
    }

    pub fn SetOffset(&self, offset: i64) {
        self.offset.store(offset, Ordering::SeqCst);
    }

    pub fn Rdtsc(&self) -> i64 {
        return Self::RawRdtsc() - self.offset.load(Ordering::Relaxed);
    }
}