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

#![allow(dead_code)]
#![allow(non_snake_case)]
extern crate alloc;
extern crate bit_field;
extern crate errno;
extern crate core_affinity;

#[macro_use]
pub mod asm;

pub mod qlib;
mod memmgr;
mod heap_alloc;
mod qcall;
mod vmspace;
mod kvm_vcpu;
mod syncmgr;
pub mod namespace;
pub mod elf_loader;
pub mod runc;
pub mod ucall;
pub mod console;
pub mod util;
pub mod amd64_def;
pub mod perflog;
//pub mod uring;

use spin::Mutex;
use lazy_static::lazy_static;
use core::sync::atomic::AtomicU64;

use self::qlib::buddyallocator::MemAllocator;
use self::qlib::{addr};
use self::qlib::qmsg::*;
use self::vmspace::hostfdnotifier::*;
use self::vmspace::host_pma_keeper::*;
use self::vmspace::kernel_io_thread::*;
use vmspace::*;
use self::vmspace::uringMgr::*;

const LOWER_TOP: u64 = 0x00007fffffffffff;
const UPPER_BOTTOM: u64 = 0xffff800000000000;

lazy_static! {
    pub static ref SHARE_SPACE : AtomicU64 = AtomicU64::new(0);
    pub static ref VMS: Mutex<VMSpace> = Mutex::new(VMSpace::Init());
    pub static ref PAGE_ALLOCATOR: MemAllocator = MemAllocator::New();
    pub static ref FD_NOTIFIER: HostFdNotifier = HostFdNotifier::New();
    pub static ref SYNC_MGR: Mutex<syncmgr::SyncMgr> = Mutex::new(syncmgr::SyncMgr::New());
    pub static ref PMA_KEEPER: HostPMAKeeper = HostPMAKeeper::New();
    pub static ref URING_MGR: Mutex<UringMgr> = Mutex::new(UringMgr::New(64));
    pub static ref KERNEL_IO_THREAD: KIOThread = KIOThread::New();
    pub static ref GLOCK: Mutex<()> = Mutex::new(());
}
