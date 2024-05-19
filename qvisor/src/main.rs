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
#![allow(deref_nullptr)]
#![allow(bare_trait_objects)]
#![allow(non_camel_case_types)]
#![allow(deprecated)]
#![allow(dead_code)]
#![recursion_limit = "256"]
#![feature(unix_socket_ancillary_data)]
#![feature(allocator_api)]
#![feature(stmt_expr_attributes)]
#![allow(invalid_reference_casting)]
#![feature(btreemap_alloc)]

extern crate alloc;
extern crate bit_field;
extern crate core_affinity;
extern crate errno;

#[macro_use]
extern crate serde_derive;
extern crate cache_padded;
extern crate serde;
extern crate serde_json;

#[macro_use]
extern crate clap;

#[macro_use]
extern crate scopeguard;

#[macro_use]
extern crate lazy_static;

extern crate capabilities;
extern crate caps;
extern crate fs2;
extern crate libc;
extern crate regex;
extern crate simplelog;
extern crate spin;
extern crate tabwriter;

#[macro_use]
pub mod asm;

#[macro_use]
pub mod print;
#[cfg(target_arch="x86_64")]
pub mod amd64_def;
pub mod console;
pub mod elf_loader;
pub mod heap_alloc;
pub mod kernel_def;
mod kvm_vcpu;
#[cfg(target_arch = "aarch64")]
mod kvm_vcpu_aarch64;
#[cfg(target_arch = "x86_64")]
mod kvm_vcpu_x86;
mod memmgr;
pub mod namespace;
mod qcall;
pub mod qlib;
pub mod rdma_def;
pub mod runc;
mod syncmgr;
pub mod ucall;
pub mod unix_socket_def;
pub mod util;
mod vmspace;

use alloc::sync::Arc;
use lazy_static::lazy_static;
use spin::Mutex;
use std::cell::RefCell;
use std::env;

use self::qlib::addr;
use self::qlib::buddyallocator::MemAllocator;
use self::qlib::config::*;
use self::qlib::mem::list_allocator::*;
use self::qlib::qmsg::*;
use self::qlib::ShareSpaceRef;
use self::runc::cmd::command::*;
use self::runc::sandbox::sandbox::*;
use self::runc::shim::service::*;
use self::vmspace::hibernate::*;
use self::vmspace::host_pma_keeper::*;
use self::vmspace::hostfdnotifier::*;
use self::vmspace::kernel_io_thread::*;
use crate::qlib::linux_def::MemoryDef;
use self::qlib::kernel::kernel::kernel::Kernel;
use spin::rwlock::RwLock;

use self::vmspace::uringMgr::*;
use crate::kvm_vcpu::KVMVcpu;
use vmspace::*;

const LOWER_TOP: u64 = 0x00007fffffffffff;
const UPPER_BOTTOM: u64 = 0xffff800000000000;

pub fn AllocatorPrint(_class: usize) -> String {
    return "".to_string();
}

pub static IS_GUEST: bool = false;
pub const ENABLE_EMULATION_CC: bool = false;

pub static SHARE_SPACE: ShareSpaceRef = ShareSpaceRef::New();

thread_local!(static THREAD_ID: RefCell<i32> = RefCell::new(0));
thread_local!(static VCPU: RefCell<Option<Arc<KVMVcpu>>> = RefCell::new(None));



pub fn ThreadId() -> i32 {
    let mut i = 0;
    THREAD_ID.with(|f| {
        i = *f.borrow();
    });
    return i;
}

pub fn LocalVcpu() -> Option<Arc<KVMVcpu>> {
    let mut vcpu = None;
    VCPU.with(|f| {
        vcpu = f.borrow().clone();
    });
    return vcpu;
}

#[cfg(not(feature = "cc"))]
lazy_static! {
    pub static ref SHARE_SPACE_STRUCT: Arc<Mutex<crate::qlib::ShareSpace>> =
    Arc::new(Mutex::new(crate::qlib::ShareSpace::New()));

}

lazy_static! {
    pub static ref GLOBAL_LOCK: Mutex<()> = Mutex::new(());
    pub static ref SWAP_FILE: Mutex<SwapFile> = Mutex::new(SwapFile::Init().unwrap());
    pub static ref VMS: RwLock<VMSpace> = RwLock::new(VMSpace::Init());
    pub static ref ROOT_CONTAINER_ID: Mutex<String> = Mutex::new(String::new());
    pub static ref PAGE_ALLOCATOR: MemAllocator = MemAllocator::New();
    pub static ref FD_NOTIFIER: HostFdNotifier = HostFdNotifier::New();
    pub static ref SYNC_MGR: Mutex<syncmgr::SyncMgr> = Mutex::new(syncmgr::SyncMgr::New());
    pub static ref PMA_KEEPER: HostPMAKeeper = HostPMAKeeper::New();
    pub static ref QUARK_CONFIG: Mutex<Config> = {
        let mut config = Config::default();
        config.Load();
        Mutex::new(config)
    };
    pub static ref URING_MGR: Arc<Mutex<UringMgr>> = {
        let uringQueueSize = 1024;

        debug!("init URING_MGR");
        Arc::new(Mutex::new(UringMgr::New(uringQueueSize)))
    };
    pub static ref KERNEL_IO_THREAD: KIOThread = KIOThread::New();
    pub static ref GLOCK: Mutex<()> = Mutex::new(());
    pub static ref NIVIDIA_CONTAINER_NAME: Mutex<String> = Mutex::new(String::new());
    pub static ref SANDBOX: Mutex<Sandbox> = Mutex::new(Sandbox::default());

    pub static ref URING: Mutex::<io_uring::IoUring> = Mutex::new(io_uring::IoUring::new(MemoryDef::QURING_SIZE as u32).expect("setup io_Uring fail"));
    // used for keep consistancy
    pub static ref GUEST_KERNEL: Mutex<Option<Kernel>> = Mutex::new(None);

    pub static ref PRIVATE_VCPU_LOCAL_HOLDER: Box<PrivateCPULocal> = Box::new(PrivateCPULocal::New());  
}

pub const LOG_FILE: &'static str = "/var/log/quark/quark.log";

pub fn InitSingleton() {
    self::qlib::InitSingleton();
}

//pub static ALLOCATOR: HostAllocator = HostAllocator::New();
#[global_allocator]
pub static GLOBAL_ALLOCATOR: HostAllocator = HostAllocator::New();
pub static GUEST_HOST_SHARED_ALLOCATOR: GuestHostSharedAllocator = GuestHostSharedAllocator::New();

fn main() {
    InitSingleton();

    let cmd;

    {
        let mut str = "".to_string();
        let args: Vec<String> = env::args().collect();
        cmd = args[1].clone();
        for s in &args {
            str.push_str(s);
            str.push_str(" ");
        }
        info!("commandline args is {}", str);
    }

    let shimMode = QUARK_CONFIG.lock().ShimMode;
    if shimMode == true && &cmd != "boot" {
        error!("*********shim mode***************");
        containerd_shim::run::<Service>("io.containerd.empty.v1", None)
    } else {
        let mut args = match Parse() {
            Ok(args) => args,
            Err(e) => {
                error!("the parse error is {:?}", e);
                panic!("exitting...")
            }
        };
        match Run(&mut args) {
            Err(e) => {
                error!("the error is {:?}", e);
                ::std::process::exit(-1);
            }
            Ok(()) => {
                ::std::process::exit(0);
            }
        }
    }
}
