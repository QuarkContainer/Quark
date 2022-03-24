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
#![feature(proc_macro_hygiene)]
#![feature(naked_functions)]
#![allow(bare_trait_objects)]
#![feature(map_first_last)]
#![allow(non_camel_case_types)]
#![feature(llvm_asm)]
#![allow(deprecated)]
#![feature(thread_id_value)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![feature(core_intrinsics)]


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

extern crate libc;
extern crate spin;
extern crate x86_64;
extern crate capabilities;
extern crate caps;
extern crate fs2;
extern crate regex;
extern crate simplelog;
extern crate tabwriter;

#[macro_use]
pub mod asm;

#[macro_use]
pub mod print;

pub mod amd64_def;
pub mod console;
pub mod elf_loader;
pub mod heap_alloc;
mod kvm_vcpu;
mod memmgr;
pub mod namespace;
mod qcall;
pub mod qlib;
pub mod runc;
mod syncmgr;
pub mod ucall;
pub mod util;
mod vmspace;
pub mod kernel_def;

use alloc::sync::Arc;
use lazy_static::lazy_static;
use spin::Mutex;
use std::env;
use std::cell::RefCell;

use self::heap_alloc::*;
use self::qlib::addr;
use self::qlib::buddyallocator::MemAllocator;
use self::qlib::config::*;
use self::qlib::qmsg::*;
use self::qlib::ShareSpace;
use self::qlib::ShareSpaceRef;
use self::runc::cmd::command::*;
use self::vmspace::host_pma_keeper::*;
use self::vmspace::hostfdnotifier::*;
use self::vmspace::kernel_io_thread::*;
use self::runc::shim::service::*;

use self::vmspace::uringMgr::*;
use vmspace::*;

const LOWER_TOP: u64 = 0x00007fffffffffff;
const UPPER_BOTTOM: u64 = 0xffff800000000000;

pub fn AllocatorPrint(_class: usize) -> String {
    return "".to_string();
}

pub static SHARE_SPACE: ShareSpaceRef = ShareSpaceRef::New();

thread_local!(static THREAD_ID: RefCell<i32> = RefCell::new(-1));

pub fn ThreadId() -> i32 {
    let mut i = 0;
    THREAD_ID.with(|f| {
        i = *f.borrow();
    });
    return i;
}

lazy_static! {
    pub static ref SHARE_SPACE_STRUCT: Arc<Mutex<ShareSpace>> =
        Arc::new(Mutex::new(ShareSpace::New()));
    pub static ref VMS: Mutex<VMSpace> = Mutex::new(VMSpace::Init());
    pub static ref ROOT_CONTAINER_ID: Mutex<String> = Mutex::new(String::new());
    pub static ref PAGE_ALLOCATOR: MemAllocator = MemAllocator::New();
    pub static ref FD_NOTIFIER: HostFdNotifier = HostFdNotifier::New();
    pub static ref IO_MGR: vmspace::HostFileMap::IOMgr =
        vmspace::HostFileMap::IOMgr::Init().expect("Init IOMgr fail");
    pub static ref SYNC_MGR: Mutex<syncmgr::SyncMgr> = Mutex::new(syncmgr::SyncMgr::New());
    pub static ref PMA_KEEPER: HostPMAKeeper = HostPMAKeeper::New();
    pub static ref QUARK_CONFIG: Mutex<Config> = {
        let mut config = Config::default();
        config.Load();
        Mutex::new(config)
    };
    pub static ref URING_MGR: Arc<Mutex<UringMgr>> = {
        let config = QUARK_CONFIG.lock();
        let uringSize = config.UringSize;
        Arc::new(Mutex::new(UringMgr::New(uringSize)))
    };
    pub static ref KERNEL_IO_THREAD: KIOThread = KIOThread::New();
    pub static ref GLOCK: Mutex<()> = Mutex::new(());
}

pub const LOG_FILE: &'static str = "/var/log/quark/quark.log";

pub fn InitSingleton() {
    self::qlib::InitSingleton();
}

#[global_allocator]
static GLOBAL: HostAllocator = HostAllocator::New();

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
    if shimMode == true && &cmd != "boot"  {
        error!("*********shim mode***************");
        containerd_shim::run::<Service>("io.containerd.empty.v1", None)

    } else {
        let mut args = Parse().unwrap();
        match Run(&mut args) {
            Err(e) => {
                error!("the error is {:?}", e);
                ::std::process::exit(-1);
            }
            Ok(()) => {
                //error!("successfully ...");
                ::std::process::exit(0);
            }
        }
    }
}
