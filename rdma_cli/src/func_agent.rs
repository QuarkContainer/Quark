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

#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(deref_nullptr)]
#![feature(proc_macro_hygiene)]
#![feature(naked_functions)]
#![allow(bare_trait_objects)]
#![feature(map_first_last)]
#![allow(non_camel_case_types)]
#![allow(deprecated)]
#![feature(thread_id_value)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![feature(core_intrinsics)]
#![recursion_limit = "256"]


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
#[macro_use]
extern crate log;
extern crate caps;
extern crate fs2;
extern crate regex;
extern crate simplelog;
extern crate tabwriter;

#[macro_use]
pub mod print;

#[macro_use]
pub mod asm;
pub mod kernel_def;
pub mod qlib;

pub mod common;
pub mod constants;
pub mod rdma_ctrlconn;
pub mod rdma_def;
pub mod ingress_informer;
pub mod rdma_ingress_informer;
pub mod service_informer;
pub mod unix_socket_def;
use qlib::common::*;

pub mod funclib;

use self::qlib::ShareSpaceRef;
use alloc::slice;
use alloc::sync::Arc;
use fs2::FileExt;
use spin::Mutex;
use std::collections::HashMap;
use std::io;
use std::io::prelude::*;
use std::io::Error;
use std::net::{IpAddr, Ipv4Addr, TcpListener, TcpStream};
use std::os::unix::io::{AsRawFd, RawFd};
pub static SHARE_SPACE: ShareSpaceRef = ShareSpaceRef::New();
use self::qlib::mem::list_allocator::*;
use crate::qlib::rdma_share::*;
use common::EpollEvent;
use common::*;
use qlib::linux_def::*;
use qlib::rdma_svc_cli::*;
use qlib::socket_buf::{SocketBuff, SocketBuffIntern};
use qlib::unix_socket::UnixSocket;
use std::str::FromStr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::{env, mem, ptr, thread, time};
use rdma_ctrlconn::*;
use ingress_informer::IngressInformer;
use rdma_ingress_informer::RdmaIngressInformer;
use service_informer::ServiceInformer;
use crate::constants::*;

use crate::funclib::func_agent::*;
use crate::funclib::agent::*;

pub static GLOBAL_ALLOCATOR: HostAllocator = HostAllocator::New();

lazy_static! {
    pub static ref GLOBAL_LOCK: Mutex<()> = Mutex::new(());
    pub static ref RDMA_CTLINFO: CtrlInfo = CtrlInfo::default();
}

#[tokio::main]
async fn main() -> Result<()> {
    tokio::select! {
        _a = Execution() => {}
        _b = ChannelProcess() => {}
    };

    return Ok(())
}
