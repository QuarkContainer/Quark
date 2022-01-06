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
#![feature(proc_macro_hygiene, asm)]
#![feature(naked_functions)]
#![allow(bare_trait_objects)]
#![feature(map_first_last)]
#![allow(non_camel_case_types)]
#![feature(llvm_asm)]
#![allow(deprecated)]
#![feature(thread_id_value)]

#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate serde;
extern crate cache_padded;

#[macro_use]
extern crate clap;

#[macro_use]
extern crate scopeguard;

#[macro_use]
extern crate lazy_static;

extern crate alloc;
extern crate spin;
extern crate x86_64;
extern crate errno;
extern crate libc;
#[macro_use]
extern crate log;
extern crate simplelog;
extern crate capabilities;
extern crate regex;
extern crate fs2;
extern crate caps;
extern crate tabwriter;

#[macro_use]
pub mod print;

pub mod kvmlib;

use std::env;

use kvmlib::heap_alloc::*;

pub const LOG_FILE : &'static str = "/var/log/quark/quark.log";

pub fn InitSingleton() {
    kvmlib::qlib::InitSingleton();
}

#[global_allocator]
static GLOBAL: HostAllocator = HostAllocator::New();

fn main() {
    use self::kvmlib::runc::cmd::command::*;

    InitSingleton();

    {
        let mut str = "".to_string();
        let args : Vec<String> = env::args().collect();
        for s in &args {
            str.push_str(s);
            str.push_str(" ");
        }
        info!("commandline args is {}", str);
    }

    let mut args = Parse().unwrap();
    match Run(&mut args) {
        Err(e) => {
            error!("the error is {:?}", e);
            ::std::process::exit(-1);
        },
        Ok(()) => {
            //error!("successfully ...");
            ::std::process::exit(0);
        }
    }
}
