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

#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate serde;

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

pub mod kvmlib;

use simplelog::*;
use std::fs::OpenOptions;
use std::env;

pub const LOG_FILE : &'static str = "/var/log/quark/quark.log";

pub fn StartLog(log: &str, level: LevelFilter) {
    //std::fs::remove_file(log).ok();

    //let termLog = TermLogger::new(LevelFilter::Error, Config::default(), TerminalMode::Mixed).unwrap();

    let mut build = ConfigBuilder::new();
    build.set_time_format_str("%H:%M:%S%.3f");
    //build.set_time_format_str("");
    let config = build.build();

    let commonLogfile = OpenOptions::new()
        .append(true)
        .create(true)
        .open(LOG_FILE).unwrap();

    if log == LOG_FILE {
        CombinedLogger::init(
            vec![
                //termLog,
                WriteLogger::new(level, config, commonLogfile)
            ]
        ).unwrap();
    } else {
        let mut build1 = ConfigBuilder::new();
        build1.set_time_format_str("%H:%M:%S%.3f");
        let config1 = build1.build();

        let logfile = OpenOptions::new()
            .append(true)
            .create(true)
            .open(log).unwrap();

        CombinedLogger::init(
            vec![
                //termLog,
                WriteLogger::new(level, config, commonLogfile),
                WriteLogger::new(level, config1, logfile)
            ]
        ).unwrap();
    }
}

fn main() {
    use self::kvmlib::runc::cmd::command::*;

    #[global_allocator]
    static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

    StartLog(LOG_FILE, LevelFilter::Info);

    {
        let mut str = "".to_string();
        let args : Vec<String> = env::args().collect();
        for s in &args {
            str.push_str(s);
            str.push_str(" ");
        }
        error!("commandline args is {}", str);
    }

    let mut args = Parse().unwrap();

    //StartLog(&args.config.DebugLog, args.config.DebugLevel.ToLevelFilter());
    //StartLog(&args.config.DebugLog, LevelFilter::Info);

    /*use std::fs::File;
    use std::io::Write;

    let mut file = File::create(&args.config.DebugLog).unwrap();

    // Write a &str in the file (ignoring the result).
    let log = r#"{"Level": "Debug", "Time":"2018-04-09T23:00:00Z", "Msg": "asdfasdf"}"#;
    writeln!(&mut file, "{}", log).unwrap();*/

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
