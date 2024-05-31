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

use alloc::string::String;
use alloc::vec::Vec;
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};
use kvm_ioctls::Kvm;
use std::fs;

use super::super::super::qlib::common::*;
use super::super::super::qlib::config::*;
use super::super::cmd::config::*;
use super::super::runtime::loader::*;
use super::super::runtime::vm::*;
use super::command::*;

#[derive(Debug)]
pub struct CmdCmd {
    pub cmd: Vec<String>,
}

impl CmdCmd {
    pub fn Init(cmd_matches: &ArgMatches) -> Result<Self> {
        return Ok(Self {
            cmd: match cmd_matches.values_of("cmd") {
                None => Vec::new(),
                Some(iter) => iter.map(|s| s.to_string()).collect(),
            },
        });
    }

    pub fn SubCommand<'a, 'b>(_common: &CommonArgs<'a, 'b>) -> App<'a, 'b> {
        return SubCommand::with_name("cmd")
            .setting(AppSettings::ColoredHelp)
            .setting(AppSettings::TrailingVarArg)
            .arg(
                Arg::with_name("cmd")
                    .help("Compatibility (ignored)")
                    .multiple(true),
            )
            .about("Signal a (previously created) container");
    }

    pub fn Run(&self, _gCfg: &GlobalConfig) -> Result<()> {
        let kvmfd = Kvm::open_with_cloexec(false).expect("can't open kvm");

        let mut args = Args::default();
        args.KvmFd = kvmfd;
        args.AutoStart = true;
        args.ID = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".to_string();

        for a in &self.cmd {
            args.Spec.process.args.push(a.to_string());
        }

        match VirtualMachine::Init(args) {
            Ok(mut vm) => {
                vm.run().expect("vm.run() fail");
            }
            Err(e) => info!("error is {:?}", e),
        }

        return Ok(());
    }
}

impl Config {
    pub const CONFIG_FILE: &'static str = "/etc/quark/config.json";

    // if the config file exist, load file and return true; otherwise return false
    pub fn Load(&mut self) -> bool {
        let contents = match fs::read_to_string(Self::CONFIG_FILE) {
            Ok(c) => c,
            _ => return false,
        };

        let mut config: Config = serde_json::from_str(&contents).expect("configuration wrong format");

        if config.CCMode > CCMode::None {
            config.EnableRDMA = false;
            config.EnableTsot = false;
        }

        *self = config;
        return true;
    }

    pub fn Print(&self) {
        let c = serde_json::to_string(self).unwrap();
        error!("config is {}", c);
    }
}
