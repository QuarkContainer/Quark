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

use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};
use serde_json;
use std::fs::File;
use std::os::unix::io::FromRawFd;

use super::super::super::qlib::common::*;
use super::super::cmd::config::*;
use super::super::runtime::sandbox_process::*;
use super::command::*;

#[derive(Debug)]
pub struct BootCmd {
    pub pipefd: i32,
}

impl BootCmd {
    pub fn Init(cmd_matches: &ArgMatches) -> Result<Self> {
        return Ok(Self {
            pipefd: cmd_matches
                .value_of("pipefd")
                .unwrap()
                .to_string()
                .parse()
                .unwrap(),
        });
    }

    pub fn SubCommand<'a, 'b>(_common: &CommonArgs<'a, 'b>) -> App<'a, 'b> {
        return SubCommand::with_name("boot")
            .setting(AppSettings::ColoredHelp)
            .arg(
                Arg::with_name("pipefd")
                    .required(true)
                    .takes_value(true)
                    .long("pipefd")
                    .help("pipe fd with the container parameters"),
            )
            .about("Create a container (to be started later)");
    }

    pub fn Run(&self, _gCfg: &GlobalConfig) -> Result<()> {
        let pipefile = unsafe { File::from_raw_fd(self.pipefd) };

        let process: SandboxProcess = serde_json::from_reader(&pipefile)
            .map_err(|e| Error::IOError(format!("BootCmd io::error is {:?}", e)))?;

        process.Child()?;
        return Ok(());
    }
}
