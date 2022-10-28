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
use nix::fcntl::OFlag;
use nix::sys::stat::Mode;

use crate::runc::container::container::RunAction;
use crate::runc::runtime::util::{Open, Write};
use crate::runc::shim::container_io::ContainerIO;

use super::command::*;
use super::super::cmd::config::*;
use super::super::runtime::sandbox_process::*;
use super::super::super::qlib::common::*;

#[derive(Debug)]
pub struct SandboxCmd {
    pub id: String,
    pub task_socket: String,
    pub pid_file: String,
}

impl SandboxCmd {
    pub fn Init(cmd_matches: &ArgMatches) -> Result<Self> {
        return Ok(Self {
            id: cmd_matches
                .value_of("id")
                .unwrap()
                .to_string(),
            task_socket: cmd_matches
                .value_of("task-socket")
                .unwrap()
                .to_string(),
            pid_file: cmd_matches
                .value_of("pid-file")
                .unwrap()
                .to_string(),
        });
    }

    pub fn SubCommand<'a, 'b>(_common: &CommonArgs<'a, 'b>) -> App<'a, 'b> {
        return SubCommand::with_name("sandbox")
            .setting(AppSettings::ColoredHelp)
            .arg(
                Arg::with_name("id")
                    .required(true)
                    .takes_value(true)
                    .long("id")
                    .help("sandbox id"),
            )
            .arg(
                Arg::with_name("task-socket")
                    .required(true)
                    .takes_value(true)
                    .long("task-socket")
                    .help("task socket address for quark to listen"),
            )
            .arg(
                Arg::with_name("pid-file")
                    .required(true)
                    .takes_value(true)
                    .long("pid-file")
                    .help("pid file path"),
            )
            .about("Create a sandbox");
    }

    pub fn Run(&self, gCfg: &GlobalConfig) -> Result<()> {
        let bundleDir = if gCfg.RootDir.is_empty() {
            "".to_string()
        } else {
            gCfg.RootDir.to_string()
        };
        // TODO get pivot_root config from runtime options
        let mut process = SandboxProcess::New(gCfg, RunAction::Create, &self.id, &bundleDir, true)?;
        process.TaskSocket = Some(self.task_socket.clone());
        process.SandboxRootDir = bundleDir;
        let pid = process.Execv1(&ContainerIO::None)?;
        let pid_file_fd = Open(&self.pid_file, OFlag::O_CREAT|OFlag::O_WRONLY, Mode::empty())?;
        Write(pid_file_fd, pid.to_string().as_bytes())?;
        return Ok(());
    }
}
