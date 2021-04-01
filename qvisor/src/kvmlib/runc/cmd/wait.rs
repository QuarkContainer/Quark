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

use clap::{App, AppSettings, SubCommand, ArgMatches, Arg};
use alloc::string::String;

use super::super::super::qlib::common::*;
use super::super::cmd::config::*;
use super::super::container::container::*;
use super::command::*;

#[derive(Debug)]
pub struct WaitCmd  {
    pub id: String,
    pub pid: i32,
    pub rootPid: i32,
}

impl WaitCmd {
    pub fn Init(cmd_matches: &ArgMatches) -> Result<Self> {
        let pidStr = cmd_matches.value_of("pid").unwrap().to_string();
        let rootPidStr = cmd_matches.value_of("rootpid").unwrap().to_string();

        let pid = match pidStr.parse::<i32>() {
            Err(_e) => return Err(Error::Common(format!("pid {} cant not be parsed as int type", pidStr))),
            Ok(v) => v,
        };

        let rootPid = match rootPidStr.parse::<i32>() {
            Err(_e) => return Err(Error::Common(format!("root {} cant not be parsed as int type", rootPidStr))),
            Ok(v) => v,
        };

        return Ok(Self {
            id: cmd_matches.value_of("id").unwrap().to_string(),
            pid: pid,
            rootPid: rootPid,
        })
    }

    pub fn SubCommand<'a, 'b>(common: &CommonArgs<'a, 'b>) -> App<'a, 'b> {
        return SubCommand::with_name("wait")
            .setting(AppSettings::ColoredHelp)
            .arg(&common.id_arg)
            .arg(
                Arg::with_name("rootpid")
                    .default_value("-1")
                    .long("rootpid")
                    .required(true)
                    .takes_value(true)
                    .help("Signal to send to container"),
            )
            .arg(
                Arg::with_name("pid")
                    .default_value("-1")
                    .long("pid")
                    .takes_value(true)
                    .help("send the specified signal to a specific process"),
            )
            .about("wait a container");
    }

    pub fn Run(&self, gCfg: &GlobalConfig) -> Result<()> {
        info!("Container:: Wait ....");
        let id = &self.id;
        let rootPid = self.rootPid;
        let pid = self.pid;

        if rootPid != -1 && pid != -1 {
            panic!("only one of -pid and -rootPid can be set")
        }

        let mut container = Container::Load(&gCfg.RootDir, id)?;

        let res;

        if rootPid == -1 && pid == -1 {
            res  = container.Wait()?;
        } else if rootPid != -1 {
            res = container.WaitRootPID(rootPid, true)?;
        } else { //pid != -1
            res = container.WaitPid(pid, true)?;
        }

        let ret = waitResult {
            id: id.to_string(),
            exitStatus: res,
        };

        println!("{:?}", ret);
        return Ok(())
    }
}

#[derive(Debug)]
pub struct waitResult {
    pub id: String,
    pub exitStatus: u32,
}