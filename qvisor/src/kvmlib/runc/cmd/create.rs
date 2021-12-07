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

use clap::{App, AppSettings, SubCommand, ArgMatches};
use alloc::string::String;

use super::super::super::qlib::common::*;
use super::super::super::qlib::path::*;
use super::super::cmd::config::*;
use super::super::oci::*;
use super::super::container::container::*;
use super::command::*;

#[derive(Debug)]
pub struct CreateCmd  {
    pub id: String,
    pub bundleDir: String,
    pub consoleSocket: String,
    pub pivot: bool,
    pub pid: String,
}

impl CreateCmd {
    pub fn Init(cmd_matches: &ArgMatches) -> Result<Self> {
        return Ok(Self {
            id: cmd_matches.value_of("id").unwrap().to_string(),
            bundleDir: cmd_matches.value_of("bundle").unwrap().to_string(),
            consoleSocket: cmd_matches.value_of("console-socket").unwrap().to_string(),
            pivot: !cmd_matches.is_present("no-pivot"),
            pid: cmd_matches.value_of("p").unwrap().to_string(),
        })
    }

    pub fn SubCommand<'a, 'b>(common: &CommonArgs<'a, 'b>) -> App<'a, 'b> {
        return SubCommand::with_name("create")
            .setting(AppSettings::ColoredHelp)
            .arg(&common.id_arg)
            .arg(&common.bundle_arg)
            .arg(&common.consoleSocket_arg)
            .arg(&common.no_pivot_arg)
            .arg(&common.pid_arg)
            .arg(&common.init_arg)
            .arg(&common.user_log_arg)
            .about("Create a container (to be started later)");
    }

    pub fn Run(&self, gCfg: &GlobalConfig) -> Result<()> {
        let specfile = Join(&self.bundleDir, "config.json");
        let spec = Spec::load(&specfile).unwrap();

        Container::Create(
            &self.id,
            RunAction::Create,
            spec,
            gCfg,
            &self.bundleDir,
            &self.consoleSocket,
            &self.pid,
            "",
            true,
            self.pivot
        )?;

        //eprintln!("Application error: teststasdfasfd");
        //::std::process::exit(1);

        return Ok(())
    }
}