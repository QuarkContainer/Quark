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
use super::super::container::status::*;
use super::command::*;

#[derive(Default, Debug)]
pub struct DeleteCmd  {
    pub id: String,
    pub force: bool,
}

impl DeleteCmd {
    pub fn Init(cmd_matches: &ArgMatches) -> Result<Self> {
        return Ok(Self {
            id: cmd_matches.value_of("id").unwrap().to_string(),
            force: cmd_matches.is_present("force"),
        })
    }

    pub fn SubCommand<'a, 'b>(common: &CommonArgs<'a, 'b>) -> App<'a, 'b> {
        return SubCommand::with_name("delete")
            .setting(AppSettings::ColoredHelp)
            .arg(&common.id_arg)
            .arg(
                Arg::with_name("force")
                    .long("force")
                    .help("terminate container if running"),
            )
            .about("delete resources held by a container");
    }

    pub fn Run(&self, gCfg: &GlobalConfig) -> Result<()> {
        info!("Container:: Delete ....");
        let mut container = Container::Load(&gCfg.RootDir, &self.id)?;

        if !self.force && container.Status != Status::Created && container.Status != Status::Stopped {
            return Err(Error::Common("cannot delete container that is not stopped without --force flag".to_string()));
        }

        return container.Destroy()
    }
}

