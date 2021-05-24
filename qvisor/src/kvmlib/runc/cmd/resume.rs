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
use super::super::cmd::config::*;
use super::super::container::container::*;
use super::command::*;

#[derive(Debug)]
pub struct ResumeCmd  {
    pub id: String,
}

impl ResumeCmd {
    pub fn Init(cmd_matches: &ArgMatches) -> Result<Self> {
        return Ok(Self {
            id: cmd_matches.value_of("id").unwrap().to_string(),
        })
    }

    pub fn SubCommand<'a, 'b>(common: &CommonArgs<'a, 'b>) -> App<'a, 'b> {
        return SubCommand::with_name("resume")
            .setting(AppSettings::ColoredHelp)
            .arg(&common.id_arg)
            .about("Resume unpauses a paused container");
    }

    pub fn Run(&self, gCfg: &GlobalConfig) -> Result<()> {
        info!("Container:: resume ....");
        let id = &self.id;

        let mut container = Container::Load(&gCfg.RootDir, id)?;
        container.Resume()?;

        return Ok(())
    }
}