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

use super::super::super::qlib::common::*;
use super::config::GlobalConfig;
use super::command::CommonArgs;
use super::super::container::container::Container;
use serde_json;
use std::io::{self, Write};

#[derive(Debug)]
pub struct StateCmd {
    pub id: String,
}

impl StateCmd {
    pub fn Init(cmd_matches: &ArgMatches) -> Result<Self> {
        return Ok (Self {
            id: cmd_matches.value_of("id").unwrap().to_string(),
        })
    }

    pub fn SubCommand<'a, 'b>(common: &CommonArgs<'a, 'b>) -> App<'a, 'b> {
        return SubCommand::with_name("state")
            .setting(AppSettings::ColoredHelp)
            .about("Query the state properties of a container, as defined by OCI")
            .arg(&common.id_arg);
    }
    
    pub fn Run(&self, gCfg: &GlobalConfig) -> Result<()> {
        info!("Container:: state ....");
        let id = &self.id;

        let container = Container::Load(&gCfg.RootDir, id)?;
        let state = container.State();

        debug!("container state: {:?}", &state);
        match serde_json::to_string(&state) {
            Ok(str) => {
                if let Err(e) = io::stdout().write_all(str.as_bytes()) {
                    return Err(Error::IOError(e.to_string()));
                }
            }
            Err(e) => {
                return Err(Error::Common(e.to_string()))
            }
        }
        
        return Ok(())
    }
}

