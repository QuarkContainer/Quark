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
use chrono::{DateTime, Local};
use std::io::Write;
use tabwriter::TabWriter;

use super::super::super::qlib::common::*;
use super::super::cmd::config::*;
use super::super::container::container::*;
use super::command::*;

#[derive(Default, Debug)]
pub struct ListCmd  {
    pub format: String,
    pub quiet: bool,
}

impl ListCmd {
    pub fn Init(cmd_matches: &ArgMatches) -> Result<Self> {
        return Ok(Self {
            format: cmd_matches.value_of("format").unwrap_or_default().to_string(),
            quiet: cmd_matches.is_present("quiet"),
        })
    }

    pub fn SubCommand<'a, 'b>(common: &CommonArgs<'a, 'b>) -> App<'a, 'b> {
        return SubCommand::with_name("list")
            .setting(AppSettings::ColoredHelp)
            .arg(&common.format_arg)
            .arg(
                Arg::with_name("quiet")
                    .long("quiet")
                    .short("q")
                    .help("display only container IDs"),
            )
            .about("lists containers started with the given root");
    }

    pub fn Run(&self, gCfg: &GlobalConfig) -> Result<()> {
        let ids = ContainerList(&gCfg.RootDir)?;

        info!("list is {:?}, cfg is {:?}, ids is {:?}", self, gCfg, &ids);
        if self.quiet {
            for id in ids {
                info!("{}", id);
            }
            return Ok(())
        }

        let mut containers = Vec::new();

        for id in ids {
            let c = Container::Load(&gCfg.RootDir, &id)?;
            containers.push(c);
        }

        if &self.format == "table" {
            let mut tw = TabWriter::new(vec![]).minwidth(12).padding(3);

            write!(&mut tw, "ID\tPID\tSTATUS\tBUNDLE\tCREATED\tOWNER\n").unwrap();

            for c in containers {
                let datetime: DateTime<Local> = c.CreateTime().into();
                write!(tw, "{}\t{}\t{:?}\t{}\t{:?}\t{}\n",
                         &c.ID,
                         c.SandboxPid(),
                         &c.Status,
                         &c.BundleDir,
                         &datetime,
                         &c.Owner,
                ).unwrap();
            }

            tw.flush().unwrap();

            let written = String::from_utf8(tw.into_inner().unwrap()).unwrap();
            println!("{}", written);
        } else if &self.format == "json" {
            let mut states = Vec::new();
            for c in containers {
                states.push(c.State());
            }

            println!("{:#?}", states);
        } else {
            println!("unknown format {}", self.format);
        }

        return Ok(())
    }
}