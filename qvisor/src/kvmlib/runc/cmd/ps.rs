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
use std::io::Write;
use tabwriter::TabWriter;

use super::super::super::qlib::common::*;
use super::super::super::qlib::control_msg::*;
use super::super::cmd::config::*;
use super::super::container::container::*;
use super::command::*;

#[derive(Debug, PartialEq, Eq)]
pub enum Format {
    Json,
    Table
}

#[derive(Debug)]
pub struct PsCmd  {
    pub id: String,
    pub format: Format,
}

impl PsCmd {
    pub fn Init(cmd_matches: &ArgMatches) -> Result<Self> {
        let ret = Self {
            id: cmd_matches.value_of("id").unwrap().to_string(),
            format: match cmd_matches.value_of("format").unwrap() {
                "table" => Format::Table,
                "json" => Format::Json,
                _ => return Err(Error::Common("invalid format option".to_string()))
            }
        };

        return Ok(ret)
    }

    pub fn SubCommand<'a, 'b>(common: &CommonArgs<'a, 'b>) -> App<'a, 'b> {
        return SubCommand::with_name("ps")
            .setting(AppSettings::ColoredHelp)
            .arg(&common.id_arg)
            .arg(
                Arg::with_name("format")
                    .help("current working directory")
                    .default_value("table")
                    .takes_value(true)
                    .long("format")
                    .short("f"),
            )
            .about("ps displays the processes running inside a container") ;
    }

    pub fn Run(&mut self, gCfg: &GlobalConfig) -> Result<()> {
        info!("Container:: PS ....");
        let container = Container::Load(&gCfg.RootDir, &self.id)?;

        let plist = container.Processes()?;

        if self.format == Format::Table {
            PrintProcessListToTable(&plist);
        } else {
            PrintPIDsJson(&plist);
        }

        return Ok(())
    }
}

pub fn PrintProcessListToTable(pl: &[ProcessInfo]) {
    let mut tw = TabWriter::new(vec![]).minwidth(10).padding(3);

    write!(&mut tw, "UID\tPID\tPPID\tC\tSTIME\tTIME\tCMD\n").unwrap();
    for d in pl {
        write!(&mut tw, "\n{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
               d.UID.0,
               d.PID,
               d.PPID,
               d.Utilization,
               d.STime,
               d.Time,
               d.Cmd).unwrap();
    }
    tw.flush().unwrap();

    let written = String::from_utf8(tw.into_inner().unwrap()).unwrap();
    println!("{}", written);
}

pub fn PrintPIDsJson(pl: &[ProcessInfo]) {
    let mut pids = Vec::with_capacity(pl.len());
    for d in pl {
        pids.push(d.PID);
    }

    println!("{:?}", pids);
}