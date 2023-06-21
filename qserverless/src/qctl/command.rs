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

use std::env;
use clap::{App, AppSettings};

use qobjs::common::*;

use crate::create_pypackage::CreatePyPackageCmd;

pub struct Arguments {
    pub cmd: Command,
}

#[derive(Debug)]
pub enum Command {
    CreatePyPackage(CreatePyPackageCmd),
}

pub async fn Run(args: &mut Arguments) -> Result<()> {
    match &mut args.cmd {
        Command::CreatePyPackage(cmd) => return cmd.Run().await,
    }
}

fn get_args() -> Vec<String> {
    return env::args().collect();
}

pub fn Parse() -> Result<Arguments> {
    //let common = CommonArgs::New();

    let matches = App::new("qvisor")
    .about("qctl - qserverless client command line tool")
    .setting(AppSettings::ColoredHelp)
    .author(crate_authors!("\n"))
    .setting(AppSettings::SubcommandRequired)
    .version(crate_version!())
    .subcommand(CreatePyPackageCmd::SubCommand())
    .get_matches_from(get_args());
        
    let args = match matches.subcommand() {
        ("createpy", Some(cmd_matches)) => Arguments {
            cmd: Command::CreatePyPackage(CreatePyPackageCmd::Init(&cmd_matches)?),
        },
        // We should never reach here because clap already enforces this
        x => panic!("command not recognized {:?}", x),
    };
    
    return Ok(args);
}