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

use clap::{App, AppSettings, ArgMatches, SubCommand, Arg};

use qobjs::{common::*, types::QMETASVC_ADDR, system_types::FuncPackage};

use crate::package_mgr::PackageMgr;

#[derive(Debug)]
pub struct CreatePyPackageCmd {
    pub manifestfile: String,
    pub zipfolder: String,
}

impl CreatePyPackageCmd {
    pub fn Init(cmd_matches: &ArgMatches) -> Result<Self> {
        return Ok(Self {
            manifestfile: cmd_matches.value_of("manifest").unwrap().to_string(),
            zipfolder: cmd_matches.value_of("func_directory").unwrap().to_string(),
        });
    }

    pub fn SubCommand<'a, 'b>() -> App<'a, 'b> {
        return SubCommand::with_name("createpy")
            .setting(AppSettings::ColoredHelp)
            .arg(
                Arg::with_name("manifest")
                    .required(true)
                    .help("func package definition")
                    .long("manifest")
                    .short("f")
                    .takes_value(true),
            )    
            .arg(
                Arg::with_name("func_directory")
                    .required(true)
                    .help("the directory contains functions files")
                    .long("funcs_dir")
                    .short("d")
                    .takes_value(true),
            )
            .about("Create a python function package");
    }

    pub async fn Run(&self) -> Result<()> {
        println!("CreatePyPackageCmd is {:?}", self);
        let addr = format!("http://{}", QMETASVC_ADDR);
        let mut packageMgr = match PackageMgr::New(&addr).await {
            Err(e) => {
                println!("can't connect the qservereless service {} with error {:?}", &addr, e);
                return Ok(())
            }
            Ok(m) => m
        };

        let packageStr = match std::fs::read(&self.manifestfile) {
            Err(e) => {
                println!("can't open manifest file {} with error {:?}", &self.manifestfile, e);
                return Ok(())
            }
            Ok(s) => s 
        };

        let package : FuncPackage = match serde_json::from_slice(&packageStr) {
            Err(e) => {
                println!("can't deserialize manifest {} with error {:?}", &self.manifestfile, e);
                return Ok(())
            }
            Ok(p) => p 
        };

        match packageMgr.CreatePyPackage(package, &self.zipfolder).await {
            Err(e) => {
                println!("can't create package with error {:?}", e);
                return Ok(());
            }
            Ok(_) => {
                return Ok(());
            }
        }
        
    }
}