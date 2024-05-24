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

#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

#[macro_use]
extern crate log;

#[macro_use]
extern crate clap;

pub mod package_mgr;
pub mod command;
pub mod create_pypackage;
pub mod list;
pub mod get;
pub mod get_object;
pub mod delete;

use command::{Parse, Run};
use qobjs::{common::*, zip::ZipMgr};

#[tokio::main]
async fn main() -> Result<()> {
    log4rs::init_file("fp_logging_config.yaml", Default::default()).unwrap();
    let mut args = match Parse() {
        Ok(args) => args,
        Err(e) => {
            error!("the parse error is {:?}", e);
            panic!("exiting...")
        }
    };

    Run(&mut args).await?;

    return Ok(())
}

pub fn ZipTest() -> Result<()> {
    log4rs::init_file("fp_logging_config.yaml", Default::default()).unwrap();
    error!("test ...");

    let buf = ZipMgr::ZipFolder("/home/brad/rust/Quark/qserverless/deployment")?;
    ZipMgr::Unzip("./test", buf)?;

    return Ok(())
}
