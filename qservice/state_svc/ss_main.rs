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
#![allow(deprecated)]

#[macro_use]
extern crate log;
extern crate simple_logging;

pub mod state_svc;

use qshare::common::*;

use self::state_svc::StateService;


#[tokio::main]
async fn main() -> Result<()> {
    log4rs::init_file("/etc/quark/ss_logging_config.yaml", Default::default()).unwrap();
    
    StateService().await.unwrap();

    return Ok(())
}