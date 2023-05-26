// Copyright (c) 2021 Quark Container Authors / 2014 The Kubernetes Authors
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

pub mod func_def;
pub mod func_mgr;
pub mod funcagent_client;
pub mod funcall_mgr;

use funcagent_client::FuncAgentClient;
use once_cell::sync::OnceCell;

use qobjs::common::*;

use crate::func_mgr::FuncMgr;

lazy_static::lazy_static! {
    pub static ref FUNC_MGR: FuncMgr = {
        FuncMgr::Init()
    };
}

pub static FUNC_AGENT_CLIENT: OnceCell<FuncAgentClient> = OnceCell::new();


#[tokio::main]
async fn main() -> Result<()> {
    log4rs::init_file("logging_config.yaml", Default::default()).unwrap();
    
    FUNC_AGENT_CLIENT.set(FuncAgentClient::Init("http://27.0.0.1:1234").await?).unwrap();


    let mgr = FuncMgr::Init();
    println!("{:?}", mgr.Call("add", "gtest").await);
    println!("{:?}", mgr.Call("sub", "gtest").await);
    println!("{:?}", mgr.Call("sub1", "gtest").await);
    println!("Hello, world!");
    
    return Ok(());
}
