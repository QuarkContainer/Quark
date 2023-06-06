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
pub mod blob_mgr;

use blob_mgr::BlobMgr;
use funcagent_client::FuncAgentClient;
use once_cell::sync::OnceCell;

use qobjs::common::*;

use crate::funcall_mgr::*;

lazy_static::lazy_static! {
    pub static ref FUNC_CALL_MGR: FuncCallMgr = {
        FuncCallMgr::Init()
    };

    pub static ref BLOB_MGR: BlobMgr = {
        BlobMgr::default()
    }; 
}

pub static FUNC_AGENT_CLIENT: OnceCell<FuncAgentClient> = OnceCell::new();


#[tokio::main]
async fn main() -> Result<()> {
    log4rs::init_file("fp_logging_config.yaml", Default::default()).unwrap();
    FUNC_AGENT_CLIENT.set(FuncAgentClient::Init("http://192.168.0.22:8892").await?).unwrap();
    FUNC_CALL_MGR.Process().await?;
    
    return Ok(());
}

#[cfg(test)]
mod tests {
    use qobjs::func_client::FuncClient;

    //use super::*;

    #[actix_rt::test]
    async fn TestDirectFuncCallAdd() {
        log4rs::init_file("logging_config.yaml", Default::default()).unwrap();
        error!("TestDirectFuncCall 1");
        let mut client = FuncClient::Init("http://192.168.0.22:8892").await.unwrap();
        error!("TestDirectFuncCall 2");
        //let ret = client.Call("ns1", "package1", "sub", "", 1).await;
        let ret = client.Call("ns1", "package1", "add", "", 1).await;
        error!("ret is {:?}", ret);
        assert!(ret.is_ok());
    }

    #[actix_rt::test]
    async fn TestDirectFuncCallSub() {
        log4rs::init_file("logging_config.yaml", Default::default()).unwrap();
        error!("TestDirectFuncCall 1");
        let mut client = FuncClient::Init("http://192.168.0.22:8892").await.unwrap();
        error!("TestDirectFuncCall 2");
        let ret = client.Call("ns1", "package1", "sub", "", 1).await;
        error!("ret is {:?}", ret);
        assert!(ret.is_ok());
    }

    #[actix_rt::test]
    async fn TestDirectFuncCallSimple() {
        log4rs::init_file("logging_config.yaml", Default::default()).unwrap();
        error!("TestDirectFuncCall 1");
        let mut client = FuncClient::Init("http://192.168.0.22:8892").await.unwrap();
        error!("TestDirectFuncCall 2");
        let ret = client.Call("ns1", "package1", "simple", "", 1).await;
        error!("ret is {:?}", ret);
        assert!(ret.is_ok());
    }
}