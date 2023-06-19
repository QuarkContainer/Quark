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
#![feature(map_first_last)]

#[macro_use]
extern crate scopeguard;

#[macro_use]
extern crate log;

use func_call::FuncCallMgr;
use func_node::FuncNodeMgr;
use func_pod::FuncPodMgr;
use func_svc::FuncSvc;
use lazy_static::lazy_static;

pub mod func_svc;
pub mod func_conn;
pub mod func_call;
pub mod task_queue;
pub mod scheduler;
pub mod func_pod;
pub mod package;
pub mod func_node;
pub mod message;
pub mod grpc_svc;

use package::PackageMgr;
use qobjs::{common::*, cacher_client::CacherClient, types::DataObject, informer_factory::InformerFactory, selection_predicate::ListOption};
use scheduler::Scheduler;

lazy_static! {
    pub static ref PACKAGE_MGR: PackageMgr = {
        PackageMgr::New()
    };

    pub static ref FUNC_POD_MGR: FuncPodMgr = {
        FuncPodMgr::New()
    };

    pub static ref FUNC_NODE_MGR: FuncNodeMgr = {
        FuncNodeMgr::New()
    };

    pub static ref FUNC_CALL_MGR: FuncCallMgr = {
        FuncCallMgr::default()
    };

    pub static ref FUNC_SVC_MGR: FuncSvc = {
        FuncSvc::New()
    };

    pub static ref SCHEDULER: Scheduler = {
        Scheduler::New("http://127.0.0.1:8890")
    };
}

#[tokio::main]
async fn main() -> Result<()> {
    use std::sync::Arc;

    log4rs::init_file("fs_logging_config.yaml", Default::default()).unwrap();

    error!("init func svc");

    let factory = InformerFactory::New("http://127.0.0.1:8890", "").await.unwrap();
    factory.AddInformer("package", &ListOption::default()).await.unwrap();
    let informer = factory.GetInformer("package").await.unwrap();
    let _id1 = informer.AddEventHandler(Arc::new(PACKAGE_MGR.clone())).await.unwrap();

    use crate::package::PackageId;
    let packageId = PackageId {
        namespace: "ns1".to_string(),
        packageName: "package1".to_string(),
    };

    if PACKAGE_MGR.Get(&packageId).is_err() {
        let client = CacherClient::New("http://127.0.0.1:8890".into()).await.unwrap();
        let obj = DataObject::NewFuncPackage1("ns1", "package1").unwrap();
        client.Create("package", obj.Obj()).await.unwrap();
    }

    let packageId = PackageId {
        namespace: "ns1".to_string(),
        packageName: "pypackage1".to_string(),
    };

    if PACKAGE_MGR.Get(&packageId).is_err() {
        let client = CacherClient::New("http://127.0.0.1:8890".into()).await.unwrap();
        let obj = DataObject::NewFuncPyPackage("ns1", "pypackage1").unwrap();
        client.Delete("package", "ns1", "pypackage1").await.ok();
        error!("create new package {:#?}", &obj);
        client.Create("package", obj.Obj()).await.unwrap();
    }

    grpc_svc::FuncSvcGrpcService().await.unwrap();
    Ok(())
}


#[cfg(test)]
mod tests {
    use qobjs::audit::func_audit::*;

    #[test]
    fn test_create() {
        let mut audit = SqlFuncAudit::New("postgresql://testuser:123456@localhost/testdb1").unwrap();
        let id = uuid::Uuid::new_v4().to_string();
        
        audit.CreateFunc(
            &id, 
            &uuid::Uuid::new_v4().to_string(), 
            "package1", 
            &uuid::Uuid::new_v4().to_string()
        ).unwrap();
        
        assert!(false);
    }

    #[test]
    fn test_update() {
        let mut audit = SqlFuncAudit::New("postgresql://testuser:123456@localhost/testdb1").unwrap();
        let id = uuid::Uuid::new_v4().to_string();
        
        audit.CreateFunc(
            &id, 
            &uuid::Uuid::new_v4().to_string(), 
            "package1", 
            &uuid::Uuid::new_v4().to_string()
        ).unwrap();
        
        audit.FinishFunc(
            &id, 
            "Finish"
        ).unwrap();
        assert!(false);
    }
}