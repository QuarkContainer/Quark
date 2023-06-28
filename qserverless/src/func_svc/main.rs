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
use qobjs::{common::*, informer_factory::InformerFactory, selection_predicate::ListOption, audit::audit_agent::AuditAgent};
use scheduler::Scheduler;
use qobjs::types::*;

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
        Scheduler::New(QMETASVC_ADDR)
    };

    pub static ref AUDIT_AGENT: AuditAgent = {
        AuditAgent::New(AUDITDB_ADDR)
    };
}

#[tokio::main]
async fn main() -> Result<()> {
    use std::sync::Arc;

    use qobjs::types::*;

    log4rs::init_file("fs_logging_config.yaml", Default::default()).unwrap();

    error!("init func svc");

    let qmetaSvcAddr = &format!("http://{}", QMETASVC_ADDR);
    let factory = InformerFactory::New(qmetaSvcAddr, "").await.unwrap();
    factory.AddInformer("package", &ListOption::default()).await.unwrap();
    let informer = factory.GetInformer("package").await.unwrap();
    let _id1 = informer.AddEventHandler(Arc::new(PACKAGE_MGR.clone())).await.unwrap();

    grpc_svc::FuncSvcGrpcService().await.unwrap();
    Ok(())
}


#[cfg(test)]
mod tests {
    use qobjs::audit::func_audit::*;
    use qobjs::object_mgr::*;
    use qobjs::types::*;

    #[actix_rt::test]
    async fn test_create() {
        let audit = SqlFuncAudit::New(AUDITDB_ADDR).await.unwrap();
        let id = uuid::Uuid::new_v4().to_string();
        
        audit.CreateFunc(
            &id, 
            &uuid::Uuid::new_v4().to_string(), 
            "ns1",
            "package1",
            "testfunc1", 
            &uuid::Uuid::new_v4().to_string()
        ).await.unwrap();
        
        assert!(false);
    }

    #[actix_rt::test]
    async fn test_update() {
        let audit = SqlFuncAudit::New(AUDITDB_ADDR).await.unwrap();
        let id = uuid::Uuid::new_v4().to_string();
        
        audit.CreateFunc(
            &id, 
            &uuid::Uuid::new_v4().to_string(), 
            "ns1",
            "package1",
            "testfunc1", 
            &uuid::Uuid::new_v4().to_string()
        ).await.unwrap();
        
        audit.FinishFunc(
            &id, 
            "Finish"
        ).await.unwrap();
        assert!(false);
    }

    #[actix_rt::test]
    async fn test_blob() {
        let blob = SqlObjectMgr::New(OBJECTDB_ADDR).await.unwrap();
        let namespace = "ns1";
        let name = "object1";
        let datastr = "asdfasdfasdfdsafd";
        blob.PutObject(
            namespace,
            name, 
            &datastr.as_bytes()
        ).await.unwrap();

        let data = blob.ReadObject(
            namespace,
            name
        ).await.unwrap();

        println!("data is {}", std::str::from_utf8(&data).unwrap());

        let objs = blob.ListObjects(namespace, "obj").await.unwrap();
        println!("list is {:?}", objs);
        assert!(objs.len()==1);

        let objs = blob.ListObjects(namespace, "xxx").await.unwrap();
        assert!(objs.len()==0);

        blob.DeleteObject(
            namespace,
            name
        ).await.unwrap();

        assert!(datastr==std::str::from_utf8(&data).unwrap());
        assert!(false);
    }
}