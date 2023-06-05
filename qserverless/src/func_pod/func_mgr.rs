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

use std::sync::Arc;
use std::collections::BTreeMap;

use qobjs::func;
use qobjs::types::*;

use crate::{func_def::*, FUNC_CALL_MGR};

//#[derive(Default)]
pub struct FuncMgr {
    pub funcPodId: String,
    pub namespace: String,
    pub packageName: String,
    pub funcs: Arc<BTreeMap<String, Arc<dyn QSFunc>>>,
}

impl FuncMgr {
    pub fn FuncPodId(&self) -> String {
        return self.funcPodId.clone();
    }

    pub fn GetPodIdFromEnvVar() -> String {
        use std::env;
        match env::var(EnvVarNodeMgrPodId) {
            Ok(val) => return val,
            Err(_e) => {
                error!("can't get pod id from environment variable {}, generate a random pid id", EnvVarNodeMgrPodId);
                return uuid::Uuid::new_v4().to_string()
            }

        };
    }

    pub fn RegisterMsg(&self) -> func::FuncPodRegisterReq {
        return func::FuncPodRegisterReq {
            func_pod_id: self.funcPodId.clone(),
            namespace: self.namespace.clone(),
            package_name: self.packageName.clone(),
        }
    }
    
    pub fn Init() -> Self {
        let mut funcs: BTreeMap<String, Arc<dyn QSFunc>> = BTreeMap::new();
        

        funcs.insert("add".to_string(), Arc::new(Add{}));
        funcs.insert("sub".to_string(), Arc::new(Sub{}));

        return Self {
            funcPodId: Self::GetPodIdFromEnvVar(),
            namespace: "ns1".to_owned(),
            packageName: "package1".to_owned(),
            funcs: Arc::new(funcs),
        }
    }

    pub async fn Call(&self, name: &str, parameters: &str) -> QSResult {
        let f = match self.funcs.get(name) {
            None => return Err(format!("There is no func named {} has names {:?}", name, self.funcs.keys())),
            Some(f) => f.clone(),
        };

        return f.func(parameters.to_string()).await;
    }
}

#[derive(Debug)]
pub struct Add {}

#[async_trait::async_trait]
impl QSFunc for Add {
    async fn func(&self, _parameters: String) -> Result<String, String> {
        let ret = FUNC_CALL_MGR.RemoteCall(
            "ns1".to_string(), 
            "package1".to_string(), 
            "sub".to_string(), 
            "call from Add".to_string(), 
            1
        ).await;
        Ok(format!("add with sub result {:?}", ret))
    }
}

#[derive(Debug)]
pub struct Sub {}

#[async_trait::async_trait]
impl QSFunc for Sub {
    async fn func(&self, parameters: String) -> Result<String, String> {

        Ok(format!("sub with para {}", parameters))
    }
    
}
