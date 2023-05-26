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

use std::collections::BTreeMap;
use futures::future::FutureExt; 

use crate::func_def::*;


#[derive(Default)]
pub struct FuncMgr {
    pub funcs: BTreeMap<String, QServerlessFn>,
}

impl FuncMgr {
    pub fn Init() -> Self {
        let mut map = Self::default();

        map.AddFunc("add", Box::new(|a| add(a).boxed()));
        map.AddFunc("sub", Box::new(|a| sub(a).boxed()));

        return map;
    }

    pub fn AddFunc(&mut self, name: &str, f: QServerlessFn) {
        self.funcs.insert(name.to_string(), f);
    }

    pub async fn Call(&self, name: &str, parameters: &str) -> Result<String, String> {
        let f = match self.funcs.get(name) {
            None => return Err(format!("There is no func named {}", name)),
            Some(f) => f,
        };

        return f(parameters.to_string()).await;
    }
}

async fn add(_parameters: String) -> Result<String, String> {
    Ok("add".to_string())
}

async fn sub(_parameters: String) -> Result<String, String> {
    Err("sub".to_string())
}
