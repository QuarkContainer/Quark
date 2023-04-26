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

//use std::collections::BTreeMap;

use k8s_openapi::api::core::v1::{self as k8s};

use qobjs::common::*;

use super::k8s_types::EnvVar;

pub fn MakeEnvironmentVariables(
    pod: &k8s::Pod, 
    container: &k8s::Container, 
    _v1ConfigMaps: &k8s::ConfigMap,  
    _v1Secrets: &[k8s::Secret],
    _podIP: &str,
    _podIPs: &[String]
) -> Result<Vec<EnvVar>> {
    if pod.spec.as_ref().unwrap().enable_service_links.is_none() {
        return Err(Error::CommonError(format!("nil pod.spec.enableServiceLinks encountered, cannot construct envvars")));
    }

    let mut result : Vec<EnvVar> = Vec::new();
    //let mut configMaps :BTreeMap<String, k8s::ConfigMap> = BTreeMap::new();
    //let mut secrets: BTreeMap<String, k8s::Secret> = BTreeMap::new();
    //let mut tmpEnv: BTreeMap<String, String> = BTreeMap::new();

    if container.env.is_none() {
        return Ok(result);
    }

    for e in container.env.as_ref().unwrap() {
        let env = EnvVar {
            name: e.name.to_string(),
            value: e.value.as_deref().unwrap_or("").to_string(),
        };

        result.push(env);
    }

    return Ok(result)
}