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

use qshare::{common::*, node::ContainerDef};

use super::k8s_types::EnvVar;

pub fn MakeEnvironmentVariables(
    container: &ContainerDef, 
) -> Result<Vec<EnvVar>> {
    // todo: ?
    /*if pod.spec.as_ref().unwrap().enable_service_links.is_none() {
        return Err(Error::CommonError(format!("nil pod.spec.enableServiceLinks encountered, cannot construct envvars")));
    }*/

    let mut result : Vec<EnvVar> = Vec::new();
    //let mut configMaps :BTreeMap<String, k8s::ConfigMap> = BTreeMap::new();
    //let mut secrets: BTreeMap<String, k8s::Secret> = BTreeMap::new();
    //let mut tmpEnv: BTreeMap<String, String> = BTreeMap::new();

    for e in &container.envs {
        let env = EnvVar {
            name: e.0.to_string(),
            value: e.1.to_string(),
        };

        result.push(env);
    }

    return Ok(result)
}