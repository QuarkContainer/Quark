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

use std::collections::{HashMap};

use qobjs::k8s;
use qobjs::v1alpha2::{self as cri};

use crate::runtime::k8s_util::*;

// determinePodSandboxIP determines the IP addresses of the given pod sandbox.
pub fn DeterminePodSandboxIPs(podNamespace: &str, podName: &str, podSandbox: &cri::PodSandboxStatus) -> Vec<String> {
    let mut podIps = Vec::new();
    match &podSandbox.network {
        None => {
            info!("Pod Sandbox status doesn't have network information, cannot report IPs pod namespace {} name {}", podNamespace, podName);
            return podIps;
        }
        Some(network) => {
            // pick primary IP
            if network.ip.len() != 0 {
                podIps.push(network.ip.clone());
            }

            for ip in &network.additional_ips {
                podIps.push(ip.ip.clone());
            }
        }
    }

    return podIps;
}

pub fn AddPodSecurityContext(pod: &k8s::Pod, lpsc: &mut cri::LinuxPodSandboxConfig) {
    let spec = pod.spec.as_ref().unwrap();
    
    let mut sysctls = HashMap::new();

    if let Some(sc) = &spec.security_context {
        if let Some(ctls) = &sc.sysctls {
            for c in ctls {
                sysctls.insert(c.name.clone(), c.value.clone());
            }
        }

        lpsc.sysctls = sysctls;

        if let Some(run_as_user) = sc.run_as_user {
            lpsc.security_context.as_mut().unwrap().run_as_user = Some(cri::Int64Value {
                value: run_as_user,
            });
        }

        if let Some(run_as_group) = sc.run_as_group {
            lpsc.security_context.as_mut().unwrap().run_as_group = Some(cri::Int64Value {
                value: run_as_group,
            });
        }

        lpsc.security_context.as_mut().unwrap().namespace_options = Some(NamespacesForPod(pod));

        if let Some(fsgroup) = sc.fs_group {
            lpsc.security_context.as_mut().unwrap().supplemental_groups.push(fsgroup);
        }

        if let Some(fsgroups) = &sc.supplemental_groups {
            for sg in fsgroups {
                lpsc.security_context.as_mut().unwrap().supplemental_groups.push(*sg);
            }
        }

        if let Some(opts) = &sc.se_linux_options {
            lpsc.security_context.as_mut().unwrap().selinux_options = Some(cri::SeLinuxOption {
                user: opts.user.as_deref().unwrap_or("").to_string(),
                role: opts.role.as_deref().unwrap_or("").to_string(),
                r#type: opts.type_.as_deref().unwrap_or("").to_string(),
                level: opts.level.as_deref().unwrap_or("").to_string(),

            })
        }
    }

}