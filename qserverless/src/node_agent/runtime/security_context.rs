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

use std::collections::BTreeMap;
use lazy_static::lazy_static;

use k8s_openapi::api::core::v1::{self as k8s};
use qobjs::v1alpha2::{self as cri};

use crate::runtime::k8s_const;
use crate::runtime::k8s_util;

// SeccompProfileTypeUnconfined indicates no seccomp profile is applied (A.K.A. unconfined).
pub const SeccompProfileTypeUnconfined: &str = "Unconfined";
// SeccompProfileTypeRuntimeDefault represents the default container runtime seccomp profile.
pub const SeccompProfileTypeRuntimeDefault: &str = "RuntimeDefault";
// SeccompProfileTypeLocalhost indicates a profile defined in a file on the node should be used.
// The file's location relative to <kubelet-root-dir>/seccomp.
pub const SeccompProfileTypeLocalhost: &str = "Localhost";

pub fn DetermineEffectiveSecurityContext(
    pod: &k8s::Pod, 
    container: &k8s::Container, 
    _uid: Option<i64>, 
    _username: &str, 
    seccompDefault: bool, 
    seccompProfileRoot: &str
) -> cri::LinuxContainerSecurityContext {
    let effectiveSc = determineEffectiveSecurityContext(pod, container);
    let mut synthesized = ConvertToRuntimeSecurityContext(&effectiveSc);

    let map = BTreeMap::new();
    let annotation = match &pod.metadata.annotations {
        None => &map,
        Some(m) => m 
    };
    synthesized.seccomp_profile_path = GetSeccompProfilePath(
        annotation, 
        &container.name, 
        &pod.spec.as_ref().unwrap().security_context, 
        &container.security_context, 
        seccompDefault, 
        seccompProfileRoot
    );

    synthesized.seccomp = Some(GetSeccompProfile(
        seccompProfileRoot, 
        annotation, 
        &container.name, 
        &pod.spec.as_ref().unwrap().security_context, 
        &container.security_context, 
        seccompDefault
    ));

    synthesized.namespace_options = Some(k8s_util::NamespacesForPod(pod));

    let podSc = &pod.spec.as_ref().unwrap().security_context;
    if podSc != &None {
        let podSc = podSc.as_ref().unwrap();
        if podSc.fs_group.is_some() {
            synthesized.supplemental_groups.push(podSc.fs_group.clone().unwrap());
        }

        if let Some(groups) = &podSc.supplemental_groups {
            for sg in groups {
                synthesized.supplemental_groups.push(*sg);
            }
        }
    }

    
    synthesized.masked_paths = ConvertToRuntimeMaskedPaths(&effectiveSc.proc_mount);
    synthesized.readonly_paths = ConvertToRuntimeMaskedPaths(&effectiveSc.proc_mount);
    synthesized.no_new_privs = AddNoNewPrivileges(&Some(effectiveSc)); 

    return synthesized;
}

pub fn AddNoNewPrivileges(sc: &Option<k8s::SecurityContext>) -> bool {
    match sc {
        None => return false,
        Some(sc) => {
            if sc.allow_privilege_escalation.is_none() {
                return false;
            }
    
            return !sc.allow_privilege_escalation.unwrap()
        }
    }
}

lazy_static! {
    pub static ref defaultMaskedPaths : Vec<String> = vec![
        "/proc/acpi".to_string(),
        "/proc/kcore".to_string(),
        "/proc/keys".to_string(),
        "/proc/latency_stats".to_string(),
        "/proc/timer_list".to_string(),
        "/proc/timer_stats".to_string(),
        "/proc/sched_debug".to_string(),
        "/proc/scsi".to_string(),
        "/sys/firmware".to_string(),
    ];

    pub static ref defaultReadonlyPaths : Vec<String> = vec![
		"/proc/asound".to_string(),
		"/proc/bus".to_string(),
		"/proc/fs".to_string(),
		"/proc/irq".to_string(),
		"/proc/sys".to_string(),
		"/proc/sysrq-trigger".to_string(),
    ];
}

//pub type ProcMountType = &str;

// DefaultProcMount uses the container runtime defaults for readonly and masked
// paths for /proc.  Most container runtimes mask certain paths in /proc to avoid
// accidental security exposure of special devices or information.
pub const DefaultProcMount: &str = "Default";

// UnmaskedProcMount bypasses the default masking behavior of the container
// runtime and ensures the newly created /proc the container stays in tact with
// no modifications.
pub const UnmaskedProcMount: &str = "Unmasked";

// ConvertToRuntimeMaskedPaths converts the ProcMountType to the specified or default
// masked paths.
pub fn ConvertToRuntimeMaskedPaths(opt: &Option<String>) -> Vec<String> {
	if opt.is_some() && opt.as_ref().unwrap() == UnmaskedProcMount {
		// Unmasked proc mount should have no paths set as masked.
		return Vec::new();
	}

	// Otherwise, add the default masked paths to the runtime security context.
	return defaultMaskedPaths.to_vec()
}

// ConvertToRuntimeReadonlyPaths converts the ProcMountType to the specified or default
// readonly paths.
pub fn ConvertToRuntimeReadonlyPaths(opt: &Option<String>) -> Vec<String> {
	if opt.is_some() && opt.as_ref().unwrap() == UnmaskedProcMount {
		// Unmasked proc mount should have no paths set as readonly.
		return Vec::new();
	}

	// Otherwise, add the default readonly paths to the runtime security context.
	return defaultReadonlyPaths.to_vec();
}

pub fn SecurityContextFromPodSecurityContext(pod: &k8s::Pod) -> Option<k8s::SecurityContext> {
    let mut synthesized = k8s::SecurityContext::default();

    let spec = pod.spec.as_ref().unwrap();

    if spec.security_context.is_none() {
        return None;
    }

    let context = spec.security_context.as_ref().unwrap();

    synthesized.run_as_user = context.run_as_user.clone();
    synthesized.run_as_group = context.run_as_group.clone();
    synthesized.run_as_non_root = context.run_as_non_root.clone();


    return Some(synthesized);
}

// DetermineEffectiveSecurityContext returns a synthesized SecurityContext for reading effective configurations
// from the provided pod's and container's security context. Container's fields take precedence in cases where both
// are set
pub fn determineEffectiveSecurityContext(pod: &k8s::Pod, container: &k8s::Container) -> k8s::SecurityContext {
    let effectiveSc = SecurityContextFromPodSecurityContext(pod);
    let containerSc = &container.security_context;

    if effectiveSc.is_none() && containerSc.is_none() {
        return k8s::SecurityContext::default();
    }

    if effectiveSc.is_some() && containerSc.is_none() {
        return effectiveSc.unwrap();
    }

    if effectiveSc.is_none() && containerSc.is_some() {
        return containerSc.clone().unwrap();
    }

    let mut effectiveSc = effectiveSc.unwrap();
    let containerSc = containerSc.as_ref().unwrap();

    effectiveSc.se_linux_options = containerSc.se_linux_options.clone();
    effectiveSc.capabilities = containerSc.capabilities.clone();
    effectiveSc.privileged = containerSc.privileged.clone();
    effectiveSc.run_as_user = containerSc.run_as_user.clone();
    effectiveSc.run_as_group = containerSc.run_as_group.clone();
    effectiveSc.run_as_non_root = containerSc.run_as_non_root.clone();
    effectiveSc.read_only_root_filesystem = containerSc.read_only_root_filesystem.clone();
    effectiveSc.allow_privilege_escalation = containerSc.allow_privilege_escalation.clone();
    effectiveSc.proc_mount = containerSc.proc_mount.clone();

    return effectiveSc;
    
}

// convertToRuntimeSecurityContext converts v1.SecurityContext to criv1.SecurityContext.
pub fn ConvertToRuntimeSecurityContext(securityContext: &k8s::SecurityContext) -> cri::LinuxContainerSecurityContext {
    let mut sc = cri::LinuxContainerSecurityContext {
        capabilities: ConvertToRuntimeCapabilities(&securityContext.capabilities),
        selinux_options: ConvertToRuntimeSELinuxOption(&securityContext.se_linux_options),
        ..Default::default()
    };

    match &securityContext.run_as_user {
        None => (),
        Some(r) => {
            sc.run_as_user = Some(cri::Int64Value{
                value: *r,
            })
        }
    }

    match &securityContext.run_as_group {
        None => (),
        Some(r) => {
            sc.run_as_group = Some(cri::Int64Value{
                value: *r,
            })
        }
    }

    match &securityContext.privileged {
        None => (),
        Some(r) => {
            sc.privileged = *r;
        }
    }

    match &securityContext.read_only_root_filesystem {
        None => (),
        Some(r) => {
            sc.readonly_rootfs = *r;
        }
    }

    return sc;
}

pub fn ConvertToRuntimeSELinuxOption(opts: &Option<k8s::SELinuxOptions>) -> Option<cri::SeLinuxOption> {
    if opts.is_none() {
        return None;
    }

    let opts = opts.as_ref().unwrap();

    let option = cri::SeLinuxOption {
        user: opts.user.as_deref().unwrap_or("").to_string(),
        role: opts.role.as_deref().unwrap_or("").to_string(),
        r#type: opts.type_.as_deref().unwrap_or("").to_string(),
        level: opts.level.as_deref().unwrap_or("").to_string(),
    };

    return Some(option)
}

// convertToRuntimeCapabilities converts v1.Capabilities to criv1.Capability.
pub fn ConvertToRuntimeCapabilities(opts: &Option<k8s::Capabilities>) -> Option<cri::Capability> {
    if opts.is_none() {
        return None;
    }

    let mut capacity = cri::Capability::default();

    let opts = opts.as_ref().unwrap();
    for value in opts.add.as_ref().unwrap() {
        capacity.add_capabilities.push(value.to_string());
    }

    for value in opts.drop.as_ref().unwrap() {
        capacity.drop_capabilities.push(value.to_string());
    }

    return Some(capacity);
}

pub fn FieldProfile(scmp: &Option<k8s::SeccompProfile>, profileRootPath: &str, fallbackToRuntimeDefault: bool) -> String {
    if scmp.is_none() {
        if fallbackToRuntimeDefault {
            return k8s_const::SeccompProfileRuntimeDefault.to_string()
        }

        return "".to_string();
    }

    let scmp = scmp.as_ref().unwrap();
    if &scmp.type_ == SeccompProfileTypeRuntimeDefault {
        return k8s_const::SeccompProfileRuntimeDefault.to_string();
    }

    if &scmp.type_ == SeccompProfileTypeLocalhost 
        && scmp.localhost_profile.is_some() 
        && scmp.localhost_profile.as_ref().unwrap().len() > 0 {
        let fname = format!("{}/{}", profileRootPath, scmp.localhost_profile.as_ref().unwrap());
        return format!("{}/{}", k8s_const::SeccompLocalhostProfileNamePrefix, fname);
    }

    if &scmp.type_ == SeccompProfileTypeUnconfined {
        return k8s_const::SeccompProfileNameUnconfined.to_string();
    }

    return "".to_string();
}

pub fn AnnotationProfile(profile: &str, profileRootPath: &str) -> String {
    if profile.starts_with(k8s_const::SeccompLocalhostProfileNamePrefix) {
        let name = profile.strip_prefix(k8s_const::SeccompLocalhostProfileNamePrefix).unwrap();
        let fname = format!("{}/{}", profileRootPath, name);
        return k8s_const::SeccompLocalhostProfileNamePrefix.to_string() + &fname;
    }

    return profile.to_string();
}

pub fn GetSeccompProfilePath(
    annotations: &BTreeMap<String, String>, 
    containerName: &str,
	podSecContext: &Option<k8s::PodSecurityContext>, 
    containerSecContext: &Option<k8s::SecurityContext>, 
    fallbackToRuntimeDefault: bool, 
    seccompProfileRoot: &str) -> String {
    // container fields are applied first
    if containerSecContext.is_some() && containerSecContext.as_ref().unwrap().seccomp_profile.is_some() {
        return FieldProfile(
            &containerSecContext.as_ref().unwrap().seccomp_profile, 
            seccompProfileRoot, 
            fallbackToRuntimeDefault
        );
    }

    // if container field does not exist, try container annotation (deprecated)
	if containerName !=  "" {
        match annotations.get(&(k8s_const::SeccompContainerAnnotationKeyPrefix.to_string() + containerName)) {
            None => (),
            Some(profile) => {
                return AnnotationProfile(profile, seccompProfileRoot);
            }
        }
    }

    // when container seccomp is not defined, try to apply from pod field
	if podSecContext.is_some() && podSecContext.as_ref().unwrap().seccomp_profile.is_some() {
        return FieldProfile(
            &podSecContext.as_ref().unwrap().seccomp_profile, 
            seccompProfileRoot, 
            fallbackToRuntimeDefault
        );
    }

    // as last resort, try to apply pod annotation (deprecated)
	match annotations.get(k8s_const::SeccompPodAnnotationKey) {
        None => (),
        Some(profile) => {
            return AnnotationProfile(profile, seccompProfileRoot);
        }
    }

    if fallbackToRuntimeDefault {
        return k8s_const::SeccompProfileRuntimeDefault.to_string();
    }

    return "".to_string();
}

pub fn FieldSeccompProfile(scmp: &Option<k8s::SeccompProfile>, profileRootPath: &str, fallbackToRuntimeDefault: bool) -> cri::SecurityProfile {
    match scmp {
        None => {
            if fallbackToRuntimeDefault {
                return cri::SecurityProfile {
                    profile_type: cri::security_profile::ProfileType::RuntimeDefault as i32,
                    ..Default::default()
                }
            }

            return cri::SecurityProfile {
                profile_type: cri::security_profile::ProfileType::Unconfined as i32,
                ..Default::default()
            }
        }
        Some(t) => {
            if t.type_ == SeccompProfileTypeRuntimeDefault {
                return cri::SecurityProfile {
                    profile_type: cri::security_profile::ProfileType::RuntimeDefault as i32,
                    ..Default::default()
                }
            }

            if t.type_ == SeccompProfileTypeLocalhost 
                && t.localhost_profile.is_some() 
                && t.localhost_profile.as_ref().unwrap().len() > 0 {
                let fname = format!("{}/{}", profileRootPath, t.localhost_profile.as_ref().unwrap());
                return cri::SecurityProfile {
                    profile_type: cri::security_profile::ProfileType::RuntimeDefault as i32,
                    localhost_ref: fname,
                }
            }

            return cri::SecurityProfile {
                profile_type: cri::security_profile::ProfileType::Unconfined as i32,
                ..Default::default()
            }
        }
    }
}

pub fn GetSeccompProfile(
    seccompProfileRoot: &str, 
    _annotations: &BTreeMap<String, String>, 
    _containerName: &str,
	podSecContext: &Option<k8s::PodSecurityContext>, 
    containerSecContext: &Option<k8s::SecurityContext>, 
    fallbackToRuntimeDefault: bool) -> cri::SecurityProfile {
    
    // container fields are applied first
    if containerSecContext.is_some() && containerSecContext.as_ref().unwrap().seccomp_profile.is_some() {
        return FieldSeccompProfile(
            &containerSecContext.as_ref().unwrap().seccomp_profile,
            seccompProfileRoot, 
            fallbackToRuntimeDefault
        );
    }

    // when container seccomp is not defined, try to apply from pod field
	if podSecContext.is_some() && podSecContext.as_ref().unwrap().seccomp_profile.is_some() {
        return FieldSeccompProfile(
            &podSecContext.as_ref().unwrap().seccomp_profile,
            seccompProfileRoot, 
            fallbackToRuntimeDefault
        );
    }

    if fallbackToRuntimeDefault {
        return cri::SecurityProfile {
            profile_type: cri::security_profile::ProfileType::RuntimeDefault as i32,
            ..Default::default()
        }
    }

    return cri::SecurityProfile {
        profile_type: cri::security_profile::ProfileType::Unconfined as i32,
        ..Default::default()
    }
}
    