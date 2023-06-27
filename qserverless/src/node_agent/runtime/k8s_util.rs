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
use std::fs;
use std::path::Path;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::sync::RwLock;
use qobjs::types::DefaultNodeFuncLogFolder;
use rand::prelude::*;
use std::fs::Permissions;

use qobjs::k8s;
use qobjs::crictl;
use qobjs::common::*;

use qobjs::config::*;
use crate::FUNCDIR_MGR;
use crate::NODEAGENT_CONFIG;
use crate::pod::PodType;
use crate::runtime::k8s_quantity::*;
use crate::runtime::security_context::*;

use super::k8s_types::RunContainerOptions;

pub fn CleanupPodDataDirs(rootPath: &str, pod: &k8s::Pod) -> Result<()> {
    let uid = pod.metadata.uid.as_ref().unwrap();
    fs::remove_dir_all(&GetPodDir(rootPath, uid)).ok();
    fs::remove_dir_all(&GetPodVolumesDir(rootPath, uid)).ok();
    fs::remove_dir_all(&GetPodPluginsDir(rootPath, uid)).ok();
    return Ok(())
}

pub fn MakePodDataDir(rootPath: &str, pod: &k8s::Pod) -> Result<()> {
    let uid = pod.metadata.uid.as_ref().unwrap();
    fs::create_dir_all(&GetPodDir(rootPath, uid)).ok();
    fs::create_dir_all(&GetPodVolumesDir(rootPath, uid)).ok();
    fs::create_dir_all(&GetPodPluginsDir(rootPath, uid)).ok();
    return Ok(())
}

pub fn CleanupPodLogDir(rootPath: &str, pod: &k8s::Pod) -> Result<()> {
    let namespace = pod.metadata.namespace.as_ref().unwrap();
    let name = pod.metadata.name.as_ref().unwrap();
    let uid = pod.metadata.uid.as_ref().unwrap();
    fs::remove_dir_all(&GetPodLogDir(rootPath, namespace, name, uid)).ok();
    return Ok(())
}

pub fn MakePodLogDir(rootPath: &str, pod: &k8s::Pod) -> Result<()> {
    let namespace = pod.metadata.namespace.as_ref().unwrap();
    let name = pod.metadata.name.as_ref().unwrap();
    let uid = pod.metadata.uid.as_ref().unwrap();
    fs::create_dir_all(&GetPodLogDir(rootPath, namespace, name, uid)).ok();
    return Ok(())
}

pub fn GetPullSecretsForPod(_pod: &k8s::Pod) -> Vec<k8s::Secret> {
	let pullSecrets = Vec::new();

	// for _, secretRef := range pod.Spec.ImagePullSecrets {
	//  if len(secretRef.Name) == 0 {
	//    // API validation permitted entries with empty names (http://issue.k8s.io/99454#issuecomment-787838112).
	//    // Ignore to avoid unnecessary warnings.
	//    continue
	//  }
	//  secret, err := secretManager.GetSecret(pod.Namespace, secretRef.Name)
	//  if err != nil {
	//    klog.InfoS("Unable to retrieve pull secret, the image pull may not succeed.", QUARK_POD, klog.KObj(pod), "secret", klog.KObj(secret), "err", err)
	//    continue
	//  }
	//
	//  pullSecrets = append(pullSecrets, *secret)
	// }

	return pullSecrets
}

pub fn IsHostNetworkPod(pod: &k8s::Pod) -> bool {
    return pod.spec.as_ref().unwrap().host_network.is_some() && 
        pod.spec.as_ref().unwrap().host_network.clone().unwrap();
}

pub fn ContainerLogFileName(containerName: &str, restartCount: i32) -> String {
	return format!("{}/{}", containerName, restartCount);
}

pub fn BuildContainerLogsDirectory(pod: &k8s::Pod, containerName: &str) -> Result<String> {
    let podNamespace = pod.metadata.namespace.as_ref().unwrap();
    let podName = pod.metadata.name.as_ref().unwrap();
    let podUID = pod.metadata.uid.as_ref().unwrap();
    let podPath = GetPodLogDir(DefaultPodLogsRootPath, podNamespace, podName, podUID);
    let containerPath = format!("{}/{}", podPath, containerName);

    if !Path::new(&containerPath).exists() {
        fs::create_dir(&containerPath)?;
        fs::set_permissions(&containerPath, fs::Permissions::from_mode(0o755))?;
    }

    return Ok(containerPath)
}

pub fn BuildPodLogsDirectory(podNamespace: &str, podName: &str, podUid: &str) -> Result<String> {
    let podPath = GetPodLogDir(DefaultPodLogsRootPath, podNamespace, podName, podUid);
    
    if !Path::new(&podPath).exists() {
        fs::create_dir(&podPath)?;
        fs::set_permissions(&podPath, fs::Permissions::from_mode(0o755))?;
    }

    return Ok(podPath)
}

pub type ContainerType = i32;
pub const Containers: ContainerType = 1 << 0;
pub const InitContainers: ContainerType = 1 << 0;
pub const EphemeralContainers: ContainerType = 1 << 0;
pub const AllContainers: ContainerType = Containers | InitContainers | EphemeralContainers;

pub fn AllFeatureEnabledContainers() -> ContainerType {
    return AllContainers;
}

pub type ContainerVisitor = fn(container: &k8s::Container, containerType: ContainerType) -> bool;

pub fn HasPrivileged(c: &k8s::Container, _containerType: ContainerType) -> bool {
    if c.security_context.is_some() 
        && c.security_context.as_ref().unwrap().privileged.is_some() 
        && c.security_context.as_ref().unwrap().privileged.unwrap() {
        return true;
    }
    return false;
}

pub fn HasPrivileged1(c: &k8s::EphemeralContainer, _containerType: ContainerType) -> bool {
    if c.security_context.is_some() 
        && c.security_context.as_ref().unwrap().privileged.is_some() 
        && c.security_context.as_ref().unwrap().privileged.unwrap() {
        return true;
    }
    return false;
}

pub fn HasPrivilegedContainer(podSpec: &k8s::PodSpec) -> bool {
    let mask = AllContainers;
    if mask & InitContainers != 0 && podSpec.init_containers.is_some() {
        for i in podSpec.init_containers.as_ref().unwrap() {{
            if HasPrivileged(i, InitContainers) {
                return true;
            }
        }}
    }

    if mask & Containers != 0 {
        for i in &podSpec.containers {{
            if HasPrivileged(i, Containers) {
                return true;
            }
        }}
    }

    if mask & EphemeralContainers != 0 && podSpec.ephemeral_containers.is_some() {
        for i in podSpec.ephemeral_containers.as_ref().unwrap() {{
            if HasPrivileged1(&i, EphemeralContainers) {
                return true;
            }
        }}
    }

    return false;
}

// ProtocolTCP is the TCP protocol.
pub const ProtocolTCP: &str = "TCP";
// ProtocolUDP is the UDP protocol.
pub const ProtocolUDP: &str = "UDP";
// ProtocolSCTP is the SCTP protocol.
pub const ProtocolSCTP: &str = "SCTP";

pub fn MakePortMappings(container: &k8s::Container) -> Vec<crictl::PortMapping> {
    let mut pms = Vec::new();
    if container.ports.is_none() {
        return pms;
    }

    for p in container.ports.as_ref().unwrap() {
        let pm = crictl::PortMapping {
            host_ip: p.host_ip.as_deref().unwrap_or("").to_string(),
            host_port: p.host_port.clone().unwrap_or(0), 
            container_port: p.container_port,
            protocol: ToRuntimeProtocol(p.protocol.as_deref().unwrap_or("ProtocolTCP")),
        };
        pms.push(pm);
    }

    return pms;
}

pub fn ToRuntimeProtocol(protocol: &str) -> i32 {
    match protocol {
        ProtocolTCP => return crictl::Protocol::Tcp as i32,
        ProtocolUDP => return crictl::Protocol::Udp as i32,
        ProtocolSCTP => return crictl::Protocol::Sctp as i32,
        _ => return crictl::Protocol::Tcp as i32,
    }
}

// namespacesForPod returns the criv1.NamespaceOption for a given pod.
// An empty or nil pod can be used to get the namespace defaults for v1.Pod.
pub fn NamespacesForPod(pod: &k8s::Pod) -> crictl::NamespaceOption {
	return crictl::NamespaceOption {
		ipc:     IpcNamespaceForPod(pod),
		network: NetworkNamespaceForPod(pod),
		pid:     PidNamespaceForPod(pod),
        ..Default::default()
	}
}

// A NamespaceMode describes the intended namespace configuration for each
// of the namespaces (Network, PID, IPC) in NamespaceOption. Runtimes should
// map these modes as appropriate for the technology underlying the runtime.
pub type NamespaceMode = i32;

// A POD namespace is common to all containers in a pod.
// For example, a container with a PID namespace of POD expects to view
// all of the processes in all of the containers in the pod.
pub const NamespaceMode_POD: NamespaceMode = 0;
// A CONTAINER namespace is restricted to a single container.
// For example, a container with a PID namespace of CONTAINER expects to
// view only the processes in that container.
pub const NamespaceMode_CONTAINER: NamespaceMode = 1;
// A NODE namespace is the namespace of the Kubernetes node.
// For example, a container with a PID namespace of NODE expects to view
// all of the processes on the host running the kubelet.
pub const NamespaceMode_NODE: NamespaceMode = 2;
// TARGET targets the namespace of another container. When this is specified,
// a target_id must be specified in NamespaceOption and refer to a container
// previously created with NamespaceMode CONTAINER. This containers namespace
// will be made to match that of container target_id.
// For example, a container with a PID namespace of TARGET expects to view
// all of the processes that container target_id can view.
pub const NamespaceMode_TARGET: NamespaceMode = 3;

pub fn IpcNamespaceForPod(pod: &k8s::Pod) -> NamespaceMode {
	if *pod.spec.as_ref().unwrap().host_ipc.as_ref().unwrap_or(&false) {
		return NamespaceMode_NODE
	}
	return NamespaceMode_POD
}

pub fn NetworkNamespaceForPod(pod: &k8s::Pod) -> NamespaceMode {
	if *pod.spec.as_ref().unwrap().host_network.as_ref().unwrap() {
		return NamespaceMode_NODE
	}
	return NamespaceMode_POD
}

pub fn PidNamespaceForPod(pod: &k8s::Pod) -> NamespaceMode {
	if *pod.spec.as_ref().unwrap().host_pid.as_ref().unwrap_or(&false) {
		return NamespaceMode_NODE
	}

    if pod.spec.as_ref().unwrap().share_process_namespace.is_some()  
        && *pod.spec.as_ref().unwrap().share_process_namespace.as_ref().unwrap() {
		return NamespaceMode_POD
	}

	// Note that PID does not default to the zero value for v1.Pod
	return NamespaceMode_CONTAINER
}

pub fn ConvertOverheadToLinuxResources(nodeConfig: &NodeConfigurationInner, pod: &k8s::Pod) -> crictl::LinuxContainerResources {
    let resource = crictl::LinuxContainerResources::default();
    
    let spec = pod.spec.as_ref().unwrap();
    if let Some(overhead) = &spec.overhead {
        let resource = QuarkResource::New(overhead).unwrap();

        return calculateLinuxResources(nodeConfig, resource.cpu, resource.cpu, resource.memory);
    }

    return resource;
}

pub fn ApplySandboxResources(nodeConfig: &NodeConfigurationInner, pod: &k8s::Pod, psc: &mut crictl::PodSandboxConfig) -> Result<()> {
    if psc.linux.is_none() {
        return Ok(());
    }

    let mut linux = psc.linux.as_mut().unwrap();
    linux.resources = Some(CalculateSandboxResources(nodeConfig, pod));
    linux.overhead = Some(ConvertOverheadToLinuxResources(nodeConfig, pod));
    return Ok(())
}

pub fn CalculateSandboxResources(nodeConfig: &NodeConfigurationInner, pod: &k8s::Pod) -> crictl::LinuxContainerResources {
    let (req, lim) = PodRequestsAndLimitsWithoutOverhead(pod);
    return calculateLinuxResources(nodeConfig, req.cpu, lim.cpu, lim.memory);
}

pub fn ContainerResource(container: &k8s::Container) -> (QuarkResource, QuarkResource) {
    if container.resources.is_none() {
        return (QuarkResource::default(), QuarkResource::default());
    }
    let req = if container.resources.as_ref().unwrap().requests.is_none() {
        QuarkResource::default()
    } else {
        QuarkResource::New(container.resources.as_ref().unwrap().requests.as_ref().unwrap()).unwrap()
    };
        
    let lim = if container.resources.as_ref().unwrap().requests.is_none() {
        QuarkResource::default()
    } else {
        QuarkResource::New(container.resources.as_ref().unwrap().limits.as_ref().unwrap()).unwrap()
    };
    return (req, lim);
}

pub fn PodRequestsAndLimitsWithoutOverhead(pod: &k8s::Pod) -> (QuarkResource, QuarkResource) {
    let mut reqs = QuarkResource::default();
    let mut limits = QuarkResource::default();
    for container in &pod.spec.as_ref().unwrap().containers {
        let (tmpReqs, tmpLimits) = ContainerResource(container);

        reqs.Add(&tmpReqs);
        limits.Add(&tmpLimits);
    }

    return (reqs, limits)
}

pub fn calculateLinuxResources(nodeConfig: &NodeConfigurationInner, cpureq: i64, cpulimit: i64, memlimit: i64) -> crictl::LinuxContainerResources {
    let mut resources = crictl::LinuxContainerResources::default();

    let cpushare;
    if cpureq == 0 && !cpulimit != 0 {
        cpushare = cpulimit;
    } else {
        cpushare = MilliCPUToShares(cpureq) as i64;
    }

    if memlimit != 0 {
        resources.memory_limit_in_bytes = memlimit;
    }

    resources.cpu_shares = cpushare;

    if nodeConfig.CPUCFSQuota {
        let cpuPeriod = quotaPeriod;
        /*
        if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.CPUCFSQuotaPeriod) {
			cpuPeriod = int64(nodeConfig.CPUCFSQuotaPeriod / time.Microsecond)
		} 
        */
        let cpuQuota = MilliCPUToQuota(cpulimit, cpuPeriod);
		resources.cpu_quota = cpuQuota;
		resources.cpu_period = cpuPeriod;
    }

    return resources;
}

pub const MinShares: i64 = 2;
pub const MaxShares: i64 = 262144;

pub const SharesPerCPU: i64 = 1024;
pub const MilliCPUToCPU: i64 = 1000;

pub const milliCPUToCPU: i64 = 1000;

// 100000 is equivalent to 100ms
pub const quotaPeriod: i64    = 100000;
pub const minQuotaPeriod: i64 = 1000;

pub fn MilliCPUToShares(milliCPU: i64) -> u64 {
    if milliCPU == 0 {
        // Docker converts zero milliCPU to unset, which maps to kernel default
		// for unset: 1024. Return 2 here to really match kernel default for
		// zero milliCPU.
        return 0;
    }

    // Conceptually (milliCPU / milliCPUToCPU) * sharesPerCPU, but factored to improve rounding.
	let shares = (milliCPU * SharesPerCPU) / MilliCPUToCPU;
	if shares < MinShares {
		return MinShares as u64
	}
	if shares > MaxShares {
		return MaxShares as u64
	}
	return shares as u64
}

pub fn MilliCPUToQuota(milliCPU: i64, period: i64) -> i64 {
    // CFS quota is measured in two values:
	//  - cfs_period_us=100ms (the amount of time to measure usage across)
	//  - cfs_quota=20ms (the amount of cpu time allowed to be used across a period)
	// so in the above example, you are limited to 20% of a single CPU
	// for multi-cpu environments, you just scale equivalent amounts
	// see https://www.kernel.org/doc/Documentation/scheduler/sched-bwc.txt for details
	if milliCPU == 0 {
		return 0
	}

	// we then convert your milliCPU to a value normalized over a period
	let mut quota = (milliCPU * period) / milliCPUToCPU;

	// quota needs to be a minimum of 1ms.
	if quota < minQuotaPeriod {
		quota = minQuotaPeriod;
	}

	return quota;
}

// systemdSuffix is the cgroup name suffix for systemd
pub const systemdSuffix: &str = ".slice";
// MemoryMin is memory.min for cgroup v2
pub const MemoryMin: &str = "memory.min";
// MemoryHigh is memory.high for cgroup v2
pub const MemoryHigh: &str = "memory.high";

pub fn generateLinuxContainerConfig(
    nodeConfig: &NodeConfigurationInner, 
    container: &k8s::Container, 
    pod: &Arc<RwLock<k8s::Pod>>, 
    uid: Option<i64>, 
    username: &str, 
    enforceMemoryQoS: bool
) -> crictl::LinuxContainerConfig {
    let pod = &pod.read().unwrap();
    let mut lc = crictl::LinuxContainerConfig {
        resources: Some(crictl::LinuxContainerResources::default()),
        security_context: Some(DetermineEffectiveSecurityContext(pod, container, uid, username, nodeConfig.SeccompDefault, &nodeConfig.SeccompProfileRoot)),
    };

    let (req, lim) = ContainerResource(container);

    let mem = lc.resources.as_ref().unwrap().memory_limit_in_bytes;

    lc.resources = Some(calculateLinuxResources(nodeConfig, req.cpu, lim.cpu, lim.memory));
    lc.resources.as_mut().unwrap().oom_score_adj = nodeConfig.OOMScoreAdj as i64;
    //lc.resources.as_mut().unwrap().hugepage_limits = nodeConfig.OOMScoreAdj as i64;
    lc.resources.as_mut().unwrap().memory_swap_limit_in_bytes = mem;

    if enforceMemoryQoS {
        let mut unified = BTreeMap::new();
        let (req, lim) = ContainerResource(container);
        let memoryRequest = req.memory;
        let memoryLimit = lim.memory;
        if memoryRequest != 0 {
            unified.insert(MemoryMin.to_string(), format!("{}", memoryRequest));
        }

        // If container sets limits.memory, we set memory.high=pod.spec.containers[i].resources.limits[memory] * memory_throttling_factor
		// for container level cgroup if memory.high>memory.min.
		// If container doesn't set limits.memory, we set memory.high=node_allocatable_memory * memory_throttling_factor
		// for container level cgroup.
		let mut memoryHigh: i64 = 0;
        if memoryLimit != 0 {
            memoryHigh = (memoryLimit as f64 * DefaultMemoryThrottlingFactor) as i64;
        } else {
            // allocatable := m.getNodeAllocatable()
			// allocatableMemory, ok := allocatable[v1.ResourceMemory]
			// if ok && allocatableMemory.Value() > 0 {
			//  memoryHigh = int64(float64(allocatableMemory.Value()) * m.memoryThrottlingFactor)
			// }
        }

        if memoryHigh > memoryRequest {
            unified.insert(MemoryHigh.to_string(), format!("{}", memoryHigh));
        }

        if unified.len() > 0 {
            for (k, v) in &unified {
                lc.resources.as_mut().unwrap().unified.insert(k.to_string(), v.to_string());
            }
        }
    }

    return lc
} 

pub fn MakeDevice(opts: &RunContainerOptions) -> Vec<crictl::Device> {
    let mut devices = Vec::with_capacity(opts.devices.len());

    for dev in &opts.devices {
        devices.push(crictl::Device {
            host_path: dev.pathOnHost.clone(),
            container_path: dev.pathInContainer.clone(),
            permissions: dev.permissions.clone(),
        })
    }

    return devices;
}


pub async fn MakeMounts(opts: &RunContainerOptions, container: &k8s::Container, namespace: &str, podType: &PodType) -> Result<Vec<crictl::Mount>> {
    let mut volumeMounts = Vec::new();

    for v in &opts.mounts {
        let mount = crictl::Mount {
            host_path: v.hostPath.clone(),
            container_path: v.containerPath.clone(),
            readonly: v.readOnly,
            selinux_relabel: v.SELinuxRelabel,
            propagation: v.Propagation as i32,
        };

        volumeMounts.push(mount);
    }

    let mount = crictl::Mount {
        host_path: NODEAGENT_CONFIG.FuncAgentSvcSocketAddr(),
        container_path: NODEAGENT_CONFIG.FuncAgentSvcSocketLocalAddr(),
        selinux_relabel: false,
        ..Default::default()
    };
    volumeMounts.push(mount);

    let logfolder = format!("{}/func/{}", DefaultNodeFuncLogFolder, namespace);
    std::fs::create_dir_all(&logfolder).unwrap();
    
    let mount = crictl::Mount {
        host_path: logfolder,
        container_path: DefaultNodeFuncLogFolder.to_owned(),
        selinux_relabel: false,
        ..Default::default()
    };
    volumeMounts.push(mount);

    match podType {
        PodType::Normal => (),
        PodType::Python(objName) => {
            let funcDir = FUNCDIR_MGR.get().unwrap().GetFuncDir(namespace, objName).await?;
            let mount = crictl::Mount {
                host_path: funcDir.to_owned(),
                container_path: "/app/src/qserverless/func".to_string(),
                selinux_relabel: false,
                ..Default::default()
            };
            volumeMounts.push(mount);
        }
    }

    // The reason we create and mount the log file in here (not in kubelet) is because
	// the file's location depends on the ID of the container, and we need to create and
	// mount the file before actually starting the container.
    if opts.podContainerDir.len() != 0 && container.termination_message_path.as_ref().unwrap().len() != 0 {
        // Because the PodContainerDir contains pod uid and container name which is unique enough,
		// here we just add a random id to make the path unique for different instances
		// of the same container.
		let cid = MakeUID();
        let containerLogPath = format!("{}/{}", &opts.podContainerDir, cid);
        fs::create_dir_all(&containerLogPath).unwrap();
        let perms = Permissions::from_mode(0o666);
        fs::set_permissions(&containerLogPath, perms).unwrap();

        let containerLogPath = containerLogPath;
        let terminationMessagePath = container.termination_message_path.clone();
        let selinuxRelabel = false; //selinux.GetEnabled()
        volumeMounts.push(crictl::Mount {
            host_path: containerLogPath,
            container_path: terminationMessagePath.as_deref().unwrap_or("").to_string(),
            selinux_relabel: selinuxRelabel,
            ..Default::default()
        });
    }

    return Ok(volumeMounts);
}

pub fn MakeAbsolutePath(_goos: &str, path: &str) -> String {
    return path.to_string();
}

pub fn MakeUID() -> String {
    let mut rng = rand::thread_rng();

    let n: u64 = rng.gen();
    return format!("{:0<8}", n);
}