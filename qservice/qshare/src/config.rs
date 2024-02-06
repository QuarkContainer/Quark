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

use std::{collections::{BTreeMap, BTreeSet}, time::Duration};
use std::path::Path;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use core::ops::Deref;
use serde::{Deserialize, Serialize};

use crate::common::*;

pub const SYSTEM_CONFIGS : &str = r#"
{
    "product": {
        "nodeAgentConfig" : {
            "rootPath"      : "/var/lib/quark/nodeagent",
            "cidr"          : "10.1.1.0/8",
            "nodeName"      : "",
            "hostIP"        : "127.0.0.1",
            "nodeAgentPort" : 1235,
            "blobSvcPort"   : 8892,
            "funcSvcAddr"   : "http://127.0.0.1:8891",
            "nodeMgrAddrs"  : ["http://127.0.0.1:8888"]
        },
        "testConfig" : {
            "nodeAgentUnixSocket": "/var/lib/quark/nodeagent/sock"
        }
    },
    "node1": {
        "nodeAgentConfig" : {
            "rootPath"      : "/var/lib/quark/nodeagent",
            "cidr"          : "10.1.2.0/8",
            "nodeName"      : "node1",
            "hostIP"        : "127.0.0.1",
            "nodeAgentPort" : 1235,
            "blobSvcPort"   : 8892,
            "funcSvcAddr"   : "http://127.0.0.1:8891",
            "nodeMgrAddrs"  : ["http://127.0.0.1:8888"]
        },
        "testConfig" : {
            "nodeAgentUnixSocket": "/var/lib/quark/nodeagent/node1/sock"
        }
    },
    "node2": {
        "nodeAgentConfig" : {
            "rootPath"      : "/var/lib/quark/nodeagent",
            "cidr"          : "10.1.3.0/8",
            "nodeName"      : "node2",
            "hostIP"        : "127.0.0.1",
            "nodeAgentPort" : 1236,
            "blobSvcPort"   : 8893,
            "funcSvcAddr"   : "http://127.0.0.1:8891",
            "nodeMgrAddrs"  : ["http://127.0.0.1:8888"]
        },
        "testConfig" : {
            "nodeAgentUnixSocket": "/var/lib/quark/nodeagent/node2/sock"
        }
    }
}"#;    

pub const TEST_CONFIG_NAME : &str = "node1";
//pub const SYSTEM_CONFIG : &str = SYSTEM_CONFIG_SIMPLE;

pub const TSOT_CNI_PORT: u16 = 1234;
pub const TSOT_CONNECTION_PORT: u16 = 1235;

#[derive(Debug, Deserialize, Serialize)]
pub struct SystemConfig {
    pub nodeAgentConfig: NodeAgentConfig,
    pub testConfig: TestConfig,
}

impl SystemConfig {
    pub fn NodeAgentConfig(&self) -> NodeAgentConfig {
        return self.nodeAgentConfig.clone();
    }

    pub fn TestConfig(&self) -> TestConfig {
        return self.testConfig.clone();
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct NodeAgentConfig {
    pub rootPath: String,
    pub cidr: String,
    pub nodeName: String,
    pub hostIP: String,
    pub nodeAgentPort: u16,
    pub blobSvcPort: i32,
    pub funcSvcAddr: String,
    pub nodeMgrAddrs: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TestConfig {
    pub nodeAgentUnixSocket: String,
}

impl NodeAgentConfig {
    pub fn Base(&self) -> String {
        if self.nodeName.len() == 0 {
            return format!("{}", &self.rootPath);
        }

        return format!("{}/{}", &self.rootPath, &self.nodeName); 
    }

    pub fn FuncAgentSvcSocketLocalAddr(&self) -> String {
        return format!("{}/sock", &self.rootPath); 
    }

    pub fn FuncAgentSvcSocketAddr(&self) -> String {
        return format!("{}/sock", self.Base());
    }

    pub fn BlobSvcAddr(&self) -> String {
        return format!("{}:{}", self.hostIP, self.blobSvcPort);
    }

    pub fn BlobStoreMetaPath(&self) -> String {
        return format!("{}/blobstore/meta", self.Base());
    }

    pub fn BlobStoreDataPath(&self) -> String {
        return format!("{}/blobstore/data", self.Base());
    }

    pub fn FuncSvcAddr(&self) -> String {
        return self.funcSvcAddr.clone();
    }

    pub fn nodeMgrAddrs(&self) -> Vec<String> {
        return self.nodeMgrAddrs.clone();
    }

    pub fn NodeName(&self) -> String {
        if self.nodeName.len() == 0 {
            return hostname::get().unwrap().to_str().unwrap().to_string();
        } else {
            return self.nodeName.clone();
        }
    }
}

//use k8s_openapi::api::core::v1::{self as k8s};

pub const DecimalExponent : &str = "DecimalExponent"; // e.g., 12e6
pub const BinarySI        : &str = "BinarySI";        // e.g., 12Mi (12 * 2^20)
pub const DecimalSI       : &str = "DecimalSI";       // e.g., 12M  (12 * 10^6)

pub const CPU                               : &str = "CPU";
pub const Memory                            : &str = "Memory";
pub const PID                               : &str = "PID";
pub const DefaultRootPath                   : &str = "/var/lib/quark/nodeagent";
pub const DefaultDBName                     : &str = "nodeagent.sqlite";
pub const DefaultContainerRuntimeEndpoint   : &str = "/run/containerd/containerd.sock";
pub const DefaultMaxPods                    : i32 = 2000;
pub const DefaultPodPidLimits               : i32 = -1;
pub const DefaultCgroupRoot                 : &str = "/";
pub const DefaultCgroupDriver               : &str = "cgroupfs";
pub const DefaultMaxContainerPerPod         : i32 = 10;
pub const DefaultMounter                    : &str = "mount";
pub const DefaultPodsPerCore                : i32 = 0;
pub const DefaultNodeAgentCgroupName        : &str = "";
pub const DefaultSystemCgroupName           : &str = "";
pub const DefaultPodsDirName                : &str = "pods";
pub const DefaultPodLogsRootPath            : &str = "/var/log/pods";
pub const DefaultVolumesDirName             : &str = "volumes";
pub const DefaultVolumeSubpathsDirName      : &str = "volume-subpaths";
pub const DefaultVolumeDevicesDirName       : &str = "volumeDevices";
pub const DefaultPluginsDirName             : &str = "plugins";
pub const DefaultPluginsRegistrationDirName : &str = "plugins_registry";
pub const DefaultContainersDirName          : &str = "containers";
pub const DefaultPluginContainersDirName    : &str = "plugin-containers";
pub const DefaultPodResourcesDirName        : &str = "pod-resources";
pub const DefaultMemoryThrottlingFactor     : f64 = 0.8;
pub const DefaultSessionServicePort         : i32 = 1022;
pub const DefaultNodePortStartingNum        : i32 = 1024;
pub const KubeletPluginsDirSELinuxLabel     : &str = "system_u:object_r:container_file_t:s0";
pub const DefaultPodCgroupName              : &str = "containers";
pub const DefaultRuntimeHandler             : &str = "quarkd"; // "runc";
pub const DefaultPodConcurrency             : i32 = 5;

#[derive(Debug, PartialEq, Eq)]
pub enum ResourceName {
    	// CPU, in cores. (500m = .5 cores)
	ResourceCPU, // ResourceName = "cpu"
	// Memory, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
	ResourceMemory, // ResourceName = "memory"
	// Volume size, in bytes (e,g. 5Gi = 5GiB = 5 * 1024 * 1024 * 1024)
	ResourceStorage, // ResourceName = "storage"
	// Local ephemeral storage, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
	// The resource name for ResourceEphemeralStorage is alpha and it can change across releases.
	ResourceEphemeralStorage, // ResourceName = "ephemeral-storage"
}

pub type ResourceList = BTreeMap<ResourceName, String>;

#[derive(Debug, Default)]
pub struct NodeConfigurationInner {
    pub ReservedMem              : u64,
    pub ReservedCpuCores         : u32,
    pub ContainerRuntime         : String,
	pub ContainerRuntimeEndpoint : String,
	pub CgroupRoot               : String,
	pub CgroupDriver             : String,
	pub DatabaseURL              : String, // /var/lib/nodeagent/db/nodeagent.sqlite
	pub NodeMgreUrls           : Vec<String>,
	pub Hostname                 : String,
	pub MemoryQoS                : bool,
	pub DisableSwap              : bool,
	pub MaxPods                  : i32,
	pub MaxContainerPerPod       : i32,
	pub MounterPath              : String, // a mounter bin path, leave it empty if use default
	pub NodeIP                   : String,
	pub NodeAgentCgroupName      : String,
	pub OOMScoreAdj              : i32,
	pub QOSReserved              : BTreeMap<ResourceName, i64>,
	pub PodLogRootPath           : String,
	pub PodPidLimits             : i32, // default 100
	pub PodsPerCore              : i32,
	pub PodCgroupName            : String,
	pub RootPath                 : String, // node agent state root, /var/lib/nodeagent/
	pub RuntimeHandler           : String,
	pub ProtectKernelDefaults    : bool,
	pub SystemCgroupName         : String,
	pub EnforceCPULimits         : bool,
	pub CPUCFSQuota              : bool,
	pub CPUCFSQuotaPeriod        : Duration,
	pub ReservedSystemCPUs       : BTreeSet<i32>,
	pub EnforceNodeAllocatable   : BTreeSet<String>,
	pub NodeAgentReserved        : ResourceList,
	pub SystemReserved           : ResourceList,
	pub SeccompProfileRoot       : String,
	pub SeccompDefault           : bool,
	pub NodePortStartingNo       : i32,
	pub SessionServicePort       : i32,
	pub PodConcurrency           : i32,
}

impl NodeConfigurationInner {
    pub fn DefaultNodeConfiguration() -> Result<Self> {
        use gethostname::gethostname;
        use local_ip_address::local_ip;

        let hostname = gethostname().to_str().unwrap_or("").to_string();
        let localIp = local_ip().unwrap().to_string();

        return Ok(Self {
            ReservedMem:  4 * 1024 * 1024 * 1024,
            ReservedCpuCores:         4, 
            ContainerRuntime:         "remote".to_string(),
            ContainerRuntimeEndpoint: DefaultContainerRuntimeEndpoint.to_string(),
            CgroupRoot:               DefaultCgroupRoot.to_string(),
            CgroupDriver:             DefaultCgroupDriver.to_string(),
            DatabaseURL:              format!("file:{}/db/{}?cache=shared&mode=rwc", DefaultRootPath, DefaultDBName),
            NodeMgreUrls:             Vec::new(),
            Hostname:                 hostname,
            MaxPods:                  DefaultMaxPods,
            MaxContainerPerPod:       DefaultMaxContainerPerPod,
            MounterPath:              DefaultMounter.to_string(),
            NodeIP:                   localIp,
            NodeAgentCgroupName:      DefaultNodeAgentCgroupName.to_string(),
            OOMScoreAdj:              -999,
            QOSReserved:              BTreeMap::new(),
            PodConcurrency:           DefaultPodConcurrency,
            PodLogRootPath:           DefaultPodLogsRootPath.to_string(),
            PodPidLimits:             DefaultPodPidLimits,
            PodsPerCore:              DefaultPodsPerCore,
            PodCgroupName:            DefaultPodCgroupName.to_string(),
            RootPath:                 DefaultRootPath.to_string(),
            RuntimeHandler:           DefaultRuntimeHandler.to_string(),
            SeccompProfileRoot:       format!("{}/{}", DefaultRootPath, "seccomp"),
            NodePortStartingNo:       DefaultNodePortStartingNum,
            SessionServicePort:       DefaultSessionServicePort,
            SeccompDefault:           false,
            ProtectKernelDefaults:    false,
            SystemCgroupName:         DefaultSystemCgroupName.to_string(),
            MemoryQoS:                false,
            DisableSwap:              true,
            EnforceCPULimits:         true,
            CPUCFSQuota:              true,
            CPUCFSQuotaPeriod:        Duration::from_millis(100),
            ReservedSystemCPUs:       BTreeSet::new(),
            EnforceNodeAllocatable:   BTreeSet::new(),
            NodeAgentReserved:        BTreeMap::new(),
            SystemReserved:           BTreeMap::new(),
        })
    }

    pub fn ValidateNodeConfiguration(&self) -> Result<()> {
        if !Path::new(&self.RootPath).exists() {
            fs::create_dir(&self.RootPath)?;
            fs::set_permissions(&self.RootPath, fs::Permissions::from_mode(0o655))?;
        }

        if !Path::new(&self.PodLogRootPath).exists() {
            fs::create_dir(&self.PodLogRootPath)?;
            fs::set_permissions(&self.PodLogRootPath, fs::Permissions::from_mode(0o655))?;
        }

        return Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct NodeConfiguration(Arc<NodeConfigurationInner>);

impl Deref for NodeConfiguration {
    type Target = Arc<NodeConfigurationInner>;

    fn deref(&self) -> &Arc<NodeConfigurationInner> {
        &self.0
    }
}

impl NodeConfiguration {
    pub fn Default() -> Result<Self> {
        let inner = NodeConfigurationInner::DefaultNodeConfiguration()?;
        return Ok(Self(Arc::new(inner)));
    }
}

pub fn GetPodsDir(rootPath: &str) -> String {
    return format!("{}/{}", rootPath, DefaultPodsDirName);
}

// getPodLogDir returns the full path to the pod log dir
pub fn GetPodLogDir(rootPath: &str, podNamespace: &str, podName: &str, podUID: &str) -> String {
	return format!("{}/{}/{}/{}", rootPath, podNamespace, podName, podUID);
}

// getPluginsDir returns the full path to the directory under which plugin
// directories are created.  Plugins can use these directories for data that
// they need to persist.  Plugins should create subdirectories under this named
// after their own names.
pub fn GetPluginsDir(rootPath: &str) -> String {
	return format!("{}/{}", rootPath, DefaultPluginsDirName)
}

// getPluginsRegistrationDir returns the full path to the directory under which
// plugins socket should be placed to be registered.
// More information is available about plugin registration in the pluginwatcher
// module
pub fn GetPluginsRegistrationDir(rootPath: &str) -> String {
	return format!("{}/{}", rootPath, DefaultPluginsRegistrationDirName)
}

// getPluginDir returns a data directory name for a given plugin name.
// Plugins can use these directories to store data that they need to persist.
// For per-pod plugin data, see getPodPluginDir.
pub fn GetPluginDir(rootPath: &str, pluginName: &str) -> String {
	return format!("{}/{}", rootPath, pluginName)
}

// getVolumeDevicePluginsDir returns the full path to the directory under which plugin
// directories are created.  Plugins can use these directories for data that
// they need to persist.  Plugins should create subdirectories under this named
// after their own names.
pub fn GetVolumeDevicePluginsDir(rootPath: &str) -> String {
	return format!("{}/{}", rootPath, DefaultPluginsDirName)
}

// getPodDir returns the full path to the per-pod directory for the pod with
// the given UID.
pub fn GetPodDir(rootPath: &str, podUID: &str) -> String {
	return format!("{}/{}", GetPodsDir(rootPath), podUID)
}

// getPodVolumesSubpathsDir returns the full path to the per-pod subpaths directory under
// which subpath volumes are created for the specified pod.  This directory may not
// exist if the pod does not exist or subpaths are not specified.
pub fn GetPodVolumeSubpathsDir(rootPath: &str, podUID: &str) -> String {
	return format!("{}/{}", GetPodDir(rootPath, podUID), DefaultVolumeSubpathsDirName)
}

// getPodVolumesDir returns the full path to the per-pod data directory under
// which volumes are created for the specified pod.  This directory may not
// exist if the pod does not exist.
pub fn GetPodVolumesDir(rootPath: &str, podUID: &str) -> String {
	return format!("{}/{}", GetPodDir(rootPath, podUID), DefaultVolumesDirName)
}

// getPodVolumeDir returns the full path to the directory which represents the
// named volume under the named plugin for specified pod.  This directory may not
// exist if the pod does not exist.
pub fn GetPodVolumeDir(rootPath: &str, podUID: &str, pluginName: &str, volumeName: &str) -> String {
	return format!("{}/{}/{}", GetPodVolumesDir(rootPath, podUID), pluginName, volumeName);
}

// getPodVolumeDevicesDir returns the full path to the per-pod data directory under
// which volumes are created for the specified pod. This directory may not
// exist if the pod does not exist.
pub fn GetPodVolumeDevicesDir(rootPath: &str, podUID: &str) -> String {
	return format!("{}/{}", GetPodDir(rootPath, podUID), DefaultVolumeDevicesDirName)
}

// getPodVolumeDeviceDir returns the full path to the directory which represents the
// named plugin for specified pod. This directory may not exist if the pod does not exist.
pub fn GetPodVolumeDeviceDir(rootPath: &str, podUID: &str, pluginName: &str) -> String {
	return format!("{}/{}", GetPodDir(rootPath, podUID), pluginName)
}

// getPodPluginsDir returns the full path to the per-pod data directory under
// which plugins may store data for the specified pod.  This directory may not
// exist if the pod does not exist.
pub fn GetPodPluginsDir(rootPath: &str, podUID: &str) -> String {
	return format!("{}/{}", GetPodDir(rootPath, podUID), DefaultPluginsDirName)
}

// getPodPluginDir returns a data directory name for a given plugin name for a
// given pod UID.  Plugins can use these directories to store data that they
// need to persist.  For non-per-pod plugin data, see getPluginDir.
pub fn GetPodPluginDir(rootPath: &str, podUID: &str, pluginName: &str) -> String {
	return format!("{}/{}", GetPodPluginsDir(rootPath, podUID), pluginName)
}

// getPodContainerDir returns the full path to the per-pod data directory under
// which container data is held for the specified pod.  This directory may not
// exist if the pod or container does not exist.
pub fn GetPodContainerDir(rootPath: &str, podUID: &str, ctrName: &str) -> String {
	return format!("{}/{}", GetPodDir(rootPath, podUID), ctrName);
}

// getPodResourcesSocket returns the full path to the directory containing the pod resources socket
pub fn GetPodResourcesDir(rootPath: &str) -> String {
	return format!("{}/{}", rootPath, DefaultPodResourcesDirName)
}

