// Copyright (c) 2021 Quark Container Authors/Google 2014
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

use chrono::{DateTime, Utc};
use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

mod go_date_format {
    use chrono::{DateTime, Utc, TimeZone};
    use serde::{self, Deserialize, Serializer, Deserializer};

    //const FORMAT: &'static str = "%Y-%m-%dT%H:%M:%S.f%Z";
    const FORMAT: &'static str = "%Y-%m-%dT%H:%M:%S%.9f%Z";

    // The signature of a serialize_with function must follow the pattern:
    //
    //    fn serialize<S>(&T, S) -> Result<S::Ok, S::Error>
    //    where
    //        S: Serializer
    //
    // although it may also be generic over the input types T.
    pub fn serialize<S>(
        date: &DateTime<Utc>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = format!("{}", date.format(FORMAT));
        serializer.serialize_str(&s)
    }

    // The signature of a deserialize_with function must follow the pattern:
    //
    //    fn deserialize<'de, D>(D) -> Result<T, D::Error>
    //    where
    //        D: Deserializer<'de>
    //
    // although it may also be generic over the output types T.
    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<DateTime<Utc>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Utc.datetime_from_str(&s, FORMAT).map_err(serde::de::Error::custom)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MachineInfo {
    // The time of this information point.
    #[serde(default, rename = "timestamp", with = "go_date_format")]
	pub Timestamp: DateTime<Utc>, // `json:"timestamp"`

	// Vendor id of CPU.
	#[serde(default, rename = "vendor_id")]
	pub CPUVendorID: String, // `json:"vendor_id"`

	// The number of cores in this machine.
	#[serde(rename = "num_cores")]
	pub NumCores: i32, // `json:"num_cores"`

	// The number of physical cores in this machine.
	#[serde(default, rename = "num_physical_cores")]
	pub NumPhysicalCores: i32, // `json:"num_physical_cores"`

	// The number of cpu sockets in this machine.
	#[serde(default, rename = "num_sockets")]
	pub NumSockets: i32, // `json:"num_sockets"`

	// Maximum clock speed for the cores, in KHz.
	#[serde(rename = "cpu_frequency_khz")]
	pub CpuFrequency: u64, // `json:"cpu_frequency_khz"`

	// The amount of memory (in bytes) in this machine
	#[serde(rename = "memory_capacity")]
	pub MemoryCapacity: u64, // `json:"memory_capacity"`

	// The amount of swap (in bytes) in this machine
	#[serde(default, rename = "swap_capacity")]
	pub SwapCapacity : u64, //`json:"swap_capacity"`

	// Memory capacity and number of DIMMs by memory type
	#[serde(default, rename = "memory_by_type")]
	pub MemoryByType: BTreeMap<String, MemoryInfo>, //`json:"memory_by_type"`

	// #[serde(rename = "nvm")]
	// pub NVMInfo: NVMInfo, // `json:"nvm"`

	// HugePages on this machine.
	#[serde(rename = "hugepages")]
	pub HugePages: Vec<HugePagesInfo>, // `json:"hugepages"`

	// The machine id
	#[serde(rename = "machine_id")]
	pub MachineID: String, // `json:"machine_id"`

	// The system uuid
	#[serde(rename = "system_uuid")]
	pub SystemUUID: String, // `json:"system_uuid"`

	// The boot id
	#[serde(rename = "boot_id")]
	pub BootID: String, // `json:"boot_id"`

	// Filesystems on this machine.
	#[serde(rename = "filesystems")]
	pub Filesystems: Vec<FsInfo>, // `json:"filesystems"`

	// Disk map
	#[serde(rename = "disk_map")]
	pub DiskMap: BTreeMap<String, DiskInfo>, // `json:"disk_map"`

	// Network devices
	#[serde(rename = "network_devices")]
	pub NetworkDevices: Vec<NetInfo>, // `json:"network_devices"`

	// Machine Topology
	// Describes cpu/memory layout and hierarchy.
	// #[serde(default, rename = "topology")]
	// pub Topology: Vec<Node>, // `json:"topology"`

	// Cloud provider the machine belongs to.
	#[serde(rename = "cloud_provider")]
	pub CloudProvider: String, // `json:"cloud_provider"`

	// Type of cloud instance (e.g. GCE standard) the machine is.
	#[serde(rename = "instance_type")]
	pub InstanceType: String, // `json:"instance_type"`

	// ID of cloud instance (e.g. instance-1) given to it by the cloud provider.
	#[serde(rename = "instance_id")]
	pub InstanceID: String, // `json:"instance_id"`
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DiskInfo {
	// device name
	#[serde(rename = "name")]
	pub Name: String, // `json:"name"`

	// Major number
	#[serde(rename = "major")]
	pub Major: u64, // `json:"major"`

	// Minor number
	#[serde(rename = "minor")]
	pub Minor: u64, // `json:"minor"`

	// Size in bytes
	#[serde(rename = "size")]
	pub Size: u64, // `json:"size"`

	// I/O Scheduler - one of "none", "noop", "cfq", "deadline"
	#[serde(rename = "scheduler")]
	pub Scheduler: String, // `json:"scheduler"`
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NetInfo {
	// Device name
	#[serde(rename = "name")]
	pub Name: String, // `json:"name"`

	// Mac Address
	#[serde(rename = "mac_address")]
	pub MacAddress: String, // `json:"mac_address"`

	// Speed in MBits/s
	#[serde(rename = "speed")]
	pub Speed: i64, // `json:"speed"`

	// Maximum Transmission Unit
	#[serde(rename = "mtu")]
	pub Mtu: i64, // `json:"mtu"`
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Node {
	// Device name
	#[serde(rename = "node_id")]
	pub Id: i32, // `json:"node_id"`
	
    // Per-node memory
	// Device name
	#[serde(rename = "memory")]
	pub Memory: u64, //        `json:"memory"`
	
    // Device name
	#[serde(rename = "hugepages")]
	pub HugePages: Vec<HugePagesInfo>, // `json:"hugepages"`
	
    // Device name
	#[serde(rename = "cores")]
	pub Cores: Vec<Core>, //          `json:"cores"`
	
    // Device name
	#[serde(rename = "caches")]
	pub Caches: Vec<Cache>, //         `json:"caches"`
	
    // Device name
	#[serde(default, rename = "distances")]
	pub Distances: Vec<u64>, //        `json:"distances"`
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Core {
	#[serde(rename = "core_id")]
	pub Id : i32, //     `json:"core_id"`
	#[serde(rename = "thread_ids")]
	pub Threads: Vec<i32>, //   `json:"thread_ids"`
	#[serde(rename = "caches")]
	pub Caches: Vec<Cache>, // `json:"caches"`
	#[serde(default, rename = "uncore_caches")]
	pub UncoreCaches: Vec<Cache>, // `json:"uncore_caches"`
	#[serde(default, rename = "socket_id")]
	pub SocketID: i32, //     `json:"socket_id"`
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Cache {
	// Id of memory cache
	#[serde(default, rename = "id")]
	pub Id: i32, // `json:"id"`
	// Size of memory cache in bytes.
	#[serde(rename = "size")]
	pub Size: u64, // `json:"size"`
	// Type of memory cache: data, instruction, or unified.
	#[serde(rename = "type")]
	pub Type: String, // `json:"type"`
	// Level (distance from cpus) in a multi-level cache hierarchy.
	#[serde(rename = "level")]
	pub Level: i32, // `json:"level"`
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FsInfo {
	// Block device associated with the filesystem.
	#[serde(rename = "device")]
	pub Device: String, // `json:"device"`
	// DeviceMajor is the major identifier of the device, used for correlation with blkio stats
	/*#[serde(rename = "-")]
	pub DeviceMajor: u64, // `json:"-"`
	// DeviceMinor is the minor identifier of the device, used for correlation with blkio stats
	#[serde(rename = "-")]
	pub DeviceMinor: u64, // `json:"-"`*/

	// Total number of bytes available on the filesystem.
	#[serde(rename = "capacity")]
	pub Capacity: u64, // `json:"capacity"`

	// Type of device.
	#[serde(rename = "type")]
	pub Type: String, //`json:"type"`

	// Total number of inodes available on the filesystem.
	#[serde(rename = "inodes")]
	pub Inodes: u64, // `json:"inodes"`

	// HasInodes when true, indicates that Inodes info will be available.
	#[serde(rename = "has_inodes")]
	pub HasInodes: bool, // `json:"has_inodes"`
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HugePagesInfo {
	// huge page size (in kB)
	#[serde(rename = "page_size")]
	pub PageSize: u64, // `json:"page_size"`

	// number of huge pages
	#[serde(rename = "num_pages")]
	pub NumPages: u64, // `json:"num_pages"`
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MemoryInfo {
    // The amount of memory (in bytes).
    #[serde(rename = "capacity")]
	pub Capacity: u64, // `json:"capacity"`

	// Number of memory DIMMs.
	#[serde(rename = "dimm_count")]
	pub DimmCount: u32, // `json:"dimm_count"`
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NVMInfo {
	// The total NVM capacity in bytes for memory mode.
	#[serde(rename = "memory_mode_capacity")]
	pub MemoryModeCapacity: u64, // `json:"memory_mode_capacity"`

	//The total NVM capacity in bytes for app direct mode.
	#[serde(default, rename = "direct_mode_capacity")]
	pub AppDirectModeCapacity: u64, // `json:"app direct_mode_capacity"`

	// Average power budget in watts for NVM devices configured in BIOS.
	#[serde(rename = "avg_power_budget")]
	pub AvgPowerBudget: u32, // `json:"avg_power_budget"`
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct VersionInfo {
	// Kernel version.
    #[serde(rename = "memory_mode_capacity")]
    pub KernelVersion: String, // `json:"kernel_version"`

    // OS image being used for cadvisor container, or host image if running on host directly.
    #[serde(rename = "container_os_version")]
    pub ContainerOsVersion: String, // `json:"container_os_version"`

    // Docker version.
    #[serde(rename = "docker_version")]
    pub DockerVersion: String, // `json:"docker_version"`

    // Docker API Version
    #[serde(rename = "docker_api_version")]
    pub DockerAPIVersion: String, // `json:"docker_api_version"`

    // cAdvisor version.
    #[serde(rename = "cadvisor_version")]
    pub CadvisorVersion: String, // `json:"cadvisor_version"`

    // cAdvisor git revision.
    #[serde(rename = "cadvisor_revision")]
    pub CadvisorRevision: String, // `json:"cadvisor_revision"`
}