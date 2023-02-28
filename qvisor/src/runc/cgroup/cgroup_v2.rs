// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors / Runc Authors / containerd Authors
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

use std::fs::File;
use std::io::prelude::*;
use std::io::SeekFrom;

use crate::qlib::common::*;
use crate::runc::oci::*;

use super::cgroup::*;

pub const SUBTREE_CONTROL: &str = "cgroup.subtree_control";
pub const CONTROLLERS_FILE: &str = "cgroup.controllers";
pub const CGROUP2_KEY: &str = "cgroup2";

// https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html
pub const DEFAULT_PERIOD: u64 = 100000;

/* pub trait Controllerv2 : Controller {
    fn generateProperties(spec: &LinuxResources)
} */

pub struct Cpu2 {}

impl Controller for Cpu2 {
    fn Optional(&self) -> bool {
        return false;
    }

    fn Skip(&self, _linuxResource: &Option<LinuxResources>) -> Result<()> {
        panic!("cgroup controller is not optional");
    }

    fn Set(&self, spec: &Option<LinuxResources>, path: &str) -> Result<()> {
        match spec {
            None => return Ok(()),
            Some(ref spec) => match spec.cpu {
                None => return Ok(()),
                Some(ref cpu) => {
                    match cpu.shares {
                        None => (),
                        Some(ref shares) => {
                            let weight = convertCPUSharesToCgroupV2Value(*shares);
                            if weight != 0 {
                                SetValue(path, "cpu.weight", &format!("{}", weight))?;
                            }
                        }
                    }

                    if cpu.period.is_some() || cpu.quota.is_some() {
                        let mut v = "max".to_string();

                        match cpu.quota {
                            None => (),
                            Some(quota) => {
                                if quota != 0 {
                                    v = format!("{}", quota);
                                }
                            }
                        }

                        let mut period = 0u64;
                        match cpu.period {
                            None => (),
                            Some(p) => {
                                if p != 0 {
                                    period = p;
                                } else {
                                    period = DEFAULT_PERIOD;
                                }
                            }
                        }

                        v = v + " ";
                        v = v + &format!("{}", period);
                        SetValue(path, "cpu.max", &v)?;
                    }

                    return Ok(());
                }
            },
        }
    }
}

pub struct CpuSet2 {}

impl Controller for CpuSet2 {
    fn Optional(&self) -> bool {
        return false;
    }

    fn Skip(&self, _linuxResource: &Option<LinuxResources>) -> Result<()> {
        panic!("cgroup controller is not optional");
    }

    fn Set(&self, spec: &Option<LinuxResources>, path: &str) -> Result<()> {
        match spec {
            None => return Ok(()),
            Some(ref spec) => match spec.cpu {
                None => return Ok(()),
                Some(ref cpu) => {
                    if &cpu.cpus != "" {
                        SetValue(path, "cpuset.cpus", &cpu.cpus)?;
                    }

                    if &cpu.mems != "" {
                        SetValue(path, "cpuset.mems", &cpu.mems)?;
                    }
                }
            },
        }

        return Ok(());
    }
}

pub struct Memory2 {}

impl Controller for Memory2 {
    fn Optional(&self) -> bool {
        return false;
    }

    fn Skip(&self, _linuxResource: &Option<LinuxResources>) -> Result<()> {
        panic!("cgroup controller is not optional");
    }

    fn Set(&self, spec: &Option<LinuxResources>, path: &str) -> Result<()> {
        match spec {
            None => return Ok(()),
            Some(ref spec) => {
                match spec.memory {
                    None => return Ok(()),
                    Some(ref memory) => {
                        // in cgroup v2, we set memory and swap separately, but the spec specifies
                        // Swap field as memory+swap, so we need memory limit here to be set in
                        // order to get the correct swap value.
                        match memory.swap {
                            None => (),
                            Some(Swap) => match memory.limit {
                                None => {
                                    return Err(Error::Common(
                                        "cgroup: Memory.Swap is set without Memory.Limit"
                                            .to_string(),
                                    ));
                                }
                                Some(limit) => {
                                    let swap = ConvertMemorySwapToCgroupV2Value(Swap, limit)?;
                                    let mut swapStr = NumToStr(swap);
                                    if &swapStr == "" && swap == 0 && Swap > 0 {
                                        swapStr = "0".to_string();
                                    }

                                    if &swapStr != "" {
                                        SetValue(path, "memory.swap.max", &swapStr)?;
                                    }
                                }
                            },
                        }

                        match memory.limit {
                            None => (),
                            Some(limit) => {
                                let val = NumToStr(limit);
                                if &val != "" {
                                    SetValue(path, "memory.max", &val)?;
                                }
                            }
                        }

                        match memory.reservation {
                            None => (),
                            Some(reservation) => {
                                let val = NumToStr(reservation);
                                if &val != "" {
                                    SetValue(path, "memory.low", &val)?;
                                }
                            }
                        }
                    }
                }
            }
        }

        return Ok(());
    }
}

// Since the OCI spec is designed for cgroup v1, in some cases
// there is need to convert from the cgroup v1 configuration to cgroup v2
// the formula for cpuShares is y = (1 + ((x - 2) * 9999) / 262142)
// convert from [2-262144] to [1-10000]
// 262144 comes from Linux kernel definition "#define MAX_SHARES (1UL << 18)"
pub fn convertCPUSharesToCgroupV2Value(cpuShares: u64) -> u64 {
    if cpuShares == 0 {
        return 0;
    }

    return 1 + ((cpuShares - 2) * 9999) / 262142;
}

// convertMemorySwapToCgroupV2Value converts MemorySwap value from OCI spec
// for use by cgroup v2 drivers. A conversion is needed since
// Resources.MemorySwap is defined as memory+swap combined, while in cgroup v2
// swap is a separate value.
pub fn ConvertMemorySwapToCgroupV2Value(memorySwap: i64, memory: i64) -> Result<i64> {
    // for compatibility with cgroup1 controller, set swap to unlimited in
    // case the memory is set to unlimited, and swap is not explicitly set,
    // treating the request as "set both memory and swap to unlimited".
    if memory == -1 && memorySwap == 0 {
        return Ok(-1);
    }

    if memorySwap == -1 || memorySwap == 0 {
        // -1 is "max", 0 is "unset", so treat as is.
        return Ok(memorySwap);
    }

    // sanity checks
    if memory == 0 || memory == -1 {
        return Err(Error::Common(
            "unable to set swap limit without memory limit".to_string(),
        ));
    }
    if memory < 0 {
        return Err(Error::Common(format!("invalid memory value: {}", memory)));
    }
    if memorySwap < memory {
        return Err(Error::Common(
            "memory+swap limit should be >= memory limit".to_string(),
        ));
    }

    return Ok(memorySwap - memory);
}

// Since the OCI spec is designed for cgroup v1, in some cases
// there is need to convert from the cgroup v1 configuration to cgroup v2
// the formula for BlkIOWeight to IOWeight is y = (1 + (x - 10) * 9999 / 990)
// convert linearly from [10-1000] to [1-10000]
pub fn ConvertBlkIOToIOWeightValue(blkIoWeight: u16) -> u64 {
    if blkIoWeight == 0 {
        return 0;
    }
    return 1 + (blkIoWeight as u64 - 100) * 9999 / 990;
}

pub fn NumToStr(value: i64) -> String {
    if value == 0 {
        return "".to_string();
    } else if value == -1 {
        return "max".to_string();
    } else {
        return format!("{}", value);
    }
}

// bfqDeviceWeightSupported checks for per-device BFQ weight support (added
// in kernel v5.4, commit 795fe54c2a8) by reading from "io.bfq.weight".
pub fn bfqDeviceWeightSupported(bfq: &mut File) -> bool {
    match bfq.seek(SeekFrom::Start(0)) {
        Ok(_) => (),
        _ => return false,
    }

    let mut buffer = String::new();

    // read the whole file
    match bfq.read_to_string(&mut buffer) {
        Ok(_) => (),
        _ => return false,
    }

    // If only a single number (default weight) if read back, we have older
    // kernel.
    match buffer.parse::<u64>() {
        Ok(_) => return true,
        _ => return false,
    }
}

// parseKeyValue parses a space-separated "name value" kind of cgroup
// parameter and returns its key as a string, and its value as uint64
// (ParseUint is used to convert the value). For example,
// "io_service_bytes 1234" will be returned as "io_service_bytes", 1234.
pub fn ParseKeyValue(t: &str) -> Result<(String, u64)> {
    let parts: Vec<&str> = t.splitn(3, " ").collect();
    if parts.len() != 2 {
        return Err(Error::Common(format!(
            "line {} is not in key value format",
            t
        )));
    }

    let value = ParseUint(parts[1], 10, 64)?;
    return Ok((parts[0].to_string(), value));
}

// parseUint converts a string to an uint64 integer.
// Negative values are returned at zero as, due to kernel bugs,
// some of the memory cgroup stats can be negative.
pub fn ParseUint(s: &str, base: u32, bitSize: u32) -> Result<u64> {
    match u64::from_str_radix(s, base) {
        Ok(v) => return Ok(v),
        _ => {
            // 1. Handle negative values greater than MinInt64 (and)
            // 2. Handle negative values lesser than MinInt64
            match i64::from_str_radix(s, base) {
                Ok(v) => {
                    if v < 0 {
                        return Ok(0);
                    }

                    return Ok(v as u64);
                }
                _ => {
                    return Err(Error::Common(format!(
                        "ParseUint fail {}/{}/{}",
                        s, base, bitSize
                    )))
                }
            }
        }
    }
}
