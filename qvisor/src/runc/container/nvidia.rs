// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
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

use std::fs;

use crate::qlib::common::*;
use crate::runc::oci::Spec;

const NVD_ENV_VAR : &str = "NVIDIA_VISIBLE_DEVICES";

pub fn FindAllGPUDevices() -> Result<Vec<u32>> {
    let paths = fs::read_dir("/dev/").unwrap();

    let mut list = Vec::new();
    for path in paths {
        let filename = path.unwrap().file_name().to_str().unwrap().to_string();
        if filename.starts_with("nvidia") {
            let id = filename.strip_prefix("nvidia").unwrap();
            match id.parse() {
                Err(_) => (),
                Ok(idx) => {
                    list.push(idx);
                }
            }
        }
    }
    
    return Ok(list);
}

// NvidiaDeviceList returns the list of devices that should be visible to the
// sandbox. In Docker mode, this is the set of devices specified in
// NVIDIA_VISIBLE_DEVICES. In non-Docker mode, this is all Nvidia devices, as
// we cannot know the set of usable GPUs until subcontainer creation.
pub fn NvidiaDeviceList(spec: &Spec) -> Result<String> {
    for env in &spec.process.env {
        let parts : Vec<&str> = env.split("=").collect();
        if parts.len() != 2 {
            continue;
        } 

        if parts[0] == NVD_ENV_VAR {
            let nvd = parts[1];

            if nvd == "none" {
                return Ok("".to_owned());
            }

            if nvd == "all" {
                return Ok("all".to_owned());
            }

            let gpus : Vec<&str> = nvd.split(",").collect();
            for gpu in gpus {
                // Validate gpuDev. We only support the following formats for now:
                // * GPU indices (e.g. 0,1,2)
                // * GPU UUIDs (e.g. GPU-fef8089b)
                //
                // We do not support MIG devices yet.
                if gpu.starts_with("GPU-") {
                    return Err(Error::Common(format!("We do not support Nvidia device {}", gpu)));
                }

                if gpu.parse::<u32>().is_err() {
                    return Err(Error::Common(format!("We do not support Nvidia device {}", gpu)));
                }
            }

            return Ok(nvd.to_owned())
        }        
    }

    return Ok("".to_owned());
}

