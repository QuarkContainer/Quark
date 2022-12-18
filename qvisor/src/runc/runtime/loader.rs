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

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use simplelog::*;

use super::super::oci::*;

#[derive(Default, Debug)]
pub struct Args {
    // Id is the sandbox ID.
    pub ID: String,

    // Spec is the sandbox specification.
    pub Spec: Spec,

    // Conf is the system configuration.
    pub Conf: Box<Config>,

    // StdioFDs is the stdio for the application.
    pub StdioFDs: Vec<i32>,

    // UserLogFD is the file descriptor to write user logs to.
    pub UserLogFD: i32,

    pub KvmFd: i32,

    pub AutoStart: bool,

    pub BundleDir: String,

    pub Pivot: bool,

    pub ControlSock: i32,

    pub Rootfs: String,

    pub RDMASvcCliSock: i32,
}

impl Args {
    pub fn GetCpuCount(&self) -> usize {
        match &self.Spec.linux {
            None => return 0,
            Some(linux) => {
                match &linux.resources {
                    None => return 0,
                    Some(resources) => {
                        match &resources.cpu {
                            None => return 0,
                            Some(cpu) => {
                                let quota = match cpu.quota {
                                    None => return 0,
                                    Some(q) => q,
                                };

                                let period = match cpu.period {
                                    None => return 0,
                                    Some(p) => p,
                                };

                                let count = (quota as u64 + period - 1) / period;
                                return count as usize;
                            }
                        }
                    }
                }
            }
        };
    } 
}