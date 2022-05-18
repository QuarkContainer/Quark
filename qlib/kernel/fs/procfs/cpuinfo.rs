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

use crate::qlib::mutex::*;
use alloc::string::ToString;
use alloc::sync::Arc;

use super::super::super::kernel::kernel::*;
use super::super::super::task::*;
use super::super::inode::*;
use super::super::mount::*;
use super::inode::*;

pub fn NewCPUInfo(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let kernel = GetKernel();
    let features = kernel.featureSet.clone();

    let mut cpuInfo = "".to_string();
    for i in 0..kernel.applicationCores {
        cpuInfo += &features.lock().CPUInfo(i as u32);
    }

    return NewStaticProcInode(task, msrc, &Arc::new(cpuInfo.as_bytes().to_vec()));
}
