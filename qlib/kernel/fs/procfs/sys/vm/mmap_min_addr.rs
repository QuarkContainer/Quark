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

use alloc::sync::Arc;
use crate::qlib::mutex::*;

use super::super::super::super::super::kernel::kernel::*;
use super::super::super::super::super::task::*;
use super::super::super::super::mount::*;
use super::super::super::super::inode::*;
use super::super::super::inode::*;

pub fn NewMinAddrData(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let kernel = GetKernel();
    let minaddr = format!("{}\n", kernel.platform.MinUserAddress());

    return NewStaticProcInode(task,
                              msrc,
                              &Arc::new(minaddr.as_bytes().to_vec()))
}