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
use spin::Mutex;
use alloc::string::String;
use alloc::string::ToString;

use super::super::super::qlib::common::*;
use super::super::super::task::*;
use super::super::filesystems::*;
use super::super::inode::*;
use super::super::mount::*;
use super::sys::*;

pub struct SysFileSystem {}

impl Filesystem for SysFileSystem {
    fn Name(&self) -> String {
        return "sysfs".to_string();
    }

    fn Flags(&self) -> FilesystemFlags {
        return 0;
    }

    fn Mount(&mut self, task: &Task, _device: &str, flags: &MountSourceFlags, _data: &str) -> Result<Inode> {
        info!("sysfs file system mount ...");

        let msrc = MountSource::NewCachingMountSource(self, flags);
        let inode = NewSys(task, &Arc::new(Mutex::new(msrc)));
        return Ok(inode)
    }

    fn AllowUserMount(&self) -> bool {
        return true;
    }

    fn AllowUserList(&self) -> bool {
        return true;
    }
}