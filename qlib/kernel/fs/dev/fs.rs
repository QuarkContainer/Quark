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
use alloc::string::String;
use alloc::string::ToString;

use super::super::super::super::common::*;
use super::super::super::task::*;
use super::super::filesystems::*;
use super::super::inode::*;
use super::super::mount::*;
use super::dev::*;

pub struct DevTmpfs {}

impl Filesystem for DevTmpfs {
    fn Name(&self) -> String {
        return "devtmpfs".to_string();
    }

    fn Flags(&self) -> FilesystemFlags {
        return 0;
    }

    fn Mount(&mut self, task: &Task, _device: &str, flags: &MountSourceFlags, _data: &str) -> Result<Inode> {
        //let options = FileSystem::GenericMountSourceOptions(data);

        let mountSource = MountSource::NewCachingMountSource(self, flags);
        let inode = NewDev(task, &Arc::new(QMutex::new(mountSource)));
        return Ok(inode)
    }

    fn AllowUserMount(&self) -> bool {
        return true;
    }

    fn AllowUserList(&self) -> bool {
        return true;
    }
}