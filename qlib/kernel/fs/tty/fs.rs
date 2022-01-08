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
use ::qlib::mutex::*;
use alloc::string::String;
use alloc::string::ToString;
use core::any::Any;

use super::super::super::super::common::*;
use super::super::super::task::*;
use super::super::super::super::linux_def::*;
use super::super::filesystems::*;
use super::super::inode::*;
use super::super::mount::*;
use super::super::dirent::*;
use super::dir::*;

pub struct PtsTmpfs {}

impl Filesystem for PtsTmpfs {
    fn Name(&self) -> String {
        return "devpts".to_string();
    }

    fn Flags(&self) -> FilesystemFlags {
        return 0;
    }

    fn Mount(&mut self, task: &Task, _device: &str, flags: &MountSourceFlags, data: &str) -> Result<Inode> {
        if data != "" {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        let mops = Arc::new(QMutex::new(PtsSuperOperations {}));
        let msrc = MountSource::NewPtsMountSource(&mops, self, flags);
        let dir = NewDir(task, &Arc::new(QMutex::new(msrc)));
        return Ok(dir)
    }

    fn AllowUserMount(&self) -> bool {
        return false;
    }

    fn AllowUserList(&self) -> bool {
        return true;
    }
}

pub struct PtsSuperOperations {}

impl MountSourceOperations for PtsSuperOperations {
    fn as_any(&self) -> &Any {
        return self
    }

    fn Destroy(&mut self) {}
    fn ResetInodeMappings(&mut self) {}
    fn SaveInodeMapping(&mut self, _inode: &Inode, _path: &str) {}
}

impl DirentOperations for PtsSuperOperations {
    fn Revalidate(&self, _name: &str, _parent: &Inode, _child: &Inode) -> bool {
        return true
    }

    fn Keep(&self, _dirent: &Dirent) -> bool {
        return false;
    }

    fn CacheReadDir(&self) -> bool {
        return false;
    }
}