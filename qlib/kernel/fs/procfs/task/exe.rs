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
use alloc::string::String;
use alloc::sync::Arc;

use super::super::super::super::super::common::*;
use super::super::super::super::super::linux_def::*;
use super::super::super::super::task::*;
use super::super::super::super::threadmgr::thread::*;
use super::super::super::dirent::*;
use super::super::super::inode::*;
use super::super::super::mount::*;
use super::super::super::ramfs::symlink::*;
use super::super::symlink_proc::*;

pub struct ExeNode {
    pub thread: Thread,
}

impl ExeNode {
    pub fn Executable(&self) -> Result<Dirent> {
        let mm = self.thread.lock().memoryMgr.clone();
        let ret = mm.metadata.lock().executable.clone();
        match ret {
            None => return Err(Error::SysError(SysErr::ENOENT)),
            Some(d) => Ok(d),
        }
    }
}

impl ReadLinkNodeTrait for ExeNode {
    fn ReadLink(&self, _link: &Symlink, task: &Task, _dir: &Inode) -> Result<String> {
        let exe = self.Executable()?;

        let root = task.Root();
        let (name, _) = exe.FullName(&root);
        return Ok(name);
    }

    fn GetLink(&self, link: &Symlink, task: &Task, dir: &Inode) -> Result<Dirent> {
        return link.GetLink(task, dir);
    }
}

pub fn NewExe(task: &Task, thread: &Thread, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let node = ExeNode {
        thread: thread.clone(),
    };

    return SymlinkNode::New(task, msrc, node.into(), Some(thread.clone()));
}
