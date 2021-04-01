// Copyright (c) 2021 Quark Container Authors
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
use alloc::string::String;
use spin::Mutex;

use super::super::super::super::qlib::common::*;
use super::super::super::super::qlib::linux_def::*;
use super::super::super::super::task::*;
use super::super::super::ramfs::symlink::*;
use super::super::super::dirent::*;
use super::super::super::mount::*;
use super::super::super::inode::*;
use super::super::super::super::threadmgr::thread::*;
use super::super::symlink_proc::*;

pub struct NamespaceNode {
    pub thread: Thread,
}

impl ReadLinkNode for NamespaceNode {
    fn ReadLink(&self, link: &Symlink, task: &Task, dir: &Inode) -> Result<String> {
        return link.ReadLink(task, dir)
    }

    fn GetLink(&self, link: &Symlink, task: &Task, dir: &Inode) -> Result<Dirent> {
        return link.GetLink(task, dir);
    }
}

pub fn NewExe(task: &Task, msrc: &Arc<Mutex<MountSource>>) -> Inode {
    let node = ExeNode {
        thread: task.Thread(),
    };

    return SymlinkNode::New(task, msrc, node)
}
