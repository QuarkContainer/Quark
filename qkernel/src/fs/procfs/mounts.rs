// Copyright (c) 2021 QuarkSoft LLC
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
use alloc::string::ToString;
use spin::Mutex;

use super::super::super::qlib::common::*;
use super::super::super::task::*;
use super::super::ramfs::symlink::*;
use super::super::dirent::*;
use super::super::mount::*;
use super::super::inode::*;
use super::symlink_proc::*;

pub struct MountsNode {}

impl ReadLinkNode for MountsNode {
    fn ReadLink(&self, _link: &Symlink, _task: &Task, _dir: &Inode) -> Result<String> {
        return Ok("self/mounts".to_string())
    }

    fn GetLink(&self, link: &Symlink, task: &Task, dir: &Inode) -> Result<Dirent> {
        return link.GetLink(task, dir);
    }
}

pub fn NewMounts(task: &Task, msrc: &Arc<Mutex<MountSource>>) -> Inode {
    let node = MountsNode {};

    return SymlinkNode::New(task, msrc, node,  None)
}
