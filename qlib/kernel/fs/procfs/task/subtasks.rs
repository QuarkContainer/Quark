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
use alloc::collections::btree_map::BTreeMap;
use alloc::string::ToString;
use alloc::sync::Arc;

use super::super::super::super::super::auth::*;
use super::super::super::super::super::common::*;
use super::super::super::super::super::device::*;
use super::super::super::super::super::linux_def::*;
use super::super::super::super::fs::dentry::*;
use super::super::super::super::fs::fsutil::file::dynamic_dir_file_operations::*;
use super::super::super::super::fs::mount::*;
use super::super::super::super::task::*;
use super::super::super::super::threadmgr::pid_namespace::*;
use super::super::super::super::threadmgr::thread::*;
use super::super::super::attr::*;
use super::super::super::dirent::*;
use super::super::super::file::*;
use super::super::super::flags::*;
use super::super::super::inode::*;
use super::super::super::ramfs::dir::*;
use super::super::dir_proc::*;
use super::super::inode::*;
use super::super::proc::*;

impl ProcNode {
    pub fn NewSubTasksDir(
        &self,
        task: &Task,
        thread: &Thread,
        msrc: &Arc<QMutex<MountSource>>,
    ) -> Inode {
        let contents = BTreeMap::new();
        let subTasksNode = SubTasksNode {
            thread: thread.clone(),
            procNode: self.clone(),
        };

        let subTaskDir = DirNode {
            dir: Dir::New(
                task,
                contents,
                &ROOT_OWNER,
                &FilePermissions::FromMode(FileMode(0o0555)),
            ),
            data: subTasksNode,
        };

        return NewProcInode(
            &Arc::new(subTaskDir),
            msrc,
            InodeType::SpecialDirectory,
            Some(thread.clone()),
        );
    }
}

// subtasks represents a /proc/TID/task directory.\
pub struct SubTasksNode {
    pub thread: Thread,
    pub procNode: ProcNode,
}

impl DirDataNode for SubTasksNode {
    fn Lookup(&self, _d: &Dir, task: &Task, dir: &Inode, name: &str) -> Result<Dirent> {
        let tid = match name.parse::<i32>() {
            Err(_) => return Err(Error::SysError(SysErr::ENOENT)),
            Ok(id) => id,
        };

        let pidns = self.procNode.lock().pidns.clone();
        let thread = match pidns.TaskWithID(tid) {
            None => return Err(Error::SysError(SysErr::ENOENT)),
            Some(t) => t,
        };

        if thread.ThreadGroup() != self.thread.ThreadGroup() {
            return Err(Error::SysError(SysErr::ENOENT));
        }

        let ms = dir.lock().MountSource.clone();
        let td = self.procNode.NewTaskDir(task, &self.thread, &ms, false);
        return Ok(Dirent::New(&td, name));
    }

    fn GetFile(
        &self,
        _d: &Dir,
        _task: &Task,
        _dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        let pidns = self.procNode.lock().pidns.clone();
        return Ok(File::New(
            dirent,
            &flags,
            NewSubTasksFile(&self.thread, &pidns).into(),
        ));
    }
}

pub fn NewSubTasksFile(
    thread: &Thread,
    pidns: &PIDNamespace,
) -> DynamicDirFileOperations {
    let subTaskFile = SubTasksFileNode {
        thread: thread.clone(),
        pidns: pidns.clone(),
    };

    return DynamicDirFileOperations { node: subTaskFile.into() };
}

pub struct SubTasksFileNode {
    pub thread: Thread,
    pub pidns: PIDNamespace,
}

impl DynamicDirFileNodeTrait for SubTasksFileNode {
    fn ReadDir(
        &self,
        task: &Task,
        _f: &File,
        offset: i64,
        serializer: &mut DentrySerializer,
    ) -> Result<i64> {
        let mut dirCtx = DirCtx {
            Serializer: serializer,
            DirCursor: "".to_string(),
        };

        if offset == 0 {
            let root = task.mountNS.root.clone();
            let (dot, dotdot) = root.GetDotAttrs(&root);

            dirCtx.DirEmit(task, &".".to_string(), &dot)?;
            dirCtx.DirEmit(task, &"..".to_string(), &dotdot)?;
        }

        let tasks = self.thread.ThreadGroup().MemberIDs(&self.pidns);
        let idx = match tasks.binary_search(&(offset as i32)) {
            Ok(i) => i,
            Err(i) => i,
        };

        if idx == tasks.len() {
            return Ok(offset);
        }

        let taskInts = &tasks[idx..];
        let mut tid = 0;
        for id in taskInts {
            tid = *id;
            let name = format!("{}", tid);
            let attr = DentAttr::GenericDentAttr(InodeType::SpecialDirectory, &PROC_DEVICE);
            dirCtx.DirEmit(task, &name, &attr)?;
        }

        return Ok(tid as i64 + 1);
    }
}
