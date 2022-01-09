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
use alloc::string::ToString;
use crate::qlib::mutex::*;
use alloc::collections::btree_map::BTreeMap;

use super::super::super::super::super::common::*;
use super::super::super::super::super::linux_def::*;
use super::super::super::super::super::auth::*;
use super::super::super::super::task::*;
use super::super::super::attr::*;
use super::super::super::file::*;
use super::super::super::flags::*;
use super::super::super::dirent::*;
use super::super::super::mount::*;
use super::super::super::inode::*;
use super::super::super::ramfs::dir::*;
use super::super::super::super::threadmgr::pid_namespace::*;
use super::super::super::super::threadmgr::thread::*;
use super::super::dir_proc::*;
use super::super::proc::*;
use super::super::inode::*;
use super::auxvec::*;
use super::exe::*;
use super::exec_args::*;
use super::comm::*;
use super::fds::*;
use super::uid_pid_map::*;
use super::io::*;
use super::maps::*;
use super::mounts::*;
use super::stat::*;
use super::statm::*;
use super::status::*;

// taskDir represents a task-level directory.
pub struct TaskDirNode {
    pub pidns: Option<PIDNamespace>,
    pub thread: Thread,
}

impl DirDataNode for TaskDirNode {
    fn Lookup(&self, d: &Dir, task: &Task, dir: &Inode, name: &str) -> Result<Dirent> {
        return d.Lookup(task, dir, name);
    }

    fn GetFile(&self, d: &Dir, task: &Task, dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        return d.GetFile(task, dir, dirent, flags)
    }
}

impl ProcNode {
    pub fn NewTaskDir(&self, task: &Task, thread: &Thread, msrc: &Arc<QMutex<MountSource>>, showSubtasks: bool) -> Inode {
        let mut contents = BTreeMap::new();
        contents.insert("auxv".to_string(), NewAUXVec(task, thread, msrc));
        contents.insert("cmdline".to_string(), NewExecArg(task, thread, msrc, ExecArgType::CmdlineExecArg));
        contents.insert("comm".to_string(), NewComm(task, thread, msrc));
        contents.insert("environ".to_string(), NewExecArg(task, thread, msrc, ExecArgType::EnvironExecArg));
        contents.insert("exe".to_string(), NewExe(task, thread, msrc));
        contents.insert("fd".to_string(), NewFdDir(task, thread, msrc));
        contents.insert("fdinfo".to_string(), NewFdInfoDir(task, thread, msrc));
        contents.insert("gid_map".to_string(), NewIdMap(task, thread, msrc, true));
        contents.insert("io".to_string(), NewIO(task, thread, msrc));
        contents.insert("maps".to_string(), NewMaps(task, thread, msrc));
        contents.insert("mountinfo".to_string(), NewMountInfoFile(task, thread, msrc));
        contents.insert("mounts".to_string(), NewMountsFile(task, thread, msrc));
        contents.insert("stat".to_string(), NewStat(task, thread, showSubtasks, self.lock().pidns.clone(), msrc));
        contents.insert("statm".to_string(), NewStatm(task, thread, msrc));
        contents.insert("status".to_string(), NewStatus(task, thread, msrc));
        contents.insert("uid_map".to_string(), NewIdMap(task, thread, msrc, false));

        if showSubtasks {
            contents.insert("task".to_string(), self.NewSubTasksDir(task, thread, msrc));
        }

        let taskDir = DirNode {
            dir: Dir::New(task, contents, &ROOT_OWNER, &FilePermissions::FromMode(FileMode(0o0555))),
            data: TaskDirNode {
                pidns: None,
                thread: thread.clone(),
            }
        };

        return NewProcInode(&Arc::new(taskDir), msrc, InodeType::SpecialDirectory, Some(thread.clone()))
    }
}

