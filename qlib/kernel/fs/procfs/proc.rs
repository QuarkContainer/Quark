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
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use core::any::Any;
use core::ops::Deref;

use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::device::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::task_mgr::*;
use super::super::super::fs::dentry::*;
use super::super::super::fs::fsutil::file::*;
use super::super::super::kernel::kernel::*;
use super::super::super::kernel::waiter::*;
use super::super::super::task::*;
use super::super::super::threadmgr::pid_namespace::*;
use super::super::attr::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::host::hostinodeop::*;
use super::super::inode::*;
use super::super::mount::*;
use super::super::ramfs::dir::*;
use super::super::ramfs::symlink::*;
use super::dir_proc::*;
use super::inode::*;
use super::symlink_proc::*;
use super::sys::sys::*;

use super::cpuinfo::*;
use super::filesystems::*;
use super::loadavg::*;
use super::meminfo::*;
use super::mounts::*;
use super::net::*;
use super::stat::*;
use super::uptime::*;

pub struct ProcNodeInternal {
    pub kernel: Kernel,
    pub pidns: PIDNamespace,
    pub cgroupControllers: Arc<QMutex<BTreeMap<String, String>>>,
}

#[derive(Clone)]
pub struct ProcNode(Arc<QMutex<ProcNodeInternal>>);

impl Deref for ProcNode {
    type Target = Arc<QMutex<ProcNodeInternal>>;

    fn deref(&self) -> &Arc<QMutex<ProcNodeInternal>> {
        &self.0
    }
}

impl DirDataNodeTrait for ProcNode {
    fn Lookup(&self, d: &Dir, task: &Task, dir: &Inode, name: &str) -> Result<Dirent> {
        let err = match d.Lookup(task, dir, name) {
            Ok(dirent) => return Ok(dirent),
            Err(e) => e,
        };

        let tid = match name.parse::<i32>() {
            Ok(tid) => tid,
            _ => return Err(err),
        };

        let otherThread = match self.lock().pidns.TaskWithID(tid) {
            None => return Err(err),
            Some(t) => t,
        };

        let otherTask = TaskId::New(otherThread.lock().taskId).GetTask();

        let ms = dir.lock().MountSource.clone();
        let td = self.NewTaskDir(&otherTask, &otherThread, &ms, true);

        return Ok(Dirent::New(&td, name));
    }

    fn GetFile(
        &self,
        d: &Dir,
        _task: &Task,
        _dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        let p = DirNode {
            dir: d.clone(),
            data: self.clone().into(),
        };

        return Ok(File::New(dirent, &flags, RootProcFile { iops: p }.into()));
    }
}

pub fn NewProc(
    task: &Task,
    msrc: &Arc<QMutex<MountSource>>,
    cgroupControllers: BTreeMap<String, String>,
) -> Inode {
    let mut contents = BTreeMap::new();

    let kernel = GetKernel();
    let pidns = kernel.RootPIDNamespace();

    contents.insert("cpuinfo".to_string(), NewCPUInfo(task, msrc));
    contents.insert("filesystems".to_string(), NewFileSystem(task, msrc));
    contents.insert("loadavg".to_string(), NewLoadAvg(task, msrc));
    contents.insert("mounts".to_string(), NewMounts(task, msrc));
    contents.insert("net".to_string(), NewNetDir(task, msrc));
    contents.insert("self".to_string(), NewProcessSelf(task, &pidns, msrc));
    contents.insert("stat".to_string(), NewStatData(task, msrc));
    contents.insert("thread-self".to_string(), NewThreadSelf(task, &pidns, msrc));
    contents.insert("uptime".to_string(), NewUptime(task, msrc));
    contents.insert("meminfo".to_string(), NewMeminfo(task, msrc));

    contents.insert("sys".to_string(), NewSys(task, msrc));

    let iops = Dir::New(
        task,
        contents,
        &ROOT_OWNER,
        &FilePermissions::FromMode(FileMode(0o0555)),
    );
    let kernel = GetKernel();
    let pidns = kernel.RootPIDNamespace();

    let proc = ProcNodeInternal {
        kernel: kernel,
        pidns: pidns,
        cgroupControllers: Arc::new(QMutex::new(cgroupControllers)),
    };

    let p = DirNode {
        dir: iops,
        data: ProcNode(Arc::new(QMutex::new(proc))).into(),
    };

    return NewProcInode(p.into(), msrc, InodeType::SpecialDirectory, None);
}

#[derive(Clone)]
pub struct ProcessSelfNode {
    pub pidns: PIDNamespace,
}

impl ReadLinkNodeTrait for ProcessSelfNode {
    fn ReadLink(&self, _link: &Symlink, task: &Task, _dir: &Inode) -> Result<String> {
        let thread = task.Thread();
        let tg = thread.ThreadGroup();
        let tgid = self.pidns.IDOfThreadGroup(&tg);

        let str = format!("{}", tgid);

        return Ok(str);
    }

    fn GetLink(&self, link: &Symlink, task: &Task, dir: &Inode) -> Result<Dirent> {
        return link.GetLink(task, dir);
    }
}

pub fn NewProcessSelf(task: &Task, pidns: &PIDNamespace, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let node = ProcessSelfNode {
        pidns: pidns.clone(),
    };

    return SymlinkNode::New(task, msrc, node.into(), None);
}

#[derive(Clone)]
pub struct ThreadSelfNode {
    pub pidns: PIDNamespace,
}

impl ReadLinkNodeTrait for ThreadSelfNode {
    fn ReadLink(&self, _link: &Symlink, task: &Task, _dir: &Inode) -> Result<String> {
        let thread = task.Thread();
        let tg = thread.ThreadGroup();
        let tgid = self.pidns.IDOfThreadGroup(&tg);
        let tid = self.pidns.IDOfTask(&thread);

        let str = format!("{}/task/{}", tgid, tid);

        return Ok(str);
    }

    fn GetLink(&self, link: &Symlink, task: &Task, dir: &Inode) -> Result<Dirent> {
        return link.GetLink(task, dir);
    }
}

pub fn NewThreadSelf(task: &Task, pidns: &PIDNamespace, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let node = ThreadSelfNode {
        pidns: pidns.clone(),
    };

    return SymlinkNode::New(task, msrc, node.into(), None);
}

#[derive(Clone)]
pub struct RootProcFile {
    pub iops: DirNode,
}

impl Waitable for RootProcFile {}

impl SpliceOperations for RootProcFile {}

impl FileOperations for RootProcFile {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::RootProcFile;
    }

    fn Seekable(&self) -> bool {
        return true;
    }

    fn Seek(&self, task: &Task, f: &File, whence: i32, current: i64, offset: i64) -> Result<i64> {
        return SeekWithDirCursor(task, f, whence, current, offset, None);
    }

    fn ReadAt(
        &self,
        _task: &Task,
        _f: &File,
        _dsts: &mut [IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOSYS));
    }

    fn WriteAt(
        &self,
        _task: &Task,
        _f: &File,
        _srcs: &[IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOSYS));
    }

    fn Append(&self, _task: &Task, _f: &File, _srcs: &[IoVec]) -> Result<(i64, i64)> {
        return Err(Error::SysError(SysErr::ENOSYS));
    }

    fn Fsync(
        &self,
        _task: &Task,
        _f: &File,
        _start: i64,
        _end: i64,
        _syncType: SyncType,
    ) -> Result<()> {
        return Ok(());
    }

    fn Flush(&self, _task: &Task, _f: &File) -> Result<()> {
        return Ok(());
    }

    fn Ioctl(&self, _task: &Task, _f: &File, _fd: i32, _request: u64, _val: u64) -> Result<u64> {
        return Err(Error::SysError(SysErr::ENOTTY));
    }

    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        let inode = f.Dirent.Inode().clone();
        return inode.UnstableAttr(task);
    }

    fn Mappable(&self) -> Result<MMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
    }

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

        // Get normal directory contents from ramfs dir.
        let mut map = self.iops.dir.Children();

        let root = task.Root();

        let (dot, dotdot) = root.GetDotAttrs(&root);
        map.insert(".".to_string(), dot);
        map.insert("..".to_string(), dotdot);

        let procNode = self.iops.data.ProcNode().unwrap();

        let pidns = procNode.lock().pidns.clone();
        for tg in &pidns.ThreadGroups() {
            if tg.Leader().is_some() {
                let name = format!("{}", tg.ID());
                map.insert(
                    name,
                    DentAttr::GenericDentAttr(InodeType::SpecialDirectory, &PROC_DEVICE),
                );
            }
        }

        if offset > map.len() as i64 {
            return Ok(offset);
        }

        let mut cnt = 0;
        for (name, entry) in &map {
            if cnt >= offset {
                dirCtx.DirEmit(task, name, entry)?
            }

            cnt += 1;
        }

        return Ok(map.len() as i64);
    }

    fn IterateDir(
        &self,
        _task: &Task,
        _d: &Dirent,
        _dirCtx: &mut DirCtx,
        _offset: i32,
    ) -> (i32, Result<i64>) {
        return (0, Err(Error::SysError(SysErr::ENOTDIR)));
    }
}

impl SockOperations for RootProcFile {}
