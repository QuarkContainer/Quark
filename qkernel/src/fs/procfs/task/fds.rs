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
use alloc::string::String;
use alloc::string::ToString;
use spin::Mutex;
use alloc::collections::btree_map::BTreeMap;

use super::super::super::super::qlib::auth::*;
use super::super::super::super::qlib::common::*;
use super::super::super::super::qlib::linux_def::*;
use super::super::super::super::qlib::device::*;
use super::super::super::super::kernel::fd_table::*;
use super::super::super::super::task::*;
use super::super::super::ramfs::dir::*;
use super::super::super::ramfs::symlink::*;
use super::super::super::fsutil::file::dynamic_dir_file_operations::*;
use super::super::super::attr::*;
use super::super::super::file::*;
use super::super::super::flags::*;
use super::super::super::dirent::*;
use super::super::super::dentry::*;
use super::super::super::mount::*;
use super::super::super::inode::*;
use super::super::super::super::threadmgr::thread::*;
use super::super::symlink_proc::*;
use super::super::inode::*;
use super::super::dir_proc::*;

pub fn NewFd(task: &Task, thread: &Thread, msrc: &Arc<Mutex<MountSource>>, f: &File) -> Inode {
    let node = FdNode {
        file: f.Downgrade(),
    };

    return SymlinkNode::New(task, msrc, node, Some(thread.clone()))
}

pub struct FdNode {
    file: FileWeak,
}

impl ReadLinkNode for FdNode {
    fn ReadLink(&self, _link: &Symlink, task: &Task, _dir: &Inode) -> Result<String> {
        let kernel = task.Thread().lock().k.clone();
        let root = kernel.RootDir();
        let file = match self.file.Upgrade() {
            None => return Err(Error::SysError(SysErr::ENOENT)),
            Some(f) => f,
        };
        let dirent = file.Dirent.clone();
        let (name, _) = dirent.FullName(&root);
        return Ok(name)
    }

    fn GetLink(&self, _link: &Symlink, _task: &Task, _dir: &Inode) -> Result<Dirent> {
        let file = match self.file.Upgrade() {
            None => return Err(Error::SysError(SysErr::ENOENT)),
            Some(f) => f,
        };
        let dirent = file.Dirent.clone();
        return Ok(dirent)
    }

    fn GetFile(&self, _link: &Symlink, _task: &Task, _dir: &Inode, _dirent: &Dirent, _flags: FileFlags) -> Result<File> {
        let file = match self.file.Upgrade() {
            None => return Err(Error::SysError(SysErr::ENOENT)),
            Some(f) => f,
        };
        return Ok(file)
    }
}

fn WalkDescriptors(task: &Task, p: &str, toInode: &mut FnMut(&File, &FDFlags) -> Inode) -> Result<Inode> {
    let n : i32 = match p.parse() {
        Err(_) => return Err(Error::SysError(SysErr::ENOENT)),
        Ok(n) => n,
    };

    let (file, fdFlags) = match task.GetDescriptor(n) {
        Err(_) => return Err(Error::SysError(SysErr::ENOENT)),
        Ok(f) => f,
    };

    return Ok(toInode(&file, &fdFlags))
}

fn WalkDescriptors2(task: &Task, p: &str, msrc: &Arc<Mutex<MountSource>>) -> Result<Inode> {
    let n : i32 = match p.parse() {
        Err(_) => return Err(Error::SysError(SysErr::ENOENT)),
        Ok(n) => n,
    };

    let (file, fdFlags) = match task.GetDescriptor(n) {
        Err(_) => return Err(Error::SysError(SysErr::ENOENT)),
        Ok(f) => f,
    };

    let flags = file.flags.lock().0.ToLinux() | fdFlags.ToLinuxFileFlags();
    let content = format!("flags:\t{:o}\n", flags);
    return Ok(NewStaticProcInode(task, msrc, &Arc::new(content.as_bytes().to_vec())))
}

fn ReadDescriptors(task: &Task, c: &mut DirCtx, offset: i64, typ: InodeType) -> Result<i64> {
    let fds = task.fdTbl.lock().GetFDs();

    let mut fdInts = &fds[..];
    let idx = match fds.binary_search(&(offset as i32)) {
        Err(idx) => idx,
        Ok(idx) => idx,
    };

    if idx == fdInts.len() {
        return Ok(offset)
    }

    fdInts = &fdInts[idx..];

    let mut ret = offset as i32;
    for i in 0..fdInts.len() {
        let fd = fdInts[i];
        let name = format!("{}", fd);
        match c.DirEmit(task, &name, &DentAttr::GenericDentAttr(typ, &PROC_DEVICE)) {
            Err(e) => {
                if i > 0 {
                    return Ok(fd as i64)
                }
                return Err(e)
            }
            Ok(()) => (),
        }
        ret = fd;
    }

    return Ok((ret + 1) as i64)
}

pub struct FdDirFile {
    pub IsInfoFile: bool,
    pub thread: Thread,
}

impl DynamicDirFileNode for FdDirFile {
    fn ReadDir(&self, task: &Task, _f: &File, offset: i64, serializer: &mut DentrySerializer) -> Result<i64> {
        let mut dirCtx = DirCtx {
            Serializer: serializer,
            DirCursor: "".to_string(),
        };

        let typ = if self.IsInfoFile {
            InodeType::Symlink
        } else {
            InodeType::RegularFile
        };

        return ReadDescriptors(task, &mut dirCtx, offset, typ);
    }
}

pub fn NewFdDirFile(IsInfoFile: bool, thread: &Thread) -> DynamicDirFileOperations<FdDirFile> {
    let fdDirFile = FdDirFile {
        IsInfoFile: IsInfoFile,
        thread: thread.clone(),
    };

    return DynamicDirFileOperations {
        node: fdDirFile
    }
}

pub struct FdDirNode {
    pub thread: Thread,
}

impl DirDataNode for FdDirNode {
    // Check implements InodeOperations.Check.
    //
    // This is to match Linux, which uses a special permission handler to guarantee
    // that a process can still access /proc/self/fd after it has executed
    // setuid. See fs/proc/fd.c:proc_fd_permission.
    fn Check(&self, _d: &Dir, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        if ContextCanAccessFile(task, inode, reqPerms)? {
            return Ok(true)
        }

        let thread = match &task.thread {
            None => return Ok(false),
            Some(t) => t.clone(),
        };

        let tg = self.thread.ThreadGroup();
        if tg == thread.ThreadGroup() {
            return Ok(true);
        }

        return Ok(false);
    }

    fn Lookup(&self, _d: &Dir, task: &Task, dir: &Inode, name: &str) -> Result<Dirent> {
        let msrc = dir.lock().MountSource.clone();
        let inode = WalkDescriptors(task, name, &mut |file: &File, _fdFlags: &FDFlags| {
            return NewFd(task, &self.thread, &msrc, file)
        })?;

        return Ok(Dirent::New(&inode, name))
    }

    fn GetFile(&self, _d: &Dir, _task: &Task, _dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        let fops = NewFdDirFile(true, &self.thread);

        return Ok(File::New(dirent, &flags, fops))
    }
}

pub struct FdInfoDirNode {
    pub thread: Thread,
}

impl DirDataNode for FdInfoDirNode {
    fn Lookup(&self, _d: &Dir, task: &Task, dir: &Inode, name: &str) -> Result<Dirent> {
        let msrc = dir.lock().MountSource.clone();
        let inode = WalkDescriptors(task, name, &mut |file: &File, fdFlags: &FDFlags| {
            let flags = file.flags.lock().0.ToLinux() | fdFlags.ToLinuxFileFlags();
            let content = format!("flags:\t0{:o}\n", flags);
            return NewStaticProcInode(task, &msrc, &Arc::new(content.as_bytes().to_vec()))

        })?;

        return Ok(Dirent::New(&inode, name))
    }

    fn GetFile(&self, _d: &Dir, _task: &Task, _dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        let fops = NewFdDirFile(true, &self.thread);

        return Ok(File::New(dirent, &flags, fops))
    }
}

pub fn NewFdDir(task: &Task, thread: &Thread, msrc: &Arc<Mutex<MountSource>>) -> Inode {
    let contents = BTreeMap::new();
    let f = DirNode {
        dir: Dir::New(task, contents, &ROOT_OWNER, &FilePermissions::FromMode(FileMode(0o0555))),
        data: FdDirNode {
            thread: thread.clone(),
        }
    };

    return NewProcInode(&Arc::new(f), msrc, InodeType::SpecialDirectory, Some(thread.clone()))
}

pub fn NewFdInfoDir(task: &Task, thread: &Thread, msrc: &Arc<Mutex<MountSource>>) -> Inode {
    let contents = BTreeMap::new();
    let f = DirNode {
        dir: Dir::New(task, contents, &ROOT_OWNER, &FilePermissions::FromMode(FileMode(0o0555))),
        data: FdInfoDirNode {
            thread: thread.clone(),
        }
    };

    return NewProcInode(&Arc::new(f), msrc, InodeType::SpecialDirectory, Some(thread.clone()))
}