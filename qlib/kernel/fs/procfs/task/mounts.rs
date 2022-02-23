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
use alloc::vec::Vec;
use alloc::string::ToString;

use super::super::super::super::super::common::*;
use super::super::super::super::super::linux_def::*;
use super::super::super::super::super::auth::*;
use super::super::super::fsutil::file::readonly_file::*;
use super::super::super::fsutil::inode::simple_file_inode::*;
use super::super::super::super::task::*;
use super::super::super::attr::*;
use super::super::super::file::*;
use super::super::super::flags::*;
use super::super::super::dirent::*;
use super::super::super::mount::*;
use super::super::super::inode::*;
use super::super::super::super::threadmgr::thread::*;
use super::super::inode::*;

pub fn ForEachMount(thread: &Thread, mountns: MountNs, f: &mut FnMut(&str, &Arc<QMutex<Mount>>)) {
    let fsctx = thread.lock().fsc.clone();

    let rootDir = fsctx.RootDirectory();

    let mnt = match mountns.FindMount(&rootDir) {
        None => return,
        Some(mnt) => mnt,
    };

    let ms = mountns.AllMountsUnder(&mnt);

    for m in ms {
        let mroot = m.lock().Root();
        let (mountPath, desc) = mroot.FullName(&rootDir);

        if !desc {
            continue;
        }

        f(&mountPath, &m)
    }
}

pub fn NewMountInfoFile(task: &Task, thread: &Thread, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let v = NewMountInfoFileSimpleFileInode(task, thread, &ROOT_OWNER, &FilePermissions::FromMode(FileMode(0o400)), FSMagic::PROC_SUPER_MAGIC);
    return NewProcInode(&Arc::new(v), msrc, InodeType::SpecialFile, Some(thread.clone()))

}

pub fn NewMountInfoFileSimpleFileInode(task: &Task,
                               thread: &Thread,
                               owner: &FileOwner,
                               perms: &FilePermissions,
                               typ: u64)
                               -> SimpleFileInode<MountInfoFile> {
    let io = MountInfoFile{thread: thread.clone()};

    return SimpleFileInode::New(task, owner, perms, typ, false, io)
}

pub struct MountInfoFile {
    thread: Thread,
}

impl MountInfoFile {
    pub fn GenSnapshot(&self, task: &Task) -> Vec<u8> {
        info!("MountInfoFile GenSnapshot...");
        let mut ret = "".to_string();

        let mountns = task.mountNS.clone();
        ForEachMount(&self.thread, mountns,  &mut |mountPath: &str, m: &Arc<QMutex<Mount>>| {
            // Format:
            // 36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 - ext3 /dev/root rw,errors=continue
            // (1)(2)(3)   (4)   (5)      (6)      (7)   (8) (9)   (10)         (11)

            // (1) MountSource ID.
            ret += &format!("{} ", m.lock().Id);

            // (2)  Parent ID (or this ID if there is no parent).
            let mut pId = m.lock().Id;
            let isroot = m.lock().IsRoot();
            let isundo = m.lock().IsUndo();
            if !isroot && !isundo {
                pId = m.lock().Pid;
            }

            ret += &format!("{} ", pId);

            // (3) Major:Minor device ID. We don't have a superblock, so we
            // just use the root inode device number.
            let mroot = m.lock().Root();
            let sa = mroot.Inode().lock().StableAttr;
            ret += &format!("{}:{} ", sa.DeviceFileMajor, sa.DeviceFileMinor);

            // (4) Root: the pathname of the directory in the filesystem
            // which forms the root of this mount.
            //
            // NOTE(b/78135857): This will always be "/" until we implement
            // bind mounts.
            ret += "/ ";

            // (5) Mount point (relative to process root).
            ret += mountPath;

            // (6) Mount options.
            let mountSource = mroot.Inode().lock().MountSource.clone();
            let flags = mountSource.lock().Flags;
            let mut opts = "rw".to_string();
            if flags.ReadOnly {
                opts = "ro".to_string();
            }

            if flags.NoAtime {
                opts += ",noatime";
            }

            if flags.NoExec {
                opts += ",noexec";
            }

            ret += &format!("{} ", opts);

            // (7) Optional fields: zero or more fields of the form "tag[:value]".
            // (8) Separator: the end of the optional fields is marked by a single hyphen.
            ret += "- ";

            // (9) Filesystem type.
            ret += &format!("{} ", &mountSource.lock().FileSystemType);

            // (10) Mount source: filesystem-specific information or "none".
            ret += "none ";

            // (11) Superblock options. Only "ro/rw" is supported for now,
            // and is the same as the filesystem option.
            ret += &format!("{}\n", opts);
        });

        return ret.as_bytes().to_vec();
    }
}

impl SimpleFileTrait for MountInfoFile {
    fn GetFile(&self, task: &Task, _dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        let fops = NewSnapshotReadonlyFileOperations(self.GenSnapshot(task));
        let file = File::New(dirent, &flags, fops);
        return Ok(file);
    }
}

pub fn NewMountsFile(task: &Task, thread: &Thread, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let v = NewMountsFileSimpleFileInode(task, thread, &ROOT_OWNER, &FilePermissions::FromMode(FileMode(0o400)), FSMagic::PROC_SUPER_MAGIC);
    return NewProcInode(&Arc::new(v), msrc, InodeType::SpecialFile, Some(thread.clone()))

}

pub fn NewMountsFileSimpleFileInode(task: &Task,
                                       thread: &Thread,
                                       owner: &FileOwner,
                                       perms: &FilePermissions,
                                       typ: u64)
                                       -> SimpleFileInode<MountsFile> {
    let io = MountsFile{thread: thread.clone()};

    return SimpleFileInode::New(task, owner, perms, typ, false, io)
}

pub struct MountsFile {
    thread: Thread,
}

impl MountsFile {
    pub fn GenSnapshot(&self, task: &Task) -> Vec<u8> {
        let mut ret = "".to_string();

        let mountns = task.mountNS.clone();
        ForEachMount(&self.thread, mountns, &mut |mountPath: &str, m: &Arc<QMutex<Mount>>| {
            let mroot = m.lock().Root();
            let mountSource = mroot.Inode().lock().MountSource.clone();
            let flags = mountSource.lock().Flags;

            let opts = if flags.ReadOnly {
                "ro"
            } else {
                "rw"
            };

            ret += &format!("{} {} {} {} {} {}\n",
                            "none",
                            mountPath,
                            mountSource.lock().FileSystemType,
                            opts,
                            0,
                            0);
        });

        return ret.as_bytes().to_vec();
    }
}

impl SimpleFileTrait for MountsFile {
    fn GetFile(&self, task: &Task, _dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        let fops = NewSnapshotReadonlyFileOperations(self.GenSnapshot(task));
        let file = File::New(dirent, &flags, fops);
        return Ok(file);
    }
}