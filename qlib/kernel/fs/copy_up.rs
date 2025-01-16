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

use alloc::vec::Vec;

use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::kernel::time::*;
use super::super::task::*;
use super::attr::*;
use super::dirent::*;
use super::file_overlay::*;
use super::flags::*;
use super::inode::*;
use super::overlay::*;
use crate::qlib::kernel::fs::file::FileOperations;
use crate::GUEST_HOST_SHARED_ALLOCATOR;

pub fn copyUp(task: &Task, d: &Dirent) -> Result<()> {
    let _a = RENAME.read();

    return CopyUpLockedForRename(task, d);
}

pub fn CopyUpLockedForRename(task: &Task, d: &Dirent) -> Result<()> {
    loop {
        {
            let inode = d.inode.clone();
            let inodelock = inode.lock();
            let overlay = inodelock.Overlay.as_ref().unwrap().read();
            if overlay.upper.is_some() {
                return Ok(());
            }
        }

        let next = findNextCopyup(task, d);
        doCopyup(task, &next)?;
    }
}

fn findNextCopyup(_task: &Task, d: &Dirent) -> Dirent {
    let mut next = d.clone();
    let mut parent = next.Parent().as_ref().unwrap().clone();

    loop {
        {
            let inodeLock = parent.inode.lock();
            let overlay = inodeLock.Overlay.as_ref().unwrap().read();
            if overlay.upper.is_some() {
                return next;
            }
        }

        next = parent;
        parent = next.Parent().as_ref().unwrap().clone();
    }
}

fn doCopyup(task: &Task, next: &Dirent) -> Result<()> {
    let nextInode = next.inode.lock();
    let nextOverlay = nextInode.Overlay.as_ref().unwrap().read();
    let t = nextOverlay.lower.as_ref().unwrap().StableAttr().Type;

    if !(t == InodeType::RegularFile || t == InodeType::Directory || t == InodeType::Symlink) {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if nextOverlay.upper.is_some() {
        // We raced with another doCopyUp, no problem.
        return Ok(());
    }

    let parent = next.Parent().as_ref().unwrap().clone();

    let attrs = nextOverlay.lower.as_ref().unwrap().UnstableAttr(task)?;

    let parentOverlay = parent.Inode().lock().Overlay.as_ref().unwrap().clone();
    let mut parentUpper = parentOverlay.read().upper.as_ref().unwrap().clone();

    let mut childUpperInode;

    let root = task.Root();
    let nextStableAttr = nextInode.StableAttr().clone();
    match nextStableAttr.Type {
        InodeType::RegularFile => {
            let childFile = parentUpper.Create(
                task,
                &root,
                &next.Name(),
                &FileFlags {
                    Read: true,
                    Write: true,
                    ..Default::default()
                },
                &attrs.Perms,
            )?;
            childUpperInode = childFile.Dirent.Inode();
        }
        InodeType::Directory => {
            parentUpper.CreateDirectory(task, &root, &next.Name(), &attrs.Perms)?;
            let childUpper = match parentUpper.Lookup(task, &next.Name()) {
                Err(e) => {
                    info!("copy up failed to lookup directory: {:?}", e);
                    cleanupUpper(task, &mut parentUpper, &next.Name());
                    return Err(e);
                }
                Ok(n) => n,
            };

            childUpperInode = childUpper.Inode();
        }
        InodeType::Symlink => {
            let childLower = next
                .inode
                .lock()
                .Overlay
                .as_ref()
                .unwrap()
                .read()
                .lower
                .as_ref()
                .unwrap()
                .clone();
            let link = childLower.ReadLink(task)?;

            parentUpper.CreateLink(task, &root, &link, &next.Name())?;
            let childUpper = match parentUpper.Lookup(task, &next.Name()) {
                Err(e) => {
                    info!("copy up failed to lookup directory: {:?}", e);
                    cleanupUpper(task, &mut parentUpper, &next.Name());
                    return Err(e);
                }
                Ok(n) => n,
            };

            childUpperInode = childUpper.Inode();
        }
        _ => {
            panic!(
                "copy up of invalid type {:?} on {}",
                nextStableAttr.Type,
                &next.Name()
            )
        }
    }

    let lower = nextOverlay.lower.as_ref().unwrap().clone();
    copyAttributesLocked(task, &mut childUpperInode, &lower)?;
    copyContentsLocked(task, &mut childUpperInode, &lower, attrs.Size)?;

    //todo: handle map

    return Ok(());
}

pub fn cleanupUpper(task: &Task, parent: &mut Inode, name: &str) {
    let iops = parent.lock().InodeOp.clone();
    match iops.Remove(task, parent, name) {
        Ok(()) => return,
        Err(e) => panic!("overlay filesystem is in an inconsistent state: failed to remove {} from upper filesystem: {:?}", name, e)
    }
}

fn copyContentsLocked(task: &Task, upper: &Inode, lower: &Inode, size: i64) -> Result<()> {
    if lower.StableAttr().Type != InodeType::RegularFile {
        return Ok(());
    }

    let upperFile = OverlayFile(
        task,
        upper,
        &FileFlags {
            Write: true,
            ..Default::default()
        },
    )?;
    let lowerFile = OverlayFile(
        task,
        lower,
        &FileFlags {
            Read: true,
            ..Default::default()
        },
    )?;

    let buf: [u8; 4096] = [0; 4096];

    let upperOps = upperFile.FileOp.clone();
    let lowerOps = lowerFile.FileOp.clone();

    let iov = IoVec::New(&buf);
    let mut iovs: [IoVec; 1] = [iov];

    let mut offset: i64 = 0;
    loop {
        let nr = lowerOps.ReadAt(task, &lowerFile, &mut iovs, offset, false)?;
        if nr == 0 {
            if offset != size {
                panic!(
                    "filesystem is in an inconsistent state: wrote only {} bytes of {} sized file",
                    offset, size
                )
            }

            return Ok(());
        }

        let tmpIov = IoVec::New(&buf[..nr as usize]);
        let tmpIovs: [IoVec; 1] = [tmpIov];

        let nw = upperOps.WriteAt(task, &upperFile, &tmpIovs, offset, false)?;
        assert!(nw == nr, "copyContentsLocked nr nw mismatch");

        offset += nw;
    }
}

fn copyAttributesLocked(task: &Task, upper: &mut Inode, lower: &Inode) -> Result<()> {
    let lowerAttr = lower.UnstableAttr(task)?;
    let lowerXattr = match lower.Listxattr(Xattr::XATTR_SIZE_MAX) {
        Err(Error::SysError(SysErr::EOPNOTSUPP)) => Vec::new(),
        Ok(a) => a,
        Err(e) => return Err(e),
    };

    let upperInodeOp = upper.lock().InodeOp.clone();
    upperInodeOp.SetOwner(task, upper, &lowerAttr.Owner)?;
    upperInodeOp.SetTimestamps(
        task,
        upper,
        &InterTimeSpec {
            ATime: lowerAttr.AccessTime,
            MTime: lowerAttr.ModificationTime,
            ..Default::default()
        },
    )?;

    for name in &lowerXattr {
        if IsXattrOverlay(name) {
            continue;
        }

        let value = lower
            .Getxattr(task, name, Xattr::XATTR_SIZE_MAX)?
            .to_vec_in(GUEST_HOST_SHARED_ALLOCATOR);

        upperInodeOp.Setxattr(upper, name, &value, 0)?;
    }

    return Ok(());
}
