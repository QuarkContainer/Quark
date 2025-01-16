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

use crate::qlib::kernel::util::sharedcstring::SharedCString;
use crate::qlib::mutex::*;
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::*;

use super::super::super::auth::*;
use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::super::path::*;
use super::super::kernel::time::*;
use super::super::socket::unix::transport::unix::*;
use super::super::task::*;
use super::attr::*;
use super::copy_up::*;
use super::dirent::*;
use super::file::*;
use super::file_overlay::*;
use super::flags::*;
use super::inode::*;
use super::mount::*;
use super::overlay::*;

pub fn OverlayHasWhiteout(task: &Task, parent: &Inode, name: &str) -> bool {
    match parent.Getxattr(task, &XattrOverlayWhiteout(name), 1) {
        Ok(s) => {
            if s.len() != 0 {
                return false;
            }

            return s[0] == 'y' as u8;
        }
        _ => return false,
    }
}

pub fn overlayCreateWhiteout(parent: &mut Inode, name: &str) -> Result<()> {
    let iops = parent.lock().InodeOp.clone();
    
    return iops.Setxattr(
        parent,
        &XattrOverlayWhiteout(name),
        SharedCString::New(&"y").Slice(),
        0,
    );
}

pub fn overlayLookup(
    task: &Task,
    parent: &Arc<RwLock<OverlayEntry>>,
    inode: &Inode,
    name: &str,
) -> Result<(Dirent, bool)> {
    let parent = parent.read();

    if parent.upper.is_none() && parent.lower.is_none() {
        panic!("invalid overlayEntry, needs at least one Inode")
    }

    let mut upperInode: Option<Inode> = None;
    let mut lowerInode: Option<Inode> = None;

    if parent.upper.is_some() {
        let upper = parent.upper.as_ref().unwrap().clone();
        match upper.Lookup(task, name) {
            Ok(child) => {
                upperInode = Some(child.Inode());
            }
            Err(Error::SysError(SysErr::ENOENT)) => {
                upperInode = None;
            }
            Err(e) => return Err(e),
        }

        if OverlayHasWhiteout(task, &upper, name) {
            if upperInode.is_none() {
                return Err(Error::SysError(SysErr::ENOENT));
            }

            let entry = OverlayEntry::New(task, upperInode, None, false)?;
            let oinode = NewOverlayInode(task, entry, &inode.lock().MountSource);
            let d = Dirent::New(&oinode, name);
            return Ok((d, true));
        }
    }

    if parent.lower.is_some() {
        let lower = parent.lower.as_ref().unwrap().clone();
        match lower.Lookup(task, name) {
            Ok(child) => {
                if upperInode.is_none() {
                    lowerInode = Some(child.Inode());
                } else {
                    let childInode = child.Inode();
                    if upperInode.as_ref().unwrap().StableAttr().Type
                        == childInode.StableAttr().Type
                        || upperInode.as_ref().unwrap().StableAttr().IsDir()
                            && childInode.StableAttr().IsDir()
                    {
                        lowerInode = Some(child.Inode());
                    }
                }
            }
            Err(Error::SysError(SysErr::ENOENT)) => {
                lowerInode = None;
            }
            Err(e) => return Err(e),
        }
    }

    if upperInode.is_none() && lowerInode.is_none() {
        return Err(Error::SysError(SysErr::ENOENT));
    }

    let lowerExists = lowerInode.is_some();

    if upperInode.is_some() && lowerInode.is_some() {
        upperInode.as_ref().unwrap().lock().StableAttr =
            lowerInode.as_ref().unwrap().lock().StableAttr;

        if upperInode.as_ref().unwrap().StableAttr().IsDir() {
            lowerInode = None;
        }
    }

    let upperIsSome = upperInode.is_some();

    let entry = OverlayEntry::New(task, upperInode, lowerInode, lowerExists)?;
    let oinode = NewOverlayInode(task, entry, &inode.lock().MountSource);
    let d = Dirent::New(&oinode, name);

    return Ok((d, upperIsSome));
}

pub fn OverlayCreate(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    parent: &Dirent,
    name: &str,
    flags: &FileFlags,
    perm: &FilePermissions,
) -> Result<File> {
    CopyUpLockedForRename(task, parent)?;

    let mut upper = o.read().upper.as_ref().unwrap().clone();
    let upperInodeOp = upper.lock().InodeOp.clone();
    let upperFile = upperInodeOp.Create(task, &mut upper, name, flags, perm)?;

    let upperFileInode = upperFile.Dirent.Inode();
    let entry = match OverlayEntry::New(task, Some(upperFileInode.clone()), None, false) {
        Ok(e) => e,
        Err(e) => {
            cleanupUpper(task, &mut upper, name);
            return Err(e);
        }
    };

    //let mut upperDirent = Dirent::NewTransient(&upperFileInode);
    upperFile.Dirent.main.lock().Inode = upperFileInode;
    upperFile.Dirent.main.lock().Parent = None;

    let parentInode = parent.Inode();
    let overlayInode = NewOverlayInode(task, entry, &parentInode.lock().MountSource);
    let overlayDirent = Dirent::New(&overlayInode, name);

    let mut oFlags = *flags;
    oFlags.Pread = upperFile.Flags().Pread;
    oFlags.PWrite = upperFile.Flags().PWrite;
    let overlayFile = File::New(
        &overlayDirent,
        &oFlags,
        OverlayFileOperations(Arc::new(OverlayFileOperationsInner {
            upper: QMutex::new(Some(upperFile)),
            ..Default::default()
        }))
        .into(),
    );

    return Ok(overlayFile);
}

pub fn overlayCreateDirectory(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    parent: &Dirent,
    name: &str,
    perm: &FilePermissions,
) -> Result<()> {
    CopyUpLockedForRename(task, parent)?;

    let mut inode = o.read().upper.as_ref().unwrap().clone();
    let iops = inode.lock().InodeOp.clone();
    let res = iops.CreateDirectory(task, &mut inode, name, perm);
    return res;
}

pub fn overlayCreateLink(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    parent: &Dirent,
    oldname: &str,
    newname: &str,
) -> Result<()> {
    CopyUpLockedForRename(task, parent)?;

    let mut inode = o.read().upper.as_ref().unwrap().clone();
    let iops = inode.lock().InodeOp.clone();
    let res = iops.CreateLink(task, &mut inode, oldname, newname);
    return res;
}

pub fn overlayCreateHardLink(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    parent: &Dirent,
    target: &Dirent,
    name: &str,
) -> Result<()> {
    CopyUpLockedForRename(task, parent)?;
    CopyUpLockedForRename(task, target)?;

    let mut inode = o.read().upper.as_ref().unwrap().clone();
    let iops = inode.lock().InodeOp.clone();

    let tmpInode = target.Inode();
    let targetInode = tmpInode
        .lock()
        .Overlay
        .as_ref()
        .unwrap()
        .read()
        .upper
        .as_ref()
        .unwrap()
        .clone();
    let res = iops.CreateHardLink(task, &mut inode, &targetInode, name);
    return res;
}

pub fn overlayCreateFifo(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    parent: &Dirent,
    name: &str,
    perm: &FilePermissions,
) -> Result<()> {
    CopyUpLockedForRename(task, parent)?;

    let mut inode = o.read().upper.as_ref().unwrap().clone();
    let iops = inode.lock().InodeOp.clone();
    let res = iops.CreateFifo(task, &mut inode, name, perm);
    return res;
}

pub fn overlayRemove(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    parent: &Dirent,
    child: &Dirent,
) -> Result<()> {
    CopyUpLockedForRename(task, parent)?;

    let childinode = child.Inode();
    let overlay = childinode.lock().Overlay.as_ref().unwrap().clone();
    let overlaylock = overlay.read();

    if overlaylock.upper.is_some() {
        let mut oupper = o.read().upper.as_ref().unwrap().clone();
        let oupperOps = oupper.lock().InodeOp.clone();
        if childinode.StableAttr().Type == InodeType::Directory {
            oupperOps.RemoveDirectory(task, &mut oupper, &child.Name())?
        } else {
            oupperOps.Remove(task, &mut oupper, &child.Name())?
        }
    }

    if overlaylock.LowerExists {
        let mut oupper = o.read().upper.as_ref().unwrap().clone();
        return overlayCreateWhiteout(&mut oupper, &child.Name());
    }

    return Ok(());
}

pub fn overlayRename(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    oldParent: &Dirent,
    renamed: &Dirent,
    newParent: &Dirent,
    newName: &str,
    replacement: bool,
) -> Result<()> {
    let renamedInode = renamed.Inode();
    let oldParentInode = oldParent.Inode();
    let newParentInode = newParent.Inode();
    if renamedInode.lock().Overlay.is_none()
        || oldParentInode.lock().Overlay.is_none()
        || newParentInode.lock().Overlay.is_none()
    {
        return Err(Error::SysError(SysErr::EXDEV));
    }

    let mut replacement = replacement;

    if replacement {
        let newParentInode = newParent.Inode();
        let newParentOverlay = newParentInode.lock().Overlay.as_ref().unwrap().clone();
        match overlayLookup(task, &newParentOverlay, &newParentInode, newName) {
            Ok((replaced, inUpper)) => {
                if !inUpper {
                    replacement = false;
                }

                let replacedInode = replaced.Inode();
                let isDir = replacedInode.StableAttr().IsDir();
                if isDir {
                    let children = ReaddirOne(task, &replaced)?;

                    if children.len() > 0 {
                        return Err(Error::SysError(SysErr::ENOTEMPTY));
                    }
                }
            }
            Err(Error::SysError(SysErr::ENOENT)) => (),
            Err(e) => return Err(e),
        }
    }

    CopyUpLockedForRename(task, renamed)?;
    CopyUpLockedForRename(task, newParent)?;

    let oldName = renamed.Name();

    let overlayUpper = o.read().upper.as_ref().unwrap().clone();
    let overlayUpperOps = overlayUpper.lock().InodeOp.clone();

    let renamedInode = renamed.Inode();
    let mut renamedUpper = renamedInode
        .lock()
        .Overlay
        .as_ref()
        .unwrap()
        .read()
        .upper
        .as_ref()
        .unwrap()
        .clone();
    let oldParentInode = oldParent.Inode();
    let mut oldParentUpper = oldParentInode
        .lock()
        .Overlay
        .as_ref()
        .unwrap()
        .read()
        .upper
        .as_ref()
        .unwrap()
        .clone();
    let newParentUpper = newParent
        .Inode()
        .lock()
        .Overlay
        .as_ref()
        .unwrap()
        .read()
        .upper
        .as_ref()
        .unwrap()
        .clone();

    overlayUpperOps.Rename(
        task,
        &mut renamedUpper,
        &oldParentUpper,
        &oldName,
        &newParentUpper,
        newName,
        replacement,
    )?;

    let lowerExists = renamedInode
        .lock()
        .Overlay
        .as_ref()
        .unwrap()
        .read()
        .LowerExists;

    if lowerExists {
        return overlayCreateWhiteout(&mut oldParentUpper, &oldName);
    }

    return Ok(());
}

pub fn overlayBind(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    name: &str,
    parent: &Dirent,
    data: &BoundEndpoint,
    perm: &FilePermissions,
) -> Result<Dirent> {
    let overlay = o.write();

    // We do not support doing anything exciting with sockets unless there
    // is already a directory in the upper filesystem.
    if overlay.upper.is_none() {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    let upperInode = overlay.upper.as_ref().unwrap().clone();
    let iops = upperInode.lock().InodeOp.clone();
    let d = iops.Bind(task, &upperInode, name, data, perm)?;

    let inode = d.Inode();

    let msrc = parent.Inode().lock().MountSource.clone();
    // Create a new overlay entry and dirent for the socket.
    let entry = OverlayEntry::New(task, Some(inode), None, false)?;

    let oInode = NewOverlayInode(task, entry, &msrc);
    return Ok(Dirent::New(&oInode, name));
}

pub fn overlayBoundEndpoint(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    path: &str,
) -> Option<BoundEndpoint> {
    let overlay = o.read();

    if overlay.upper.is_some() {
        let upperInode = overlay.upper.as_ref().unwrap().clone();
        let iops = upperInode.lock().InodeOp.clone();
        return iops.BoundEndpoint(task, &upperInode, path);
    }

    let lower = overlay.lower.as_ref().unwrap().clone();
    let overlay = lower.lock().Overlay.clone();
    match overlay {
        None => {
            let iops = lower.lock().InodeOp.clone();
            return iops.BoundEndpoint(task, &lower, path);
        }
        Some(overlay) => {
            // Lower is not an overlay. Call BoundEndpoint directly.
            return overlayBoundEndpoint(task, &overlay, path);
        }
    }
}

pub fn overlayGetFile(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    d: &Dirent,
    flags: &FileFlags,
) -> Result<File> {
    if flags.Write {
        copyUp(task, d)?
    }

    let mut flags = flags.clone();
    let overlay = o.read();

    if overlay.upper.is_some() {
        let upperInode = overlay.upper.as_ref().unwrap().clone();
        let upper = OverlayFile(task, &upperInode, &flags)?;

        flags.Pread = upper.Flags().Pread;
        flags.PWrite = upper.Flags().PWrite;

        let overlayFileOps = OverlayFileOperations(Arc::new(OverlayFileOperationsInner {
            upper: QMutex::new(Some(upper)),
            ..Default::default()
        }));

        let f = File::New(d, &flags, overlayFileOps.into());
        return Ok(f);
    }

    let lowerInode = overlay.lower.as_ref().unwrap().clone();
    let lower = OverlayFile(task, &lowerInode, &flags)?;

    flags.Pread = lower.Flags().Pread;
    flags.PWrite = lower.Flags().PWrite;

    let overlayFileOps = OverlayFileOperations(Arc::new(OverlayFileOperationsInner {
        upper: QMutex::new(Some(lower)),
        ..Default::default()
    }));

    let f = File::New(d, &flags, overlayFileOps.into());
    return Ok(f);
}

pub fn overlayUnstableAttr(task: &Task, o: &Arc<RwLock<OverlayEntry>>) -> Result<UnstableAttr> {
    let overlay = o.read();
    if overlay.upper.is_some() {
        let upperInode = overlay.upper.as_ref().unwrap().clone();
        return upperInode.UnstableAttr(task);
    } else {
        let lowerInode = overlay.lower.as_ref().unwrap().clone();
        return lowerInode.UnstableAttr(task);
    }
}

pub fn overlayStableAttr(o: &Arc<RwLock<OverlayEntry>>) -> StableAttr {
    let overlay = o.read();
    if overlay.upper.is_some() {
        let upperInode = overlay.upper.as_ref().unwrap().clone();
        return upperInode.StableAttr();
    } else {
        let lowerInode = overlay.lower.as_ref().unwrap().clone();
        return lowerInode.StableAttr();
    }
}

pub fn overlayGetxattr(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    name: &str,
    size: usize,
) -> Result<Vec<u8>> {
    if HasPrefix(name, &XATTR_OVERLAY_PREFIX.to_string()) {
        return Err(Error::SysError(SysErr::ENODATA));
    }

    let overlay = o.read();
    if overlay.upper.is_some() {
        let upperInode = overlay.upper.as_ref().unwrap().clone();
        return upperInode.Getxattr(task, name, size);
    } else {
        let lowerInode = overlay.lower.as_ref().unwrap().clone();
        return lowerInode.Getxattr(task, name, size);
    }
}

pub fn overlaySetxattr(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    d: &Dirent,
    name: &str,
    value: &[u8],
    flags: u32,
) -> Result<()> {
    if HasPrefix(name, &XATTR_OVERLAY_PREFIX.to_string()) {
        return Err(Error::SysError(SysErr::ENODATA));
    }

    copyUp(task, d)?;

    let overlay = o.read();
    let mut upperInode = overlay.upper.as_ref().unwrap().clone();
    let upperInodeOps = upperInode.lock().InodeOp.clone();
    return upperInodeOps.Setxattr(&mut upperInode, name, value, flags);
}

pub fn overlayListxattr(o: &Arc<RwLock<OverlayEntry>>, size: usize) -> Result<Vec<String>> {
    let overlay = o.read();
    let names = if overlay.upper.is_some() {
        let upperInode = overlay.upper.as_ref().unwrap().clone();
        upperInode.Listxattr(size)?
    } else {
        let lowerInode = overlay.lower.as_ref().unwrap().clone();
        lowerInode.Listxattr(size)?
    };

    let overlayPrefix = XATTR_OVERLAY_PREFIX.to_string();

    let mut res = Vec::new();
    for name in names {
        if !HasPrefix(&name, &overlayPrefix) {
            res.push(name)
        }
    }

    return Ok(res);
}

pub fn overlayRemovexattr(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    d: &Dirent,
    name: &str,
) -> Result<()> {
    // Don't allow changes to overlay xattrs through a removexattr syscall.
    if IsXattrOverlay(name) {
        return Err(Error::SysError(SysErr::EPERM));
    }

    copyUp(task, d)?;

    let overlay = o.read();
    let mut upperInode = overlay.upper.as_ref().unwrap().clone();
    let upperInodeOps = upperInode.lock().InodeOp.clone();
    return upperInodeOps.Removexattr(&mut upperInode, name);
}

pub fn overlayCheck(task: &Task, o: &Arc<RwLock<OverlayEntry>>, p: &PermMask) -> Result<()> {
    let overlay = o.read();
    if overlay.upper.is_some() {
        let upperInode = overlay.upper.as_ref().unwrap().clone();
        return upperInode.CheckPermission(task, p);
    } else {
        let mut p = *p;
        if p.write {
            p.write = false;
            p.read = true;
        }

        let lowerInode = overlay.lower.as_ref().unwrap().clone();
        return lowerInode.check(task, &p);
    }
}

pub fn overlaySetPermissions(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    d: &Dirent,
    f: FilePermissions,
) -> bool {
    match copyUp(task, d) {
        Err(_) => return false,
        Ok(()) => (),
    };

    let overlay = o.read();
    let mut upperInode = overlay.upper.as_ref().unwrap().clone();
    let upperInodeOps = upperInode.lock().InodeOp.clone();
    return upperInodeOps.SetPermissions(task, &mut upperInode, f);
}

pub fn overlaySetOwner(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    d: &Dirent,
    owner: &FileOwner,
) -> Result<()> {
    copyUp(task, d)?;

    let overlay = o.read();
    let mut upperInode = overlay.upper.as_ref().unwrap().clone();
    let upperInodeOps = upperInode.lock().InodeOp.clone();
    return upperInodeOps.SetOwner(task, &mut upperInode, owner);
}

pub fn overlaySetTimestamps(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    d: &Dirent,
    ts: &InterTimeSpec,
) -> Result<()> {
    copyUp(task, d)?;

    let overlay = o.read();
    let mut upperInode = overlay.upper.as_ref().unwrap().clone();
    let upperInodeOps = upperInode.lock().InodeOp.clone();
    return upperInodeOps.SetTimestamps(task, &mut upperInode, ts);
}

pub fn overlayTruncate(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    d: &Dirent,
    size: i64,
) -> Result<()> {
    copyUp(task, d)?;

    let overlay = o.read();
    let mut upperInode = overlay.upper.as_ref().unwrap().clone();
    let upperInodeOps = upperInode.lock().InodeOp.clone();
    return upperInodeOps.Truncate(task, &mut upperInode, size);
}

pub fn overlayAllocate(
    task: &Task,
    o: &Arc<RwLock<OverlayEntry>>,
    d: &Dirent,
    offset: i64,
    length: i64,
) -> Result<()> {
    copyUp(task, d)?;

    let overlay = o.read();
    let mut upperInode = overlay.upper.as_ref().unwrap().clone();
    let upperInodeOps = upperInode.lock().InodeOp.clone();
    return upperInodeOps.Allocate(task, &mut upperInode, offset, length);
}

pub fn overlayReadlink(task: &Task, o: &Arc<RwLock<OverlayEntry>>) -> Result<String> {
    let overlay = o.read();
    if overlay.upper.is_some() {
        let upperInode = overlay.upper.as_ref().unwrap().clone();
        return upperInode.ReadLink(task);
    } else {
        let lowerInode = overlay.lower.as_ref().unwrap().clone();
        return lowerInode.ReadLink(task);
    }
}

pub fn overlayGetlink(task: &Task, o: &Arc<RwLock<OverlayEntry>>) -> Result<Dirent> {
    let overlay = o.read();
    let _dirent = if overlay.upper.is_some() {
        let upperInode = overlay.upper.as_ref().unwrap().clone();
        upperInode.GetLink(task)?
    } else {
        let lowerInode = overlay.lower.as_ref().unwrap().clone();
        lowerInode.GetLink(task)?
    };

    //todo: fix it
    /*
    if dirent != nil {
        // This dirent is likely bogus (its Inode likely doesn't contain
        // the right overlayEntry). So we're forced to drop it on the
        // ground and claim that jumping around the filesystem like this
        // is not supported.
        name, _ := dirent.FullName(nil)
        dirent.DecRef()

        // Claim that the path is not accessible.
        err = syserror.EACCES
        log.Warningf("Getlink not supported in overlay for %q", name)
    }
    return nil, err
    */
    panic!("overlayGetlink: get dirent");
}

pub fn overlayStatFS(task: &Task, o: &Arc<RwLock<OverlayEntry>>) -> Result<FsInfo> {
    let overlay = o.read();
    let mut info = if overlay.upper.is_some() {
        let upperInode = overlay.upper.as_ref().unwrap().clone();
        upperInode.StatFS(task)?
    } else {
        let lowerInode = overlay.lower.as_ref().unwrap().clone();
        lowerInode.StatFS(task)?
    };

    info.Type = FSMagic::OVERLAYFS_SUPER_MAGIC;
    return Ok(info);
}
