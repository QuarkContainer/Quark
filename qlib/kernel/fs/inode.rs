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

use alloc::string::String;
use alloc::vec::Vec;
use spin::*;
//use alloc::string::ToString;
use crate::qlib::mutex::*;
use alloc::sync::Arc;
use core::any::Any;
use core::ops::Deref;

use super::super::super::auth::*;
use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::kernel::time::*;
use super::super::socket::unix::transport::unix::*;
use super::super::task::*;
use super::super::uid::*;

use super::attr::*;
use super::dentry::*;
use super::dirent::*;
use super::file::*;
use super::flags::*;
use super::host::hostinodeop::*;
use super::inode_overlay::*;
use super::lock::*;
use super::mount::*;
use super::overlay::*;
use super::inotify::*;

pub fn ContextCanAccessFile(task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
    let creds = task.creds.clone();
    let uattr = inode.UnstableAttr(task)?;

    //info!("ContextCanAccessFile 1, perms is {:?}", &uattr.Perms);
    let mut p = &uattr.Perms.Other;
    {
        let creds = creds.lock();
        if uattr.Owner.UID == creds.EffectiveKUID {
            p = &uattr.Perms.User
        } else if creds.InGroup(uattr.Owner.GID) {
            p = &uattr.Perms.Group
        }
    }

    //info!("ContextCanAccessFile 2");
    if inode.StableAttr().IsFile()
        && reqPerms.execute
        && inode.lock().MountSource.lock().Flags.NoExec
    {
        return Ok(false);
    }

    //info!("ContextCanAccessFile 3, p is {:?}, reqPerms is {:?}", &p, reqPerms);
    if p.SupersetOf(reqPerms) {
        return Ok(true);
    }

    //info!("ContextCanAccessFile 4");
    if inode.StableAttr().IsDir() {
        if CheckCapability(&creds, Capability::CAP_DAC_OVERRIDE, &uattr) {
            return Ok(true);
        }
    }

    //info!("ContextCanAccessFile 5");
    if reqPerms.OnlyRead() && CheckCapability(&creds, Capability::CAP_DAC_READ_SEARCH, &uattr) {
        return Ok(true);
    }

    //info!("ContextCanAccessFile 6");
    return Ok(false);
}

#[derive(Debug)]
pub enum IopsType {
    MockInodeOperations,
    FullDevice,
    NullDevice,
    RandomDevice,
    TTYDevice,
    ZeroDevice,
    HostInodeOp,
    TaskOwnedInodeOps,
    StaticFileInodeOps,
    SeqFile,
    Dir,
    SocketInodeOps,
    Symlink,
    TmpfsDir,
    TmpfsFifoInodeOp,
    TmpfsFileInodeOp,
    TmpfsSocket,
    TmpfsSymlink,
    DirInodeOperations,
    MasterInodeOperations,
    SlaveInodeOperations,
    PipeIops,
    DirNode,
    SymlinkNode,
    SimpleFileInode,
}

pub trait InodeOperations: Sync + Send {
    fn as_any(&self) -> &Any;
    fn IopsType(&self) -> IopsType;

    fn InodeType(&self) -> InodeType;
    fn InodeFileType(&self) -> InodeFileType;
    fn WouldBlock(&self) -> bool;
    fn Lookup(&self, task: &Task, dir: &Inode, name: &str) -> Result<Dirent>;
    fn Create(
        &self,
        task: &Task,
        dir: &mut Inode,
        name: &str,
        flags: &FileFlags,
        perm: &FilePermissions,
    ) -> Result<File>;
    fn CreateDirectory(
        &self,
        task: &Task,
        dir: &mut Inode,
        name: &str,
        perm: &FilePermissions,
    ) -> Result<()>;
    fn CreateLink(&self, task: &Task, dir: &mut Inode, oldname: &str, newname: &str) -> Result<()>;
    fn CreateHardLink(
        &self,
        task: &Task,
        dir: &mut Inode,
        target: &Inode,
        name: &str,
    ) -> Result<()>;
    fn CreateFifo(
        &self,
        task: &Task,
        dir: &mut Inode,
        name: &str,
        perm: &FilePermissions,
    ) -> Result<()>;
    //fn RemoveDirent(&mut self, dir: &mut InodeStruStru, remove: &Arc<QMutex<Dirent>>) -> Result<()> ;
    fn Remove(&self, task: &Task, dir: &mut Inode, name: &str) -> Result<()>;
    fn RemoveDirectory(&self, task: &Task, dir: &mut Inode, name: &str) -> Result<()>;
    fn Rename(
        &self,
        task: &Task,
        dir: &mut Inode,
        oldParent: &Inode,
        oldname: &str,
        newParent: &Inode,
        newname: &str,
        replacement: bool,
    ) -> Result<()>;
    fn Bind(
        &self,
        _task: &Task,
        _dir: &Inode,
        _name: &str,
        _data: &BoundEndpoint,
        _perms: &FilePermissions,
    ) -> Result<Dirent>;
    fn BoundEndpoint(&self, _task: &Task, inode: &Inode, path: &str) -> Option<BoundEndpoint>;
    fn GetFile(&self, task: &Task, dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File>;
    fn UnstableAttr(&self, task: &Task) -> Result<UnstableAttr>;
    fn Getxattr(&self, dir: &Inode, name: &str) -> Result<String>;
    fn Setxattr(&self, dir: &mut Inode, name: &str, value: &str) -> Result<()>;
    fn Listxattr(&self, dir: &Inode) -> Result<Vec<String>>;
    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool>;
    fn SetPermissions(&self, task: &Task, dir: &mut Inode, f: FilePermissions) -> bool;
    fn SetOwner(&self, task: &Task, dir: &mut Inode, owner: &FileOwner) -> Result<()>;
    fn SetTimestamps(&self, task: &Task, dir: &mut Inode, ts: &InterTimeSpec) -> Result<()>;
    fn Truncate(&self, task: &Task, dir: &mut Inode, size: i64) -> Result<()>;
    fn Allocate(&self, task: &Task, dir: &mut Inode, offset: i64, length: i64) -> Result<()>;
    fn ReadLink(&self, _task: &Task, dir: &Inode) -> Result<String>;
    fn GetLink(&self, _task: &Task, dir: &Inode) -> Result<Dirent>;
    fn AddLink(&self, _task: &Task);
    fn DropLink(&self, _task: &Task);
    fn IsVirtual(&self) -> bool;
    fn Sync(&self) -> Result<()>;
    fn StatFS(&self, task: &Task) -> Result<FsInfo>;
    fn Mappable(&self) -> Result<HostInodeOp>;
}

// LockCtx is an Inode's lock context and contains different personalities of locks; both
// Posix and BSD style locks are supported.
//
// Note that in Linux fcntl(2) and flock(2) locks are _not_ cooperative, because race and
// deadlock conditions make merging them prohibitive. We do the same and keep them oblivious
// to each other but provide a "context" as a convenient container.
#[derive(Clone, Default)]
pub struct LockCtx {
    // Posix is a set of POSIX-style regional advisory locks, see fcntl(2).
    pub Posix: Locks,

    // BSD is a set of BSD-style advisory file wide locks, see flock(2).
    pub BSD: Locks,
}

#[derive(Clone)]
pub struct Inode(pub Arc<QMutex<InodeIntern>>);

impl Default for Inode {
    fn default() -> Self {
        return Self(Arc::new(QMutex::new(InodeIntern::New())));
    }
}

impl Deref for Inode {
    type Target = Arc<QMutex<InodeIntern>>;

    fn deref(&self) -> &Arc<QMutex<InodeIntern>> {
        &self.0
    }
}

impl Drop for Inode {
    fn drop(&mut self) {
        if Arc::strong_count(&self.0) == 1 {
            let watches = self.Watches();

            // If this inode is being destroyed because it was unlinked, queue a
            // deletion event. This may not be the case for inodes being revalidated.
            let unlinked = watches.read().unlinked;
            if unlinked {
                watches.Notify("", InotifyEvent::IN_DELETE_SELF, 0);
            }

            // Remove references from the watch owners to the watches on this inode,
            // since the watches are about to be GCed. Note that we don't need to worry
            // about the watch pins since if there were any active pins, this inode
            // wouldn't be in the destructor.
            watches.TargetDestroyed();
        }
    }
}

impl Inode {
    pub fn New<T: InodeOperations + 'static>(
        InodeOp: &Arc<T>,
        MountSource: &Arc<QMutex<MountSource>>,
        StableAttr: &StableAttr,
    ) -> Self {
        let inodeInternal = InodeIntern {
            UniqueId: NewUID(),
            InodeOp: InodeOp.clone(),
            StableAttr: StableAttr.clone(),
            LockCtx: LockCtx::default(),
            watches: Watches::default(),
            MountSource: MountSource.clone(),
            Overlay: None,
        };

        return Self(Arc::new(QMutex::new(inodeInternal)));
    }

    pub fn WouldBlock(&self) -> bool {
        return self.lock().InodeOp.WouldBlock();
    }

    pub fn NewHostInode(
        msrc: &Arc<QMutex<MountSource>>,
        fd: i32,
        fstat: &LibcStat,
        writeable: bool,
    ) -> Result<Self> {
        //info!("after fstat: {:?}", fstat.StableAttr());

        //println!("the stable attr is {:?}", &fstat.StableAttr());
        let iops = HostInodeOp::New(
            &msrc.lock().MountSourceOperations.clone(),
            fd,
            fstat.WouldBlock(),
            &fstat,
            writeable,
        );

        return Ok(Self(Arc::new(QMutex::new(InodeIntern {
            UniqueId: NewUID(),
            InodeOp: Arc::new(iops),
            StableAttr: fstat.StableAttr(),
            LockCtx: LockCtx::default(),
            watches: Watches::default(),
            MountSource: msrc.clone(),
            Overlay: None,
        }))));
    }

    pub fn InodeType(&self) -> InodeType {
        return self.lock().InodeOp.InodeType();
    }

    pub fn ID(&self) -> u64 {
        return self.lock().UniqueId;
    }

    pub fn Lookup(&self, task: &Task, name: &str) -> Result<Dirent> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            let (dirent, _) = overlayLookup(task, &overlay, self, name)?;
            return Ok(dirent);
        }

        let iops = self.lock().InodeOp.clone();
        let res = iops.Lookup(task, self, name);
        return res;
    }

    pub fn Create(
        &mut self,
        task: &Task,
        d: &Dirent,
        name: &str,
        flags: &FileFlags,
        perm: &FilePermissions,
    ) -> Result<File> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return OverlayCreate(task, &overlay, d, name, flags, perm);
        }

        let op = self.lock().InodeOp.clone();
        let res = op.Create(task, self, name, flags, perm);
        return res;
    }

    pub fn CreateDirectory(
        &mut self,
        task: &Task,
        d: &Dirent,
        name: &str,
        perm: &FilePermissions,
    ) -> Result<()> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayCreateDirectory(task, &overlay, d, name, perm);
        }

        let op = self.lock().InodeOp.clone();
        let res = op.CreateDirectory(task, self, name, &perm);
        return res;
    }

    pub fn CreateLink(
        &mut self,
        task: &Task,
        d: &Dirent,
        oldname: &str,
        newname: &str,
    ) -> Result<()> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayCreateLink(task, &overlay, d, oldname, newname);
        }

        let op = self.lock().InodeOp.clone();
        let res = op.CreateLink(task, self, oldname, newname);
        return res;
    }

    pub fn CreateHardLink(
        &mut self,
        task: &Task,
        d: &Dirent,
        target: &Dirent,
        name: &str,
    ) -> Result<()> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayCreateHardLink(task, &overlay, d, target, name);
        }

        let op = self.lock().InodeOp.clone();
        let inode = target.Inode();
        let res = op.CreateHardLink(task, self, &inode, name);
        return res;
    }

    pub fn CreateFifo(
        &mut self,
        task: &Task,
        d: &Dirent,
        name: &str,
        perm: &FilePermissions,
    ) -> Result<()> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayCreateFifo(task, &overlay, d, name, perm);
        }

        let op = self.lock().InodeOp.clone();
        let res = op.CreateFifo(task, self, name, perm);
        return res;
    }

    pub fn Remove(&mut self, task: &Task, d: &Dirent, remove: &Dirent) -> Result<()> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayRemove(task, &overlay, d, remove);
        }

        let name = (remove.0).0.lock().Name.clone();
        let removeInode = remove.Inode();
        let typ = removeInode.StableAttr().Type;
        let op = self.lock().InodeOp.clone();

        let res = match typ {
            InodeType::Directory | InodeType::SpecialDirectory => {
                op.RemoveDirectory(task, self, &name)
            }
            _ => op.Remove(task, self, &name),
        };

        return res;
    }

    pub fn Rename(
        &mut self,
        task: &Task,
        oldParent: &Dirent,
        renamed: &Dirent,
        newParent: &Dirent,
        newname: &str,
        replacement: bool,
    ) -> Result<()> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayRename(
                task,
                &overlay,
                oldParent,
                renamed,
                newParent,
                newname,
                replacement,
            );
        }

        let oldInode = oldParent.Inode();
        let newInode = newParent.Inode();

        let oldname = (renamed.0).0.lock().Name.clone();

        let op = self.lock().InodeOp.clone();
        let res = op.Rename(
            task,
            self,
            &oldInode,
            &oldname,
            &newInode,
            newname,
            replacement,
        );
        return res;
    }

    pub fn Bind(
        &self,
        task: &Task,
        name: &str,
        data: &BoundEndpoint,
        perms: &FilePermissions,
    ) -> Result<Dirent> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayBind(task, &overlay, name, data, perms);
        }

        let op = self.lock().InodeOp.clone();
        let res = op.Bind(task, self, name, data, perms);
        return res;
    }

    pub fn BoundEndpoint(&self, task: &Task, path: &str) -> Option<BoundEndpoint> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayBoundEndpoint(task, &overlay, path);
        }

        let op = self.lock().InodeOp.clone();
        let res = op.BoundEndpoint(task, self, path);
        return res;
    }

    pub fn GetFile(&self, task: &Task, dirent: &Dirent, flags: &FileFlags) -> Result<File> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayGetFile(task, &overlay, dirent, flags);
        }

        let op = self.lock().InodeOp.clone();
        let res = op.GetFile(task, self, dirent, *flags);
        return res;
    }

    pub fn UnstableAttr(&self, task: &Task) -> Result<UnstableAttr> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayUnstableAttr(task, &overlay);
        }

        let op = self.lock().InodeOp.clone();
        let res = op.UnstableAttr(task);
        return res;
    }

    pub fn Getxattr(&self, name: &str) -> Result<String> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayGetxattr(&overlay, name);
        }

        let op = self.lock().InodeOp.clone();
        let res = op.Getxattr(self, name);
        return res;
    }

    pub fn Setxattr(&mut self, name: &str, value: &str) -> Result<()> {
        let op = self.lock().InodeOp.clone();
        op.Setxattr(self, name, value)?;
        return Ok(());
    }

    pub fn Listxattr(&self) -> Result<Vec<String>> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayListxattr(&overlay);
        }

        let op = self.lock().InodeOp.clone();
        let res = op.Listxattr(self);
        return res;
    }

    pub fn SetPermissions(&mut self, task: &Task, d: &Dirent, f: FilePermissions) -> bool {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlaySetPermissions(task, &overlay, d, f);
        }

        let op = self.lock().InodeOp.clone();
        let res = op.SetPermissions(task, self, f);
        return res;
    }

    pub fn SetOwner(&mut self, task: &Task, d: &Dirent, owner: &FileOwner) -> Result<()> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlaySetOwner(task, &overlay, d, owner);
        }

        let op = self.lock().InodeOp.clone();
        op.SetOwner(task, self, owner)?;
        return Ok(());
    }

    pub fn SetTimestamps(&mut self, task: &Task, d: &Dirent, ts: &InterTimeSpec) -> Result<()> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlaySetTimestamps(task, &overlay, d, ts);
        }

        let op = self.lock().InodeOp.clone();
        op.SetTimestamps(task, self, ts)?;
        return Ok(());
    }

    pub fn Truncate(&mut self, task: &Task, d: &Dirent, size: i64) -> Result<()> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayTruncate(task, &overlay, d, size);
        }

        let op = self.lock().InodeOp.clone();
        op.Truncate(task, self, size)?;
        return Ok(());
    }

    pub fn Allocate(&mut self, task: &Task, d: &Dirent, offset: i64, length: i64) -> Result<()> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayAllocate(task, &overlay, d, offset, length);
        }

        let op = self.lock().InodeOp.clone();
        op.Allocate(task, self, offset, length)?;
        return Ok(());
    }

    pub fn ReadLink(&self, task: &Task) -> Result<String> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayReadlink(task, &overlay);
        }

        let op = self.lock().InodeOp.clone();
        let res = op.ReadLink(task, self);
        return res;
    }

    pub fn GetLink(&self, task: &Task) -> Result<Dirent> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayGetlink(task, &overlay);
        }

        let op = self.lock().InodeOp.clone();
        let res = op.GetLink(task, self);
        return res;
    }

    pub fn AddLink(&self, task: &Task) {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            // This interface is only used by ramfs to update metadata of
            // children. These filesystems should _never_ have overlay
            // Inodes cached as children. So explicitly disallow this
            // scenario and avoid plumbing Dirents through to do copy up.
            panic!("overlay Inodes cached in ramfs directories are not supported")
        }

        let op = self.lock().InodeOp.clone();
        let res = op.AddLink(task);
        return res;
    }

    pub fn DropLink(&self, task: &Task) {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            // This interface is only used by ramfs to update metadata of
            // children. These filesystems should _never_ have overlay
            // Inodes cached as children. So explicitly disallow this
            // scenario and avoid plumbing Dirents through to do copy up.
            panic!("overlay Inodes cached in ramfs directories are not supported")
        }

        let op = self.lock().InodeOp.clone();
        let res = op.DropLink(task);
        return res;
    }

    pub fn StableAttr(&self) -> StableAttr {
        let overlay = self.lock().Overlay.clone();
        match overlay {
            None => return self.lock().StableAttr.clone(),
            Some(overlay) => {
                return overlayStableAttr(&overlay);
            }
        }
    }

    pub fn CheckPermission(&self, task: &Task, p: &PermMask) -> Result<()> {
        if p.write && self.lock().MountSource.lock().Flags.ReadOnly {
            return Err(Error::SysError(SysErr::EROFS));
        }

        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let mountSource = self.lock().MountSource.clone();
            if p.write && overlayUpperMountSource(&mountSource).lock().Flags.ReadOnly {
                return Err(Error::SysError(SysErr::EROFS));
            }
        }

        return self.check(task, p);
    }

    pub fn check(&self, task: &Task, p: &PermMask) -> Result<()> {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            let overlay = self.lock().Overlay.as_ref().unwrap().clone();
            return overlayCheck(task, &overlay, p);
        }

        let op = self.lock().InodeOp.clone();
        if !op.Check(task, self, p)? {
            return Err(Error::SysError(SysErr::EACCES));
        }

        return Ok(());
    }

    pub fn CheckOwnership(&mut self, task: &Task) -> bool {
        info!("CheckOwnership 1");
        let uattr = match self.UnstableAttr(task) {
            Err(_) => return false,
            Ok(u) => u,
        };

        let creds = task.creds.lock();
        info!(
            "CheckOwnership 2, uattr.Owner.UID is {:?}, cred is {:?}",
            uattr.Owner.UID, &creds
        );
        if uattr.Owner.UID == creds.EffectiveKUID {
            return true;
        }

        info!("CheckOwnership 3");
        if creds.HasCapability(Capability::CAP_FOWNER)
            && creds.UserNamespace.MapFromKUID(uattr.Owner.UID).Ok()
        {
            return true;
        }

        info!("CheckOwnership 4");
        return false;
    }

    pub fn IsVirtual(&self) -> bool {
        let isOverlay = self.lock().Overlay.is_some();
        if isOverlay {
            return false;
        }

        let iops = self.lock().InodeOp.clone();
        return iops.IsVirtual();
    }

    // hasChildren is a helper method that determines whether an arbitrary inode
    // (not necessarily ramfs) has any children.
    pub fn HasChildren(&self, task: &Task) -> (bool, Result<()>) {
        let d = Dirent::NewTransient(self);

        let file = match self.GetFile(
            task,
            &d,
            &FileFlags {
                Read: true,
                ..Default::default()
            },
        ) {
            Err(e) => return (false, Err(e)),
            Ok(f) => f,
        };

        let mut ser = CollectEntriesSerilizer::New();
        match file.ReadDir(task, &mut ser) {
            Err(e) => return (false, Err(e)),
            _ => (),
        }

        if ser.Written() > 2 {
            return (true, Ok(()));
        }

        return (false, Ok(()));
    }

    pub fn CheckCapability(self: &Self, task: &Task, cp: u64) -> bool {
        let uattr = match self.UnstableAttr(task) {
            Err(_) => return false,
            Ok(a) => a,
        };

        let creds = task.creds.clone();

        return CheckCapability(&creds, cp, &uattr);
    }

    pub fn StatFS(&self, task: &Task) -> Result<FsInfo> {
        let overlay = self.lock().Overlay.clone();
        let isOverlay = overlay.is_some();
        if isOverlay {
            let overlay = overlay.as_ref().unwrap().clone();
            return overlayStatFS(task, &overlay);
        }

        let inodeOp = self.lock().InodeOp.clone();
        return inodeOp.StatFS(task);
    }

    pub fn Watches(&self) -> Watches {
        return self.lock().watches.clone();
    }
}

//#[derive(Clone, Default, Debug, Copy)]
pub struct InodeIntern {
    pub UniqueId: u64,
    pub InodeOp: Arc<InodeOperations>,
    pub StableAttr: StableAttr,
    pub LockCtx: LockCtx,
    pub watches: Watches,
    pub MountSource: Arc<QMutex<MountSource>>,
    pub Overlay: Option<Arc<RwLock<OverlayEntry>>>,
}

impl Default for InodeIntern {
    fn default() -> Self {
        return Self {
            UniqueId: NewUID(),
            InodeOp: Arc::new(HostInodeOp::default()),
            StableAttr: Default::default(),
            LockCtx: LockCtx::default(),
            watches: Watches::default(),
            MountSource: Arc::new(QMutex::new(MountSource::default())),
            Overlay: None,
        };
    }
}

impl InodeIntern {
    pub fn New() -> Self {
        return Self {
            UniqueId: NewUID(),
            InodeOp: Arc::new(HostInodeOp::default()),
            StableAttr: Default::default(),
            LockCtx: LockCtx::default(),
            watches: Watches::default(),
            MountSource: Arc::new(QMutex::new(MountSource::default())),
            Overlay: None,
        };
    }

    pub fn StableAttr(&self) -> &StableAttr {
        return &self.StableAttr;
    }
}

pub fn CheckCapability(creds: &Credentials, cp: u64, uattr: &UnstableAttr) -> bool {
    let creds = creds.lock();
    if !creds.UserNamespace.MapFromKUID(uattr.Owner.UID).Ok() {
        return false;
    }

    if !creds.UserNamespace.MapFromKGID(uattr.Owner.GID).Ok() {
        return false;
    }

    return creds.HasCapability(cp);
}
