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
use alloc::string::ToString;
use alloc::collections::btree_map::BTreeMap;
use alloc::vec::Vec;
use spin::RwLock;
use ::qlib::mutex::*;
use core::any::Any;
use alloc::sync::Arc;
use core::ops::Deref;

use super::super::host::hostinodeop::*;
use super::super::fsutil::file::*;
use super::super::attr::*;
use super::super::flags::*;
use super::super::inode::*;
use super::super::dentry::*;
use super::super::file::*;
use super::super::dirent::*;
use super::super::mount::*;
use super::super::super::task::*;
use super::super::super::kernel::time::*;
use super::super::super::kernel::waiter::*;
use super::super::super::kernel::waiter::qlock::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::auth::*;
use super::super::super::uid::*;
use super::super::super::socket::unix::transport::unix::*;

#[derive(Clone, Default)]
pub struct CreateOps {
    pub NewDir: Option<fn(task: &Task, dir: &Inode, perms: &FilePermissions) -> Result<Inode>>,
    pub NewFile: Option<fn(task: &Task, dir: &Inode, perms: &FilePermissions) -> Result<Inode>>,
    pub NewSymlink: Option<fn(task: &Task, dir: &Inode, target: &str) -> Result<Inode>>,
    pub NewBoundEndpoint: Option<fn(task: &Task, dir: &Inode, ep: &BoundEndpoint, perms: &FilePermissions) -> Result<Inode>>,
    pub NewFifo: Option<fn(task: &Task, dir: &Inode, perms: &FilePermissions) -> Result<Inode>>,
}

pub struct DirInternal {
    pub CreateOps: CreateOps,
    pub children: BTreeMap<String, Inode>,
    pub dentryMap: DentMap,

    pub fsType: u64,
    pub unstable: UnstableAttr,

    pub xattrs: BTreeMap<String, String>
}

impl DirInternal {
    pub fn walk(&self, p: &str) -> Result<Inode> {
        match self.children.get(p) {
            None => Err(Error::SysError(SysErr::ENOENT)),
            Some(i) => Ok(i.clone())
        }
    }

    pub fn addChild(&mut self, task: &Task, name: &str, inode: &Inode) {
        let stableAttr = inode.lock().StableAttr().clone();

        self.children.insert(name.to_string(), inode.clone());
        self.dentryMap.Add(name, &DentAttr {
            Type: stableAttr.Type,
            InodeId: stableAttr.InodeId,
        });

        if stableAttr.IsDir() {
            self.addLink()
        }

        inode.AddLink(task);
    }

    pub fn removeChild(&mut self, task: &Task, name: &str) -> Result<Inode> {
        let inode = {
            let inode = match self.children.remove(name) {
                None => return Err(Error::SysError(SysErr::EACCES)),
                Some(i) => i,
            };

            self.dentryMap.Remove(name);

            inode
        };

        if inode.lock().StableAttr().IsDir() {
            self.dropLink();
        }

        inode.DropLink(task);

        return Ok(inode)
    }

    pub fn dropLink(&mut self) {
        self.unstable.Links -= 1;
    }

    pub fn addLink(&mut self) {
        self.unstable.Links += 1;
    }

    fn createInodeOperationsCommon(&mut self, task: &Task, name: &str, makeInodeOperations: &mut FnMut() -> Result<Inode>) -> Result<Inode> {
        if name.len() > NAME_MAX {
            return Err(Error::SysError(SysErr::ENAMETOOLONG))
        }

        let mut inode = makeInodeOperations()?;

        self.addChild(task, name, &mut inode);
        return Ok(inode)
    }
}

#[derive(Clone)]
pub struct Dir(pub Arc<RwLock<DirInternal>>);

impl Deref for Dir {
    type Target = Arc<RwLock<DirInternal>>;

    fn deref(&self) -> &Arc<RwLock<DirInternal>> {
        &self.0
    }
}

impl Dir {
    pub fn New(task: &Task, contents: BTreeMap<String, Inode>, owner: &FileOwner, perms: &FilePermissions) -> Self {
        let unstable = WithCurrentTime(task, &UnstableAttr {
            Owner: *owner,
            Perms: *perms,
            ..Default::default()
        });

        let mut dmap = BTreeMap::new();
        for (name, inode) in &contents {
            let stable = inode.lock().StableAttr().clone();
            dmap.insert(name.clone(), DentAttr {
                Type: stable.Type,
                InodeId: stable.InodeId,
            });
        }

        let d = DirInternal {
            CreateOps: CreateOps::default(),
            children: contents,
            dentryMap: DentMap::New(dmap),

            fsType: FSMagic::RAMFS_MAGIC,
            unstable: unstable,

            xattrs: BTreeMap::new(),
        };

        let ret = Dir(Arc::new(RwLock::new(d)));
        ret.AddLink(task);
        return ret;
    }

    pub fn AddChild(&self, task: &Task, name: &str, inode: &mut Inode) {
        let mut dir = self.write();

        dir.addChild(task, name, inode);
    }

    pub fn FindChild(&self, name: &str) -> Option<Inode> {
        match self.read().children.get(name) {
            None => None,
            Some(i) => Some(i.clone()),
        }
    }

    pub fn Children(&self) -> BTreeMap<String, DentAttr> {
        let d = self.read();

        let all = d.dentryMap.GetAll();
        let mut entries = BTreeMap::new();
        for (name, entry) in all {
            entries.insert(name.to_string(), *entry);
        }

        return entries
    }
}

impl InodeOperations for Dir {
    fn as_any(&self) -> &Any {
        return self
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::Dir;
    }

    fn InodeType(&self) -> InodeType {
        return InodeType::Directory;
    }

    fn InodeFileType(&self) -> InodeFileType{
        return InodeFileType::RamDir;
    }

    fn WouldBlock(&self) -> bool {
        return false;
    }

    fn Lookup(&self, _task: &Task, _dir: &Inode, name: &str) -> Result<Dirent> {
        if name.len() > NAME_MAX {
            return Err(Error::SysError(SysErr::ENAMETOOLONG))
        }

        let d = self.read();

        let inode = d.walk(name)?;

        return Ok(Dirent::New(&inode, name))
    }

    fn Create(&self, task: &Task, dir: &mut Inode, name: &str, flags: &FileFlags, perms: &FilePermissions) -> Result<File> {
        let mut d = self.write();

        let newFile = match d.CreateOps.NewFile {
            None => return Err(Error::SysError(SysErr::EACCES)),
            Some(newFile) => {
                newFile
            }
        };

        let inode = d.createInodeOperationsCommon(task, name, &mut || -> Result<Inode> {
            return newFile(task, dir, perms)
        })?;

        let created = Dirent::New(&inode, name);
        return inode.GetFile(task, &created, flags);
    }

    fn CreateDirectory(&self, task: &Task, dir: &mut Inode, name: &str, perms: &FilePermissions) -> Result<()> {
        let mut d = self.write();

        let newDir = match d.CreateOps.NewDir {
            None => return Err(Error::SysError(SysErr::EACCES)),
            Some(newDir) => {
                newDir
            }
        };

        let _inode = d.createInodeOperationsCommon(task, name, &mut || -> Result<Inode> {
            return newDir(task, dir, perms)
        })?;

        return Ok(())
    }

    fn Bind(&self, task: &Task, dir: &Inode, name: &str, ep: &BoundEndpoint, perms: &FilePermissions) -> Result<Dirent> {
        let mut d = self.write();

        let newep = match d.CreateOps.NewBoundEndpoint {
            None => return Err(Error::SysError(SysErr::EACCES)),
            Some(NewBoundEndpoint) => {
                NewBoundEndpoint
            }
        };

        let res = d.createInodeOperationsCommon(task, name, &mut || -> Result<Inode> {
            return newep(task, dir, ep, perms)
        });

        let inode = match res {
            Err(Error::SysError(SysErr::EEXIST)) => return Err(Error::SysError(SysErr::EADDRINUSE)),
            Err(e) => return Err(e),
            Ok(inode) => inode,
        };

        return Ok(Dirent::New(&inode, name));
    }

    fn CreateLink(&self, task: &Task, dir: &mut Inode, oldname: &str, newname: &str) -> Result<()> {
        let mut d = self.write();

        let newSymlink = match d.CreateOps.NewSymlink {
            None => return Err(Error::SysError(SysErr::EACCES)),
            Some(newSymlink) => {
                newSymlink
            }
        };

        let _ = d.createInodeOperationsCommon(task, newname, &mut || -> Result<Inode> {
            return newSymlink(task, dir, oldname)
        })?;

        return Ok(())
    }

    fn CreateHardLink(&self, task: &Task, _dir: &mut Inode, target: &Inode, name: &str) -> Result<()> {
        if name.len() > NAME_MAX {
            return Err(Error::SysError(SysErr::ENAMETOOLONG))
        }

        let mut d = self.write();

        d.addChild(task, name, target);

        return Ok(())
    }

    fn CreateFifo(&self, task: &Task, dir: &mut Inode, name: &str, perms: &FilePermissions) -> Result<()> {
        let mut d = self.write();

        let newFifo = match d.CreateOps.NewFifo {
            None => return Err(Error::SysError(SysErr::EACCES)),
            Some(newFifo) => {
                newFifo
            }
        };

        let _inode = d.createInodeOperationsCommon(task, name, &mut || -> Result<Inode> {
            return newFifo(task, dir, perms)
        })?;

        return Ok(())
    }

    //fn RemoveDirent(&mut self, dir: &mut InodeStruStru, remove: &Arc<QMutex<Dirent>>) -> Result<()> ;
    fn Remove(&self, task: &Task, _dir: &mut Inode, name: &str) -> Result<()> {
        if name.len() > NAME_MAX {
            return Err(Error::SysError(SysErr::ENAMETOOLONG))
        }

        let _ = self.write().removeChild(task, name)?;
        return Ok(())
    }

    fn RemoveDirectory(&self, task: &Task, _dir: &mut Inode, name: &str) -> Result<()> {
        if name.len() > NAME_MAX {
            return Err(Error::SysError(SysErr::ENAMETOOLONG))
        }

        let mut d = self.write();

        let childInode = d.walk(name)?;

        let (ok, err) = childInode.HasChildren(task);

        match err {
            Err(e) => return Err(e),
            _ => if ok {
                return Err(Error::SysError(SysErr::ENOTEMPTY))
            }
        }

        let _inode = d.removeChild(task, name)?;

        return Ok(())
    }

    fn Rename(&self, task: &Task, _dir: &mut Inode, oldParent: &Inode, oldname: &str, newParent: &Inode, newname: &str, replacement: bool) -> Result<()> {
        let oldParentInodeOp = oldParent.lock().InodeOp.clone();
        let newParentInodeOp = newParent.lock().InodeOp.clone();

        return Rename(task, oldParentInodeOp, oldname, newParentInodeOp, newname, replacement)
    }

    fn BoundEndpoint(&self, _task: &Task, _inode: &Inode, _path: &str) -> Option<BoundEndpoint> {
        return None
    }

    fn GetFile(&self, _task: &Task, _dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        let mut flags = flags;
        flags.Pread = true;

        let dirOps = DirFileOperation::New(self);
        let file = FileInternal {
            UniqueId: NewUID(),
            Dirent: dirent.clone(),
            flags: QMutex::new((flags, None)),
            offset: QLock::New(0),
            FileOp: Arc::new(dirOps),
        };

        return Ok(File(Arc::new(file)))
    }

    fn UnstableAttr(&self, _task: &Task, _dir: &Inode) -> Result<UnstableAttr> {
        let u = self.read().unstable;
        return Ok(u)
    }

    fn Getxattr(&self, _dir: &Inode, name: &str) -> Result<String> {
        match self.read().xattrs.get(name) {
            None => Err(Error::SysError(SysErr::ENOATTR)),
            Some(s) => Ok(s.clone())
        }
    }

    fn Setxattr(&self, _dir: &mut Inode, name: &str, value: &str) -> Result<()> {
        self.write().xattrs.insert(name.to_string(), value.to_string());
        return Ok(())
    }

    fn Listxattr(&self, _dir: &Inode) -> Result<Vec<String>> {
        let mut res = Vec::new();
        for (name, _) in &self.read().xattrs {
            res.push(name.clone());
        }

        return Ok(res)
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return ContextCanAccessFile(task, inode, reqPerms)
    }

    fn SetPermissions(&self, task: &Task, _dir: &mut Inode, p: FilePermissions) -> bool {
        self.write().unstable.SetPermissions(task, &p);
        return true;
    }

    fn SetOwner(&self, task: &Task, _dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        self.write().unstable.SetOwner(task, owner);
        return Ok(())
    }

    fn SetTimestamps(&self, task: &Task, _dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        self.write().unstable.SetTimestamps(task, ts);
        return Ok(())
    }

    fn Truncate(&self, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EISDIR))
    }

    fn Allocate(&self, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EISDIR))
    }

    fn ReadLink(&self, _task: &Task,_dir: &Inode) -> Result<String> {
        return Err(Error::SysError(SysErr::ENOLINK))
    }

    fn GetLink(&self, _task: &Task, _dir: &Inode) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOLINK))
    }

    fn AddLink(&self, _task: &Task) {
        self.write().addLink();
    }

    fn DropLink(&self, _task: &Task) {
        self.write().dropLink();
    }

    fn IsVirtual(&self) -> bool {
        return true
    }

    fn Sync(&self) -> Result<()> {
        return Ok(())
    }

    fn StatFS(&self, _task: &Task) -> Result<FsInfo> {
        return Err(Error::SysError(SysErr::ENOSYS))
    }

    fn Mappable(&self) -> Result<HostInodeOp> {
        return Err(Error::SysError(SysErr::ENODEV))
    }
}


pub struct DirFileOperation {
    pub dirCursor: QMutex<String>,
    pub dir: Dir,
}

impl DirFileOperation {
    pub fn New(dir: &Dir) -> Self {
        return Self {
            dirCursor: QMutex::new("".to_string()),
            dir: dir.clone(),
        }
    }
}

impl Waitable for DirFileOperation {}

impl SpliceOperations for DirFileOperation {}

impl FileOperations for DirFileOperation {
    fn as_any(&self) -> &Any {
        return self
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::DirFileOperation
    }

    fn Seekable(&self) -> bool {
        return true;
    }

    fn Seek(&self, task: &Task, f: &File, whence: i32, current: i64, offset: i64) -> Result<i64> {
        return SeekWithDirCursor(task, f, whence, current, offset, Some(&mut self.dirCursor.lock()))
    }

    fn ReadDir(&self, task: &Task, f: &File, offset: i64, serializer: &mut DentrySerializer) -> Result<i64> {
        let root = task.Root();

        let mut dirCtx = DirCtx::New(serializer, &mut self.dirCursor.lock());
        let res = DirentReadDir(task, &f.Dirent, self, &root, &mut dirCtx, offset);
        *self.dirCursor.lock() = dirCtx.DirCursor.to_string();
        return res;
    }

    fn ReadAt(&self, _task: &Task, _f: &File, _dsts: &mut [IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
        return Err(Error::SysError(SysErr::EISDIR))
    }

    fn WriteAt(&self, _task: &Task, _f: &File, _srcs: &[IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
        return Err(Error::SysError(SysErr::EISDIR))
    }

    fn Append(&self, task: &Task, f: &File, srcs: &[IoVec]) -> Result<(i64, i64)> {
        let n = self.WriteAt(task, f, srcs, 0, false)?;
        return Ok((n, 0))
    }

    fn Fsync(&self, _task: &Task, _f: &File, _start: i64, _end: i64, _syncType: SyncType) -> Result<()> {
        return Ok(())
    }

    fn Flush(&self, _task: &Task, _f: &File) -> Result<()> {
        return Ok(())
    }

    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        let inode = f.Dirent.Inode();
        return inode.UnstableAttr(task);

    }

    fn Ioctl(&self, _task: &Task, _f: &File, _fd: i32, _request: u64, _val: u64) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTTY))
    }

    fn IterateDir(&self, task: &Task, _d: &Dirent, dirCtx: &mut DirCtx, offset: i32) -> (i32, Result<i64>) {
        let d = self.dir.read();

        let n = match dirCtx.ReadDir(task, &d.dentryMap) {
            Err(e) => return (offset, Err(e)),
            Ok(n) => n,
        };

        return (offset + n as i32, Ok(n as i64))
    }

    fn Mappable(&self) -> Result<HostInodeOp> {
        return Err(Error::SysError(SysErr::ENODEV))
    }
}

impl SockOperations for DirFileOperation {}

pub fn Rename(task: &Task, oldParent: Arc<InodeOperations>, oldName: &str, newParent: Arc<InodeOperations>, newName: &str, replacement: bool) -> Result<()> {
    let op = match oldParent.as_any().downcast_ref::<Dir>() {
        None => return Err(Error::SysError(SysErr::EXDEV)),
        Some(d) => d.clone(),
    };

    let np = match newParent.as_any().downcast_ref::<Dir>() {
        None => return Err(Error::SysError(SysErr::EXDEV)),
        Some(d) => d.clone(),
    };

    if newName.len() > NAME_MAX {
        return Err(Error::SysError(SysErr::ENAMETOOLONG))
    }

    let mut npLocked = np.write();

    if replacement {
        let replaced = npLocked.children.get(newName).expect("Dirent claims rename is replacement, the newName is not find");

        if replaced.StableAttr().IsDir() {
            let (ok, err) = replaced.HasChildren(task);

            match err {
                Err(e) => return Err(e),
                _ => if ok {
                    return Err(Error::SysError(SysErr::ENOTEMPTY))
                }
            }

            npLocked.removeChild(task, newName)?;
        }
    }

    if !Arc::ptr_eq(&op, &np) {
        let mut opLocked = op.write();

        let mut n = opLocked.children.get(oldName).unwrap().clone();
        opLocked.removeChild(task, oldName).unwrap();
        npLocked.addChild(task, newName, &mut n);
    } else {
        let mut n = npLocked.children.get(oldName).unwrap().clone();
        npLocked.removeChild(task, oldName).unwrap();
        npLocked.addChild(task, newName, &mut n);
    }

    return Ok(())
}