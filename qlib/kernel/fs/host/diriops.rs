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
use crate::GUEST_HOST_SHARED_ALLOCATOR;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::sync::Weak;
use alloc::vec::Vec;
use core::any::Any;
use core::ops::Deref;

use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use crate::qlib::kernel::fs::host::dirent::Dirent64;
//use super::super::super::super::device::*;
use super::super::super::kernel::time::*;
pub use super::super::super::memmgr::vma::MMappable;
use super::super::super::socket::unix::transport::unix::*;
use super::super::super::socket::unix::unix::*;
use super::super::super::task::*;
use super::super::super::util::cstring::*;
use super::super::super::Kernel::HostSpace;
use super::super::super::SHARESPACE;
use super::super::attr::*;
use super::super::dentry::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::inode::*;
use super::hostdirfops::*;
use super::util::*;
use super::*;

pub const MAX_FILENAME_LEN: usize = (1 << 16) - 1;

pub struct HostDirOpIntern {
    pub mops: Arc<QMutex<MountSourceOperations>>,
    //this should be SuperOperations
    pub HostFd: i32,
    pub sattr: StableAttr,
    pub errorcode: i64,

    pub overrides: BTreeMap<String, Inode>,
    pub readdirCache: Option<DentMap>,
}

impl Default for HostDirOpIntern {
    fn default() -> Self {
        return Self {
            mops: Arc::new(QMutex::new(SimpleMountSourceOperations::default())),
            HostFd: -1,
            sattr: StableAttr::default(),
            errorcode: 0,
            overrides: BTreeMap::new(),
            readdirCache: None,
        };
    }
}

impl Drop for HostDirOpIntern {
    fn drop(&mut self) {
        if self.HostFd == -1 {
            //default fd
            return;
        }

        HostSpace::Close(self.HostFd);
    }
}

impl HostDirOpIntern {
    pub fn New(mops: &Arc<QMutex<MountSourceOperations>>, fd: i32, fstat: &LibcStat) -> Self {
        return Self {
            mops: mops.clone(),
            HostFd: fd,
            sattr: fstat.StableAttr(),
            errorcode: 0,
            readdirCache: None,
            overrides: BTreeMap::new(),
        };
    }

    pub fn ReadDirAll(&self, _task: &Task) -> Result<DentMap> {
        let fd = self.HostFd();

        let mut buf = Box::new_in([0u8; 4096 * 4], GUEST_HOST_SHARED_ALLOCATOR);// 16KB in shared heap
        //let deviceId = self.sattr.DeviceId;
        let mut entries = BTreeMap::new();
        let mut reset = true;
        loop {
            let res = HostSpace::ReadDir(fd, &mut *buf, reset);
            if res < 0 {
                return Err(Error::SysError(-res as i32));
            }

            reset = false;

            if res == 0 {
                break;
            }

            let addr = &buf[0] as *const _ as u64;
            let cnt: u64 = res as u64;
            let mut pos: u64 = 0;
            while pos < cnt {
                let name;
                let dType;
                let inode;
                unsafe {
                    let d: *const Dirent64 = (addr + pos) as *const Dirent64;
                    name = (*d).name;
                    dType = (*d).type_;
                    inode = (*d).ino;
                    pos += (*d).reclen as u64;
                }

                // Construct the key to find the virtual inode.
                // Directory entries reside on the same Device
                // and SecondaryDevice as their parent.
                let dentry = DentAttr {
                    Type: InodeType(DType::ModeType(dType) as u32),
                    InodeId: inode,
                    //InodeId: HOSTFILE_DEVICE.lock().Map(MultiDeviceKey::New(deviceId, "".to_string(), inode)),
                };

                let pathname = CString::FromAddr(&name[0] as *const _ as u64);
                entries.insert(pathname.Str().unwrap().to_string(), dentry);
            }
        }

        return Ok(DentMap::New(entries));
    }

    pub fn IterateDir(
        &mut self,
        task: &Task,
        dirCtx: &mut DirCtx,
        offset: i32,
    ) -> (i32, Result<i64>) {
        if SHARESPACE.config.read().ReaddirCache {
            if self.readdirCache.is_none() {
                let dentryMap = match self.ReadDirAll(task) {
                    Err(e) => return (offset, Err(e)),
                    Ok(entires) => entires,
                };

                self.readdirCache = Some(dentryMap);
            }

            return match dirCtx.ReadDir(task, self.readdirCache.as_ref().unwrap()) {
                Err(e) => (offset, Err(e)),
                Ok(count) => (offset + count as i32, Ok(0)),
            };
        } else {
            let dentryMap = match self.ReadDirAll(task) {
                Err(e) => return (offset, Err(e)),
                Ok(entires) => entires,
            };

            return match dirCtx.ReadDir(task, &dentryMap) {
                Err(e) => (offset, Err(e)),
                Ok(count) => (offset + count as i32, Ok(0)),
            };
        }
    }

    pub fn SetMaskedAttributes(&self, mask: &AttrMask, attr: &UnstableAttr) -> Result<()> {
        return SetMaskedAttributes(self.HostFd, mask, attr);
    }

    pub fn Sync(&self) -> Result<()> {
        let ret = Fsync(self.HostFd);
        if ret < 0 {
            return Err(Error::SysError(-ret));
        }

        return Ok(());
    }

    pub fn HostFd(&self) -> i32 {
        return self.HostFd;
    }

    pub fn StableAttr(&self) -> StableAttr {
        return self.sattr;
    }

    pub fn InodeType(&self) -> InodeType {
        return self.sattr.Type;
    }
}

#[derive(Clone)]
pub struct HostDirOpWeak(pub Weak<QMutex<HostDirOpIntern>>);

impl HostDirOpWeak {
    pub fn Upgrade(&self) -> Option<HostDirOp> {
        let f = match self.0.upgrade() {
            None => return None,
            Some(f) => f,
        };

        return Some(HostDirOp(f));
    }
}

#[derive(Clone)]
pub struct HostDirOp(pub Arc<QMutex<HostDirOpIntern>>);

impl PartialEq for HostDirOp {
    fn eq(&self, other: &Self) -> bool {
        return Arc::ptr_eq(&self.0, &other.0);
    }
}

impl Eq for HostDirOp {}

impl Default for HostDirOp {
    fn default() -> Self {
        return Self(Arc::new(QMutex::new(HostDirOpIntern::default())));
    }
}

impl Deref for HostDirOp {
    type Target = Arc<QMutex<HostDirOpIntern>>;

    fn deref(&self) -> &Arc<QMutex<HostDirOpIntern>> {
        &self.0
    }
}

impl HostDirOp {
    pub fn New(mops: &Arc<QMutex<MountSourceOperations>>, fd: i32, fstat: &LibcStat) -> Self {
        let intern = Arc::new(QMutex::new(HostDirOpIntern::New(mops, fd, fstat)));

        let ret = Self(intern);
        return ret;
    }

    pub fn SyncFs(&self) -> Result<()> {
        let fd = self.HostFd();

        let ret = HostSpace::SyncFs(fd);
        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        return Ok(());
    }

    pub fn Downgrade(&self) -> HostDirOpWeak {
        return HostDirOpWeak(Arc::downgrade(&self.0));
    }

    pub fn HostFd(&self) -> i32 {
        return self.lock().HostFd;
    }

    pub fn StableAttr(&self) -> StableAttr {
        return self.lock().sattr;
    }

    pub fn GetHostFileOp(&self, _task: &Task) -> HostDirFops {
        let hostDirOp = HostDirFops {
            DirOp: self.clone(),
            DirCursor: Arc::new(QMutex::new("".to_string())),
            //Buf: HostFileBuf::None,
        };
        return hostDirOp;
    }

    pub fn Fsync(
        &self,
        _task: &Task,
        _f: &File,
        _start: i64,
        _end: i64,
        syncType: SyncType,
    ) -> Result<()> {
        let fd = self.HostFd();
        let datasync = if syncType == SyncType::SyncData {
            true
        } else {
            false
        };

        let ret = if datasync {
            HostSpace::FDataSync(fd)
        } else {
            HostSpace::FSync(fd)
        };

        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        return Ok(());
    }

    pub fn FD(&self) -> i32 {
        let ret = self.lock().HostFd;
        return ret;
    }
}

impl InodeOperations for HostDirOp {
    fn as_any(&self) -> &Any {
        self
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::HostDirOp;
    }

    fn InodeType(&self) -> InodeType {
        return self.lock().sattr.Type;
    }

    fn InodeFileType(&self) -> InodeFileType {
        return InodeFileType::HostDir;
    }

    fn WouldBlock(&self) -> bool {
        return false;
    }

    fn Lookup(&self, task: &Task, parent: &Inode, name: &str) -> Result<Dirent> {
        let skiprw = true;
        let (fd, writeable, fstat) = match TryOpenAt(self.HostFd(), name, skiprw) {
            Err(Error::SysError(SysErr::ENOENT)) => {
                let inode = match self.lock().overrides.get(name) {
                    None => return Err(Error::SysError(SysErr::ENOENT)),
                    Some(i) => i.clone(),
                };

                return Ok(Dirent::New(&inode, name));
            }
            Err(e) => return Err(e),
            Ok(d) => d,
        };

        let ms = parent.lock().MountSource.clone();
        let inode = Inode::NewHostInode(task, &ms, fd, &fstat, writeable, skiprw, false)?;
        
        let ret = Ok(Dirent::New(&inode, name));
        return ret;
    }

    fn Create(
        &self,
        task: &Task,
        dir: &mut Inode,
        name: &str,
        flags: &FileFlags,
        perm: &FilePermissions,
    ) -> Result<File> {
        //let fd = openAt(self.HostFd(), name, (LibcConst::O_RDWR | LibcConst::O_CREAT | LibcConst::O_EXCL) as i32, perm.LinuxMode());

        let owner = task.FileOwner();

        let mut newFlags = *flags;

        // the fd might be use for other read/write operations todo: handle this more elegant
        newFlags.Read = true;
        newFlags.Write = true;

        let (fd, fstat) = createAt(
            self.HostFd(),
            name,
            newFlags.ToLinux() | LibcConst::O_CREAT as i32,
            perm.LinuxMode(),
            owner.UID.0,
            owner.GID.0,
        )?;

        self.lock().readdirCache = None;

        let mountSource = dir.lock().MountSource.clone();

        let inode = Inode::NewHostInode(task, &mountSource, fd, &fstat, true, false, false)?;
        let dirent = Dirent::New(&inode, name);

        let file = inode.GetFile(task, &dirent, flags)?;
        return Ok(file);
    }

    fn CreateDirectory(
        &self,
        task: &Task,
        _dir: &mut Inode,
        name: &str,
        perm: &FilePermissions,
    ) -> Result<()> {
        let owner = task.FileOwner();

        let ret = Mkdirat(
            self.HostFd(),
            name,
            perm.LinuxMode(),
            owner.UID.0,
            owner.GID.0,
        );
        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        self.lock().readdirCache = None;

        return Ok(());
    }

    fn CreateLink(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        oldname: &str,
        newname: &str,
    ) -> Result<()> {
        let ret = SymLinkAt(oldname, self.HostFd(), newname);

        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        self.lock().readdirCache = None;
        return Ok(());
    }

    fn CreateHardLink(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        target: &Inode,
        name: &str,
    ) -> Result<()> {
        let iops = match target.lock().InodeOp.HostInodeOp() {
            Some(p) => p.clone(),
            None => return Err(Error::SysError(SysErr::EPERM)),
        };

        let ret = LinkAt(
            iops.HostFd(),
            "",
            self.HostFd(),
            name,
            ATType::AT_EMPTY_PATH,
        );

        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        self.lock().readdirCache = None;
        return Ok(());
    }

    fn CreateFifo(
        &self,
        task: &Task,
        _dir: &mut Inode,
        name: &str,
        perm: &FilePermissions,
    ) -> Result<()> {
        let owner = task.FileOwner();

        let ret = Mkfifoat(
            self.HostFd(),
            name,
            perm.LinuxMode(),
            owner.UID.0,
            owner.GID.0,
        );
        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        self.lock().readdirCache = None;

        return Ok(());
    }

    fn Remove(&self, _task: &Task, _dir: &mut Inode, name: &str) -> Result<()> {
        match self.lock().overrides.remove(name) {
            None => (),
            Some(_) => return Ok(()),
        }

        let flags = 0; //ATType::AT_REMOVEDIR

        let ret = UnLinkAt(self.HostFd(), name, flags);

        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        self.lock().readdirCache = None;
        return Ok(());
    }

    fn RemoveDirectory(&self, _task: &Task, _dir: &mut Inode, name: &str) -> Result<()> {
        let flags = ATType::AT_REMOVEDIR;

        let ret = UnLinkAt(self.HostFd(), name, flags);

        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        self.lock().readdirCache = None;
        return Ok(());
    }

    fn Rename(
        &self,
        task: &Task,
        dir: &mut Inode,
        oldParent: &Inode,
        oldname: &str,
        newParent: &Inode,
        newname: &str,
        replacement: bool,
    ) -> Result<()> {
        return Rename(
            task,
            dir,
            oldParent,
            oldname,
            newParent,
            newname,
            replacement,
        );
    }

    fn Bind(
        &self,
        task: &Task,
        dir: &Inode,
        name: &str,
        ep: &BoundEndpoint,
        perms: &FilePermissions,
    ) -> Result<Dirent> {
        if name.len() > MAX_FILENAME_LEN {
            return Err(Error::SysError(SysErr::ENAMETOOLONG));
        }

        let msrc = dir.lock().MountSource.clone();
        let inode = NewUnixSocketInode(task, ep, &task.FileOwner(), perms, &msrc);
        let dirent = Dirent::New(&inode, name);
        self.lock().overrides.insert(String::from(name), inode);

        return Ok(dirent);
    }

    fn BoundEndpoint(&self, _task: &Task, _inode: &Inode, _path: &str) -> Option<BoundEndpoint> {
        return None;
    }

    fn GetFile(
        &self,
        task: &Task,
        _dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        let fops = self.GetHostFileOp(task);

        let inode = dirent.Inode();
        let wouldBlock = inode.lock().InodeOp.WouldBlock();

        return Ok(File::NewHostFile(dirent, &flags, fops.into(), wouldBlock));
    }

    fn UnstableAttr(&self, task: &Task) -> Result<UnstableAttr> {
        let mops = self.lock().mops.clone();
        let fd = self.HostFd();

        return UnstableAttr(fd, task, &mops);
    }

    //fn StableAttr(&self) -> &StableAttr;
    fn Getxattr(&self, _dir: &Inode, name: &str, _size: usize) -> Result<Vec<u8>> {
        return Getxattr(self.HostFd(), name);
    }

    fn Setxattr(&self, _dir: &mut Inode, name: &str, value: &[u8], flags: u32) -> Result<()> {
        return Setxattr(self.HostFd(), name, value, flags);
    }

    fn Listxattr(&self, _dir: &Inode, _size: usize) -> Result<Vec<String>> {
        return Listxattr(self.HostFd());
    }

    fn Removexattr(&self, _dir: &Inode, name: &str) -> Result<()> {
        return Removexattr(self.HostFd(), name);
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return ContextCanAccessFile(task, inode, reqPerms);
    }

    fn SetPermissions(&self, _task: &Task, _dir: &mut Inode, f: FilePermissions) -> bool {
        return Fchmod(self.HostFd(), f.LinuxMode()) == 0;
    }

    fn SetOwner(&self, _task: &Task, _dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        let ret = FChown(self.HostFd(), owner.UID.0, owner.GID.0);

        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        };

        return Ok(());
    }

    fn SetTimestamps(&self, _task: &Task, _dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        return SetTimestamps(self.HostFd(), ts);
    }

    fn Truncate(&self, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EISDIR));
    }

    fn Allocate(&self, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EISDIR));
    }

    fn ReadLink(&self, _task: &Task, _dir: &Inode) -> Result<String> {
        return ReadLinkAt(self.HostFd(), &"".to_string());
    }

    fn GetLink(&self, _task: &Task, dir: &Inode) -> Result<Dirent> {
        if !dir.StableAttr().IsSymlink() {
            return Err(Error::SysError(SysErr::ENOLINK));
        }

        return Err(Error::ErrResolveViaReadlink);
    }

    fn AddLink(&self, _task: &Task) {
        //return Err(Error::None)
    }

    fn DropLink(&self, _task: &Task) {
        //return Err(Error::None)
    }

    fn IsVirtual(&self) -> bool {
        false
    }

    fn Sync(&self) -> Result<()> {
        return self.lock().Sync();
    }

    fn StatFS(&self, _task: &Task) -> Result<FsInfo> {
        return StatFS(self.HostFd());
    }

    fn Mappable(&self) -> Result<MMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}
