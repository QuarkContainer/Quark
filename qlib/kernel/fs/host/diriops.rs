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
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::sync::Weak;
use alloc::vec::Vec;
use core::any::Any;
use core::ops::Deref;

use crate::qlib::kernel::fs::host::dirent::Dirent64;
use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::linux::time::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::device::*;
use super::super::super::util::cstring::*;
use super::super::super::guestfdnotifier::*;
use super::super::super::kernel::time::*;
pub use super::super::super::memmgr::vma::HostIopsMappable;
use super::super::super::kernel::waiter::queue::*;
use super::super::super::socket::unix::transport::unix::*;
use super::super::super::task::*;
use super::super::super::Kernel::HostSpace;
use super::super::super::IOURING;
use super::super::super::SHARESPACE;
use super::super::attr::*;
use super::super::dirent::*;
use super::super::dentry::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::inode::*;
use super::hostdirfops::*;
use super::util::*;
use super::*;


pub struct HostDirOpIntern {
    pub mops: Arc<QMutex<MountSourceOperations>>,
    //this should be SuperOperations
    pub HostFd: i32,
    pub sattr: StableAttr,
    pub queue: Queue,
    pub errorcode: i64,

    pub readdirCache: Option<DentMap>,
}

impl Default for HostDirOpIntern {
    fn default() -> Self {
        return Self {
            mops: Arc::new(QMutex::new(SimpleMountSourceOperations::default())),
            HostFd: -1,
            sattr: StableAttr::default(),
            queue: Queue::default(),
            errorcode: 0,
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
    pub fn New(
        mops: &Arc<QMutex<MountSourceOperations>>,
        fd: i32,
        fstat: &LibcStat,
    ) -> Self {
        let ret = Self {
            mops: mops.clone(),
            HostFd: fd,
            sattr: fstat.StableAttr(),
            queue: Queue::default(),
            errorcode: 0,
            readdirCache: None,
        };

        return ret;
    }

    pub fn ReadDirAll(&self, _task: &Task) -> Result<DentMap> {
        let fd = self.HostFd();

        let mut buf: [u8; 4096 * 4] = [0; 4096 * 4]; // 16KB in stack

        let deviceId = self.sattr.DeviceId;
        let mut entries = BTreeMap::new();
        let mut reset = true;
        loop {
            let res = HostSpace::ReadDir(fd, &mut buf, reset);
            if res < 0 {
                return Err(Error::SysError(-res as i32));
            }

            reset = false;

            if res == 0 {
                break;
            }

            let addr = &buf[0] as * const _ as u64;
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
                    InodeId: HOSTFILE_DEVICE.lock().Map(MultiDeviceKey {
                        Device: deviceId, //ft.deviceId,
                        Inode: inode,
                        SecondaryDevice: "".to_string(),
                        // todo: do we need this?
                        //SecondaryDevice: f.inodeOperations.fileState.key.SecondaryDevice,

                    }),
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
        if mask.Empty() {
            return Ok(());
        }

        if mask.UID || mask.GID {
            return Err(Error::SysError(SysErr::EPERM));
        }

        if mask.Perms {
            let ret = Fchmod(self.HostFd, attr.Perms.LinuxMode()) as i32;
            if ret < 0 {
                return Err(Error::SysError(-ret));
            }
        }

        if mask.Size {
            let ret = Ftruncate(self.HostFd, attr.Size) as i32;
            if ret < 0 {
                return Err(Error::SysError(-ret));
            }
        }

        if mask.AccessTime || mask.ModificationTime {
            let ts = InterTimeSpec {
                ATime: attr.AccessTime,
                ATimeOmit: !mask.AccessTime,
                MTime: attr.ModificationTime,
                MTimeOmit: !mask.ModificationTime,
                ..Default::default()
            };

            return SetTimestamps(self.HostFd, &ts);
        }

        return Ok(());
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
    pub fn New(
        mops: &Arc<QMutex<MountSourceOperations>>,
        fd: i32,
        fstat: &LibcStat,
    ) -> Self {
        let intern = Arc::new(QMutex::new(HostDirOpIntern::New(
            mops, fd, fstat
        )));

        let ret = Self(intern);
        SetWaitInfo(fd, ret.lock().queue.clone());
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

    pub fn SyncFileRange(&self, offset: i64, nbytes: i64, flags: u32) -> Result<()> {
        let fd = self.HostFd();

        let ret = HostSpace::SyncFileRange(fd, offset, nbytes, flags);
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

    pub fn Queue(&self) -> Queue {
        return self.lock().queue.clone();
    }

    pub fn GetHostFileOp(&self, _task: &Task) -> Arc<HostDirFops> {
        let hostDirOp = HostDirFops {
            DirOp: self.clone(),
            DirCursor: QMutex::new("".to_string()),
            //Buf: HostFileBuf::None,
        };
        return Arc::new(hostDirOp);
    }

    // return (st_size, st_blocks)
    pub fn Size(&self) -> Result<(i64, i64)> {
        return Err(Error::SysError(SysErr::ENODEV));
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

    fn Lookup(&self, _task: &Task, parent: &Inode, name: &str) -> Result<Dirent> {
        let (fd, writeable, fstat) = TryOpenAt(self.HostFd(), name)?;

        let ms = parent.lock().MountSource.clone();
        let inode = Inode::NewHostInode(&ms, fd, &fstat, writeable)?;

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

        let inode = Inode::NewHostInode(&mountSource, fd, &fstat, true)?;
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
        _target: &Inode,
        _name: &str,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::EPERM));
    }

    fn CreateFifo(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _name: &str,
        _perm: &FilePermissions,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::EPERM));
    }

    fn Remove(&self, _task: &Task, _dir: &mut Inode, name: &str) -> Result<()> {
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
        _task: &Task,
        _dir: &mut Inode,
        oldParent: &Inode,
        oldname: &str,
        newParent: &Inode,
        newname: &str,
        _replacement: bool,
    ) -> Result<()> {
        let oldParent = match oldParent
            .lock()
            .InodeOp
            .as_any()
            .downcast_ref::<HostDirOp>()
            {
                Some(p) => p.clone(),
                None => panic!("&InodeOp isn't a HostDirOp! {:?}", oldParent.InodeType()),
            };

        let newParent = match newParent
            .lock()
            .InodeOp
            .as_any()
            .downcast_ref::<HostDirOp>()
            {
                Some(p) => p.clone(),
                None => panic!("&InodeOp isn't a HostDirOp! {:?}", newParent.InodeType()),
            };

        let ret = RenameAt(oldParent.HostFd(), oldname, newParent.HostFd(), newname);

        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        oldParent.lock().readdirCache = None;
        newParent.lock().readdirCache = None;
        return Ok(());
    }

    fn Bind(
        &self,
        _task: &Task,
        _dir: &Inode,
        _name: &str,
        _data: &BoundEndpoint,
        _perms: &FilePermissions,
    ) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOTDIR));
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

        return Ok(File::NewHostFile(dirent, &flags, fops, wouldBlock));
    }

    fn UnstableAttr(&self, task: &Task) -> Result<UnstableAttr> {
        let uringStatx = SHARESPACE.config.read().UringStatx;

        // the statx uring call sometime become very slow. todo: root cause this.
        if !uringStatx {
            let mut s: LibcStat = Default::default();
            let hostfd = self.lock().HostFd;
            let ret = Fstat(hostfd, &mut s) as i32;
            if ret < 0 {
                return Err(Error::SysError(-ret as i32));
            }

            let mops = self.lock().mops.clone();
            return Ok(s.UnstableAttr(&mops));
        } else {
            let mut s: Statx = Default::default();
            let hostfd = self.lock().HostFd;

            let str = CString::New("");
            let ret = IOURING.Statx(
                task,
                hostfd,
                str.Ptr(),
                &mut s as *mut _ as u64,
                ATType::AT_EMPTY_PATH,
                StatxMask::STATX_BASIC_STATS,
            );

            if ret < 0 {
                return Err(Error::SysError(-ret as i32));
            }

            let mops = self.lock().mops.clone();
            return Ok(s.UnstableAttr(&mops));
        }
    }

    //fn StableAttr(&self) -> &StableAttr;
    fn Getxattr(&self, _dir: &Inode, name: &str, _size: usize) -> Result<Vec<u8>> {
        let str = CString::New(name);
        let val : &mut[u8; Xattr::XATTR_NAME_MAX]= &mut [0; Xattr::XATTR_NAME_MAX];
        let ret  = HostSpace::FGetXattr(self.FD(),
                                        str.Ptr(),
                                        &val[0] as * const _ as u64,
                                        val.len()) as i32;
        if ret < 0 {
            return Err(Error::SysError(-ret))
        };

        return Ok(val[0..ret as usize].to_vec())
    }

    fn Setxattr(&self, _dir: &mut Inode, name: &str, value: &[u8], flags: u32) -> Result<()> {
        let name = CString::New(name);
        let addr = if value.len() == 0 {
            0
        } else {
            &value[0] as * const _ as u64
        };

        let ret  = HostSpace::FSetXattr(self.FD(),
                                        name.Ptr(),
                                        addr,
                                        value.len(),
                                        flags) as i32;

        if ret < 0 {
            return Err(Error::SysError(-ret))
        };

        return Ok(())
    }

    fn Listxattr(&self, _dir: &Inode, _size: usize) -> Result<Vec<String>> {
        let val : &mut[u8; Xattr::XATTR_NAME_MAX]= &mut [0; Xattr::XATTR_NAME_MAX];
        let ret  = HostSpace::FListXattr(self.FD(),
                                         &val[0] as * const _ as u64,
                                         val.len()) as i32;
        if ret < 0 {
            return Err(Error::SysError(-ret))
        };

        let mut res = Vec::new();
        let mut cur = 0;
        for i in 0..ret as usize {
            if val[i] == 0 {
                let str = String::from_utf8(val[cur..i].to_vec()).map_err(|e| Error::Common(format!("Getxattr fail {}", e)))?;
                res.push(str);
                cur = i+1;
            }
        }

        return Ok(res)
    }

    fn Removexattr(&self, _dir: &Inode, name: &str) -> Result<()> {
        let name = CString::New(name);
        let ret  = HostSpace::FRemoveXattr(self.FD(),
                                           name.Ptr()) as i32;

        if ret < 0 {
            return Err(Error::SysError(-ret))
        };

        return Ok(())
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
        if ts.ATimeOmit && ts.MTimeOmit {
            return Ok(());
        }

        let mut sts: [Timespec; 2] = [Timespec::default(); 2];

        sts[0] = TimespecFromTimestamp(ts.ATime, ts.ATimeOmit, ts.ATimeSetSystemTime);
        sts[1] = TimespecFromTimestamp(ts.MTime, ts.MTimeOmit, ts.MTimeSetSystemTime);

        let ret = HostSpace::Futimens(self.HostFd(), &sts as *const _ as u64);
        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        return Ok(());
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
        let mut statfs = LibcStatfs::default();

        let fd = self.HostFd();
        let ret = HostSpace::Fstatfs(fd, &mut statfs as *mut _ as u64);
        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        let mut fsInfo = FsInfo::default();
        fsInfo.Type = statfs.Type;
        fsInfo.TotalBlocks = statfs.Blocks;
        fsInfo.FreeBlocks = statfs.BlocksFree;
        fsInfo.TotalFiles = statfs.Files;
        fsInfo.FreeFiles = statfs.FilesFree;

        return Ok(fsInfo);
    }

    fn Mappable(&self) -> Result<HostIopsMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}
