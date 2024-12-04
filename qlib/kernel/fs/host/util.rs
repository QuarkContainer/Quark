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
use alloc::sync::Arc;
use alloc::vec::Vec;

use super::super::super::super::auth::id::*;
use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::device::*;
use super::super::super::super::linux::time::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::qmsg::qcall::TryOpenStruct;
use super::super::super::kernel::time::*;
use super::super::super::task::*;
use super::super::super::util::sharedcstring::*;
use crate::GUEST_HOST_SHARED_ALLOCATOR;
use Box;

use super::super::super::super::path;
use super::super::super::Kernel::HostSpace;
use super::super::super::IOURING;
use super::super::super::SHARESPACE;
use super::super::attr::*;
use super::diriops::*;
use super::*;

impl Statx {
    pub fn InodeType(&self) -> InodeType {
        let x = self.stx_mode as u16 & ModeType::S_IFMT;

        match x {
            ModeType::S_IFLNK => InodeType::Symlink,
            ModeType::S_IFIFO => InodeType::Pipe,
            ModeType::S_IFCHR => InodeType::CharacterDevice,
            ModeType::S_IFBLK => InodeType::BlockDevice,
            ModeType::S_IFSOCK => InodeType::Socket,
            ModeType::S_IFDIR => InodeType::Directory,
            ModeType::S_IFREG => InodeType::RegularFile,
            _ => {
                info!("unknow host file type {}: assuming regular", x);
                return InodeType::RegularFile;
            }
        }
    }

    pub fn WouldBlock(&self) -> bool {
        let iType = self.InodeType();
        return iType == InodeType::Pipe
            || iType == InodeType::Socket
            || iType == InodeType::CharacterDevice;
    }

    pub fn StableAttr(&self) -> StableAttr {
        let devId = ID {
            Major: self.stx_dev_major as u16,
            Minor: self.stx_dev_minor,
        };

        let deviceId = devId.DeviceID();
        let InodeId = self.stx_ino;

        /*let deviceId = HOSTFILE_DEVICE.lock().DeviceID();
        let InodeId =  HOSTFILE_DEVICE.lock().Map(MultiDeviceKey::New(
            devId.DeviceID(),
            "".to_string(),
            self.stx_ino,
        ));*/

        let major = self.stx_rdev_major;
        let minor = self.stx_rdev_minor;

        return StableAttr {
            Type: self.InodeType(),
            DeviceId: deviceId,
            InodeId: InodeId,
            BlockSize: self.stx_blksize as i64,
            DeviceFileMajor: major as u16,
            DeviceFileMinor: minor,
        };
    }

    pub fn Owner(&self, mo: Arc<QMutex<MountSourceOperations>>) -> FileOwner {
        //todo: info!("we don't handle dontTranslateOwnership, fix it");
        //let mut dontTranslateOwnership = mo.lock().as_any().downcast_ref::<SuperOperations>().expect("Owner: not SuperOperations").dontTranslateOwnership;
        let mounter = mo
            .lock()
            .as_any()
            .downcast_ref::<SuperOperations>()
            .expect("Owner: not SuperOperations")
            .mounter
            .clone();

        let dontTranslateOwnership = true;
        if dontTranslateOwnership {
            return FileOwner {
                UID: KUID(self.stx_uid),
                GID: KGID(self.stx_gid),
            };
        }

        let mut owner = FileOwner {
            UID: OVERFLOW_KUID,
            GID: OVERFLOW_KGID,
        };

        if self.stx_uid == HOST_AUTH_ID.lock().uid {
            owner.UID = mounter.UID;
        }

        for gid in &HOST_AUTH_ID.lock().gids {
            if self.stx_gid == *gid {
                owner.GID = mounter.GID;
                break;
            }
        }

        return owner;
    }

    pub fn UnstableAttr(&self, mo: &Arc<QMutex<MountSourceOperations>>) -> UnstableAttr {
        return UnstableAttr {
            Size: self.stx_size as i64,
            Usage: self.stx_blocks as i64 * 512,
            Perms: FileMode(self.stx_mode as u16).FilePerms(),
            Owner: self.Owner(mo.clone()),
            AccessTime: Time::FromStatxTimestamp(&self.stx_atime),
            ModificationTime: Time::FromStatxTimestamp(&self.stx_mtime),
            StatusChangeTime: Time::FromStatxTimestamp(&self.stx_ctime),
            Links: self.stx_nlink as u64,
        };
    }
}

pub fn InodeType(st_mode: u32) -> InodeType {
    let x = st_mode as u16 & ModeType::S_IFMT;

    match x {
        ModeType::S_IFLNK => InodeType::Symlink,
        ModeType::S_IFIFO => InodeType::Pipe,
        ModeType::S_IFCHR => InodeType::CharacterDevice,
        ModeType::S_IFBLK => InodeType::BlockDevice,
        ModeType::S_IFSOCK => InodeType::Socket,
        ModeType::S_IFDIR => InodeType::Directory,
        ModeType::S_IFREG => InodeType::RegularFile,
        _ => {
            info!("unknow host file type {}: assuming regular", x);
            return InodeType::RegularFile;
        }
    }
}

impl LibcStat {
    pub fn InodeType(&self) -> InodeType {
        return InodeType(self.st_mode);
    }

    pub fn WouldBlock(&self) -> bool {
        let iType = self.InodeType();
        return iType == InodeType::Pipe
            || iType == InodeType::Socket
            || iType == InodeType::CharacterDevice;
    }

    pub fn StableAttr(&self) -> StableAttr {
        let deviceId = self.st_dev;
        let InodeId = self.st_ino;

        /*let deviceId = HOSTFILE_DEVICE.lock().DeviceID();
        let InodeId = HOSTFILE_DEVICE.lock().Map(MultiDeviceKey::New(
            self.st_dev,
            "".to_string(),
            self.st_ino,
        ));*/

        let (major, minor) = DecodeDeviceId(self.st_rdev as u32);
        #[cfg(target_arch = "aarch64")]
        let st_blksize = self.st_blksize as i64;
        #[cfg(target_arch = "x86_64")]
        let st_blksize = self.st_blksize;
        return StableAttr {
            Type: self.InodeType(),
            DeviceId: deviceId,
            InodeId: InodeId,
            BlockSize: st_blksize,
            DeviceFileMajor: major,
            DeviceFileMinor: minor,
        };
    }

    pub fn Owner(&self, mo: Arc<QMutex<MountSourceOperations>>) -> FileOwner {
        //todo: info!("we don't handle dontTranslateOwnership, fix it");
        //let mut dontTranslateOwnership = mo.lock().as_any().downcast_ref::<SuperOperations>().expect("Owner: not SuperOperations").dontTranslateOwnership;
        let mounter = mo
            .lock()
            .as_any()
            .downcast_ref::<SuperOperations>()
            .expect("Owner: not SuperOperations")
            .mounter
            .clone();

        let dontTranslateOwnership = true;
        if dontTranslateOwnership {
            return FileOwner {
                UID: KUID(self.st_uid),
                GID: KGID(self.st_gid),
            };
        }

        let mut owner = FileOwner {
            UID: OVERFLOW_KUID,
            GID: OVERFLOW_KGID,
        };

        if self.st_uid == HOST_AUTH_ID.lock().uid {
            owner.UID = mounter.UID;
        }

        for gid in &HOST_AUTH_ID.lock().gids {
            if self.st_gid == *gid {
                owner.GID = mounter.GID;
                break;
            }
        }

        return owner;
    }

    pub fn UnstableAttr(&self, mo: &Arc<QMutex<MountSourceOperations>>) -> UnstableAttr {
        #[cfg(target_arch = "aarch64")]
        let st_nlink = self.st_nlink as u64;
        #[cfg(target_arch = "x86_64")]
        let st_nlink = self.st_nlink;
        return UnstableAttr {
            Size: self.st_size,
            Usage: self.st_blocks * 512,
            Perms: FileMode(self.st_mode as u16).FilePerms(),
            Owner: self.Owner(mo.clone()),
            AccessTime: Time::FromUnix(self.st_atime, self.st_atime_nsec),
            ModificationTime: Time::FromUnix(self.st_mtime, self.st_mtime_nsec),
            StatusChangeTime: Time::FromUnix(self.st_ctime, self.st_ctime_nsec),
            Links: st_nlink,
        };
    }
}

pub fn TimespecFromTimestamp(t: Time, omit: bool, setSysTime: bool) -> Timespec {
    if omit {
        return Timespec {
            tv_sec: 0,
            tv_nsec: Utime::UTIME_OMIT,
        };
    }

    if setSysTime {
        return Timespec {
            tv_sec: 0,
            tv_nsec: Utime::UTIME_NOW,
        };
    }

    return Timespec::FromNs(t.0);
}

//if dirfd ==-100, there is no parent
//return (fd, writeable)
pub fn TryOpenAt(dirfd: i32, name: &str, skiprw: bool) -> Result<(i32, bool, LibcStat)> {
    if dirfd == -100 && !path::IsAbs(name) {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let name = path::Clean(name);
    let fstat = LibcStat::default();
    let mut tryopen = TryOpenStruct {
        fstat: &fstat,
        writeable: false,
    };
    let cstr = SharedCString::New(&name);

    let ret = HostSpace::TryOpenAt(dirfd, cstr.Ptr(), &mut tryopen as *mut TryOpenStruct as u64, skiprw);

    if ret < 0 {
        return Err(Error::SysError(-ret as i32));
    }

    return Ok((ret as i32, tryopen.writeable, fstat));
}

pub fn OpenAt(dirfd: i32, name: &str, flags: i32) -> Result<(i32, LibcStat)> {
    if dirfd == -100 && !path::IsAbs(name) {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let name = path::Clean(name);
    let fstat = LibcStat::default();
    let mut tryopen = TryOpenStruct {
        fstat: &fstat,
        writeable: false,
    };
    let cstr = SharedCString::New(&name);

    let ret = HostSpace::OpenAt(
        dirfd,
        cstr.Ptr(),
        flags,
        &mut tryopen as *mut TryOpenStruct as u64,
    );

    if ret < 0 {
        return Err(Error::SysError(-ret as i32));
    }

    return Ok((ret as i32, fstat));
}

pub fn Fstat(fd: i32, fstat: &mut LibcStat) -> i64 {
    return HostSpace::Fstat(fd, fstat as *mut _ as u64);
}

pub fn Fstatat(dirfd: i32, pathname: &str, fstat: &mut LibcStat, flags: i32) -> i64 {
    let cstr = SharedCString::New(pathname);
    return HostSpace::Fstatat(dirfd, cstr.Ptr(), fstat as *mut _ as u64, flags);
}

pub fn Fcntl(fd: i32, cmd: i32, arg: u64) -> i64 {
    return HostSpace::Fcntl(fd, cmd, arg);
}

pub fn Mkdirat(fd: i32, name: &str, perm: u32, uid: u32, gid: u32) -> i64 {
    let cstr = SharedCString::New(name);
    let res = HostSpace::Mkdirat(fd, cstr.Ptr(), perm, uid, gid);
    return res;
}

pub fn Mkfifoat(fd: i32, name: &str, perm: u32, uid: u32, gid: u32) -> i64 {
    let cstr = SharedCString::New(name);
    let res = HostSpace::Mkfifoat(fd, cstr.Ptr(), perm, uid, gid);
    return res;
}

pub fn LinkAt(olddirfd: i32, oldpath: &str, newdirfd: i32, newpath: &str, flags: i32) -> i64 {
    let oldpath = SharedCString::New(oldpath);
    let newpath = SharedCString::New(newpath);

    return HostSpace::LinkAt(olddirfd, oldpath.Ptr(), newdirfd, newpath.Ptr(), flags);
}

pub fn SymLinkAt(oldpath: &str, newdirfd: i32, newpath: &str) -> i64 {
    let oldpath = SharedCString::New(oldpath);
    let newpath = SharedCString::New(newpath);

    return HostSpace::SymLinkAt(oldpath.Ptr(), newdirfd, newpath.Ptr());
}

pub fn UnLinkAt(dirfd: i32, pathname: &str, flags: i32) -> i64 {
    let cstr = SharedCString::New(pathname);
    return HostSpace::Unlinkat(dirfd, cstr.Ptr(), flags);
}

pub fn RenameAt(olddirfd: i32, oldpath: &str, newdirfd: i32, newpath: &str) -> i64 {
    let oldpath = SharedCString::New(oldpath);
    let newpath = SharedCString::New(newpath);

    return HostSpace::RenameAt(olddirfd, oldpath.Ptr(), newdirfd, newpath.Ptr());
}

pub fn Fchmod(fd: i32, mode: u32) -> i64 {
    return HostSpace::Fchmod(fd, mode);
}

pub fn FChown(fd: i32, owner: u32, group: u32) -> i64 {
    return HostSpace::FChown(fd, owner, group);
}

pub fn Ftruncate(fd: i32, size: i64) -> i64 {
    return HostSpace::Ftruncate(fd, size);
}

pub fn Fallocate(fd: i32, mode: i32, offset: i64, len: i64) -> i64 {
    return HostSpace::Fallocate(fd, mode, offset, len);
}

pub fn ReadLinkAt(dirfd: i32, path: &str) -> Result<String> {
    let mut buf: [u8; 1024] = [0; 1024];
    let cstr = SharedCString::New(path);

    let ret = HostSpace::ReadLinkAt(dirfd, cstr.Ptr(), &mut buf[0] as *mut _ as u64, 1024);

    if ret < 0 {
        return Err(Error::SysError(-ret as i32));
    }

    assert!(ret < 1024, "ReadLinkAt has no enough buffer");
    return Ok(String::from_utf8(buf[..ret as usize].to_vec()).unwrap());
}

pub fn createAt(
    dirfd: i32,
    name: &str,
    flags: i32,
    perm: u32,
    uid: u32,
    gid: u32,
) -> Result<(i32, LibcStat)> {
    let cstr = SharedCString::New(name);
    let mut fstat = LibcStat::default();

    let ret = HostSpace::CreateAt(
        dirfd,
        cstr.Ptr(),
        flags,
        perm as i32,
        uid,
        gid,
        &mut fstat as *mut _ as u64,
    ) as i32;

    if ret < 0 {
        return Err(Error::SysError(-ret));
    }

    return Ok((ret, fstat));
}

pub fn Ioctl(fd: i32, cmd: u64, argp: u64, argplen: usize) -> i32 {
    return HostSpace::IoCtl(fd, cmd, argp, argplen) as i32;
}

pub fn Fsync(fd: i32) -> i32 {
    return HostSpace::FSync(fd) as i32;
}

pub fn SetTimestamps(fd: i32, ts: &InterTimeSpec) -> Result<()> {
    if ts.ATimeOmit && ts.MTimeOmit {
        return Ok(());
    }

    let mut sts = Box::new_in([Timespec::default(); 2], GUEST_HOST_SHARED_ALLOCATOR);
    
    sts[0] = TimespecFromTimestamp(ts.ATime, ts.ATimeOmit, ts.ATimeSetSystemTime);
    sts[1] = TimespecFromTimestamp(ts.MTime, ts.MTimeOmit, ts.MTimeSetSystemTime);

    let ret = HostSpace::Futimens(fd, &*sts as *const _ as u64);

    if ret < 0 {
        return Err(Error::SysError(-ret as i32));
    }

    return Ok(());
}

pub fn Seek(fd: i32, offset: i64, whence: i32) -> i64 {
    return HostSpace::Seek(fd, offset, whence);
}

pub fn SetMaskedAttributes(fd: i32, mask: &AttrMask, attr: &UnstableAttr) -> Result<()> {
    if mask.Empty() {
        return Ok(());
    }

    if mask.UID || mask.GID {
        return Err(Error::SysError(SysErr::EPERM));
    }

    if mask.Perms {
        let ret = Fchmod(fd, attr.Perms.LinuxMode()) as i32;
        if ret < 0 {
            return Err(Error::SysError(-ret));
        }
    }

    if mask.Size {
        let ret = Ftruncate(fd, attr.Size) as i32;
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

        return SetTimestamps(fd, &ts);
    }

    return Ok(());
}

pub fn Rename(
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
        None => panic!("&InodeOp isn't a HostInodeOp!"),
    };

    let newParent = match newParent
        .lock()
        .InodeOp
        .as_any()
        .downcast_ref::<HostDirOp>()
    {
        Some(p) => p.clone(),
        None => panic!("&InodeOp isn't a HostInodeOp!"),
    };

    let ret = RenameAt(oldParent.HostFd(), oldname, newParent.HostFd(), newname);

    if ret < 0 {
        return Err(Error::SysError(-ret as i32));
    }

    oldParent.lock().readdirCache = None;
    newParent.lock().readdirCache = None;
    return Ok(());
}

pub fn UnstableAttr(
    hostfd: i32,
    task: &Task,
    mo: &Arc<QMutex<MountSourceOperations>>,
) -> Result<UnstableAttr> {
    let uringStatx = SHARESPACE.config.read().UringStatx;

    // the statx uring call sometime become very slow. todo: root cause this.
    if !uringStatx {
        let mut s: LibcStat = Default::default();
        let ret = Fstat(hostfd, &mut s) as i32;
        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        return Ok(s.UnstableAttr(mo));
    } else {
        let s = Box::new_in(Statx::default(), GUEST_HOST_SHARED_ALLOCATOR);
        let addr = &*s as *const  _ as u64;
        let str =  SharedCString::New("");
        let ret = IOURING.Statx(
            task,
            hostfd,
            str.Ptr(),
            addr,
            ATType::AT_EMPTY_PATH,
            StatxMask::STATX_BASIC_STATS,
        );

        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        return Ok(s.UnstableAttr(mo));
    }
}

pub fn Getxattr(fd: i32, name: &str) -> Result<Vec<u8>> {
    let str = SharedCString::New(name);
    let val: &mut [u8; Xattr::XATTR_NAME_MAX] = &mut [0; Xattr::XATTR_NAME_MAX];
    let ret = HostSpace::FGetXattr(fd, str.Ptr(), &val[0] as *const _ as u64, val.len()) as i32;
    if ret < 0 {
        return Err(Error::SysError(-ret));
    };

    return Ok(val[0..ret as usize].to_vec());
}

pub fn Setxattr(fd: i32, name: &str, value: &[u8], flags: u32) -> Result<()> {
    let name = SharedCString::New(name);
    let addr = if value.len() == 0 {
        0
    } else {
        &value[0] as *const _ as u64
    };

    let ret = HostSpace::FSetXattr(fd, name.Ptr(), addr, value.len(), flags) as i32;

    if ret < 0 {
        return Err(Error::SysError(-ret));
    };

    return Ok(());
}

pub fn Listxattr(fd: i32) -> Result<Vec<String>> {
    let val: &mut [u8; Xattr::XATTR_NAME_MAX] = &mut [0; Xattr::XATTR_NAME_MAX];
    let ret = HostSpace::FListXattr(fd, &val[0] as *const _ as u64, val.len()) as i32;
    if ret < 0 {
        return Err(Error::SysError(-ret));
    };

    let mut res = Vec::new();
    let mut cur = 0;
    for i in 0..ret as usize {
        if val[i] == 0 {
            let str = String::from_utf8(val[cur..i].to_vec())
                .map_err(|e| Error::Common(format!("Getxattr fail {}", e)))?;
            res.push(str);
            cur = i + 1;
        }
    }

    return Ok(res);
}

pub fn Removexattr(fd: i32, name: &str) -> Result<()> {
    let name = SharedCString::New(name);
    let ret = HostSpace::FRemoveXattr(fd, name.Ptr()) as i32;

    if ret < 0 {
        return Err(Error::SysError(-ret));
    };

    return Ok(());
}

pub fn StatFS(fd: i32) -> Result<FsInfo> {
    let mut statfs = LibcStatfs::default();

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
