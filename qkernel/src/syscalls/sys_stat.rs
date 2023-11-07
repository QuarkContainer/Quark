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

use super::super::fs::attr::*;
use super::super::fs::dirent::*;
use super::super::fs::file::*;
use super::super::qlib::common::*;
use super::super::qlib::device::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;
use super::sys_file::*;

pub fn SysStat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let statAddr = args.arg1 as u64;

    return Stat(task, addr, statAddr);
}

pub fn Stat(task: &Task, addr: u64, statAddr: u64) -> Result<i64> {
    let (path, dirPath) = copyInPath(task, addr, false)?;
    info!("Stat path is {}", &path);

    fileOpOn(
        task,
        ATType::AT_FDCWD,
        &path,
        true,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            return stat(task, d, dirPath, statAddr);
        },
    )?;

    return Ok(0);
}

pub fn SysFstatat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let statAddr = args.arg2 as u64;
    let flags = args.arg3 as i32;

    return Fstatat(task, fd, addr, statAddr, flags);
}

pub fn Fstatat(task: &Task, fd: i32, addr: u64, statAddr: u64, flags: i32) -> Result<i64> {
    let (path, dirPath) = copyInPath(task, addr, flags & ATType::AT_EMPTY_PATH != 0)?;

    info!(
        "Fstatat path is {} dirPath {}, flags & ATType::AT_SYMLINK_NOFOLLOW {:x}",
        &path,
        dirPath,
        flags & ATType::AT_SYMLINK_NOFOLLOW
    );
    if path.len() == 0 {
        let file = task.GetFile(fd)?;

        fstat(task, &file, statAddr)?;
        return Ok(0);
    }

    let resolve = dirPath || flags & ATType::AT_SYMLINK_NOFOLLOW == 0;

    let ret = fileOpOn(
        task,
        fd,
        &path,
        resolve,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            return stat(task, d, dirPath, statAddr);
        },
    );

    match ret {
        Err(e) => {
            //error!("Fstatat fail path is {}, error is {:?}", &path, &e);
            return Err(e);
        }
        _ => (),
    }

    return Ok(0);
}

pub fn SysLstat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let statAddr = args.arg1 as u64;

    return Lstat(task, addr, statAddr);
}

pub fn Lstat(task: &Task, addr: u64, statAddr: u64) -> Result<i64> {
    let (path, dirPath) = copyInPath(task, addr, false)?;

    info!("Lstat path is {}", &path);
    let resolve = dirPath;

    fileOpOn(
        task,
        ATType::AT_FDCWD,
        &path,
        resolve,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            return stat(task, d, dirPath, statAddr);
        },
    )?;

    return Ok(0);
}

pub fn SysFstat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let statAddr = args.arg1 as u64;

    return Fstat(task, fd, statAddr);
}

pub fn SysStatx(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let pathAddr = args.arg1 as u64;
    let flags = args.arg2 as i32;
    let mask = args.arg3 as u32;
    let statxAddr = args.arg4 as u64;

    if mask & StatxMask::STATX__RESERVED != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if flags
        & !(ATType::AT_SYMLINK_NOFOLLOW
            | ATType::AT_EMPTY_PATH
            | StatxFlags::AT_STATX_SYNC_TYPE as i32)
        != 0
    {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if flags as u32 & StatxFlags::AT_STATX_SYNC_TYPE == StatxFlags::AT_STATX_SYNC_TYPE {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let (path, dirPath) = copyInPath(task, pathAddr, flags & ATType::AT_EMPTY_PATH != 0)?;

    if path.len() == 0 {
        let file = task.GetFile(fd)?;

        let uattr = file.UnstableAttr(task)?;
        let inode = file.Dirent.Inode();
        let sattr = inode.StableAttr();
        statx(task, &sattr, &uattr, statxAddr)?;
        return Ok(0);
    }

    let resolve = dirPath || flags & ATType::AT_SYMLINK_NOFOLLOW == 0;

    fileOpOn(
        task,
        fd,
        &path,
        resolve,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            let inode = d.Inode();
            let sattr = inode.StableAttr();

            if dirPath && !sattr.IsDir() {
                return Err(Error::SysError(SysErr::ENOTDIR));
            }

            let uattr = inode.UnstableAttr(task)?;

            statx(task, &sattr, &uattr, statxAddr)?;
            return Ok(());
        },
    )?;

    return Ok(0);
}

pub fn Fstat(task: &Task, fd: i32, statAddr: u64) -> Result<i64> {
    let file = task.GetFile(fd)?;

    fstat(task, &file, statAddr)?;
    return Ok(0);
}

pub fn SysStatfs(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let statfsAddr = args.arg1 as u64;

    let res = Statfs(task, addr, statfsAddr)?;
    return Ok(res as i64);
}

pub fn Statfs(task: &Task, addr: u64, statfsAddr: u64) -> Result<u64> {
    let (path, _dirPath) = copyInPath(task, addr, false)?;

    info!("Statfs path is {}", &path);
    fileOpOn(
        task,
        ATType::AT_FDCWD,
        &path,
        true,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            return statfsImpl(task, d, statfsAddr);
        },
    )?;

    return Ok(0);
}

pub fn SysFstatfs(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let statfsAddr = args.arg1 as u64;

    let file = task.GetFile(fd)?;

    statfsImpl(task, &file.Dirent, statfsAddr)?;
    return Ok(0);
}

fn stat(task: &Task, d: &Dirent, dirPath: bool, statAddr: u64) -> Result<()> {
    let inode = d.Inode();

    if dirPath && !inode.StableAttr().IsDir() {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    let uattr = inode.UnstableAttr(task)?;
    let sattr = inode.StableAttr();

    copyOutStat(task, statAddr, &sattr, &uattr)?;
    return Ok(());
}

fn fstat(task: &Task, f: &File, statAddr: u64) -> Result<()> {
    let inode = f.Dirent.Inode();

    let uattr = f.UnstableAttr(task)?;
    let sattr = inode.StableAttr();

    copyOutStat(task, statAddr, &sattr, &uattr)?;
    return Ok(());
}

fn copyOutStat(task: &Task, statAddr: u64, sattr: &StableAttr, uattr: &UnstableAttr) -> Result<()> {
    //let mut s: &mut LibcStat = task.GetTypeMut(statAddr)?;

    let mut s: LibcStat = LibcStat::default();
    //*s = LibcStat::default();
    let creds = task.creds.clone();
    let ns = creds.lock().UserNamespace.clone();

    #[cfg(target_arch = "aarch64")]
    let st_nlink = uattr.Links as u32;
    #[cfg(target_arch = "x86_64")]
    let st_nlink = uattr.Links;

    #[cfg(target_arch = "aarch64")]
    let st_blksize = uattr.Links as i32;
    #[cfg(target_arch = "x86_64")]
    let st_blksize = uattr.Links;

    s.st_dev = sattr.DeviceId as u64;
    s.st_ino = sattr.InodeId;
    s.st_nlink = st_nlink;
    s.st_mode = sattr.Type.LinuxType() as u32 | uattr.Perms.LinuxMode();
    s.st_uid = uattr.Owner.UID.In(&ns).OrOverflow().0;
    s.st_gid = uattr.Owner.GID.In(&ns).OrOverflow().0;
    s.st_rdev = MakeDeviceID(sattr.DeviceFileMajor, sattr.DeviceFileMinor) as u64;
    s.st_size = uattr.Size;
    s.st_blksize = st_blksize;
    s.st_blocks = uattr.Usage / 512;

    let atime = uattr.AccessTime.Timespec();
    s.st_atime = atime.tv_sec;
    s.st_atime_nsec = atime.tv_nsec;

    let mtime = uattr.ModificationTime.Timespec();
    s.st_mtime = mtime.tv_sec;
    s.st_mtime_nsec = mtime.tv_nsec;

    let ctime = uattr.StatusChangeTime.Timespec();
    s.st_ctime = ctime.tv_sec;
    s.st_ctime_nsec = ctime.tv_nsec;

    task.CopyOutObj(&s, statAddr)?;
    //info!("copyOutStat stat is {:x?}", s);
    return Ok(());
}

fn statx(task: &Task, sattr: &StableAttr, uattr: &UnstableAttr, statxAddr: u64) -> Result<()> {
    let (devMajor, devMinor) = DecodeDeviceId(sattr.DeviceId as u32);

    let creds = task.creds.clone();
    let ns = creds.lock().UserNamespace.clone();

    //let out: &mut Statx = task.GetTypeMut::<Statx>(statxAddr)?;

    let s = Statx {
        stx_mask: StatxMask::STATX_BASIC_STATS,
        stx_blksize: sattr.BlockSize as u32,
        stx_attributes: 0,
        stx_nlink: uattr.Links as u32,
        stx_uid: uattr.Owner.UID.In(&ns).OrOverflow().0,
        stx_gid: uattr.Owner.GID.In(&ns).OrOverflow().0,
        stx_mode: (sattr.Type.LinuxType() as u32 | uattr.Perms.LinuxMode()) as u16,
        __statx_pad1: [0; 1],
        stx_ino: sattr.InodeId,
        stx_size: uattr.Size as u64,
        stx_attributes_mask: 0,
        stx_blocks: uattr.Usage as u64 / 512,
        stx_atime: uattr.AccessTime.StatxTimestamp(),
        stx_btime: StatxTimestamp::default(),
        stx_ctime: uattr.StatusChangeTime.StatxTimestamp(),
        stx_mtime: uattr.ModificationTime.StatxTimestamp(),
        stx_rdev_major: sattr.DeviceFileMajor as u32,
        stx_rdev_minor: sattr.DeviceFileMinor,
        stx_dev_major: devMajor as u32,
        stx_dev_minor: devMinor,
        __statx_pad2: [0; 14],
    };

    //*out = s;

    task.CopyOutObj(&s, statxAddr)?;
    return Ok(());
}

fn statfsImpl(task: &Task, d: &Dirent, addr: u64) -> Result<()> {
    let inode = d.Inode();
    let sattr = inode.lock().StableAttr().clone();

    let info = inode.StatFS(task)?;

    let statfs = LibcStatfs {
        Type: info.Type,
        BlockSize: sattr.BlockSize,
        Blocks: info.TotalBlocks,
        BlocksFree: info.FreeBlocks,
        BlocksAvailable: info.FreeBlocks,
        Files: info.TotalFiles,
        FilesFree: info.FreeFiles,
        NameLength: NAME_MAX as u64,
        FragmentSize: sattr.BlockSize,
        ..Default::default()
    };

    //let out: &mut LibcStatfs = task.GetTypeMut::<LibcStatfs>(addr)?;
    //*out = statfs;

    task.CopyOutObj(&statfs, addr)?;

    return Ok(());
}
