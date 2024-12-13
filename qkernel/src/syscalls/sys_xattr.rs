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

use super::super::fs::dirent::*;
use super::super::fs::inode::*;
use super::super::fs::inotify::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::path::*;
use super::super::task::*;
use super::sys_file::*;
use super::syscalls::*;

// GetXattr implements linux syscall getxattr(2).
pub fn SysGetXattr(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    return getXattrFromPath(task, args, true);
}

// LGetXattr implements linux syscall lgetxattr(2).
pub fn SysLGetXattr(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    return getXattrFromPath(task, args, false);
}

// FGetXattr implements linux syscall fgetxattr(2).
pub fn SysFGetXattr(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let nameAddr = args.arg1 as u64;
    let valueAddr = args.arg2 as u64;
    let size = args.arg3 as u32 as u64;

    // Return EBADF if the fd was opened with O_PATH.
    let file = task.GetFile(fd)?;

    if file.Flags().Path {
        return Err(Error::SysError(SysErr::EBADF));
    }

    let n = GetXAttr(task, &file.Dirent, nameAddr, valueAddr, size as usize)?;
    return Ok(n);
}

pub fn getXattrFromPath(
    task: &mut Task,
    args: &SyscallArguments,
    resolveSymlink: bool,
) -> Result<i64> {
    let pathAddr = args.arg0 as u64;
    let nameAddr = args.arg1 as u64;
    let valueAddr = args.arg2 as u64;
    let size = args.arg3 as u32;

    let (path, dirPath) = copyInPath(task, pathAddr, false)?;

    let mut n = 0;
    fileOpOn(
        task,
        ATType::AT_FDCWD,
        &path,
        resolveSymlink,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            let inode = d.Inode();

            if dirPath && !inode.StableAttr().IsDir() {
                return Err(Error::SysError(SysErr::ENOTDIR));
            }

            n = GetXAttr(task, d, nameAddr, valueAddr, size as usize)?;
            return Ok(());
        },
    )?;

    return Ok(n as i64);
}

pub fn GetXAttr(
    task: &Task,
    d: &Dirent,
    nameAddr: u64,
    valueAddr: u64,
    size: usize,
) -> Result<i64> {
    let name = CopyInXattrName(task, nameAddr)?;

    let inode = d.Inode();
    CheckXattrPermissons(
        task,
        &inode,
        &PermMask {
            read: true,
            ..Default::default()
        },
    )?;

    if !HasPrefix(&name, Xattr::XATTR_USER_PREFIX) {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    // If getxattr(2) is called with size 0, the size of the value will be
    // returned successfully even if it is nonzero. In that case, we need to
    // retrieve the entire attribute value so we can return the correct size.
    let mut requestdatasize = size;
    if size == 0 || size > Xattr::XATTR_SIZE_MAX {
        requestdatasize = Xattr::XATTR_SIZE_MAX;
    }

    let value = inode.Getxattr(task, &name, requestdatasize)?;

    let n = value.len();
    if n > requestdatasize {
        return Err(Error::SysError(SysErr::ERANGE));
    }

    // Don't copy out the attribute value if size is 0.
    if size == 0 {
        return Ok(n as i64);
    }

    task.CopyOutSlice(&value, valueAddr, n)?;
    return Ok(n as i64);
}

// SetXattr implements linux syscall setxattr(2).
pub fn SysSetXattr(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    return setXattrFromPath(task, args, true);
}

// LSetXattr implements linux syscall lsetxattr(2).
pub fn SysLSetXattr(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    return setXattrFromPath(task, args, false);
}

// FGetXattr implements linux syscall fgetxattr(2).
pub fn SysFSetXattr(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let nameAddr = args.arg1 as u64;
    let valueAddr = args.arg2 as u64;
    let size = args.arg3 as u64;
    let flags = args.arg4 as u32;

    let file = task.GetFile(fd)?;
    if file.Flags().Path {
        return Err(Error::SysError(SysErr::EBADF));
    }

    SetXAttr(
        task,
        &file.Dirent,
        nameAddr,
        valueAddr,
        size as usize,
        flags,
    )?;
    return Ok(0);
}

pub fn setXattrFromPath(
    task: &mut Task,
    args: &SyscallArguments,
    resolveSymlink: bool,
) -> Result<i64> {
    let pathAddr = args.arg0 as u64;
    let nameAddr = args.arg1 as u64;
    let valueAddr = args.arg2 as u64;
    let size = args.arg3 as u64;
    let flags = args.arg4 as u32;

    let (path, dirPath) = copyInPath(task, pathAddr, false)?;

    fileOpOn(
        task,
        ATType::AT_FDCWD,
        &path,
        resolveSymlink,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            let inode = d.Inode();

            if dirPath && !inode.StableAttr().IsDir() {
                return Err(Error::SysError(SysErr::ENOTDIR));
            }

            SetXAttr(task, d, nameAddr, valueAddr, size as usize, flags)?;
            return Ok(());
        },
    )?;

    return Ok(0);
}

pub fn SetXAttr(
    task: &Task,
    d: &Dirent,
    nameAddr: u64,
    valueAddr: u64,
    size: usize,
    flags: u32,
) -> Result<()> {
    if flags & !(Xattr::XATTR_CREATE | Xattr::XATTR_REPLACE) != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let name = CopyInXattrName(task, nameAddr)?;

    let mut inode = d.Inode();
    CheckXattrPermissons(
        task,
        &inode,
        &PermMask {
            write: true,
            ..Default::default()
        },
    )?;

    if size > Xattr::XATTR_SIZE_MAX {
        return Err(Error::SysError(SysErr::E2BIG));
    }

    let buf = task.CopyInVecShared(valueAddr, size)?;

    if !HasPrefix(&name, Xattr::XATTR_USER_PREFIX) {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    inode.Setxattr(task, d, &name, &buf, flags)?;
    d.InotifyEvent(InotifyEvent::IN_ATTRIB, 0, EventType::InodeEvent);
    return Ok(());
}

pub fn CopyInXattrName(task: &Task, nameAddr: u64) -> Result<String> {
    let (name, err) = task
        .mm
        .CopyInString(task, nameAddr, Xattr::XATTR_NAME_MAX + 1);
    match err {
        Err(Error::SysError(SysErr::ENAMETOOLONG)) => return Err(Error::SysError(SysErr::ERANGE)),
        Err(e) => return Err(e),
        Ok(()) => (),
    }

    if name.len() == 0 {
        return Err(Error::SysError(SysErr::ERANGE));
    }

    return Ok(name);
}

// Restrict xattrs to regular files and directories.
//
// In Linux, this restriction technically only applies to
// xattrs in the "user.*" namespace. Make file type checks specific to the
// namespace once we allow other xattr prefixes.
pub fn XattrFileTypeOk(i: &Inode) -> bool {
    return i.StableAttr().IsDir() || i.StableAttr().IsRegular();
}

pub fn CheckXattrPermissons(task: &Task, i: &Inode, perms: &PermMask) -> Result<()> {
    // Restrict xattrs to regular files and directories.
    if !XattrFileTypeOk(i) {
        if perms.write {
            return Err(Error::SysError(SysErr::EPERM));
        }
        return Err(Error::SysError(SysErr::ENODATA));
    }

    return i.CheckPermission(task, &perms);
}

// ListXattr implements linux syscall listxattr(2).
pub fn SysListXattr(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    return ListXattrFromPath(task, args, true);
}

// LListXattr implements linux syscall llistxattr(2).
pub fn SysLListXattr(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    return ListXattrFromPath(task, args, false);
}

// FListXattr implements linux syscall flistxattr(2).
pub fn SysFListXattr(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let listAddr = args.arg1 as u64;
    let size = args.arg2 as u32 as u64;

    let file = task.GetFile(fd)?;
    if file.Flags().Path {
        return Err(Error::SysError(SysErr::EBADF));
    }

    let n = ListXAttr(task, &file.Dirent, listAddr, size as usize)?;
    return Ok(n);
}

pub fn ListXattrFromPath(
    task: &mut Task,
    args: &SyscallArguments,
    resolveSymlink: bool,
) -> Result<i64> {
    let pathAddr = args.arg0 as u64;
    let listAddr = args.arg1 as u64;
    let size = args.arg2 as u32;

    let (path, dirPath) = copyInPath(task, pathAddr, false)?;
    let mut n = 0;
    fileOpOn(
        task,
        ATType::AT_FDCWD,
        &path,
        resolveSymlink,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            let inode = d.Inode();

            if dirPath && !inode.StableAttr().IsDir() {
                return Err(Error::SysError(SysErr::ENOTDIR));
            }

            n = ListXAttr(task, d, listAddr, size as usize)?;
            return Ok(());
        },
    )?;

    return Ok(n as i64);
}

pub fn ListXAttr(task: &Task, d: &Dirent, addr: u64, size: usize) -> Result<i64> {
    let inode = d.Inode();
    if !XattrFileTypeOk(&inode) {
        return Ok(0);
    }

    // If listxattr(2) is called with size 0, the buffer size needed to contain
    // the xattr list will be returned successfully even if it is nonzero. In
    // that case, we need to retrieve the entire list so we can compute and
    // return the correct size.
    let mut requestdatasize = size;
    if size == 0 || size > Xattr::XATTR_SIZE_MAX {
        requestdatasize = Xattr::XATTR_SIZE_MAX;
    }

    let xattrs = inode.Listxattr(requestdatasize)?;

    let mut listSize = 0;
    for name in &xattrs {
        // todo: support namespaces other than "user".
        if HasPrefix(&name, Xattr::XATTR_USER_PREFIX) {
            listSize += name.len() + 1;
        }
    }

    if listSize > Xattr::XATTR_SIZE_MAX {
        return Err(Error::SysError(SysErr::E2BIG));
    }

    if listSize > requestdatasize {
        return Err(Error::SysError(SysErr::ERANGE));
    }

    // Don't copy out the attributes if size is 0.
    if size == 0 {
        return Ok(listSize as _);
    }

    let mut buf = Vec::new();
    for name in xattrs {
        // todo: support namespaces other than "user".
        if HasPrefix(&name, Xattr::XATTR_USER_PREFIX) {
            buf.append(&mut name.as_bytes().to_vec());
            buf.push(0);
        }
    }

    task.CopyOutSlice(&buf, addr, listSize)?;

    return Ok(listSize as _);
}

// RemoveXattr implements linux syscall removexattr(2).
pub fn SysRemoveXattr(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    return RemoveXattrFromPath(task, args, true);
}

// LRemoveXattr implements linux syscall lremovexattr(2).
pub fn SysLRemoveXattr(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    return RemoveXattrFromPath(task, args, false);
}

// FRemoveXattr implements linux syscall fremovexattr(2).
pub fn SysFRemoveXattr(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let nameAddr = args.arg1 as u64;

    let file = task.GetFile(fd)?;
    if file.Flags().Path {
        return Err(Error::SysError(SysErr::EBADF));
    }

    RemoveAttr(task, &file.Dirent, nameAddr)?;
    return Ok(0);
}

pub fn RemoveXattrFromPath(
    task: &mut Task,
    args: &SyscallArguments,
    resolveSymlink: bool,
) -> Result<i64> {
    let pathAddr = args.arg0 as u64;
    let nameAddr = args.arg1 as u64;

    let (path, dirPath) = copyInPath(task, pathAddr, false)?;

    fileOpOn(
        task,
        ATType::AT_FDCWD,
        &path,
        resolveSymlink,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            let inode = d.Inode();

            if dirPath && !inode.StableAttr().IsDir() {
                return Err(Error::SysError(SysErr::ENOTDIR));
            }

            RemoveAttr(task, d, nameAddr)?;
            return Ok(());
        },
    )?;

    return Ok(0);
}

pub fn RemoveAttr(task: &Task, d: &Dirent, nameAddr: u64) -> Result<()> {
    let name = CopyInXattrName(task, nameAddr)?;

    let mut inode = d.Inode();
    CheckXattrPermissons(
        task,
        &inode,
        &PermMask {
            write: true,
            ..Default::default()
        },
    )?;

    if !HasPrefix(&name, Xattr::XATTR_USER_PREFIX) {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    inode.Removexattr(task, d, &name)?;
    d.InotifyEvent(InotifyEvent::IN_ATTRIB, 0, EventType::InodeEvent);
    return Ok(());
}
