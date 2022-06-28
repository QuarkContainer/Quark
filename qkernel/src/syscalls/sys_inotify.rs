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

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::kernel::fs::inotify::*;
use super::super::qlib::kernel::fs::anon::*;
use super::super::qlib::kernel::fs::dirent::*;
use super::super::qlib::kernel::fs::flags::*;
use super::super::qlib::kernel::fs::file::*;
use super::super::kernel::fd_table::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;
use super::sys_file::*;


const ALL_FLAGS : i32 = (InotifyEvent::IN_NONBLOCK | InotifyEvent::IN_CLOEXEC) as i32;

pub fn InotifyInit1(task: &mut Task, flags: i32) -> Result<i64> {
    if flags & !ALL_FLAGS != 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    // name matches fs/eventfd.c:eventfd_file_create.
    let inode = NewAnonInode(task);
    let dirent = Dirent::New(&inode, "anon_inode:[timerfd]");

    let fileFlags = FileFlags {
        Read: true,
        Write: true,
        NonBlocking: flags & InotifyEvent::IN_NONBLOCK as i32 != 0,
        ..Default::default()
    };

    let file = File::New(&dirent, &fileFlags, Inotify::New());
    let fd = task.NewFDFrom(
        0,
        &file,
        &FDFlags {
            CloseOnExec: flags & InotifyEvent::IN_CLOEXEC as i32 != 0,
        },
    )?;

    return Ok(fd as i64)
}

// InotifyInit1 implements the inotify_init1() syscalls.
pub fn SysInotifyInit1(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let flags = args.arg0 as i32;
    return InotifyInit1(task, flags);
}

// InotifyInit implements the inotify_init() syscalls.
pub fn SysInotifyInit(task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    let flags = 0;
    return InotifyInit1(task, flags);
}

// fdToInotify resolves an fd to an inotify object. If successful, the file will
// have an extra ref and the caller is responsible for releasing the ref.
pub fn FdToInotify(task: &Task, fd: i32) -> Result<(Inotify, File)> {
    let file = task.GetFile(fd)?;
    let inotify = match file.FileOp.as_any().downcast_ref::<Inotify>() {
        Some(tf) => tf.clone(),
        None => return Err(Error::SysError(SysErr::EINVAL)),
    };

    return Ok((inotify, file))
}

// InotifyAddWatch implements the inotify_add_watch() syscall.
pub fn SysInotifyAddWatch(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let mask = args.arg2 as i32;

    // "IN_DONT_FOLLOW: Don't dereference pathname if it is a symbolic link."
    //  -- inotify(7)
    let resolve = mask & InotifyEvent::IN_DONT_FOLLOW as i32 == 0;

    // "EINVAL: The given event mask contains no valid events."
    // -- inotify_add_watch(2)
    let validBits = mask & InotifyEvent::ALL_INOTIFY_BITS as i32;
    if validBits == 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    let (ino, _file) = FdToInotify(task, fd)?;

    let (path, _) = copyInPath(task, addr, false)?;

    let mut wd = 0;
    fileOpOn(
        task,
        ATType::AT_FDCWD,
        &path,
        resolve,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            let onlyDir = mask & InotifyEvent::IN_ONLYDIR as i32 != 0;
            let inode = d.Inode();
            if onlyDir && !inode.StableAttr().IsDir() {
                return Err(Error::SysError(SysErr::ENOTDIR))
            }

            wd = ino.AddWatch(d, mask as u32);
            return Ok(())
        })?;

    return Ok(wd as i64)
}

// InotifyRmWatch implements the inotify_rm_watch() syscall.
pub fn SysInotifyRmWatch(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let wd = args.arg1 as i32;

    let (ino, _file) = FdToInotify(task, fd)?;
    ino.RmWatch(wd)?;
    return Ok(0)
}