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

use super::super::fs::dirent::*;
use super::super::fs::inotify::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;
use crate::qlib::kernel::util::sharedcstring::SharedCString;
use super::super::Kernel::HostSpace;
use super::sys_file::*;

fn Chmod(task: &Task, d: &Dirent, mode: FileMode) -> Result<()> {
    // Must own file to change mode.
    let mut inode = d.Inode();
    if !inode.CheckOwnership(task) {
        return Err(Error::SysError(SysErr::EPERM));
    }

    let mode = FileMode(mode.0 & (!task.Umask() as u16));
    let p = FilePermissions::FromMode(mode);
    if !inode.SetPermissions(task, d, p) {
        return Err(Error::SysError(SysErr::EPERM));
    }

    // File attribute changed, generate notification.
    d.InotifyEvent(InotifyEvent::IN_ATTRIB, 0, EventType::InodeEvent);

    return Ok(());
}

fn ChmodAt(task: &Task, fd: i32, addr: u64, mode: FileMode) -> Result<()> {
    let (path, _) = copyInPath(task, addr, false)?;

    let mode = FileMode(mode.0 & !(task.Umask() as u16));
    return fileOpOn(task, fd, &path, true, &mut |_root: &Dirent,
                                                 d: &Dirent,
                                                 _remainingTraversals: u32|
     -> Result<()> {
        let ret = Chmod(task, d, mode);
        return ret;
    });
}

// Chmod implements linux syscall chmod(2).
pub fn SysChmod(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let mode = args.arg1 as u16 as u32;

    let mode = FileMode(mode as u16);

    match ChmodAt(task, AT_FDCWD, addr, mode) {
        Err(Error::CHMOD) => {
            // todo: this is workaround. Need to good way to handle non file chmod.
            info!("chmod.....");
            let (path, _) = copyInPath(task, addr, false)?;
            let path = SharedCString::New(&path);
            let ret = HostSpace::Chmod(path.Ptr(), mode.0 as u32);
            return Ok(ret);
            //return Ok(task.Chmod(addr as u64, mode.0 as u64));
        }
        Err(e) => {
            info!("chmod...error {:?}", e);
            return Err(e);
        }
        Ok(()) => return Ok(0),
    }
}

// Fchmod implements linux syscall fchmod(2).
pub fn SysFchmod(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let mode = args.arg1 as u16 as u32;

    let file = task.GetFile(fd)?;
    if file.Flags().Path {
        return Err(Error::SysError(SysErr::EBADF));
    }

    let mode = FileMode(mode as u16);
    Chmod(task, &file.Dirent, mode)?;
    return Ok(0);
}

// Fchmodat implements linux syscall fchmodat(2).
pub fn SysFchmodat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let mode = args.arg2 as u16 as u32;

    let mode = FileMode(mode as u16);
    ChmodAt(task, fd, addr, mode)?;
    return Ok(0);
}
