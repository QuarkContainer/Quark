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

use super::super::fs::dirent::*;
use super::super::fs::file::*;
use super::super::fs::flags::*;
use super::super::fs::inode::*;
use super::super::fs::inotify::*;
use super::super::fs::lock::*;
use super::super::kernel::fasync::*;
use super::super::kernel::fd_table::*;
use super::super::kernel::pipe::reader::*;
use super::super::kernel::pipe::reader_writer::*;
use super::super::kernel::pipe::writer::*;
use super::super::kernel::time::*;
//use super::super::kernel_def::*;
use super::super::qlib::auth::cap_set::*;
use super::super::qlib::auth::id::*;
use super::super::qlib::auth::*;
use super::super::qlib::common::*;
use super::super::qlib::limits::*;
use super::super::qlib::linux::fcntl::*;
use super::super::qlib::linux::time::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::path::*;
use super::super::qlib::range::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;
use super::super::util::cstring::*;
use crate::qlib::kernel::util::sharedcstring::SharedCString;
use fs::host::hostinodeop::HostInodeOp;
use fs::host::util::Fcntl;

fn fileOpAt(
    task: &Task,
    dirFd: i32,
    path: &str,
    func: &mut FnMut(&Dirent, &Dirent, &str, u32) -> Result<()>,
) -> Result<()> {
    let (dir, name) = SplitLast(path);

    if dir == "/" {
        return func(
            &task.Root(),
            &task.Root(),
            &name.to_string(),
            MAX_SYMLINK_TRAVERSALS,
        );
    } else if dir == "." && dirFd == AT_FDCWD {
        return func(
            &task.Root(),
            &task.Workdir(),
            &name.to_string(),
            MAX_SYMLINK_TRAVERSALS,
        );
    }

    return fileOpOn(
        task,
        dirFd,
        &dir.to_string(),
        true,
        &mut |root: &Dirent, d: &Dirent, remainingTraversals: u32| -> Result<()> {
            return func(root, d, &name.to_string(), remainingTraversals);
        },
    );
}

pub fn fileOpOn(
    task: &Task,
    dirFd: i32,
    path: &str,
    resolve: bool,
    func: &mut FnMut(&Dirent, &Dirent, u32) -> Result<()>,
) -> Result<()> {
    let d: Dirent;
    let wd: Dirent;
    let mut rel: Option<Dirent> = None;

    if path.len() > 0 && path.as_bytes()[0] == '/' as u8 {
        // Absolute path; rel can be nil.
    } else if dirFd == ATType::AT_FDCWD {
        wd = task.Workdir();
        rel = Some(wd.clone());
    } else {
        let file = task.GetFile(dirFd)?;

        let dirent = file.Dirent.clone();
        let inode = dirent.Inode();
        if !inode.StableAttr().IsDir() {
            return Err(Error::SysError(SysErr::ENOTDIR));
        }

        rel = Some(file.Dirent.clone());
    }

    let root = task.Root();
    let mut remainTraversals = MAX_SYMLINK_TRAVERSALS;

    d = task
        .mountNS
        .FindDirent(task, &root, rel, path, &mut remainTraversals, resolve)?;

    return func(&root, &d, remainTraversals);
}

//return (path, whether it is dir)
pub fn copyInPath(task: &Task, addr: u64, allowEmpty: bool) -> Result<(String, bool)> {
    let str = CString::ToString(task, addr)?;

    if &str == "" && !allowEmpty {
        return Err(Error::SysError(SysErr::ENOENT));
    }

    let (path, dirPath) = TrimTrailingSlashes(&str);

    return Ok((path.to_string(), dirPath));
}

pub fn SysOpenAt(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let dirFd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let flags = args.arg2 as i32;
    let mode = args.arg3 as u16 as u32;

    let flags = CleanOpenFlags(flags)? as u32;

    if flags & Flags::O_CREAT as u32 != 0 {
        let res = createAt(task, dirFd, addr, flags, FileMode(mode as u16))?;
        return Ok(res as i64);
    }

    let res = openAt(task, dirFd, addr, flags)?;
    return Ok(res as i64);
}

pub fn CleanOpenFlags(flags: i32) -> Result<i32> {
    let mut flags = flags
        & (Flags::O_ACCMODE
            | Flags::O_CREAT
            | Flags::O_EXCL
            | Flags::O_NOCTTY
            | Flags::O_TRUNC
            | Flags::O_APPEND
            | Flags::O_NONBLOCK
            | Flags::O_DSYNC
            | Flags::O_ASYNC
            | Flags::O_DIRECT
            | Flags::O_LARGEFILE
            | Flags::O_DIRECTORY
            | Flags::O_NOFOLLOW
            | Flags::O_NOATIME
            | Flags::O_SYNC
            | Flags::O_PATH
            | Flags::O_TMPFILE);

    // Linux's __O_SYNC (which we call linux.O_SYNC) implies O_DSYNC.
    if flags & Flags::O_SYNC != 0 {
        flags |= Flags::O_DSYNC;
    }

    // Linux's __O_TMPFILE (which we call linux.O_TMPFILE) must be specified
    // with O_DIRECTORY and a writable access mode (to ensure that it fails on
    // filesystem implementations that do not support it).
    if flags & Flags::O_TMPFILE != 0 {
        if flags & Flags::O_DIRECTORY == 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }
        if flags & Flags::O_CREAT != 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }
        if flags & Flags::O_ACCMODE == Flags::O_RDONLY {
            return Err(Error::SysError(SysErr::EINVAL));
        }
    }

    // we can't read/write will readonly or writeonly
    // work around. todo: find better solution
    if flags & Flags::O_RDWR != 0 && (flags & Flags::O_RDONLY != 0 || flags & Flags::O_WRONLY != 0)
    {
        //return Err(Error::SysError(SysErr::EINVAL));
        flags |= Flags::O_PATH;
    }

    // O_PATH causes most other flags to be ignored.
    if flags & Flags::O_PATH != 0 {
        flags &= Flags::O_DIRECTORY | Flags::O_NOFOLLOW | Flags::O_PATH;
    }

    return Ok(flags);
}

pub fn SysOpen(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let flags = args.arg1 as i32;
    let mode = args.arg2 as u16 as u32;

    let flags = CleanOpenFlags(flags)?;

    if flags & Flags::O_CREAT != 0 {
        let res = createAt(
            task,
            ATType::AT_FDCWD,
            addr,
            flags as u32,
            FileMode(mode as u16),
        )?;
        return Ok(res as i64);
    }

    let res = openAt(task, ATType::AT_FDCWD, addr, flags as u32)?;
    return Ok(res as i64);
}

pub fn SysCreate(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let mode = args.arg1 as u16 as u32;

    let res = createAt(
        task,
        ATType::AT_FDCWD,
        addr,
        (Flags::O_WRONLY | Flags::O_TRUNC) as u32,
        FileMode(mode as u16),
    )?;
    return Ok(res as i64);
}

pub fn openAt(task: &Task, dirFd: i32, addr: u64, flags: u32) -> Result<i32> {
    //task.PerfGoto(PerfType::Open);
    //defer!(task.PerfGofrom(PerfType::Open));

    let (path, dirPath) = copyInPath(task, addr, false)?;

    info!(
        "openat path is {}, the perm is {:?}, current is {}",
        &path,
        &PermMask::FromFlags(flags),
        task.fsContext.WorkDirectory().MyFullName()
    );

    let mut fileFlags = FileFlags::FromFlags(flags);
    let resolve = !fileFlags.NoFollow && !fileFlags.Path;
    let mut fd = -1;

    fileOpOn(
        task,
        dirFd,
        &path,
        resolve,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            let mut inode = d.Inode();

            // if needs write to file, check whether the inode is opened as writable
            if inode.StableAttr().IsRegular() && fileFlags.Write {
                let iops = inode.lock().InodeOp.clone();
                match iops.HostInodeOp() {
                    Some(iops) => {
                        let mut lkiops = iops.lock();
                        if lkiops.SkipRw() {
                            let parent = d.Parent().unwrap();
                            let parentIops = parent.Inode().lock().InodeOp.clone();
                            
                            let dirfd = parentIops.HostDirOp().expect(&format!("inodeop type is {:?}", parentIops.InodeType())).HostFd();
                            let name = d.Name();
                            let cstr = SharedCString::New(&name);
                            lkiops.TryOpenWrite(dirfd, cstr.Ptr())?;
                        }
                    }
                    None => (),
                }
            }

            if !fileFlags.Path {
                inode.CheckPermission(task, &PermMask::FromFlags(flags))?;
            }

            if inode.StableAttr().IsSymlink() && !resolve && !fileFlags.Path {
                return Err(Error::SysError(SysErr::ELOOP));
            }


            // Linux always adds the O_LARGEFILE flag when running in 64-bit mode.
            if !fileFlags.Path {
                fileFlags.LargeFile = true;
            }

            if inode.StableAttr().IsDir() {
                if fileFlags.Write {
                    return Err(Error::SysError(SysErr::EISDIR));
                }
            } else {
                if fileFlags.Directory {
                    return Err(Error::SysError(SysErr::ENOTDIR));
                }

                if dirPath {
                    return Err(Error::SysError(SysErr::ENOTDIR));
                }
            }

            if inode.StableAttr().IsSocket() {
                if !fileFlags.Path {
                    return Err(Error::SysError(SysErr::ENXIO));
                } else if fileFlags.Read || fileFlags.Write {
                    return Err(Error::SysError(SysErr::ENXIO));
                }
            }

            if flags & Flags::O_TRUNC as u32 != 0 {
                if inode.StableAttr().IsDir() {
                    return Err(Error::SysError(SysErr::EISDIR));
                }

                inode.Truncate(task, d, 0)?;
            }

            let file = match inode.GetFile(task, &d, &fileFlags) {
                Ok(f) => f,
                Err(Error::ErrInterrupted) => {
                    return Err(Error::SysError(SysErr::ERESTARTSYS));
                }
                Err(e) => {
                    return Err(e);
                }
            };

            let newFd = task.NewFDFrom(
                0,
                &file,
                &FDFlags {
                    CloseOnExec: flags & Flags::O_CLOEXEC as u32 != 0,
                },
            )?;

            fd = newFd;

            d.InotifyEvent(InotifyEvent::IN_OPEN, 0, EventType::InodeEvent);

            return Ok(());
        },
    )?;

    return Ok(fd);
}

// Mknod implements the linux syscall mknod(2).
pub fn SysMknode(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let path = args.arg0 as u64;
    let mode = args.arg1 as u16;

    // We don't need this argument until we support creation of device nodes.
    //let _dev = args.arg2 as u16 as u32;

    mknodeAt(task, ATType::AT_FDCWD, path, FileMode(mode))?;
    return Ok(0);
}

// Mknodat implements the linux syscall mknodat(2).
pub fn SysMknodeat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let dirFD = args.arg0 as i32;
    let path = args.arg1 as u64;
    let mode = args.arg2 as u16;

    // We don't need this argument until we support creation of device nodes.
    //let _dev = args.arg3 as u16 as u32;

    mknodeAt(task, dirFD, path, FileMode(mode))?;
    return Ok(0);
}

pub fn mknodeAt(task: &Task, dirFd: i32, addr: u64, mode: FileMode) -> Result<()> {
    let (path, dirPath) = copyInPath(task, addr, false)?;

    if dirPath {
        return Err(Error::SysError(SysErr::ENOENT));
    }

    return fileOpAt(task, dirFd, &path, &mut |root: &Dirent,
                                              d: &Dirent,
                                              name: &str,
                                              _remainingTraversals: u32|
     -> Result<()> {
        let inode = d.Inode();
        inode.CheckPermission(
            task,
            &PermMask {
                write: true,
                execute: true,
                ..Default::default()
            },
        )?;

        let perms = FilePermissions::FromMode(FileMode(mode.0 & !task.Umask() as u16));

        match mode.FileType().0 {
            0 | ModeType::MODE_REGULAR => {
                let flags = FileFlags {
                    Read: true,
                    Write: true,
                    ..Default::default()
                };

                let _file = d.Create(task, root, name, &flags, &perms)?;
                return Ok(());
            }
            ModeType::MODE_NAMED_PIPE => {
                return d.CreateFifo(task, root, name, &perms);
            }
            ModeType::MODE_SOCKET => return Err(Error::SysError(SysErr::EOPNOTSUPP)),
            ModeType::MODE_CHARACTER_DEVICE | ModeType::MODE_BLOCK_DEVICE => {
                return Err(Error::SysError(SysErr::EPERM))
            }
            _ => return Err(Error::SysError(SysErr::EINVAL)),
        }
    });
}

pub fn createAt(task: &Task, dirFd: i32, addr: u64, flags: u32, mode: FileMode) -> Result<i32> {
    let (path, dirPath) = copyInPath(task, addr, false)?;

    info!(
        "createAt path is {}, current is {}, flags is {}",
        &path,
        task.fsContext.WorkDirectory().MyFullName(),
        flags
    );
    if dirPath {
        return Err(Error::SysError(SysErr::EISDIR));
    }

    let mut fileFlags = FileFlags::FromFlags(flags);
    // Linux always adds the O_LARGEFILE flag when running in 64-bit mode.
    if !fileFlags.Path {
        fileFlags.LargeFile = true;
    }

    // the io_uring write will fail with EAGAIN even for disk file. Work around to make sure the file is opened without nonblocking
    fileFlags.NonBlocking = false;

    let mut fd = 0;
    let mnt = task.mountNS.clone();

    fileOpAt(task, dirFd, &path, &mut |root: &Dirent,
                                       parent: &Dirent,
                                       name: &str,
                                       remainingTraversals: u32|
     -> Result<()> {
        let mut found = parent.clone();

        let mut remainingTraversals = remainingTraversals;
        let mut parent = parent.clone();
        let mut name = name.to_string();
        let mut err = Error::None;

        loop {
            let parentInode = parent.Inode();
            if !parentInode.StableAttr().IsDir() {
                return Err(Error::SysError(SysErr::ENOTDIR));
            }

            found = match mnt.FindDirent(
                task,
                root,
                Some(parent.clone()),
                &name,
                &mut remainingTraversals,
                false,
            ) {
                Ok(d) => d,
                Err(e) => {
                    err = e;
                    break;
                }
            };

            if flags & Flags::O_EXCL as u32 != 0 {
                return Err(Error::SysError(SysErr::EEXIST));
            }

            let foundInode = found.Inode();
            if foundInode.StableAttr().IsDir() && fileFlags.Write {
                return Err(Error::SysError(SysErr::EISDIR));
            }

            if !foundInode.StableAttr().IsSymlink() {
                break;
            }

            if flags & Flags::O_NOFOLLOW as u32 != 0 {
                return Err(Error::SysError(SysErr::ELOOP));
            }

            match foundInode.GetLink(task) {
                Err(Error::ErrResolveViaReadlink) => (),
                Err(e) => return Err(e),
                Ok(_) => break,
            };

            if remainingTraversals == 0 {
                return Err(Error::SysError(SysErr::ELOOP));
            }

            let path = match foundInode.ReadLink(task) {
                Err(e) => {
                    err = e;
                    break;
                }
                Ok(p) => p,
            };

            remainingTraversals -= 1;

            let (newParentPath, newName) = SplitLast(&path);
            let newParent = match mnt.FindDirent(
                task,
                root,
                Some(parent.clone()),
                &newParentPath.to_string(),
                &mut remainingTraversals,
                true,
            ) {
                Err(e) => {
                    err = e;
                    break;
                }
                Ok(p) => p,
            };

            parent = newParent;
            name = newName.to_string();
        }

        let newFile = match err {
            Error::None => {
                let mut foundInode = found.Inode();

                if foundInode.StableAttr().IsRegular() {
                    let iops = foundInode.lock().InodeOp.clone();
                    match iops.HostInodeOp() {
                        Some(iops) => {
                            let mut lkiops = iops.lock();
                            let dirfd = parent.Inode().lock().InodeOp.HostDirOp().unwrap().HostFd();
                            if lkiops.SkipRw() {
                                let cstr = SharedCString::New(&name);
                                lkiops.TryOpenWrite(dirfd, cstr.Ptr())?;
                            }
                        }
                        None => (),
                    }
                }

                if flags & Flags::O_TRUNC as u32 != 0 {
                    if foundInode.StableAttr().IsDir() {
                        return Err(Error::SysError(SysErr::EISDIR))
                    }
                    



                    foundInode.Truncate(task, &found, 0)?;
                }

                let inode = found.Inode();
                let newFile = match inode.GetFile(task, &found, &fileFlags) {
                    Err(e) => {
                        return Err(ConvertIntr(e, Error::SysError(SysErr::ERESTARTSYS)))
                    },
                    Ok(f) => {
                        f
                    },
                };

                newFile
            }
            Error::SysError(SysErr::ENOENT) |
            //todo: this is a workaround, we can only get EPERM failure instead of ENOENT, fix this later
            Error::SysError(SysErr::EPERM) => {
                // File does not exist. Proceed with creation.

                // Do we have write permissions on the parent?
                let parentInode = parent.Inode();
                parentInode.CheckPermission(task, &PermMask {
                    write: true,
                    execute: true,
                    ..Default::default()
                })?;

                let perms = FilePermissions::FromMode(FileMode(mode.0 & !task.Umask() as u16));
                let newFile = parent.Create(task, root, &name, &fileFlags, &perms)?;
                //found = newFile.lock().Dirent.clone()();
                newFile
            }
            e => return Err(e)
        };

        let newFd = task.NewFDFrom(
            0,
            &newFile,
            &FDFlags {
                CloseOnExec: flags & Flags::O_CLOEXEC as u32 != 0,
            },
        )?;

        fd = newFd;

        // Queue the open inotify event. The creation event is
        // automatically queued when the dirent is found. The open
        // events are implemented at the syscall layer so we need to
        // manually queue one here.
        let newDirent = newFile.Dirent.clone();
        newDirent.InotifyEvent(InotifyEvent::IN_OPEN, 0, EventType::InodeEvent);
        //found.InotifyEvent(InotifyEvent::IN_OPEN, 0);

        return Ok(());
    })?;

    return Ok(fd);
}

pub fn SysAccess(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let pathName = args.arg0 as u64;
    let mode = args.arg1 as u32;

    accessAt(task, ATType::AT_FDCWD, pathName, mode)?;
    return Ok(0);
}

pub fn SysFaccessat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let dirfd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let mode = args.arg2 as u16 as u32;

    accessAt(task, dirfd, addr, mode)?;
    return Ok(0);
}

pub fn accessAt(task: &Task, dirFd: i32, addr: u64, mode: u32) -> Result<()> {
    const R_OK: u32 = 4;
    const W_OK: u32 = 2;
    const X_OK: u32 = 1;

    let (path, _) = copyInPath(task, addr, false)?;

    info!("accessAt dirfd is {}, path is {}", dirFd, &path);
    if mode & !(R_OK | W_OK | X_OK) != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    return fileOpOn(
        task,
        dirFd,
        &path.to_string(),
        true,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            {
                let creds = task.Creds().Fork();
                let mut creds = creds.lock();

                creds.EffectiveKUID = creds.RealKUID;
                creds.EffectiveKGID = creds.RealKGID;

                if creds.EffectiveKGID.In(&creds.UserNamespace).0 == ROOT_UID.0 {
                    creds.EffectiveCaps = creds.PermittedCaps
                } else {
                    creds.EffectiveCaps = CapSet::New(0)
                }
            }

            let inode = d.Inode();
            return inode.CheckPermission(
                task,
                &PermMask {
                    read: mode & R_OK != 0,
                    write: mode & W_OK != 0,
                    execute: mode & X_OK != 0,
                },
            );
        },
    );
}

pub fn SysIoctl(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let request = args.arg1 as u64;
    let val = args.arg2 as u64;

    Ioctl(task, fd, request, val)?;
    return Ok(0);
}

pub fn Ioctl(task: &mut Task, fd: i32, request: u64, val: u64) -> Result<u64> {
    let file = task.GetFile(fd)?;

    if file.Flags().Path {
        return Err(Error::SysError(SysErr::EBADF));
    }

    //let fops = file.FileOp.clone();
    //let inode = file.Dirent.Inode();
    //error!("Ioctl inodetype is {:?}, fopstype is {:?}", inode.InodeType(), fops.FopsType());

    match request {
        IoCtlCmd::FIONCLEX => {
            task.SetFlags(fd, &FDFlags { CloseOnExec: false })?;

            return Ok(0);
        }
        IoCtlCmd::FIOCLEX => {
            task.SetFlags(fd, &FDFlags { CloseOnExec: true })?;

            return Ok(0);
        }
        IoCtlCmd::FIONBIO => {
            let set: u32 = task.CopyInObj(val)?;

            let mut flags = file.Flags();

            if set != 0 {
                flags.NonBlocking = true;
            } else {
                flags.NonBlocking = false;
            }

            file.SetFlags(task, flags.SettableFileFlags());
            return Ok(0);
        }
        IoCtlCmd::FIOASYNC => {
            let set: u32 = task.CopyInObj(val)?;

            let mut flags = file.Flags();

            if set != 0 {
                flags.Async = true;
            } else {
                flags.Async = false;
            }

            file.SetFlags(task, flags.SettableFileFlags());
            return Ok(0);
        }
        IoCtlCmd::FIOSETOWN | IoCtlCmd::SIOCSPGRP => {
            let set: i32 = task.CopyInObj(val)?;
            FSetOwner(task, fd, &file, set)?;
            return Ok(0);
        }
        IoCtlCmd::FIOGETOWN | IoCtlCmd::SIOCGPGRP => {
            let who = FGetOwn(task, &file);
            //*task.GetTypeMut(val)? = who;
            task.CopyOutObj(&who, val)?;
            return Ok(0);
        }
        _ => return file.Ioctl(task, fd, request, val),
    }
}

pub fn SysGetcwd(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let size = args.arg1 as usize;

    let cwd = task.Workdir();
    let root = task.Root();

    let (mut s, reachable) = cwd.FullName(&root);
    if !reachable {
        s = "(unreachable)".to_string() + &s
    }

    if s.len() >= size {
        return Err(Error::SysError(SysErr::ERANGE));
    }

    let len = if s.len() + 1 > size {
        size
    } else {
        s.len() + 1
    };

    task.CopyOutString(addr, len, &s)?;
    return Ok(len as i64);
}

pub fn SysChroot(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;

    if !task.Creds().HasCapability(Capability::CAP_SYS_CHROOT) {
        return Err(Error::SysError(SysErr::EPERM));
    }

    let (path, _) = copyInPath(task, addr, false)?;
    let mut dir = task.Root();

    let res = fileOpOn(
        task,
        ATType::AT_FDCWD,
        &path,
        true,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            let inode = d.Inode();
            if !inode.StableAttr().IsDir() {
                return Err(Error::SysError(SysErr::ENOTDIR));
            }

            inode.CheckPermission(
                task,
                &PermMask {
                    execute: true,
                    ..Default::default()
                },
            )?;

            dir = d.clone();

            Ok(())
        },
    );

    match res {
        Err(e) => return Err(e),
        Ok(_) => {
            task.fsContext.SetRootDirectory(&dir);
            return Ok(0);
        }
    }
}

pub fn SysChdir(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;

    let (path, _) = copyInPath(task, addr, false)?;
    info!("SysChdir path is {}", &path);
    let mut dir = task.Workdir();

    let res = fileOpOn(
        task,
        ATType::AT_FDCWD,
        &path,
        true,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            let inode = d.Inode();
            if !inode.StableAttr().IsDir() {
                return Err(Error::SysError(SysErr::ENOTDIR));
            }

            inode.CheckPermission(
                task,
                &PermMask {
                    execute: true,
                    ..Default::default()
                },
            )?;

            dir = d.clone();

            Ok(())
        },
    );

    match res {
        Err(e) => Err(e),
        Ok(_) => {
            task.fsContext.SetWorkDirectory(&dir);
            return Ok(0);
        }
    }
}

pub fn SysFchdir(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let file = task.GetFile(fd)?;

    let dirent = file.Dirent.clone();
    info!("SysFchdir dir is {}", dirent.MyFullName());
    let inode = dirent.Inode();

    if !inode.StableAttr().IsDir() {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    inode.CheckPermission(
        task,
        &PermMask {
            execute: true,
            ..Default::default()
        },
    )?;

    task.fsContext.SetWorkDirectory(&dirent);

    return Ok(0);
}

// CloseRange implements linux syscall close_range(2).
pub fn SysCloseRange(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let first = args.arg0 as i32;
    let last = args.arg1 as i32;
    let flags = args.arg2 as i32;

    if first < 0 || last < 0 || first > last {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if flags & !(Cmd::CLOSE_RANGE_CLOEXEC | Cmd::CLOSE_RANGE_UNSHARE) != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let cloexec = flags & Cmd::CLOSE_RANGE_CLOEXEC != 0;
    let unshare = flags & Cmd::CLOSE_RANGE_UNSHARE != 0;

    if unshare {
        // If possible, we don't want to copy FDs to the new unshared table, because those FDs will
        // be promptly closed and no longer used. So in the case where we know the range extends all
        // the way to the end of the FdTable, we can simply copy the FdTable only up to the start of
        // the range that we are closing.
        let lastfd = task.fdTbl.GetLastFd();
        if !cloexec && last > lastfd {
            task.UnshareFdTable(first);
        } else {
            task.UnshareFdTable(i32::MAX);
        }
    }

    if cloexec {
        let flagToApply = FDFlags { CloseOnExec: true };

        task.fdTbl.SetFlagsForRange(first, last + 1, flagToApply)?;
        return Ok(0);
    }

    let files = task.fdTbl.RemoveRange(first, last + 1);
    for f in files {
        match f.Flush(task) {
            Ok(_) => (),
            Err(_) => {
                // Per the close_range(2) documentation, errors upon closing file descriptors are ignored.
            }
        }
    }

    return Ok(0);
}

// Close implements linux syscall close(2).
pub fn SysClose(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    close(task, fd)?;
    return Ok(0);
}

pub fn close(task: &Task, fd: i32) -> Result<()> {
    let file = task.RemoveFile(fd)?;

    file.Flush(task)?;
    Ok(())
}

pub fn SysDup(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;

    let file = task.GetFile(fd)?;

    let newfd = task.NewFDFrom(0, &file, &FDFlags::default())?;
    return Ok(newfd as i64);
}

pub fn SysDup2(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let oldfd = args.arg0 as i32;
    let newfd = args.arg1 as i32;

    if oldfd == newfd {
        let _oldfile = task.GetFile(oldfd)?;
        return Ok(newfd as i64);
    }

    return Dup3(task, oldfd, newfd, 0);
}

pub fn SysDup3(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let oldfd = args.arg0 as i32;
    let newfd = args.arg1 as i32;
    let flags = args.arg2 as u32;

    if oldfd == newfd {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    return Dup3(task, oldfd, newfd, flags);
}

pub fn Dup3(task: &mut Task, oldfd: i32, newfd: i32, flags: u32) -> Result<i64> {
    let oldFile = task.GetFile(oldfd)?;

    task.NewFDAt(
        newfd,
        &oldFile,
        &FDFlags {
            CloseOnExec: flags & Flags::O_CLOEXEC as u32 != 0,
        },
    )?;

    return Ok(newfd as i64);
}

pub fn SysLseek(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let offset = args.arg1 as i64;
    let whence = args.arg2 as i32;
    let res = Lseek(task, fd, offset, whence)?;
    return Ok(res as i64);
}

pub fn Lseek(task: &mut Task, fd: i32, offset: i64, whence: i32) -> Result<i64> {
    let file = task.GetFile(fd)?;

    if whence < SeekWhence::SEEK_SET || whence > SeekWhence::SEEK_END {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if file.Flags().Path {
        return Err(Error::SysError(SysErr::EBADF));
    }

    let res = file.Seek(task, whence, offset);
    return res;
}

pub fn FGetOwnEx(task: &mut Task, file: &File) -> FOwnerEx {
    let ma = match file.Async(task, None) {
        None => return FOwnerEx::default(),
        Some(a) => a,
    };

    match ma.Owner() {
        Recipient::PG(processgroup) => {
            let processgroup = match processgroup.Upgrade() {
                None => return FOwnerEx::default(),
                Some(pg) => pg,
            };
            return FOwnerEx {
                Type: F_OWNER_PGRP,
                PID: task.Thread().PIDNamespace().IDOfProcessGroup(&processgroup),
            };
        }
        Recipient::TG(threadgroup) => {
            let threadgroup = match threadgroup.Upgrade() {
                None => return FOwnerEx::default(),
                Some(tg) => tg,
            };
            return FOwnerEx {
                Type: F_OWNER_PID,
                PID: task.Thread().PIDNamespace().IDOfThreadGroup(&threadgroup),
            };
        }
        Recipient::Thread(thread) => {
            let thread = match thread.Upgrade() {
                None => return FOwnerEx::default(),
                Some(t) => t,
            };
            return FOwnerEx {
                Type: F_OWNER_TID,
                PID: task.Thread().PIDNamespace().IDOfTask(&thread),
            };
        }
        Recipient::None => {
            return FOwnerEx::default();
        }
    }
}

pub fn FGetOwn(task: &mut Task, file: &File) -> i32 {
    let owner = FGetOwnEx(task, file);
    if owner.Type == F_OWNER_PGRP {
        return -owner.PID;
    }

    return owner.PID;
}

// fSetOwn sets the file's owner with the semantics of F_SETOWN in Linux.
//
// If who is positive, it represents a PID. If negative, it represents a PGID.
// If the PID or PGID is invalid, the owner is silently unset.
pub fn FSetOwner(task: &Task, fd: i32, file: &File, who: i32) -> Result<()> {
    // F_SETOWN flips the sign of negative values, an operation that is guarded
    // against overflow.
    if who == core::i32::MIN {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let a = file.Async(task, Some(FileAsync::New(fd))).unwrap();
    if who == 0 {
        a.Unset(task);
        return Ok(());
    }

    if who < 0 {
        let pg = task.Thread().PIDNamespace().ProcessGroupWithID(-who);
        if pg.is_none() {
            return Err(Error::SysError(SysErr::ESRCH));
        }

        a.SetOwnerProcessGroup(task, pg);
        return Ok(());
    }

    let tg = task.Thread().PIDNamespace().ThreadGroupWithID(who);
    if tg.is_none() {
        return Err(Error::SysError(SysErr::ESRCH));
    }

    a.SetOwnerThreadGroup(task, tg);
    return Ok(());
}

pub fn PosixLock(task: &Task, flockAddr: u64, file: &File, block: bool) -> Result<()> {
    let inode = file.Dirent.Inode();
    // In Linux the file system can choose to provide lock operations for an inode.
    // Normally pipe and socket types lack lock operations. We diverge and use a heavy
    // hammer by only allowing locks on files and directories.
    /*if !inode.StableAttr().IsFile() && !inode.StableAttr().IsDir() {
        return Err(Error::SysError(SysErr::EBADF));
    }*/

    let flock: Flock = task.CopyInObj(flockAddr)?;

    let rng = file.ComputeLockRange(task, flock.Start, flock.Len, flock.Whence as _)?;

    // The lock uid is that of the fdtble's UniqueId.
    let lockUniqueID = task.fdTbl.Id();

    // These locks don't block; execute the non-blocking operation using the inode's lock
    // context directly.
    let fflags = file.Flags();

    let pid = task.Thread().ThreadGroup().ID();
    match flock.Type as u64 {
        LibcConst::F_RDLCK => {
            if !fflags.Read {
                return Err(Error::SysError(SysErr::EBADF));
            }

            let lock = inode.lock().LockCtx.Posix.clone();
            if !lock.LockRegion(
                task,
                lockUniqueID,
                OwnerInfo::New(pid),
                LockType::ReadLock,
                &rng,
                block,
            )? {
                return Err(Error::SysError(SysErr::EAGAIN));
            }

            return Ok(());
        }
        LibcConst::F_WRLCK => {
            if !fflags.Write {
                return Err(Error::SysError(SysErr::EBADF));
            }

            let lock = inode.lock().LockCtx.Posix.clone();
            if !lock.LockRegion(
                task,
                lockUniqueID,
                OwnerInfo::New(pid),
                LockType::WriteLock,
                &rng,
                block,
            )? {
                return Err(Error::SysError(SysErr::EAGAIN));
            }

            return Ok(());
        }
        LibcConst::F_UNLCK => {
            let lock = inode.lock().LockCtx.Posix.clone();
            lock.UnlockRegion(task, lockUniqueID, &rng);

            return Ok(());
        }
        _ => return Err(Error::SysError(SysErr::EINVAL)),
    }
}

pub fn PosixTestLock(task: &Task, flockAddr: u64, file: &File) -> Result<()> {
    let flock: Flock = task.CopyInObj(flockAddr)?;

    let typ = match flock.Type as i32 {
        F_RDLCK => LockType::ReadLock,
        F_WRLCK => LockType::WriteLock,
        _ => return Err(Error::SysError(SysErr::EINVAL)),
    };

    let r = file.ComputeLockRange(task, flock.Start, flock.Len, flock.Whence as _)?;

    // The lock uid is that of the fdtble's UniqueId.
    let lockUniqueID = task.fdTbl.Id();
    let inode = file.Dirent.Inode();
    let lock = inode.lock().LockCtx.Posix.clone();
    let newFlock = lock.TestRegion(task, lockUniqueID, typ, &r);

    task.CopyOutObj(&newFlock, flockAddr)?;
    return Ok(());
}

pub fn SysFcntl(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let cmd = args.arg1 as i32;
    let val = args.arg2 as u64;

    let (file, flags) = task.GetFileAll(fd)?;

    match cmd {
        Cmd::F_DUPFD | Cmd::F_DUPFD_CLOEXEC => {
            let from = val as i32;
            let fd = task.NewFDFrom(
                from,
                &file,
                &FDFlags {
                    CloseOnExec: cmd == Cmd::F_DUPFD_CLOEXEC,
                },
            )?;
            return Ok(fd as i64);
        }
        Cmd::F_GETFD => Ok(flags.ToLinuxFDFlags() as i64),
        Cmd::F_SETFD => {
            let flags = val as u32;
            task.SetFlags(
                fd,
                &FDFlags {
                    CloseOnExec: flags & LibcConst::FD_CLOEXEC as u32 != 0,
                },
            )?;
            Ok(0)
        }
        Cmd::F_GETFL => Ok(file.Flags().ToLinux() as i64),
        Cmd::F_SETFL => {
            if file.Flags().Path {
                return Err(Error::SysError(SysErr::EBADF));
            }

            let flags = val as u32;
            file.SetFlags(task, FileFlags::FromFlags(flags).SettableFileFlags());
            Ok(0)
        }
        Cmd::F_SETLK => {
            if file.Flags().Path {
                return Err(Error::SysError(SysErr::EBADF));
            }

            PosixLock(task, val, &file, false)?;
            return Ok(0);
        }
        Cmd::F_SETLKW => {
            if file.Flags().Path {
                return Err(Error::SysError(SysErr::EBADF));
            }

            PosixLock(task, val, &file, true)?;
            return Ok(0);
        }
        Cmd::F_GETLK => {
            if file.Flags().Path {
                return Err(Error::SysError(SysErr::EBADF));
            }

            PosixTestLock(task, val, &file)?;
            return Ok(0);
        }
        Cmd::F_GETOWN => {
            if file.Flags().Path {
                return Err(Error::SysError(SysErr::EBADF));
            }
            return Ok(FGetOwn(task, &file) as i64);
        }
        Cmd::F_SETOWN => {
            if file.Flags().Path {
                return Err(Error::SysError(SysErr::EBADF));
            }
            FSetOwner(task, fd, &file, val as i32)?;
            return Ok(0);
        }
        Cmd::F_GETOWN_EX => {
            if file.Flags().Path {
                return Err(Error::SysError(SysErr::EBADF));
            }
            let addr = val;
            let owner = FGetOwnEx(task, &file);
            //*task.GetTypeMut(addr)? = owner;
            task.CopyOutObj(&owner, addr)?;
            return Ok(0);
        }
        Cmd::F_SETOWN_EX => {
            if file.Flags().Path {
                return Err(Error::SysError(SysErr::EBADF));
            }
            let addr = val;
            let owner: FOwnerEx = task.CopyInObj(addr)?;
            let a = file.Async(task, Some(FileAsync::New(fd))).unwrap();

            match owner.Type {
                F_OWNER_TID => {
                    if owner.PID == 0 {
                        a.Unset(task);
                        return Ok(0);
                    }

                    let thread = task.Thread().PIDNamespace().TaskWithID(owner.PID);
                    match thread {
                        None => return Err(Error::SysError(SysErr::ESRCH)),
                        Some(thread) => {
                            a.SetOwnerTask(task, Some(thread));
                            return Ok(0);
                        }
                    }
                }
                F_OWNER_PID => {
                    if owner.PID == 0 {
                        a.Unset(task);
                        return Ok(0);
                    }

                    let tg = task.Thread().PIDNamespace().ThreadGroupWithID(owner.PID);
                    match tg {
                        None => return Err(Error::SysError(SysErr::ESRCH)),
                        Some(tg) => {
                            a.SetOwnerThreadGroup(task, Some(tg));
                            return Ok(0);
                        }
                    }
                }
                F_OWNER_PGRP => {
                    if owner.PID == 0 {
                        a.Unset(task);
                        return Ok(0);
                    }

                    let pg = task.Thread().PIDNamespace().ProcessGroupWithID(owner.PID);
                    match pg {
                        None => return Err(Error::SysError(SysErr::ESRCH)),
                        Some(pg) => {
                            a.SetOwnerProcessGroup(task, Some(pg));
                            return Ok(0);
                        }
                    }
                }
                _ => return Err(Error::SysError(SysErr::EINVAL)),
            }
        }
        Cmd::F_GET_SEALS => {
            let iops = file.Dirent.Inode().lock().InodeOp.clone();
            if let Some(ops) = iops.as_any().downcast_ref::<HostInodeOp>() {
                let hostf = ops.HostFd();
                let ret = Fcntl(hostf, cmd, val);
                if ret < 0 {
                    return Err(Error::SysError(-ret as _));
                }

                return Ok(ret);
            }

            return Err(Error::SysError(SysErr::EINVAL));
        }
        Cmd::F_ADD_SEALS => {
            let iops = file.Dirent.Inode().lock().InodeOp.clone();
            if let Some(ops) = iops.as_any().downcast_ref::<HostInodeOp>() {
                let hostf = ops.HostFd();
                let ret = Fcntl(hostf, cmd, val);
                if ret < 0 {
                    return Err(Error::SysError(-ret as _));
                }

                return Ok(ret);
            }

            return Err(Error::SysError(SysErr::EINVAL));
        }
        Cmd::F_GETPIPE_SZ => {
            let mut fops = file.FileOp.clone();

            if fops.FopsType() == FileOpsType::OverlayFileOperations {
                fops = if let Some(ops) = fops.OverlayFileOperations() {
                    ops.into()
                } else {
                    panic!("F_GETPIPE_SZ OverlayFileOperations fail");
                }
            };

            let pipe = if let Some(ops) = fops.as_any().downcast_ref::<Reader>() {
                ops.pipe.clone()
            } else if let Some(ops) = fops.as_any().downcast_ref::<Writer>() {
                ops.pipe.clone()
            } else if let Some(ops) = fops.as_any().downcast_ref::<ReaderWriter>() {
                ops.pipe.clone()
            } else {
                return Err(Error::SysError(SysErr::EINVAL));
            };

            let n = pipe.PipeSize();
            return Ok(n as i64);
        }
        Cmd::F_SETPIPE_SZ => {
            let mut fops = file.FileOp.clone();

            if fops.FopsType() == FileOpsType::OverlayFileOperations {
                fops = if let Some(ops) = fops.OverlayFileOperations() {
                    ops.into()
                } else {
                    panic!("F_SETPIPE_SZ OverlayFileOperations fail");
                }
            };

            let pipe = if let Some(ops) = fops.as_any().downcast_ref::<Reader>() {
                ops.pipe.clone()
            } else if let Some(ops) = fops.as_any().downcast_ref::<Writer>() {
                ops.pipe.clone()
            } else if let Some(ops) = fops.as_any().downcast_ref::<ReaderWriter>() {
                ops.pipe.clone()
            } else {
                return Err(Error::SysError(SysErr::EINVAL));
            };

            let n = pipe.SetPipeSize(val as i64)?;
            return Ok(n as i64);
        }
        Cmd::F_SETSIG => {
            if file.Flags().Path {
                return Err(Error::SysError(SysErr::EBADF));
            }
            match file.Async(task, Some(FileAsync::New(fd))) {
                None => return Ok(0),
                Some(async) => {
                    async.SetSignal(val as i32)?;
                    return Ok(0);
                }
            }
        }
        Cmd::F_GETSIG => {
            if file.Flags().Path {
                return Err(Error::SysError(SysErr::EBADF));
            }
            match file.Async(task, Some(FileAsync::New(fd))) {
                None => return Ok(0),
                Some(async) => {
                    return Ok(async.Signal().0 as i64);
                }
            }
        }
        _ => return Err(Error::SysError(SysErr::EINVAL)),
    }
}

const _FADV_NORMAL: i32 = 0;
const _FADV_RANDOM: i32 = 1;
const _FADV_SEQUENTIAL: i32 = 2;
const _FADV_WILLNEED: i32 = 3;
const _FADV_DONTNEED: i32 = 4;
const _FADV_NOREUSE: i32 = 5;

pub fn SysFadvise64(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let _offset = args.arg1 as i64;
    let len = args.arg2 as i64;
    let advice = args.arg3 as i32;

    if len < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let file = task.GetFile(fd)?;
    if file.Flags().Path {
        return Err(Error::SysError(SysErr::EBADF));
    }

    let inode = file.Dirent.Inode();
    if inode.StableAttr().IsPipe() {
        return Err(Error::SysError(SysErr::ESPIPE));
    }

    match advice {
        _FADV_NORMAL | _FADV_RANDOM | _FADV_SEQUENTIAL | _FADV_WILLNEED | _FADV_DONTNEED
        | _FADV_NOREUSE => return Ok(0),
        _ => return Err(Error::SysError(SysErr::EINVAL)),
    }
}

fn mkdirAt(task: &Task, dirFd: i32, addr: u64, mode: FileMode) -> Result<i64> {
    let (path, _) = copyInPath(task, addr, false)?;
    info!("mkdirAt path is {}", &path);

    fileOpAt(
        task,
        dirFd,
        &path.to_string(),
        &mut |root: &Dirent, d: &Dirent, name: &str, _remainingTraversals: u32| -> Result<()> {
            let inode = d.Inode();
            if !inode.StableAttr().IsDir() {
                return Err(Error::SysError(SysErr::ENOTDIR));
            }

            let mut remainingTraversals = MAX_SYMLINK_TRAVERSALS;
            let res = task.mountNS.FindDirent(
                task,
                root,
                Some(d.clone()),
                name,
                &mut remainingTraversals,
                true,
            );

            match res {
                Ok(_) => Err(Error::SysError(SysErr::EEXIST)),
                Err(Error::SysError(SysErr::EACCES)) => {
                    return Err(Error::SysError(SysErr::EACCES))
                }
                _ => {
                    let perms = {
                        inode.CheckPermission(
                            task,
                            &PermMask {
                                write: true,
                                execute: true,
                                ..Default::default()
                            },
                        )?;

                        FilePermissions::FromMode(FileMode(mode.0 & !task.Umask() as u16))
                    };

                    return d.CreateDirectory(task, root, name, &perms);
                }
            }
        },
    )?;

    return Ok(0);
}

pub fn SysMkdir(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let mode = args.arg1 as u16;

    return mkdirAt(task, ATType::AT_FDCWD, addr, FileMode(mode));
}

pub fn SysMkdirat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let dirfd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let mode = args.arg2 as u16;

    return mkdirAt(task, dirfd, addr, FileMode(mode));
}

fn rmdirAt(task: &Task, dirFd: i32, addr: u64) -> Result<i64> {
    let (path, _) = copyInPath(task, addr, false)?;
    info!("rmdirAt path is {}", &path);

    if path.as_str() == "/" {
        return Err(Error::SysError(SysErr::EBUSY));
    }

    fileOpAt(
        task,
        dirFd,
        &path.to_string(),
        &mut |root: &Dirent, d: &Dirent, name: &str, _remainingTraversals: u32| -> Result<()> {
            let inode = d.Inode();
            if !inode.StableAttr().IsDir() {
                return Err(Error::SysError(SysErr::ENOTDIR));
            }

            match name {
                "." => return Err(Error::SysError(SysErr::EINVAL)),
                ".." => return Err(Error::SysError(SysErr::ENOTEMPTY)),
                _ => (),
            }

            d.MayDelete(task, root, name)?;

            return d.RemoveDirectory(task, root, name);
        },
    )?;

    return Ok(0);
}

pub fn SysRmdir(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    return rmdirAt(task, ATType::AT_FDCWD, addr);
}

fn symlinkAt(task: &Task, newAddr: u64, dirFd: i32, oldAddr: u64) -> Result<u64> {
    let (newPath, dirPath) = copyInPath(task, newAddr, false)?;
    if dirPath {
        return Err(Error::SysError(SysErr::ENOENT));
    }

    let (oldPath, err) = task.CopyInString(oldAddr, PATH_MAX);
    match err {
        Err(e) => return Err(e),
        _ => (),
    }

    if oldPath.as_str() == "" {
        return Err(Error::SysError(SysErr::ENOENT));
    }

    info!("symlinkAt newpath is {}, oldpath is {}", &newPath, &oldPath);

    fileOpAt(
        task,
        dirFd,
        &newPath.to_string(),
        &mut |root: &Dirent, d: &Dirent, name: &str, _remainingTraversals: u32| -> Result<()> {
            let inode = d.Inode();
            if !inode.StableAttr().IsDir() {
                return Err(Error::SysError(SysErr::ENOTDIR));
            }

            inode.CheckPermission(
                task,
                &PermMask {
                    write: true,
                    execute: true,
                    ..Default::default()
                },
            )?;

            return d.CreateLink(task, root, &oldPath, name);
        },
    )?;

    return Ok(0);
}

pub fn SysSymlink(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let oldAddr = args.arg0 as u64;
    let newAddr = args.arg1 as u64;

    symlinkAt(task, newAddr, ATType::AT_FDCWD, oldAddr)?;
    return Ok(0);
}

pub fn SysSymlinkat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let oldAddr = args.arg0 as u64;
    let dirfd = args.arg1 as i32;
    let newAddr = args.arg2 as u64;

    symlinkAt(task, newAddr, dirfd, oldAddr)?;
    return Ok(0);
}

fn mayLinkAt(task: &Task, target: &mut Inode) -> Result<()> {
    if target.CheckOwnership(task) {
        return Ok(());
    }

    if !target.StableAttr().IsRegular() {
        return Err(Error::SysError(SysErr::EPERM));
    }

    let res = target.CheckPermission(
        task,
        &PermMask {
            write: true,
            execute: true,
            ..Default::default()
        },
    );

    match res {
        Err(_) => Err(Error::SysError(SysErr::EPERM)),
        _ => Ok(()),
    }
}

fn linkAt(
    task: &Task,
    oldDirfd: i32,
    oldAddr: u64,
    newDirfd: i32,
    newAddr: u64,
    resolve: bool,
    allowEmpty: bool,
) -> Result<i64> {
    let (oldPath, _) = copyInPath(task, oldAddr, allowEmpty)?;
    let (newPath, dirPath) = copyInPath(task, newAddr, false)?;

    if dirPath {
        return Err(Error::SysError(SysErr::ENOENT));
    }

    if allowEmpty && oldPath == "" {
        let target = task.GetFile(oldDirfd)?;

        let mut inode = target.Dirent.Inode();
        mayLinkAt(task, &mut inode)?;

        fileOpAt(
            task,
            newDirfd,
            &newPath.to_string(),
            &mut |root: &Dirent,
                  newParent: &Dirent,
                  name: &str,
                  _remainingTraversals: u32|
             -> Result<()> {
                let inode = newParent.Inode();
                if !inode.StableAttr().IsDir() {
                    return Err(Error::SysError(SysErr::ENOTDIR));
                }

                inode.CheckPermission(
                    task,
                    &PermMask {
                        write: true,
                        execute: true,
                        ..Default::default()
                    },
                )?;

                return newParent.CreateHardLink(task, root, &target.Dirent.clone(), name);
            },
        )?;

        return Ok(0);
    };

    fileOpOn(
        task,
        oldDirfd,
        &oldPath.to_string(),
        resolve,
        &mut |_root: &Dirent, target: &Dirent, _remainingTraversals: u32| -> Result<()> {
            let mut inode = target.Inode();
            mayLinkAt(task, &mut inode)?;

            return fileOpAt(
                task,
                newDirfd,
                &newPath.to_string(),
                &mut |root: &Dirent,
                      newParent: &Dirent,
                      newName: &str,
                      _remainingTraversals: u32|
                 -> Result<()> {
                    let inode = newParent.Inode();
                    if !inode.StableAttr().IsDir() {
                        return Err(Error::SysError(SysErr::ENOTDIR));
                    }

                    inode.CheckPermission(
                        task,
                        &PermMask {
                            write: true,
                            execute: true,
                            ..Default::default()
                        },
                    )?;

                    return newParent.CreateHardLink(task, root, target, newName);
                },
            );
        },
    )?;

    return Ok(0);
}

pub fn SysLink(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let oldAddr = args.arg0 as u64;
    let newAddr = args.arg1 as u64;

    let resolve = false;
    return linkAt(
        task,
        ATType::AT_FDCWD,
        oldAddr,
        ATType::AT_FDCWD,
        newAddr,
        resolve,
        false,
    );
}

pub fn SysLinkat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let oldDirfd = args.arg0 as i32;
    let oldAddr = args.arg1 as u64;
    let newDirfd = args.arg2 as i32;
    let newAddr = args.arg3 as u64;

    // man linkat(2):
    // By default, linkat(), does not dereference oldpath if it is a
    // symbolic link (like link(2)). Since Linux 2.6.18, the flag
    // AT_SYMLINK_FOLLOW can be specified in flags to cause oldpath to be
    // dereferenced if it is a symbolic link.
    let flags = args.arg4 as i32;

    if flags & !(ATType::AT_SYMLINK_FOLLOW | ATType::AT_EMPTY_PATH) != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let resolve = flags & ATType::AT_SYMLINK_FOLLOW == ATType::AT_SYMLINK_FOLLOW;
    let allowEmpty = flags & ATType::AT_EMPTY_PATH == ATType::AT_EMPTY_PATH;

    {
        let creds = task.Creds();
        let userNS = creds.lock().UserNamespace.clone();
        if allowEmpty && !creds.HasCapabilityIn(Capability::CAP_DAC_READ_SEARCH, &userNS) {
            return Err(Error::SysError(SysErr::ENOENT));
        }
    }

    return linkAt(
        task, oldDirfd, oldAddr, newDirfd, newAddr, resolve, allowEmpty,
    );
}

fn readlinkAt(task: &Task, dirFd: i32, addr: u64, bufAddr: u64, size: u32) -> Result<i64> {
    let (path, dirPath) = copyInPath(task, addr, false)?;
    if dirPath {
        return Err(Error::SysError(SysErr::ENOENT));
    }

    info!("readlinkAt path is {}", &path);
    let mut copied = 0;
    let size = size as usize;

    fileOpOn(
        task,
        dirFd,
        &path,
        false,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            let inode = d.Inode();
            inode.CheckPermission(
                task,
                &PermMask {
                    read: true,
                    ..Default::default()
                },
            )?;

            let s = match inode.ReadLink(task) {
                Err(Error::SysError(SysErr::ENOLINK)) => {
                    return Err(Error::SysError(SysErr::EINVAL))
                }
                Err(Error::SysError(SysErr::ENOENT)) => {
                    // there is such interesting result happen when running mariadb with "/tmp" folder
                    // work around this now
                    // todo: find better solution later
                    return Err(Error::SysError(SysErr::EINVAL));
                }
                Err(e) => return Err(e),
                Ok(s) => s,
            };

            info!("readlinkAt 1 path is {}, target is {}", &path, &s);
            let mut buffer = s.as_bytes();
            if buffer.len() > size {
                buffer = &buffer[..size]
            }

            task.CopyOutSlice(buffer, bufAddr, buffer.len())?;
            copied = buffer.len();
            Ok(())
        },
    )?;

    return Ok(copied as i64);
}

// Readlink implements linux syscall readlink(2).
pub fn SysReadLink(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let bufAddr = args.arg1 as u64;
    let size = args.arg2 as u64;

    let size = size as u32;
    return readlinkAt(task, ATType::AT_FDCWD, addr, bufAddr, size);
}

// Readlinkat implements linux syscall readlinkat(2).
pub fn SysReadLinkAt(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let dirfd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let bufAddr = args.arg2 as u64;
    let size = args.arg3 as u64;

    let size = size as u32;
    return readlinkAt(task, dirfd, addr, bufAddr, size);
}

pub fn ReadLinkAt(task: &Task, dirFd: i32, addr: u64, bufAddr: u64, size: u64) -> Result<i64> {
    let size = size as u32;

    return readlinkAt(task, dirFd, addr, bufAddr, size);
}

fn unlinkAt(task: &Task, dirFd: i32, addr: u64) -> Result<i64> {
    let (path, dirPath) = copyInPath(task, addr, false)?;

    info!("unlinkAt path is {}", &path);
    if dirPath {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fileOpAt(
        task,
        dirFd,
        &path.to_string(),
        &mut |root: &Dirent, d: &Dirent, name: &str, _remainingTraversals: u32| -> Result<()> {
            let inode = d.Inode();
            if !inode.StableAttr().IsDir() {
                return Err(Error::SysError(SysErr::ENOTDIR));
            }

            d.MayDelete(task, root, name)?;

            return d.Remove(task, root, name, dirPath);
        },
    )?;

    return Ok(0);
}

pub fn SysUnlink(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;

    return unlinkAt(task, ATType::AT_FDCWD, addr);
}

pub fn SysUnlinkat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let dirfd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let flags = args.arg2 as u32;

    if flags & ATType::AT_REMOVEDIR as u32 != 0 {
        return rmdirAt(task, dirfd, addr);
    }

    return unlinkAt(task, dirfd, addr);
}

pub fn SysTruncate(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let len = args.arg1 as i64;

    if len < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let (path, dirPath) = copyInPath(task, addr, false)?;

    if dirPath {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let rlimitSize = task
        .Thread()
        .ThreadGroup()
        .Limits()
        .Get(LimitType::FileSize)
        .Cur;
    if len as u64 > rlimitSize {
        return Err(Error::ErrExceedsFileSizeLimit);
    }

    fileOpOn(
        task,
        ATType::AT_FDCWD,
        &path,
        true,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            let mut inode = d.Inode();
            if inode.StableAttr().IsDir() {
                return Err(Error::SysError(SysErr::EISDIR));
            }

            if !inode.StableAttr().IsFile() {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            inode.CheckPermission(
                task,
                &PermMask {
                    write: true,
                    ..Default::default()
                },
            )?;

            inode.Truncate(task, d, len)?;

            // File length modified, generate notification.
            d.InotifyEvent(InotifyEvent::IN_MODIFY, 0, EventType::InodeEvent);
            return Ok(());
        },
    )?;

    return Ok(0);
}

pub fn SysFtruncate(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let len = args.arg1 as i64;

    let file = task.GetFile(fd)?;
    if !file.Flags().Write {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let mut inode = file.Dirent.Inode();
    if !inode.StableAttr().IsFile() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if len < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let rlimitSize = task
        .Thread()
        .ThreadGroup()
        .Limits()
        .Get(LimitType::FileSize)
        .Cur;
    if len as u64 > rlimitSize {
        return Err(Error::ErrExceedsFileSizeLimit);
    }

    let dirent = file.Dirent.clone();
    inode.Truncate(task, &dirent, len)?;

    // File length modified, generate notification.
    file.Dirent
        .InotifyEvent(InotifyEvent::IN_MODIFY, 0, EventType::InodeEvent);

    return Ok(0);
}

pub fn SysUmask(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let mask = args.arg0 as u32;
    let mask = task.fsContext.SwapUmask(mask & 0o777);
    return Ok(mask as i64);
}

fn chown(task: &Task, d: &Dirent, uid: UID, gid: GID) -> Result<i64> {
    let mut owner = FileOwner {
        UID: KUID(NO_ID),
        GID: KGID(NO_ID),
    };

    let creds = task.Creds();
    let inode = d.Inode();
    let uattr = inode.UnstableAttr(task)?;
    let hasCap = CheckCapability(&creds, Capability::CAP_CHOWN, &uattr);

    let c = creds.lock();
    let isOwner = uattr.Owner.UID == c.EffectiveKUID;
    if uid.Ok() {
        let kuid = c.UserNamespace.MapToKUID(uid);
        if !kuid.Ok() {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let isNoop = uattr.Owner.UID == kuid;
        if !(hasCap || (isOwner && isNoop)) {
            return Err(Error::SysError(SysErr::EPERM));
        }

        owner.UID = kuid;
    }

    if gid.Ok() {
        let kgid = c.UserNamespace.MapToKGID(gid);
        if !kgid.Ok() {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let isNoop = uattr.Owner.GID == kgid;
        let isMemberGroup = c.InGroup(kgid);
        if !(hasCap || (isOwner && (isNoop || isMemberGroup))) {
            return Err(Error::SysError(SysErr::EPERM));
        }

        owner.GID = kgid;
    }

    // This is racy; the inode's owner may have changed in
    // the meantime. (Linux holds i_mutex while calling
    // fs/attr.c:notify_change() => inode_operations::setattr =>
    // inode_change_ok().)
    info!(
        "workaround enable setowner for host inode, the owner is {:?}",
        &owner
    );
    let mut inode = d.Inode();
    inode.SetOwner(task, d, &owner)?;

    // When the owner or group are changed by an unprivileged user,
    // chown(2) also clears the set-user-ID and set-group-ID bits, but
    // we do not support them.
    return Ok(0);
}

fn chownAt(
    task: &Task,
    fd: i32,
    addr: u64,
    resolve: bool,
    allowEmpty: bool,
    uid: UID,
    gid: GID,
) -> Result<i64> {
    let (path, _) = copyInPath(task, addr, allowEmpty)?;

    if path == "" {
        let file = task.GetFile(fd)?;
        let dirent = file.Dirent.clone();
        chown(task, &dirent, uid, gid)?;
        return Ok(0);
    }

    fileOpOn(
        task,
        fd,
        &path,
        resolve,
        &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
            chown(task, d, uid, gid)?;
            Ok(())
        },
    )?;

    return Ok(0);
}

pub fn SysChown(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let uid = args.arg1 as u32;
    let gid = args.arg2 as u32;

    let uid = UID(uid as u32);
    let gid = GID(gid as u32);

    let ret = chownAt(task, ATType::AT_FDCWD, addr, true, false, uid, gid)?;
    return Ok(ret as i64);
}

pub fn SysLchown(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let addr = args.arg0 as u64;
    let uid = args.arg1 as u32;
    let gid = args.arg2 as u32;

    let uid = UID(uid as u32);
    let gid = GID(gid as u32);

    return chownAt(task, ATType::AT_FDCWD, addr, false, false, uid, gid);
}

pub fn SysFchown(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let uid = UID(args.arg1 as u32);
    let gid = GID(args.arg2 as u32);

    let file = task.GetFile(fd)?;
    if file.Flags().Path {
        return Err(Error::SysError(SysErr::EBADF));
    }
    let dirent = file.Dirent.clone();
    return chown(task, &dirent, uid, gid);
}

pub fn SysFchownat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let dirfd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let uid = UID(args.arg2 as u32);
    let gid = GID(args.arg3 as u32);
    let flags = args.arg4 as i32;

    if flags & !(ATType::AT_EMPTY_PATH | ATType::AT_SYMLINK_NOFOLLOW) != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    return chownAt(
        task,
        dirfd,
        addr,
        flags & ATType::AT_SYMLINK_NOFOLLOW == 0,
        flags & ATType::AT_EMPTY_PATH != 0,
        uid,
        gid,
    );
}

fn utime(task: &Task, dirfd: i32, addr: u64, ts: &InterTimeSpec, resolve: bool) -> Result<i64> {
    let setTimestamp = &mut |_root: &Dirent, d: &Dirent, _remainingTraversals: u32| -> Result<()> {
        let mut inode = d.Inode();
        if !inode.CheckOwnership(task) {
            if (ts.ATimeOmit || !ts.ATimeSetSystemTime) && (ts.MTimeOmit || !ts.MTimeSetSystemTime)
            {
                return Err(Error::SysError(SysErr::EPERM));
            }

            inode.CheckPermission(
                task,
                &PermMask {
                    write: true,
                    ..Default::default()
                },
            )?;
        }

        inode.SetTimestamps(task, d, ts)?;

        // File attribute changed, generate notification.
        if ts.ATimeOmit {
            d.InotifyEvent(InotifyEvent::IN_MODIFY, 0, EventType::InodeEvent);
        } else if ts.MTimeOmit {
            d.InotifyEvent(InotifyEvent::IN_ACCESS, 0, EventType::InodeEvent);
        } else {
            d.InotifyEvent(InotifyEvent::IN_ATTRIB, 0, EventType::InodeEvent);
        }

        return Ok(());
    };

    if addr == 0 && dirfd != ATType::AT_FDCWD {
        if !resolve {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let f = task.GetFile(dirfd)?;

        let root = task.Root().clone();
        setTimestamp(&root, &f.Dirent.clone(), MAX_SYMLINK_TRAVERSALS)?;
        return Ok(0);
    }

    let (path, _) = copyInPath(task, addr, false)?;

    fileOpOn(task, dirfd, &path.to_string(), resolve, setTimestamp)?;
    return Ok(0);
}

pub fn SysUtime(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let filenameAddr = args.arg0 as u64;
    let timesAddr = args.arg1 as u64;

    let mut ts = InterTimeSpec::default();

    if timesAddr != 0 {
        let times: Utime = task.CopyInObj(timesAddr)?;
        ts.ATime = Time::FromSec(times.Actime);
        ts.ATimeSetSystemTime = false;
        ts.MTime = Time::FromSec(times.Modtime);
        ts.MTimeSetSystemTime = false;
    }

    return utime(task, ATType::AT_FDCWD, filenameAddr, &ts, true);
}

pub fn SysUtimes(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let filenameAddr = args.arg0 as u64;
    let timesAddr = args.arg1 as u64;

    let mut ts = InterTimeSpec::default();

    if timesAddr != 0 {
        let times: [Timeval; 2] = task.CopyInObj(timesAddr)?;
        ts.ATime = Time::FromTimeval(&times[0]);
        ts.ATimeSetSystemTime = false;
        ts.MTime = Time::FromTimeval(&times[1]);
        ts.MTimeSetSystemTime = false;
    }

    return utime(task, ATType::AT_FDCWD, filenameAddr, &ts, true);
}

// timespecIsValid checks that the timespec is valid for use in utimensat.
pub fn TimespecIsValid(ts: &Timespec) -> bool {
    return ts.tv_nsec == Utime::UTIME_OMIT
        || ts.tv_nsec == Utime::UTIME_NOW
        || ts.tv_nsec < 1_000_000_000;
}

pub fn SysUtimensat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let dirfd = args.arg0 as i32;
    let filenameAddr = args.arg1 as u64;
    let timesAddr = args.arg2 as u64;
    let flags = args.arg3 as i32;

    let mut ts = InterTimeSpec::default();

    if timesAddr != 0 {
        let times: [Timespec; 2] = task.CopyInObj(timesAddr)?;

        if !TimespecIsValid(&times[0]) || !TimespecIsValid(&times[1]) {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        // If both are UTIME_OMIT, this is a noop.
        if times[0].tv_nsec == Utime::UTIME_OMIT && times[1].tv_nsec == Utime::UTIME_OMIT {
            return Ok(0);
        }

        ts = InterTimeSpec {
            ATime: Time::FromTimespec(&times[0]),
            ATimeOmit: times[0].tv_nsec == Utime::UTIME_OMIT,
            ATimeSetSystemTime: times[0].tv_nsec == Utime::UTIME_NOW,
            MTime: Time::FromTimespec(&times[1]),
            MTimeOmit: times[1].tv_nsec == Utime::UTIME_OMIT,
            MTimeSetSystemTime: times[1].tv_nsec == Utime::UTIME_NOW,
        }
    }

    return utime(
        task,
        dirfd,
        filenameAddr,
        &ts,
        flags & ATType::AT_SYMLINK_NOFOLLOW == 0,
    );
}

pub fn SysFutimesat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let dirfd = args.arg0 as i32;
    let filenameAddr = args.arg1 as u64;
    let timesAddr = args.arg2 as u64;

    let mut ts = InterTimeSpec::default();
    const E6: i64 = 1_000_000;

    if timesAddr != 0 {
        let times: [Timeval; 2] = task.CopyInObj(timesAddr)?;

        if times[0].Usec > E6 || times[0].Usec < 0 || times[1].Usec > E6 || times[1].Usec < 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        ts.ATime = Time::FromTimeval(&times[0]);
        ts.ATimeSetSystemTime = false;
        ts.MTime = Time::FromTimeval(&times[1]);
        ts.MTimeSetSystemTime = false;
    }

    return utime(task, dirfd, filenameAddr, &ts, true);
}

fn renameAt(task: &Task, oldDirfd: i32, oldAddr: u64, newDirfd: i32, newAddr: u64) -> Result<i64> {
    let (newPath, _) = copyInPath(task, newAddr, false)?;
    let (oldPath, _) = copyInPath(task, oldAddr, false)?;

    debug!("renameAt old {}, new {}", &oldPath, newPath);

    fileOpAt(
        task,
        oldDirfd,
        &oldPath.to_string(),
        &mut |_root: &Dirent,
              oldParent: &Dirent,
              oldName: &str,
              _remainingTraversals: u32|
         -> Result<()> {
            let inode = oldParent.Inode();
            if !inode.StableAttr().IsDir() {
                return Err(Error::SysError(SysErr::ENOTDIR));
            }

            match oldName {
                "" | "." | ".." => return Err(Error::SysError(SysErr::EBUSY)),
                _ => (),
            }

            return fileOpAt(
                task,
                newDirfd,
                &newPath.to_string(),
                &mut |root: &Dirent,
                      newParent: &Dirent,
                      newName: &str,
                      _remainingTraversals: u32|
                 -> Result<()> {
                    let inode = newParent.Inode();
                    if !inode.StableAttr().IsDir() {
                        return Err(Error::SysError(SysErr::ENOTDIR));
                    }

                    match newName {
                        "" | "." | ".." => return Err(Error::SysError(SysErr::EBUSY)),
                        _ => (),
                    }

                    return Dirent::Rename(task, root, oldParent, oldName, newParent, newName);
                },
            );
        },
    )?;

    Ok(0)
}

pub fn SysRename(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let oldAddr = args.arg0 as u64;
    let newAddr = args.arg1 as u64;

    return renameAt(task, ATType::AT_FDCWD, oldAddr, ATType::AT_FDCWD, newAddr);
}

pub fn SysRenameat(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let oldDirfd = args.arg0 as i32;
    let oldAddr = args.arg1 as u64;
    let newDirfd = args.arg2 as i32;
    let newAddr = args.arg3 as u64;

    return renameAt(task, oldDirfd, oldAddr, newDirfd, newAddr);
}

// Fallocate implements linux system call fallocate(2).
pub fn SysFallocate(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let mode = args.arg1 as i64;
    let offset = args.arg2 as i64;
    let len = args.arg3 as i64;

    let file = task.GetFile(fd)?;

    if offset < 0 || len <= 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if mode != 0 {
        //t.Kernel().EmitUnimplementedEvent(t)
        return Err(Error::SysError(SysErr::ENOTSUP));
    }

    if !file.Flags().Write || file.Flags().Path {
        return Err(Error::SysError(SysErr::EBADF));
    }

    let mut inode = file.Dirent.Inode();
    if inode.StableAttr().IsPipe() {
        return Err(Error::SysError(SysErr::ESPIPE));
    }

    if inode.StableAttr().IsDir() {
        return Err(Error::SysError(SysErr::EISDIR));
    }

    if !inode.StableAttr().IsRegular() {
        return Err(Error::SysError(SysErr::ENODEV));
    }

    let size = offset + len;

    if size < 0 {
        return Err(Error::SysError(SysErr::EFBIG));
    }

    let rlimitSize = task
        .Thread()
        .ThreadGroup()
        .Limits()
        .Get(LimitType::FileSize)
        .Cur;
    if len as u64 > rlimitSize {
        return Err(Error::ErrExceedsFileSizeLimit);
    }

    let dirent = file.Dirent.clone();
    inode.Allocate(task, &dirent, offset, len)?;

    file.Dirent
        .InotifyEvent(InotifyEvent::IN_MODIFY, 0, EventType::InodeEvent);

    Ok(0)
}

pub fn SysFlock(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let mut operation = args.arg1 as i32;

    let file = task.GetFile(fd)?;

    if file.Flags().Path {
        return Err(Error::SysError(SysErr::EBADF));
    }

    let nonblocking = operation & LibcConst::LOCK_NB as i32 != 0;
    operation &= !(LibcConst::LOCK_NB as i32);

    // flock(2):
    // Locks created by flock() are associated with an open file table entry. This means that
    // duplicate file descriptors (created by, for example, fork(2) or dup(2)) refer to the
    // same lock, and this lock may be modified or released using any of these descriptors. Furthermore,
    // the lock is released either by an explicit LOCK_UN operation on any of these duplicate
    // descriptors, or when all such descriptors have been closed.
    //
    // If a process uses open(2) (or similar) to obtain more than one descriptor for the same file,
    // these descriptors are treated independently by flock(). An attempt to lock the file using
    // one of these file descriptors may be denied by a lock that the calling process has already placed via
    // another descriptor.
    //
    // We use the File UniqueID as the lock UniqueID because it needs to reference the same lock across dup(2)
    // and fork(2).
    let lockUniqueId = file.UniqueId();

    let rng = Range::New(0, MAX_RANGE);
    let inode = file.Dirent.Inode();
    let bsd = inode.lock().LockCtx.BSD.clone();

    match operation as u64 {
        LibcConst::LOCK_EX => {
            if nonblocking {
                // Since we're nonblocking we pass a nil lock.Blocker implementation.
                if !bsd.LockRegion(
                    task,
                    lockUniqueId,
                    OwnerInfo::default(),
                    LockType::WriteLock,
                    &rng,
                    false,
                )? {
                    return Err(Error::SysError(SysErr::EWOULDBLOCK));
                }
            } else {
                // Because we're blocking we will pass the task to satisfy the lock.Blocker interface.
                if !bsd.LockRegion(
                    task,
                    lockUniqueId,
                    OwnerInfo::default(),
                    LockType::WriteLock,
                    &rng,
                    true,
                )? {
                    return Err(Error::SysError(SysErr::EINTR));
                }
            }
        }
        LibcConst::LOCK_SH => {
            if nonblocking {
                // Since we're nonblocking we pass a nil lock.Blocker implementation.
                if !bsd.LockRegion(
                    task,
                    lockUniqueId,
                    OwnerInfo::default(),
                    LockType::ReadLock,
                    &rng,
                    false,
                )? {
                    return Err(Error::SysError(SysErr::EWOULDBLOCK));
                }
            } else {
                // Because we're blocking we will pass the task to satisfy the lock.Blocker interface.
                if !bsd.LockRegion(
                    task,
                    lockUniqueId,
                    OwnerInfo::default(),
                    LockType::ReadLock,
                    &rng,
                    true,
                )? {
                    return Err(Error::SysError(SysErr::EINTR));
                }
            }
        }
        LibcConst::LOCK_UN => bsd.UnlockRegion(task, lockUniqueId, &rng),
        _ => {
            // flock(2): EINVAL operation is invalid.
            return Err(Error::SysError(SysErr::EINVAL));
        }
    }

    return Ok(0);
}
