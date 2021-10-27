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

use ::qlib::mutex::*;
use alloc::string::String;
use alloc::string::ToString;
use alloc::collections::btree_map::BTreeMap;
use alloc::vec::Vec;
use alloc::sync::Arc;
use core::any::Any;

use super::super::super::guestfdnotifier::*;
use super::super::super::kernel::waiter::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::util::cstring::*;
use super::super::super::qlib::device::*;
use super::super::super::qlib::pagetable::*;
use super::super::super::qlib::range::*;
use super::super::super::qlib::addr::*;
use super::super::super::qlib::bytestream::*;
use super::super::super::Kernel::HostSpace;
use super::super::super::task::*;
use super::super::super::kernel::async_wait::*;
use super::super::super::IOURING;
use super::super::super::SHARESPACE;
use super::super::super::qlib::qmsg::qcall::*;
//use super::super::super::BUF_MGR;

use super::super::file::*;
use super::super::fsutil::file::*;
use super::super::dentry::*;
use super::super::dirent::*;
use super::super::attr::*;
use super::super::inode::*;
use super::super::host::hostinodeop::*;

use super::util::*;
use super::dirent::*;

pub enum HostFileBuf {
    None,
    TTYOut(Arc<QMutex<ByteStream>>),
}

pub struct HostFileOp {
    pub InodeOp: HostInodeOp,
    pub DirCursor: QMutex<String>,
    //pub Buf: HostFileBuf,
}

#[derive(Clone)]
pub struct FutureFileType {
    pub dirfd: i32,
    pub pathname: u64,
    pub future: Future<Statx>,
}

impl HostFileOp {
    fn ReadDirAll(&self, task: &Task) -> Result<BTreeMap<String, DentAttr>> {
        let uringStatx = SHARESPACE.config.read().UringStatx;

        let buf: [u8; 4096] = [0; 4096];

        let fd = self.InodeOp.HostFd();

        let res = Seek(fd, 0, SeekWhence::SEEK_SET) as i32;

        if res < 0 {
            if -res == SysErr::ESPIPE {
                return Err(Error::SysError(SysErr::ENOTDIR))
            }

            return Err(Error::SysError(-res))
        }

        let mut names: Vec<CString> = Vec::new();
        loop {
            let addr = &buf[0] as *const _ as u64;
            let cnt = GetDents(fd, addr, buf.len() as u32);

            if cnt < 0 {
                return Err(Error::SysError(-cnt as i32))
            }

            if cnt == 0 {
                break;
            }

            let cnt: u64 = cnt as u64;
            let mut pos: u64 = 0;
            while pos < cnt {
                unsafe {
                    let d: *const Dirent64 = (addr + pos) as *const Dirent64;
                    let name = (*d).name;
                    let str = CString::ToString(task, &name[0] as *const _ as u64).expect("ReadDirAll fail1");
                    names.push(CString::New(&str));
                    pos += (*d).reclen as u64;
                }
            }
        }

        let mut entries = BTreeMap::new();

        if names.len() == 0 {
            return Ok(entries);
        }

        if !uringStatx {
            let mut fts = Vec::with_capacity(names.len());
            for name in &names {
                fts.push(FileType {
                    dirfd: fd,
                    pathname: name.Ptr(),
                    mode: 0,
                    device: 0,
                    inode: 0,
                    ret: 0,
                })
            }

            HostSpace::BatchFstatat(&mut fts);

            for i in 0 .. fts.len() {
                let ft = &fts[i];
                let ret = ft.ret;
                if ret < 0 {
                    if -ret == SysErr::ENOENT {
                        continue
                    }

                    return Err(Error::SysError(-ret))
                }

                let dentry = DentAttr {
                    Type: InodeType(ft.mode),
                    InodeId: HOSTFILE_DEVICE.lock().Map(MultiDeviceKey {
                        Device: ft.device,
                        Inode: ft.inode,
                        SecondaryDevice: "".to_string(),
                    })
                };

                let name = CString::ToString(task, names[i].Ptr()).expect("ReadDirAll fail2");
                entries.insert(name, dentry);
            }

        } else {
            let mw = MultiWait::New(task.GetTaskIdQ());

            let mut fts = Vec::with_capacity(names.len());
            for name in &names {
                let mask = StatxMask::STATX_MODE | StatxMask::STATX_INO;
                let future = IOURING.AsyncStatx(fd, name.Ptr(), ATType::AT_SYMLINK_NOFOLLOW, mask, &mw);
                fts.push(FutureFileType {
                    dirfd: fd,
                    pathname: name.Ptr(),
                    future: future,
                })
            }

            mw.Wait();

            for i in 0 .. fts.len() {
                let ft = &fts[i];

                let ret = match ft.future.Wait() {
                    Err(Error::SysError(SysErr::ENOENT)) => {
                        continue
                    }
                    Err(e) => return Err(e),
                    Ok(ret) => ret,
                };

                let dentry = DentAttr {
                    Type: InodeType(ret.stx_mode as _),
                    InodeId: HOSTFILE_DEVICE.lock().Map(MultiDeviceKey {
                        Device: MakeDeviceID(ret.stx_dev_major as u16, ret.stx_dev_minor as _) as u64,
                        Inode: ret.stx_ino as _,
                        SecondaryDevice: "".to_string(),
                    })
                };

                let name = CString::ToString(task, names[i].Ptr()).expect("ReadDirAll fail2");
                entries.insert(name, dentry);
            }
        }

        return Ok(entries);
    }
}

impl Waitable for HostFileOp {
    fn Readiness(&self, _task: &Task,mask: EventMask) -> EventMask {
        // somehow, a normal host file could also be polled.
        assert!(self.InodeOp.lock().WouldBlock, "HostFileOp::EventRegister is not supported");

        let fd = self.InodeOp.FD();
        return NonBlockingPoll(fd, mask);
    }

    fn EventRegister(&self, task: &Task,e: &WaitEntry, mask: EventMask) {
        assert!(self.InodeOp.lock().WouldBlock, "HostFileOp::EventRegister is not supported");

        /*if !self.InodeOp.lock().WouldBlock {
            return
        }*/

        let queue = self.InodeOp.Queue();
        queue.EventRegister(task, e, mask);
        let fd = self.InodeOp.FD();
        UpdateFD(fd).unwrap();
    }

    fn EventUnregister(&self, task: &Task,e: &WaitEntry) {
        assert!(self.InodeOp.lock().WouldBlock, "HostFileOp::EventRegister is not supported");
        /*if !self.InodeOp.lock().WouldBlock {
            return
        }*/

        let queue = self.InodeOp.Queue();
        queue.EventUnregister(task, e);
        let fd = self.InodeOp.FD();
        UpdateFD(fd).unwrap();
    }
}

impl SpliceOperations for HostFileOp {}

impl FileOperations for HostFileOp {
    fn as_any(&self) -> &Any {
        return self
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::HostFileOp
    }

    fn Seekable(&self) -> bool {
        return true;
    }

    fn Seek(&self, task: &Task, f: &File, whence: i32, current: i64, offset: i64) -> Result<i64> {
        let mut dirCursor = self.DirCursor.lock();
        let mut cursor = "".to_string();
        let newOffset = SeekWithDirCursor(task, f, whence, current, offset, Some(&mut cursor))?;
        *dirCursor = cursor;
        return Ok(newOffset)
    }

    fn ReadDir(&self, task: &Task, file: &File, offset: i64, serializer: &mut DentrySerializer) -> Result<i64> {
        let root = task.Root();
        let mut dirCursor = self.DirCursor.lock();

        let mut dirCtx = DirCtx {
            Serializer: serializer,
            DirCursor: (*dirCursor).to_string(),
        };

        let res = DirentReadDir(task, &file.Dirent, self, &root, &mut dirCtx, offset)?;
        *dirCursor = dirCtx.DirCursor;
        return Ok(res);
    }

    fn ReadAt(&self, task: &Task, f: &File, dsts: &mut [IoVec], offset: i64, blocking: bool) -> Result<i64> {
        let hostIops = self.InodeOp.clone();

        hostIops.ReadAt(task, f, dsts, offset, blocking)
    }

    fn WriteAt(&self, task: &Task, f: &File, srcs: &[IoVec], offset: i64, blocking: bool) -> Result<i64> {
        let hostIops = self.InodeOp.clone();

        hostIops.WriteAt(task, f, srcs, offset, blocking)
    }

    fn Append(&self, task: &Task, f: &File, srcs: &[IoVec]) -> Result<(i64, i64)> {
        let hostIops = self.InodeOp.clone();

        return hostIops.Append(task, f, srcs);
    }

    fn Fsync(&self, task: &Task, f: &File, start: i64, end: i64, syncType: SyncType) -> Result<()> {
        let hostIops = self.InodeOp.clone();

        return hostIops.Fsync(task, f, start, end, syncType)
    }

    fn Flush(&self, task: &Task, f: &File) -> Result<()> {
        if self.InodeOp.InodeType() == InodeType::RegularFile {
            return self.Fsync(task, f, 0, 0, SyncType::SyncAll);
        }

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
        let entries = match self.ReadDirAll(task) {
            Err(e) => return (offset, Err(e)),
            Ok(entires) => entires,
        };

        let dentryMap = DentMap {
            Entries: entries,
        };

        return match dirCtx.ReadDir(task, &dentryMap) {
            Err(e) => (offset, Err(e)),
            Ok(count) => (offset + count as i32, Ok(0))
        }
    }

    fn Mappable(&self) -> Result<HostInodeOp> {
        return self.InodeOp.Mappable();
    }
}

impl SockOperations for HostFileOp {}

impl PageTables {
    //Reset the cow page to the orginal file page, it is used for the file truncate
    pub fn ResetFileMapping(&self, task: &Task, addr: u64, f: &HostInodeOp, fr: &Range, at: &AccessType) -> Result<()> {
        return self.MapFile(task, addr, f, fr, at, false);
    }

    pub fn MapFile(&self, task: &Task, addr: u64, f: &HostInodeOp, fr: &Range, at: &AccessType, _precommit: bool) -> Result<()> {
        let bs = f.MapInternal(task, fr)?;
        let mut addr = addr;

        let pt = self;
        for b in &bs {
            //todo: handle precommit
            /*if precommit {
                let s = b.ToSlice();
                let mut offset = 0;
                while offset < s.len() as u64 {
                    let _ = s[offset];// Touch to commit.
                    offset += MemoryDef::PAGE_SIZE;
                }
            }*/

            pt.MapHost(task, addr, b, at, true)?;
            addr += b.Len() as u64;
        }

        return Ok(());
    }

    pub fn RemapFile(&self, task: &Task, addr: u64, f: &HostInodeOp, fr: &Range, oldar: &Range, at: &AccessType, _precommit: bool) -> Result<()> {
        let bs = f.MapInternal(task, fr)?;
        let mut addr = addr;

        let pt = self;
        for b in &bs {
            //todo: handle precommit
            /*if precommit {
                let s = b.ToSlice();
                let mut offset = 0;
                while offset < s.len() as u64 {
                    let _ = s[offset];// Touch to commit.
                    offset += MemoryDef::PAGE_SIZE;
                }
            }*/

            pt.RemapHost(task, addr, b, oldar, at, true)?;
            addr += b.Len() as u64;
        }

        return Ok(());
    }
}

