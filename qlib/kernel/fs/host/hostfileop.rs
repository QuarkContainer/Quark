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
use alloc::collections::btree_map::BTreeMap;
use alloc::vec::Vec;
use alloc::sync::Arc;
use core::any::Any;

use super::super::super::guestfdnotifier::*;
use super::super::super::kernel::waiter::*;
use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::device::*;
use super::super::super::super::pagetable::*;
use super::super::super::super::range::*;
use super::super::super::super::addr::*;
use super::super::super::super::bytestream::*;
use super::super::super::Kernel::HostSpace;
use super::super::super::task::*;
use super::super::super::kernel::async_wait::*;
use super::super::super::super::qmsg::qcall::*;

use super::super::file::*;
use super::super::fsutil::file::*;
use super::super::dentry::*;
use super::super::dirent::*;
use super::super::attr::*;
use super::super::inode::*;
use super::super::host::hostinodeop::*;

use super::util::*;

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

impl HostInodeOp {
    fn ReadDirAll(&self, _task: &Task) -> Result<BTreeMap<String, DentAttr>> {
        let fd = self.HostFd();

        let mut fts = FileTypes {
            fileTypes: Vec::new()
        };

        let res = HostSpace::ReadDir(fd, &mut fts);
        if res < 0 {
            return Err(Error::SysError(-res as i32))
        }


        let mut entries = BTreeMap::new();
        for ft in &fts.fileTypes {
            let dentry = DentAttr {
                Type: InodeType(DType::ModeType(ft.dType) as u32),
                InodeId: HOSTFILE_DEVICE.lock().Map(MultiDeviceKey {
                    Device: ft.device,
                    Inode: ft.inode,
                    SecondaryDevice: "".to_string(),
                })
            };

            entries.insert(ft.pathname.Str().unwrap().to_string(), dentry);
        }

        return Ok(entries);
    }
}

impl HostFileOp {
    fn ReadDirAll(&self, task: &Task) -> Result<BTreeMap<String, DentAttr>> {
        return self.InodeOp.ReadDirAll(task)
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

