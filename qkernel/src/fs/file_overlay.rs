// Copyright (c) 2021 Quark Container Authors
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
use alloc::sync::Arc;
use spin::Mutex;
use spin::RwLock;
use core::any::Any;
use alloc::collections::btree_map::BTreeMap;
use alloc::borrow::ToOwned;

use super::super::kernel::waiter::*;
use super::super::qlib::common::*;
use super::super::task::*;
use super::super::qlib::linux_def::*;
use super::inode::*;
use super::flags::*;
use super::file::*;
use super::dirent::*;
use super::dentry::*;
use super::attr::*;
use super::overlay::*;
use super::inode_overlay::*;
use super::host::hostinodeop::*;

pub fn OverlayFile(task: &Task, inode: &Inode, flags: &FileFlags) -> Result<File> {
    let dirent = Dirent::NewTransient(inode);
    let f = inode.GetFile(task, &dirent, flags)?;
    return Ok(f)
}

pub struct OverlayFileOperations {
    pub upper: Mutex<Option<File>>,
    pub lower: Mutex<Option<File>>,
    pub dirCursor: Mutex<String>,
    pub dirCache: Mutex<DentMap>,
}

impl Default for OverlayFileOperations {
    fn default() -> Self {
        return Self {
            upper: Mutex::new(None),
            lower: Mutex::new(None),
            dirCursor: Mutex::new("".to_owned()),
            dirCache: Mutex::new(DentMap::default())
        }
    }
}

impl OverlayFileOperations {
    fn OnTop(&self, task: &Task, file: &File, mut func: impl FnMut(&File, Arc<FileOperations>) -> Result<()>) -> Result<()> {
        let inode = file.Dirent.Inode();
        let overlay = inode.lock().Overlay.as_ref().unwrap().clone();
        let o = overlay.read();
        if o.upper.is_none() {
            let lower = self.lower.lock().as_ref().unwrap().clone();
            let lowerfops = lower.FileOp.clone();
            return func(&lower, lowerfops)
        }

        {
            let mut upper = self.upper.lock();
            if upper.is_none() {
                let upperInode = o.upper.as_ref().unwrap().clone();
                let upperFile = OverlayFile(task, &upperInode, &file.flags.lock().0)?;
                *upper = Some(upperFile);
            }
        }

        let upper = self.upper.lock().as_ref().unwrap().clone();
        let upperfops = upper.FileOp.clone();
        return func(&upper, upperfops)
    }

    fn FileOps(&self) -> Arc<FileOperations> {
        let ops = if self.upper.lock().is_some() {
            self.upper.lock().as_ref().unwrap().FileOp.clone()
        } else if self.lower.lock().is_some() {
            self.lower.lock().as_ref().unwrap().FileOp.clone()
        } else {
            panic!("OverlayFileOperations::FileOps both upper and lower are none");
        };

        return ops;
    }

    fn Dirent(&self) -> Dirent {
        let dirent = if self.upper.lock().is_some() {
            self.upper.lock().as_ref().unwrap().Dirent.clone()
        } else if self.lower.lock().is_some() {
            self.lower.lock().as_ref().unwrap().Dirent.clone()
        } else {
            panic!("OverlayFileOperations::Dirent both upper and lower are none");
        };

        return dirent;
    }
}

impl Waitable for OverlayFileOperations {
    fn Readiness(&self, task: &Task, mask: EventMask) -> EventMask {
        let ops = self.FileOps();
        return ops.Readiness(task, mask);
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        let ops = self.FileOps();
        return ops.EventRegister(task, e, mask);
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        let ops = self.FileOps();
        return ops.EventUnregister(task, e);
    }
}

impl SpliceOperations for OverlayFileOperations {}

impl FileOperations for OverlayFileOperations {
    fn as_any(&self) -> &Any {
        return self
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::OverlayFileOperations
    }

    fn Seekable(&self) -> bool {
        return true;
    }

    fn Seek(&self, task: &Task, f: &File, whence: i32, current: i64, offset: i64) -> Result<i64> {
        let seekDir;
        let mut cursor = self.dirCursor.lock();

        let ops = self.FileOps();
        let n = ops.Seek(task, f, whence, current, offset)?;
        let dirent = self.Dirent();
        let inode = dirent.Inode();
        seekDir = inode.StableAttr().IsDir();

        if seekDir && whence == SeekWhence::SEEK_SET && offset == 0 {
            *cursor = "".to_string();
        }

        return Ok(n);
    }

    fn ReadDir(&self, task: &Task, file: &File, offset: i64, serializer: &mut DentrySerializer) -> Result<i64> {
        let root = task.Root();
        let mut dirCursor = self.dirCursor.lock();

        let mut dirCtx = DirCtx {
            Serializer: serializer,
            DirCursor: (*dirCursor).to_string(),
        };

        let dirent = file.Dirent.clone();
        let frozen = (dirent.0).0.lock().frozen;
        if frozen {
            let res = DirentReadDir(task, &dirent, self, &root, &mut dirCtx, offset)?;
            *dirCursor = dirCtx.DirCursor;
            return Ok(res);
        }

        let inode = dirent.Inode();
        let o = inode.lock().Overlay.as_ref().expect("OverlayFileOperations:Readdir get none overlay").clone();
        let dirCache = ReaddirEntries(task, o)?;

        *self.dirCache.lock() = dirCache;

        let res = DirentReadDir(task, &dirent, self, &root, &mut dirCtx, offset)?;
        *dirCursor = dirCtx.DirCursor;
        return Ok(res);
    }

    fn ReadAt(&self, task: &Task, f: &File, dsts: &mut [IoVec], offset: i64, blocking: bool) -> Result<i64> {
        let mut n = 0;
        self.OnTop(task, f, |file, ops| -> Result<()> {
            n = ops.ReadAt(task, file, dsts, offset, blocking)?;
            return Ok(())
        })?;

        return Ok(n)
    }

    fn WriteAt(&self, task: &Task, f: &File, srcs: &[IoVec], offset: i64, blocking: bool) -> Result<i64> {
        let ops = self.upper.lock().as_ref().unwrap().FileOp.clone();
        let res = ops.WriteAt(task, f, srcs, offset, blocking);
        return res;
    }

    fn Append(&self, task: &Task, f: &File, srcs: &[IoVec]) -> Result<(i64, i64)> {
        let ops = self.upper.lock().as_ref().unwrap().FileOp.clone();
        let res = ops.Append(task, f, srcs);
        return res;
    }

    fn Fsync(&self, task: &Task, f: &File, start: i64, end: i64, syncType: SyncType) -> Result<()> {
        let ops = self.FileOps();
        return ops.Fsync(task, f, start, end, syncType);
    }

    fn Flush(&self, task: &Task, f: &File) -> Result<()> {
        let ops = self.FileOps();
        return ops.Flush(task, f);
    }


    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        let ops = self.FileOps();
        return ops.UnstableAttr(task, f);
    }

    fn Ioctl(&self, task: &Task, f: &File, fd: i32, request: u64, val: u64) -> Result<()> {
        let upper = {
            let upper = self.upper.lock();
            if upper.is_none() {
                return Err(Error::SysError(SysErr::ENOTTY))
            }

            upper.as_ref().unwrap().clone()
        };

        return upper.FileOp.Ioctl(task, f, fd, request, val)
    }

    fn IterateDir(&self, task: &Task, _d: &Dirent, dirCtx: &mut DirCtx, offset: i32) -> (i32, Result<i64>) {
        let cache = self.dirCache.lock();
        let n = match dirCtx.ReadDir(task, &cache) {
            Err(e) => return (offset, Err(e)),
            Ok(n) => n,
        };

        return (offset + n as i32, Ok(n as i64))
    }

    fn Mappable(&self) -> Result<HostInodeOp> {
        let ops = self.FileOps();
        return ops.Mappable();
    }
}

impl SockOperations for OverlayFileOperations {}

fn ReaddirEntries(task: &Task, o: Arc<RwLock<OverlayEntry>>) -> Result<DentMap> {
    if o.read().upper.is_none() && o.read().lower.is_none() {
        panic!("invalid overlayEntry, needs at least one Inode")
    }

    let upper = o.read().upper.clone();
    let mut entries = if upper.is_some() {
        ReaddirOne(task, &Dirent::NewTransient(upper.as_ref().unwrap()))?
    } else {
        BTreeMap::new()
    };

    let lower = o.read().lower.clone();
    if lower.is_some() {
        let lowerEntries = ReaddirOne(task, &Dirent::NewTransient(lower.as_ref().unwrap()))?;
        for (name, entry) in lowerEntries {
            if upper.is_some() {
                if OverlayHasWhiteout(upper.as_ref().unwrap(), &name) {
                    continue
                }
            }

            if !entries.contains_key(&name) {
                entries.insert(name, entry);
            }
        }
    }

    return Ok(DentMap { Entries: entries })
}

pub fn ReaddirOne(task: &Task, d: &Dirent) -> Result<BTreeMap<String, DentAttr>> {
    let inode = d.Inode();
    let dir = inode.GetFile(task, d, &FileFlags { Read: true, ..Default::default() })?;

    let mut serializer = CollectEntriesSerilizer::New();
    dir.ReadDir(task, &mut serializer)?;

    serializer.Entries.remove(&".".to_owned());
    serializer.Entries.remove(&"..".to_owned());
    return Ok(serializer.Entries)
}
