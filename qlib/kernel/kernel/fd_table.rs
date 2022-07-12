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
use alloc::collections::btree_map::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Bound::*;

use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::super::limits::*;
use super::super::super::range::*;
use super::super::task::*;
use super::super::fs::file::*;
use super::super::uid::*;

#[derive(Clone, Default, Debug)]
pub struct FDFlags {
    pub CloseOnExec: bool,
}

impl FDFlags {
    pub fn ToLinuxFileFlags(&self) -> i32 {
        if self.CloseOnExec {
            return Flags::O_CLOEXEC;
        }

        return 0;
    }

    pub fn ToLinuxFDFlags(&self) -> u32 {
        if self.CloseOnExec {
            return LibcConst::FD_CLOEXEC as u32;
        }

        return 0;
    }
}

#[derive(Clone)]
pub struct Descriptor {
    pub file: File,
    pub flags: FDFlags,
}


#[derive(Debug, Clone, Default)]
pub struct GapMgr {
    pub range: Range,
    pub map: BTreeMap<u64, u64>,
    //store the gaps, key: gapStart, val: gapLen
}

impl GapMgr {
    pub fn New(start: u64, len: u64) -> Self {
        let mut map = BTreeMap::new();
        let range = Range { start, len };
        map.insert(range.End()-1, len);
        return GapMgr {
            range: range,
            map,
        };
    }

    pub fn Fork(&self) -> Self {
        let range = Range::New(self.range.start, self.range.len);
        let mut map = BTreeMap::new();
        for (end, len) in &self.map {
            map.insert(*end, *len);
        }

        return GapMgr { range, map };
    }

    pub fn Print(&self) {
        info!(
        "GapMgr: the full range is {:x} to {:x}",
        self.range.start,
        self.range.End()
        );
        for (start, len) in &self.map {
            info!(
            "the gap start is {:x}, len is {:x}, end is {:x}",
            *start,
            *len,
            *start + *len
            );
        }
    }

    pub fn Take(&mut self, id: u64) {
        let mut remove = None;
        let mut insert = None;

        for (gEnd, gLen) in self.map.range_mut((Included(id), Unbounded)) {
            let gR = Range {
                start: *gEnd - (*gLen - 1),
                len: *gLen,
            };

            assert!(gR.Contains(id), "Take fail {}/{:?}/{:?}", id, &gR, &self.map);

            // This is the only one in the range
            if gR.Len() == 1 {
                remove = Some(*gEnd);
                break
            }

             // there is left range need to add
            if gR.Start() < id {
                insert = Some(Range::New(gR.Start(), id - gR.Start()));
            }

            // the right range need to remove
            if gR.End() - 1 == id {
                remove = Some(*gEnd);
            } else {
                // right range need to shrink
                *gLen = *gEnd - id;
            }

            break;
        }

        match remove {
            None => (),
            Some(end) => {
                self.map.remove(&end);
            },
        }

        match insert {
            None => (),
            Some(r) => {
                self.map.insert(r.End()-1, r.Len());
            }
        }
    }

    pub fn AllocAfter(&mut self, id: u64) -> Option<u64> {
        let mut firstRange = None;
        for (gEnd, gLen) in self.map.range((Included(id), Unbounded)) {
            firstRange = Some(Range::New(*gEnd - (*gLen - 1), *gLen));
            break;
        }

        match firstRange {
            None => return None,
            Some(r) => {
                if r.Start() < id {
                    self.Take(id);
                    return Some(id)
                } else {
                    if r.Len() > 1 {
                        self.map.insert(r.End()-1, r.Len()-1);
                    } else {
                        self.map.remove(&(r.End()-1));
                    }

                    return Some(r.Start())
                }
            }
        }
    }

    pub fn Alloc(&mut self) -> Option<u64> {
        return self.AllocAfter(self.range.start);
    }

    pub fn Free(&mut self, id: u64) {
        let leftRange = if id == 0 {
            None
        } else {
            match self.map.get(&(id - 1)) {
                None => None,
                Some(len) => Some(Range::New(id - 1 - (*len - 1), *len)),
            }
        };

        let mut rightRange = None;

        for (gEnd, gLen) in self.map.range((Excluded(id), Unbounded)) {
            let range = Range::New(*gEnd - (*gLen - 1), *gLen);
            if range.Start() == id + 1 {
                rightRange = Some(range);
            } else {
                rightRange = None;
            }
            break;
        }

        if leftRange.is_none() {
            match rightRange {
                None => {
                    self.map.insert(id, 1);
                }
                Some(r) => {
                    self.map.insert(r.End()-1, r.Len() + 1);
                }
            }
        } else {
            let leftRange = leftRange.unwrap();
            self.map.remove(&(leftRange.End()-1));
            match rightRange {
                None => {
                    self.map.insert(leftRange.End(), leftRange.Len() + 1);
                }
                Some(r) => {
                    self.map.insert(r.End()-1, r.Len() + leftRange.Len() + 1);
                }
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct FDTable {
    id: u64,
    data: Arc<QMutex<FDTableInternal>>,
}

impl Drop for FDTable {
    fn drop(&mut self) {
        if self.RefCount() == 1 {
            self.Clear();
        }
    }
}

impl FDTable {
    pub fn New() -> Self {
        return Self {
            id: NewUID(),
            data: Default::default(),
        }
    }

    pub fn Id(&self) -> u64 {
        return self.id;
    }

    pub fn GetFDs(&self) -> Vec<i32> {
        let intern = self.data.lock();
        let mut fds = Vec::with_capacity(intern.descTbl.len());

        for (fd, _) in &intern.descTbl {
            fds.push(*fd)
        }

        return fds;
    }


    pub fn Remove(&self, fd: i32) -> Option<File> {
        return self.data.lock().Remove(self.id, fd);
    }

    pub fn RemoveRange(&self, startfd: i32, endfd: i32) -> Vec<File> {
        let mut intern = self.data.lock();
        let mut ids = Vec::new();
        for (fd, _) in intern.descTbl.range((Included(&startfd), Excluded(&endfd))) {
            ids.push(*fd)
        };

        let mut ret = Vec::new();
        for fd in ids {
            match intern.Remove(self.id, fd) {
                None => error!("impossible in RemoveRange"),
                Some(f)=> ret.push(f)
            }
        }

        return ret;
    }

    pub fn SetFlagsForRange(&mut self, startfd: i32, endfd: i32, flags: FDFlags) -> Result<()> {
        let mut intern = self.data.lock();
        if startfd < 0 || startfd >= endfd {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        for (_, d) in intern.descTbl.range_mut((Included(&startfd), Excluded(&endfd))) {
            d.flags = flags.clone();
        }

        return Ok(())
    }

    pub fn RemoveCloseOnExec(&self) {
        let mut intern = self.data.lock();
        let mut removed = Vec::new();
        for (fd, desc) in &intern.descTbl {
            if desc.flags.CloseOnExec {
                removed.push(*fd);
            }
        }

        for fd in &removed {
            intern.gaps.Free(*fd as u64);
            let desc = intern.descTbl.remove(fd).unwrap();
            intern.Drop(self.id, &desc.file);
        }

        //self.Verify();
    }

    pub fn SetFlags(&self, fd: i32, flags: &FDFlags) -> Result<()> {
        if fd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        let mut intern = self.data.lock();
        let file = intern.descTbl.get_mut(&fd);

        match file {
            None => return Err(Error::SysError(SysErr::EBADF)),
            Some(fdesc) => fdesc.flags = flags.clone(),
        }

        return Ok(());
    }

    pub fn GetLastFd(&self) -> i32 {
        let intern = self.data.lock();
        for (i, _) in intern.descTbl.iter().rev() {
            return *i;
        }

        return 0;
    }

    pub fn Get(&self, fd: i32) -> Result<(File, FDFlags)> {
        let intern = self.data.lock();

        let f = intern.descTbl.get(&fd);
        match f {
            None => return Err(Error::SysError(SysErr::EBADF)),
            Some(f) => Ok((f.file.clone(), f.flags.clone())),
        }
    }

    pub fn Dup(&self, task: &Task, fd: i32) -> Result<i32> {
        if fd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        let (f, flags) = self.Get(fd)?;
        return self.NewFDFrom(task, 0, &f, &flags);
    }

    pub fn Dup2(&self, task: &Task, oldfd: i32, newfd: i32) -> Result<i32> {
        if oldfd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        if newfd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        let (f, flags) = self.Get(oldfd)?;
        self.NewFDAt(task, newfd, &f, &flags)?;
        return Ok(newfd);
    }

    pub fn Dup3(&self, task: &Task, oldfd: i32, newfd: i32, flags: i32) -> Result<i32> {
        if oldfd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        if newfd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        let closeOnExec = Flags(flags).CloseOnExec();

        let (f, mut flags) = self.Get(oldfd)?;
        flags.CloseOnExec = closeOnExec;
        self.NewFDAt(task, newfd, &f, &flags)?;
        return Ok(newfd);
    }

    pub fn NewFDFrom(&self, task: &Task, fd: i32, file: &File, flags: &FDFlags) -> Result<i32> {
        if fd < 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        // Default limit.
        let mut end = i32::MAX;

        let lim = task.Thread().ThreadGroup().Limits().Get(LimitType::NumberOfFiles).Cur;
        if lim != u64::MAX {
            end = lim as i32;
        }

        if fd + 1 > end {
            return Err(Error::SysError(SysErr::EMFILE));
        }

        let mut tbl = self.data.lock();

        let newfd = match tbl.gaps.AllocAfter(fd as u64) {
            None => return Err(Error::SysError(SysErr::EMFILE)),
            Some(newfd) => newfd as i32,
        };

        if newfd >= end {
            tbl.gaps.Free(newfd as u64);
            return Err(Error::SysError(SysErr::EMFILE));
        }

        tbl.set(self.id, newfd, &file, flags);

        //self.Verify();
        return Ok(newfd);
    }

    pub fn NewFDAt(&self, task: &Task, fd: i32, file: &File, flags: &FDFlags) -> Result<()> {
        if fd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        let lim = task.Thread().ThreadGroup().Limits().Get(LimitType::NumberOfFiles).Cur;
        if fd as u64 >= lim {
            return Err(Error::SysError(SysErr::EMFILE));
        }

        return self.data.lock().NewFDAt(task, self.id, fd, file, flags);
    }

    // Fork returns an independent FDTable, cloning all FDs up to maxFds (non-inclusive).
    pub fn Fork(&self, maxFds: i32) -> FDTable {
        let intern = self.data.lock();
        let mut tbl = FDTableInternal {
            gaps: intern.gaps.clone(),
            descTbl: BTreeMap::new(),
        };

        for (fd, file) in &intern.descTbl {
            if *fd >= maxFds {
                break;
            }
            tbl.set(self.id, *fd, &file.file, &file.flags)
        }

        return Self {
            id: NewUID(),
            data: Arc::new(QMutex::new(tbl)),
        }
    }

    pub fn Clear(&self) {
        self.data.lock().RemoveAll(self.id);
    }

    pub fn Count(&self) -> usize {
        return self.data.lock().descTbl.len();
    }

    pub fn RefCount(&self) -> usize {
        return Arc::strong_count(&self.data);
    }
}

pub struct FDTableInternal {
    pub gaps: GapMgr,
    pub descTbl: BTreeMap<i32, Descriptor>,
}

impl Default for FDTableInternal {
    fn default() -> Self {
        return Self::New();
    }
}

impl FDTableInternal {
    pub fn New() -> Self {
        return Self {
            gaps: GapMgr::New(0, i32::MAX as u64),
            descTbl: BTreeMap::new(),
        };
    }

    // test function
    pub fn Verify(&self) {
        let mut cur = self.gaps.range.Start();
        for (end, len) in &self.gaps.map {
            let start = *end - (*len - 1);
            while cur < start {
                assert!(self.descTbl.contains_key(&(cur as i32)),
                    "verify {}/{}->{}/{:?}/{:?}",
                    cur, start, *len, &self.descTbl.keys(), &self.gaps.map);
                cur += 1;
            }
            cur = *end + 1;
        }
    }

    pub fn Drop(&self, lockUniqueID: u64, file: &File) {
        if Arc::strong_count(&file.0) == 1 {
            let d = file.Dirent.clone();
            let mut ev = 0;
            let inode = d.Inode();
            let lockCtx = inode.lock().LockCtx.clone();

            let task = Task::Current();
            lockCtx
                .Posix
                .UnlockRegion(task, lockUniqueID, &Range::Max());

            if inode.StableAttr().IsDir() {
                ev |= InotifyEvent::IN_ISDIR;
            }

            if file.Flags().Write {
                ev |= InotifyEvent::IN_CLOSE_WRITE;
            } else {
                ev |= InotifyEvent::IN_CLOSE_NOWRITE;
            }
            d.InotifyEvent(ev, 0);
        }
    }

    pub fn Print(&self) {
        for (id, d) in &self.descTbl {
            info!(
                "FDTableInternal::Print [{}], refcount is {}, id is {}",
                *id,
                Arc::strong_count(&d.file.0),
                d.file.0.UniqueId
            )
        }
    }

    pub fn Size(&self) -> usize {
        return self.descTbl.len();
    }

    fn set(&mut self, uid: u64, fd: i32, file: &File, flags: &FDFlags) {
        let fdesc = Descriptor {
            file: file.clone(),
            flags: flags.clone(),
        };

        match self.descTbl.insert(fd, fdesc) {
            None => (),
            Some(f) => self.Drop(uid, &f.file),
        }
    }

    fn NewFDAt(&mut self, _task: &Task, uid: u64, fd: i32, file: &File, flags: &FDFlags) -> Result<()> {
        //self.Verify();
        match self.descTbl.remove(&fd) {
            None => {
                self.gaps.Take(fd as u64);
            },
            Some(desc) => {
                self.Drop(uid, &desc.file);
            }
        }

        self.set(uid, fd, file, flags);
        return Ok(());
    }

    pub fn SetFlags(&mut self, fd: i32, flags: &FDFlags) -> Result<()> {
        if fd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        let file = self.descTbl.get_mut(&fd);

        match file {
            None => return Err(Error::SysError(SysErr::EBADF)),
            Some(fdesc) => fdesc.flags = flags.clone(),
        }

        return Ok(());
    }

    pub fn GetFiles(&self) -> Vec<File> {
        let mut files = Vec::with_capacity(self.descTbl.len());

        for (_, file) in &self.descTbl {
            files.push(file.file.clone())
        }

        return files;
    }

    pub fn Remove(&mut self, uid: u64, fd: i32) -> Option<File> {
        if fd < 0 {
            return None;
        }

        let file = self.descTbl.remove(&fd);

        match file {
            None => return None,
            Some(f) => return {
                self.gaps.Free(fd as u64);
                //self.Verify();
                self.Drop(uid, &f.file);
                Some(f.file)
            },
        }
    }

    pub fn RemoveAll(&mut self, uid: u64) {
        let mut removed = Vec::new();
        for (fd, _) in &self.descTbl {
            removed.push(*fd);
        }

        for fd in &removed {
            // the whole gaps state will be removed. No need update gaps to improve performance
            //self.gaps.Free(*fd as u64);
            let desc = self.descTbl.remove(fd).unwrap();
            self.Drop(uid, &desc.file);
        }

        //self.Verify();

    }
}

