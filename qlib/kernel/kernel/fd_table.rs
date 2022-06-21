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
use core::ops::Deref;
use core::ops::Bound::*;

use super::super::super::common::*;
use super::super::super::linux_def::*;
//use super::super::super::limits::*;
//use super::super::super::range::*;
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


#[derive(Clone, Default)]
pub struct FDTable((Arc<QMutex<FDTableInternal>>, u64));

impl Deref for FDTable {
    type Target = Arc<QMutex<FDTableInternal>>;

    fn deref(&self) -> &Arc<QMutex<FDTableInternal>> {
        &(self.0).0
    }
}

impl FDTable {
    pub fn ID(&self) -> u64 {
        return (self.0).1;
    }

    pub fn NewFDFrom(&self, task: &Task, fd: i32, files: &File, flags: &FDFlags) -> Result<i32> {
        return self.lock().NewFDFrom(task, fd, files, flags)
    }

    pub fn NewFDAt(&self, task: &Task, fd: i32, file: &File, flags: &FDFlags) -> Result<()> {
        return self.lock().NewFDAt(task, fd, file, flags);
    }

    pub fn Dup(&self, task: &Task, fd: i32) -> Result<i32> {
        return self.lock().Dup(task, fd)
    }

    pub fn Dup2(&self, task: &Task, oldfd: i32, newfd: i32) -> Result<i32> {
        return self.lock().Dup2(task, oldfd, newfd)
    }

    pub fn Dup3(&self, task: &Task, oldfd: i32, newfd: i32, flags: i32) -> Result<i32> {
        return self.lock().Dup3(task, oldfd, newfd, flags)
    }

    // Fork returns an independent FDTable, cloning all FDs up to maxFds (non-inclusive).
    pub fn Fork(&self, maxFds: i32) -> FDTable {
        let internal = self.lock().Fork(maxFds);

        return FDTable((Arc::new(QMutex::new(internal)), NewUID()));
    }

    pub fn Clear(&self) {
        self.lock().RemoveAll();
    }

    pub fn Count(&self) -> usize {
        return self.lock().descTbl.len();
    }

    pub fn RefCount(&self) -> usize {
        return Arc::strong_count(&(self.0).0);
    }
}

pub struct FDTableInternal {
    pub next: i32,
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
            next: 0,
            descTbl: BTreeMap::new(),
        };
    }

    pub fn GetLastFd(&self) -> i32 {
        for (i, _) in self.descTbl.iter().rev() {
            return *i;
        }

        return 0;
    }

    pub fn SetFlagsForRange(&mut self, startfd: i32, endfd: i32, flags: FDFlags) -> Result<()> {
        if startfd < 0 || startfd >= endfd {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        for (_, d) in self.descTbl.range_mut((Included(&startfd), Excluded(&endfd))) {
            d.flags = flags.clone();
        }

        return Ok(())
    }

    pub fn Drop(&self, file: &File) {
        if Arc::strong_count(&file.0) == 1 {
            let d = file.Dirent.clone();
            let mut ev = 0;
            if d.Inode().StableAttr().IsDir() {
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

    fn set(&mut self, fd: i32, file: &File, flags: &FDFlags) {
        let fdesc = Descriptor {
            file: file.clone(),
            flags: flags.clone(),
        };

        match self.descTbl.insert(fd, fdesc) {
            None => (),
            Some(f) => self.Drop(&f.file),
        }
    }

    pub fn NewFDFrom(&mut self, _task: &Task, fd: i32, files: &File, flags: &FDFlags) -> Result<i32> {
        if fd < 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        // Default limit.
        let end = i32::MAX;

        /*let lim = task.Thread().ThreadGroup().Limits().Get(LimitType::NumberOfFiles).Cur;
        if lim != u64::MAX {
            end = lim as i32;
        }

        if fd + 1 > end {
            return Err(Error::SysError(SysErr::EMFILE));
        }*/

        let mut fd = fd;
        let mut reset = false;
        if fd < self.next {
            fd = self.next;
            reset = true;
        }

        let mut found = false;
        for i in fd..end {
            let curr = self.descTbl.get(&i);

            match curr {
                None => {
                    self.set(i, &files, flags);
                    fd = i;
                    found = true;
                    break;
                }
                _ => (),
            }
        }

        if !found {
            return Err(Error::SysError(SysErr::EMFILE));
        }

        if reset {
            self.next = fd + 1;
        }

        return Ok(fd);
    }

    pub fn NewFDAt(&mut self, _task: &Task, fd: i32, file: &File, flags: &FDFlags) -> Result<()> {
        if fd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        /*let lim = task.Thread().ThreadGroup().Limits().Get(LimitType::NumberOfFiles).Cur;
        if fd as u64 >= lim {
            return Err(Error::SysError(SysErr::EMFILE));
        }*/

        self.set(fd, file, flags);
        return Ok(());
    }

    pub fn Dup(&mut self, task: &Task, fd: i32) -> Result<i32> {
        if fd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        let (f, flags) = self.Get(fd)?;
        return self.NewFDFrom(task, 0, &f, &flags);
    }

    pub fn Dup2(&mut self, task: &Task, oldfd: i32, newfd: i32) -> Result<i32> {
        if oldfd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        if newfd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        self.Remove(newfd);

        let (f, flags) = self.Get(oldfd)?;
        self.NewFDAt(task, newfd, &f, &flags)?;
        return Ok(newfd);
    }

    pub fn Dup3(&mut self, task: &Task, oldfd: i32, newfd: i32, flags: i32) -> Result<i32> {
        if oldfd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        if newfd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        self.Remove(newfd);
        let closeOnExec = Flags(flags).CloseOnExec();

        let (f, mut flags) = self.Get(oldfd)?;
        flags.CloseOnExec = closeOnExec;
        self.NewFDAt(task, newfd, &f, &flags)?;
        return Ok(newfd);
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

    pub fn GetFDs(&self) -> Vec<i32> {
        let mut fds = Vec::with_capacity(self.descTbl.len());

        for (fd, _) in &self.descTbl {
            fds.push(*fd)
        }

        return fds;
    }

    pub fn GetFiles(&self) -> Vec<File> {
        let mut files = Vec::with_capacity(self.descTbl.len());

        for (_, file) in &self.descTbl {
            files.push(file.file.clone())
        }

        return files;
    }

    pub fn Get(&self, fd: i32) -> Result<(File, FDFlags)> {
        let f = self.descTbl.get(&fd);
        match f {
            None => return Err(Error::SysError(SysErr::EBADF)),
            Some(f) => Ok((f.file.clone(), f.flags.clone())),
        }
    }

    // Fork returns an independent FDTable, cloning all FDs up to maxFds (non-inclusive).
    pub fn Fork(&self, maxFds: i32) -> FDTableInternal {
        let mut tbl = FDTableInternal {
            next: self.next,
            descTbl: BTreeMap::new(),
        };

        for (fd, file) in &self.descTbl {
            if *fd >= maxFds {
                break;
            }
            tbl.set(*fd, &file.file, &file.flags)
        }

        return tbl;
    }

    pub fn RemoveRange(&mut self, startfd: i32, endfd: i32) -> Vec<File> {
        let mut ids = Vec::new();
        for (fd, _) in self.descTbl.range((Included(&startfd), Excluded(&endfd))) {
            ids.push(*fd)
        };

        let mut ret = Vec::new();
        for fd in ids {
            match self.Remove(fd) {
                None => error!("impossible in RemoveRange"),
                Some(f)=> ret.push(f)
            }
        }

        return ret;
    }

    pub fn Remove(&mut self, fd: i32) -> Option<File> {
        if fd < 0 {
            return None;
        }

        if fd < self.next {
            self.next = fd;
        }

        let file = self.descTbl.remove(&fd);

        match file {
            None => return None,
            Some(f) => return {
                self.Drop(&f.file);
                Some(f.file)
            },
        }
    }

    pub fn RemoveCloseOnExec(&mut self) {
        let mut removed = Vec::new();
        for (fd, desc) in &self.descTbl {
            if desc.flags.CloseOnExec {
                removed.push(*fd);
            }
        }

        for fd in &removed {
            let desc = self.descTbl.remove(fd).unwrap();
            self.Drop(&desc.file);
        }
    }

    pub fn RemoveAll(&mut self) {
        let mut removed = Vec::new();
        for (fd, _) in &self.descTbl {
            removed.push(*fd);
        }

        for fd in &removed {
            let desc = self.descTbl.remove(fd).unwrap();
            self.Drop(&desc.file);
        }
    }
}

/*

#[derive(Clone, Default)]
pub struct FDTable((Arc<QMutex<FDTableInternal>>, u64));

impl Deref for FDTable {
    type Target = Arc<QMutex<FDTableInternal>>;

    fn deref(&self) -> &Arc<QMutex<FDTableInternal>> {
        &(self.0).0
    }
}

impl FDTable {
    pub fn ID(&self) -> u64 {
        return (self.0).1;
    }

    pub fn Dup(&self, task: &Task, fd: i32) -> Result<i32> {
        if fd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        let (f, flags) = self.lock().Get(fd)?;
        return self.NewFDFrom(task, 0, &f, &flags);
    }

    pub fn Dup2(&self, task: &Task, oldfd: i32, newfd: i32) -> Result<i32> {
        if oldfd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        if newfd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        let (f, flags) = self.lock().Get(oldfd)?;
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


        let (f, mut flags) = self.lock().Get(oldfd)?;
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

        return self.lock().NewFDFrom(task, fd, file, flags);
    }

    pub fn NewFDAt(&self, task: &Task, fd: i32, file: &File, flags: &FDFlags) -> Result<()> {
        if fd < 0 {
            return Err(Error::SysError(SysErr::EBADF));
        }

        let lim = task.Thread().ThreadGroup().Limits().Get(LimitType::NumberOfFiles).Cur;
        if fd as u64 >= lim {
            return Err(Error::SysError(SysErr::EMFILE));
        }

        return self.lock().NewFDAt(task, fd, file, flags);
    }

    // Fork returns an independent FDTable, cloning all FDs up to maxFds (non-inclusive).
    pub fn Fork(&self, maxFds: i32) -> FDTable {
        let internal = self.lock().Fork(maxFds);

        return FDTable((Arc::new(QMutex::new(internal)), NewUID()));
    }

    pub fn Clear(&self) {
        self.lock().RemoveAll();
    }

    pub fn Count(&self) -> usize {
        return self.lock().descTbl.len();
    }

    pub fn RefCount(&self) -> usize {
        return Arc::strong_count(&(self.0).0);
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

    pub fn GetLastFd(&self) -> i32 {
        for (i, _) in self.descTbl.iter().rev() {
            return *i;
        }

        return 0;
    }

    pub fn SetFlagsForRange(&mut self, startfd: i32, endfd: i32, flags: FDFlags) -> Result<()> {
        if startfd < 0 || startfd >= endfd {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        for (_, d) in self.descTbl.range_mut((Included(&startfd), Excluded(&endfd))) {
            d.flags = flags.clone();
        }

        return Ok(())
    }

    pub fn Drop(&self, file: &File) {
        if Arc::strong_count(&file.0) == 1 {
            let d = file.Dirent.clone();
            let mut ev = 0;
            if d.Inode().StableAttr().IsDir() {
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

    fn set(&mut self, fd: i32, file: &File, flags: &FDFlags) {
        let fdesc = Descriptor {
            file: file.clone(),
            flags: flags.clone(),
        };

        match self.descTbl.insert(fd, fdesc) {
            None => (),
            Some(f) => self.Drop(&f.file),
        }
    }

    fn NewFDFrom(&mut self, _task: &Task, fd: i32, files: &File, flags: &FDFlags) -> Result<i32> {
        let newfd = match self.gaps.AllocAfter(fd as u64, 1, 0) {
            Err(_) => return Err(Error::SysError(SysErr::EMFILE)),
            Ok(newfd) => newfd as i32,
        };

        self.set(newfd, &files, flags);

        return Ok(newfd);
    }

    fn NewFDAt(&mut self, _task: &Task, fd: i32, file: &File, flags: &FDFlags) -> Result<()> {
        self.gaps.Take(fd as u64, 1);
        self.set(fd, file, flags);
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

    pub fn GetFDs(&self) -> Vec<i32> {
        let mut fds = Vec::with_capacity(self.descTbl.len());

        for (fd, _) in &self.descTbl {
            fds.push(*fd)
        }

        return fds;
    }

    pub fn GetFiles(&self) -> Vec<File> {
        let mut files = Vec::with_capacity(self.descTbl.len());

        for (_, file) in &self.descTbl {
            files.push(file.file.clone())
        }

        return files;
    }

    pub fn Get(&self, fd: i32) -> Result<(File, FDFlags)> {
        let f = self.descTbl.get(&fd);
        match f {
            None => return Err(Error::SysError(SysErr::EBADF)),
            Some(f) => Ok((f.file.clone(), f.flags.clone())),
        }
    }

    // Fork returns an independent FDTable, cloning all FDs up to maxFds (non-inclusive).
    pub fn Fork(&self, maxFds: i32) -> FDTableInternal {
        let mut tbl = FDTableInternal {
            gaps: GapMgr::New(0, i32::MAX as u64),
            descTbl: BTreeMap::new(),
        };

        for (fd, file) in &self.descTbl {
            if *fd >= maxFds {
                break;
            }
            tbl.gaps.Take(*fd as u64, 1);
            tbl.set(*fd, &file.file, &file.flags)
        }

        return tbl;
    }

    pub fn RemoveRange(&mut self, startfd: i32, endfd: i32) -> Vec<File> {
        let mut ids = Vec::new();
        for (fd, _) in self.descTbl.range((Included(&startfd), Excluded(&endfd))) {
            ids.push(*fd)
        };

        let mut ret = Vec::new();
        for fd in ids {
            match self.Remove(fd) {
                None => error!("impossible in RemoveRange"),
                Some(f)=> ret.push(f)
            }
        }

        return ret;
    }

    pub fn Remove(&mut self, fd: i32) -> Option<File> {
        if fd < 0 {
            return None;
        }

        self.gaps.Free(fd as u64, 1);
        let file = self.descTbl.remove(&fd);

        match file {
            None => return None,
            Some(f) => return {
                self.Drop(&f.file);
                Some(f.file)
            },
        }
    }

    pub fn RemoveCloseOnExec(&mut self) {
        let mut removed = Vec::new();
        for (fd, desc) in &self.descTbl {
            if desc.flags.CloseOnExec {
                removed.push(*fd);
            }
        }

        for fd in &removed {
            self.gaps.Free(*fd as u64, 1);
            let desc = self.descTbl.remove(fd).unwrap();
            self.Drop(&desc.file);
        }
    }

    pub fn RemoveAll(&mut self) {
        let mut removed = Vec::new();
        for (fd, _) in &self.descTbl {
            removed.push(*fd);
        }

        for fd in &removed {
            self.gaps.Free(*fd as u64, 1);
            let desc = self.descTbl.remove(fd).unwrap();
            self.Drop(&desc.file);
        }
    }
}

*/