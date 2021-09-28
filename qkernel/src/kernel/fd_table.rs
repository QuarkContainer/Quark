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

use alloc::collections::btree_map::BTreeMap;
use alloc::vec::Vec;
use spin::Mutex;
use alloc::sync::Arc;
use core::ops::Deref;

use super::super::qlib::linux_def::*;
use super::super::qlib::common::*;
use super::super::fs::file::*;
use super::super::uid::*;

#[derive(Clone, Default, Debug)]
pub struct FDFlags {
    pub CloseOnExec: bool,
}

impl FDFlags {
    pub fn ToLinuxFileFlags(&self) -> i32 {
        if self.CloseOnExec {
            return Flags::O_CLOEXEC
        }

        return 0;
    }

    pub fn ToLinuxFDFlags(&self) -> u32 {
        if self.CloseOnExec {
            return LibcConst::FD_CLOEXEC as u32
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
pub struct FDTable((Arc<Mutex<FDTableInternal>>, u64));

impl Deref for FDTable {
    type Target = Arc<Mutex<FDTableInternal>>;

    fn deref(&self) -> &Arc<Mutex<FDTableInternal>> {
        &(self.0).0
    }
}

impl FDTable {
    pub fn ID(&self) -> u64 {
        return (self.0).1;
    }

    pub fn Fork(&self) -> FDTable {
        let internal = self.lock().Fork();

        return FDTable((Arc::new(Mutex::new(internal)), NewUID()));
    }

    pub fn Clear(&self) {
        self.lock().descTbl.clear();
    }

    pub fn Count(&self) -> usize {
        return self.lock().descTbl.len();
    }

    pub fn RefCount(&self) -> usize {
        return Arc::strong_count(&(self.0).0)
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
        }
    }

    pub fn Print(&self) {
        for (id, d) in &self.descTbl {
            info!("FDTableInternal::Print [{}], refcount is {}, id is {}",
            *id, Arc::strong_count(&d.file.0), d.file.0.UniqueId)
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

        self.descTbl.insert(fd, fdesc);
    }

    pub fn NewFDFrom(&mut self, fd: i32, file: &File, flags: &FDFlags) -> Result<i32> {
        let fds = self.NewFDs(fd, &[file.clone()], flags)?;
        return Ok(fds[0])
    }

    pub fn NewFDs(&mut self, fd: i32, files: &[File], flags: &FDFlags) -> Result<Vec<i32>> {
        if fd < 0 {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        let mut fd = fd;
        if fd < self.next {
            fd = self.next;
        }

        let end = core::i32::MAX;

        let mut fds = Vec::new();
        let mut i = fd;

        while i < end && fds.len() < files.len() {
            let fd = self.descTbl.get(&i);

            match fd {
                None => {
                    self.set(i, &files[fds.len()], flags);
                    fds.push(i);
                }
                _ => ()
            }
            i += 1;
        }

        //fail, undo the change
        if fds.len() < files.len() {
            for i in &fds {
                self.descTbl.remove(i);
            }

            return Err(Error::SysError(SysErr::EMFILE))
        }

        if fd == self.next {
            self.next = fds[fds.len() - 1] + 1;
        }

        return Ok(fds)
    }

    pub fn NewFDAt(&mut self, fd: i32, file: &File, flags: &FDFlags) -> Result<()> {
        if fd < 0 {
            return Err(Error::SysError(SysErr::EBADF))
        }

        self.set(fd, file, flags);
        return Ok(())
    }

    pub fn Dup(&mut self, fd: i32) -> Result<i32> {
        if fd < 0 {
            return Err(Error::SysError(SysErr::EBADF))
        }

        let (f, flags) = self.Get(fd)?;
        return self.NewFDFrom(0, &f, &flags);
    }

    pub fn Dup2(&mut self, oldfd: i32, newfd: i32) -> Result<i32> {
        if oldfd < 0 {
            return Err(Error::SysError(SysErr::EBADF))
        }

        if newfd < 0 {
            return Err(Error::SysError(SysErr::EBADF))
        }

        self.Remove(newfd);

        let (f, flags) = self.Get(oldfd)?;
        self.NewFDAt(newfd, &f, &flags)?;
        return Ok(newfd)
    }

    pub fn Dup3(&mut self, oldfd: i32, newfd: i32, flags: i32) -> Result<i32> {
        if oldfd < 0 {
            return Err(Error::SysError(SysErr::EBADF))
        }

        if newfd < 0 {
            return Err(Error::SysError(SysErr::EBADF))
        }

        self.Remove(newfd);
        let closeOnExec = Flags(flags).CloseOnExec();

        let (f, mut flags) = self.Get(oldfd)?;
        flags.CloseOnExec = closeOnExec;
        self.NewFDAt(newfd, &f, &flags)?;
        return Ok(newfd)
    }

    pub fn SetFlags(&mut self, fd: i32, flags: &FDFlags) -> Result<()> {
        if fd < 0 {
            return Err(Error::SysError(SysErr::EBADF))
        }

        let file = self.descTbl.get_mut(&fd);

        match file {
            None => return Err(Error::SysError(SysErr::EBADF)),
            Some(fdesc) => fdesc.flags = flags.clone(),
        }

        return Ok(())
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

    pub fn Fork(&self) -> FDTableInternal {
        let mut tbl = FDTableInternal {
            next: self.next,
            descTbl: BTreeMap::new(),
        };

        for (fd, file) in &self.descTbl {
            tbl.set(*fd, &file.file, &file.flags)
        }

        return tbl
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
            Some(f) => return Some(f.file)
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
            inotifyFileClose(&desc.file);
        }
    }

    pub fn RemoveAll(&mut self) {
        let mut removed = Vec::new();
        for (fd, _) in &self.descTbl {
            removed.push(*fd);
        }

        for fd in &removed {
            let desc = self.descTbl.remove(fd).unwrap();
            inotifyFileClose(&desc.file);
        }
    }
}

pub fn inotifyFileClose(_f: &File) {
    //todo: will implement it later
}