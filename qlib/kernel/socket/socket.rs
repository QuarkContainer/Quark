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
use crate::GUEST_HOST_SHARED_ALLOCATOR;
use alloc::boxed::Box;
use alloc::collections::btree_map::BTreeMap;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::AtomicI64;
use core::sync::atomic::Ordering;

//use super::unix::transport::unix::*;
use super::super::super::common::*;
use super::super::super::device::*;
use super::super::super::linux_def::*;
use super::super::super::singleton::*;
use super::super::fs::dirent::*;
use super::super::fs::file::*;
use super::super::fs::filesystems::*;
use super::super::fs::host::fs::*;
use super::super::fs::host::util::*;
use super::super::fs::inode::*;
use super::super::fs::mount::*;
use super::super::task::*;

pub static FAMILIAES: Singleton<QRwLock<Families>> = Singleton::<QRwLock<Families>>::New();
pub static SOCKET_DEVICE: Singleton<Arc<QMutex<Device>>> = Singleton::<Arc<QMutex<Device>>>::New();
pub static UNIX_SOCKET_DEVICE: Singleton<Arc<QMutex<Device>>> =
    Singleton::<Arc<QMutex<Device>>>::New();

// for fd recieved from host unix socket
pub static HOST_FILE_DEVICE: Singleton<Arc<QMutex<Device>>> =
    Singleton::<Arc<QMutex<Device>>>::New();


pub unsafe fn InitSingleton() {
    FAMILIAES.Init(QRwLock::new(Families::New()));
    SOCKET_DEVICE.Init(NewAnonDevice());
    UNIX_SOCKET_DEVICE.Init(NewAnonDevice());
    HOST_FILE_DEVICE.Init(NewAnonDevice());
}

/*
lazy_static! {
    pub static ref FAMILIAES: QRwLock<Families> = QRwLock::new(Families::New());
    pub static ref SOCKET_DEVICE : Arc<QMutex<Device>> = NewAnonDevice();
    pub static ref UNIX_SOCKET_DEVICE : Arc<QMutex<Device>> = NewAnonDevice();
}*/

pub trait Provider: Send + Sync {
    fn Socket(&self, task: &Task, stype: i32, protocol: i32) -> Result<Option<Arc<File>>>;
    fn Pair(
        &self,
        task: &Task,
        stype: i32,
        protocol: i32,
    ) -> Result<Option<(Arc<File>, Arc<File>)>>;
}

pub fn NewSocket(task: &Task, family: i32, stype: i32, protocol: i32) -> Result<Arc<File>> {
    return FAMILIAES.read().NewSocket(task, family, stype, protocol);
}

pub fn NewPair(
    task: &Task,
    family: i32,
    stype: i32,
    protocol: i32,
) -> Result<(Arc<File>, Arc<File>)> {
    return FAMILIAES.read().NewPair(task, family, stype, protocol);
}

pub struct Families {
    map: BTreeMap<i32, Vec<Box<Provider>>>,
}

impl Families {
    pub fn New() -> Self {
        return Self {
            map: BTreeMap::new(),
        };
    }

    pub fn RegisterProvider(&mut self, family: i32, provider: Box<Provider>) {
        if !self.map.contains_key(&family) {
            self.map.insert(family, Vec::new());
        }

        let arr = self.map.get_mut(&family).unwrap();
        arr.push(provider);
    }

    pub fn NewSocket(
        &self,
        task: &Task,
        family: i32,
        stype: i32,
        protocol: i32,
    ) -> Result<Arc<File>> {
        let arr = match self.map.get(&family) {
            None => return Err(Error::SysError(SysErr::EAFNOSUPPORT)),
            Some(a) => a,
        };

        for p in arr {
            let s = p.Socket(task, stype, protocol)?;
            match s {
                None => (),
                Some(s) => return Ok(s),
            }
        }

        return Err(Error::SysError(SysErr::EAFNOSUPPORT));
    }

    pub fn NewPair(
        &self,
        task: &Task,
        family: i32,
        stype: i32,
        protocol: i32,
    ) -> Result<(Arc<File>, Arc<File>)> {
        let arr = match self.map.get(&family) {
            None => return Err(Error::SysError(SysErr::EAFNOSUPPORT)),
            Some(a) => a,
        };

        for p in arr {
            let s = p.Pair(task, stype, protocol)?;
            match s {
                None => (),
                Some(s) => return Ok(s),
            }
        }

        return Err(Error::SysError(SysErr::EAFNOSUPPORT));
    }
}

pub fn NewSocketDirent(task: &Task, _d: Arc<QMutex<Device>>, fd: i32) -> Result<Dirent> {
    let msrc = MountSource::NewHostMountSource(
        &"/".to_string(),
        &task.FileOwner(),
        &WhitelistFileSystem::New(),
        &MountSourceFlags::default(),
        false,
    );

    let mut fstat = Box::new_in(LibcStat::default(), GUEST_HOST_SHARED_ALLOCATOR);
    let ret = Fstat(fd, &mut *fstat);
    if ret < 0 {
        return Err(Error::SysError(-ret as i32));
    }
    let inode = Inode::NewHostInode(task, &Arc::new(QMutex::new(msrc)), fd, &*fstat, true, false, false)?;

    let name = format!("socket:[{}]", fd);
    return Ok(Dirent::New(&inode, &name.to_string()));
}

// for fd recieved from host unix socket
pub fn NewHostfileDirent(task: &Task, _d: Arc<QMutex<Device>>, fd: i32) -> Result<Dirent> {
    let msrc = MountSource::NewHostMountSource(
        &"/".to_string(),
        &task.FileOwner(),
        &WhitelistFileSystem::New(),
        &MountSourceFlags::default(),
        false,
    );

    let mut fstat = Box::new_in(LibcStat::default(), GUEST_HOST_SHARED_ALLOCATOR);
    let ret = Fstat(fd, &mut *fstat);
    if ret < 0 {
        return Err(Error::SysError(-ret as i32));
    }
    let inode = Inode::NewHostInode(task, &Arc::new(QMutex::new(msrc)), fd, &*fstat, true, false, false)?;

    let name = format!("file:[{}]", fd);
    return Ok(Dirent::New(&inode, &name.to_string()));
}

#[derive(Default)]
pub struct SendReceiveTimeout {
    pub send: AtomicI64,
    pub recv: AtomicI64,
}

impl SendReceiveTimeout {
    pub fn SetRecvTimeout(&mut self, ns: i64) {
        self.recv.store(ns, Ordering::Relaxed)
    }

    pub fn SetSendTimeout(&mut self, ns: i64) {
        self.send.store(ns, Ordering::Relaxed)
    }

    pub fn RecvTimeout(&self) -> i64 {
        return self.recv.load(Ordering::Relaxed);
    }

    pub fn SendTimeout(&self) -> i64 {
        return self.send.load(Ordering::Relaxed);
    }
}
