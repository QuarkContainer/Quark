// Copyright (c) 2021 QuarkSoft LLC
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
use alloc::sync::Arc;
use spin::Mutex;
use alloc::collections::btree_map::BTreeMap;
use alloc::vec::Vec;
use lazy_static::lazy_static;

use super::super::qlib::common::*;
use super::super::task::*;
use super::inode::*;

pub type FilesystemFlags = i32;

// FilesystemRequiresDev indicates that the file system requires a device name
// on mount. It is used to construct the output of /proc/filesystems.
pub const FILESYSTEM_REQUIRES_DEV : FilesystemFlags = 1;

lazy_static! {
    pub static ref FILESYSTEMS: Mutex<FileSystems> = Mutex::new(FileSystems::New());
}

pub fn FindFilesystem(name: &str) -> Option<Arc<Mutex<Filesystem>>> {
    return FILESYSTEMS.lock().FindFilesystem(name);
}

pub fn RegisterFilesystem<T: Filesystem + 'static>(f: &Arc<Mutex<T>>) {
    FILESYSTEMS.lock().RegisterFilesystem(f)
}

pub fn GetFilesystems() -> Vec<Arc<Mutex<Filesystem>>> {
    return FILESYSTEMS.lock().GetFilesystems()
}

#[derive(Debug, Clone, Default, Copy)]
pub struct MountSourceFlags {
    pub ReadOnly: bool,
    pub NoAtime: bool,
    pub ForcePageCache: bool,
    pub NoExec: bool,
}

pub struct FileSystems {
    pub registered: BTreeMap<String, Arc<Mutex<Filesystem>>>
}

impl FileSystems {
    pub fn New() -> Self {
        return Self {
            registered: BTreeMap::new(),
        }
    }

    pub fn RegisterFilesystem<T: Filesystem + 'static>(&mut self, f: &Arc<Mutex<T>>) {
        let name = f.lock().Name();
        if let Some(_) = self.registered.get(&name) {
            panic!("filesystem already registered at {}", name)
        }

        self.registered.insert(name, f.clone());
    }

    pub fn FindFilesystem(&self, name: &str) -> Option<Arc<Mutex<Filesystem>>> {
        match self.registered.get(name) {
            None => None,
            Some(f) => Some(f.clone())
        }
    }

    pub fn GetFilesystems(&self) -> Vec<Arc<Mutex<Filesystem>>> {
        let mut res = Vec::new();
        for (_, f) in &self.registered {
            res.push(f.clone())
        }

        return res;
    }
}

pub trait Filesystem: Send {
    fn Name(&self) -> String;
    fn Flags(&self) -> FilesystemFlags;
    fn Mount(&mut self, task: &Task, device: &str, flags: &MountSourceFlags, data: &str) -> Result<Inode>;
    fn AllowUserMount(&self) -> bool;
    fn AllowUserList(&self) -> bool;
}
