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
use alloc::collections::btree_map::BTreeMap;
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use alloc::boxed::Box;

use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::path::*;
use super::super::super::task::*;
use super::super::filesystems::*;
use super::super::inode::*;
use super::super::mount::*;
use super::util::*;

pub const MAX_TRAVERSALS: u32 = 10;
pub const FILESYSTEM_NAME: &str = "whitelistfs";
pub const WHITELIST_KEY: &str = "whitelist";
pub const ROOT_PATH_KEY: &str = "root";
pub const DONT_TRANSLATE_OWNERSHIP_KEY: &str = "dont_translate_ownership";

pub struct WhitelistFileSystem {
    pub paths: Vec<String>,
}

impl WhitelistFileSystem {
    pub fn New() -> Self {
        return Self { paths: Vec::new() };
    }

    pub fn GenericMountSourceOptions(data: &str) -> BTreeMap<String, String> {
        let mut options = BTreeMap::new();

        if data.len() == 0 {
            return options;
        }

        let v: Vec<&str> = data.split(',').collect();
        for opt in v {
            if opt.len() > 0 {
                let res: Vec<&str> = opt.split('=').collect();
                if res.len() == 2 {
                    options.insert(res[0].to_string(), res[1].to_string());
                } else {
                    options.insert(opt.to_string(), "".to_string());
                }
            }
        }

        return options;
    }

    pub fn InstallWhitelist(&self, task: &Task, m: &MountNs) -> Result<()> {
        InstallWhitelist(task, m, &self.paths)
    }
}

pub fn InstallWhitelist(task: &Task, m: &MountNs, inputPaths: &Vec<String>) -> Result<()> {
    if inputPaths.len() == 0 || (inputPaths.len() == 1 && inputPaths[0].as_str() == "") {
        return Ok(());
    }

    let mut done = BTreeMap::new();

    let root = m.Root();

    let mut paths = Vec::new();
    for p in inputPaths {
        paths.push(p.clone())
    }

    let mut i = 0;
    while i < paths.len() {
        let p = paths[i].clone();

        i += 1;

        if !IsAbs(&p) {
            return Err(Error::Common(
                "InstallWhitelist: path should not absoluted".to_string(),
            ));
        }

        let s = p.as_bytes();
        for j in 1..s.len() + 1 {
            if j < s.len() && s[j] != '/' as u8 {
                continue;
            }

            let current = String::from_utf8(s[..j].to_vec()).unwrap();

            let mut remainingTraversals = MAX_TRAVERSALS;
            let d = m.FindDirent(task, &root, None, &current, &mut remainingTraversals, false);
            let dir = match d {
                Err(e) => {
                    info!("populate failed for {}, {:?}", current, e);
                    continue;
                }
                Ok(d) => d,
            };

            if current == p {
                let inode = dir.Inode();
                let sattr = inode.lock().StableAttr().clone();

                if sattr.IsDir() {
                    for (name, _) in dir.ChildDenAttrs(task)? {
                        paths.push(Join(&current, &name))
                    }
                }

                if sattr.IsSymlink() {
                    match done.get(&current) {
                        None => continue,
                        Some(_) => (),
                    }

                    let s = match inode.ReadLink(task) {
                        Err(_) => continue,
                        Ok(s) => s,
                    };

                    if IsAbs(&s) {
                        paths.push(s)
                    } else {
                        let target = Join(&Dir(&current), &s);
                        paths.push(target);
                    }
                }
            }

            if !done.contains_key(&current) {
                done.insert(current, dir.clone());
            }
        }
    }

    return Ok(());
}

impl Filesystem for WhitelistFileSystem {
    fn Name(&self) -> String {
        return FILESYSTEM_NAME.to_string();
    }

    fn Flags(&self) -> FilesystemFlags {
        return 0;
    }

    fn Mount(
        &mut self,
        task: &Task,
        _device: &str,
        flags: &MountSourceFlags,
        data: &str,
    ) -> Result<Inode> {
        let mut options = Self::GenericMountSourceOptions(data);

        let remove = if let Some(ref wl) = options.get(&WHITELIST_KEY.to_string()) {
            let v: Vec<&str> = wl.split('|').collect();
            self.paths = v.iter().map(|x| x.to_string()).collect();
            true
        } else {
            false
        };

        if remove {
            options.remove(&WHITELIST_KEY.to_string());
        }

        let mut rootPath = "/".to_string();

        let remove = if let Some(rp) = options.get(&ROOT_PATH_KEY.to_string()) {
            rootPath = rp.clone();

            for i in 0..self.paths.len() {
                let p = &self.paths[i];
                let rel = Rel(&rootPath, p)?;
                self.paths[i] = Join(&"/".to_string(), &rel);
            }

            true
        } else {
            false
        };

        if remove {
            options.remove(&ROOT_PATH_KEY.to_string());
        }

        let (fd, writable, _) = TryOpenAt(-100, &rootPath, false)?;

        if fd < 0 {
            return Err(Error::SysError(-fd));
        }

        let mut dontTranslateOwnership = false;
        let remove = if let Some(ref v) = options.get(&DONT_TRANSLATE_OWNERSHIP_KEY.to_string()) {
            let b = ParseBool(v)?;
            dontTranslateOwnership = b;
            true
        } else {
            false
        };

        if remove {
            options.remove(&DONT_TRANSLATE_OWNERSHIP_KEY.to_string());
        }

        if options.len() > 0 {
            return Err(Error::Common("unsupported mount options".to_string()));
        }

        let owner = task.Creds().FileOwner();

        let msrc =
            MountSource::NewHostMountSource(&rootPath, &owner, self, flags, dontTranslateOwnership);

        let mut fstat = Box::new_in(LibcStat::default(), GUEST_HOST_SHARED_ALLOCATOR);
        let ret = Fstat(fd, &mut *fstat);
        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }
        let inode = Inode::NewHostInode(
            task,
            &Arc::new(QMutex::new(msrc)),
            fd,
            &*fstat,
            writable,
            false,
            false,
        )?;

        return Ok(inode);
    }

    fn AllowUserMount(&self) -> bool {
        return false;
    }

    fn AllowUserList(&self) -> bool {
        return true;
    }
}
