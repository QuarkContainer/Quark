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

use alloc::sync::Arc;
use spin::Mutex;
use alloc::string::String;
use alloc::string::ToString;
use alloc::collections::btree_map::BTreeMap;

use super::super::super::qlib::auth::id::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::task::*;
use super::super::filesystems::*;
use super::super::host::fs::*;
use super::super::inode::*;
use super::super::mount::*;
use super::tmpfs_dir::*;

// Set initial permissions for the root directory.
pub const MODE_KEY :&str = "mode";

// UID for the root directory.
pub const ROOT_UIDKEY :&str = "uid";

// GID for the root directory.
pub const ROOT_GIDKEY :&str = "gid";

// Permissions that exceed modeMask will be rejected.
pub const MODE_MASK :u16 = 0o1777;

// Default permissions are read/write/execute.
pub const DEFAULT_MODE :u16 = 0o777;

pub struct TmpfsFileSystem {}

impl Filesystem for TmpfsFileSystem {
    fn Name(&self) -> String {
        return "tmpfs".to_string();
    }

    fn Flags(&self) -> FilesystemFlags {
        return 0;
    }

    fn Mount(&mut self, task: &Task, _device: &str, flags: &MountSourceFlags, data: &str) -> Result<Inode> {
        info!("tmfps file system mount ...");

        // Parse generic comma-separated key=value options, this file system expects them.
        let mut options = WhitelistFileSystem::GenericMountSourceOptions(data);

        // Parse the root directory permissions.
        let mut perms = FilePermissions::FromMode(FileMode(DEFAULT_MODE));

        match options.remove(MODE_KEY) {
            None => (),
            Some(m) => {
                let i = match m.parse::<u16>() {
                    Ok(v) => v,
                    Err(e) => {
                        info!("mode value not parsable 'mode={}': {:?}", m, e);
                        return Err(Error::SysError(SysErr::EINVAL))
                    }
                };

                if i & !MODE_MASK != 0 {
                    info!("invalid mode {}: must be less than {}", m, MODE_MASK);
                    return Err(Error::SysError(SysErr::EINVAL))
                }

                perms = FilePermissions::FromMode(FileMode(i as u16))
            }
        }

        let creds = task.Creds();
        let userns = creds.lock().UserNamespace.clone();
        let mut owner = task.FileOwner();

        match options.remove(ROOT_UIDKEY) {
            None => (),
            Some(uidstr) => {
                let uid = match uidstr.parse::<u32>() {
                    Ok(v) => v,
                    Err(e) => {
                        info!("uid value not parsable 'uid={}': {:?}", uidstr, e);
                        return Err(Error::SysError(SysErr::EINVAL))
                    }
                };

                owner.UID = userns.MapToKUID(UID(uid));
            }
        }

        match options.remove(ROOT_GIDKEY) {
            None => (),
            Some(gidstr) => {
                let gid = match gidstr.parse::<u32>() {
                    Ok(v) => v,
                    Err(e) => {
                        info!("gid value not parsable 'gid={}': {:?}", gidstr, e);
                        return Err(Error::SysError(SysErr::EINVAL))
                    }
                };

                owner.GID = userns.MapToKGID(GID(gid));
            }
        }

        // Fail if the caller passed us more options than we can parse. They may be
        // expecting us to set something we can't set.
        if options.len() > 0 {
            info!("unsupported mount options: {:?}", options);
            return Err(Error::SysError(SysErr::EINVAL))
        }

        let msrc = MountSource::NewCachingMountSource(self, flags);

        let inode = NewTmpfsDir(task, BTreeMap::new(), &owner, &perms, Arc::new(Mutex::new(msrc)));
        return Ok(inode)
    }

    fn AllowUserMount(&self) -> bool {
        return true;
    }

    fn AllowUserList(&self) -> bool {
        return true;
    }
}
