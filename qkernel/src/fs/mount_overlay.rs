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
use core::any::Any;

use super::super::qlib::common::*;
use super::super::task::*;
use super::mount::*;
use super::inode::*;
use super::dirent::*;
use super::filesystems::*;

pub struct OverlayMountSourceOperations {
    pub upper: Arc<Mutex<MountSource>>,
    pub lower: Arc<Mutex<MountSource>>,
}

pub fn NewOverlayMountSource(upper: &Arc<Mutex<MountSource>>, lower: &Arc<Mutex<MountSource>>, flags: &MountSourceFlags) -> Arc<Mutex<MountSource>> {
    let mut msrc = MountSource::NewOverlayMountSource(&Arc::new(Mutex::new(OverlayMountSourceOperations {
        upper: upper.clone(),
        lower: lower.clone(),
    })), &OverLayFileSystem {}, flags);

    let mut size = lower.lock().fscache.MaxSize();
    if size > upper.lock().fscache.MaxSize() {
        size = upper.lock().fscache.MaxSize()
    }

    msrc.fscache.SetMaxSize(size);
    return Arc::new(Mutex::new(msrc))
}

impl DirentOperations for OverlayMountSourceOperations {
    fn Revalidate(&self, name: &str, parent: &Inode, child: &Inode) -> bool {
        let c = match &child.lock().Overlay {
            None => panic!("overlay cannot revalidate inode that is not an overlay"),
            Some(c) => c.clone(),
        };

        let p = match &parent.lock().Overlay {
            None => panic!("trying to revalidate an overlay inode but the parent is not an overlay"),
            Some(p) => p.clone(),
        };

        match &c.read().lower {
            Some(ref _l) => {
                let parentLower = match &p.read().lower {
                    Some(ref l) => l.clone(),
                    _ => panic!("Revalidatef fail"),
                };

                let childLower = match &c.read().lower {
                    Some(ref l) => l.clone(),
                    _ => panic!("Revalidatef fail"),
                };

                if self.lower.lock().Revalidate(name, &parentLower, &childLower) {
                    panic!("an overlay cannot revalidate file objects from the lower fs")
                }
            }
            None => (),
        }

        match &c.read().upper {
            None => return false,
            _ => (),
        }

        let parentUpper = match &p.read().upper {
            Some(ref u) => u.clone(),
            _ => panic!("Revalidatef fail"),
        };

        let childUpper = match &c.read().upper {
            Some(ref u) => u.clone(),
            _ => panic!("Revalidatef fail"),
        };

        return self.upper.lock().Revalidate(name, &parentUpper, &childUpper)
    }

    fn Keep(&self, dirent: &Dirent) -> bool {
        return self.upper.lock().Keep(dirent)
    }

    fn CacheReadDir(&self) -> bool {
        panic!("not implemented")
    }
}

impl MountSourceOperations for OverlayMountSourceOperations {
    fn as_any(&self) -> &Any {
        return self
    }

    fn Destroy(&mut self) {}

    fn ResetInodeMappings(&mut self) {
        self.upper.lock().MountSourceOperations.lock().ResetInodeMappings();
        self.lower.lock().MountSourceOperations.lock().ResetInodeMappings();
    }

    fn SaveInodeMapping(&mut self, _inode: &Inode, _path: &str) {
        panic!("not implemented")
    }
}

pub struct OverLayFileSystem {}

impl Filesystem for OverLayFileSystem {
    fn Name(&self) -> String {
        return "overlayfs".to_string();
    }

    fn Flags(&self) -> FilesystemFlags {
        return 0;
    }

    fn Mount(&mut self, _task: &Task, _device: &str, _flags: &MountSourceFlags, _date: &str) -> Result<Inode> {
        panic!("overlayFilesystem.Mount should not be called!");
    }

    fn AllowUserMount(&self) -> bool {
        return false;
    }

    fn AllowUserList(&self) -> bool {
        return true;
    }
}


