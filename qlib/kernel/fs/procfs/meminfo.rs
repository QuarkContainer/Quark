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

use alloc::sync::Arc;
use alloc::string::ToString;
use crate::qlib::mutex::*;

use super::super::attr::*;
use super::super::file::*;
use super::super::inode::*;
use super::super::mount::*;
use super::super::flags::*;
use super::super::dirent::*;
//use super::super::super::super::linux::time::*;
use super::super::super::task::*;
use super::super::super::super::auth::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::common::*;
use super::super::fsutil::inode::simple_file_inode::*;
use super::super::fsutil::file::readonly_file::*;
use super::inode::*;

pub struct MeminfoFileNode {}

impl ReadonlyFileNode for MeminfoFileNode {
    fn ReadAt(&self, task: &Task, _f: &File, dsts: &mut [IoVec], offset: i64, _blocking: bool) -> Result<i64> {
        if offset < 0 {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        let mut s = "".to_string();

        // this is just dummy meminfo
        // todo: fix this.
        let gb : u64 = 1024 * 1024 * 1024;
        s += &format!("MemTotal:       {:08} kB\n", 12 * gb / 1024);
        s += &format!("MemFree:        {:08} kB\n", 8 * gb / 1024);
        s += &format!("MemAvailable:   {:08} kB\n", 4 * gb / 1024);
        s += &format!("Buffers:        {:08} kB\n", 1 * gb / 1024); // memory usage by block devices
        s += &format!("Cached:         {:08} kB\n", 1 * gb / 1024);
        // Emulate a system with no swap, which disables inactivation of anon pages.
        s += &format!("SwapCache:             0 kB\n");
        s += &format!("Active:         {:08} kB\n", 1 * gb / 1024);
        s += &format!("Inactive:       {:08} kB\n", 1 * gb / 1024);
        s += &format!("Active(anon):   {:08} kB\n", 1 * gb / 1024);
        s += &format!("Inactive(anon):        0 kB\n");
        s += &format!("Active(file):   {:08} kB\n", 1 * gb / 1024);
        s += &format!("Inactive(file): {:08} kB\n", 1 * gb / 1024);
        s += &format!("Unevictable:           0 kB\n");
        s += &format!("Mlocked:               0 kB\n");
        s += &format!("SwapTotal:             0 kB\n");
        s += &format!("SwapFree:              0 kB\n");
        s += &format!("Dirty:                 0 kB\n");
        s += &format!("Writeback:             0 kB\n");
        s += &format!("AnonPages:      {:08} kB\n", 1 * gb / 1024);
        s += &format!("Mapped:         {:08} kB\n", 1 * gb / 1024);
        s += &format!("Shmem:          {:08} kB\n", 1 * gb / 1024);

        let bytes = s.as_bytes();
        if offset as usize > bytes.len() {
            return Ok(0)
        }

        let n = task.CopyDataOutToIovs(&bytes[offset as usize..], dsts)?;

        return Ok(n as i64)
    }
}

pub struct MeminfoInode {}

impl SimpleFileTrait for MeminfoInode {
    fn GetFile(&self, _task: &Task, _dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        let fops = ReadonlyFileOperations {
            node: MeminfoFileNode{},
        };

        let file = File::New(dirent, &flags, fops);
        return Ok(file)
    }
}

pub fn NewMeminfo(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let node = SimpleFileInode::New (
        task,
        &ROOT_OWNER,
        &FilePermissions{
            User : PermMask {
                read: true,
                write: false,
                execute: false,
            },
            Group : PermMask {
                read: true,
                write: false,
                execute: false,
            },
            Other : PermMask {
                read: true,
                write: false,
                execute: false,
            },
            ..Default::default()
        },
        FSMagic::ANON_INODE_FS_MAGIC,
        false,
        MeminfoInode{}
    );

    return NewProcInode(&Arc::new(node), msrc, InodeType::SpecialFile, None)
}