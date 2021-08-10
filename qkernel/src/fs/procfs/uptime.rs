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
use spin::Mutex;

use super::super::attr::*;
use super::super::file::*;
use super::super::inode::*;
use super::super::mount::*;
use super::super::flags::*;
use super::super::dirent::*;
use super::super::super::kernel::kernel::*;
use super::super::super::qlib::linux::time::*;
use super::super::super::task::*;
use super::super::super::qlib::auth::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::common::*;
use super::super::fsutil::inode::simple_file_inode::*;
use super::super::fsutil::file::readonly_file::*;
use super::inode::*;

pub struct UptimeFileNode {}

impl ReadonlyFileNode for UptimeFileNode {
    fn ReadAt(&self, task: &Task, _f: &File, dsts: &mut [IoVec], offset: i64, _blocking: bool) -> Result<i64> {
        if offset < 0 {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        let kernel = GetKernel();
        let startTime = kernel.startTime;
        let now = task.Now();

        let dur = now.Sub(startTime) / SECOND;
        let s = format!("{}", dur);
        let bytes = s.as_bytes();
        if offset as usize > bytes.len() {
            return Ok(0)
        }

        let n = task.CopyDataOutToIovs(&bytes[offset as usize..], dsts)?;

        return Ok(n as i64)
    }
}

pub struct UptimeInode {}

impl SimpleFileTrait for UptimeInode {
    fn GetFile(&self, _task: &Task, _dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        let fops = ReadonlyFileOperations {
            node: UptimeFileNode{},
        };

        let file = File::New(dirent, &flags, fops);
        return Ok(file)
    }
}

pub fn NewUptime(task: &Task, msrc: &Arc<Mutex<MountSource>>) -> Inode {
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
        UptimeInode{}
    );

    return NewProcInode(&Arc::new(node), msrc, InodeType::SpecialFile, None)
}