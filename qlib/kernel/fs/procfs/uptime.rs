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
use alloc::sync::Arc;

use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::kernel::kernel::*;
use super::super::super::task::*;
use super::super::attr::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::fsutil::file::readonly_file::*;
use super::super::fsutil::inode::simple_file_inode::*;
use super::super::inode::*;
use super::super::mount::*;
use super::inode::*;

#[derive(Clone)]
pub struct UptimeFileNode {}

impl ReadonlyFileNodeTrait for UptimeFileNode {
    fn ReadAt(
        &self,
        task: &Task,
        _f: &File,
        dsts: &mut [IoVec],
        offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        if offset < 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let kernel = GetKernel();
        let startTime = kernel.startTime;
        let now = task.Now();

        let val = now.Sub(startTime) / 1000_000;
        let second = val / 1000;
        let ms = val % 1000 / 10;
        let s = format!("{}.{} 0.00", second, ms);
        let bytes = s.as_bytes();
        if offset as usize > bytes.len() {
            return Ok(0);
        }

        let n = task.CopyDataOutToIovs(&bytes[offset as usize..], dsts, true)?;

        return Ok(n as i64);
    }
}

pub struct UptimeInode {}

impl SimpleFileTrait for UptimeInode {
    fn GetFile(
        &self,
        _task: &Task,
        _dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        let fops = ReadonlyFileOperations {
            node: UptimeFileNode {}.into(),
        };

        let file = File::New(dirent, &flags, fops.into());
        return Ok(file);
    }
}

pub fn NewUptime(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let node = SimpleFileInode::New(
        task,
        &ROOT_OWNER,
        &FilePermissions {
            User: PermMask {
                read: true,
                write: false,
                execute: false,
            },
            Group: PermMask {
                read: true,
                write: false,
                execute: false,
            },
            Other: PermMask {
                read: true,
                write: false,
                execute: false,
            },
            ..Default::default()
        },
        FSMagic::ANON_INODE_FS_MAGIC,
        false,
        UptimeInode {}.into(),
    );

    return NewProcInode(node.into(), msrc, InodeType::SpecialFile, None);
}
