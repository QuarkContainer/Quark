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

use alloc::sync::Arc;
use spin::Mutex;

use super::super::task::*;
use super::super::qlib::auth::*;
use super::super::qlib::device::*;
use super::super::qlib::linux_def::*;
use super::fsutil::inode::simple_file_inode::*;
use super::attr::*;
use super::inode::*;
use super::mount::*;

// NewInode constructs an anonymous Inode that is not associated
// with any real filesystem. Some types depend on completely pseudo
// "anon" inodes (eventfds, epollfds, etc).
pub fn NewAnonInode(task: &Task) -> Inode {
    let perm = FilePermissions {
        User: PermMask {
            read: true,
            write: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let iops = SimpleFileInode::New(task,
                                    &ROOT_OWNER,
                                    &perm,
                                    FSMagic::ANON_INODE_FS_MAGIC,
                                    true,
                                    SimpleFileNode{});

    let deviceId = PSEUDO_DEVICE.lock().id.DeviceID();
    let inodeId = PSEUDO_DEVICE.lock().NextIno();

    let sattr = StableAttr {
        Type: InodeType::Anonymous,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: 4096,
        DeviceFileMajor: 0,
        DeviceFileMinor: 0,
    };

    return Inode::New(&Arc::new(iops),
                      &Arc::new(Mutex::new(MountSource::NewPseudoMountSource())),
                      &sattr);
}