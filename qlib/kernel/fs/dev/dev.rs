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
use alloc::collections::btree_map::BTreeMap;
use alloc::string::ToString;
use alloc::sync::Arc;

use super::super::super::super::auth::*;
use super::super::super::super::device::*;
use super::super::super::super::linux_def::*;
use super::super::super::task::*;
use super::super::super::uid::NewUID;
use super::super::attr::*;
use super::super::inode::*;
use super::super::mount::*;
use super::super::ramfs::dir::*;
use super::super::ramfs::symlink::*;
use super::full::*;
use super::null::*;
use super::random::*;
use super::tty::*;
use super::zero::*;
use super::proxyfile::*;

const MEM_DEV_MAJOR: u16 = 1;

// Mem device minors.
const NULL_DEV_MINOR: u32 = 3;
const ZERO_DEV_MINOR: u32 = 5;
const FULL_DEV_MINOR: u32 = 7;
const RANDOM_DEV_MINOR: u32 = 8;
const URANDOM_DEV_MINOR: u32 = 9;

fn NewTTYDevice(iops: TTYDevice, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let deviceId = DEV_DEVICE.lock().id.DeviceID();
    let inodeId = DEV_DEVICE.lock().NextIno();

    let stableAttr = StableAttr {
        Type: InodeType::CharacterDevice,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: MemoryDef::PAGE_SIZE as i64,
        DeviceFileMajor: 5,
        DeviceFileMinor: 0,
    };

    let inodeInternal = InodeIntern {
        UniqueId: NewUID(),
        InodeOp: iops.into(),
        StableAttr: stableAttr,
        LockCtx: LockCtx::default(),
        MountSource: msrc.clone(),
        Overlay: None,
        ..Default::default()
    };

    return Inode(Arc::new(QMutex::new(inodeInternal)));
}

fn NewNullDevice(iops: NullDevice, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let deviceId = DEV_DEVICE.lock().id.DeviceID();
    let inodeId = DEV_DEVICE.lock().NextIno();

    let stableAttr = StableAttr {
        Type: InodeType::CharacterDevice,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: MemoryDef::PAGE_SIZE as i64,
        DeviceFileMajor: MEM_DEV_MAJOR,
        DeviceFileMinor: NULL_DEV_MINOR,
    };

    let inodeInternal = InodeIntern {
        UniqueId: NewUID(),
        InodeOp: iops.into(),
        StableAttr: stableAttr,
        LockCtx: LockCtx::default(),
        MountSource: msrc.clone(),
        Overlay: None,
        ..Default::default()
    };

    return Inode(Arc::new(QMutex::new(inodeInternal)));
}

fn NewTestProxyDevice(iops: ProxyDevice, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let deviceId = DEV_DEVICE.lock().id.DeviceID();
    let inodeId = DEV_DEVICE.lock().NextIno();

    let stableAttr = StableAttr {
        Type: InodeType::CharacterDevice,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: MemoryDef::PAGE_SIZE as i64,
        DeviceFileMajor: MEM_DEV_MAJOR,
        DeviceFileMinor: NULL_DEV_MINOR,
    };

    let inodeInternal = InodeIntern {
        UniqueId: NewUID(),
        InodeOp: iops.into(),
        StableAttr: stableAttr,
        LockCtx: LockCtx::default(),
        MountSource: msrc.clone(),
        Overlay: None,
        ..Default::default()
    };

    return Inode(Arc::new(QMutex::new(inodeInternal)));
}

fn NewZeroDevice(iops: ZeroDevice, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let deviceId = DEV_DEVICE.lock().id.DeviceID();
    let inodeId = DEV_DEVICE.lock().NextIno();

    let stableAttr = StableAttr {
        Type: InodeType::CharacterDevice,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: MemoryDef::PAGE_SIZE as i64,
        DeviceFileMajor: MEM_DEV_MAJOR,
        DeviceFileMinor: ZERO_DEV_MINOR,
    };

    let inodeInternal = InodeIntern {
        UniqueId: NewUID(),
        InodeOp: iops.into(),
        StableAttr: stableAttr,
        LockCtx: LockCtx::default(),
        MountSource: msrc.clone(),
        Overlay: None,
        ..Default::default()
    };

    return Inode(Arc::new(QMutex::new(inodeInternal)));
}

fn NewFullDevice(iops: FullDevice, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let deviceId = DEV_DEVICE.lock().id.DeviceID();
    let inodeId = DEV_DEVICE.lock().NextIno();

    let stableAttr = StableAttr {
        Type: InodeType::CharacterDevice,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: MemoryDef::PAGE_SIZE as i64,
        DeviceFileMajor: MEM_DEV_MAJOR,
        DeviceFileMinor: FULL_DEV_MINOR,
    };

    let inodeInternal = InodeIntern {
        UniqueId: NewUID(),
        InodeOp: iops.into(),
        StableAttr: stableAttr,
        LockCtx: LockCtx::default(),
        MountSource: msrc.clone(),
        Overlay: None,
        ..Default::default()
    };

    return Inode(Arc::new(QMutex::new(inodeInternal)));
}

fn NewRandomDevice(iops: RandomDevice, msrc: &Arc<QMutex<MountSource>>, minor: u32) -> Inode {
    let deviceId = DEV_DEVICE.lock().id.DeviceID();
    let inodeId = DEV_DEVICE.lock().NextIno();

    let stableAttr = StableAttr {
        Type: InodeType::CharacterDevice,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: MemoryDef::PAGE_SIZE as i64,
        DeviceFileMajor: MEM_DEV_MAJOR,
        DeviceFileMinor: minor,
    };

    let inodeInternal = InodeIntern {
        UniqueId: NewUID(),
        InodeOp: iops.into(),
        StableAttr: stableAttr,
        LockCtx: LockCtx::default(),
        MountSource: msrc.clone(),
        Overlay: None,
        ..Default::default()
    };

    return Inode(Arc::new(QMutex::new(inodeInternal)));
}

fn NewDirectory(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let iops = Dir::New(
        task,
        BTreeMap::new(),
        &ROOT_OWNER,
        &FilePermissions::FromMode(FileMode(0o0555)),
    );

    let deviceId = PROC_DEVICE.lock().id.DeviceID();
    let inodeId = PROC_DEVICE.lock().NextIno();

    let stableAttr = StableAttr {
        Type: InodeType::Directory,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: MemoryDef::PAGE_SIZE as i64,
        DeviceFileMajor: 0,
        DeviceFileMinor: 0,
    };

    let inodeInternal = InodeIntern {
        UniqueId: NewUID(),
        InodeOp: iops.into(),
        StableAttr: stableAttr,
        LockCtx: LockCtx::default(),
        MountSource: msrc.clone(),
        Overlay: None,
        ..Default::default()
    };

    return Inode(Arc::new(QMutex::new(inodeInternal)));
}

fn NewSymlink(task: &Task, target: &str, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let iops = Symlink::New(task, &ROOT_OWNER, target);

    let deviceId = DEV_DEVICE.lock().id.DeviceID();
    let inodeId = DEV_DEVICE.lock().NextIno();

    let stableAttr = StableAttr {
        Type: InodeType::Symlink,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: MemoryDef::PAGE_SIZE as i64,
        DeviceFileMajor: 0,
        DeviceFileMinor: 0,
    };

    let inodeInternal = InodeIntern {
        UniqueId: NewUID(),
        InodeOp: iops.into(),
        StableAttr: stableAttr,
        LockCtx: LockCtx::default(),
        MountSource: msrc.clone(),
        Overlay: None,
        ..Default::default()
    };

    return Inode(Arc::new(QMutex::new(inodeInternal)));
}

pub fn NewDev(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let mut contents = BTreeMap::new();

    contents.insert(
        "fd".to_string(),
        NewSymlink(task, &"/proc/self/fd".to_string(), msrc),
    );
    contents.insert(
        "stdin".to_string(),
        NewSymlink(task, &"/proc/self/fd/0".to_string(), msrc),
    );
    contents.insert(
        "stdout".to_string(),
        NewSymlink(task, &"/proc/self/fd/1".to_string(), msrc),
    );
    contents.insert(
        "stderr".to_string(),
        NewSymlink(task, &"/proc/self/fd/2".to_string(), msrc),
    );

    contents.insert(
        "null".to_string(),
        NewNullDevice(
            NullDevice::New(task, &ROOT_OWNER, &FileMode(0o0666)),
            msrc,
        ),
    );

    contents.insert(
        "proxy".to_string(),
        NewTestProxyDevice(
            ProxyDevice::New(task, &ROOT_OWNER, &FileMode(0o0666)),
            msrc,
        ),
    );

    contents.insert(
        "zero".to_string(),
        NewZeroDevice(
            ZeroDevice::New(task, &ROOT_OWNER, &FileMode(0o0666)),
            msrc,
        ),
    );
    contents.insert(
        "full".to_string(),
        NewFullDevice(
            FullDevice::New(task, &ROOT_OWNER, &FileMode(0o0666)),
            msrc,
        ),
    );

    contents.insert(
        "shm".to_string(),
        Inode::NewTmpDirInode(task, "/dev/shm").expect("create /dev/shm fail"),
    );

    contents.insert(
        "random".to_string(),
        NewRandomDevice(
            RandomDevice::New(task, &ROOT_OWNER, &FileMode(0o0666)),
            msrc,
            RANDOM_DEV_MINOR,
        ),
    );
    contents.insert(
        "urandom".to_string(),
        NewRandomDevice(
            RandomDevice::New(task, &ROOT_OWNER, &FileMode(0o0666)),
            msrc,
            URANDOM_DEV_MINOR,
        ),
    );

    // A devpts is typically mounted at /dev/pts to provide
    // pseudoterminal support. Place an empty directory there for
    // the devpts to be mounted over.
    //contents.insert("pts".to_string(), NewDirectory(task, msrc));

    // Similarly, applications expect a ptmx device at /dev/ptmx
    // connected to the terminals provided by /dev/pts/. Rather
    // than creating a device directly (which requires a hairy
    // lookup on open to determine if a devpts exists), just create
    // a symlink to the ptmx provided by devpts. (The Linux devpts
    // documentation recommends this).
    //
    // If no devpts is mounted, this will simply be a dangling
    // symlink, which is fine.
    contents.insert(
        "ptmx".to_string(),
        NewSymlink(task, &"pts/ptmx".to_string(), msrc),
    );

    let ttyDevice = TTYDevice::New(task, &ROOT_OWNER, &FileMode(0o0666));
    contents.insert("tty".to_string(), NewTTYDevice(ttyDevice, msrc));

    let iops = Dir::New(
        task,
        contents,
        &ROOT_OWNER,
        &FilePermissions::FromMode(FileMode(0o0555)),
    );

    let deviceId = DEV_DEVICE.lock().id.DeviceID();
    let inodeId = DEV_DEVICE.lock().NextIno();

    let stableAttr = StableAttr {
        Type: InodeType::Directory,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: MemoryDef::PAGE_SIZE as i64,
        DeviceFileMajor: 0,
        DeviceFileMinor: 0,
    };

    let inodeInternal = InodeIntern {
        UniqueId: NewUID(),
        InodeOp: iops.into(),
        StableAttr: stableAttr,
        LockCtx: LockCtx::default(),
        MountSource: msrc.clone(),
        Overlay: None,
        ..Default::default()
    };

    return Inode(Arc::new(QMutex::new(inodeInternal)));
}
