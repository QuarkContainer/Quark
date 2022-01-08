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

use super::super::super::linux_def::*;
use super::super::kernel::time::*;
use super::super::task::*;
use super::super::super::auth::*;
use super::dentry::*;


#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum InodeFileType {
    Host,
    Mock,
    Full,
    Null,
    Random,
    TTY,
    Zero,
    TaskOwned,
    StaticFile,
    SeqFile,
    RamDir,
    Socket,
    Symlink,
    TmpfsDir,
    TmpfsFifo,
    TmpfsFile,
    TmpfsSocket,
    TmpfsSymlink,
    TTYDir,
    TTYMaster,
    TTYSlave,
    Pipe,
    SimpleFileInode,
    SymlinkNode,
    DirNode,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum InodeType {
    //Dummy InodeType, should not happen
    None,

    // RegularFile is a regular file.
    RegularFile,

    // SpecialFile is a file that doesn't support SeekEnd. It is used for
    // things like proc files.
    SpecialFile,

    // Directory is a directory.
    Directory,

    // SpecialDirectory is a directory that *does* support SeekEnd. It's
    // the opposite of the SpecialFile scenario above. It similarly
    // supports proc files.
    SpecialDirectory,

    // Symlink is a symbolic link.
    Symlink,

    // Pipe is a pipe (named or regular).
    Pipe,

    // Socket is a socket.
    Socket,

    // CharacterDevice is a character device.
    CharacterDevice,

    // BlockDevice is a block device.
    BlockDevice,

    // Anonymous is an anonymous type when none of the above apply.
    // Epoll fds and event-driven fds fit this category.
    Anonymous,
}

impl Default for InodeType {
    fn default() -> Self {
        return InodeType::RegularFile
    }
}

impl InodeType {
    pub fn LinuxType(&self) -> u16 {
        match self {
            Self::RegularFile | Self::SpecialFile => return ModeType::MODE_REGULAR,
            Self::Directory | Self::SpecialDirectory => return ModeType::MODE_DIRECTORY,
            Self::Symlink => return ModeType::MODE_SYMLINK,
            Self::Pipe => return ModeType::MODE_NAMED_PIPE,
            Self::CharacterDevice => return ModeType::MODE_CHARACTER_DEVICE,
            Self::BlockDevice => return ModeType::MODE_BLOCK_DEVICE,
            Self::Socket => return ModeType::MODE_SOCKET,
            _ => return 0,
        }
    }

    pub fn ToType(&self) -> u8 {
        match self {
            Self::RegularFile | Self::SpecialFile => return DType::DT_REG,
            Self::Symlink => return DType::DT_LNK,
            Self::Directory | Self::SpecialDirectory => return DType::DT_DIR,
            Self::Pipe => return DType::DT_FIFO,
            Self::CharacterDevice => return DType::DT_CHR,
            Self::BlockDevice => return DType::DT_BLK,
            Self::Socket => return DType::DT_SOCK,
            _ => return DType::DT_UNKNOWN,
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct StableAttr {
    pub Type: InodeType,
    pub DeviceId: u64,
    pub InodeId: u64,
    pub BlockSize: i64,
    pub DeviceFileMajor: u16,
    pub DeviceFileMinor: u32,
}

impl StableAttr {
    pub fn IsDir(&self) -> bool {
        return self.Type == InodeType::Directory || self.Type == InodeType::SpecialDirectory
    }

    pub fn IsFile(&self) -> bool {
        return self.Type == InodeType::RegularFile || self.Type == InodeType::SpecialFile
    }

    pub fn IsRegular(&self) -> bool {
        return self.Type == InodeType::RegularFile
    }

    pub fn IsSymlink(&self) -> bool {
        return self.Type == InodeType::Symlink
    }

    pub fn IsPipe(&self) -> bool {
        return self.Type == InodeType::Pipe
    }

    pub fn IsSocket(&self) -> bool {
        return self.Type == InodeType::Socket
    }

    pub fn IsCharDevice(&self) -> bool {
        return self.Type == InodeType::CharacterDevice
    }

    pub fn DentAttr(&self) -> DentAttr {
        return DentAttr {
            Type: self.Type,
            InodeId: self.InodeId,
        }
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub struct UnstableAttr {
    pub Size: i64,
    pub Usage: i64,
    pub Perms: FilePermissions,
    pub Owner: FileOwner,
    pub AccessTime: Time,
    pub ModificationTime: Time,
    pub StatusChangeTime: Time,
    pub Links: u64,
}

impl UnstableAttr {
    pub fn SetOwner(&mut self, task: &Task, owner: &FileOwner) {
        if owner.UID.Ok() {
            self.Owner.UID = owner.UID
        }

        if owner.GID.Ok() {
            self.Owner.GID = owner.GID
        }

        self.StatusChangeTime = task.Now();
    }

    pub fn SetPermissions(&mut self, task: &Task, p: &FilePermissions) {
        self.Perms = *p;
        self.StatusChangeTime = task.Now();
    }

    pub fn SetTimestamps(&mut self, task: &Task, ts: &InterTimeSpec) {
        if ts.ATimeOmit && ts.MTimeOmit {
            return
        }

        let now = task.Now();

        if !ts.ATimeOmit {
            if ts.ATimeSetSystemTime {
                self.AccessTime = now;
            } else {
                self.AccessTime = ts.ATime;
            }
        }

        if !ts.MTimeOmit {
            if ts.MTimeSetSystemTime {
                self.ModificationTime = now;
            } else {
                self.ModificationTime = ts.MTime;
            }
        }

        self.StatusChangeTime = now;
    }
}

pub fn WithCurrentTime(task: &Task, u: &UnstableAttr) -> UnstableAttr {
    let t = task.Now();
    let mut res = *u;

    res.AccessTime = t;
    res.ModificationTime = t;
    res.StatusChangeTime = t;
    return res;
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct AttrMask {
    pub Typ: bool,
    pub DeviceID: bool,
    pub InodeID: bool,
    pub BlockSize: bool,
    pub Size: bool,
    pub Usage: bool,
    pub Perms: bool,
    pub UID: bool,
    pub GID: bool,
    pub AccessTime: bool,
    pub ModificationTime: bool,
    pub StatusChangeTime: bool,
    pub Links: bool
}

impl AttrMask {
    pub fn Empty(&self) -> bool {
        return *self == Self::default()
    }
}

