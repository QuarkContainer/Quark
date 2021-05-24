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

pub mod no_readwrite_file;
pub mod static_file;
pub mod static_dir_file_operations;
pub mod fileopsutil;
pub mod readonly_file;
pub mod dynamic_dir_file_operations;

pub use self::static_file::*;
pub use self::no_readwrite_file::*;

use alloc::vec::Vec;
use alloc::string::String;
use alloc::string::ToString;

use super::super::file::*;
use super::super::attr::*;
use super::super::dentry::*;
use super::super::dirent::*;
use super::super::super::kernel::waiter::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::task::*;
use super::super::super::qlib::mem::seq::*;
use super::super::super::qlib::mem::io::*;

pub fn SeekWithDirCursor(task: &Task, f: &File, whence: i32, current: i64, offset: i64, dirCursor: Option<&mut String>) -> Result<i64> {
    let inode = f.Dirent.Inode();

    if inode.StableAttr().IsPipe() || inode.StableAttr().IsSocket() {
        return Err(Error::SysError(SysErr::ESPIPE))
    }

    if inode.StableAttr().IsCharDevice() {
        return Ok(0)
    }

    let fileType = inode.StableAttr().Type;

    if whence == SeekWhence::SEEK_SET {
        match fileType {
            InodeType::RegularFile | InodeType::SpecialFile | InodeType::BlockDevice => {
                if offset < 0 {
                    return Err(Error::SysError(SysErr::EINVAL))
                }

                return Ok(offset)
            }
            InodeType::Directory | InodeType::SpecialDirectory => {
                if offset != 0 {
                    return Err(Error::SysError(SysErr::EINVAL))
                }

                if let Some(cursor) = dirCursor {
                    *cursor = "".to_string();
                }

                return Ok(0)
            }
            _ => return Err(Error::SysError(SysErr::EINVAL))
        }
    } else if whence == SeekWhence::SEEK_CUR {
        match fileType {
            InodeType::RegularFile | InodeType::SpecialFile | InodeType::BlockDevice => {
                if current + offset < 0 {
                    return Err(Error::SysError(SysErr::EINVAL))
                }

                return Ok(current + offset)
            }
            InodeType::Directory | InodeType::SpecialDirectory => {
                if offset != 0 {
                    return Err(Error::SysError(SysErr::EINVAL))
                }

                if let Some(cursor) = dirCursor {
                    *cursor = "".to_string()
                }

                return Ok(current)
            }
            _ => return Err(Error::SysError(SysErr::EINVAL))
        }
    } else if whence == SeekWhence::SEEK_END {
        match fileType {
            InodeType::RegularFile | InodeType::BlockDevice => {
                let sz = inode.UnstableAttr(task).unwrap().Size;

                if core::i64::MAX - sz < offset {
                    return Err(Error::SysError(SysErr::EINVAL))
                }

                return Ok(sz + offset)
            }
            InodeType::SpecialDirectory => {
                if offset != 0 {
                    return Err(Error::SysError(SysErr::EINVAL))
                }

                return Ok(FILE_MAX_OFFSET)
            }
            _ => return Err(Error::SysError(SysErr::EINVAL))
        }
    }

    return Ok(current)
}

pub struct FileGenericSeek {}

impl FileGenericSeek {
    fn Seek(&self, task: &Task, f: &File, whence: i32, current: i64, offset: i64) -> Result<i64> {
        return SeekWithDirCursor(task, f, whence, current, offset, None)
    }
}

pub struct FileZeroSeek {}

impl FileZeroSeek {
    fn Seek(&self, _task: &Task, _f: &File, _whence: i32, _current: i64, _offset: i64) -> Result<i64> {
        return Ok(0)
    }
}

pub struct FileNoSeek {}

impl FileNoSeek {
    fn Seek(&self, _task: &Task, _f: &File, _whence: i32, _current: i64, _offset: i64) -> Result<i64> {
        return Err(Error::SysError(SysErr::EINVAL))
    }
}

pub struct FilePipeSeek {}

impl FilePipeSeek {
    fn Seek(&self, _task: &Task, _f: &File, _whence: i32, _current: i64, _offset: i64) -> Result<i64> {
        return Err(Error::SysError(SysErr::ESPIPE))
    }
}

pub struct FileNotDirReaddir {}

impl FileNotDirReaddir {
    fn ReadDir(&self, _task: &Task, _f: &File, _offset: i64, _serializer: &mut DentrySerializer) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn IterateDir(&self, _task: &Task, _d: &Dirent, _dirCtx: &mut DirCtx, _offset: i32) -> (i32, Result<i64>) {
        return (0, Err(Error::SysError(SysErr::ENOTDIR)))
    }
}

pub struct FileNoFsync {}

impl FileNoFsync {
    fn Fsync(&self, _task: &Task, _f: &File, _start: i64, _end: i64, _syncType: SyncType) -> Result<()> {
        return Err(Error::SysError(SysErr::EINVAL))
    }
}

pub struct FileNoopFsync {}

impl FileNoopFsync {
    fn Fsync(&self, _task: &Task, _f: &File, _start: i64, _end: i64, _syncType: SyncType) -> Result<()> {
        return Ok(())
    }
}

pub struct FileNoopFlush {}

impl FileNoopFlush {
    fn Flush(&self, _task: &Task, _f: &File) -> Result<()> {
        return Ok(())
    }
}

pub struct FileNoIoctl {}

impl FileNoIoctl {
    fn Ioctl(&self, _task: &Task, _f: &File, _fd: i32, _request: u64, _val: u64) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTTY))
    }
}

pub struct FileNoSplice {}

impl FileNoSplice {
    fn ReadAt(&self, _task: &Task, _f: &File, _dsts: &mut [IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOSYS))
    }

    fn WriteAt(&self, _task: &Task, _f: &File, _srcs: &[IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOSYS))
    }
}

pub struct FileNoopWrite {}

impl FileNoopWrite {
    fn WriteAt(&self, _task: &Task, _f: &File, srcs: BlockSeq, _offset: i64) -> Result<i64> {
        return Ok(srcs.NumBytes() as i64)
    }
}

pub struct FileNoWrite {}

impl FileNoWrite {
    fn WriteAt(&self, _task: &Task, _f: &File, _srcs: BlockSeq, _offset: i64) -> Result<i64> {
        return Err(Error::SysError(SysErr::EINVAL))
    }
}

pub struct FileNoopRead {}

impl FileNoopRead {
    fn ReadAt(&self, _task: &Task, _f: &File, _dsts: BlockSeq, _offset: i64) -> Result<i64> {
        return Ok(0)
    }
}

pub struct FileNoRead {}

impl FileNoRead {
    fn ReadAt(&self, _task: &Task, _f: &File, _dsts: BlockSeq, _offset: i64) -> Result<i64> {
        return Err(Error::SysError(SysErr::EINVAL))
    }
}

pub struct DirFileOperations {}

impl DirFileOperations {
    fn Seek(&self, task: &Task, f: &File, whence: i32, current: i64, offset: i64) -> Result<i64> {
        return SeekWithDirCursor(task, f, whence, current, offset, None)
    }

    fn ReadAt(&self, _task: &Task, _f: &File, _dsts: BlockSeq, _offset: i64) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOSYS))
    }

    fn WriteAt(&self, _task: &Task, _f: &File, _srcs: BlockSeq, _offset: i64) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOSYS))
    }

    fn Fsync(&self, _task: &Task, _f: &File, _start: i64, _end: i64, _syncType: SyncType) -> Result<()> {
        return Ok(())
    }

    fn Flush(&self, _task: &Task, _f: &File) -> Result<()> {
        return Ok(())
    }

    fn Ioctl(&self, _task: &Task, _f: &File, _fd: i32, _request: u64, _val: u64) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTTY))
    }
}

pub struct FileUseInodeUnstableAttr {}

impl FileUseInodeUnstableAttr {
    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        let inode = f.Dirent.Inode();
        return inode.UnstableAttr(task);
    }
}

pub struct FileStaticContentReader {
    pub content: Vec<u8>,
}

impl FileStaticContentReader {
    fn ReadAt(&self, _task: &Task, _f: &File, dsts: BlockSeq, offset: i64) -> Result<i64> {
        if offset < 0 {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        if offset as usize > self.content.len() {
            return Ok(0)
        }

        let mut bsw = BlockSeqWriter(dsts);
        let mut ioWriter = ToIOWriter { writer: &mut bsw };
        return ioWriter.Write(&self.content[offset as usize..]);
    }
}

pub struct FileNoMMap {}

impl FileNoMMap {
    fn Mmap(&self, _task: &Task, _f: &File, _len: u64, _hugePage: bool, _offset: u64, _share: bool, _prot: u64) -> Result<u64> {
        return Err(Error::SysError(SysErr::ENODEV))
    }
}

pub struct AlwaysReady {}

impl AlwaysReady {
    fn Readiness(&self, _task: &Task,mask: EventMask) -> EventMask {
        return mask
    }

    fn EventRegister(&self, _task: &Task,_e: &WaitEntry, _mask: EventMask) {

    }

    fn EventUnregister(&self, _task: &Task,_e: &WaitEntry) {

    }
}