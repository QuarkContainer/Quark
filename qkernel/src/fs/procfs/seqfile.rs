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

use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use ::qlib::mutex::*;
use spin::RwLock;
use alloc::vec::Vec;
use core::any::Any;
use core::ops::Deref;
use alloc::collections::btree_map::BTreeMap;

use socket::unix::transport::unix::BoundEndpoint;
use super::super::host::hostinodeop::*;
use super::super::fsutil::file::*;
use super::super::attr::*;
use super::super::flags::*;
use super::super::inode::*;
use super::super::dentry::*;
use super::super::file::*;
use super::super::dirent::*;
use super::super::mount::*;
use super::super::super::kernel::waiter::*;
use super::super::super::task::*;
use super::super::super::kernel::time::*;
use super::super::super::kernel::waiter::qlock::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::auth::*;
use super::super::super::uid::*;

#[derive(Clone, Copy)]
pub enum SeqHandle {
    None,
    Simple(u64),
}

#[derive(Clone)]
pub struct SeqData {
    pub Buf: Vec<u8>,
    pub Handle: SeqHandle,
}

pub trait SeqSource {
    fn NeedsUpdate(&mut self, generation: i64) -> bool;
    fn ReadSeqFileData(&mut self, task: &Task, handle: SeqHandle) -> (Vec<SeqData>, i64);
}

pub struct SeqGenerationCounter {
    pub generation: i64,
}

impl SeqGenerationCounter {
    pub fn SetGeneration(&mut self, generation: i64) {
        self.generation = generation;
    }

    pub fn Update(&mut self) {
        self.generation += 1;
    }

    pub fn Generation(&self) -> i64 {
        return self.generation
    }

    pub fn IsCurrent(&self, generation: i64) -> bool {
        return self.generation == generation
    }
}

pub struct SeqFileInternal {
    pub xattrs: BTreeMap<String, String>,

    pub fsType: u64,
    pub unstable: UnstableAttr,

    pub seqSource: Arc<QMutex<SeqSource>>,

    pub source: Vec<SeqData>,
    pub generation: i64,
    pub lastRead: i64,
}

impl SeqFileInternal {
    pub fn UpdateSource(&mut self, task: &Task, record: usize) {
        info!("SeqFileInternal UpdateSource");

        let h = if record == 0 {
            SeqHandle::None
        } else {
            self.source[record - 1].Handle
        };

        self.source = self.source[..record].to_vec();

        let (mut newSource, generation) = self.seqSource.lock().ReadSeqFileData(task, h);
        self.generation = generation;
        self.source.append(&mut newSource);
    }
}

fn findIndexAndOffset(data: &[SeqData], offset: usize) -> (usize, usize) {
    let mut i = 0;

    let mut offset = offset;
    for buf in data {
        let l = buf.Buf.len();
        if offset < l {
            return (i, offset)
        }

        offset -= l;
        i += 1;
    }

    return (data.len(), offset)
}

#[derive(Clone)]
pub struct SeqFile(Arc<RwLock<SeqFileInternal>>);

unsafe impl Send for SeqFile {}

unsafe impl Sync for SeqFile {}

impl Deref for SeqFile {
    type Target = Arc<RwLock<SeqFileInternal>>;

    fn deref(&self) -> &Arc<RwLock<SeqFileInternal>> {
        &self.0
    }
}

impl SeqFile {
    pub fn New(task: &Task, source: Arc<QMutex<SeqSource>>) -> Self {
        let unstable = WithCurrentTime(task, &UnstableAttr {
            Owner: ROOT_OWNER,
            Perms: FilePermissions::FromMode(FileMode(0o444)),
            ..Default::default()
        });

        let internal = SeqFileInternal {
            xattrs: BTreeMap::new(),
            fsType: FSMagic::PROC_SUPER_MAGIC,
            unstable: unstable,
            seqSource: source,
            source: Vec::new(),
            generation: 0,
            lastRead: 0,
        };

        return Self(Arc::new(RwLock::new(internal)))
    }
}

impl InodeOperations for SeqFile {
    fn as_any(&self) -> &Any {
        return self
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::SeqFile;
    }

    fn InodeType(&self) -> InodeType {
        return InodeType::SpecialFile;
    }

    fn InodeFileType(&self) -> InodeFileType{
        return InodeFileType::SeqFile;
    }

    fn WouldBlock(&self) -> bool {
        return false;
    }

    fn Lookup(&self, _task: &Task, _dir: &Inode, _name: &str) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn Create(&self, _task: &Task, _dir: &mut Inode, _name: &str, _flags: &FileFlags, _perm: &FilePermissions) -> Result<File> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn CreateDirectory(&self, _task: &Task, _dir: &mut Inode, _name: &str, _perm: &FilePermissions) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn CreateLink(&self, _task: &Task, _dir: &mut Inode, _oldname: &str, _newname: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn CreateHardLink(&self, _task: &Task, _dir: &mut Inode, _target: &Inode, _name: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn CreateFifo(&self, _task: &Task, _dir: &mut Inode, _name: &str, _perm: &FilePermissions) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn Remove(&self, _task: &Task, _dir: &mut Inode, _name: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn RemoveDirectory(&self, _task: &Task, _dir: &mut Inode, _name: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn Rename(&self, _task: &Task, _dir: &mut Inode, _oldParent: &Inode, _oldname: &str, _newParent: &Inode, _newname: &str, _replacement: bool) -> Result<()> {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    fn Bind(&self, _task: &Task, _dir: &Inode, _name: &str, _data: &BoundEndpoint, _perms: &FilePermissions) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn BoundEndpoint(&self, _task: &Task, _inode: &Inode, _path: &str) -> Option<BoundEndpoint> {
        return None
    }

    fn GetFile(&self, _task: &Task, _dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        let fops = Arc::new(SeqFileOperations {
            seqFile: self.clone()
        });

        let internal = FileInternal {
            UniqueId: NewUID(),
            Dirent: dirent.clone(),
            flags: QMutex::new((flags, None)),
            offset: QLock::New(0),
            FileOp: fops,
        };

        return Ok(File(Arc::new(internal)))
    }

    fn UnstableAttr(&self, task: &Task, _dir: &Inode) -> Result<UnstableAttr> {
        let mut u = self.write().unstable;

        u.ModificationTime = task.Now();
        return Ok(u)
    }

    fn Getxattr(&self, _dir: &Inode, name: &str) -> Result<String> {
        match self.read().xattrs.get(name) {
            None => Err(Error::SysError(SysErr::ENOATTR)),
            Some(s) => Ok(s.clone())
        }
    }

    fn Setxattr(&self, _dir: &mut Inode, name: &str, value: &str) -> Result<()> {
        self.write().xattrs.insert(name.to_string(), value.to_string());
        return Ok(())
    }

    fn Listxattr(&self, _dir: &Inode) -> Result<Vec<String>> {
        let mut res = Vec::new();
        for (name, _) in &self.read().xattrs {
            res.push(name.clone());
        }

        return Ok(res)
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return ContextCanAccessFile(task, inode, reqPerms)
    }

    fn SetPermissions(&self, task: &Task, _dir: &mut Inode, p: FilePermissions) -> bool {
        self.write().unstable.SetPermissions(task, &p);
        return true;
    }

    fn SetOwner(&self, task: &Task, _dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        self.write().unstable.SetOwner(task, owner);
        return Ok(())
    }

    fn SetTimestamps(&self, task: &Task, _dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        self.write().unstable.SetTimestamps(task, ts);
        return Ok(())
    }

    fn Truncate(&self, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    fn Allocate(&self, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP))
    }

    fn ReadLink(&self, _task: &Task,_dir: &Inode) -> Result<String> {
        return Err(Error::SysError(SysErr::ENOLINK))
    }

    fn GetLink(&self, _task: &Task, _dir: &Inode) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOLINK))
    }

    fn AddLink(&self, _task: &Task) {
        self.write().unstable.Links += 1;
    }

    fn DropLink(&self, _task: &Task) {
        self.write().unstable.Links -= 1;
    }

    fn IsVirtual(&self) -> bool {
        return true
    }

    fn Sync(&self) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOSYS));
    }

    fn StatFS(&self, _task: &Task) -> Result<FsInfo> {
        if self.read().fsType == 0 {
            return Err(Error::SysError(SysErr::ENOSYS))
        }

        return Ok(FsInfo { Type: self.read().fsType, ..Default::default() })
    }

    fn Mappable(&self) -> Result<HostInodeOp> {
        return Err(Error::SysError(SysErr::ENODEV))
    }
}

pub struct SeqFileOperations {
    pub seqFile: SeqFile,
}

impl Waitable for SeqFileOperations {}

impl SpliceOperations for SeqFileOperations {}

impl FileOperations for SeqFileOperations {
    fn as_any(&self) -> &Any {
        return self
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::SeqFileOperations
    }

    fn Seekable(&self) -> bool {
        return true;
    }

    fn Seek(&self, task: &Task, f: &File, whence: i32, current: i64, offset: i64) -> Result<i64> {
        return SeekWithDirCursor(task, f, whence, current, offset, None)
    }

    fn ReadDir(&self, _task: &Task, _f: &File, _offset: i64, _serializer: &mut DentrySerializer) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn ReadAt(&self, task: &Task, _f: &File, dsts: &mut [IoVec], offset: i64, _blocking: bool) -> Result<i64> {
        let size = IoVec::NumBytes(dsts);
        let dataBuf = DataBuff::New(size);
        let bs = dataBuf.BlockSeq();

        let mut file = self.seqFile.write();

        let mut updated = false;
        let (mut i, mut recordOffset) = findIndexAndOffset(&file.source, offset as usize);

        if i == file.source.len() {
            if !file.seqSource.lock().NeedsUpdate(file.generation) {
                file.lastRead = offset;
                return Ok(0)
            }

            let oldlen = file.source.len();
            file.UpdateSource(task, oldlen);
            updated = true;

            let tmp = findIndexAndOffset(&file.source[oldlen..], recordOffset as usize);
            i = tmp.0;
            recordOffset = tmp.1;
            i += oldlen;

            if i == file.source.len() {
                file.lastRead = offset;
                return Ok(0)
            }
        }

        let mut bs = bs;
        let mut done = 0;
        if recordOffset != 0 {
            let n = bs.CopyOut(&file.source[i].Buf[recordOffset..]);
            done += n;
            bs = bs.DropFirst(n as u64);
            if bs.NumBytes() == 0 {
                file.lastRead = offset;
                task.CopyDataOutToIovs(&dataBuf.buf[0..done], dsts)?;
                return Ok(done as i64)
            }

            i += 1;
        }

        if !updated && (file.seqSource.lock().NeedsUpdate(file.generation) || file.lastRead > offset) {
            file.UpdateSource(task, i);
        }

        for buf in &file.source[i..] {
            let n = bs.CopyOut(&buf.Buf);
            done += n;

            bs = bs.DropFirst(n as u64);
            if bs.NumBytes() == 0 {
                file.lastRead = offset;
                task.CopyDataOutToIovs(&dataBuf.buf[0..done], dsts)?;
                return Ok(done as i64)
            }
        }

        info!("SeqFileOperations ReadAt 5, done = {}", done);
        file.lastRead = offset;
        task.CopyDataOutToIovs(&dataBuf.buf[0..done], dsts)?;
        return Ok(done as i64)
    }

    fn WriteAt(&self, _task: &Task, _f: &File, _srcs: &[IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
        return Err(Error::SysError(SysErr::EACCES))
    }

    fn Append(&self, task: &Task, f: &File, srcs: &[IoVec]) -> Result<(i64, i64)> {
        let n = self.WriteAt(task, f, srcs, 0, false)?;
        return Ok((n, 0))
    }

    fn Fsync(&self, _task: &Task, _f: &File, _start: i64, _end: i64, _syncType: SyncType) -> Result<()> {
        return Ok(())
    }

    fn Flush(&self, _task: &Task, _f: &File) -> Result<()> {
        return Ok(())
    }

    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        let inode = f.Dirent.Inode();
        return inode.UnstableAttr(task);

    }

    fn Ioctl(&self, _task: &Task, _f: &File, _fd: i32, _request: u64, _val: u64) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTTY))
    }

    fn IterateDir(&self, _task: &Task, _d: &Dirent, _dirCtx: &mut DirCtx, _offset: i32) -> (i32, Result<i64>) {
        return (0, Err(Error::SysError(SysErr::ENOTDIR)))
    }

    fn Mappable(&self) -> Result<HostInodeOp> {
        return Err(Error::SysError(SysErr::ENODEV))
    }
}

impl SockOperations for SeqFileOperations {}