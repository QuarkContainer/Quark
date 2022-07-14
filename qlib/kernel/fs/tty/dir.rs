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
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::Any;
use core::ops::Deref;

use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::device::*;
use super::super::super::super::linux_def::*;
use super::super::super::kernel::time::*;
use super::super::super::kernel::waiter::*;
use super::super::super::socket::unix::transport::unix::*;
use super::super::super::task::*;
use super::super::super::uid::NewUID;
use super::super::attr::*;
use super::super::dentry::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::fsutil::file::*;
use super::super::host::hostinodeop::*;
use super::super::inode::*;
use super::super::mount::*;
use super::master::*;
use super::terminal::*;

pub fn NewDir(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let unstable = WithCurrentTime(
        task,
        &UnstableAttr {
            Owner: ROOT_OWNER,
            Perms: FilePermissions::FromMode(FileMode(0o555)),
            ..Default::default()
        },
    );

    let d = DirInodeOperations(Arc::new(QMutex::new(DirInodeOperationsInternal {
        fsType: FSMagic::DEVPTS_SUPER_MAGIC as i64,
        unstable: unstable,
        msrc: msrc.clone(),
        master: Inode::default(),
        slaves: BTreeMap::new(),
        dentryMap: DentMap::New(BTreeMap::new()),
        next: 0,
    })));

    let master = NewMasterNode(
        task,
        &d,
        &ROOT_OWNER,
        &FilePermissions::FromMode(FileMode(0o666)),
    );

    let stableAttr = master.lock().StableAttr().clone();
    {
        let mut dLocked = d.lock();

        dLocked.master = master;
        dLocked.dentryMap.Add(
            &"ptmx".to_string(),
            &DentAttr {
                Type: stableAttr.Type,
                InodeId: stableAttr.InodeId,
            },
        );
    }

    let deviceId = PTS_DEVICE.lock().id.DeviceID();
    let inodeId = PTS_DEVICE.lock().NextIno();

    let stableAttr = StableAttr {
        Type: InodeType::Directory,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: 4096,
        DeviceFileMajor: 0,
        DeviceFileMinor: 0,
    };

    let inodeInternal = InodeIntern {
        UniqueId: NewUID(),
        InodeOp: Arc::new(d),
        StableAttr: stableAttr,
        LockCtx: LockCtx::default(),
        MountSource: msrc.clone(),
        Overlay: None,
    };

    return Inode(Arc::new(QMutex::new(inodeInternal)));
}

pub struct DirInodeOperationsInternal {
    pub fsType: i64,
    pub unstable: UnstableAttr,
    pub msrc: Arc<QMutex<MountSource>>,
    pub master: Inode,
    pub slaves: BTreeMap<u32, Inode>,
    pub dentryMap: DentMap,
    pub next: u32,
}

#[derive(Clone)]
pub struct DirInodeOperations(Arc<QMutex<DirInodeOperationsInternal>>);

impl Deref for DirInodeOperations {
    type Target = Arc<QMutex<DirInodeOperationsInternal>>;

    fn deref(&self) -> &Arc<QMutex<DirInodeOperationsInternal>> {
        &self.0
    }
}

impl DirInodeOperations {
    pub fn allocateTerminal(&self, task: &Task) -> Result<Terminal> {
        let mut internal = self.lock();

        let n = internal.next;
        if n == (1 << 32 - 1) as u32 {
            //MaxUint32
            return Err(Error::SysError(SysErr::ENOMEM));
        }

        if internal.slaves.contains_key(&n) {
            panic!("pty index collision; index {} already exists", n);
        }

        let t = Terminal::New(self, n);
        internal.next += 1;

        let creds = task.creds.clone();
        let _uid = creds.lock().EffectiveKUID;
        let _gid = creds.lock().EffectiveKGID;

        return Ok(t);
    }
}

impl InodeOperations for DirInodeOperations {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::DirInodeOperations;
    }

    fn InodeType(&self) -> InodeType {
        return InodeType::Directory;
    }

    fn InodeFileType(&self) -> InodeFileType {
        return InodeFileType::TTYDir;
    }

    fn WouldBlock(&self) -> bool {
        return false;
    }

    fn Lookup(&self, _task: &Task, _dir: &Inode, name: &str) -> Result<Dirent> {
        let internal = self.lock();

        if name == "ptmx" {
            return Ok(Dirent::New(&internal.master, name));
        }

        let n: u32 = match name.parse::<u32>() {
            Err(_) => return Err(Error::SysError(SysErr::ENOENT)),
            Ok(n) => n,
        };

        let s = match &internal.slaves.get(&n) {
            Some(s) => s.clone(),
            _ => return Err(Error::SysError(SysErr::ENOENT)),
        };

        return Ok(Dirent::New(s, name));
    }

    fn Create(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _name: &str,
        _flags: &FileFlags,
        _perm: &FilePermissions,
    ) -> Result<File> {
        return Err(Error::SysError(SysErr::EACCES));
    }

    fn CreateDirectory(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _name: &str,
        _perm: &FilePermissions,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::EACCES));
    }

    fn CreateLink(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _oldname: &str,
        _newname: &str,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::EACCES));
    }

    fn CreateHardLink(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _target: &Inode,
        _name: &str,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::EACCES));
    }

    fn CreateFifo(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _name: &str,
        _perm: &FilePermissions,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::EACCES));
    }

    fn Remove(&self, _task: &Task, _dir: &mut Inode, _name: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::EPERM));
    }

    fn RemoveDirectory(&self, _task: &Task, _dir: &mut Inode, _name: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::EPERM));
    }

    fn Rename(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _oldParent: &Inode,
        _oldname: &str,
        _newParent: &Inode,
        _newname: &str,
        _replacement: bool,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn Bind(
        &self,
        _task: &Task,
        _dir: &Inode,
        _name: &str,
        _data: &BoundEndpoint,
        _perms: &FilePermissions,
    ) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn BoundEndpoint(&self, _task: &Task, _inode: &Inode, _path: &str) -> Option<BoundEndpoint> {
        return None;
    }

    fn GetFile(
        &self,
        _task: &Task,
        _dir: &Inode,
        _dirent: &Dirent,
        _flags: FileFlags,
    ) -> Result<File> {
        return Err(Error::SysError(SysErr::EPERM));
    }

    fn UnstableAttr(&self, _task: &Task) -> Result<UnstableAttr> {
        let u = self.0.lock().unstable;
        return Ok(u);
    }

    fn Getxattr(&self, _dir: &Inode, _name: &str, _size: usize) -> Result<Vec<u8>> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    fn Setxattr(&self, _dir: &mut Inode, _name: &str, _value: &[u8], _flags: u32) -> Result<()> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    fn Listxattr(&self, _dir: &Inode, _size: usize) -> Result<Vec<String>> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return ContextCanAccessFile(task, inode, reqPerms);
    }

    fn SetPermissions(&self, task: &Task, _dir: &mut Inode, p: FilePermissions) -> bool {
        self.lock().unstable.SetPermissions(task, &p);
        return true;
    }

    fn SetOwner(&self, task: &Task, _dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        self.lock().unstable.SetOwner(task, owner);
        return Ok(());
    }

    fn SetTimestamps(&self, task: &Task, _dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        self.lock().unstable.SetTimestamps(task, ts);
        return Ok(());
    }

    fn Truncate(&self, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EISDIR));
    }

    fn Allocate(&self, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
        return Err(Error::SysError(SysErr::EISDIR));
    }

    fn ReadLink(&self, _task: &Task, _dir: &Inode) -> Result<String> {
        return Err(Error::SysError(SysErr::ENOLINK));
    }

    fn GetLink(&self, _task: &Task, _dir: &Inode) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOLINK));
    }

    fn AddLink(&self, _task: &Task) {
        self.0.lock().unstable.Links += 1;
    }

    fn DropLink(&self, _task: &Task) {
        self.0.lock().unstable.Links -= 1;
    }

    fn IsVirtual(&self) -> bool {
        return true;
    }

    fn Sync(&self) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOSYS));
    }

    fn StatFS(&self, _task: &Task) -> Result<FsInfo> {
        if self.lock().fsType == 0 {
            return Err(Error::SysError(SysErr::ENOSYS));
        }

        return Ok(FsInfo {
            Type: self.lock().fsType as u64,
            ..Default::default()
        });
    }

    fn Mappable(&self) -> Result<MMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

pub struct DirFileOperations {
    pub di: DirInodeOperations,
    pub DirCursor: QMutex<String>,
}

impl Waitable for DirFileOperations {
    fn Readiness(&self, _task: &Task, _mask: EventMask) -> EventMask {
        panic!("DirFileOperations doesn't support Waitable::Readiness");
    }

    fn EventRegister(&self, _task: &Task, _e: &WaitEntry, _mask: EventMask) {
        panic!("DirFileOperations doesn't support Waitable::EventRegister");
    }

    fn EventUnregister(&self, _task: &Task, _e: &WaitEntry) {
        panic!("DirFileOperations doesn't support Waitable::EventUnregister");
    }
}

impl SpliceOperations for DirFileOperations {}

impl FileOperations for DirFileOperations {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::DirFileOperations;
    }

    fn Seekable(&self) -> bool {
        return true;
    }

    fn Seek(&self, task: &Task, f: &File, whence: i32, current: i64, offset: i64) -> Result<i64> {
        return SeekWithDirCursor(task, f, whence, current, offset, None);
    }

    fn ReadDir(
        &self,
        task: &Task,
        file: &File,
        offset: i64,
        serializer: &mut DentrySerializer,
    ) -> Result<i64> {
        let root = task.Root();
        let mut dirCursor = self.DirCursor.lock();

        let mut dirCtx = DirCtx {
            Serializer: serializer,
            DirCursor: (*dirCursor).to_string(),
        };

        let res = DirentReadDir(task, &file.Dirent, self, &root, &mut dirCtx, offset)?;
        *dirCursor = dirCtx.DirCursor;
        return Ok(res);
    }

    fn ReadAt(
        &self,
        _task: &Task,
        _f: &File,
        _dsts: &mut [IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::EISDIR));
    }

    fn WriteAt(
        &self,
        _task: &Task,
        _f: &File,
        _srcs: &[IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::EISDIR));
    }

    fn Append(&self, task: &Task, f: &File, srcs: &[IoVec]) -> Result<(i64, i64)> {
        let n = self.WriteAt(task, f, srcs, 0, false)?;
        return Ok((n, 0));
    }

    fn Fsync(
        &self,
        _task: &Task,
        _f: &File,
        _start: i64,
        _end: i64,
        _syncType: SyncType,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn Flush(&self, _task: &Task, _f: &File) -> Result<()> {
        return Ok(());
    }

    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        let inode = f.Dirent.Inode();
        return inode.UnstableAttr(task);
    }

    fn Ioctl(&self, _task: &Task, _f: &File, _fd: i32, _request: u64, _val: u64) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTTY));
    }

    fn IterateDir(
        &self,
        task: &Task,
        _d: &Dirent,
        dirCtx: &mut DirCtx,
        offset: i32,
    ) -> (i32, Result<i64>) {
        let ops = self
            .di
            .as_any()
            .downcast_ref::<DirInodeOperations>()
            .expect("DirInodeOperations convert fail")
            .lock();

        return match dirCtx.ReadDir(task, &ops.dentryMap) {
            Err(e) => (offset, Err(e)),
            Ok(count) => (offset + count as i32, Ok(0)),
        };
    }

    fn Mappable(&self) -> Result<MMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

impl SockOperations for DirFileOperations {}
