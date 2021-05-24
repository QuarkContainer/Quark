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
use alloc::sync::Arc;
use spin::Mutex;
use alloc::collections::btree_map::BTreeMap;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::any::Any;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use core::ops::Deref;

//use socket::unix::transport::unix::BoundEndpoint;
use super::super::qlib::linux_def::*;
use super::super::qlib::common::*;
use super::super::qlib::path::*;
use super::super::task::*;
use super::super::qlib::auth::*;
use super::super::qlib::auth::userns::*;
use super::filesystems::*;
use super::inode::*;
use super::host::*;
use super::dirent::*;
use super::tty::fs::*;
use super::mount_overlay::*;
use super::super::qlib::lrc_cache::*;

pub struct Mount {
    pub Id: u64,
    pub Pid: u64,
    pub root: Dirent,
    pub prev: Option<Arc<Mutex<Mount>>>,
}

impl Mount {
    pub const INVALID_MOUNT_ID: u64 = core::u64::MAX;

    pub fn New(id: u64, pid: u64, root: &Dirent) -> Self {
        return Self {
            Id: id,
            Pid: pid,
            root: root.clone(),
            prev: None,
        }
    }

    pub fn NewRootMount(id: u64, root: &Dirent) -> Self {
        return Self {
            Id: id,
            Pid: Self::INVALID_MOUNT_ID,
            root: root.clone(),
            prev: None,
        }
    }

    pub fn NewUndoMount(root: &Dirent) -> Self {
        return Self {
            Id: Self::INVALID_MOUNT_ID,
            Pid: Self::INVALID_MOUNT_ID,
            root: root.clone(),
            prev: None,
        }
    }

    pub fn Root(&self) -> Dirent {
        return self.root.clone();
    }

    //whether it has no parent
    pub fn IsRoot(&self) -> bool {
        return !self.IsUndo() && self.Pid == Self::INVALID_MOUNT_ID;
    }

    pub fn IsUndo(&self) -> bool {
        if self.Id == Self::INVALID_MOUNT_ID {
            assert!(self.Pid != Self::INVALID_MOUNT_ID, "Undo mount with valid parentID");
            return true;
        }

        return false
    }
}

pub struct MountNsInternal {
    pub userns: UserNameSpace,
    pub root: Dirent,
    pub mounts: Mutex<BTreeMap<u64, Arc<Mutex<Mount>>>>,
    pub mountId: AtomicU64,
}

impl Default for MountNsInternal {
    fn default() -> Self {
        return Self {
            userns: UserNameSpace::default(),
            root: Dirent::default(),
            mounts: Mutex::new(BTreeMap::new()),
            mountId: AtomicU64::new(0),
        }
    }
}

#[derive(Default, Clone)]
pub struct MountNs(Arc<MountNsInternal>);

impl Deref for MountNs {
    type Target = Arc<MountNsInternal>;

    fn deref(&self) -> &Arc<MountNsInternal> {
        &self.0
    }
}

impl MountNs {
    pub fn New(task: &Task, root: &Inode) -> Self {
        let d = Dirent::New(&root, &"/".to_string());  //(Arc::new(Mutex::new(InterDirent::New(root.clone(), &"/".to_string()))));
        let mut mounts = BTreeMap::new();
        let rootMount = Arc::new(Mutex::new(Mount::NewRootMount(1, &d)));
        mounts.insert(d.ID(), rootMount);
        let internal = MountNsInternal {
            userns: task.creds.lock().UserNamespace.clone(),
            root: d,
            mounts: Mutex::new(mounts),
            mountId: AtomicU64::new(2),
        };

        return Self(Arc::new(internal));
    }

    pub fn UserNamespace(&self) -> UserNameSpace {
        return self.userns.clone();
    }

    pub fn Root(&self) -> Dirent {
        return self.root.clone();
    }

    pub fn Freeze(&self) {
        self.root.Freeze()
    }

    pub fn Mount(&self, mountPoint: &Dirent, inode: &Inode) -> Result<()> {
        let replacement = mountPoint.Mount(inode)?;

        let parentMnt = self.FindMount(mountPoint).unwrap();
        let mut childMnt = Mount::New(self.mountId.fetch_add(1, Ordering::SeqCst), parentMnt.lock().Id, &replacement);

        mountPoint.clone().DropExtendedReference();

        let mntId = mountPoint.ID();
        let mut mounts = self.mounts.lock();
        let prev = mounts.get(&mntId);

        let havePre = match prev {
            Some(_) => {
                true
            }
            _ => false
        };

        if havePre {
            childMnt.prev = Some(prev.unwrap().clone());
            mounts.remove(&mntId);
            mounts.insert(replacement.ID(), Arc::new(Mutex::new(childMnt)));
            return Ok(())
        }

        childMnt.prev = Some(Arc::new(Mutex::new(Mount::NewUndoMount(mountPoint))));
        mounts.insert(replacement.ID(), Arc::new(Mutex::new(childMnt)));
        return Ok(())
    }

    pub fn Unmount(&self, node: &Dirent, detachOnly: bool) -> Result<()> {
        let mut mounts = self.mounts.lock();
        let orig = mounts.get(&node.ID());
        let orig = match orig {
            None => return Err(Error::SysError(SysErr::EINVAL)),
            Some(n) => n,
        };

        let prev = match &orig.lock().prev {
            None => panic!("cannot unmount initial dirent"),
            Some(prev) => prev.clone(),
        };

        let m = node.Inode().lock().MountSource.clone();
        if !detachOnly && Arc::strong_count(&m) != 2 {
            return Err(Error::SysError(SysErr::EBUSY))
        }

        node.UnMount(&prev.lock().root)?;

        let isNone = match prev.lock().prev {
            None => true,
            _ => false,
        };

        if isNone {
            if !prev.lock().IsUndo() {
                panic!("Last mount in the chain must be a undo mount")
            }
        } else {
            let prevRoot = prev.lock().root.ID();
            mounts.insert(prevRoot, prev);
        }

        mounts.remove(&node.ID());

        return Ok(())
    }

    pub fn FindMount(&self, d: &Dirent) -> Option<Arc<Mutex<Mount>>> {
        let mut d = d.clone();
        let mounts = self.mounts.lock();
        loop {
            let id = d.ID();

            match mounts.get(&id) {
                None => (),
                Some(mount) => return Some(mount.clone())
            }

            let tmp;
            match &(d.0).0.lock().Parent {
                None => return None,
                Some(ref p) => tmp = p.clone(),
            }

            d = tmp;
        }
    }

    pub fn AllMountsUnder(&self, parent: &Arc<Mutex<Mount>>) -> Vec<Arc<Mutex<Mount>>> {
        let mut ret: Vec<Arc<Mutex<Mount>>> = Vec::new();

        for (_, mp) in self.mounts.lock().iter() {
            if mp.lock().IsUndo() {
                continue;
            }

            let root = mp.lock().root.clone();
            let parentRoot = parent.lock().root.clone();
            if root.DescendantOf(&parentRoot) {
                ret.push(mp.clone())
            }
        }

        return ret;
    }

    pub fn FindLink(&self, task: &Task, root: &Dirent, wd: Option<Dirent>, path: &str, remainingTraversals: &mut u32) -> Result<Dirent> {
        if path.len() == 0 {
            return Err(Error::SysError(SysErr::ENOENT))
        }

        let (mut first, mut remain) = SplitFirst(path);

        let mut current = match wd {
            None => root.clone(),
            Some(wd) => wd.clone(),
        };

        while first == "/" {
            if remain == "" {
                return Ok(root.clone())
            }

            current = root.clone();
            let (tfirst, tremain) = SplitFirst(remain);
            first = tfirst;
            remain = tremain;
        }

        loop {
            let currentIode = current.Inode();
            if !Arc::ptr_eq(&current, root) {
                if !currentIode.StableAttr().IsDir() {
                    return Err(Error::SysError(SysErr::ENOTDIR))
                }

                currentIode.CheckPermission(task, &PermMask {
                    execute: true,
                    ..Default::default()
                })?
            }

            let next = match current.Walk(task, root, first) {
                Err(e) => {
                    current.ExtendReference();
                    return Err(e);
                }
                Ok(n) => n,
            };

            if remain != "" {
                current = self.resolve(task, root, &next, remainingTraversals)?;
            } else {
                next.ExtendReference();
                return Ok(next)
            }

            let (tfirst, tremain) = SplitFirst(remain);
            first = tfirst;
            remain = tremain;
        }
    }

    pub fn FindInode(&self, task: &Task, root: &Dirent, wd: Option<Dirent>, path: &str, remainingTraversals: &mut u32) -> Result<Dirent> {
        let d = self.FindLink(task, root, wd, path, remainingTraversals)?;
        let ret = self.resolve(task, root, &d, remainingTraversals);
        return ret;
    }

    pub fn resolve(&self, task: &Task, root: &Dirent, node: &Dirent, remainingTraversals: &mut u32) -> Result<Dirent> {
        let inode = node.Inode();
        let target = inode.GetLink(task);

        match target {
            Ok(target) => {
                if *remainingTraversals == 0 {
                    return Err(Error::SysError(SysErr::ELOOP))
                }

                return Ok(target)
            }
            Err(Error::SysError(SysErr::ENOLINK)) => {
                return Ok(node.clone())
            }
            Err(Error::ErrResolveViaReadlink) => {
                if *remainingTraversals == 0 {
                    return Err(Error::SysError(SysErr::ELOOP))
                }

                let targetPath = inode.ReadLink(task)?;
                *remainingTraversals -= 1;

                let wd = match &(node.0).0.lock().Parent {
                    None => None,
                    Some(ref wd) => Some(wd.clone()),
                };

                let d = self.FindInode(task, root, wd, &targetPath, remainingTraversals)?;

                return Ok(d)
            }
            Err(err) => Err(err)
        }
    }

    pub fn ResolveExecutablePath(&self, task: &Task, wd: &str, name: &str, paths: &Vec<String>) -> Result<String> {
        if IsAbs(name) {
            return Ok(name.to_string());
        }

        let mut wd = wd.clone();
        if let Some(idx) = name.find('/') {
            if idx > 0 {
                if wd.len() == 0 {
                    wd = "/";
                }

                if !IsAbs(&wd) {
                    return Ok(Join(&wd, name))
                }
            }
        }

        let root = task.Root();

        for p in paths {
            let binPath = Join(p, name);
            let mut traversals = MAX_SYMLINK_TRAVERSALS;

            let d = self.FindInode(task, &root, None, &binPath, &mut traversals);

            let d = match d {
                Err(Error::SysError(SysErr::ENOENT)) | Err(Error::SysError(SysErr::EACCES)) => continue,
                Err(error) => return Err(error),
                Ok(d) => d,
            };

            let inode = d.Inode();
            if !inode.StableAttr().IsRegular() {
                continue;
            }

            let err = inode.CheckPermission(&task, &PermMask {
                read: true,
                execute: true,
                write: false,
            });

            match err {
                Err(_) => continue,
                Ok(_) => (),
            }

            return Ok(Join(&Join(&"/".to_string(), p), name))
        }

        return Err(Error::SysError(SysErr::ENOENT))
    }
}

const PREFIX : &str = "PATH=";
pub fn GetPath(env : &[String]) -> Vec<String> {
    for e in env {
        if HasPrefix(e, PREFIX) {
            let v = TrimPrefix(e, PREFIX);
            let ret = v.split(':').map(|s|s.to_string()).collect();
            return ret;
        }
    }

    return Vec::new();
}

#[derive(Clone)]
pub enum MountOptions {
    Host(Arc<Mutex<SuperOperations>>),
    Default,
}

impl Default for MountOptions {
    fn default() -> Self {
        return Self::Default
    }
}

impl MountOptions {
    pub fn HostOptions(&self) -> Result<Arc<Mutex<SuperOperations>>> {
        match self {
            MountOptions::Host(o) => Ok(o.clone()),
            _ => Err(Error::InvalidInput)
        }
    }
}

const DEFAULT_DIRENT_CACHE_SIZE: u64 = 1024;

//#[derive(Clone)]
pub struct MountSource {
    pub FileSystemType: String,
    pub Flags: MountSourceFlags,
    pub MountSourceOperations: Arc<Mutex<MountSourceOperations>>,
    pub fscache: LruCache<Dirent>,
    frozen: Vec<Dirent>,
}

impl Default for MountSource {
    fn default() -> Self {
        return Self {
            FileSystemType: "".to_string(),
            Flags: MountSourceFlags::default(),
            MountSourceOperations: Arc::new(Mutex::new(SimpleMountSourceOperations::default())),
            fscache: LruCache::New(DEFAULT_DIRENT_CACHE_SIZE),
            frozen: Vec::new(),
        }
    }
}

impl DirentOperations for MountSource {
    fn Revalidate(&self, name: &str, parent: &Inode, child: &Inode) -> bool {
        return self.MountSourceOperations.lock().Revalidate(name, parent, child)
    }

    fn Keep(&self, dirent: &Dirent) -> bool {
        return self.MountSourceOperations.lock().Keep(dirent)
    }

    fn CacheReadDir(&self) -> bool {
        return self.MountSourceOperations.lock().CacheReadDir();
    }
}

impl MountSource {
    pub fn New(mops: &Arc<Mutex<MountSourceOperations>>, filesystem: &Filesystem, flags: &MountSourceFlags) -> Self {
        /*let mut fsType = "none".to_string();
        if let fs = Some(filesystem) {
            fsType = filesystem.Name()
        }*/

        let fsType = filesystem.Name();
        return Self {
            Flags: flags.clone(),
            FileSystemType: fsType.to_string(),
            MountSourceOperations: mops.clone(),
            fscache: LruCache::New(DEFAULT_DIRENT_CACHE_SIZE),
            frozen: Vec::new(),
        }
    }

    pub fn NewPtsMountSource(mops: &Arc<Mutex<PtsSuperOperations>>, filesystem: &Filesystem, flags: &MountSourceFlags) -> Self {
        let fsType = filesystem.Name();
        return Self {
            Flags: flags.clone(),
            FileSystemType: fsType.to_string(),
            MountSourceOperations: mops.clone(),
            fscache: LruCache::New(DEFAULT_DIRENT_CACHE_SIZE),
            frozen: Vec::new(),
        }
    }

    pub fn NewOverlayMountSource(mops: &Arc<Mutex<OverlayMountSourceOperations>>, filesystem: &Filesystem, flags: &MountSourceFlags) -> Self {
        let fsType = filesystem.Name();
        return Self {
            Flags: flags.clone(),
            FileSystemType: fsType.to_string(),
            MountSourceOperations: mops.clone(),
            fscache: LruCache::New(DEFAULT_DIRENT_CACHE_SIZE),
            frozen: Vec::new(),
        }
    }

    pub fn NewHostMountSource(root: &str, mounter: &FileOwner, filesystem: &Filesystem, flags: &MountSourceFlags, dontTranslateOwnership: bool) -> Self {
        let mops = Arc::new(Mutex::new(SuperOperations {
            mountSourceOperations: Default::default(),
            root: root.to_string(),
            inodeMapping: BTreeMap::new(),
            mounter: mounter.clone(),
            dontTranslateOwnership: dontTranslateOwnership,
        }));

        let fsType = filesystem.Name();

        return Self {
            Flags: flags.clone(),
            FileSystemType: fsType.to_string(),
            MountSourceOperations: mops,
            fscache: LruCache::New(DEFAULT_DIRENT_CACHE_SIZE),
            frozen: Vec::new(),
        };
    }

    pub fn NewCachingMountSource(filesystem: &Filesystem, flags: &MountSourceFlags) -> Self {
        let mops = Arc::new(Mutex::new(SimpleMountSourceOperations {
            keep: false,
            revalidate: false,
            cacheReaddir: false,
        }));

        let fsType = filesystem.Name();

        return Self {
            Flags: flags.clone(),
            FileSystemType: fsType.to_string(),
            MountSourceOperations: mops,
            fscache: LruCache::New(DEFAULT_DIRENT_CACHE_SIZE),
            frozen: Vec::new(),
        };
    }

    pub fn NewNonCachingMountSource(filesystem: &Filesystem, flags: &MountSourceFlags) -> Self {
        let mops = Arc::new(Mutex::new(SimpleMountSourceOperations {
            keep: false,
            revalidate: false,
            cacheReaddir: true,
        }));

        let fsType = filesystem.Name();

        return Self {
            Flags: flags.clone(),
            FileSystemType: fsType.to_string(),
            MountSourceOperations: mops,
            fscache: LruCache::New(DEFAULT_DIRENT_CACHE_SIZE),
            frozen: Vec::new(),
        };
    }

    pub fn NewRevalidatingMountSource(filesystem: &Filesystem, flags: &MountSourceFlags) -> Self {
        let mops = Arc::new(Mutex::new(SimpleMountSourceOperations {
            keep: true,
            revalidate: true,
            cacheReaddir: false,
        }));

        let fsType = filesystem.Name();

        return Self {
            Flags: flags.clone(),
            FileSystemType: fsType.to_string(),
            MountSourceOperations: mops,
            fscache: LruCache::New(DEFAULT_DIRENT_CACHE_SIZE),
            frozen: Vec::new(),
        };
    }

    pub fn NewPseudoMountSource() -> Self {
        let mops = Arc::new(Mutex::new(SimpleMountSourceOperations {
            keep: false,
            revalidate: false,
            cacheReaddir: false,
        }));

        return Self {
            Flags: MountSourceFlags::default(),
            FileSystemType: "".to_string(),
            MountSourceOperations: mops,
            fscache: LruCache::New(DEFAULT_DIRENT_CACHE_SIZE),
            frozen: Vec::new(),
        };
    }

    #[cfg(test)]
    pub fn ContainsKey(&self, key: u64) -> bool {
        return self.fscache.ContainsKey(key)
    }

    pub fn FlashDirentRefs(&mut self) {
        self.fscache.Clear();
    }

    pub fn ExtendReference(&mut self, dirent: &Dirent) {
        let id = dirent.ID();
        self.fscache.Add(id, dirent.clone())
    }

    pub fn DropExtendReference(&mut self, dirent: &Dirent) {
        let id = dirent.ID();
        self.fscache.Remove(id);
    }

    pub fn Froze(&mut self, dirent: &Dirent) {
        self.frozen.push(dirent.clone());
    }
}

pub trait DirentOperations {
    fn Revalidate(&self, name: &str, parent: &Inode, child: &Inode) -> bool;
    fn Keep(&self, dirent: &Dirent) -> bool;
    fn CacheReadDir(&self) -> bool;
}

pub trait MountSourceOperations: DirentOperations + Send {
    fn as_any(&self) -> &Any;

    fn Destroy(&mut self);
    fn ResetInodeMappings(&mut self);
    fn SaveInodeMapping(&mut self, inode: &Inode, path: &str);
}

#[derive(Clone, Default, Debug)]
pub struct SimpleMountSourceOperations {
    pub keep: bool,
    pub revalidate: bool,
    pub cacheReaddir: bool,
}

impl DirentOperations for SimpleMountSourceOperations {
    fn Revalidate(&self, _name: &str, _parent: &Inode, _child: &Inode) -> bool {
        return self.revalidate
    }

    fn Keep(&self, _dirent: &Dirent) -> bool {
        return self.keep
    }

    fn CacheReadDir(&self) -> bool {
        return self.cacheReaddir
    }
}

impl MountSourceOperations for SimpleMountSourceOperations {
    fn as_any(&self) -> &Any {
        return self
    }

    fn Destroy(&mut self) {}

    fn ResetInodeMappings(&mut self) {}

    fn SaveInodeMapping(&mut self, _inode: &Inode, _path: &str) {}
}

// Info defines attributes of a filesystem.
#[derive(Clone, Default, Debug, Copy)]
pub struct FsInfo {
    // Type is the filesystem type magic value.
    pub Type: u64,

    // TotalBlocks is the total data blocks in the filesystem.
    pub TotalBlocks: u64,

    // FreeBlocks is the number of free blocks available.
    pub FreeBlocks: u64,

    // TotalFiles is the total file nodes in the filesystem.
    pub TotalFiles: u64,

    // FreeFiles is the number of free file nodes.
    pub FreeFiles: u64,
}

#[cfg(test1)]
mod tests {
    use alloc::sync::Arc;
    use spin::Mutex;
    use core::any::Any;
    use alloc::rc::Rc;
    use core::cell::RefCell;
    use core::ops::Deref;

    use super::*;
    use super::super::flags::*;
    use super::super::dentry::*;
    use super::super::file::*;
    use super::super::super::auth::*;
    use super::super::super::mem::seq::*;
    //use super::super::super::Common::*;
    //use super::super::super::libcDef::*;

    fn NewMockInode(msrc: &Arc<Mutex<MountSource>>, sattr: &StableAttr) -> Inode {
        let iops = Arc::new(NewMockInodeOperations());
        let inodeInternal = InodeIntern {
            InodeOp: iops,
            StableAttr: *sattr,
            MountSource: msrc.clone(),
            ..Default::default()
        };

        return Inode(Arc::new(Mutex::new(inodeInternal)))
    }

    fn NewMockInodeOperations() -> MockInodeOperations {
        let internal = MockInodeOperationsIntern {
            UAttr: UnstableAttr {
                Perms: FilePermissions::FromMode(FileMode(0o777)),
                ..Default::default()
            },
            ..Default::default()
        };

        return MockInodeOperations(Arc::new(Mutex::new(internal)))
    }

    fn NewMockMountSource(cacheSize: u64) -> MountSource {
        let msops = Arc::new(Mutex::new(MockMountSourceOps {
            keep: true,
            revalidate: false,
        }));

        return MountSource {
            MountSourceOperations: msops,
            fscache: LruCache::New(cacheSize),
            ..Default::default()
        }
    }

    struct MockMountSourceOps {
        keep: bool,
        revalidate: bool,
    }

    impl DirentOperations for MockMountSourceOps {
        fn Revalidate(&self, _name: &str, _parent: &Inode, _child: &Inode) -> bool {
            return self.revalidate
        }

        fn Keep(&self, _dirent: &Dirent) -> bool {
            return self.keep;
        }

        fn CacheReadDir(&self) -> bool {
            return self.keep
        }
    }

    impl MountSourceOperations for MockMountSourceOps {
        fn as_any(&self) -> &Any {
            return self
        }

        fn Destroy(&mut self) {}

        fn ResetInodeMappings(&mut self) {}

        fn SaveInodeMapping(&mut self, _inode: &Inode, _path: &str) {}
    }

    #[derive(Debug, Clone, Default)]
    struct MockFileOperations {}

    impl SpliceOperations for MockFileOperations {}

    impl FileOperations for MockFileOperations {
        fn as_any(&self) -> &Any {
            return self
        }

        fn FopsType(&self) -> FileOpsType {
            return FileOpsType::MockFileOperations
        }

        fn Seekable(&self) -> bool {
            return true;
        }

        fn Seek(&mut self, _task: &Task, _f: &mut File, _whence: i32, _offset: i64) -> Result<i64> {
            return Ok(0)
        }

        fn ReadDir(&self, _task: &Task, _f: &mut File, _serializer: &mut DentrySerializer) -> Result<i64> {
            return Ok(0)
        }

        fn ReadAt(&self, _task: &Task, _f: &mut File, _dsts: BlockSeq, _offset: i64) -> Result<i64> {
            return Ok(0)
        }

        fn WriteAt(&self, _task: &Task, _f: &mut File, _srcs: BlockSeq, _offset: i64) -> Result<i64> {
            return Ok(0)
        }

        fn Append(&self, _task: &Task, _f: &File, _srcs: &[IoVec]) -> Result<(i64, i64)> {
            return Err(Error::SysError(SysErr::ESPIPE))
        }

        fn Fsync(&self, _task: &Task, _f: &mut File, _start: i64, _end: i64, _syncType: SyncType) -> Result<()> {
            return Ok(())
        }

        fn Flush(&self, _task: &Task, _f: &mut File) -> Result<()> {
            return Ok(())
        }

        fn UnstableAttr(&self, _task: &Task, _f: &File) -> Result<UnstableAttr> {
            return Ok(UnstableAttr::default())
        }

        fn Ioctl(&mut self, _task: &Task, _f: &mut File, _fd: i32, _request: u64, _val: u64) -> Result<()> {
            return Ok(())
        }

        fn IterateDir(&self, _task: &Task,_d: &Dirent, _dirCtx: &mut DirCtx, _offset: i32) -> (i32, Result<i64>) {
            return (0, Ok(0))
        }
    }

    #[derive(Debug, Clone, Default)]
    struct MockInodeOperationsIntern {
        UAttr: UnstableAttr,
        createCalled: bool,
        createDirectoryCalled: bool,
        createLinkCalled: bool,
        renameCalled: bool,
        walkCalled: bool,
    }

    struct MockInodeOperations(pub Arc<Mutex<MockInodeOperationsIntern>>);

    impl Default for MockInodeOperations {
        fn default() -> Self {
            return Self(Arc::new(Mutex::new(Default::default())))
        }
    }

    impl Deref for MockInodeOperations {
        type Target = Arc<Mutex<MockInodeOperationsIntern>>;

        fn deref(&self) -> &Arc<Mutex<MockInodeOperationsIntern>> {
            &self.0
        }
    }

    impl InodeOperations for MockInodeOperations {
        fn as_any(&self) -> &Any {
            return self
        }

        fn IopsType(&self) -> IopsType {
            return IopsType::MockInodeOperations;
        }

        fn InodeFileType(&self) -> InodeFileType{
            return InodeFileType::Mock;
        }

        fn Lookup(&self, _task: &Task, dir: &Inode, name: &str) -> Result<Dirent> {
            self.lock().walkCalled = true;
            let inodeInternal = InodeIntern {
                InodeOp: Arc::new(Self::default()),
                StableAttr: StableAttr::default(),
                MountSource: dir.lock().MountSource.clone(),
                ..Default::default()
            };

            let inode = Inode(Arc::new(Mutex::new(inodeInternal)));
            let dirent = Dirent::New(&inode, name);
            return Ok(dirent)
        }

        fn Create(&self, _task: &Task, dir: &mut Inode, name: &str, _flags: &FileFlags, _perm: &FilePermissions) -> Result<Arc<Mutex<File>>> {
            self.lock().createCalled = true;
            let inodeInternal = InodeIntern {
                InodeOp: Arc::new(Self::default()),
                StableAttr: StableAttr::default(),
                MountSource: dir.lock().MountSource.clone(),
                ..Default::default()
            };

            let inode = Inode(Arc::new(Mutex::new(inodeInternal)));
            let dirent = Dirent::New(&inode, name);

            return Ok(Arc::new(Mutex::new(File {
                UniqueId: 0,
                Dirent: dirent,
                flags: FileFlags::default(),
                offset: 0,
                FileOp: Rc::new(RefCell::new(MockFileOperations::default())),
            })))
        }

        fn CreateDirectory(&self, _task: &Task, _dir: &mut Inode, _name: &str, _perm: &FilePermissions) -> Result<()> {
            self.lock().createDirectoryCalled = true;
            return Ok(())
        }

        fn CreateLink(&self, _task: &Task, _dir: &mut Inode, _oldname: &str, _newname: &str) -> Result<()> {
            self.lock().createLinkCalled = true;
            return Ok(())
        }

        fn CreateHardLink(&self, _task: &Task, _dir: &mut Inode, _target: &Inode, _name: &str) -> Result<()> {
            return Err(Error::None)
        }

        fn CreateFifo(&self, _task: &Task, _dir: &mut Inode, _name: &str, _perm: &FilePermissions) -> Result<()> {
            return Err(Error::None)
        }

        //fn RemoveDirent(&mut self, dir: &mut InodeStruStru, remove: &Arc<Mutex<Dirent>>) -> Result<()> ;
        fn Remove(&self, _task: &Task, _dir: &mut Inode, _name: &str) -> Result<()> {
            return Ok(())
        }

        fn RemoveDirectory(&self, _task: &Task, _dir: &mut Inode, _name: &str) -> Result<()> {
            return Ok(())
        }

        fn Rename(&self, _task: &Task, _dir: &mut Inode, _oldParent: &Inode, _oldname: &str, _newParent: &Inode, _newname: &str, _replacement: bool) -> Result<()> {
            self.lock().renameCalled = true;
            return Ok(())
        }

        fn Bind(&self, _task: &Task, _dir: &Inode, _name: &str, _data: &BoundEndpoint, _perms: &FilePermissions) -> Result<Dirent>;
        fn BoundEndpoint(&self, _task: &Task, inode: &Inode, path: &str) -> Option<BoundEndpoint>;


        fn GetFile(&self, _dir: &Inode, _dirent: &Dirent, _flags: FileFlags) -> Result<Arc<Mutex<File>>> {
            return Err(Error::None)
        }

        fn UnstableAttr(&self, _dir: &Inode) -> Result<UnstableAttr> {
            return Ok(self.lock().UAttr)
        }

        fn Getxattr(&self, _dir: &Inode, _name: &str) -> Result<String> {
            return Err(Error::None)
        }

        fn Setxattr(&self, _dir: &mut Inode, _name: &str, _value: &str) -> Result<()> {
            return Err(Error::None)
        }

        fn Listxattr(&self, _dir: &Inode) -> Result<Vec<String>> {
            return Err(Error::None)
        }

        fn Check(&self, task: &Task, dir: &Inode, reqPerms: PermMask) -> Result<bool> {
            return ContextCanAccessFile(task, dir, reqPerms)
        }

        fn SetPermissions(&self, _task: &Task, _dir: &mut Inode, _f: FilePermissions) -> bool {
            return false;
        }

        fn SetOwner(&self, _task: &Task, _dir: &mut Inode, _owner: &FileOwner) -> Result<()> {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        fn SetTimestamps(&self, _task: &Task, _dir: &mut Inode, _ts: &InterTimeSpec) -> Result<()> {
            return Ok(())
        }

        fn Truncate(&self, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
            return Ok(())
        }

        fn Allocate(&self, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
            return Ok(())
        }

        fn ReadLink(&self, _task: &Task,_dir: &Inode) -> Result<String> {
            return Err(Error::None)
        }

        fn GetLink(&self, _task: &Task, _dir: &Inode) -> Result<Dirent> {
            return Err(Error::SysError(SysErr::ENOLINK))
        }

        fn AddLink(&self, _task: &Task) {}

        fn DropLink(&self, _task: &Task) {}

        fn IsVirtual(&self) -> bool {
            return false;
        }

        fn Sync(&self) -> Result<()> {
            return Ok(())
        }

        fn StatFS(&self, _task: &Task) -> Result<FsInfo> {
            return Err(Error::SysError(SysErr::ENOSYS))
        }
    }

    #[test]
    fn TestMountSourceOnlyCachedOnce() {
        let task = Task::default();

        let ms = Arc::new(Mutex::new(NewMockMountSource(100)));
        let rootInode = NewMockInode(&ms, &StableAttr {
            Type: InodeType::Directory,
            ..Default::default()
        });

        let mut mm = MountNs::New(&rootInode);
        let rootDirent = mm.Root();

        let child = rootDirent.Walk(&task, &rootDirent, &"child".to_string()).unwrap();
        child.ExtendReference();

        assert!(ms.lock().ContainsKey(child.lock().Id));

        let subms = Arc::new(Mutex::new(NewMockMountSource(100)));
        let submountInode = NewMockInode(&subms, &StableAttr {
            Type: InodeType::Directory,
            ..Default::default()
        });


        mm.Mount(&child, &submountInode).unwrap();

        let child2 = rootDirent.Walk(&task, &rootDirent, &"child".to_string()).unwrap();
        child2.ExtendReference();

        assert!(!child.lock().Id != child2.lock().Id);
        assert!(!subms.lock().ContainsKey(child.lock().Id));
        assert!(subms.lock().ContainsKey(child2.lock().Id));
    }
}