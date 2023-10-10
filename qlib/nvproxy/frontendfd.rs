// Copyright (c) 2021 Quark Container Authors
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

use core::sync::atomic::AtomicBool;

use alloc::borrow::ToOwned;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::Any;
use core::ops::Deref;
use core::sync::atomic::Ordering;

use crate::qlib::cstring::CString;
use crate::qlib::kernel::Kernel::HostSpace;
use crate::qlib::kernel::fs::file::*;
use crate::qlib::kernel::fs::inode::*;
use crate::qlib::kernel::guestfdnotifier::NonBlockingPoll;
use crate::qlib::kernel::guestfdnotifier::UpdateFD;
use crate::qlib::mutex::*;
use crate::qlib::auth::*;
use crate::qlib::common::*;
use crate::qlib::linux_def::*;
use crate::qlib::kernel::kernel::time::*;
use crate::qlib::kernel::kernel::waiter::qlock::*;
use crate::qlib::kernel::kernel::waiter::*;
use crate::qlib::kernel::socket::unix::transport::unix::*;
use crate::qlib::kernel::task::*;
use crate::qlib::kernel::uid::*;
use crate::qlib::kernel::fs::host::hostinodeop::*;

use crate::qlib::kernel::fs::attr::*;
use crate::qlib::kernel::fs::dentry::*;
use crate::qlib::kernel::fs::dirent::*;
//use crate::qlib::kernel::fs::file::*;
use crate::qlib::kernel::fs::flags::*;
//use crate::qlib::kernel::fs::fsutil::file::*;
use crate::qlib::kernel::fs::fsutil::inode::*;
//use crate::qlib::kernel::fs::inode::*;
use crate::qlib::kernel::fs::mount::*;
use crate::qlib::nvproxy::frontend::*;
use crate::qlib::path;
use crate::qlib::nvproxy::classes::Nv0005AllocParameters;
use crate::qlib::nvproxy::frontend_type::*;
use crate::qlib::nvproxy::nvproxy::NVProxy;
use crate::qlib::nvproxy::nvgpu::NV_CONTROL_DEVICE_MINOR;

pub struct NvFrontendDevice {
    pub nvp: NVProxy,
    pub minor: u16,
    pub attr: Arc<QRwLock<InodeSimpleAttributesInternal>>,
}


impl InodeOperations for NvFrontendDevice {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::NvFrontendDevice;
    }

    fn InodeType(&self) -> InodeType {
        return InodeType::CharacterDevice;
    }

    fn InodeFileType(&self) -> InodeFileType {
        return InodeFileType::Null;
    }

    fn WouldBlock(&self) -> bool {
        return true;
    }

    fn Lookup(&self, _task: &Task, _dir: &Inode, _name: &str) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn Create(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _name: &str,
        _flags: &FileFlags,
        _perm: &FilePermissions,
    ) -> Result<File> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn CreateDirectory(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _name: &str,
        _perm: &FilePermissions,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn CreateLink(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _oldname: &str,
        _newname: &str,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn CreateHardLink(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _target: &Inode,
        _name: &str,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn CreateFifo(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _name: &str,
        _perm: &FilePermissions,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn Remove(&self, _task: &Task, _dir: &mut Inode, _name: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn RemoveDirectory(&self, _task: &Task, _dir: &mut Inode, _name: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
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
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        let hostpath = if self.minor == NV_CONTROL_DEVICE_MINOR {
            "/dev/nvidiactl".to_owned()
        } else {
            format!("/dev/nvidia{}", self.minor)
        };

        let name = path::Clean(&hostpath);
        let cstr = CString::New(&name);

        let fd = HostSpace::OpenDevFile(-1, cstr.Ptr(), flags.ToLinux()) as i32;
        if fd < 0 {
            return Err(Error::SysError(-fd));
        }
        
        let inner = NvFrontendFileOptionsInner {
            fd: fd,
            queue: Queue::default(),
            isControl: self.minor == NV_CONTROL_DEVICE_MINOR,
            hasMmapContext: AtomicBool::new(false),
            nvp: self.nvp.clone(),
        };

        let fops = NvFrontendFileOptions(Arc::new(inner));

        let f = FileInternal {
            UniqueId: NewUID(),
            Dirent: dirent.clone(),
            flags: QMutex::new((flags, None)),
            offset: QLock::New(0),
            FileOp: fops.into(),
        };

        return Ok(File(Arc::new(f)));
    }

    fn UnstableAttr(&self, _task: &Task) -> Result<UnstableAttr> {
        let u = self.attr.read().unstable;
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
        self.attr.write().unstable.SetPermissions(task, &p);
        return true;
    }

    fn SetOwner(&self, task: &Task, _dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        self.attr.write().unstable.SetOwner(task, owner);
        return Ok(());
    }

    fn SetTimestamps(&self, task: &Task, _dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        self.attr.write().unstable.SetTimestamps(task, ts);
        return Ok(());
    }

    fn Truncate(&self, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
        return Ok(());
    }

    fn Allocate(&self, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
        return Ok(());
    }

    fn ReadLink(&self, _task: &Task, _dir: &Inode) -> Result<String> {
        return Err(Error::SysError(SysErr::ENOLINK));
    }

    fn GetLink(&self, _task: &Task, _dir: &Inode) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOLINK));
    }

    fn AddLink(&self, _task: &Task) {
        self.attr.write().unstable.Links += 1;
    }

    fn DropLink(&self, _task: &Task) {
        self.attr.write().unstable.Links -= 1;
    }

    fn IsVirtual(&self) -> bool {
        return true;
    }

    fn Sync(&self) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOSYS));
    }

    fn StatFS(&self, _task: &Task) -> Result<FsInfo> {
        return Err(Error::SysError(SysErr::ENOSYS));
    }

    fn Mappable(&self) -> Result<MMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

pub struct NvFrontendFileOptionsInner {
    pub fd: i32,
    pub queue: Queue,
    pub isControl: bool,
    pub hasMmapContext: AtomicBool,
    pub nvp: NVProxy, 
}

#[derive(Clone)]
pub struct NvFrontendFileOptions(Arc<NvFrontendFileOptionsInner>);

impl Deref for NvFrontendFileOptions {
    type Target = Arc<NvFrontendFileOptionsInner>;

    fn deref(&self) -> &Arc<NvFrontendFileOptionsInner> {
        &self.0
    }
}

impl Waitable for NvFrontendFileOptions {
    fn Readiness(&self, _task: &Task, mask: EventMask) -> EventMask {
        let fd = self.fd;
        return NonBlockingPoll(fd, mask);
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        let queue = self.queue.clone();
        queue.EventRegister(task, e, mask);
        let fd = self.fd;
        UpdateFD(fd).unwrap();
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        let queue = self.queue.clone();
        queue.EventUnregister(task, e);
        let fd = self.fd;
        UpdateFD(fd).unwrap();
    }
}

impl SpliceOperations for NvFrontendFileOptions {}

impl FileOperations for NvFrontendFileOptions {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::NvFrontendFileOptions;
    }

    fn Seekable(&self) -> bool {
        return false;
    }

    fn Seek(&self, _task: &Task, _f: &File, _whence: i32, _current: i64, _offset: i64) -> Result<i64> {
        return Err(Error::SysError(SysErr::ESPIPE));
    }

    fn ReadDir(
        &self,
        _task: &Task,
        _f: &File,
        _offset: i64,
        _serializer: &mut DentrySerializer,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn ReadAt(
        &self,
        _task: &Task,
        _f: &File,
        _dsts: &mut [IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn WriteAt(
        &self,
        _task: &Task,
        _f: &File,
        _srcs: &[IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOSPC));
    }

    fn Append(&self, _task: &Task, _f: &File, _srcs: &[IoVec]) -> Result<(i64, i64)> {
        return Err(Error::SysError(SysErr::ESPIPE));
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

    fn UnstableAttr(&self, _task: &Task, _f: &File) -> Result<UnstableAttr> {
        return Err(Error::SysError(SysErr::ENOTSUP));
    }

    fn Ioctl(&self, _task: &Task, _f: &File, _fd: i32, _request: u64, _val: u64) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTTY));
    }

    fn IterateDir(
        &self,
        _task: &Task,
        _d: &Dirent,
        _dirCtx: &mut DirCtx,
        _offset: i32,
    ) -> (i32, Result<i64>) {
        return (0, Err(Error::SysError(SysErr::ENOTDIR)));
    }

    fn Mappable(&self) -> Result<MMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

impl SockOperations for NvFrontendFileOptions {}

pub fn RMAllocSimple<Params: Sized + Clone + Copy>(
    fi: &FrontendIoctlState, 
    ioctlParams: &NVOS64Parameters,
    isNVOS64: bool
) -> Result<u64> {
    if ioctlParams.allocParms == 0 {
        return RMAllocInvoke::<u8>(fi, ioctlParams, None, isNVOS64)
    }

    let allocParams: Params = fi.task.CopyInObj(ioctlParams.allocParms)?;

    let n = RMAllocInvoke::<Params>(fi, ioctlParams, Some(&allocParams), isNVOS64)?;

    fi.task.CopyOutObj(&allocParams, ioctlParams.allocParms)?;

    return Ok(n)
}

pub fn RMAllocNoParams(
    fi: &FrontendIoctlState, 
    ioctlParams: &NVOS64Parameters,
    isNVOS64: bool
) -> Result<u64> {
    return RMAllocInvoke::<u8>(fi, ioctlParams, None, isNVOS64);
}

pub fn RMAllocEventOSEvent(
    fi: &FrontendIoctlState, 
    ioctlParams: &NVOS64Parameters,
    isNVOS64: bool
) -> Result<u64> {
    let allocParams : Nv0005AllocParameters = fi.task.CopyInObj(ioctlParams.allocParms)?;
    let eventFileGeneric = match fi.task.GetFile(allocParams.data as i32) {
        Err(_) => return Err(Error::SysError(SysErr::EINVAL)),
        Ok(f) => f
    };

    let eventFile = match &eventFileGeneric.FileOp {
        FileOps::NvFrontendFileOptions(nvfops) => {
            nvfops.clone()
        }
        _ => {
            return Err(Error::SysError(SysErr::EINVAL))
        }
    };

    let mut allocParamsTmp = allocParams;
    allocParamsTmp.data = eventFile.fd as u64;

    let n = RMAllocInvoke(fi, ioctlParams, Some(&allocParamsTmp), isNVOS64)?;

    let mut outAllocParams = allocParamsTmp;
    outAllocParams.data = allocParams.data;
    fi.task.CopyOutObj(&outAllocParams, ioctlParams.allocParms)?;
    return Ok(n)
}

pub fn RMVidHeapControl(fi: &FrontendIoctlState) -> Result<u64> {
    if fi.ioctlParamsSize as usize != core::mem::size_of::<NVOS32Parameters>() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    // Function determines the type of Data.
    let ioctlParams: NVOS32Parameters = fi.task.CopyInObj(fi.ioctlParamsAddr)?;
    debug!("nvproxy: VID_HEAP_CONTROL function {}", ioctlParams.function);

    // See
	// src/nvidia/interface/deprecated/rmapi_deprecated_vidheapctrl.c:rmVidHeapControlTable
	// for implementation.

    if ioctlParams.function == NVOS32_FUNCTION_ALLOC_SIZE {
        return RMVidHeapControlAllocSize(fi, &ioctlParams);
    } 

    warn!("nvproxy: unknown VID_HEAP_CONTROL function {}", ioctlParams.function);
    return Err(Error::SysError(SysErr::EINVAL));
}

pub fn RMMapMemory(fi: &FrontendIoctlState) -> Result<u64> {
    if fi.ioctlParamsSize as usize != core::mem::size_of::<IoctlNVOS33ParametersWithFD>() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let ioctlParams: IoctlNVOS33ParametersWithFD = fi.task.CopyInObj(fi.ioctlParamsAddr)?;

    let mapFileGeneric = match fi.task.GetFile(ioctlParams.fd) {
        Err(_) => return Err(Error::SysError(SysErr::EINVAL)),
        Ok(f) => f
    };

    let mapfile = match &mapFileGeneric.FileOp {
        FileOps::NvFrontendFileOptions(nvfops) => {
            nvfops.clone()
        }
        _ => {
            return Err(Error::SysError(SysErr::EINVAL))
        }
    };

    if mapfile.hasMmapContext.load(Ordering::Relaxed) ||
        !mapfile.hasMmapContext.compare_and_swap(false, true, Ordering::SeqCst) {
        warn!("nvproxy: attempted to reuse FD {} for NV_ESC_RM_MAP_MEMORY", ioctlParams.fd);
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let mut ioctlParamsTmp = ioctlParams;
    ioctlParamsTmp.fd = mapfile.fd;

    let n = FrontendIoctlInvoke(fi, &ioctlParamsTmp)?;
    let mut outIoctlParams = ioctlParamsTmp;
    outIoctlParams.fd = ioctlParams.fd;
    fi.task.CopyOutObj(&outIoctlParams, fi.ioctlParamsAddr)?;
    
    return Ok(n)    
}