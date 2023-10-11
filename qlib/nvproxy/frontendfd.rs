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

use crate::qlib::addr::Addr;
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
use crate::qlib::linux::ioctl::*;

use crate::qlib::kernel::fs::attr::*;
use crate::qlib::kernel::fs::dentry::*;
use crate::qlib::kernel::fs::dirent::*;
use crate::qlib::kernel::fs::flags::*;
use crate::qlib::kernel::fs::fsutil::inode::*;
use crate::qlib::kernel::fs::mount::*;
use crate::qlib::nvproxy::classes::*;
use crate::qlib::nvproxy::frontend::*;
use crate::qlib::nvproxy::nvgpu::*;
use crate::qlib::nvproxy::nvproxy::OSDescMem;
use crate::qlib::path;
use crate::qlib::nvproxy::classes::Nv0005AllocParameters;
use crate::qlib::nvproxy::frontend_type::*;
use crate::qlib::nvproxy::nvproxy::NVProxy;
use crate::qlib::range::Range;
//use crate::qlib::range::Range;

use super::nvgpu::NV_ERR_NOT_SUPPORTED;

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

    fn Ioctl(&self, task: &Task, _f: &File, _fd: i32, request: u64, val: u64) -> Result<u64> {
        let cmd = request as u32;
        let nr = IOCBits::IOC_NR(cmd);
        let argPtr = val;
        let argSize = IOCBits::IOC_SIZE(cmd);

        let fi = FrontendIoctlState {
            fd: self.clone(),
            task: task,
            nr: nr,
            ioctlParamsAddr: argPtr,
            ioctlParamsSize: argSize,
        };

        // nr determines the argument type.
        // Don't log nr since it's already visible as the last byte of cmd in
        // strace logging.
        // Implementors:
        // - To map nr to a symbol, look in
        // src/nvidia/arch/nvalloc/unix/include/nv_escape.h,
        // kernel-open/common/inc/nv-ioctl-numbers.h, and
        // kernel-open/common/inc/nv-ioctl-numa.h.
        // - To determine the parameter type, find the implementation in
        // kernel-open/nvidia/nv.c:nvidia_ioctl() or
        // src/nvidia/arch/nvalloc/unix/src/escape.c:RmIoctl().
        // - Add symbol and parameter type definitions to //pkg/abi/nvgpu.
        // - Add filter to seccomp_filters.go.
        // - Add handling below.
        match nr {
            NV_ESC_CARD_INFO |                     // nv_ioctl_card_info_t
            NV_ESC_CHECK_VERSION_STR |             // nv_rm_api_version_t
            NV_ESC_SYS_PARAMS |                    // nv_ioctl_sys_params_t
            NV_ESC_RM_DUP_OBJECT |                 // NVOS55_PARAMETERS
            NV_ESC_RM_SHARE |                      // NVOS57_PARAMETERS
            NV_ESC_RM_UNMAP_MEMORY |               // NVOS34_PARAMETERS
            NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO => { // NVOS56_PARAMETERS
                return FrontendIoctlSimple(&fi);
            }
            NV_ESC_REGISTER_FD => {
                return FrontendRegisterFD(&fi);
            }
            _ => {
                warn!("nvproxy: unknown frontend ioctl {} == {:x?} (argSize={}, cmd={:x?})", nr, nr, argSize, cmd);
                return Err(Error::SysError(SysErr::EINVAL));
            }
        }
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

// FrontendIoctlSimple implements a frontend ioctl whose parameters don't
// contain any pointers requiring translation, file descriptors, or special
// cases or effects, and consequently don't need to be typed by the qvsior.
pub fn FrontendIoctlSimple(fi: &FrontendIoctlState) -> Result<u64> {
    if fi.ioctlParamsSize == 0 {
        return FrontendIoctlInvoke::<u8>(fi, None);
    }

    let ioctlParams: Vec<u8> = fi.task.CopyInVec(fi.ioctlParamsAddr, fi.ioctlParamsSize as usize)?;
    let n = FrontendIoctlInvoke(fi, Some(&ioctlParams[0]))?;

    fi.task.CopyOutSlice(&ioctlParams, fi.ioctlParamsAddr, fi.ioctlParamsSize as usize)?;

    return Ok(n)
}

pub fn FrontendRegisterFD(fi: &FrontendIoctlState) -> Result<u64> {
    if fi.ioctlParamsSize as usize != core::mem::size_of::<IoctlRegisterFD>() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let ioctlParams: IoctlRegisterFD = fi.task.CopyInObj(fi.ioctlParamsAddr)?;

    let ctlFileGeneric = match fi.task.GetFile(ioctlParams.ctlFD) {
        Err(_) => return Err(Error::SysError(SysErr::EINVAL)),
        Ok(f) => f
    };

    let ctlFile = match &ctlFileGeneric.FileOp {
        FileOps::NvFrontendFileOptions(nvfops) => {
            nvfops.clone()
        }
        _ => {
            return Err(Error::SysError(SysErr::EINVAL))
        }
    };

    let ioctlParamsTmp = IoctlRegisterFD {
        ctlFD: ctlFile.fd
    };

    return FrontendIoctlInvoke(fi, Some(&ioctlParamsTmp))
}

pub fn RMAllocOSEvent(fi: &FrontendIoctlState) -> Result<u64> {
    if fi.ioctlParamsSize as usize != core::mem::size_of::<IoctlAllocOSEvent>() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let ioctlParams: IoctlAllocOSEvent = fi.task.CopyInObj(fi.ioctlParamsAddr)?;

    let eventFileGeneric = match fi.task.GetFile(ioctlParams.fd as i32) {
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

    let mut ioctlParamsTmp = ioctlParams;
    ioctlParamsTmp.fd = eventFile.fd as u32;

    let n = FrontendIoctlInvoke(fi, Some(&ioctlParamsTmp))?;

    let mut outIoctlParams = ioctlParamsTmp;
    outIoctlParams.fd = ioctlParams.fd;

    fi.task.CopyOutObj(&outIoctlParams, fi.ioctlParamsAddr)?;
    
    return Ok(n)
}

pub fn RMFreeOSEvent(fi: &FrontendIoctlState) -> Result<u64> {
    if fi.ioctlParamsSize as usize != core::mem::size_of::<IoctlFreeOSEvent>() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let ioctlParams: IoctlFreeOSEvent = fi.task.CopyInObj(fi.ioctlParamsAddr)?;

    let eventFileGeneric = match fi.task.GetFile(ioctlParams.fd as i32) {
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

    let mut ioctlParamsTmp = ioctlParams;
    ioctlParamsTmp.fd = eventFile.fd as u32;

    let n = FrontendIoctlInvoke(fi, Some(&ioctlParamsTmp))?;

    let mut outIoctlParams = ioctlParamsTmp;
    outIoctlParams.fd = ioctlParams.fd;

    fi.task.CopyOutObj(&outIoctlParams, fi.ioctlParamsAddr)?;
    
    return Ok(n)
}

pub fn RMAllocMemory(fi: &FrontendIoctlState) -> Result<u64> {
    if fi.ioctlParamsSize as usize != core::mem::size_of::<IoctlNVOS02ParametersWithFD>() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let ioctlParams: IoctlNVOS02ParametersWithFD = fi.task.CopyInObj(fi.ioctlParamsAddr)?;

    error!("nvproxy: NV_ESC_RM_ALLOC_MEMORY class {}", ioctlParams.params.class);

    // See src/nvidia/arch/nvalloc/unix/src/escape.c:RmIoctl() and
	// src/nvidia/interface/deprecated/rmapi_deprecated_allocmemory.c:rmAllocMemoryTable
	// for implementation.
    if ioctlParams.params.class == NV01_MEMORY_SYSTEM_OS_DESCRIPTOR {
        return RMAllocOSDescriptor(fi, &ioctlParams)
    }

    warn!("nvproxy: unknown NV_ESC_RM_ALLOC_MEMORY class {:#?}", ioctlParams.params.class);

    return Err(Error::SysError(SysErr::EINVAL))
}

pub fn FailWithStatus(fi: &FrontendIoctlState, ioctlParams: &IoctlNVOS02ParametersWithFD, status: u32) -> Result<u64> {
    let mut outIoctlParams = *ioctlParams;
    outIoctlParams.params.status = status;
    fi.task.CopyOutObj(&outIoctlParams, fi.ioctlParamsAddr)?;
    return Ok(0)
}

pub fn RMAllocOSDescriptor(fi: &FrontendIoctlState, ioctlParams: &IoctlNVOS02ParametersWithFD) -> Result<u64> {
    let appAddr = ioctlParams.params.memory;
    match Addr(appAddr).PageAligned() {
        Err(_) => return FailWithStatus(fi, ioctlParams, NV_ERR_NOT_SUPPORTED),
        Ok(_) => ()
    }

    let arLen = ioctlParams.params.limit as i64;
    if arLen == -1 {
        return FailWithStatus(fi, ioctlParams, NV_ERR_INVALID_LIMIT);
    }

    let arLen = match Addr(ioctlParams.params.limit + 1).RoundUp() {
        Err(_) => return FailWithStatus(fi, ioctlParams, NV_ERR_INVALID_ADDRESS),
        Ok(a) => a.0
    };

    let prs = fi.task.mm.Pin(fi.task, appAddr, arLen)?;
    let ret = HostSpace::RemapGuestMemRanges(
        arLen, 
        unsafe { &*(&prs as * const _ as u64 as * const &[Range])}
    );
    if ret < 0 {
        return Err(Error::SysError(ret as i32));
    }

    let hostAddr = ret as u64;

    let mut ioctlParamsTmp = *ioctlParams;
    ioctlParamsTmp.params.memory = hostAddr;
    // NV01_MEMORY_SYSTEM_OS_DESCRIPTOR shouldn't use ioctlParams.FD; clobber
	// it to be sure.
	ioctlParamsTmp.fd = -1;

    let o = OSDescMem {
        pinnedRange: prs,
    };

    let n;
    {
        let mut inner = fi.fd.nvp.lock();
        n = FrontendIoctlInvoke(fi, Some(&ioctlParamsTmp))?;

        inner.objs.insert(ioctlParamsTmp.params.objectNew, o.into());
    }

    info!("nvproxy: pinned pages for OS descriptor with handle {:x?}", ioctlParamsTmp.params.objectNew);
    // Unmap the reserved range, which is no longer required.
	HostSpace::UnmapGuestMemRange(hostAddr, arLen);
    
    let mut outIoctlParams = ioctlParamsTmp;
    outIoctlParams.params.memory = ioctlParams.params.memory;
    outIoctlParams.fd = ioctlParams.fd;

    fi.task.CopyOutObj(&outIoctlParams, fi.ioctlParamsAddr)?;

    return Ok(n)
}

pub fn RMFree(fi: &FrontendIoctlState) -> Result<u64> {
    if fi.ioctlParamsSize as usize != core::mem::size_of::<NVOS00Parameters>() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let ioctlParams: NVOS00Parameters = fi.task.CopyInObj(fi.ioctlParamsAddr)?;

    let n;
    {
        let mut inner = fi.fd.nvp.lock();
        n = FrontendIoctlInvoke(fi, Some(&ioctlParams))?;
        inner.objs.remove(&ioctlParams.objectOld);
    }

    fi.task.CopyOutObj(&ioctlParams, fi.ioctlParamsAddr)?;
    return Ok(n)
}

pub fn RMControl(fi: &FrontendIoctlState) -> Result<u64> {
    if fi.ioctlParamsSize as usize != core::mem::size_of::<NVOS54Parameters>() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let ioctlParams: NVOS54Parameters = fi.task.CopyInObj(fi.ioctlParamsAddr)?;
    debug!("nvproxy: control command {:#?}", ioctlParams.cmd);

    if ioctlParams.cmd & RM_GSS_LEGACY_MASK != 0 {
        // This is a "legacy GSS control" that is implemented by the GPU System
		// Processor (GSP). Conseqeuently, its parameters cannot reasonably
		// contain application pointers, and the control is in any case
		// undocumented.
		// See
		// src/nvidia/src/kernel/rmapi/entry_points.c:_nv04ControlWithSecInfo()
		// =>
		// src/nvidia/interface/deprecated/rmapi_deprecated_control.c:RmDeprecatedGetControlHandler()
		// =>
		// src/nvidia/interface/deprecated/rmapi_gss_legacy_control.c:RmGssLegacyRpcCmd().
        return RMControlSimple(fi, &ioctlParams);
    }

    // Implementors:
	// - Top two bytes of Cmd specifies class; third byte specifies category;
	// fourth byte specifies "message ID" (command within class/category).
	//   e.g. 0x800288:
	//   - Class 0x0080 => look in
	//   src/common/sdk/nvidia/inc/ctrl/ctrl0080/ctrl0080base.h for categories.
	//   - Category 0x02 => NV0080_CTRL_GPU => look in
	//   src/common/sdk/nvidia/inc/ctrl/ctrl0080/ctrl0080gpu.h for
	//   `#define NV0080_CTRL_CMD_GPU_QUERY_SW_STATE_PERSISTENCE (0x800288)`
	//   and accompanying documentation, parameter type.
	// - If this fails, or to find implementation, grep for `methodId=.*0x<Cmd
	// in lowercase hex without leading 0s>` to find entry in g_*_nvoc.c;
	// implementing function is is "pFunc".
	// - Add symbol definition to //pkg/abi/nvgpu. Parameter type definition is
	// only required for non-simple commands.
	// - Add handling below.
    match ioctlParams.cmd {
        NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE |
		NV0000_CTRL_CMD_CLIENT_SET_INHERITED_SHARE_POLICY |
		NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS |
		NV0000_CTRL_CMD_GPU_GET_ID_INFO |
		NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2 |
		NV0000_CTRL_CMD_GPU_GET_PROBED_IDS |
		NV0000_CTRL_CMD_GPU_ATTACH_IDS |
		NV0000_CTRL_CMD_GPU_DETACH_IDS |
		NV0000_CTRL_CMD_GPU_GET_PCI_INFO |
		NV0000_CTRL_CMD_GPU_QUERY_DRAIN_STATE |
		NV0000_CTRL_CMD_GPU_GET_MEMOP_ENABLE |
		NV0000_CTRL_CMD_SYNC_GPU_BOOST_GROUP_INFO |
		NV0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS |
		NV0000_CTRL_CMD_SYSTEM_GET_FABRIC_STATUS |
		NV0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS_MATRIX |
		NV0080_CTRL_CMD_FB_GET_CAPS_V2 |
		NV0080_CTRL_CMD_GPU_GET_NUM_SUBDEVICES |
		NV0080_CTRL_CMD_GPU_QUERY_SW_STATE_PERSISTENCE |
		NV0080_CTRL_CMD_GPU_GET_VIRTUALIZATION_MODE |
		0x80028b | // unknown | paramsSize == 1
		NV0080_CTRL_CMD_GPU_GET_CLASSLIST_V2 |
		NV0080_CTRL_CMD_HOST_GET_CAPS_V2 |
		NV2080_CTRL_CMD_BUS_GET_PCI_INFO |
		NV2080_CTRL_CMD_BUS_GET_PCI_BAR_INFO |
		NV2080_CTRL_CMD_BUS_GET_INFO_V2 |
		NV2080_CTRL_CMD_BUS_GET_PCIE_SUPPORTED_GPU_ATOMICS |
		NV2080_CTRL_CMD_CE_GET_ALL_CAPS |
		NV2080_CTRL_CMD_FB_GET_INFO_V2 |
		NV2080_CTRL_CMD_GPU_GET_INFO_V2 |
		NV2080_CTRL_CMD_GPU_GET_NAME_STRING |
		NV2080_CTRL_CMD_GPU_GET_SHORT_NAME_STRING |
		NV2080_CTRL_CMD_GPU_GET_SIMULATION_INFO |
		NV2080_CTRL_CMD_GPU_QUERY_ECC_STATUS |
		NV2080_CTRL_CMD_GPU_QUERY_COMPUTE_MODE_RULES |
		NV2080_CTRL_CMD_GPU_ACQUIRE_COMPUTE_MODE_RESERVATION |
		NV2080_CTRL_CMD_GPU_RELEASE_COMPUTE_MODE_RESERVATION |
		NV2080_CTRL_CMD_GPU_GET_GID_INFO |
		NV2080_CTRL_CMD_GPU_GET_ENGINES_V2 |
		NV2080_CTRL_CMD_GPU_GET_ACTIVE_PARTITION_IDS |
		NV2080_CTRL_CMD_GPU_GET_COMPUTE_POLICY_CONFIG |
		NV2080_CTRL_CMD_GET_GPU_FABRIC_PROBE_INFO |
		NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE |
		NV2080_CTRL_CMD_GR_GET_CTX_BUFFER_SIZE |
		NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER |
		NV2080_CTRL_CMD_GR_GET_CAPS_V2 |
		NV2080_CTRL_CMD_GR_GET_GPC_MASK |
		NV2080_CTRL_CMD_GR_GET_TPC_MASK |
		NV2080_CTRL_CMD_GSP_GET_FEATURES |
		NV2080_CTRL_CMD_MC_GET_ARCH_INFO |
		NV2080_CTRL_CMD_MC_SERVICE_INTERRUPTS |
		NV2080_CTRL_CMD_NVLINK_GET_NVLINK_STATUS |
		NV2080_CTRL_CMD_PERF_BOOST |
		NV2080_CTRL_CMD_RC_GET_WATCHDOG_INFO |
		NV2080_CTRL_CMD_RC_RELEASE_WATCHDOG_REQUESTS |
		NV2080_CTRL_CMD_RC_SOFT_DISABLE_WATCHDOG |
		NV2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO |
		NV503C_CTRL_CMD_REGISTER_VA_SPACE |
		NV503C_CTRL_CMD_REGISTER_VIDMEM |
		NV503C_CTRL_CMD_UNREGISTER_VIDMEM |
		NV83DE_CTRL_CMD_DEBUG_SET_EXCEPTION_MASK |
		NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES |
		NV83DE_CTRL_CMD_DEBUG_CLEAR_ALL_SM_ERROR_STATES |
		NV906F_CTRL_CMD_RESET_CHANNEL |
		NV90E6_CTRL_CMD_MASTER_GET_ERROR_INTR_OFFSET_MASK |
		NV90E6_CTRL_CMD_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK |
		NVC36F_CTRL_GET_CLASS_ENGINEID |
		NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN |
		NVA06C_CTRL_CMD_GPFIFO_SCHEDULE |
		NVA06C_CTRL_CMD_SET_TIMESLICE |
		NVA06C_CTRL_CMD_PREEMPT => {
            return RMControlSimple(fi, &ioctlParams);
        }
        NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION => {
            return CtrlClientSystemGetBuildVersion(fi, &ioctlParams);
        }
        NV0080_CTRL_CMD_FIFO_GET_CHANNELLIST => {
            return CtrlDevFIFOGetChannelList(fi, &ioctlParams);
        }
        NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS => {
            return CtrlSubdevFIFODisableChannels(fi, &ioctlParams);
        }
        NV2080_CTRL_CMD_GR_GET_INFO => {
            return CtrlSubdevGRGetInfo(fi, &ioctlParams);
        }
        _ => {
            warn!("nvproxy: unknown control command {:x?} (paramsSize={})", ioctlParams.cmd, ioctlParams.paramsSize);
            return Err(Error::SysError(SysErr::EINVAL));
        }
    }
}

pub fn RMControlSimple(fi: &FrontendIoctlState, ioctlParams: &NVOS54Parameters) -> Result<u64> {
    if ioctlParams.paramsSize == 0 {
        if ioctlParams.params != 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        return RMControlInvoke::<u8>(fi, ioctlParams, None);
    }

    if ioctlParams.params == 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let ctrlParams: Vec<u8> = fi.task.CopyInVec(ioctlParams.params, ioctlParams.paramsSize as usize)?;
    let n = RMControlInvoke(fi, &ioctlParams, Some(&ctrlParams))?;

    fi.task.CopyOutSlice(&ctrlParams, ioctlParams.params, ioctlParams.paramsSize as usize)?;
    return Ok(n)
}

pub fn CtrlClientSystemGetBuildVersion(fi: &FrontendIoctlState, ioctlParams: &NVOS54Parameters) -> Result<u64> {
    if fi.ioctlParamsSize as usize != core::mem::size_of::<Nv0000CtrlSystemGetBuildVersionParams>() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let ctrlParams: Nv0000CtrlSystemGetBuildVersionParams = fi.task.CopyInObj(fi.ioctlParamsAddr)?;
    if ctrlParams.driverVersionBuffer == 0 || ctrlParams.versionBuffer == 0 || ctrlParams.titleBuffer == 0 {
        // No strings are written if any are null. See
		// src/nvidia/interface/deprecated/rmapi_deprecated_control.c:V2_CONVERTER(_NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION).
		return CtrlClientSystemGetBuildVersionInvoke(fi, ioctlParams, &ctrlParams, 0, 0, 0)
    }

    // Need to buffer strings for copy-out.
	if ctrlParams.sizeOfStrings == 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let mut driverVersionBuf: Vec<u8> = Vec::with_capacity(ctrlParams.sizeOfStrings as usize);
    driverVersionBuf.resize(ctrlParams.sizeOfStrings as usize, 0);
    let mut versionBuf: Vec<u8> = Vec::with_capacity(ctrlParams.sizeOfStrings as usize);
    versionBuf.resize(ctrlParams.sizeOfStrings as usize, 0);
    let mut titleBuf: Vec<u8> = Vec::with_capacity(ctrlParams.sizeOfStrings as usize);
    titleBuf.resize(ctrlParams.sizeOfStrings as usize, 0);

    let n = CtrlClientSystemGetBuildVersionInvoke(
        fi, 
        ioctlParams, 
        &ctrlParams, 
        &driverVersionBuf[0] as * const _ as u64, 
        &versionBuf[0] as * const _ as u64,  
        &titleBuf[0] as * const _ as u64
    )?;

    fi.task.CopyOutSlice(&driverVersionBuf, ctrlParams.driverVersionBuffer, ctrlParams.sizeOfStrings as usize)?;
    fi.task.CopyOutSlice(&versionBuf, ctrlParams.versionBuffer, ctrlParams.sizeOfStrings as usize)?;
    fi.task.CopyOutSlice(&titleBuf, ctrlParams.titleBuffer, ctrlParams.sizeOfStrings as usize)?;

    return Ok(n)
}

pub fn CtrlSubdevFIFODisableChannels(fi: &FrontendIoctlState, ioctlParams: &NVOS54Parameters) -> Result<u64> {
    if fi.ioctlParamsSize as usize != core::mem::size_of::<Nv2080CtrlFifoDisableChannelsParams>() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let ctrlParams: Nv2080CtrlFifoDisableChannelsParams = fi.task.CopyInObj(fi.ioctlParamsAddr)?;

    // This pointer must be NULL; see
	// src/nvidia/src/kernel/gpu/fifo/kernel_fifo_ctrl.c:subdeviceCtrlCmdFifoDisableChannels_IMPL().
	// Consequently, we don't need to translate it, but we do want to ensure
	// that it actually is NULL.
    if ctrlParams.runlistPreemptEvent != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }
    let n = RMControlInvoke(fi, ioctlParams, Some(&ctrlParams))?;
    fi.task.CopyOutObj(&ctrlParams, ioctlParams.params)?;
    return Ok(n)
}

pub fn RMAlloc(fi: &FrontendIoctlState) -> Result<u64> {
    let ioctlParams: NVOS64Parameters;
    let mut isNVOS64 = false;
    match fi.ioctlParamsSize {
        SIZEOF_NVOS21_PARAMETERS => {
            let buf : NVOS21Parameters = fi.task.CopyInObj(fi.ioctlParamsAddr)?;
            ioctlParams = NVOS64Parameters {
                root: buf.root,
                objectParent: buf.objectParent,
                objectNew: buf.objectNew,
                class: buf.class,
                allocParms: buf.allocParms,
                status: buf.status,
                rightsRequested: 0,
                flags: 0,
            };
        }
        SIZEOF_NVOS64_PARAMETERS => {
            ioctlParams = fi.task.CopyInObj(fi.ioctlParamsAddr)?;
            isNVOS64 = true;
        }
        _ => {
            return Err(Error::SysError(SysErr::EINVAL));
        }
    }

    // hClass determines the type of pAllocParms.
    debug!("nvproxy: allocation class {:#?}", ioctlParams.class);

    // Implementors:
	// - To map hClass to a symbol, look in
	// src/nvidia/generated/g_allclasses.h.
	// - See src/nvidia/src/kernel/rmapi/resource_list.h for table mapping class
	// ("External Class") to the type of pAllocParms ("Alloc Param Info") and
	// the class whose constructor interprets it ("Internal Class").
	// - Add symbol and parameter type definitions to //pkg/abi/nvgpu.
	// - Add handling below.
    match ioctlParams.class {
        NV01_ROOT |
        NV01_ROOT_NON_PRIV |
        NV01_ROOT_CLIENT => {
            return RMAllocSimple::<Handle>(fi, &ioctlParams, isNVOS64);
        }
        NV01_EVENT_OS_EVENT => {
            return RMAllocEventOSEvent(fi, &ioctlParams, isNVOS64);
        }
        NV01_DEVICE_0 => {
            return RMAllocSimple::<Nv0080AllocParameters>(fi, &ioctlParams, isNVOS64);
        }
        NV20_SUBDEVICE_0 => {
            return RMAllocSimple::<Nv2080AllocParameters>(fi, &ioctlParams, isNVOS64);
        }
        NV50_THIRD_PARTY_P2P => {
            return RMAllocSimple::<Nv503cAllocParameters>(fi, &ioctlParams, isNVOS64);
        }
        GT200_DEBUGGER => {
            return RMAllocSimple::<Nv83deAllocParameters>(fi, &ioctlParams, isNVOS64);
        }
        FERMI_CONTEXT_SHARE_A => {
            return RMAllocSimple::<NvCtxshareAllocationParameters>(fi, &ioctlParams, isNVOS64);
        }
        FERMI_VASPACE_A => {
            return RMAllocSimple::<NvVaspaceAllocationParameters>(fi, &ioctlParams, isNVOS64);
        }
        KEPLER_CHANNEL_GROUP_A => {
            return RMAllocSimple::<NvChannelGroupAllocationParameters>(fi, &ioctlParams, isNVOS64);
        }
        VOLTA_CHANNEL_GPFIFO_A |
        TURING_CHANNEL_GPFIFO_A |
        AMPERE_CHANNEL_GPFIFO_A => {
            return RMAllocSimple::<NvChannelAllocParams>(fi, &ioctlParams, isNVOS64);
        }
        VOLTA_DMA_COPY_A |
        TURING_DMA_COPY_A |
        AMPERE_DMA_COPY_A |
        AMPERE_DMA_COPY_B |
        HOPPER_DMA_COPY_A => {
            return RMAllocSimple::<Nvb0b5AllocationParameters>(fi, &ioctlParams, isNVOS64);
        }
        VOLTA_COMPUTE_A |
        TURING_COMPUTE_A |
        AMPERE_COMPUTE_A |
        AMPERE_COMPUTE_B |
        ADA_COMPUTE_A |
        HOPPER_COMPUTE_A => {
            return RMAllocSimple::<NvGrAllocationParameters>(fi, &ioctlParams, isNVOS64);
        }
        HOPPER_USERMODE_A => {
            return RMAllocSimple::<NvHopperUsermodeAParams>(fi, &ioctlParams, isNVOS64);
        }
        GF100_SUBDEVICE_MASTER |
        VOLTA_USERMODE_A |
        TURING_USERMODE_A => {
            return RMAllocNoParams(fi, &ioctlParams, isNVOS64);
        }
        NV_MEMORY_FABRIC => {
            return RMAllocSimple::<Nv00f8AllocationParameters>(fi, &ioctlParams, isNVOS64);
        }
        _ => {
            warn!("nvproxy: unknown allocation class {:#?}", ioctlParams.class);
            return Err(Error::SysError(SysErr::EINVAL));
        }
    }
}

// Unlike frontendIoctlSimple and rmControlSimple, rmAllocSimple requires the
// parameter type since the parameter's size is otherwise unknown.
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

    let n = FrontendIoctlInvoke(fi, Some(&ioctlParamsTmp))?;
    let mut outIoctlParams = ioctlParamsTmp;
    outIoctlParams.fd = ioctlParams.fd;
    fi.task.CopyOutObj(&outIoctlParams, fi.ioctlParamsAddr)?;
    
    return Ok(n)    
}