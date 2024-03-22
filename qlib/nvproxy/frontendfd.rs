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
use crate::qlib::kernel::fs::inode::*;
use crate::qlib::kernel::util::sharedstring::SharedString;
use crate::qlib::kernel::Kernel::HostSpace;
use crate::qlib::kernel::fs::file::*;
use crate::qlib::kernel::guestfdnotifier::NonBlockingPoll;
use crate::qlib::kernel::guestfdnotifier::UpdateFD;
use crate::qlib::kernel::memmgr::mm::MemoryManager;
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

#[derive(Debug, Clone)]
pub struct NvFrontendDevice {
    pub nvp: NVProxy,
    pub minor: u32,
    pub attr: Arc<QRwLock<InodeSimpleAttributesInternal>>,
}

impl NvFrontendDevice {
    pub fn New(task: &Task, nvp: &NVProxy, minor: u32, owner: &FileOwner, mode: &FileMode) -> Self {
        let attr = InodeSimpleAttributesInternal::New(
            task,
            owner,
            &FilePermissions::FromMode(*mode),
            FSMagic::TMPFS_MAGIC,
        );

        return Self {
            nvp: nvp.clone(),
            minor: minor,
            attr: Arc::new(QRwLock::new(attr)),
        }
    }
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
        let cstr = SharedString::New(&name);

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
            mapRange: QMutex::new(None),
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

#[derive(Clone, Debug)]
pub struct NvFrontendMapRange {
    pub fileOffset: u64,
    pub phyAddr: u64,
    pub len: u64,
}

pub struct NvFrontendFileOptionsInner {
    pub fd: i32,
    pub queue: Queue,
    pub isControl: bool,
    pub hasMmapContext: AtomicBool,
    pub nvp: NVProxy, 
    pub mapRange: QMutex<Option<NvFrontendMapRange>>,
}

impl NvFrontendFileOptionsInner {
    pub fn MapInternal(&self, _task: &Task, fr: &Range, writeable: bool) -> Result<IoVec> {
        assert!(fr.start == 0);

        let prot = if writeable {
            (MmapProt::PROT_WRITE | MmapProt::PROT_READ) as i32
        } else {
            MmapProt::PROT_READ as i32
        };

        let ret = HostSpace::MMapFile(fr.len, self.fd, fr.start, prot);
        
        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        let phyAddr = ret as u64;

        assert!(self.mapRange.lock().is_none());

        *self.mapRange.lock() = Some(NvFrontendMapRange {
            fileOffset: fr.start,
            phyAddr: phyAddr,
            len: fr.len
        });
        
        return Ok(IoVec { start: phyAddr, len: fr.len as usize });
    }

    pub fn Unmap( 
        &self,
        _ms: &MemoryManager,
        ar: &Range,
        offset: u64
    ) -> Result<()> {
        let mapRange = match self.mapRange.lock().take() {
            None => return Err(Error::SysError(SysErr::EINVAL)),
            Some(mr) => mr,
        };

        assert!(mapRange.fileOffset == offset);
        assert!(mapRange.len == ar.len);
        HostSpace::MUnmap(mapRange.phyAddr, mapRange.len);
        
        return Ok(())
    }
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
        let handler = match self.nvp.lock().frontendIoctl.get(&nr) {
            Some(h) => h.clone(),
            None => {
                error!("nvproxy: unknown frontend ioctl {} == {:x?} (argSize={}, cmd={:x?})", nr, nr, argSize, cmd);
                return Err(Error::SysError(SysErr::EINVAL));
            }
        };

        return handler(&fi);
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
        return Ok(MMappable::FromNvFrontendFops(self.clone()));
    }
}

impl PartialEq for NvFrontendFileOptions {
    fn eq(&self, other: &Self) -> bool {
        return Arc::ptr_eq(&self.0, &other.0);
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

pub fn RMNumaInfo(_fi: &FrontendIoctlState) -> Result<u64> {
    // The CPU topology seen by the host driver differs from the CPU
	// topology presented by the sentry to the application, so reject this
	// ioctl; doing so is non-fatal.
    debug!("nvproxy: ignoring NV_ESC_NUMA_INFO");
    return Err(Error::SysError(SysErr::EINVAL));
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
        FileOps::OverlayFileOperations(of) => {
            match of.FileOps() {
                FileOps::NvFrontendFileOptions(nvfops) => {
                    nvfops.clone()
                }
                _ => {
                    return Err(Error::SysError(SysErr::EINVAL))
                }
            }
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
        FileOps::OverlayFileOperations(of) => {
            match of.FileOps() {
                FileOps::NvFrontendFileOptions(nvfops) => {
                    nvfops.clone()
                }
                _ => {
                    return Err(Error::SysError(SysErr::EINVAL))
                }
            }
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
        FileOps::OverlayFileOperations(of) => {
            match of.FileOps() {
                FileOps::NvFrontendFileOptions(nvfops) => {
                    nvfops.clone()
                }
                _ => {
                    return Err(Error::SysError(SysErr::EINVAL))
                }
            }
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

    // See src/nvidia/arch/nvalloc/unix/src/escape.c:RmIoctl() and
	// src/nvidia/interface/deprecated/rmapi_deprecated_allocmemory.c:rmAllocMemoryTable
	// for implementation.
    if ioctlParams.params.class == NV01_MEMORY_SYSTEM_OS_DESCRIPTOR {
        return RMAllocOSDescriptor(fi, &ioctlParams)
    }

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
        &prs[0] as * const _ as u64,
        prs.len()
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
    debug!("nvproxy: control command {:#x?}", ioctlParams);

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
    let handler = match fi.fd.nvp.lock().controlCmd.get(&ioctlParams.cmd) {
        Some(handler) => handler.clone(),
        None => {
            warn!("nvproxy: unknown control command {:x?} (paramsSize={})", ioctlParams.cmd, ioctlParams.paramsSize);
            return Err(Error::SysError(SysErr::EINVAL));
        }
    };

    return handler(fi, &ioctlParams);
}

pub fn RMControlSimple(fi: &FrontendIoctlState, ioctlParams: &NVOS54Parameters) -> Result<u64> {
    if ioctlParams.paramsSize == 0 {
        if ioctlParams.params != 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        return RMControlInvoke(fi, ioctlParams, 0);
    }

    if ioctlParams.params == 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let ctrlParams: Vec<u8> = fi.task.CopyInVec(ioctlParams.params, ioctlParams.paramsSize as usize)?;
    let n = RMControlInvoke(fi, &ioctlParams, &ctrlParams[0] as * const _ as u64)?;
    
    fi.task.CopyOutSlice(&ctrlParams, ioctlParams.params, ioctlParams.paramsSize as usize)?;
    return Ok(n)
}

pub fn CtrlClientSystemGetBuildVersion(fi: &FrontendIoctlState, ioctlParams: &NVOS54Parameters) -> Result<u64> {
    // if fi.ioctlParamsSize as usize != core::mem::size_of::<Nv0000CtrlSystemGetBuildVersionParams>() {
    //     return Err(Error::SysError(SysErr::EINVAL));
    // }

    let ctrlParams: Nv0000CtrlSystemGetBuildVersionParams = fi.task.CopyInObj(ioctlParams.params)?;
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

    // error!("CtrlClientSystemGetBuildVersion driverVersionBuf is {:?}", &driverVersionBuf);
    // error!("CtrlClientSystemGetBuildVersion versionBuf is {:?}", &versionBuf);
    // error!("CtrlClientSystemGetBuildVersion titleBuf is {:?}", &titleBuf);

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
    let n = RMControlInvoke(fi, ioctlParams, &ctrlParams as * const _ as u64)?;
    fi.task.CopyOutObj(&ctrlParams, ioctlParams.params)?;
    return Ok(n)
}

pub fn RMAlloc(fi: &FrontendIoctlState) -> Result<u64> {
    let isV535 = fi.fd.nvp.lock().useRmAllocParamsV535;
    let isNVOS64;
    if isV535 {
        match fi.ioctlParamsSize {
            SIZEOF_NVOS21_PARAMETERS_V535 => {
                isNVOS64 = false;
            }
            SIZEOF_NVOS64_PARAMETERS_V535 => {
                isNVOS64 = true;
            }
            _ => return Err(Error::SysError(SysErr::EINVAL)), 
        }
    } else {
        match fi.ioctlParamsSize {
            SIZEOF_NVOS21_PARAMETERS => {
                isNVOS64 = false;
            }
            SIZEOF_NVOS64_PARAMETERS => {
                isNVOS64 = true;
            }
            _ => return Err(Error::SysError(SysErr::EINVAL)), 
        }
    }

    let ioctlParams = if isNVOS64 {
        if isV535 {
            let p : NVOS64ParametersV535 = fi.task.CopyInObj(fi.ioctlParamsAddr)?;
            p.ToOS64V535()
        } else {
            let p : NVOS64Parameters = fi.task.CopyInObj(fi.ioctlParamsAddr)?;
            p.ToOS64V535()
        }
    } else {
        if isV535 {
            let p : NVOS21ParametersV535 = fi.task.CopyInObj(fi.ioctlParamsAddr)?;
            p.ToOS64V535()
        } else {
            let p : NVOS21Parameters = fi.task.CopyInObj(fi.ioctlParamsAddr)?;
            p.ToOS64V535()
        }
    };

    let handler = match fi.fd.nvp.lock().allocationClass.get(&ioctlParams.class) {
        Some(handler) => {
            handler.clone()
        }
        None => {
            warn!("nvproxy: unknown allocation class {:x}", ioctlParams.class);
            return Err(Error::SysError(SysErr::EINVAL));
        }
    };

    return handler(fi, &ioctlParams, isNVOS64);
}

// Unlike frontendIoctlSimple and rmControlSimple, rmAllocSimple requires the
// parameter type since the parameter's size is otherwise unknown.
pub fn RMAllocSimple<Params: Sized + Clone + Copy>(
    fi: &FrontendIoctlState, 
    ioctlParams: &NVOS64ParametersV535,
    isNVOS64: bool
) -> Result<u64> {
    if ioctlParams.allocParms == 0 {
        return RMAllocInvoke(fi, ioctlParams, 0, isNVOS64)
    }

    let allocParams: Params = fi.task.CopyInObj(ioctlParams.allocParms)?;

    let n = RMAllocInvoke(fi, ioctlParams, &allocParams as * const _ as u64, isNVOS64)?;

    fi.task.CopyOutObj(&allocParams, ioctlParams.allocParms)?;

    return Ok(n)
}

pub fn RMAllocNoParams(
    fi: &FrontendIoctlState, 
    ioctlParams: &NVOS64ParametersV535,
    isNVOS64: bool
) -> Result<u64> {
    return RMAllocInvoke(fi, ioctlParams, 0, isNVOS64);
}

pub fn RMAllocEventOSEvent(
    fi: &FrontendIoctlState, 
    ioctlParams: &NVOS64ParametersV535,
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
        FileOps::OverlayFileOperations(of) => {
            match of.FileOps() {
                FileOps::NvFrontendFileOptions(nvfops) => {
                    nvfops.clone()
                }
                _ => {
                    return Err(Error::SysError(SysErr::EINVAL))
                }
            }
        }
        _ => {
            return Err(Error::SysError(SysErr::EINVAL))
        }
    };

    let mut allocParamsTmp = allocParams;
    allocParamsTmp.data = eventFile.fd as u64;

    let n = RMAllocInvoke(fi, ioctlParams, &allocParamsTmp as * const _ as u64, isNVOS64)?;

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
        FileOps::OverlayFileOperations(of) => {
            match of.FileOps() {
                FileOps::NvFrontendFileOptions(nvfops) => {
                    nvfops.clone()
                }
                _ => {
                    return Err(Error::SysError(SysErr::EINVAL))
                }
            }
        }
        _ => {
            return Err(Error::SysError(SysErr::EINVAL))
        }
    };

    if mapfile.hasMmapContext.load(Ordering::Relaxed) ||
        mapfile.hasMmapContext.compare_and_swap(false, true, Ordering::SeqCst) {
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