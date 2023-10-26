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

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::Any;
use core::ops::Deref;

use crate::qlib::cstring::CString;
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

use crate::qlib::kernel::fs::attr::*;
use crate::qlib::kernel::fs::dentry::*;
use crate::qlib::kernel::fs::dirent::*;
use crate::qlib::kernel::fs::flags::*;
use crate::qlib::kernel::fs::fsutil::inode::*;
use crate::qlib::kernel::fs::mount::*;
use crate::qlib::path;
use crate::qlib::nvproxy::nvproxy::NVProxy;
use crate::qlib::kernel::fs::inode::*;
use crate::qlib::nvproxy::nvgpu::*;
use crate::qlib::range::Range;

use super::uvm::*;

#[derive(Debug, Clone)]
pub struct UvmDevice {
    pub nvp: NVProxy,
    pub attr: Arc<QRwLock<InodeSimpleAttributesInternal>>,
}

impl UvmDevice {
    pub fn New(task: &Task, nvp: &NVProxy, owner: &FileOwner, mode: &FileMode) -> Self {
        let attr = InodeSimpleAttributesInternal::New(
            task,
            owner,
            &FilePermissions::FromMode(*mode),
            FSMagic::TMPFS_MAGIC,
        );

        return UvmDevice {
            nvp: nvp.clone(),
            attr: Arc::new(QRwLock::new(attr)),
        }
    }
}

impl InodeOperations for UvmDevice {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::UvmDevice;
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
        let hostpath = "/dev/nvidia-uvm";

        let name = path::Clean(&hostpath);
        let cstr = CString::New(&name);

        let fd = HostSpace::OpenDevFile(-1, cstr.Ptr(), flags.ToLinux()) as i32;
        if fd < 0 {
            return Err(Error::SysError(-fd));
        }
        
        let inner = UvmFileOptionsInner {
            fd: fd,
            queue: Queue::default(),
            nvp: self.nvp.clone(),
            mapRange: QMutex::new(None),
        };

        let fops = UvmFileOptions(Arc::new(inner));

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
        error!("UvmDevice Mappable 1");
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

#[derive(Clone, Debug)]
pub struct UvmMapRange {
    pub fileOffset: u64,
    pub phyAddr: u64,
    pub len: u64,
}

pub struct UvmFileOptionsInner {
    pub fd: i32,
    pub queue: Queue,
    pub nvp: NVProxy, 
    pub mapRange: QMutex<Option<UvmMapRange>>,
}

impl UvmFileOptionsInner {
    pub fn MapInternal(&self, _task: &Task, addr: u64, fr: &Range, writeable: bool) -> Result<IoVec> {
        error!("uvmMapInternal 1 addr is {:x} writeable {}", addr, writeable);

        
        let prot = if writeable {
            (MmapProt::PROT_WRITE | MmapProt::PROT_READ) as i32
        } else {
            MmapProt::PROT_READ as i32
        };

        let ret = if MemoryDef::NVIDIA_START_ADDR <= addr && addr < MemoryDef::NVIDIA_START_ADDR + MemoryDef::NVIDIA_ADDR_SIZE {
            error!("uvmMapInternal 2"); 
            let _flags = (MmapFlags::MAP_FIXED | MmapFlags::MAP_SHARED) as i32;
            HostSpace::NvidiaMMap(addr, fr.len, prot, 0x11, self.fd, fr.start)
        } else {
            error!("uvmMapInternal 3"); 
            HostSpace::MMapFile(fr.len, self.fd, fr.start, prot)
        };

        error!("uvmMapInternal 4 {:x}", ret); 
        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        let phyAddr = ret as u64;

        assert!(self.mapRange.lock().is_none());

        error!("uvmMapInternal 5 {:x}", phyAddr);
        *self.mapRange.lock() = Some(UvmMapRange {
            fileOffset: fr.start,
            phyAddr: phyAddr,
            len: fr.len
        });
        error!("uvmMapInternal 5 ");
        
        return Ok(IoVec { start: phyAddr, len: fr.len as usize });
    }

    pub fn Unmap( 
        &self,
        _ms: &MemoryManager,
        ar: &Range,
        offset: u64
    ) -> Result<()> {
        error!("uvm Unmap 1");
        let mapRange = match self.mapRange.lock().take() {
            None => return Ok(()),
            Some(mr) => mr,
        };

        error!("uvm Unmap 2 {:x?} {:x?} {:x}", &mapRange, ar, offset);
        assert!(mapRange.fileOffset == offset);
        error!("uvm Unmap 3");
        assert!(mapRange.len == ar.len);
        error!("uvm Unmap 4");
        HostSpace::MUnmap(mapRange.phyAddr, mapRange.len);
        error!("uvm Unmap 1");
        
        return Ok(())
    }
}

#[derive(Clone)]
pub struct UvmFileOptions(Arc<UvmFileOptionsInner>);

impl Deref for UvmFileOptions {
    type Target = Arc<UvmFileOptionsInner>;

    fn deref(&self) -> &Arc<UvmFileOptionsInner> {
        &self.0
    }
}

impl Waitable for UvmFileOptions {
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

impl SpliceOperations for UvmFileOptions {}

impl FileOperations for UvmFileOptions {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::UvmFileOptions;
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
        let argPtr = val;
        
        let ui = UvmIoctlState {
            fd: self.clone(),
            cmd: cmd,
            ioctlParamsAddr: argPtr,
        };

        error!("nvmfd ioctl cmd is {:x}/{}", cmd, cmd);
        
        let handler = match self.nvp.lock().uvmIoctl.get(&cmd) {
            Some(h) => {
                h.clone()
            }
            None => {
                warn!("nvproxy: unknown uvm ioctl {}", cmd);
                return Err(Error::SysError(SysErr::EINVAL));
            }
        };

        match handler(task, &ui) {
            Err(e) => {
                error!("uvm ioctl error {:?}", &e);
                return Err(e);
            }
            Ok(v) => {
                return Ok(v);
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
        error!("uvm mmap...");
        return Ok(MMappable::FromUvmFops(self.clone()));
    }
}

impl PartialEq for UvmFileOptions {
    fn eq(&self, other: &Self) -> bool {
        return Arc::ptr_eq(&self.0, &other.0);
    }
}

impl SockOperations for UvmFileOptions {}

pub fn UvmIoctlInvoke<Params: Sized>(
    ui: &UvmIoctlState, 
    params: Option<&Params>
) -> Result<u64> {
    let paramsAddr = match params {
        None => 0,
        Some(p) => p as * const _ as u64
    };

    let n = HostSpace::IoCtl(
        ui.fd.fd, 
        ui.cmd as u64, 
        paramsAddr
    ); 
    if n < 0 {
        return Err(Error::SysError(n as i32));
    }

    return Ok(n as u64)
}

pub fn UvmIoctlSimple<Params: Sized + Copy + alloc::fmt::Debug>(task: &Task, ui: &UvmIoctlState) -> Result<u64> {
    let ioctlParams: Params = task.CopyInObj(ui.ioctlParamsAddr)?;

    error!("UvmIoctlSimple {:x?}", &ioctlParams);
    let n = UvmIoctlInvoke(ui, Some(&ioctlParams))?;
    
	task.CopyOutObj(&ioctlParams, ui.ioctlParamsAddr)?;
    return Ok(n);
}

pub fn UvmIoctlNoParams(_task: &Task, ui: &UvmIoctlState) -> Result<u64> {
    return UvmIoctlInvoke::<u8>(ui, None);
}

pub fn UvmInitialize(task: &Task, ui: &UvmIoctlState) -> Result<u64> {
    let ioctlParams : UvmInitializeParams = task.CopyInObj(ui.ioctlParamsAddr)?;

    let mut ioctlParamsTmp = ioctlParams;
    // This is necessary to share the host UVM FD between sentry and
	// application processes.
	ioctlParamsTmp.flags = ioctlParams.flags | UVM_INIT_FLAGS_MULTI_PROCESS_SHARING_MODE;

    let n = UvmIoctlInvoke(ui, Some(&ioctlParamsTmp))?;
    let mut outIoctlParams = ioctlParamsTmp;
    outIoctlParams.flags &= ioctlParams.flags | UVM_INIT_FLAGS_MULTI_PROCESS_SHARING_MODE;
    task.CopyOutObj(&outIoctlParams, ui.ioctlParamsAddr)?;
    return Ok(n)
}

pub fn UvmMMInitialize(task: &Task, ui: &UvmIoctlState) -> Result<u64> {
    let ioctlParams : UvmMmInitializeParams = task.CopyInObj(ui.ioctlParamsAddr)?;

    error!("UvmMMInitialize 0 {}", ioctlParams.uvmFD);
    let FailWithStatus = |status: u32| -> Result<u64> {
        let mut outIoctlParams = ioctlParams;
        outIoctlParams.status = status;
        task.CopyOutObj(&outIoctlParams, ui.ioctlParamsAddr)?;
        return Ok(0)
    };

    error!("UvmMMInitialize 0.1"); 
    let uvmFileGeneric = match task.GetFile(ioctlParams.uvmFD) {
        Err(_) => {
            return FailWithStatus(NV_ERR_INVALID_ARGUMENT);
        }
        Ok(f) => f
    };

    error!("UvmMMInitialize 0.2"); 
    let uvmFile = match &uvmFileGeneric.FileOp {
        FileOps::UvmFileOptions(ops) => {
            ops.clone()
        }
        FileOps::OverlayFileOperations(of) => {
            match of.FileOps() {
                FileOps::UvmFileOptions(ops) => {
                    ops.clone()
                }
                _ => {
                    return FailWithStatus(NV_ERR_INVALID_ARGUMENT);
                }
            }
        }
        _ => {
            return FailWithStatus(NV_ERR_INVALID_ARGUMENT);
        }
    };
    
    let mut ioctlParamsTmp = ioctlParams;
    error!("UvmMMInitialize 1 ioctlParamsTmp is {:?}", &ioctlParamsTmp);
    ioctlParamsTmp.uvmFD = uvmFile.fd;
    let n = UvmIoctlInvoke(ui, Some(&ioctlParamsTmp))?;

    let mut outIoctlParams = ioctlParamsTmp;
    outIoctlParams.uvmFD = ioctlParams.uvmFD;

    error!("UvmMMInitialize 2 outIoctlParams is {:?}", &outIoctlParams);

    task.CopyOutObj(&outIoctlParams, ui.ioctlParamsAddr)?;

    return Ok(n)
}

pub fn uvmIoctlHasRMCtrlFD<Params: Sized + Copy + HasRMCtrlFD>(task: &Task, ui: &UvmIoctlState) -> Result<u64> {
    let ioctlParams : Params = task.CopyInObj(ui.ioctlParamsAddr)?;

    let rmCtrlFD = ioctlParams.GetRMCtrlFD();
    if rmCtrlFD < 0 {
        let n = UvmIoctlInvoke(ui, Some(&ioctlParams))?;
        task.CopyOutObj(&ioctlParams, ui.ioctlParamsAddr)?;
        return Ok(n)
    }

    let ctlFileGeneric = match task.GetFile(rmCtrlFD) {
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

    let mut ioctlParamsTmp = ioctlParams;
    ioctlParamsTmp.SetRMCtrlFD(ctlFile.fd);
    let n = UvmIoctlInvoke(ui, Some(&ioctlParamsTmp))?;
    let mut outIoctlParams = ioctlParamsTmp;
    outIoctlParams.SetRMCtrlFD(rmCtrlFD);

    task.CopyOutObj(&outIoctlParams, ui.ioctlParamsAddr)?;
    return Ok(n)
} 