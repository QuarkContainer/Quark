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
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::sync::Weak;
use alloc::vec::Vec;
use core::any::Any;
use core::ops::Deref;

use super::super::super::super::addr::*;
use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::mem::areaset::*;
use super::super::super::super::range::*;
use super::super::super::fd::*;
use super::super::super::guestfdnotifier::*;
use super::super::super::kernel::time::*;
use super::super::super::kernel::waiter::qlock::*;
use super::super::super::kernel::waiter::queue::*;
use super::super::super::memmgr::mapping_set::*;
use super::super::super::memmgr::mm::*;
pub use super::super::super::memmgr::vma::MMappable;
use super::super::super::memmgr::*;
use super::super::super::socket::unix::transport::unix::*;
use super::super::super::task::*;
use super::super::super::Kernel::HostSpace;
use super::super::super::IOURING;
use super::super::super::SHARESPACE;
use super::super::attr::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::filesystems::*;
use super::super::flags::*;
use super::super::inode::*;
use super::fs::*;
use super::hostfileop::*;
use super::util::*;
use super::*;

#[cfg(feature = "cc")]
use crate::qlib::kernel::Kernel::is_cc_enabled;

pub struct MappableInternal {
    //addr mapping from file offset to physical address
    pub f2pmap: BTreeMap<u64, u64>,

    // mappings tracks mappings of the cached file object into
    // memmap.MappingSpaces.
    pub mapping: AreaSet<MappingsOfRange>,

    // file offset to ref count mapping
    pub chunkrefs: BTreeMap<u64, i32>,

    //addr mapping for shared pages from shared memory to private memory
    //need to write back to shared pages when munmap in cc. the value is (private memory,fileOffset,writeable)
    #[cfg(feature = "cc")]
    pub p2pmap: BTreeMap<u64, (u64, u64, bool)>,
}

impl MappableInternal {
    #[cfg(feature = "cc")]
    pub fn WritebackPage(&self, phyAddr: u64) {
        match self.p2pmap.get(&phyAddr) {
            None => (),
            Some((newAddr, _, writeable)) => unsafe {
                if *writeable {
                    core::ptr::copy_nonoverlapping(
                        *newAddr as *const u8,
                        phyAddr as *mut u8,
                        PAGE_SIZE as usize,
                    );
                }
            },
        }
    }

    #[cfg(feature = "cc")]
    pub fn SyncWrite(&self, offset: i64, srcs: &[IoVec]) {
        for i in 0..srcs.len() {
            let start_page = offset as u64 & !PAGE_MASK;
            let start_offset = offset as u64 & PAGE_MASK;
            let end_page = (offset as usize + srcs[i].len - 1) as u64 & !PAGE_MASK;
            let end_offset = (offset as usize + srcs[i].len - 1) as u64 & PAGE_MASK;
            for (_, (newAddr, fileoffset, _)) in &self.p2pmap {
                if *fileoffset >= start_page && *fileoffset <= end_page {
                    let start = if *fileoffset == start_page {
                        start_offset
                    } else {
                        0
                    };
                    let end = if *fileoffset == end_page {
                        end_offset
                    } else {
                        PAGE_MASK
                    };
                    let src_offset = srcs[i].start + *fileoffset + start - offset as u64;
                    let count = end - start + 1;
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            (src_offset) as *const u8,
                            (*newAddr + start) as *mut u8,
                            count as usize,
                        );
                    }
                }
            }
        }
    }

    #[cfg(feature = "cc")]
    pub fn WritebackAllPages(&self) {
        for (phyAddr, (newAddr, _, writeable)) in &self.p2pmap {
            if *writeable {
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        *newAddr as *const u8,
                        *phyAddr as *mut u8,
                        PAGE_SIZE as usize,
                    );
                }
            }
        }
    }

    pub fn Clear(&mut self) {
        #[cfg(feature = "cc")]
        if is_cc_enabled() {
            for (phyAddr, (newAddr, _, writeable)) in &self.p2pmap {
                if *writeable {
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            *newAddr as *const u8,
                            *phyAddr as *mut u8,
                            PAGE_SIZE as usize,
                        );
                    }
                }
            }
        }

        for (_offset, phyAddr) in &self.f2pmap {
            //error!("MappableInternal clean phyAddr {:x?}/{:x?}", phyAddr, offset);
            HostSpace::MUnmap(*phyAddr, CHUNK_SIZE);
        }
    }

    pub fn IncrRefOn(&mut self, fr: &Range) {
        let mut chunkStart = fr.Start() & !CHUNK_MASK;
        while chunkStart < fr.End() {
            let mut refs = match self.chunkrefs.get(&chunkStart) {
                None => 0,
                Some(v) => *v,
            };

            refs += PagesInChunk(fr, chunkStart);
            self.chunkrefs.insert(chunkStart, refs);

            chunkStart += CHUNK_SIZE;
        }
    }

    pub fn DecrRefOn(&mut self, fr: &Range) {
        let mut chunkStart = fr.Start() & !CHUNK_MASK;
        while chunkStart < fr.End() {
            let mut refs = match self.chunkrefs.get(&chunkStart) {
                None => 0,
                Some(v) => *v,
            };

            refs -= PagesInChunk(fr, chunkStart);

            if refs == 0 {
                let phyAddr = match self.f2pmap.get(&chunkStart) {
                    None => {
                        //info!("HostMappable::RemovePhysicalMapping fileOffset {:x} doesn't exist", chunkStart);
                        //for kernel pma registation, tehre is no phymapping,
                        chunkStart += CHUNK_SIZE;
                        continue;
                    }
                    Some(offset) => *offset,
                };

                #[cfg(feature = "cc")]
                if is_cc_enabled(){
                    for i in 0..CHUNK_SIZE / PAGE_SIZE {
                        match self.p2pmap.remove(&(phyAddr + i * PAGE_SIZE)) {
                            None => (),
                            Some((newAddr, _, writeable)) => unsafe {
                                if writeable {
                                    core::ptr::copy_nonoverlapping(
                                        newAddr as *const u8,
                                        (phyAddr + i * PAGE_SIZE) as *mut u8,
                                        PAGE_SIZE as usize,
                                    );
                                }
                            },
                        }
                    }
                }

                HostSpace::MUnmap(phyAddr, CHUNK_SIZE);

                /*error!("DecrRefOn 1 {:x}/{:x}", phyAddr,  phyAddr + CHUNK_SIZE);
                let mut curr = phyAddr;
                while curr < phyAddr + CHUNK_SIZE {
                    Clflush(curr);
                    error!("DecrRefOn 1.1 {:x}", curr);
                    curr += 64;
                }
                error!("DecrRefOn 2");*/
                self.f2pmap.remove(&chunkStart);
            } else if refs > 0 {
                self.chunkrefs.insert(chunkStart, refs);
            } else {
                panic!(
                    "Mappable::DecrRefOn get negative refs {}, pages is {}, fr is {:x?}",
                    refs,
                    PagesInChunk(fr, chunkStart),
                    fr
                )
            }

            chunkStart += CHUNK_SIZE;
        }
    }
}

pub fn PagesInChunk(r: &Range, chunkStart: u64) -> i32 {
    assert!(
        chunkStart & CHUNK_MASK == 0,
        "chunkStart is {:x}",
        chunkStart
    );
    let chunkRange = Range::New(chunkStart, CHUNK_SIZE);
    return (r.Intersect(&chunkRange).Len() / MemoryDef::PAGE_SIZE) as i32;
}

impl Default for MappableInternal {
    fn default() -> Self {
        return Self {
            //lock: QLock::default(),
            f2pmap: BTreeMap::new(),
            mapping: AreaSet::New(0, core::u64::MAX),
            chunkrefs: BTreeMap::new(),
            #[cfg(feature = "cc")]
            p2pmap: BTreeMap::new(),
        };
    }
}

#[derive(Default, Clone)]
pub struct Mappable(Arc<QMutex<MappableInternal>>);

impl Deref for Mappable {
    type Target = Arc<QMutex<MappableInternal>>;

    fn deref(&self) -> &Arc<QMutex<MappableInternal>> {
        &self.0
    }
}

pub struct HostInodeOpIntern {
    pub mops: Arc<QMutex<MountSourceOperations>>,
    //this should be SuperOperations
    pub HostFd: i32,
    pub WouldBlock: bool,
    pub Writeable: bool,
    pub skiprw: bool,
    pub sattr: StableAttr,
    pub queue: Queue,
    pub errorcode: i64,

    // this Size is only used for mmap len check. It might not be consistent with host file size.
    // when write to the file, the size is not updated.
    // todo: fix this
    pub size: i64,

    pub mappable: Option<Mappable>,
    pub bufWriteLock: QAsyncLock,
    pub hasMappable: bool,

    pub isMemfd: bool,
}

impl Default for HostInodeOpIntern {
    fn default() -> Self {
        return Self {
            mops: Arc::new(QMutex::new(SimpleMountSourceOperations::default())),
            HostFd: -1,
            WouldBlock: false,
            Writeable: false,
            skiprw: true,
            sattr: StableAttr::default(),
            queue: Queue::default(),
            errorcode: 0,
            mappable: None,
            size: 0,
            bufWriteLock: QAsyncLock::default(),
            hasMappable: false,
            isMemfd: false,
        };
    }
}

impl Drop for HostInodeOpIntern {
    fn drop(&mut self) {
        if self.HostFd == -1 {
            //default fd
            return;
        }

        let task = Task::Current();

        let _l = if self.BufWriteEnable() {
            Some(self.BufWriteLock().Lock(task))
        } else {
            None
        };

        if SHARESPACE.config.read().MmapRead {
            match self.mappable.take() {
                None => (),
                Some(mapable) => {
                    mapable.lock().Clear();
                }
            }
        }

        HostSpace::Close(self.HostFd);
    }
}

impl HostInodeOpIntern {
    pub fn New(
        mops: &Arc<QMutex<MountSourceOperations>>,
        fd: i32,
        wouldBlock: bool,
        fstat: &LibcStat,
        writeable: bool,
        skiprw: bool,
        isMemfd: bool,
    ) -> Self {
        let mut ret = Self {
            mops: mops.clone(),
            HostFd: fd,
            WouldBlock: wouldBlock,
            Writeable: writeable,
            skiprw: skiprw,
            sattr: fstat.StableAttr(),
            queue: Queue::default(),
            errorcode: 0,
            mappable: None,
            size: fstat.st_size,
            bufWriteLock: QAsyncLock::default(),
            hasMappable: false,
            isMemfd: isMemfd,
        };

        if ret.CanMap() {
            ret.mappable = Some(Mappable::default());
        }

        return ret;
    }

    pub fn SkipRw(&self) -> bool {
        return self.skiprw;
    }

    pub fn TryOpenWrite(&mut self, dirfd: i32, name: u64) -> Result<()> {
        if !self.skiprw {
            if !self.Writeable {
                return Err(Error::SysError(SysErr::EPERM));
            }

            return Ok(())
        }
        let ret = HostSpace::TryOpenWrite(dirfd, self.HostFd, name);
        self.skiprw = false;
        self.Writeable = true;
        if ret < 0 {
            // assume the user has no write permission
            return Err(Error::SysError(SysErr::EPERM));
        }

        return Ok(())
    }

    /*********************************start of mappable****************************************************************/
    fn Mappable(&mut self) -> Mappable {
        return self.mappable.clone().unwrap();
    }

    //add mapping between physical address and file offset, offset must be hugepage aligned
    pub fn AddPhyMapping(&mut self, phyAddr: u64, offset: u64) {
        assert!(
            offset & CHUNK_MASK == 0,
            "HostMappable::AddPhysicalMap offset should be hugepage aligned"
        );

        let mappable = self.Mappable();
        let mut mappableLock = mappable.lock();
        mappableLock.f2pmap.insert(offset, phyAddr);
    }

    pub fn IncrRefOn(&mut self, fr: &Range) {
        let mappable = self.Mappable();
        let mut mappableLock = mappable.lock();
        return mappableLock.IncrRefOn(fr);
    }

    pub fn DecrRefOn(&mut self, fr: &Range) {
        let mappable = self.Mappable();
        let mut mappableLock = mappable.lock();
        return mappableLock.DecrRefOn(fr);
    }

    //get phyaddress ranges for the file range
    pub fn MapInternal(&mut self, task: &Task, fr: &Range) -> Result<Vec<IoVec>> {
        let mut chunkStart = fr.Start() & !HUGE_PAGE_MASK;

        self.Fill(task, chunkStart, fr.End())?;
        let mut res = Vec::new();

        let mappable = self.Mappable();
        let mappableLock = mappable.lock();

        while chunkStart < fr.End() {
            let phyAddr = mappableLock.f2pmap.get(&chunkStart).unwrap();
            let mut startOffset = 0;
            if chunkStart < fr.Start() {
                startOffset = fr.Start() - chunkStart;
            }

            let mut endOff = CHUNK_SIZE;
            if chunkStart + CHUNK_SIZE > fr.End() {
                endOff = fr.End() - chunkStart;
            }

            res.push(IoVec::NewFromAddr(
                phyAddr + startOffset,
                (endOff - startOffset) as usize,
            ));
            chunkStart += CHUNK_SIZE;
        }

        return Ok(res);
    }

    // map one page from file offsetFile to phyAddr
    pub fn MapFilePage(&mut self, task: &Task, fileOffset: u64) -> Result<u64> {
        let filesize = self.size as u64;
        if filesize <= fileOffset {
            return Err(Error::FileMapError);
        }

        let chunkStart = fileOffset & !HUGE_PAGE_MASK;
        self.Fill(task, chunkStart, fileOffset + PAGE_SIZE)?;

        let mappable = self.Mappable();
        let mappableLock = mappable.lock();

        let phyAddr = mappableLock.f2pmap.get(&chunkStart).unwrap();
        return Ok(phyAddr + (fileOffset - chunkStart));
    }

    #[cfg(feature = "cc")]
    pub fn MapSharedPage(&mut self, phyAddr: u64, newAddr: u64, offset: u64, writeable: bool) {
        let mappable = self.Mappable();
        let mut mappableLock = mappable.lock();
        mappableLock
            .p2pmap
            .insert(phyAddr, (newAddr, offset, writeable));
    }

    //fill the holes for the file range by mmap
    //start must be Hugepage aligned
    fn Fill(&mut self, _task: &Task, start: u64, end: u64) -> Result<()> {
        let mappable = self.Mappable();

        let mut start = start;

        let mut holes = Vec::new();

        while start < end {
            match mappable.lock().f2pmap.get(&start) {
                None => holes.push(start),
                Some(_) => (),
            }

            start += HUGE_PAGE_SIZE;
        }

        for offset in holes {
            self.MMapChunk(offset)?;
        }
        return Ok(());
    }

    pub fn MMapChunk(&mut self, offset: u64) -> Result<u64> {
        let writeable = self.Writeable;

        let prot = if writeable {
            (MmapProt::PROT_WRITE | MmapProt::PROT_READ) as i32
        } else {
            MmapProt::PROT_READ as i32
        };

        let phyAddr = self.MapFileChunk(offset, prot)?;
        self.AddPhyMapping(phyAddr, offset);
        return Ok(phyAddr);
    }

    pub fn MapFileChunk(&self, offset: u64, prot: i32) -> Result<u64> {
        assert!(
            offset & CHUNK_MASK == 0,
            "MapFile offset must be chunk aligned"
        );

        let fd = self.HostFd();
        let ret = HostSpace::MMapFile(CHUNK_SIZE, fd, offset, prot);

        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        let phyAddr = ret as u64;

        return Ok(phyAddr);
    }

    /*********************************end of mappable****************************************************************/

    pub fn SetMaskedAttributes(&self, mask: &AttrMask, attr: &UnstableAttr) -> Result<()> {
        return SetMaskedAttributes(self.HostFd, mask, attr);
    }

    pub fn Sync(&self) -> Result<()> {
        let ret = Fsync(self.HostFd);
        if ret < 0 {
            return Err(Error::SysError(-ret));
        }

        return Ok(());
    }

    pub fn HostFd(&self) -> i32 {
        return self.HostFd;
    }

    pub fn BufWriteEnable(&self) -> bool {
        return SHARESPACE.config.read().FileBufWrite && !self.hasMappable;
    }

    pub fn BufWriteLock(&self) -> QAsyncLock {
        return self.bufWriteLock.clone();
    }

    pub fn WouldBlock(&self) -> bool {
        return self.WouldBlock;
    }

    pub fn StableAttr(&self) -> StableAttr {
        return self.sattr;
    }

    pub fn CanMap(&self) -> bool {
        return self.sattr.Type == InodeType::RegularFile
            || self.sattr.Type == InodeType::SpecialFile;
    }

    pub fn InodeType(&self) -> InodeType {
        return self.sattr.Type;
    }
}

#[derive(Clone)]
pub struct HostInodeOpWeak(pub Weak<QMutex<HostInodeOpIntern>>);

impl HostInodeOpWeak {
    pub fn Upgrade(&self) -> Option<HostInodeOp> {
        let f = match self.0.upgrade() {
            None => return None,
            Some(f) => f,
        };

        return Some(HostInodeOp(f));
    }
}

#[derive(Clone)]
pub struct HostInodeOp(pub Arc<QMutex<HostInodeOpIntern>>);

impl PartialEq for HostInodeOp {
    fn eq(&self, other: &Self) -> bool {
        return Arc::ptr_eq(&self.0, &other.0);
    }
}

impl Eq for HostInodeOp {}

impl Default for HostInodeOp {
    fn default() -> Self {
        return Self(Arc::new(QMutex::new(HostInodeOpIntern::default())));
    }
}

impl Deref for HostInodeOp {
    type Target = Arc<QMutex<HostInodeOpIntern>>;

    fn deref(&self) -> &Arc<QMutex<HostInodeOpIntern>> {
        &self.0
    }
}

impl HostInodeOp {
    pub fn New(
        mops: &Arc<QMutex<MountSourceOperations>>,
        fd: i32,
        wouldBlock: bool,
        fstat: &LibcStat,
        writeable: bool,
        skiprw: bool,
        isMemfd: bool,
    ) -> Self {
        let intern = Arc::new(QMutex::new(HostInodeOpIntern::New(
            mops, fd, wouldBlock, fstat, writeable, skiprw, isMemfd,
        )));

        let ret = Self(intern);
        SetWaitInfo(fd, ret.lock().queue.clone());
        return ret;
    }

    pub fn NewMemfdIops(len: i64) -> Result<Self> {
        let fd = HostSpace::CreateMemfd(len, 0) as i32;
        if fd < 0 {
            return Err(Error::SysError(-fd as i32));
        }

        let mut fstat = LibcStat::default();

        let ret = Fstat(fd, &mut fstat) as i32;
        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        let msrc = MountSource::NewHostMountSource(
            &"/".to_string(),
            &ROOT_OWNER,
            &WhitelistFileSystem::New(),
            &MountSourceFlags::default(),
            false,
        );
        let intern = Arc::new(QMutex::new(HostInodeOpIntern::New(
            &msrc.MountSourceOperations.clone(),
            fd,
            false,
            &fstat,
            true,
            false,
            true,
        )));

        let ret = Self(intern);
        return Ok(ret);
    }

    pub fn SyncFs(&self) -> Result<()> {
        let fd = self.HostFd();

        let ret = HostSpace::SyncFs(fd);
        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        return Ok(());
    }

    pub fn SyncFileRange(&self, offset: i64, nbytes: i64, flags: u32) -> Result<()> {
        let fd = self.HostFd();

        let ret = HostSpace::SyncFileRange(fd, offset, nbytes, flags);
        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        return Ok(());
    }

    pub fn Downgrade(&self) -> HostInodeOpWeak {
        return HostInodeOpWeak(Arc::downgrade(&self.0));
    }

    pub fn HostFd(&self) -> i32 {
        return self.lock().HostFd;
    }

    pub fn UpdateMaxLen(&self, size: i64) {
        let mut h = self.lock();
        if h.size < size {
            h.size = size;
        }
    }

    pub fn StableAttr(&self) -> StableAttr {
        return self.lock().sattr;
    }

    pub fn Queue(&self) -> Queue {
        return self.lock().queue.clone();
    }

    pub fn GetHostFileOp(&self, _task: &Task) -> HostFileOp {
        let hostFileOp = HostFileOp {
            InodeOp: self.clone(),
            DirCursor: Arc::new(QMutex::new("".to_string())),
            //Buf: HostFileBuf::None,
        };
        return hostFileOp;
    }

    // return (st_size, st_blocks)
    pub fn Size(&self) -> Result<(i64, i64)> {
        let mut s: LibcStat = Default::default();
        let hostfd = self.lock().HostFd;
        let ret = Fstat(hostfd, &mut s) as i32;
        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        return Ok((s.st_size, s.st_blocks));
    }

    /*********************************start of fileoperation *******************/

    pub fn BufWriteEnable(&self) -> bool {
        return self.lock().BufWriteEnable();
    }

    // ReadEndOffset returns an exclusive end offset for a read operation
    // so that the read does not overflow an int64 nor size.
    //
    // Parameters:
    // - offset: the starting offset of the read.
    // - length: the number of bytes to read.
    // - size:   the size of the file.
    //
    // Postconditions: The returned offset is >= offset.
    pub fn ReadEndOffset(offset: i64, len: i64, size: i64) -> i64 {
        if offset >= size {
            return offset;
        }

        let mut end = offset + len;
        if end < offset || end > size {
            end = size;
        }

        return end;
    }

    pub fn ReadAt(
        &self,
        task: &Task,
        _f: &File,
        dsts: &mut [IoVec],
        offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        let hostIops = self.clone();

        let size = IoVec::NumBytes(dsts);
        let size = if size >= MemoryDef::HUGE_PAGE_SIZE as usize {
            MemoryDef::HUGE_PAGE_SIZE as usize
        } else {
            size
        };
        let buf = DataBuff::New(size);

        let iovs = buf.Iovs(size);
        let inodeType = self.InodeType();

        if inodeType != InodeType::RegularFile && inodeType != InodeType::CharacterDevice {
            let ret = IORead(hostIops.HostFd(), &iovs)?;

            // todo: handle partial write
            task.CopyDataOutToIovs(&buf.buf[0..ret as usize], dsts, false)?;
            return Ok(ret as i64);
        } else {
            if inodeType == InodeType::RegularFile && SHARESPACE.config.read().MmapRead {
                let mut intern = self.lock();
                if offset > intern.size {
                    return Ok(0);
                }

                let end = Self::ReadEndOffset(offset, size as i64, intern.size);
                if end == offset {
                    return Ok(0);
                }

                let srcIovs =
                    intern.MapInternal(task, &Range::New(offset as u64, (end - offset) as u64))?;
                let count = task.CopyIovsOutToIovs(&srcIovs, dsts, true)?;

                return Ok(count as i64);
            }

            if SHARESPACE.config.read().UringIO {
                if self.BufWriteEnable() {
                    // try to gain the lock once, release immediately
                    self.BufWriteLock().Lock(task);
                }

                let ret = IOURING.Read(
                    task,
                    hostIops.HostFd(),
                    buf.Ptr(),
                    buf.Len() as u32,
                    offset as i64,
                );

                if ret < 0 {
                    if ret as i32 != -SysErr::EINVAL {
                        return Err(Error::SysError(-ret as i32));
                    }
                } else if ret >= 0 {
                    task.CopyDataOutToIovs(&buf.buf[0..ret as usize], dsts, true)?;
                    return Ok(ret as i64);
                }

                // if ret == SysErr::EINVAL, the file might be tmpfs file, io_uring can't handle this
                // fallback to normal case
                // todo: handle tmp file elegant
            }

            let offset = if inodeType == InodeType::CharacterDevice {
                let ret = IOTTYRead(hostIops.HostFd(), &iovs)?;
                task.CopyDataOutToIovs(&buf.buf[0..ret as usize], dsts, false)?;
                return Ok(ret as i64);
            } else {
                offset
            };

            let ret = IOReadAt(hostIops.HostFd(), &iovs, offset as u64)?;
            task.CopyDataOutToIovs(&buf.buf[0..ret as usize], dsts, true)?;
            return Ok(ret as i64);
        }
    }

    pub fn BufWriteLock(&self) -> QAsyncLock {
        return self.lock().BufWriteLock();
    }

    pub fn WriteAt(
        &self,
        task: &Task,
        _f: &File,
        srcs: &[IoVec],
        offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        let hostIops: HostInodeOp = self.clone();
        if !hostIops.lock().Writeable {
            error!("writeat {}", hostIops.lock().HostFd);
        }
        
        assert!(hostIops.lock().Writeable);

        let size = IoVec::NumBytes(srcs);
        if size == 0 {
            return Ok(0);
        }

        let size = if size >= MemoryDef::HUGE_PAGE_SIZE as usize {
            MemoryDef::HUGE_PAGE_SIZE as usize
        } else {
            size
        };

        #[cfg(feature = "cc")]
        if is_cc_enabled() {
            if let Some(mappable) = self.lock().mappable.clone() {
                mappable.lock().SyncWrite(offset, srcs);
            }
        }

        let mut buf = DataBuff::New(size);
        let len = task.CopyDataInFromIovs(&mut buf.buf, srcs, true)?;
        let iovs = buf.Iovs(len);

        let inodeType = self.InodeType();

        if inodeType != InodeType::RegularFile && inodeType != InodeType::CharacterDevice {
            let ret = IOWrite(hostIops.HostFd(), &iovs)?;
            return Ok(ret as i64);
        } else {
            let offset = if inodeType == InodeType::CharacterDevice {
                -1
            } else {
                offset
            };

            if SHARESPACE.config.read().UringIO {
                let ret = if self.BufWriteEnable() {
                    let lock = self.BufWriteLock().Lock(task);
                    let count = IOURING.BufFileWrite(hostIops.HostFd(), buf, offset, lock);
                    count
                } else {
                    IOURING.Write(
                        task,
                        hostIops.HostFd(),
                        buf.Ptr(),
                        buf.Len() as u32,
                        offset as i64,
                    )
                };

                if ret < 0 {
                    if ret as i32 != -SysErr::EINVAL {
                        return Err(Error::SysError(-ret as i32));
                    }
                } else if ret >= 0 {
                    if inodeType != InodeType::CharacterDevice {
                        hostIops.UpdateMaxLen(offset + ret);
                    }

                    return Ok(ret as i64);
                }

                // if ret == SysErr::EINVAL, the file might be tmpfs file, io_uring can't handle this
                // fallback to normal case
                // todo: handle tmp file elegant
            }

            match IOWriteAt(hostIops.HostFd(), &iovs, offset as u64) {
                Err(e) => return Err(e),
                Ok(ret) => {
                    hostIops.UpdateMaxLen(offset + ret);
                    return Ok(ret);
                }
            }
        }
    }

    pub fn Append(&self, task: &Task, f: &File, srcs: &[IoVec]) -> Result<(i64, i64)> {
        let hostIops = self.clone();

        let inodeType = hostIops.InodeType();
        if inodeType == InodeType::RegularFile || inodeType == InodeType::SpecialFile {
            let size = IoVec::NumBytes(srcs);
            /*let size = if size >= MemoryDef::HUGE_PAGE_SIZE as usize {
                MemoryDef::HUGE_PAGE_SIZE as usize
            } else {
                size
            };*/
            let mut buf = DataBuff::New(size);

            let len = task.CopyDataInFromIovs(&mut buf.buf, srcs, true)?;
            let iovs = buf.Iovs(len);

            let iovsAddr = &iovs[0] as *const _ as u64;
            let iovcnt = 1;

            let (count, len) = HostSpace::IOAppend(hostIops.HostFd(), iovsAddr, iovcnt);
            if count < 0 {
                return Err(Error::SysError(-count as i32));
            }

            if inodeType == InodeType::RegularFile {
                hostIops.UpdateMaxLen(len);
            }

            return Ok((count, len));
        } else {
            let n = self.WriteAt(task, f, srcs, 0, true)?;
            return Ok((n, 0));
        }
    }

    pub fn Fsync(
        &self,
        task: &Task,
        _f: &File,
        _start: i64,
        _end: i64,
        syncType: SyncType,
    ) -> Result<()> {
        if self.lock().isMemfd {
            return Ok(());
        }

        let fd = self.HostFd();
        let datasync = if syncType == SyncType::SyncData {
            true
        } else {
            false
        };

        #[cfg(feature = "cc")]
        if is_cc_enabled(){
            self.lock().Mappable().lock().WritebackAllPages();
        }

        let ret = if SHARESPACE.config.read().UringIO && self.InodeType() == InodeType::RegularFile
        {
            if self.BufWriteEnable() {
                // try to gain the lock once, release immediately
                self.BufWriteLock().Lock(task);
            }

            IOURING.Fsync(task, fd, datasync)
        } else {
            if self.BufWriteEnable() {
                // try to gain the lock once, release immediately
                self.BufWriteLock().Lock(task);
            }

            if datasync {
                HostSpace::FDataSync(fd)
            } else {
                HostSpace::FSync(fd)
            }
        };

        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        return Ok(());
    }

    /*********************************start of mappable****************************************************************/

    //add mapping between file offset and the <MappingSpace(i.e. memorymanager), Virtual Address>
    pub fn AddMapping(
        &self,
        ms: &MemoryManager,
        ar: &Range,
        offset: u64,
        writeable: bool,
    ) -> Result<()> {
        self.lock().hasMappable = true;

        // todo: if there is bufwrite ongoing, should we wait for it?
        /*let _= if self.BufWriteEnable() {
            let task = Task::Current();
            error!("AddMapping 1");
            let lock = self.lock().bufWriteLock.Lock(task);
            error!("AddMapping 2");
            Some(lock)
        } else {
            None
        };*/

        let mappable = self.lock().Mappable();
        let mut mappableLock = mappable.lock();

        mappableLock.mapping.AddMapping(ms, ar, offset, writeable);
        mappableLock.IncrRefOn(&Range::New(offset, ar.Len()));
        return Ok(());
    }

    pub fn RemoveMapping(
        &self,
        ms: &MemoryManager,
        ar: &Range,
        offset: u64,
        writeable: bool,
    ) -> Result<()> {
        let mappable = self.lock().Mappable();
        let mut mappableLock = mappable.lock();

        mappableLock
            .mapping
            .RemoveMapping(ms, ar, offset, writeable);
        mappableLock.DecrRefOn(&Range::New(offset, ar.Len()));
        return Ok(());
    }

    pub fn CopyMapping(
        &self,
        ms: &MemoryManager,
        _srcAr: &Range,
        dstAR: &Range,
        offset: u64,
        writeable: bool,
    ) -> Result<()> {
        return self.AddMapping(ms, dstAR, offset, writeable);
    }

    pub fn FD(&self) -> i32 {
        let ret = self.lock().HostFd;
        return ret;
    }

    //get phyaddress ranges for the file range
    pub fn MapInternal(&self, task: &Task, fr: &Range) -> Result<Vec<IoVec>> {
        return self.lock().MapInternal(task, fr);
    }

    // map one page from file offsetFile to phyAddr
    pub fn MapFilePage(&self, task: &Task, fileOffset: u64) -> Result<u64> {
        return self.lock().MapFilePage(task, fileOffset);
    }

    #[cfg(feature = "cc")]
    pub fn MapSharedPage(&self, phyAddr: u64, newAddr: u64, offset: u64, writeable: bool) {
        self.lock()
            .MapSharedPage(phyAddr, newAddr, offset, writeable);
    }

    pub fn MSync(&self, fr: &Range, msyncType: MSyncType) -> Result<()> {
        let ranges = self.GetPhyRanges(fr);
        #[cfg(feature = "cc")]
        if is_cc_enabled(){
            self.WritebackRanges(&ranges)?;
        }
        for r in &ranges {
            let ret = HostSpace::MSync(r.Start(), r.Len() as usize, msyncType.MSyncFlags());
            if ret < 0 {
                return Err(Error::SysError(ret as i32));
            }
        }

        return Ok(());
    }

    pub fn MAdvise(&self, start: u64, len: u64, advise: i32) -> Result<()> {
        let ranges = self.GetPhyRanges(&Range::New(start, len));
        for r in &ranges {
            let ret = HostSpace::Madvise(r.Start(), r.Len() as usize, advise);
            if ret < 0 {
                return Err(Error::SysError(ret as i32));
            }
        }

        return Ok(());
    }

    pub fn Mlock(&self, start: u64, len: u64, mode: MLockMode) -> Result<()> {
        let ranges = self.GetPhyRanges(&Range::New(start, len));
        for r in &ranges {
            match mode {
                MLockMode::MlockNone => {
                    let ret = HostSpace::MUnlock(r.Start(), r.Len());
                    if ret < 0 {
                        return Err(Error::SysError(ret as i32));
                    }
                }
                MLockMode::MlockEager => {
                    let flags = 0;
                    let ret = HostSpace::Mlock2(r.Start(), r.Len(), flags);
                    if ret < 0 {
                        return Err(Error::SysError(ret as i32));
                    }
                }
                MLockMode::MlockLazy => {
                    let flags = MLOCK_ONFAULT;
                    let ret = HostSpace::Mlock2(r.Start(), r.Len(), flags);
                    if ret < 0 {
                        return Err(Error::SysError(ret as i32));
                    }
                }
            }
        }

        return Ok(());
    }

    pub fn GetPhyRanges(&self, fr: &Range) -> Vec<Range> {
        let mut chunkStart = fr.Start() & !HUGE_PAGE_MASK;
        let mut rs = Vec::new();

        let mappable = self.lock().Mappable();
        let mappableLock = mappable.lock();

        while chunkStart < fr.End() {
            match mappableLock.f2pmap.get(&chunkStart) {
                None => (),
                Some(phyAddr) => {
                    let fChunckRange = Range::New(chunkStart, HUGE_PAGE_SIZE);
                    let startOffset = if fChunckRange.Contains(fr.Start()) {
                        fr.Start() - fChunckRange.Start()
                    } else {
                        0
                    };

                    let len = if fChunckRange.Contains(fr.End()) {
                        fr.End() - fChunckRange.Start()
                    } else {
                        HUGE_PAGE_SIZE
                    };
                    rs.push(Range::New(*phyAddr + startOffset, len));
                }
            }

            chunkStart += CHUNK_SIZE;
        }

        return rs;
    }

    #[cfg(feature = "cc")]
    pub fn WritebackRanges(&self, ranges: &Vec<Range>) -> Result<()> {
        let mappable = self.lock().Mappable();
        let mappableLock = mappable.lock();

        for range in ranges {
            assert!(range.start & PAGE_MASK == 0);
            let mut PageStart = range.start;
            while PageStart < range.End() {
                mappableLock.WritebackPage(PageStart);
                PageStart += PAGE_SIZE;
            }
        }
        return Ok(());
    }

    /*********************************end of mappable****************************************************************/
}

impl InodeOperations for HostInodeOp {
    fn as_any(&self) -> &Any {
        self
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::HostInodeOp;
    }

    fn InodeType(&self) -> InodeType {
        return self.lock().sattr.Type;
    }

    fn InodeFileType(&self) -> InodeFileType {
        return InodeFileType::Host;
    }

    fn WouldBlock(&self) -> bool {
        return self.lock().WouldBlock;
    }

    fn Lookup(&self, _task: &Task, _parent: &Inode, _name: &str) -> Result<Dirent> {
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
        task: &Task,
        dir: &mut Inode,
        oldParent: &Inode,
        oldname: &str,
        newParent: &Inode,
        newname: &str,
        replacement: bool,
    ) -> Result<()> {
        return Rename(
            task,
            dir,
            oldParent,
            oldname,
            newParent,
            newname,
            replacement,
        );
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
        task: &Task,
        _dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        let fops = self.GetHostFileOp(task);

        let inode = dirent.Inode();
        let wouldBlock = inode.lock().InodeOp.WouldBlock();

        return Ok(File::NewHostFile(dirent, &flags, fops.into(), wouldBlock));
    }

    fn UnstableAttr(&self, task: &Task) -> Result<UnstableAttr> {
        if self.BufWriteEnable() {
            // try to gain the lock once, release immediately
            self.BufWriteLock().Lock(task);
        }

        let mops = self.lock().mops.clone();
        let fd = self.HostFd();

        return UnstableAttr(fd, task, &mops);
    }

    //fn StableAttr(&self) -> &StableAttr;
    fn Getxattr(&self, _dir: &Inode, name: &str, _size: usize) -> Result<Vec<u8>> {
        return Getxattr(self.HostFd(), name);
    }

    fn Setxattr(&self, _dir: &mut Inode, name: &str, value: &[u8], flags: u32) -> Result<()> {
        return Setxattr(self.HostFd(), name, value, flags);
    }

    fn Listxattr(&self, _dir: &Inode, _size: usize) -> Result<Vec<String>> {
        return Listxattr(self.HostFd());
    }

    fn Removexattr(&self, _dir: &Inode, name: &str) -> Result<()> {
        return Removexattr(self.HostFd(), name);
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return ContextCanAccessFile(task, inode, reqPerms);
    }

    fn SetPermissions(&self, _task: &Task, _dir: &mut Inode, f: FilePermissions) -> bool {
        return Fchmod(self.HostFd(), f.LinuxMode()) == 0;
    }

    fn SetOwner(&self, _task: &Task, _dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        let ret = FChown(self.HostFd(), owner.UID.0, owner.GID.0);

        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        };

        return Ok(());
    }

    fn SetTimestamps(&self, _task: &Task, _dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        return SetTimestamps(self.HostFd(), ts);
    }

    fn Truncate(&self, task: &Task, _dir: &mut Inode, size: i64) -> Result<()> {
        let uattr = self.UnstableAttr(task)?;
        let oldSize = uattr.Size;
        assert!(oldSize == self.lock().size);
        if size == oldSize {
            return Ok(());
        }

        if self.lock().CanMap() {
            if size < oldSize {
                let mappable = self.Mappable()?.HostIops().unwrap().lock().Mappable();
                let ranges = mappable.lock().mapping.InvalidateRanges(
                    task,
                    &Range::New(size as u64, oldSize as u64 - size as u64),
                    true,
                );
                for r in &ranges {
                    r.invalidate(task, true);
                }
            }
        }

        let ret = Ftruncate(self.HostFd(), size);
        
        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        self.lock().size = size;

        return Ok(());
    }

    fn Allocate(&self, task: &Task, _dir: &mut Inode, offset: i64, length: i64) -> Result<()> {
        let ret = Fallocate(self.HostFd(), 0, offset, length);

        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }

        let uattr = self.UnstableAttr(task)?;
        self.lock().size = uattr.Size;

        return Ok(());
    }

    fn ReadLink(&self, _task: &Task, _dir: &Inode) -> Result<String> {
        return ReadLinkAt(self.HostFd(), &"".to_string());
    }

    fn GetLink(&self, _task: &Task, dir: &Inode) -> Result<Dirent> {
        if !dir.StableAttr().IsSymlink() {
            return Err(Error::SysError(SysErr::ENOLINK));
        }

        return Err(Error::ErrResolveViaReadlink);
    }

    fn AddLink(&self, _task: &Task) {
        //return Err(Error::None)
    }

    fn DropLink(&self, _task: &Task) {
        //return Err(Error::None)
    }

    fn IsVirtual(&self) -> bool {
        false
    }

    fn Sync(&self) -> Result<()> {
        return self.lock().Sync();
    }

    fn StatFS(&self, _task: &Task) -> Result<FsInfo> {
        return StatFS(self.HostFd());
    }

    fn Mappable(&self) -> Result<MMappable> {
        let inodeType = self.lock().InodeType();
        if inodeType == InodeType::RegularFile {
            return Ok(MMappable::FromHostIops(self.clone()));
        } else {
            return Err(Error::SysError(SysErr::ENODEV));
        }
    }
}
