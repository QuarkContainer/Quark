// Copyright (c) 2021 Quark Container Authors
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

use alloc::sync::Arc;
use alloc::sync::Weak;
use spin::RwLock;
use spin::Mutex;
use core::ops::Deref;
use lazy_static::lazy_static;
use core::sync::atomic::AtomicU64;
use core::sync::atomic;
use alloc::string::String;
use alloc::string::ToString;
use alloc::slice;
use x86_64::structures::paging::PageTableFlags;
use alloc::vec::Vec;
use super::super::arch::x86_64::context::*;

use super::super::PAGE_MGR;
use super::super::KERNEL_PAGETABLE;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::range::*;
use super::super::qlib::addr::*;
use super::super::qlib::stack::*;
use super::super::qlib::mem::seq::*;
use super::super::task::*;
use super::super::qlib::mem::stackvec::*;
use super::super::qlib::pagetable::*;
use super::super::qlib::limits::*;
use super::super::qlib::perf_tunning::*;
use super::super::kernel::aio::aio_context::*;
use super::super::fs::dirent::*;
use super::super::mm::*;
use super::super::qlib::mem::areaset::*;
use super::arch::*;
use super::vma::*;
use super::metadata::*;
use super::syscalls::*;
use super::*;

pub struct MMMapping {
    pub vmas: AreaSet<VMA>,

    // brk is the mm's brk, which is manipulated using the brk(2) system call.
    // The brk is initially set up by the loader which maps an executable
    // binary into the mm.
    pub brkInfo: BrkInfo,

    // usageAS is vmas.Span(), cached to accelerate RLIMIT_AS checks.
    pub usageAS: u64,

    // lockedAS is the combined size in bytes of all vmas with vma.mlockMode !=
    // memmap.MLockNone.
    pub lockedAS: u64,

    // New VMAs created by MMap use whichever of memmap.MMapOpts.MLockMode or
    // defMLockMode is greater.
    pub defMLockMode: MLockMode,
}

impl Default for MMMapping {
    fn default() -> Self {
        let vmas = AreaSet::New(0, MemoryDef::LOWER_TOP);
        let mm = Self {
            vmas: vmas,
            brkInfo: BrkInfo::default(),
            usageAS: 0,
            lockedAS: 0,
            defMLockMode: MLockMode::MlockNone,
        };
        return mm;
    }
}

#[derive(Default)]
pub struct MMMetadata {
    // argv is the application argv. This is set up by the loader and may be
    // modified by prctl(PR_SET_MM_ARG_START/PR_SET_MM_ARG_END). No
    // requirements apply to argv; we do not require that argv.WellFormed().
    pub argv: Range,

    // envv is the application envv. This is set up by the loader and may be
    // modified by prctl(PR_SET_MM_ENV_START/PR_SET_MM_ENV_END). No
    // requirements apply to envv; we do not require that envv.WellFormed().
    pub envv: Range,

    // auxv is the ELF's auxiliary vector.
    pub auxv: Vec<AuxEntry>,

    // executable is the executable for this MemoryManager. If executable
    // is not nil, it holds a reference on the Dirent.
    pub executable: Option<Dirent>,

    // dumpability describes if and how this MemoryManager may be dumped to
    // userspace.
    //
    pub dumpability: Dumpability,
}

#[derive(Default)]
pub struct MMPagetable {
    pub pt: PageTables,

    pub sharedLoadsOffset: u64,

    // curRSS is pmas.Span(), cached to accelerate updates to maxRSS. It is
    // reported as the MemoryManager's RSS.
    //
    // maxRSS should be modified only via insertRSS and removeRSS, not
    // directly.
    pub curRSS: u64,

    // maxRSS is the maximum resident set size in bytes of a MemoryManager.
    // It is tracked as the application adds and removes mappings to pmas.
    //
    // maxRSS should be modified only via insertRSS, not directly.
    pub maxRSS: u64,
}

#[derive(Default)]
pub struct MemoryManagerInternal {
    pub uid: UniqueID,
    pub inited: bool,

    pub mappingLock: Arc<RwLock<()>>,
    pub mapping: Mutex<MMMapping>,

    pub pagetable: RwLock<MMPagetable>,

    pub metadataLock: Arc<Mutex<()>>,
    pub metadata: Mutex<MMMetadata>,

    pub layout: Mutex<MmapLayout>,
    pub aioManager: AIOManager,
}

#[derive(Clone)]
pub struct MemoryManager(Arc<MemoryManagerInternal>);

impl Deref for MemoryManager {
    type Target = Arc<MemoryManagerInternal>;

    fn deref(&self) -> &Arc<MemoryManagerInternal> {
        &self.0
    }
}

#[derive(Clone)]
pub struct MemoryManagerWeak {
    pub uid: UniqueID,
    pub data: Weak<MemoryManagerInternal>,
}

impl MemoryManagerWeak{
    pub fn ID(&self) -> UniqueID {
        return self.uid;
    }

    pub fn Upgrade(&self) -> MemoryManager {
        return MemoryManager(self.data.upgrade().expect("MemoryManagerWeak upgrade fail"))
    }
}

impl MemoryManager {
    pub fn Init() -> Self {
        let mut vmas = AreaSet::New(0, !0);
        let vma = VMA {
            mappable: None,
            offset: 0,
            fixed: true,
            realPerms: AccessType::ReadWrite(),
            effectivePerms: AccessType::ReadWrite(),
            maxPerms: AccessType::ReadWrite(),
            private: true,
            growsDown: false,
            dontfork: false,
            mlockMode: MLockMode::MlockNone,
            kernel: true,
            hint: String::from("Kernel Space"),
            id: None,
            numaPolicy: 0,
            numaNodemask: 0,
        };

        let gap = vmas.FindGap(MemoryDef::PHY_LOWER_ADDR);
        vmas.Insert(&gap, &Range::New(MemoryDef::PHY_LOWER_ADDR, MemoryDef::PHY_UPPER_ADDR - MemoryDef::PHY_LOWER_ADDR), vma);

        let mapping = MMMapping {
            vmas: vmas,
            brkInfo: BrkInfo::default(),
            usageAS: 0,
            lockedAS: 0,
            defMLockMode: MLockMode::MlockNone,
        };

        let metadata = MMMetadata {
            argv: Range::default(),
            envv: Range::default(),
            auxv: Vec::new(),
            executable: None,
            dumpability: NOT_DUMPABLE,
        };

        let pt = KERNEL_PAGETABLE.Fork(&*PAGE_MGR).unwrap();
        let pagetable = MMPagetable {
            pt: pt,
            sharedLoadsOffset: MemoryDef::SHARED_START,
            curRSS: 0,
            maxRSS: 0,
        };

        let layout = MmapLayout {
            MinAddr: MemoryDef::VIR_MMAP_START,
            MaxAddr: MemoryDef::LOWER_TOP,
            BottomUpBase: MemoryDef::VIR_MMAP_START,
            TopDownBase: MemoryDef::LOWER_TOP,
            ..Default::default()
        };

        let internal = MemoryManagerInternal {
            uid: NewUID(),
            inited: true,
            mappingLock: Arc::new(RwLock::new(())),
            mapping: Mutex::new(mapping),
            pagetable: RwLock::new(pagetable),
            metadataLock: Arc::new(Mutex::new(())),
            metadata: Mutex::new(metadata),
            layout: Mutex::new(layout),
            aioManager: AIOManager::default(),
        };

        return Self(Arc::new(internal))
    }

    pub fn Downgrade(&self) -> MemoryManagerWeak {
        return MemoryManagerWeak {
            uid: self.uid,
            data: Arc::downgrade(&self.0)
        }
    }

    pub fn MapStackAddr(&self) -> u64 {
        return self.layout.lock().MapStackAddr();
    }

    pub fn SetupStack(&self, stackLayout: &StackLayout, entries: &[AuxEntry]) {
        let mut meta = self.metadata.lock();
        for entry in entries {
            meta.auxv.push(*entry);
        }

        meta.argv = Range::New(stackLayout.ArgvStart, stackLayout.ArgvEnd - stackLayout.ArgvStart);
        meta.envv = Range::New(stackLayout.EnvvStart, stackLayout.EvvvEnd - stackLayout.EnvvStart)
    }

    //Remove virtual memory to the phy mem mapping
    pub fn RemoveVMAsLocked(&self, ar: &Range) -> Result<()> {
        let mut mapping = self.mapping.lock();
        let (mut vseg, vgap) = mapping.vmas.Find(ar.Start());
        if vgap.Ok() {
            vseg = vgap.NextSeg();
        }

        while vseg.Ok() && vseg.Range().Start() < ar.End() {
            vseg = mapping.vmas.Isolate(&vseg, &ar);
            let r = vseg.Range();
            let vma = vseg.Value();

            if vma.mappable.is_some() {
                let mappable = vma.mappable.clone().unwrap();
                mappable.RemoveMapping(self, &r, vma.offset, vma.CanWriteMappableLocked())?;
            }

            mapping.usageAS -= r.Len();
            if vma.mlockMode != MLockMode::MlockNone {
                mapping.lockedAS -= r.Len();
            }

            let mut pt = self.pagetable.write();

            pt.pt.MUnmap(r.Start(), r.Len())?;
            pt.curRSS -= r.Len();
            let vgap = mapping.vmas.Remove(&vseg);
            vseg = vgap.NextSeg();
        }

        return Ok(())
    }

    pub fn MappingLock(&self) -> Arc<RwLock<()>> {
        return self.mappingLock.clone();
    }

    pub fn BrkSetup(&self, addr: u64) {
        let ml = self.MappingLock();
        let _ml = ml.write();
        self.mapping.lock().brkInfo = BrkInfo {
            brkStart : addr,
            brkEnd: addr,
            brkMemEnd: addr,
        }
    }

    pub fn AddRssLock(&self, ar: &Range) {
        let mut pt = self.pagetable.write();
        pt.curRSS += ar.Len();
        if pt.curRSS > pt.maxRSS {
            pt.maxRSS = pt.curRSS;
        }
    }

    pub fn RemoveRssLock(&self, ar: &Range) {
        let mut pt = self.pagetable.write();
        pt.curRSS -= ar.Len();
    }

    pub fn GenStatmSnapshot(&self, _task: &Task) -> Vec<u8> {
        let pt = self.pagetable.read();
        let vss = pt.curRSS;
        let rss = pt.maxRSS;

        let res = format!("{} {} 0 0 0 0 0\n",
                          vss/MemoryDef::PAGE_SIZE, rss/MemoryDef::PAGE_SIZE);

        return res.as_bytes().to_vec();
    }

    pub const DEV_MINOR_BITS : usize = 20;
    pub const VSYSCALLEND: u64 = 0xffffffffff601000;
    pub const VSYSCALL_MAPS_ENTRY : &'static str  = "ffffffffff600000-ffffffffff601000 --xp 00000000 00:00 0                  [vsyscall]\n";

    pub fn GetSnapshotLocked(&self, task: &Task, skipKernel: bool) -> String {
        let internal = self.mapping.lock();
        let mut seg = internal.vmas.FirstSeg();
        let mut ret = "".to_string();
        loop {
            if seg.IsTail() {
                break;
            }

            let vma = seg.Value();
            if vma.kernel && skipKernel {
                seg = seg.NextSeg();
                continue;
            }

            let range = seg.Range();

            let private = if vma.private {
                "p"
            } else {
                "s"
            };

            let (dev, inodeId) = match &vma.id {
                None => (0, 0),
                Some(ref mapping) => {
                    (mapping.DeviceID(), mapping.InodeID())
                }
            };

            let devMajor = (dev >> Self::DEV_MINOR_BITS) as u32;
            let devMinor = (dev & ((1 <<Self::DEV_MINOR_BITS) - 1)) as u32;

            let mut s = if vma.hint.len() == 0 {
                vma.hint.to_string()
            } else {
                match &vma.id {
                    None => "".to_string(),
                    //todo: seems that mappedName doesn't work. Fix it
                    Some(ref id) => id.MappedName(task),
                }
            };

            let str = format!("{:08x}-{:08x} {}{} {:08x} {:02x}:{:02x} {} ",
                              range.Start(),
                              range.End(),
                              vma.realPerms.String(),
                              private,
                              vma.offset,
                              devMajor,
                              devMinor,
                              inodeId
            );

            if s.len() != 0 && str.len() < 73 {
                let pad = String::from_utf8(vec![b' '; 73 - str.len()]).unwrap();
                s = pad + &s;
            }

            ret += &str;
            ret += &s;
            ret += "\n";

            seg = seg.NextSeg();
        }

        ret += Self::VSYSCALL_MAPS_ENTRY;

        return ret;
        //return ret.as_bytes().to_vec();
    }

    pub fn GenMapsSnapshot(&self, task: &Task) -> Vec<u8> {
        let ret = self.GetSnapshotLocked(task, true);

        return ret.as_bytes().to_vec();
    }

    pub fn SetExecutable(&self, dirent: &Dirent) {
        self.metadata.lock().executable = Some(dirent.clone());
    }

    //remove all the user vmas, used for execve
    pub fn Clear(&self) -> Result<()> {
        let ml = self.MappingLock();
        let _ml = ml.write();

        let mut pt = self.pagetable.write();

        // if we are clearing memory manager in current pagetable,
        // need to switch to kernel pagetable to avoid system crash
        let isCurrent = pt.pt.IsActivePagetable();
        if isCurrent {
            super::super::KERNEL_PAGETABLE.SwitchTo();
        }

        let mut mm = self.mapping.lock();
        let mut vseg = mm.vmas.FirstSeg();

        while vseg.Ok() {
            let r = vseg.Range();
            let vma = vseg.Value();

            if vma.kernel == true {
                vseg = vseg.NextSeg();
                continue;
            }

            if vma.mappable.is_some() {
                let mappable = vma.mappable.clone().unwrap();
                mappable.RemoveMapping(self, &r, vma.offset, vma.CanWriteMappableLocked())?;
            }

            pt.pt.MUnmap(r.Start(), r.Len())?;
            let vgap = mm.vmas.Remove(&vseg);
            vseg = vgap.NextSeg();
        }

        return Ok(())
    }

    pub fn MinCore(&self, _task: &Task, r: &Range) -> Vec<u8> {
        let ml = self.MappingLock();
        let _ml = ml.read();

        let mut res = Vec::with_capacity((r.Len() / MemoryDef::PAGE_SIZE) as usize);
        let mut addr = r.Start();
        while addr < r.End() {
            match self.VirtualToPhyLocked(addr) {
                Ok(_) => res.push(1),
                Err(_) => res.push(0),
            }
            addr += MemoryDef::PAGE_SIZE;
        }

        return res;
    }

    // MLock implements the semantics of Linux's mlock()/mlock2()/munlock(),
    // depending on mode.
    pub fn Mlock(&self, _task: &Task, addr: u64, len: u64, mode: MLockMode) -> Result<()> {
        let la = match Addr(len + Addr(addr).PageOffset()).RoundUp() {
            Ok(l) => l.0,
            Err(_) => return Err(Error::SysError(SysErr::EINVAL))
        };

        let ar = match Addr(addr).RoundDown().unwrap().ToRange(la) {
            Ok(r) => r,
            Err(_) => return Err(Error::SysError(SysErr::EINVAL))
        };

        let ml = self.MappingLock();
        let _ml = ml.write();

        if ar.Len() == 0 {
            return Ok(())
        }

        let mut unmapped = false;

        let mut mapping = self.mapping.lock();
        let mut vseg = mapping.vmas.FindSeg(ar.Start());
        loop {
            if !vseg.Ok() {
                unmapped = true;
                break;
            }

            vseg = mapping.vmas.Isolate(&vseg, &ar);
            let mut vma = vseg.Value();
            let prevMode = vma.mlockMode;
            vma.mlockMode = mode;
            vseg.SetValue(vma);
            if mode != MLockMode::MlockNone && prevMode == MLockMode::MlockNone {
                mapping.lockedAS += vseg.Range().Len();
            } else if mode == MLockMode::MlockNone && prevMode != MLockMode::MlockNone {
                mapping.lockedAS -= vseg.Range().Len();
            }

            if ar.End() <= vseg.Range().End() {
                break;
            }
            let (vsegTmp, _) = vseg.NextNonEmpty();
            vseg = vsegTmp;
        }

        mapping.vmas.MergeRange(&ar);
        mapping.vmas.MergeAdjacent(&ar);
        if unmapped {
            return Err(Error::SysError(SysErr::ENOMEM))
        }

        // todo: populate the pagetable
        // if mode == MLockMode::MlockEager {}

        let mut vseg = mapping.vmas.FindSeg(ar.Start());
        while vseg.Ok() && vseg.Range().Start() < ar.End() {
            let vma = vseg.Value();
            // Linux: mm/gup.c:__get_user_pages() returns EFAULT in this
            // case, which is converted to ENOMEM by mlock.
            if !vma.effectivePerms.Any() {
                return Err(Error::SysError(SysErr::ENOMEM))
            }

            if let Some(iops) = vma.mappable.clone() {
                let mr = ar.Intersect(&vseg.Range());
                let fstart = mr.Start() - vseg.Range().Start() + vma.offset;

                // todo: fix the Munlock, when there are multiple process lock/unlock a memory range.
                // with current implementation, the first unlock will work.
                iops.Mlock(fstart, mr.Len(), mode)?;
            }

            vseg = vseg.NextSeg()
        }

        return Ok(())
    }

    // MLockAll implements the semantics of Linux's mlockall()/munlockall(),
    // depending on opts.
    pub fn MlockAll(&self, _task: &Task, opts: &MLockAllOpts) -> Result<()> {
        if !opts.Current && !opts.Future {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        // todo: fully support opts.Current and opts.Future
        // it is not supported now
        let mode = opts.Mode;
        let ml = self.MappingLock();
        let _ml = ml.write();

        let mapping = self.mapping.lock();
        let mut vseg = mapping.vmas.FirstSeg();
        while vseg.Ok() {
            let mut vma = vseg.Value();
            vma.mlockMode = mode;
            vseg.SetValue(vma.clone());

            if !vma.effectivePerms.Any() {
                vseg = vseg.NextSeg();
                continue;
            }

            if let Some(iops) = vma.mappable.clone() {
                let mr = vseg.Range();
                let fstart = mr.Start() - vseg.Range().Start() + vma.offset;

                // todo: fix the Munlock, when there are multiple process lock/unlock a memory range.
                // with current implementation, the first unlock will work.
                iops.Mlock(fstart, mr.Len(), mode)?;
            }

            vseg = vseg.NextSeg();
        }

        return Ok(())
    }

    pub fn MSync(&self, _task: &Task, addr: u64, length: u64, opts: &MSyncOpts) -> Result<()> {
        if addr != Addr(addr).RoundDown()?.0 {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        if length == 0 {
            return Ok(())
        }

        let la = match Addr(length).RoundUp() {
            Err(_) => return Err(Error::SysError(SysErr::ENOMEM)),
            Ok(l) => l.0
        };

        let ar = Range::New(addr, la);

        let ml = self.MappingLock();
        let _ml = ml.read();

        let mapping = self.mapping.lock();
        let mut vseg = mapping.vmas.LowerBoundSeg(ar.Start());

        if !vseg.Range().Contains(ar.Start()) {
            return Err(Error::SysError(SysErr::ENOMEM))
        }

        let mut unmaped = false;
        let mut lastEnd = ar.Start();
        loop {
            if !vseg.Ok() {
                unmaped = true;
                break;
            }

            if lastEnd < vseg.Range().Start() {
                unmaped = true;
            }

            lastEnd = vseg.Range().End();
            let vma = vseg.Value();
            if opts.Invalidate && vma.mlockMode != MLockMode::MlockNone {
                return Err(Error::SysError(SysErr::EBUSY))
            }

            // It's only possible to have dirtied the Mappable through a shared
            // mapping. Don't check if the mapping is writable, because mprotect
            // may have changed this, and also because Linux doesn't.
            if vma.mappable.is_some() && !vma.private {
                let msyncType = if opts.Sync {
                    MSyncType::MsSync
                } else {
                    MSyncType::MsAsync
                };

                let fops = vma.mappable.clone().unwrap();

                let mr = ar.Intersect(&vseg.Range());

                let fstart = mr.Start() - vseg.Range().Start() + vma.offset;
                fops.MSync(&Range::New(fstart, mr.Len()), msyncType)?;

                if lastEnd >= ar.End() {
                    break;
                }

                vseg = mapping.vmas.LowerBoundSeg(lastEnd)
            } else {
                if lastEnd >= ar.End() {
                    break;
                }
                vseg = vseg.NextSeg();
            }
        }

        if unmaped {
            return Err(Error::SysError(SysErr::ENOMEM))
        }

        return Ok(())
    }

    pub fn SetMmapLayout(&self, minUserAddr: u64, maxUserAddr: u64, r: &LimitSet) -> Result<MmapLayout> {
        let layout = Context64::NewMmapLayout(minUserAddr, maxUserAddr, r)?;
        *self.layout.lock() = layout;
        return Ok(layout)
    }

    pub fn GetRoot(&self) -> u64 {
        return self.pagetable.read().pt.GetRoot();
    }

    pub fn GetVmaAndRangeLocked(&self, addr: u64) -> Option<(VMA, Range)> {
        let mapping = self.mapping.lock();
        let vseg = mapping.vmas.FindSeg(addr);
        if !vseg.Ok() {
            return None;
        }

        return Some((vseg.Value(), vseg.Range()))
    }

    pub fn MapPageLocked(&self, vaddr: Addr, phyAddr: Addr, flags: PageTableFlags) -> Result<bool> {
        let pt = self.pagetable.write();
        return pt.pt.MapPage(vaddr, phyAddr, flags, &*PAGE_MGR);
    }

    pub fn VirtualToPhy(&self, vAddr: u64) -> Result<(u64, AccessType)> {
        let ml = self.MappingLock();
        let _ml = ml.read();

        if vAddr == 0 {
            return Err(Error::SysError(SysErr::EFAULT))
        }

        let pagetable = self.pagetable.read();
        return pagetable.pt.VirtualToPhy(vAddr);
    }

    pub fn VirtualToPhyLocked(&self, vAddr: u64) -> Result<(u64, AccessType)> {
        if vAddr == 0 {
            return Err(Error::SysError(SysErr::EFAULT))
        }

        let pagetable = self.pagetable.read();
        return pagetable.pt.VirtualToPhy(vAddr);
    }

    pub fn InstallPageWithAddrLocked(&self, task: &Task, pageAddr: u64) -> Result<()> {
        let (vma, range) = match self.GetVmaAndRangeLocked(pageAddr) {
            None => return Err(Error::SysError(SysErr::EFAULT)),
            Some(data) => data
        };

        return self.InstallPageLocked(task, &vma, pageAddr, &range);
    }

    pub fn InstallPageLocked(&self, task: &Task, vma: &VMA, pageAddr: u64, range: &Range) -> Result<()> {
        match self.VirtualToPhyLocked(pageAddr) {
            Err(_) => (),
            Ok(_) => return Ok(())
        }

        if !vma.effectivePerms.Any() {
            return Err(Error::SysError(SysErr::EFAULT))
        }

        let exec = vma.effectivePerms.Exec();
        match &vma.mappable {
            Some(ref mappable) => {
                let vmaOffset = pageAddr - range.Start();
                let fileOffset = vmaOffset + vma.offset; // offset in the file
                let phyAddr = mappable.MapFilePage(task, fileOffset)?;
                //error!("fault 2.1, vma.mappable.is_some() is {}, vaddr is {:x}, paddr is {:x}",
                //      vma.mappable.is_some(), pageAddr, phyAddr);

                if vma.private {
                    self.MapPageReadLocked(pageAddr, phyAddr, exec);
                } else {
                    let writeable = vma.effectivePerms.Write();
                    if writeable {
                        self.MapPageWriteLocked(pageAddr, phyAddr, exec);
                    } else {
                        self.MapPageReadLocked(pageAddr, phyAddr, exec);
                    }
                }

                return Ok(())
            },
            None => {
                //let vmaOffset = pageAddr - range.Start();
                //let phyAddr = vmaOffset + vma.offset; // offset in the phyAddr

                let phyAddr = super::super::PAGE_MGR.AllocPage(true).unwrap();
                if vma.private {
                    self.MapPageReadLocked(pageAddr, phyAddr, exec);
                } else {
                    let writeable = vma.effectivePerms.Write();
                    if writeable {
                        self.MapPageWriteLocked(pageAddr, phyAddr, exec);
                    } else {
                        self.MapPageReadLocked(pageAddr, phyAddr, exec);
                    }
                }

                return Ok(())
            }
        }
    }

    pub fn MapPageWriteLocked(&self, vAddr: u64, pAddr: u64, exec: bool) {
        let pt = self.pagetable.write();
        pt.pt.MapPage(Addr(vAddr), Addr(pAddr), PageOpts::New(true, true, exec).Val(), &*PAGE_MGR).unwrap();
    }

    pub fn MapPageReadLocked(&self, vAddr: u64, pAddr: u64, exec: bool) {
        let pt = self.pagetable.write();
        pt.pt.MapPage(Addr(vAddr), Addr(pAddr), PageOpts::New(true, false, exec).Val(), &*PAGE_MGR).unwrap();
    }

    pub fn EnableWriteLocked(&self, addr: u64, exec: bool) {
        let pt = self.pagetable.write();
        pt.pt.SetPageFlags(Addr(addr), PageOpts::New(true, true, exec).Val());
    }

    pub fn CopyOnWriteLocked(&self, pageAddr: u64, vma: &VMA) {
        let (phyAddr, permission) = self.VirtualToPhyLocked(pageAddr).expect(&format!("addr is {:x}", pageAddr));

        if permission.Write() {
            // another thread has cow, return
            return;
        }

        let refCount = super::super::PAGE_MGR.GetRef(phyAddr)
            .expect(&format!("CopyOnWrite PAGE_MGR GetRef addr {:x} fail", phyAddr));

        let exec = vma.effectivePerms.Exec();
        if refCount == 1 && vma.mappable.is_none() {
            //print!("CopyOnWriteLocked enable write ... pageaddr is {:x}", pageAddr);
            self.EnableWriteLocked(pageAddr, exec);
        } else {
            // Copy On Write
            let page = { super::super::PAGE_MGR.AllocPage(true).unwrap() };
            CopyPage(pageAddr, page);
            self.MapPageWriteLocked(pageAddr, page, exec);
        }

        unsafe { llvm_asm!("invlpg ($0)" :: "r" (pageAddr): "memory" ) };
    }

    pub fn CopyOnWrite(&self, pageAddr: u64, vma: &VMA) {
        let ml = self.MappingLock();
        let _ml = ml.write();

        PerfGoto(PerfType::PageFault);
        self.CopyOnWriteLocked(pageAddr, vma);
        PerfGofrom(PerfType::PageFault);
    }

    // check whether the address range is legal.
    // 1. whether the range belong to user's space
    // 2. Whether the read/write permission meet requirement
    // 3. if need cow, fix the page.
    // 4. return max allowed len
    pub fn FixPermission(&self, task: &Task, vAddr: u64, len: u64, writeReq: bool, allowPartial: bool) -> Result<u64> {
        if core::u64::MAX - vAddr < len || vAddr == 0 {
            return Err(Error::SysError(SysErr::EFAULT))
        }

        let ml = self.MappingLock();
        let _ml = ml.write();

        self.FixPermissionLocked(task, vAddr, len ,writeReq, allowPartial)
    }

    // check whether the address range is legal.
    // 1. whether the range belong to user's space
    // 2. Whether the read/write permission meet requirement
    // 3. if need cow, fix the page.
    // 4. return max allowed len
    pub fn FixPermissionLocked(&self, task: &Task, vAddr: u64, len: u64, writeReq: bool, allowPartial: bool) -> Result<u64> {
        let mut addr = Addr(vAddr).RoundDown()?.0;
        //error!("FixPermission vaddr {:x} addr {:x} len is {:x}", vAddr, addr, len);
        while addr <= vAddr + len - 1 {
            let (_, permission) = match self.VirtualToPhyLocked(addr) {
                Err(Error::AddressNotMap(_)) => {
                    match self.InstallPageWithAddrLocked(task, addr) {
                        Err(_) => {
                            if !allowPartial || addr < vAddr {
                                return Err(Error::SysError(SysErr::EFAULT))
                            }
                            return Ok(addr - vAddr)
                        }
                        Ok(()) => (),
                    }
                    self.VirtualToPhyLocked(addr)?
                }
                Err(e) => {
                    return Err(e)
                }
                Ok(ret) => ret,
            };
            if writeReq && !permission.Write() {
                let (vma, _) = match self.GetVmaAndRangeLocked(addr) {
                    None => {
                        if !allowPartial || addr < vAddr {
                            return Err(Error::SysError(SysErr::EFAULT))
                        }

                        return Ok(addr - vAddr)
                    },
                    Some(vma) => vma.clone(),
                };

                if !vma.effectivePerms.Write() {
                    if !allowPartial || addr < vAddr {
                        return Err(Error::SysError(SysErr::EFAULT))
                    }

                    return Ok(addr - vAddr)
                }

                self.CopyOnWriteLocked(addr, &vma);
            }

            addr += MemoryDef::PAGE_SIZE;
        }

        return Ok(len);
    }

    pub fn PopulateVMALocked(&self, task: &Task, vmaSeg: &AreaSeg<VMA>, ar: &Range, precommit: bool, vdso: bool) -> Result<()> {
        let vma = vmaSeg.Value();
        let mut perms = vma.effectivePerms;

        //if it is filemapping and private, need cow.
        // if it is anon share, first marks it as writeable. When clone, mark it as readonly.
        if vma.private & vma.mappable.is_some() {
            perms.ClearWrite();
        }

        self.pagetable.write().pt.MUnmap(ar.Start(), ar.Len())?;
        let segAr = vmaSeg.Range();
        match vma.mappable {
            None => {
                //anonymous mapping
                if !vdso {
                    self.AddRssLock(ar);
                } else {
                    //vdso: the phyaddress has been allocated and the address is vma.offset
                    self.pagetable.write().pt.MapHost(task, ar.Start(), &IoVec::NewFromAddr(vma.offset, ar.Len() as usize), &perms, true)?;
                }
            }
            Some(mappable) => {
                //host file mapping
                // the map file mapfile cost is high. Only pre-commit it when the size < 4MB.
                // todo: improve that later

                if precommit && segAr.Len() < 0x200000 {
                    self.pagetable.write().pt.MapFile(task, ar.Start(), &mappable, &Range::New(vma.offset + ar.Start() - segAr.Start(), ar.Len()), &perms, precommit)?;
                }
                self.AddRssLock(ar);
            }
        }

        return Ok(())
    }

    pub fn PopulateVMARemapLocked(&self, task: &Task, vmaSeg: &AreaSeg<VMA>, ar: &Range, oldar: &Range, _precommit: bool) -> Result<()> {
        //let segAr = vmaSeg.Range();
        let vma = vmaSeg.Value();
        let mut perms = vma.effectivePerms;

        if vma.private & vma.mappable.is_some() { //if it is filemapping and private, need cow.
            perms.ClearWrite();
        }

        let pt = self.pagetable.write();

        let len = if ar.Len() > oldar.Len() {
            oldar.Len()
        } else {
            ar.Len()
        };

        // todo: change the name to pt.Remap
        pt.pt.RemapAna(task, &Range::New(ar.Start(), len), oldar.Start(), &perms, true)?;

        return Ok(())
    }

    pub fn ApplicationAddrRange(&self) -> Range {
        let layout = self.layout.lock();
        return Range::New(layout.MinAddr, layout.MaxAddr - layout.MinAddr);
    }

    pub fn Fork(&self) -> Result<Self> {
        let ml = self.MappingLock();
        let _ml = ml.read();

        let layout = *self.layout.lock();
        let mmIntern2 = MemoryManagerInternal {
            uid: NewUID(),
            inited: true,
            layout: Mutex::new(layout),
            ..Default::default()
        };

        let mm2 = MemoryManager(Arc::new(mmIntern2));
        {
            let mappingInternal1 = self.mapping.lock();
            let mut mappingInternal2 = mm2.mapping.lock();

            mappingInternal2.brkInfo = mappingInternal1.brkInfo;
            mappingInternal2.usageAS = mappingInternal1.usageAS;
            mappingInternal2.lockedAS = 0;
            let range = mappingInternal1.vmas.range;
            mappingInternal2.vmas.Reset(range.Start(), range.Len());

            let ptInternal1 = self.pagetable.write();
            let mut ptInternal2 = mm2.pagetable.write();

            ptInternal2.sharedLoadsOffset = ptInternal1.sharedLoadsOffset;
            ptInternal2.curRSS = ptInternal1.curRSS;
            ptInternal2.maxRSS = ptInternal1.maxRSS;
            ptInternal2.pt = ptInternal1.pt.Fork(&*PAGE_MGR)?;


            let mut srcvseg = mappingInternal1.vmas.FirstSeg();
            let mut dstvgap = mappingInternal2.vmas.FirstGap();

            let meta1 = self.metadata.lock();
            let mut meta2 = mm2.metadata.lock();
            for aux in &meta1.auxv {
                meta2.auxv.push(*aux);
            }
            meta2.argv = meta1.argv;
            meta2.envv = meta1.envv;
            meta2.executable = meta1.executable.clone();

            while srcvseg.Ok() {
                let mut vma = srcvseg.Value();

                if vma.dontfork {
                    mappingInternal2.usageAS -= srcvseg.Range().Len();
                    let tmp = srcvseg.NextSeg();
                    srcvseg = tmp;
                    continue;
                }

                let vmaAR = srcvseg.Range();

                if vma.mappable.is_some() {
                    let mappable = vma.mappable.clone().unwrap();

                    match mappable.AddMapping(&mm2, &vmaAR, vma.offset, vma.CanWriteMappableLocked()) {
                        Err(e) => {
                            let appRange = mm2.ApplicationAddrRange();
                            mm2.RemoveVMAsLocked(&appRange)?;
                            return Err(e)
                        }
                        _ => (),
                    }
                }

                vma.mlockMode = MLockMode::MlockNone;

                if vma.kernel == false {
                    //info!("vma kernel is {}, private is {}, hint is {}", vma.kernel, vma.private, vma.hint);
                    if vma.private {
                        //cow
                        ptInternal1.pt.ForkRange(&ptInternal2.pt, vmaAR.Start(), vmaAR.Len(), &*PAGE_MGR)?;
                    } else {
                        ptInternal1.pt.CopyRange(&ptInternal2.pt, vmaAR.Start(), vmaAR.Len(), &*PAGE_MGR)?;
                    }
                }

                dstvgap = mappingInternal2.vmas.Insert(&dstvgap, &vmaAR, vma).NextGap();

                let tmp = srcvseg.NextSeg();
                srcvseg = tmp;
            }
        }

        return Ok(mm2);
    }

    //used by file truncate, remove the cow related mapping
    /*pub fn ResetFileMapping(&self, task: &Task, ar: &Range, _invalidatePrivate: bool) {
        //return self.MUnmap(task, ar.Start(), ar.Len()).unwrap();
        let mm = self.read();

        let vseg = mm.vmas.FindSeg(ar.Start());
        if !vseg.Ok() || vseg.Value().mappable.is_none() || vseg.Range().IsSupersetOf(ar) {
            panic!("MemoryManager::RetsetFileMapping invalid input")
        };

        let pt = &mm.pt;

        let vr = vseg.Range();

        let offset = ar.Start() - vr.Start();
        let vma = vseg.Value();
        let mappable = vma.mappable.unwrap();

        //reset the filerange
        pt.ResetFileMapping(task, ar.Start(), &mappable, &Range::New(vma.offset + offset, ar.Len()), &vma.realPerms).unwrap();
    }

        fn GetBlocks(&self, start: u64, len: u64, bs: &mut StackVec<IoVec>, writeable: bool) -> Result<()> {
        let alignedStart = Addr(start).RoundDown()?.0;
        let aligntedEnd = Addr(start + len).RoundUp()?.0;

        let pages = ((aligntedEnd - alignedStart) / MemoryDef::PAGE_SIZE) as usize;
        let mut vec = StackVec::New(pages);

        let mm = self.read();
        let pt = &mm.pt;

        if writeable {
            pt.GetAddresses(Addr(alignedStart), Addr(aligntedEnd), &mut vec)?;
        } else {
            pt.GetAddresses(Addr(alignedStart), Addr(aligntedEnd), &mut vec)?;
        }

        ToBlocks(bs, vec.Slice());

        let mut slice = bs.SliceMut();

        let startOff = start - alignedStart;
        slice[0].start += startOff;
        slice[0].len -= startOff as usize;

        let endOff = aligntedEnd - (start + len);
        slice[slice.len() - 1].len -= endOff as usize;

        return Ok(())
    }

    //get an array of readonly blocks, return entries count put in bs
    pub fn GetReadonlyBlocks(&self, start: u64, len: u64, bs: &mut StackVec<IoVec>) -> Result<()> {
        return self.GetBlocks1(start, len, bs, false);
    }

    pub fn GetAddressesWithCOW(&self, start: u64, len: u64, bs: &mut StackVec<IoVec>) -> Result<()> {
        return self.GetBlocks1(start, len, bs, true);
    }
    */

    pub fn ID(&self) -> u64 {
        return self.uid;
    }

    pub fn V2PIov(&self, task: &Task, start: u64, len: u64, output: &mut Vec<IoVec>, writable: bool) -> Result<()> {
        let ml = self.MappingLock();
        let _ml = ml.read();
        return self.V2PIovLocked(task, start, len, output, writable)
    }

    pub fn V2PIovLocked(&self, task: &Task, start: u64, len: u64, output: &mut Vec<IoVec>, writable: bool) -> Result<()> {
        self.FixPermissionLocked(task, start, len, writable, false)?;

        let mut start = start;
        let end = start + len;

        while start < end {
            let next = if Addr(start).IsPageAligned() {
                start + MemoryDef::PAGE_SIZE
            } else {
                Addr(start).RoundUp().unwrap().0
            };

            match self.VirtualToPhyLocked(start) {
                Err(e) => {
                    info!("convert to phyaddress fail, addr = {:x} e={:?}", start, e);
                    return Err(Error::SysError(SysErr::EFAULT))
                }
                Ok((pAddr, _)) => {
                    output.push(IoVec {
                        start: pAddr,
                        len: if end < next {
                            (end - start) as usize
                        } else {
                            (next - start) as usize
                        }, //iov.len,
                    });

                }
            }

            start = next;
        }

        return Ok(())
    }

    //Copy an Object to user memory, it is used only for the task_clone
    pub fn CopyOutObj<T: Sized + Copy>(&self, task: &Task, src: &T, dst: u64) -> Result<()> {
        let len = core::mem::size_of::<T>();

        let mut dsts = Vec::new();
        self.V2PIov(task, dst, len as u64, &mut dsts, true)?;
        let dsts = BlockSeq::NewFromSlice(&dsts);

        let srcAddr = src as * const _ as u64 as * const u8;
        let src = unsafe { slice::from_raw_parts(srcAddr, len) };

        dsts.CopyOut(src);
        return Ok(())
    }
}

pub type UniqueID = u64;
lazy_static! {
    static ref UID: AtomicU64 = AtomicU64::new(1);
}

pub fn NewUID() -> u64 {
    return UID.fetch_add(1, atomic::Ordering::SeqCst);
}

pub fn ToBlocks(bs: &mut StackVec<IoVec>, arr: &[u64]) {
    let mut begin = arr[0];
    let mut expect = begin + MemoryDef::PAGE_SIZE;
    for i in 1..arr.len() {
        if arr[i] == expect {
            expect += MemoryDef::PAGE_SIZE;
        } else {
            bs.Push(IoVec::NewFromAddr(begin, (expect - begin) as usize));
            begin = arr[i];
            expect = begin + MemoryDef::PAGE_SIZE;
        }
    }

    bs.Push(IoVec::NewFromAddr(begin, (expect - begin) as usize));
}

// MLockAllOpts holds options to MLockAll.
pub struct MLockAllOpts {
    // If Current is true, change the memory-locking behavior of all mappings
    // to Mode. If Future is true, upgrade the memory-locking behavior of all
    // future mappings to Mode. At least one of Current or Future must be true.
    pub Current: bool,
    pub Future: bool,
    pub Mode: MLockMode
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn TestToBlocks() {
        let mut bs = StackVec::New(100);

        let arr = [MemoryDef::PAGE_SIZE, 2 * MemoryDef::PAGE_SIZE, 3 * MemoryDef::PAGE_SIZE, 5 * MemoryDef::PAGE_SIZE];
        ToBlocks(&mut bs, &arr);

        let slice = bs.Slice();
        assert_eq!(slice[0], Block::NewFromAddr(MemoryDef::PAGE_SIZE, 3 * MemoryDef::PAGE_SIZE as usize));
        assert_eq!(slice[1], Block::NewFromAddr(5 * MemoryDef::PAGE_SIZE, MemoryDef::PAGE_SIZE as usize));
    }
}