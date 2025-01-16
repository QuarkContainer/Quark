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
use alloc::sync::Weak;
use alloc::vec::Vec;
use core::ops::Deref;
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;

use crate::kernel_def::Invlpg;
use crate::qlib::mutex::*;

use super::super::super::addr::*;
use super::super::super::auxv::*;
use super::super::super::common::*;
use super::super::super::limits::*;
use super::super::super::linux_def::*;
use super::super::super::mem::areaset::*;
use super::super::super::pagetable::PageTableFlags;
use super::super::super::pagetable::*;
use super::super::super::range::*;
use super::super::super::vcpu_mgr::CPULocal;
use super::super::arch::__arch::context::Context64;
use super::super::fs::dirent::*;
use super::super::kernel::aio::aio_context::*;
use super::super::mm::*;
use super::super::stack::*;
use super::super::task::*;
use super::super::uid::*;
use super::super::Kernel::HostSpace;
use super::super::KERNEL_PAGETABLE;
use super::super::PAGE_MGR;
use super::arch::*;
use super::metadata::*;
use super::syscalls::*;
use super::vma::*;
use super::*;
use crate::qlib::kernel::{SHARESPACE, asm::*};
use crate::qlib::vcpu_mgr::VcpuMode;

use crate::qlib::kernel::arch::tee::{is_cc_active, guest_physical_address};

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

impl MMMetadata {
    pub fn Fork(&self) -> Self {
        let mut auxv = Vec::new();
        for a in &self.auxv {
            auxv.push(*a)
        }

        return Self {
            argv: self.argv,
            envv: self.envv,
            auxv: auxv,
            executable: self.executable.clone(),
            dumpability: self.dumpability,
        };
    }
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

    // store whether the vcpu are working on the memory manager
    pub vcpuMapping: AtomicU64,
    pub tlbShootdownMask: AtomicU64,

    pub userVDSOBase: AtomicU64,
    pub mappingLock: QUpgradableLock,
    pub mapping: QMutex<MMMapping>,

    pub pagetable: QRwLock<MMPagetable>,

    pub metadataLock: Arc<QMutex<()>>,
    pub metadata: QMutex<MMMetadata>,

    pub layout: QMutex<MmapLayout>,
    pub aioManager: AIOManager,
    pub membarrierPrivateEnabled: AtomicBool,
}

#[derive(Clone)]
pub struct MemoryManager(Arc<MemoryManagerInternal>);

impl Deref for MemoryManager {
    type Target = Arc<MemoryManagerInternal>;

    fn deref(&self) -> &Arc<MemoryManagerInternal> {
        &self.0
    }
}

impl Drop for MemoryManager {
    fn drop(&mut self) {
        if Arc::strong_count(&self.0) == 1 {
            SHARESPACE.hiberMgr.RemoveMemMgr(self);
            let _ml = self.MappingWriteLock();
            self.CleanVMAs().unwrap();
        }
    }
}

#[derive(Clone)]
pub struct MemoryManagerWeak {
    pub uid: UniqueID,
    pub data: Weak<MemoryManagerInternal>,
}

impl MemoryManagerWeak {
    pub fn ID(&self) -> UniqueID {
        return self.uid;
    }

    pub fn Upgrade(&self) -> MemoryManager {
        return MemoryManager(self.data.upgrade().expect("MemoryManagerWeak upgrade fail"));
    }
}

impl MemoryManager {
    pub fn Init(kernel: bool) -> Self {
        let mut vmas = AreaSet::New(0, MemoryDef::LOWER_TOP);
        let vma = VMA {
            mappable: MMappable::None,
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
        // kernel memory
        let kernel_range_start;
		#[cfg(target_arch="x86_64")]{
			kernel_range_start = MemoryDef::PHY_LOWER_ADDR;
		}

		#[cfg(target_arch="aarch64")]{
			kernel_range_start = MemoryDef::HYPERCALL_MMIO_BASE;
		}

        let kernel_range_length = MemoryDef::PHY_UPPER_ADDR - kernel_range_start;

        vmas.Insert(
            &gap,
            &Range::New(
                kernel_range_start,
                kernel_range_length,
            ),
            vma.clone(),
        );

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

        let pt = if kernel {
            KERNEL_PAGETABLE.Clone()
        } else {
            KERNEL_PAGETABLE.Fork(&*PAGE_MGR).unwrap()
        };

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
            vcpuMapping: AtomicU64::new(0),
            tlbShootdownMask: AtomicU64::new(0),
            userVDSOBase: AtomicU64::new(0),
            mappingLock: QUpgradableLock::default(),
            mapping: QMutex::new(mapping),
            pagetable: QRwLock::new(pagetable),
            metadataLock: Arc::new(QMutex::new(())),
            metadata: QMutex::new(metadata),
            layout: QMutex::new(layout),
            aioManager: AIOManager::default(),
            membarrierPrivateEnabled: AtomicBool::new(false),
        };

        let mm = Self(Arc::new(internal));

        SHARESPACE.hiberMgr.AddMemMgr(&mm);

        return mm;
    }

    pub fn SetUserVDSOBase(&self, addr: u64) {
        return self.userVDSOBase.store(addr, Ordering::Release);
    }

    pub fn GetUserVDSOBase(&self) -> u64 {
        return self.userVDSOBase.load(Ordering::Acquire);
    }

    pub fn EnableMembarrierPrivate(&self) {
        return self.membarrierPrivateEnabled.store(true, Ordering::Release);
    }

    pub fn IsMembarrierPrivateEnabled(&self) -> bool {
        return self.membarrierPrivateEnabled.load(Ordering::Acquire);
    }

    pub fn Downgrade(&self) -> MemoryManagerWeak {
        return MemoryManagerWeak {
            uid: self.uid,
            data: Arc::downgrade(&self.0),
        };
    }

    pub fn MaskTlbShootdown(&self, vcpuId: u64) {
        self.tlbShootdownMask
            .fetch_or(1 << vcpuId, Ordering::SeqCst);
    }

    pub fn UnmaskTlbShootdown(&self, vcpuId: u64) -> u64 {
        return self
            .tlbShootdownMask
            .fetch_and(!(1 << vcpuId), Ordering::SeqCst);
    }

    pub fn SetVcpu(&self, vcpu: usize) {
        assert!(vcpu < 64);
        self.vcpuMapping.fetch_or(1 << vcpu, Ordering::SeqCst);
    }

    pub fn TlbShootdown(&self) {
        if self.pagetable.read().pt.TlbShootdown() {
            let mask = self.GetVcpuMapping();
            if mask > 0 {
                self.tlbShootdownMask.fetch_or(mask, Ordering::SeqCst);
                let mut interrupt_mask = 0u64;
                let vcpu_len = SHARESPACE.scheduler.VcpuArr.len();
                for i in 0..vcpu_len {
                    if (1 << i) & mask != 0 {
                        if SHARESPACE.scheduler.VcpuArr[i].GetMode() == VcpuMode::User {
                            interrupt_mask = interrupt_mask | (1 << i);
                        }
                    }
                }
                if interrupt_mask != 0 {
                    HostSpace::TlbShootdown(interrupt_mask);
                }
            }

            //CPULocal::Myself().pageAllocator.lock().Clean();
        }

        self.pagetable.read().pt.FreePages();
    }

    pub fn ClearVcpu(&self, vcpu: usize) {
        assert!(vcpu < 64);
        self.vcpuMapping.fetch_and(!(1 << vcpu), Ordering::SeqCst);
    }

    pub fn GetVcpuMapping(&self) -> u64 {
        let mask = self.vcpuMapping.load(Ordering::SeqCst);
        let vcpu = GetVcpuId();
        return mask & !(1 << vcpu);
    }

    pub fn VcpuEnter(&self) {
        let vcpu = GetVcpuId();
        self.SetVcpu(vcpu);
    }

    pub fn VcpuLeave(&self) {
        let vcpu = GetVcpuId();
        self.ClearVcpu(vcpu);
    }

    pub fn MapStackAddr(&self) -> u64 {
        return self.layout.lock().MapStackAddr();
    }

    pub fn SetupStack(&self, stackLayout: &StackLayout, entries: &[AuxEntry]) {
        let mut meta = self.metadata.lock();
        for entry in entries {
            meta.auxv.push(*entry);
        }

        meta.argv = Range::New(
            stackLayout.ArgvStart,
            stackLayout.ArgvEnd - stackLayout.ArgvStart,
        );
        meta.envv = Range::New(
            stackLayout.EnvvStart,
            stackLayout.EvvvEnd - stackLayout.EnvvStart,
        )
    }

    pub fn CleanVMAs(&self) -> Result<()> {
        let mut mapping = self.mapping.lock();
        let (mut vseg, vgap) = mapping.vmas.Find(0);
        if vgap.Ok() {
            vseg = vgap.NextSeg();
            debug!("VM: Start clean VMAs - KEY.0 ->\nvseg:{:?}\nvgap:{:?}", vseg.Value(), vgap.Range());
        } else {
            debug!("VM: No VMAs to cleanup.");
        }

        while vseg.Ok() {
            let r = vseg.Range();
            let vma = vseg.Value();

            if !vma.kernel {
                vma.mappable
                    .RemoveMapping(self, &r, vma.offset, vma.CanWriteMappableLocked())?;
            }
            let vgap = mapping.vmas.Remove(&vseg);
            vseg = vgap.NextSeg();
        }

        return Ok(());
    }

    //Remove virtual memory to the phy mem mapping
    pub fn MFree(&self, ar: &Range) -> Result<()> {
        let mut mapping = self.mapping.lock();
        let (mut vseg, vgap) = mapping.vmas.Find(ar.Start());
        if vgap.Ok() {
            vseg = vgap.NextSeg();
        }

        while vseg.Ok() && vseg.Range().Start() < ar.End() {
            vseg = mapping.vmas.Isolate(&vseg, &ar);
            let r = vseg.Range();
            let vma = vseg.Value();

            if !vma.kernel {
                /*if vma.mappable.is_some() {
                    let mappable = vma.mappable.clone().unwrap();
                    // todo: fix the Madvise/MADV_DONTNEED, when there are multiple process MAdviseOp::MADV_DONTNEED
                    // with current implementation, the first Madvise/MADV_DONTNEED will work.
                    mappable.RemoveMapping(self, &r, vma.offset, vma.CanWriteMappableLocked())?;
                }

                mapping.usageAS -= r.Len();
                if vma.mlockMode != MLockMode::MlockNone {
                    mapping.lockedAS -= r.Len();
                }*/

                let mut pt = self.pagetable.write();

                pt.pt.MUnmap(r.Start(), r.Len())?;
                pt.curRSS -= r.Len();
            }
            //let vgap = mapping.vmas.Remove(&vseg);
            vseg = vgap.NextSeg();
        }

        return Ok(());
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

            if !vma.kernel {
                vma.mappable
                    .RemoveMapping(self, &r, vma.offset, vma.CanWriteMappableLocked())?;

                mapping.usageAS -= r.Len();
                if vma.mlockMode != MLockMode::MlockNone {
                    mapping.lockedAS -= r.Len();
                }

                let mut pt = self.pagetable.write();

                pt.pt.MUnmap(r.Start(), r.Len())?;
                pt.curRSS -= r.Len();
            }
            let vgap = mapping.vmas.Remove(&vseg);
            vseg = vgap.NextSeg();
        }

        return Ok(());
    }

    // SHOULD be called before return to user space,
    // to make sure the tlb flushed
    pub fn HandleTlbShootdown(&self) {
        let localTLBEpoch = CPULocal::Myself().tlbEpoch.load(Ordering::Acquire);
        let currTLBEpoch = self.TLBEpoch();

        if localTLBEpoch != currTLBEpoch {
            CPULocal::Myself()
                .tlbEpoch
                .store(currTLBEpoch, Ordering::Release);
            #[cfg(target_arch = "aarch64")]
            let curr = CurrentUserTable();
            #[cfg(target_arch = "x86_64")]
            let curr = CurrentCr3();
            PageTables::Switch(curr);
        }
    }

    pub fn MappingReadLock(&self) -> QUpgradableLockGuard {
        let lock = self.mappingLock.Read();
        return lock;
    }

    pub fn MappingWriteLock(&self) -> QUpgradableLockGuard {
        let lock = self.mappingLock.Write();
        return lock;
    }

    pub fn BrkSetup(&self, addr: u64) {
        let _ml = self.MappingReadLock();
        self.mapping.lock().brkInfo = BrkInfo {
            brkStart: addr,
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

        let res = format!(
            "{} {} 0 0 0 0 0\n",
            vss / MemoryDef::PAGE_SIZE,
            rss / MemoryDef::PAGE_SIZE
        );

        return res.as_bytes().to_vec();
    }

    pub const DEV_MINOR_BITS: usize = 20;
    pub const VSYSCALLEND: u64 = 0xffffffffff601000;
    pub const VSYSCALL_MAPS_ENTRY: &'static str =
        "ffffffffff600000-ffffffffff601000 --xp 00000000 00:00 0                  [vsyscall]\n";

    pub fn PrintVma(&self, task: &Task, vma: &VMA, range: &Range) -> String {
        let private = if vma.private { "p" } else { "s" };

        let (dev, inodeId) = match &vma.id {
            None => (0, 0),
            Some(ref mapping) => (mapping.DeviceID(), mapping.InodeID()),
        };

        let devMajor = (dev >> Self::DEV_MINOR_BITS) as u32;
        let devMinor = (dev & ((1 << Self::DEV_MINOR_BITS) - 1)) as u32;

        let mut s = if vma.hint.len() == 0 {
            vma.hint.to_string()
        } else {
            match &vma.id {
                None => "".to_string(),
                //todo: seems that mappedName doesn't work. Fix it
                Some(ref id) => id.MappedName(task),
            }
        };

        let str = format!(
            "{:08x}-{:08x} {}{} {:08x} {:02x}:{:02x} {} ",
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

        return str + &s;
    }

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

            let private = if vma.private { "p" } else { "s" };

            let (dev, inodeId) = match &vma.id {
                None => (0, 0),
                Some(ref mapping) => (mapping.DeviceID(), mapping.InodeID()),
            };

            let devMajor = (dev >> Self::DEV_MINOR_BITS) as u32;
            let devMinor = (dev & ((1 << Self::DEV_MINOR_BITS) - 1)) as u32;

            let mut s = if vma.hint.len() == 0 {
                vma.hint.to_string()
            } else {
                match &vma.id {
                    None => "".to_string(),
                    //todo: seems that mappedName doesn't work. Fix it
                    Some(ref id) => id.MappedName(task),
                }
            };

            let str = format!(
                "{:08x}-{:08x} {}{} {:08x} {:02x}:{:02x} {} ",
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

        #[cfg(target_arch = "x86_64")]
        {
            ret += Self::VSYSCALL_MAPS_ENTRY;
        }

        return ret;
    }

    pub fn GenMapsSnapshot(&self, task: &Task) -> Vec<u8> {
        let ret = self.GetSnapshotLocked(task, true);

        return ret.as_bytes().to_vec();
    }

    pub fn SetExecutable(&self, dirent: &Dirent) {
        self.metadata.lock().executable = Some(dirent.clone());
    }

    pub fn MinCore(&self, _task: &Task, r: &Range) -> Vec<u8> {
        let _ml = self.MappingReadLock();

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

    pub fn mlockedBytesRangeLocked(&self, mr: &Range) -> u64 {
        let mut total = 0;
        let mapping = self.mapping.lock();
        let mut seg = mapping.vmas.LowerBoundSeg(mr.Start());
        while seg.Ok() && seg.Range().Start() < mr.End() {
            let vma = seg.Value();
            if vma.mlockMode != MLockMode::MlockNone {
                let segMR = seg.Range();
                total += segMR.Intersect(&mr).Len();
            }
            seg = seg.NextSeg();
        }

        return total;
    }

    // MLock implements the semantics of Linux's mlock()/mlock2()/munlock(),
    // depending on mode.
    pub fn Mlock(&self, task: &Task, addr: u64, len: u64, mode: MLockMode) -> Result<()> {
        let la = match Addr(len + Addr(addr).PageOffset()).RoundUp() {
            Ok(l) => l.0,
            Err(_) => return Err(Error::SysError(SysErr::EINVAL)),
        };

        let ar = match Addr(addr).RoundDown().unwrap().ToRange(la) {
            Ok(r) => r,
            Err(_) => return Err(Error::SysError(SysErr::EINVAL)),
        };

        let _ml = self.MappingWriteLock();

        if mode != MLockMode::MlockNone {
            let creds = task.creds.clone();
            let userns = creds.lock().UserNamespace.Root();
            if !creds.HasCapabilityIn(Capability::CAP_IPC_LOCK, &userns) {
                let mlockLimit = task
                    .Thread()
                    .ThreadGroup()
                    .Limits()
                    .Get(LimitType::MemoryLocked)
                    .Cur;
                if mlockLimit == 0 {
                    return Err(Error::SysError(SysErr::EPERM));
                }

                let lockedAS = self.mapping.lock().lockedAS;
                let newLockedAS = lockedAS + ar.Len() + self.mlockedBytesRangeLocked(&ar);
                if newLockedAS > mlockLimit {
                    return Err(Error::SysError(SysErr::ENOMEM));
                }
            }
        }

        if ar.Len() == 0 {
            return Ok(());
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
            return Err(Error::SysError(SysErr::ENOMEM));
        }

        // todo: populate the pagetable
        // if mode == MLockMode::MlockEager {}

        let mut vseg = mapping.vmas.FindSeg(ar.Start());
        while vseg.Ok() && vseg.Range().Start() < ar.End() {
            let vma = vseg.Value();
            // Linux: mm/gup.c:__get_user_pages() returns EFAULT in this
            // case, which is converted to ENOMEM by mlock.
            if !vma.effectivePerms.Any() {
                return Err(Error::SysError(SysErr::ENOMEM));
            }

            if let Some(iops) = vma.mappable.HostIops() {
                let mr = ar.Intersect(&vseg.Range());
                let fstart = mr.Start() - vseg.Range().Start() + vma.offset;

                // todo: fix the Munlock, when there are multiple process lock/unlock a memory range.
                // with current implementation, the first unlock will work.
                iops.Mlock(fstart, mr.Len(), mode)?;
            }

            vseg = vseg.NextSeg()
        }

        return Ok(());
    }

    // MLockAll implements the semantics of Linux's mlockall()/munlockall(),
    // depending on opts.
    pub fn MlockAll(&self, task: &Task, opts: &MLockAllOpts) -> Result<()> {
        if !opts.Current && !opts.Future {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        // todo: fully support opts.Current and opts.Future
        // it is not supported now
        let mode = opts.Mode;
        let _ml = self.MappingWriteLock();

        if opts.Current {
            if mode != MLockMode::MlockNone {
                let creds = task.creds.clone();
                let userns = creds.lock().UserNamespace.Root();
                if !creds.HasCapabilityIn(Capability::CAP_IPC_LOCK, &userns) {
                    let mlockLimit = task
                        .Thread()
                        .ThreadGroup()
                        .Limits()
                        .Get(LimitType::MemoryLocked)
                        .Cur;
                    if mlockLimit == 0 {
                        return Err(Error::SysError(SysErr::EPERM));
                    }

                    if self.mapping.lock().vmas.Span() > mlockLimit {
                        return Err(Error::SysError(SysErr::EPERM));
                    }
                }
            }
        }

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

            if let Some(iops) = vma.mappable.HostIops() {
                let mr = vseg.Range();
                let fstart = mr.Start() - vseg.Range().Start() + vma.offset;

                // todo: fix the Munlock, when there are multiple process lock/unlock a memory range.
                // with current implementation, the first unlock will work.
                iops.Mlock(fstart, mr.Len(), mode)?;
            }

            vseg = vseg.NextSeg();
        }

        return Ok(());
    }

    pub fn MSync(&self, _task: &Task, addr: u64, length: u64, opts: &MSyncOpts) -> Result<()> {
        if addr != Addr(addr).RoundDown()?.0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        if length == 0 {
            return Ok(());
        }

        let la = match Addr(length).RoundUp() {
            Err(_) => return Err(Error::SysError(SysErr::ENOMEM)),
            Ok(l) => l.0,
        };

        let ar = Range::New(addr, la);

        let _ml = self.MappingReadLock();

        let mapping = self.mapping.lock();
        let mut vseg = mapping.vmas.LowerBoundSeg(ar.Start());

        if !vseg.Range().Contains(ar.Start()) {
            return Err(Error::SysError(SysErr::ENOMEM));
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
                return Err(Error::SysError(SysErr::EBUSY));
            }

            // It's only possible to have dirtied the Mappable through a shared
            // mapping. Don't check if the mapping is writable, because mprotect
            // may have changed this, and also because Linux doesn't.
            if vma.mappable.HostIops().is_some() && !vma.private {
                let msyncType = if opts.Sync {
                    MSyncType::MsSync
                } else {
                    MSyncType::MsAsync
                };

                let fops = vma.mappable.HostIops().unwrap();

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
            return Err(Error::SysError(SysErr::ENOMEM));
        }

        return Ok(());
    }

    pub fn SetMmapLayout(
        &self,
        minUserAddr: u64,
        maxUserAddr: u64,
        r: &LimitSet,
    ) -> Result<MmapLayout> {
        let layout = Context64::NewMmapLayout(minUserAddr, maxUserAddr, r)?;
        *self.layout.lock() = layout;
        return Ok(layout);
    }

    pub fn TLBEpoch(&self) -> u64 {
        return self.pagetable.read().pt.TLBEpoch();
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

        return Some((vseg.Value(), vseg.Range()));
    }

    pub fn MapPageLocked(
        &self,
        vaddr: Addr,
        phyAddr: Addr,
        flags: PageTableFlags,
    ) -> Result<(PageTableEntry, bool)> {
        let pt = self.pagetable.write();
        return pt.pt.MapPage(vaddr, phyAddr, flags, &*PAGE_MGR);
    }

    pub fn VirtualToPhy(&self, vAddr: u64) -> Result<(u64, AccessType)> {
        let _ml = self.MappingReadLock();

        if vAddr == 0 {
            return Err(Error::SysError(SysErr::EFAULT));
        }

        let pagetable = self.pagetable.read();
        return pagetable.pt.VirtualToPhy(vAddr);
    }

    pub fn VirtualToPhyLocked(&self, vAddr: u64) -> Result<(u64, AccessType)> {
        if vAddr == 0 {
            return Err(Error::SysError(SysErr::EFAULT));
        }

        let pagetable = self.pagetable.read();
        return pagetable.pt.VirtualToPhy(vAddr);
    }

    pub fn VirtualToEntryLocked(&self, vAddr: u64) -> Result<PageTableEntry> {
        if vAddr == 0 {
            return Err(Error::SysError(SysErr::EFAULT));
        }

        let pagetable = self.pagetable.read();
        let entry = pagetable.pt.VirtualToEntry(vAddr)?;
        return Ok(entry.clone());
    }
    pub fn InstallPageLocked(
        &self,
        task: &Task,
        vma: &VMA,
        pageAddr: u64,
        range: &Range,
        needWrite: bool,
    ) -> Result<(PageTableEntry, bool)> {
        match self.VirtualToEntryLocked(pageAddr) {
            Err(_) => (),
            Ok(entry) => return Ok((entry, true)),
        }

        if !vma.effectivePerms.Any() {
            return Err(Error::SysError(SysErr::EFAULT));
        }

        let exec = vma.effectivePerms.Exec();
        match &vma.mappable.HostIops() {
            Some(iops) => {
                let vmaOffset = pageAddr - range.Start();
                let fileOffset = vmaOffset + vma.offset; // offset in the file
                debug!("VM: Install Page - HostIO - vma-offset:{:#0x} - file-offset:{:#0x}", vmaOffset, fileOffset);
                let phyAddr = iops.MapFilePage(task, fileOffset)?;
                if is_cc_active() {
                    let writeable = vma.effectivePerms.Write();
                    let page = { super::super::PAGE_MGR.AllocPage(true).unwrap() };
                    debug!("VM: Install Page - copy pha:{:#0x} to page:{:#0x}", phyAddr, page);
                    CopyPage(page, phyAddr);
                    debug!("VM: Install Page - copy done.");

                    let ret;
                    if writeable {
                        ret = self.MapPageWriteLocked(pageAddr, page, exec);
                    } else {
                        ret = self.MapPageReadLocked(pageAddr, page, exec);
                    }

                    if !vma.private {
                        iops.MapSharedPage(phyAddr, page, fileOffset, writeable);
                    }
                    super::super::PAGE_MGR.DerefPage(page);
                    return ret;
                }
                //error!("fault 2.1, vma.mappable.is_some() is {}, vaddr is {:x}, paddr is {:x}",
                //      vma.mappable.is_some(), pageAddr, phyAddr);

                if vma.private {
                    let writeable = vma.effectivePerms.Write();

                    if needWrite && writeable {
                        let page = { super::super::PAGE_MGR.AllocPage(true).unwrap() };
                        CopyPage(page, phyAddr);
                        let ret = self.MapPageWriteLocked(pageAddr, page, exec);
                        super::super::PAGE_MGR.DerefPage(page);
                        return ret;
                    } else {
                        return self.MapPageReadLocked(pageAddr, phyAddr, exec);
                    }
                } else {
                    let writeable = vma.effectivePerms.Write();
                    if writeable {
                        return self.MapPageWriteLocked(pageAddr, phyAddr, exec);
                    } else {
                        return self.MapPageReadLocked(pageAddr, phyAddr, exec);
                    }
                }
            }
            None => {
                // for mmappable socket
                match vma.mappable.ByteStream() {
                    Some(b) => {
                        let vmaOffset = pageAddr - range.Start();
                        let fileOffset = vmaOffset + vma.offset; // offset in the file
                        let (phyAddr, len) = b.lock().GetRawBuf();
                        assert!(len - 4096 >= fileOffset as usize);
                        let phyAddr = phyAddr + fileOffset;
                        let writeable = vma.effectivePerms.Write();
                        if writeable {
                            return self.MapPageWriteLocked(pageAddr, phyAddr, exec);
                        } else {
                            return self.MapPageWriteLocked(pageAddr, phyAddr, exec);
                        }
                    }
                    None => (),
                }

                //let vmaOffset = pageAddr - range.Start();
                //let phyAddr = vmaOffset + vma.offset; // offset in the phyAddr

                let phyAddr = super::super::PAGE_MGR.AllocPage(true).unwrap();
                let writeable = vma.effectivePerms.Write();
                if writeable {
                    let ret = self.MapPageWriteLocked(pageAddr, phyAddr, exec);
                    super::super::PAGE_MGR.DerefPage(phyAddr);
                    return ret;
                } else {
                    let ret = self.MapPageReadLocked(pageAddr, phyAddr, exec);
                    super::super::PAGE_MGR.DerefPage(phyAddr);
                    return ret;
                }
            }
        }
    }

    pub fn MapPageWriteLocked(
        &self,
        vAddr: u64,
        pAddr: u64,
        exec: bool,
    ) -> Result<(PageTableEntry, bool)> {
        let pt = self.pagetable.write();
        return pt.pt.MapPage(
            Addr(vAddr),
            Addr(pAddr),
            PageOpts::New(true, true, exec).Val(),
            &*PAGE_MGR,
        );
    }

    pub fn MapPageReadLocked(
        &self,
        vAddr: u64,
        pAddr: u64,
        exec: bool,
    ) -> Result<(PageTableEntry, bool)> {
        let pt = self.pagetable.write();
        return pt.pt.MapPage(
            Addr(vAddr),
            Addr(pAddr),
            PageOpts::New(true, false, exec).Val(),
            &*PAGE_MGR,
        );
    }

    pub fn EnableWriteLocked(&self, addr: u64, exec: bool) {
        let pt = self.pagetable.write();
        pt.pt
            .SetPageFlags(Addr(addr), PageOpts::New(true, true, exec).Val());
    }

    pub fn CopyOnWriteLocked(&self, pageAddr: u64, vma: &VMA) -> u64 {
        let (phyAddr, permission) = self
            .VirtualToPhyLocked(pageAddr)
            .expect(&format!("addr is {:x}", pageAddr));

        if permission.Write() {
            // another thread has cow, return
            Invlpg(pageAddr);
            return phyAddr;
        }

        let exec = vma.effectivePerms.Exec();
        let page = { super::super::PAGE_MGR.AllocPage(false).unwrap() };
        CopyPage(page, phyAddr);
        let _ = self.MapPageWriteLocked(pageAddr, page, exec);
        return page;
    }

    pub fn CopyOnWrite(&self, pageAddr: u64, vma: &VMA) {
        let _ml = self.MappingWriteLock();

        //PerfGoto(PerfType::PageFault);
        self.CopyOnWriteLocked(pageAddr, vma);
        //PerfGofrom(PerfType::PageFault);
    }

    pub fn V2P(
        &self,
        task: &Task,
        start: u64,
        len: u64,
        writable: bool,
        allowPartial: bool,
    ) -> Result<Vec<IoVec>> {
        if len == 0 {
            return Ok(Vec::new());
        }

        if start == 0 {
            return Err(Error::SysError(SysErr::EFAULT));
        }

        let rl = self.MappingReadLock();

        return self.V2PLocked(task, &rl, start, len, writable, allowPartial);
    }

    pub fn FixPermissionForIovs(&self, task: &Task, iovs: &[IoVec], writable: bool) -> Result<()> {
        let rl = self.MappingReadLock();
        for iov in iovs {
            self.FixPermissionLocked(
                task,
                &rl,
                iov.Start(),
                iov.Len() as u64,
                writable,
                None,
                false,
            )?;
        }

        return Ok(());
    }

    pub fn V2PLocked(
        &self,
        task: &Task,
        rlock: &QUpgradableLockGuard,
        start: u64,
        len: u64,
        writable: bool,
        allowPartial: bool,
    ) -> Result<Vec<IoVec>> {
        if MemoryDef::PHY_LOWER_ADDR <= start && start <= MemoryDef::PHY_UPPER_ADDR {
            // Kernel phy address
            let end = start + len;
            assert!(MemoryDef::PHY_LOWER_ADDR <= end && end <= MemoryDef::PHY_UPPER_ADDR);
            let mut output = Vec::with_capacity(1);
            output.push(IoVec {
                start: start,
                len: len as usize,
            });
            return Ok(output);
        }

        let mut output =
            Vec::with_capacity(((len + MemoryDef::PAGE_SIZE - 1) / MemoryDef::PAGE_SIZE) as usize);
        let len = self.FixPermissionLocked(
            task,
            rlock,
            start,
            len,
            writable,
            Some(&mut output),
            allowPartial,
        )?;

        if output.len() > 0 {
            let offset = Addr(start).PageOffset();
            output[0].start += offset;
            output[0].len -= offset as usize;

            let last = output.len() - 1;
            let end: u64 = start + len;
            let endoffset = Addr(end).PageOffset();
            if endoffset != 0 {
                output[last].len -= (MemoryDef::PAGE_SIZE - endoffset) as usize;
            }
        }

        return Ok(output);
    }

    // check whether the address range is legal.
    // 1. whether the range belong to user's space
    // 2. Whether the read/write permission meet requirement
    // 3. if need cow, fix the page.
    // 4. return max allowed len
    pub fn FixPermission(
        &self,
        task: &Task,
        vAddr: u64,
        len: u64,
        writeReq: bool,
        allowPartial: bool,
    ) -> Result<u64> {
        if core::u64::MAX - vAddr < len || vAddr == 0 {
            return Err(Error::SysError(SysErr::EFAULT));
        }

        let rl = self.MappingReadLock();

        self.FixPermissionLocked(task, &rl, vAddr, len, writeReq, None, allowPartial)
    }

    pub fn Pin(&self, task: &Task, vAddr: u64, len: u64) -> Result<Vec<Range>> {
        let rl = self.MappingReadLock();
        return self.PinLocked(task, &rl, vAddr, len);
    }

    pub fn PinLocked(
        &self,
        task: &Task,
        rlock: &QUpgradableLockGuard,
        vAddr: u64,
        len: u64,
    ) -> Result<Vec<Range>> {
        self.FixPermissionLocked(task, rlock, vAddr, len, false, None, false)?;

        let startAddr = vAddr;
        let mut vaddr = vAddr;
        let mut lastAddr = 0;
        let mut lastLen = 0;
        let mut ranges = Vec::new();
        while vaddr < startAddr + len {
            let (paddr, _) = self.VirtualToPhyLocked(vaddr)?;
            super::super::PAGE_MGR.RefPage(paddr);
            if paddr != lastAddr + lastLen {
                //+ MemoryDef::PAGE_SIZE {
                if lastAddr != 0 {
                    ranges.push(Range::New(lastAddr, lastLen));
                }

                lastAddr = paddr;
                lastLen = MemoryDef::PAGE_SIZE;
            } else {
                lastLen += MemoryDef::PAGE_SIZE;
            }

            vaddr += MemoryDef::PAGE_SIZE;
        }

        assert!(lastAddr != 0 && lastLen != 0);
        ranges.push(Range::New(lastAddr, lastLen));
        return Ok(ranges);
    }

    pub fn AppendOutput(output: &mut Vec<IoVec>, paddr: u64) {
        let iov = IoVec {
            start: paddr,
            len: MemoryDef::PAGE_SIZE as usize,
        };

        let cnt = output.len();
        if cnt > 0 && output[cnt - 1].End() == iov.start {
            // use the last entry
            output[cnt - 1].len += iov.len;
        } else {
            output.push(iov);
        }
    }

    // check whether the address range is legal.
    // 1. whether the range belong to user's space
    // 2. Whether the read/write permission meet requirement
    // 3. if need cow, fix the page.
    // 4. return max allowed len
    pub fn FixPermissionLocked(
        &self,
        task: &Task,
        rlock: &QUpgradableLockGuard,
        vAddr: u64,
        len: u64,
        writeReq: bool,
        paddrRanges: Option<&mut Vec<IoVec>>,
        allowPartial: bool,
    ) -> Result<u64> {
        assert!(!rlock.Writable());

        defer!({
            if rlock.Writable() {
                rlock.Downgrade()
            }
        });

        if (vAddr as i64) < 0 {
            return Err(Error::SysError(SysErr::EFAULT));
        }

        // todo: fix the security check issue
        if MemoryDef::PHY_LOWER_ADDR <= vAddr && vAddr <= MemoryDef::PHY_UPPER_ADDR {
            // Kernel phy address
            let end = vAddr + len;
            assert!(MemoryDef::PHY_LOWER_ADDR <= end && end <= MemoryDef::PHY_UPPER_ADDR);
            return Ok(len);
        }

        let mut addr = Addr(vAddr).RoundDown()?.0;
        let mut needTLBShootdown = false;

        let mut vma = VMA::default();
        let mut range = Range::default();
        let mut vec = Vec::new();

        let (output, needOutput) = match paddrRanges {
            Some(output) => (output, true),
            None => (&mut vec, false),
        };

        while addr <= vAddr + len - 1 {
            let (paddr, permission) = match self.VirtualToPhyLocked(addr) {
                Err(Error::AddressNotMap(_)) => {
                    if !rlock.Writable() {
                        rlock.Upgrade();
                    }

                    if addr >= range.End() {
                        match self.GetVmaAndRangeLocked(addr) {
                            None => {
                                if !allowPartial || addr < vAddr {
                                    return Err(Error::SysError(SysErr::EFAULT));
                                }

                                if needTLBShootdown {
                                    self.TlbShootdown();
                                }

                                return Ok(addr - vAddr);
                            }
                            Some((v, r)) => {
                                vma = v;
                                range = r;
                            }
                        };
                    }

                    let entry = match self.InstallPageLocked(task, &vma, addr, &range, writeReq) {
                        Err(_) => {
                            if !allowPartial || addr < vAddr {
                                return Err(Error::SysError(SysErr::EFAULT));
                            }

                            if needTLBShootdown {
                                self.TlbShootdown();
                            }
                            return Ok(addr - vAddr);
                        }
                        Ok((entry, _)) => entry,
                    };

                    (
                        guest_physical_address(entry.addr().as_u64()),
                        AccessType::NewFromPageFlags(entry.flags()),
                    )
                }
                Err(e) => return Err(e),
                Ok(ret) => ret,
            };

            if !permission.Read() {
                // No read permission
                if !allowPartial || addr < vAddr {
                    return Err(Error::SysError(SysErr::EFAULT));
                }

                if needTLBShootdown {
                    self.TlbShootdown();
                }

                if needOutput {
                    Self::AppendOutput(output, paddr);
                }
                return Ok(addr - vAddr);
            }

            if writeReq && !permission.Write() {
                if addr >= range.End() {
                    match self.GetVmaAndRangeLocked(addr) {
                        None => {
                            if !allowPartial || addr < vAddr {
                                return Err(Error::SysError(SysErr::EFAULT));
                            }

                            if needTLBShootdown {
                                self.TlbShootdown();
                            }

                            if needOutput {
                                Self::AppendOutput(output, paddr);
                            }

                            return Ok(addr - vAddr);
                        }
                        Some((v, r)) => {
                            vma = v;
                            range = r;
                        }
                    };
                }

                if vma.effectivePerms.Write() {
                    if !rlock.Writable() {
                        rlock.Upgrade();
                    }

                    let paddr = self.CopyOnWriteLocked(addr, &vma);

                    if needOutput {
                        Self::AppendOutput(output, paddr);
                    }
                    needTLBShootdown = true;
                } else {
                    if !allowPartial || addr < vAddr {
                        return Err(Error::SysError(SysErr::EFAULT));
                    }

                    if needTLBShootdown {
                        self.TlbShootdown();
                    }

                    if needOutput {
                        Self::AppendOutput(output, paddr);
                    }
                    return Ok(addr - vAddr);
                }
            } else {
                if needOutput {
                    Self::AppendOutput(output, paddr);
                }
            }

            addr += MemoryDef::PAGE_SIZE;
        }

        if needTLBShootdown {
            self.TlbShootdown();
        }
        return Ok(len);
    }

    pub fn PopulateVMALocked(
        &self,
        task: &Task,
        vmaSeg: &AreaSeg<VMA>,
        ar: &Range,
        precommit: bool,
        vdso: bool,
    ) -> Result<()> {
        let vma = vmaSeg.Value();
        let mut perms = vma.effectivePerms;

        //if it is filemapping and private, need cow.
        // if it is anon share, first marks it as writeable. When clone, mark it as readonly.
        if vma.private & vma.mappable.HostIops().is_some() {
            perms.ClearWrite();
        }

        self.pagetable.write().pt.MUnmap(ar.Start(), ar.Len())?;
        let segAr = vmaSeg.Range();
        let iops = match &vma.mappable {
            MMappable::None => {
                //anonymous mapping
                self.AddRssLock(ar);
                if !vdso {
                    //
                } else {
                    //vdso: the phyaddress has been allocated and the address is vma.offset
                    self.pagetable.write().pt.MapHost(
                        task,
                        ar.Start(),
                        &IoVec::NewFromAddr(vma.offset, ar.Len() as usize),
                        &perms,
                        true,
                    )?;
                }
                return Ok(());
            }
            MMappable::HostIops(iops) => iops.clone(),
            MMappable::Shm(shm) => shm.HostIops(),
            _ => {
                return Err(Error::SysError(SysErr::EINVAL));
            }
        };

        //host file mapping
        // the map file mapfile cost is high. Only pre-commit it when the size < 4MB.
        // todo: improve that later
        let currPerm = if !vma.private {
            perms
        } else {
            // disable write for private range, so that it could trigger cow in pagefault
            AccessType(perms.0 & !MmapProt::PROT_WRITE)
        };

        if precommit && segAr.Len() < 0x200000 {
            self.pagetable.write().pt.MapFile(
                task,
                ar.Start(),
                &iops,
                &Range::New(vma.offset + ar.Start() - segAr.Start(), ar.Len()),
                &currPerm,
                precommit,
            )?;
        }
        self.AddRssLock(ar);

        return Ok(());
    }

    pub fn PopulateVMARemapLocked(
        &self,
        task: &Task,
        vmaSeg: &AreaSeg<VMA>,
        ar: &Range,
        oldar: &Range,
        _precommit: bool,
    ) -> Result<()> {
        //let segAr = vmaSeg.Range();
        let vma = vmaSeg.Value();
        let mut perms = vma.effectivePerms;

        if vma.private & vma.mappable.HostIops().is_some() {
            //if it is filemapping and private, need cow.
            perms.ClearWrite();
        }

        let pt = self.pagetable.write();

        let len = if ar.Len() > oldar.Len() {
            oldar.Len()
        } else {
            ar.Len()
        };

        // todo: change the name to pt.Remap
        pt.pt.RemapAna(
            task,
            &Range::New(ar.Start(), len),
            oldar.Start(),
            &perms,
            true,
        )?;

        return Ok(());
    }

    pub fn ApplicationAddrRange(&self) -> Range {
        let layout = self.layout.lock();
        return Range::New(layout.MinAddr, layout.MaxAddr - layout.MinAddr);
    }

    pub fn Fork(&self) -> Result<Self> {
        let _ml = self.MappingWriteLock();
        let layout = *self.layout.lock();
        let mmIntern2 = MemoryManagerInternal {
            uid: NewUID(),
            inited: true,
            layout: QMutex::new(layout),
            metadata: QMutex::new(self.metadata.lock().Fork()),
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

            let vdsoAddr = self.GetUserVDSOBase();
            mm2.SetUserVDSOBase(vdsoAddr);

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

                match vma.mappable.AddMapping(
                    &mm2,
                    &vmaAR,
                    vma.offset,
                    vma.CanWriteMappableLocked(),
                ) {
                    Err(e) => {
                        let appRange = mm2.ApplicationAddrRange();
                        mm2.RemoveVMAsLocked(&appRange)?;
                        return Err(e);
                    }
                    _ => (),
                }

                vma.mlockMode = MLockMode::MlockNone;

                if vma.kernel == false {
                    //info!("vma kernel is {}, private is {}, hint is {}", vma.kernel, vma.private, vma.hint);
                    if vma.private {
                        //cow
                        ptInternal1.pt.ForkRange(
                            &ptInternal2.pt,
                            vmaAR.Start(),
                            vmaAR.Len(),
                            &*PAGE_MGR,
                        )?;
                    } else {
                        ptInternal1.pt.CopyRange(
                            &ptInternal2.pt,
                            vmaAR.Start(),
                            vmaAR.Len(),
                            &*PAGE_MGR,
                        )?;
                    }
                }

                dstvgap = mappingInternal2
                    .vmas
                    .Insert(&dstvgap, &vmaAR, vma)
                    .NextGap();

                let tmp = srcvseg.NextSeg();
                srcvseg = tmp;
            }
        }

        self.TlbShootdown();

        SHARESPACE.hiberMgr.AddMemMgr(&mm2);
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
}

// MLockAllOpts holds options to MLockAll.
pub struct MLockAllOpts {
    // If Current is true, change the memory-locking behavior of all mappings
    // to Mode. If Future is true, upgrade the memory-locking behavior of all
    // future mappings to Mode. At least one of Current or Future must be true.
    pub Current: bool,
    pub Future: bool,
    pub Mode: MLockMode,
}
