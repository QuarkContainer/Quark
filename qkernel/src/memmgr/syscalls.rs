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

use core::u64;

use super::super::kernel::futex::*;
use super::super::memmgr::mm::*;
use super::super::memmgr::vma::*;
use super::super::task::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::addr::*;
use super::super::qlib::range::*;
use super::super::qlib::linux::limits::*;
use super::*;

#[derive(Debug)]
pub struct MSyncOpts {
    // Sync has the semantics of MS_SYNC.
    pub Sync: bool,

    // Invalidate has the semantics of MS_INVALIDATE.
    pub Invalidate: bool,
}

impl MemoryManager {
    // MMap establishes a memory mapping.
    pub fn MMap(&self, task: &Task, opts: &mut MMapOpts) -> Result<u64> {
        let lock = self.Lock();
        let _l = lock.lock();

        if opts.Length == 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let length = match Addr(opts.Length).RoundUp() {
            Err(_) => return Err(Error::SysError(SysErr::ENOMEM)),
            Ok(l) => l.0,
        };

        opts.Length = length;

        if opts.Mappable.is_some() {
            // Offset must be aligned.
            if Addr(opts.Offset).RoundDown()?.0 != opts.Offset {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            // Offset + length must not overflow.
            if u64::MAX - opts.Length < opts.Offset {
                return Err(Error::SysError(SysErr::ENOMEM));
            }
        } else if !opts.VDSO { //not vdso
            opts.Offset = 0;
        }

        if Addr(opts.Addr).RoundDown()?.0 != opts.Addr {
            // MAP_FIXED requires addr to be page-aligned; non-fixed mappings
            // don't.
            if opts.Fixed {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            opts.Addr = Addr(opts.Addr).RoundDown()?.0;
        }

        if !opts.MaxPerms.SupersetOf(&opts.Perms) {
            return Err(Error::SysError(SysErr::EACCES));
        }

        if opts.Unmap && !opts.Fixed {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        if opts.GrowsDown && opts.Mappable.is_some() {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let mm = self;
        let (vseg, ar) = mm.write().CreateVMAlocked(task, opts, self)?;

        mm.PopulateVMA(task, &vseg, &ar, opts.Precommit, opts.VDSO)?;

        return Ok(ar.Start());
    }

    // MapStack allocates the initial process stack.
    pub fn MapStack(&self, task: &Task) -> Result<Range> {
        let lock = self.Lock();
        let _l = lock.lock();

        // maxStackSize is the maximum supported process stack size in bytes.
        //
        // This limit exists because stack growing isn't implemented, so the entire
        // process stack must be mapped up-front.
        const MAX_STACK_SIZE: u64 = 128 << 20; //128 MB

        let sz = DEFAULT_STACK_SOFT_LIMIT;

        let mm = self;
        //todo: add random
        // stackEnd := mm.layout.MaxAddr - usermem.Addr(mrand.Int63n(int64(mm.layout.MaxStackRand))).RoundDown()
        let stackEnd = mm.write().layout.MaxAddr;

        if stackEnd < sz {
            return Err(Error::SysError(SysErr::ENOMEM));
        }

        let stackStart = stackEnd - sz;
        let (vseg, ar) = mm.write().CreateVMAlocked(task, &MMapOpts {
            Length: sz,
            Addr: stackStart,
            Offset: 0,
            Fixed: true,
            Unmap: false,
            Map32Bit: false,
            Perms: AccessType::ReadWrite(),
            MaxPerms: AccessType::AnyAccess(),
            Private: true,
            VDSO: false,
            GrowsDown: true,
            Precommit: false,
            MLockMode: MLockMode::default(),
            Kernel: false,
            Mapping: None,
            Mappable: None,
            Hint: "[stack]".to_string(),
        }, self)?;

        mm.PopulateVMA(task, &vseg, &ar, false, false)?;

        return Ok(ar)
    }

    // MUnmap implements the semantics of Linux's munmap(2).
    pub fn MUnmap(&self, _task: &Task, addr: u64, length: u64) -> Result<()> {
        let lock = self.Lock();
        let _l = lock.lock();

        if addr != Addr(addr).RoundDown()?.0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        if length == 0 {
            //|| length != Addr(length).RoundDown()?.0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let length = Addr(length).RoundUp()?.0;

        let ar = Addr(addr).ToRange(length)?;

        return self.write().RemoveVMAsLocked(self, &ar);
    }

    // MRemap implements the semantics of Linux's mremap(2).
    pub fn MRemap(&self, task: &Task, oldAddr: u64, oldSize: u64, newSize: u64, opts: &MRemapOpts) -> Result<u64> {
        let lock = self.Lock();
        let _l = lock.lock();

        // "Note that old_address has to be page aligned." - mremap(2)
        if oldAddr != Addr(oldAddr).RoundDown()?.0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        // Linux treats an old_size that rounds up to 0 as 0, which is otherwise a
        // valid size. However, new_size can't be 0 after rounding.
        let oldSize = Addr(oldSize).RoundUp()?.0;
        let newSize = Addr(newSize).RoundUp()?.0;
        if newSize == 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let mut oldEnd = Addr(oldAddr).AddLen(oldSize)?.0;

        let mut mm = self.write();
        // All cases require that a vma exists at oldAddr.
        let mut vseg = mm.vmas.FindSeg(oldAddr);
        if !vseg.Ok() {
            return Err(Error::SysError(SysErr::EFAULT));
        }

        // Behavior matrix:
        //
        // Move     | oldSize = 0 | oldSize < newSize | oldSize = newSize | oldSize > newSize
        // ---------+-------------+-------------------+-------------------+------------------
        //   NoMove | ENOMEM [1]  | Grow in-place     | No-op             | Shrink in-place
        //  MayMove | Copy [1]    | Grow in-place or  | No-op             | Shrink in-place
        //          |             |   move            |                   |
        // MustMove | Copy        | Move and grow     | Move              | Shrink and move
        //
        // [1] In-place growth is impossible because the vma at oldAddr already
        // occupies at least part of the destination. Thus the NoMove case always
        // fails and the MayMove case always falls back to copying.

        if opts.Move != MREMAP_MUST_MOVE {
            // Handle no-ops and in-place shrinking. These cases don't care if
            // [oldAddr, oldEnd) maps to a single vma, or is even mapped at all
            // (aside from oldAddr).
            if newSize <= oldSize {
                if newSize < oldSize {
                    // If oldAddr+oldSize didn't overflow, oldAddr+newSize can't
                    // either.
                    let newEnd = oldAddr + newSize;
                    mm.RemoveVMAsLocked(self, &Range::New(newEnd, oldSize - newSize))?;
                }

                return Ok(oldAddr)
            }

            // Handle in-place growing.

            // Check that oldEnd maps to the same vma as oldAddr.
            if vseg.Range().End() < oldEnd {
                return Err(Error::SysError(SysErr::EFAULT));
            }

            // "Grow" the existing vma by creating a new mergeable one.
            let vma = vseg.Value();
            let mut newOffset = 0;
            if vma.mappable.is_some() {
                newOffset = vseg.MappableRange().End();
            }

            match mm.CreateVMAlocked(task, &MMapOpts {
                Length: newSize - oldSize,
                Addr: oldEnd,
                Offset: newOffset,
                Fixed: true,
                Unmap: false,
                Map32Bit: false,
                Perms: vma.realPerms,
                MaxPerms: vma.maxPerms,
                Private: vma.private,
                VDSO: false,
                GrowsDown: vma.growsDown,
                Precommit: false,
                MLockMode: MLockMode::default(),
                Kernel: false,
                Mapping: vma.id.clone(),
                Mappable: vma.mappable.clone(),
                Hint: vma.hint.to_string(),
            }, self) {
                Ok((vseg, ar)) => {
                    core::mem::drop(mm);
                    self.PopulateVMA(task, &vseg, &ar, false, false)?;//to true?
                    return Ok(oldAddr);
                }
                Err(e) => {
                    // In-place growth failed. In the MRemapMayMove case, fall through to
                    // copying/moving below.
                    if opts.Move == MREMAP_NO_MOVE {
                        return Err(e)
                    }
                }
            }
        }

        // Find a location for the new mapping.
        let newAR;
        match opts.Move {
            MREMAP_MAY_MOVE => {
                let newAddr = mm.FindAvailableLocked(newSize, &mut FindAvailableOpts::default())?;
                newAR = Range::New(newAddr, newSize);
            }
            MREMAP_MUST_MOVE => {
                let newAddr = opts.NewAddr;
                if Addr(newAddr).RoundDown()?.0 != newAddr {
                    return Err(Error::SysError(SysErr::EINVAL));
                }

                match Addr(newAddr).ToRange(newSize) {
                    Err(_) => return Err(Error::SysError(SysErr::EINVAL)),
                    Ok(r) => newAR = r,
                }

                if Range::New(oldAddr, oldEnd - oldAddr).Overlaps(&newAR) {
                    return Err(Error::SysError(SysErr::EINVAL));
                }

                // Unmap any mappings at the destination.
                mm.RemoveVMAsLocked(self, &newAR)?;

                // If the sizes specify shrinking, unmap everything between the new and
                // old sizes at the source. Unmapping before the following checks is
                // correct: compare Linux's mm/mremap.c:mremap_to() => do_munmap(),
                // vma_to_resize().
                if newSize < oldSize {
                    let oldNewEnd = oldAddr + newSize;
                    mm.RemoveVMAsLocked(self, &Range::New(oldNewEnd, oldEnd - oldNewEnd))?;
                    oldEnd = oldNewEnd;
                }

                // unmapLocked may have invalidated vseg; look it up again.
                vseg = mm.vmas.FindSeg(oldAddr);
            }
            _ => {
                //unreachable
                panic!("impossible to reach...");
            }
        }

        let oldAR = Range::New(oldAddr, oldEnd - oldAddr);

        // Check that oldEnd maps to the same vma as oldAddr.
        if vseg.Range().End() < oldEnd {
            return Err(Error::SysError(SysErr::EFAULT));
        }

        let vma = vseg.Value();
        if vma.mappable.is_some() {
            if core::u64::MAX - vma.offset < newAR.Len() {
                return Err(Error::SysError(SysErr::EINVAL));
            }

            // Inform the Mappable, if any, of the new mapping.
            let mappable = vma.mappable.clone().unwrap();
            let offsetat = vseg.MappableOffsetAt(oldAR.Start());
            mappable.CopyMapping(self, &oldAR, &newAR, offsetat, vma.CanWriteMappableLocked())?;
        }

        if oldSize == 0 {
            // Handle copying.
            //
            // We can't use createVMALocked because it calls Mappable.AddMapping,
            // whereas we've already called Mappable.CopyMapping (which is
            // consistent with Linux). Call vseg.Value() (rather than
            // vseg.ValuePtr()) to make a copy of the vma.
            let mut vma = vseg.Value();
            if vma.mappable.is_some() {
                vma.offset = vseg.MappableOffsetAt(oldAR.Start());
            }

            let gap = mm.vmas.FindGap(newAR.Start());
            let vseg = mm.vmas.Insert(&gap, &newAR, vma);
            mm.usageAS += newAR.Len();
            core::mem::drop(mm);
            self.PopulateVMA(task, &vseg, &newAR, false, false)?;
            return Ok(newAR.Start())
        }

        // Handle moving.
        //
        // Remove the existing vma before inserting the new one to minimize
        // iterator invalidation. We do this directly (instead of calling
        // removeVMAsLocked) because:
        //
        // 1. We can't drop the reference on vma.id, which will be transferred to
        // the new vma.
        //
        // 2. We can't call vma.mappable.RemoveMapping, because pmas are still at
        // oldAR, so calling RemoveMapping could cause us to miss an invalidation
        // overlapping oldAR.
        //
        // Call vseg.Value() (rather than vseg.ValuePtr()) to make a copy of the
        // vma.
        let vseg = mm.vmas.Isolate(&vseg, &oldAR);
        let vma = vseg.Value();
        mm.vmas.Remove(&vseg);
        let gap = mm.vmas.FindGap(newAR.Start());
        let vseg = mm.vmas.Insert(&gap, &newAR, vma.clone());

        mm.usageAS = mm.usageAS - oldAR.Len() + newAR.Len();

        // Now that pmas have been moved to newAR, we can notify vma.mappable that
        // oldAR is no longer mapped.
        if vma.mappable.is_some() {
            let mappable = vma.mappable.clone().unwrap();
            mappable.RemoveMapping(self, &oldAR, vma.offset, vma.CanWriteMappableLocked())?;
        }

        core::mem::drop(mm);
        self.PopulateVMARemap(task, &vseg, &newAR, &Range::New(oldAddr, oldSize), true)?;

        return Ok(newAR.Start())
    }

    pub fn MProtect(&self, addr: u64, len: u64, realPerms: &AccessType, growsDown: bool) -> Result<()> {
        let lock = self.Lock();
        let _l = lock.lock();

        if Addr(addr).RoundDown()?.0 != addr {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        if len == 0 {
            return Ok(())
        }

        let rlen = match Addr(len).RoundUp() {
            Err(_) => return Err(Error::SysError(SysErr::ENOMEM)),
            Ok(l) => l.0,
        };

        let mut ar = Addr(addr).ToRange(rlen)?;

        let effectivePerms = realPerms.Effective();

        let mut mm = self.write();
        // Non-growsDown mprotect requires that all of ar is mapped, and stops at
        // the first non-empty gap. growsDown mprotect requires that the first vma
        // be growsDown, but does not require it to extend all the way to ar.Start;
        // vmas after the first must be contiguous but need not be growsDown, like
        // the non-growsDown case.
        let vseg = mm.vmas.LowerBoundSeg(ar.Start());
        if !vseg.Ok() {
            return Err(Error::SysError(SysErr::ENOMEM));
        }

        if growsDown {
            if !vseg.Value().growsDown {
                return Err(Error::SysError(SysErr::EINVAL));
            }
            if ar.End() <= vseg.Range().Start() {
                return Err(Error::SysError(SysErr::ENOMEM));
            }
            ar.start = vseg.Range().Start();
        } else {
            if ar.Start() < vseg.Range().Start() {
                return Err(Error::SysError(SysErr::ENOMEM));
            }
        }

        let mut vseg = vseg;
        loop {
            let vma = vseg.Value();
            // Check for permission validity before splitting vmas, for consistency
            // with Linux.
            if !vma.maxPerms.SupersetOf(&effectivePerms) {
                return Err(Error::SysError(SysErr::EACCES));
            }

            //error!("MProtect: vseg range is {:x?}, vma.mappable.is_some() is {}", vseg.Range(), vma.mappable.is_some());

            vseg = mm.vmas.Isolate(&vseg, &ar);
            // Update vma permissions.
            let mut vma = vseg.Value();
            vma.realPerms = *realPerms;
            vma.effectivePerms = effectivePerms;

            vseg.SetValue(vma);
            let range = vseg.Range();

            let pageopts = if effectivePerms.Write() {
                PageOpts::UserReadWrite().Val()
            } else if effectivePerms.Read() || effectivePerms.Exec() {
                PageOpts::UserReadOnly().Val()
            } else {
                PageOpts::UserNonAccessable().Val()
            };

            //change pagetable permission
            let mut end = range.End();
            if ar.End() < end {
                end = ar.End();
            }

            mm.pt.write().MProtect(Addr(range.Start()), Addr(end), pageopts, false)?;
            if ar.End() <= range.End() {
                break;
            }

            let (segtmp, _) = vseg.NextNonEmpty();
            vseg = segtmp;
            if !vseg.Ok() {
                return Err(Error::SysError(SysErr::ENOMEM));
            }
        }

        mm.vmas.MergeRange(&ar);
        mm.vmas.MergeAdjacent(&ar);

        return Ok(())
    }

    pub fn NumaPolicy(&self, addr: u64) -> Result<(i32, u64)> {
        let mm = self.read();
        let vseg = mm.vmas.FindSeg(addr);
        if !vseg.Ok() {
            return Err(Error::SysError(SysErr::EFAULT));
        }

        let vma = vseg.Value();
        return Ok((vma.numaPolicy, vma.numaNodemask))
    }

    pub fn SetNumaPolicy(&self, addr: u64, len: u64, policy: i32, nodemask: u64) -> Result<()> {
        let lock = self.Lock();
        let _l = lock.lock();

        if !Addr(addr).IsPageAligned() {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        // Linux allows this to overflow.
        let la = Addr(len).RoundUp()?.0;
        let ar = Range::New(addr, la);

        if ar.Len() == 0 {
            return Ok(())
        }

        let mut mm = self.write();
        let mut vseg = mm.vmas.LowerBoundSeg(ar.Start());
        let mut lastEnd = ar.Start();

        loop {
            if !vseg.Ok() || lastEnd < vseg.Range().Start() {
                // "EFAULT: ... there was an unmapped hole in the specified memory
                // range specified [sic] by addr and len." - mbind(2)
                return Err(Error::SysError(SysErr::EFAULT))
            }
            vseg = mm.vmas.Isolate(&vseg, &ar);
            let mut vma = vseg.Value();
            vma.numaPolicy = policy;
            vma.numaNodemask = nodemask;
            vseg.SetValue(vma);
            lastEnd = vseg.Range().End();
            if ar.End() <= lastEnd {
                mm.vmas.MergeRange(&ar);
                mm.vmas.MergeAdjacent(&ar);
                return Ok(())
            }
            let (tmpVseg, _) = vseg.NextNonEmpty();
            vseg = tmpVseg;
        }
    }

    // BrkSetup sets mm's brk address to addr and its brk size to 0.
    pub fn BrkSetup(&self, addr: u64) {
        let mut mm = self.write();

        if mm.brkInfo.brkStart != mm.brkInfo.brkEnd {
            panic!("BrkSetup get nonempty brk");
        }

        mm.brkInfo.brkStart = addr;
        mm.brkInfo.brkEnd = addr;
        mm.brkInfo.brkMemEnd = addr;
    }

    // Brk implements the semantics of Linux's brk(2), except that it returns an
    // error on failure.
    pub fn Brk(&self, task: &Task, addr: u64) -> Result<u64> {
        let lock = self.Lock();
        let _l = lock.lock();

        let mm = self;

        if addr == 0 || addr == -1 as i64 as u64 {
            return Ok(mm.read().brkInfo.brkEnd);
        }

        if addr < mm.read().brkInfo.brkStart {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let oldbrkpg = Addr(mm.write().brkInfo.brkEnd).RoundUp()?.0;

        let newbrkpg = match Addr(addr).RoundUp() {
            Err(_e) => return Err(Error::SysError(SysErr::EFAULT)),
            Ok(v) => v.0,
        };

        if oldbrkpg < newbrkpg {
            let (vseg, ar) = mm.write().CreateVMAlocked(task, &MMapOpts {
                Length: newbrkpg - oldbrkpg,
                Addr: oldbrkpg,
                Offset: 0,
                Fixed: true,
                Unmap: false,
                Map32Bit: false,
                Perms: AccessType::ReadWrite(),
                MaxPerms: AccessType::AnyAccess(),
                Private: true,
                VDSO: false,
                GrowsDown: false,
                Precommit: false,
                MLockMode: MLockMode::default(),
                Kernel: false,
                Mapping: None,
                Mappable: None,
                Hint: "[Heap]".to_string(),
            }, self)?;

            mm.PopulateVMA(task, &vseg, &ar, false, false)?;
            mm.write().brkInfo.brkEnd = addr;
        } else {
            if newbrkpg < oldbrkpg {
                mm.write().RemoveVMAsLocked(self, &Range::New(newbrkpg, oldbrkpg - newbrkpg))?;
            }

            mm.write().brkInfo.brkEnd = addr;
        }

        return Ok(addr);
    }

    pub fn GetSharedFutexKey(&self, _task: &Task, addr: u64) -> Result<Key> {
        let lock = self.Lock();
        let _l = lock.lock();

        let ar = match Addr(addr).ToRange(4) {
            Ok(r) => r,
            Err(_) => return Err(Error::SysError(SysErr::EFAULT)),
        };

        let mm = self.read();
        let (vseg, _, err) = mm.GetVMAsLocked(&ar, &AccessType::ReadOnly(), false);
        match err {
            Ok(()) => (),
            Err(e) => return Err(e),
        }

        let vma = vseg.Value();
        if vma.private {
            return Ok(Key {
                Kind: KeyKind::KindSharedPrivate,
                Addr: addr,
            })
        }

        let (phyAddr, _) = mm.VirtualToPhy(addr)?;

        return Ok(Key {
            Kind: KeyKind::KindSharedMappable,
            Addr: phyAddr,
        })
    }

    pub fn MAdvise(&self, _task: &Task, addr: u64, length: u64, advise: i32) -> Result<()> {
        let ar = match Addr(addr).ToRange(length) {
            Err(_) => return Err(Error::SysError(SysErr::EINVAL)),
            Ok(r) => r
        };

        let lock = self.Lock();
        let _l = lock.lock();

        let mut vseg = self.read().vmas.LowerBoundSeg(ar.Start());
        while vseg.Ok() && vseg.Range().Start() < ar.End() {
            let vma = vseg.Value();
            if vma.mlockMode != MLockMode::MlockNone && advise == MAdviseOp::MADV_DONTNEED {
                return Err(Error::SysError(SysErr::EINVAL))
            }

            let mr = ar.Intersect(&vseg.Range());
            self.write().pt.write().MUnmap(mr.Start(), mr.Len())?;

            if let Some(iops) = vma.mappable.clone() {
                let fstart = mr.Start() - vseg.Range().Start() + vma.offset;

                // todo: fix the Madvise/MADV_DONTNEED, when there are multiple process MAdviseOp::MADV_DONTNEED
                // with current implementation, the first Madvise/MADV_DONTNEED will work.
                iops.MAdvise(fstart, mr.Len(), advise)?;
            }

            vseg = vseg.NextSeg();
        }

        return Ok(())
    }

    pub fn VirtualMemorySizeRange(&self, ar: &Range) -> u64 {
        return self.read().vmas.SpanRange(&ar);
    }

    pub fn VirtualMemorySize(&self) -> u64 {
        return self.read().usageAS;
    }

    pub fn ResidentSetSize(&self) -> u64 {
        return self.read().curRSS;
    }

    pub fn MaxResidentSetSize(&self) -> u64 {
        return self.read().maxRSS;
    }
}

// MRemapOpts specifies options to MRemap.
#[derive(Debug)]
pub struct MRemapOpts {
    // Move controls whether MRemap moves the remapped mapping to a new address.
    pub Move: MRemapMoveMode,

    // NewAddr is the new address for the remapping. NewAddr is ignored unless
    // Move is MMRemapMustMove.
    pub NewAddr: u64,
}

pub type MRemapMoveMode = i32;

// MRemapNoMove prevents MRemap from moving the remapped mapping.
pub const MREMAP_NO_MOVE: MRemapMoveMode = 0;

// MRemapMayMove allows MRemap to move the remapped mapping.
pub const MREMAP_MAY_MOVE: MRemapMoveMode = 1;

// MRemapMustMove requires MRemap to move the remapped mapping to
// MRemapOpts.NewAddr, replacing any existing mappings in the remapped
// range.
pub const MREMAP_MUST_MOVE: MRemapMoveMode = 2;