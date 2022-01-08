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

use alloc::sync::Arc;
use alloc::string::String;
use alloc::string::ToString;
use core::fmt;

use super::super::fs::host::hostinodeop::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::task::*;
use super::super::qlib::addr::*;
//use super::super::task::*;
use super::*;
use super::super::qlib::range::*;
use super::super::qlib::mem::areaset::*;
use super::mm::*;
use super::arch::*;

// map32Start/End are the bounds to which MAP_32BIT mappings are constrained,
// and are equivalent to Linux's MAP32_BASE and MAP32_MAX respectively.
pub const MAP32_START: u64 = 0x40000000;
pub const MAP32_END: u64 = 0x80000000;

#[derive(Clone, Default, Debug)]
pub struct FindAvailableOpts {
    // These fields are equivalent to those in MMapOpts, except that:
    //
    // - Addr must be page-aligned.
    //
    // - Unmap allows existing guard pages in the returned range.

    pub Addr: u64,
    pub Fixed: bool,
    pub Unmap: bool,
    pub Map32Bit: bool,
    pub Kernel: bool,
}

impl MemoryManager {
    pub fn FindLowestAvailableLocked(&self, length: u64, alignment: u64, bounds: &Range) -> Result<u64> {
        let mapping = self.mapping.lock();
        let mut gap = mapping.vmas.LowerBoundGap(bounds.Start());

        while gap.Ok() && gap.Range().Start() < bounds.End() {
            let gr = gap.Range().Intersect(bounds);
            if gr.Len() > length {
                // Can we shift up to match the alignment?
                let offset = gr.Start() % alignment;
                if offset != 0 {
                    if gr.Len() >= length + alignment - offset {
                        return Ok(gr.Start() + (alignment - offset))
                    }
                }

                // Either aligned perfectly, or can't align it.
                return Ok(gr.Start())
            }

            let tmp = gap.NextGap();
            gap = tmp;
        }

        return Err(Error::SysError(SysErr::ENOMEM));
    }

    pub fn FindHighestAvailableLocked(&self, length: u64, alignment: u64, bounds: &Range) -> Result<u64> {
        let mapping = self.mapping.lock();
        let mut gap = mapping.vmas.UpperBoundGap(bounds.End());

        while gap.Ok() && gap.Range().End() > bounds.Start() {
            let gr = gap.Range().Intersect(bounds);
            if gr.Len() > length {
                // Can we shift up to match the alignment?
                let start = gr.End() - length;
                let offset = gr.Start() % alignment;
                if offset != 0 {
                    if gr.Start() >= start - offset {
                        return Ok(start - offset)
                    }
                }

                // Either aligned perfectly, or can't align it.
                return Ok(start)
            }

            let tmp = gap.NextGap();
            gap = tmp;
        }

        return Err(Error::SysError(SysErr::ENOMEM));
    }

    // getVMAsLocked ensures that vmas exist for all addresses in ar, and support
    // access of type (at, ignorePermissions). It returns:
    //
    // - An iterator to the vma containing ar.Start. If no vma contains ar.Start,
    // the iterator is unspecified.
    //
    // - An iterator to the gap after the last vma containing an address in ar. If
    // vmas exist for no addresses in ar, the iterator is to a gap that begins
    // before ar.Start.
    //
    // - An error that is non-nil if vmas exist for only a subset of ar.
    //
    // Preconditions: mm.mappingMu must be locked for reading; it may be
    // temporarily unlocked. ar.Length() != 0.
    pub fn GetVMAsLocked(&self, r: &Range, at: &AccessType, ignorePermissions: bool) -> (AreaSeg<VMA>, AreaGap<VMA>, Result<()>) {
        let mapping = self.mapping.lock();
        let (mut vbegin, mut vgap) = mapping.vmas.Find(r.Start());

        if !vbegin.Ok() {
            vbegin = vgap.NextSeg()
        } else {
            vgap = vbegin.PrevGap();
        }

        let mut addr = r.Start();
        let mut vseg = vbegin.clone();

        while vseg.Ok() {
            // Loop invariants: vgap = vseg.PrevGap(); addr < vseg.End().
            if addr < vseg.Range().Start() {
                return (vbegin, vgap, Err(Error::SysError(SysErr::EFAULT)));
            }

            {
                let vseglock = vseg.lock();
                let vma = vseglock.Value();

                let mut perms = vma.effectivePerms;
                if ignorePermissions {
                    perms = vma.maxPerms;
                }

                if !perms.SupersetOf(at) {
                    return (vbegin, vgap, Err(Error::SysError(SysErr::EPERM)));
                }
            }

            addr = vseg.Range().End();
            vgap = vseg.NextGap();
            if addr >= r.End() {
                return (vbegin, vgap, Ok(()))
            }

            vseg = vseg.NextSeg();
        }

        // Ran out of vmas before ar.End.
        return (vbegin, vgap, Err(Error::SysError(SysErr::EFAULT)));
    }

    pub fn FindAvailableLocked(&self, length: u64, opts: &mut FindAvailableOpts) -> Result<u64> {
        if opts.Fixed {
            opts.Map32Bit = false;
        }

        let mut allowedRange = if opts.Kernel {
            Range::New(0, !0)
        } else {
            self.ApplicationAddrRange()
        };

        if opts.Map32Bit {
            allowedRange = allowedRange.Intersect(&Range::New(MAP32_START, MAP32_END - MAP32_START));
        }

        // Does the provided suggestion work?
        match Addr(opts.Addr).ToRange(length) {
            Ok(r) => {
                if allowedRange.IsSupersetOf(&r) {
                    if opts.Unmap {
                        return Ok(r.Start());
                    }

                    let vgap = self.mapping.lock().vmas.FindGap(r.Start());
                    if vgap.Ok() && vgap.AvailableRange().IsSupersetOf(&r) {
                        return Ok(r.Start())
                    }
                }
            }
            Err(_) => ()
        }

        // Fixed mappings accept only the requested address.
        if opts.Fixed {
            return Err(Error::SysError(SysErr::ENOMEM))
        }

        // Prefer hugepage alignment if a hugepage or more is requested.
        let mut alignment = MemoryDef::PAGE_SIZE;
        if length > MemoryDef::HUGE_PAGE_SIZE {
            alignment = MemoryDef::HUGE_PAGE_SIZE;
        }

        if opts.Map32Bit {
            return self.FindLowestAvailableLocked(length, alignment, &allowedRange);
        }

        let layout = *self.layout.lock();
        if layout.DefaultDirection == MMAP_BOTTOM_UP {
            return self.FindLowestAvailableLocked(length, alignment,
                                                  &Range::New(layout.BottomUpBase, layout.MaxAddr - layout.BottomUpBase));
        }

        return self.FindHighestAvailableLocked(length, alignment,
                                               &Range::New(layout.MinAddr, layout.TopDownBase - layout.MinAddr));
    }

    pub fn CreateVMAlocked(&self, _task: &Task, opts: &MMapOpts) -> Result<(AreaSeg<VMA>, Range)> {
        if opts.MaxPerms != opts.MaxPerms.Effective() {
            panic!("Non-effective MaxPerms {:?} cannot be enforced", opts.MaxPerms);
        }

        // Find a useable range.
        let mut findopts = FindAvailableOpts {
            Addr: opts.Addr,
            Fixed: opts.Fixed,
            Unmap: opts.Unmap,
            Map32Bit: opts.Map32Bit,
            Kernel: opts.Kernel,
        };
        let addr = self.FindAvailableLocked(opts.Length, &mut findopts)?;

        let ar = Range::New(addr, opts.Length);

        // todo: Check against RLIMIT_AS.
        /*let mut newUsageAS = self.usageAS + opts.Length;
        if opts.Unmap {
            newUsageAS -= self.vmas.SpanRange(&ar);
        }*/

        // Remove overwritten mappings. This ordering is consistent with Linux:
        // compare Linux's mm/mmap.c:mmap_region() => do_munmap(),
        // file->f_op->mmap().
        if opts.Unmap {
            self.RemoveVMAsLocked(&ar)?;
        }

        let mut mapping = self.mapping.lock();
        let gap = mapping.vmas.FindGap(ar.Start());

        if opts.Mappable.is_some() {
            let mappable = opts.Mappable.clone().unwrap();
            mappable.AddMapping(self, &ar, opts.Offset, !opts.Private && opts.MaxPerms.Write())?;
        }

        let vma = VMA {
            mappable: opts.Mappable.clone(),
            offset: opts.Offset,
            fixed: opts.Fixed,
            realPerms: opts.Perms,
            effectivePerms: opts.Perms.Effective(),
            maxPerms: opts.MaxPerms,
            private: opts.Private,
            growsDown: opts.GrowsDown,
            dontfork: false,
            mlockMode: opts.MLockMode,
            kernel: opts.Kernel,
            hint: opts.Hint.to_string(),
            id: opts.Mapping.clone(),
            numaPolicy: 0,
            numaNodemask: 0,
        };

        mapping.usageAS += opts.Length;

        let vseg = mapping.vmas.Insert(&gap, &ar, vma);
        let nextvseg = vseg.NextSeg();
        assert!(vseg.Range().End() <= nextvseg.Range().Start(), "vseg end < vseg.next.start");
        return Ok((vseg, ar))
    }

    //find free seg with enough len
    pub fn FindAvailableSeg(&self, _task: &Task, offset: u64, len: u64) -> Result<u64> {
        let _ml = self.MappingWriteLock();

        let mut findopts = FindAvailableOpts {
            Addr: offset,
            Fixed: false,
            Unmap: false,
            Map32Bit: false,
            Kernel: false,
        };

        let addr = self.FindAvailableLocked(len, &mut findopts)?;
        return Ok(addr);
    }

}

#[derive(Clone, Default)]
pub struct VMA {
    pub mappable: Option<HostInodeOp>,
    pub offset: u64, //file offset when the mappable is not null, phyaddr for other
    pub fixed: bool,

    // realPerms are the memory permissions on this vma, as defined by the
    // application.
    pub realPerms: AccessType,

    // effectivePerms are the memory permissions on this vma which are
    // actually used to control access.
    //
    // Invariant: effectivePerms == realPerms.Effective().
    pub effectivePerms: AccessType,

    // maxPerms limits the set of permissions that may ever apply to this
    // memory
    pub maxPerms: AccessType,

    // private is true if this is a MAP_PRIVATE mapping, such that writes to
    // the mapping are propagated to a copy.
    pub private: bool,

    // growsDown is true if the mapping may be automatically extended downward
    // under certain conditions. If growsDown is true, mappable must be nil.
    //
    // There is currently no corresponding growsUp flag; in Linux, the only
    // architectures that can have VM_GROWSUP mappings are ia64, parisc, and
    // metag, none of which we currently support.
    pub growsDown: bool,

    // dontfork is the MADV_DONTFORK setting for this vma configured by madvise().
    pub dontfork: bool,

    pub mlockMode: MLockMode,

    pub kernel: bool,
    pub hint: String,
    pub id: Option<Arc<Mapping>>,

    // numaPolicy is the NUMA policy for this vma set by mbind().
    pub numaPolicy: i32,

    // numaNodemask is the NUMA nodemask for this vma set by mbind().
    pub numaNodemask: u64,
}

impl fmt::Debug for VMA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VMA")
            .field("offset", &self.offset)
            .field("realPerms", &self.realPerms)
            .field("effectivePerms", &self.effectivePerms)
            .field("maxPerms", &self.maxPerms)
            .field("private", &self.private)
            .field("growsDown", &self.growsDown)
            .field("kernel", &self.kernel)
            .field("hint", &self.hint)
            .finish()
    }
}

impl VMA {
    pub fn Copy(&self) -> Self {
        let copy = VMA {
            mappable: self.mappable.clone(),
            offset: self.offset,
            fixed: self.fixed,
            realPerms: self.realPerms,
            effectivePerms: self.effectivePerms,
            maxPerms: self.maxPerms,
            private: self.private,
            growsDown: self.growsDown,
            dontfork: self.dontfork,
            mlockMode: self.mlockMode,
            kernel: self.kernel,
            hint: self.hint.to_string(),
            id: self.id.clone(),
            numaPolicy: 0,
            numaNodemask: 0,
        };

        return copy
    }

    // canWriteMappableLocked returns true if it is possible for vma.mappable to be
    // written to via this vma, i.e. if it is possible that
    // vma.mappable.Translate(at.Write=true) may be called as a result of this vma.
    // This includes via I/O with usermem.IOOpts.IgnorePermissions = true, such as
    // PTRACE_POKEDATA.
    //
    // canWriteMappableLocked is equivalent to Linux's VM_SHARED.
    pub fn CanWriteMappableLocked(&self) -> bool {
        !self.private && self.maxPerms.Write()
    }
}

impl AreaSeg<VMA> {
    //virtual address to mappable offset
    pub fn MappableOffsetAt(&self, vaddr: u64) -> u64 {
        let entry = self.lock();
        let vma = entry.Value();
        let vstart = entry.range.Start();
        return vma.offset + (vaddr - vstart);
    }

    //virtual address range to mappable range
    pub fn MappableRangeOf(&self, r: &Range) -> Range {
        return Range::New(self.MappableOffsetAt(r.Start()), r.Len())
    }

    pub fn MappableRange(&self) -> Range {
        return self.MappableRangeOf(&self.Range())
    }

    //mappable offset to virtual address
    pub fn AddrRangeAt(&self, offset: u64) -> u64 {
        let entry = self.lock();
        let vma = entry.Value();
        let vstart = entry.range.Start();
        return vstart + (offset - vma.offset);
    }

    //mappable range to virtual range
    pub fn AddrRangeOf(&self, r: &Range) -> Range {
        let start = self.AddrRangeAt(r.Start());
        return Range::New(start, r.Len())
    }

    //find first area which range.end is large than addr
    pub fn SeekNextLowerBound(&self, addr: u64) -> AreaSeg<VMA> {
        let mut seg = self.clone();

        while seg.Ok() && addr >= seg.Range().End() {
            seg = seg.NextSeg();
        }

        return seg;
    }
}

pub const GUARD_BYTES: u64 = 256 * MemoryDef::PAGE_SIZE;

impl AreaGap<VMA> {
    // availableRange returns the subset of vgap.Range() in which new vmas may be
    // created without MMapOpts.Unmap == true.
    pub fn AvailableRange(&self) -> Range {
        let r = self.Range();
        let next = self.NextSeg();

        //no next
        if !next.Ok() || !next.lock().Value().growsDown {
            return r
        }

        // Exclude guard pages.
        if r.Len() < GUARD_BYTES {
            return Range::New(r.Start(), 0);
        }

        return Range::New(r.Start(), r.Len() - GUARD_BYTES)
    }
}

pub fn MinKey() -> u64 {
    return 0;
}

pub fn MaxKey() -> u64 {
    return !0;
}

impl AreaValue for VMA {
    fn Merge(&self, r1: &Range, _r2: &Range, vma2: &VMA) -> Option<VMA> {
        let vma1 = self;
        if vma1.mappable.is_some() && vma1.offset + r1.Len() != vma2.offset {
            return None;
        }

        if vma1.mappable != vma2.mappable ||
            vma1.realPerms != vma2.realPerms ||
            vma1.maxPerms != vma2.maxPerms ||
            vma1.effectivePerms != vma2.effectivePerms ||
            vma1.private != vma2.private ||
            vma1.growsDown != vma2.growsDown ||
            vma1.dontfork != vma2.dontfork ||
            vma1.mlockMode != vma2.mlockMode ||
            vma1.kernel != vma2.kernel ||
            vma1.numaPolicy != vma2.numaPolicy ||
            vma1.numaNodemask != vma2.numaNodemask ||
            vma1.hint != vma2.hint {
            return None;
        }

        return Some(vma1.Copy())
    }

    fn Split(&self, r: &Range, split: u64) -> (VMA, VMA) {
        let v = self;
        let mut v2 = v.clone();

        v2.offset += split - r.Start();

        return (v.clone(), v2)
    }
}
