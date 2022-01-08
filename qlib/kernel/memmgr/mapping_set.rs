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
use alloc::vec::Vec;
use ::qlib::mutex::*;
use alloc::collections::btree_set::BTreeSet;
use core::cmp::Ordering;
use core::ops::Deref;

use super::super::super::mem::areaset::*;
use super::super::memmgr::mm::*;
use super::super::super::addr::*;
use super::super::super::range::*;
use super::super::task::*;

// MappingOfRange represents a mapping of a MappableRange.
#[derive(Clone)]
pub struct MappingOfRange {
    pub MappingSpace: MemoryManagerWeak,
    pub AddrRange: Range,
    pub Writeable: bool,
}

impl Ord for MappingOfRange {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.MappingSpace.ID() != other.MappingSpace.ID() {
            return self.MappingSpace.ID().cmp(&other.MappingSpace.ID())
        } else if self.AddrRange.Start() != other.AddrRange.Start() {
            return self.AddrRange.Start().cmp(&other.AddrRange.Start())
        } else if self.AddrRange.End() != other.AddrRange.End() {
            return self.AddrRange.End().cmp(&other.AddrRange.End())
        } else {
            return self.Writeable.cmp(&other.Writeable)
        }
    }
}

impl Eq for MappingOfRange {}

impl PartialEq for MappingOfRange {
    fn eq(&self, other: &Self) -> bool {
        return self.AddrRange.Start() == other.AddrRange.Start() &&
            self.AddrRange.End() == other.AddrRange.End() &&
            self.Writeable == other.Writeable;
    }
}

impl PartialOrd for MappingOfRange {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl MappingOfRange {
    pub fn invalidate(&self, task: &Task, _invalidatePrivate: bool) {
        //self.MappingSpace.Upgrade().ResetFileMapping(task, &self.AddrRange, invalidatePrivate);
        let start = Addr(self.AddrRange.Start()).RoundUp().unwrap().0;
        let end = Addr(self.AddrRange.End()).RoundUp().unwrap().0;
        if start >= end {
            return
        }
        self.MappingSpace.Upgrade().MUnmap(task, start, end - start).unwrap();
    }
}

#[derive(Clone, Default)]
pub struct MappingsOfRange(pub Arc<QMutex<BTreeSet<MappingOfRange>>>);

impl Deref for MappingsOfRange {
    type Target = Arc<QMutex<BTreeSet<MappingOfRange>>>;

    fn deref(&self) -> &Arc<QMutex<BTreeSet<MappingOfRange>>> {
        &self.0
    }
}

impl MappingsOfRange {
    pub fn New() -> Self {
        return Self(Arc::new(QMutex::new(BTreeSet::new())));
    }
}

impl AreaValue for MappingsOfRange {
    fn Merge(&self, _r1: &Range, r2: &Range, v2: &Self) -> Option<Self> {
        let v1 = self.lock();
        let v2 = v2.lock();
        if v1.len() != v2.len() {
            return None;
        }

        let merged = Self::New();
        {
            let mut m = merged.lock();

            // Each MappingOfRange in val1 must have a matching region in val2, forming
            // one contiguous region.
            for k1 in v1.iter() {
                let k2 = MappingOfRange {
                    MappingSpace: k1.MappingSpace.clone(),
                    AddrRange: Range {
                        start: k1.AddrRange.End(),
                        len: r2.Len(),
                    },
                    Writeable: k1.Writeable,
                };

                if !v2.contains(&k2) {
                    return None;
                }

                m.insert(MappingOfRange {
                    MappingSpace: k1.MappingSpace.clone(),
                    AddrRange: Range {
                        start: k1.AddrRange.Start(),
                        len: k2.AddrRange.End() - k1.AddrRange.Start(),
                    },
                    Writeable: k1.Writeable,
                });
            }
        }

        return Some(merged)
    }

    fn Split(&self, r: &Range, split: u64) -> (Self, Self) {
        let val = self.lock();
        if split <= r.Start() || split >= r.End() {
            panic!("split is not within range {:?}", r);
        }

        let m1 = Self::New();
        let m2 = Self::New();

        // split is a value in MappableRange, we need the offset into the
        // corresponding MappingsOfRange.
        let offset = split - r.Start();
        for k in val.iter() {
            let k1 = MappingOfRange {
                MappingSpace: k.MappingSpace.clone(),
                AddrRange: Range {
                    start: k.AddrRange.Start(),
                    len: offset,
                },
                Writeable: k.Writeable,
            };

            m1.lock().insert(k1);

            let k2 = MappingOfRange {
                MappingSpace: k.MappingSpace.clone(),
                AddrRange: Range {
                    start: k.AddrRange.Start() + offset,
                    len: k.AddrRange.Len() - offset,
                },
                Writeable: k.Writeable,
            };
            m2.lock().insert(k2);
        }

        return (m1, m2)
    }
}

// subsetMapping returns the MappingOfRange that maps subsetRange, given that
// ms maps wholeRange beginning at addr.
//
// For instance, suppose wholeRange = [0x0, 0x2000) and addr = 0x4000,
// indicating that ms maps addresses [0x4000, 0x6000) to MappableRange [0x0,
// 0x2000). Then for subsetRange = [0x1000, 0x2000), subsetMapping returns a
// MappingOfRange for which AddrRange = [0x5000, 0x6000).
fn SubsetMapping(wholeRange: &Range, subsetRange: &Range, ms: &MemoryManagerWeak, addr: u64, writeable: bool) -> MappingOfRange {
    if !wholeRange.IsSupersetOf(&subsetRange) {
        panic!("{:?} is not a superset of {:?}", wholeRange, subsetRange);
    }

    let offset = subsetRange.Start() - wholeRange.Start();
    let start = addr + offset;

    return MappingOfRange {
        MappingSpace: ms.clone(),
        AddrRange: Range {
            start: start,
            len: subsetRange.Len(),
        },
        Writeable: writeable,
    }
}

impl AreaSet<MappingsOfRange> {
    // AddMapping adds the given mapping and returns the set of Range that
    // previously had no mappings.
    //
    // Preconditions: As for Mappable.AddMapping.
    pub fn AddMapping(&mut self, ms: &MemoryManager, ar: &Range, offset: u64, writeable: bool) -> Vec<Range> {
        let mr = Range::New(offset, ar.Len());
        let mut mapped = Vec::new();

        let (mut seg, mut gap) = self.Find(mr.Start());
        loop {
            if seg.Ok() && seg.Range().Start() < mr.End() {
                seg = self.Isolate(&seg, &mr);
                let val = seg.Value();
                val.lock().insert(SubsetMapping(&mr, &seg.Range(), &ms.Downgrade(), ar.Start(), writeable));
                let (stmp, gtmp) = seg.NextNonEmpty();
                seg = stmp;
                gap = gtmp;
            } else if gap.Ok() && gap.Range().Start() < mr.End() {
                let gapMR = gap.Range().Intersect(&mr);
                mapped.push(gapMR.clone());
                // Insert a set and continue from the above case.
                seg = self.Insert(&gap, &gapMR, MappingsOfRange::New());
                gap = AreaGap::default();
            } else {
                return mapped
            }
        }
    }

    // RemoveMapping removes the given mapping and returns the set of
    // MappableRanges that now have no mappings.
    //
    // Preconditions: As for Mappable.RemoveMapping.
    pub fn RemoveMapping(&mut self, ms: &MemoryManager, ar: &Range, offset: u64, writeable: bool) -> Vec<Range> {
        let mr = Range::New(offset, ar.Len());
        let mut unmapped = Vec::new();

        let mut seg = self.FindSeg(mr.Start());
        if !seg.Ok() {
            panic!("MappingSet.RemoveMapping({:?}): no segment containing {:x}", mr, mr.Start());
        }

        while seg.Ok() && seg.Range().Start() < mr.End() {
            // Ensure this segment is limited to our range.
            seg = self.Isolate(&seg, &mr);

            // Remove this part of the mapping.
            let mappings = seg.Value();
            mappings.lock().remove(&SubsetMapping(&mr, &seg.Range(), &ms.Downgrade(), ar.Start(), writeable));

            let len = mappings.lock().len();
            if len == 0 {
                unmapped.push(seg.Range());
                seg = self.Remove(&seg).NextSeg();
            } else {
                seg = seg.NextSeg();
            }
        }

        self.MergeAdjacent(&mr);
        return unmapped;
    }

    // Invalidate calls MappingSpace.Invalidate for all mappings of offsets in mr.
    pub fn InvalidateRanges(&mut self, _task: &Task, mr: &Range, _invalidatePrivate: bool) -> Vec<MappingOfRange> {
        let mut ranges = Vec::new();
        let mut seg = self.LowerBoundSeg(mr.Start());
        while seg.Ok() && seg.Range().Start() < mr.End() {
            let segMR = seg.Range();
            for m in seg.Value().lock().iter() {
                let region = SubsetMapping(&segMR, &segMR.Intersect(mr), &m.MappingSpace, m.AddrRange.Start(), m.Writeable);
                ranges.push(region);
            }

            seg = seg.NextSeg();
        }

        return ranges;
    }

    // InvalidateAll calls MappingSpace.Invalidate for all mappings of s.
    pub fn InvalidateAll(&mut self, task: &Task, invalidatePrivate: bool) {
        let mut seg = self.FirstSeg();
        while seg.Ok() {
            for m in seg.Value().lock().iter() {
                m.invalidate(task, invalidatePrivate);
            }

            seg = seg.NextSeg();
        }
    }
}