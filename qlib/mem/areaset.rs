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

use super::super::mutex::*;
use alloc::collections::btree_map::BTreeMap;
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::sync::Weak;
use core::ops::Bound::*;
use core::ops::Deref;

use super::super::range::*;

pub trait AreaValue: Send + Sync + Clone + Default {
    fn Merge(&self, r1: &Range, _r2: &Range, vma2: &Self) -> Option<Self>;
    fn Split(&self, r: &Range, split: u64) -> (Self, Self);
}

pub enum Area<T: AreaValue> {
    AreaSeg(AreaSeg<T>),
    AreaGap(AreaGap<T>),
}

impl<T: AreaValue> Area<T> {
    pub fn NewSeg(entry: &AreaEntry<T>) -> Self {
        return Self::AreaSeg(AreaSeg(entry.clone()));
    }

    pub fn NewGap(entry: &AreaEntry<T>) -> Self {
        return Self::AreaGap(AreaGap(entry.clone()));
    }

    pub fn AreaGap(&self) -> AreaGap<T> {
        match self {
            Self::AreaGap(ref gap) => gap.clone(),
            _ => AreaGap::default(),
        }
    }

    pub fn AreaSeg(&self) -> AreaSeg<T> {
        match self {
            Self::AreaSeg(ref seg) => seg.clone(),
            _ => AreaSeg::default(),
        }
    }

    pub fn IsSeg(&self) -> bool {
        match self {
            Self::AreaSeg(_) => true,
            Self::AreaGap(_) => false,
        }
    }

    pub fn IsGap(&self) -> bool {
        return !self.IsSeg();
    }
}

#[derive(Clone, Default)]
pub struct AreaSeg<T: AreaValue>(pub AreaEntry<T>);

impl<T: AreaValue> Deref for AreaSeg<T> {
    type Target = AreaEntry<T>;

    fn deref(&self) -> &AreaEntry<T> {
        &self.0
    }
}

impl<T: AreaValue> PartialEq for AreaSeg<T> {
    fn eq(&self, other: &Self) -> bool {
        return self.0 == other.0;
    }
}

impl<T: AreaValue> Eq for AreaSeg<T> {}

impl<T: AreaValue> AreaSeg<T> {
    pub fn New(entry: AreaEntry<T>) -> Self {
        return Self(entry);
    }

    pub fn Value(&self) -> T {
        return self.0.lock().Value().clone();
    }

    pub fn SetValue(&self, value: T) {
        self.0.lock().value = Some(value);
    }

    pub fn NextNonEmpty(&self) -> (AreaSeg<T>, AreaGap<T>) {
        let gap = self.NextGap();
        if gap.Range().Len() != 0 {
            return (AreaSeg::default(), gap);
        }

        return (gap.NextSeg(), AreaGap::default());
    }

    pub fn Ok(&self) -> bool {
        return self.0.Ok();
    }

    pub fn PrevGap(&self) -> AreaGap<T> {
        if self.0.IsHead() {
            return AreaGap::default();
        }

        return AreaGap(self.0.PrevEntry().unwrap());
    }

    pub fn NextGap(&self) -> AreaGap<T> {
        if self.0.IsTail() {
            return AreaGap::default();
        }

        return AreaGap(self.0.clone());
    }

    pub fn NextSeg(&self) -> AreaSeg<T> {
        if self.0.IsTail() {
            return self.clone();
        }

        return AreaSeg(self.0.NextEntry().unwrap());
    }

    pub fn PrevSeg(&self) -> AreaSeg<T> {
        if self.0.IsHead() {
            return self.clone();
        }

        return AreaSeg(self.0.PrevEntry().unwrap());
    }
}

#[derive(Clone, Default)]
pub struct AreaGap<T: AreaValue>(pub AreaEntry<T>); //the entry before the gap

impl<T: AreaValue> AreaGap<T> {
    pub fn New(entry: AreaEntry<T>) -> Self {
        return Self(entry);
    }

    pub fn Ok(&self) -> bool {
        return !self.0.IsTail(); //use the tail for invalid AreaGap
    }

    pub fn PrevSeg(&self) -> AreaSeg<T> {
        if !self.Ok() {
            return AreaSeg::default();
        }

        return AreaSeg(self.0.clone()); //Gap's prevseg is always exist
    }

    pub fn NextSeg(&self) -> AreaSeg<T> {
        if !self.Ok() {
            return AreaSeg::default();
        }

        return AreaSeg(self.0.NextEntry().unwrap()); //Gap's entry won't be tail
    }

    pub fn PrevGap(&self) -> Self {
        if !self.0.Ok() {
            return Self::default();
        }

        let prevEntry = self.0.PrevEntry().unwrap(); //Gap's entry can't be tail;
        return Self(prevEntry);
    }

    pub fn NextGap(&self) -> Self {
        if !self.Ok() {
            return Self::default();
        }

        let nextEntry = self.0.NextEntry().unwrap(); //Gap's entry can't be tail;
        return Self(nextEntry);
    }

    pub fn Range(&self) -> Range {
        if !self.Ok() {
            return Range::New(0, 0);
        }

        let start = self.0.Range().End();
        let end = self.0.NextEntry().unwrap().Range().Start();
        return Range::New(start, end - start);
    }

    // IsEmpty returns true if the iterated gap is empty (that is, the "gap" is
    // between two adjacent segments.)
    pub fn IsEmpty(&self) -> bool {
        return self.Range().Len() == 0;
    }
}

#[derive(Default)]
pub struct AreaEntryInternal<T: AreaValue> {
    pub range: Range,
    pub value: Option<T>,
    pub prev: Option<AreaEntryWeak<T>>,
    pub next: Option<AreaEntry<T>>,
}

impl<T: AreaValue> AreaEntryInternal<T> {
    pub fn Value(&self) -> &T {
        return self
            .value
            .as_ref()
            .expect("AreaEntryInternal get None, it is head/tail");
    }
}

#[derive(Clone, Default)]
pub struct AreaEntryWeak<T: AreaValue>(pub Weak<QMutex<AreaEntryInternal<T>>>);

impl<T: AreaValue> AreaEntryWeak<T> {
    pub fn Upgrade(&self) -> Option<AreaEntry<T>> {
        let c = match self.0.upgrade() {
            None => return None,
            Some(c) => c,
        };

        return Some(AreaEntry(c));
    }
}

#[derive(Clone, Default)]
pub struct AreaEntry<T: AreaValue>(pub Arc<QMutex<AreaEntryInternal<T>>>);

impl<T: AreaValue> Deref for AreaEntry<T> {
    type Target = Arc<QMutex<AreaEntryInternal<T>>>;

    fn deref(&self) -> &Arc<QMutex<AreaEntryInternal<T>>> {
        &self.0
    }
}

impl<T: AreaValue> PartialEq for AreaEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        return Arc::ptr_eq(&self.0, &other.0);
    }
}

impl<T: AreaValue> Eq for AreaEntry<T> {}

impl<T: AreaValue> AreaEntry<T> {
    pub fn Downgrade(&self) -> AreaEntryWeak<T> {
        let c = Arc::downgrade(&self.0);
        return AreaEntryWeak(c);
    }

    pub fn Dummy(start: u64, len: u64) -> Self {
        let internal = AreaEntryInternal {
            range: Range::New(start, len),
            ..Default::default()
        };

        return Self(Arc::new(QMutex::new(internal)));
    }

    pub fn New(start: u64, len: u64, vma: T) -> Self {
        let internal = AreaEntryInternal {
            range: Range::New(start, len),
            value: Some(vma),
            ..Default::default()
        };

        return Self(Arc::new(QMutex::new(internal)));
    }

    pub fn Remove(&self) {
        let mut curr = self.lock();
        let prev = curr.prev.take().expect("prev is null");
        let next = curr.next.take().expect("next is null");

        prev.Upgrade().unwrap().lock().next = Some(next.clone());
        (*next).lock().prev = Some(prev);
    }

    pub fn InsertAfter(&self, r: &Range, vma: T) -> Self {
        let n = Self::New(r.Start(), r.Len(), vma);

        let next = self.lock().next.take().expect("next is null");
        self.lock().next = Some(n.clone());
        n.lock().prev = Some(self.Downgrade());

        next.lock().prev = Some(n.Downgrade());
        n.lock().next = Some(next.clone());

        return n;
    }

    //not head/tail
    pub fn Ok(&self) -> bool {
        let cur = self.lock();
        return !(cur.prev.is_none() || cur.next.is_none());
    }

    pub fn IsHead(&self) -> bool {
        return self.lock().prev.is_none();
    }

    pub fn IsTail(&self) -> bool {
        return self.lock().next.is_none();
    }

    pub fn Range(&self) -> Range {
        return self.lock().range;
    }

    //return next gap which not empty
    pub fn NextNonEmptyGap(&self) -> Option<AreaGap<T>> {
        let mut cur = self.clone();
        while !cur.IsTail() {
            //not tail
            let next = cur.NextEntry().unwrap();
            let end = next.Range().Start();
            let start = cur.Range().End();
            if start != end {
                return Some(AreaGap(cur));
            }

            cur = next;
        }

        return None;
    }

    //return NextEntry of current
    pub fn NextEntry(&self) -> Option<Self> {
        if self.IsTail() {
            return None;
        }

        let tmp = self.lock().next.clone().unwrap();
        return Some(tmp);
    }

    //return Prev Area of current
    pub fn PrevEntry(&self) -> Option<Self> {
        if self.IsHead() {
            return None;
        }

        let tmp = self.lock().prev.clone().unwrap();
        return tmp.Upgrade();
    }

    //return <prev Gap before the Area, the area before the Gap>
    pub fn PrevNoneEmptyGap(&self) -> Option<AreaGap<T>> {
        let mut cur = self.clone();
        while !cur.IsHead() {
            let prev = cur.PrevEntry().unwrap();
            let start = prev.Range().End();
            let end = cur.Range().Start();
            if start != end {
                return Some(AreaGap(prev));
            }
            cur = prev
        }

        return None;
    }
}

#[derive(Clone)]
pub struct AreaSet<T: AreaValue> {
    pub range: Range,

    pub head: AreaEntry<T>,
    pub tail: AreaEntry<T>,

    //<start, <len, val>>
    pub map: BTreeMap<u64, AreaEntry<T>>,
}

impl<T: AreaValue> AreaSet<T> {
    pub fn New(start: u64, len: u64) -> Self {
        let head = AreaEntry::Dummy(start, 0);
        let tail = if len != core::u64::MAX {
            AreaEntry::Dummy(start + len, 0)
        } else {
            AreaEntry::Dummy(core::u64::MAX, 0)
        };

        head.lock().next = Some(tail.clone());
        tail.lock().prev = Some(head.Downgrade());

        return Self {
            range: Range::New(start, len),
            head: head,
            tail: tail,
            map: BTreeMap::new(),
        };
    }

    pub fn Print(&self) -> String {
        let mut output = "".to_string();
        for (_, e) in &self.map {
            output += &format!("range is {:x?}\n", e.Range())
        }
        return output;
    }

    pub fn Reset(&mut self, start: u64, len: u64) {
        self.head.lock().range = Range::New(start, 0);
        self.tail.lock().range = Range::New(start + len, 0);
        self.range = Range::New(start, len);
    }

    // IsEmpty returns true if the set contains no segments.
    pub fn IsEmpty(&self) -> bool {
        return self.map.len() == 0;
    }

    // IsEmptyRange returns true iff no segments in the set overlap the given
    // range.
    pub fn IsEmptyRange(&self, range: &Range) -> bool {
        if range.Len() == 0 {
            return true;
        }

        let (_seg, gap) = self.Find(range.Start());

        if !gap.Ok() {
            return false;
        }

        return range.End() <= gap.Range().End();
    }

    // Span returns the total size of all segments in the set.
    pub fn Span(&self) -> u64 {
        let mut seg = self.FirstSeg();

        let mut sz = 0;
        while seg.Ok() {
            sz += seg.Range().Len();
            seg = seg.NextSeg();
        }

        return sz;
    }

    // SpanRange returns the total size of the intersection of segments in the set
    // with the given range.
    pub fn SpanRange(&self, r: &Range) -> u64 {
        if r.Len() == 0 {
            return 0;
        }

        let mut res = 0;

        let mut seg = self.LowerBoundSeg(r.Start());

        while seg.Ok() && seg.Range().Start() < r.End() {
            res += seg.Range().Intersect(r).Len();
            seg = seg.NextSeg();
        }

        return res;
    }

    //return first valid seg, i.e. not include head/tail
    pub fn FirstSeg(&self) -> AreaSeg<T> {
        let first = self.head.NextEntry().unwrap();
        return AreaSeg(first);
    }

    //return last valid seg, i.e. not include head/tail
    pub fn LastSeg(&self) -> AreaSeg<T> {
        let last = self.tail.PrevEntry().unwrap();
        return AreaSeg(last);
    }

    pub fn FirstGap(&self) -> AreaGap<T> {
        return AreaGap(self.head.clone());
    }

    pub fn LastGap(&self) -> AreaGap<T> {
        let prev = self.tail.PrevEntry().unwrap();
        return AreaGap(prev);
    }

    pub fn FindArea(&self, key: u64) -> Area<T> {
        if key < self.range.Start() {
            return Area::AreaSeg(AreaSeg(self.head.clone()));
        }

        if key > self.range.End() {
            return Area::AreaSeg(AreaSeg(self.tail.clone()));
        }

        let mut iter = self.map.range((Unbounded, Included(key))).rev();
        let entry = match iter.next() {
            None => self.head.clone(),
            Some((_, v)) => v.clone(),
        };

        if entry.Ok() && entry.Range().Contains(key) {
            return Area::AreaSeg(AreaSeg(entry));
        }

        return Area::AreaGap(AreaGap(entry));
    }

    pub fn Find(&self, key: u64) -> (AreaSeg<T>, AreaGap<T>) {
        let area = self.FindArea(key);
        return (area.AreaSeg(), area.AreaGap());
    }

    // FindSegment returns the segment whose range contains the given key. If no
    // such segment exists, FindSegment returns a terminal iterator.
    pub fn FindSeg(&self, key: u64) -> AreaSeg<T> {
        let area = self.FindArea(key);
        return area.AreaSeg();
    }

    // LowerBoundSeg returns the segment with the lowest range that contains a
    // key greater than or equal to min. If no such segment exists,
    // LowerBoundSegment returns a terminal iterator.
    pub fn LowerBoundSeg(&self, key: u64) -> AreaSeg<T> {
        let (seg, gap) = self.Find(key);
        if seg.Ok() {
            return seg;
        }

        if gap.Ok() {
            return gap.NextSeg();
        }

        return AreaSeg::default();
    }

    // UpperBoundSegment returns the segment with the highest range that contains a
    // key less than or equal to max. If no such segment exists, UpperBoundSegment
    // returns a terminal iterator.
    pub fn UpperBoundSeg(&self, key: u64) -> AreaSeg<T> {
        let (seg, gap) = self.Find(key);
        if seg.Ok() {
            return seg;
        }

        if gap.Ok() {
            return gap.PrevSeg();
        }

        return AreaSeg::default();
    }

    //return the Gap containers the key
    pub fn FindGap(&self, key: u64) -> AreaGap<T> {
        let area = self.FindArea(key);
        return area.AreaGap();
    }

    // LowerBoundGap returns the gap with the lowest range that is greater than or
    // equal to min.
    pub fn LowerBoundGap(&self, key: u64) -> AreaGap<T> {
        let (seg, gap) = self.Find(key);
        if gap.Ok() {
            return gap;
        }

        if seg.Ok() {
            return seg.NextGap();
        }

        return AreaGap::default();
    }

    // UpperBoundGap returns the gap with the highest range that is less than or
    // equal to max.
    pub fn UpperBoundGap(&self, key: u64) -> AreaGap<T> {
        let (seg, gap) = self.Find(key);
        if gap.Ok() {
            return gap;
        }

        if seg.Ok() {
            return seg.PrevGap();
        }

        return AreaGap::default();
    }

    // Add inserts the given segment into the set and returns true. If the new
    // segment can be merged with adjacent segments, Add will do so. If the new
    // segment would overlap an existing segment, Add returns false. If Add
    // succeeds, all existing iterators are invalidated.
    pub fn Add(&mut self, r: &Range, val: T) -> bool {
        let gap = self.FindGap(r.Start());
        if !gap.Ok() {
            return false;
        }

        if r.End() > gap.Range().End() {
            return false;
        }

        self.Insert(&gap, r, val);
        return true;
    }

    // AddWithoutMerging inserts the given segment into the set and returns true.
    // If it would overlap an existing segment, AddWithoutMerging does nothing and
    // returns false. If AddWithoutMerging succeeds, all existing iterators are
    // invalidated.
    pub fn AddWithoutMerging(&mut self, r: &Range, val: T) -> bool {
        let gap = self.FindGap(r.Start());
        if !gap.Ok() {
            return false;
        }

        if r.End() > gap.Range().End() {
            return false;
        }

        self.InsertWithoutMergingUnchecked(&gap, r, val);
        return true;
    }

    // Insert inserts the given segment into the given gap. If the new segment can
    // be merged with adjacent segments, Insert will do so. Insert returns an
    // iterator to the segment containing the inserted value (which may have been
    // merged with other values). All existing iterators (including gap, but not
    // including the returned iterator) are invalidated.
    //
    // If the gap cannot accommodate the segment, or if r is invalid, Insert panics.
    //
    // Insert is semantically equivalent to a InsertWithoutMerging followed by a
    // Merge, but may be more efficient. Note that there is no unchecked variant of
    // Insert since Insert must retrieve and inspect gap's predecessor and
    // successor segments regardless.
    pub fn Insert(&mut self, gap: &AreaGap<T>, r: &Range, val: T) -> AreaSeg<T> {
        let prev = gap.PrevSeg();
        let next = gap.NextSeg();

        if prev.Ok() && prev.Range().End() > r.Start() {
            panic!(
                "new segment {:x?} overlaps predecessor {:x?}",
                r,
                prev.Range()
            )
        }

        if next.Ok() && next.Range().Start() < r.End() {
            panic!(
                "new segment {:x?} overlaps successor {:x?}",
                r,
                next.Range()
            )
        }

        //can't enable merge segment as the return merged segment might override the COW page
        //todo: fix this
        let r1 = prev.Range();
        if prev.Ok() && prev.Range().End() == r.Start() {
            let mval = prev.lock().Value().Merge(&r1, &r, &val);
            if mval.is_some() {
                prev.lock().range.len += r.Len();
                prev.lock().value = mval.clone();
                if next.Ok() && next.Range().Start() == r.End() {
                    let val = mval.clone().unwrap();
                    let r1 = prev.Range();
                    let r2 = next.Range();
                    let mval = val.Merge(&r1, &r2, next.lock().Value());
                    if mval.is_some() {
                        prev.lock().range.len += next.Range().Len();
                        prev.lock().value = mval;
                        return self.Remove(&next).PrevSeg();
                    }
                }

                return prev;
            }
        }

        if next.Ok() && next.Range().Start() == r.End() {
            let r2 = next.Range();
            let mval = val.Merge(&r, &r2, next.lock().Value());
            if mval.is_some() {
                next.lock().range.start = r.Start();
                next.lock().range.len += r.Len();
                next.lock().value = mval;
                self.map.remove(&r.End());
                self.map.insert(r.Start(), next.0.clone());
                return next;
            }
        }

        return self.InsertWithoutMergingUnchecked(gap, r, val);
    }

    // InsertWithoutMerging inserts the given segment into the given gap and
    // returns an iterator to the inserted segment. All existing iterators
    // (including gap, but not including the returned iterator) are invalidated.
    //
    // If the gap cannot accommodate the segment, or if r is invalid,
    // InsertWithoutMerging panics.
    pub fn InsertWithoutMerging(&mut self, gap: &AreaGap<T>, r: &Range, val: T) -> AreaSeg<T> {
        let gr = gap.Range();
        if !gr.IsSupersetOf(r) {
            panic!(
                "cannot insert segment range {:?} into gap range {:?}",
                r, gr
            );
        }

        return self.InsertWithoutMergingUnchecked(gap, r, val);
    }

    // InsertWithoutMergingUnchecked inserts the given segment into the given gap
    // and returns an iterator to the inserted segment. All existing iterators
    // (including gap, but not including the returned iterator) are invalidated.
    pub fn InsertWithoutMergingUnchecked(
        &mut self,
        gap: &AreaGap<T>,
        r: &Range,
        val: T,
    ) -> AreaSeg<T> {
        let prev = gap.PrevSeg();
        let n = prev.InsertAfter(r, val);
        self.map.insert(r.Start(), n.clone());
        return AreaSeg(n);
    }

    // Remove removes the given segment and returns an iterator to the vacated gap.
    // All existing iterators (including seg, but not including the returned
    // iterator) are invalidated.
    pub fn Remove(&mut self, e: &AreaSeg<T>) -> AreaGap<T> {
        assert!(e.Ok(), "Remove: can't remove head/tail entry");
        let prev = e.PrevSeg();

        self.map.remove(&e.Range().Start());
        e.Remove();

        return prev.NextGap();
    }

    // RemoveAll removes all segments from the set. All existing iterators are
    // invalidated.
    pub fn RemoveAll(&mut self) {
        for (_, e) in &self.map {
            e.Remove();
        }

        self.map.clear();
    }

    // RemoveRange removes all segments in the given range. An iterator to the
    // newly formed gap is returned, and all existing iterators are invalidated.
    pub fn RemoveRange(&mut self, r: &Range) -> AreaGap<T> {
        let (mut seg, mut gap) = self.Find(r.Start());

        if seg.Ok() {
            seg = self.Isolate(&seg, r);
            gap = self.Remove(&seg);
        }

        seg = gap.NextSeg();
        while seg.Ok() && seg.Range().Start() < r.End() {
            seg = self.Isolate(&seg, r);
            gap = self.Remove(&seg);
        }

        return gap;
    }

    // Merge attempts to merge two neighboring segments. If successful, Merge
    // returns an iterator to the merged segment, and all existing iterators are
    // invalidated. Otherwise, Merge returns a terminal iterator.
    //
    // If first is not the predecessor of second, Merge panics.
    pub fn Merge(&mut self, first: &AreaSeg<T>, second: &AreaSeg<T>) -> AreaSeg<T> {
        if first.NextSeg() != second.clone() {
            panic!(
                "attempt to merge non-neighboring segments {:?}, {:?}",
                first.Range(),
                second.Range()
            )
        }

        return self.MergeUnchecked(first, second);
    }

    // MergeUnchecked attempts to merge two neighboring segments. If successful,
    // MergeUnchecked returns an iterator to the merged segment, and all existing
    // iterators are invalidated. Otherwise, MergeUnchecked returns a terminal
    // iterator.
    pub fn MergeUnchecked(&mut self, first: &AreaSeg<T>, second: &AreaSeg<T>) -> AreaSeg<T> {
        if !first.Ok() || !second.Ok() {
            //can't merge head or tail
            return AreaSeg::default();
        }

        if first.Range().End() == second.Range().Start() {
            let r1 = first.Range();
            let r2 = second.Range();

            let vma = first.lock().Value().Merge(&r1, &r2, second.lock().Value());
            if vma.is_some() {
                first.lock().range.len += second.lock().range.len;
                first.lock().value = vma;
                self.Remove(&second);
                return first.clone();
            }
        }

        return AreaSeg::default();
    }

    // MergeAll attempts to merge all adjacent segments in the set. All existing
    // iterators are invalidated.
    pub fn MergeAll(&mut self) {
        let mut seg = self.FirstSeg();
        if !seg.Ok() {
            return;
        }

        let mut next = seg.NextSeg();

        while next.Ok() {
            let mseg = self.MergeUnchecked(&seg, &next);
            if mseg.Ok() {
                seg = mseg;
                next = seg.NextSeg();
            } else {
                seg = next;
                next = seg.NextSeg();
            }
        }
    }

    // MergeRange attempts to merge all adjacent segments that contain a key in the
    // specific range.
    pub fn MergeRange(&mut self, r: &Range) {
        let mut seg = self.LowerBoundSeg(r.Start());
        if !seg.Ok() {
            return;
        }

        let mut next = seg.NextSeg();

        while next.Ok() && next.Range().Start() < r.End() {
            let mseg = self.MergeUnchecked(&seg, &next);
            if mseg.Ok() {
                seg = mseg;
                next = seg.NextSeg();
            } else {
                seg = next;
                next = seg.NextSeg();
            }
        }
    }

    // MergeAdjacent attempts to merge the segment containing r.Start with its
    // predecessor, and the segment containing r.End-1 with its successor.
    pub fn MergeAdjacent(&mut self, r: &Range) {
        let first = self.FindSeg(r.Start());
        if first.Ok() {
            let prev = first.PrevSeg();
            if prev.Ok() {
                self.Merge(&prev, &first);
            }
        }

        let last = self.FindSeg(r.End() - 1);
        if last.Ok() {
            let next = last.NextSeg();
            if next.Ok() {
                self.Merge(&last, &next);
            }
        }
    }

    // Split splits the given segment at the given key and returns iterators to the
    // two resulting segments. All existing iterators (including seg, but not
    // including the returned iterators) are invalidated.
    //
    // If the segment cannot be split at split (because split is at the start or
    // end of the segment's range, so splitting would produce a segment with zero
    // length, or because split falls outside the segment's range altogether),
    // Split panics.
    pub fn Split(&mut self, seg: &AreaSeg<T>, split: u64) -> (AreaSeg<T>, AreaSeg<T>) {
        let r = seg.Range();
        if !r.CanSplitAt(split) {
            panic!("can't split {:?} at {:?}", r, split);
        }

        return self.SplitUnchecked(seg, split);
    }

    // SplitUnchecked splits the given segment at the given key and returns
    // iterators to the two resulting segments. All existing iterators (including
    // seg, but not including the returned iterators) are invalidated.
    pub fn SplitUnchecked(&mut self, seg: &AreaSeg<T>, split: u64) -> (AreaSeg<T>, AreaSeg<T>) {
        let range = seg.Range();
        let (val1, val2) = seg.lock().Value().Split(&range, split);
        let end2 = range.End();
        seg.lock().range.len = split - range.Start();
        seg.lock().value = Some(val1);
        let gap = seg.NextNonEmptyGap().unwrap();
        let seg2 = self.InsertWithoutMergingUnchecked(&gap, &Range::New(split, end2 - split), val2);

        let prev = seg2.PrevSeg();
        return (prev, seg2);
    }

    // SplitAt splits the segment straddling split, if one exists. SplitAt returns
    // true if a segment was split and false otherwise. If SplitAt splits a
    // segment, all existing iterators are invalidated.
    pub fn SplitAt(&mut self, split: u64) -> bool {
        let seg = self.FindSeg(split);

        if !seg.Ok() {
            return false;
        }

        let r = seg.Range();
        if r.CanSplitAt(split) {
            self.SplitUnchecked(&seg, split);
            return true;
        }

        return false;
    }

    // Isolate ensures that the given segment's range does not escape r by
    // splitting at r.Start and r.End if necessary, and returns an updated iterator
    // to the bounded segment. All existing iterators (including seg, but not
    // including the returned iterators) are invalidated.
    pub fn Isolate(&mut self, seg: &AreaSeg<T>, r: &Range) -> AreaSeg<T> {
        let mut seg = seg.clone();

        let curr = seg.Range();
        if curr.CanSplitAt(r.Start()) {
            let (_, tmp) = self.SplitUnchecked(&seg, r.Start());
            seg = tmp;
        }

        let curr = seg.Range();
        if curr.CanSplitAt(r.End()) {
            let (tmp, _) = self.SplitUnchecked(&seg, r.End());
            seg = tmp;
        }

        return seg;
    }

    // ApplyContiguous applies a function to a contiguous range of segments,
    // splitting if necessary. The function is applied until the first gap is
    // encountered, at which point the gap is returned. If the function is applied
    // across the entire range, a terminal gap is returned. All existing iterators
    // are invalidated.
    pub fn ApplyContiguous(&mut self, r: &Range, mut f: impl FnMut(&AreaEntry<T>)) -> AreaGap<T> {
        let (mut seg, mut gap) = self.Find(r.Start());

        if !seg.Ok() {
            return gap;
        }

        loop {
            seg = self.Isolate(&seg, r);
            f(&seg);

            if seg.Range().End() > r.End() {
                return AreaGap::default();
            }

            gap = seg.NextGap();
            if !gap.IsEmpty() {
                return gap;
            }

            seg = gap.NextSeg();
            if !seg.Ok() {
                return AreaGap::default();
            }
        }
    }
}
