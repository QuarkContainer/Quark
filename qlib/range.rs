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

use alloc::collections::btree_map::BTreeMap;
use core::ops::Bound::*;
use alloc::vec::Vec;
use core::cmp::*;
use core::ops::Deref;
use spin::Mutex;

use super::addr::*;
use super::common::{Error, Result};

pub const MAX_RANGE : u64 = core::u64::MAX;

#[derive(Copy, Clone, Default, Debug)]
pub struct Range {
    pub start: u64,
    pub len: u64
}

impl Ord for Range {
    fn cmp(&self, other: &Self) -> Ordering {
        let startCmp = self.start.cmp(&other.start);
        if startCmp != Ordering::Equal {
            return startCmp;
        }

        return self.len.cmp(&other.len);
    }
}

impl PartialOrd for Range {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Range {
    fn eq(&self, other: &Self) -> bool {
        return self.start == other.start && self.len == self.len
    }
}

impl Eq for Range {}

impl Range {
    pub fn New(start: u64, len: u64) -> Self {
        return Range { start, len }
    }

    pub fn Max() -> Self {
        return Range {start: 0, len: MAX_RANGE}
    }

    pub fn Start(&self) -> u64 {
        return self.start
    }

    pub fn End(&self) -> u64 {
        if self.len == MAX_RANGE {
            return MAX_RANGE;
        }

        return self.start + self.len;
    }

    pub fn Print(&self) {
        info!("{:x}:{:x}", self.start, self.len);
    }

    pub fn Len(&self) -> u64 {
        return self.len;
    }

    //alloc memory space,
    //len: mem size
    //aligmentOrd: mem aligment order, 2^ord
    pub fn Alloc(&self, len: u64, aligmentOrd: u64) -> Option<u64> {
        let mut start = self.start;
        let size = 1 << aligmentOrd;
        let mask = size - 1;

        if start & mask != 0 {
            start = start & (!mask);
            start += size;
        }

        if self.End() > start && self.End() - start >= len {
            return Some(start)
        }

        return None
    }

    pub fn Contains(&self, x: u64) -> bool {
        return x >= self.start && x < self.start + self.len
    }

    pub fn Overlaps(&self, r: &Range) -> bool {
        return self.start < r.End() && r.start < self.End()
    }

    pub fn IsSupersetOf(&self, r: &Range) -> bool {
        return self.start <= r.start && self.End() >= r.End()
    }

    pub fn Intersect(&self, r: &Range) -> Range {
        let mut start = self.start;
        let mut end = self.End();

        if self.start < r.start {
            start = r.start
        }

        if self.End() > r.End() {
            end = r.End();
        }

        return Range {
            start,
            len: end - start,
        }
    }

    pub fn CanSplitAt(&self, x: u64) -> bool {
        return self.start < x && x < self.End()
    }

    pub fn IsPageAigned(&self) -> bool {
        return Addr(self.start).IsPageAligned() && Addr(self.len).IsPageAligned()
    }
}

pub struct AreaMgr<T: core::clone::Clone> {
    //<start, (len, val)>
    pub map: BTreeMap<u64, (u64, T)>
}

impl<T: core::clone::Clone> AreaMgr<T> {
    pub fn New() -> Self {
        return AreaMgr {
            map: BTreeMap::new(),
        }
    }

    pub fn Span(&self) -> u64 {
        let mut res = 0;
        for (_, (len, _)) in &self.map {
            res += *len;
        }

        return res;
    }

    pub fn Fork(&self) -> Self {
        let mut res = Self::New();
        for (start, (len, val)) in &self.map {
            res.map.insert(*start, (*len, val.clone()));
        }

        return res;
    }

    pub fn Print(&self) {
        for (start, (len, _)) in &self.map {
            info!("AreaMgr: the start is {:x}, len is {:x}, end is {:x}", *start, *len, *start + *len);
        }
    }

    pub fn RemoveRange(&mut self, start: u64, len: u64) {
        let r = Range { start, len };

        //let mut inserts : Vec<(u64, u64, Arc<RwLock<Box<T>>>)> = Vec::new();
        let mut inserts: Vec<(u64, u64, T)> = Vec::new();
        let mut removes: Vec<u64> = Vec::new();

        for (cStart, (cLen, cData)) in self.map.range_mut((Unbounded, Excluded(r.End()))).rev() {
            let cR = Range { start: *cStart, len: *cLen };
            if r.Contains(cR.Start()) {
                removes.push(cR.Start()); //remove the current area
                if !r.Contains(cR.End()) && cR.End() - r.End() != 0 {
                    // add remain part of current area
                    inserts.push((r.End(), cR.End() - r.End(), cData.clone()));
                }
            } else {
                if cR.Contains(r.Start()) && r.Start() - cR.Start() != 0 {
                    *cLen = r.Start() - cR.Start(); //change the current area len
                }
            }

            if cR.End() < start {
                break;
            }
        }

        for (_, removeKey) in removes.iter().enumerate() {
            self.map.remove(removeKey);
        }

        for (_, (iStart, iLen, iData)) in inserts.iter().enumerate() {
            self.map.insert(*iStart, (*iLen, iData.clone()));
        }
    }

    //pub fn Add(&mut self, start: u64, len: u64, data: &Arc<RwLock<Box<T>>>) {
    pub fn Add(&mut self, start: u64, len: u64, data: &T) {
        self.RemoveRange(start, len);
        self.map.insert(start, (len, data.clone()));
    }

    //for mProtect, change the data of the range
    pub fn Protect(&mut self, addr: u64, len: u64, f: impl Fn(&T) -> Result<T>) -> Result<()> {
        let r = Range::New(addr, len);

        let mut hasPrevious = false;
        let mut pEnd = 0;
        let mut finish = false;

        for (cStart, (cLen, cData)) in self.map.range((Unbounded, Excluded(r.End()))) {
            let cR = Range::New(*cStart, *cLen);
            if cR.Overlaps(&r) {
                if !hasPrevious {
                    hasPrevious = true;
                } else {
                    if cR.Start() != pEnd {
                        return Err(Error::InvalidInput);
                    }
                }

                pEnd = cR.End();

                if cR.End() >= r.End() {
                    finish = true;
                }

                f(&*cData)?;
            }
        }

        if !finish {
            return Err(Error::InvalidInput);
        }

        let mut inserts = Vec::new();
        for (cStart, (cLen, cData)) in self.map.range((Unbounded, Excluded(r.End()))) {
            let cR = Range::New(*cStart, *cLen);
            if cR.Overlaps(&r) {
                let iR = cR.Intersect(&r);
                let data = f(&*cData)?;
                inserts.push((iR.Start(), iR.Len(), data));
                //self.Add(iR.Start(), iR.Len(), &data);
            }
        }

        for i in 0..inserts.len() {
            let (start, len, data) = &inserts[i];
            self.Add(*start, *len, &*data);
        }

        return Ok(())
    }

    pub fn Delete(&mut self, start: u64) -> Result<()> {
        //todo: add more validation
        if !self.map.contains_key(&start) {
            return Err(Error::InvalidInput)
        }

        self.map.remove(&start);
        return Ok(())
    }

    //pub fn Get(&self, key: u64) -> Option<(u64, u64, Arc<RwLock<Box<T>>>)> {
    pub fn Get(&self, key: u64) -> Option<(u64, u64, &T)> {
        let mut iter = self.map.range((Unbounded, Included(key))).rev();
        match iter.next() {
            None => return None,
            Some((start, (len, data))) => {
                let r = Range { start: *start, len: *len };
                if !r.Contains(key) {
                    return None
                } else {
                    return Some((*start, *len, &*data))
                }
            }
        }
    }

    pub fn GetMut(&mut self, key: u64) -> Option<(u64, u64, &mut T)> {
        let mut iter = self.map.range_mut((Unbounded, Included(key))).rev();
        match iter.next() {
            None => return None,
            Some((start, (len, data))) => {
                let r = Range { start: *start, len: *len };
                if !r.Contains(key) {
                    return None
                } else {
                    return Some((*start, *len, &mut *data))
                }
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct BufMgr (Mutex<BufMgrIntern>);

impl Deref for BufMgr {
    type Target = Mutex<BufMgrIntern>;

    fn deref(&self) -> &Mutex<BufMgrIntern> {
        &self.0
    }
}

impl BufMgr {
    pub fn Init(&self, start: u64, len: u64) {
        self.lock().Init(start, len);
    }

    pub fn New() -> Self {
        let intern = BufMgrIntern::New();
        return Self(Mutex::new(intern))
    }

    pub fn Alloc(&self, len: u64) -> Result<u64> {
        return self.lock().Alloc(len);
    }

    pub fn Free(&self, start: u64, len: u64) {
        return self.lock().Free(start, len);
    }
}

#[derive(Debug, Clone, Default)]
pub struct BufMgrIntern {
    pub next: u64,
    pub gapMgr: GapMgr,
    pub buf: Vec<u8>,
}

impl BufMgrIntern {
    pub fn Init(&mut self, start: u64, len: u64) {
        self.next = start;
        self.gapMgr.Init(start, len);
    }

    pub fn New() -> Self {
        let memoryOrd = 23; // 8 mb
        let size = 1 << memoryOrd;
        let mut buf = Vec::with_capacity(size);
        unsafe {
            buf.set_len(size);
        }

        let start = &buf[0] as * const _ as u64;

        return Self {
            next: start,
            gapMgr: GapMgr::New(start, size as u64),
            buf: buf,
        }
    }

    pub fn Alloc(&mut self, len: u64) -> Result<u64> {
        match self.gapMgr.AllocAfter(self.next, len, 0) {
            Ok(addr) => {
                self.next = addr + len;
                return Ok(addr)
            }
            Err(e) => {
                if self.next == self.gapMgr.range.Start() {
                    return Err(e)
                }
            }
        }

        match self.gapMgr.AllocAfter(self.gapMgr.range.Start(), len, 0) {
            Ok(addr) => {
                self.next = addr + len;
                return Ok(addr)
            }
            Err(e) => return Err(e)
        }
    }

    pub fn Free(&mut self, start: u64, len: u64) {
        //print!("PrintStrRespHandler free start is {:x}, len is {:x}", start, len);
        self.gapMgr.Free(start, len)
    }
}

#[derive(Debug, Clone, Default)]
pub struct GapMgr {
    pub range: Range,
    pub map: BTreeMap<u64, u64>,
    //store the gaps, key: gapStart, val: gapLen
}

impl GapMgr {
    pub fn New(start: u64, len: u64) -> Self {
        let mut map = BTreeMap::new();
        map.insert(start, len);
        return GapMgr {
            range: Range { start, len },
            map,
        }
    }

    pub fn Init(&mut self, start: u64, len: u64) {
        assert!(self.map.len() == 0);
        self.range = Range::New(start, len);
        self.map.insert(start, len);
    }

    pub fn Fork(&self) -> Self {
        let range = Range::New(self.range.start, self.range.len);
        let mut map = BTreeMap::new();
        for (start, len) in &self.map {
            map.insert(*start, *len);
        }

        return GapMgr {
            range,
            map,
        }
    }

    pub fn Print(&self) {
        info!("GapMgr: the full range is {:x} to {:x}", self.range.start, self.range.End());
        for (start, len) in &self.map {
            info!("the gap start is {:x}, len is {:x}, end is {:x}", *start, *len, *start + *len);
        }
    }

    //test function
    //return range with (gapStart, gapEnd)
    pub fn Find(&self, key: u64) -> Option<(u64, u64)> {
        let mut before = self.map.range((Included(self.range.Start()), Included(key)));

        match before.next_back() {
            None => None,
            Some((start, len)) => {
                return Some((*start, *start + *len))
            }
        }
    }

    //remove the gap belong to the input range
    pub fn Take(&mut self, start: u64, len: u64) {
        let r = Range { start, len };

        let mut removes = Vec::new();
        let mut needInsert = false;
        let mut insertStart = 0;
        let mut insertLen = 0;

        for (gStart, gLen) in self.map.range_mut((Unbounded, Excluded(r.End()))).rev() {
            let gR = Range { start: *gStart, len: *gLen };
            if gR.Start() < r.End() && r.End() < gR.End() {
                needInsert = true;
                insertStart = r.End();
                insertLen = gR.End() - r.End();
            }

            if r.Start() <= gR.Start() && gR.Start() < r.End() {
                removes.push(*gStart)
            }

            if gR.Start() <= r.Start() && r.Start() < gR.End() {
                *gLen = r.Start() - gR.Start();
            }

            if gR.End() <= r.Start() {
                break;
            }
        }

        for i in 0..removes.len() {
            let val = removes[i];
            self.map.remove(&val);
        }

        if needInsert {
            self.map.insert(insertStart, insertLen);
        }
    }

    pub fn AllocAfter(&mut self, addr: u64, len: u64, aligmentOrd: u64) -> Result<u64> {
        let mut res = 0;
        let mut hasResult = false;
        let mut cStart = 0;
        let mut cLen = 0;

        for (gStart, gLen) in self.map.range((Included(addr), Unbounded)).rev() {
            //todo: optimize
            if *gStart + *gLen < addr + len || *gLen < len {
                continue;
            }

            //tmp = max(*start, addr)
            let mut tmp = *gStart;
            if tmp < addr {
                tmp = addr;
            }

            let ret = Range { start: tmp, len: *gStart + *gLen - tmp }.Alloc(len, aligmentOrd);
            match ret {
                Some(start) => {
                    //info!("the start is {}, len is {}, alignment is {}", start, len, aligmentOrd);

                    hasResult = true;
                    res = start;

                    cStart = *gStart;
                    cLen = *gLen;
                    break;
                }
                _ => ()
            }
        }

        //self.Print();

        if hasResult {
            if cStart == res {
                self.map.remove(&cStart);
            } else {
                *self.map.get_mut(&cStart).unwrap() = res - cStart;
            }

            //self.Print();
            //info!("the addr is {:x} the res is {:x}, the len is {:x}, the cLen is {:x}, cStart is {:x}, cStart+cLen is {:x}, res+len={:x}, MemoryDef::PHY_UPPER_ADDR is {:x}",
            //         addr, res, len, cLen, cStart, cStart+cLen, res + len, super::libcDef::MemoryDef::PHY_UPPER_ADDR);
            if res + len != cStart + cLen {
                self.map.insert(res + len, cStart + cLen - (res + len));
            }

            return Ok(res)
        }

        return Err(Error::NoEnoughSpace)
    }

    pub fn Alloc(&mut self, len: u64, aligmentOrd: u64) -> Result<u64> {
        return self.AllocAfter(self.range.start, len, aligmentOrd);
    }

    pub fn Free(&mut self, start: u64, len: u64) {
        let range = Range { start, len };

        let mut before = self.map.range((Unbounded, Excluded(range.Start())));
        let mut after = self.map.range((Excluded(range.Start()), Unbounded));

        let mut nStart = range.Start();
        let mut nEnd = range.End();

        let mut removeBefore = false;
        let mut bKey = 0;
        match before.next_back() {
            Some((bStart, bLen)) => {
                if *bStart + *bLen == range.Start() {
                    removeBefore = true;
                    bKey = *bStart;
                    nStart = *bStart;
                }
            }
            None => ()
        }

        let mut removeAfter = false;
        let mut aKey = 0;
        match after.next() {
            Some((aStart, aLen)) => {
                if range.End() == *aStart {
                    removeAfter = true;
                    aKey = *aStart;
                    nEnd = *aStart + *aLen;
                }
            }

            None => ()
        }

        if removeBefore {
            self.map.remove(&bKey);
        }

        if removeAfter {
            self.map.remove(&aKey);
        }

        self.map.insert(nStart, nEnd - nStart);
    }
}

pub struct IdMgr<T: core::clone::Clone> {
    pub gaps: GapMgr,
    pub map: BTreeMap<u64, T>,
    pub start: u64,
    pub len: u64,
    pub last: u64,
}

impl<T: core::clone::Clone> IdMgr<T> {
    pub fn Init(start: u64, len: u64) -> Self {
        return IdMgr {
            gaps: GapMgr::New(start, len),
            map: BTreeMap::new(),
            start: start,
            len: len,
            last: 0,
        }
    }

    pub fn AllocId(&mut self) -> Result<u64> {
        let id = match self.gaps.AllocAfter(self.last, 1, 0) {
            Ok(id) => id,
            _ => {
                self.gaps.AllocAfter(0, 1, 0)?
            }
        };

        self.last = id;
        return Ok(id)
    }

    pub fn Add(&mut self, id: u64, data: T) {
        /*let id = match self.gaps.AllocAfter(self.last, 1, 0) {
            Ok(id) => id,
            _ => {
                self.gaps.AllocAfter(0, 1, 0)?
            }
        };

        self.last = id;*/
        self.map.insert(id, data);
        //return Ok(id)
    }

    pub fn Get(&self, id: u64) -> Result<&T> {
        match self.map.get(&id) {
            None => return Err(Error::NotExist),
            Some(data) => return Ok(data)
        }
    }

    pub fn GetMut(&mut self, id: u64) -> Option<&mut T> {
        return self.map.get_mut(&id)
    }

    pub fn Remove(&mut self, id: u64) -> Option<T> {
        if self.map.contains_key(&id) {
            self.gaps.Free(id, 1);
            return self.map.remove(&id)
        } else {
            return None
        }
    }
}
