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
use core::ops::Bound::{Excluded, Included, Unbounded};

use super::range::*;

#[derive(Debug, Clone, Default)]
pub struct IdAllocator {
    range: Range,
    map: BTreeMap<u64, u64>,
    prevAllocatedId: u64,
}

impl IdAllocator {
    pub fn New(start: u64, len: u64) -> Self {
        let mut map = BTreeMap::new();
        let range = Range { start, len };
        map.insert(range.End() - 1, len);
        return IdAllocator {
            range: range,
            map,
            prevAllocatedId: start - 1,
        };
    }

    pub fn LastGap(&self) -> Option<Range> {
        for (end, len) in self.map.iter().rev() {
            return Some(Range::New(*end - *len + 1, *len));
        }

        return None;
    }

    pub fn Fork(&self) -> Self {
        let range = Range::New(self.range.start, self.range.len);
        let mut map = BTreeMap::new();
        for (end, len) in &self.map {
            map.insert(*end, *len);
        }

        return IdAllocator {
            range,
            map,
            prevAllocatedId: self.range.start - 1,
        };
    }

    pub fn Print(&self) {
        info!(
            "GapMgr: the full range is {:x} to {:x}",
            self.range.start,
            self.range.End()
        );
        for (start, len) in &self.map {
            info!(
                "the gap start is {:x}, len is {:x}, end is {:x}",
                *start,
                *len,
                *start + *len
            );
        }
    }

    pub fn Take(&mut self, id: u64) {
        let mut remove = None;
        let mut insert = None;

        for (gEnd, gLen) in self.map.range_mut((Included(id), Unbounded)) {
            let gR = Range {
                start: *gEnd - (*gLen - 1),
                len: *gLen,
            };

            assert!(
                gR.Contains(id),
                "Take fail {}/{:?}/{:?}",
                id,
                &gR,
                &self.map
            );

            // This is the only one in the range
            if gR.Len() == 1 {
                remove = Some(*gEnd);
                break;
            }

            // there is left range need to add
            if gR.Start() < id {
                insert = Some(Range::New(gR.Start(), id - gR.Start()));
            }

            // the right range need to remove
            if gR.End() - 1 == id {
                remove = Some(*gEnd);
            } else {
                // right range need to shrink
                *gLen = *gEnd - id;
            }

            break;
        }

        match remove {
            None => (),
            Some(end) => {
                self.map.remove(&end);
            }
        }

        match insert {
            None => (),
            Some(r) => {
                self.map.insert(r.End() - 1, r.Len());
            }
        }
    }

    fn AllocAfterInternal(&mut self, id: u64) -> Option<u64> {
        let mut firstRange = None;
        for (gEnd, gLen) in self.map.range((Included(id), Unbounded)) {
            firstRange = Some(Range::New(*gEnd - (*gLen - 1), *gLen));
            break;
        }

        match firstRange {
            None => return None,
            Some(r) => {
                if r.Start() < id {
                    self.Take(id);
                    return Some(id);
                } else {
                    if r.Len() > 1 {
                        self.map.insert(r.End() - 1, r.Len() - 1);
                    } else {
                        self.map.remove(&(r.End() - 1));
                    }

                    return Some(r.Start());
                }
            }
        }
    }

    pub fn AllocFromStart(&mut self) -> Option<u64> {
        return self.AllocAfterInternal(self.range.start);
    }

    pub fn AllocFromCurrent(&mut self) -> Option<u64> {
        let mut id = self.AllocAfterInternal(self.prevAllocatedId + 1);
        if id.is_none() && self.prevAllocatedId != self.range.start - 1 {
            id = self.AllocFromStart();
        }
        if id.is_some() {
            self.prevAllocatedId = id.unwrap();
        }
        return id;
    }

    pub fn Free(&mut self, id: u64) {
        let leftRange = if id == 0 {
            None
        } else {
            match self.map.get(&(id - 1)) {
                None => None,
                Some(len) => Some(Range::New(id - 1 - (*len - 1), *len)),
            }
        };

        let mut rightRange = None;

        for (gEnd, gLen) in self.map.range((Excluded(id), Unbounded)) {
            let range = Range::New(*gEnd - (*gLen - 1), *gLen);
            if range.Start() == id + 1 {
                rightRange = Some(range);
            } else {
                rightRange = None;
            }
            break;
        }

        if leftRange.is_none() {
            match rightRange {
                None => {
                    self.map.insert(id, 1);
                }
                Some(r) => {
                    self.map.insert(r.End() - 1, r.Len() + 1);
                }
            }
        } else {
            let leftRange = leftRange.unwrap();
            self.map.remove(&(leftRange.End() - 1));
            match rightRange {
                None => {
                    self.map.insert(leftRange.End(), leftRange.Len() + 1);
                }
                Some(r) => {
                    self.map.insert(r.End() - 1, r.Len() + leftRange.Len() + 1);
                }
            }
        }
    }
}
