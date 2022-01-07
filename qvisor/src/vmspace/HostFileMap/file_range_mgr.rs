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

use super::super::super::qlib::range::*;

#[derive(Default, Debug)]
pub struct RangeMgr {
    pub map : BTreeMap<Range, bool>,
}

impl RangeMgr {
    pub fn Conflict(&self, range: &Range, write: bool) -> bool {
        let start = Range::New(range.start, 0);
        let end = Range::New(range.End(), 0);
        for (&r, &value) in self.map.range((Included(&start), Excluded(&end))) {
            // if there is existing same range, just take it as conflict
            // in theory, read of same range doesn't conflict.
            // the implementation is just for performance
            if r == *range {
                return true
            }

            if r.Overlaps(range) {
                if write {
                    // for write ops, any overlap is conflict
                    return true
                } else {
                    // for read ops, existing write conflict
                    if value {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    pub fn AddRange(&mut self, range: &Range, write: bool) {
        self.map.insert(*range, write);
    }

    pub fn RemoveRange(&mut self, range: &Range) -> bool {
        match self.map.remove(range) {
            None => false,
            _ => true,
        }
    }
}