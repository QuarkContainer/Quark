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

use alloc::vec::Vec;

use super::super::linux_def::*;

pub const ALL_CAP: CapSet = CapSet((1 << Capability::CAP_LAST_CAP) - 1);

#[derive(Serialize, Deserialize, Default, Debug, Copy, Clone, Eq, PartialEq)]
pub struct CapSet(pub u64);

// A CapSet is a set of capabilities, implemented as a bitset
impl CapSet {
    pub fn New(cap: u64) -> Self {
        return Self(1 << cap);
    }

    pub fn NewWithCaps(caps: &Vec<u64>) -> Self {
        let mut mask: u64 = 0;

        for cap in caps {
            mask |= (1 as u64) << cap;
        }

        return Self(mask);
    }
}

pub fn MaskOf64(i: usize) -> u64 {
    return 1 << i;
}

pub fn CapSetOf(cp: u64) -> CapSet {
    return CapSet(MaskOf64(cp as usize));
}

pub fn CapabilitySetOfMany(cps: &[u64]) -> CapSet {
    let mut cs = 0;
    for cp in cps {
        cs |= MaskOf64(*cp as usize)
    }

    return CapSet(cs);
}

// TaskCapabilities represents all the capability sets for a task. Each of these
// sets is explained in greater detail in capabilities(7).
#[derive(Serialize, Deserialize, Default, Debug, Copy, Clone, Eq, PartialEq)]
pub struct TaskCaps {
    pub PermittedCaps: CapSet,
    pub InheritableCaps: CapSet,
    pub EffectiveCaps: CapSet,
    pub BoundingCaps: CapSet,
    pub AmbientCaps: CapSet,
}
