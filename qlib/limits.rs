// Copyright (c) 2021 QuarkSoft LLC
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
use alloc::sync::Arc;
use spin::Mutex;
use core::ops::Deref;
use core::u64;
use lazy_static::lazy_static;

use super::common::*;
use super::linux_def::*;
use super::linux::limits::*;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Ord, PartialOrd, Eq, PartialEq)]
#[repr(i32)]
pub enum LimitType {
    CPU = 0,
    FileSize,
    Data,
    Stack,
    Core,
    Rss,
    ProcessCount,
    NumberOfFiles,
    MemoryLocked,
    AS,
    Locks,
    SignalsPending,
    MessageQueueBytes,
    Nice,
    RealTimePriority,
    Rttime,
}

lazy_static! {
    pub static ref FROM_LINUX_RESOURCE : BTreeMap<i32, LimitType> = [
        (RLIMIT_CPU,        LimitType::CPU),
        (RLIMIT_FSIZE,      LimitType::FileSize),
        (RLIMIT_DATA,       LimitType::Data),
        (RLIMIT_STACK,      LimitType::Stack),
        (RLIMIT_CORE,       LimitType::Core),
        (RLIMIT_RSS,        LimitType::Rss),
        (RLIMIT_NPROC,      LimitType::ProcessCount),
        (RLIMIT_NOFILE,     LimitType::NumberOfFiles),
        (RLIMIT_MEMLOCK,    LimitType::MemoryLocked),
        (RLIMIT_AS,         LimitType::AS),
        (RLIMIT_LOCKS,      LimitType::Locks),
        (RLIMIT_SIGPENDING, LimitType::SignalsPending),
        (RLIMIT_MSGQUEUE,   LimitType::MessageQueueBytes),
        (RLIMIT_NICE,       LimitType::Nice),
        (RLIMIT_RTPRIO,     LimitType::RealTimePriority),
        (RLIMIT_RTTIME,     LimitType::Rttime),
    ].iter().cloned().collect();
}

pub const INFINITY: u64 = u64::MAX;

pub fn FromLinux(rl: u64) -> u64 {
    if rl == RLIM_INFINITY {
        return INFINITY
    }

    return rl
}

pub fn ToLinux(l: u64) -> u64 {
    if l == INFINITY {
        return RLIM_INFINITY
    }

    return l;
}

pub fn NewLinuxLimitSet() -> LimitSet {
    let ls = LimitSet::default();
    for (rlt, rl) in &*INIT_RLIMITS {
        let lt = FROM_LINUX_RESOURCE.get(rlt).expect("unknown rlimit type");
        ls.SetUnchecked(*lt, Limit {
            Cur: FromLinux(rl.Cur),
            Max: FromLinux(rl.Max),
        });
    }

    return ls
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limit {
    pub Cur: u64,
    pub Max: u64,
}

impl Default for Limit {
    fn default() -> Self {
        return Self {
            Cur: INFINITY,
            Max: INFINITY,
        }
    }
}

#[derive(Serialize, Deserialize, Default, Debug, Eq, PartialEq)]
pub struct LimitSetInternal {
    pub data: BTreeMap<LimitType, Limit>,
}

#[derive(Clone)]
pub struct LimitSet(pub Arc<Mutex<LimitSetInternal>>);

impl Deref for LimitSet {
    type Target = Arc<Mutex<LimitSetInternal>>;

    fn deref(&self) -> &Arc<Mutex<LimitSetInternal>> {
        &self.0
    }
}

impl Default for LimitSet {
    fn default() -> Self {
        return Self(Arc::new(Mutex::new(LimitSetInternal {
            data: BTreeMap::new()
        })))
    }
}

impl LimitSet {
    pub fn GetCopy(&self) -> Self {
        let internal = self.lock();
        let mut data = BTreeMap::new();
        for (k, v) in &internal.data {
            data.insert(*k, *v);
        }

        return Self(Arc::new(Mutex::new(LimitSetInternal {
            data: data
        })))
    }

    pub fn GetInternalCopy(&self) -> LimitSetInternal {
        let internal = self.lock();
        let mut data = BTreeMap::new();
        for (k, v) in &internal.data {
            data.insert(*k, *v);
        }

        LimitSetInternal {
            data: data
        }
    }

    pub fn Get(&self, t: LimitType) -> Limit {
        let internal = self.lock();
        match internal.data.get(&t) {
            None => {
                Limit::default()
            }
            Some(v) => v.clone(),
        }
    }

    pub fn GetCapped(&self, t: LimitType, max: u64) -> u64 {
        let s = self.Get(t);
        if s.Cur == INFINITY || s.Cur > max {
            return max
        }

        return s.Cur;
    }

    pub fn SetUnchecked(&self, t: LimitType, v: Limit) {
        let mut internal = self.lock();
        internal.data.insert(t, v);
    }

    pub fn Set(&self, t: LimitType, v: Limit, privileged: bool) -> Result<Limit> {
        let mut internal = self.lock();

        match internal.data.get_mut(&t) {
            Some(l) => {
                info!("current is {:?}, the neww is {:?}", l, &v);
                if l.Max < v.Max && !privileged {
                    return Err(Error::SysError(SysErr::EPERM));
                }

                info!("Set ....");
                if l.Cur > v.Max {
                    return Err(Error::SysError(SysErr::EINVAL));
                }

                let old = *l;
                *l = v;
                return Ok(old)
            }
            None => ()
        }

        internal.data.insert(t, v);
        return Ok(Limit::default())
    }
}