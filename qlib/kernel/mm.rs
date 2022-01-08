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
use crate::qlib::mutex::*;
use core::ops::Deref;

#[derive(Clone, Default, Debug, Copy)]
pub struct ActivityInternal {
    // curRSS is pmas.Span(), cached to accelerate updates to maxRSS. It is
    // reported as the MemoryManager's RSS.
    //
    // maxRSS should be modified only via insertRSS and removeRSS, not
    // directly.
    //
    // maxRSS is protected by activeMu.
    pub curRSS: u64,

    // maxRSS is the maximum resident set size in bytes of a MemoryManager.
    // It is tracked as the application adds and removes mappings to pmas.
    //
    // maxRSS should be modified only via insertRSS, not directly.
    //
    // maxRSS is protected by activeMu.
    pub maxRSS: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Activity(Arc<QMutex<ActivityInternal>>);

impl Deref for Activity {
    type Target = Arc<QMutex<ActivityInternal>>;

    fn deref(&self) -> &Arc<QMutex<ActivityInternal>> {
        &self.0
    }
}

impl Activity {
    pub fn Clone(&self) -> Self {
        let a = self.lock();
        let internal = ActivityInternal {
            curRSS: a.curRSS,
            maxRSS: a.maxRSS,
        };

        return Self(Arc::new(QMutex::new(internal)))
    }
}

#[derive(Clone, Default, Debug, Copy)]
pub struct BrkInfo {
    pub brkStart: u64,
    pub brkEnd: u64,
    pub brkMemEnd: u64,
}
