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

use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;

use super::taskMgr::*;

// SeqCountEpoch tracks writer critical sections in a SeqCount.
pub struct SeqCountEpoch {
    pub val: u64,
}

#[derive(Default, Debug)]
pub struct SeqCount {
    // epoch is incremented by BeginWrite and EndWrite, such that epoch is odd
    // if a writer critical section is active, and a read from data protected
    // by this SeqCount is atomic iff epoch is the same even value before and
    // after the read.
    pub epoch: AtomicU64,
}

impl SeqCount {
    pub fn BeginRead(&self) -> SeqCountEpoch {
        let mut epoch = self.epoch.load(Ordering::SeqCst);

        while epoch & 1 != 0 {
            Yield();
            epoch = self.epoch.load(Ordering::SeqCst);
        }

        return SeqCountEpoch { val: epoch };
    }

    // ReadOk returns true if the reader critical section initiated by a previous
    // call to BeginRead() that returned epoch did not race with any writer critical
    // sections.
    //
    // ReadOk may be called any number of times during a reader critical section.
    // Reader critical sections do not need to be explicitly terminated; the last
    // call to ReadOk is implicitly the end of the reader critical section.
    pub fn ReadOk(&self, epoch: SeqCountEpoch) -> bool {
        return self.epoch.load(Ordering::SeqCst) == epoch.val;
    }

    pub fn BeginWrite(&self) {
        let epoch = self.epoch.fetch_add(1, Ordering::SeqCst);
        if epoch & 1 != 1 {
            panic!("SeqCount.BeginWrite during writer critical section")
        }
    }

    pub fn EndWrite(&self) {
        let epoch = self.epoch.fetch_add(1, Ordering::SeqCst);
        if epoch & 1 != 0 {
            panic!("SeqCount.EndWrite outside writer critical section")
        }
    }
}
