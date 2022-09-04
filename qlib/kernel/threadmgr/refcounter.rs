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

use core::i64;
use core::sync::atomic::AtomicI64;
use core::sync::atomic::Ordering;

#[derive(Debug)]
pub struct AtomicRefCount {
    refCount: AtomicI64,
}

impl Default for AtomicRefCount {
    fn default() -> Self {
        return Self {
            refCount: AtomicI64::new(1),
        };
    }
}

impl AtomicRefCount {
    pub fn ReadRefs(&self) -> i64 {
        return self.refCount.load(Ordering::SeqCst);
    }

    pub fn IncRef(&mut self) {
        self.refCount.fetch_add(1, Ordering::SeqCst);
    }

    pub fn DecRefWithDesctructor(&mut self, mut f: impl FnMut()) {
        let v = self.refCount.fetch_sub(1, Ordering::SeqCst);
        if v == 1 {
            f();
        }
    }
}
