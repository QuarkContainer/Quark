// Copyright (c) 2021 Quark Container Authors
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

use super::mm::*;

// Dumpability describes if and how core dumps should be created.
pub type Dumpability = i32;

// NotDumpable indicates that core dumps should never be created.
pub const NOT_DUMPABLE : Dumpability = 0;

// UserDumpable indicates that core dumps should be created, owned by
// the current user.
pub const USER_DUMPABLE : Dumpability = 1;

// RootDumpable indicates that core dumps should be created, owned by
// root.
pub const ROOT_DUMPABLE : Dumpability = 2;

impl MemoryManager1 {
    pub fn Dumpability(&self) -> Dumpability {
        return self.read().dumpability;
    }

    pub fn SetDumpability(&self, d: Dumpability) {
        self.write().dumpability = d;
    }
}

impl MemoryManager {
    pub fn Dumpability(&self) -> Dumpability {
        return self.metadata.lock().dumpability;
    }

    pub fn SetDumpability(&self, d: Dumpability) {
        self.metadata.lock().dumpability = d;
    }
}