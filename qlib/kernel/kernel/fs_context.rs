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

use crate::qlib::mutex::*;
use alloc::sync::Arc;
use core::ops::Deref;

use super::super::fs::dirent::*;

#[derive(Default)]
pub struct FSContextInternal {
    pub root: Dirent,
    pub cwd: Dirent,
    pub umask: u32,
}

#[derive(Clone, Default)]
pub struct FSContext(Arc<QMutex<FSContextInternal>>);

impl Deref for FSContext {
    type Target = Arc<QMutex<FSContextInternal>>;

    fn deref(&self) -> &Arc<QMutex<FSContextInternal>> {
        &self.0
    }
}

impl FSContext {
    pub fn New(root: &Dirent, cwd: &Dirent, umask: u32) -> Self {
        let internal = FSContextInternal {
            root: root.clone(),
            cwd: cwd.clone(),
            umask: umask,
        };

        return Self(Arc::new(QMutex::new(internal)))
    }

    pub fn Fork(&self) -> Self {
        let me = self.lock();
        return Self::New(&me.root, &me.cwd, me.umask)
    }

    pub fn WorkDirectory(&self) -> Dirent {
        return self.lock().cwd.clone();
    }

    pub fn SetWorkDirectory(&self, d: &Dirent) {
        self.lock().cwd = d.clone();
    }

    pub fn RootDirectory(&self) -> Dirent {
        return self.lock().root.clone();
    }

    pub fn SetRootDirectory(&self, d: &Dirent) {
        self.lock().root = d.clone();
    }

    pub fn Umask(&self) -> u32 {
        return self.lock().umask
    }

    pub fn SwapUmask(&self, mask: u32) -> u32 {
        let mut me = self.lock();
        let old = me.umask;
        me.umask = mask;
        return old;
    }
}