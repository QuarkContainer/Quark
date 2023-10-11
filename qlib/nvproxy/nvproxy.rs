// Copyright (c) 2021 Quark Container Authors
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

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;

use crate::qlib::kernel::PAGE_MGR;
use crate::qlib::linux_def::MemoryDef;
use crate::qlib::mutex::QMutex;
use crate::qlib::nvproxy::nvgpu;
use crate::qlib::range::Range;

pub struct NVProxyInner {
    pub objs: BTreeMap<nvgpu::Handle, NVObject>
}

#[derive(Clone)]
pub struct NVProxy(Arc<QMutex<NVProxyInner>>);

impl Deref for NVProxy {
    type Target = Arc<QMutex<NVProxyInner>>;

    fn deref(&self) -> &Arc<QMutex<NVProxyInner>> {
        &self.0
    }
}

pub struct OSDescMem {
    pub pinnedRange: Vec<Range>
}

impl Drop for OSDescMem {
    fn drop(&mut self) {
        for r in &self.pinnedRange {
            let mut paddr = r.start;
            while paddr < r.End() {
                PAGE_MGR.DerefPage(paddr);
                paddr += MemoryDef::PAGE_SIZE;
            }
        }
    }
}

pub enum NVObject {
    OSDescMem(OSDescMem)
}

impl From<OSDescMem> for NVObject {
    fn from(o: OSDescMem) -> Self {
        NVObject::OSDescMem(o)
    }
}
