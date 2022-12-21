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

use alloc::collections::BTreeMap;
use core::ops::Deref;
use alloc::vec::Vec;

use crate::qlib::kernel::memmgr::mm::*;
use crate::qlib::mutex::*;
use crate::qlib::linux_def::IoVec;

#[derive(Default)]
pub struct ReapSwapFile {
    pub fd: i32,            // the file fd 
    pub iovs: Vec<IoVec>,
}

#[derive(Default)]
pub struct HiberMgrIntern {
	pub pageMap: BTreeMap<u64, u64>, // pageAddr --> file offset 
	pub memmgrs: BTreeMap<u64, MemoryManagerWeak>,
	pub reap: bool,
	pub reapSwapFile: ReapSwapFile,
}

#[derive(Default)]
pub struct HiberMgr(QMutex<HiberMgrIntern>);
 
impl Deref for HiberMgr {
    type Target = QMutex<HiberMgrIntern>;

    fn deref(&self) -> &QMutex<HiberMgrIntern> {
        &self.0
    }
}

impl HiberMgr {
	pub fn AddMemMgr(&self, mm: &MemoryManager) {
		let uid = mm.uid;
		self.lock().memmgrs.insert(uid, mm.Downgrade());
	}

	pub fn RemoveMemMgr(&self, mm: &MemoryManager) -> bool {
		let uid = mm.uid;
		match self.lock().memmgrs.remove(&uid) {
			None => return false,
			Some(_) => return true,
		}
	}

    pub fn ContainersPage(&self, phyAddr: u64) -> bool {
        let intern = self.lock();
        return intern.pageMap.contains_key(&phyAddr);
    }
}
