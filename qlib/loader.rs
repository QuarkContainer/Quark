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

use alloc::string::String;
use alloc::vec::Vec;

use super::auth::cap_set::*;
use super::limits::*;

use crate::GLOBAL_ALLOCATOR;

#[derive(Serialize, Deserialize, Default, Debug, Eq, PartialEq, Clone)]
pub struct Process {
    //user
    pub UID: u32,
    pub GID: u32,
    pub AdditionalGids: Vec<u32>,

    pub Terminal: bool,
    pub Args: Vec<String>,
    pub Envs: Vec<String>,
    pub Cwd: String,

    //caps
    pub Caps: TaskCaps,

    pub NoNewPrivileges: bool,

    //host
    pub NumCpu: u32,
    pub HostName: String,

    //Container
    pub limitSet: LimitSetInternal,
    pub ID: String,

    pub Root: String,
    pub Stdiofds: [i32; 3],
    pub ExecId: Option<String>,
}
/* 
impl Process {
    pub fn default_in_shared() -> Self{
        let shared_allocator = GLOBAL_ALLOCATOR.GuestHostSharedAllocator();
        let size = core::mem::size_of::<Process>();
        let process_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size,0x80) as *mut Process };
        unsafe {
            *process_ptr = Process{
                AdditionalGids: Vec::new_in(shared_allocator),
                Args: Vec::new_in(shared_allocator),
                Envs: Vec::new_in(shared_allocator),
                ..Default::default()
            }
        }
        return unsafe{*process_ptr};
    }
}
*/