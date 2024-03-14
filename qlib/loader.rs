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

impl Process {
    //Cannot use default trait clone here, clone uses clone method for the allocator in vec,
    //which is the global allocator of the qvisor, not the kernel
    pub fn clone_from_shared(&mut self, process_ptr:*mut Process){
        let shared_process = unsafe{&*process_ptr };
        info!("Process: {:#?}\n",shared_process);
        let cloned_value = shared_process.AdditionalGids.clone();
        let addr = core::ptr::addr_of!(cloned_value) as u64;
        info!("Cloned value addr: 0x{:x}",addr);
        self.UID = shared_process.UID;
        self.GID = shared_process.GID;
        self.AdditionalGids.extend(shared_process.AdditionalGids.iter());
        self.Terminal = shared_process.Terminal;
        //FIXME: if there is a more elegant way to initiate a String with global allocator from another String
        for str in shared_process.Args.iter(){
            unsafe{self.Args.push(String::from_utf8_unchecked(str.as_bytes().to_vec()));}
        }
        for str in shared_process.Envs.iter(){
            unsafe{self.Envs.push(String::from_utf8_unchecked(str.as_bytes().to_vec()));}
        }
        self.Cwd = unsafe{String::from_utf8_unchecked((&shared_process.Cwd).as_bytes().to_vec())};
        self.Caps = shared_process.Caps;
        self.NoNewPrivileges = shared_process.NoNewPrivileges;
        self.NumCpu = shared_process.NumCpu;
        self.HostName = unsafe{String::from_utf8_unchecked((&shared_process.HostName).as_bytes().to_vec())};
        for (k, v) in &shared_process.limitSet.data {
            self.limitSet.data.insert(*k, *v);
        }
        self.Stdiofds = shared_process.Stdiofds;
        self.ExecId = match &shared_process.ExecId{
            Some(str) => unsafe{Some(String::from_utf8_unchecked(str.as_bytes().to_vec()))},
            _ => None,
        };
    }
}