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

use super::super::runc::container::container::*;
use super::super::qlib::control_msg::*;

// ControlSocketAddr generates an abstract unix socket name for the given ID.
pub fn ControlSocketAddr(id: &str) -> String {
    return format!("\x00qvisor-sandbox.{}", id)
}

pub const UCALL_BUF_LEN : usize = 4096;
type Cid = String;

#[derive(Serialize, Deserialize, Debug)]
pub enum UCallReq {
    RootContainerStart(RootContainerStart),
    ExecProcess(ExecArgs),
    Pause,
    Unpause,
    Ps(Cid),
    WaitContainer(Cid),
    WaitPid(WaitPid),
    Signal(SignalArgs),
    ContainerDestroy(Cid),
    CreateSubContainer(CreateArgs),
    StartSubContainer(StartArgs),
    WaitAll,
}

impl FileDescriptors for UCallReq {
    // todo: need to implement this so that fds can be passed
    fn GetFds(&self) -> Option<&[i32]> {
        match self {
            UCallReq::ExecProcess(args) => return args.GetFds(),
            UCallReq::CreateSubContainer(args) => {
                if args.fds[0] == -1 {
                    return None
                } else {
                    return Some(&args.fds)
                }
            },
            _ => return None,
        }
    }

    fn SetFds(&mut self, fds: &[i32]) {
        match self {
            UCallReq::ExecProcess(ref mut args) => return args.SetFds(fds),
            _ => ()
        }
    }
}

pub trait FileDescriptors {
    fn GetFds(&self) -> Option<&[i32]>;
    fn SetFds(&mut self, fds: &[i32]);
}