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

use alloc::vec::Vec;

use super::super::super::qlib::linux_def::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::super::task::*;
use super::super::super::socket::control::*;
use super::super::super::Kernel::HostSpace;

pub struct HostSCMRights (Vec<i32>);

impl Drop for HostSCMRights {
    fn drop(&mut self) {
        for fd in &self.0 {
            HostSpace::AsyncClose(*fd)
        }
    }
}

impl SCMRights for HostSCMRights {
    fn Files(&mut self, task: &Task, max: usize) -> (SCMRights, bool) {
        let mut n = max;
        let mut trunc = false;

        let l = self.0.len();
        if n > l {
            n = l;
        } else if n < l {
            n = l;
            trunc = true;
        }

        let rf = SCMRights(FdsToFiles(task, &self.0[..n]));
        self.0 = self.0.split_off(n);

        return (rf, trunc)
    }
}

impl HostSCMRights {
    pub fn New(fds: Vec<i32>) -> Self {
        return Self(fds);
    }

    // Clone implements transport.RightsControlMessage.Clone.
    pub fn Clone(&self) -> Option<SCMRights> {
        // Host rights never need to be cloned.
        return None;
    }
}

// If an error is encountered, only files created before the error will be
// returned. This is what Linux does.
fn FdsToFiles(task: &Task, fds: &[i32]) -> Vec<File> {
    let mut files = Vec::new();

    for fd in fds {
        let mut fileFlags : i32 = 0;
        let ret = HostSpace::Fcntl(*fd, Cmd::F_GETFL, &mut fileFlags as * mut _ as u64);
        if ret < 0 {
            info!("Error retrieving host FD flags: {}", -ret);
            break;
        }

        let file = match File::NewFileFromFd(task, *fd, &task.FileOwner(), false) {
            Err(e) => {
                info!("Error creating file from host FD: {:?}", e);
                break;
            }
            Ok(f) => f,
        };

        // Set known flags.
        file.SetFlags(task, SettableFileFlags {
            NonBlocking: fileFlags & Flags::O_NONBLOCK != Flags::O_NONBLOCK,
            ..Default::default()
        });

        files.push(file);
    }

    return files;
}