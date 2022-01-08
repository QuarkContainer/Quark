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

use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::task::*;

pub type SyscallRestartErrno = i32;

// SyscallRestartErrnoFromReturn returns the SyscallRestartErrno represented by
// rv, the value in a syscall return register.
pub fn SyscallRestartErrnoFromReturn(rv: u64) -> (SyscallRestartErrno, bool) {
    let err = rv as i64 as i32;

    match -err {
        SysErr::ERESTARTSYS => return (SysErr::ERESTARTSYS, true),
        SysErr::ERESTARTNOINTR => return (SysErr::ERESTARTNOINTR, true),
        SysErr::ERESTARTNOHAND => return (SysErr::ERESTARTNOHAND, true),
        SysErr::ERESTART_RESTARTBLOCK => return (SysErr::ERESTART_RESTARTBLOCK, true),
        _ => return (0, false)
    }
}

// SyscallRestartBlock represents the restart block for a syscall restartable
// with a custom function. It encapsulates the state required to restart a
// syscall across a S/R.

pub trait SyscallRestartBlock : Sync + Send {
    fn Restart(&self, task: &mut Task) -> Result<i64>;
}