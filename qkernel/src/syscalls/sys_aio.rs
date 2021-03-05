// Copyright (c) 2021 QuarkSoft LLC
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


use super::super::task::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::syscalls::syscalls::*;

pub fn SysIoSetup(_task: &mut Task, _args: &SyscallArguments) -> Result<i64> {
    return Err(Error::SysError(SysErr::ENOSYS))
}