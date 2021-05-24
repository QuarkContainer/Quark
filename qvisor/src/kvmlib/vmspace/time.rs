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

use libc::*;

use super::super::qlib::linux::time;
use super::super::qlib::common::*;

pub struct HostTime {}

impl HostTime {
    pub fn Realtime() -> Result<i64> {
        let ts = time::Timespec::default();

        let res = unsafe {
            clock_gettime(CLOCK_REALTIME, &ts as *const _ as u64 as *mut timespec) as i64
        };

        if res == -1 {
            return Err(Error::SysError(errno::errno().0))
        }

        return Ok(ts.ToNs()?)
    }

    pub fn Monotime() -> Result<i64> {
        let ts = time::Timespec::default();

        let res = unsafe {
            clock_gettime(CLOCK_MONOTONIC, &ts as *const _ as u64 as *mut timespec) as i64
        };

        if res == -1 {
            return Err(Error::SysError(errno::errno().0))
        }

        return Ok(ts.ToNs()?)
    }
}