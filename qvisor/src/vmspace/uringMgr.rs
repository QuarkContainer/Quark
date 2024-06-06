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
use super::super::print::*;
use super::super::*;

//#[derive(Debug)]
pub struct UringMgr {
    pub fds: Vec<i32>,
    pub uringSize: usize,
}

impl Drop for UringMgr {
    fn drop(&mut self) {
        self.Close();
    }
}

pub const FDS_SIZE: usize = 1024 * 16;

impl UringMgr {
    pub fn New(size: usize) -> Self {
        let fdsSize = if QUARK_CONFIG.lock().UringFixedFile {
            FDS_SIZE
        } else {
            0
        };

        let mut fds = Vec::with_capacity(fdsSize);
        for _i in 0..fdsSize {
            fds.push(-1);
        }

        let ret = Self {
            fds: fds,
            uringSize: size,
        };

        return ret;
    }

    pub fn Close(&mut self) {
        let logfd = LOG.Logfd();
        for fd in &self.fds {
            if *fd >= 0 && *fd != logfd {
                unsafe {
                    libc::close(*fd);
                }
            }
        }
    }
}
