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

use super::super::qlib::common::*;
use super::super::*;

pub struct UringMgr {
    pub uringSize: usize,
}

impl Drop for UringMgr {
    fn drop(&mut self) {
        self.Close();
    }
}

impl UringMgr {
    pub fn New(size: usize) -> Self {
        Self { uringSize: size }
    }

    /// No-op hook retained for RDMA socket setup (`rdma_socket.rs`).
    pub fn Addfd(&mut self, _host_fd: i32) -> Result<()> {
        Ok(())
    }

    pub fn Close(&mut self) {}
}
