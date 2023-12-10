// Copyright (c) 2021 Quark Container Authors
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

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Ord, Eq)]
#[repr(u64)]
pub enum ProxyCommand {
    None = 0 as u64,
    CudaSetDevice,
    CudaDeviceSynchronize,
    CudaMalloc,
    CudaMemcpy,
}

impl Default for ProxyCommand {
    fn default() -> Self {
        return Self::None;
    }
}

#[derive(Copy, Clone, Default, Debug)]
#[repr(C)]
pub struct ProxyParameters {
    pub para1: u64,
    pub para2: u64,
    pub para3: u64,
    pub para4: u64,
    pub para5: u64,
    pub para6: u64,
    pub para7: u64,
}

