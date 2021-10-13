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

// kernel to host
#[derive(Debug, Copy, Clone)]
#[repr(align(128))]
pub enum HostOutputMsg {
    QCall(u64),
    MUnmap(MUnmap),
}


#[derive(Clone, Default, Debug, Copy)]
pub struct MUnmap {
    pub addr: u64,
    pub len: u64,
}
