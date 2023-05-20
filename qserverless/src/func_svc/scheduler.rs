// Copyright (c) 2021 Quark Container Authors
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

use std::ops::Add;
use std::ops::Sub;

#[derive(Debug, Clone, Default, Copy)]
pub struct Resource {
    pub mem: u64,
    pub cpu: u64,
}

impl Add for Resource {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            mem: self.mem + other.mem,
            cpu: self.cpu + other.cpu,
        }
    }
}

impl Sub for Resource {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            mem: self.mem - other.mem,
            cpu: self.cpu - other.cpu,
        }
    }
}

impl Resource {
    pub fn Fullfil(&self, req: &Self) -> bool {
        return req.mem <= self.mem && req.cpu <= self.cpu;
    }
}