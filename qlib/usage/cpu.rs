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

#[derive(Debug, Clone, Default, Copy)]
pub struct CPUStats {
    // UserTime is the amount of time spent executing application code.
    pub UserTime: i64,

    pub SysTime: i64,

    // VoluntarySwitches is the number of times control has been voluntarily
    // ceded due to blocking, etc.
    pub VoluntarySwitches: u64,
}

impl CPUStats {
    pub fn Accumulate(&mut self, s2: &CPUStats) {
        self.UserTime += s2.UserTime;
        self.SysTime += s2.SysTime;
        self.VoluntarySwitches += s2.VoluntarySwitches;
    }
}
