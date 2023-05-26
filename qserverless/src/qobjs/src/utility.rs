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

use std::time::SystemTime;

use std::time::Duration;

use crate::func;

pub struct SystemTimeProto {
    pub seconds: u64,
    pub nanos: u32,
}

impl SystemTimeProto {
    pub fn FromTimestamp(ts: &func::Timestamp) -> Self {
        return Self {
            seconds: ts.seconds,
            nanos: ts.nanos,
        }
    }

    pub fn ToTimeStamp(&self) -> func::Timestamp {
        return func::Timestamp { 
            seconds: self.seconds, 
            nanos: self.nanos 
        };
    }

    pub fn FromSystemTime(time: SystemTime) -> Self {
        let dur = time.duration_since(SystemTime::UNIX_EPOCH).unwrap();
        return Self {
            seconds: dur.as_secs(),
            nanos: dur.subsec_nanos(),
        }
    }

    pub fn ToSystemTime(&self) -> SystemTime {
        let dur = Duration::from_secs(self.seconds).checked_add(Duration::from_nanos(self.nanos as u64)).unwrap();
        return SystemTime::UNIX_EPOCH.checked_add(dur).unwrap();
    }
}