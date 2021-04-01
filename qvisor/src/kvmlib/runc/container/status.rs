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

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Copy, Clone)]
pub enum Status {
    // Created indicates "the runtime has finished the create operation and
    // the container process has neither exited nor executed the
    // user-specified program".
    Created = 0,

    // Creating indicates "the container is being created".
    Creating = 1,

    // Paused indicates that the process within the container has been
    // suspended.
    Paused = 2,

    // Running indicates "the container process has executed the
    // user-specified program but has not exited".
    Running = 3,

    // Stopped indicates "the container process has exited".
    Stopped = 4,
}

impl Default for Status {
    fn default() -> Status {
        Status::Created
    }
}

impl Status {
    pub fn String(&self) -> String {
        match self {
            Self::Created => "created".to_string(),
            Self::Creating => "creating".to_string(),
            Self::Paused => "pause".to_string(),
            Self::Running => "running".to_string(),
            Self::Stopped => "stopped".to_string(),
        }
    }
}