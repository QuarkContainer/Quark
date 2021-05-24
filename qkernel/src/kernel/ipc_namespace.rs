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

use super::semaphore;
//use super::shm;
use super::super::qlib::auth::userns::*;

#[derive(Clone, Default)]
pub struct IPCNamespace {
    pub userNS: UserNameSpace,
    pub semphores: semaphore::Registry,
    //pub shms: shm::Registry,
}

impl IPCNamespace {
    pub fn New(userNS: &UserNameSpace) -> Self {
        return Self {
            userNS: userNS.clone(),
            semphores: semaphore::Registry::New(userNS),
            //shms: shm::Registry::New(userNS)
        }
    }

    pub fn SemaphoreRegistry(&self) -> semaphore::Registry {
        return self.semphores.clone()
    }

    /*pub fn ShmRegistry(&self) -> shm::Registry {
        return self.shms.clone()
    }*/
}