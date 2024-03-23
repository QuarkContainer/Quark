// Copyright (c) 2021 Quark Container Authors / 2014 The Kubernetes Authors
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

use core::ops::Deref;
use std::sync::Arc;
use std::sync::Mutex;

use qshare::common::*;

#[derive(Debug, Default)]
pub struct PodSandboxInner {
    pub podUid: String,
    pub namespace: String,
    pub name: String,
    pub ip: IpAddress,
}

#[derive(Debug, Clone)]
pub struct PodSandbox(Arc<Mutex<PodSandboxInner>>);

impl Deref for PodSandbox {
    type Target = Arc<Mutex<PodSandboxInner>>;

    fn deref(&self) -> &Arc<Mutex<PodSandboxInner>> {
        &self.0
    }
}

impl PodSandbox {
    pub fn New(uid: &str, namespace: &str, name: &str, addr: IpAddress) -> Self {
        let inner = PodSandboxInner {
            podUid: uid.to_owned(),
            namespace: namespace.to_owned(),
            name: name.to_owned(),
            ip: addr,
        };

        return Self(Arc::new(Mutex::new(inner)));
    }
}
