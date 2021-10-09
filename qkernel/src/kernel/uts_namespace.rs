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

use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use ::qlib::mutex::*;
use core::ops::Deref;

use super::super::qlib::auth::userns::*;

#[derive(Default)]
pub struct UTSNamespaceInternal {
    pub hostName: String,
    pub domainName: String,
    pub userns: UserNameSpace,
}

#[derive(Clone, Default)]
pub struct UTSNamespace(Arc<QMutex<UTSNamespaceInternal>>);

impl Deref for UTSNamespace {
    type Target = Arc<QMutex<UTSNamespaceInternal>>;

    fn deref(&self) -> &Arc<QMutex<UTSNamespaceInternal>> {
        &self.0
    }
}

impl UTSNamespace {
    pub fn New(hostName: String, domainName: String, userns: UserNameSpace) -> Self {
        let internal = UTSNamespaceInternal {
            hostName: hostName,
            domainName: domainName,
            userns: userns
        };

        return Self(Arc::new(QMutex::new(internal)))
    }

    pub fn HostName(&self) -> String {
        return self.lock().hostName.to_string();
    }

    pub fn SetHostName(&self, host: String) {
        self.lock().hostName = host;
    }

    pub fn DomainName(&self) -> String {
        return self.lock().domainName.to_string();
    }

    pub fn SetDomainName(&self, domain: String) {
        self.lock().domainName = domain;
    }

    pub fn UserNamespace(&self) -> UserNameSpace {
        return self.lock().userns.clone();
    }

    pub fn Fork(&self, userns: &UserNameSpace) -> Self {
        let me = self.lock();
        let internal = UTSNamespaceInternal {
            hostName: me.hostName.to_string(),
            domainName: me.domainName.to_string(),
            userns: userns.clone(),
        };

        return Self(Arc::new(QMutex::new(internal)))
    }
}