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

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use core::ops::Deref;

use crate::qlib::{proxy::nvgpu, mutex::QMutex};

pub struct NVProxyInner {
    pub objs: BTreeMap<nvgpu::Handle, NVObject>
}

#[derive(Clone)]
pub struct NVProxy(Arc<QMutex<NVProxyInner>>);

impl Deref for NVProxy {
    type Target = Arc<QMutex<NVProxyInner>>;

    fn deref(&self) -> &Arc<QMutex<NVProxyInner>> {
        &self.0
    }
}

pub struct NVObject {

}