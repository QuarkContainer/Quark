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

pub mod dev;
pub mod fs;
pub mod full;
pub mod null;
pub mod proxyfile;
pub mod random;
pub mod tty;
pub mod zero;

use crate::qlib::mutex::*;
use alloc::sync::Arc;

use super::filesystems::*;

pub fn Init() {
    RegisterFilesystem(&Arc::new(QMutex::new(self::fs::DevTmpfs {})));
}
