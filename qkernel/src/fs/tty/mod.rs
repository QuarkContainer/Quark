// Copyright (c) 2021 QuarkSoft LLC
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

pub mod line_discipline;
pub mod queue;
mod utf8;
pub mod dir;
pub mod terminal;
pub mod master;
pub mod slave;
pub mod fs;

use alloc::sync::Arc;
use spin::Mutex;

use super::filesystems::*;

pub fn Init() {
    RegisterFilesystem(&Arc::new(Mutex::new(self::fs::PtsTmpfs {})));
}
