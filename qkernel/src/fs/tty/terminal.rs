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

use alloc::sync::Arc;
use spin::Mutex;

use super::super::host::tty::*;
use super::dir::*;
use super::line_discipline::*;

pub struct Terminal {
    pub n: u32,
    pub d: DirInodeOperations,
    pub ld: Arc<Mutex<LineDiscipline>>,
}

impl Terminal {
    pub fn New(d: &DirInodeOperations, n: u32) -> Self {
        return Self {
            d: d.clone(),
            n: n,
            ld: Arc::new(Mutex::new(LineDiscipline::New(DEFAULT_SLAVE_TERMIOS)))
        }
    }
}
