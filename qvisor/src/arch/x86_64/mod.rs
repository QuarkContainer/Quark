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

use crate::arch::vCPU;
use crate::qlib::common::Error;

pub mod context;

#[derive(Default, Debug)]
pub struct x86_64vCPU {
    gtdAddr: u64,
    idtAddr: u64,
    tssIntStackStart: u64,
    tssAddr: u64,
}

pub type ArchvCPU = x86_64vCPU;

impl vCPU for x86_64vCPU {
    fn new () -> Self {
        Self {
            ..Default::default()
        }
    }

    fn init(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn run(&self) -> Result<(), Error> {
        Ok(())
    }

}
