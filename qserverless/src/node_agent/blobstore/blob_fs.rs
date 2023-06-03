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

use std::fs::File;

use qobjs::common::*;

pub struct BlobFs {
    pub root: String,
}

impl BlobFs {
    pub fn New(root: &str) -> Self {
        return Self {
            root: root.to_string()
        }
    }

    pub fn Create(&self, addr: &str) -> Result<File> {
        let file = File::create(&self.FileName(addr))?;
        return Ok(file)
    }

    pub fn Open(&self, addr: &str) -> Result<File> { 
        let file = File::open(&self.FileName(addr))?;
        return Ok(file)
    }

    pub fn FileName(&self, addr: &str) -> String {
        return format!("{}{}", &self.root, addr);
    }

    pub fn Remove(&self, _addr: &str) -> Result<()> {
        unimplemented!();
    }
}