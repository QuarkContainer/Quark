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


//
//NOTE: Every thing here is placeholder.
//

// Aarch64 FP state
// NOTE: Placeholder
#[derive(Debug)]
pub struct Aarch64FPState {
    pub data: [u64; 4069],
}

impl Default for Aarch64FPState {
    fn default() -> Self {
        Aarch64FPState {
            data: [0u64; 4069]
        }
    }
}

pub type ArchFPState = Aarch64FPState;

impl Aarch64FPState{
    pub const fn Init() -> Self {
        Aarch64FPState{
            data: [0u64; 4069]
        }
    }

	pub fn Fork(&self) -> Self {
        Aarch64FPState{
            data: [0u64; 4069]
        }
    }

    pub fn Slice(&self) -> &'static mut [u8]{
		 static mut dummy:  &mut [u8] = &mut [0u8];
		unsafe{
		 dummy as &'static mut [u8]
		} 
    }

    pub fn SanitizeUser(&self) {}
    pub fn Reset(&self)  {}
    pub fn SaveFp(&self) {}
    pub fn RestoreFp(&self) {}
}
