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
// General Purpose Registers
// ref: ARM Cortex-A Series Programmer's Guide for ARMv8-A.
//      Procedure Call Standart for the Arm 64-bit Architecture.
//
//
#[derive(Debug, Copy, Clone, Default)]
pub struct Arm64SysRegs {
    //
    // Non Preserved Registers
    //
    X0  : u64,
    X1  : u64,
    X2  : u64,
    X3  : u64,
    X4  : u64,
    X5  : u64,
    X6  : u64,
    X7  : u64,

    X8  : u64,
    //
    // Local variables, Caller-saved Registers
    //
    X9  : u64,
    X10 : u64,
    X11 : u64,
    X12 : u64,
    X13 : u64,
    X14 : u64,
    X15 : u64,
    //
    // End Caller-saved Registers
    //
    X16 : u64,
    X17 : u64,
    X18 : u64,
    //
    // Calle-saved Registers
    //
    X19 : u64,
    X20 : u64,
    X21 : u64,
    X22 : u64,
    X23 : u64,
    X24 : u64,
    X25 : u64,
    X26 : u64,
    X27 : u64,
    X28 : u64,
    //
    // End Calle-saved Registers
    X29 : u64, //FramePointer
    X30 : u64, //LinkRegister
    X31 : u64, //StackPointer
}

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
