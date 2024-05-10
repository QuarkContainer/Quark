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


use core::{slice, sync::atomic::{AtomicUsize, Ordering}};
use alloc::sync::Arc;

use crate::qlib as qlib;
use qlib::{linux_def::MemoryDef, QMutex, kernel::FP_STATE};
use super::super::super::asm;
//
// General Purpose Registers
// ref: ARM Cortex-A Series Programmer's Guide for ARMv8-A.
//      Procedure Call Standart for the Arm 64-bit Architecture.
//
//
#[derive(Debug, Copy, Clone, Default)]
pub struct Aarch64Regs {
    //
    // Parameter/Result Registers for subroutine call
    //
    pub X0  : u64,
    pub X1  : u64,
    pub X2  : u64,
    pub X3  : u64,
    pub X4  : u64,
    pub X5  : u64,
    pub X6  : u64,
    pub X7  : u64,
    //
    // Indirect result location Register
    //
    pub X8  : u64,
    //
    // Local variables, Caller-saved Registers
    //
    pub X9  : u64,
    pub X10 : u64,
    pub X11 : u64,
    pub X12 : u64,
    pub X13 : u64,
    pub X14 : u64,
    pub X15 : u64,
    //
    // IPC Temporary Registers
    //
    pub X16 : u64, // IP0
    pub X17 : u64, // IP1
    //
    // Platform Register (if needed) || Temporary Register
    //
    pub X18 : u64,
    //
    // Calle-saved Registers
    //    
    pub X19 : u64,
    pub X20 : u64,
    pub X21 : u64,
    pub X22 : u64,
    pub X23 : u64,
    pub X24 : u64,
    pub X25 : u64,
    pub X26 : u64,
    pub X27 : u64,
    pub X28 : u64,
    //
    // End Calle-saved Registers
    //
    pub X29 : u64, // FramePointer
    pub X30 : u64, // LinkRegister
    pub X31 : u64, // StackPointer
    //
    // Thread Pointer / ID Register
    //
    pub TPIDR_EL0: u64,
    //
    // Process state information Register
    //
    pub PSTATE: u64, // See also SPSR
    pub TTBR0_EL1: u64,
    //NOTE: Possible ASID (note used for now)
}

// Aarch64 FP state
#[derive(Debug)]
pub struct Aarch64FPState {
    pub data: [u8; MemoryDef::PAGE_SIZE as usize],
    pub size: AtomicUsize,
}

impl Default for Aarch64FPState {    fn default() -> Self {
        Aarch64FPState {
            data: [0u8; MemoryDef::PAGE_SIZE as usize],
            size: AtomicUsize::new(MemoryDef::PAGE_SIZE as usize),
        }
    }
}

pub type ArchFPState = Aarch64FPState;

impl Aarch64FPState{
    pub const fn Init() -> Self {
        Aarch64FPState{
            data: [0u8; MemoryDef::PAGE_SIZE as usize],
            size: AtomicUsize::new(MemoryDef::PAGE_SIZE as usize),
        }
    }

    fn New() -> Self {
        //
        // NOTE No Aarch64-cpuid present -> TODO
        //
        Self {
            ..Default::default()
        }
    }

    pub fn Load() -> Self {
        FP_STATE.Fork()
    }

    pub fn Size(&self) -> usize {
        self.size.load(Ordering::SeqCst)
    }

    pub fn Fork(&self) -> Self {
        let mut fork = Self::New();

        fork.data.copy_from_slice(&self.data[0..Self::Size(self)]);
        fork.size.store(self.size.load(Ordering::SeqCst), Ordering::SeqCst);

        fork
    }

    pub fn Slice(&self) -> &'static mut [u8]{
        let data_prt = self.data.as_ptr() as *mut u8;
        let _slice = unsafe {
            slice::from_raw_parts_mut(data_prt, self.Size())
        };

        _slice
    }

    pub fn FloatingPointData(&self) -> u64 {
        self.data.as_ptr() as u64
    }

    pub fn SaveFp(&self) {
        asm::SaveFloatingPoint(self.FloatingPointData());
    }

    pub fn RestoreFp(&self) {
        asm::LoadFloatingPoint(self.FloatingPointData());
    }
    //
    //NOTE: Every thing here is placeholder.
    //
    //
    // TODO
    // How to proceed...
    //
    pub fn SanitizeUser(&self) {}

    //
    // TODO
    // In which context is this used?
    //
    pub fn Reset(&self)  {}
}

pub struct State {
    pub Regs: &'static mut Aarch64Regs,
    pub archFPState: Arc<QMutex<Aarch64FPState>>,
}

impl State {
    //
    // NOTE It does not seem to be used at all...
    pub fn FullRestore(&self) -> bool { true }

    pub fn Fork(&self, regs: &'static mut Aarch64Regs) -> Self{
        Self {
            Regs: regs,
            archFPState: Arc::new(QMutex::new(self.archFPState.lock().Fork()))
        }
    }
}
