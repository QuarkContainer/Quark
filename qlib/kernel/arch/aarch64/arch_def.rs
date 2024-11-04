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

use core::{slice, sync::atomic::{AtomicUsize, AtomicU64, Ordering}};
use alloc::sync::Arc;

use crate::qlib::{linux_def::MemoryDef, QMutex, kernel::FP_STATE};
use super::super::super::asm;
use super::super::super::SignalDef::*;

#[derive(Debug)]
pub struct Aarch64FPState {
    pub data: [u8; MemoryDef::PAGE_SIZE as usize],
    pub size: AtomicUsize,
}

impl Default for Aarch64FPState {
    fn default() -> Self {
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
    // TODO
    pub fn SanitizeUser(&self) {
        todo!("Aarch64FPState::SanitizeUser not implemented");
    }

    // TODO FPState imcomplete for aarch64
    pub fn Reset(&self)  {}
}

pub struct State {
    pub Regs: &'static mut PtRegs,
    pub archFPState: Arc<QMutex<Aarch64FPState>>,
}

impl State {
    pub fn FullRestore(&self) -> bool { true }

    pub fn Fork(&self, regs: &'static mut PtRegs) -> Self{
        Self {
            Regs: regs,
            archFPState: Arc::new(QMutex::new(self.archFPState.lock().Fork()))
        }
    }
}

pub type Context = Aarch64Context;

#[derive(Debug)]
#[repr(C)]
pub struct Aarch64Context {
    pub x19: u64,
    pub x20: u64,
    pub x21: u64,
    pub x22: u64,
    pub x23: u64,
    pub x24: u64,
    pub x25: u64,
    pub x26: u64,
    pub x27: u64,
    pub x28: u64,
    pub fp: u64,
    pub sp: u64,
    pub pc: u64,
    pub x0: u64,
    pub ready: AtomicU64,
    pub tls: u64,
}

impl Aarch64Context {
    pub fn New() -> Self {
        return Self {
         x19:    0,
         x20:    0,
         x21:    0,
         x22:    0,
         x23:    0,
         x24:    0,
         x25:    0,
         x26:    0,
         x27:    0,
         x28:    0,
         fp:     0,
         sp:     0,
         pc:     0,
         x0:     0,
         ready:  AtomicU64::new(1),
         tls:    0,
        };
    }

    pub fn set_tls(&mut self, tls: u64) {
        self.tls = tls;
    }

    pub fn get_tls(&self) -> u64 {
        self.tls
    }

    pub fn set_sp(&mut self, sp: u64) {
        self.sp = sp;
    }

    pub fn get_sp(&self) -> u64 {
        self.sp
    }

    pub unsafe fn place_on_stack(&mut self, addr: u64) {
        *(self.sp as *mut u64) = addr;
    }

    pub fn set_para(&mut self, para: u64) {
        self.x0 = para;
    }

    pub fn get_pc(&self) -> u64 {
        self.pc
    }

    pub fn set_pc(&mut self, pc: u64) {
        self.pc = pc;
    }

    pub fn set_ready(&self, val: u64) {
        self.ready.store(val, Ordering::Release);
    }

    pub fn get_ready(&self) -> u64 {
        return self.ready.load(Ordering::Acquire);
    }
}
