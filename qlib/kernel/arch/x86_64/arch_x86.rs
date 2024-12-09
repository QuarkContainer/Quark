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

use crate::qlib::mutex::*;
use alloc::sync::Arc;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;

use super::super::super::super::cpuid::*;
use super::super::super::asm::*;
use super::super::super::SignalDef::*;
use super::super::super::FP_STATE;

// System-related constants for x86.

// SyscallWidth is the width of syscall, sysenter, and int 80 insturctions.
pub const SYSCALL_WIDTH: usize = 2;

// EFLAGS register bits.

// EFLAGS_CF is the mask for the carry flag.
pub const EFLAGS_CF: u64 = 1 << 0;
// EFLAGS_PF is the mask for the parity flag.
pub const EFLAGS_PF: u64 = 1 << 2;
// EFLAGS_AF is the mask for the auxiliary carry flag.
pub const EFLAGS_AF: u64 = 1 << 4;
// EFLAGS_ZF is the mask for the zero flag.
pub const EFLAGS_ZF: u64 = 1 << 6;
// EFLAGS_SF is the mask for the sign flag.
pub const EFLAGS_SF: u64 = 1 << 7;
// EFLAGS_TF is the mask for the trap flag.
pub const EFLAGS_TF: u64 = 1 << 8;
// EFLAGS_IF is the mask for the interrupt flag.
pub const EFLAGS_IF: u64 = 1 << 9;
// EFLAGS_DF is the mask for the direction flag.
pub const EFLAGS_DF: u64 = 1 << 10;
// EFLAGS_OF is the mask for the overflow flag.
pub const EFLAGS_OF: u64 = 1 << 11;
// EFLAGS_IOPL is the mask for the I/O privilege level.
pub const EFLAGS_IOPL: u64 = 3 << 12;
// EFLAGS_NT is the mask for the nested task bit.
pub const EFLAGS_NT: u64 = 1 << 14;
// EFLAGS_RF is the mask for the resume flag.
pub const EFLAGS_RF: u64 = 1 << 16;
// EFLAGS_VM is the mask for the virtual mode bit.
pub const EFLAGS_VM: u64 = 1 << 17;
// EFLAGS_AC is the mask for the alignment check / access control bit.
pub const EFLAGS_AC: u64 = 1 << 18;
// EFLAGS_VIF is the mask for the virtual interrupt flag.
pub const EFLAGS_VIF: u64 = 1 << 19;
// EFLAGS_VIP is the mask for the virtual interrupt pending bit.
pub const EFLAGS_VIP: u64 = 1 << 20;
// EFLAGS_ID is the mask for the CPUID detection bit.
pub const EFLAGS_ID: u64 = 1 << 21;

// EFLAGS_PTRACE_MUTABLE is the mask for the set of EFLAGS that may be
// changed by ptrace(PTRACE_SETREGS). EFLAGS_PTRACE_MUTABLE is analogous to
// Linux's FLAG_MASK.
pub const EFLAGS_PTRACE_MUTABLE: u64 = EFLAGS_CF
    | EFLAGS_PF
    | EFLAGS_AF
    | EFLAGS_ZF
    | EFLAGS_SF
    | EFLAGS_TF
    | EFLAGS_DF
    | EFLAGS_OF
    | EFLAGS_RF
    | EFLAGS_AC
    | EFLAGS_NT;

// EFLAGS_RESTORABLE is the mask for the set of EFLAGS that may be changed by
// SignalReturn. EFLAGS_RESTORABLE is analogous to Linux's FIX_EFLAGS.
pub const EFLAGS_RESTORABLE: u64 = EFLAGS_AC
    | EFLAGS_OF
    | EFLAGS_DF
    | EFLAGS_TF
    | EFLAGS_SF
    | EFLAGS_ZF
    | EFLAGS_AF
    | EFLAGS_PF
    | EFLAGS_CF
    | EFLAGS_RF;

// TrapInstruction is the x86 trap instruction.
pub const TRAP_INSTRUCTION: [u8; 1] = [0xcc];

// CPUIDInstruction is the x86 CPUID instruction.
pub const CPUIDINSTRUCTION: [u8; 2] = [0xf, 0xa2];

// X86TrapFlag is an exported const for use by other packages.
pub const X86_TRAP_FLAG: u64 = 1 << 8;

// Segment selectors. See arch/x86/include/asm/segment.h.
pub const USER_CS: u64 = 0x33; // guest ring 3 code selector
pub const USER32_CS: u64 = 0x23; // guest ring 3 32 bit code selector
pub const USER_DS: u64 = 0x2b; // guest ring 3 data selector

pub const FS_TLS_SEL: u64 = 0x63; // Linux FS thread-local storage selector
pub const GS_TLS_SEL: u64 = 0x6b; // Linux GS thread-local storage selector

// MXCSR_DEFAULT is the reset value of MXCSR (Intel SDM Vol. 2, Ch. 3.2
// "LDMXCSR")
pub const MXCSR_DEFAULT: u32 = 0x1f80;

// MXCSR_OFFSET is the offset in bytes of the MXCSR field from the start of the
// FXSAVE/XSAVE area. (Intel SDM Vol. 1, Table 10-2 "Format of an FXSAVE Area")
pub const MXCSR_OFFSET: usize = 24;

// x86FPState is x86 floating point state.
#[repr(align(4096))]
#[repr(C)]
#[derive(Debug)]
pub struct X86fpstate {
    pub data: [u8; 4096],
    pub size: AtomicUsize,
}


impl Default for X86fpstate {
    fn default() -> Self {
        return Self::Load();
    }
}
pub type ArchFPState = X86fpstate;

impl X86fpstate {
    // minXstateBytes is the minimum size in bytes of an x86 XSAVE area, equal
    // to the size of the XSAVE legacy area (512 bytes) plus the size of the
    // XSAVE header (64 bytes). Equivalently, minXstateBytes is GDB's
    // X86_XSTATE_SSE_SIZE.
    pub const MIN_XSTATE_BYTES: usize = 512 + 64;

    // userXstateXCR0Offset is the offset in bytes of the USER_XSTATE_XCR0_WORD
    // field in Linux's struct user_xstateregs, which is the type manipulated
    // by ptrace(PTRACE_GET/SETREGSET, NT_X86_XSTATE). Equivalently,
    // userXstateXCR0Offset is GDB's I386_LINUX_XSAVE_XCR0_OFFSET.
    pub const USER_XSTATE_XCR0_OFFSET: usize = 464;

    // xstateBVOffset is the offset in bytes of the XSTATE_BV field in an x86
    // XSAVE area.
    pub const XSTATE_BVOFFSET: usize = 512;

    // xsaveHeaderZeroedOffset and xsaveHeaderZeroedBytes indicate parts of the
    // XSAVE header that we coerce to zero: "Bytes 15:8 of the XSAVE header is
    // a state-component bitmap called XCOMP_BV. ... Bytes 63:16 of the XSAVE
    // header are reserved." - Intel SDM Vol. 1, Section 13.4.2 "XSAVE Header".
    // Linux ignores XCOMP_BV, but it's able to recover from XRSTOR #GP
    // exceptions resulting from invalid values; we aren't. Linux also never
    // uses the compacted format when doing XSAVE and doesn't even define the
    // compaction extensions to XSAVE as a CPU feature, so for simplicity we
    // assume no one is using them.
    pub const XSAVE_HEADER_ZEROED_OFFSET: usize = 512 + 8;
    pub const XSAVE_HEADER_ZEROED_BYTES: usize = 64 - 8;

    // mxcsrOffset is the offset in bytes of the MXCSR field from the start of
    // the FXSAVE area. (Intel SDM Vol. 1, Table 10-2 "Format of an FXSAVE
    // Area")
    pub const MXCSR_OFFSET: usize = 24;

    // mxcsrMaskOffset is the offset in bytes of the MXCSR_MASK field from the
    // start of the FXSAVE area.
    pub const MXCSR_MASK_OFFSET: usize = 28;

    fn New() -> Self {
        let (size, _align) = HostFeatureSet().ExtendedStateSize();

        if size > 4096 {
            panic!("X86fpstate capacity size({}) > 4096", size);
        }

        return Self {
            data: [0; 4096],
            size: AtomicUsize::new(size as usize),
        };
    }

    pub fn SanitizeUser(&self) {
        // Force reserved bits in MXCSR to 0. This is consistent with Linux.
        self.SanitizeMXCSR();

        if self.Size() >= Self::MIN_XSTATE_BYTES {
            // Users can't enable *more* XCR0 bits than what we, and the CPU, support.
            let xstateBVAddr = &self.data[Self::XSTATE_BVOFFSET] as *const _ as u64;
            let mut xstateBV: u64 = unsafe { *(xstateBVAddr as *const u64) };

            xstateBV &= HostFeatureSet().ValidXCR0Mask();
            unsafe {
                *(xstateBVAddr as *mut u64) = xstateBV;
            };

            let addr = &self.data[Self::XSAVE_HEADER_ZEROED_OFFSET] as *const _ as u64;
            let ptr = addr as *mut u8;
            let slice =
                unsafe { core::slice::from_raw_parts_mut(ptr, Self::XSAVE_HEADER_ZEROED_BYTES) };

            for i in 0..Self::XSAVE_HEADER_ZEROED_BYTES {
                slice[i] = 0;
            }
        }
    }

    pub fn mxcsrMask(&self) -> u32 {
        let mxcsrAddr = &self.data[Self::MXCSR_MASK_OFFSET] as *const _ as u64;
        let mxcsrMask: u32 = unsafe { *(mxcsrAddr as *const u32) };
        return mxcsrMask;
    }

    pub fn SanitizeMXCSR(&self) {
        let mxcsrAddr = &self.data[Self::MXCSR_OFFSET] as *const _ as u64;
        let mxcsr: u32 = unsafe { *(mxcsrAddr as *const u32) };

        let mut mxcsrMask = FP_STATE.mxcsrMask();
        if mxcsrMask == 0 {
            // "If the value of the MXCSR_MASK field is 00000000H, then the
            // MXCSR_MASK value is the default value of 0000FFBFH." - Intel SDM
            // Vol. 1, Section 11.6.6 "Guidelines for Writing to the MXCSR
            // Register"
            mxcsrMask = 0xffbf
        }

        let mxcsr = mxcsr & mxcsrMask;
        unsafe {
            *(mxcsrAddr as *mut u32) = mxcsr;
        };
    }

    pub fn Load() -> Self {
        return FP_STATE.Fork();
    }

    pub fn Size(&self) -> usize {
        return self.size.load(Ordering::SeqCst);
    }

    pub fn Slice(&self) -> &'static mut [u8] {
        let ptr = &self.data[0] as *const _ as u64 as *mut u8;
        let buf = unsafe { core::slice::from_raw_parts_mut(ptr, self.Size()) };
        return buf;
    }

    pub const fn Init() -> Self {
        return Self {
            data: [0; 4096],
            size: AtomicUsize::new(4096),
        };
    }

    pub fn Reset(&self) {
        let (size, _align) = HostFeatureSet().ExtendedStateSize();
        self.size.store(size as usize, Ordering::SeqCst);
        self.SaveFp();
    }

    pub fn Fork(&self) -> Self {
        let mut f = Self::New();

        for i in 0..self.size.load(Ordering::Relaxed) {
            f.data[i] = self.data[i];
        }
        f.size
            .store(self.size.load(Ordering::SeqCst), Ordering::SeqCst);

        return f;
    }

    pub fn FloatingPointData(&self) -> u64 {
        return &self.data[0] as *const _ as u64;
    }

    pub fn SaveFp(&self) {
        SaveFloatingPoint(self.FloatingPointData());
    }

    pub fn RestoreFp(&self) {
        LoadFloatingPoint(self.FloatingPointData());
    }
}

pub struct State {
    // The system registers.
    pub Regs: &'static mut PtRegs,

    // Our floating point state.
    pub x86FPState: Arc<QMutex<X86fpstate>>,
}

impl State {
    pub fn FullRestore(&self) -> bool {
        // A fast system call return is possible only if
        //
        // * RCX matches the instruction pointer.
        // * R11 matches our flags value.
        // * Usermode does not expect to set either the resume flag or the
        //   virtual mode flags (unlikely.)
        // * CS and SS are set to the standard selectors.
        //
        // That is, SYSRET results in the correct final state.
        let fastRestore = self.Regs.rcx == self.Regs.rip
            && self.Regs.eflags == self.Regs.r11
            && (self.Regs.eflags & EFLAGS_RF) == 0
            && (self.Regs.eflags & EFLAGS_VM) == 0
            && self.Regs.cs == USER_CS
            && self.Regs.ss == USER_DS;

        return !fastRestore;
    }

    pub fn Fork(&self, regs: &'static mut PtRegs) -> Self {
        return Self {
            Regs: regs,
            x86FPState: Arc::new(QMutex::new(self.x86FPState.lock().Fork())),
        };
    }
}

pub type Context = X86Context;

#[derive(Debug)]
#[repr(C)]
pub struct X86Context {
    pub rsp: u64,
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub rbx: u64,
    pub rbp: u64,
    pub rdi: u64,
    pub ready: AtomicU64,
    pub fs: u64,
}

impl X86Context {
    pub fn New() -> Self {
        return Self {
            rsp: 0,
            r15: 0,
            r14: 0,
            r13: 0,
            r12: 0,
            rbx: 0,
            rbp: 0,
            rdi: 0,
            ready: AtomicU64::new(1),
            fs: 0,
        };
    }

    pub fn set_tls(&mut self, tls: u64) {
        self.fs = tls;
    }

    pub fn get_tls(&self) -> u64 {
        self.fs
    }

    pub fn set_sp(&mut self, sp: u64) {
        self.rsp = sp;
    }

    pub fn get_sp(&self) -> u64 {
        self.rsp
    }

    pub fn set_para(&mut self, para: u64) {
        self.rdi = para;
    }

    pub fn set_ready(&self, val: u64) {
        self.ready.store(val, Ordering::Release);
    }

    pub fn get_ready(&self) -> u64 {
        return self.ready.load(Ordering::Acquire);
    }
}
