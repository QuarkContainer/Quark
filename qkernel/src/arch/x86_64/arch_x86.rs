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

use alloc::sync::Arc;
use spin::Mutex;

use super::super::super::qlib::cpuid::*;
use super::super::super::SignalDef::*;

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
pub const EFLAGS_PTRACE_MUTABLE: u64 = EFLAGS_CF | EFLAGS_PF | EFLAGS_AF | EFLAGS_ZF | EFLAGS_SF | EFLAGS_TF | EFLAGS_DF | EFLAGS_OF | EFLAGS_RF | EFLAGS_AC | EFLAGS_NT;

// EFLAGS_RESTORABLE is the mask for the set of EFLAGS that may be changed by
// SignalReturn. EFLAGS_RESTORABLE is analogous to Linux's FIX_EFLAGS.
pub const EFLAGS_RESTORABLE: u64 = EFLAGS_AC | EFLAGS_OF | EFLAGS_DF | EFLAGS_TF | EFLAGS_SF | EFLAGS_ZF | EFLAGS_AF | EFLAGS_PF | EFLAGS_CF | EFLAGS_RF;

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

pub fn initX86FPState(data: u64, useXsave: bool) {
    unsafe {
        llvm_asm!("
            // Save MXCSR (callee-save)
            STMXCSR	-0x08(%rsp)

            // Save x87 CW (callee-save)
	        FSTCW -0x10(%rsp)

	        MOVBQZX $1, %rax
	        TESTQ   %rax, %rax
	        JZ      1f

	        // Set MXCSR to the default value.
	        MOVL    $$0x1f80, 24(%rdi)

	        // Initialize registers with XRSTOR.
	        MOVL	$$0xffffffff, %rax
	        MOVL	$$0xffffffff, %rdi
	        XRSTOR64 0(%rdi)

	        // Now that all the state has been reset, write it back out to the
	        // XSAVE area.
	        XSAVE64 0(%rdi)

	        JMP     2f
	        1:
	        // Clear out existing X values.
	        PXOR    %x0
	        MOVQ    %x0, %x1
	        MOVQ    %x0, %x2
	        MOVQ    %x0, %x3
	        MOVQ    %x0, %x4
	        MOVQ    %x0, %x5
	        MOVQ    %x0, %x6
	        MOVQ    %x0, %x7
	        MOVQ    %x0, %x8
	        MOVQ    %x0, %x9
	        MOVQ    %x0, %x10
	        MOVQ    %x0, %x11
	        MOVQ    %x0, %x12
	        MOVQ    %x0, %x13
	        MOVQ    %x0, %x14
	        MOVQ    %x0, %x15

	        // Zero out %rax and store into MMX registers. MMX registers are
            // an alias of 8x64 bits of the 8x80 bits used for the original
            // x87 registers. Storing zero into them will reset the FPU registers
            // to bits [63:0] = 0, [79:64] = 1. But the contents aren't too
            // important, just the fact that we have reset them to a known value.
	        XORQ    %rax
	        MOVQ    %rax, %m0
	        MOVQ    %rax, %m1
	        MOVQ    %rax, %m2
	        MOVQ    %rax, %m3
	        MOVQ    %rax, %m4
	        MOVQ    %rax, %m5
	        MOVQ    %rax, %m6
	        MOVQ    %rax, %m7

            //  - Reset FPU control word to 0x037f
            //  - Clear FPU status word
            //  - Reset FPU tag word to 0xffff
            //  - Clear FPU data pointer
            //  - Clear FPU instruction pointer
            FNINIT

            // Reset MXCSR.
            MOVL    $$0x1f80, -24(%rsp)
            LDMXCSR -24(%rsp)

            // Save the floating point state with fxsave.
            FXSAVE64 0(%rdi)

	        2:
	        // Restore MXCSR.
	        LDMXCSR -8(%rsp)

	        // Restore x87 CW.
	        FLDCW   -16(%rsp)

            RET
            "
        :
        : "rdi"(data), "r"(useXsave)
        :
        :
        );
    }
}

// x86FPState is x86 floating point state.
#[repr(align(4096))]
pub struct X86fpstate {
    pub data: [u8; 4096],
    pub size: usize,
}

impl X86fpstate {
    pub fn New() -> Self {
        let (size, _align) = HostFeatureSet().ExtendedStateSize();

        if size > 4096 {
            panic!("X86fpstate capacity size({}) > 4096", size);
        }

        return Self {
            data: [0; 4096],
            size: size as usize,
        }
    }

    pub fn NewX86FPState() -> Self {
        let f = Self::New();

        initX86FPState(f.FloatingPointData(), HostFeatureSet().UseXsave());
        return f;
    }

    pub fn Fork(&self) -> Self {
        let mut f = Self::New();

        for i in 0..f.data.len() {
            f.data[i] = self.data[i];
        }

        return f;
    }

    pub fn FloatingPointData(&self) -> u64 {
        return &self.data[0] as *const _ as u64;
    }
}

pub struct State {
    // The system registers.
    pub Regs: &'static mut PtRegs,

    // Our floating point state.
    pub x86FPState: Arc<Mutex<X86fpstate>>,
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
        let fastRestore = self.Regs.rcx == self.Regs.rip &&
            self.Regs.eflags == self.Regs.r11 &&
            (self.Regs.eflags & EFLAGS_RF) == 0 &&
            (self.Regs.eflags & EFLAGS_VM) == 0 &&
            self.Regs.cs == USER_CS &&
            self.Regs.ss == USER_DS;

        return !fastRestore;
    }

    pub fn Fork(&self, regs: &'static mut PtRegs) -> Self {
        return Self {
            Regs: regs,
            x86FPState: Arc::new(Mutex::new(self.x86FPState.lock().Fork())),
        }
    }
}