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

//use core::mem;
//use alloc::sync::Arc;
//use spin::Mutex;

//use super::super::super::SignalDef::*;
//use super::super::super::stack::*;
//use super::super::super::qlib::common::*;
//use super::super::super::qlib::linux_def::*;
use super::context::*;
//use super::arch_x86::*;

const FP_XSTATE_MAGIC2_SIZE: usize = 4;

impl Context64 {
    fn fpuFrameSize(&self) -> (u64, bool) {
        let mut size = self.state.x86FPState.lock().size;
        let mut useXsave = false;
        if size > 512 {
            // Make room for the magic cookie at the end of the xsave frame.
            size += FP_XSTATE_MAGIC2_SIZE;
            useXsave = true;
        }

        return (size as u64, useXsave)
    }

    // SignalSetup implements Context.SignalSetup. (Compare to Linux's
    // arch/x86/kernel/signal.c:__setup_rt_frame().)
    /*pub fn SignalSetup(&mut self, st: &mut Stack, act: &SigAct, info: &mut SignalInfo, alt: &SignalStack, sigset: SignalSet) -> Result<()> {
        let mut sp = st.sp;

        // "The 128-byte area beyond the location pointed to by %rsp is considered
        // to be reserved and shall not be modified by signal or interrupt
        // handlers. ... leaf functions may use this area for their entire stack
        // frame, rather than adjusting the stack pointer in the prologue and
        // epilogue." - AMD64 ABI
        //
        // (But this doesn't apply if we're starting at the top of the signal
        // stack, in which case there is no following stack frame.)
        if !(alt.IsEnable() && sp == alt.Top()) {
            sp -= 128;
        }

        let (fpSize, _) = self.fpuFrameSize();
        sp = (sp - fpSize) & !(64 - 1);

        let regs = &self.state.Regs;
        // Construct the UContext64 now since we need its size.
        let mut uc = UContext {
            // No _UC_FP_XSTATE: see Fpstate above.
            // No _UC_STRICT_RESTORE_SS: we don't allow SS changes.
            Flags: UC_SIGCONTEXT_SS,
            Link: 0,
            Stack: *alt,
            MContext: SigContext::New(regs, sigset.0, 0, 0),
            Sigset: sigset.0,
        };

        // based on the fault that caused the signal. For now, leave Err and
        // Trapno unset and assume CR2 == info.Addr() for SIGSEGVs and
        // SIGBUSes.
        if info.Signo == Signal::SIGSEGV || info.Signo == Signal::SIGBUS {
            uc.MContext.cr2 = info.SigFault().addr;
        }

        // "... the value (%rsp+8) is always a multiple of 16 (...) when
        // control is transferred to the function entry point." - AMD64 ABI
        let ucSize = mem::size_of::<UContext>();

        let frameSize = 8 + ucSize + 128;
        let frameBottom = (sp - frameSize as u64) & !15 - 8;
        sp = frameBottom + frameSize as u64;
        st.sp = sp;

        // Prior to proceeding, figure out if the frame will exhaust the range
        // for the signal stack. This is not allowed, and should immediately
        // force signal delivery (reverting to the default handler).
        if act.flags.IsOnStack() && alt.IsEnable() && !alt.Contains(frameBottom) {
            return Err(Error::SysError(SysErr::EFAULT))
        }

        // Adjust the code.
        info.FixSignalCodeForUser();

        let infoAddr = st.PushType::<SignalInfo>(info);
        let ucAddr = st.PushType::<UContext>(&uc);

        if act.flags.HasRestorer() {
            st.PushU64(act.restorer);
        } else {
            return Err(Error::SysError(SysErr::EFAULT))
        }

        self.state.Regs.rip = act.handler;
        self.state.Regs.rsp = st.sp;
        self.state.Regs.rdi = info.Signo as u64;
        self.state.Regs.rsi = infoAddr;
        self.state.Regs.rdx = ucAddr;
        self.state.Regs.rax = 0;

        // Save the thread's floating point state.
        self.sigFPState.push(self.state.x86FPState.clone());

        self.state.x86FPState = Arc::new(Mutex::new(X86fpstate::NewX86FPState()));

        return Ok(())
    }*/
}