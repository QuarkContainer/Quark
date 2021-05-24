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

use alloc::vec::Vec;
use alloc::sync::Arc;
use spin::Mutex;

use super::super::super::qlib::limits::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::SignalDef::*;
use super::super::super::qlib::addr::*;
use super::super::super::memmgr::arch::*;
use super::super::super::kernel_util::*;
use super::arch_x86::*;

// These constants come directly from Linux.


// MAX_ADDR64 is the maximum userspace address. It is TASK_SIZE in Linux
// for a 64-bit process.
pub const MAX_ADDR64: u64 = (1 << 47) - MemoryDef::PAGE_SIZE;

// MAX_STACK_RAND64 is the maximum randomization to apply to the stack.
// It is defined by arch/x86/mm/mmap.c:stack_maxrandom_size in Linux.
pub const MAX_STACK_RAND64: u64 = 16 << 30; // 16 GB

// MAX_MMAP_RAND64 is the maximum randomization to apply to the mmap
// layout. It is defined by arch/x86/mm/mmap.c:arch_mmap_rnd in Linux.
pub const MAX_MMAP_RAND64: u64 = (1 << 28) * MemoryDef::PAGE_SIZE;

// MIN_GAP64 is the minimum gap to leave at the top of the address space
// for the stack. It is defined by arch/x86/mm/mmap.c:MIN_GAP in Linux.
pub const MIN_GAP64: u64 = (128 << 20) + MAX_STACK_RAND64;

// PREFERRED_PIELOAD_ADDR is the standard Linux position-independent
// executable base load address. It is ELF_ET_DYN_BASE in Linux.
//
// The Platform {Min,Max}UserAddress() may preclude loading at this
// address. See other preferredFoo comments below.
pub const PREFERRED_PIELOAD_ADDR: u64 = MAX_ADDR64 / 3 * 2;

// Select a preferred minimum TopDownBase address.
//
// Some applications (TSAN and other *SANs) are very particular about
// the way the Linux mmap allocator layouts out the address space.
//
// TSAN in particular expects top down allocations to be made in the
// range [0x7e8000000000, 0x800000000000).
//
// The minimum TopDownBase on Linux would be:
// 0x800000000000 - MIN_GAP64 - MAX_MMAP_RAND64 = 0x7efbf8000000.
//
// (MIN_GAP64 because TSAN uses a small RLIMIT_STACK.)
//
// 0x7e8000000000 is selected arbitrarily by TSAN to leave room for
// allocations below TopDownBase.
//
// N.B. ASAN and MSAN are more forgiving; ASAN allows allocations all
// the way down to 0x10007fff8000, and MSAN down to 0x700000000000.
//
// Of course, there is no hard minimum to allocation; an allocator can
// search all the way from TopDownBase to Min. However, TSAN declared
// their range "good enough".
//
// We would like to pick a TopDownBase such that it is unlikely that an
// allocator will select an address below TSAN's minimum. We achieve
// this by trying to leave a sizable gap below TopDownBase.
//
// This is all "preferred" because the layout min/max address may not
// allow us to select such a TopDownBase, in which case we have to fall
// back to a layout that TSAN may not be happy with.
pub const PREFERRED_TOP_DOWN_ALLOC_MIN: u64 = 0x7e8000000000;
pub const PREFERRED_ALLOCATION_GAP: u64 = 128 << 30; // 128 GB
pub const PREFERRED_TOP_DOWN_BASE_MIN: u64 = PREFERRED_TOP_DOWN_ALLOC_MIN + PREFERRED_ALLOCATION_GAP;

// MIN_MMAP_RAND64 is the smallest we are willing to make the
// randomization to stay above PREFERRED_TOP_DOWN_BASE_MIN.
pub const MIN_MMAP_RAND64: u64 = (1 << 26) * MemoryDef::PAGE_SIZE;

pub struct Context64 {
    pub state: State,
    pub sigFPState: Vec<Arc<Mutex<X86fpstate>>>,
}

impl Context64 {
    pub fn New() -> Self {
        return Self {
            state: State {
                Regs: unsafe {
                    &mut *(0 as *mut PtRegs)
                },
                x86FPState: Arc::new(Mutex::new(X86fpstate::New())),
            },
            sigFPState: Vec::new(),
        }
    }

    pub fn CopySigFPState(&self) -> Vec<Arc<Mutex<X86fpstate>>> {
        let mut sigfs = Vec::with_capacity(self.sigFPState.len());

        for s in &self.sigFPState {
            sigfs.push(Arc::new(Mutex::new(s.lock().Fork())));
        }

        return sigfs
    }

    // Fork returns an exact copy of this context.
    pub fn Fork(&self, regs: &'static mut PtRegs) -> Self {
        return Self {
            state: self.state.Fork(regs),
            sigFPState: self.CopySigFPState(),
        }
    }

    // Return returns the current syscall return value.
    pub fn Return(&self) -> u64 {
        return self.state.Regs.rax;
    }

    // Return returns the current syscall return value.
    pub fn SetReturn(&mut self, val: u64) {
        self.state.Regs.rax = val;
    }

    // IP returns the current instruction pointer.
    pub fn IP(&self) -> u64 {
        return self.state.Regs.rip;
    }

    // SetIP sets the current instruction pointer.
    pub fn SetIP(&mut self, val: u64) {
        self.state.Regs.rip = val;
    }

    // Stack returns the current stack pointer.
    pub fn Stack(&self) -> u64 {
        return self.state.Regs.rsp;
    }

    // SetStack sets the current stack pointer.
    pub fn SetStack(&mut self, val: u64) {
        self.state.Regs.rsp = val;
    }

    // NewMmapLayout implements Context.NewMmapLayout consistently with Linux.
    pub fn NewMmapLayout(min: u64, max: u64, r: &LimitSet) -> Result<MmapLayout> {
        let min = Addr(min).RoundUp()?.0;

        let mut max = if max > MAX_ADDR64 {
            MAX_ADDR64
        } else {
            max
        };

        max = Addr(max).RoundDown()?.0;

        if min > max {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let stackSize = r.Get(LimitType::Stack);

        // MAX_GAP in Linux.
        let maxGap = (max / 6) * 5;
        let mut gap = stackSize.Cur;
        if gap < MIN_GAP64 {
            gap = MIN_GAP64;
        }

        if gap > maxGap {
            gap = maxGap;
        }

        let mut defaultDir = MMAP_TOP_DOWN;
        if stackSize.Cur == INFINITY {
            defaultDir = MMAP_BOTTOM_UP;
        }

        let topDownMin = max - gap - MAX_MMAP_RAND64;
        let mut maxRand = MAX_MMAP_RAND64;
        if topDownMin < PREFERRED_TOP_DOWN_BASE_MIN {
            // Try to keep TopDownBase above preferredTopDownBaseMin by
            // shrinking maxRand.
            let maxAdjust = maxRand - MIN_MMAP_RAND64;
            let needAdjust = PREFERRED_TOP_DOWN_BASE_MIN - topDownMin;
            if needAdjust <= maxAdjust {
                maxRand -= needAdjust;
            }
        }

        let rnd = MMapRand(maxRand)?;
        let l = MmapLayout {
            MinAddr: min,
            MaxAddr: max,
            // TASK_UNMAPPED_BASE in Linux.
            BottomUpBase: Addr(max / 3 + rnd).RoundDown()?.0,
            TopDownBase: Addr(max - gap - rnd).RoundDown()?.0,
            DefaultDirection: defaultDir,
            // We may have reduced the maximum randomization to keep
            // TopDownBase above preferredTopDownBaseMin while maintaining
            // our stack gap. Stack allocations must use that max
            // randomization to avoiding eating into the gap.
            MaxStackRand: maxRand,
            sharedLoadsOffset: 0,
        };

        // Final sanity check on the layout.
        if !l.Valid() {
            panic!("Invalid MmapLayout: {:?}", l);
        }

        return Ok(l)
    }

    // PIELoadAddress implements Context.PIELoadAddress.
    pub fn PIELoadAddress(l: &MmapLayout) -> Result<u64> {
        let mut base = PREFERRED_PIELOAD_ADDR;

        let max = match Addr(base).AddLen(MAX_MMAP_RAND64) {
            Err(_) => panic!("preferredPIELoadAddr {} too large", base),
            Ok(addr) => addr.0
        };

        if max > l.MaxAddr {
            // preferredPIELoadAddr won't fit; fall back to the standard
            // Linux behavior of 2/3 of TopDownBase. TSAN won't like this.
            //
            // Don't bother trying to shrink the randomization for now.
            base = l.TopDownBase / 3 * 2;
        }

        let addr = base + MMapRand(MAX_MMAP_RAND64)?;


        return Ok(Addr(addr).RoundDown().unwrap().0);
    }
}

// mmapRand returns a random adjustment for randomizing an mmap layout.
pub fn MMapRand(max: u64) -> Result<u64> {
    let addr = RandU64()? % max;
    return Ok(Addr(addr).RoundDown().unwrap().0)
}

