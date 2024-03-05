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

use core::sync::atomic::Ordering;
use libc::*;
use crate::arch::vCPU;

use super::*;
use super::qlib::common::*;
use super::qlib::pagetable::*;
use super::runc::runtime::vm::*;

#[repr(C)]
pub struct SignalMaskStruct {
    length: u32,
    mask1: u32,
    mask2: u32,
    _pad: u32,
}

pub struct HostPageAllocator {
    pub allocator: AlignedAllocator,
}

impl HostPageAllocator {
    pub fn New() -> Self {
        return Self {
            allocator: AlignedAllocator::New(0x1000, 0x10000),
        };
    }
}

impl Allocator for HostPageAllocator {
    fn AllocPage(&self, _incrRef: bool) -> Result<u64> {
        let ret = self.allocator.Allocate()?;
        ZeroPage(ret);
        return Ok(ret);
    }
}

impl RefMgr for HostPageAllocator {
    fn Ref(&self, _addr: u64) -> Result<u64> {
        //panic!("HostPageAllocator doesn't support Ref");
        return Ok(1);
    }

    fn Deref(&self, _addr: u64) -> Result<u64> {
        panic!("HostPageAllocator doesn't support Deref");
    }

    fn GetRef(&self, _addr: u64) -> Result<u64> {
        panic!("HostPageAllocator doesn't support GetRef");
    }
}

impl KVMVcpu {
    pub const KVM_INTERRUPT: u64 = 0x4004ae86;
    pub fn InterruptGuest(&self) {
        let bounce: u32 = 20; //VirtualizationException
        let ret = unsafe {
            ioctl(
                self.vcpu.as_raw_fd(),
                Self::KVM_INTERRUPT,
                &bounce as *const _ as u64,
            )
        };

        assert!(
            ret == 0,
            "InterruptGuest ret is {}/{}/{}",
            ret,
            errno::errno().0,
            self.vcpu.as_raw_fd()
        );
    }

    pub fn run(&self, tgid: i32) -> Result<()> {
        SetExitSignal();
        self.SignalMask();
        let tid = unsafe { gettid() };
        self.threadid.store(tid as u64, Ordering::SeqCst);
        self.tgid.store(tgid as u64, Ordering::SeqCst);

        if self.cordId > 0 {
            let coreid = core_affinity::CoreId {
                id: self.cordId as usize,
            };
            // print cpu id
            core_affinity::set_for_current(coreid);
        }

        if !super::runc::runtime::vm::IsRunning() {
            info!("The VM is not running.");
            return Ok(());
        }

        info!(
            "Start enter guest[{}]: entry is {:x}, stack is {:x}",
            self.id, self.entry, self.topStackAddr
        );
        self.arch_vcpu.run()?;

        Ok(())
    }

    pub fn VcpuWait(&self) -> i64 {
        let sharespace = &SHARE_SPACE;
        loop {
            if !super::runc::runtime::vm::IsRunning() {
                return -1;
            }

            {
                sharespace.IncrHostProcessor();
                Self::GuestMsgProcess(sharespace);

                defer!({
                    // last processor in host
                    if sharespace.DecrHostProcessor() == 0 {
                        Self::GuestMsgProcess(sharespace);
                    }
                });
            }

            let ret = sharespace.scheduler.WaitVcpu(sharespace, self.id, true);
            match ret {
                Ok(taskId) => return taskId as i64,
                Err(Error::Exit) => return -1,
                Err(e) => panic!("HYPERCALL_HLT wait fail with error {:?}", e),
            }
        }
    }



    pub fn dump(&self) -> Result<()> {
        if !Dump(self.id) {
            return Ok(());
        }
        defer!(ClearDump(self.id));
        let regs = self
            .vcpu
            .get_regs()
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        let sregs = self
            .vcpu
            .get_sregs()
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        error!("vcpu regs: {:#x?}", regs);
        let ss = sregs.ss.selector as u64;
        let isUser = (ss & 0x3) != 0;
        if isUser {
            error!("vcpu {} is in user mode, skip", self.id);
            return Ok(());
        }
        let kernelMemRegionSize = QUARK_CONFIG.lock().KernelMemSize;
        let mut frames = String::new();
        crate::qlib::backtracer::trace(regs.rip, regs.rsp, regs.rbp, &mut |frame| {
            frames.push_str(&format!("{:#x?}\n", frame));
            if frame.rbp < MemoryDef::PHY_LOWER_ADDR
                || frame.rbp >= MemoryDef::PHY_LOWER_ADDR + kernelMemRegionSize * MemoryDef::ONE_GB
            {
                false
            } else {
                true
            }
        });
        error!("vcpu {} stack: {}", self.id, frames);
        Ok(())
    }
}
