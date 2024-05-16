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

use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;
use super::asm::*;
#[cfg(target_arch = "aarch64")]
use core::arch::asm;
//use super::IOURING;
use super::super::singleton::*;
use super::super::vcpu_mgr::*;
use super::SHARESPACE;

pub static VCPU_COUNT: Singleton<AtomicUsize> = Singleton::<AtomicUsize>::New();
pub static CPU_LOCAL: Singleton<&'static [CPULocal]> = Singleton::<&'static [CPULocal]>::New();

pub fn SetVCPCount(cpuCnt: usize) {
    VCPU_COUNT.store(cpuCnt, Ordering::SeqCst)
}

#[derive(Debug)]
#[allow(non_camel_case_types)]
pub enum MSR {
    MSR_EFER = 0xc0000080,
    /* extended feature register */
    MSR_STAR = 0xc0000081,
    /* legacy mode SYSCALL target */
    MSR_LSTAR = 0xc0000082,
    /* long mode SYSCALL target */
    MSR_CSTAR = 0xc0000083,
    /* compat mode SYSCALL target */
    MSR_SYSCALL_MASK = 0xc0000084,
    /* EFLAGS mask for syscall */
    MSR_FS_BASE = 0xc0000100,
    /* 64bit FS base */
    MSR_GS_BASE = 0xc0000101,
    /* 64bit GS base */
    MSR_KERNEL_GS_BASE = 0xc0000102,
    /* SwapGS GS shadow */
    MSR_TSC_AUX = 0xc0000103,
    /* Auxiliary TSC */
}

#[derive(Debug)]
#[allow(non_camel_case_types)]
#[repr(u16)]
pub enum PrCtlEnum {
    ARCH_SET_GS = 0x1001,
    ARCH_SET_FS = 0x1002,
    ARCH_GET_FS = 0x1003,
    ARCH_GET_GS = 0x1004,
}

#[cfg(target_arch = "x86_64")]
pub fn RegisterSysCall(addr: u64) {
    //WriteMsr(MSR::MSR_STAR as u32, 0x00200008<<32);
    WriteMsr(MSR::MSR_STAR as u32, 0x00100008 << 32);
    WriteMsr(MSR::MSR_SYSCALL_MASK as u32, 0x3f7fd5);
    WriteMsr(MSR::MSR_LSTAR as u32, addr);
}

// this replaces RegisterSysCall
#[cfg(target_arch="aarch64")]
pub fn RegisterExceptionTable(addr: u64) {
    unsafe{
        asm!(
            "MSR VBAR_EL1, {}",
            in(reg) addr,
            );
    }
}


#[cfg(target_arch = "x86_64")]
#[inline]
pub fn SetTLS(addr: u64) {
    //println!("SetFs from {:x} to {:x}", GetFs(), addr);
    WriteMsr(MSR::MSR_FS_BASE as u32, addr);
    //println!("the input value is {:x}, the get fs result is {:x}", addr, ReadMsr(MSR::MSR_FS_BASE as u32));
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn SetTLS(addr: u64) {
    tpidr_el0_write(addr);
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn GetFs() -> u64 {
    return ReadMsr(MSR::MSR_FS_BASE as u32);
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn SetGs(addr: u64) {
    WriteMsr(MSR::MSR_KERNEL_GS_BASE as u32, addr);
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn GetGs() -> u64 {
    return ReadMsr(MSR::MSR_KERNEL_GS_BASE as u32);
}

impl CPULocal {
    pub fn Myself() -> &'static Self {
        return &CPU_LOCAL[Self::CpuId() as usize];
    }

    pub fn NextUringIdx(cnt: u64) -> usize {
        let cpuId = Self::CpuId() as usize;
        return CPU_LOCAL[Self::CpuId() as usize].IncrUringMsgCnt(cnt) as usize + cpuId;
    }

    pub fn SetKernelStack(task: u64) {
        Self::Myself().kernelStack.store(task, Ordering::Relaxed); //the data only read in current thread
    }

    pub fn KernelStack() -> u64 {
        return Self::Myself().kernelStack.load(Ordering::Relaxed); //the data only read in current thread
    }

    pub fn SwitchCount() -> u64 {
        Self::Myself().switchCount.load(Ordering::Relaxed)
    }

    pub fn IncreaseSwitchCount() -> u64 {
        return Self::Myself().switchCount.fetch_add(1, Ordering::Relaxed) + 1;
    }

    pub fn SetUserStack(task: u64) {
        Self::Myself().userStack.store(task, Ordering::Relaxed); //the data only read in current thread
    }

    pub fn UserStack() -> u64 {
        return Self::Myself().userStack.load(Ordering::SeqCst);
    }

    pub fn SetWaitTask(task: u64) {
        Self::Myself().waitTask.store(task, Ordering::SeqCst);
    }

    pub fn WaitTask() -> u64 {
        return Self::Myself().waitTask.load(Ordering::SeqCst);
    }

    pub fn SetCurrentTask(task: u64) {
        Self::Myself().currentTask.store(task, Ordering::SeqCst);
    }

    pub fn CurrentTask() -> u64 {
        return Self::Myself().currentTask.load(Ordering::SeqCst);
    }

    pub fn SetPendingFreeStack(stack: u64) {
        Self::Myself()
            .pendingFreeStack
            .store(stack, Ordering::SeqCst);
    }

    pub fn PendingFreeStack() -> u64 {
        return Self::Myself().pendingFreeStack.load(Ordering::SeqCst);
    }

    pub fn CPUState() -> VcpuState {
        return Self::Myself().State();
    }

    pub fn GetCPUState(cpuId: usize) -> VcpuState {
        return CPU_LOCAL[cpuId].State();
    }

    pub fn SwitchToRunning(&self) {
        let _searchingCnt = self.ToRunning(&SHARESPACE);
        /*if searchingCnt == 0 {
            SHARESPACE.scheduler.WakeOne();
        }*/
    }
}
