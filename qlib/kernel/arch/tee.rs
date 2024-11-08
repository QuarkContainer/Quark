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

use core::sync::atomic::{AtomicU8, Ordering};
use lazy_static::lazy_static;
use crate::qlib::linux_def::MemoryDef;
use crate::qlib::config::CCMode;

lazy_static! {
    //TODO: It should be only set once
    static ref TEE_TYPE: AtomicU8 = AtomicU8::new(CCMode::None as u8);
}

/// Sets TEE_TYPE to a valid TEE CCMode, if not previously set.
pub fn set_tee_type(mode: CCMode) {
    static mut SET: bool = false;
    match mode {
        CCMode::None => return,
        _ => {
            // We care only for the first write by vCPU0
            unsafe {
                if SET == false {
                    TEE_TYPE.store(mode as u8, Ordering::Relaxed);
                    SET = true;
                }
            }
        }
    }
}

/// It return the CC-mode the kernel is running.
pub fn get_tee_type() -> CCMode {
    let mode = TEE_TYPE.load(Ordering::Relaxed);
    CCMode::from(mode as u64)
}

/// It returns true only if the CC-mode is backed by real HW
pub fn is_hw_tee() -> bool {
    let mode = TEE_TYPE.load(Ordering::Relaxed) as u64;
    CCMode::tee_backedup(mode)
}

pub fn is_cc_active() -> bool {
    let mode = CCMode::from(TEE_TYPE.load(Ordering::Acquire) as u64);
    match mode {
        CCMode::None => { return false; }
        _ => { return true; }
    }
}

/// Depending on TEE architecture, the guest physical address should be
/// marked as shared("untrusted")/private("trusted"). The actual set/unset
/// bit(s) on the GPA is implementation defined by the particular TEE.
pub fn gpa_adjust_shared_bit(_address: &mut u64, _protect: bool) {
    if is_hw_tee() {
        //
        // Impliment according to architecture
        //
    }
}

/// Before the guest can reason on a GPA, the information on the IPA that
/// regards the TEE should be removed.
pub fn guest_physical_address(ipa_address: u64) -> u64 {
    #![allow(unused_mut)]
    let mut address_guest = ipa_address;
    if is_hw_tee() {
        //
        // Impliment according to architecture
        //
    }
    address_guest
}

/// For Guest Physical Address
pub fn is_protected_address(gpa: u64) -> bool {
    let mut res = true;
    #[cfg(target_arch = "aarch64")]
    if gpa == MemoryDef::HYPERCALL_MMIO_BASE {
        res = false;
    }
    if (gpa >= MemoryDef::FILE_MAP_OFFSET &&
        gpa < MemoryDef::FILE_MAP_OFFSET + MemoryDef::FILE_MAP_SIZE)
        || (gpa >= MemoryDef::GUEST_HOST_SHARED_HEAP_OFFSET &&
        gpa < MemoryDef::GUEST_HOST_SHARED_HEAP_END) {
            res = false;
    }
    res
}

/// Usage: On aarch64-Realm - invoke Hypercall
/// NOTE: Unstable signature - may change in the future.
pub fn call_host(_hcall_type: u64, _arg1: u64, _arg2: u64, _arg3: u64, _arg4: u64) {
        //
        // Impliment according to architecture
        //
}

/// Entry call to the booting flow of other vCPUS.
/// Internals are architecture depended.
pub fn boot_others(_boot_help_data: u64, _vcpu_count: u64, _pc: u64) {
        //
        // Impliment according to architecture
        //
}
