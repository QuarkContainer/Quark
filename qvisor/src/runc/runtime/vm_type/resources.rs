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

use std::{string::String, fmt};

use hashbrown::HashMap;

#[derive(PartialEq, Eq, Hash)]
pub enum MemAreaType {
    PrivateHeapArea,
    SharedHeapArea,
    KernelArea,
    FileMapArea,
    HypercallMmioArea,
}

impl fmt::Display for MemAreaType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MemAreaType::PrivateHeapArea => write!(f, "Guest-private heap area"),
            MemAreaType::SharedHeapArea => write!(f,  "Host-shared heap area"),
            MemAreaType::KernelArea => write!(f, "Guest-kernel code area"),
            MemAreaType::FileMapArea => write!(f, "Host-shared file-map area"),
            MemAreaType::HypercallMmioArea => write!(f, "Guest hypercall-mmio area"),
        }
    }
}

#[derive(Copy, Clone)]
pub(in super) struct MemArea {
    pub base_host: u64,
    pub base_guest: u64,
    pub size: u64,
    pub guest_private: bool,
    pub host_backedup: bool,
}

impl fmt::Debug for MemArea {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "host base:{:#x}, guest base:{:#x}, size:{:#x}, guest private:{}, on host backedup:{}",
                self.base_host,
                self.base_guest,
                self.size,
                self.guest_private,
                self.host_backedup))
    }
}

pub(in super) struct MemLayoutConfig {
    pub mem_area_map: HashMap<MemAreaType, MemArea>,
    pub kernel_stack_size: usize,
    pub guest_mem_size: u64,
}

impl fmt::Debug for MemLayoutConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (_area_type, _area) in &self.mem_area_map {
            let _ = f.write_fmt(format_args!("\n{} - {:?}", _area_type.to_string(), _area));
        }
        let _ = f.write_fmt(format_args!("\nKernel init region size:{:#x}", self.guest_mem_size));
        f.write_fmt(format_args!("\nStack size:{:#x}", self.kernel_stack_size))
    }
}

pub struct VmResources {
    pub min_vcpu_amount: usize,
    pub kernel_bin_path: String,
    pub vdso_bin_path: String,
    pub sandbox_uid_name: String,
    pub pod_id: String,
    pub(in super) mem_layout: MemLayoutConfig,
}

impl VmResources {
    pub fn mem_area_info(&self, mem_area: MemAreaType) -> Option<(u64, u64, u64)> {
        if let Some((_, _mem_area)) = self.mem_layout.mem_area_map.get_key_value(&mem_area) {
            return Some((_mem_area.base_host, _mem_area.base_guest, _mem_area.size));
        }
        None
    }
}

impl fmt::Debug for VmResources {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\nVmResources[\nMin vcpu amount:{},\nPath to kernel bin:{:?},\n\
              Path to vdso bin:{},\nSandbox UID:{},\nPod ID:{},\nMemory layout:[{:?}]]",
              self.min_vcpu_amount, self.kernel_bin_path, self.vdso_bin_path,
              self.sandbox_uid_name, self.pod_id, self.mem_layout)
    }
}
