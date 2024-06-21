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

pub enum MemArea {
    PrivateHeapArea,
    SharedHeapArea,
    KernelArea,
    FileMapArea,
    HypercallMmioArea,
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
    pub fn mem_area_info(&self, mem_area: MemArea) -> Option<(u64, u64)> {
        match mem_area {
            #[cfg(feature = "cc")]
            MemArea::PrivateHeapArea =>
               Some((self.mem_layout.private_heap_mem_base,
                    self.mem_layout.private_heap_mem_size)),
            MemArea::SharedHeapArea =>
                Some((self.mem_layout.shared_heap_mem_base,
                      self.mem_layout.shared_heap_mem_size)),
            MemArea::KernelArea =>
                Some((self.mem_layout.kernel_base,
                      self.mem_layout.kernel_size)),
            MemArea::FileMapArea =>
                Some((self.mem_layout.file_map_area_base,
                      self.mem_layout.file_map_area_size)),
            #[cfg(target_arch = "aarch64")]
            MemArea::HypercallMmioArea =>
                Some((self.mem_layout.hypercall_mmio_base,
                      self.mem_layout.hypercall_mmio_size)),
            _ => None
        }
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
pub(in super) struct MemLayoutConfig {
    #[cfg(feauture = "cc")]
    pub(in super) private_heap_mem_base: u64,
    #[cfg(feauture = "cc")]
    pub(in super) private_heap_mem_size: u64,
    pub(in super) shared_heap_mem_base: u64,
    pub(in super) shared_heap_mem_size: u64,
    pub(in super) kernel_base: u64,
    pub(in super) kernel_size: u64,
    pub(in super) file_map_area_base: u64,
    pub(in super) file_map_area_size: u64,
    #[cfg(target_arch = "aarch64")]
    pub(in super) hypercall_mmio_base: u64,
    #[cfg(target_arch = "aarch64")]
    pub(in super) hypercall_mmio_size: u64,
    pub(in super) stack_size: usize,
}

impl fmt::Debug for MemLayoutConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        //NOTE: This is bad, don't know how to handel it...
        let mut pmb: u64 = std::u64::MAX;
        let mut pms: u64 = std::u64::MAX;
        #[cfg(feature = "cc")]{
            pmb = self.private_heap_mem_base;
            pms = self.private_heap_mem_size;
        }

        let mut hcb: u64 = std::u64::MAX;
        let mut hcs: u64 = std::u64::MAX;
        #[cfg(target_arch = "aarch64")] {
            hcb = self.hypercall_mmio_base;
            hcs = self.hypercall_mmio_size;
        }

        write!(f, "\nPrivate-Heap Memory base:{:#x},\nPrivate-Heap Memory size:{:#x},\n\
              Shared-Heap Memory base:{:#x},\nShared-Heap Memory size:{:#x},\nKernel base:{:#x},\n\
              Kernel size:{:#x},\nFile-Map base:{:#x},\n File-Map size:{:#x},\n\
              Hyperacall-MMIO base:{:#x},\nHypercall-MMIO size:{:#x},\nStack size:{:#x}", pmb, pms,
              self.shared_heap_mem_base, self.shared_heap_mem_size, self.kernel_base,
              self.kernel_size, self.file_map_area_base, self.file_map_area_size, hcb, hcs,
              self.stack_size)
    }
}
