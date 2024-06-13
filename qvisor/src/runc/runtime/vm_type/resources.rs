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
    pub fn mem_area_info(&self, mem_area: MemArea) -> Option<(u64, u64, u64)> {
        match mem_area {
            MemArea::PrivateHeapArea =>
               if self.mem_layout.guest_private_mem_layout.is_some() {
                   Some((self.mem_layout.guest_private_mem_layout
                         .unwrap().private_heap_mem_base_host,
                         self.mem_layout.guest_private_mem_layout
                         .unwrap().private_heap_mem_base_guest,
                         self.mem_layout.guest_private_mem_layout
                         .unwrap().private_heap_total_mem_size))
            } else {
                None
            },
            MemArea::SharedHeapArea =>
                Some((self.mem_layout.shared_heap_mem_base_host,
                      self.mem_layout.shared_heap_mem_base_guest,
                      self.mem_layout.shared_heap_mem_size)),
            MemArea::KernelArea =>
                Some((self.mem_layout.kernel_base_host,
                      self.mem_layout.kernel_base_guest,
                      self.mem_layout.kernel_init_region_size)),
            MemArea::FileMapArea =>
                Some((self.mem_layout.file_map_area_base_host,
                      self.mem_layout.file_map_area_base_guest,
                      self.mem_layout.file_map_area_size)),
            #[cfg(target_arch = "aarch64")]
            MemArea::HypercallMmioArea =>
                Some((u64::MAX, //Not backed on host
                       self.mem_layout.hypercall_mmio_base,
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

#[derive(Default, Copy, Clone)]
pub(in super) struct GuestPrivateMemLayout {
    pub(in super) private_heap_mem_base_host: u64,
    pub(in super) private_heap_mem_base_guest: u64,
    pub(in super) private_heap_init_mem_size: u64,
    pub(in super) private_heap_total_mem_size: u64,
}

#[derive(Default, Copy, Clone)]
pub(in super) struct MemLayoutConfig {
    pub(in super) guest_private_mem_layout: Option<GuestPrivateMemLayout>,
    pub(in super) shared_heap_mem_base_guest: u64,
    pub(in super) shared_heap_mem_base_host: u64,
    pub(in super) shared_heap_mem_size: u64,
    pub(in super) kernel_base_guest: u64,
    pub(in super) kernel_base_host: u64,
    pub(in super) kernel_init_region_size: u64,
    pub(in super) file_map_area_base_guest: u64,
    pub(in super) file_map_area_base_host: u64,
    pub(in super) file_map_area_size: u64,
    #[cfg(target_arch = "aarch64")]
    pub(in super) hypercall_mmio_base: u64,
    #[cfg(target_arch = "aarch64")]
    pub(in super) hypercall_mmio_size: u64,
    pub(in super) stack_size: usize,
}

impl fmt::Debug for MemLayoutConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[cfg(target_arch = "aarch64")] {
            f.write_fmt(format_args!("\nHyperacall-MMIO base gurst:{:#x}",
                self.hypercall_mmio_base));
            f.write_fmt(format_args!("\nHyperacall-MMIO size:{:#x}",
                self.hypercall_mmio_size));
        }
        if self.guest_private_mem_layout.is_some() {
            f.write_fmt(format_args!("\nPrivate-Heap Memory base host:{:#x}",
                self.guest_private_mem_layout.unwrap().private_heap_mem_base_host));
            f.write_fmt(format_args!("\nPrivate-Heap Memory base guest:{:#x}",
                self.guest_private_mem_layout.unwrap().private_heap_mem_base_guest));
            f.write_fmt(format_args!("\nPrivate-Heap Memory init size:{:#x}",
                self.guest_private_mem_layout.unwrap().private_heap_init_mem_size));
            f.write_fmt(format_args!("\nPrivate-Heap Memory total size:{:#x}",
                self.guest_private_mem_layout.unwrap().private_heap_total_mem_size));
        }
        f.write_fmt(format_args!("\nShared-Heap Memory base host:{:#x}",
            self.shared_heap_mem_base_host));
        f.write_fmt(format_args!("\nShared-Heap Memory base guest:{:#x}",
            self.shared_heap_mem_base_guest));
        f.write_fmt(format_args!("\nShared-Heap Memory size:{:#x}", self.shared_heap_mem_size));
        f.write_fmt(format_args!("\nKernel base host:{:#x}", self.kernel_base_host));
        f.write_fmt(format_args!("\nKernel base guest:{:#x}", self.kernel_base_guest));
        f.write_fmt(format_args!("\nKernel initial region:{:#x}", self.kernel_init_region_size));
        f.write_fmt(format_args!("\nFile-Map Memory base host:{:#x}",
            self.file_map_area_base_host));
        f.write_fmt(format_args!("\nFile-Map Memory base guest:{:#x}",
            self.file_map_area_base_guest));
        f.write_fmt(format_args!("\nFile-Map region:{:#x}", self.file_map_area_size));
        f.write_fmt(format_args!("\nGuest kernel stack size:{:#x}", self.stack_size))
    }
}
