// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,x
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::mem::size_of;

use kvm_bindings::kvm_device_attr;
use vmm_sys_util::ioctl;

const KVM_IO: u32 = 0xAE;
const KVM_MEM_GUEST_MEMFD: u32 = 1 << 2;

pub const KVM_SET_DEVICE_ATTR: u64 = ioctl::ioctl_expr(
    ioctl::_IOC_WRITE,
    KVM_IO,
    0xE1,
    size_of::<kvm_device_attr>() as u32,
) as u64;

use vmm_sys_util::ioctl_ioc_nr;
vmm_sys_util::ioctl_io_nr!(KVM_CHECK_EXTENSION, KVM_IO, 0x03);

#[repr(C)]
pub struct KvmUserSpaceMemoryRegion2 {
    slot: u32,
    flags: u32,
    guest_phys_addr: u64,
    memory_size: u64,
    userspace_addr: u64,
    guest_memfd_offset: u64,
    guest_memfd: u32,
    __pad1: u32,
    __pad2: [u64; 14]
}

impl KvmUserSpaceMemoryRegion2 {
    pub fn new(_slot: u32, _flags: u32, _guest_phys_addr: u64, _memory_size: u64,
        _userspace_addr: u64, _guest_memfd_offset: u64, _guest_memfd: u32) -> Self {
        Self {
            slot: _slot,
            flags: _flags | KVM_MEM_GUEST_MEMFD,
            guest_phys_addr: _guest_phys_addr,
            memory_size: _memory_size,
            userspace_addr: _userspace_addr,
            guest_memfd_offset: _guest_memfd_offset,
            guest_memfd: _guest_memfd,
            __pad1: 0,
            __pad2: [0u64; 14]
        }
    }
}

#[repr(C)]
pub struct KvmCreateGuestMemFd {
   size: u64,
    flags: u64,
    __pad: [u64; 6],
}

impl KvmCreateGuestMemFd {
    pub fn new(_size: u64, _flags: u64) -> Self {
        Self {
            size: _size,
            flags: _flags,
            __pad: [0u64; 6]
        }
    }
}

const KVM_CREATE_GUEST_MEMFD: u64 = vmm_sys_util::ioctl::ioctl_expr(
    vmm_sys_util::ioctl::_IOC_WRITE | vmm_sys_util::ioctl::_IOC_READ,
    KVM_IO,
    0xD4,
    std::mem::size_of::<KvmCreateGuestMemFd>() as u32,
) as u64;

const KVM_SET_USER_MEMORY_REGION2: u64 = vmm_sys_util::ioctl::ioctl_expr(
    vmm_sys_util::ioctl::_IOC_WRITE,
    KVM_IO,
    0x49,
    std::mem::size_of::<KvmUserSpaceMemoryRegion2>() as u32
) as u64;

pub mod kvm_ioctl {
    use kvm_ioctls::{VmFd, Kvm};
    use libc::c_ulong;
    use vmm_sys_util::ioctl::ioctl_with_mut_ref;
    use crate::qlib::common::Error;
    use super::{KvmCreateGuestMemFd, KVM_CREATE_GUEST_MEMFD, KvmUserSpaceMemoryRegion2,
        KVM_SET_USER_MEMORY_REGION2, KVM_CHECK_EXTENSION};

    pub fn kvm_create_guest_memfd(vm_fd: &VmFd, guest_memfd: &mut KvmCreateGuestMemFd)
        -> Result<u32, Error> {
        let res = unsafe {
            ioctl_with_mut_ref(vm_fd, KVM_CREATE_GUEST_MEMFD, guest_memfd)
        };
        if res < 0 {
            let os_error = std::io::Error::last_os_error();
            error!("VM: Failed to create guest-memfd with error-{os_error:?}");
            return Err(Error::IOError(String::from("Kvm ioctl - failed")));
        }
        Ok(res as u32)
    }

    pub fn kvm_set_user_memory_region2(vm_fd: &VmFd, memory_region: &mut KvmUserSpaceMemoryRegion2)
        -> Result<(), Error> {
        let res = unsafe {
            ioctl_with_mut_ref(vm_fd, KVM_SET_USER_MEMORY_REGION2, memory_region)
        };
        if res < 0 {
            let os_error = std::io::Error::last_os_error();
            error!("VM: Failed to set user memory region with error-{os_error:?}");
            return Err(Error::IOError(String::from("Kvm ioctl - failed")));
        }
        Ok(())
    }

    pub fn kvm_supports_extension(kvm_fd: &Kvm, extension: u32) -> u64 {
        let ret = unsafe {
            vmm_sys_util::ioctl::ioctl_with_val(kvm_fd, KVM_CHECK_EXTENSION(), extension as c_ulong)
        };
        if ret < 0 {
            panic!("KVM: Failed to check extension:{} with error:{:?}",
                extension, std::io::Error::last_os_error());
        }
        ret as u64
    }
}
