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

use kvm_bindings::*;
use libc;
use libc::*;
use std::os::unix::io::AsRawFd;

pub const _IOC_NRBITS: c_uint = 8;
pub const _IOC_TYPEBITS: c_uint = 8;
pub const _IOC_SIZEBITS: c_uint = 14;
pub const _IOC_DIRBITS: c_uint = 2;
pub const _IOC_NRMASK: c_uint = 255;
pub const _IOC_TYPEMASK: c_uint = 255;
pub const _IOC_SIZEMASK: c_uint = 16383;
pub const _IOC_DIRMASK: c_uint = 3;
pub const _IOC_NRSHIFT: c_uint = 0;
pub const _IOC_TYPESHIFT: c_uint = 8;
pub const _IOC_SIZESHIFT: c_uint = 16;
pub const _IOC_DIRSHIFT: c_uint = 30;
pub const _IOC_NONE: c_uint = 0;
pub const _IOC_WRITE: c_uint = 1;
pub const _IOC_READ: c_uint = 2;
pub const IOC_IN: c_uint = 1_073_741_824;
pub const IOC_OUT: c_uint = 2_147_483_648;
pub const IOC_INOUT: c_uint = 3_221_225_472;
pub const IOCSIZE_MASK: c_uint = 1_073_676_288;
pub const IOCSIZE_SHIFT: c_uint = 16;

/// Raw macro to declare a function that returns an ioctl number.
#[macro_export]
macro_rules! ioctl_ioc_nr {
    ($name:ident, $dir:expr, $ty:expr, $nr:expr, $size:expr) => {
        #[allow(non_snake_case)]
        pub fn $name() -> ::std::os::raw::c_ulong {
            u64::from(
                ($dir << _IOC_DIRSHIFT)
                    | ($ty << _IOC_TYPESHIFT)
                    | ($nr << _IOC_NRSHIFT)
                    | ($size << _IOC_SIZESHIFT),
            )
        }
    };
}

/// Declare an ioctl that transfers no data.
#[macro_export]
macro_rules! ioctl_io_nr {
    ($name:ident, $ty:expr, $nr:expr) => {
        ioctl_ioc_nr!($name, _IOC_NONE, $ty, $nr, 0);
    };
}

/// Declare an ioctl that reads data.
#[macro_export]
macro_rules! ioctl_ior_nr {
    ($name:ident, $ty:expr, $nr:expr, $size:ty) => {
        ioctl_ioc_nr!(
            $name,
            _IOC_READ,
            $ty,
            $nr,
            ::std::mem::size_of::<$size>() as u32
        );
    };
}

/// Declare an ioctl that writes data.
#[macro_export]
macro_rules! ioctl_iow_nr {
    ($name:ident, $ty:expr, $nr:expr, $size:ty) => {
        ioctl_ioc_nr!(
            $name,
            _IOC_WRITE,
            $ty,
            $nr,
            ::std::mem::size_of::<$size>() as u32
        );
    };
}

/// Declare an ioctl that reads and writes data.
#[macro_export]
macro_rules! ioctl_iowr_nr {
    ($name:ident, $ty:expr, $nr:expr, $size:ty) => {
        ioctl_ioc_nr!(
            $name,
            _IOC_READ | _IOC_WRITE,
            $ty,
            $nr,
            ::std::mem::size_of::<$size>() as u32
        );
    };
}

ioctl_iow_nr!(KVM_INTERRUPT, KVMIO, 0x86, kvm_interrupt);
ioctl_iow_nr!(KVM_SET_REGS, KVMIO, 0x82, kvm_regs);

/// Run an ioctl with no arguments.
pub unsafe fn ioctl<F: AsRawFd>(fd: &F, req: c_ulong) -> c_int {
    libc::ioctl(fd.as_raw_fd(), req, 0)
}

/// Run an ioctl with a single value argument.
pub unsafe fn ioctl_with_val<F: AsRawFd>(fd: &F, req: c_ulong, arg: c_ulong) -> c_int {
    libc::ioctl(fd.as_raw_fd(), req, arg)
}

/// Run an ioctl with an immutable reference.
pub unsafe fn ioctl_with_ref<F: AsRawFd, T>(fd: &F, req: c_ulong, arg: &T) -> c_int {
    libc::ioctl(fd.as_raw_fd(), req, arg as *const T as *const c_void)
}

/// Run an ioctl with a mutable reference.
pub unsafe fn ioctl_with_mut_ref<F: AsRawFd, T>(fd: &F, req: c_ulong, arg: &mut T) -> c_int {
    libc::ioctl(fd.as_raw_fd(), req, arg as *mut T as *mut c_void)
}

/// Run an ioctl with a raw pointer.
pub unsafe fn ioctl_with_ptr<F: AsRawFd, T>(fd: &F, req: c_ulong, arg: *const T) -> c_int {
    libc::ioctl(fd.as_raw_fd(), req, arg as *const c_void)
}

/// Run an ioctl with a mutable raw pointer.
pub unsafe fn ioctl_with_mut_ptr<F: AsRawFd, T>(fd: &F, req: c_ulong, arg: *mut T) -> c_int {
    libc::ioctl(fd.as_raw_fd(), req, arg as *mut c_void)
}

pub fn QueueInterrupt(vcpu: &kvm_ioctls::VcpuFd, irq: u32) -> c_int {
    unsafe { ioctl_with_ptr(vcpu, KVM_INTERRUPT() as c_ulong, &kvm_interrupt { irq }) }
}

pub fn QueueTimer(vcpu: &kvm_ioctls::VcpuFd) -> c_int {
    //info!("the KVM_INTERRUPT id is {:x}", KVM_INTERRUPT());
    //info!("the KVM_SET_REGS id is {:x}", KVM_SET_REGS());
    let mut res = QueueInterrupt(vcpu, 32);

    if res != 0 {
        info!(
            "the io::Error::last_os_error() is {}",
            std::io::Error::last_os_error()
        );
        res = std::io::Error::last_os_error().raw_os_error().unwrap();
    }

    info!("after enque timer interrupt: res={}", res);
    return res;
}
