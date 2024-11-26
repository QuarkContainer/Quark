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

use core::mem::size_of;

use self::range::Range;
use alloc::boxed::Box;

use super::super::common::*;
use super::super::config::*;
use super::super::linux_def::*;
use super::super::loader::*;
use super::super::qmsg;
use super::super::qmsg::*;
use super::super::socket_buf::*;
use super::super::tsot_msg::*;
use super::super::*;
use super::Kernel::*;
use crate::kernel_def::HyperCall64;
use crate::qlib::control_msg::ControlMsg;
use crate::qlib::proxy::*;
use crate::GLOBAL_ALLOCATOR;
use crate::GUEST_HOST_SHARED_ALLOCATOR;
use crate::qlib::kernel::arch::tee::is_cc_active;

impl HostSpace {
    pub fn WakeupVcpu(vcpuId: u64) {
        HyperCall64(HYPERCALL_WAKEUP_VCPU, vcpuId, 0, 0, 0);
    }

    pub fn IOWait() {
        HyperCall64(HYPERCALL_IOWAIT, 0, 0, 0, 0);
    }

    pub fn Hlt() {
        HyperCall64(HYPERCALL_HLT, 0, 0, 0, 0);
    }

    pub fn LoadProcessKernel(processAddr: u64) -> i64 {
        if is_cc_active() {
            let mut new_process = Box::new_in(Process::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let process_ptr = &mut *new_process as *mut _;
            let mut msg = Box::new_in(
                Msg::LoadProcessKernel(LoadProcessKernel {
                    processAddr: process_ptr as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;
            let private_process = unsafe { &mut *(processAddr as *mut Process) };
            *private_process = (*new_process).clone();
            return ret;
        } else {
            let mut msg = Msg::LoadProcessKernel(LoadProcessKernel {
                processAddr: processAddr,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn CreateMemfd(len: i64, flags: u32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::CreateMemfd(CreateMemfd {
                    len: len,
                    flags: flags,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::CreateMemfd(CreateMemfd {
                len: len,
                flags: flags,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Fallocate(fd: i32, mode: i32, offset: i64, len: i64) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::Fallocate(Fallocate {
                    fd,
                    mode,
                    offset,
                    len,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Fallocate(Fallocate {
                fd,
                mode,
                offset,
                len,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Sysinfo(addr: u64) -> i64 {
        if is_cc_active() {
            let mut new_info = Box::new_in(LibcSysinfo::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let info_ptr = &mut *new_info as *mut _;
            let mut msg = Box::new_in(
                Msg::Sysinfo(Sysinfo {
                    addr: info_ptr as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;
            unsafe {
                *(addr as *mut LibcSysinfo) = *info_ptr;
            }
            return ret;
        } else {
            let mut msg = Msg::Sysinfo(Sysinfo { addr });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn EventfdWrite(fd: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::EventfdWrite(EventfdWrite { fd }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::EventfdWrite(EventfdWrite { fd });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn RenameAt(olddirfd: i32, oldpath: u64, newdirfd: i32, newpath: u64) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::RenameAt(RenameAt {
                    olddirfd,
                    oldpath,
                    newdirfd,
                    newpath,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::RenameAt(RenameAt {
                olddirfd,
                oldpath,
                newdirfd,
                newpath,
            });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn HostMemoryBarrier() -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::HostMemoryBarrier(HostMemoryBarrier {}),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::HostMemoryBarrier(HostMemoryBarrier {});

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Ftruncate(fd: i32, len: i64) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::Ftruncate(Ftruncate { fd, len }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Ftruncate(Ftruncate { fd, len });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Rdtsc() -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(Msg::Rdtsc(Rdtsc {}), GUEST_HOST_SHARED_ALLOCATOR);

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Rdtsc(Rdtsc {});

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn SetTscOffset(offset: i64) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::SetTscOffset(SetTscOffset { offset: offset }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::SetTscOffset(SetTscOffset { offset: offset });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn TlbShootdown(vcpuMask: u64) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::TlbShootdown(TlbShootdown { vcpuMask: vcpuMask }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::TlbShootdown(TlbShootdown { vcpuMask: vcpuMask });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn IORead(fd: i32, iovs: u64, iovcnt: i32) -> i64 {
        if is_cc_active() {
            let iovs_size = (size_of::<IoVec>()) * (iovcnt as usize);
            let iovs_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(iovs_size, 0x8) as *mut IoVec };
            unsafe {
                core::ptr::copy_nonoverlapping(iovs as *const u8, iovs_ptr as *mut u8, iovs_size);
            }

            let mut msg = Box::new_in(
                Msg::IORead(IORead {
                    fd,
                    iovs: iovs_ptr as u64,
                    iovcnt,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;

            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(iovs_ptr as *mut u8, iovs_size, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::IORead(IORead { fd, iovs, iovcnt });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn IOTTYRead(fd: i32, iovs: u64, iovcnt: i32) -> i64 {
        if is_cc_active() {
            let iovs_size = (size_of::<IoVec>()) * (iovcnt as usize);
            let iovs_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(iovs_size, 0x8) as *mut IoVec };
            unsafe {
                core::ptr::copy_nonoverlapping(iovs as *const u8, iovs_ptr as *mut u8, iovs_size);
            }

            let mut msg = Box::new_in(
                Msg::IOTTYRead(IOTTYRead {
                    fd,
                    iovs: iovs_ptr as u64,
                    iovcnt,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;

            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(iovs_ptr as *mut u8, iovs_size, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::IOTTYRead(IOTTYRead { fd, iovs, iovcnt });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn IOWrite(fd: i32, iovs: u64, iovcnt: i32) -> i64 {
        if is_cc_active() {
            let iovs_size = (size_of::<IoVec>()) * (iovcnt as usize);
            let iovs_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(iovs_size, 0x8) as *mut IoVec };
            unsafe {
                core::ptr::copy_nonoverlapping(iovs as *const u8, iovs_ptr as *mut u8, iovs_size);
            }

            let mut msg = Box::new_in(
                Msg::IOWrite(IOWrite {
                    fd,
                    iovs: iovs_ptr as u64,
                    iovcnt,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;

            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(iovs_ptr as *mut u8, iovs_size, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::IOWrite(IOWrite { fd, iovs, iovcnt });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn IOReadAt(fd: i32, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        if is_cc_active() {
            let iovs_size = (size_of::<IoVec>()) * (iovcnt as usize);
            let iovs_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(iovs_size, 0x8) as *mut IoVec };
            unsafe {
                core::ptr::copy_nonoverlapping(iovs as *const u8, iovs_ptr as *mut u8, iovs_size);
            }

            let mut msg = Box::new_in(
                Msg::IOReadAt(IOReadAt {
                    fd,
                    iovs: iovs_ptr as u64,
                    iovcnt,
                    offset,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;

            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(iovs_ptr as *mut u8, iovs_size, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::IOReadAt(IOReadAt {
                fd,
                iovs,
                iovcnt,
                offset,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn IOWriteAt(fd: i32, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        if is_cc_active() {
            let iovs_size = (size_of::<IoVec>()) * (iovcnt as usize);
            let iovs_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(iovs_size, 0x8) as *mut IoVec };
            unsafe {
                core::ptr::copy_nonoverlapping(iovs as *const u8, iovs_ptr as *mut u8, iovs_size);
            }

            let mut msg = Box::new_in(
                Msg::IOWriteAt(IOWriteAt {
                    fd,
                    iovs: iovs_ptr as u64,
                    iovcnt,
                    offset,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;

            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(iovs_ptr as *mut u8, iovs_size, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::IOWriteAt(IOWriteAt {
                fd,
                iovs,
                iovcnt,
                offset,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn IOAppend(fd: i32, iovs: u64, iovcnt: i32) -> (i64, i64) {
        if is_cc_active() {
            let iovs_size = (size_of::<IoVec>()) * (iovcnt as usize);
            let iovs_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(iovs_size, 0x8) as *mut IoVec };
            unsafe {
                core::ptr::copy_nonoverlapping(iovs as *const u8, iovs_ptr as *mut u8, iovs_size);
            }

            let mut file_len = Box::new_in(0i64, GUEST_HOST_SHARED_ALLOCATOR);
            let file_len_ptr = &mut *file_len as *mut _;

            let mut msg = Box::new_in(
                Msg::IOAppend(IOAppend {
                    fd,
                    iovs: iovs_ptr as u64,
                    iovcnt,
                    fileLenAddr: file_len_ptr as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;
            if ret < 0 {
                return (ret, 0);
            }

            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(iovs_ptr as *mut u8, iovs_size, 0x8);
            }

            return (ret, *file_len);
        } else {
            let mut fileLen: i64 = 0;
            let mut msg = Msg::IOAppend(IOAppend {
                fd,
                iovs,
                iovcnt,
                fileLenAddr: &mut fileLen as *mut _ as u64,
            });

            let ret = HostSpace::Call(&mut msg, false) as i64;
            if ret < 0 {
                return (ret, 0);
            }

            return (ret, fileLen);
        }
    }

    pub fn IOAccept(fd: i32, addr: u64, addrlen: u64) -> i64 {
        if is_cc_active() {
            let mut new_socket = Box::new_in(TcpSockAddr::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let mut new_len =
                Box::new_in((*new_socket).data.len() as u32, GUEST_HOST_SHARED_ALLOCATOR);
            let socket_ptr = &mut *new_socket as *mut _;
            let len_ptr = &mut *new_len as *mut _;

            let mut msg = Box::new_in(
                Msg::IOAccept(IOAccept {
                    fd,
                    addr: (socket_ptr as u64),
                    addrlen: (len_ptr as u64),
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::HCall(&mut msg, false) as i64;
            let socket = unsafe { &mut *(addr as *mut TcpSockAddr) };
            socket.data = new_socket.data;
            let len = unsafe { &mut *(addrlen as *mut u32) };
            *len = *new_len;
            return ret;
        } else {
            let mut msg = Msg::IOAccept(IOAccept { fd, addr, addrlen });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn IOConnect(fd: i32, addr: u64, addrlen: u32) -> i64 {
        if is_cc_active() {
            let socket_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(addrlen as usize, 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(addr as *const u8, socket_ptr, addrlen as usize);
            }

            let mut msg = Box::new_in(
                Msg::IOConnect(IOConnect {
                    fd,
                    addr: socket_ptr as u64,
                    addrlen,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;

            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(socket_ptr as *mut u8, addrlen as usize, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::IOConnect(IOConnect { fd, addr, addrlen });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn IORecvMsg(fd: i32, msghdr: u64, flags: i32, blocking: bool) -> i64 {
        if is_cc_active() {
            let mut hasName = false;
            let mut hasControl = false;
            let mut new_name_buff = core::ptr::null::<u8>() as *mut u8;
            let mut new_control_buff = core::ptr::null::<u8>() as *mut u8;
            let mut nameLen = 0u32;
            let mut msgControlLen = 0usize;
            let mut new_msghdr = Box::new_in(MsgHdr::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let new_msghdr_ptr = &mut *new_msghdr as *mut _;
            let private_msghdr = unsafe { &mut *(msghdr as *mut MsgHdr) };
            *new_msghdr = *private_msghdr;

            //new_msghdr.msgName is an array if new_msghdr.nameLen is not null;
            if new_msghdr.nameLen != 0 {
                nameLen = new_msghdr.nameLen;
                hasName = true;
                new_name_buff =
                    unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(new_msghdr.nameLen as usize, 0x8) };
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        new_msghdr.msgName as *const u8,
                        new_name_buff,
                        nameLen as usize,
                    );
                }
                new_msghdr.msgName = new_name_buff as u64;
            }

            //new_msghdr.msgControl is a vec in private memory,if is null msgHdr.msgControl = ptr::null::<u8>() as u64;
            if !(new_msghdr.msgControl == core::ptr::null::<u8>() as u64) {
                msgControlLen = new_msghdr.msgControlLen;
                hasControl = true;
                new_control_buff = unsafe {
                    GLOBAL_ALLOCATOR.AllocSharedBuf(new_msghdr.msgControlLen as usize, 0x8)
                };
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        new_msghdr.msgControl as *const u8,
                        new_control_buff,
                        msgControlLen,
                    );
                }
                new_msghdr.msgControl = new_control_buff as u64;
            }

            let mut new_iov = Box::new_in(IoVec::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let private_iov = unsafe { *(private_msghdr.iov as *const IoVec) };
            *new_iov = private_iov;
            new_msghdr.iov = &mut *new_iov as *const _ as u64;

            let mut msg = Box::new_in(
                Msg::IORecvMsg(IORecvMsg {
                    fd,
                    msghdr: new_msghdr_ptr as u64,
                    flags,
                    blocking,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;

            if hasName {
                let updated_len = new_msghdr.nameLen;
                private_msghdr.nameLen = updated_len;
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        new_name_buff as *const u8,
                        private_msghdr.msgName as *mut u8,
                        nameLen as usize,
                    )
                };
                unsafe {
                    GLOBAL_ALLOCATOR.DeallocShareBuf(
                        new_name_buff as *mut u8,
                        nameLen as usize,
                        0x8,
                    );
                }
            }
            if hasControl {
                let updated_len = new_msghdr.msgControlLen;
                private_msghdr.msgControlLen = updated_len;
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        new_control_buff as *const u8,
                        private_msghdr.msgControlLen as *mut u8,
                        msgControlLen,
                    )
                };
                unsafe {
                    GLOBAL_ALLOCATOR.DeallocShareBuf(
                        new_control_buff as *mut u8,
                        msgControlLen,
                        0x8,
                    );
                }
            }

            private_msghdr.msgFlags = new_msghdr.msgFlags;
            return ret;
        } else {
            let mut msg = Msg::IORecvMsg(IORecvMsg {
                fd,
                msghdr,
                flags,
                blocking,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn IORecvfrom(fd: i32, buf: u64, size: usize, flags: i32, addr: u64, len: u64) -> i64 {
        if is_cc_active() {
            let mut hasName = false;
            let mut new_name_buff = core::ptr::null::<u8>() as *mut u8;
            let mut new_len_ptr = core::ptr::null::<u32>() as *mut u32;
            let mut nameLen = 0u32;
            let addrlen = unsafe { &mut *(len as *mut u32) };
            if (*addrlen) != 0 {
                nameLen = *addrlen;
                hasName = true;
                new_name_buff =
                    unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf((*addrlen) as usize, 0x8) };
                new_len_ptr =
                    unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size_of::<u32>(), 0x8) as *mut u32 };
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        addr as *const u8,
                        new_name_buff,
                        nameLen as usize,
                    );
                }
                unsafe {
                    *new_len_ptr = nameLen;
                }
            }
            let mut msg = Box::new_in(
                Msg::IORecvfrom(IORecvfrom {
                    fd,
                    buf,
                    size,
                    flags,
                    addr: if hasName { new_name_buff as u64 } else { addr },
                    len: if hasName { new_len_ptr as u64 } else { len },
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;

            if hasName {
                let updated_len = unsafe { *new_len_ptr };
                (*addrlen) = updated_len;
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        new_name_buff as *const u8,
                        addr as *mut u8,
                        nameLen as usize,
                    )
                };
                unsafe {
                    GLOBAL_ALLOCATOR.DeallocShareBuf(
                        new_name_buff as *mut u8,
                        nameLen as usize,
                        0x8,
                    );
                }
                unsafe {
                    GLOBAL_ALLOCATOR.DeallocShareBuf(new_len_ptr as *mut u8, size_of::<u32>(), 0x8);
                }
            }
            return ret;
        } else {
            let mut msg = Msg::IORecvfrom(IORecvfrom {
                fd,
                buf,
                size,
                flags,
                addr,
                len,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn IOSendMsg(fd: i32, msghdr: u64, flags: i32, blocking: bool) -> i64 {
        if is_cc_active() {
            let mut hasName = false;
            let mut hasControl = false;
            let mut new_name_buff = core::ptr::null::<u8>() as *mut u8;
            let mut new_control_buff = core::ptr::null::<u8>() as *mut u8;
            let mut nameLen = 0u32;
            let mut msgControlLen = 0usize;
            let mut new_msghdr = Box::new_in(MsgHdr::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let new_msghdr_ptr = &mut *new_msghdr as *mut _;
            let private_msghdr = unsafe { &mut *(msghdr as *mut MsgHdr) };
            *new_msghdr = *private_msghdr;

            if new_msghdr.nameLen != 0 {
                nameLen = new_msghdr.nameLen;
                hasName = true;
                new_name_buff =
                    unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(new_msghdr.nameLen as usize, 0x8) };
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        new_msghdr.msgName as *const u8,
                        new_name_buff,
                        nameLen as usize,
                    );
                }
                new_msghdr.msgName = new_name_buff as u64;
            }

            if !(new_msghdr.msgControl == core::ptr::null::<u8>() as u64) {
                msgControlLen = new_msghdr.msgControlLen;
                hasControl = true;
                new_control_buff = unsafe {
                    GLOBAL_ALLOCATOR.AllocSharedBuf(new_msghdr.msgControlLen as usize, 0x8)
                };
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        new_msghdr.msgControl as *const u8,
                        new_control_buff,
                        msgControlLen,
                    );
                }
                new_msghdr.msgControl = new_control_buff as u64;
            }

            let mut new_iov = Box::new_in(IoVec::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let private_iov = unsafe { *(private_msghdr.iov as *const IoVec) };
            *new_iov = private_iov;
            new_msghdr.iov = &mut *new_iov as *const _ as u64;

            let mut msg = Box::new_in(
                Msg::IOSendMsg(IOSendMsg {
                    fd,
                    msghdr: new_msghdr_ptr as u64,
                    flags,
                    blocking,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;

            if hasName {
                let updated_len = new_msghdr.nameLen;
                private_msghdr.nameLen = updated_len;
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        new_name_buff as *const u8,
                        private_msghdr.msgName as *mut u8,
                        nameLen as usize,
                    )
                };
                unsafe {
                    GLOBAL_ALLOCATOR.DeallocShareBuf(
                        new_name_buff as *mut u8,
                        nameLen as usize,
                        0x8,
                    );
                }
            }

            if hasControl {
                let updated_len = new_msghdr.msgControlLen;
                private_msghdr.msgControlLen = updated_len;
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        new_control_buff as *const u8,
                        private_msghdr.msgControlLen as *mut u8,
                        msgControlLen,
                    )
                };
                unsafe {
                    GLOBAL_ALLOCATOR.DeallocShareBuf(
                        new_control_buff as *mut u8,
                        msgControlLen,
                        0x8,
                    );
                }
            }

            private_msghdr.msgFlags = new_msghdr.msgFlags;
            return ret;
        } else {
            let mut msg = Msg::IOSendMsg(IOSendMsg {
                fd,
                msghdr,
                flags,
                blocking,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn IOSendto(fd: i32, buf: u64, size: usize, flags: i32, addr: u64, len: u32) -> i64 {
        if is_cc_active() {
            let mut hasName = false;
            let mut new_name_buff = core::ptr::null::<u8>() as *mut u8;
            if len != 0 {
                hasName = true;
                new_name_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len as usize, 0x8) };
                unsafe {
                    core::ptr::copy_nonoverlapping(addr as *const u8, new_name_buff, len as usize);
                }
            }

            let mut msg = Box::new_in(
                Msg::IOSendto(IOSendto {
                    fd,
                    buf,
                    size,
                    flags,
                    addr: if hasName { new_name_buff as u64 } else { addr },
                    len,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;
            if hasName {
                unsafe {
                    GLOBAL_ALLOCATOR.DeallocShareBuf(new_name_buff as *mut u8, len as usize, 0x8);
                }
            }

            return ret;
        } else {
            let mut msg = Msg::IOSendto(IOSendto {
                fd,
                buf,
                size,
                flags,
                addr,
                len,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn GetTimeOfDay(tv: u64, tz: u64) -> i64 {
        if is_cc_active() {
            let mut new_tv = Box::new_in(
                super::super::linux::time::Timeval::default(),
                GUEST_HOST_SHARED_ALLOCATOR,
            );
            let new_tv_ptr = &mut *new_tv as *mut _;
            let mut new_tz = Box::new_in([0u32; 2], GUEST_HOST_SHARED_ALLOCATOR);
            let new_tz_ptr = &mut *new_tz as *mut _;
            let mut msg = Box::new_in(
                Msg::GetTimeOfDay(GetTimeOfDay {
                    tv: new_tv_ptr as u64,
                    tz: new_tz_ptr as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;

            let private_tv = unsafe { &mut *(tv as *mut super::super::linux::time::Timeval) };
            let private_tz = unsafe { &mut *(tz as *mut [u32; 2]) };
            *private_tv = *new_tv;
            *private_tz = *new_tz;
            return ret;
        } else {
            let mut msg = Msg::GetTimeOfDay(GetTimeOfDay { tv, tz });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn ReadLinkAt(dirfd: i32, path: u64, buf: u64, bufsize: u64) -> i64 {
        if is_cc_active() {
            let new_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(bufsize as usize, 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(
                    buf as *const u8,
                    new_buff as *mut u8,
                    bufsize as usize,
                );
            }
            let mut msg = Box::new_in(
                Msg::ReadLinkAt(ReadLinkAt {
                    dirfd,
                    path,
                    buf: new_buff as u64,
                    bufsize,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;
            unsafe {
                core::ptr::copy_nonoverlapping(
                    new_buff as *const u8,
                    buf as *mut u8,
                    bufsize as usize,
                );
            }
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_buff as *mut u8, bufsize as usize, 0x8);
            }

            return ret;
        } else {
            let mut msg = Msg::ReadLinkAt(ReadLinkAt {
                dirfd,
                path,
                buf,
                bufsize,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Fcntl(fd: i32, cmd: i32, arg: u64) -> i64 {
        if is_cc_active() {
            //Here arg can be used directly, flock is implemented inside the kernel
            let mut msg = Box::new_in(
                Msg::Fcntl(Fcntl { fd, cmd, arg }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Fcntl(Fcntl { fd, cmd, arg });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn IoCtl(fd: i32, cmd: u64, argp: u64, argplen: usize) -> i64 {
        if is_cc_active() {
            let new_argp = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(argplen, 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(argp as *const u8, new_argp as *mut u8, argplen);
            }
            let mut msg = Box::new_in(
                Msg::IoCtl(IoCtl {
                    fd,
                    cmd,
                    argp: new_argp as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );
            let ret = HostSpace::Call(&mut msg, false) as i64;
            unsafe {
                core::ptr::copy_nonoverlapping(new_argp as *const u8, argp as *mut u8, argplen);
            }
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_argp as *mut u8, argplen, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::IoCtl(IoCtl { fd, cmd, argp });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Fstatfs(fd: i32, buf: u64) -> i64 {
        if is_cc_active() {
            let mut new_buff = Box::new_in(LibcStatfs::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let new_buff_ptr = &mut *new_buff as *mut _;

            let mut msg = Box::new_in(
                Msg::Fstatfs(Fstatfs {
                    fd,
                    buf: new_buff_ptr as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;

            let private_buff = unsafe { &mut *(buf as *mut LibcStatfs) };
            *private_buff = *new_buff;
            return ret;
        } else {
            let mut msg = Msg::Fstatfs(Fstatfs { fd, buf });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn NewSocket(fd: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::NewSocket(NewSocket { fd }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, true) as i64;
        } else {
            let mut msg = Msg::NewSocket(NewSocket { fd });

            return HostSpace::HCall(&mut msg, true) as i64;
        }
    }

    //not used?
    pub fn FAccessAt(dirfd: i32, pathname: u64, mode: i32, flags: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::FAccessAt(FAccessAt {
                    dirfd,
                    pathname,
                    mode,
                    flags,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::FAccessAt(FAccessAt {
                dirfd,
                pathname,
                mode,
                flags,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Fstat(fd: i32, buff: u64) -> i64 {
        if is_cc_active() {
            let mut new_buff = Box::new_in(LibcStat::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let new_buff_ptr = &mut *new_buff as *mut _;

            let mut msg = Box::new_in(
                Msg::Fstat(Fstat {
                    fd,
                    buff: new_buff_ptr as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = Self::HCall(&mut msg, false) as i64;

            let private_buff = unsafe { &mut *(buff as *mut LibcStat) };
            *private_buff = *new_buff;
            return ret;
        } else {
            let mut msg = Msg::Fstat(Fstat { fd, buff });

            return Self::HCall(&mut msg, false) as i64;
        }
    }

    pub fn Fstatat(dirfd: i32, pathname: u64, buff: u64, flags: i32) -> i64 {
        if is_cc_active() {
            let mut new_buff = Box::new_in(LibcStat::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let new_buff_ptr = &mut *new_buff as *mut _;

            let mut msg = Box::new_in(
                Msg::Fstatat(Fstatat {
                    dirfd,
                    pathname,
                    buff: new_buff_ptr as u64,
                    flags,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;

            let private_buff = unsafe { &mut *(buff as *mut LibcStat) };
            *private_buff = *new_buff;
            return ret;
        } else {
            let mut msg = Msg::Fstatat(Fstatat {
                dirfd,
                pathname,
                buff,
                flags,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Unlinkat(dirfd: i32, pathname: u64, flags: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::Unlinkat(Unlinkat {
                    dirfd,
                    pathname,
                    flags,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Unlinkat(Unlinkat {
                dirfd,
                pathname,
                flags,
            });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn Mkdirat(dirfd: i32, pathname: u64, mode_: u32, uid: u32, gid: u32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::Mkdirat(Mkdirat {
                    dirfd,
                    pathname,
                    mode_,
                    uid,
                    gid,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Mkdirat(Mkdirat {
                dirfd,
                pathname,
                mode_,
                uid,
                gid,
            });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn Mkfifoat(dirfd: i32, name: u64, mode: u32, uid: u32, gid: u32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::Mkfifoat(Mkfifoat {
                    dirfd,
                    name,
                    mode,
                    uid,
                    gid,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Mkfifoat(Mkfifoat {
                dirfd,
                name,
                mode,
                uid,
                gid,
            });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn Proxy(cmd: ProxyCommand, parameters: ProxyParameters) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::Proxy(Proxy { cmd, parameters }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Proxy(Proxy { cmd, parameters });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn SwapInPage(addr: u64) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::SwapInPage(SwapInPage { addr }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::SwapInPage(SwapInPage { addr });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn SwapOut() -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(Msg::SwapOut(SwapOut {}), GUEST_HOST_SHARED_ALLOCATOR);

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::SwapOut(SwapOut {});

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn SwapIn() -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(Msg::SwapIn(SwapIn {}), GUEST_HOST_SHARED_ALLOCATOR);

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::SwapIn(SwapIn {});

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn SysSync() -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(Msg::SysSync(SysSync {}), GUEST_HOST_SHARED_ALLOCATOR);

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::SysSync(SysSync {});

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn SyncFs(fd: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(Msg::SyncFs(SyncFs { fd }), GUEST_HOST_SHARED_ALLOCATOR);

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::SyncFs(SyncFs { fd });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn SyncFileRange(fd: i32, offset: i64, nbytes: i64, flags: u32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::SyncFileRange(SyncFileRange {
                    fd,
                    offset,
                    nbytes,
                    flags,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::SyncFileRange(SyncFileRange {
                fd,
                offset,
                nbytes,
                flags,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn FSync(fd: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(Msg::FSync(FSync { fd }), GUEST_HOST_SHARED_ALLOCATOR);

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::FSync(FSync { fd });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn MSync(addr: u64, len: usize, flags: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::MSync(MSync { addr, len, flags }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::MSync(MSync { addr, len, flags });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn Madvise(addr: u64, len: usize, advise: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::MAdvise(MAdvise { addr, len, advise }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::MAdvise(MAdvise { addr, len, advise });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn FDataSync(fd: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::FDataSync(FDataSync { fd }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::FDataSync(FDataSync { fd });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Seek(fd: i32, offset: i64, whence: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::Seek(Seek { fd, offset, whence }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Seek(Seek { fd, offset, whence });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn ReadDir(dirfd: i32, data: &mut [u8], reset: bool) -> i64 {
        if is_cc_active() {
            let new_data_buff_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(data.len(), 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(
                    &mut data[0] as *const u8,
                    new_data_buff_ptr,
                    data.len(),
                );
            }
            let mut msg = Box::new_in(
                Msg::ReadDir(ReadDir {
                    dirfd,
                    addr: new_data_buff_ptr as u64,
                    len: data.len(),
                    reset: reset,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::HCall(&mut msg, false) as i64;
            unsafe {
                core::ptr::copy_nonoverlapping(
                    new_data_buff_ptr as *const u8,
                    &mut data[0] as *mut _,
                    data.len(),
                );
            }
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_data_buff_ptr, data.len(), 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::ReadDir(ReadDir {
                dirfd,
                addr: &mut data[0] as *mut _ as u64,
                len: data.len(),
                reset: reset,
            });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn FSetXattr(fd: i32, name: u64, value: u64, size: usize, flags: u32) -> i64 {
        if is_cc_active() {
            let new_value_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size, 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(value as *const u8, new_value_ptr, size);
            }

            let mut msg = Box::new_in(
                Msg::FSetXattr(FSetXattr {
                    fd,
                    name,
                    value: new_value_ptr as u64,
                    size,
                    flags,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_value_ptr, size, 0x8);
            }

            return ret;
        } else {
            let mut msg = Msg::FSetXattr(FSetXattr {
                fd,
                name,
                value,
                size,
                flags,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn FGetXattr(fd: i32, name: u64, value: u64, size: usize) -> i64 {
        if is_cc_active() {
            let new_value_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size, 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(value as *const u8, new_value_ptr, size);
            }

            let mut msg = Box::new_in(
                Msg::FGetXattr(FGetXattr {
                    fd,
                    name,
                    value: new_value_ptr as u64,
                    size,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            // FGetXattr has to be hcall as it will also be called by
            // inode::lookup --> OverlayHasWhiteout which might be called by create and hold a lock
            let ret = HostSpace::HCall(&mut msg, false) as i64;
            unsafe {
                core::ptr::copy_nonoverlapping(new_value_ptr, value as *mut u8, size);
            }
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_value_ptr, size, 0x8);
            }

            return ret;
        } else {
            let mut msg = Msg::FGetXattr(FGetXattr {
                fd,
                name,
                value,
                size,
            });

            // FGetXattr has to be hcall as it will also be called by
            // inode::lookup --> OverlayHasWhiteout which might be called by create and hold a lock
            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn FRemoveXattr(fd: i32, name: u64) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::FRemoveXattr(FRemoveXattr { fd, name }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::FRemoveXattr(FRemoveXattr { fd, name });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn FListXattr(fd: i32, list: u64, size: usize) -> i64 {
        if is_cc_active() {
            let new_list_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size, 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(list as *const u8, new_list_addr, size);
            }

            let mut msg = Box::new_in(
                Msg::FListXattr(FListXattr {
                    fd,
                    list: new_list_addr as u64,
                    size,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );
            let ret = HostSpace::Call(&mut msg, false) as i64;

            unsafe {
                core::ptr::copy_nonoverlapping(new_list_addr, list as *mut u8, size);
            }
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_list_addr, size, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::FListXattr(FListXattr { fd, list, size });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn GetRandom(buf: u64, len: u64, flags: u32) -> i64 {
        if is_cc_active() {
            let new_buf_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len as usize, 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(buf as *const u8, new_buf_addr, len as usize);
            }

            let mut msg = Box::new_in(
                Msg::GetRandom(GetRandom {
                    buf: new_buf_addr as u64,
                    len,
                    flags,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;

            unsafe {
                core::ptr::copy_nonoverlapping(new_buf_addr, buf as *mut u8, len as usize);
            }
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_buf_addr, len as usize, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::GetRandom(GetRandom { buf, len, flags });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Statm(statm: &mut StatmInfo) -> i64 {
        if is_cc_active() {
            let mut new_info = Box::new_in(StatmInfo::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let new_info_ptr = &mut *new_info as *mut _;
            let mut msg = Box::new_in(
                Msg::Statm(Statm {
                    buf: new_info_ptr as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;
            *statm = *new_info;
            return ret;
        } else {
            let mut msg = Msg::Statm(Statm {
                buf: statm as *const _ as u64,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Socket(domain: i32, type_: i32, protocol: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::Socket(Socket {
                    domain,
                    type_,
                    protocol,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Socket(Socket {
                domain,
                type_,
                protocol,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn UnblockedSocket(domain: i32, type_: i32, protocol: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::Socket(Socket {
                    domain,
                    type_,
                    protocol,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Socket(Socket {
                domain,
                type_,
                protocol,
            });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn GetSockName(sockfd: i32, addr: u64, addrlen: u64) -> i64 {
        if is_cc_active() {
            let len_size = size_of::<i32>();
            let len = unsafe { *(addrlen as *const i32) };
            let new_len_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len_size, 0x8) };
            let new_buff_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len as usize, 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(addr as *const u8, new_buff_addr, len as usize);
            }
            unsafe {
                core::ptr::copy_nonoverlapping(addrlen as *const u8, new_len_addr, len_size);
            }
            let mut msg = Box::new_in(
                Msg::GetSockName(GetSockName {
                    sockfd,
                    addr: new_buff_addr as u64,
                    addrlen: new_len_addr as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;
            unsafe {
                core::ptr::copy_nonoverlapping(new_buff_addr, addr as *mut u8, len as usize);
            }
            unsafe {
                core::ptr::copy_nonoverlapping(new_len_addr, addrlen as *mut u8, len_size);
            }
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_buff_addr, len as usize, 0x8);
            }
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_len_addr, len_size, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::GetSockName(GetSockName {
                sockfd,
                addr,
                addrlen,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn GetPeerName(sockfd: i32, addr: u64, addrlen: u64) -> i64 {
        if is_cc_active() {
            let len_size = size_of::<i32>();
            let len = unsafe { *(addrlen as *const i32) };
            let new_len_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len_size, 0x8) };
            let new_buff_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len as usize, 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(addr as *const u8, new_buff_addr, len as usize);
            }
            unsafe {
                core::ptr::copy_nonoverlapping(addrlen as *const u8, new_len_addr, len_size);
            }
            let mut msg = Box::new_in(
                Msg::GetPeerName(GetPeerName {
                    sockfd,
                    addr: new_buff_addr as u64,
                    addrlen: new_len_addr as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;
            unsafe {
                core::ptr::copy_nonoverlapping(new_buff_addr, addr as *mut u8, len as usize);
            }
            unsafe {
                core::ptr::copy_nonoverlapping(new_len_addr, addrlen as *mut u8, len_size);
            }
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_buff_addr, len as usize, 0x8);
            }
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_len_addr, len_size, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::GetPeerName(GetPeerName {
                sockfd,
                addr,
                addrlen,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn GetSockOpt(sockfd: i32, level: i32, optname: i32, optval: u64, optlen: u64) -> i64 {
        if is_cc_active() {
            let val_size = unsafe { *(optlen as *const i32) } as usize;
            let len_size = size_of::<i32>();
            let new_val_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(val_size, 0x8) };
            let new_len_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len_size, 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(optval as *const u8, new_val_addr, val_size);
            }
            unsafe {
                core::ptr::copy_nonoverlapping(optlen as *const u8, new_len_addr, len_size);
            }
            let mut msg = Box::new_in(
                Msg::GetSockOpt(GetSockOpt {
                    sockfd,
                    level,
                    optname,
                    optval: new_val_addr as u64,
                    optlen: new_len_addr as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;
            unsafe {
                core::ptr::copy_nonoverlapping(new_len_addr, optlen as *mut u8, len_size);
            }
            unsafe {
                core::ptr::copy_nonoverlapping(new_val_addr, optval as *mut u8, val_size);
            }
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_len_addr, len_size, 0x8);
            }
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_val_addr, val_size, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::GetSockOpt(GetSockOpt {
                sockfd,
                level,
                optname,
                optval,
                optlen,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn SetSockOpt(sockfd: i32, level: i32, optname: i32, optval: u64, optlen: u32) -> i64 {
        if is_cc_active() {
            let new_val_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(optlen as usize, 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(optval as *const u8, new_val_addr, optlen as usize);
            }
            let mut msg = Box::new_in(
                Msg::SetSockOpt(SetSockOpt {
                    sockfd,
                    level,
                    optname,
                    optval: new_val_addr as u64,
                    optlen,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            //return Self::HCall_cc(&mut msg) as i64;
            let ret = HostSpace::Call(&mut msg, false) as i64;
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_val_addr, optlen as usize, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::SetSockOpt(SetSockOpt {
                sockfd,
                level,
                optname,
                optval,
                optlen,
            });

            //return Self::HCall(&mut msg) as i64;
            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Bind(sockfd: i32, addr: u64, addrlen: u32, umask: u32) -> i64 {
        if is_cc_active() {
            let new_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(addrlen as usize, 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(addr as *const u8, new_addr, addrlen as usize);
            }

            let mut msg = Box::new_in(
                Msg::IOBind(IOBind {
                    sockfd,
                    addr: new_addr as u64,
                    addrlen,
                    umask,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_addr, addrlen as usize, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::IOBind(IOBind {
                sockfd,
                addr,
                addrlen,
                umask,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Listen(sockfd: i32, backlog: i32, block: bool) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::IOListen(IOListen {
                    sockfd,
                    backlog,
                    block,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::IOListen(IOListen {
                sockfd,
                backlog,
                block,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn RDMAListen(sockfd: i32, backlog: i32, block: bool, acceptQueue: AcceptQueue) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::RDMAListen(RDMAListen {
                    sockfd,
                    backlog,
                    block,
                    acceptQueue,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::RDMAListen(RDMAListen {
                sockfd,
                backlog,
                block,
                acceptQueue,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn RDMANotify(sockfd: i32, typ: RDMANotifyType) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::RDMANotify(RDMANotify { sockfd, typ }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::RDMANotify(RDMANotify { sockfd, typ });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Shutdown(sockfd: i32, how: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::IOShutdown(IOShutdown { sockfd, how }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::IOShutdown(IOShutdown { sockfd, how });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn ExitVM(exitCode: i32) {
        HyperCall64(HYPERCALL_EXIT_VM, exitCode as u64, 0, 0, 0);
        //Self::AQCall(qmsg::HostOutputMsg::ExitVM(exitCode));
    }

    pub fn Panic(str: &str) {
        if is_cc_active() {
            //copy the &str to shared buffer
            let bytes = str.as_bytes();
            let len = bytes.len();
            let new_str_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len, 0x8) };

            let dest_ptr: *mut u8 = new_str_ptr;
            let src_ptr: *const u8 = bytes.as_ptr();
            unsafe {
                core::ptr::copy_nonoverlapping(src_ptr, dest_ptr, len);
            }
            let new_str = unsafe {
                alloc::str::from_utf8_unchecked(core::slice::from_raw_parts(dest_ptr, len))
            };

            let mut msg = Box::new_in(
                Print {
                    level: DebugLevel::Error,
                    str: new_str,
                },
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let msg_ptr = &mut *msg as *mut _;

            HyperCall64(HYPERCALL_PANIC, msg_ptr as u64, 0, 0, 0);
            //Here need to drop mannually, otherwise the compiler will drop ealier, causing error
            drop(msg);
        } else {
            let msg = Print {
                level: DebugLevel::Error,
                str: str,
            };

            HyperCall64(HYPERCALL_PANIC, &msg as *const _ as u64, 0, 0, 0);
        }
    }

    pub fn TryOpenWrite(dirfd: i32, oldfd: i32, name: u64) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::TryOpenWrite(TryOpenWrite {
                    dirfd: dirfd,
                    oldfd: oldfd,
                    name: name,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = Self::HCall(&mut msg, false) as i64;
            return ret;
        } else {
            let mut msg = Msg::TryOpenWrite(TryOpenWrite {
                dirfd: dirfd,
                oldfd: oldfd,
                name: name,
            });

            let ret = Self::HCall(&mut msg, false) as i64;
            return ret;
        }
    }

    pub fn TryOpenAt(dirfd: i32, name: u64, addr: u64, skiprw: bool) -> i64 {
        if is_cc_active() {
            let new_libcstat = Box::new_in(LibcStat::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let old_tryopen = unsafe { &mut *(addr as *mut TryOpenStruct) };
            let old_libcstat = unsafe { &mut *(old_tryopen.fstat as *const _ as *mut LibcStat) };
            let mut new_tryopen = Box::new_in(
                TryOpenStruct {
                    fstat: &*new_libcstat,
                    writeable: false,
                },
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let new_tryopen_ptr = &mut *new_tryopen as *mut TryOpenStruct;
            let mut msg = Box::new_in(
                Msg::TryOpenAt(TryOpenAt {
                    dirfd: dirfd,
                    name: name,
                    addr: new_tryopen_ptr as u64,
                    skiprw: skiprw,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = Self::HCall(&mut msg, false) as i64;
            *old_libcstat = *new_libcstat;
            old_tryopen.writeable = new_tryopen.writeable;

            return ret;
        } else {
            let mut msg = Msg::TryOpenAt(TryOpenAt {
                dirfd: dirfd,
                name: name,
                addr: addr,
                skiprw: skiprw,
            });

            let ret = Self::HCall(&mut msg, false) as i64;
            return ret;
        }
    }

    pub fn OpenAt(dirfd: i32, name: u64, flags: i32, addr: u64) -> i64 {
        if is_cc_active() {
            let new_libcstat = Box::new_in(LibcStat::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let old_tryopen = unsafe { &mut *(addr as *mut TryOpenStruct) };
            let old_libcstat = unsafe { &mut *(old_tryopen.fstat as *const _ as *mut LibcStat) };
            let mut new_tryopen = Box::new_in(
                TryOpenStruct {
                    fstat: &*new_libcstat,
                    writeable: false,
                },
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let new_tryopen_ptr = &mut *new_tryopen as *mut TryOpenStruct;

            let mut msg = Box::new_in(
                Msg::OpenAt(OpenAt {
                    dirfd: dirfd,
                    name: name,
                    flags: flags,
                    addr: new_tryopen_ptr as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = Self::HCall(&mut msg, false) as i64;
            *old_libcstat = *new_libcstat;
            old_tryopen.writeable = new_tryopen.writeable;
            return ret;
        } else {
            let mut msg = Msg::OpenAt(OpenAt {
                dirfd: dirfd,
                name: name,
                flags: flags,
                addr: addr,
            });

            let ret = Self::HCall(&mut msg, false) as i64;
            return ret;
            //return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn OpenDevFile(dirfd: i32, name: u64, flags: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::OpenDevFile(OpenDevFile {
                    dirfd: dirfd,
                    name: name,
                    flags: flags,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = Self::HCall(&mut msg, false) as i64;
            return ret;
        } else {
            let mut msg = Msg::OpenDevFile(OpenDevFile {
                dirfd: dirfd,
                name: name,
                flags: flags,
            });

            let ret = Self::HCall(&mut msg, false) as i64;
            return ret;
            //return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn RemapGuestMemRanges(len: u64, addr: u64, count: usize) -> i64 {
        if is_cc_active() {
            let range_size = size_of::<Range>();
            let new_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(range_size * count, 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(addr as *const u8, new_addr, range_size);
            }
            let mut msg = Box::new_in(
                Msg::RemapGuestMemRanges(RemapGuestMemRanges {
                    len: len,
                    addr: new_addr as u64,
                    count: count,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = Self::Call(&mut msg, false) as i64;
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_addr, range_size * count, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::RemapGuestMemRanges(RemapGuestMemRanges {
                len: len,
                addr: addr,
                count: count,
            });

            let ret = Self::Call(&mut msg, false) as i64;
            return ret;
        }
    }

    pub fn UnmapGuestMemRange(start: u64, len: u64) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::UnmapGuestMemRange(UnmapGuestMemRange {
                    start: start,
                    len: len,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = Self::Call(&mut msg, false) as i64;
            return ret;
        } else {
            let mut msg = Msg::UnmapGuestMemRange(UnmapGuestMemRange {
                start: start,
                len: len,
            });

            let ret = Self::Call(&mut msg, false) as i64;
            return ret;
        }
    }

    pub fn HostUnixConnect(type_: i32, addr: u64, len: usize) -> i64 {
        if is_cc_active() {
            let new_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len, 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(addr as *const u8, new_addr, len);
            }
            let mut msg = Box::new_in(
                Msg::HostUnixConnect(HostUnixConnect {
                    type_: type_,
                    addr: new_addr as u64,
                    len: len,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = Self::Call(&mut msg, false) as i64;
            return ret;
        } else {
            let mut msg = Msg::HostUnixConnect(HostUnixConnect {
                type_: type_,
                addr: addr,
                len: len,
            });

            let ret = Self::Call(&mut msg, false) as i64;
            return ret;
        }
    }

    pub fn HostUnixRecvMsg(fd: i32, msghdr: u64, flags: i32) -> i64 {
        if is_cc_active() {
            let mut hasName = false;
            let mut hasControl = false;
            let mut new_name_buff = core::ptr::null::<u8>() as *mut u8;
            let mut new_control_buff = core::ptr::null::<u8>() as *mut u8;
            let mut nameLen = 0u32;
            let mut msgControlLen = 0usize;
            let mut new_msghdr = Box::new_in(MsgHdr::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let new_msghdr_ptr = &mut *new_msghdr as *mut _;
            let private_msghdr = unsafe { &mut *(msghdr as *mut MsgHdr) };
            *new_msghdr = *private_msghdr;

            //new_msghdr.msgName is an array if new_msghdr.nameLen is not null;
            if new_msghdr.nameLen != 0 {
                nameLen = new_msghdr.nameLen;
                hasName = true;
                new_name_buff =
                    unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(new_msghdr.nameLen as usize, 0x8) };
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        new_msghdr.msgName as *const u8,
                        new_name_buff,
                        nameLen as usize,
                    );
                }
                new_msghdr.msgName = new_name_buff as u64;
            }

            //new_msghdr.msgControl is a vec in private memory,if is null msgHdr.msgControl = ptr::null::<u8>() as u64;
            if !new_msghdr.msgControl == core::ptr::null::<u8>() as u64 {
                msgControlLen = new_msghdr.msgControlLen;
                hasControl = true;
                new_control_buff = unsafe {
                    GLOBAL_ALLOCATOR.AllocSharedBuf(new_msghdr.msgControlLen as usize, 0x8)
                };
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        new_msghdr.msgControl as *const u8,
                        new_control_buff,
                        msgControlLen,
                    );
                }
                new_msghdr.msgControl = new_control_buff as u64;
            }

            let mut new_iov = Box::new_in(IoVec::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let private_iov = unsafe { *(private_msghdr.iov as *const IoVec) };
            *new_iov = private_iov;
            new_msghdr.iov = &mut *new_iov as *const _ as u64;

            let mut msg = Box::new_in(
                Msg::HostUnixRecvMsg(HostUnixRecvMsg {
                    fd: fd,
                    msghdr: new_msghdr_ptr as u64,
                    flags: flags,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = Self::Call(&mut msg, false) as i64;

            if hasName {
                let updated_len = new_msghdr.nameLen;
                private_msghdr.nameLen = updated_len;
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        new_name_buff as *const u8,
                        private_msghdr.msgName as *mut u8,
                        nameLen as usize,
                    )
                };
                unsafe {
                    GLOBAL_ALLOCATOR.DeallocShareBuf(
                        new_name_buff as *mut u8,
                        nameLen as usize,
                        0x8,
                    );
                }
            }
            if hasControl {
                let updated_len = new_msghdr.msgControlLen;
                private_msghdr.msgControlLen = updated_len;
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        new_control_buff as *const u8,
                        private_msghdr.msgControlLen as *mut u8,
                        msgControlLen,
                    )
                };
                unsafe {
                    GLOBAL_ALLOCATOR.DeallocShareBuf(
                        new_control_buff as *mut u8,
                        msgControlLen,
                        0x8,
                    );
                }
            }

            private_msghdr.msgFlags = new_msghdr.msgFlags;
            return ret;
        } else {
            let mut msg = Msg::HostUnixRecvMsg(HostUnixRecvMsg {
                fd: fd,
                msghdr: msghdr,
                flags: flags,
            });

            let ret = Self::Call(&mut msg, false) as i64;
            return ret;
        }
    }

    pub fn TsotRecvMsg(msgAddr: u64) -> i64 {
        if is_cc_active() {
            let mut new_msg = Box::new_in(TsotMessage::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let new_msg_ptr = &mut *new_msg as *mut _;
            let private_msg = unsafe { &mut *(msgAddr as *mut TsotMessage) };
            let mut msg = Box::new_in(
                Msg::TsotRecvMsg(TsotRecvMsg {
                    msgAddr: new_msg_ptr as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            // TsotRecvMsg will be called in uring async process, must use HCall
            let ret = Self::HCall(&mut msg, false) as i64;
            *private_msg = *new_msg;
            return ret;
        } else {
            let mut msg = Msg::TsotRecvMsg(TsotRecvMsg { msgAddr: msgAddr });

            // TsotRecvMsg will be called in uring async process, must use HCall
            let ret = Self::HCall(&mut msg, false) as i64;
            return ret;
        }
    }

    pub fn TsotSendMsg(msgAddr: u64) -> i64 {
        if is_cc_active() {
            let mut new_msg = Box::new_in(TsotMessage::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let new_msg_ptr = &mut *new_msg as *mut _;
            let private_msg = unsafe { &mut *(msgAddr as *mut TsotMessage) };
            *new_msg = *private_msg;

            let mut msg = Box::new_in(
                Msg::TsotSendMsg(TsotSendMsg {
                    msgAddr: new_msg_ptr as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            // TsotSendMsg might be called in uring async process, must use HCall
            let ret = Self::HCall(&mut msg, false) as i64;
            *private_msg = *new_msg;
            return ret;
        } else {
            let mut msg = Msg::TsotSendMsg(TsotSendMsg { msgAddr: msgAddr });

            // TsotSendMsg might be called in uring async process, must use HCall
            let ret = Self::HCall(&mut msg, false) as i64;
            return ret;
        }
    }

    pub fn CreateAt(
        dirfd: i32,
        pathName: u64,
        flags: i32,
        mode: i32,
        uid: u32,
        gid: u32,
        fstatAddr: u64,
    ) -> i64 {
        if is_cc_active() {
            let mut new_buff = Box::new_in(LibcStat::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let new_buff_ptr = &mut *new_buff as *mut _;
            let private_buff = unsafe { &mut *(fstatAddr as *mut LibcStat) };
            let mut msg = Box::new_in(
                Msg::CreateAt(CreateAt {
                    dirfd,
                    pathName,
                    flags,
                    mode,
                    uid,
                    gid,
                    fstatAddr: new_buff_ptr as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::HCall(&mut msg, false) as i64;
            *private_buff = *new_buff;
            return ret;
        } else {
            let mut msg = Msg::CreateAt(CreateAt {
                dirfd,
                pathName,
                flags,
                mode,
                uid,
                gid,
                fstatAddr,
            });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn SchedGetAffinity(pid: i32, cpuSetSize: u64, mask: u64) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::SchedGetAffinity(SchedGetAffinity {
                    pid,
                    cpuSetSize,
                    mask,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::SchedGetAffinity(SchedGetAffinity {
                pid,
                cpuSetSize,
                mask,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Fchdir(fd: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(Msg::Fchdir(Fchdir { fd }), GUEST_HOST_SHARED_ALLOCATOR);

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Fchdir(Fchdir { fd });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Fadvise(fd: i32, offset: u64, len: u64, advice: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::Fadvise(Fadvise {
                    fd,
                    offset,
                    len,
                    advice,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Fadvise(Fadvise {
                fd,
                offset,
                len,
                advice,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Mlock2(addr: u64, len: u64, flags: u32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::Mlock2(Mlock2 { addr, len, flags }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Mlock2(Mlock2 { addr, len, flags });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn MUnlock(addr: u64, len: u64) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::MUnlock(MUnlock { addr, len }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::MUnlock(MUnlock { addr, len });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn NonBlockingPoll(fd: i32, mask: EventMask) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::NonBlockingPoll(NonBlockingPoll { fd, mask }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            //return HostSpace::Call(&mut msg, false) as i64;
            let ret = Self::HCall(&mut msg, false) as i64;
            //error!("NonBlockingPoll2 fd is {} ret is {}", fd, ret);

            return ret;
        } else {
            let mut msg = Msg::NonBlockingPoll(NonBlockingPoll { fd, mask });

            //return HostSpace::Call(&mut msg, false) as i64;
            let ret = Self::HCall(&mut msg, false) as i64;
            //error!("NonBlockingPoll2 fd is {} ret is {}", fd, ret);

            return ret;
        }
    }

    pub fn HostEpollWaitProcess() -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::HostEpollWaitProcess(HostEpollWaitProcess {}),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = Self::HCall(&mut msg, false) as i64;
            return ret;
        } else {
            let mut msg = Msg::HostEpollWaitProcess(HostEpollWaitProcess {});

            let ret = Self::HCall(&mut msg, false) as i64;
            return ret;
        }
    }

    pub fn VcpuWait() -> TaskId {
        if is_cc_active() {
            let mut next = Box::new_in(TaskId::New(0, 0), GUEST_HOST_SHARED_ALLOCATOR);
            let next_ptr = &mut *next as *mut _;
            HyperCall64(HYPERCALL_VCPU_WAIT, 0, 0, next_ptr as u64, 0);
            assert!(next.PrivateTaskAddr() != 0);
            assert!(next.SharedTaskAddr() != 0);
            return *next;
        } else {
            let mut next: TaskId = TaskId::New(0, 0);
            HyperCall64(HYPERCALL_VCPU_WAIT, 0, 0, &mut next as *mut _ as u64, 0);
            assert!(next.PrivateTaskAddr() != 0);
            assert!(next.SharedTaskAddr() != 0);
            return next;
        }
    }
    
    pub fn NewTmpfsFile(typ: TmpfsFileType, addr: u64) -> i64 {
        if is_cc_active() {
            let mut new_buff = Box::new_in(LibcStat::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let new_buff_ptr = &mut *new_buff as *mut _;
            let private_buff = unsafe { &mut *(addr as *mut LibcStat) };
            let mut msg = Box::new_in(
                Msg::NewTmpfsFile(NewTmpfsFile {
                    typ,
                    addr: new_buff_ptr as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;
            *private_buff = *new_buff;
            return ret;
        } else {
            let mut msg = Msg::NewTmpfsFile(NewTmpfsFile { typ, addr });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Chown(pathname: u64, owner: u32, group: u32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::Chown(Chown {
                    pathname,
                    owner,
                    group,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Chown(Chown {
                pathname,
                owner,
                group,
            });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn FChown(fd: i32, owner: u32, group: u32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::FChown(FChown { fd, owner, group }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::FChown(FChown { fd, owner, group });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Chmod(pathname: u64, mode: u32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::Chmod(Chmod { pathname, mode }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Chmod(Chmod { pathname, mode });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn Fchmod(fd: i32, mode: u32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::Fchmod(Fchmod { fd, mode }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Fchmod(Fchmod { fd, mode });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn LinkAt(olddirfd: i32, oldpath: u64, newdirfd: i32, newpath: u64, flags: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::LinkAt(LinkAt {
                    olddirfd,
                    oldpath,
                    newdirfd,
                    newpath,
                    flags,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::LinkAt(LinkAt {
                olddirfd,
                oldpath,
                newdirfd,
                newpath,
                flags,
            });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn SymLinkAt(oldpath: u64, newdirfd: i32, newpath: u64) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::SymLinkAt(SymLinkAt {
                    oldpath,
                    newdirfd,
                    newpath,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::SymLinkAt(SymLinkAt {
                oldpath,
                newdirfd,
                newpath,
            });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn ReadControlMsg(fd: i32, addr: u64) -> i64 {
        if is_cc_active() {
            let mut new_buff = Box::new_in(ControlMsg::default(), GUEST_HOST_SHARED_ALLOCATOR);
            let new_buff_ptr = &mut *new_buff as *mut _;
            let private_buff = unsafe { &mut *(addr as *mut ControlMsg) };
            let mut msg = Box::new_in(
                Msg::ReadControlMsg(ReadControlMsg {
                    fd,
                    addr: new_buff_ptr as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;
            *private_buff = *new_buff;
            return ret;
        } else {
            let mut msg = Msg::ReadControlMsg(ReadControlMsg { fd, addr });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn WriteControlMsgResp(fd: i32, addr: u64, len: usize, close: bool) -> i64 {
        if is_cc_active() {
            let new_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len, 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(addr as *const u8, new_buff as *mut u8, len);
            }
            let mut msg = Box::new_in(
                Msg::WriteControlMsgResp(WriteControlMsgResp {
                    fd: fd,
                    addr: new_buff as u64,
                    len: len,
                    close: close,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::HCall(&mut msg, false) as i64;
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_buff, len, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::WriteControlMsgResp(WriteControlMsgResp {
                fd: fd,
                addr: addr,
                len: len,
                close: close,
            });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn UpdateWaitInfo(fd: i32, waitinfo: FdWaitInfo) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::UpdateWaitInfo(UpdateWaitInfo {
                    fd: fd,
                    waitinfo: waitinfo,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::HCall(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::UpdateWaitInfo(UpdateWaitInfo {
                fd: fd,
                waitinfo: waitinfo,
            });

            return HostSpace::HCall(&mut msg, false) as i64;
        }
    }

    pub fn Futimens(fd: i32, times: u64) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::Futimens(Futimens { fd, times }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            return HostSpace::Call(&mut msg, false) as i64;
        } else {
            let mut msg = Msg::Futimens(Futimens { fd, times });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn GetStdfds(addr: u64) -> i64 {
        if is_cc_active() {
            let len = size_of::<i32>() * 3;
            let new_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len, 0x8) };
            unsafe {
                core::ptr::copy_nonoverlapping(addr as *const u8, new_buff as *mut u8, len);
            }
            let mut msg = Box::new_in(
                Msg::GetStdfds(GetStdfds {
                    addr: new_buff as u64,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let ret = HostSpace::Call(&mut msg, false) as i64;
            unsafe {
                core::ptr::copy_nonoverlapping(new_buff as *const u8, addr as *mut u8, len);
            }
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(new_buff, len, 0x8);
            }
            return ret;
        } else {
            let mut msg = Msg::GetStdfds(GetStdfds { addr });

            return HostSpace::Call(&mut msg, false) as i64;
        }
    }

    pub fn MMapFile(len: u64, fd: i32, offset: u64, prot: i32) -> i64 {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::MMapFile(MMapFile {
                    len,
                    fd,
                    offset,
                    prot,
                }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            let res = HostSpace::HCall(&mut msg, true) as i64;
            return res;
        } else {
            let mut msg = Msg::MMapFile(MMapFile {
                len,
                fd,
                offset,
                prot,
            });

            let res = HostSpace::HCall(&mut msg, true) as i64;
            return res;
        }
    }

    pub fn MUnmap(addr: u64, len: u64) {
        if is_cc_active() {
            let mut msg = Box::new_in(
                Msg::MUnmap(qmsg::qcall::MUnmap { addr, len }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            HostSpace::HCall(&mut msg, true);
        } else {
            let mut msg = Msg::MUnmap(qmsg::qcall::MUnmap { addr, len });

            HostSpace::HCall(&mut msg, true);
        }
    }

    pub fn EventfdWriteAsync(fd: i32) {
        if is_cc_active() {
            let msg = Box::new_in(
                HostOutputMsg::EventfdWriteAsync(EventfdWriteAsync { fd }),
                GUEST_HOST_SHARED_ALLOCATOR,
            );

            super::SHARESPACE.AQCall(&*msg);
        } else {
            let msg = HostOutputMsg::EventfdWriteAsync(EventfdWriteAsync { fd });

            super::SHARESPACE.AQCall(&msg);
        }
    }

    pub fn SyncPrint(level: DebugLevel, str: &str) {
        if is_cc_active() {
            //copy the &str to shared buffer
            let bytes = str.as_bytes();
            let len: usize = bytes.len();
            let new_str_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len, 0x8) };
            let dest_ptr: *mut u8 = new_str_ptr;
            let src_ptr: *const u8 = bytes.as_ptr();
            unsafe {
                core::ptr::copy_nonoverlapping(src_ptr, dest_ptr, len);
            }
            let new_str = unsafe {
                alloc::str::from_utf8_unchecked(core::slice::from_raw_parts(dest_ptr, len))
            };

            let mut msg = Box::new_in(
                Print {
                    level: level,
                    str: new_str,
                },
                GUEST_HOST_SHARED_ALLOCATOR,
            );
            let msg_ptr = &mut *msg as *mut _;
            HyperCall64(HYPERCALL_PRINT, msg_ptr as u64, 0, 0, 0);
            //Here need to drop mannually, otherwise the compiler will drop ealier, causing error
            drop(msg);
            unsafe {
                GLOBAL_ALLOCATOR.DeallocShareBuf(dest_ptr, len, 0x8);
            }
        } else {
            let msg = Print { level, str };
            HyperCall64(HYPERCALL_PRINT, &msg as *const _ as u64, 0, 0, 0);
        }
    }

    pub fn Kprint(str: &str) {
        let bytes = str.as_bytes();
        let trigger = super::SHARESPACE.Log(bytes);
        if trigger {
            super::IOURING.LogFlush();
        }
    }

    #[inline(always)]
    pub fn KernelMsg(id: u64, val1: u64, val2: u64, val3: u64) {
        HyperCall64(HYPERCALL_MSG, id, val1, val2, val3)
    }

    pub fn KernelOOM(size: u64, alignment: u64) {
        HyperCall64(HYPERCALL_OOM, size, alignment, 0, 0)
    }

    pub fn KernelGetTime(clockId: i32) -> Result<i64> {
        if is_cc_active() {
            let size = size_of::<GetTimeCall>();
            let call_ptr =
                unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size, 0x8) as *mut GetTimeCall };
            let call = unsafe { &mut *call_ptr };
            call.clockId = clockId;
            call.res = 0;

            let addr = call_ptr as *const _ as u64;
            HyperCall64(HYPERCALL_GETTIME, addr, 0, 0, 0);

            use self::common::*;

            if call.res < 0 {
                return Err(Error::SysError(-call.res as i32));
            }

            let ret = call.res;
            unsafe { GLOBAL_ALLOCATOR.DeallocShareBuf(call_ptr as *mut u8, size, 0x8) };
            return Ok(ret);
        } else {
            let call = GetTimeCall {
                clockId,
                ..Default::default()
            };

            let addr = &call as *const _ as u64;
            HyperCall64(HYPERCALL_GETTIME, addr, 0, 0, 0);

            use self::common::*;

            if call.res < 0 {
                return Err(Error::SysError(-call.res as i32));
            }

            return Ok(call.res);
        }
    }

    pub fn KernelVcpuFreq() -> i64 {
        if is_cc_active() {
            let size = size_of::<VcpuFeq>();
            let call_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size, 0x8) as *mut VcpuFeq };
            let call = unsafe { &mut *call_ptr };
            call.res = 0;

            let addr = call_ptr as *const _ as u64;
            HyperCall64(HYPERCALL_VCPU_FREQ, addr, 0, 0, 0);

            let ret = call.res;
            unsafe { GLOBAL_ALLOCATOR.DeallocShareBuf(call_ptr as *mut u8, size, 0x8) };
            return ret;
        } else {
            let call = VcpuFeq::default();

            let addr = &call as *const _ as u64;
            HyperCall64(HYPERCALL_VCPU_FREQ, addr, 0, 0, 0);

            return call.res;
        }
    }

    pub fn VcpuYield() {
        HyperCall64(HYPERCALL_VCPU_YIELD, 0, 0, 0, 0);
    }

    #[inline]
    pub fn VcpuDebug() {
        HyperCall64(HYPERCALL_VCPU_DEBUG, 0, 0, 0, 0);
    }

    pub fn VcpuPrint() {
        HyperCall64(HYPERCALL_VCPU_PRINT, 0, 0, 0, 0);
    }
}
