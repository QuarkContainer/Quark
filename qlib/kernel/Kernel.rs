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

use super::super::tsot_msg::*;
use super::super::common::*;
use super::super::config::*;
use super::super::linux_def::*;
use super::super::qmsg;
use super::super::qmsg::*;
use super::super::socket_buf::*;
use super::super::*;
use super::super::loader::*;
use crate::kernel_def::HyperCall64;
use crate::qlib::control_msg::ControlMsg;
use crate::qlib::nvproxy::frontend_type::RMAPIVersion;
use crate::qlib::proxy::*;
use crate::GLOBAL_ALLOCATOR;

extern "C" {
    pub fn rdtsc() -> i64;
}

pub struct HostSpace {}

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
    
    //Here exists memory leak, vec in shared struct cannot be dealloc
    pub fn LoadProcessKernel(processAddr: u64) -> i64 {
        let process_size = size_of::<Process>();
        let process_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(process_size,0x8) as *mut Process };
        let mut msg = Msg::LoadProcessKernel(LoadProcessKernel {
            processAddr: process_ptr as u64,
        });
        let ret = HostSpace::Call(&mut msg, false) as i64;
        let private_process = unsafe{&mut *(processAddr as *mut Process)};
        private_process.clone_from_shared(process_ptr);
        //info!("###Cloned Process:{:#?}\n",private_process);
        //Here could have memory leak since 
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(process_ptr as *mut u8, process_size, 0x8);}
        return ret;
    }
    
    /* 
    pub fn LoadProcessKernel(processAddr: u64) -> i64 {
        let mut msg = Msg::LoadProcessKernel(LoadProcessKernel {
            processAddr: processAddr,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }
    */

    pub fn CreateMemfd(len: i64, flags: u32) -> i64 {
        let mut msg = Msg::CreateMemfd(CreateMemfd {
            len: len,
            flags: flags,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Fallocate(fd: i32, mode: i32, offset: i64, len: i64) -> i64 {
        let mut msg = Msg::Fallocate(Fallocate {
            fd,
            mode,
            offset,
            len,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }
    
    pub fn Sysinfo(addr: u64) -> i64 {
        let info_size = size_of::<LibcSysinfo>();
        let info_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(info_size,0x8) as *mut LibcSysinfo };
        let mut msg = Msg::Sysinfo(Sysinfo {
            addr: info_ptr as u64,
        });
        
        let ret = HostSpace::Call(&mut msg, false) as i64;
        unsafe{
            *(addr as *mut LibcSysinfo) = *info_ptr;
        }
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(info_ptr as *mut u8, info_size, 0x8);}
        return ret;
    }

    pub fn EventfdWrite(fd: i32) -> i64 {
        let mut msg = Msg::EventfdWrite(EventfdWrite { fd });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn RenameAt(olddirfd: i32, oldpath: u64, newdirfd: i32, newpath: u64) -> i64 {
        let mut msg = Msg::RenameAt(RenameAt {
            olddirfd,
            oldpath,
            newdirfd,
            newpath,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn HostMemoryBarrier() -> i64 {
        let mut msg = Msg::HostMemoryBarrier(HostMemoryBarrier {});

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Ftruncate(fd: i32, len: i64) -> i64 {
        let mut msg = Msg::Ftruncate(Ftruncate { fd, len });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Rdtsc() -> i64 {
        let mut msg = Msg::Rdtsc(Rdtsc {});

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn SetTscOffset(offset: i64) -> i64 {
        let mut msg = Msg::SetTscOffset(SetTscOffset { offset: offset });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn TlbShootdown(vcpuMask: u64) -> i64 {
        let mut msg = Msg::TlbShootdown(TlbShootdown { vcpuMask: vcpuMask });

        return HostSpace::HCall(&mut msg, false) as i64;
    }
    
    pub fn IORead(fd: i32, iovs: u64, iovcnt: i32) -> i64 {
        let iovs_size = (size_of::<IoVec>())*(iovcnt as usize);
        let iovs_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(iovs_size,0x8) as *mut IoVec};
        unsafe { core::ptr::copy_nonoverlapping(iovs as *const u8, iovs_ptr as *mut u8, iovs_size);}

        let mut msg = Msg::IORead(IORead { fd, iovs: iovs_ptr as u64, iovcnt });

        let ret = HostSpace::Call(&mut msg, false) as i64;

        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(iovs_ptr as *mut u8, iovs_size, 0x8);}
        return ret;
    }

    pub fn IOTTYRead(fd: i32, iovs: u64, iovcnt: i32) -> i64 {
        let iovs_size = (size_of::<IoVec>())*(iovcnt as usize);
        let iovs_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(iovs_size,0x8) as *mut IoVec};
        unsafe { core::ptr::copy_nonoverlapping(iovs as *const u8, iovs_ptr as *mut u8, iovs_size);}

        let mut msg = Msg::IOTTYRead(IOTTYRead { fd, iovs: iovs_ptr as u64, iovcnt });

        let ret = HostSpace::Call(&mut msg, false) as i64;

        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(iovs_ptr as *mut u8, iovs_size, 0x8);}
        return ret;
        
    }

    pub fn IOWrite(fd: i32, iovs: u64, iovcnt: i32) -> i64 {
        let iovs_size = (size_of::<IoVec>())*(iovcnt as usize);
        let iovs_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(iovs_size,0x8) as *mut IoVec};
        unsafe { core::ptr::copy_nonoverlapping(iovs as *const u8, iovs_ptr as *mut u8, iovs_size);}

        let mut msg = Msg::IOWrite(IOWrite { fd, iovs: iovs_ptr as u64, iovcnt });

        let ret = HostSpace::Call(&mut msg, false) as i64;

        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(iovs_ptr as *mut u8, iovs_size, 0x8);}
        return ret;
    }

    pub fn IOReadAt(fd: i32, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let iovs_size = (size_of::<IoVec>())*(iovcnt as usize);
        let iovs_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(iovs_size,0x8) as *mut IoVec};
        unsafe { core::ptr::copy_nonoverlapping(iovs as *const u8, iovs_ptr as *mut u8, iovs_size);}

        let mut msg = Msg::IOReadAt(IOReadAt {
            fd,
            iovs: iovs_ptr as u64,
            iovcnt,
            offset,
        });

        let ret = HostSpace::Call(&mut msg, false) as i64;

        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(iovs_ptr as *mut u8, iovs_size, 0x8);}
        return ret;
    }

    pub fn IOWriteAt(fd: i32, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let iovs_size = (size_of::<IoVec>())*(iovcnt as usize);
        let iovs_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(iovs_size,0x8) as *mut IoVec};
        unsafe { core::ptr::copy_nonoverlapping(iovs as *const u8, iovs_ptr as *mut u8, iovs_size);}

        let mut msg = Msg::IOWriteAt(IOWriteAt {
            fd,
            iovs: iovs_ptr as u64,
            iovcnt,
            offset,
        });

        let ret = HostSpace::Call(&mut msg, false) as i64;

        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(iovs_ptr as *mut u8, iovs_size, 0x8);}
        return ret;
    }

    pub fn IOAppend(fd: i32, iovs: u64, iovcnt: i32) -> (i64, i64) {
        let iovs_size = (size_of::<IoVec>())*(iovcnt as usize);
        let iovs_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(iovs_size,0x8) as *mut IoVec};
        unsafe { core::ptr::copy_nonoverlapping(iovs as *const u8, iovs_ptr as *mut u8, iovs_size);}

        let file_len_size = size_of::<i64>();
        let file_len_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(file_len_size,0x8) as *mut i64};
        unsafe{*file_len_ptr = 0;}

        let mut msg = Msg::IOAppend(IOAppend {
            fd,
            iovs: iovs_ptr as u64,
            iovcnt,
            fileLenAddr: file_len_ptr as u64,
        });

        let ret = HostSpace::Call(&mut msg, false) as i64;
        if ret < 0 {
            return (ret, 0);
        }

        let fileLen = unsafe{*file_len_ptr};
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(iovs_ptr as *mut u8, iovs_size, 0x8);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(file_len_ptr as *mut u8, file_len_size, 0x8);}
        return (ret, fileLen);
    }

    pub fn IOAccept(fd: i32, addr: u64, addrlen: u64) -> i64 {
        let socket_size = size_of::<TcpSockAddr>();
        let socket_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(socket_size,0x8) as *mut TcpSockAddr };
        unsafe{
            (*socket_ptr).data = [0; UNIX_PATH_MAX + 2];
        }
        let len_size = size_of::<u32>();
        let len_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len_size,0x8) as *mut u32 };
        unsafe{
            (*len_ptr) = (*socket_ptr).data.len() as _;
        }

        let mut msg = Msg::IOAccept(IOAccept { fd, addr: (socket_ptr as u64), addrlen: (len_ptr as u64) });
        
        let ret = HostSpace::HCall(&mut msg, false) as i64;
        let mut socket = unsafe{ &mut *(addr as *mut TcpSockAddr)} ;
        socket.data = unsafe {(*socket_ptr).data};
        let len = addrlen as *mut u32;
        unsafe{
            *len = *len_ptr;
        }

        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(socket_ptr as *mut u8 ,size_of::<TcpSockAddr>(),0x8);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(len_ptr as *mut u8 ,size_of::<u32>(),0x8);}

        return ret;
    }

    pub fn IOConnect(fd: i32, addr: u64, addrlen: u32) -> i64 {
        let socket_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(addrlen as usize,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(addr as *const u8, socket_ptr, addrlen as usize);}

        let mut msg = Msg::IOConnect(IOConnect { fd, addr: socket_ptr as u64, addrlen });

        let ret = HostSpace::Call(&mut msg, false) as i64;
        
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(socket_ptr as *mut u8 ,addrlen as usize,0x8);}
        return ret;
    }

    pub fn IORecvMsg(fd: i32, msghdr: u64, flags: i32, blocking: bool) -> i64 {
        let mut hasName = false;
        let mut hasControl = false;
        let mut new_name_buff = core::ptr::null::<u8>() as *mut u8;
        let mut new_control_buff = core::ptr::null::<u8>() as *mut u8;
        let mut nameLen =0u32;
        let mut msgControlLen = 0usize;
        let new_msghdr_size = size_of::<MsgHdr>();
        let new_msghdr_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(new_msghdr_size,0x8) as *mut MsgHdr};
        unsafe { core::ptr::copy_nonoverlapping(msghdr as *const u8, new_msghdr_ptr as *mut u8, new_msghdr_size);}
        let new_msghdr = unsafe{&mut *new_msghdr_ptr};
        
        //new_msghdr.msgName is an array if new_msghdr.nameLen is not null;
        if new_msghdr.nameLen != 0 {
            nameLen = new_msghdr.nameLen;
            hasName = true;
            new_name_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(new_msghdr.nameLen as usize,0x8)};
            unsafe { core::ptr::copy_nonoverlapping(new_msghdr.msgName as *const u8, new_name_buff, nameLen as usize);}
            new_msghdr.msgName = new_name_buff as u64;
        }

        //new_msghdr.msgControl is a vec in private memory,if is null msgHdr.msgControl = ptr::null::<u8>() as u64;
        if !new_msghdr.msgControl == core::ptr::null::<u8>() as u64 {
            msgControlLen = new_msghdr.msgControlLen;
            hasControl = true;
            new_control_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(new_msghdr.msgControlLen as usize,0x8)};
            unsafe { core::ptr::copy_nonoverlapping(new_msghdr.msgControl as *const u8, new_control_buff, msgControlLen);}
            new_msghdr.msgControl = new_control_buff as u64;
        }

        let mut msg = Msg::IORecvMsg(IORecvMsg {
            fd,
            msghdr: new_msghdr_ptr as u64,
            flags,
            blocking,
        });

        let ret =  HostSpace::Call(&mut msg, false) as i64;

        let private_msghdr = unsafe {&mut *(msghdr as *mut MsgHdr)};
        if hasName {
            let updated_len = new_msghdr.nameLen;
            private_msghdr.nameLen = updated_len;
            unsafe{core::ptr::copy_nonoverlapping(new_name_buff as *const u8, private_msghdr.msgName as *mut u8, nameLen as usize)};
            unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_name_buff as *mut u8 ,nameLen as usize,0x8);}
        }
        if hasControl{
            let updated_len = new_msghdr.msgControlLen;
            private_msghdr.msgControlLen = updated_len;
            unsafe{core::ptr::copy_nonoverlapping(new_control_buff as *const u8, private_msghdr.msgControlLen as *mut u8, msgControlLen)};
            unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_control_buff as *mut u8 ,msgControlLen,0x8);}
        }

        private_msghdr.msgFlags = new_msghdr.msgFlags;
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_msghdr_ptr as *mut u8 ,new_msghdr_size,0x8);}
        return ret;
    }

    pub fn IORecvfrom(fd: i32, buf: u64, size: usize, flags: i32, addr: u64, len: u64) -> i64 {
        let mut hasName = false;
        let mut new_name_buff = core::ptr::null::<u8>() as *mut u8;
        let mut new_len_ptr = core::ptr::null::<u32>() as *mut u32;
        let mut nameLen =0u32;
        let addrlen = unsafe{&mut *(len as *mut u32)};
        if (*addrlen) != 0 {
            nameLen = *addrlen;
            hasName = true;
            new_name_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf((*addrlen) as usize,0x8)};
            new_len_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size_of::<u32>(),0x8) as *mut u32};
            unsafe { core::ptr::copy_nonoverlapping(addr as *const u8, new_name_buff, nameLen as usize);}
            unsafe{*new_len_ptr = nameLen;}
        }
        let mut msg = Msg::IORecvfrom(IORecvfrom {
            fd,
            buf,
            size,
            flags,
            addr: if hasName {new_name_buff as u64} else {addr},
            len: if hasName { new_len_ptr as u64} else {len},
        });
        
        let ret =  HostSpace::Call(&mut msg, false) as i64;

        if hasName {
            let updated_len = unsafe{*new_len_ptr};
            (*addrlen) = updated_len;
            unsafe{core::ptr::copy_nonoverlapping(new_name_buff as *const u8, addr as *mut u8, nameLen as usize)};
            unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_name_buff as *mut u8 ,nameLen as usize,0x8);}
            unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_len_ptr as *mut u8 ,size_of::<u32>(),0x8);}
        }
        return ret;
    }

    pub fn IOSendMsg(fd: i32, msghdr: u64, flags: i32, blocking: bool) -> i64 {
        let mut hasName = false;
        let mut hasControl = false;
        let mut new_name_buff = core::ptr::null::<u8>() as *mut u8;
        let mut new_control_buff = core::ptr::null::<u8>() as *mut u8;
        let mut nameLen =0u32;
        let mut msgControlLen = 0usize;
        let new_msghdr_size = size_of::<MsgHdr>();
        let new_msghdr_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(new_msghdr_size,0x8) as *mut MsgHdr};
        unsafe { core::ptr::copy_nonoverlapping(msghdr as *const u8, new_msghdr_ptr as *mut u8, new_msghdr_size);}
        let new_msghdr = unsafe{&mut *new_msghdr_ptr};

        if new_msghdr.nameLen != 0 {
            nameLen = new_msghdr.nameLen;
            hasName = true;
            new_name_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(new_msghdr.nameLen as usize,0x8)};
            unsafe { core::ptr::copy_nonoverlapping(new_msghdr.msgName as *const u8, new_name_buff, nameLen as usize);}
            new_msghdr.msgName = new_name_buff as u64;
        }

        if !new_msghdr.msgControl == core::ptr::null::<u8>() as u64 {
            msgControlLen = new_msghdr.msgControlLen;
            hasControl = true;
            new_control_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(new_msghdr.msgControlLen as usize,0x8)};
            unsafe { core::ptr::copy_nonoverlapping(new_msghdr.msgControl as *const u8, new_control_buff, msgControlLen);}
            new_msghdr.msgControl = new_control_buff as u64;
        }

        let mut msg = Msg::IOSendMsg(IOSendMsg {
            fd,
            msghdr: new_msghdr_ptr as u64,
            flags,
            blocking,
        });


        let ret =  HostSpace::Call(&mut msg, false) as i64;

        let private_msghdr = unsafe {&mut *(msghdr as *mut MsgHdr)};
        if hasName {
            let updated_len = new_msghdr.nameLen;
            private_msghdr.nameLen = updated_len;
            unsafe{core::ptr::copy_nonoverlapping(new_name_buff as *const u8, private_msghdr.msgName as *mut u8, nameLen as usize)};
            unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_name_buff as *mut u8 ,nameLen as usize,0x8);}
        }

        if hasControl{
            let updated_len = new_msghdr.msgControlLen;
            private_msghdr.msgControlLen = updated_len;
            unsafe{core::ptr::copy_nonoverlapping(new_control_buff as *const u8, private_msghdr.msgControlLen as *mut u8, msgControlLen)};
            unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_control_buff as *mut u8 ,msgControlLen,0x8);}
        }

        private_msghdr.msgFlags = new_msghdr.msgFlags;
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_msghdr_ptr as *mut u8 ,new_msghdr_size,0x8);}
        return ret;
    }

    pub fn IOSendto(fd: i32, buf: u64, size: usize, flags: i32, addr: u64, len: u32) -> i64 {
        let mut hasName = false;
        let mut new_name_buff = core::ptr::null::<u8>() as *mut u8;
        if len != 0 {
            hasName = true;
            new_name_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len as usize,0x8)};
            unsafe { core::ptr::copy_nonoverlapping(addr as *const u8, new_name_buff, len as usize);}
        }

        let mut msg = Msg::IOSendto(IOSendto {
            fd,
            buf,
            size,
            flags,
            addr:if hasName {new_name_buff as u64} else {addr},
            len,
        });
        
        let ret =HostSpace::Call(&mut msg, false) as i64;
        if hasName {
            unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_name_buff as *mut u8 ,len as usize,0x8);}
        }

        return ret;
    }

    pub fn GetTimeOfDay(tv: u64, tz: u64) -> i64 {
        let tv_size = size_of::<super::super::linux::time::Timeval>();
        let new_tv_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(tv_size,0x8) as *mut super::super::linux::time::Timeval};
        unsafe { core::ptr::copy_nonoverlapping(tv as *const u8, new_tv_ptr as *mut u8, tv_size);}

        let tz_size = size_of::<u32>()*2;
        let new_tz_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(tz_size,0x8) as *mut u32};
        unsafe { core::ptr::copy_nonoverlapping(tz as *const u8, new_tz_ptr as *mut u8, tz_size);}
        let mut msg = Msg::GetTimeOfDay(GetTimeOfDay { tv: new_tv_ptr as u64, tz: new_tz_ptr as u64 });

        let ret =HostSpace::Call(&mut msg, false) as i64;

        unsafe { core::ptr::copy_nonoverlapping(new_tv_ptr as *const u8, tv as *mut u8, tv_size);}
        unsafe { core::ptr::copy_nonoverlapping(new_tz_ptr as *const u8, tz as *mut u8, tz_size);}

        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_tv_ptr as *mut u8 ,tv_size ,0x8);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_tz_ptr as *mut u8 ,tz_size ,0x8);}
        return ret;
    }

    pub fn ReadLinkAt(dirfd: i32, path: u64, buf: u64, bufsize: u64) -> i64 {
        let new_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(bufsize as usize,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(buf as *const u8, new_buff as *mut u8, bufsize as usize);}
        let mut msg = Msg::ReadLinkAt(ReadLinkAt {
            dirfd,
            path,
            buf: new_buff as u64,
            bufsize,
        });

        let ret = HostSpace::Call(&mut msg, false) as i64;
        unsafe { core::ptr::copy_nonoverlapping(new_buff as *const u8, buf as *mut u8, bufsize as usize);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_buff as *mut u8 ,bufsize as usize,0x8);}

        return ret;
    }

    pub fn Fcntl(fd: i32, cmd: i32, arg: u64) -> i64 {
        //Here arg can be used directly, flock is implemented inside the kernel
        let mut msg = Msg::Fcntl(Fcntl { fd, cmd, arg });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn IoCtl(fd: i32, cmd: u64, argp: u64, argplen: usize) -> i64 {
        let new_argp = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(argplen,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(argp as *const u8, new_argp as *mut u8, argplen);}
        let mut msg = Msg::IoCtl(IoCtl { fd, cmd, argp:new_argp as u64 });
        let ret = HostSpace::Call(&mut msg, false) as i64;
        unsafe { core::ptr::copy_nonoverlapping(new_argp as *const u8, argp as *mut u8, argplen);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_argp as *mut u8 ,argplen,0x8);}
        return ret;
    }

    pub fn Fstatfs(fd: i32, buf: u64) -> i64 {
        let buff_size = size_of::<LibcStatfs>();
        let new_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(buff_size,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(buf as *const u8, new_buff as *mut u8, buff_size);}

        let mut msg = Msg::Fstatfs(Fstatfs { fd, buf: new_buff as u64 });

        let ret =  HostSpace::Call(&mut msg, false) as i64;

        unsafe{ core::ptr::copy_nonoverlapping(new_buff as *const u8, buf as *mut u8, buff_size);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_buff as *mut u8 ,buff_size,0x8);}
        return ret;
    }

    pub fn NewSocket(fd: i32) -> i64 {
        let mut msg = Msg::NewSocket(NewSocket { fd });

        return HostSpace::HCall(&mut msg, true) as i64;
    }

    //not used?
    pub fn FAccessAt(dirfd: i32, pathname: u64, mode: i32, flags: i32) -> i64 {
        let mut msg = Msg::FAccessAt(FAccessAt {
            dirfd,
            pathname,
            mode,
            flags,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Fstat(fd: i32, buff: u64) -> i64 {

        let new_buff_size = size_of::<LibcStat>();
        let new_buff_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(new_buff_size,0x8) as *mut LibcStat };
        unsafe { core::ptr::copy_nonoverlapping(buff as *const u8, new_buff_ptr as *mut u8, new_buff_size);}

        let mut msg = Msg::Fstat(Fstat { fd, buff: new_buff_ptr as u64});

        let ret = Self::HCall(&mut msg, false) as i64;

        unsafe{ core::ptr::copy_nonoverlapping(new_buff_ptr as *const u8, buff as *mut u8, new_buff_size);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_buff_ptr as *mut u8 ,size_of::<LibcStat>(),0x8);}
        return ret;
        //return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Fstatat(dirfd: i32, pathname: u64, buff: u64, flags: i32) -> i64 {

        let new_buff_size = size_of::<LibcStat>();
        let new_buff_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(new_buff_size,0x8) as *mut LibcStat };
        unsafe { core::ptr::copy_nonoverlapping(buff as *const u8, new_buff_ptr as *mut u8, new_buff_size);}

        let mut msg = Msg::Fstatat(Fstatat {
            dirfd,
            pathname,
            buff: new_buff_ptr as u64,
            flags,
        });

        let ret = HostSpace::Call(&mut msg, false) as i64;
        
        unsafe{ core::ptr::copy_nonoverlapping(new_buff_ptr as *const u8, buff as *mut u8, new_buff_size);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_buff_ptr as *mut u8 ,size_of::<LibcStat>(),0x8);}
        return ret;
    }

    pub fn Unlinkat(dirfd: i32, pathname: u64, flags: i32) -> i64 {
        let mut msg = Msg::Unlinkat(Unlinkat {
            dirfd,
            pathname,
            flags,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn Mkdirat(dirfd: i32, pathname: u64, mode_: u32, uid: u32, gid: u32) -> i64 {
        let mut msg = Msg::Mkdirat(Mkdirat {
            dirfd,
            pathname,
            mode_,
            uid,
            gid,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn Mkfifoat(dirfd: i32, name: u64, mode: u32, uid: u32, gid: u32) -> i64 {
        let mut msg = Msg::Mkfifoat(Mkfifoat {
            dirfd,
            name,
            mode,
            uid,
            gid,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn Proxy(cmd: ProxyCommand, parameters: ProxyParameters) -> i64 {
        let mut msg = Msg::Proxy(Proxy {
            cmd,
            parameters
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn SwapInPage(addr: u64) -> i64 {
        let mut msg = Msg::SwapInPage(SwapInPage { addr });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn SwapOut() -> i64 {
        let mut msg = Msg::SwapOut(SwapOut {});

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn SwapIn() -> i64 {
        let mut msg = Msg::SwapIn(SwapIn {});

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn SysSync() -> i64 {
        let mut msg = Msg::SysSync(SysSync {});

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn SyncFs(fd: i32) -> i64 {
        let mut msg = Msg::SyncFs(SyncFs { fd });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn SyncFileRange(fd: i32, offset: i64, nbytes: i64, flags: u32) -> i64 {
        let mut msg = Msg::SyncFileRange(SyncFileRange {
            fd,
            offset,
            nbytes,
            flags,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn FSync(fd: i32) -> i64 {
        let mut msg = Msg::FSync(FSync { fd });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn MSync(addr: u64, len: usize, flags: i32) -> i64 {
        let mut msg = Msg::MSync(MSync { addr, len, flags });
        
        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn Madvise(addr: u64, len: usize, advise: i32) -> i64 {
        let mut msg = Msg::MAdvise(MAdvise { addr, len, advise });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn FDataSync(fd: i32) -> i64 {
        let mut msg = Msg::FDataSync(FDataSync { fd });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Seek(fd: i32, offset: i64, whence: i32) -> i64 {
        let mut msg = Msg::Seek(Seek { fd, offset, whence });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn ReadDir(dirfd: i32, data: &mut [u8], reset: bool) -> i64 {
        let mut msg = Msg::ReadDir(ReadDir {
            dirfd,
            addr: &mut data[0] as *mut _ as u64,
            len: data.len(),
            reset: reset,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn FSetXattr(fd: i32, name: u64, value: u64, size: usize, flags: u32) -> i64 {
        let new_value_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(value as *const u8, new_value_ptr, size);}
        
        let mut msg = Msg::FSetXattr(FSetXattr {
            fd,
            name,
            value: new_value_ptr as u64,
            size,
            flags,
        });

        let ret = HostSpace::Call(&mut msg, false) as i64;
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_value_ptr,size,0x8);}

        return ret;
    }

    pub fn FGetXattr(fd: i32, name: u64, value: u64, size: usize) -> i64 {
        let new_value_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(value as *const u8, new_value_ptr, size);}

        let mut msg = Msg::FGetXattr(FGetXattr {
            fd,
            name,
            value: new_value_ptr as u64,
            size,
        });

        // FGetXattr has to be hcall as it will also be called by
        // inode::lookup --> OverlayHasWhiteout which might be called by create and hold a lock
        let ret = HostSpace::HCall(&mut msg, false) as i64;
        unsafe { core::ptr::copy_nonoverlapping(new_value_ptr, value as *mut u8, size);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_value_ptr,size,0x8);}

        return ret;
    }

    pub fn FRemoveXattr(fd: i32, name: u64) -> i64 {
        let mut msg = Msg::FRemoveXattr(FRemoveXattr { fd, name });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn FListXattr(fd: i32, list: u64, size: usize) -> i64 {
        let new_list_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(list as *const u8, new_list_addr, size);}

        let mut msg = Msg::FListXattr(FListXattr { fd, list: new_list_addr as u64, size });
        let ret = HostSpace::Call(&mut msg, false) as i64;

        unsafe { core::ptr::copy_nonoverlapping(new_list_addr, list as *mut u8, size);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_list_addr,size,0x8);}
        return ret;
    }

    pub fn GetRandom(buf: u64, len: u64, flags: u32) -> i64 {
        let new_buf_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len as usize,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(buf as *const u8, new_buf_addr, len as usize);}

        let mut msg = Msg::GetRandom(GetRandom { buf: new_buf_addr as u64, len, flags });

        let ret = HostSpace::Call(&mut msg, false) as i64;

        unsafe { core::ptr::copy_nonoverlapping(new_buf_addr, buf as *mut u8, len as usize);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_buf_addr,len as usize,0x8);}
        return ret;
    }

    pub fn Statm(statm: &mut StatmInfo) -> i64 {
        let size = size_of::<StatmInfo>();
        let info_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(statm as *const _ as u64 as *const u8, info_addr, size);}

        let mut msg = Msg::Statm(Statm {
            buf: info_addr as u64,
        });

        let ret = HostSpace::Call(&mut msg, false) as i64;
        unsafe { core::ptr::copy_nonoverlapping(info_addr, statm as *const _ as u64 as *mut u8, size);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(info_addr,size,0x8);}
        return ret;
    }

    pub fn Socket(domain: i32, type_: i32, protocol: i32) -> i64 {
        let mut msg = Msg::Socket(Socket {
            domain,
            type_,
            protocol,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn UnblockedSocket(domain: i32, type_: i32, protocol: i32) -> i64 {
        let mut msg = Msg::Socket(Socket {
            domain,
            type_,
            protocol,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn GetSockName(sockfd: i32, addr: u64, addrlen: u64) -> i64 {
        let len_size = size_of::<i32>();
        let len = unsafe{*(addrlen as *const i32)};
        let new_len_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len_size,0x8)};
        let new_buff_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len as usize,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(addr as *const u8, new_buff_addr, len as usize);}
        unsafe { core::ptr::copy_nonoverlapping(addrlen as *const u8, new_len_addr, len_size);}
        let mut msg = Msg::GetSockName(GetSockName {
            sockfd,
            addr: new_buff_addr as u64,
            addrlen: new_len_addr as u64,
        });

        let ret = HostSpace::Call(&mut msg, false) as i64;
        unsafe { core::ptr::copy_nonoverlapping(new_buff_addr, addr as *mut u8, len as usize);}
        unsafe { core::ptr::copy_nonoverlapping(new_len_addr, addrlen as *mut u8, len_size);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_buff_addr,len as usize,0x8);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_len_addr,len_size,0x8);}
        return ret;
    }

    pub fn GetPeerName(sockfd: i32, addr: u64, addrlen: u64) -> i64 {
        let len_size = size_of::<i32>();
        let len = unsafe{*(addrlen as *const i32)};
        let new_len_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len_size,0x8)};
        let new_buff_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len as usize,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(addr as *const u8, new_buff_addr, len as usize);}
        unsafe { core::ptr::copy_nonoverlapping(addrlen as *const u8, new_len_addr, len_size);}
        let mut msg = Msg::GetPeerName(GetPeerName {
            sockfd,
            addr: new_buff_addr as u64,
            addrlen: new_len_addr as u64,
        });

        let ret = HostSpace::Call(&mut msg, false) as i64;
        unsafe { core::ptr::copy_nonoverlapping(new_buff_addr, addr as *mut u8, len as usize);}
        unsafe { core::ptr::copy_nonoverlapping(new_len_addr, addrlen as *mut u8, len_size);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_buff_addr,len as usize,0x8);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_len_addr,len_size,0x8);}
        return ret;
    }

    pub fn GetSockOpt(sockfd: i32, level: i32, optname: i32, optval: u64, optlen: u64) -> i64 {
        let val_size = unsafe{*(optlen as *const i32)} as usize;
        let len_size = size_of::<i32>();
        let new_val_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(val_size,0x8)};
        let new_len_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len_size,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(optval as *const u8, new_val_addr, val_size);}
        unsafe { core::ptr::copy_nonoverlapping(optlen as *const u8, new_len_addr, len_size);}
        let mut msg = Msg::GetSockOpt(GetSockOpt {
            sockfd,
            level,
            optname,
            optval: new_val_addr as u64,
            optlen: new_len_addr as u64,
        });

        let ret = HostSpace::Call(&mut msg, false) as i64;
        unsafe { core::ptr::copy_nonoverlapping(new_len_addr, optlen as *mut u8, len_size);}
        unsafe { core::ptr::copy_nonoverlapping(new_val_addr, optval as *mut u8, val_size);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_len_addr,len_size,0x8);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_val_addr,val_size,0x8);}
        return ret;
    }

    pub fn SetSockOpt(sockfd: i32, level: i32, optname: i32, optval: u64, optlen: u32) -> i64 {
        let new_val_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(optlen as usize,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(optval as *const u8, new_val_addr, optlen as usize);}
        let mut msg = Msg::SetSockOpt(SetSockOpt {
            sockfd,
            level,
            optname,
            optval: new_val_addr as u64,
            optlen,
        });

        //return Self::HCall(&mut msg) as i64;
        let ret =  HostSpace::Call(&mut msg, false) as i64;
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_val_addr,optlen as usize,0x8);}
        return ret;
    }

    pub fn Bind(sockfd: i32, addr: u64, addrlen: u32, umask: u32) -> i64 {
        let new_addr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(addrlen as usize,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(addr as *const u8, new_addr, addrlen as usize);}
        let mut msg = Msg::IOBind(IOBind {
            sockfd,
            addr: new_addr as u64,
            addrlen,
            umask,
        });

        let ret = HostSpace::Call(&mut msg, false) as i64;
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_addr,addrlen as usize,0x8);}
        return ret;
    }

    pub fn Listen(sockfd: i32, backlog: i32, block: bool) -> i64 {
        let mut msg = Msg::IOListen(IOListen {
            sockfd,
            backlog,
            block,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }
    
    pub fn RDMAListen(sockfd: i32, backlog: i32, block: bool, acceptQueue: AcceptQueue) -> i64 {
        let mut msg = Msg::RDMAListen(RDMAListen {
            sockfd,
            backlog,
            block,
            acceptQueue,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn RDMANotify(sockfd: i32, typ: RDMANotifyType) -> i64 {
        let mut msg = Msg::RDMANotify(RDMANotify { sockfd, typ });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Shutdown(sockfd: i32, how: i32) -> i64 {
        let mut msg = Msg::IOShutdown(IOShutdown { sockfd, how });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn ExitVM(exitCode: i32) {
        HyperCall64(HYPERCALL_EXIT_VM, exitCode as u64, 0, 0, 0);
        //Self::AQCall(qmsg::HostOutputMsg::ExitVM(exitCode));
    }

    pub fn Panic(str: &str) {
        
        //copy the &str to shared buffer
        let bytes = str.as_bytes();
        let len  = bytes.len();
        let new_str_ptr =  unsafe{ GLOBAL_ALLOCATOR.AllocSharedBuf(len,0x8) };
        
        let dest_ptr: *mut u8 = new_str_ptr;
        let src_ptr: *const u8 = bytes.as_ptr();
        unsafe { core::ptr::copy_nonoverlapping(src_ptr, dest_ptr, len);}
        let new_str = unsafe {alloc::str::from_utf8_unchecked(core::slice::from_raw_parts(dest_ptr,len))}; 
        
        let size = size_of::<Print>();
        let msg_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size,0x8) as *mut Print };
        let mut msg = unsafe {&mut *msg_ptr};
        msg.level = DebugLevel::Error;
        msg.str = new_str;
        HyperCall64(HYPERCALL_PANIC, msg_ptr as u64, 0, 0, 0);
        
    }

    pub fn TryOpenWrite(dirfd: i32, oldfd: i32, name: u64) -> i64 {
        let mut msg = Msg::TryOpenWrite(TryOpenWrite {
            dirfd: dirfd,
            oldfd: oldfd,
            name: name,
        });

        let ret = Self::HCall(&mut msg, false) as i64;
        return ret;
    }

    pub fn TryOpenAt(dirfd: i32, name: u64, addr: u64, skiprw: bool) -> i64 {
        let tryopen_size = size_of::<TryOpenStruct>();
        let libcstat_size = size_of::<LibcStat>();
        let new_tryopen_ptr =  unsafe{ GLOBAL_ALLOCATOR.AllocSharedBuf(tryopen_size,0x8) };
        let new_libcstat = unsafe{ GLOBAL_ALLOCATOR.AllocSharedBuf(libcstat_size,0x8) };
        let new_tryopen = unsafe{&mut *(new_tryopen_ptr as *mut TryOpenStruct)};
        unsafe { core::ptr::copy_nonoverlapping(addr as *const u8, new_tryopen_ptr, tryopen_size);}
        let old_libcstat = new_tryopen.fstat as *const _ as *mut u8;
        unsafe { core::ptr::copy_nonoverlapping(old_libcstat, new_libcstat, libcstat_size);}
        new_tryopen.fstat = unsafe{&*(new_libcstat as *mut LibcStat)};

        let mut msg = Msg::TryOpenAt(TryOpenAt {
            dirfd: dirfd,
            name: name,
            addr: new_tryopen_ptr as u64,
            skiprw: skiprw,
        });

        let ret = Self::HCall(&mut msg, false) as i64;
        unsafe { core::ptr::copy_nonoverlapping(new_libcstat, old_libcstat, libcstat_size);}
        unsafe { core::ptr::copy_nonoverlapping(new_tryopen_ptr as *const u8, addr as *mut u8, tryopen_size);}
        let old_tryopen = unsafe{&mut *(addr as *mut TryOpenStruct)};
        old_tryopen.fstat = unsafe{&*(old_libcstat as *mut LibcStat)};
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_tryopen_ptr,tryopen_size,0x8);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_libcstat,libcstat_size,0x8);}
        return ret;
        //return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn OpenAt(dirfd: i32, name: u64, flags: i32, addr: u64) -> i64 {
        let tryopen_size = size_of::<TryOpenStruct>();
        let libcstat_size = size_of::<LibcStat>();
        let new_tryopen_ptr =  unsafe{ GLOBAL_ALLOCATOR.AllocSharedBuf(tryopen_size,0x8) };
        let new_libcstat = unsafe{ GLOBAL_ALLOCATOR.AllocSharedBuf(libcstat_size,0x8) };
        let new_tryopen = unsafe{&mut *(new_tryopen_ptr as *mut TryOpenStruct)};
        unsafe { core::ptr::copy_nonoverlapping(addr as *const u8, new_tryopen_ptr, tryopen_size);}
        let old_libcstat = new_tryopen.fstat as *const _ as *mut u8;
        unsafe { core::ptr::copy_nonoverlapping(old_libcstat, new_libcstat, libcstat_size);}
        new_tryopen.fstat = unsafe{&*(new_libcstat as *mut LibcStat)};

        let mut msg = Msg::OpenAt(OpenAt {
            dirfd: dirfd,
            name: name,
            flags: flags,
            addr: new_tryopen_ptr as u64,
        });

        let ret = Self::HCall(&mut msg, false) as i64;
        unsafe { core::ptr::copy_nonoverlapping(new_libcstat, old_libcstat, libcstat_size);}
        unsafe { core::ptr::copy_nonoverlapping(new_tryopen_ptr as *const u8, addr as *mut u8, tryopen_size);}
        let old_tryopen = unsafe{&mut *(addr as *mut TryOpenStruct)};
        old_tryopen.fstat = unsafe{&*(old_libcstat as *mut LibcStat)};
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_tryopen_ptr,tryopen_size,0x8);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_libcstat,libcstat_size,0x8);}
        return ret;
        //return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn OpenDevFile(dirfd: i32, name: u64, flags: i32) -> i64 {
        let mut msg = Msg::OpenDevFile(OpenDevFile {
            dirfd: dirfd,
            name: name,
            flags: flags,
        });

        let ret = Self::HCall(&mut msg, false) as i64;
        return ret;
        //return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn RemapGuestMemRanges(len: u64, addr: u64, count: usize) -> i64 {
        let range_size = size_of::<Range>();
        let new_addr =  unsafe{ GLOBAL_ALLOCATOR.AllocSharedBuf(range_size*count,0x8) };
        unsafe { core::ptr::copy_nonoverlapping(addr as *const u8, new_addr, range_size);}
        let mut msg = Msg::RemapGuestMemRanges(RemapGuestMemRanges {
            len: len,
            addr: new_addr as u64,
            count: count
        });

        let ret = Self::Call(&mut msg, false) as i64;
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_addr,range_size*count,0x8);}
        return ret;
    }

    pub fn UnmapGuestMemRange(start: u64, len: u64) -> i64 {
        let mut msg = Msg::UnmapGuestMemRange(UnmapGuestMemRange {
            start: start,
            len: len,
        });

        let ret = Self::Call(&mut msg, false) as i64;
        return ret;
    }

    pub fn NividiaDriverVersion(version: &RMAPIVersion) -> i64 {
        let version_size = size_of::<RMAPIVersion>();
        let new_version = unsafe{ GLOBAL_ALLOCATOR.AllocSharedBuf(version_size,0x8) };
        unsafe { core::ptr::copy_nonoverlapping(version as * const _ as *const u8 , new_version, version_size);}
        let mut msg = Msg::NividiaDriverVersion(NividiaDriverVersion {
            ioctlParamsAddr: new_version as u64
        });

        let ret = Self::Call(&mut msg, false) as i64;
        unsafe { core::ptr::copy_nonoverlapping(new_version, version as * const _ as *mut u8, version_size);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_version,version_size,0x8);}
        return ret;
    }

    pub fn NvidiaMMap(addr: u64, len: u64, prot: i32, flags: i32, fd: i32, offset: u64) -> i64 {
        let mut msg = Msg::NvidiaMMap(NvidiaMMap {
            addr: addr,
            len: len,
            prot: prot,
            flags: flags,
            fd: fd,
            offset: offset
        });

        let ret = Self::Call(&mut msg, false) as i64;
        return ret;
    }

    pub fn HostUnixConnect(type_: i32, addr: u64, len: usize) -> i64 {
        let new_addr = unsafe{ GLOBAL_ALLOCATOR.AllocSharedBuf(len,0x8) };
        unsafe { core::ptr::copy_nonoverlapping(addr as *const u8 , new_addr, len);}
        let mut msg = Msg::HostUnixConnect(HostUnixConnect {
            type_: type_,
            addr: new_addr as u64,
            len: len,
        });

        let ret = Self::Call(&mut msg, false) as i64;
        return ret;
    }

    pub fn HostUnixRecvMsg(fd: i32, msghdr: u64, flags: i32) -> i64 {
        let mut hasName = false;
        let mut hasControl = false;
        let mut new_name_buff = core::ptr::null::<u8>() as *mut u8;
        let mut new_control_buff = core::ptr::null::<u8>() as *mut u8;
        let mut nameLen =0u32;
        let mut msgControlLen = 0usize;
        let new_msghdr_size = size_of::<MsgHdr>();
        let new_msghdr_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(new_msghdr_size,0x8) as *mut MsgHdr};
        unsafe { core::ptr::copy_nonoverlapping(msghdr as *const u8, new_msghdr_ptr as *mut u8, new_msghdr_size);}
        let new_msghdr = unsafe{&mut *new_msghdr_ptr};
        
        //new_msghdr.msgName is an array if new_msghdr.nameLen is not null;
        if new_msghdr.nameLen != 0 {
            nameLen = new_msghdr.nameLen;
            hasName = true;
            new_name_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(new_msghdr.nameLen as usize,0x8)};
            unsafe { core::ptr::copy_nonoverlapping(new_msghdr.msgName as *const u8, new_name_buff, nameLen as usize);}
            new_msghdr.msgName = new_name_buff as u64;
        }

        //new_msghdr.msgControl is a vec in private memory,if is null msgHdr.msgControl = ptr::null::<u8>() as u64;
        if !new_msghdr.msgControl == core::ptr::null::<u8>() as u64 {
            msgControlLen = new_msghdr.msgControlLen;
            hasControl = true;
            new_control_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(new_msghdr.msgControlLen as usize,0x8)};
            unsafe { core::ptr::copy_nonoverlapping(new_msghdr.msgControl as *const u8, new_control_buff, msgControlLen);}
            new_msghdr.msgControl = new_control_buff as u64;
        }
        let mut msg = Msg::HostUnixRecvMsg(HostUnixRecvMsg {
            fd: fd,
            msghdr: new_msghdr_ptr as u64,
            flags: flags
        });

        let ret = Self::Call(&mut msg, false) as i64;

        let private_msghdr = unsafe {&mut *(msghdr as *mut MsgHdr)};
        if hasName {
            let updated_len = new_msghdr.nameLen;
            private_msghdr.nameLen = updated_len;
            unsafe{core::ptr::copy_nonoverlapping(new_name_buff as *const u8, private_msghdr.msgName as *mut u8, nameLen as usize)};
            unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_name_buff as *mut u8 ,nameLen as usize,0x8);}
        }
        if hasControl{
            let updated_len = new_msghdr.msgControlLen;
            private_msghdr.msgControlLen = updated_len;
            unsafe{core::ptr::copy_nonoverlapping(new_control_buff as *const u8, private_msghdr.msgControlLen as *mut u8, msgControlLen)};
            unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_control_buff as *mut u8 ,msgControlLen,0x8);}
        }

        private_msghdr.msgFlags = new_msghdr.msgFlags;
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_msghdr_ptr as *mut u8 ,new_msghdr_size,0x8);}

        return ret;
    }

    pub fn TsotRecvMsg(msgAddr: u64) -> i64 {
        let message_size = size_of::<TsotMessage>();
        let new_msg_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(message_size,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(msgAddr as *const u8, new_msg_ptr as *mut u8, message_size);}
        let mut msg = Msg::TsotRecvMsg(TsotRecvMsg {
            msgAddr: new_msg_ptr as u64,
        });

        // TsotRecvMsg will be called in uring async process, must use HCall
        let ret = Self::HCall(&mut msg, false) as i64;
        unsafe { core::ptr::copy_nonoverlapping(new_msg_ptr as *const u8, msgAddr as *mut u8, message_size);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_msg_ptr as *mut u8 ,message_size,0x8);}
        return ret;
    }

    pub fn TsotSendMsg(msgAddr: u64) -> i64 {
        let message_size = size_of::<TsotMessage>();
        let new_msg_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(message_size,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(msgAddr as *const u8, new_msg_ptr as *mut u8, message_size);}
        let mut msg = Msg::TsotSendMsg(TsotSendMsg {
            msgAddr: msgAddr,
        });

        // TsotSendMsg might be called in uring async process, must use HCall
        let ret = Self::HCall(&mut msg, false) as i64;
        unsafe { core::ptr::copy_nonoverlapping(new_msg_ptr as *const u8, msgAddr as *mut u8, message_size);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_msg_ptr as *mut u8 ,message_size,0x8);}
        return ret;
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
        let buff_size = size_of::<LibcStat>();
        let new_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(buff_size,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(fstatAddr as *const u8, new_buff as *mut u8, buff_size);}

        let mut msg = Msg::CreateAt(CreateAt {
            dirfd,
            pathName,
            flags,
            mode,
            uid,
            gid,
            fstatAddr: new_buff as u64,
        });

        let ret = HostSpace::HCall(&mut msg, false) as i64;
        unsafe { core::ptr::copy_nonoverlapping(new_buff as *const u8, fstatAddr as *mut u8, buff_size);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_buff, buff_size,0x8);}
        return ret;
    }

    pub fn SchedGetAffinity(pid: i32, cpuSetSize: u64, mask: u64) -> i64 {
        let mut msg = Msg::SchedGetAffinity(SchedGetAffinity {
            pid,
            cpuSetSize,
            mask,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Fchdir(fd: i32) -> i64 {
        let mut msg = Msg::Fchdir(Fchdir { fd });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Fadvise(fd: i32, offset: u64, len: u64, advice: i32) -> i64 {
        let mut msg = Msg::Fadvise(Fadvise {
            fd,
            offset,
            len,
            advice,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Mlock2(addr: u64, len: u64, flags: u32) -> i64 {
        let mut msg = Msg::Mlock2(Mlock2 { addr, len, flags });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn MUnlock(addr: u64, len: u64) -> i64 {
        let mut msg = Msg::MUnlock(MUnlock { addr, len });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn NonBlockingPoll(fd: i32, mask: EventMask) -> i64 {
        let mut msg = Msg::NonBlockingPoll(NonBlockingPoll { fd, mask });

        //return HostSpace::Call(&mut msg, false) as i64;
        let ret = Self::HCall(&mut msg, false) as i64;
        //error!("NonBlockingPoll2 fd is {} ret is {}", fd, ret);

        return ret;
    }

    pub fn HostEpollWaitProcess() -> i64 {
        let mut msg = Msg::HostEpollWaitProcess(HostEpollWaitProcess {});

        let ret = Self::HCall(&mut msg, false) as i64;
        return ret;
    }

    pub fn VcpuWait() -> i64 {
        let ret_ptr =  unsafe{ GLOBAL_ALLOCATOR.AllocSharedBuf(size_of::<i64>(),0x8) as *mut i64};
        HyperCall64(HYPERCALL_VCPU_WAIT, 0, 0, ret_ptr as u64, 0);
        let ret = unsafe{ *ret_ptr };
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(ret_ptr as *mut u8 ,size_of::<i64>(),0x8);}
        return ret as i64;
    }

    pub fn NewTmpfsFile(typ: TmpfsFileType, addr: u64) -> i64 {
        let buff_size = size_of::<LibcStat>();
        let new_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(buff_size,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(addr as *const u8, new_buff as *mut u8, buff_size);}
        let mut msg = Msg::NewTmpfsFile(NewTmpfsFile { typ, addr:new_buff as u64 });

        let ret = HostSpace::Call(&mut msg, false) as i64;
        unsafe { core::ptr::copy_nonoverlapping(new_buff as *const u8, addr as *mut u8, buff_size);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_buff, buff_size,0x8);}
        return ret;
    }
    
    pub fn Chown(pathname: u64, owner: u32, group: u32) -> i64 {
        let mut msg = Msg::Chown(Chown {
            pathname,
            owner,
            group,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn FChown(fd: i32, owner: u32, group: u32) -> i64 {
        let mut msg = Msg::FChown(FChown { fd, owner, group });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Chmod(pathname: u64, mode: u32) -> i64 {
        let mut msg = Msg::Chmod(Chmod { pathname, mode });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Fchmod(fd: i32, mode: u32) -> i64 {
        let mut msg = Msg::Fchmod(Fchmod { fd, mode });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn LinkAt(olddirfd: i32, oldpath: u64, newdirfd: i32, newpath: u64, flags: i32) -> i64 {
        let mut msg = Msg::LinkAt(LinkAt {
            olddirfd,
            oldpath,
            newdirfd,
            newpath,
            flags,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn SymLinkAt(oldpath: u64, newdirfd: i32, newpath: u64) -> i64 {
        let mut msg = Msg::SymLinkAt(SymLinkAt {
            oldpath,
            newdirfd,
            newpath,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn ReadControlMsg(fd: i32, addr: u64) -> i64 {
        let msg_size = size_of::<ControlMsg>();
        let new_buff = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(msg_size,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(addr as *const u8, new_buff as *mut u8, msg_size);}
        let mut msg = Msg::ReadControlMsg(ReadControlMsg { fd, addr: new_buff as u64});

        let ret = HostSpace::Call(&mut msg, false) as i64;
        unsafe { core::ptr::copy_nonoverlapping(new_buff as *const u8, addr as *mut u8, msg_size);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_buff, msg_size,0x8);}
        return ret;
    }   

    pub fn WriteControlMsgResp(fd: i32, addr: u64, len: usize, close: bool) -> i64 {
        let new_buff =  unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(addr as *const u8, new_buff as *mut u8, len);}
        let mut msg = Msg::WriteControlMsgResp(WriteControlMsgResp {
            fd: fd,
            addr: new_buff as u64,
            len: len,
            close: close,
        });

        let ret = HostSpace::HCall(&mut msg, false) as i64;
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_buff, len,0x8);}
        return ret;
    }

    //not used yet, use an arc as shared struct, cannot used in cvm
    pub fn UpdateWaitInfo(fd: i32, waitinfo: FdWaitInfo) -> i64 {
        let mut msg = Msg::UpdateWaitInfo(UpdateWaitInfo {
            fd: fd,
            waitinfo: waitinfo,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn Futimens(fd: i32, times: u64) -> i64 {
        let mut msg = Msg::Futimens(Futimens { fd, times });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn GetStdfds(addr: u64) -> i64 {
        let len = size_of::<i32>()*3;
        let new_buff =  unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(len,0x8)};
        unsafe { core::ptr::copy_nonoverlapping(addr as *const u8, new_buff as *mut u8, len);}
        let mut msg = Msg::GetStdfds(GetStdfds { addr:new_buff as u64 });

        let ret = HostSpace::Call(&mut msg, false) as i64;
        unsafe { core::ptr::copy_nonoverlapping(new_buff as *const u8, addr as *mut u8, len);}
        unsafe{ GLOBAL_ALLOCATOR.DeallocShareBuf(new_buff, len,0x8);}
        return ret;
    }

    pub fn MMapFile(len: u64, fd: i32, offset: u64, prot: i32) -> i64 {
        // assert!(
        //     len % MemoryDef::PMD_SIZE == 0,
        //     "offset is {:x}, len is {:x}",
        //     offset,
        //     len
        // );
        // assert!(
        //     offset % MemoryDef::PMD_SIZE == 0,
        //     "offset is {:x}, len is {:x}",
        //     offset,
        //     len
        // );
        let mut msg = Msg::MMapFile(MMapFile {
            len,
            fd,
            offset,
            prot,
        });

        let res = HostSpace::HCall(&mut msg, true) as i64;
        //assert!(res as u64 % MemoryDef::PMD_SIZE == 0, "res {:x}", res);
        return res;
    }

    pub fn MUnmap(addr: u64, len: u64) {
        // assert!(
        //     addr % MemoryDef::PMD_SIZE == 0,
        //     "addr is {:x}, len is {:x}",
        //     addr,
        //     len
        // );
        // assert!(
        //     len % MemoryDef::PMD_SIZE == 0,
        //     "addr is {:x}, len is {:x}",
        //     addr,
        //     len
        // );
        let mut msg = Msg::MUnmap(qmsg::qcall::MUnmap { addr, len });

        HostSpace::HCall(&mut msg, true);
    }

    pub fn EventfdWriteAsync(fd: i32) {
        let msg = HostOutputMsg::EventfdWriteAsync(EventfdWriteAsync { fd });

        super::SHARESPACE.AQCall(&msg);
    }

    pub fn SyncPrint(level: DebugLevel, str: &str) {
        //copy the &str to shared buffer
        let bytes = str.as_bytes();
        let len: usize  = bytes.len();
        let new_str_ptr =  unsafe{ GLOBAL_ALLOCATOR.AllocSharedBuf(len,0x8) };
        let dest_ptr: *mut u8 = new_str_ptr;
        let src_ptr: *const u8 = bytes.as_ptr();
        unsafe { core::ptr::copy_nonoverlapping(src_ptr, dest_ptr, len);}
        let new_str = unsafe {alloc::str::from_utf8_unchecked(core::slice::from_raw_parts(dest_ptr,len))}; 
        
        let size = size_of::<Print>();
        let msg_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size,0x8) as *mut Print };
        let mut msg = unsafe {&mut *msg_ptr};
        msg.level = level;
        msg.str = new_str;
        
        HyperCall64(HYPERCALL_PRINT, msg_ptr as u64, 0, 0, 0);

        unsafe{
            GLOBAL_ALLOCATOR.DeallocShareBuf(dest_ptr, len, 0x8);
            GLOBAL_ALLOCATOR.DeallocShareBuf(msg_ptr as *mut u8, size, 0x8);
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
        let size = size_of::<GetTimeCall>();
        let call_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size,0x8) as *mut GetTimeCall };
        let mut call = unsafe {&mut *call_ptr};
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
    }

    pub fn KernelVcpuFreq() -> i64 {
        let size = size_of::<VcpuFeq>();
        let call_ptr = unsafe { GLOBAL_ALLOCATOR.AllocSharedBuf(size,0x8) as *mut VcpuFeq };
        let mut call = unsafe {&mut *call_ptr};
        call.res=0;

        let addr = call_ptr as *const _ as u64;
        HyperCall64(HYPERCALL_VCPU_FREQ, addr, 0, 0, 0);

        let ret = call.res;
        unsafe { GLOBAL_ALLOCATOR.DeallocShareBuf(call_ptr as *mut u8, size, 0x8) };
        return ret;
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

pub fn GetSockOptI32(sockfd: i32, level: i32, optname: i32) -> Result<i32> {
    let mut val: i32 = 0;
    let len: i32 = 4;
    let res = HostSpace::GetSockOpt(
        sockfd,
        level,
        optname,
        &mut val as *mut i32 as u64,
        &len as *const i32 as u64,
    ) as i32;

    if res < 0 {
        return Err(Error::SysError(-res as i32));
    }

    return Ok(val);
}
