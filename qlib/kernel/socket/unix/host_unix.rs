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

use core::sync::atomic::{AtomicI64, Ordering};
use core::ptr;
use core::ops::Deref;

use alloc::slice;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use alloc::boxed::Box;

use crate::GUEST_HOST_SHARED_ALLOCATOR;
use crate::GuestHostSharedAllocator;
use crate::qlib::common::*;
use crate::qlib::kernel::Kernel;
use crate::qlib::kernel::fd::{IORead, IOWrite};
use crate::qlib::kernel::fs::file::{File, FileOps};
use crate::qlib::kernel::fs::filesystems::MountSourceFlags;
use crate::qlib::kernel::fs::flags::FileFlags;
use crate::qlib::kernel::fs::host::fs::WhitelistFileSystem;
use crate::qlib::kernel::fs::host::hostinodeop::HostInodeOp;
use crate::qlib::kernel::fs::host::util::Fstat;
use crate::qlib::kernel::fs::mount::MountSource;
use crate::qlib::kernel::guestfdnotifier::{NonBlockingPoll, UpdateFD};
use crate::qlib::kernel::kernel::fd_table::FDFlags;
use crate::qlib::kernel::kernel::time::Time;
use crate::qlib::kernel::kernel::waiter::{Queue, Waitable, WaitEntry};
use crate::qlib::kernel::socket::control::*;
use crate::qlib::kernel::socket::hostinet::socket::{HostIoctlIFReq, HostIoctlIFConf};
use crate::qlib::kernel::socket::socket::{NewHostfileDirent, HOST_FILE_DEVICE};
use crate::qlib::kernel::task::Task;
use crate::qlib::kernel::tcpip::tcpip::*;
use crate::qlib::linux_def::*;
use crate::qlib::mutex::QMutex;
use crate::qlib::kernel::socket::control::ControlMessageRights;

// pub fn NewHostUnixSocketFile(
//     task: &Task,
//     family: i32,
//     fd: i32,
//     stype: i32,
//     nonblock: bool
// ) -> Result<File> {
//     let dirent = NewSocketDirent(task, SOCKET_DEVICE.clone(), fd)?;
//     let inode = dirent.Inode();
//     let iops = inode.lock().InodeOp.clone();
//     let hostiops = iops.HostInodeOp().unwrap();
//     let s = HostUnixSocketOperations::New(
//         family, 
//         fd, 
//         stype, 
//         hostiops.Queue(), 
//         hostiops.clone()
//     )?;

//     let file = File::New(
//         &dirent,
//         &&FileFlags {
//             NonBlocking: nonblock,
//             Read: true,
//             Write: true,
//             ..Default::default()
//         },
//         s.into(),
//     );

//     GetKernel().sockets.AddSocket(&file);
//     return Ok(file);
// }

#[derive(Clone)]
pub struct HostUnixSocketOperations(Arc<HostUnixSocketOperationsIntern>);

impl Deref for HostUnixSocketOperations {
    type Target = Arc<HostUnixSocketOperationsIntern>;

    fn deref(&self) -> &Arc<HostUnixSocketOperationsIntern> {
        &self.0
    }
}

impl HostUnixSocketOperations {
    pub fn New(
        task: &Task,
        fd: i32,
        stype: i32
    ) -> Result<Self> {
        let msrc = MountSource::NewHostMountSource(
            &"/".to_string(),
            &task.FileOwner(),
            &WhitelistFileSystem::New(),
            &MountSourceFlags::default(),
            false,
        );

        let mut fstat = Box::new_in(LibcStat::default(), GUEST_HOST_SHARED_ALLOCATOR);
        let ret = Fstat(fd, &mut *fstat);
        if ret < 0 {
            return Err(Error::SysError(-ret as i32));
        }
        let msrc = Arc::new(QMutex::new(msrc));

        let iops = HostInodeOp::New(
            &msrc.lock().MountSourceOperations.clone(),
            fd,
            fstat.WouldBlock(),
            &*fstat,
            true,
            false,
            false,
        );
        let queue = iops.Queue();

        let ret = HostUnixSocketOperationsIntern {
            send: AtomicI64::new(0),
            recv: AtomicI64::new(0),
            stype,
            fd,
            queue,
            iops: iops
        };

        let ret = Self(Arc::new(ret));
        return Ok(ret);
    }
}

pub struct HostUnixSocketOperationsIntern {
    pub send: AtomicI64,
    pub recv: AtomicI64,
    pub stype: i32,
    pub fd: i32,
    pub queue: Queue,
    pub iops: HostInodeOp,
}

impl Waitable for HostUnixSocketOperations {
    fn Readiness(&self, _task: &Task, mask: EventMask) -> EventMask {
        let fd = self.fd;
        return NonBlockingPoll(fd, mask);
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        let queue = self.queue.clone();
        queue.EventRegister(task, e, mask);
        let fd = self.fd;
        UpdateFD(fd).unwrap();
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        let queue = self.queue.clone();
        queue.EventUnregister(task, e);
        let fd = self.fd;
        UpdateFD(fd).unwrap();
    }
}

impl HostUnixSocketOperations {
    pub fn ReadAt(
        &self,
        task: &Task,
        dsts: &mut [IoVec]
    ) -> Result<i64> {
        let size = IoVec::NumBytes(dsts);
        let buf = DataBuff::New(size);
        let iovs = buf.Iovs(size);
        let ret = IORead(self.fd, &iovs)?;

        // handle partial memcopy
        task.CopyDataOutToIovs(&buf.buf[0..ret as usize], dsts, false)?;
        return Ok(ret);
    }

    pub fn WriteAt(
        &self,
        task: &Task,
        srcs: &[IoVec]
    ) -> Result<i64> {
        let size = IoVec::NumBytes(srcs);
        if size == 0 {
            return Ok(0);
        }

        let size = IoVec::NumBytes(srcs);
        let mut buf = DataBuff::New(size);
        let len = task.CopyDataInFromIovs(&mut buf.buf, srcs, true)?;
        let iovs = buf.Iovs(len);
        return IOWrite(self.fd, &iovs);
    }

    pub fn Ioctl(&self, task: &Task, _f: &File, _fd: i32, request: u64, val: u64) -> Result<u64> {
        let flags = request as i32;

        let hostfd = self.fd;
        match flags as u64 {
            LibcConst::SIOCGIFFLAGS
            | LibcConst::SIOCGIFBRDADDR
            | LibcConst::SIOCGIFDSTADDR
            | LibcConst::SIOCGIFHWADDR
            | LibcConst::SIOCGIFINDEX
            | LibcConst::SIOCGIFMAP
            | LibcConst::SIOCGIFMETRIC
            | LibcConst::SIOCGIFMTU
            | LibcConst::SIOCGIFNAME
            | LibcConst::SIOCGIFNETMASK
            | LibcConst::SIOCGIFTXQLEN => {
                let addr = val;
                HostIoctlIFReq(task, hostfd, request, addr)?;

                return Ok(0);
            }
            LibcConst::SIOCGIFCONF => {
                let addr = val;
                HostIoctlIFConf(task, hostfd, request, addr)?;

                return Ok(0);
            }
            LibcConst::TIOCINQ => {
                let tmp = Box::new_in(0i32, GUEST_HOST_SHARED_ALLOCATOR);
                let res = Kernel::HostSpace::IoCtl(self.fd, request, &tmp as *const _ as u64);
                if res < 0 {
                    return Err(Error::SysError(-res as i32));
                }
                task.CopyOutObj(&*tmp, val)?;
                return Ok(0);
            }
            _ => {
                let tmp = Box::new_in(0i32, GUEST_HOST_SHARED_ALLOCATOR);
                let res = Kernel::HostSpace::IoCtl(self.fd, request, &tmp as *const _ as u64);
            if res < 0 {
                    return Err(Error::SysError(-res as i32));
                }
                task.CopyOutObj(&*tmp, val)?;
                return Ok(0);
            }
        }
    }

    pub fn GetSockOpt(&self, _task: &Task, level: i32, name: i32, opt: &mut [u8]) -> Result<i64> {
        let mut optLen = Box::new_in(opt.len(), GUEST_HOST_SHARED_ALLOCATOR);
        let res = if *optLen == 0 {
            Kernel::HostSpace::GetSockOpt(
                self.fd,
                level,
                name,
                ptr::null::<u8>() as u64,
                &mut *optLen as *mut _ as u64,
            )
        } else {
            Kernel::HostSpace::GetSockOpt(
                self.fd,
                level,
                name,
                &mut opt[0] as *mut _ as u64,
                &mut *optLen as *mut _ as u64,
            )
        };

        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        return Ok(*optLen as i64);
    }


    pub fn GetSockName(&self, _task: &Task, socketaddr: &mut [u8]) -> Result<i64> {
        let len = Box::new_in(socketaddr.len() as i32, GUEST_HOST_SHARED_ALLOCATOR);

        let res = Kernel::HostSpace::GetSockName(
            self.fd,
            &socketaddr[0] as *const _ as u64,
            &*len as *const _ as u64,
        );
        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        return Ok(*len as i64);
    }

    pub fn GetPeerName(&self, _task: &Task, socketaddr: &mut [u8]) -> Result<i64> {
        let len = Box::new_in(socketaddr.len() as i32, GUEST_HOST_SHARED_ALLOCATOR);
        let res = Kernel::HostSpace::GetPeerName(
            self.fd,
            &socketaddr[0] as *const _ as u64,
            &*len as *const _ as u64,
        );
        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        return Ok(*len as i64);
    }
    
    pub fn RecvMsg(
        &self,
        task: &Task,
        dsts: &mut [IoVec],
        flags: i32,
        deadline: Option<Time>,
        _senderRequested: bool,
        controlDataLen: usize,
    ) -> Result<(i64, i32, Option<(SockAddr, usize)>, Vec<u8>)> {
        if flags
            & !(MsgType::MSG_DONTWAIT
                | MsgType::MSG_PEEK
                | MsgType::MSG_TRUNC
                | MsgType::MSG_ERRQUEUE
                | MsgType::MSG_CTRUNC
                | MsgType::MSG_WAITALL
                | MsgType::MSG_CMSG_CLOEXEC
            )
            != 0
        {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let size = IoVec::NumBytes(dsts);
        let buf = DataBuff::New(size);
        let iovs = buf.Iovs(size);

        let mut msgHdr = Box::new_in(MsgHdr::default(), GUEST_HOST_SHARED_ALLOCATOR);
        msgHdr.iov = &iovs[0] as *const _ as u64;
        msgHdr.iovLen = iovs.len();

        // let mut addr: [u8; SIZEOF_SOCKADDR_UNIX] = [0; SIZEOF_SOCKADDR_UNIX];
        // if senderRequested {
        //     msgHdr.msgName = &mut addr[0] as *mut _ as u64;
        //     msgHdr.nameLen = SIZEOF_SOCKADDR_UNIX as u32;
        // }

        let mut controlVec: Vec<u8, GuestHostSharedAllocator> = Vec::with_capacity_in(controlDataLen, GUEST_HOST_SHARED_ALLOCATOR);
        controlVec.resize(controlDataLen, 0);
        msgHdr.msgControlLen = controlDataLen;
        if msgHdr.msgControlLen != 0 {
            msgHdr.msgControl = &mut controlVec[0] as *mut _ as u64;
        } else {
            msgHdr.msgControl = ptr::null::<u8>() as u64;
        }

        let general = task.blocker.generalEntry.clone();
        self.EventRegister(task, &general, EVENT_READ);
        defer!(self.EventUnregister(task, &general));

        let mut res = Kernel::HostSpace::HostUnixRecvMsg(
            self.fd,
            &mut *msgHdr as *mut _ as u64,
            flags | MsgType::MSG_DONTWAIT
        ) as i32;

        while res == -SysErr::EWOULDBLOCK
            && flags & (MsgType::MSG_DONTWAIT | MsgType::MSG_ERRQUEUE) == 0
        {
            match task.blocker.BlockWithMonoTimer(true, deadline) {
                Err(Error::ErrInterrupted) => {
                    return Err(Error::SysError(SysErr::ERESTARTSYS));
                }
                Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                    return Err(Error::SysError(SysErr::EAGAIN));
                }
                Err(e) => {
                    return Err(e);
                }
                _ => (),
            }

            res = Kernel::HostSpace::IORecvMsg(
                self.fd,
                &mut *msgHdr as *mut _ as u64,
                flags | MsgType::MSG_DONTWAIT,
                false,
            ) as i32;
        }

        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        let msgFlags = msgHdr.msgFlags & !MsgType::MSG_CTRUNC;
        // let senderAddr = if senderRequested
        //     // for tcp connect, recvmsg get nameLen=0 msg
        //     && msgHdr.nameLen >= 4
        // {
        //     let addr = GetAddr(addr[0] as i16, &addr[0..msgHdr.nameLen as usize])?;
        //     let l = addr.Len();
        //     Some((addr, l))
        // } else {
        //     None
        // };

        let senderAddr = None;
        
        controlVec.resize(msgHdr.msgControlLen, 0);
        let totalLen = controlVec.len();

        if msgHdr.msgControlLen > 0 {
            let ctrlMsg = Parse(&controlVec)?;
            let mut fds = Vec::new();
            match &ctrlMsg.Rights {
                Some(right) => {
                    for fd in &right.0 {
                        let fd = *fd;
                        let file = NewHostFile(task, fd)?;

                        let newFd = task.NewFDFrom(
                            0,
                            &file,
                            &FDFlags {
                                CloseOnExec: true,
                            },
                        )?;
        
                        fds.push(newFd);
                    }
                }
                None => ()
            }

            let controlData = &mut controlVec[..];
            let (controlData, _) = ControlMessageRights(fds).EncodeInto(controlData, msgHdr.msgFlags);

            let new_size = totalLen - controlData.len();
            msgHdr.msgControlLen = new_size;
            controlVec.resize(msgHdr.msgControlLen, 0);
        }

        // todo: need to handle partial copy
        let count = if res < buf.buf.len() as i32 {
            res
        } else {
            buf.buf.len() as i32
        };
        let _len = task.CopyDataOutToIovs(&buf.buf[0..count as usize], dsts, false)?;
        return Ok((res as i64, msgFlags, senderAddr, controlVec.to_vec()));
    }

    pub fn SendMsg(
        &self,
        task: &Task,
        srcs: &[IoVec],
        flags: i32,
        msgHdr: &mut MsgHdr,
        deadline: Option<Time>,
    ) -> Result<i64> {
        if flags
            & !(MsgType::MSG_DONTWAIT
                | MsgType::MSG_EOR
                | MsgType::MSG_FASTOPEN
                | MsgType::MSG_MORE
                | MsgType::MSG_NOSIGNAL)
            != 0
        {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let size = IoVec::NumBytes(srcs);
        let mut buf = DataBuff::New(size);
        let len = task.CopyDataInFromIovs(&mut buf.buf, srcs, true)?;
        let iovs = buf.Iovs(len);

        msgHdr.iov = &iovs[0] as *const _ as u64;
        msgHdr.iovLen = iovs.len();
        msgHdr.msgFlags = 0;

        if msgHdr.msgControlLen > 0 {
            let controlVec = unsafe {
                slice::from_raw_parts_mut(msgHdr.msgControl as *mut u8, msgHdr.msgControlLen)
            };

            let ctrlMsg = Parse(controlVec)?;
            let mut fds = Vec::new();
            match &ctrlMsg.Rights {
                Some(right) => {
                    for fd in &right.0 {
                        let file = task.GetFile(*fd)?;
                        let fops = file.FileOp.clone();
                        match fops {
                            FileOps::OverlayFileOperations(h) => {
                                match h.FileOps() {
                                    FileOps::HostFileOp(h) => {
                                        fds.push(h.InodeOp.HostFd());
                                    }
                                    FileOps::TTYFileOps(h) => {
                                        let hfops = h.lock().fileOps.clone();
                                        fds.push(hfops.InodeOp.HostFd());
                                    }
                                    _ => {
                                        // we only support hostfile sending through host unix socket
                                        return Err(Error::SysError(SysErr::EINVAL));
                                    }
                                }
                            }
                            FileOps::HostFileOp(h) => {
                                fds.push(h.InodeOp.HostFd());
                            }
                            FileOps::TTYFileOps(h) => {
                                let hfops = h.lock().fileOps.clone();
                                fds.push(hfops.InodeOp.HostFd());
                            }
                            _ => {
                                // we only support hostfile sending through host unix socket
                                return Err(Error::SysError(SysErr::EINVAL));
                            }
                        }
                    }
                }
                None => ()
            }
            
            let totalLen = controlVec.len();
            let controlData = &mut controlVec[..];
            let (controlData, _) = ControlMessageRights(fds).EncodeInto(controlData, msgHdr.msgFlags);

            let new_size = totalLen - controlData.len();
            msgHdr.msgControlLen = new_size;
        }

        let mut res = if msgHdr.msgControlLen > 0 {
            Kernel::HostSpace::IOSendMsg(
                self.fd,
                msgHdr as *const _ as u64,
                flags | MsgType::MSG_DONTWAIT,
                false,
            ) as i32
        } else {
            Kernel::HostSpace::IOSendto(
                self.fd,
                buf.Ptr(),
                len,
                flags | MsgType::MSG_DONTWAIT,
                msgHdr.msgName,
                msgHdr.nameLen,
            ) as i32
        };

        while res == -SysErr::EWOULDBLOCK && flags & MsgType::MSG_DONTWAIT == 0 {
            let general = task.blocker.generalEntry.clone();

            self.EventRegister(task, &general, EVENT_WRITE);
            defer!(self.EventUnregister(task, &general));
            match task.blocker.BlockWithMonoTimer(true, deadline) {
                Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                    return Err(Error::SysError(SysErr::EAGAIN))
                }
                Err(e) => {
                    return Err(e);
                }
                _ => (),
            }

            res = if msgHdr.msgControlLen > 0 {
                Kernel::HostSpace::IOSendMsg(
                    self.fd,
                    msgHdr as *const _ as u64,
                    flags | MsgType::MSG_DONTWAIT,
                    false,
                ) as i32
            } else {
                Kernel::HostSpace::IOSendto(
                    self.fd,
                    buf.Ptr(),
                    len,
                    flags | MsgType::MSG_DONTWAIT,
                    msgHdr.msgName,
                    msgHdr.nameLen,
                ) as i32
            };
        }

        if res < 0 {
            return Err(Error::SysError(-res as i32));
        }

        return Ok(res as i64);
    }

    pub fn SetRecvTimeout(&self, ns: i64) {
        self.recv.store(ns, Ordering::Relaxed)
    }

    pub fn SetSendTimeout(&self, ns: i64) {
        self.send.store(ns, Ordering::Relaxed)
    }

    pub fn RecvTimeout(&self) -> i64 {
        return self.recv.load(Ordering::Relaxed);
    }

    pub fn SendTimeout(&self) -> i64 {
        return self.send.load(Ordering::Relaxed);
    }
}

pub fn NewHostFile(
    task: &Task,
    fd: i32
) -> Result<File> {
    let dirent = NewHostfileDirent(task, HOST_FILE_DEVICE.clone(), fd)?;
    let inode = dirent.Inode();
    
    let fileflags = FileFlags {
        Read: true,
        Write: true,
        ..Default::default()
    };
    
    let file = match inode.GetFile(task, &dirent, &fileflags) {
        Ok(f) => f,
        Err(Error::ErrInterrupted) => {
            return Err(Error::SysError(SysErr::ERESTARTSYS));
        }
        Err(e) => {
            return Err(e);
        }
    };
    
    return Ok(file);
}
