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

use alloc::vec::Vec;
use alloc::boxed::Box;

use crate::GuestHostSharedAllocator;
use crate::GUEST_HOST_SHARED_ALLOCATOR;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::socket::socket::*;
use super::super::task::*;
//use super::super::socket::control::*;
//use super::super::socket::control::ControlMessage;
use super::super::fs::file::*;
use super::super::fs::flags::*;
use super::super::kernel::fd_table::*;
use super::super::kernel::time::*;
use super::super::qlib::linux::time::*;
use super::super::syscalls::syscalls::*;
//use super::super::qlib::linux::socket::*;
use super::super::kernel::timer::*;

// minListenBacklog is the minimum reasonable backlog for listening sockets.
const MIN_LISTEN_BACKLOG: u32 = 8;

// maxListenBacklog is the maximum allowed backlog for listening sockets.
const MAX_LISTEN_BACKLOG: u32 = 1024;

// maxAddrLen is the maximum socket address length we're willing to accept.
const MAX_ADDR_LEN: u32 = 200;

// maxOptLen is the maximum sockopt parameter length we're willing to accept.
const MAX_OPT_LEN: u32 = 1024;

// maxControlLen is the maximum length of the msghdr.msg_control buffer we're
// willing to accept. Note that this limit is smaller than Linux, which allows
// buffers upto INT_MAX.
const MAX_CONTROL_LEN: usize = 10 * 1024 * 1024;

// nameLenOffset is the offset from the start of the MessageHeader64 struct to
// the NameLen field.
const NAME_LEN_OFFSET: u32 = 8;

// controlLenOffset is the offset form the start of the MessageHeader64 struct
// to the ControlLen field.
const CONTROL_LEN_OFFSET: u32 = 40;

// flagsOffset is the offset form the start of the MessageHeader64 struct
// to the Flags field.
const FLAGS_OFFSET: u32 = 48;

pub fn SysSocket(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let domain = args.arg0 as i32;
    let stype = args.arg1 as i32;
    let protocol = args.arg2 as i32;

    if stype & !(0xf | SocketFlags::SOCK_CLOEXEC | SocketFlags::SOCK_NONBLOCK) != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let s = NewSocket(task, domain, stype, protocol)?;

    let flags = SettableFileFlags {
        NonBlocking: stype & Flags::O_NONBLOCK != 0,
        ..Default::default()
    };

    s.SetFlags(task, flags);
    s.flags.lock().0.NonSeekable = true;
    let fd = task.NewFDFrom(
        0,
        &s,
        &FDFlags {
            CloseOnExec: stype & SocketFlags::SOCK_CLOEXEC != 0,
        },
    )?;

    return Ok(fd as i64);
}

pub fn SysSocketPair(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let domain = args.arg0 as i32;
    let stype = args.arg1 as i32;
    let protocol = args.arg2 as i32;
    let socks = args.arg3 as u64;

    if stype & !(0xf | SocketFlags::SOCK_CLOEXEC | SocketFlags::SOCK_NONBLOCK) != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let fileFlags = SettableFileFlags {
        NonBlocking: stype & Flags::O_NONBLOCK != 0,
        ..Default::default()
    };

    let fdFlags = FDFlags {
        CloseOnExec: stype & SocketFlags::SOCK_CLOEXEC != 0,
    };

    let (s1, s2) = NewPair(task, domain, stype & 0xf, protocol)?;

    s1.SetFlags(task, fileFlags);
    s1.flags.lock().0.NonSeekable = true;
    s2.SetFlags(task, fileFlags);
    s2.flags.lock().0.NonSeekable = true;

    let fd1 = task.NewFDFrom(0, &s1, &fdFlags)?;
    let fd2 = task.NewFDFrom(0, &s2, &fdFlags)?;

    let fds = [fd1, fd2];
    task.CopyOutSlice(&fds, socks, 2)?;

    return Ok(0);
}

pub fn CaptureAddress(task: &Task, addr: u64, addrlen: u32) -> Result<Vec<u8, GuestHostSharedAllocator>> {
    if addrlen > MAX_ADDR_LEN {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    //task.CheckPermission(addr, addrlen as u64, false, false)?;

    return task.CopyInVecShared(addr, addrlen as usize);
}

#[derive(Debug)]
pub struct SockaddrIn {
    pub sin_family: u16,
    pub sin_port: u16,
    pub sin_addr: [u8; 4],
    pub sin_zero: [u8; 8],
}

pub fn SysConnect(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let addrlen = args.arg2 as u32;

    if addrlen > MAX_ADDR_LEN {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let file = task.GetFile(fd)?;

    let blocking = !file.Flags().NonBlocking;
    let sock = file.FileOp.clone();

    if addrlen > MAX_ADDR_LEN as u32 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let addrstr = CaptureAddress(task, addr, addrlen)?;

    sock.Connect(task, &addrstr, blocking)?;

    return Ok(0);
}

pub fn SysAccept4(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let addrlen = args.arg2 as u64;
    let flags = args.arg3 as i32;

    return Accept4(task, fd, addr, addrlen, flags);
}

pub fn Accept4(task: &Task, fd: i32, addr: u64, addrlen: u64, flags: i32) -> Result<i64> {
    if flags & !(SocketFlags::SOCK_CLOEXEC | SocketFlags::SOCK_NONBLOCK) != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let file = task.GetFile(fd)?;

    let sock = file.FileOp.clone();

    let blocking = !file.Flags().NonBlocking;

    let len = if addrlen == 0 {
        0
    } else {
        let len = task.CopyInObj::<i32>(addrlen)?;

        if len < 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }
        len as u32
    };

    let mut addrstr: [u8; MAX_ADDR_LEN as usize] = [0; MAX_ADDR_LEN as usize];

    let mut len = if len < MAX_ADDR_LEN {
        len
    } else {
        MAX_ADDR_LEN as u32
    };

    let lenCopy = len;
    let peerRequested = len != 0;

    let addrstr = &mut addrstr[..len as usize];

    let nfd = match sock.Accept(task, addrstr, &mut len, flags, blocking) {
        Err(Error::ErrInterrupted) => return Err(Error::SysError(SysErr::ERESTARTSYS)),
        Err(e) => return Err(e),
        Ok(nfd) => nfd,
    };

    if peerRequested {
        task.CopyOutSlice(addrstr, addr, lenCopy as usize)?;
        //*task.GetTypeMut::<i32>(addrlen)? = len as i32;
        task.CopyOutObj(&(len as i32), addrlen)?
    }

    return Ok(nfd);
}

pub fn SysAccept(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let addrlen = args.arg2 as u64;

    return Accept4(task, fd, addr, addrlen, 0);
}

pub fn SysBind(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let addrlen = args.arg2 as usize;

    let file = task.GetFile(fd)?;

    let sock = file.FileOp.clone();

    if addrlen > MAX_ADDR_LEN as usize {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let addrstr = task.CopyInVecShared(addr, addrlen as usize)?;
    let res = sock.Bind(task, &addrstr);

    return res;
}

pub fn SysListen(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let backlog = args.arg1 as u32;

    let file = task.GetFile(fd)?;

    let sock = file.FileOp.clone();
    let mut backlog = backlog;

    if backlog >= MAX_LISTEN_BACKLOG {
        backlog = MAX_LISTEN_BACKLOG;
    }

    // Accept one more than the configured listen backlog to keep in parity with
    // Linux. Ref, because of missing equality check here:
    // https://github.com/torvalds/linux/blob/7acac4b3196/include/net/sock.h#L937
    //
    // In case of unix domain sockets, the following check
    // https://github.com/torvalds/linux/blob/7d6beb71da3/net/unix/af_unix.c#L1293
    // will allow 1 connect through since it checks for a receive queue len >
    // backlog and not >=.
    let backlog = backlog + 1;

    let res = sock.Listen(task, backlog as i32);
    return res;
}

pub fn SysShutdown(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let how = args.arg1 as i32;

    let file = task.GetFile(fd)?;

    let sock = file.FileOp.clone();

    let how = how as u64;
    if !(how == LibcConst::SHUT_RD || how == LibcConst::SHUT_WR || how == LibcConst::SHUT_RDWR) {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let res = sock.Shutdown(task, how as i32);
    return res;
}

pub fn SysGetSockOpt(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let level = args.arg1 as i32;
    let name = args.arg2 as i32;
    let optValAddr = args.arg3 as u64;
    let optLenAddr = args.arg4 as u64;

    let file = task.GetFile(fd)?;

    let sock = file.FileOp.clone();

    let optlen = if optLenAddr != 0 {
        let optlen = task.CopyInObj::<i32>(optLenAddr)?;

        if optlen < 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        optlen
    } else {
        0
    };

    let mut optVal = Box::new_in([0u8; MAX_OPT_LEN as usize], GUEST_HOST_SHARED_ALLOCATOR);
    let res = sock.GetSockOpt(task, level, name, &mut optVal[..optlen as usize])?;

    if res < 0 {
        panic!("GetSockOpt: get negative optlen")
    }

    let len = res as usize;
    task.CopyOutSlice(&optVal[..len], optValAddr, len)?;
    //*task.GetTypeMut(optLenAddr)? = len as i32;
    task.CopyOutObj(&(len as i32), optLenAddr)?;

    return Ok(0);
}

pub fn SysSetSockOpt(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let level = args.arg1 as i32;
    let name = args.arg2 as i32;
    let optValAddr = args.arg3 as u64;
    let optLen = args.arg4 as i32;

    let file = task.GetFile(fd)?;

    let sock = file.FileOp.clone();

    // Linux allows optlen = 0, which is equivalent to optval = 0,
    // see `do_ip_setsockopt` in linux/source/net/ipv4/ip_sockglue.c
    if optLen < 0 || optLen > MAX_OPT_LEN as i32 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let optVal = task.CopyInVecShared(optValAddr, optLen as usize)?;
    let res = sock.SetSockOpt(task, level, name, &optVal[..optLen as usize])?;

    return Ok(res);
}

pub fn SysGetSockName(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let addrlen = args.arg2 as u64;

    let file = task.GetFile(fd)?;

    let sock = file.FileOp.clone();

    let mut buf = Box::new_in([0u8; MAX_ADDR_LEN as usize], GUEST_HOST_SHARED_ALLOCATOR);
    let len = task.CopyInObj::<i32>(addrlen)?;

    let len = if len > MAX_ADDR_LEN as i32 {
        MAX_ADDR_LEN as i32
    } else {
        len
    };

    let mut outputlen = sock.GetSockName(task, &mut buf[0..len as usize])? as usize;
    //*(task.GetTypeMut::<i32>(addrlen)?) = outputlen as i32;
    task.CopyOutObj(&(outputlen as i32), addrlen)?;

    if len < outputlen as i32 {
        outputlen = len as usize;
    }

    task.CopyOutSlice(&buf[..outputlen as usize], addr, outputlen as usize)?;

    return Ok(0);
}

pub fn SysGetPeerName(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let addrlen = args.arg2 as u64;

    let file = task.GetFile(fd)?;

    let sock = file.FileOp.clone();
    let mut buf = Box::new_in([0u8; MAX_ADDR_LEN as usize], GUEST_HOST_SHARED_ALLOCATOR);

    let mut outputlen = sock.GetPeerName(task, &mut *buf)? as usize;

    //info!("SysGetPeerName buf is {}", &buf[..outputlen as usize]);

    let len = task.CopyInObj::<i32>(addrlen)?;
    if len < outputlen as i32 {
        outputlen = len as usize;
    }

    task.CopyOutSlice(&buf[..outputlen as usize], addr, addrlen as usize)?;
    //*(task.GetTypeMut::<i32>(addrlen)?) = outputlen as i32;
    task.CopyOutObj(&(outputlen as i32), addrlen)?;
    return Ok(0);
}

fn recvSingleMsg(
    task: &Task,
    sock: &FileOps,
    msgPtr: u64,
    flags: i32,
    deadline: Option<Time>,
) -> Result<i64> {
    //let msg = task.GetTypeMut::<MsgHdr>(msgPtr)?;

    let mut msg: MsgHdr = task.CopyInObj(msgPtr)?;
    if msg.iovLen > UIO_MAXIOV {
        return Err(Error::SysError(SysErr::EMSGSIZE));
    }

    if msg.msgControl == 0 {
        msg.msgControlLen = 0;
    }

    if msg.msgName == 0 {
        msg.nameLen = 0;
    }

    let mut dst = task.IovsFromAddr(msg.iov, msg.iovLen)?;

    // Fast path when no control message nor name buffers are provided.
    if msg.msgControlLen == 0 && msg.nameLen == 0 {
        let (n, mut mflags, _, controlMessageBuffer) =
            sock.RecvMsg(task, &mut dst, flags, deadline, false, 0)?;

        if controlMessageBuffer.len() != 0 {
            mflags |= MsgType::MSG_CTRUNC;
        }

        msg.msgFlags = mflags;
        task.CopyOutObj(&msg, msgPtr)?;

        return Ok(n as i64);
    }

    if msg.msgControlLen > MAX_CONTROL_LEN {
        return Err(Error::SysError(SysErr::ENOBUFS));
    }

    let mut addressVec: Vec<u8> = vec![0; msg.nameLen as usize];
    //let mut controlVec: Vec<u8> = vec![0; msg.msgControlLen as usize];

    let (n, mflags, sender, controlMessageBuffer) = sock.RecvMsg(
        task,
        &mut dst,
        flags,
        deadline,
        msg.nameLen != 0,
        msg.msgControlLen,
    )?;

    /*
     let controlData = &mut controlVec[..];

    //todo: handle Timestamp ControlMessage

    let mut credType : [u8; 4] = [0; 4];
    let controlData = if let Ok(_) = sock.GetSockOpt(task, SOL_SOCKET, LibcConst::SO_PASSCRED as i32, &mut credType) {
        if credType[0] != 0 {
            match cms.Credentials {
                // Edge case: user set SO_PASSCRED but the sender didn't set it in control massage
                None => {
                    let (data, flags) = ControlMessageCredentials::Empty().EncodeInto(controlData, mflags);
                    mflags = flags;
                    data
                }
                Some(ref creds) => {
                    let (data, flags) = creds.Credentials().EncodeInto(controlData, mflags);
                    mflags = flags;
                    data
                },
            }
        } else {
            controlData
        }
    } else {
        controlData
    };

    let controlData = match cms.Rights {
        None => controlData,
        Some(ref mut rights) => {
            let maxFDs = (controlData.len() as isize - SIZE_OF_CONTROL_MESSAGE_HEADER as isize) / 4;
            if maxFDs < 0 {
                mflags |= MsgType::MSG_CTRUNC;
                controlData
            } else {
                let (fds, trunc) = rights.RightsFDs(task, flags & MsgType::MSG_CMSG_CLOEXEC != 0, maxFDs as usize);
                if trunc {
                    mflags |= MsgType::MSG_CTRUNC;
                }
                let (data, _) = ControlMessageRights(fds).EncodeInto(controlData, mflags);
                data
            }
        },
    };

    msg.msgControlLen = msg.msgControlLen - controlData.len();
    */
    msg.msgControlLen = controlMessageBuffer.len();

    if msg.nameLen != 0 && msg.msgName != 0 && sender.is_some() {
        let (sender, senderLen) = sender.unwrap();
        if msg.nameLen < senderLen as u32 {
            return Err(Error::SysError(SysErr::ERANGE));
        }
        sender.Marsh(&mut addressVec[..], senderLen)?;
        task.CopyOutSlice(&addressVec[0..senderLen], msg.msgName, msg.nameLen as usize)?;
        msg.nameLen = senderLen as u32;
    }
    if msg.msgControl != 0 && msg.msgControlLen != 0 {
        task.CopyOutSlice(
            &controlMessageBuffer[0..msg.msgControlLen as usize],
            msg.msgControl,
            msg.msgControlLen,
        )?;
    } else {
        msg.msgControlLen = 0;
    }

    msg.msgFlags = mflags;

    task.CopyOutObj(&msg, msgPtr)?;
    return Ok(n);
}

fn sendSingleMsg(
    task: &Task,
    sock: &FileOps,
    msgPtr: u64,
    flags: i32,
    deadline: Option<Time>,
) -> Result<i64> {
    let msg = task.CopyInObj::<MsgHdr>(msgPtr)?;

    if msg.msgControlLen > MAX_CONTROL_LEN as usize {
        return Err(Error::SysError(SysErr::ENOBUFS));
    }

    if msg.iovLen > UIO_MAXIOV {
        return Err(Error::SysError(SysErr::EMSGSIZE));
    }

    let msgVec: Vec<u8, GuestHostSharedAllocator> = task.CopyInVecShared(msg.msgName, msg.nameLen as usize)?;
    let controlVec: Vec<u8, GuestHostSharedAllocator> = task.CopyInVecShared(msg.msgControl, msg.msgControlLen as usize)?;

    let mut pMsg = Box::new_in(msg, GUEST_HOST_SHARED_ALLOCATOR);
    if msg.nameLen > 0 {
        pMsg.msgName = &msgVec[0] as *const _ as u64;
    }

    if msg.msgControlLen > 0 {
        pMsg.msgControl = &controlVec[0] as *const _ as u64;
    }

    let src = task.IovsFromAddr(msg.iov, msg.iovLen)?;

    let res = sock.SendMsg(task, &src, flags, &mut *pMsg, deadline)?;
    task.CopyOutObj(&msg, msgPtr)?;
    return Ok(res);
}

pub fn SysRecvMsg(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let msgPtr = args.arg1 as u64;
    let mut flags = args.arg2 as i32;

    let file = task.GetFile(fd)?;

    let sock = file.FileOp.clone();

    if flags
        & !(MsgType::BASE_RECV_FLAGS
            | MsgType::MSG_PEEK
            | MsgType::MSG_CMSG_CLOEXEC
            | MsgType::MSG_ERRQUEUE)
        != 0
    {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if !file.Blocking() {
        flags |= MsgType::MSG_DONTWAIT
    }

    let mut deadline = None;
    let dl = file.FileOp.RecvTimeout();
    if dl > 0 {
        let now = MonotonicNow();
        deadline = Some(Time(now + dl));
    } else if dl < 0 {
        flags |= MsgType::MSG_DONTWAIT
    }

    let res = recvSingleMsg(task, &sock, msgPtr, flags, deadline)?;
    return Ok(res);
}

pub fn SysRecvMMsg(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let msgPtr = args.arg1 as u64;
    let vlen = args.arg2 as i32;
    let mut flags = args.arg3 as i32;
    let timeout = args.arg4 as u64;

    let file = task.GetFile(fd)?;

    let sock = file.FileOp.clone();

    if flags & !(MsgType::BASE_RECV_FLAGS | MsgType::MSG_CMSG_CLOEXEC | MsgType::MSG_ERRQUEUE) != 0
    {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if vlen < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let mut vlen = vlen as u32;

    if vlen > UIO_MAXIOV as u32 {
        vlen = UIO_MAXIOV as u32;
    }

    let mut deadline = None;
    if timeout != 0 {
        let timePtr = task.CopyInObj::<Timespec>(timeout)?;

        let now = MonotonicNow();
        deadline = Some(Time(now + timePtr.ToNs()?));
    }

    if deadline.is_none() {
        let dl = file.FileOp.RecvTimeout();
        if dl > 0 {
            let now = MonotonicNow();
            deadline = Some(Time(now + dl));
        } else if dl < 0 {
            flags |= MsgType::MSG_DONTWAIT;
        }
    }

    let mut count = 0;
    let mut res = 0;
    //let msgs = task.GetSliceMut::<MMsgHdr>(msgPtr, vlen as usize)?;
    let mut msgs = task.CopyInVec::<MMsgHdr>(msgPtr, vlen as usize)?;

    //info!("SysRecvMMsg 1 vlen is {}", vlen);
    for i in 0..vlen as usize {
        res = match recvSingleMsg(
            task,
            &sock,
            &(msgs[i].msgHdr) as *const MsgHdr as u64,
            flags,
            deadline,
        ) {
            Err(e) => {
                if count > 0 {
                    break;
                }

                return Err(e);
            }
            Ok(n) => n,
        };

        if res < 0 {
            break;
        }

        msgs[i].msgLen = res as u32;
        count += 1;
    }

    if count == 0 {
        return Err(Error::SysError(-res as i32));
    }

    task.CopyOutSlice(&msgs, msgPtr, vlen as usize)?;

    return Ok(count);
}

pub const BASE_RECV_FLAGS: i32 = MsgType::MSG_OOB
    | MsgType::MSG_DONTROUTE
    | MsgType::MSG_DONTWAIT
    | MsgType::MSG_NOSIGNAL
    | MsgType::MSG_WAITALL
    | MsgType::MSG_TRUNC
    | MsgType::MSG_CTRUNC;

pub fn SysRecvFrom(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let bufPtr = args.arg1 as u64;
    let buflen = args.arg2 as i64;
    let flags = args.arg3 as i32;
    let namePtr = args.arg4 as u64;
    let nameLenPtr = args.arg5 as u64;

    let file = task.GetFile(fd)?;

    let sock = file.FileOp.clone();

    if buflen < 0 {
        return Err(Error::SysError(-SysErr::EINVAL));
    }

    if flags & !(BASE_RECV_FLAGS | MsgType::MSG_PEEK | MsgType::MSG_CONFIRM) != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let mut flags = flags;
    if !file.Blocking() {
        flags |= MsgType::MSG_DONTWAIT
    }

    let iov = IoVec::NewFromAddr(bufPtr, buflen as usize);
    let mut iovs: [IoVec; 1] = [iov];

    let mut nameLen: i32 = 0;
    if nameLenPtr != 0 {
        nameLen = task.CopyInObj(nameLenPtr)?;
    }

    if (namePtr == 0 || nameLen == 0)
        && (file.Flags().NonBlocking || flags & MsgType::MSG_DONTWAIT != 0)
        && flags & !MsgType::MSG_DONTWAIT == 0
    {
        match file.Readv(task, &mut iovs) {
            Err(Error::ErrInterrupted) => return Err(Error::SysError(SysErr::ERESTARTSYS)),
            Err(e) => {
                return Err(e);
            }
            Ok(n) => {
                return Ok(n);
            }
        };
    }

    let mut pMsg = MsgHdr::default();

    //todo: handle the msg.nameLen > 1024
    let _msgVec = if namePtr != 0 {
        let msgVec: Vec<u8> = vec![0; nameLen as usize];
        pMsg.msgName = &msgVec[0] as *const _ as u64;
        msgVec
    } else {
        Vec::new()
    };

    let mut deadline = None;
    let dl = file.FileOp.RecvTimeout();
    if dl > 0 {
        let now = MonotonicNow();
        deadline = Some(Time(now + dl));
    } else if dl < 0 {
        flags |= MsgType::MSG_DONTWAIT;
    }

    let (bytes, _, sender, _) =
        sock.RecvMsg(task, &mut iovs, flags, deadline, nameLenPtr != 0, 0)?;

    if nameLenPtr != 0 && sender.is_some() {
        let (sender, senderLen) = sender.unwrap();
        if senderLen != 2 {
            if nameLen < senderLen as i32 {
                return Err(Error::SysError(SysErr::ERANGE));
            }
            //let slices = task.GetSliceMut::<u8>(namePtr, nameLen as usize)?;
            //sender.Marsh(slices, senderLen)?;
            let mut dataBuf = DataBuff::New(nameLen as usize);
            sender.Marsh(&mut dataBuf.buf, senderLen)?;
            task.CopyOutSlice(&mut dataBuf.buf, namePtr, nameLen as usize)?;
            //task.CopyOutSlice(&msgVec[0..pMsg.nameLen as usize], namePtr, nameLen as usize)?;
            task.CopyOutObj(&(senderLen as u32), nameLenPtr)?;
        } else {
            // has only type
            let len: u32 = 0;
            task.CopyOutObj(&len, nameLenPtr)?;
        }
    }

    return Ok(bytes as i64);
}

pub fn SysSendMsg(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let msgPtr = args.arg1 as u64;
    let mut flags = args.arg2 as i32;

    let file = task.GetFile(fd)?;

    let sock = file.FileOp.clone();

    if flags
        & !(MsgType::MSG_DONTWAIT | MsgType::MSG_EOR | MsgType::MSG_MORE | MsgType::MSG_NOSIGNAL)
        != 0
    {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if !file.Blocking() {
        flags |= MsgType::MSG_DONTWAIT;
    }

    let mut deadline = None;
    let dl = sock.SendTimeout();
    if dl > 0 {
        let now = MonotonicNow();
        deadline = Some(Time(now + dl));
    } else if dl < 0 {
        flags |= MsgType::MSG_DONTWAIT
    }

    let res = sendSingleMsg(task, &sock, msgPtr, flags, deadline)?;
    return Ok(res);
}

pub fn SysSendMMsg(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let msgPtr = args.arg1 as u64;
    let vlen = args.arg2 as i32;
    let mut flags = args.arg3 as i32;

    let file = task.GetFile(fd)?;

    let sock = file.FileOp.clone();

    if flags
        & !(MsgType::MSG_DONTWAIT | MsgType::MSG_EOR | MsgType::MSG_MORE | MsgType::MSG_NOSIGNAL)
        != 0
    {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if vlen < 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let mut vlen = vlen as u32;

    if vlen > UIO_MAXIOV as u32 {
        vlen = UIO_MAXIOV as u32;
    }

    if !file.Blocking() {
        flags |= MsgType::MSG_DONTWAIT;
    }

    let mut deadline = None;
    let dl = sock.SendTimeout();
    if dl > 0 {
        let now = MonotonicNow();
        deadline = Some(Time(now + dl));
    } else if dl < 0 {
        flags |= MsgType::MSG_DONTWAIT
    }

    let mut count = 0;
    let mut res = 0;
    //let msgs = task.GetSliceMut::<MMsgHdr>(msgPtr, vlen as usize)?;
    let mut msgs = task.CopyInVec::<MMsgHdr>(msgPtr, vlen as usize)?;
    for i in 0..vlen as usize {
        res = sendSingleMsg(
            task,
            &sock,
            &(msgs[i].msgHdr) as *const MsgHdr as u64,
            flags,
            deadline,
        )?;

        if res < 0 {
            break;
        }

        msgs[i].msgLen = res as u32;
        count += 1;
    }

    if count == 0 {
        return Err(Error::SysError(-res as i32));
    }

    task.CopyOutSlice(&msgs, msgPtr, vlen as usize)?;

    return Ok(count);
}

pub fn SysSendTo(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let bufPtr = args.arg1 as u64;
    let buflen = args.arg2 as i64;
    let mut flags = args.arg3 as i32;
    let namePtr = args.arg4 as u64;
    let nameLen = args.arg5 as u32;

    let file = task.GetFile(fd)?;

    let sock = file.FileOp.clone();

    if buflen < 0 {
        return Err(Error::SysError(-SysErr::EINVAL));
    }

    task.CheckPermission(bufPtr, buflen as u64, false, false)?;
    let iov = IoVec::NewFromAddr(bufPtr, buflen as usize);
    let iovs: [IoVec; 1] = [iov];

    if (namePtr == 0 || nameLen == 0)
        && (file.Flags().NonBlocking || flags & MsgType::MSG_DONTWAIT != 0)
    {
        match file.Writev(task, &iovs) {
            Err(Error::ErrInterrupted) => return Err(Error::SysError(SysErr::ERESTARTSYS)),
            Err(e) => {
                return Err(e);
            }
            Ok(n) => {
                return Ok(n);
            }
        };
    }

    let mut pMsg = Box::new_in(MsgHdr::default(), GUEST_HOST_SHARED_ALLOCATOR);

    let _msgVec = if namePtr != 0 && nameLen > 0 {
        //let vec = task.GetSlice::<u8>(namePtr, nameLen as usize)?.to_vec();
        let vec = task.CopyInVecShared::<u8>(namePtr, nameLen as usize)?;
        pMsg.msgName = vec.as_ptr() as u64;
        pMsg.nameLen = nameLen;
        Some(vec)
    } else {
        pMsg.msgName = 0;
        pMsg.nameLen = 0;
        None
    };

    let mut deadline = None;

    let dl = sock.SendTimeout();
    if dl > 0 {
        let now = MonotonicNow();
        deadline = Some(Time(now + dl));
    } else if dl < 0 {
        flags |= MsgType::MSG_DONTWAIT
    }

    if !file.Blocking() {
        flags |= MsgType::MSG_DONTWAIT;
    }

    let res = sock.SendMsg(task, &iovs, flags, &mut pMsg, deadline)?;
    return Ok(res);
}
