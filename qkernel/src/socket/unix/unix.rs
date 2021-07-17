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

use core::any::Any;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::AtomicI64;
use core::sync::atomic::Ordering;
use alloc::string::String;
use alloc::string::ToString;
use spin::Mutex;
use core::ops::*;
use alloc::boxed::Box;

use super::super::socket::*;
use super::super::super::fs::attr::*;
use super::super::super::fs::file::*;
use super::super::super::fs::flags::*;
use super::super::super::fs::dentry::*;
use super::super::super::fs::dirent::*;
//use super::super::super::fs::attr::*;
use super::super::super::fs::host::hostinodeop::*;
use super::super::super::kernel::fd_table::*;
use super::super::super::kernel::abstract_socket_namespace::*;
use super::super::super::kernel::waiter::*;
use super::super::super::kernel::time::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux::socket::*;
use super::super::super::task::*;
//use super::super::super::qlib::mem::io::*;
use super::super::super::qlib::mem::seq::*;
use super::super::super::qlib::path::*;
//use super::super::super::Kernel;
use super::super::super::Kernel::HostSpace;
use super::super::super::qlib::linux_def::*;
//use super::super::super::fd::*;
use super::super::super::tcpip::tcpip::*;
use super::transport::unix::*;
use super::transport::connectioned::*;
use super::transport::connectionless::*;
use super::super::super::socket::control::*;
use super::super::super::socket::epsocket::epsocket::*;

pub fn NewUnixSocket(task: &Task, ep: BoundEndpoint, stype: i32, hostfd: i32) -> Result<File> {
    //assert!(family == AFType::AF_UNIX, "NewUnixSocket family is not AF_UNIX");
    let dirent = NewSocketDirent(task, UNIX_SOCKET_DEVICE.clone(), hostfd)?;
    let fileFlags = FileFlags {
        Read: true,
        Write: true,
        ..Default::default()
    };

    return Ok(File::New(&dirent, &fileFlags, UnixSocketOperations::New(ep, stype, hostfd)))
}

pub struct UnixSocketOperations {
    pub ep: BoundEndpoint,
    pub stype: i32,
    pub send: AtomicI64,
    pub recv: AtomicI64,
    pub name: Mutex<Option<Vec<u8>>>,
    pub hostfd: i32,
}

impl UnixSocketOperations {
    pub fn New(ep: BoundEndpoint, stype: i32, hostfd: i32) -> Self {
        let ret = Self {
            ep: ep,
            stype: stype,
            send: AtomicI64::new(0),
            recv: AtomicI64::new(0),
            name: Mutex::new(None),
            hostfd: hostfd,
        };

        return ret;
    }

    pub fn State(&self) -> i32 {
        return self.ep.State();
    }

    pub fn IsPacket(&self) -> bool {
        if self.stype == SockType::SOCK_DGRAM || self.stype == SockType::SOCK_SEQPACKET {
            return true;
        }

        if self.stype == SockType::SOCK_STREAM {
            return false;
        }

        // We shouldn't have allowed any other socket types during creation.
        panic!("Invalid socket type {}", self.stype);
    }

    // GetPeerName implements the linux syscall getpeername(2) for sockets backed by
    // a transport.Endpoint.
    pub fn GetPeer(&self, _task: &Task) -> Result<(SockAddr, u32)> {
        let addr = self.ep.GetRemoteAddress()?;
        let l = addr.Len();
        return Ok((SockAddr::Unix(addr), l as u32))
    }

    // GetSockName implements the linux syscall getsockname(2) for sockets backed by
    // a transport.Endpoint.
    pub fn GetSockName(&self, _task: &Task) -> Result<(SockAddr, u32)> {
        let addr = self.ep.GetLocalAddress()?;

        let l = addr.Len();
        return Ok((SockAddr::Unix(addr), l as u32))
    }

    // blockingAccept implements a blocking version of accept(2), that is, if no
    // connections are ready to be accept, it will block until one becomes ready.
    pub fn BlockingAccept(&self, task: &Task) -> Result<ConnectionedEndPoint> {
        let entry = task.blocker.generalEntry.clone();
        self.EventRegister(task, &entry, EVENT_IN);

        defer!(self.EventUnregister(task, &entry));

        // Try to accept the connection; if it fails, then wait until we get a
        // notification.
        loop {
            match self.ep.Accept() {
                Ok(ep) => return Ok(ep),
                Err(Error::SysError(SysErr::EWOULDBLOCK)) => (),
                Err(e) => return Err(e),
            }

            task.blocker.BlockGeneral()?;
        }
    }

    fn encodeControlMsg(&self, task: &Task, mut ctrls: SCMControlMessages, controlDataLen: usize, mflags: &mut i32, cloexec: bool) -> Vec<u8> {
        // fill this with logic currently in sys_socket.....
        let mut controlVec: Vec<u8> = vec![0; controlDataLen];
        let controlData = &mut controlVec[..];

        let mut opt = SockOpt::PasscredOption(0);
        
        let controlData = if let Ok(_) = self.ep.GetSockOpt(&mut opt) {
            match opt {
                SockOpt::PasscredOption(0) => controlData,
                _ => match ctrls.Credentials {
                    // Edge case: user set SO_PASSCRED but the sender didn't set it in control massage
                    None => {
                        let (data, flags) = ControlMessageCredentials::Empty().EncodeInto(controlData, *mflags);
                        *mflags = flags;
                        data
                    },
                    Some(ref creds) => {
                        let (data, flags) = creds.Credentials().EncodeInto(controlData, *mflags);
                        *mflags = flags;
                        data
                    },
                }
            }
        } else {
            controlData
        };

        let controlData = match ctrls.Rights {
            None => controlData,
            Some(ref mut rights) => {
                let maxFDs = (controlData.len() as isize - SIZE_OF_CONTROL_MESSAGE_HEADER as isize) / 4;
                if maxFDs < 0 {
                    *mflags |= MsgType::MSG_CTRUNC;
                    controlData
                } else {
                    let (fds, trunc) = rights.RightsFDs(task, cloexec, maxFDs as usize);
                    if trunc {
                        *mflags |= MsgType::MSG_CTRUNC;
                    }
                    let (controlData, _) = ControlMessageRights(fds).EncodeInto(controlData, *mflags);
                    controlData
                }
            },
        };

        let new_size = controlDataLen - controlData.len();
        controlVec.resize(new_size, 0);
        return controlVec;
    }
}

impl Drop for UnixSocketOperations {
    fn drop(&mut self) {
        self.ep.Close();
    }
}

impl Passcred for UnixSocketOperations {
    fn Passcred(&self) -> bool {
        return self.ep.Passcred();
    }
}

impl ConnectedPasscred  for UnixSocketOperations {
    fn ConnectedPasscred(&self) -> bool {
        return self.ep.ConnectedPasscred();
    }
}

impl Waitable for UnixSocketOperations {
    fn Readiness(&self, task: &Task, mask: EventMask) -> EventMask {
        self.ep.Readiness(task, mask)
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        self.ep.EventRegister(task, e, mask)
    }

    fn EventUnregister(&self, task: &Task,e: &WaitEntry) {
        self.ep.EventUnregister(task, e)
    }
}

// extractPath extracts and validates the address.
pub fn ExtractPath(sockAddr: &[u8]) -> Result<Vec<u8>> {
    let addr = GetAddr(AFType::AF_UNIX as i16, sockAddr)?;
    let p = if let SockAddr::Unix(addr) = addr {
        addr.Path.as_bytes().to_vec()
    } else {
        panic!("impossible")
    };

    if p.len() == 0 {
        // Not allowed.
        return Err(Error::SysError(SysErr::EINVAL))
    }

    if p[p.len()-1] == '/' as u8 {
        // Weird, they tried to bind '/a/b/c/'?
        return Err(Error::SysError(SysErr::EISDIR))
    }

    return Ok(p);
}

impl SpliceOperations for UnixSocketOperations {}

impl FileOperations for UnixSocketOperations {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::UnixSocketOperations
    }

    fn Seekable(&self) -> bool {
        return false;
    }

    fn Seek(&self, _task: &Task, _f: &File, _whence: i32, _current: i64, _offset: i64) -> Result<i64> {
        return Err(Error::SysError(SysErr::ESPIPE))
    }

    fn ReadDir(&self, _task: &Task, _f: &File, _offset: i64, _serializer: &mut DentrySerializer) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn ReadAt(&self, _task: &Task, _f: &File, dsts: &mut [IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
        let count = {
            let srcs = BlockSeq::NewFromSlice(dsts);
            srcs.NumBytes()
        };

        if count == 0 {
            return Ok(0)
        }

        match self.ep.RecvMsg(dsts, false, 0, false, None) {
            Err(Error::ErrClosedForReceive) => {
                // todo: fix this. How to handle ErrClosedForReceive, If again, Read will wait for it
                /*if self.IsPacket() {
                    return Err(Error::SysError(SysErr::EAGAIN))
                }*/
                return Ok(0)
            }
            Err(e) => return Err(e),
            Ok((n, _, _, _)) => {
                return Ok(n as i64)
            }
        }
    }

    fn WriteAt(&self, task: &Task, _f: &File, srcs: &[IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
        let ctrl = if self.ep.ConnectedPasscred() || self.ep.Passcred() {
            NewControlMessage(task, Some(self.ep.clone()), None)
        } else {
            SCMControlMessages::default()
        };

        let count = {
            let srcs = BlockSeq::NewFromSlice(srcs);
            srcs.NumBytes()
        };

        if count == 0 {
            let nInt = self.ep.SendMsg(srcs, &ctrl, &None)?;
            return Ok(nInt as i64)
        }

        match self.ep.SendMsg(srcs, &ctrl, &None) {
            Err(e) => return Err(e),
            Ok(n) => return Ok(n as i64)
        }
    }

    fn Append(&self, task: &Task, f: &File, srcs: &[IoVec]) -> Result<(i64, i64)> {
        let n = self.WriteAt(task, f, srcs, 0, false)?;
        return Ok((n, 0))
    }

    fn Fsync(&self, _task: &Task, _f: &File, _start: i64, _end: i64, _syncType: SyncType) -> Result<()> {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    fn Flush(&self, _task: &Task, _f: &File) -> Result<()> {
        return Ok(())
    }

    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        let inode = f.Dirent.Inode();
        return inode.UnstableAttr(task);
    }

    fn Ioctl(&self, task: &Task, _f: &File, fd: i32, request: u64, val: u64) -> Result<()> {
        return Ioctl(task, &self.ep, fd, request, val)
    }

    fn IterateDir(&self, _task: &Task, _d: &Dirent, _dirCtx: &mut DirCtx, _offset: i32) -> (i32, Result<i64>) {
        return (0, Err(Error::SysError(SysErr::ENOTDIR)))
    }

    fn Mappable(&self) -> Result<HostInodeOp> {
        return Err(Error::SysError(SysErr::ENODEV))
    }
}

// extractEndpoint retrieves the transport.BoundEndpoint associated with a Unix
// socket path. The Release must be called on the transport.BoundEndpoint when
// the caller is done with it.
pub fn ExtractEndpoint(task: &Task, sockAddr: &[u8]) -> Result<BoundEndpoint> {
    let path = ExtractPath(sockAddr)?;

    //info!("unix socket path is {}", String::from_utf8(path.to_vec()).unwrap());

    // Is it abstract?
    if path[0] == 0 {
        let ep = match BoundEndpoint(&path) {
            None => return Err(Error::SysError(SysErr::ECONNREFUSED)),
            Some(ep) => ep,
        };

        return Ok(ep)
    }

    let path = String::from_utf8(path).unwrap();

    // Find the node in the filesystem.
    let root = task.fsContext.RootDirectory();
    let cwd = task.fsContext.WorkDirectory();
    let mut remainingTraversals = 10; //DefaultTraversalLimit
    let mns = task.Thread().MountNamespace();
    let d = mns.FindInode(task, &root, Some(cwd), &path, &mut remainingTraversals)?;

    // Extract the endpoint if one is there.
    let inode = d.Inode();
    let iops = inode.lock().InodeOp.clone();
    let fullName = d.MyFullName();

    //if it is host unix socket, the ep is in virtual unix
    if iops.InodeType() == InodeType::Socket
        && iops.as_any().downcast_ref::<HostInodeOp>().is_some() {
        let ep = match BoundEndpoint(&fullName.into_bytes()) {
            None => return Err(Error::SysError(SysErr::ECONNREFUSED)),
            Some(ep) => ep,
        };

        return Ok(ep)
    }

    let ep = match inode.BoundEndpoint(task, &path) {
        None => return Err(Error::SysError(SysErr::ECONNREFUSED)),
        Some(ep) => ep,
    };

    return Ok(ep)
}

impl SockOperations for UnixSocketOperations {
    fn Connect(&self, task: &Task, socketaddr: &[u8], _blocking: bool) -> Result<i64> {
        let ep = ExtractEndpoint(task, socketaddr)?;

        // Connect the server endpoint.
        match self.ep.Connect(task, &ep) {
            Err(Error::SysError(SysErr::EPROTOTYPE)) => {
                // Linux for abstract sockets returns ErrConnectionRefused
                // instead of ErrWrongProtocolForSocket.
                let path = ExtractPath(socketaddr)?;
                if path[0] == 0 {
                    return Err(Error::SysError(SysErr::ECONNREFUSED))
                } else {
                    return Err(Error::SysError(SysErr::EPROTOTYPE))
                }
            }
            Err(e) => return Err(e),
            _ => (),
        }
        return Ok(0)
    }

    // Accept implements the linux syscall accept(2) for sockets backed by
    // a transport.Endpoint.
    fn Accept(&self, task: &Task, addr: &mut [u8], addrlen: &mut u32, flags: i32, blocking: bool) -> Result<i64> {
        let ep = match self.ep.Accept() {
            Err(Error::SysError(SysErr::EWOULDBLOCK)) => {
                if !blocking {
                    return Err(Error::SysError(SysErr::EWOULDBLOCK));
                }

                self.BlockingAccept(task)?
            }
            Err(e) => return Err(e),
            Ok(ep) => ep,
        };

        let ep = BoundEndpoint::Connected(ep);

        let fd = HostSpace::Socket(AFType::AF_UNIX, self.stype, 0) as i32;
        if fd < 0 {
            return Err(Error::SysError(-fd))
        }

        let ns = NewUnixSocket(task, ep, self.stype, fd)?;
        ns.flags.lock().0.NonSeekable = true;
        if flags & SocketFlags::SOCK_NONBLOCK != 0 {
            let mut fflags = ns.Flags();
            fflags.NonBlocking = true;
            ns.SetFlags(task, fflags.SettableFileFlags());
        }

        if *addrlen != 0 {
            *addrlen = ns.FileOp.GetPeerName(task, addr)? as u32;
        }

        let fdFlags = FDFlags {
            CloseOnExec: flags & SocketFlags::SOCK_CLOEXEC != 0,
        };

        let fd = task.NewFDFrom(0, &ns, &fdFlags)?;
        return Ok(fd as i64);
    }

    fn Bind(&self, task: &Task, socketaddr: &[u8]) -> Result<i64> {
        let p = ExtractPath(socketaddr)?;

        info!("Bind p is {:?}", &p);
        let bep = self.ep.clone();

        let addr = SockAddrUnix::New(core::str::from_utf8(&p).expect("Bind to string fail"));
        self.ep.Bind(&addr)?;

        let root = task.fsContext.RootDirectory();

        // Is it abstract?
        if p[0] == 0 {
            Bind(p.clone(), &bep)?;
            *(self.name.lock()) = Some(p);
        } else {
            let p = String::from_utf8(p).unwrap();
            info!("bind address is {}", &p);

            let cwd = task.fsContext.WorkDirectory();

            let d;
            let name;
            if !p.contains('/') {
                d = cwd;
                name = &p[..];
            } else {
                // Find the last path component, we know that something follows
                // that final slash, otherwise extractPath() would have failed.
                let lastSlash = LastIndex(&p, '/' as u8);
                assert!(lastSlash != -1);
                let subpath = if lastSlash == 0 {
                    // Fix up subpath in case file is in root.
                    "/"
                } else {
                    &p[0..lastSlash as usize]
                };

                let mut remainingTraversals = 10;
                d = task.Thread().MountNamespace().FindInode(task, &root, Some(cwd), &subpath.to_string(), &mut remainingTraversals)?;
                name = &p[lastSlash as usize + 1..];
            }

            // Create the socket.
            let permisson = FilePermissions {
                User: PermMask {read: true, ..Default::default()},
                ..Default::default()
            };

            let inode = d.Inode();
            let iops = inode.lock().InodeOp.clone();

            //if it is host folder, create shadow host unix socket bind
            if iops.InodeType() == InodeType::Directory
                && iops.as_any().downcast_ref::<HostInodeOp>().is_some() {

                let fullName = d.MyFullName() + "/" + &name.to_string();

                let hostfd = self.hostfd;
                let addr = SockAddrUnix::New(&fullName).ToNative();

                let ret = HostSpace::Bind(hostfd, &addr as * const _ as u64, (UNIX_PATH_MAX + 2) as u32, task.Umask());
                if ret < 0 {
                    return Err(Error::SysError(-ret as i32))
                }

                // handle the host unix socket as virtual unix socket
                Bind(fullName.into_bytes(), &bep)?;
                *(self.name.lock()) = Some(p.into_bytes());
            } else {
                match d.Bind(task, &root, &name.to_string(), &bep, &permisson) {
                    Err(_) => return Err(Error::SysError(SysErr::EADDRINUSE)),
                    Ok(_) => (),
                }
            }
        }

        return Ok(0)
    }

    fn Listen(&self, _task: &Task, backlog: i32) -> Result<i64> {
        self.ep.Listen(backlog)?;
        return Ok(0);
    }

    fn Shutdown(&self, _task: &Task, how: i32) -> Result<i64> {
        let f = ConvertShutdown(how)?;

        self.ep.Shutdown(f)?;
        return Ok(0)
    }

    fn GetSockOpt(&self, task: &Task, level: i32, name: i32, opt: &mut [u8]) -> Result<i64> {
        let ret = GetSockOpt(task, self, &self.ep, AFType::AF_UNIX, self.ep.Type(), level, name, opt.len())?;
        let size = ret.Marsh(opt)?;
        return Ok(size as i64)
    }

    fn SetSockOpt(&self, task: &Task, level: i32, name: i32, opt: &[u8]) -> Result<i64> {
        SetSockOpt(task, self, &self.ep, level, name, opt)?;
        return Ok(0)
    }

    fn GetSockName(&self, _task: &Task, socketaddr: &mut [u8]) -> Result<i64> {
        let addr = self.ep.GetLocalAddress()?;

        let l = addr.Len();
        SockAddr::Unix(addr).Marsh(socketaddr, l)?;

        return Ok(l as i64)
    }

    fn GetPeerName(&self, _task: &Task, socketaddr: &mut [u8]) -> Result<i64> {
        let addr = self.ep.GetRemoteAddress()?;

        let l = addr.Len();
        SockAddr::Unix(addr).Marsh(socketaddr, l as usize)?;

        return Ok(l as i64)
    }

    fn RecvMsg(&self, task: &Task, dsts: &mut [IoVec], flags: i32, deadline: Option<Time>, senderRequested: bool, controlDataLen: usize)
               -> Result<(i64, i32, Option<(SockAddr, usize)>, Vec<u8>)>  {
        let trunc = flags & MsgType::MSG_TRUNC != 0;
        let peek = flags & MsgType::MSG_PEEK != 0;
        let dontWait = flags & MsgType::MSG_DONTWAIT != 0;
        let waitAll = flags & MsgType::MSG_WAITALL != 0;
        let cloexec = flags & MsgType::MSG_CMSG_CLOEXEC != 0;

        // Calculate the number of FDs for which we have space and if we are
        // requesting credentials.
        let mut wantCreds = false;
        let mut msgFlags = 0;
        let mut rightsLen = controlDataLen as isize - SIZEOF_CMSGHDR as isize;
        if self.Passcred() {
            // Credentials take priority if they are enabled and there is space.
            wantCreds = rightsLen >= 0;
            if !wantCreds {
                msgFlags |= MsgType::MSG_CTRUNC;
            }

            let credlen = CMsgSpace(SIZEOF_UCRED);
            rightsLen -= credlen as isize;
        }

        // FDs are 32 bit (4 byte) ints.
        let mut numRights = rightsLen / 4;
        if numRights < 0 {
            numRights = 0;
        }

        let mut unixAddr = SockAddrUnix::default();

        let mut total = 0;
        let mut sender = None;
        let mut outputctrls = SCMControlMessages::default();
        let mut ControlVec = self.encodeControlMsg(task, outputctrls, controlDataLen, &mut msgFlags, cloexec);
        let mut dsts = BlockSeq::ToBlocks(dsts);
        match self.ep.RecvMsg(&mut dsts, wantCreds, numRights as u64, peek, Some(&mut unixAddr)) {
            Err(Error::SysError(SysErr::EAGAIN)) => {
                if dontWait {
                    return Err(Error::SysError(SysErr::EAGAIN))
                }
            }
            Err(Error::ErrClosedForReceive) => {
                if self.IsPacket() {
                    return Err(Error::SysError(SysErr::EAGAIN))
                }
                return Ok((total, msgFlags, sender, ControlVec))
            }
            Err(e) => {
                return Err(e)
            }
            Ok((mut n, ms, ctrls, ctrunc)) => {
                sender = if senderRequested {
                    let fromLen = unixAddr.Len();
                    Some((SockAddr::Unix(unixAddr), fromLen))
                } else {
                    None
                };

                outputctrls = ctrls;
                ControlVec = self.encodeControlMsg(task, outputctrls, controlDataLen, &mut msgFlags, cloexec);
                if ctrunc {
                    msgFlags |= MsgType::MSG_CTRUNC;
                }

                let seq = BlockSeq::NewFromSlice(&dsts);

                if self.IsPacket() && n < ms {
                    msgFlags |= MsgType::MSG_TRUNC;
                }

                if trunc {
                    n = ms
                }

                if dontWait || !waitAll || self.IsPacket() || n >= seq.NumBytes() as usize {
                    if self.IsPacket() && n < ms {
                        msgFlags |= MsgType::MSG_TRUNC;
                    }

                    if trunc {
                        n = ms;
                    }

                    return Ok((n as i64, msgFlags, sender, ControlVec))
                }

                let seq = seq.DropFirst(n as u64);
                dsts = BlockSeqToIoVecs(seq);
                total += n as i64;
            }
        }

        let general = task.blocker.generalEntry.clone();
        self.EventRegister(task, &general, EVENT_READ);
        defer!(self.EventUnregister(task, &general));

        loop {
            let mut unixAddr = SockAddrUnix::default();
            match self.ep.RecvMsg(&mut dsts, wantCreds, numRights as u64, peek, Some(&mut unixAddr)) {
                Err(Error::SysError(SysErr::EAGAIN)) => (),
                Err(Error::ErrClosedForReceive) => {
                    return Ok((total as i64, msgFlags, sender, ControlVec))
                }
                Err(e) => {
                    if total > 0 {
                        return Ok((total as i64, msgFlags, sender, ControlVec))
                    }

                    return Err(e)
                },
                Ok((n, ms, ctrls, ctrunc)) => {
                    let sender = if senderRequested {
                        let fromLen = unixAddr.Len();
                        Some((SockAddr::Unix(unixAddr), fromLen))
                    } else {
                        None
                    };

                    if ctrunc {
                        msgFlags |= MsgType::MSG_CTRUNC;
                    }

                    if trunc {
                        total += ms as i64
                    } else {
                        total += n as i64
                    }

                    // todo: handle waitAll
                    let seq = BlockSeq::NewFromSlice(&dsts);
                    if waitAll && n < seq.NumBytes() as usize {
                        info!("RecvMsg get waitall, but return partial......")
                    }

                    if self.IsPacket() && n < ms {
                        msgFlags |= MsgType::MSG_TRUNC;
                    }

                    let seq = BlockSeq::NewFromSlice(&dsts);
                    if !waitAll || self.IsPacket() || n >= seq.NumBytes() as usize {
                        if self.IsPacket() && n < ms {
                            msgFlags |= MsgType::MSG_TRUNC;
                        }
                        let ControlVector = self.encodeControlMsg(task, ctrls, controlDataLen, &mut msgFlags, cloexec);
                        return Ok((total, msgFlags, sender, ControlVector))
                    }

                    let seq = seq.DropFirst(n as u64);
                    dsts = BlockSeqToIoVecs(seq);
                }
            }

            match task.blocker.BlockWithMonoTimer(true, deadline) {
                Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                    if total > 0 {
                        return Ok((total as i64, msgFlags, sender, ControlVec))
                    }
                    return Err(Error::SysError(SysErr::EAGAIN))
                }
                Err(e) => return Err(e),
                _ =>(),
            }
        }
    }

    fn SendMsg(&self, task: &Task, srcs: &[IoVec], flags: i32, msgHdr: &mut MsgHdr, deadline: Option<Time>) -> Result<i64> {
        let to: Vec<u8> = if msgHdr.msgName != 0 {
            if self.stype == SockType::SOCK_SEQPACKET {
                Vec::new()
            } else if self.stype == SockType::SOCK_STREAM {
                if self.State() == SS_CONNECTED {
                    return Err(Error::SysError(SysErr::EISCONN))
                }

                return Err(Error::SysError(SysErr::EOPNOTSUPP));
            } else {
                task.CopyIn(msgHdr.msgName, msgHdr.nameLen as usize)?
            }
        } else {
            Vec::new()
        };

        let controlVec: Vec<u8> = if msgHdr.msgControl != 0 {
            task.CopyIn(msgHdr.msgControl, msgHdr.msgControlLen as usize)?
        } else {
            Vec::new()
        };

        let toEp = if to.len() > 0 {
            let ep = ExtractEndpoint(task, &to)?;
            Some(ep)
        } else {
            None
        };

        let ctrlMsg = if controlVec.len() > 0 {
             Parse(&controlVec)?
        } else {
            ControlMessages::default()
        };

        let scmCtrlMsg = ctrlMsg.ToSCMUnix(task, &self.ep, &toEp)?;

        let n = match self.ep.SendMsg(srcs, &scmCtrlMsg, &toEp) {
            Err(Error::SysError(SysErr::EAGAIN)) => {
                if flags & MsgType::MSG_DONTWAIT != 0 {
                    return Err(Error::SysError(SysErr::EAGAIN))
                }
                0
            }
            Err(e) => return Err(e),
            Ok(n) => {
                if flags & MsgType::MSG_DONTWAIT != 0 {
                    return Ok(n as i64)
                }
                n
            },
        };

        // We'll have to block. Register for notification and keep trying to
        // send all the data.
        let general = task.blocker.generalEntry.clone();
        self.EventRegister(task, &general, EVENT_OUT);
        defer!(self.EventUnregister(task, &general));

        let mut total = n;

        let bs = BlockSeq::NewFromSlice(srcs);
        let totalLen = bs.Len();
        while total < totalLen {
            let left = bs.DropFirst(total as u64);
            let srcs = left.ToIoVecs();
            let n = match self.ep.SendMsg(&srcs, &scmCtrlMsg, &toEp) {
                Err(Error::SysError(SysErr::EAGAIN)) => {
                    0
                }
                Err(e) => {
                    if total > 0 {
                        return Ok(total as i64)
                    }
                    return Err(e)
                },
                Ok(n) => n
            };

            total += n;

            match task.blocker.BlockWithMonoTimer(true, deadline) {
                Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                    return Err(Error::SysError(SysErr::EAGAIN))
                }
                Err(e) => {
                    return Err(e);
                }
                _ => ()
            }
        }

        return Ok(total as i64)
    }

    fn SetRecvTimeout(&self, ns: i64) {
        self.recv.store(ns, Ordering::Relaxed)
    }

    fn SetSendTimeout(&self, ns: i64) {
        self.send.store(ns, Ordering::Relaxed)
    }

    fn RecvTimeout(&self) -> i64 {
        return self.recv.load(Ordering::Relaxed)
    }

    fn SendTimeout(&self) -> i64 {
        return self.send.load(Ordering::Relaxed)
    }
}

pub struct UnixSocketProvider {
}

impl Provider for UnixSocketProvider {
    fn Socket(&self, task: &Task, stype: i32, protocol: i32) -> Result<Option<Arc<File>>> {
        if protocol != 0 && protocol != AFType::AF_UNIX {
            return Err(Error::SysError(SysErr::EPROTONOSUPPORT))
        }

        let fd = HostSpace::Socket(AFType::AF_UNIX, stype, protocol) as i32;
        if fd < 0 {
            return Err(Error::SysError(-fd))
        }

        // Create the endpoint and socket.
        match stype {
            SockType::SOCK_DGRAM => {
                let ep = ConnectionLessEndPoint::New(fd);
                let ep = BoundEndpoint::ConnectLess(ep);
                return Ok(Some(Arc::new(NewUnixSocket(task, ep, stype, fd)?)))
            }
            SockType::SOCK_SEQPACKET => {
                let ep = ConnectionedEndPoint::New(stype, fd);
                let ep = BoundEndpoint::Connected(ep);
                return Ok(Some(Arc::new(NewUnixSocket(task, ep, stype, fd)?)))
            }
            SockType::SOCK_STREAM => {
                let ep = ConnectionedEndPoint::New(stype, fd);
                let ep = BoundEndpoint::Connected(ep);
                return Ok(Some(Arc::new(NewUnixSocket(task, ep, stype, fd)?)))
            }
            _ => return Err(Error::SysError(SysErr::EINVAL))
        }
    }

    fn Pair(&self, task: &Task, stype: i32, protocol: i32) -> Result<Option<(Arc<File>, Arc<File>)>> {
        if protocol != 0 && protocol != AFType::AF_UNIX {
            return Err(Error::SysError(SysErr::EPROTONOSUPPORT))
        }

        match stype {
            SockType::SOCK_STREAM => (),
            SockType::SOCK_DGRAM | SockType::SOCK_SEQPACKET => (),
            _ => return Err(Error::SysError(SysErr::EINVAL))
        }

        let fd1 = HostSpace::Socket(AFType::AF_UNIX, stype, protocol) as i32;
        if fd1 < 0 {
            return Err(Error::SysError(-fd1))
        }

        let fd2 = HostSpace::Socket(AFType::AF_UNIX, stype, protocol) as i32;
        if fd2 < 0 {
            return Err(Error::SysError(-fd2))
        }

        // Create the endpoints and sockets.
        let (ep1, ep2) = ConnectionedEndPoint::NewPair(stype, fd1, fd2);
        let ep1 = BoundEndpoint::Connected(ep1);
        let ep2 = BoundEndpoint::Connected(ep2);
        let s1 = NewUnixSocket(task, ep1, stype, fd1)?;
        let s2 = NewUnixSocket(task, ep2, stype, fd2)?;

        return Ok(Some((Arc::new(s1), Arc::new(s2))))
    }
}

pub fn Init() {
    FAMILIAES.write().RegisterProvider(AFType::AF_UNIX, Box::new(UnixSocketProvider { }))
}