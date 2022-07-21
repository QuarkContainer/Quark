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

use crate::qlib::mutex::*;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::Any;
use core::ops::*;
use core::sync::atomic::AtomicI64;
use core::sync::atomic::Ordering;

use super::super::super::fs::attr::*;
use super::super::super::fs::dentry::*;
use super::super::super::fs::dirent::*;
use super::super::super::fs::file::*;
use super::super::super::fs::flags::*;
use super::super::socket::*;
//use super::super::super::fs::attr::*;
use super::super::super::super::common::*;
use super::super::super::super::linux::socket::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::device::*;
use super::super::super::super::auth::*;
use super::super::super::fs::host::hostinodeop::*;
use super::super::super::fs::inode::*;
use super::super::super::fs::fsutil::inode::*;
use super::super::super::fs::mount::*;
use super::super::super::kernel::abstract_socket_namespace::*;
use super::super::super::kernel::kernel::GetKernel;
use super::super::super::kernel::fd_table::*;
use super::super::super::kernel::time::*;
use super::super::super::kernel::waiter::*;
use super::super::super::task::*;
//use super::super::super::super::mem::io::*;
use super::super::super::super::mem::seq::*;
use super::super::super::super::path::*;
use super::super::super::socket::control::*;
use super::super::super::socket::epsocket::epsocket::*;
use super::super::super::tcpip::tcpip::*;
use super::transport::connectioned::*;
use super::transport::connectionless::*;
use super::transport::unix::*;

pub fn NewUnixSocket(task: &Task, ep: BoundEndpoint, stype: i32) -> Result<File> {
    //assert!(family == AFType::AF_UNIX, "NewUnixSocket family is not AF_UNIX");
    let dirent = NewUnixSocketDummyDirent(task, UNIX_SOCKET_DEVICE.clone())?;
    let fileFlags = FileFlags {
        Read: true,
        Write: true,
        ..Default::default()
    };

    let file = File::New(
        &dirent,
        &fileFlags,
        UnixSocketOperations::New(ep, stype),
    );

    GetKernel().sockets.AddSocket(&file);

    return Ok(file);
}

pub fn NewUnixSocketDummyDirent(task: &Task,
                                d: Arc<QMutex<Device>>) -> Result<Dirent> {
    let ino = d.lock().NextIno();

    let iops = SimpleFileInode::New(task,
                                    &task.FileOwner(),
                                    &FilePermissions{User: PermMask::NewReadWrite(), ..Default::default()},
                                    FSMagic::SOCKFS_MAGIC,
                                    true,
                                    Dummy{});

    let deviceId = d.lock().DeviceID();
    let inodeId = d.lock().NextIno();
    let attr = StableAttr {
        Type: InodeType::Socket,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: MemoryDef::PAGE_SIZE as i64,
        DeviceFileMajor: 0,
        DeviceFileMinor: 0,
    };

    let msrc = MountSource::NewPseudoMountSource();
    let inode = Inode::New(&Arc::new(iops), &Arc::new(QMutex::new(msrc)), &attr);

    let name = format!("socket:[{}]", ino);
    return Ok(Dirent::New(&inode, &name.to_string()));
}

pub fn NewUnixSocketDirent(task: &Task,
                           ep: &BoundEndpoint) -> Result<Dirent> {
    let msrc = MountSource::NewPseudoMountSource();
    let iops = SocketInodeOps::New(task,
                                   ep,
                                   &task.FileOwner(),
                                   &FilePermissions{User: PermMask::NewReadWrite(), ..Default::default()});
    let deviceId = UNIX_SOCKET_DEVICE.lock().DeviceID();
    let inodeId = UNIX_SOCKET_DEVICE.lock().NextIno();
    let attr = StableAttr {
        Type: InodeType::Socket,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: MemoryDef::PAGE_SIZE as i64,
        DeviceFileMajor: 0,
        DeviceFileMinor: 0,
    };

    let inode = Inode::New(&Arc::new(iops), &Arc::new(QMutex::new(msrc)), &attr);

    let name = format!("socket:[{}]", inodeId);
    return Ok(Dirent::New(&inode, &name.to_string()));
}

pub fn NewUnixSocketInode(task: &Task,
                          ep: &BoundEndpoint,
                          owner: &FileOwner,
                          perms: &FilePermissions,
                          msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let iops = SocketInodeOps::New(task, ep, owner, perms);
    let deviceId = UNIX_SOCKET_DEVICE.lock().DeviceID();
    let inodeId = UNIX_SOCKET_DEVICE.lock().NextIno();
    let attr = StableAttr {
        Type: InodeType::Socket,
        DeviceId: deviceId,
        InodeId: inodeId,
        BlockSize: MemoryDef::PAGE_SIZE as i64,
        DeviceFileMajor: 0,
        DeviceFileMinor: 0,
    };

    let inode = Inode::New(&Arc::new(iops), msrc, &attr);

    return inode;
}


pub struct UnixSocketOperations {
    pub ep: BoundEndpoint,
    pub stype: i32,
    pub send: AtomicI64,
    pub recv: AtomicI64,
    pub name: QMutex<Option<Vec<u8>>>
}

impl UnixSocketOperations {
    pub fn New(ep: BoundEndpoint, stype: i32) -> Self {
        let ret = Self {
            ep: ep,
            stype: stype,
            send: AtomicI64::new(0),
            recv: AtomicI64::new(0),
            name: QMutex::new(None),
        };

        return ret;
    }

    /*pub fn State(&self) -> i32 {
        return self.ep.State();
    }*/

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
        return Ok((SockAddr::Unix(addr), l as u32));
    }

    // GetSockName implements the linux syscall getsockname(2) for sockets backed by
    // a transport.Endpoint.
    pub fn GetSockName(&self, _task: &Task) -> Result<(SockAddr, u32)> {
        let addr = self.ep.GetLocalAddress()?;

        let l = addr.Len();
        return Ok((SockAddr::Unix(addr), l as u32));
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

    fn encodeControlMsg(
        &self,
        task: &Task,
        mut ctrls: SCMControlMessages,
        controlDataLen: usize,
        mflags: &mut i32,
        cloexec: bool,
    ) -> Vec<u8> {
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
                        let (data, flags) =
                            ControlMessageCredentials::Empty().EncodeInto(controlData, *mflags);
                        *mflags = flags;
                        data
                    }
                    Some(ref creds) => {
                        let (data, flags) = creds.Credentials().EncodeInto(controlData, *mflags);
                        *mflags = flags;
                        data
                    }
                },
            }
        } else {
            controlData
        };

        let controlData = match ctrls.Rights {
            None => controlData,
            Some(ref mut rights) => {
                let maxFDs =
                    (controlData.len() as isize - SIZE_OF_CONTROL_MESSAGE_HEADER as isize) / 4;
                if maxFDs < 0 {
                    *mflags |= MsgType::MSG_CTRUNC;
                    controlData
                } else {
                    let (fds, trunc) = rights.RightsFDs(task, cloexec, maxFDs as usize);
                    if trunc {
                        *mflags |= MsgType::MSG_CTRUNC;
                    }
                    let (controlData, _) =
                        ControlMessageRights(fds).EncodeInto(controlData, *mflags);
                    controlData
                }
            }
        };

        let new_size = controlDataLen - controlData.len();
        controlVec.resize(new_size, 0);
        return controlVec;
    }
}

impl Drop for UnixSocketOperations {
    fn drop(&mut self) {
       match *self.name.lock() {
            None => (),
            Some(ref name) => {
                if name[0] == 0 {
                    ABSTRACT_SOCKET.Remove(name, &self.ep);
                } else {
                    UNIX_SOCKET_PINS.Unpin(name);
                }

            }
        }

        self.ep.Close();
    }
}

impl Passcred for UnixSocketOperations {
    fn Passcred(&self) -> bool {
        return self.ep.Passcred();
    }
}

impl ConnectedPasscred for UnixSocketOperations {
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

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
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
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if p[p.len() - 1] == '/' as u8 {
        // Weird, they tried to bind '/a/b/c/'?
        return Err(Error::SysError(SysErr::EISDIR));
    }

    return Ok(p);
}

impl SpliceOperations for UnixSocketOperations {}

impl FileOperations for UnixSocketOperations {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::UnixSocketOperations;
    }

    fn Seekable(&self) -> bool {
        return false;
    }

    fn Seek(
        &self,
        _task: &Task,
        _f: &File,
        _whence: i32,
        _current: i64,
        _offset: i64,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::ESPIPE));
    }

    fn ReadDir(
        &self,
        _task: &Task,
        _f: &File,
        _offset: i64,
        _serializer: &mut DentrySerializer,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn ReadAt(
        &self,
        _task: &Task,
        _f: &File,
        dsts: &mut [IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        let count = IoVec::NumBytes(dsts);

        if count == 0 {
            return Ok(0);
        }

        match self.ep.RecvMsg(dsts, false, 0, false, None) {
            Err(Error::ErrClosedForReceive) => {
                // todo: fix this. How to handle ErrClosedForReceive, If again, Read will wait for it
                /*if self.IsPacket() {
                    return Err(Error::SysError(SysErr::EAGAIN))
                }*/
                return Ok(0);
            }
            Err(e) => return Err(e),
            Ok((n, _, _, _)) => return Ok(n as i64),
        }
    }

    fn WriteAt(
        &self,
        task: &Task,
        _f: &File,
        srcs: &[IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        let ConnectedPasscred = self.ep.ConnectedPasscred();
        let Passcred = self.ep.Passcred();
        let ctrl = if ConnectedPasscred || Passcred {
            NewControlMessage(task, Some(self.ep.clone()), None)
        } else {
            SCMControlMessages::default()
        };

        let count = IoVec::NumBytes(srcs);

        if count == 0 {
            let nInt = self.ep.SendMsg(srcs, &ctrl, &None)?;
            return Ok(nInt as i64);
        }

        match self.ep.SendMsg(srcs, &ctrl, &None) {
            Err(e) => return Err(e),
            Ok(n) => return Ok(n as i64),
        }
    }

    fn Append(&self, task: &Task, f: &File, srcs: &[IoVec]) -> Result<(i64, i64)> {
        let n = self.WriteAt(task, f, srcs, 0, false)?;
        return Ok((n, 0));
    }

    fn Fsync(
        &self,
        _task: &Task,
        _f: &File,
        _start: i64,
        _end: i64,
        _syncType: SyncType,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn Flush(&self, _task: &Task, _f: &File) -> Result<()> {
        return Ok(());
    }

    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        let inode = f.Dirent.Inode();
        return inode.UnstableAttr(task);
    }

    fn Ioctl(&self, task: &Task, _f: &File, fd: i32, request: u64, val: u64) -> Result<()> {
        return Ioctl(task, &self.ep, fd, request, val);
    }

    fn IterateDir(
        &self,
        _task: &Task,
        _d: &Dirent,
        _dirCtx: &mut DirCtx,
        _offset: i32,
    ) -> (i32, Result<i64>) {
        return (0, Err(Error::SysError(SysErr::ENOTDIR)));
    }

    fn Mappable(&self) -> Result<MMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
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
        let ep = match ABSTRACT_SOCKET.BoundEndpoint(&path) {
            None => return Err(Error::SysError(SysErr::ECONNREFUSED)),
            Some(ep) => ep,
        };

        return Ok(ep);
    }

    let path = String::from_utf8(path).unwrap();

    // Find the node in the filesystem.
    let root = task.fsContext.RootDirectory();
    let cwd = task.fsContext.WorkDirectory();
    let mut remainingTraversals = 10; //DefaultTraversalLimit
    let mns = task.mountNS.clone();
    let d = mns.FindDirent(
        task,
        &root,
        Some(cwd),
        &path,
        &mut remainingTraversals,
        true,
    )?;

    // Extract the endpoint if one is there.
    let inode = d.Inode();
    let iops = inode.lock().InodeOp.clone();

    match iops.as_any().downcast_ref::<SocketInodeOps>() {
        None => return Err(Error::SysError(SysErr::ECONNREFUSED)),
        Some(iops) => {
            return Ok(iops.ep.clone());
        }
    }
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
                    return Err(Error::SysError(SysErr::ECONNREFUSED));
                } else {
                    return Err(Error::SysError(SysErr::EPROTOTYPE));
                }
            }
            Err(e) => {
                return Err(e)
            },
            _ => (),
        }
        return Ok(0);
    }

    // Accept implements the linux syscall accept(2) for sockets backed by
    // a transport.Endpoint.
    fn Accept(
        &self,
        task: &Task,
        addr: &mut [u8],
        addrlen: &mut u32,
        flags: i32,
        blocking: bool,
    ) -> Result<i64> {
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

        let ns = NewUnixSocket(task, ep, self.stype)?;
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
            ABSTRACT_SOCKET.Bind(p.clone(), &bep)?;
            *(self.name.lock()) = Some(p);
        } else {
            let p = String::from_utf8(p).unwrap();
            
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
                d = task.mountNS.clone().FindDirent(
                    task,
                    &root,
                    Some(cwd),
                    &subpath.to_string(),
                    &mut remainingTraversals,
                    true,
                )?;
                name = &p[lastSlash as usize + 1..];
            }

            // Create the socket.
            /*let permisson = FilePermissions {
                User: PermMask {
                    read: true,
                    ..Default::default()
                },
                ..Default::default()
            };*/

            // todo: check for right solution

            let permisson = FilePermissions {
                User: PermMask {
                    write: true,
                    read: true,
                    ..Default::default()
                },
                Group: PermMask {
                    write: true,
                    read: true,
                    ..Default::default()
                },
                Other: PermMask {
                    write: true,
                    read: true,
                    ..Default::default()
                },
                ..Default::default()
            };


            match d.Bind(task, &root, &name.to_string(), &bep, &permisson) {
                Err(_) => return Err(Error::SysError(SysErr::EADDRINUSE)),
                Ok(childDirent) => {
                    let dir = d.MyFullName();
                    let fullname = format!("{}/{}", dir, name);
                    UNIX_SOCKET_PINS.Pin(fullname.as_bytes().to_vec(), &childDirent);
                    *(self.name.lock()) = Some(fullname.as_bytes().to_vec());
                },
            }
        }

        return Ok(0);
    }

    fn Listen(&self, _task: &Task, backlog: i32) -> Result<i64> {
        self.ep.Listen(backlog)?;
        return Ok(0);
    }

    fn Shutdown(&self, _task: &Task, how: i32) -> Result<i64> {
        let f = ConvertShutdown(how)?;

        self.ep.Shutdown(f)?;
        return Ok(0);
    }

    fn GetSockOpt(&self, task: &Task, level: i32, name: i32, opt: &mut [u8]) -> Result<i64> {
        let ret = GetSockOpt(
            task,
            self,
            &self.ep,
            AFType::AF_UNIX,
            self.ep.Type(),
            level,
            name,
            opt.len(),
        )?;
        let size = ret.Marsh(opt)?;
        return Ok(size as i64);
    }

    fn SetSockOpt(&self, task: &Task, level: i32, name: i32, opt: &[u8]) -> Result<i64> {
        SetSockOpt(task, self, &self.ep, level, name, opt)?;
        return Ok(0);
    }

    fn GetSockName(&self, _task: &Task, socketaddr: &mut [u8]) -> Result<i64> {
        let addr = self.ep.GetLocalAddress()?;

        let l = addr.Len();
        SockAddr::Unix(addr).Marsh(socketaddr, l)?;

        return Ok(l as i64);
    }

    fn GetPeerName(&self, _task: &Task, socketaddr: &mut [u8]) -> Result<i64> {
        let addr = self.ep.GetRemoteAddress()?;

        let l = addr.Len();
        SockAddr::Unix(addr).Marsh(socketaddr, l as usize)?;

        return Ok(l as i64);
    }

    fn RecvMsg(
        &self,
        task: &Task,
        dsts: &mut [IoVec],
        flags: i32,
        deadline: Option<Time>,
        senderRequested: bool,
        controlDataLen: usize,
    ) -> Result<(i64, i32, Option<(SockAddr, usize)>, Vec<u8>)> {
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

        let mut total: i64 = 0;
        let mut sender = None;
        let mut outputctrls = SCMControlMessages::default();
        let mut ControlVec =
            self.encodeControlMsg(task, outputctrls, controlDataLen, &mut msgFlags, cloexec);
        let size = IoVec::NumBytes(dsts);
        let buf = DataBuff::New(size);

        let mut bs = BlockSeqToIoVecs(buf.BlockSeq());
        match self.ep.RecvMsg(
            &mut bs,
            wantCreds,
            numRights as u64,
            peek,
            Some(&mut unixAddr),
        ) {
            Err(Error::SysError(SysErr::EAGAIN)) => {
                if dontWait {
                    return Err(Error::SysError(SysErr::EAGAIN));
                }
            }
            Err(Error::ErrClosedForReceive) => {
                if self.IsPacket() {
                    return Err(Error::SysError(SysErr::EAGAIN));
                }
                let len = task.CopyDataOutToIovs(&buf.buf[0..total as usize], dsts, false)?;
                return Ok((len as i64, msgFlags, sender, ControlVec));
            }
            Err(e) => return Err(e),
            Ok((mut n, ms, ctrls, ctrunc)) => {
                sender = if senderRequested {
                    let fromLen = unixAddr.Len();
                    Some((SockAddr::Unix(unixAddr), fromLen))
                } else {
                    None
                };

                outputctrls = ctrls;
                ControlVec = self.encodeControlMsg(
                    task,
                    outputctrls,
                    controlDataLen,
                    &mut msgFlags,
                    cloexec,
                );
                if ctrunc {
                    msgFlags |= MsgType::MSG_CTRUNC;
                }

                let seq = BlockSeq::NewFromSlice(&bs);

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

                    let count = if (n as usize) < buf.buf.len() {
                        n as usize
                    } else {
                        buf.buf.len()
                    };

                    let _len = task.CopyDataOutToIovs(&buf.buf[0..count as usize], dsts, false)?;
                    return Ok((n as i64, msgFlags, sender, ControlVec));
                }

                let seq = seq.DropFirst(n as u64);
                bs = BlockSeqToIoVecs(seq);
                total += n as i64;
            }
        }

        let general = task.blocker.generalEntry.clone();
        self.EventRegister(task, &general, EVENT_READ);
        defer!(self.EventUnregister(task, &general));

        loop {
            let mut unixAddr = SockAddrUnix::default();
            match self.ep.RecvMsg(
                &mut bs,
                wantCreds,
                numRights as u64,
                peek,
                Some(&mut unixAddr),
            ) {
                Err(Error::SysError(SysErr::EAGAIN)) => (),
                Err(Error::ErrClosedForReceive) => {
                    let count = if total > buf.buf.len() as i64 {
                        buf.buf.len() as i64
                    } else {
                        total
                    };
                    if trunc {
                        return Ok((total as i64, msgFlags, sender, ControlVec));
                    }
                    let len = task.CopyDataOutToIovs(&buf.buf[0..count as usize], dsts, false)?;
                    return Ok((len as i64, msgFlags, sender, ControlVec));
                }
                Err(e) => {
                    if total > 0 {
                        let count = if total > buf.buf.len() as i64 {
                            buf.buf.len() as i64
                        } else {
                            total
                        };
                        if trunc {
                            return Ok((total as i64, msgFlags, sender, ControlVec));
                        }
                        let len = task.CopyDataOutToIovs(&buf.buf[0..count as usize], dsts, false)?;
                        return Ok((len as i64, msgFlags, sender, ControlVec));
                    }

                    return Err(e);
                }
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
                    let seq = BlockSeq::NewFromSlice(&bs);
                    if waitAll && n < seq.NumBytes() as usize {
                        info!("RecvMsg get waitall, but return partial......")
                    }

                    if self.IsPacket() && n < ms {
                        msgFlags |= MsgType::MSG_TRUNC;
                    }

                    let seq = BlockSeq::NewFromSlice(&bs);
                    if !waitAll || self.IsPacket() || n >= seq.NumBytes() as usize {
                        if self.IsPacket() && n < ms {
                            msgFlags |= MsgType::MSG_TRUNC;
                        }
                        let ControlVector = self.encodeControlMsg(
                            task,
                            ctrls,
                            controlDataLen,
                            &mut msgFlags,
                            cloexec,
                        );
                        let count = if total > buf.buf.len() as i64 {
                            buf.buf.len() as i64
                        } else {
                            total
                        };
                        let len = task.CopyDataOutToIovs(&buf.buf[0..count as usize], dsts, false)?;

                        if trunc {
                            return Ok((total as i64, msgFlags, sender, ControlVector));
                        }
                        return Ok((len as i64, msgFlags, sender, ControlVector));
                    }

                    let seq = seq.DropFirst(n as u64);
                    bs = BlockSeqToIoVecs(seq);
                }
            }

            match task.blocker.BlockWithMonoTimer(true, deadline) {
                Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                    if total > 0 {
                        let len = task.CopyDataOutToIovs(&buf.buf[0..total as usize], dsts, false)?;
                        return Ok((len as i64, msgFlags, sender, ControlVec));
                    }
                    return Err(Error::SysError(SysErr::EAGAIN));
                }
                Err(e) => return Err(e),
                _ => (),
            }
        }
    }

    fn SendMsg(
        &self,
        task: &Task,
        srcs: &[IoVec],
        flags: i32,
        msgHdr: &mut MsgHdr,
        deadline: Option<Time>,
    ) -> Result<i64> {
        let to: Vec<u8> = if msgHdr.msgName != 0 {
            if self.stype == SockType::SOCK_SEQPACKET {
                Vec::new()
            } else if self.stype == SockType::SOCK_STREAM {
                if self.State() == SS_CONNECTED as u32 {
                    return Err(Error::SysError(SysErr::EISCONN));
                }

                return Err(Error::SysError(SysErr::EOPNOTSUPP));
            } else {
                task.CopyInVec(msgHdr.msgName, msgHdr.nameLen as usize)?
            }
        } else {
            Vec::new()
        };

        let controlVec: Vec<u8> = if msgHdr.msgControl != 0 {
            task.CopyInVec(msgHdr.msgControl, msgHdr.msgControlLen as usize)?
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

        let size = IoVec::NumBytes(srcs);
        let mut buf = DataBuff::New(size);
        let len = task.CopyDataInFromIovs(&mut buf.buf, srcs, true)?;
        let n = match self.ep.SendMsg(&buf.Iovs(len), &scmCtrlMsg, &toEp) {
            Err(Error::SysError(SysErr::EAGAIN)) => {
                if flags & MsgType::MSG_DONTWAIT != 0 {
                    return Err(Error::SysError(SysErr::EAGAIN));
                }
                0
            }
            Err(e) => return Err(e),
            Ok(n) => {
                if flags & MsgType::MSG_DONTWAIT != 0 {
                    return Ok(n as i64);
                }
                n
            }
        };

        // only send control message once
        let scmCtrlMsg = SCMControlMessages::default();

        // We'll have to block. Register for notification and keep trying to
        // send all the data.
        let general = task.blocker.generalEntry.clone();
        self.EventRegister(task, &general, WRITEABLE_EVENT);
        defer!(self.EventUnregister(task, &general));

        let mut total = n;

        let bs = buf.BlockSeq();
        let totalLen = bs.Len();
        while total < totalLen {
            let left = bs.DropFirst(total as u64);
            let srcs = left.ToIoVecs();
            let n = match self.ep.SendMsg(&srcs, &scmCtrlMsg, &toEp) {
                Err(Error::SysError(SysErr::EAGAIN)) => 0,
                Err(e) => {
                    if total > 0 {
                        return Ok(total as i64);
                    }
                    return Err(e);
                }
                Ok(n) => n,
            };

            total += n;

            match task.blocker.BlockWithMonoTimer(true, deadline) {
                Err(Error::SysError(SysErr::ETIMEDOUT)) => {
                    return Err(Error::SysError(SysErr::EAGAIN))
                }
                Err(e) => {
                    return Err(e);
                }
                _ => (),
            }
        }

        return Ok(total as i64);
    }

    fn SetRecvTimeout(&self, ns: i64) {
        self.recv.store(ns, Ordering::Relaxed)
    }

    fn SetSendTimeout(&self, ns: i64) {
        self.send.store(ns, Ordering::Relaxed)
    }

    fn RecvTimeout(&self) -> i64 {
        return self.recv.load(Ordering::Relaxed);
    }

    fn SendTimeout(&self) -> i64 {
        return self.send.load(Ordering::Relaxed);
    }

    fn State(&self) -> u32 {
        return self.ep.State() as u32
    }

    fn Type(&self) -> (i32, i32, i32) {
        return (AFType::AF_UNIX, self.stype, 0)
    }
}

pub struct Dummy{}

impl SimpleFileTrait for Dummy {}


pub struct SocketInodeOps {
    pub ep: BoundEndpoint,
    pub simpleAttributes: Arc<InodeSimpleAttributes>,
    pub simpleExtendedAttribute: Arc<InodeSimpleExtendedAttributes>,
}

impl SocketInodeOps {
    pub fn New(
        task: &Task,
        ep: &BoundEndpoint,
        owner: &FileOwner,
        perms: &FilePermissions,
    ) -> Self {
        return Self {
            ep: ep.clone(),
            simpleAttributes: Arc::new(InodeSimpleAttributes::New(
                task,
                owner,
                perms,
                FSMagic::SOCKFS_MAGIC,
            )),
            simpleExtendedAttribute: Arc::new(InodeSimpleExtendedAttributes::default()),
        };
    }
}

impl InodeOperations for SocketInodeOps {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn IopsType(&self) -> IopsType {
        return IopsType::SocketInodeOps;
    }

    fn InodeType(&self) -> InodeType {
        return InodeType::Socket;
    }

    fn InodeFileType(&self) -> InodeFileType {
        return InodeFileType::Socket;
    }

    fn WouldBlock(&self) -> bool {
        return true;
    }

    fn Lookup(&self, _task: &Task, _dir: &Inode, _name: &str) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn Create(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _name: &str,
        _flags: &FileFlags,
        _perm: &FilePermissions,
    ) -> Result<File> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn CreateDirectory(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _name: &str,
        _perm: &FilePermissions,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn Bind(
        &self,
        _task: &Task,
        _dir: &Inode,
        _name: &str,
        _data: &BoundEndpoint,
        _perms: &FilePermissions,
    ) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn BoundEndpoint(&self, _task: &Task, _inode: &Inode, _path: &str) -> Option<BoundEndpoint> {
        return Some(self.ep.clone());
    }

    fn CreateLink(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _oldname: &str,
        _newname: &str,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn CreateHardLink(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _target: &Inode,
        _name: &str,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn CreateFifo(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _name: &str,
        _perm: &FilePermissions,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn Remove(&self, _task: &Task, _dir: &mut Inode, _name: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn RemoveDirectory(&self, _task: &Task, _dir: &mut Inode, _name: &str) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn Rename(
        &self,
        _task: &Task,
        _dir: &mut Inode,
        _oldParent: &Inode,
        _oldname: &str,
        _newParent: &Inode,
        _newname: &str,
        _replacement: bool,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    fn GetFile(
        &self,
        _task: &Task,
        _dir: &Inode,
        _dirent: &Dirent,
        _flags: FileFlags,
    ) -> Result<File> {
        return Err(Error::SysError(SysErr::ENXIO))
    }

    fn UnstableAttr(&self, _task: &Task) -> Result<UnstableAttr> {
        let u = self.simpleAttributes.read().unstable;
        return Ok(u);
    }

    fn Getxattr(&self, dir: &Inode, name: &str, size: usize) -> Result<Vec<u8>> {
        return self.simpleExtendedAttribute.Getxattr(dir, name, size);
    }

    fn Setxattr(&self, dir: &mut Inode, name: &str, value: &[u8], flags: u32) -> Result<()> {
        return self.simpleExtendedAttribute.Setxattr(dir, name, value, flags);
    }

    fn Listxattr(&self, dir: &Inode, size: usize) -> Result<Vec<String>> {
        return self.simpleExtendedAttribute.Listxattr(dir, size);
    }

    fn Check(&self, task: &Task, inode: &Inode, reqPerms: &PermMask) -> Result<bool> {
        return ContextCanAccessFile(task, inode, reqPerms);
    }

    fn SetPermissions(&self, task: &Task, _dir: &mut Inode, p: FilePermissions) -> bool {
        self.simpleAttributes
            .write()
            .unstable
            .SetPermissions(task, &p);
        return true;
    }

    fn SetOwner(&self, task: &Task, _dir: &mut Inode, owner: &FileOwner) -> Result<()> {
        self.simpleAttributes.write().unstable.SetOwner(task, owner);
        return Ok(());
    }

    fn SetTimestamps(&self, task: &Task, _dir: &mut Inode, ts: &InterTimeSpec) -> Result<()> {
        self.simpleAttributes
            .write()
            .unstable
            .SetTimestamps(task, ts);
        return Ok(());
    }

    fn Truncate(&self, _task: &Task, _dir: &mut Inode, _size: i64) -> Result<()> {
        return Ok(());
    }

    fn Allocate(&self, _task: &Task, _dir: &mut Inode, _offset: i64, _length: i64) -> Result<()> {
        return Ok(());
    }

    fn ReadLink(&self, _task: &Task, _dir: &Inode) -> Result<String> {
        return Err(Error::SysError(SysErr::ENOLINK));
    }

    fn GetLink(&self, _task: &Task, _dir: &Inode) -> Result<Dirent> {
        return Err(Error::SysError(SysErr::ENOLINK));
    }

    fn AddLink(&self, _task: &Task) {
        self.simpleAttributes.write().unstable.Links += 1;
    }

    fn DropLink(&self, _task: &Task) {
        self.simpleAttributes.write().unstable.Links -= 1;
    }

    fn IsVirtual(&self) -> bool {
        return true;
    }

    fn Sync(&self) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOSYS));
    }

    fn StatFS(&self, _task: &Task) -> Result<FsInfo> {
        return Err(Error::SysError(SysErr::ENOSYS));
    }

    fn Mappable(&self) -> Result<MMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

pub struct UnixSocketProvider {}

impl Provider for UnixSocketProvider {
    fn Socket(&self, task: &Task, stype: i32, protocol: i32) -> Result<Option<Arc<File>>> {
        if protocol != 0 && protocol != AFType::AF_UNIX {
            return Err(Error::SysError(SysErr::EPROTONOSUPPORT));
        }

        // Create the endpoint and socket.
        match stype {
            SockType::SOCK_DGRAM => {
                let ep = ConnectionLessEndPoint::New();
                let ep = BoundEndpoint::ConnectLess(ep);
                return Ok(Some(Arc::new(NewUnixSocket(task, ep, stype)?)));
            }
            SockType::SOCK_SEQPACKET => {
                let ep = ConnectionedEndPoint::New(stype);
                let ep = BoundEndpoint::Connected(ep);
                return Ok(Some(Arc::new(NewUnixSocket(task, ep, stype)?)));
            }
            SockType::SOCK_STREAM => {
                let ep = ConnectionedEndPoint::New(stype);
                let ep = BoundEndpoint::Connected(ep);
                return Ok(Some(Arc::new(NewUnixSocket(task, ep, stype)?)));
            }
            _ => return Err(Error::SysError(SysErr::EINVAL)),
        }
    }

    fn Pair(
        &self,
        task: &Task,
        stype: i32,
        protocol: i32,
    ) -> Result<Option<(Arc<File>, Arc<File>)>> {
        if protocol != 0 && protocol != AFType::AF_UNIX {
            return Err(Error::SysError(SysErr::EPROTONOSUPPORT));
        }

        match stype {
            SockType::SOCK_STREAM => (),
            SockType::SOCK_DGRAM | SockType::SOCK_SEQPACKET => (),
            _ => return Err(Error::SysError(SysErr::EINVAL)),
        }

        // Create the endpoints and sockets.
        let (ep1, ep2) = ConnectionedEndPoint::NewPair(stype);
        let ep1 = BoundEndpoint::Connected(ep1);
        let ep2 = BoundEndpoint::Connected(ep2);
        let s1 = NewUnixSocket(task, ep1, stype)?;
        let s2 = NewUnixSocket(task, ep2, stype)?;

        return Ok(Some((Arc::new(s1), Arc::new(s2))));
    }
}

pub fn Init() {
    FAMILIAES
        .write()
        .RegisterProvider(AFType::AF_UNIX, Box::new(UnixSocketProvider {}))
}
