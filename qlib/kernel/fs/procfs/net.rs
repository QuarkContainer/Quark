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
use alloc::collections::btree_map::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::string::ToString;
use alloc::sync::Arc;

use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::linux::time::*;
use super::super::super::super::kernel::kernel::kernel::GetKernel;
use super::super::super::super::kernel::socket::control::*;
use super::super::super::super::kernel::socket::unix::unix::*;
use super::super::super::tcpip::tcpip::*;
use super::super::super::task::*;
use super::super::super::socket::unix::transport::unix::*;
use super::super::attr::*;
use super::super::fsutil::inode::simple_file_inode::*;
use super::super::fsutil::file::readonly_file::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::inode::*;
use super::super::mount::*;
use super::super::ramfs::dir::*;
use super::dir_proc::*;
use super::inode::*;

pub struct NetDirNode {}

impl DirDataNode for NetDirNode {
    fn Lookup(&self, d: &Dir, task: &Task, dir: &Inode, name: &str) -> Result<Dirent> {
        return d.Lookup(task, dir, name);
    }

    fn GetFile(
        &self,
        d: &Dir,
        task: &Task,
        dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        return d.GetFile(task, dir, dirent, flags);
    }
}

pub fn NewNetDir(
    task: &Task,
    msrc: &Arc<QMutex<MountSource>>,
) -> Inode {
    let mut contents = BTreeMap::new();

    // The following files are simple stubs until they are
    // implemented in netstack, if the file contains a
    // header the stub is just the header otherwise it is
    // an empty file.
    let arp = "IP address       HW type     Flags       HW address            Mask     Device\n";
    contents.insert("arp".to_string(), NewStaticProcInode(task, msrc, &Arc::new(arp.as_bytes().to_vec())));

    let netlink = "sk       Eth Pid    Groups   Rmem     Wmem     Dump     Locks     Drops     Inode\n";
    contents.insert("netlink".to_string(), NewStaticProcInode(task, msrc, &Arc::new(netlink.as_bytes().to_vec())));

    let netstat = "TcpExt: SyncookiesSent SyncookiesRecv SyncookiesFailed EmbryonicRsts PruneCalled RcvPruned OfoPruned OutOfWindowIcmps LockDroppedIcmps ArpFilter TW TWRecycled TWKilled PAWSPassive PAWSActive PAWSEstab DelayedACKs DelayedACKLocked DelayedACKLost ListenOverflows ListenDrops TCPPrequeued TCPDirectCopyFromBacklog TCPDirectCopyFromPrequeue TCPPrequeueDropped TCPHPHits TCPHPHitsToUser TCPPureAcks TCPHPAcks TCPRenoRecovery TCPSackRecovery TCPSACKReneging TCPFACKReorder TCPSACKReorder TCPRenoReorder TCPTSReorder TCPFullUndo TCPPartialUndo TCPDSACKUndo TCPLossUndo TCPLostRetransmit TCPRenoFailures TCPSackFailures TCPLossFailures TCPFastRetrans TCPForwardRetrans TCPSlowStartRetrans TCPTimeouts TCPLossProbes TCPLossProbeRecovery TCPRenoRecoveryFail TCPSackRecoveryFail TCPSchedulerFailed TCPRcvCollapsed TCPDSACKOldSent TCPDSACKOfoSent TCPDSACKRecv TCPDSACKOfoRecv TCPAbortOnData TCPAbortOnClose TCPAbortOnMemory TCPAbortOnTimeout TCPAbortOnLinger TCPAbortFailed TCPMemoryPressures TCPSACKDiscard TCPDSACKIgnoredOld TCPDSACKIgnoredNoUndo TCPSpuriousRTOs TCPMD5NotFound TCPMD5Unexpected TCPMD5Failure TCPSackShifted TCPSackMerged TCPSackShiftFallback TCPBacklogDrop TCPMinTTLDrop TCPDeferAcceptDrop IPReversePathFilter TCPTimeWaitOverflow TCPReqQFullDoCookies TCPReqQFullDrop TCPRetransFail TCPRcvCoalesce TCPOFOQueue TCPOFODrop TCPOFOMerge TCPChallengeACK TCPSYNChallenge TCPFastOpenActive TCPFastOpenActiveFail TCPFastOpenPassive TCPFastOpenPassiveFail TCPFastOpenListenOverflow TCPFastOpenCookieReqd TCPSpuriousRtxHostQueues BusyPollRxPackets TCPAutoCorking TCPFromZeroWindowAdv TCPToZeroWindowAdv TCPWantZeroWindowAdv TCPSynRetrans TCPOrigDataSent TCPHystartTrainDetect TCPHystartTrainCwnd TCPHystartDelayDetect TCPHystartDelayCwnd TCPACKSkippedSynRecv TCPACKSkippedPAWS TCPACKSkippedSeq TCPACKSkippedFinWait2 TCPACKSkippedTimeWait TCPACKSkippedChallenge TCPWinProbe TCPKeepAlive TCPMTUPFail TCPMTUPSuccess\n\n";
    contents.insert("netstat".to_string(), NewStaticProcInode(task, msrc, &Arc::new(netstat.as_bytes().to_vec())));

    let packet = "sk       RefCnt Type Proto  Iface R Rmem   User   Inode\n";
    contents.insert("packet".to_string(), NewStaticProcInode(task, msrc, &Arc::new(packet.as_bytes().to_vec())));

    let protocols = "protocol  size sockets  memory press maxhdr  slab module     cl co di ac io in de sh ss gs se re sp bi br ha uh gp em\n";
    contents.insert("protocols".to_string(), NewStaticProcInode(task, msrc, &Arc::new(protocols.as_bytes().to_vec())));

    // Linux sets psched values to: nsec per usec, psched
    // tick in ns, 1000000, high res timer ticks per sec
    // (ClockGetres returns 1ns resolution).
    let psched = format!("{:08x} {:08x} {:08x} {:08x}\n", MICROSECOND/NANOSECOND, 64, 1000000, SECOND/NANOSECOND);
    contents.insert("psched".to_string(), NewStaticProcInode(task, msrc, &Arc::new(psched.as_bytes().to_vec())));

    let ptype = "Type Device      Function\n";
    contents.insert("ptype".to_string(), NewStaticProcInode(task, msrc, &Arc::new(ptype.as_bytes().to_vec())));

    contents.insert("tcp".to_string(), NewNetTCP(task, msrc));
    contents.insert("udp".to_string(), NewNetUDP(task, msrc));
    contents.insert("unix".to_string(), NewNetUnix(task, msrc));

    let taskDir = DirNode {
        dir: Dir::New(
            task,
            contents,
            &ROOT_OWNER,
            &FilePermissions::FromMode(FileMode(0o0555)),
        ),
        data: NetDirNode {},
    };

    return NewProcInode(
        &Arc::new(taskDir),
        msrc,
        InodeType::SpecialDirectory,
        None,
    );
}

pub fn NetworkToHost16(n: u16) -> u16 {
    let low = n & 0xff;
    let high = (n >> 8) & 0xff;
    return (low << 8) | high;
}

pub fn WriteInetAddr(addr: &SockAddr) -> String {
    match addr {
        SockAddr::Inet(addr) => {
            let ipAddr = unsafe {
                *(&addr.Addr[0] as * const _ as * const u32)
            };
            return format!("{:08X}:{:04X} ", ipAddr, NetworkToHost16(addr.Port))
        }
        SockAddr::Inet6(addr) => {
            let ipAddr0 = unsafe {
                *(&addr.Addr[0] as * const _ as * const u32)
            };
            let ipAddr1 = unsafe {
                *(&addr.Addr[4] as * const _ as * const u32)
            };
            let ipAddr2 = unsafe {
                *(&addr.Addr[8] as * const _ as * const u32)
            };
            let ipAddr3 = unsafe {
                *(&addr.Addr[12] as * const _ as * const u32)
            };
            return format!("{:08X}{:08X}{:08X}{:08X}:{:04X} ",
                           ipAddr0, ipAddr1, ipAddr2, ipAddr3, NetworkToHost16(addr.Port))
        }
        _ => panic!("WriteInetAddr doesn't support address {:x?}", addr)
    }
}

pub struct NetTCP {}

impl SimpleFileTrait for NetTCP {
    fn GetFile(
        &self,
        _task: &Task,
        _dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        let fops = ReadonlyFileOperations {
            node: NetTCPReadonlyFileNode {}.into(),
        };

        let file = File::New(dirent, &flags, fops.into());
        return Ok(file);
    }
}

pub fn NewNetTCP(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let node = SimpleFileInode::New(
        task,
        &ROOT_OWNER,
        &FilePermissions {
            User: PermMask {
                read: true,
                write: false,
                execute: false,
            },
            Group: PermMask {
                read: true,
                write: false,
                execute: false,
            },
            Other: PermMask {
                read: true,
                write: false,
                execute: false,
            },
            ..Default::default()
        },
        FSMagic::ANON_INODE_FS_MAGIC,
        false,
        NetTCP {},
    );

    return NewProcInode(&Arc::new(node), msrc, InodeType::SpecialFile, None);
}


pub struct NetTCPReadonlyFileNode {}

impl ReadonlyFileNodeTrait for NetTCPReadonlyFileNode {
    fn ReadAt(
        &self,
        task: &Task,
        _f: &File,
        dsts: &mut [IoVec],
        offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        if offset < 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let header = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode                                                     \n";
        let bytes = GetData(task, AFType::AF_INET, header)?;
        if offset as usize > bytes.len() {
            return Ok(0);
        }

        let n = task.CopyDataOutToIovs(&bytes[offset as usize..], dsts, true)?;

        return Ok(n as i64);
    }
}

pub fn GetData(task: &Task, fa: i32, header: &str) -> Result<Vec<u8>> {
    let kernel = GetKernel();
    let sockets = kernel.sockets.ListSockets();

    let mut buf = header.to_string();
    for (id, file) in sockets {
        let fopsType = file.FileOp.FopsType();
        //&& fopsType != FileOpsType::UnixSocketOperations
        if fopsType != FileOpsType::SocketOperations {
            continue
        }

        let sockops = file.FileOp.clone();
        /*let sockops = fileops
            .as_any()
            .downcast_ref::<SocketOperations>()
            .expect("SocketOperations convert fail")
            .clone();

        let family = sockops.family;
        let stype = sockops.stype;*/

        let (family, stype, _protocol) = sockops.Type();

        if fa != family || stype != SockType::SOCK_STREAM {
            continue;
        }

        // Linux's documentation for the fields below can be found at
        // https://www.kernel.org/doc/Documentation/networking/proc_net_tcp.txt.
        // For Linux's implementation, see net/ipv4/tcp_ipv4.c:get_tcp4_sock().
        // Note that the header doesn't contain labels for all the fields.

        // Field: sl; entry number.
        buf += &format!("{:>4}: ", id);

        let mut sockBuf = [0; 256];

        // Field: local_adddress.
        let _ = sockops.GetSockName(task, &mut sockBuf)?;
        let addr = GetAddr(AFType::AF_INET as _, &sockBuf)?;
        buf += &WriteInetAddr(&addr);

        // Field: rem_address.
        let addr = {
            match sockops.GetPeerName(task, &mut sockBuf) {
                Err(_) => {
                    let inetAddr = SockAddrInet {
                        Family: family as _,
                        ..Default::default()
                    };
                    SockAddr::Inet(inetAddr)
                }
                Ok(_) => {
                    GetAddr(AFType::AF_INET as _, &sockBuf)?
                }
            }
        };

        buf += &WriteInetAddr(&addr);

        // Field: state; socket state.
        buf += &format!("{:02X} ", sockops.State());

        // Field: tx_queue, rx_queue; number of packets in the transmit and
        // receive queue. Unimplemented.
        buf += &format!("{:08X}:{:08X} ", 0, 0);

        // Field: tr, tm->when; timer active state and number of jiffies
        // until timer expires. Unimplemented.
        buf += &format!("{:02X}:{:08X} ", 0, 0);

        // Field: retrnsmt; number of unrecovered RTO timeouts.
        // Unimplemented.
        buf += &format!("{:08X} ", 0);

        // Field: uid.
        match file.Dirent.Inode().UnstableAttr(task) {
            Err(e) => {
                error!("Failed to retrieve unstable attr for socket file: {:?}", e);
                buf += &format!("{:<5} ", 0);
            }
            Ok(uattr) => {
                let creds = task.Creds();
                let usernamespace = creds.lock().UserNamespace.clone();
                buf += &format!("{:<5} ", uattr.Owner.UID.In(&usernamespace).OrOverflow().0);
            }
        }

        // Field: timeout; number of unanswered 0-window probes.
        // Unimplemented.
        buf += &format!("{:>8} ", 0);

        // Field: inode.
        let inodeId = file.Dirent.Inode().StableAttr().InodeId;
        buf += &format!("{:>8} ", inodeId);

        // Field: ref; reference count on the socket inode. Don't count the ref
        // we obtain while deferencing the weakref to this socket.
        buf += &format!("{ } ", file.ReadRefs() - 1);

        // Field: Socket struct address. Redacted due to the same reason as
        // the 'Num' field in /proc/net/unix, see netUnix.ReadSeqFileData.
        buf += &format!("{:>16} ", 0);

        // Field: retransmit timeout. Unimplemented.
        buf += &format!("{} ", 0);

        // Field: predicted tick of soft clock (delayed ACK control data).
        // Unimplemented.
        buf += &format!("{} ", 0);

        // Field: (ack.quick<<1)|ack.pingpong, Unimplemented.
        buf += &format!("{} ", 0);

        // Field: sending congestion window, Unimplemented.
        buf += &format!("{} ", 0);

        // Field: Slow start size threshold, -1 if threshold >= 0xFFFF.
        // Unimplemented, report as large threshold.
        buf += &format!("{} ", -1);

        buf += &format!("\n");
    }

    return Ok(buf.as_bytes().to_vec());
}

pub struct NetUDP {}

impl SimpleFileTrait for NetUDP {
    fn GetFile(
        &self,
        _task: &Task,
        _dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        let fops = ReadonlyFileOperations {
            node: NetUDPReadonlyFileNode {}.into(),
        };

        let file = File::New(dirent, &flags, fops.into());
        return Ok(file);
    }
}

pub fn NewNetUDP(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let node = SimpleFileInode::New(
        task,
        &ROOT_OWNER,
        &FilePermissions {
            User: PermMask {
                read: true,
                write: false,
                execute: false,
            },
            Group: PermMask {
                read: true,
                write: false,
                execute: false,
            },
            Other: PermMask {
                read: true,
                write: false,
                execute: false,
            },
            ..Default::default()
        },
        FSMagic::ANON_INODE_FS_MAGIC,
        false,
        NetUDP {},
    );

    return NewProcInode(&Arc::new(node), msrc, InodeType::SpecialFile, None);
}

pub struct NetUDPReadonlyFileNode {}

impl NetUDPReadonlyFileNode {
    pub fn GetData(&self, task: &Task, header: &str) -> Result<Vec<u8>> {
        let kernel = GetKernel();
        let sockets = kernel.sockets.ListSockets();

        let mut buf = header.to_string();
        for (id, file) in sockets {
            let fopsType = file.FileOp.FopsType();
            //&& fopsType != FileOpsType::UnixSocketOperations
            if fopsType != FileOpsType::SocketOperations {
                continue
            }

            let sockops = file.FileOp.clone();
            /*let sockops = fileops
                .as_any()
                .downcast_ref::<SocketOperations>()
                .expect("SocketOperations convert fail")
                .clone();

            let family = sockops.family;
            let stype = sockops.stype;*/

            let (family, stype, _protocol) = sockops.Type();

            if family != AFType::AF_INET || stype != SockType::SOCK_STREAM {
                continue;
            }

            // Linux's documentation for the fields below can be found at
            // https://www.kernel.org/doc/Documentation/networking/proc_net_tcp.txt.
            // For Linux's implementation, see net/ipv4/tcp_ipv4.c:get_tcp4_sock().
            // Note that the header doesn't contain labels for all the fields.

            // Field: sl; entry number.
            buf += &format!("{:<5}: ", id);

            let mut sockBuf = [0; 256];

            // Field: local_adddress.
            let _ = sockops.GetSockName(task, &mut sockBuf)?;
            let addr = GetAddr(AFType::AF_INET as _, &sockBuf)?;
            buf += &WriteInetAddr(&addr);

            // Field: rem_address.
            let _ = sockops.GetPeerName(task, &mut sockBuf)?;
            let addr = GetAddr(AFType::AF_INET as _, &sockBuf)?;
            buf += &WriteInetAddr(&addr);

            // Field: state; socket state.
            buf += &format!("{:02X} ", sockops.State());

            // Field: tx_queue, rx_queue; number of packets in the transmit and
            // receive queue. Unimplemented.
            buf += &format!("{:08X}:{:08X} ", 0, 0);

            // Field: tr, tm->when; timer active state and number of jiffies
            // until timer expires. Unimplemented.
            buf += &format!("{:02X}:{:08X} ", 0, 0);

            // Field: retrnsmt; number of unrecovered RTO timeouts.
            // Unimplemented.
            buf += &format!("{:08X} ", 0);

            // Field: uid.
            match file.Dirent.Inode().UnstableAttr(task) {
                Err(e) => {
                    error!("Failed to retrieve unstable attr for socket file: {:?}", e);
                    buf += &format!("{:<5} ", 0);
                }
                Ok(uattr) => {
                    let creds = task.Creds();
                    let usernamespace = creds.lock().UserNamespace.clone();
                    buf += &format!("{:<5} ", uattr.Owner.UID.In(&usernamespace).OrOverflow().0);
                }
            }

            // Field: timeout; number of unanswered 0-window probes.
            // Unimplemented.
            buf += &format!("{:>8} ", 0);

            // Field: inode.
            let inodeId = file.Dirent.Inode().StableAttr().InodeId;
            buf += &format!("{:>8} ", inodeId);

            // Field: ref; reference count on the socket inode. Don't count the ref
            // we obtain while deferencing the weakref to this socket.
            buf += &format!("{ } ", file.ReadRefs() - 1);

            // Field: Socket struct address. Redacted due to the same reason as
            // the 'Num' field in /proc/net/unix, see netUnix.ReadSeqFileData.
            buf += &format!("{:>16} ", 0);

            // Field: drops; number of dropped packets. Unimplemented.
            buf += &format!("{} ", 0);

            buf += &format!("\n");
        }

        return Ok(buf.as_bytes().to_vec());
    }
}

impl ReadonlyFileNodeTrait for NetUDPReadonlyFileNode {
    fn ReadAt(
        &self,
        task: &Task,
        _f: &File,
        dsts: &mut [IoVec],
        offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        if offset < 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let header = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode ref pointer drops             \n";
        let bytes = self.GetData(task, header)?;
        if offset as usize > bytes.len() {
            return Ok(0);
        }

        let n = task.CopyDataOutToIovs(&bytes[offset as usize..], dsts, true)?;

        return Ok(n as i64);
    }
}

pub struct NetUnix {}

impl SimpleFileTrait for NetUnix {
    fn GetFile(
        &self,
        _task: &Task,
        _dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        let fops = ReadonlyFileOperations {
            node: NetUnixReadonlyFileNode {}.into(),
        };

        let file = File::New(dirent, &flags, fops.into());
        return Ok(file);
    }
}

pub fn NewNetUnix(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let node = SimpleFileInode::New(
        task,
        &ROOT_OWNER,
        &FilePermissions {
            User: PermMask {
                read: true,
                write: false,
                execute: false,
            },
            Group: PermMask {
                read: true,
                write: false,
                execute: false,
            },
            Other: PermMask {
                read: true,
                write: false,
                execute: false,
            },
            ..Default::default()
        },
        FSMagic::ANON_INODE_FS_MAGIC,
        false,
        NetUnix {},
    );

    return NewProcInode(&Arc::new(node), msrc, InodeType::SpecialFile, None);
}


pub struct NetUnixReadonlyFileNode {}

impl NetUnixReadonlyFileNode {
    pub fn GetData(&self, _task: &Task, header: &str) -> Result<Vec<u8>> {
        let kernel = GetKernel();
        let sockets = kernel.sockets.ListSockets();

        let mut buf = header.to_string();
        for (id, file) in sockets {
            let fopsType = file.FileOp.FopsType();
            if fopsType != FileOpsType::UnixSocketOperations {
                continue
            }

            let fops = file.FileOp.clone();
            let (family, _stype, _protocol) = fops.Type();

            if family != AFType::AF_UNIX {
                continue;
            }

            let sockops = fops
                .as_any()
                .downcast_ref::<UnixSocketOperations>()
                .expect("SocketOperations convert fail")
                .clone();

            // Linux's documentation for the fields below can be found at
            // https://www.kernel.org/doc/Documentation/networking/proc_net_tcp.txt.
            // For Linux's implementation, see net/ipv4/tcp_ipv4.c:get_tcp4_sock().
            // Note that the header doesn't contain labels for all the fields.

            // Field: sl; entry number.
            buf += &format!("{:<5}: ", id);

            // Field: local_adddress.
            let addr = match sockops.ep.GetLocalAddress() {
                Err(e) => {
                    error!("NetUnixReadonlyFileNode Failed to retrieve socket name from {:?}", e);
                    SockAddrUnix {
                        Family: AFType::AF_UNIX as _,
                        Path: "<unknown>".to_string(),
                    }
                }
                Ok(addr) => addr
            };

            let mut sockFlags = 0;
            match &sockops.ep {
                BoundEndpoint::Connected(sock) => {
                    if sock.Listening() {
                        // For unix domain sockets, linux reports a single flag
                        // value if the socket is listening, of __SO_ACCEPTCON.
                        sockFlags = SO_ACCEPTCON;
                    }
                }
                _ => ()
            }

            let inodeId = file.Dirent.Inode().StableAttr().InodeId;

            // In the socket entry below, the value for the 'Num' field requires
            // some consideration. Linux prints the address to the struct
            // unix_sock representing a socket in the kernel, but may redact the
            // value for unprivileged users depending on the kptr_restrict
            // sysctl.
            //
            // One use for this field is to allow a privileged user to
            // introspect into the kernel memory to determine information about
            // a socket not available through procfs, such as the socket's peer.
            //
            // On gvisor, returning a pointer to our internal structures would
            // be pointless, as it wouldn't match the memory layout for struct
            // unix_sock, making introspection difficult. We could populate a
            // struct unix_sock with the appropriate data, but even that
            // requires consideration for which kernel version to emulate, as
            // the definition of this struct changes over time.
            //
            // For now, we always redact this pointer.
            buf += &format!("{:010}: {:08X} {:08X} {:08X} {:04X} {:02X} {:>5}",
                            0,
                            file.ReadRefs() - 1,
                            0,
                            sockFlags,
                            sockops.ep.Type(),
                            sockops.State(),
                            inodeId);

            if addr.Path.len() != 0 {
                if addr.Path.as_bytes()[0] == 0 {
                    buf += &format!(" @{}", &addr.Path[1..]);
                } else {
                    buf += &format!(" {}", addr.Path);
                }
            }

            buf += &format!("\n");
        }

        return Ok(buf.as_bytes().to_vec());
    }
}

impl ReadonlyFileNodeTrait for NetUnixReadonlyFileNode {
    fn ReadAt(
        &self,
        task: &Task,
        _f: &File,
        dsts: &mut [IoVec],
        offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        if offset < 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let header = "Num       RefCount Protocol Flags    Type St Inode Path\n";
        let bytes = self.GetData(task, header)?;
        if offset as usize > bytes.len() {
            return Ok(0);
        }

        let n = task.CopyDataOutToIovs(&bytes[offset as usize..], dsts, true)?;

        return Ok(n as i64);
    }
}
