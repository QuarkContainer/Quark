// Copyright (c) 2021 Quark Container Authors
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

use alloc::collections::vec_deque::VecDeque;
use alloc::slice;
use alloc::sync::Arc;
use core::sync::atomic::Ordering;
use spin::Mutex;
use std::collections::HashMap;
use std::collections::HashSet;
use std::net::Ipv4Addr;
use std::ops::{Deref, DerefMut};
use std::str::FromStr;
use std::{env, mem, ptr, thread, time};

use super::qlib::bytestream::*;
use super::qlib::common::*;
use super::qlib::linux_def::*;
use super::qlib::rdma_share::*;
use super::qlib::socket_buf::*;
use super::qlib::unix_socket::*;

pub enum SockType {
    TBD,
    SERVER,
    CLIENT,
}

pub enum RDMASockFdInfo {
    ServerSock(ServerSock),
    DataSock(DataSock),
}

pub struct ServerSock {
    pub fd: u32,
    pub srcIpAddr: u32,
    pub srcPort: u16,
    pub acceptQueue: AcceptQueue,
    pub status: SockStatus,
}

#[derive(Clone)]
pub struct DataSock {
    pub fd: u32,
    pub sockBuff: Arc<SocketBuff>,
    pub srcIpAddr: u32,
    pub srcPort: u16,
    pub dstIpAddr: u32,
    pub dstPort: u16,
    pub status: SockStatus,
    pub duplexMode: DuplexMode,
    pub channelId: u32,
}

pub struct SockFdInfo {
    pub fd: u32,
    pub sockBuff: Arc<SocketBuff>,
    pub srcIpAddr: u32,
    pub srcPort: u16,
    pub dstIpAddr: u32,
    pub dstPort: u16,
    pub status: SockStatus,
    pub duplexMode: DuplexMode,
    pub sockType: SockType,
    pub channelId: u32,
    pub acceptQueue: AcceptQueue,
    // pub acceptQueue: [RDMAChannel; 5], //object with remote information.
}

impl Default for SockFdInfo {
    fn default() -> Self {
        SockFdInfo {
            fd: 0,
            sockBuff: Arc::new(SocketBuff::default()),
            srcIpAddr: 0,
            srcPort: 0,
            dstIpAddr: 0,
            dstPort: 0,
            status: SockStatus::CLOSING,
            duplexMode: DuplexMode::SHUTDOWN_RDWR,
            sockType: SockType::SERVER,
            channelId: 0,
            acceptQueue: AcceptQueue::default(),
        }
    }
}

#[derive(Default, Debug)]
pub struct AcceptItem {
    pub sockfd: u32,
}

#[derive(Default, Clone, Debug)]
pub struct AcceptQueue(Arc<Mutex<AcceptQueueIntern>>);

impl Deref for AcceptQueue {
    type Target = Arc<Mutex<AcceptQueueIntern>>;

    fn deref(&self) -> &Arc<Mutex<AcceptQueueIntern>> {
        &self.0
    }
}

impl AcceptQueue {
    pub fn New(length: usize) -> Self {
        Self(Arc::new(Mutex::new(AcceptQueueIntern {
            queue: VecDeque::default(),
            queueLen: length,
            error: 0,
            total: 0,
        })))
    }
}

#[derive(Default, Debug)]
pub struct AcceptQueueIntern {
    pub queue: VecDeque<AcceptItem>,
    pub queueLen: usize,
    pub error: i32,
    pub total: u64,
}

impl AcceptQueueIntern {
    pub fn SetErr(&mut self, error: i32) {
        self.error = error
    }

    pub fn Err(&self) -> i32 {
        return self.error;
    }

    pub fn SetQueueLen(&mut self, len: usize) {
        self.queueLen = len;
    }

    pub fn HasSpace(&self) -> bool {
        return self.queue.len() < self.queueLen;
    }

    //return: (trigger, hasSpace)
    pub fn EnqSocket(&mut self, sockfd: u32) -> (bool, bool) {
        let item = AcceptItem { sockfd };

        self.queue.push_back(item);
        self.total += 1;
        let trigger = self.queue.len() == 1;
        return (trigger, self.queue.len() < self.queueLen);
    }

    pub fn DeqSocket(&mut self) -> (bool, Result<AcceptItem>) {
        let trigger = self.queue.len() == self.queueLen;

        match self.queue.pop_front() {
            None => {
                if self.error != 0 {
                    return (false, Err(Error::SysError(self.error)));
                }
                return (trigger, Err(Error::SysError(SysErr::EAGAIN)));
            }
            Some(item) => return (trigger, Ok(item)),
        }
    }

    pub fn Events(&self) -> EventMask {
        let mut event = EventMask::default();
        if self.queue.len() > 0 {
            event |= EVENT_IN;
        }

        if self.error != 0 {
            event |= EVENT_ERR;
        }

        return event;
    }
}

pub struct RDMASvcCliIntern {
    // agent id
    pub agentId: u32,

    // the unix socket fd between rdma client and RDMASrv
    pub cliSock: UnixSocket,

    // the memfd share memory with rdma client
    pub cliMemFd: i32,

    // the memfd share memory with rdma server
    pub srvMemFd: i32,

    // the eventfd which send notification to client
    pub cliEventFd: i32,

    // the eventfd which send notification to client
    pub srvEventFd: i32,

    // the memory region shared with client
    pub cliMemRegion: MemRegion,

    pub cliShareRegion: Mutex<&'static mut ClientShareRegion>,

    // srv memory region shared with all RDMAClient
    pub srvMemRegion: MemRegion,

    // the bitmap to expedite ready container search
    pub srvShareRegion: Mutex<&'static mut ShareRegion>,

    // // sockfd -> rdmaChannelId
    // pub rdmaChannels: HashMap<u32, u32>,

    // sockfd -> sockFdInfo
    // pub sockFdInfos: Mutex<HashMap<u32, RDMASockFdInfo>>,
    pub serverSockFdInfos: Mutex<HashMap<u32, ServerSock>>,

    pub dataSockFdInfos: Mutex<HashMap<u32, DataSock>>,

    // ipaddr -> set of used ports
    pub usedPorts: Mutex<HashMap<u32, HashSet<u16>>>,

    pub sockIPPorts: Mutex<HashMap<u32, Endpoint>>,

    pub sockIdMgr: Mutex<IdMgr>,

    pub channelToSockInfos: Mutex<HashMap<u32, DataSock>>,
}

impl Deref for RDMASvcClient {
    type Target = Arc<RDMASvcCliIntern>;

    fn deref(&self) -> &Arc<RDMASvcCliIntern> {
        &self.0
    }
}

pub struct RDMASvcClient(Arc<RDMASvcCliIntern>);

impl RDMASvcClient {
    pub fn New(
        srvEventFd: i32,
        srvMemFd: i32,
        cliEventFd: i32,
        cliMemFd: i32,
        agentId: u32,
        cliSock: UnixSocket,
    ) -> Self {
        let cliShareSize = mem::size_of::<ClientShareRegion>();
        let cliShareAddr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                cliShareSize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                cliMemFd,
                0,
            )
        };
        let cliShareRegion = unsafe { &mut (*(cliShareAddr as *mut ClientShareRegion)) };

        let cliShareRegion = Mutex::new(cliShareRegion);

        let srvShareSize = mem::size_of::<ShareRegion>();
        let srvShareAddr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                srvShareSize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                srvMemFd,
                0,
            )
        };
        let srvShareRegion = unsafe { &mut (*(srvShareAddr as *mut ShareRegion)) };
        let srvShareRegion = Mutex::new(srvShareRegion);
        Self(Arc::new(RDMASvcCliIntern {
            agentId,
            cliSock,
            cliMemFd,
            srvMemFd,
            srvEventFd,
            cliEventFd,
            cliMemRegion: MemRegion {
                addr: cliShareAddr as u64,
                len: cliShareSize as u64,
            },
            cliShareRegion,
            srvMemRegion: MemRegion {
                addr: srvShareAddr as u64,
                len: srvShareSize as u64,
            },
            srvShareRegion,
            // sockFdInfos: Mutex::new(HashMap::new()),
            serverSockFdInfos: Mutex::new(HashMap::new()),
            dataSockFdInfos: Mutex::new(HashMap::new()),
            usedPorts: Mutex::new(HashMap::new()),
            sockIPPorts: Mutex::new(HashMap::new()),
            sockIdMgr: Mutex::new(IdMgr::Init(1, 1024)),
            channelToSockInfos: Mutex::new(HashMap::new()),
        }))
    }

    pub fn initialize(path: &str) -> Self {
        let cli_sock = UnixSocket::NewClient(path).unwrap();

        let body = 1;
        let ptr = &body as *const _ as *const u8;
        let buf = unsafe { slice::from_raw_parts(ptr, 4) };
        cli_sock.WriteWithFds(buf, &[]).unwrap();

        let mut body = [0, 0];
        let ptr = &mut body as *mut _ as *mut u8;
        let buf = unsafe { slice::from_raw_parts_mut(ptr, 8) };
        let (size, fds) = cli_sock.ReadWithFds(buf).unwrap();
        if body[0] == 123 {
            println!("size: {}, fds: {:?}, agentId: {}", size, fds, body[1]);
        }

        let rdmaSvcCli = RDMASvcClient::New(fds[0], fds[1], fds[2], fds[3], body[1], cli_sock);
        rdmaSvcCli
    }

    pub fn terminate(&self) {}

    pub fn bind(&self, sockfd: u32, ipAddr: u32, port: u16) -> Result<()> {
        // TODO: handle INADDR_ANY (0)
        let mut usedPorts = self.usedPorts.lock();
        if usedPorts.contains_key(&ipAddr) {
            let usedIPPorts = usedPorts.get_mut(&ipAddr).unwrap();
            if usedIPPorts.contains(&port) {
                return Err(Error::SysError(TcpipErr::ERR_PORT_IN_USE.sysErr));
            } else {
                usedIPPorts.insert(port);
            }
        } else {
            usedPorts.insert(ipAddr, HashSet::from([port]));
        }
        self.sockIPPorts
            .lock()
            .insert(sockfd, Endpoint { ipAddr, port });
        let mut sockFdInfo = ServerSock {
            srcIpAddr: ipAddr,
            srcPort: port,
            fd: sockfd,
            acceptQueue: AcceptQueue::default(),
            status: SockStatus::BINDED,
        };

        self.serverSockFdInfos.lock().insert(sockfd, sockFdInfo);
        return Ok(());
    }

    pub fn listen(&self, sockfd: u32, waitingLen: i32) -> Result<()> {
        match self.sockIPPorts.lock().get(&sockfd) {
            Some(endpoint) => {
                let mut cliShareRegion = self.cliShareRegion.lock();
                if cliShareRegion.sq.SpaceCount() == 0 {
                    return Err(Error::NoEnoughSpace);
                } else {
                    println!("before push...");
                    let mut sockFdInfos = self.serverSockFdInfos.lock();
                    let sockFdInfo = sockFdInfos.get_mut(&sockfd).unwrap();
                    sockFdInfo.status = SockStatus::LISTENING;
                    cliShareRegion.sq.Push(RDMAReq {
                        user_data: sockfd as u64,
                        msg: RDMAReqMsg::RDMAListen(RDMAListenReq {
                            sockfd: sockfd,
                            ipAddr: endpoint.ipAddr,
                            port: endpoint.port,
                            waitingLen,
                        }),
                    });

                    self.updateBitmapAndWakeUpServerIfNecessary();

                    return Ok(());
                }
            }
            None => {
                // TODO: handle no bind or bind fail, assign random port
                println!("no binding");
                return Ok(());
            }
        }
    }

    pub fn updateBitmapAndWakeUpServerIfNecessary(&self) {
        println!("updateBitmapAndWakeUpServerIfNecessary 1 ");
        let mut srvShareRegion = self.srvShareRegion.lock();
        println!("updateBitmapAndWakeUpServerIfNecessary 2 ");
        srvShareRegion.updateBitmap(self.agentId);
        if srvShareRegion.srvBitmap.load(Ordering::Relaxed) == 1 {
            println!("before write srvEventFd");
            let data = 16u64;
            let ret = unsafe {
                libc::write(
                    self.srvEventFd,
                    &data as *const _ as *const libc::c_void,
                    mem::size_of_val(&data) as usize,
                )
            };
            println!("ret: {}", ret);
            if ret < 0 {
                println!("error: {}", std::io::Error::last_os_error());
            }
        } else {
            println!("server is not sleeping");
            self.updateBitmapAndWakeUpServerIfNecessary();
        }
    }

    pub fn connect(&self, sockfd: u32, ipAddr: u32, port: u16) -> Result<()> {
        let mut cliShareRegion = self.cliShareRegion.lock();
        if cliShareRegion.sq.SpaceCount() == 0 {
            return Err(Error::NoEnoughSpace);
        } else {
            println!("before push...");

            // TODO: figure out srcIpAddr, srcPort: 101099712, 57921
            let sockInfo = DataSock {
                srcIpAddr: u32::from(Ipv4Addr::from_str("192.168.6.6").unwrap()).to_be(),
                srcPort: 16866u16.to_be(),
                dstIpAddr: ipAddr,
                dstPort: port,
                fd: sockfd,
                status: SockStatus::CONNECTING,
                sockBuff: Arc::new(SocketBuff::NewDummySockBuf()),
                duplexMode: DuplexMode::SHUTDOWN_RDWR,
                channelId: 0,
            };

            self.dataSockFdInfos.lock().insert(sockfd, sockInfo);

            // TODO: figure out srcIpAddr, srcPort: 101099712, 57921
            cliShareRegion.sq.Push(RDMAReq {
                user_data: sockfd as u64,
                msg: RDMAReqMsg::RDMAConnect(RDMAConnectReq {
                    sockfd,
                    dstIpAddr: ipAddr,
                    dstPort: port,
                    srcIpAddr: u32::from(Ipv4Addr::from_str("192.168.6.6").unwrap()).to_be(),
                    srcPort: 16866u16.to_be(),
                }),
            });

            self.updateBitmapAndWakeUpServerIfNecessary();

            return Ok(());
        }
    }

    pub fn accept(&self, sockfd: u32, ipAddr: &mut u32, port: &mut u16) -> Result<i32> {
        let serverSockFdInfos = self.serverSockFdInfos.lock();
        match serverSockFdInfos.get(&sockfd) {
            Some(sockFdInfo) => {
                let (_trigger, res) = sockFdInfo.acceptQueue.lock().DeqSocket();
                match res {
                    Err(e) => return Err(e),
                    Ok(item) => {
                        let dataSockFdInfos = self.dataSockFdInfos.lock();
                        let dataSock = dataSockFdInfos.get(&item.sockfd);
                        match dataSock {
                            Some(dataSockFdInfo) => {
                                *ipAddr = dataSockFdInfo.dstIpAddr;
                                *port = dataSockFdInfo.dstPort;
                                Ok(item.sockfd as i32)
                            }
                            None => return Err(Error::NotExist),
                        }
                    }
                }
            }
            None => return Err(Error::NotExist),
        }
    }

    pub fn read(&self, sockfd: u32, sq: &mut RingQueue<RDMAReq>, channelId: u32) -> Result<()> {
        println!("rdmaSvcCli::read 1");
        if sq.SpaceCount() == 0 {
            println!("rdmaSvcCli::read 6");
            return Err(Error::NoEnoughSpace);
        } else {
            println!("rdmaSvcCli::read 7");
            println!("before push...");
            sq.Push(RDMAReq {
                user_data: sockfd as u64,
                msg: RDMAReqMsg::RDMARead(RDMAReadReq {
                    sockfd,
                    channelId: channelId,
                }),
            });

            self.updateBitmapAndWakeUpServerIfNecessary();
            Ok(())
        }
    }

    pub fn write(&self, sockfd: u32, sq: &mut RingQueue<RDMAReq>, channelId: u32) -> Result<()> {
        println!("rdmaSvcCli::write 1, sq.count: {}", sq.DataCount());
        if sq.SpaceCount() == 0 {
            // println!("rdmaSvcCli::write 6");
            return Err(Error::NoEnoughSpace);
        } else {
            println!("rdmaSvcCli::write 7");
            println!("before push...");
            sq.Push(RDMAReq {
                user_data: sockfd as u64,
                msg: RDMAReqMsg::RDMAWrite(RDMAWriteReq {
                    sockfd,
                    channelId: channelId,
                }),
            });

            self.updateBitmapAndWakeUpServerIfNecessary();
            Ok(())
        }
    }

    pub fn getsockname(&self, sockfd: u32) {}

    pub fn getpeername(&self, sockfd: u32) {}

    pub fn shutdown(&self, sockfd: u32, howto: i32) {}

    pub fn close(&self, sockfd: u32) {}
}
