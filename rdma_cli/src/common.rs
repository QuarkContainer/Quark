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
use spin::{Mutex, MutexGuard};
use std::collections::HashMap;
use std::collections::HashSet;
// use std::io;
// use std::io::prelude::*;
// use std::io::Error;
use std::net::Ipv4Addr;
use std::ops::{Deref, DerefMut};
use std::os::unix::io::{AsRawFd, RawFd};
use std::str::FromStr;
use std::{env, mem, ptr, thread, time};

use local_ip_address::list_afinet_netifas;
use local_ip_address::local_ip;

use super::qlib::bytestream::*;
use super::qlib::common::*;
use super::qlib::linux_def::*;
use super::qlib::rdma_share::*;
use super::qlib::rdma_svc_cli::*;
use super::qlib::socket_buf::*;
use super::qlib::unix_socket::UnixSocket;
use super::rdma_def::*;
use super::unix_socket_def::*;

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

pub struct DataSockIntern {
    pub fd: u32,
    pub sockBuff: SocketBuff,
    pub srcIpAddr: u32,
    pub srcPort: u16,
    pub dstIpAddr: u32,
    pub dstPort: u16,
    pub status: Mutex<SockStatus>,
    pub duplexMode: DuplexMode,
    pub channelId: Mutex<u32>,
    pub finReceived: Mutex<bool>,
}

#[derive(Clone)]
pub struct DataSock(Arc<DataSockIntern>);

impl Deref for DataSock {
    type Target = Arc<DataSockIntern>;

    fn deref(&self) -> &Arc<DataSockIntern> {
        &self.0
    }
}

impl DataSock {
    pub fn New(
        fd: u32,
        srcIpAddr: u32,
        srcPort: u16,
        dstIpAddr: u32,
        dstPort: u16,
        status: SockStatus,
        channelId: u32,
        sockBuff: SocketBuff,
    ) -> Self {
        Self(Arc::new(DataSockIntern {
            srcIpAddr,
            srcPort,
            dstIpAddr,
            dstPort,
            fd,
            status: Mutex::new(status), //SockStatus::CONNECTING,
            sockBuff,                   //Arc::new(SocketBuff::NewDummySockBuf()),
            duplexMode: DuplexMode::SHUTDOWN_NONE,
            channelId: Mutex::new(channelId),
            finReceived: Mutex::new(false),
        }))
    }
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

pub struct GatewayCliIntern {
    // // agent id
    // pub agentId: u32,

    // // the unix socket fd between rdma client and RDMASrv
    // pub cliSock: UnixSocket,

    // // the memfd share memory with rdma client
    // pub cliMemFd: i32,

    // // the memfd share memory with rdma server
    // pub srvMemFd: i32,

    // // the eventfd which send notification to client
    // pub cliEventFd: i32,

    // // the eventfd which send notification to client
    // pub srvEventFd: i32,

    // // the memory region shared with client
    // pub cliMemRegion: MemRegion,

    // pub cliShareRegion: Mutex<&'static mut ClientShareRegion>,

    // // srv memory region shared with all RDMAClient
    // pub srvMemRegion: MemRegion,

    // // the bitmap to expedite ready container search
    // pub srvShareRegion: Mutex<&'static mut ShareRegion>,

    // // // sockfd -> rdmaChannelId
    // // pub rdmaChannels: HashMap<u32, u32>,
    pub rdmaSvcCli: RDMASvcClient,

    // sockfd -> sockFdInfo
    // pub sockFdInfos: Mutex<HashMap<u32, RDMASockFdInfo>>,
    pub serverSockFdInfos: Mutex<HashMap<u32, ServerSock>>, // TODO: quark will maitain separate.

    pub dataSockFdInfos: Mutex<HashMap<u32, DataSock>>, // TODO: quark will maitain separate.

    // ipaddr -> set of used ports
    pub usedPorts: Mutex<HashMap<u32, HashSet<u16>>>, // TOBEDELETE

    pub sockIPPorts: Mutex<HashMap<u32, Endpoint>>, // TOBEDELETE

    pub sockIdMgr: Mutex<IdMgr>, //RDMAId.

    pub channelToSockInfos: Mutex<HashMap<u32, DataSock>>,
}

//TODO: implement default

impl Deref for GatewayClient {
    type Target = Arc<GatewayCliIntern>;

    fn deref(&self) -> &Arc<GatewayCliIntern> {
        &self.0
    }
}

pub struct GatewayClient(Arc<GatewayCliIntern>);

impl GatewayClient {
    pub fn New(rdmaSvcCli: RDMASvcClient) -> Self {
        Self(Arc::new(GatewayCliIntern {
            rdmaSvcCli,
            serverSockFdInfos: Mutex::new(HashMap::new()),
            dataSockFdInfos: Mutex::new(HashMap::new()),
            usedPorts: Mutex::new(HashMap::new()),
            sockIPPorts: Mutex::new(HashMap::new()),
            sockIdMgr: Mutex::new(IdMgr::Init(1, 1024)),
            channelToSockInfos: Mutex::new(HashMap::new()),
        }))
    }

    pub fn initialize(path: &str, clientRole: i32) -> Self {        
        let cli_sock = UnixSocket::NewClient(path).unwrap();
        let rdmaSvcCli =
            RDMASvcClient::initialize(cli_sock, 0, 0, clientRole);
        let res = GatewayClient::New(rdmaSvcCli);
        res
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
        let sockFdInfo = ServerSock {
            srcIpAddr: ipAddr,
            srcPort: port,
            fd: sockfd,
            acceptQueue: AcceptQueue::default(),
            status: SockStatus::CLOSED,
        };

        self.serverSockFdInfos.lock().insert(sockfd, sockFdInfo);
        return Ok(());
    }

    pub fn listen(&self, sockfd: u32, waitingLen: i32) -> Result<()> {
        match self.sockIPPorts.lock().get(&sockfd) {
            Some(endpoint) => match self.rdmaSvcCli.listen(sockfd, endpoint, waitingLen) {
                Ok(()) => {
                    let mut sockFdInfos = self.serverSockFdInfos.lock();
                    let sockFdInfo = sockFdInfos.get_mut(&sockfd).unwrap();
                    sockFdInfo.status = SockStatus::LISTEN;
                    Ok(())
                }
                Err(error) => return Err(error),
            },
            None => {
                // TODO: handle no bind or bind fail, assign random port
                println!("no binding");
                return Ok(());
            }
        }
    }

    pub fn connect(&self, sockfd: u32, ipAddr: u32, port: u16) -> Result<()> {
        match self.rdmaSvcCli.connect(sockfd, ipAddr, port, 0, 0) {
            Ok(()) => {
                let sockInfo = DataSock::New(
                    sockfd,
                    u32::from(Ipv4Addr::from_str("192.168.6.6").unwrap()).to_be(),
                    16866u16.to_be(),
                    ipAddr,
                    port,
                    SockStatus::SYN_SENT,
                    0,
                    SocketBuff(Arc::new(SocketBuffIntern::NewDummySockBuf())),
                );

                self.dataSockFdInfos.lock().insert(sockfd, sockInfo);
                Ok(())
            }
            Err(error) => return Err(error),
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

    pub fn getsockname(&self, _sockfd: u32) {}

    pub fn getpeername(&self, _sockfd: u32) {}

    pub fn shutdown(&self, channelId: u32, howto: u8) -> Result<()> {
        self.rdmaSvcCli.shutdown(channelId, howto)
    }

    pub fn close(&self, channelId: u32) {
        println!("rdmaSvcCli::close 1, channelId: {}", channelId);
    }

    ///// ReadFromSocket and WriteToSocket are mainly for ingress and egress
    /// TODO: extract readv and notify
    pub fn ReadFromSocket(&self, sockInfo: &mut DataSock, sockFdMappings: &HashMap<u32, i32>) {
        // println!("ReadFromSocket, 1");
        let mut buffer = sockInfo.sockBuff.writeBuf.lock();
        loop {
            let (iovsAddr, iovsCnt) = buffer.GetSpaceIovs();
            if iovsCnt == 0 {
                // println!("ReadFromSocket, break because iovsCnt == 0");
                break;
            }
            let cnt = unsafe {
                libc::readv(
                    *sockFdMappings.get(&sockInfo.fd).unwrap(),
                    iovsAddr as *const _,
                    iovsCnt as i32,
                )
            };

            if cnt > 0 {
                let trigger = buffer.Produce(cnt as usize);
                // println!("ReadFromSocket, trigger: {}", trigger);
                // let mut shareRegion = self.cliShareRegion.lock();
                if trigger {
                    let _ret = self.rdmaSvcCli.write(*sockInfo.channelId.lock());
                }
            } else {
                // println!("ReadFromSocket, break because cnt: {}", cnt);
                if cnt < 0 {
                    // println!("ReadFromSocket, error: {}", std::io::Error::last_os_error());
                } else if cnt == 0 {
                    //TODO:
                    // println!("ReadFromSocket, cnt == 0 1");
                    let _ret = self.shutdown(*sockInfo.channelId.lock(), 1);
                    // println!("ReadFromSocket, cnt == 0 2");
                    if matches!(*sockInfo.status.lock(), SockStatus::FIN_READ_FROM_BUFFER) {
                        // println!("ReadFromSocket, close socket");
                        unsafe { libc::close(*sockFdMappings.get(&sockInfo.fd).unwrap()) };

                        // clean up
                        // println!("ReadFromSocket, cnt == 0 3");
                        self.channelToSockInfos
                            .lock()
                            .remove(&sockInfo.channelId.lock());
                        // println!("ReadFromSocket, cnt == 0 4");
                        self.dataSockFdInfos.lock().remove(&sockInfo.fd);
                        // println!("ReadFromSocket, cnt == 0 5");
                        self.sockIdMgr.lock().Remove(sockInfo.fd);
                        // Send close to svc
                        let _ret = self.rdmaSvcCli.SentMsgToSvc(RDMAReqMsg::RDMAClose(
                            RDMACloseReq {
                                channelId: *sockInfo.channelId.lock(),
                            },
                        ));
                    } else {
                        *sockInfo.status.lock() = SockStatus::FIN_SENT_TO_SVC;
                        // println!("ReadFromSocket, sockInfo.status = SockStatus::FIN_SENT_TO_SVC, sockfd: {}", sockInfo.fd);
                    }
                }

                break;
            }
        }
    }

    pub fn WriteToSocket(&self, sockInfo: &mut DataSock, sockFdMappings: &HashMap<u32, i32>) {
        let mut buffer = sockInfo.sockBuff.readBuf.lock();
        // println!(
        //     "WriteToSocket, 1, sockfd: {}, status: {:?}",
        //     sockInfo.fd, sockInfo.status
        // );
        loop {
            let (iovsAddr, iovsCnt) = buffer.GetDataIovs();
            // let ioVec = unsafe { &(*(iovsAddr as *const libc::iovec)) };
            // println!(
            //     "WriteToSocket, data size 1: {}, ioVec.address: {}, ioVec::len: {}, iovsCnt: {}",
            //     buffer.AvailableDataSize(),
            //     ioVec.iov_base as u64,
            //     ioVec.iov_len,
            //     iovsCnt
            // );

            if iovsCnt == 0 {
                // println!("WriteToSocket, break because iovsCnt == 0");
                if *sockInfo.finReceived.lock() {
                    if matches!(*sockInfo.status.lock(), SockStatus::FIN_SENT_TO_SVC) {
                        // println!("WriteToSocket, close socket 1");
                        unsafe { libc::close(*sockFdMappings.get(&sockInfo.fd).unwrap()) };

                        // clean up
                        // println!("WriteToSocket, close socket 2");
                        self.channelToSockInfos
                            .lock()
                            .remove(&sockInfo.channelId.lock());
                        // println!("WriteToSocket, close socket 3");
                        self.dataSockFdInfos.lock().remove(&sockInfo.fd);
                        // println!("WriteToSocket, close socket 4");
                        self.sockIdMgr.lock().Remove(sockInfo.fd);
                        let _ret = self.rdmaSvcCli.SentMsgToSvc(RDMAReqMsg::RDMAClose(
                            RDMACloseReq {
                                channelId: *sockInfo.channelId.lock(),
                            },
                        ));
                    } else if !matches!(*sockInfo.status.lock(), SockStatus::FIN_READ_FROM_BUFFER) {
                        unsafe {
                            libc::shutdown(*sockFdMappings.get(&sockInfo.fd).unwrap(), 1);
                        }
                        // println!("WriteToSocket, shutdown socket");
                        *sockInfo.status.lock() = SockStatus::FIN_READ_FROM_BUFFER;
                    }
                }
                break;
            }

            let cnt = unsafe {
                libc::writev(
                    *sockFdMappings.get(&sockInfo.fd).unwrap(),
                    iovsAddr as *const _,
                    iovsCnt as i32,
                )
            };
            // println!("WriteToSocket, cnt: {}", cnt);
            if cnt > 0 {
                buffer.Consume(cnt as usize);
                let consumedDataSize = sockInfo.sockBuff.AddConsumeReadData(cnt as u64) as usize;
                println!(
                    "WriteToSocket::Consume, channelId: {}, sockfd: {}, cnt: {}",
                    sockInfo.channelId.lock(),
                    sockInfo.fd,
                    cnt
                );
                let bufSize = buffer.BufSize();
                // println!("WriteToSocket, consumedDataSize: {}", consumedDataSize);
                if 2 * consumedDataSize >= bufSize {
                    // let mut shareRegion = self.cliShareRegion.lock();
                    let _ret = self.rdmaSvcCli.read(*sockInfo.channelId.lock());
                }
            } else {
                // println!("WriteToSocket, break because cnt: {}", cnt);
                if cnt < 0 {
                    // println!("WriteToSocket, error: {}", std::io::Error::last_os_error());
                }

                break;
            }
        }
    }
}

// pub fn init(path: &str) -> GatewayClient {
//     let cli_sock = UnixSocket::NewClient(path).unwrap();

//     let body = 1;
//     let ptr = &body as *const _ as *const u8;
//     let buf = unsafe { slice::from_raw_parts(ptr, 4) };
//     cli_sock.WriteWithFds(buf, &[]).unwrap();

//     let mut body = [0, 0];
//     let ptr = &mut body as *mut _ as *mut u8;
//     let buf = unsafe { slice::from_raw_parts_mut(ptr, 8) };
//     let (size, fds) = cli_sock.ReadWithFds(buf).unwrap();
//     if body[0] == 123 {
//         println!("size: {}, fds: {:?}, agentId: {}", size, fds, body[1]);
//     }

//     let rdmaSvcCli = GatewayClient::New(fds[0], fds[1], fds[2], fds[3], body[1], cli_sock);
//     rdmaSvcCli
// }

#[allow(unused_macros)]
#[macro_export]
macro_rules! syscall {
    ($fn: ident ( $($arg: expr),* $(,)* ) ) => {{
        let res = unsafe { libc::$fn($($arg, )*) };
        if res == -1 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(res)
        }
    }};
}

#[repr(C)]
#[repr(packed)]
#[derive(Default, Copy, Clone, Debug)]
pub struct EpollEvent {
    pub Events: u32,
    pub U64: u64,
}

pub static READ_FLAGS: i32 = libc::EPOLLET | libc::EPOLLIN;
//const READ_FLAGS: i32 = libc::EPOLLONESHOT | libc::EPOLLIN | libc::EPOLLOUT;
pub static WRITE_FLAGS: i32 = libc::EPOLLET | libc::EPOLLOUT;
//const WRITE_FLAGS: i32 = libc::EPOLLONESHOT | libc::EPOLLIN | libc::EPOLLOUT;

pub const READ_WRITE_FLAGS: i32 = libc::EPOLLET | libc::EPOLLOUT | libc::EPOLLIN;

pub enum FdType {
    TCPSocketServer(u16),  //port
    TCPSocketConnect(u32), //sockfd maintained by RDMASvcCli
    ClientEvent,
}

#[derive(Clone)]
pub enum Srv_FdType {
    UnixDomainSocketServer(UnixSocket),
    UnixDomainSocketConnect(UnixSocket),
    TCPSocketServer,
    TCPSocketConnect(u32),
    RDMACompletionChannel,
    SrvEventFd(i32),
    NodeEventFd(NodeEvent),
}

#[derive(Clone, Debug)]
pub struct NodeEvent{
    pub is_delete: bool,
    pub ip: u32,
}

#[derive(Clone)]
pub struct PodEvent{
    pub is_delete: bool,
    pub ip: u32,
}

pub fn get_local_ip() -> u32 {
    let _my_local_ip = local_ip().unwrap();

    // println!("This is my local IP address: {:?}", my_local_ip);

    let network_interfaces = list_afinet_netifas().unwrap();

    for (_name, _ip) in network_interfaces.iter() {
        //println!("{}:\t{:?}", name, ip);
    }

    return u32::from(Ipv4Addr::from_str("172.16.1.6").unwrap());
}

pub fn epoll_create() -> std::io::Result<RawFd> {
    let fd = syscall!(epoll_create1(0))?;
    if let Ok(flags) = syscall!(fcntl(fd, libc::F_GETFD)) {
        let _ = syscall!(fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC));
    }

    Ok(fd)
}

pub fn read_event(key: u64) -> libc::epoll_event {
    libc::epoll_event {
        events: READ_FLAGS as u32,
        u64: key,
    }
}

pub fn write_event(key: u64) -> libc::epoll_event {
    libc::epoll_event {
        events: WRITE_FLAGS as u32,
        u64: key,
    }
}

pub fn read_write_event(key: u64) -> libc::epoll_event {
    libc::epoll_event {
        events: READ_WRITE_FLAGS as u32,
        u64: key,
    }
}

pub fn close(fd: RawFd) {
    let _ = syscall!(close(fd));
}

pub fn epoll_add(epoll_fd: RawFd, fd: RawFd, mut event: libc::epoll_event) -> std::io::Result<()> {
    syscall!(epoll_ctl(epoll_fd, libc::EPOLL_CTL_ADD, fd, &mut event))?;
    Ok(())
}

pub fn epoll_modify(
    epoll_fd: RawFd,
    fd: RawFd,
    mut event: libc::epoll_event,
) -> std::io::Result<()> {
    syscall!(epoll_ctl(epoll_fd, libc::EPOLL_CTL_MOD, fd, &mut event))?;
    Ok(())
}

pub fn epoll_delete(epoll_fd: RawFd, fd: RawFd) -> std::io::Result<()> {
    syscall!(epoll_ctl(
        epoll_fd,
        libc::EPOLL_CTL_DEL,
        fd,
        std::ptr::null_mut()
    ))?;
    Ok(())
}

pub fn unblock_fd(fd: i32) {
    unsafe {
        let flags = libc::fcntl(fd, Cmd::F_GETFL, 0);
        let ret = libc::fcntl(fd, Cmd::F_SETFL, flags | Flags::O_NONBLOCK);
        assert!(ret == 0, "UnblockFd fail");
    }
}
