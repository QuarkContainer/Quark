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

#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(deref_nullptr)]
#![feature(proc_macro_hygiene)]
#![feature(naked_functions)]
#![allow(bare_trait_objects)]
#![feature(map_first_last)]
#![allow(non_camel_case_types)]
#![feature(llvm_asm)]
#![allow(deprecated)]
#![feature(thread_id_value)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![feature(core_intrinsics)]

extern crate alloc;
extern crate bit_field;
extern crate core_affinity;
extern crate errno;

#[macro_use]
extern crate serde_derive;
extern crate cache_padded;
extern crate serde;
extern crate serde_json;

#[macro_use]
extern crate clap;

#[macro_use]
extern crate scopeguard;

#[macro_use]
extern crate lazy_static;

extern crate libc;
extern crate spin;
extern crate x86_64;
#[macro_use]
extern crate log;
extern crate caps;
extern crate fs2;
extern crate regex;
extern crate simplelog;
extern crate tabwriter;

#[macro_use]
pub mod print;

#[macro_use]
pub mod asm;
pub mod kernel_def;
pub mod qlib;

//pub mod rdma_bak;
//pub mod rdma_agent;
// pub mod rdma_channel;
// pub mod rdma_conn;
// pub mod rdma_ctrlconn;
pub mod rdma_service_client;
// pub mod rdma_srv;

// use crate::rdma_srv::RDMA_CTLINFO;
// use crate::rdma_srv::RDMA_SRV;

use self::qlib::ShareSpaceRef;
use alloc::slice;
use alloc::sync::Arc;
use fs2::FileExt;
use std::collections::HashMap;
use std::io;
use std::io::prelude::*;
use std::io::Error;
use std::net::{IpAddr, Ipv4Addr, TcpListener, TcpStream};
use std::os::unix::io::{AsRawFd, RawFd};
pub static SHARE_SPACE: ShareSpaceRef = ShareSpaceRef::New();
use crate::qlib::rdma_share::*;
// use crate::rdma_bak::RDMA;
use local_ip_address::list_afinet_netifas;
use local_ip_address::local_ip;
use qlib::linux_def::*;
use qlib::socket_buf::SocketBuff;
use qlib::unix_socket::UnixSocket;
// use rdma_conn::RDMAConn;
// use rdma_ctrlconn::Node;
use rdma_service_client::*;
use spin::{Mutex, MutexGuard};
use std::str::FromStr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::{env, mem, ptr, thread, time};

#[allow(unused_macros)]
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

const READ_FLAGS: i32 = libc::EPOLLET | libc::EPOLLIN;
//const READ_FLAGS: i32 = libc::EPOLLONESHOT | libc::EPOLLIN | libc::EPOLLOUT;
const WRITE_FLAGS: i32 = libc::EPOLLET | libc::EPOLLOUT;
//const WRITE_FLAGS: i32 = libc::EPOLLONESHOT | libc::EPOLLIN | libc::EPOLLOUT;

const READ_WRITE_FLAGS: i32 = libc::EPOLLET | libc::EPOLLOUT | libc::EPOLLIN;

pub enum FdType {
    UnixDomainSocketServer,
    UnixDomainSocketConnect,
    TCPSocketServer(u16),  //port
    TCPSocketConnect(u32), //sockfd maintained by RDMASvcCli
    RDMACompletionChannel,
    ClientEvent,
}

fn main() -> io::Result<()> {
    println!("RDMASvc Client...");
    let mut fds: HashMap<i32, FdType> = HashMap::new();
    let args: Vec<_> = env::args().collect();
    //eventfd_test1();
    let rdmaSvcCli: RDMASvcClient;
    rdmaSvcCli = RDMASvcClient::initialize("/home/qingming/rdma_srv");

    let cliEventFd = rdmaSvcCli.cliEventFd;
    let srvEventFd = rdmaSvcCli.srvEventFd;
    unblock_fd(cliEventFd);
    unblock_fd(rdmaSvcCli.srvEventFd);

    let epoll_fd = epoll_create().expect("can create epoll queue");
    epoll_add(epoll_fd, cliEventFd, read_event(cliEventFd as u64))?;
    fds.insert(cliEventFd, FdType::ClientEvent);

    //Bind all registered port
    //100733100, 58433
    //134654144
    let serverSockFd = rdmaSvcCli.sockIdMgr.lock().AllocId().unwrap();
    rdmaSvcCli.bind(
        serverSockFd,
        //u32::from(Ipv4Addr::from_str("172.16.1.6").unwrap()).to_be(),
        u32::from(Ipv4Addr::from_str("192.168.6.8").unwrap()).to_be(),
        16868u16.to_be(),
    );
    // let ten_millis = time::Duration::from_secs(1);
    // let _now = time::Instant::now();
    // thread::sleep(ten_millis);
    rdmaSvcCli.listen(serverSockFd, 5);
    rdmaSvcCli
        .cliShareRegion
        .lock()
        .clientBitmap
        .store(1, Ordering::SeqCst);
    wait(epoll_fd, &rdmaSvcCli, &mut fds);

    return Ok(());
}

fn wait(epoll_fd: i32, rdmaSvcCli: &RDMASvcClient, fds: &mut HashMap<i32, FdType>) {
    let mut events: Vec<EpollEvent> = Vec::with_capacity(1024);
    let mut index = 0;
    let mut randomMsgSize = 32768; //32789; //32768;
    let count = 10; //10000000;
    let mut left = randomMsgSize as u64;
    let rbSize = 65536;
    let rb: Vec<u8> = Vec::with_capacity(rbSize);
    let wb: Vec<u8> = Vec::with_capacity(randomMsgSize as usize);
    let rAddr = rb.as_ptr() as u64;
    let wAddr = rb.as_ptr() as u64;
    let mut readData = 0;

    // mapping between sockfd maintained by rdmaSvcCli and fd for connecting to external server.
    let mut sockFdMappings: HashMap<u32, i32> = HashMap::new();
    loop {
        events.clear();
        // println!("in loop");
        {
            rdmaSvcCli
                .cliShareRegion
                .lock()
                .clientBitmap
                .store(1, Ordering::SeqCst);
        }
        let res = match syscall!(epoll_wait(
            epoll_fd,
            events.as_mut_ptr() as *mut libc::epoll_event,
            1024,
            -1 as libc::c_int,
        )) {
            Ok(v) => v,
            Err(e) => panic!("error during epoll wait: {}", e),
        };

        unsafe { events.set_len(res as usize) };

        // println!("res is: {}", res);

        for ev in &events {
            // print!("u64: {}, events: {:x}", ev.U64, ev.Events);
            let event_data = fds.get(&(ev.U64 as i32));
            match event_data {
                Some(FdType::TCPSocketServer(port)) => {
                    println!("Egress gateway doesn't have this type!");
                }
                Some(FdType::TCPSocketConnect(sockfd)) => {
                    let mut sockFdInfos = rdmaSvcCli.dataSockFdInfos.lock();
                    let sockInfo = sockFdInfos.get_mut(sockfd).unwrap();

                    if !matches!(sockInfo.status, SockStatus::ESTABLISHED) {
                        println!("SockInfo.status is {:?}", sockInfo.status);
                        return;
                    }
                    let mut shareRegion = rdmaSvcCli.cliShareRegion.lock();
                    println!(
                        "FdType::TCPSocketConnect, sockfd: {}, ev.Events: {}",
                        sockfd, ev.Events
                    );
                    if ev.Events & EVENT_IN as u32 != 0 {
                        rdmaSvcCli.ReadFromSocket(sockInfo, &sockFdMappings, &mut shareRegion);
                    }
                    if ev.Events & EVENT_OUT as u32 != 0 {
                        rdmaSvcCli.WriteToSocket(sockInfo, &sockFdMappings, &mut shareRegion);
                    }
                }
                Some(FdType::RDMACompletionChannel) => {}
                Some(FdType::UnixDomainSocketConnect) => {}
                Some(FdType::UnixDomainSocketServer) => {}
                Some(FdType::ClientEvent) => {
                    let mut shareRegion = rdmaSvcCli.cliShareRegion.lock();
                    loop {
                        match shareRegion.cq.Pop() {
                            Some(cq) => match cq.msg {
                                RDMARespMsg::RDMAConnect(response) => {
                                    println!("response: {:?}", response);
                                    let ioBufIndex = response.ioBufIndex as usize;
                                    let mut sockFdInfos = rdmaSvcCli.dataSockFdInfos.lock();
                                    let sockInfo = sockFdInfos.get_mut(&response.sockfd).unwrap();
                                    sockInfo.sockBuff = Arc::new(SocketBuff::InitWithShareMemory(
                                        MemoryDef::DEFAULT_BUF_PAGE_COUNT,
                                        &shareRegion.ioMetas[ioBufIndex].readBufAtoms as *const _
                                            as u64,
                                        &shareRegion.ioMetas[ioBufIndex].writeBufAtoms as *const _
                                            as u64,
                                        &shareRegion.ioMetas[ioBufIndex].consumeReadData as *const _
                                            as u64,
                                        &shareRegion.iobufs[ioBufIndex].read as *const _ as u64,
                                        &shareRegion.iobufs[ioBufIndex].write as *const _ as u64,
                                    ));
                                    sockInfo.channelId = response.channelId;
                                    sockInfo.status = SockStatus::ESTABLISHED;

                                    rdmaSvcCli
                                        .channelToSockInfos
                                        .lock()
                                        .insert(response.channelId, sockInfo.clone());

                                    let mut writeBuf = sockInfo.sockBuff.writeBuf.lock();
                                    let (iovsAddr, iovsCnt) = writeBuf.GetSpaceIovs();

                                    let cnt = unsafe {
                                        libc::readv(
                                            *sockFdMappings.get(&sockInfo.fd).unwrap(),
                                            iovsAddr as *const _,
                                            iovsCnt as i32,
                                        )
                                    };

                                    println!(
                                        "cnt: {}, iovsCnt: {}, iovsAddr: {}",
                                        cnt, iovsCnt, iovsAddr
                                    );

                                    let mut trigger = false;

                                    if cnt > 0 {
                                        trigger = writeBuf.Produce(cnt as usize);
                                    }

                                    if trigger {
                                        println!("Send data the first time");
                                        rdmaSvcCli.write(
                                            sockInfo.fd,
                                            &mut shareRegion.sq,
                                            sockInfo.channelId,
                                        );
                                    }
                                }
                                RDMARespMsg::RDMAAccept(response) => {
                                    println!("response: {:?}", response);
                                    let mut sockFdInfos = rdmaSvcCli.serverSockFdInfos.lock();
                                    let sockInfo = sockFdInfos.get_mut(&response.sockfd).unwrap();

                                    let ioBufIndex = response.ioBufIndex as usize;
                                    let dataSockFd = rdmaSvcCli.sockIdMgr.lock().AllocId().unwrap();
                                    let dataSockInfo = DataSock {
                                        fd: dataSockFd, //Allocate fd
                                        srcIpAddr: sockInfo.srcIpAddr,
                                        srcPort: sockInfo.srcPort,
                                        dstIpAddr: response.dstIpAddr,
                                        dstPort: response.dstPort,
                                        status: SockStatus::ESTABLISHED,
                                        duplexMode: DuplexMode::SHUTDOWN_RDWR,
                                        channelId: response.channelId,
                                        sockBuff: Arc::new(SocketBuff::InitWithShareMemory(
                                            MemoryDef::DEFAULT_BUF_PAGE_COUNT,
                                            &shareRegion.ioMetas[ioBufIndex].readBufAtoms
                                                as *const _
                                                as u64,
                                            &shareRegion.ioMetas[ioBufIndex].writeBufAtoms
                                                as *const _
                                                as u64,
                                            &shareRegion.ioMetas[ioBufIndex].consumeReadData
                                                as *const _
                                                as u64,
                                            &shareRegion.iobufs[ioBufIndex].read as *const _ as u64,
                                            &shareRegion.iobufs[ioBufIndex].write as *const _
                                                as u64,
                                        )),
                                    };

                                    rdmaSvcCli
                                        .dataSockFdInfos
                                        .lock()
                                        .insert(dataSockFd, dataSockInfo.clone());
                                    sockInfo.acceptQueue.lock().EnqSocket(dataSockFd);
                                    rdmaSvcCli
                                        .channelToSockInfos
                                        .lock()
                                        .insert(response.channelId, dataSockInfo.clone());

                                    let sock_fd = unsafe {
                                        libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0)
                                    };
                                    println!("sock_fd is {}", sock_fd);
                                    unblock_fd(sock_fd);
                                    fds.insert(sock_fd, FdType::TCPSocketConnect(dataSockFd));
                                    epoll_add(epoll_fd, sock_fd, read_write_event(sock_fd as u64));

                                    unsafe {
                                        let serv_addr: libc::sockaddr_in = libc::sockaddr_in {
                                            sin_family: libc::AF_INET as u16,
                                            sin_port: 25028u16.to_be(),
                                            sin_addr: libc::in_addr {
                                                s_addr: u32::from(
                                                    Ipv4Addr::from_str("172.16.1.6").unwrap(),
                                                )
                                                .to_be(),
                                            },
                                            sin_zero: mem::zeroed(),
                                        };
                                        let ret = libc::connect(
                                            sock_fd,
                                            &serv_addr as *const libc::sockaddr_in
                                                as *const libc::sockaddr,
                                            mem::size_of_val(&serv_addr) as u32,
                                        );

                                        println!(
                                            "ret is {}, error: {}",
                                            ret,
                                            Error::last_os_error()
                                        );
                                    }
                                    sockFdMappings.insert(dataSockFd, sock_fd);

                                    println!("client connect finished!")
                                }
                                RDMARespMsg::RDMANotify(response) => {
                                    println!("RDMANotify, response: {:?}", response);
                                    if response.event & EVENT_IN != 0 {
                                        let channelToSockInfos =
                                            rdmaSvcCli.channelToSockInfos.lock();
                                        let sockInfo =
                                            channelToSockInfos.get(&response.channelId).unwrap();
                                        rdmaSvcCli.WriteToSocket(
                                            sockInfo,
                                            &sockFdMappings,
                                            &mut shareRegion,
                                        );
                                    }
                                    if response.event & EVENT_OUT != 0 {
                                        println!("RDMANotify::EVENT_OUT 1");
                                        let channelToSockInfos =
                                            rdmaSvcCli.channelToSockInfos.lock();
                                        let sockInfo =
                                            channelToSockInfos.get(&response.channelId).unwrap();

                                        rdmaSvcCli.ReadFromSocket(
                                            sockInfo,
                                            &sockFdMappings,
                                            &mut shareRegion,
                                        );
                                    }
                                }
                            },
                            None => {
                                // println!("Finish procesing response from RDMASvc");
                                break;
                            }
                        }
                    }
                }
                None => {}
            }
        }
    }
}

fn init(path: &str) -> RDMASvcClient {
    //let path = "/home/qingming/rdma_srv";
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

fn eventfd_test1() {
    let path = "/home/qingming/rdma_srv";
    let cli_sock = UnixSocket::NewClient(path).unwrap();

    let mut body = 1;
    let ptr = &body as *const _ as *const u8;
    let buf = unsafe { slice::from_raw_parts(ptr, 4) };
    cli_sock.WriteWithFds(buf, &[]).unwrap();
    body = 2;
    let ptr = &body as *const _ as *const u8;
    let buf = unsafe { slice::from_raw_parts(ptr, 4) };
    println!("2nd write");
    cli_sock.WriteWithFds(buf, &[]).unwrap();

    let mut body = 0;
    let ptr = &mut body as *mut _ as *mut u8;
    let buf = unsafe { slice::from_raw_parts_mut(ptr, 4) };
    let (size, fds) = cli_sock.ReadWithFds(buf).unwrap();
    println!("size: {}, body: {}, fds: {:?}", size, body, fds);

    let efd = fds[0];
    println!("efd: {}", efd);
    println!("sleeping ...");
    let ten_millis = time::Duration::from_secs(2);
    let _now = time::Instant::now();
    thread::sleep(ten_millis);
    println!("sleeping done...");
    let u = 10u64;
    let s = unsafe {
        libc::write(
            efd,
            &u as *const _ as *const libc::c_void,
            mem::size_of_val(&u) as usize,
        )
    };

    if s == -1 {
        println!("1 last error: {}", Error::last_os_error());
    }

    println!("before read...");
    let efd1 = fds[1];
    let u = 0u64;
    let s = unsafe {
        libc::read(
            efd1,
            &u as *const _ as *mut libc::c_void,
            mem::size_of_val(&u) as usize,
        )
    };
    if s == -1 {
        println!("2 last error: {}", Error::last_os_error());
    }
    println!("s: {}, u: {}", s, u);
}

fn eventfd_test() {
    let path = "/home/qingming/rdma_srv";
    let cli_sock = UnixSocket::NewClient(path).unwrap();
    let efd = cli_sock.RecvFd().unwrap();
    println!("efd: {}", efd);
    // println!("sleeping ...");
    // let ten_millis = time::Duration::from_secs(2);
    // let _now = time::Instant::now();
    // thread::sleep(ten_millis);
    // println!("sleeping done...");
    // let u = 10u64;
    // let s = unsafe {
    //     libc::write(
    //         efd,
    //         &u as *const _ as *const libc::c_void,
    //         mem::size_of_val(&u) as usize,
    //     )
    // };

    // if s == -1 {
    //     println!("1 last error: {}", Error::last_os_error());
    // }

    println!("before read...");
    let u = 0u64;
    let s = unsafe {
        libc::read(
            efd,
            &u as *const _ as *mut libc::c_void,
            mem::size_of_val(&u) as usize,
        )
    };
    if s == -1 {
        println!("2 last error: {}", Error::last_os_error());
    }
    println!("s: {}, u: {}", s, u);
}

fn share_client_region() {
    println!("RDMA Service is starting!");

    let path = "/home/qingming/rdma_srv";
    let cli_sock = UnixSocket::NewClient(path).unwrap();
    let fd = cli_sock.RecvFd().unwrap();
    println!("cli_sock: {}, cli_fd: {}", cli_sock.as_raw_fd(), fd);

    let size = mem::size_of::<qlib::rdma_share::ClientShareRegion>();
    let addr = unsafe {
        libc::mmap(
            ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            //libc::MAP_SHARED | libc::MAP_ANONYMOUS,
            libc::MAP_SHARED,
            fd,
            0,
        )
    };
    let eventAddr = addr as *mut ClientShareRegion; // as &mut qlib::Event;
    let clientShareRegion = unsafe { &mut (*eventAddr) };
    let sockBuf = SocketBuff::InitWithShareMemory(
        MemoryDef::DEFAULT_BUF_PAGE_COUNT,
        &clientShareRegion.ioMetas[0].readBufAtoms as *const _ as u64,
        &clientShareRegion.ioMetas[0].writeBufAtoms as *const _ as u64,
        &clientShareRegion.ioMetas[0].consumeReadData as *const _ as u64,
        &clientShareRegion.iobufs[0].read as *const _ as u64,
        &clientShareRegion.iobufs[0].write as *const _ as u64,
    );

    let consumeData = sockBuf.consumeReadData.load(Ordering::Relaxed);
    println!("consumeData: {}", consumeData);
    let consumeData = sockBuf.AddConsumeReadData(5);
    println!("consumeData: {}", consumeData);
}

fn get_local_ip() -> u32 {
    let my_local_ip = local_ip().unwrap();

    // println!("This is my local IP address: {:?}", my_local_ip);

    let network_interfaces = list_afinet_netifas().unwrap();

    for (name, ip) in network_interfaces.iter() {
        //println!("{}:\t{:?}", name, ip);
    }

    return u32::from(Ipv4Addr::from_str("172.16.1.6").unwrap());
}

fn epoll_create() -> io::Result<RawFd> {
    let fd = syscall!(epoll_create1(0))?;
    if let Ok(flags) = syscall!(fcntl(fd, libc::F_GETFD)) {
        let _ = syscall!(fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC));
    }

    Ok(fd)
}

fn read_event(key: u64) -> libc::epoll_event {
    libc::epoll_event {
        events: READ_FLAGS as u32,
        u64: key,
    }
}

fn write_event(key: u64) -> libc::epoll_event {
    libc::epoll_event {
        events: WRITE_FLAGS as u32,
        u64: key,
    }
}

fn read_write_event(key: u64) -> libc::epoll_event {
    libc::epoll_event {
        events: READ_WRITE_FLAGS as u32,
        u64: key,
    }
}

fn close(fd: RawFd) {
    let _ = syscall!(close(fd));
}

fn epoll_add(epoll_fd: RawFd, fd: RawFd, mut event: libc::epoll_event) -> io::Result<()> {
    syscall!(epoll_ctl(epoll_fd, libc::EPOLL_CTL_ADD, fd, &mut event))?;
    Ok(())
}

fn epoll_modify(epoll_fd: RawFd, fd: RawFd, mut event: libc::epoll_event) -> io::Result<()> {
    syscall!(epoll_ctl(epoll_fd, libc::EPOLL_CTL_MOD, fd, &mut event))?;
    Ok(())
}

fn epoll_delete(epoll_fd: RawFd, fd: RawFd) -> io::Result<()> {
    syscall!(epoll_ctl(
        epoll_fd,
        libc::EPOLL_CTL_DEL,
        fd,
        std::ptr::null_mut()
    ))?;
    Ok(())
}

fn unblock_fd(fd: i32) {
    unsafe {
        let flags = libc::fcntl(fd, Cmd::F_GETFL, 0);
        let ret = libc::fcntl(fd, Cmd::F_SETFL, flags | Flags::O_NONBLOCK);
        assert!(ret == 0, "UnblockFd fail");
    }
}
