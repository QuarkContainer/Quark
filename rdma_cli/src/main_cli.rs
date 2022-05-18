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

pub mod rdma_service_client;
pub mod unix_socket;

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
use local_ip_address::list_afinet_netifas;
use local_ip_address::local_ip;
use qlib::linux_def::*;
use qlib::socket_buf::SocketBuff;
use unix_socket::UnixSocket;
use rdma_service_client::*;
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
    TCPSocketServer,
    TCPSocketConnect(u32),
    RDMACompletionChannel,
}

fn main() -> io::Result<()> {
    println!("RDMASvc Client...");
    let args: Vec<_> = env::args().collect();
    //eventfd_test1();
    let rdmaSvcCli: RDMASvcClient;
    if args.len() == 1 {
        rdmaSvcCli = RDMASvcClient::initialize("/home/qingming/rdma_srv");
    } else {
        rdmaSvcCli = RDMASvcClient::initialize("/home/qingming/rdma_srv1");
    }

    let cliEventFd = rdmaSvcCli.cliEventFd;
    let srvEventFd = rdmaSvcCli.srvEventFd;
    unblock_fd(cliEventFd);
    unblock_fd(rdmaSvcCli.srvEventFd);

    let epoll_fd = epoll_create().expect("can create epoll queue");
    epoll_add(epoll_fd, cliEventFd, read_event(cliEventFd as u64))?;

    if args.len() == 1 {
        //100733100, 58433
        //134654144
        let serverSockFd = rdmaSvcCli.sockIdMgr.lock().AllocId().unwrap();
        rdmaSvcCli.bind(
            serverSockFd,
            //u32::from(Ipv4Addr::from_str("172.16.1.6").unwrap()).to_be(),
            u32::from(Ipv4Addr::from_str("192.168.6.8").unwrap()).to_be(),
            16868u16.to_be(),
        );
        let ten_millis = time::Duration::from_secs(1);
        let _now = time::Instant::now();
        thread::sleep(ten_millis);
        rdmaSvcCli.listen(serverSockFd, 5);
        rdmaSvcCli
            .cliShareRegion
            .lock()
            .clientBitmap
            .store(1, Ordering::SeqCst);
        wait(epoll_fd, &rdmaSvcCli);
    } else {
        rdmaSvcCli.connect(
            rdmaSvcCli.sockIdMgr.lock().AllocId().unwrap(),
            u32::from(Ipv4Addr::from_str("192.168.6.8").unwrap()).to_be(),
            16868u16.to_be(),
        );
        // let ten_millis = time::Duration::from_secs(2);
        // let _now = time::Instant::now();
        // thread::sleep(ten_millis);
        // rdmaSvcCli.listen(1, 5);
        {
            rdmaSvcCli
                .cliShareRegion
                .lock()
                .clientBitmap
                .store(1, Ordering::SeqCst);
        }
        wait(epoll_fd, &rdmaSvcCli);
    }

    return Ok(());
}

fn wait(epoll_fd: i32, rdmaSvcCli: &RDMASvcClient) {
    let mut events: Vec<EpollEvent> = Vec::with_capacity(1024);
    let mut index = 0;
    let mut randomMsgSize = 32789; //32768;
    let count = 10000000;
    let mut left = randomMsgSize as u64;
    let rbSize = 65536;
    let rb: Vec<u8> = Vec::with_capacity(rbSize);
    let wb: Vec<u8> = Vec::with_capacity(randomMsgSize as usize);
    let rAddr = rb.as_ptr() as u64;
    let wAddr = rb.as_ptr() as u64;
    let mut readData = 0;
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
            loop {
                match rdmaSvcCli.cliShareRegion.lock().cq.Pop() {
                    Some(cq) => match cq.msg {
                        RDMARespMsg::RDMAConnect(response) => {
                            println!("response: {:?}", response);
                            let ioBufIndex = response.ioBufIndex as usize;
                            let mut sockFdInfos = rdmaSvcCli.dataSockFdInfos.lock();
                            let sockInfo = sockFdInfos.get_mut(&response.sockfd).unwrap();
                            {
                                let mut shareRegion = rdmaSvcCli.cliShareRegion.lock();
                                let sockInfo = DataSock::New(
                                    sockInfo.fd, //Allocate fd
                                    sockInfo.srcIpAddr,
                                    sockInfo.srcPort,
                                    sockInfo.dstIpAddr,
                                    sockInfo.dstPort,
                                    SockStatus::ESTABLISHED,
                                    response.channelId,
                                    Arc::new(SocketBuff::InitWithShareMemory(
                                        MemoryDef::DEFAULT_BUF_PAGE_COUNT,
                                        &shareRegion.ioMetas[ioBufIndex].readBufAtoms as *const _
                                            as u64,
                                        &shareRegion.ioMetas[ioBufIndex].writeBufAtoms as *const _
                                            as u64,
                                        &shareRegion.ioMetas[ioBufIndex].consumeReadData as *const _
                                            as u64,
                                        &shareRegion.iobufs[ioBufIndex].read as *const _ as u64,
                                        &shareRegion.iobufs[ioBufIndex].write as *const _ as u64,
                                    )),
                                );
                                sockFdInfos.insert(sockInfo.fd, sockInfo);
                            }
                            let sockInfo = sockFdInfos.get_mut(&response.sockfd).unwrap();
                            rdmaSvcCli
                                .channelToSockInfos
                                .lock()
                                .insert(response.channelId, sockInfo.clone());

                            println!("client connect finished!");
                            println!("Writing some data to sockFdInfo!");
                            let mut writeBuf = sockInfo.sockBuff.writeBuf.lock();
                            // println!(
                            //     "mem::size_of::<RDMAAcceptResp>(): {}",
                            //     mem::size_of::<RDMAAcceptResp>()
                            // );
                            println!("writeBuf.AvailableSpace(): {}", writeBuf.AvailableSpace());
                            if mem::size_of::<RDMAAcceptResp>() > writeBuf.AvailableSpace() {
                                // return Err(Error::Timeout);
                            }

                            ////Send RDMAAcceptResp begin
                            // let msg = RDMAAcceptResp {
                            //     sockfd: 1,
                            //     ioBufIndex: 1,
                            //     channelId: 1,
                            //     dstIpAddr: 1,
                            //     dstPort: 1,
                            // };
                            // let (trigger, _len) = writeBuf.writeViaAddr(
                            //     &msg as *const _ as u64,
                            //     mem::size_of::<RDMAAcceptResp>() as u64,
                            // );
                            // let sockfd = sockInfo.fd;
                            // std::mem::drop(writeBuf);
                            // std::mem::drop(sockFdInfos);
                            // // std::mem::drop(shareRegion);
                            // println!("Writing some data to sockFdInfo 1!");
                            // if trigger {
                            //     println!("Writing some data to sockFdInfo 1.1!");
                            //     rdmaSvcCli.write(sockfd, &mut shareRegion.sq);
                            // }
                            ////Send RDMAAcceptResp end

                            let mut left = randomMsgSize as u64;
                            let r: Vec<u8> = Vec::with_capacity(randomMsgSize as usize);
                            let rAddr = r.as_ptr() as u64;
                            let (trigger, _len) = writeBuf.writeViaAddr(rAddr, randomMsgSize);
                            let sockfd = sockInfo.fd;
                            // std::mem::drop(writeBuf);
                            // std::mem::drop(sockFdInfos);
                            // std::mem::drop(shareRegion);
                            println!("Writing some data to sockFdInfo 1!");
                            if trigger {
                                println!("Writing some data to sockFdInfo 1.1!");
                                rdmaSvcCli.write(*sockInfo.channelId.lock());
                            }
                        }
                        RDMARespMsg::RDMAAccept(response) => {
                            println!("response: {:?}", response);
                            let mut sockFdInfos = rdmaSvcCli.serverSockFdInfos.lock();
                            let sockInfo = sockFdInfos.get_mut(&response.sockfd).unwrap();

                            let ioBufIndex = response.ioBufIndex as usize;
                            let dataSockFd = rdmaSvcCli.sockIdMgr.lock().AllocId().unwrap();
                            let mut shareRegion = rdmaSvcCli.cliShareRegion.lock();
                            let dataSockInfo = DataSock::New(
                                dataSockFd, //Allocate fd
                                sockInfo.srcIpAddr,
                                sockInfo.srcPort,
                                response.dstIpAddr,
                                response.dstPort,
                                SockStatus::ESTABLISHED,
                                response.channelId,
                                Arc::new(SocketBuff::InitWithShareMemory(
                                    MemoryDef::DEFAULT_BUF_PAGE_COUNT,
                                    &shareRegion.ioMetas[ioBufIndex].readBufAtoms as *const _
                                        as u64,
                                    &shareRegion.ioMetas[ioBufIndex].writeBufAtoms as *const _
                                        as u64,
                                    &shareRegion.ioMetas[ioBufIndex].consumeReadData as *const _
                                        as u64,
                                    &shareRegion.iobufs[ioBufIndex].read as *const _ as u64,
                                    &shareRegion.iobufs[ioBufIndex].write as *const _ as u64,
                                )),
                            );

                            rdmaSvcCli
                                .dataSockFdInfos
                                .lock()
                                .insert(dataSockFd, dataSockInfo.clone());
                            sockInfo.acceptQueue.lock().EnqSocket(dataSockFd);
                            rdmaSvcCli
                                .channelToSockInfos
                                .lock()
                                .insert(response.channelId, dataSockInfo.clone());

                            println!("client connect finished!")
                        }
                        RDMARespMsg::RDMANotify(response) => {
                            // println!("RDMANotify, response: {:?}", response);
                            if response.event & EVENT_IN != 0 {
                                let channelToSockInfos = rdmaSvcCli.channelToSockInfos.lock();
                                let sockInfo = channelToSockInfos.get(&response.channelId).unwrap();
                                let mut readBuf = sockInfo.sockBuff.readBuf.lock();

                                /////send RDMAAcceptResp begin
                                // let msgSize = mem::size_of::<RDMAAcceptResp>();
                                // let mut left = msgSize as u64;
                                // let r: Vec<u8> = Vec::with_capacity(msgSize as usize);
                                // let rAddr = r.as_ptr() as u64;

                                // while left != 0 {
                                //     let (trigger, len) = readBuf.readViaAddr(rAddr + (msgSize as u64 - left), left);
                                //     sockInfo.sockBuff.AddConsumeReadData(len as u64);
                                //     left = left - len as u64;
                                // }
                                // let msg = unsafe { &*(rAddr as *mut RDMAAcceptResp) };
                                // // println!("Read data, msg: {:?}", msg);
                                // let count = 1000000;
                                // if msg.sockfd < count {
                                //     index = msg.sockfd + 1;
                                //     let msg = RDMAAcceptResp {
                                //         sockfd: index,
                                //         ioBufIndex: index,
                                //         channelId: index,
                                //         dstIpAddr: index,
                                //         dstPort: index as u16,
                                //     };
                                //     let mut writeBuf = sockInfo.sockBuff.writeBuf.lock();
                                //     let (trigger, len) = writeBuf.writeViaAddr(
                                //         &msg as *const _ as u64,
                                //         mem::size_of::<RDMAAcceptResp>() as u64,
                                //     );
                                //     // println!("Client write trigger: {}, len: {}", trigger, len);
                                //     let sockfd = sockInfo.fd;
                                //     std::mem::drop(writeBuf);
                                //     // std::mem::drop(channelToSockInfos);
                                //     // std::mem::drop(shareRegion);
                                //     // println!("Writing some data to sockFdInfo 2!");
                                //     if trigger {
                                //         // println!("Writing some data to sockFdInfo 2.1!");
                                //         rdmaSvcCli.write(sockfd, &mut shareRegion.sq);
                                //     }
                                // }
                                /////send RDMAAcceptResp end

                                ////send random size begin

                                // while left != 0 {
                                //     let (trigger, len) = readBuf
                                //         .readViaAddr(rAddr + (randomMsgSize as u64 - left), left);
                                //     sockInfo.sockBuff.AddConsumeReadData(len as u64);
                                //     if len != 0 {
                                //         println!("read len: {}", len);
                                //     }

                                //     left = left - len as u64;
                                // }
                                loop {
                                    let (trigger, len) = readBuf.readViaAddr(rAddr, rbSize as u64);
                                    sockInfo.sockBuff.AddConsumeReadData(len as u64);
                                    readData += len;
                                    // if len != 0 {
                                    //     println!("read len: {}, readData: {}", len, readData);
                                    // }
                                    // else {
                                    //     break;
                                    // }
                                    if len == 0 {
                                        break;
                                    }
                                }

                                if readData >= (randomMsgSize as usize) {
                                    readData = 0;
                                    if index < count {
                                        index = index + 1;

                                        let mut writeBuf = sockInfo.sockBuff.writeBuf.lock();
                                        let (trigger, len) =
                                            writeBuf.writeViaAddr(wAddr, randomMsgSize);
                                        // if index % 10000 == 0 {
                                        println!(
                                            "Client write index: {}, trigger: {}, len: {}",
                                            index, trigger, len
                                        );
                                        // }

                                        // println!("Client write index: {}, trigger: {}, len: {}", index, trigger, len);

                                        let sockfd = sockInfo.fd;
                                        std::mem::drop(writeBuf);
                                        // std::mem::drop(channelToSockInfos);
                                        // std::mem::drop(shareRegion);
                                        // println!("Writing some data to sockFdInfo 2!");
                                        if trigger {
                                            // println!("Writing some data to sockFdInfo 2.1!");
                                            rdmaSvcCli.write(*sockInfo.channelId.lock());
                                        }
                                    }
                                    ////send random size end
                                    else {
                                        println!(">={}", count);
                                    }
                                }
                            }
                            if response.event & EVENT_OUT != 0 {
                                // let channelToSockInfos = rdmaSvcCli.channelToSockInfos.lock();
                                // let sockInfo = channelToSockInfos.get(&response.channelId).unwrap();
                                // let r: Vec<u8> = Vec::with_capacity(randomMsgSize as usize);
                                println!("EVENT_OUT is triggered");
                            }
                        }
                        RDMARespMsg::RDMAFinNotify(response) => {
                            println!("RDMAFinNotify, response: {:?}", response);
                        }
                    },
                    None => {
                        // println!("Finish procesing response from RDMASvc");
                        break;
                    }
                }
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
