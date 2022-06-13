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

pub mod common;
pub mod rdma_def;
pub mod unix_socket_def;

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
use common::EpollEvent;
use common::*;
use qlib::linux_def::*;
use qlib::rdma_svc_cli::*;
use qlib::socket_buf::SocketBuff;
use qlib::unix_socket::UnixSocket;
use std::str::FromStr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::{env, mem, ptr, thread, time};

fn main() -> io::Result<()> {
    let mut fds: HashMap<i32, FdType> = HashMap::new();
    let args: Vec<_> = env::args().collect();
    let gatewayCli: GatewayClient;
    let mut unix_sock_path = "/tmp/rdma_srv";
    if args.len() > 1 {
        unix_sock_path = args.get(1).unwrap(); //"/tmp/rdma_srv1";
    }
    gatewayCli = GatewayClient::initialize(unix_sock_path); //TODO: add 2 address from quark.

    let cliEventFd = gatewayCli.rdmaSvcCli.cliEventFd;
    unblock_fd(cliEventFd);
    unblock_fd(gatewayCli.rdmaSvcCli.srvEventFd);

    let epoll_fd = epoll_create().expect("can create epoll queue");
    epoll_add(epoll_fd, cliEventFd, read_event(cliEventFd as u64))?;
    fds.insert(cliEventFd, FdType::ClientEvent);

    // set up TCP Server to wait for incoming connection
    let server_fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
    fds.insert(server_fd, FdType::TCPSocketServer(6666));
    unblock_fd(server_fd);
    epoll_add(epoll_fd, server_fd, read_write_event(server_fd as u64))?;
    unsafe {
        // TODO: need mapping 6666 -> (172.16.1.6:8888) for testing purpose, late should come from control plane
        let serv_addr: libc::sockaddr_in = libc::sockaddr_in {
            sin_family: libc::AF_INET as u16,
            sin_port: 6666u16.to_be(),
            sin_addr: libc::in_addr {
                s_addr: u32::from_be_bytes([0, 0, 0, 0]).to_be(),
            },
            sin_zero: mem::zeroed(),
        };

        let result = libc::bind(
            server_fd,
            &serv_addr as *const libc::sockaddr_in as *const libc::sockaddr,
            mem::size_of_val(&serv_addr) as u32,
        );
        if result < 0 {
            libc::close(server_fd);
            panic!("last OS error: {:?}", Error::last_os_error());
        }
        libc::listen(server_fd, 128);
    }

    wait(epoll_fd, &gatewayCli, &mut fds);

    return Ok(());
}

fn wait(epoll_fd: i32, gatewayCli: &GatewayClient, fds: &mut HashMap<i32, FdType>) {
    let mut events: Vec<EpollEvent> = Vec::with_capacity(1024);
    let mut sockFdMappings: HashMap<u32, i32> = HashMap::new(); // mapping between sockfd maintained by rdmaSvcCli and fd for incoming requests.
    loop {
        events.clear();
        {
            gatewayCli
                .rdmaSvcCli
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

        for ev in &events {
            let event_data = fds.get(&(ev.U64 as i32));
            match event_data {
                Some(FdType::TCPSocketServer(_port)) => {
                    let mut stream_fd;
                    let mut cliaddr: libc::sockaddr_in = unsafe { mem::zeroed() };
                    let mut len = mem::size_of_val(&cliaddr) as u32;
                    loop {
                        unsafe {
                            stream_fd = libc::accept(
                                ev.U64 as i32,
                                &mut cliaddr as *mut libc::sockaddr_in as *mut libc::sockaddr,
                                &mut len,
                            );
                        }
                        if stream_fd > 0 {
                            unblock_fd(stream_fd);
                            let _ret =
                                epoll_add(epoll_fd, stream_fd, read_write_event(stream_fd as u64));

                            //TODO: use port to map to different (ip, port), hardcode for testing purpose, should come from control plane in the future
                            let sockfd = gatewayCli.sockIdMgr.lock().AllocId().unwrap(); //TODO: rename sockfd
                            let _ret = gatewayCli.connect(
                                sockfd,
                                u32::from(Ipv4Addr::from_str("192.168.6.8").unwrap()).to_be(),
                                16868u16.to_be(),
                            );
                            fds.insert(stream_fd, FdType::TCPSocketConnect(sockfd));
                            sockFdMappings.insert(sockfd, stream_fd);
                        } else {
                            break;
                        }
                    }
                }
                Some(FdType::TCPSocketConnect(sockfd)) => {
                    let mut sockInfo;
                    {
                        let mut sockFdInfos = gatewayCli.dataSockFdInfos.lock();
                        sockInfo = sockFdInfos.get_mut(sockfd).unwrap().clone();
                    }

                    if !matches!(*sockInfo.status.lock(), SockStatus::ESTABLISHED) {
                        continue;
                    }
                    if ev.Events & EVENT_IN as u32 != 0 {
                        gatewayCli.ReadFromSocket(&mut sockInfo, &sockFdMappings);
                    }
                    if ev.Events & EVENT_OUT as u32 != 0 {
                        gatewayCli.WriteToSocket(&mut sockInfo, &sockFdMappings);
                    }
                }
                Some(FdType::ClientEvent) => {
                    loop {
                        let request = gatewayCli.rdmaSvcCli.cliShareRegion.lock().cq.Pop();
                        match request {
                            Some(cq) => match cq.msg {
                                RDMARespMsg::RDMAConnect(response) => {
                                    let ioBufIndex = response.ioBufIndex as usize;
                                    let mut sockFdInfos = gatewayCli.dataSockFdInfos.lock();
                                    let sockInfo = sockFdInfos.get_mut(&response.sockfd).unwrap();
                                    // println!("RDMARespMsg::RDMAConnect, sockfd: {}, channelId: {}", sockInfo.fd, response.channelId);
                                    {
                                        let shareRegion =
                                            gatewayCli.rdmaSvcCli.cliShareRegion.lock();
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
                                                &shareRegion.ioMetas[ioBufIndex].readBufAtoms
                                                    as *const _
                                                    as u64,
                                                &shareRegion.ioMetas[ioBufIndex].writeBufAtoms
                                                    as *const _
                                                    as u64,
                                                &shareRegion.ioMetas[ioBufIndex].consumeReadData
                                                    as *const _
                                                    as u64,
                                                &shareRegion.iobufs[ioBufIndex].read as *const _
                                                    as u64,
                                                &shareRegion.iobufs[ioBufIndex].write as *const _
                                                    as u64,
                                                false,
                                            )),
                                        );
                                        sockFdInfos.insert(sockInfo.fd, sockInfo);
                                    }

                                    let sockInfo = sockFdInfos.get_mut(&response.sockfd).unwrap();
                                    gatewayCli
                                        .channelToSockInfos
                                        .lock()
                                        .insert(response.channelId, sockInfo.clone());

                                    gatewayCli.ReadFromSocket(sockInfo, &sockFdMappings);
                                }
                                RDMARespMsg::RDMAAccept(response) => {
                                    let mut sockFdInfos = gatewayCli.serverSockFdInfos.lock();
                                    let sockInfo = sockFdInfos.get_mut(&response.sockfd).unwrap();

                                    let ioBufIndex = response.ioBufIndex as usize;
                                    let dataSockFd = gatewayCli.sockIdMgr.lock().AllocId().unwrap();
                                    let shareRegion = gatewayCli.rdmaSvcCli.cliShareRegion.lock();
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
                                            false,
                                        )),
                                    );

                                    gatewayCli
                                        .dataSockFdInfos
                                        .lock()
                                        .insert(dataSockFd, dataSockInfo.clone());
                                    sockInfo.acceptQueue.lock().EnqSocket(dataSockFd);
                                    gatewayCli
                                        .channelToSockInfos
                                        .lock()
                                        .insert(response.channelId, dataSockInfo.clone());
                                }
                                RDMARespMsg::RDMANotify(response) => {
                                    if response.event & EVENT_IN != 0 {
                                        let mut sockInfo;
                                        {
                                            let mut channelToSockInfos =
                                                gatewayCli.channelToSockInfos.lock();
                                            sockInfo = channelToSockInfos
                                                .get_mut(&response.channelId)
                                                .unwrap()
                                                .clone();
                                        }
                                        gatewayCli.WriteToSocket(&mut sockInfo, &sockFdMappings);
                                    }
                                    if response.event & EVENT_OUT != 0 {
                                        let mut channelToSockInfos =
                                            gatewayCli.channelToSockInfos.lock();
                                        let sockInfo = channelToSockInfos
                                            .get_mut(&response.channelId)
                                            .unwrap();
                                        gatewayCli.ReadFromSocket(sockInfo, &sockFdMappings);
                                    }
                                }
                                RDMARespMsg::RDMAFinNotify(response) => {
                                    let mut sockInfo;
                                    {
                                        let mut channelToSockInfos =
                                            gatewayCli.channelToSockInfos.lock();
                                        sockInfo = channelToSockInfos
                                            .get_mut(&response.channelId)
                                            .unwrap()
                                            .clone();
                                    }
                                    if response.event & FIN_RECEIVED_FROM_PEER != 0 {
                                        *sockInfo.finReceived.lock() = true;
                                        gatewayCli.WriteToSocket(&mut sockInfo, &sockFdMappings);
                                    }
                                }
                            },
                            None => {
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
