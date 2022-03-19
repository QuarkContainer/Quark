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

pub mod rdma;
pub mod rdma_agent;
pub mod rdma_channel;
pub mod rdma_conn;
pub mod rdma_ctrlconn;
pub mod rdma_srv;

use crate::rdma_srv::RDMA_CTLINFO;
use crate::rdma_srv::RDMA_SRV;

use self::qlib::ShareSpaceRef;
use std::collections::HashMap;
use std::io;
use std::io::prelude::*;
use std::net::{IpAddr, Ipv4Addr, TcpListener, TcpStream};
use std::os::unix::io::{AsRawFd, RawFd};
pub static SHARE_SPACE: ShareSpaceRef = ShareSpaceRef::New();
use crate::qlib::rdma_share::ShareRegion;
use crate::rdma::RDMA;
use local_ip_address::list_afinet_netifas;
use local_ip_address::local_ip;
use qlib::linux_def::*;
use rdma_conn::RDMAConn;
use rdma_ctrlconn::Node;
use std::io::Error;
use std::str::FromStr;
use std::{env, mem};

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
    pub U64: u64
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
    println!("RDMA Service is starting!");
    println!("size of RDMAConn: {}", mem::size_of::<RDMAConn>()); 
    //TODO: make devicename and port configurable
    RDMA.Init("", 1);
    println!("size is: {}", mem::size_of::<qlib::rdma_share::ShareRegion>()); 

    // hashmap for file descriptors so that different handling can be dispatched.
    let mut fds: HashMap<i32, FdType> = HashMap::new();

    let epoll_fd = epoll_create().expect("can create epoll queue");
    let mut events: Vec<EpollEvent> = Vec::with_capacity(1024);

    let args: Vec<_> = env::args().collect();
    let server_fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
    println!("server_fd is {}", server_fd);
    unblock_fd(server_fd);
    fds.insert(server_fd, FdType::TCPSocketServer);
    epoll_add(epoll_fd, server_fd, read_write_event(server_fd as u64))?;

    unsafe {
        let mut serv_addr: libc::sockaddr_in = libc::sockaddr_in {
            sin_family: libc::AF_INET as u16,
            sin_port: 8888u16.to_be(),
            sin_addr: libc::in_addr {
                s_addr: u32::from_be_bytes([0, 0, 0, 0]).to_be(),
            },
            sin_zero: mem::zeroed(),
        };

        if args.len() > 1 {
            serv_addr = libc::sockaddr_in {
                sin_family: libc::AF_INET as u16,
                sin_port: 8889u16.to_be(),
                sin_addr: libc::in_addr {
                    s_addr: u32::from_be_bytes([0, 0, 0, 0]).to_be(),
                },
                sin_zero: mem::zeroed(),
            };
        }

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


    let local_ip = get_local_ip();
    println!("litener sock fd is {}", server_fd);

    let cur_timestamp = RDMA_CTLINFO.nodes.lock().get(&local_ip).unwrap().timestamp;
    println!("timestamp is {}", cur_timestamp);

    // connect to other RDMA service on nodes which timestamp is bigger
    // for (ipAddr, node) in RDMA_CTLINFO.nodes.lock().iter() {
    //     if cur_timestamp < node.timestamp {
    if args.len() > 1 {
        let node = Node {
            //ipAddr: u32::from(Ipv4Addr::from_str("6.1.16.172").unwrap()),
            ipAddr: u32::from(Ipv4Addr::from_str("172.16.1.6").unwrap()).to_be(),
            timestamp: 0,
            subnet: u32::from(Ipv4Addr::from_str("172.16.1.0").unwrap()),
            netmask: u32::from(Ipv4Addr::from_str("255.255.255.0").unwrap()),
        };
        let sock_fd = unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0) };
        println!("sock_fd is {}", sock_fd);
        unblock_fd(sock_fd);
        fds.insert(sock_fd, FdType::TCPSocketConnect(node.ipAddr));
        epoll_add(epoll_fd, sock_fd, read_write_event(sock_fd as u64))?;

        println!("new conn");
        let rdmaconn = RDMAConn::New(sock_fd);
        println!("before insert");
        RDMA_SRV
            .lock()
            .conns
            .insert(node.ipAddr, rdmaconn);
        println!("after insert");
        unsafe {
            let serv_addr: libc::sockaddr_in = libc::sockaddr_in {
                sin_family: libc::AF_INET as u16,
                sin_port: 8888u16.to_be(),
                sin_addr: libc::in_addr {
                    s_addr: node.ipAddr,
                },
                sin_zero: mem::zeroed(),
            };
            let ret = libc::connect(
                sock_fd,
                &serv_addr as *const libc::sockaddr_in as *const libc::sockaddr,
                mem::size_of_val(&serv_addr) as u32,
            );

            println!("ret is {}, error: {}", ret, Error::last_os_error());
        }
        // }
    }
    // }

    loop {
        events.clear();
        println!("in loop");
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

        println!("res is: {}", res);

        for ev in &events {
            print!("u64: {}, events: {:x}", ev.U64, ev.Events);
            let event_data = fds.get(&(ev.U64 as i32));
            match event_data {
                Some(FdType::TCPSocketServer) => {
                    let stream_fd;
                    let mut cliaddr: libc::sockaddr_in = unsafe { mem::zeroed() };
                    let mut len = mem::size_of_val(&cliaddr) as u32;
                    unsafe {
                        stream_fd = libc::accept(
                            ev.U64 as i32,
                            &mut cliaddr as *mut libc::sockaddr_in as *mut libc::sockaddr,
                            &mut len,
                        );
                    }
                    unblock_fd(stream_fd);
                    println!("stream_fd is: {}", stream_fd);

                    let peerIpAddrU32 = cliaddr.sin_addr.s_addr;
                    
                    fds.insert(stream_fd, FdType::TCPSocketConnect(peerIpAddrU32));

                    RDMA_SRV
                        .lock()
                        .conns
                        .insert(peerIpAddrU32, RDMAConn::New(stream_fd));
                    epoll_add(epoll_fd, stream_fd, read_write_event(stream_fd as u64))?;
                    println!("add stream fd");
                },
                Some(FdType::TCPSocketConnect(ipAddr)) => match RDMA_SRV.lock().conns.get(ipAddr) {
                    Some(rdmaConn) => {
                        rdmaConn.Notify(ev.Events as u64);
                    }
                    _ => {
                        panic!("no RDMA connection for {} found!", ipAddr)
                    }
                },
                Some(FdType::RDMACompletionChannel) => {
                    println!("xx");
                }
                Some(FdType::UnixDomainSocketConnect) => {
                    println!("xx");
                }
                Some(FdType::UnixDomainSocketServer) => {
                    println!("xx");
                }
                None => {
                    panic!("unexpected fd {} found", ev.U64);
                }
            }
        }
    }
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
