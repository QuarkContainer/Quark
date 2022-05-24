use alloc::slice;
use alloc::sync::Arc;
use core::sync::atomic::Ordering;
use spin::{Mutex, MutexGuard};
use std::net::Ipv4Addr;
use std::ops::{Deref, DerefMut};
use std::str::FromStr;
use std::{env, mem, ptr, thread, time};

use super::qlib::common::*;
use super::qlib::rdma_share::*;
use super::unix_socket::UnixSocket;

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
}

//TODO: implement default

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
        }))
    }

    pub fn init(path: &str) -> RDMASvcClient {
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

    pub fn initialize(path: &str) -> Self {
        let cli_sock = UnixSocket::NewClient(path).unwrap();

        let body = 1;
        let ptr = &body as *const _ as *const u8;
        let buf = unsafe { slice::from_raw_parts(ptr, 4) };
        cli_sock.WriteWithFds(buf, &[]).unwrap();

        let mut body = [0, 0];
        let ptr = &mut body as *mut _ as *mut u8;
        let buf = unsafe { slice::from_raw_parts_mut(ptr, 8) };
        let (_size, fds) = cli_sock.ReadWithFds(buf).unwrap();

        let rdmaSvcCli = RDMASvcClient::New(fds[0], fds[1], fds[2], fds[3], body[1], cli_sock);
        rdmaSvcCli
    }

    pub fn listen(&self, sockfd: u32, endpoint: &Endpoint, waitingLen: i32) -> Result<()> {
        let res = self.SentMsgToSvc(RDMAReqMsg::RDMAListen(RDMAListenReq {
            sockfd: sockfd,
            ipAddr: endpoint.ipAddr,
            port: endpoint.port,
            waitingLen,
        }));
        res
    }

    pub fn connect(&self, sockfd: u32, ipAddr: u32, port: u16) -> Result<()> {
        let res = self.SentMsgToSvc(RDMAReqMsg::RDMAConnect(RDMAConnectReq {
            sockfd,
            dstIpAddr: ipAddr,
            dstPort: port,
            srcIpAddr: u32::from(Ipv4Addr::from_str("192.168.6.6").unwrap()).to_be(),
            srcPort: 16866u16.to_be(),
        }));
        res
    }

    pub fn read(&self, channelId: u32) -> Result<()> {
        // println!("rdmaSvcCli::read 1");
        if self.cliShareRegion.lock().sq.Push(RDMAReq {
            user_data: 0,
            msg: RDMAReqMsg::RDMARead(RDMAReadReq {
                channelId: channelId,
            }),
        }) {
            // println!("rdmaSvcCli::read 2");
            self.updateBitmapAndWakeUpServerIfNecessary();
            Ok(())
        } else {
            // println!("rdmaSvcCli::read 3");
            return Err(Error::NoEnoughSpace);
        }
    }

    pub fn write(&self, channelId: u32) -> Result<()> {
        // println!("rdmaSvcCli::write 1");
        if self.cliShareRegion.lock().sq.Push(RDMAReq {
            user_data: 0,
            msg: RDMAReqMsg::RDMAWrite(RDMAWriteReq {
                channelId: channelId,
            }),
        }) {
            // println!("rdmaSvcCli::write 2");
            self.updateBitmapAndWakeUpServerIfNecessary();
            Ok(())
        } else {
            // println!("rdmaSvcCli::write 3");
            return Err(Error::NoEnoughSpace);
        }
    }

    pub fn shutdown(&self, channelId: u32, howto: u8) -> Result<()> {
        // println!(
        //     "rdmaSvcCli::shutdown 1, channelId: {}, howto: {}",
        //     channelId, howto
        // );
        if self.cliShareRegion.lock().sq.Push(RDMAReq {
            user_data: 0,
            msg: RDMAReqMsg::RDMAShutdown(RDMAShutdownReq {
                channelId: channelId,
                howto,
            }),
        }) {
            // println!("rdmaSvcCli::shutdown 2");
            self.updateBitmapAndWakeUpServerIfNecessary();
            Ok(())
        } else {
            // println!("rdmaSvcCli::shutdown 3");
            return Err(Error::NoEnoughSpace);
        }
    }

    pub fn updateBitmapAndWakeUpServerIfNecessary(&self) {
        // println!("updateBitmapAndWakeUpServerIfNecessary 1 ");
        let mut srvShareRegion = self.srvShareRegion.lock();
        // println!("updateBitmapAndWakeUpServerIfNecessary 2 ");
        srvShareRegion.updateBitmap(self.agentId);
        if srvShareRegion.srvBitmap.load(Ordering::Relaxed) == 1 {
            // println!("before write srvEventFd");
            let data = 16u64;
            let ret = unsafe {
                libc::write(
                    self.srvEventFd,
                    &data as *const _ as *const libc::c_void,
                    mem::size_of_val(&data) as usize,
                )
            };
            // println!("ret: {}", ret);
            if ret < 0 {
                println!("error: {}", std::io::Error::last_os_error());
            }
        } else {
            // println!("server is not sleeping");
            self.updateBitmapAndWakeUpServerIfNecessary();
        }
    }

    pub fn SentMsgToSvc(&self, msg: RDMAReqMsg) -> Result<()> {
        if self
            .cliShareRegion
            .lock()
            .sq
            .Push(RDMAReq { user_data: 0, msg })
        {
            self.updateBitmapAndWakeUpServerIfNecessary();
            Ok(())
        } else {
            return Err(Error::NoEnoughSpace);
        }
    }
}
