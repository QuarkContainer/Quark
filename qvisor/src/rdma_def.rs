use alloc::collections::BTreeMap;
use alloc::slice;
use alloc::sync::Arc;
use spin::Mutex;
use std::{mem, ptr};
use core::sync::atomic::AtomicU32;

use super::qlib::linux_def::*;
use super::qlib::rdma_share::*;
use super::qlib::rdma_svc_cli::*;
use super::qlib::unix_socket::UnixSocket;
use super::vmspace::VMSpace;

impl Drop for RDMASvcClient {
    fn drop(&mut self) {
        error!("RDMASvcClient::Drop");
    }
}

impl RDMASvcClient {
    fn New(
        srvEventFd: i32,
        srvMemFd: i32,
        cliEventFd: i32,
        cliMemFd: i32,
        agentId: u32,
        cliSock: UnixSocket,
        localShareAddr: u64,
        globalShareAddr: u64,
        podId: [u8; 64],
    ) -> Self {
        let cliShareSize = mem::size_of::<ClientShareRegion>();
        // debug!("RDMASvcClient::New, cli size: {:x}", cliShareSize);
        // debug!("RDMASvcClient::New, srv size: {:x}", mem::size_of::<ShareRegion>());
        // debug!("RDMASvcClient::New, ioBuffer: {:x}", mem::size_of::<IOBuf>());
        // debug!("RDMASvcClient::New, IOMetas: {:x}", mem::size_of::<IOMetas>());
        // debug!("RDMASvcClient::New, RingQueue<RDMAResp>: {:x}", mem::size_of::<RingQueue<RDMAResp>>());
        // debug!("RDMASvcClient::New, RingQueue<RDMAResp>: {:x}", mem::size_of::<RingQueue<RDMAReq>>());
        // debug!("RDMASvcClient::New, RDMAResp: {:x}", mem::size_of::<RDMAResp>());
        // debug!("RDMASvcClient::New, RDMAReq: {:x}", mem::size_of::<RDMAReq>());

        let cliShareAddr = unsafe {
            libc::mmap(
                if localShareAddr == 0 {
                    ptr::null_mut()
                } else {
                    localShareAddr as *mut libc::c_void
                },
                cliShareSize,
                libc::PROT_READ | libc::PROT_WRITE,
                if localShareAddr == 0 {
                    libc::MAP_SHARED
                } else {
                    libc::MAP_SHARED | libc::MAP_FIXED
                },
                cliMemFd,
                0,
            )
        };
        assert!(cliShareAddr as u64 == localShareAddr || localShareAddr == 0);

        let cliShareRegion = unsafe { &mut (*(cliShareAddr as *mut ClientShareRegion)) };

        let cliShareRegion = Mutex::new(cliShareRegion);

        let srvShareSize = mem::size_of::<ShareRegion>();
        let srvShareAddr = unsafe {
            libc::mmap(
                if globalShareAddr == 0 {
                    ptr::null_mut()
                } else {
                    globalShareAddr as *mut libc::c_void
                },
                srvShareSize,
                libc::PROT_READ | libc::PROT_WRITE,
                if globalShareAddr == 0 {
                    libc::MAP_SHARED
                } else {
                    libc::MAP_SHARED | libc::MAP_FIXED
                },
                srvMemFd,
                0,
            )
        };
        assert!(srvShareAddr as u64 == globalShareAddr || globalShareAddr == 0);

        let srvShareRegion = unsafe { &mut (*(srvShareAddr as *mut ShareRegion)) };
        let srvShareRegion = Mutex::new(srvShareRegion);
        RDMASvcClient {
            intern: Arc::new(RDMASvcCliIntern {
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
                channelToSocketMappings: Mutex::new(BTreeMap::new()),
                rdmaIdToSocketMappings: Mutex::new(BTreeMap::new()),
                nextRDMAId: AtomicU32::new(0),
                podId,
            }),
        }
    }

    // pub fn init(path: &str) -> RDMASvcClient {
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

    //     let rdmaSvcCli = RDMASvcClient::New(fds[0], fds[1], fds[2], fds[3], body[1], cli_sock);
    //     rdmaSvcCli
    // }

    pub fn initialize(cliSock: i32, localShareAddr: u64, globalShareAddr: u64, podId:[u8; 64]) -> Self {
        // let cli_sock = UnixSocket::NewClient(path).unwrap();
        let cli_sock = UnixSocket { fd: cliSock };

        let body = 1;
        let ptr = &body as *const _ as *const u8;
        let buf = unsafe { slice::from_raw_parts(ptr, 4) };
        cli_sock.WriteWithFds(buf, &[]).unwrap();

        let mut body = [0, 0];
        let ptr = &mut body as *mut _ as *mut u8;
        let buf = unsafe { slice::from_raw_parts_mut(ptr, 8) };
        let (_size, fds) = cli_sock.ReadWithFds(buf).unwrap();
        let rdmaSvcCli = RDMASvcClient::New(
            fds[0],
            fds[1],
            fds[2],
            fds[3],
            body[1],
            cli_sock,
            localShareAddr,
            globalShareAddr,
            podId,
        );
        rdmaSvcCli
    }

    pub fn wakeupSvc(&self) {
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
    }

    pub fn CreateSocket(&self) -> i64 {
        VMSpace::Socket(AFType::AF_INET, 1, 0)
    }
}