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

use alloc::sync::Arc;
use spin::Mutex;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use libc::*;
use core::mem;
use std::ops::Deref;

use super::rdma_channel::*;
use super::rdma::*;
use super::qlib::common::*;

use super::qlib::linux_def::*;

// RDMA Queue Pair
pub struct RDMAQueuePair {}

#[derive(Debug)]
#[repr(u64)]
pub enum SocketState {
    Init,
    Connect,
    WaitingForRemoteMeta,
    WaitingForRemoteReady,
    Ready,
    Error,
}

#[derive(Clone, Default)]
#[repr(C)]
pub struct RDMAInfo {
    // raddr: u64,     /* Read Buffer address */
    // rlen: u32,      /* Read Buffer len */
    // rkey: u32,      /* Read Buffer Remote key */
    qp_num: u32,    /* QP number */
    lid: u16,       /* LID of the IB port */
    // offset: u32,    //read buffer offset
    // freespace: u32, //read buffer free space size
    gid: Gid,       /* gid */
    // sending: bool,  // the writeimmediately is ongoing
}

impl RDMAInfo {
    pub fn Size() -> usize {
        return mem::size_of::<Self>();
    }
}

// RDMA connections between 2 nodes
pub struct RDMAConnInternal {
    pub fd: i32,
    pub qps: Vec<QueuePair>,
    pub ctrlChan: Option<RDMAControlChannel>,
    pub socketState: AtomicU64,
    pub localRDMAInfo: RDMAInfo,
    pub remoteRDMAInfo: Mutex<RDMAInfo>,
}

pub struct RDMAConn(Arc<RDMAConnInternal>);

impl Deref for RDMAConn {
    type Target = Arc<RDMAConnInternal>;

    fn deref(&self) -> &Arc<RDMAConnInternal> {
        &self.0
    }
}

impl RDMAConn {
    pub fn New(fd: i32) -> Self {
        let qp = RDMA.CreateQueuePair().expect("RDMADataSock create QP fail");
        println!("after create qp");
        let localRDMAInfo = RDMAInfo {
            qp_num: qp.qpNum(),
            lid: RDMA.Lid(),
            gid: RDMA.Gid(),
        };
        return Self(Arc::new(RDMAConnInternal {
            fd: fd,
            qps: vec![qp],
            ctrlChan: None,  //TODO: initialize
            socketState: AtomicU64::new(0),
            localRDMAInfo: localRDMAInfo,
            remoteRDMAInfo: Mutex::new(RDMAInfo::default()),
        }))
    }

    pub fn SetupRDMA(&self) {
        let remoteInfo = self.remoteRDMAInfo.lock();
        self.qps[0]
            .Setup(&RDMA, remoteInfo.qp_num, remoteInfo.lid, remoteInfo.gid)
            .expect("SetupRDMA fail...");
        // for _i in 0..MAX_RECV_WR {
        //     let wr = WorkRequestId::New(self.fd);
        //     self.qp
        //         .lock()
        //         .PostRecv(wr.0, self.localRDMAInfo.raddr, self.localRDMAInfo.rkey)
        //         .expect("SetupRDMA PostRecv fail");
        // }
    }

    pub fn Read(&self) {
        match self.SocketState() {
            SocketState::WaitingForRemoteMeta => {
                match self.RecvRemoteRDMAInfo() {
                    Ok(()) => { println!("Received remote RDMA Info"); },
                    _ => return,
                }
                println!("SetupRDMA");
                self.SetupRDMA();
                println!("SendAck");
                self.SendAck().unwrap(); // assume the socket is ready for send
                println!("Set state WaitingForRemoteReady");
                self.SetSocketState(SocketState::WaitingForRemoteReady);

                match self.RecvAck() {
                    Ok(()) => {
                        self.SetReady();
                    }
                    _ => (),
                }
            }
            SocketState::WaitingForRemoteReady => {
                match self.RecvAck() {
                    Ok(()) => {},
                    _ => return,
                }
                self.SetReady();
            }
            SocketState::Ready => {
                println!("Read::Ready");
            }
            _ => {
                panic!(
                    "RDMA socket read state error with state {:?}",
                    self.SocketState()
                )
            }
        }
    }

    pub fn Write(&self) {
        match self.SocketState() {
            SocketState::Init => {
                self.SendLocalRDMAInfo().unwrap();
                println!("local RDMAInfo sent");
                self.SetSocketState(SocketState::WaitingForRemoteMeta);
            }
            SocketState::WaitingForRemoteMeta => {
                println!("Write::1");
                //TODO: server side received 4(W) first and 5 (R|W) afterwards. Need more investigation to see why it's different.
            }
            SocketState::WaitingForRemoteReady => {
                println!("Write::2");
                //TODO: server side received 4(W) first and 5 (R|W) afterwards. Need more investigation to see why it's different.
            }
            SocketState::Ready => {
                println!("Write::Ready");
            }
            _ => {
                panic!(
                    "RDMA socket Write state error with state {:?}",
                    self.SocketState()
                )
            }
        }
    }

    pub fn SocketState(&self) -> SocketState {
        let state = self.socketState.load(Ordering::Relaxed);
        assert!(state <= SocketState::Ready as u64);
        let state: SocketState = unsafe { mem::transmute(state) };
        return state;
    }

    pub fn SetSocketState(&self, state: SocketState) {
        self.socketState.store(state as u64, Ordering::SeqCst)
    }

    pub fn SetReady(&self) {
        self.SetSocketState(SocketState::Ready);
        println!("Ready!!!")
    }

    pub fn SendLocalRDMAInfo(&self) -> Result<()> {
        let ret = unsafe {
            write(
                self.fd,
                &self.localRDMAInfo as *const _ as u64 as _,
                RDMAInfo::Size(),
            )
        };

        if ret < 0 {
            let errno = errno::errno().0;
            println!("SendLocalRDMAInfo, err: {}", errno);
            return Err(Error::SysError(errno));
        }

        assert!(
            ret == RDMAInfo::Size() as isize,
            "SendLocalRDMAInfo fail ret is {}, expect {}",
            ret,
            RDMAInfo::Size()
        );
        return Ok(());
    }

    pub fn RecvRemoteRDMAInfo(&self) -> Result<()> {
        let mut data = RDMAInfo::default();
        let ret = unsafe { read(self.fd, &mut data as *mut _ as u64 as _, RDMAInfo::Size()) };

        if ret < 0 {
            let errno = errno::errno().0;
            // debug!("RecvRemoteRDMAInfo, err: {}", errno);
            //self.socketBuf.SetErr(errno);
            println!("read error: {:x}", errno);
            return Err(Error::SysError(errno));
        }

        //self.socketBuf.SetErr(0); //TODO: find a better place

        assert!(
            ret == RDMAInfo::Size() as isize,
            "SendLocalRDMAInfo fail ret is {}, expect {}",
            ret,
            RDMAInfo::Size()
        );

        *self.remoteRDMAInfo.lock() = data;

        return Ok(());
    }

    pub const ACK_DATA: u64 = 0x1234567890;
    pub fn SendAck(&self) -> Result<()> {
        let data: u64 = Self::ACK_DATA;
        let ret = unsafe { write(self.fd, &data as *const _ as u64 as _, 8) };
        if ret < 0 {
            let errno = errno::errno().0;
            
            return Err(Error::SysError(errno));
        }

        assert!(ret == 8, "SendAck fail ret is {}, expect {}", ret, 8);
        return Ok(());
    }

    pub fn RecvAck(&self) -> Result<()> {
        let mut data = 0;
        let ret = unsafe { read(self.fd, &mut data as *mut _ as u64 as _, 8) };

        if ret < 0 {
            let errno = errno::errno().0;
            // debug!("RecvAck::1, err: {}", errno);
            if errno == SysErr::EAGAIN {
                return Err(Error::SysError(errno));
            }
            // debug!("RecvAck::2, err: {}", errno);
            return Err(Error::SysError(errno));
        }

        assert!(
            ret == 8 as isize,
            "RecvAck fail ret is {}, expect {}",
            ret,
            8
        );
        assert!(
            data == Self::ACK_DATA,
            "RecvAck fail data is {:x}, expect {:x}",
            ret,
            Self::ACK_DATA
        );

        return Ok(());
    }

    pub fn Notify(&self, eventmask: EventMask) {
        println!("RDMAConn::Notify 1");
        if eventmask & EVENT_WRITE != 0 {
            println!("RDMAConn::Notify 2");
            self.Write();
        }

        if eventmask & EVENT_READ != 0 {
            println!("RDMAConn::Notify 3");
            self.Read();
        }
    }
}

pub struct RDMAControlChannel(RDMAChannelWeak);