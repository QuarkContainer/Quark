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

use alloc::sync::{Arc, Weak};
use core::mem;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use libc::*;
use spin::Mutex;
use std::ops::{Deref, DerefMut};

use super::qlib::common::*;
use super::rdma::*;
use super::rdma_channel::*;

use super::qlib::linux_def::*;
use super::qlib::rdma_share::*;
use super::qlib::socket_buf::*;
use super::rdma_srv::RDMA_SRV;

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
    raddr: u64,  /* Read Buffer address */
    rlen: u32,   /* Read Buffer len */
    rkey: u32,   /* Read Buffer Remote key */
    qp_num: u32, /* QP number */
    lid: u16,    /* LID of the IB port */
    // offset: u32,    //read buffer offset
    // freespace: u32, //read buffer free space size
    gid: Gid, /* gid */
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
    //pub ctrlChan: Mutex<RDMAControlChannel>,
    pub ctrlChan: Mutex<RDMAControlChannel1>,
    pub socketState: AtomicU64,
    pub localRDMAInfo: RDMAInfo,
    pub remoteRDMAInfo: Mutex<RDMAInfo>,
}

#[derive(Clone)]
pub struct RDMAConn(Arc<RDMAConnInternal>);

impl Deref for RDMAConn {
    type Target = RDMAConnInternal;

    fn deref(&self) -> &RDMAConnInternal {
        &self.0
    }
}

// impl DerefMut for RDMAConn {
//     fn deref_mut(&mut self) -> &mut RDMAConnInternal {
//         &mut self.0
//     }
// }

impl RDMAConn {
    pub fn New(fd: i32, sockBuf: Arc<SocketBuff>, rkey: u32) -> Self {
        let qp = RDMA.CreateQueuePair().expect("RDMADataSock create QP fail");
        println!("after create qp");
        let (addr, len) = sockBuf.ReadBuf();
        let localRDMAInfo = RDMAInfo {
            qp_num: qp.qpNum(),
            lid: RDMA.Lid(),
            gid: RDMA.Gid(),
            raddr: addr,
            rlen: len as u32,
            rkey,
        };

        //TODO: may find a better place to update controlChannels.
        //RDMA_SRV.controlChannels.lock().insert(qp.qpNum(), rdmaChannel.clone());
        Self(Arc::new(RDMAConnInternal {
            fd: fd,
            qps: vec![qp],
            //ctrlChan: Mutex::new(RDMAControlChannel(Weak::new())),
            ctrlChan: Mutex::new(RDMAControlChannel1::default()),
            socketState: AtomicU64::new(0),
            localRDMAInfo: localRDMAInfo,
            remoteRDMAInfo: Mutex::new(RDMAInfo::default()),
        }))
    }
    // pub fn New(fd: i32, sockBuf: Arc<SocketBuff>, rdmaChannel: Arc<RDMAChannel>) -> Arc<Self> {
    //     let qp = RDMA.CreateQueuePair().expect("RDMADataSock create QP fail");
    //     println!("after create qp");
    //     let (addr, len) = sockBuf.ReadBuf();
    //     let localRDMAInfo = RDMAInfo {
    //         qp_num: qp.qpNum(),
    //         lid: RDMA.Lid(),
    //         gid: RDMA.Gid(),
    //         raddr: addr,
    //         rlen: len as u32,
    //         rkey: rdmaChannel.RemoteKey()
    //     };

    //     //TODO: may find a better place to update controlChannels.
    //     RDMA_SRV.controlChannels.lock().insert(qp.qpNum(), rdmaChannel.clone());
    //     let s = Arc::new(Self(RDMAConnInternal {
    //         fd: fd,
    //         qps: vec![qp],
    //         ctrlChan: rdmaChannel.clone(),
    //         socketState: AtomicU64::new(0),
    //         localRDMAInfo: localRDMAInfo,
    //         remoteRDMAInfo: Mutex::new(RDMAInfo::default()),
    //     }));
    //     rdmaChannel.SetRDMAConn(s.clone());
    //     s
    // }

    // pub fn SetRDMAControlChannel(&mut self, rdmaChannel: Arc<RDMAChannel>) {
    //     self.ctrlChan = Arc::downgrade(&rdmaChannel);
    // }

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

    pub fn GetQueuePairs(&self) -> &Vec<QueuePair> {
        &self.qps
    }

    pub fn Read(&self) {
        match self.SocketState() {
            SocketState::WaitingForRemoteMeta => {
                match self.RecvRemoteRDMAInfo() {
                    Ok(()) => {
                        println!("Received remote RDMA Info");
                    }
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
                    Ok(()) => {}
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
        self.ctrlChan
            .lock()
            .chan
            .upgrade()
            .unwrap()
            .UpdateRemoteRDMAInfo(data.raddr, data.rlen, data.rkey);
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

    pub fn RDMAWriteImm(
        &self,
        localAddr: u64,
        remoteAddr: u64,
        length: usize,
        wrId: u64,
        imm: u32,
        lkey: u32,
        rkey: u32,
    ) -> Result<()> {
        self.qps[0].WriteImm(wrId, localAddr, length as u32, lkey, remoteAddr, rkey, imm)?;
        return Ok(());
    }

    pub fn PostRecv(&self, qpNum: u32, wrId: u64, addr: u64, lkey: u32) -> Result<()> {
        //TODO: get right qp when multiple QP are used between two physical machines.
        self.qps[0].PostRecv(wrId, addr, lkey)?;
        return Ok(());
    }
}

pub struct RDMAControlChannelIntern {
    pub chan: Weak<RDMAChannelIntern>,
    pub buf: Vec<u8>,
}

#[derive(Clone)]
pub struct RDMAControlChannel1(Arc<RDMAControlChannelIntern>);

impl Default for RDMAControlChannel1 {
    fn default() -> Self {
        Self(Arc::new(RDMAControlChannelIntern {
            chan: Weak::new(),
            buf: Vec::new()
        }))
    }
}

impl Deref for RDMAControlChannel1 {
    type Target = Arc<RDMAControlChannelIntern>;

    fn deref(&self) -> &Arc<RDMAControlChannelIntern> {
        &self.0
    }
}

impl RDMAControlChannel1{
    pub fn New(rdmaChannelIntern: Arc<RDMAChannelIntern>) -> Self {
        Self(Arc::new(RDMAControlChannelIntern {
            chan: Arc::downgrade(&rdmaChannelIntern),
            buf: Vec::with_capacity(1024)
        }))
    }

    pub fn ProcessRDMAWriteImmFinish(&self) {
        println!("ProcessRDMAWriteImmFinish");
    }

    pub fn ProcessRDMARecvWriteImm(
        &self,
        qpNum: u32,
        recvCount: u64
    ) {
        println!("ProcessRDMARecvWriteImm");
    }
}

pub struct RDMAControlChannel(Weak<RDMAChannelIntern>);

impl Deref for RDMAControlChannel {
    type Target = Weak<RDMAChannelIntern>;

    fn deref(&self) -> &Weak<RDMAChannelIntern> {
        &self.0
    }
}

impl RDMAControlChannel {
    pub fn New(rdmaChannelIntern: Arc<RDMAChannelIntern>) -> Self {
        Self(Arc::downgrade(&rdmaChannelIntern))
    }
}
