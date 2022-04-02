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

use alloc::slice;
use alloc::sync::{Arc, Weak};
use core::mem;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use libc::*;
use spin::Mutex;
use std::ops::{Deref, DerefMut};
use std::{env, ptr, thread};

use super::qlib::common::*;
use super::rdma::*;
use super::rdma_channel::*;
use super::rdma_ctrlconn::*;

use super::qlib::linux_def::*;
use super::qlib::rdma_share::*;
use super::qlib::socket_buf::*;
use super::rdma_srv::*;

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
    pub ctrlChan: Mutex<RDMAControlChannel>,
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
            ctrlChan: Mutex::new(RDMAControlChannel::default()),
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
        for _i in 0..10 {
            self.qps[0]
                .PostRecv(0, self.localRDMAInfo.raddr, self.localRDMAInfo.rkey)
                .expect("SetupRDMA PostRecv fail");
        }
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
        println!("Ready!!!");
        self.ctrlChan.lock().SendData();
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
        match self.ctrlChan.lock().chan.upgrade() {
            None => {
                println!("ctrlChann is null")
            }
            _ => {
                println!("ctrlChann is not null")
            }
        }
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

    pub fn PostRecv(&self, _qpNum: u32, wrId: u64, addr: u64, lkey: u32) -> Result<()> {
        //TODO: get right qp when multiple QP are used between two physical machines.
        self.qps[0].PostRecv(wrId, addr, lkey)?;
        return Ok(());
    }
}

pub const CONTROL_MSG_SIZE: usize = 40;

pub struct RDMAControlChannelIntern {
    pub chan: Weak<RDMAChannelIntern>,
    // pub readHeadBuf: [u8; 8],
    // pub readHeadLeft: u32,
    // pub readBodyBuf: Vec<u8>,
    // pub readBodyLeft: u32,
    // pub writeBuf: Vec<u8>,
    // pub writeLeft: u32,
    pub readBuf: [u8; CONTROL_MSG_SIZE],
    pub readLeft: u8,
    pub writeBuf: [u8; CONTROL_MSG_SIZE],
    pub writeLeft: Mutex<u8>,
}

#[derive(Clone)]
pub struct RDMAControlChannel(Arc<RDMAControlChannelIntern>);

impl Default for RDMAControlChannel {
    fn default() -> Self {
        Self(Arc::new(RDMAControlChannelIntern {
            chan: Weak::new(),
            // readHeadBuf: [0; 8],
            // readHeadLeft: 0,
            // readBodyBuf: Vec::new(),
            // readBodyLeft: 0,
            // writeBuf: Vec::new(),
            // writeLeft: 0,
            readBuf: [0; CONTROL_MSG_SIZE],
            readLeft: 0,
            writeBuf: [0; CONTROL_MSG_SIZE],
            writeLeft: Mutex::new(0),
        }))
    }
}

impl Deref for RDMAControlChannel {
    type Target = Arc<RDMAControlChannelIntern>;

    fn deref(&self) -> &Arc<RDMAControlChannelIntern> {
        &self.0
    }
}

impl RDMAControlChannel {
    pub fn New(rdmaChannelIntern: Arc<RDMAChannelIntern>) -> Self {
        Self(Arc::new(RDMAControlChannelIntern {
            chan: Arc::downgrade(&rdmaChannelIntern),
            // readHeadBuf: [0; 8],
            // readHeadLeft: 0,
            // readBodyBuf: Vec::new(),
            // readBodyLeft: 0,
            // writeBuf: Vec::new(),
            // writeLeft: 0,
            readBuf: [0; CONTROL_MSG_SIZE],
            readLeft: 0,
            writeBuf: [0; CONTROL_MSG_SIZE],
            writeLeft: Mutex::new(0),
        }))
    }

    pub fn ProcessRDMAWriteImmFinish(&self) {
        self.chan.upgrade().unwrap().ProcessRDMAWriteImmFinish();
    }

    pub fn ProcessRDMARecvWriteImm(&self, qpNum: u32, _recvCount: u64) -> Result<()> {
        println!("ProcessRDMARecvWriteImm");
        let rdmaChannel = self.chan.upgrade().unwrap();
        let _res = rdmaChannel.conn.PostRecv(
            qpNum,
            rdmaChannel.localId as u64,
            rdmaChannel.raddr,
            rdmaChannel.rkey,
        );

        //TODO: handle postRecv error
        let mut readBuf = rdmaChannel.sockBuf.readBuf.lock();
        let msgSize = mem::size_of::<ControlMsgBody>();
        while readBuf.AvailableDataSize() > msgSize {
            let r: Vec<u8> = Vec::with_capacity(msgSize);
            let rAddr = r.as_ptr() as u64;
            let (_trigger, len) = readBuf.readViaAddr(rAddr, msgSize as u64);
            if len != msgSize {
                panic!("readViaAddr can't read enough content!");
            }

            let msg = unsafe { &mut *(rAddr as *mut ControlMsgBody) };
            match msg {
                ControlMsgBody::ConnectRequest(msg) => {
                    self.HandleConnectionRequest(msg);
                }
                ControlMsgBody::ConnectResponse(msg) => {
                    println!("3 remoteChannelId: {}", msg.remoteChannelId);
                }
                ControlMsgBody::ConnectConfirm(msg) => {
                    println!("1 localChannelId: {}", msg.localChannelId);
                }
                ControlMsgBody::ConsumedData(msg) => {
                    println!("3 remoteChannelId: {}", msg.remoteChannelId);
                }
            }
        }

        // let dataBuf = self
        //     .chan
        //     .upgrade()
        //     .unwrap()
        //     .sockBuf
        //     .writeBuf
        //     .lock()
        //     .GetDataBuf();
        // println!("dataBuf, addr: 0x{:x}, len: {}", dataBuf.0, dataBuf.1);
        // let x = dataBuf.0;
        // let z = unsafe { &mut *(x as *mut u8) };
        // println!("z: {}", z);
        // let z = unsafe { &mut *((x + 1) as *mut u8) };
        // println!("z: {}", z);
        // let z = unsafe { &mut *((x + 2) as *mut u8) };
        // println!("z: {}", z);
        // let controlMsg = unsafe { &mut *(x as *mut ControlMsgBody) };
        // match controlMsg {
        //     ControlMsgBody::ConnectConfirm(msg) => {
        //         println!("1 localChannelId: {}", msg.localChannelId);
        //     }
        //     ControlMsgBody::ConnectRequest(msg) => {
        //         println!("2 port: {}", msg.port);
        //     }
        //     ControlMsgBody::ConnectResponse(msg) => {
        //         println!("3 remoteChannelId: {}", msg.remoteChannelId);
        //     }
        // }

        return Ok(());
    }

    pub fn HandleConnectionRequest(&self, connectionRequest: &mut ConnectRequest) {
        let endPoint = Endpoint {
            ipAddr: connectionRequest.dstIpAddr,
            port: connectionRequest.dstPort,
        };
        let mut found = false;
        let mut agentId = 0;
        match RDMA_SRV.srvEndPoints.get(&endPoint) {
            Some(srvEndpoint) => match srvEndpoint.status {
                SrvEndPointStatus::Listening => {
                    found = true;
                    agentId = srvEndpoint.agentId;
                }
                _ => {}
            },
            None => {}
        }

        if found {
            let channelId = RDMA_SRV.channelIdMgr.lock().AllocId();
            //let rdmaChannel = RDMAChannel::New(localId, remoteId, lkey, rkey, socketBuf, rdmaConn);
        }
    }

    //Send msg using control channel
    //Where to be called:
    //1. Connect request: Trigger by SQ.
    //2. Connect Response: Trigger by RecvWriteImm
    //3. Connect Confirm: Trigger by RecvWriteImm
    //4. ConsumedData: 1) WriteImmFinish, 2) Read more than half of read buffer.
    //5. RemoteRecevieReqeustsNum ?? when to send?
    pub fn SendMsg(&self, msg: ControlMsgBody) -> Result<()> {
        let rdmaChannel = self.chan.upgrade().unwrap();
        let mut writeBuf = rdmaChannel.sockBuf.writeBuf.lock();
        if mem::size_of::<ControlMsgBody>() < writeBuf.AvailableSpace() {
            return Err(Error::Timeout);
        }
        let (trigger, _len) = writeBuf.writeViaAddr(
            &msg as *const _ as u64,
            mem::size_of::<ControlMsgBody>() as u64,
        );
        if trigger {
            //TODO: check this is good to call rdmachannel here.
            rdmaChannel.RDMASend();
        }
        // let totalLen = mem::size_of::<ControlMsgBody>();
        // if writtenLen < totalLen {
        //     let writeLeft = totalLen - writtenLen;
        //     *self.writeLeft.lock() = writeLeft as u8;
        //     let ptr = &self.writeBuf as *const _ as u64 as *mut u8;
        //     let slice = unsafe { slice::from_raw_parts_mut(ptr, CONTROL_MSG_SIZE) };
        //     let msgAddr = &msg as *const _ as u64;
        //     let msgSlice = unsafe { slice::from_raw_parts(msgAddr as *mut u8, CONTROL_MSG_SIZE) };
        //     slice[0..writeLeft]
        //         .clone_from_slice(&msgSlice[(CONTROL_MSG_SIZE - writeLeft)..CONTROL_MSG_SIZE]);
        // }
        return Ok(());
    }

    pub fn SendData(&self) {
        let dataBuf = self
            .chan
            .upgrade()
            .unwrap()
            .sockBuf
            .writeBuf
            .lock()
            .GetSpaceBuf();
        println!("dataBuf, addr: 0x{:x}, len: {}", dataBuf.0, dataBuf.1);
        let x = dataBuf.0;
        let z = unsafe { &mut *(x as *mut u8) };
        println!("z: {}", z);
        let z = unsafe { &mut *((x + 1) as *mut u8) };
        println!("z: {}", z);
        let z = unsafe { &mut *((x + 2) as *mut u8) };
        println!("z: {}", z);
        // let v: Vec<u8> = vec![101, 102, 103];
        // self.chan
        //     .upgrade()
        //     .unwrap()
        //     .sockBuf
        //     .writeBuf
        //     .lock()
        //     .writeViaAddr(v.as_ptr() as *const _ as u64, 3);
        let connectRequest = ControlMsgBody::ConnectRequest(ConnectRequest {
            remoteChannelId: 123,
            raddr: 456,
            rkey: 789,
            length: 1000,
            srcIpAddr: 0,
            srcPort: 80,
            dstIpAddr: 0,
            dstPort: 8080,
        });
        self.chan
            .upgrade()
            .unwrap()
            .sockBuf
            .writeBuf
            .lock()
            .writeViaAddr(
                &connectRequest as *const _ as u64,
                mem::size_of::<ControlMsgBody>() as u64,
            );
        self.chan.upgrade().unwrap().RDMASend();
        println!("After RDMA send");
        // let ccfd = RDMA.CompleteChannelFd();
        // println!("ccfd is {}", ccfd);
        // let mut raw_fd_set = mem::MaybeUninit::<libc::fd_set>::uninit();
        // unsafe { libc::FD_ZERO(raw_fd_set.as_mut_ptr()) };
        // let mut fd_set = unsafe { raw_fd_set.assume_init() };
        //
        // unsafe {
        //     libc::FD_SET(ccfd, &mut fd_set);
        //     println!("before select 1...");
        //     libc::select(
        //         ccfd + 1,
        //         &mut fd_set,
        //         ptr::null_mut(),
        //         ptr::null_mut(),
        //         ptr::null_mut(),
        //     );
        //     println!("after select 1...");
        //     RDMA.PollCompletionQueueAndProcess();
        // }

        // unsafe {
        //     let epollfd = libc::epoll_create1(0);
        //     let mut event = libc::epoll_event {
        //         events: libc::EPOLLET as u32 | libc::EPOLLIN as u32,
        //         u64: ccfd as u64,
        //     };
        //     epoll_ctl(epollfd, libc::EPOLL_CTL_ADD, ccfd, &mut event);
        //     let mut events: Vec<libc::epoll_event> = Vec::with_capacity(1024);
        //     println!("before epoll_wait...");
        //     let eventNum = epoll_wait(
        //         epollfd,
        //         events.as_mut_ptr() as *mut libc::epoll_event,
        //         1024,
        //         -1 as libc::c_int,
        //     );
        //     events.set_len(eventNum as usize);
        //     println!("after epoll_wait, event num is: {}, event: {:x}, fd: {}", eventNum, events[0].events, events[0].u64);
        // }
    }
}

// pub struct RDMAControlChannel(Weak<RDMAChannelIntern>);

// impl Deref for RDMAControlChannel {
//     type Target = Weak<RDMAChannelIntern>;

//     fn deref(&self) -> &Weak<RDMAChannelIntern> {
//         &self.0
//     }
// }

// impl RDMAControlChannel {
//     pub fn New(rdmaChannelIntern: Arc<RDMAChannelIntern>) -> Self {
//         Self(Arc::downgrade(&rdmaChannelIntern))
//     }
// }

/* Control channel protocol: payload */

#[derive(Clone)]
pub enum ControlMsgBody {
    ConnectRequest(ConnectRequest),
    ConnectResponse(ConnectResponse),
    ConnectConfirm(ConnectConfirm),
    ConsumedData(ConsumedData),
}

#[derive(Clone)]
pub struct ConsumedData {
    remoteChannelId: u32,
    consumedData: u32,
}

#[derive(Clone)]
pub struct ConnectRequest {
    remoteChannelId: u32,
    raddr: u64,
    rkey: u32,
    length: u32,
    dstIpAddr: u32,
    dstPort: u16,
    srcIpAddr: u32,
    srcPort: u16,
}

#[derive(Clone)]
pub struct ConnectResponse {
    remoteChannelId: u32,
    localChannelId: u32,
    raddr: u64,
    rkey: u32,
    length: u32,
}

#[derive(Clone)]
pub struct ConnectConfirm {
    remoteChannelId: u32,
    localChannelId: u32,
}

#[repr(u32)]
pub enum ControlMsgType {
    ConnectionRequest,
    ConnectionResponse,
    ConnectionConfirm,
}

pub struct ControlMsgHead {
    msgType: ControlMsgType,
    length: u32,
}

pub struct ControlMsg {
    head: ControlMsgHead,
    body: ControlMsgBody,
}
