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
use core::sync::atomic::Ordering;
use core::sync::atomic::{AtomicU32, AtomicU64};
use libc::*;
use spin::{Mutex, MutexGuard};
use std::collections::HashSet;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use std::{env, ptr, thread};

use super::qlib::common::*;
use super::rdma::*;
use super::rdma_agent::*;
use super::rdma_channel::*;
use super::rdma_ctrlconn::*;

use super::qlib::linux_def::*;
use super::qlib::rdma_share::*;
use super::qlib::socket_buf::*;
use super::rdma_srv::*;

// RDMA Queue Pair
pub struct RDMAQueuePair {}

pub const RECV_REQUEST_COUNT: u32 = 64;

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
    gid: Gid,    /* gid */
    recvRequestCount: u32,
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
    pub remoteRecvRequestCount: Mutex<u32>,
    pub localInsertedRecvRequestCount: AtomicU32,
    pub requestsQueue: Mutex<VecDeque<u32>>, //currently using channel id
    pub controlRequestsQueue: Mutex<VecDeque<u32>>, //currently using channel id
}

#[derive(Clone)]
pub struct RDMAConn(Arc<RDMAConnInternal>);

impl Deref for RDMAConn {
    type Target = RDMAConnInternal;

    fn deref(&self) -> &RDMAConnInternal {
        &self.0
    }
}

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
            recvRequestCount: RECV_REQUEST_COUNT,
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
            remoteRecvRequestCount: Mutex::new(0),
            localInsertedRecvRequestCount: AtomicU32::new(0),
            requestsQueue: Mutex::new(VecDeque::default()),
            controlRequestsQueue: Mutex::new(VecDeque::default()),
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
        for _i in 0..RECV_REQUEST_COUNT {
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
                        // println!("Received remote RDMA Info");
                    }
                    _ => return,
                }
                // println!("SetupRDMA");
                self.SetupRDMA();
                // println!("SendAck");
                self.SendAck().unwrap(); // assume the socket is ready for send
                                         // println!("Set state WaitingForRemoteReady");
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
                // println!("Read::Ready");
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
                // println!("local RDMAInfo sent");
                self.SetSocketState(SocketState::WaitingForRemoteMeta);
            }
            SocketState::WaitingForRemoteMeta => {
                // println!("Write::1");
                //TODO: server side received 4(W) first and 5 (R|W) afterwards. Need more investigation to see why it's different.
            }
            SocketState::WaitingForRemoteReady => {
                // println!("Write::2");
                //TODO: server side received 4(W) first and 5 (R|W) afterwards. Need more investigation to see why it's different.
            }
            SocketState::Ready => {
                // println!("Write::Ready");
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
        // println!("Ready!!!");
        //self.ctrlChan.lock().SendData();
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
            // println!("SendLocalRDMAInfo, err: {}", errno);
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
        println!(
            "RecvRemoteRDMAInfo. raddr: {}, rlen: {}, rkey: {}",
            data.raddr, data.rlen, data.rkey
        );
        self.ctrlChan
            .lock()
            .chan
            .upgrade()
            .unwrap()
            .UpdateRemoteRDMAInfo(0, data.raddr, data.rlen, data.rkey);
        *self.remoteRecvRequestCount.lock() = data.recvRequestCount;
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

    //Original write
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

    pub fn RDMAWrite(
        &self,
        rdmaChannel: &RDMAChannelIntern,
        remoteInfo: MutexGuard<ChannelRDMAInfo>,
    ) {
        let mut remoteRecvRequestCount = self.remoteRecvRequestCount.lock();
        // println!(
        //     "RDMAConn::RDMAWrite, channelId: {}, *remoteRecvRequestCount: {}",
        //     rdmaChannel.localId, *remoteRecvRequestCount
        // );
        if *remoteRecvRequestCount > 0 {
            *remoteRecvRequestCount -= 1;
            // println!(
            //     "CongestionControl -1,  channelId: {}, *remoteRecvRequestCount: {}",
            //     rdmaChannel.localId, *remoteRecvRequestCount
            // );
            rdmaChannel.RDMASendLocked(remoteInfo, &mut remoteRecvRequestCount);
        } else {
            if rdmaChannel.localId == 0 {
                self.controlRequestsQueue
                    .lock()
                    .push_back(rdmaChannel.localId);
                // println!(
                //     "self.controlRequestsQueue.lock(), len: {}",
                //     self.controlRequestsQueue.lock().len()
                // );
            } else {
                self.requestsQueue.lock().push_back(rdmaChannel.localId);
                // println!(
                //     "self.requestsQueue.lock(), len: {}",
                //     self.requestsQueue.lock().len()
                // );
            }

            // println!(
            //     "RDMAConn::RDMAWrite, inserted channelId: {}",
            //     rdmaChannel.localId
            // );
        }
    }

    pub fn IncreaseRemoteRequestCount(&self, count: u32) {
        // println!("IncreaseRemoteRequestCount, count: {} ", count);
        if count == 0 {
            return;
        } else {
            let mut remoteRecvRequestCount = self.remoteRecvRequestCount.lock();
            *remoteRecvRequestCount += count;
            // println!(
            //     "CongestionControl +{}, *remoteRecvRequestCount: {}",
            //     count, *remoteRecvRequestCount
            // );
            self.ProcessRequestsInQueue(remoteRecvRequestCount);
        }
    }

    fn ProcessRequestsInQueue(&self, mut remoteRecvRequestCount: MutexGuard<u32>) {
        let mut requests = self.controlRequestsQueue.lock();
        while *remoteRecvRequestCount > 0 {
            let channelId = requests.pop_front();
            match channelId {
                Some(id) => {
                    if id != 0 {
                        RDMA_SRV
                            .channels
                            .lock()
                            .get(&id)
                            .unwrap()
                            .RDMASendFromConn(&mut remoteRecvRequestCount);
                    } else {
                        RDMA_SRV
                            .controlChannels2
                            .lock()
                            .get(&self.qps[0].qpNum())
                            .unwrap()
                            .RDMASendFromConn(&mut remoteRecvRequestCount);
                    }
                    *remoteRecvRequestCount -= 1;
                }
                None => {
                    break;
                }
            }
        }
        let mut requests = self.requestsQueue.lock();
        while *remoteRecvRequestCount > 0 {
            let channelId = requests.pop_front();
            match channelId {
                Some(id) => {
                    if id != 0 {
                        RDMA_SRV
                            .channels
                            .lock()
                            .get(&id)
                            .unwrap()
                            .RDMASendFromConn(&mut remoteRecvRequestCount);
                    } else {
                        RDMA_SRV
                            .controlChannels2
                            .lock()
                            .get(&self.qps[0].qpNum())
                            .unwrap()
                            .RDMASendFromConn(&mut remoteRecvRequestCount);
                    }
                    *remoteRecvRequestCount -= 1;
                }
                None => {
                    break;
                }
            }
        }
    }

    pub fn PostRecv(&self, _qpNum: u32, wrId: u64, addr: u64, lkey: u32) -> Result<()> {
        //TODO: get right qp when multiple QP are used between two physical machines.
        self.qps[0].PostRecv(wrId, addr, lkey)?;
        let current = self
            .localInsertedRecvRequestCount
            .fetch_add(1, Ordering::SeqCst)
            + 1;
        // println!("rdma_conn::PostRecv, current: {}", current);
        if current >= RECV_REQUEST_COUNT / 2 {
            self.ctrlChan
                .lock()
                .SendControlMsg(ControlMsgBody::RecvRequestCount(RecvRequestCount {
                    count: self.localInsertedRecvRequestCount.swap(0, Ordering::SeqCst),
                }));
        }
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
    pub curControlMsg: Mutex<ControlMsgBody>,
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
            curControlMsg: Mutex::new(ControlMsgBody::DummyMsg),
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
            curControlMsg: Mutex::new(ControlMsgBody::DummyMsg),
        }))
    }

    pub fn ProcessRDMAWriteImmFinish(&self) {
        self.chan
            .upgrade()
            .unwrap()
            .ProcessRDMAWriteImmFinish(false);
    }

    pub fn ProcessRDMARecvWriteImm(&self, qpNum: u32, recvCount: u64) {
        // println!("RDMAControlChannel::ProcessRDMARecvWriteImm 1");
        let rdmaChannel = self.chan.upgrade().unwrap();
        let _res = rdmaChannel.conn.PostRecv(
            qpNum,
            rdmaChannel.localId as u64,
            rdmaChannel.raddr,
            rdmaChannel.rkey,
        );

        let (_trigger, _addr, _len) = rdmaChannel
            .sockBuf
            .ProduceAndGetFreeReadBuf(recvCount as usize);

        //TODO: handle postRecv error
        let mut readBuf = rdmaChannel.sockBuf.readBuf.lock();
        let msgSize = mem::size_of::<ControlMsgBody>();

        while readBuf.AvailableDataSize() >= msgSize {
            let r: Vec<u8> = Vec::with_capacity(msgSize);
            let rAddr = r.as_ptr() as u64;
            let (_trigger, len) = readBuf.readViaAddr(rAddr, msgSize as u64);
            if len != msgSize {
                panic!("readViaAddr can't read enough content!");
            }

            let consumedData = rdmaChannel.sockBuf.AddConsumeReadData(len as u64);
            // println!("RDMAControlChannel::ProcessRDMARecvWriteImm 3");

            if 2 * consumedData > readBuf.BufSize() as u64 {
                // println!("Control Channel to send consumed data");
                rdmaChannel.SendConsumedData();
            }

            let msg = unsafe { &*(rAddr as *mut ControlMsgBody) };
            // println!("RDMAControlChannel::ProcessRDMARecvWriteImm 4, msg: {}", msg);
            match msg {
                ControlMsgBody::ConnectRequest(msg) => {
                    self.chan
                        .upgrade()
                        .unwrap()
                        .conn
                        .IncreaseRemoteRequestCount(msg.recvRequestCount);
                    self.HandleConnectRequest(msg);
                }
                ControlMsgBody::ConnectResponse(msg) => {
                    self.chan
                        .upgrade()
                        .unwrap()
                        .conn
                        .IncreaseRemoteRequestCount(msg.recvRequestCount);
                    self.HandleConnectResponse(msg);
                }
                ControlMsgBody::ConsumedData(msg) => {
                    // println!("ControlChannel::ConsumedData: {}", msg.consumedData);
                    self.chan
                        .upgrade()
                        .unwrap()
                        .conn
                        .IncreaseRemoteRequestCount(msg.recvRequestCount);
                    self.HandleConsumedData(qpNum, msg);
                }
                ControlMsgBody::RecvRequestCount(msg) => {
                    self.chan
                        .upgrade()
                        .unwrap()
                        .conn
                        .IncreaseRemoteRequestCount(msg.count);
                }
                ControlMsgBody::DummyMsg => {
                    panic!("Control channel received dummy message!");
                }
                ControlMsgBody::ConsumedDataGroup(msg) => {
                    self.chan
                        .upgrade()
                        .unwrap()
                        .conn
                        .IncreaseRemoteRequestCount(msg.recvRequestCount);
                    self.HandleConsumedDataGroup(msg);
                }
            }
        }
    }

    pub fn HandleConsumedDataGroup(&self, consumedDataGroup: &ConsumedDataGroup) {
        let mut i = 0;
        loop {
            if i == 6 || consumedDataGroup.consumeData[i].consumedData == 0 {
                break;
            }

            let consumedData = consumedDataGroup.consumeData[i].consumedData;

            let channelId = consumedDataGroup.consumeData[i].remoteChannelId;

            if channelId == 0 {
                self.chan
                    .upgrade()
                    .unwrap()
                    .ProcessRemoteConsumedData(consumedData);
            } else {
                RDMA_SRV
                    .channels
                    .lock()
                    .get(&channelId)
                    .unwrap()
                    .ProcessRemoteConsumedData(consumedData);
            }

            i += 1;
        }
    }
    pub fn HandleConsumedData(&self, qpNum: u32, consumedData: &ConsumedData) {
        // println!("RDMAControlChannel::HandleConsumedData 1, consumedData: {:?}", consumedData);
        //  control channel
        if consumedData.remoteChannelId == 0 {
            RDMA_SRV
                .controlChannels2
                .lock()
                .get(&qpNum)
                .unwrap()
                .ProcessRemoteConsumedData(consumedData.consumedData);
        } else {
            match RDMA_SRV.channels.lock().get(&consumedData.remoteChannelId) {
                Some(rdmaChannel) => {
                    rdmaChannel.ProcessRemoteConsumedData(consumedData.consumedData);
                }
                None => {
                    println!("Channel id {} is not found!", consumedData.remoteChannelId);
                }
            }
        }
    }

    pub fn HandleConnectResponse(&self, connectResponse: &ConnectResponse) {
        // println!("handle Connect Response: {:?}", connectResponse);
        match RDMA_SRV
            .channels
            .lock()
            .get(&connectResponse.remoteChannelId)
        {
            Some(rdmaChannel) => {
                *rdmaChannel.status.lock() = ChannelStatus::ESTABLISHED;
                rdmaChannel.UpdateRemoteRDMAInfo(
                    connectResponse.localChannelId,
                    connectResponse.raddr,
                    connectResponse.rlen,
                    connectResponse.rkey,
                );
                // println!("HandleConnectResponse: before SendResponse");
                rdmaChannel.agent.SendResponse(RDMAResp {
                    user_data: 0,
                    msg: RDMARespMsg::RDMAConnect(RDMAConnectResp {
                        // sockfd: rdmaChannel.sockfd,
                        sockfd: connectResponse.remoteSockFd,
                        ioBufIndex: rdmaChannel.ioBufIndex,
                        channelId: rdmaChannel.localId,
                        dstIpAddr: rdmaChannel.dstIpAddr,
                        dstPort: rdmaChannel.dstPort,
                        srcIpAddr: rdmaChannel.srcIpAddr,
                        srcPort: rdmaChannel.srcPort,
                    }),
                })
            }
            None => {
                println!(
                    "Channel id {} is not found!",
                    connectResponse.remoteChannelId
                );
            }
        }
    }

    pub fn HandleConnectRequest(&self, connectRequest: &ConnectRequest) {
        let mut found = false;
        let mut agentId = 0;
        let mut sockfd = 0;

        match RDMA_CTLINFO
            .ipToPodIdMappings
            .lock()
            .get(&(connectRequest.dstIpAddr.to_be()))
        {
            Some(podIdStr) => {
                let mut podId: [u8; 64] = [0; 64];
                // if podIdStr.len() != podId.len() {
                //     panic!(
                //         "podId len: {} is not equal to podIdStr len: {}",
                //         podId.len(),
                //         podIdStr.len()
                //     );
                // }
                podIdStr
                    .bytes()
                    .zip(podId.iter_mut())
                    .for_each(|(b, ptr)| *ptr = b);
                let endPoint = EndpointUsingPodId {
                    podId,
                    port: connectRequest.dstPort,
                };
                match RDMA_SRV.srvPodIdEndpoints.lock().get(&endPoint) {
                    Some(srvEndpoint) => match srvEndpoint.status {
                        SrvEndPointStatus::Listening => {
                            found = true;
                            agentId = srvEndpoint.agentId;
                            sockfd = srvEndpoint.sockfd;
                        }
                        _ => {}
                    },
                    None => {
                        error!(
                            "HandleConnectRequest, pod: {} is not listening at port: {}",
                            podIdStr, connectRequest.dstPort
                        );
                    }
                }
            }
            None => {
                error!(
                    "HandleConnectRequest, podId for ip: {} is not found!!",
                    connectRequest.dstIpAddr.to_be()
                );
            }
        }

        if found {
            // println!("HandleConnectRequest 2");
            let agents = RDMA_SRV.agents.lock();
            let agent = agents.get(&agentId).unwrap();
            let rdmaChannel = agent
                .CreateServerRDMAChannel(connectRequest, self.chan.upgrade().unwrap().conn.clone());

            RDMA_SRV
                .channels
                .lock()
                .insert(rdmaChannel.localId, rdmaChannel.clone());

            self.SendControlMsg(ControlMsgBody::ConnectResponse(ConnectResponse {
                remoteChannelId: rdmaChannel.remoteChannelRDMAInfo.lock().remoteId,
                localChannelId: rdmaChannel.localId,
                raddr: rdmaChannel.raddr,
                rkey: rdmaChannel.rkey,
                rlen: rdmaChannel.length,
                recvRequestCount: self
                    .chan
                    .upgrade()
                    .unwrap()
                    .conn
                    .localInsertedRecvRequestCount
                    .swap(0, Ordering::SeqCst),
                remoteSockFd: connectRequest.sockFd,
            }));
            // .unwrap();
            // agent.sockInfos.lock().get_mut(&sockfd).unwrap().acceptQueue.lock().EnqSocket(rdmaChannel.localId);
            agent.SendResponse(RDMAResp {
                user_data: 0,
                msg: RDMARespMsg::RDMAAccept(RDMAAcceptResp {
                    sockfd,
                    ioBufIndex: rdmaChannel.ioBufIndex,
                    channelId: rdmaChannel.localId,
                    dstIpAddr: rdmaChannel.dstIpAddr,
                    dstPort: rdmaChannel.dstPort,
                    srcIpAddr: rdmaChannel.srcIpAddr,
                    srcPort: rdmaChannel.srcPort,
                }),
            });
        } else {
            println!("TODO: no server listening, need ack error back!");
        }
    }

    //Send msg using control channel
    //Where to be called:
    //1. Connect request: Trigger by SQ.
    //2. Connect Response: Trigger by RecvWriteImm
    //3. Connect Confirm: Trigger by RecvWriteImm
    //4. ConsumedData: 1) WriteImmFinish, 2) Read more than half of read buffer.
    //5. RemoteRecevieReqeustsNum ?? when to send?
    pub fn SendControlMsg(&self, msg: ControlMsgBody) {
        //-> Result<()> {
        // println!("RDMAControlChannel::SendControlMsg 0");
        let rdmaChannel = self.chan.upgrade().unwrap();
        // println!("RDMAControlChannel::SendControlMsg 1");
        let mut writeBuf = rdmaChannel.sockBuf.writeBuf.lock();
        // println!(
        //     "mem::size_of::<ControlMsgBody>(): {}",
        //     mem::size_of::<ControlMsgBody>()
        // );
        // println!("writeBuf.AvailableSpace(): {}", writeBuf.AvailableSpace());
        // if mem::size_of::<ControlMsgBody>() > writeBuf.AvailableSpace() {
        //     return Err(Error::Timeout);
        // }
        let (trigger, _len) = writeBuf.writeViaAddr(
            &msg as *const _ as u64,
            mem::size_of::<ControlMsgBody>() as u64,
        );
        // println!("RDMAControlChannel::SendControlMsg 2, trigger: {}", trigger);
        std::mem::drop(writeBuf);
        if trigger {
            rdmaChannel.RDMASend();
        }

        //TODOVIP: how to wait???
        *self.curControlMsg.lock() = msg;
        // return Ok(());
    }

    pub fn SendConsumeDataGroup(&self, channels: &mut HashSet<u32>) {
        let mut i = 0;
        let mut consumeDataGroup = ConsumedDataGroup::default();
        for channelId in channels.iter() {
            consumeDataGroup.consumeData[i].remoteChannelId = *channelId;
            if *channelId != 0 {
                consumeDataGroup.consumeData[i].consumedData = RDMA_SRV
                    .channels
                    .lock()
                    .get_mut(&channelId)
                    .unwrap()
                    .sockBuf
                    .GetAndClearConsumeReadData()
                    as u32;
            } else {
                consumeDataGroup.consumeData[i].consumedData =
                    self.chan
                        .upgrade()
                        .unwrap()
                        .sockBuf
                        .GetAndClearConsumeReadData() as u32;
            }

            i = i + 1;
            if i == 6 {
                consumeDataGroup.recvRequestCount = self
                    .chan
                    .upgrade()
                    .unwrap()
                    .conn
                    .localInsertedRecvRequestCount
                    .swap(0, Ordering::SeqCst);
                i = 0;
                self.SendControlMsg(ControlMsgBody::ConsumedDataGroup(consumeDataGroup));
                consumeDataGroup = ConsumedDataGroup::default();
            }
        }
        if i != 0 {
            consumeDataGroup.recvRequestCount = self
                .chan
                .upgrade()
                .unwrap()
                .conn
                .localInsertedRecvRequestCount
                .swap(0, Ordering::SeqCst);
            self.SendControlMsg(ControlMsgBody::ConsumedDataGroup(consumeDataGroup));
        }
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
        // println!("dataBuf, addr: 0x{:x}, len: {}", dataBuf.0, dataBuf.1);
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
            rlen: 1000,
            srcIpAddr: 0,
            srcPort: 80,
            dstIpAddr: 0,
            dstPort: 8080,
            recvRequestCount: self
                .chan
                .upgrade()
                .unwrap()
                .conn
                .localInsertedRecvRequestCount
                .swap(0, Ordering::SeqCst),
            sockFd: 123,
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
    }
}

/* Control channel protocol: payload */

#[derive(Clone)]
pub enum ControlMsgBody {
    ConnectRequest(ConnectRequest),
    ConnectResponse(ConnectResponse),
    // ConnectConfirm(ConnectConfirm),
    ConsumedDataGroup(ConsumedDataGroup),
    ConsumedData(ConsumedData),
    RecvRequestCount(RecvRequestCount),
    DummyMsg,
}

#[derive(Clone, Debug)]
pub struct ConsumedData {
    pub remoteChannelId: u32,
    pub consumedData: u32,
    pub recvRequestCount: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ConsumedDataGroup {
    pub consumeData: [ConsumeDataItem; 6],
    pub recvRequestCount: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ConsumeDataItem {
    pub remoteChannelId: u32,
    pub consumedData: u32,
}

#[derive(Clone, Debug)]
pub struct RecvRequestCount {
    pub count: u32,
}

#[derive(Clone, Debug)]
pub struct ConnectRequest {
    pub remoteChannelId: u32,
    pub raddr: u64,
    pub rkey: u32,
    pub rlen: u32,
    pub dstIpAddr: u32,
    pub dstPort: u16,
    pub srcIpAddr: u32,
    pub srcPort: u16,
    pub recvRequestCount: u32,
    pub sockFd: u32,
}

#[derive(Clone, Debug)]
pub struct ConnectResponse {
    pub remoteChannelId: u32,
    pub localChannelId: u32,
    pub raddr: u64,
    pub rkey: u32,
    pub rlen: u32,
    pub recvRequestCount: u32,
    pub remoteSockFd: u32,
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
