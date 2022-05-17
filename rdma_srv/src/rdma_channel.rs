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
use alloc::sync::Weak;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use spin::{Mutex, MutexGuard};
use std::mem;
use std::ops::{Deref, DerefMut};

use super::rdma_agent::*;
use super::rdma_conn::*;
use super::rdma_srv::*;

// RDMA Channel
use super::qlib::bytestream::*;
use super::qlib::common::*;
use super::qlib::linux_def::*;
use super::qlib::rdma_share::*;
use super::qlib::socket_buf::SocketBuff;

#[derive(Clone, Copy, Debug)]
pub enum ChannelStatus {
    // FIN_RECEIVED_FROM_CLIENT, //request from client
    // FIN_RECEIVED_FROM_PEER,
    FIN_SENT_TO_PEER,
    // CLOSE_WAIT,
    CLOSE_REQUESTED_FROM_CLIENT,
    ESTABLISHED,
    // SYN_RECEIVED,
    // LISTENING,
    CONNECTING,
    // BINDED,
    // ... to simulate TCP status
}

#[derive(Clone, Default, Debug)]
#[repr(C)]
pub struct ChannelRDMAInfo {
    pub remoteId: u32,
    pub raddr: u64,     /* Read Buffer address */
    pub rlen: u32,      /* Read Buffer len */
    pub rkey: u32,      /* Read Buffer Remote key */
    pub offset: u32,    //read buffer offset
    pub freespace: u32, //read buffer free space size
    pub sending: bool,  // the writeimmediately is ongoing
}

impl ChannelRDMAInfo {
    pub fn Size() -> usize {
        return mem::size_of::<Self>();
    }
}

pub struct RDMAChannelIntern {
    pub localId: u32,
    // pub remoteId: u32,
    // pub readBuf: Mutex<ByteStream>,
    // pub writeBuf: Mutex<ByteStream>,
    // pub consumeReadData: &'static AtomicU64,
    // pub sockfd: u32, //TODO: this is used to associate SQE and CQE, need double check it's a proper way to do it or not
    pub sockBuf: Arc<SocketBuff>,
    pub lkey: u32,
    pub rkey: u32,
    pub raddr: u64,
    pub length: u32,
    pub writeCount: AtomicUsize, //when run the writeimm, save the write bytes count here
    pub remoteChannelRDMAInfo: Mutex<ChannelRDMAInfo>,

    // ChannelRDMAInfo:
    // let localChannelRDMAInfo = ChannelRDMAInfo {
    //     raddr: addr,
    //     rlen: len as _,
    //     rkey: readMR.RKey(),
    //     offset: 0,
    //     freespace: len as u32,
    //     sending: false,
    // };

    // rdma connect to remote node
    pub conn: RDMAConn,

    // rdma agent connected to rdma client
    pub agent: RDMAAgent,

    pub vpcId: u32,
    pub srcIpAddr: u32,
    pub dstIpAddr: u32,
    pub srcPort: u16,
    pub dstPort: u16,
    pub status: Mutex<ChannelStatus>,
    pub duplexMode: Mutex<DuplexMode>,
    pub ioBufIndex: u32,
}

impl RDMAChannelIntern {
    pub fn test(&self) {
        println!("testtest");
    }

    pub fn UpdateRemoteRDMAInfo(&self, remoteId: u32, raddr: u64, rlen: u32, rkey: u32) {
        *self.remoteChannelRDMAInfo.lock() = ChannelRDMAInfo {
            remoteId,
            raddr,
            rlen,
            rkey,
            offset: 0,
            freespace: rlen,
            sending: false,
        }
    }

    pub fn RDMAWriteImm(
        &self,
        localAddr: u64,
        remoteAddr: u64,
        writeCount: usize,
        rkey: u32,
        remoteId: u32,
        wrId: u64,
    ) -> Result<()> {
        // println!("RDMAChannelIntern::RDMAWriteImm 1, localAddr: {}, remoteAddr: {}, writeCount: {}, localId: {}, remoteId: {}, lkey: {}, rkey: {}", localAddr, remoteAddr, writeCount, self.localId, remoteId, self.lkey, rkey);
        self.conn.RDMAWriteImm(
            localAddr, remoteAddr, writeCount, wrId, remoteId, self.lkey, rkey,
        )?;

        self.writeCount.store(writeCount, QOrdering::RELEASE);
        return Ok(());
    }

    pub fn ProcessRDMAWriteImmFinish(&self, finSent: bool) {
        // println!("RDMAChannel::ProcessRDMAWriteImmFinish 1");
        let mut remoteInfo = self.remoteChannelRDMAInfo.lock();
        remoteInfo.sending = false;

        let writeCount = self.writeCount.load(QOrdering::ACQUIRE);
        // debug!("ProcessRDMAWriteImmFinish::1 writeCount: {}", writeCount);

        let (trigger, addr, _len) = self
            .sockBuf
            .ConsumeAndGetAvailableWriteBuf(writeCount as usize);

        if finSent {
            if matches!(*self.status.lock(), ChannelStatus::CLOSE_REQUESTED_FROM_CLIENT) {
                println!("ProcessRDMAWriteImmFinish::2");
                RDMA_SRV.channels.lock().remove(&self.localId);
                RDMA_SRV.channelIdMgr.lock().Remove(self.localId);
                self.agent.ioBufIdMgr.lock().Remove(self.ioBufIndex);
                println!("ProcessRDMAWriteImmFinish::3, remove channel from RDMA_SRV");

            } else {
                println!("ProcessRDMAWriteImmFinish::4");
                *self.status.lock() = ChannelStatus::FIN_SENT_TO_PEER;
                // TODO: notify client.
                // self.agent.SendResponse(RDMAResp {
                //     user_data: 0,
                //     msg: RDMARespMsg::RDMAFinNotify(RDMAFinNotifyResp {
                //         // sockfd: self.sockfd,
                //         channelId: self.localId,
                //         event: FIN_SENT_TO_PEER,
                //     }),
                // });
            }

            return;
        }
        // println!(
        //         "ProcessRDMAWriteImmFinish::3, sockfd: {}, channelId: {}, len: {}, writeCount: {}, trigger: {}",
        //         self.sockfd, self.localId, _len, writeCount, trigger
        //     );
        if trigger {
            if self.localId != 0 {
                println!("ProcessRDMAWriteImmFinish: before SendResponse");
                self.agent.SendResponse(RDMAResp {
                    user_data: 0,
                    msg: RDMARespMsg::RDMANotify(RDMANotifyResp {
                        // sockfd: self.sockfd,
                        channelId: self.localId,
                        event: EVENT_OUT,
                    }),
                });
            }
        }
        // println!(
        //     "RDMAChannel::ProcessRDMAWriteImmFinish 2, localId: {}",
        //     self.localId
        // );
        if self.localId != 0 {
            self.SendConsumedDataInternal(remoteInfo.remoteId);
        } else {
            // TODO: is it needed to send consumedData for control channel here, not now!
        }
        println!("RDMAChannel::ProcessRDMAWriteImmFinish 3, addr: {}", addr);
        if addr != 0 {
            // self.RDMASendLocked(remoteInfo)
            self.conn.RDMAWrite(self, remoteInfo);
        }
    }

    fn SendConsumedDataInternal(&self, remoteChannelId: u32) {
        let readCount = self.sockBuf.GetAndClearConsumeReadData();
        // println!("SendConsumedData 1, readCount: {}", readCount);
        if readCount > 0 {
            self.conn
                .ctrlChan
                .lock()
                .SendControlMsg(ControlMsgBody::ConsumedData(ConsumedData {
                    remoteChannelId: remoteChannelId,
                    consumedData: readCount as u32,
                    recvRequestCount: self
                        .conn
                        .localInsertedRecvRequestCount
                        .swap(0, Ordering::SeqCst),
                }));
        }
    }
    pub fn SendConsumedData(&self) {
        self.SendConsumedDataInternal(self.GetRemoteChannelId());
    }

    pub fn Shutdown(&self, howto: u8) {
        if howto == 0 {
            *self.duplexMode.lock() = DuplexMode::SHUTDOWN_RD;
        } else if howto == 1 {
            *self.duplexMode.lock() = DuplexMode::SHUTDOWN_WR;
        } else if howto == 2 {
            *self.duplexMode.lock() = DuplexMode::SHUTDOWN_RDWR;
        }
        println!("RDMAChannel::Shutdown");
        self.RDMASend();
    }

    pub fn Close(&self) {
        println!("RDMAChannel::Close 1");
        println!("RDMAChannel::Close, channel status: {:?}", *self.status.lock());
        if matches!(*self.status.lock(), ChannelStatus::FIN_SENT_TO_PEER) {
            println!("RDMAChannel::Close 2");
            RDMA_SRV.channels.lock().remove(&self.localId);
            RDMA_SRV.channelIdMgr.lock().Remove(self.localId);
            self.agent.ioBufIdMgr.lock().Remove(self.ioBufIndex);
            println!("RDMAChannel::Close 3");
        } else {
            println!("RDMAChannel::Close 4");
            *self.status.lock() = ChannelStatus::CLOSE_REQUESTED_FROM_CLIENT;
            println!("RDMAChannel::Close 5");
        }
    }

    fn GetRemoteChannelId(&self) -> u32 {
        self.remoteChannelRDMAInfo.lock().remoteId
    }

    pub fn ProcessRDMARecvWriteImm(&self, qpNum: u32, recvCount: u64, finReceived: bool) {
        let _res = self
            .conn
            .PostRecv(qpNum, self.localId as u64, self.raddr, self.rkey);

        if recvCount > 0 {
            let (trigger, _addr, len) = self.sockBuf.ProduceAndGetFreeReadBuf(recvCount as usize);
            debug!("ProcessRDMARecvWriteImm::2, trigger {}", trigger);
            println!(
                "ProcessRDMARecvWriteImm::3, channelId: {}, len: {}, recvCount: {}, trigger: {}",
                // self.sockfd,
                self.localId,
                len,
                recvCount,
                trigger
            );
            if trigger {
                // TODO: notify 'client' via CQ
                // println!("ProcessRDMARecvWriteImm: before SendResponse");
                self.agent.SendResponse(RDMAResp {
                    user_data: 0,
                    msg: RDMARespMsg::RDMANotify(RDMANotifyResp {
                        // sockfd: self.sockfd,
                        channelId: self.localId,
                        event: EVENT_IN,
                    }),
                });
            } else {
                println!("ProcessRDMARecvWriteImm 4, trigger: {}", trigger);
            }
        }

        if finReceived {
            println!("ProcessRDMARecvWriteImm 5");
            // *self.status.lock() = ChannelStatus::FIN_RECEIVED_FROM_PEER;
            println!("ProcessRDMARecvWriteImm 6");
            self.agent.SendResponse(RDMAResp {
                user_data: 0,
                msg: RDMARespMsg::RDMAFinNotify(RDMAFinNotifyResp {
                    channelId: self.localId,
                    event: FIN_RECEIVED_FROM_PEER,
                }),
            });
            println!("ProcessRDMARecvWriteImm 7");
        }
    }

    pub fn ProcessRemoteConsumedData(&self, consumedCount: u32) {
        // println!("RDMAChannel::ProcessRemoteConsumedData 1");
        let trigger;
        {
            let mut remoteInfo = self.remoteChannelRDMAInfo.lock();
            trigger = remoteInfo.freespace == 0;
            remoteInfo.freespace += consumedCount as u32;
        }

        if trigger {
            self.RDMASend();
        }
    }

    pub fn RDMASend(&self) {
        println!("RDMAChannelIntern::RDMASend 1");
        let remoteInfo = self.remoteChannelRDMAInfo.lock();
        println!("RDMAChannelIntern::RDMASend 2");
        if remoteInfo.sending == true {
            return; // the sending is ongoing
        }
        println!("RDMAChannelIntern::RDMASend 3");
        //self.RDMASendLockedNew(remoteInfo);
        self.conn.RDMAWrite(self, remoteInfo);
    }

    pub fn RDMASendFromConn(&self, remoteRecvRequestCount: &mut MutexGuard<u32>) {
        let remoteInfo = self.remoteChannelRDMAInfo.lock();
        self.RDMASendLocked(remoteInfo, remoteRecvRequestCount);
    }

    fn Is_WR(&self) -> bool {
        match *self.duplexMode.lock() {
            DuplexMode::SHUTDOWN_WR => true,
            DuplexMode::SHUTDOWN_RDWR => true,
            _ => false,
        }
    }

    pub fn RDMASendLocked(
        &self,
        mut remoteInfo: MutexGuard<ChannelRDMAInfo>,
        remoteRecvRequestCount: &mut MutexGuard<u32>,
    ) {
        // println!("RDMASendLocked 1");
        let buf = self.sockBuf.writeBuf.lock();
        // println!("RDMASendLocked 3");
        let (addr, mut totalLen) = buf.GetDataBuf();
        // debug!("RDMASendLocked::1, readCount: {}, addr: {:x}, len: {}, remote.freespace: {}", readCount, addr, len, remoteInfo.freespace);
        println!(
            "RDMASendLocked 4,len: {}, remoteInfo.freespace: {}",
            totalLen, remoteInfo.freespace
        );
        if totalLen > 0 {
            let mut immData = remoteInfo.remoteId;
            let mut wrId = self.localId;
            let mut len = totalLen;
            if len > remoteInfo.freespace as usize {
                len = remoteInfo.freespace as usize;
            } else {
                if self.Is_WR() {
                    immData = immData | 0x80000000;
                    wrId = wrId | 0x80000000;
                }
            }

            println!("***********len = {}", totalLen);

            if len != 0 {
                self.RDMAWriteImm(
                    addr,
                    remoteInfo.raddr + remoteInfo.offset as u64,
                    len,
                    remoteInfo.rkey,
                    immData,
                    wrId as u64,
                )
                .expect("RDMAWriteImm fail...");
                // println!(
                //     "after calling self.RDMAWriteImm. raddr: {}, rkey: {}, len: {}",
                //     remoteInfo.raddr + remoteInfo.offset as u64,
                //     remoteInfo.rkey,
                //     len
                // );
                remoteInfo.freespace -= len as u32;
                remoteInfo.offset = (remoteInfo.offset + len as u32) % remoteInfo.rlen;
                remoteInfo.sending = true;
                // println!("RDMASendLocked::5, remoteInfo: {:?}", remoteInfo);
                //error!("RDMASendLocked::2, writeCount: {}, readCount: {}", len, readCount);
            } else {
                **remoteRecvRequestCount += 1;
            }
        } else {
            if self.Is_WR() {
                let immData = remoteInfo.remoteId | 0x80000000;
                let wrId = self.localId | 0x80000000;
                println!("RDMASendLocked::6, immData: {}, wrId: {}", immData, wrId);
                self.RDMAWriteImm(
                    addr,
                    remoteInfo.raddr + remoteInfo.offset as u64,
                    0,
                    remoteInfo.rkey,
                    immData,
                    wrId as u64,
                )
                .expect("RDMAWriteImm fail...");
            } else {
                **remoteRecvRequestCount += 1;
            }
        }
    }
}

#[derive(Clone)]
pub struct RDMAChannel(Arc<RDMAChannelIntern>);

impl Deref for RDMAChannel {
    type Target = Arc<RDMAChannelIntern>;

    fn deref(&self) -> &Arc<RDMAChannelIntern> {
        &self.0
    }
}

impl DerefMut for RDMAChannel {
    // type Target = Arc<RDMAChannelIntern>;
    fn deref_mut(&mut self) -> &mut Arc<RDMAChannelIntern> {
        &mut self.0
    }
}

impl RDMAChannel {
    // pub fn New(localId: u32, remoteId: u32, readBufHeadTailAddr: u64, writeBufHeadTailAddr: u64, consumeReadDataAddr: u64, readBufAddr: u64, writeBufAddr: u64) -> Self {
    //     let socketBuf = Arc::new(Mutex::new(SocketBuff::InitWithShareMemory(
    //         MemoryDef::DEFAULT_BUF_PAGE_COUNT,
    //         readBufHeadTailAddr,
    //         writeBufHeadTailAddr,
    //         consumeReadDataAddr,
    //         readBufAddr,
    //         writeBufAddr)));

    //     return Self(Arc::new(RDMAChannelIntern{
    //         localId : localId,
    //         remoteId : remoteId,
    //         sockBuf:socketBuf,
    //         conn: None,
    //         agent: None,
    //         vpcId: 0,
    //         srcIPAddr: 0,
    //         dstIPAddr:0,
    //         srcPort:0,
    //         dstPort:0,
    //         status: SockStatus::ESTABLISHED,
    //         duplexMode: DuplexMode::SHUTDOWN_WR,
    //     }))
    // }

    pub fn New(
        localId: u32,
        lkey: u32,
        rkey: u32,
        socketBuf: Arc<SocketBuff>,
        rdmaConn: RDMAConn,
    ) -> Self {
        let (raddr, len) = socketBuf.ReadBuf();
        Self(Arc::new(RDMAChannelIntern {
            localId: localId,
            // sockfd: 0,
            sockBuf: socketBuf,
            conn: rdmaConn,
            agent: RDMAAgent::NewDummyAgent(),
            vpcId: 0,
            srcIpAddr: 0,
            dstIpAddr: 0,
            srcPort: 0,
            dstPort: 0,
            status: Mutex::new(ChannelStatus::ESTABLISHED),
            duplexMode: Mutex::new(DuplexMode::SHUTDOWN_NONE),
            lkey,
            rkey,
            raddr: raddr,
            length: len as u32,
            remoteChannelRDMAInfo: Mutex::new(ChannelRDMAInfo::default()),
            writeCount: AtomicUsize::new(0),
            ioBufIndex: 0,
        }))
    }

    pub fn CreateRDMAChannel(
        localId: u32,
        lkey: u32,
        rkey: u32,
        socketBuf: Arc<SocketBuff>,
        rdmaConn: RDMAConn,
        connectRequest: &ConnectRequest,
        ioBufIndex: u32,
        rdmaAgent: &RDMAAgent,
    ) -> Self {
        let (raddr, len) = socketBuf.ReadBuf();
        Self(Arc::new(RDMAChannelIntern {
            localId: localId,
            // sockfd: 0,
            sockBuf: socketBuf,
            conn: rdmaConn,
            agent: rdmaAgent.clone(),
            vpcId: 0,
            srcIpAddr: connectRequest.dstIpAddr,
            dstIpAddr: connectRequest.srcIpAddr,
            srcPort: connectRequest.dstPort,
            dstPort: connectRequest.srcPort,
            status: Mutex::new(ChannelStatus::ESTABLISHED),
            duplexMode: Mutex::new(DuplexMode::SHUTDOWN_NONE),
            lkey,
            rkey,
            raddr: raddr,
            length: len as u32,
            remoteChannelRDMAInfo: Mutex::new(ChannelRDMAInfo {
                remoteId: connectRequest.remoteChannelId,
                raddr: connectRequest.raddr,
                rlen: connectRequest.rlen,
                rkey: connectRequest.rkey,
                offset: 0,
                freespace: connectRequest.rlen,
                sending: false,
            }),
            writeCount: AtomicUsize::new(0),
            ioBufIndex,
        }))
    }

    pub fn CreateClientChannel(
        localId: u32,
        // sockfd: u32,
        lkey: u32,
        rkey: u32,
        socketBuf: Arc<SocketBuff>,
        rdmaConn: RDMAConn,
        connectRequest: &RDMAConnectReq,
        ioBufIndex: u32,
        rdmaAgent: &RDMAAgent,
    ) -> Self {
        let (raddr, len) = socketBuf.ReadBuf();
        Self(Arc::new(RDMAChannelIntern {
            localId: localId,
            // sockfd: sockfd,
            sockBuf: socketBuf,
            conn: rdmaConn,
            agent: rdmaAgent.clone(),
            vpcId: 0,
            srcIpAddr: connectRequest.srcIpAddr,
            dstIpAddr: connectRequest.dstIpAddr,
            srcPort: connectRequest.srcPort,
            dstPort: connectRequest.dstPort,
            status: Mutex::new(ChannelStatus::CONNECTING),
            duplexMode: Mutex::new(DuplexMode::SHUTDOWN_NONE),
            lkey,
            rkey,
            raddr: raddr,
            length: len as u32,
            remoteChannelRDMAInfo: Mutex::new(ChannelRDMAInfo::default()),
            writeCount: AtomicUsize::new(0),
            ioBufIndex,
        }))
    }

    pub fn CreateConnectRequest(&self, sockfd: u32) -> ConnectRequest {
        ConnectRequest {
            remoteChannelId: self.localId,
            raddr: self.raddr,
            rkey: self.rkey,
            rlen: self.length,
            dstIpAddr: self.dstIpAddr,
            dstPort: self.dstPort,
            srcIpAddr: self.srcIpAddr,
            srcPort: self.srcPort,
            recvRequestCount: self
                .conn
                .localInsertedRecvRequestCount
                .swap(0, Ordering::SeqCst),
            sockFd: sockfd,
        }
    }

    // pub fn NewControlChannel(rdmaChannelIntern: Weak<RDMAChannelIntern>) -> Self {
    //     Self(rdmaChannelIntern.upgrade().unwrap())
    // }

    // pub fn NewControlChannel(
    //     localId: u32,
    //     remoteId: u32,
    //     lkey: u32,
    //     rkey: u32,
    //     socketBuf: Arc<SocketBuff>,
    // ) -> Self {
    //     Self(Arc::new(RDMAChannelIntern {
    //         localId: localId,
    //         remoteId: remoteId,
    //         sockBuf: socketBuf,
    //         conn: Weak::new(),
    //         agent: None,
    //         vpcId: 0,
    //         srcIPAddr: 0,
    //         dstIPAddr: 0,
    //         srcPort: 0,
    //         dstPort: 0,
    //         status: SockStatus::ESTABLISHED,
    //         duplexMode: DuplexMode::SHUTDOWN_WR,
    //         lkey,
    //         rkey,
    //         remoteChannelRDMAInfo: Mutex::new(ChannelRDMAInfo::default()),
    //         writeCount: AtomicUsize::new(0),
    //     }))
    // }

    pub fn RemoteKey(&self) -> u32 {
        self.rkey
    }

    // pub fn UpdateRemoteChannelRDMAInfo(&self, raddr: u64, rlen: u32, rkey: u32) {
    //     *self.remoteChannelRDMAInfo.lock() = ChannelRDMAInfo {
    //         raddr,
    //         rlen,
    //         rkey,
    //         offset: 0,
    //         freespace: rlen,
    //         sending: false,
    //     }
    // }

    // pub fn Test(&self) {
    //     self.0.test();
    // }

    // pub fn RDMAWriteImm(
    //     &self,
    //     localAddr: u64,
    //     remoteAddr: u64,
    //     writeCount: usize,
    //     rkey: u32,
    // ) -> Result<()> {
    //     self.conn.RDMAWriteImm(
    //         localAddr,
    //         remoteAddr,
    //         writeCount,
    //         self.localId as u64,
    //         self.remoteId,
    //         self.lkey,
    //         rkey,
    //     )?;

    //     self.writeCount.store(writeCount, QOrdering::RELEASE);
    //     return Ok(());
    // }

    // pub fn ProcessRDMAWriteImmFinish(&self) {
    //     let mut remoteInfo = self.remoteChannelRDMAInfo.lock();
    //     remoteInfo.sending = false;

    //     let writeCount = self.writeCount.load(QOrdering::ACQUIRE);
    //     // debug!("ProcessRDMAWriteImmFinish::1 writeCount: {}", writeCount);

    //     let (_trigger, addr, _len) = self
    //         .sockBuf
    //         .ConsumeAndGetAvailableWriteBuf(writeCount as usize);
    //     // debug!("ProcessRDMAWriteImmFinish::2, trigger: {}, addr: {}", trigger, addr);
    //     // if trigger {
    //     //     waitinfo.Notify(EVENT_OUT);
    //     // }

    //     if addr != 0 {
    //         self.RDMASendLocked(remoteInfo)
    //     }
    // }

    // pub fn ProcessRDMARecvWriteImm(
    //     &self,
    //     qpNum: u32,
    //     recvCount: u64
    // ) {
    //     let _res = self
    //         .conn
    //         .PostRecv(qpNum, self.localId as u64, self.raddr, self.rkey);

    //     // debug!("ProcessRDMARecvWriteImm::1, recvCount: {}, writeConsumeCount: {}", recvCount, writeConsumeCount);

    //     if recvCount > 0 {
    //         let (trigger, _addr, _len) =
    //             self.sockBuf.ProduceAndGetFreeReadBuf(recvCount as usize);
    //         // debug!("ProcessRDMARecvWriteImm::2, trigger {}", trigger);
    //         // if trigger {
    //         //     waitinfo.Notify(EVENT_IN);
    //         // }
    //     }

    //     // if writeConsumeCount > 0 {
    //     //     let mut remoteInfo = self.remoteChannelRDMAInfo.lock();
    //     //     let trigger = remoteInfo.freespace == 0;
    //     //     remoteInfo.freespace += writeConsumeCount as u32;

    //     //     // debug!("ProcessRDMARecvWriteImm::3, trigger {}, remoteInfo.sending: {}", trigger, remoteInfo.sending);

    //     //     if trigger && !remoteInfo.sending {
    //     //         self.RDMASendLocked(remoteInfo);
    //     //     }
    //     // }
    // }

    // pub fn RDMASendLocked(&self, mut remoteInfo: MutexGuard<ChannelRDMAInfo>) {
    //     let readCount = self.sockBuf.GetAndClearConsumeReadData();
    //     let buf = self.sockBuf.writeBuf.lock();
    //     let (addr, mut len) = buf.GetDataBuf();
    //     // debug!("RDMASendLocked::1, readCount: {}, addr: {:x}, len: {}, remote.freespace: {}", readCount, addr, len, remoteInfo.freespace);
    //     if readCount > 0 || len > 0 {
    //         if len > remoteInfo.freespace as usize {
    //             len = remoteInfo.freespace as usize;
    //         }

    //         if len != 0 || readCount > 0 {
    //             self.RDMAWriteImm(
    //                 addr,
    //                 remoteInfo.raddr + remoteInfo.offset as u64,
    //                 len,
    //                 remoteInfo.rkey,
    //             )
    //             .expect("RDMAWriteImm fail...");
    //             remoteInfo.freespace -= len as u32;
    //             remoteInfo.offset = (remoteInfo.offset + len as u32) % remoteInfo.rlen;
    //             remoteInfo.sending = true;
    //             //error!("RDMASendLocked::2, writeCount: {}, readCount: {}", len, readCount);
    //         }
    //     }
    // }
}

pub struct RDMAChannelWeak(Weak<RDMAChannelIntern>);
