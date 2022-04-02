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
use core::sync::atomic::AtomicU64;
use core::sync::atomic::AtomicUsize;
use spin::{Mutex, MutexGuard};
use std::mem;
use std::ops::{Deref, DerefMut};

use super::rdma_agent::*;
use super::rdma_conn::*;

// RDMA Channel
use super::qlib::bytestream::*;
use super::qlib::common::*;
use super::qlib::linux_def::*;
use super::qlib::socket_buf::SocketBuff;
use super::rdma::*;

pub enum SockStatus {
    CLOSE_WAIT,
    CLOSING,
    ESTABLISHED,
    SYN_RECEIVED,
    // ... to simulate TCP status
}

pub enum DuplexMode {
    SHUTDOWN_RD,
    SHUTDOWN_WR,
    SHUTDOWN_RDWR,
}

#[derive(Clone, Default)]
#[repr(C)]
pub struct RDMAInfo {
    raddr: u64,     /* Read Buffer address */
    rlen: u32,      /* Read Buffer len */
    rkey: u32,      /* Read Buffer Remote key */
    offset: u32,    //read buffer offset
    freespace: u32, //read buffer free space size
    sending: bool,  // the writeimmediately is ongoing
}

impl RDMAInfo {
    pub fn Size() -> usize {
        return mem::size_of::<Self>();
    }
}

pub struct RDMAChannelIntern {
    pub localId: u32,
    pub remoteId: u32,
    // pub readBuf: Mutex<ByteStream>,
    // pub writeBuf: Mutex<ByteStream>,
    // pub consumeReadData: &'static AtomicU64,
    pub sockBuf: Arc<SocketBuff>,
    pub lkey: u32,
    pub rkey: u32,
    pub raddr: u64,
    pub writeCount: AtomicUsize, //when run the writeimm, save the write bytes count here
    pub remoteRDMAInfo: Mutex<RDMAInfo>,

    // RDMAInfo:
    // let localRDMAInfo = RDMAInfo {
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
    pub agent: Option<RDMAAgent>,

    pub vpcId: u32,
    pub srcIPAddr: u32,
    pub dstIPAddr: u32,
    pub srcPort: u16,
    pub dstPort: u16,
    pub status: SockStatus,
    pub duplexMode: DuplexMode,
    //pub ioBufIndex: u32,
}

impl RDMAChannelIntern {
    pub fn test(&self) {
        println!("testtest");
    }

    pub fn UpdateRemoteRDMAInfo(&self, raddr: u64, rlen: u32, rkey: u32) {
        *self.remoteRDMAInfo.lock() = RDMAInfo {
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
    ) -> Result<()> {
        self.conn.RDMAWriteImm(
            localAddr,
            remoteAddr,
            writeCount,
            self.localId as u64,
            self.remoteId,
            self.lkey,
            rkey,
        )?;

        self.writeCount.store(writeCount, QOrdering::RELEASE);
        return Ok(());
    }

    pub fn ProcessRDMAWriteImmFinish(&self) {
        let mut remoteInfo = self.remoteRDMAInfo.lock();
        remoteInfo.sending = false;

        let writeCount = self.writeCount.load(QOrdering::ACQUIRE);
        // debug!("ProcessRDMAWriteImmFinish::1 writeCount: {}", writeCount);

        let (trigger, addr, _len) = self
            .sockBuf
            .ConsumeAndGetAvailableWriteBuf(writeCount as usize);
        
        if trigger {
            // waitinfo.Notify(EVENT_OUT);
            if self.localId != 0 {
                // TODO: notify 'client' to write via CQ.
            }
        }

        if addr != 0 {
            self.RDMASendLocked(remoteInfo)
        }
    }

    pub fn ProcessRDMARecvWriteImm(
        &self,
        qpNum: u32,
        recvCount: u64
    ) {
        let _res = self
            .conn
            .PostRecv(qpNum, self.localId as u64, self.raddr, self.rkey);

        // debug!("ProcessRDMARecvWriteImm::1, recvCount: {}, writeConsumeCount: {}", recvCount, writeConsumeCount);

        if recvCount > 0 {
            let (trigger, _addr, _len) =
                self.sockBuf.ProduceAndGetFreeReadBuf(recvCount as usize);
            // debug!("ProcessRDMARecvWriteImm::2, trigger {}", trigger);
            if trigger {
                // waitinfo.Notify(EVENT_IN);
                // TODO: notify 'client' via CQ
            }
        }

        // if writeConsumeCount > 0 {
        //     let mut remoteInfo = self.remoteRDMAInfo.lock();
        //     let trigger = remoteInfo.freespace == 0;
        //     remoteInfo.freespace += writeConsumeCount as u32;

        //     // debug!("ProcessRDMARecvWriteImm::3, trigger {}, remoteInfo.sending: {}", trigger, remoteInfo.sending);

        //     if trigger && !remoteInfo.sending {
        //         self.RDMASendLocked(remoteInfo);
        //     }
        // }
    }

    pub fn RDMASend(&self) {
        let remoteInfo = self.remoteRDMAInfo.lock();
        if remoteInfo.sending == true {
            return; // the sending is ongoing
        }

        self.RDMASendLocked(remoteInfo);
    }

    pub fn RDMASendLocked(&self, mut remoteInfo: MutexGuard<RDMAInfo>) {
        println!("RDMASendLocked 1");
        let readCount = self.sockBuf.GetAndClearConsumeReadData();
        let buf = self.sockBuf.writeBuf.lock();
        let (addr, mut len) = buf.GetDataBuf();
        // debug!("RDMASendLocked::1, readCount: {}, addr: {:x}, len: {}, remote.freespace: {}", readCount, addr, len, remoteInfo.freespace);
        println!("RDMASendLocked 2, readCount: {}, len: {}", readCount, len);
        if readCount > 0 || len > 0 {
            if len > remoteInfo.freespace as usize {
                len = remoteInfo.freespace as usize;
            }

            if len != 0 || readCount > 0 {
                self.RDMAWriteImm(
                    addr,
                    remoteInfo.raddr + remoteInfo.offset as u64,
                    len,
                    remoteInfo.rkey,
                )
                .expect("RDMAWriteImm fail...");
                println!("after calling self.RDMAWriteImm");
                remoteInfo.freespace -= len as u32;
                remoteInfo.offset = (remoteInfo.offset + len as u32) % remoteInfo.rlen;
                remoteInfo.sending = true;
                //error!("RDMASendLocked::2, writeCount: {}, readCount: {}", len, readCount);
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
        remoteId: u32,
        lkey: u32,
        rkey: u32,
        socketBuf: Arc<SocketBuff>,
        rdmaConn: RDMAConn,
    ) -> Self {
        let (raddr, _len) = socketBuf.ReadBuf();
        Self(Arc::new(RDMAChannelIntern {
            localId: localId,
            remoteId: remoteId,
            sockBuf: socketBuf,
            conn: rdmaConn,
            agent: None,
            vpcId: 0,
            srcIPAddr: 0,
            dstIPAddr: 0,
            srcPort: 0,
            dstPort: 0,
            status: SockStatus::ESTABLISHED,
            duplexMode: DuplexMode::SHUTDOWN_WR,
            lkey,
            rkey,
            raddr : raddr,
            remoteRDMAInfo: Mutex::new(RDMAInfo::default()),
            writeCount: AtomicUsize::new(0),
        }))
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
    //         remoteRDMAInfo: Mutex::new(RDMAInfo::default()),
    //         writeCount: AtomicUsize::new(0),
    //     }))
    // }

    pub fn RemoteKey(&self) -> u32 {
        self.rkey
    }

    // pub fn UpdateRemoteRDMAInfo(&self, raddr: u64, rlen: u32, rkey: u32) {
    //     *self.remoteRDMAInfo.lock() = RDMAInfo {
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
    //     let mut remoteInfo = self.remoteRDMAInfo.lock();
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
    //     //     let mut remoteInfo = self.remoteRDMAInfo.lock();
    //     //     let trigger = remoteInfo.freespace == 0;
    //     //     remoteInfo.freespace += writeConsumeCount as u32;

    //     //     // debug!("ProcessRDMARecvWriteImm::3, trigger {}, remoteInfo.sending: {}", trigger, remoteInfo.sending);

    //     //     if trigger && !remoteInfo.sending {
    //     //         self.RDMASendLocked(remoteInfo);
    //     //     }
    //     // }
    // }

    // pub fn RDMASendLocked(&self, mut remoteInfo: MutexGuard<RDMAInfo>) {
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


