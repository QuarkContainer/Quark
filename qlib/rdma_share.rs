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

use core::sync::atomic::AtomicU32;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
//use std::sync::atomic::AtomicI64;
use super::common::*;
use super::linux_def::*;
use std::collections::HashSet;

pub const COUNT: usize = 65536;
pub struct RingQueue<T: 'static + Default> {
    // pub struct RingQueue<T: Default> {
    // pub struct RingQueue<T: Copy> {
    pub data: [T; COUNT],
    pub ringMask: AtomicU32,
    pub head: AtomicU32,
    pub tail: AtomicU32,
}

impl<T: 'static + Default + Copy> RingQueue<T> {
    // impl<T: Default + Copy> RingQueue<T> {
    //impl<T: Copy> RingQueue<T> {
    pub fn Init(&self) {
        self.ringMask.store(COUNT as u32 - 1, Ordering::Release);
        self.head.store(0, Ordering::Release);
        self.tail.store(0, Ordering::Release);
    }

    #[inline]
    pub fn RingMask(&self) -> u32 {
        return self.ringMask.load(Ordering::Relaxed);
    }

    #[inline]
    pub fn Count(&self) -> usize {
        return self.ringMask.load(Ordering::Relaxed) as usize + 1;
    }

    // pop
    pub fn Pop(&self) -> Option<T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let available = tail.wrapping_sub(head) as usize;
        // println!("RingQueue::Pop, available: {}", available);
        if available == 0 {
            return None;
        }

        let idx = head & self.RingMask();
        let data = self.data[idx as usize];
        self.head.store(head.wrapping_add(1), Ordering::Release);
        return Some(data);
    }

    pub fn DataCount(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let available = tail.wrapping_sub(head) as usize;
        return available;
    }

    //push
    pub fn SpaceCount(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let available = tail.wrapping_sub(head) as usize;
        return self.Count() - available;
    }

    // precondition: there must be at least one free space
    pub fn Push(&mut self, data: T) {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        let available = tail.wrapping_sub(head) as usize;
        // println!("RingQueue::Push, available: {}, count: {}", available, self.Count());
        assert!(available < self.Count());

        let idx = tail & self.RingMask();
        self.data[idx as usize] = data;
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RDMAReq {
    pub user_data: u64,
    pub msg: RDMAReqMsg,
}

// impl Default for RDMAReq {
//     fn default() -> Self {
//         RDMAReq {
//             user_data: 0,
//             msg: unsafe {
//                 let addr = 0 as *mut RDMAReqMsg;
//                 &mut (*addr)
//             },
//         }
//     }
// }

#[derive(Default, Clone, Copy, Debug)]
pub struct RDMAResp {
    pub user_data: u64,
    pub msg: RDMARespMsg,
}

#[derive(Clone, Copy, Debug)]
pub enum RDMAReqMsg {
    RDMAListen(RDMAListenReq),
    RDMAConnect(RDMAConnectReq),
    RDMAWrite(RDMAWriteReq),
    RDMARead(RDMAReadReq),
    // RDMAAccept(RDMAAcceptReq), //Put connected socket on client side.
}

impl Default for RDMAReqMsg {
    fn default() -> Self {
        RDMAReqMsg::RDMAListen(RDMAListenReq::default())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum RDMARespMsg {
    RDMAConnect(RDMAConnectResp),
    RDMAAccept(RDMAAcceptResp),
    RDMANotify(RDMANotifyResp),
}

impl Default for RDMARespMsg {
    fn default() -> Self {
        RDMARespMsg::RDMAConnect(RDMAConnectResp::default())
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct RDMAConnectResp {
    pub sockfd: u32,
    pub ioBufIndex: u32,
    pub channelId: u32,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct RDMAAcceptResp {
    pub sockfd: u32,
    pub ioBufIndex: u32,
    pub channelId: u32,
    pub dstIpAddr: u32,
    pub dstPort: u16,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct RDMANotifyResp {
    pub sockfd: u32,
    pub channelId: u32,
    pub event: EventMask,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct RDMAListenReq {
    //pub vpcId: u32,
    pub sockfd: u32,
    pub ipAddr: u32,
    pub port: u16,
    pub waitingLen: i32,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct RDMAWriteReq {
    pub sockfd: u32,
    pub channelId: u32,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct RDMAReadReq {
    pub sockfd: u32,
    pub channelId: u32,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct RDMAConnectReq {
    //pub vpcId: u32,
    pub sockfd: u32,
    pub dstIpAddr: u32,
    pub dstPort: u16,
    pub srcIpAddr: u32,
    pub srcPort: u16,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct RDMAAcceptReq {
    //pub vpcId: u32,
    pub sockfd: u32,
}

#[derive(Clone, Debug)]
pub struct RDMAListenResp {
    pub ipAddr: u32,
    pub port: u16,
    pub waitingLen: i32,
}

pub struct MemRegion {
    pub addr: u64,
    pub len: u64,
}

pub const SOCKET_BUF_SIZE: usize = 64 * 1024; // 64KB

// todo: caculate this to fit ClientShareRegion in 1GB
pub const IO_BUF_COUNT: usize = 7 * 1024; //16 * 1024 - 128; // ~16K

#[repr(align(4096))]
pub struct IOBuf {
    pub read: [u8; SOCKET_BUF_SIZE],
    pub write: [u8; SOCKET_BUF_SIZE],
}

pub struct IOMetas {
    pub readBufAtoms: [AtomicU32; 2],
    pub writeBufAtoms: [AtomicU32; 2],
    pub consumeReadData: AtomicU64,
}

pub struct ClientShareRegion {
    pub clientBitmap: AtomicU64, //client sleep bit

    // the complete queue
    pub cq: RingQueue<RDMAResp>,

    // the submit queue
    pub sq: RingQueue<RDMAReq>,

    // metadata region for the sockbuf
    pub ioMetas: [IOMetas; IO_BUF_COUNT],

    // data buf for sockbuf, it will be mapped in the rdma MR
    pub iobufs: [IOBuf; IO_BUF_COUNT],
}

// total 4096 x 8 = 32KB or 8 pages
// can index about 32K x 8 = 256K containers, hope it is enough
pub const BITMAP_COUNT: usize = 64 * 8 - 9; //4096 - 4;

// pub struct TriggerBitmap {
//     // one bit map to one l2 bitmap to expedite the notification search
//     pub l1bitmap: [u64; 8],

//     // one bit map to one Quark Container
//     pub l2bitmap: [u64; BITMAP_COUNT],
// }

pub struct TriggerBitmap {
    // one bit map to one l2 bitmap to expedite the notification search
    pub l1bitmap: [AtomicU64; 8],

    // one bit map to one Quark Container
    pub l2bitmap: [AtomicU64; BITMAP_COUNT],
}

pub const MTU: usize = 1500;
pub const UDP_BUF_COUNT: usize = 512 * 1024; // 512K udp buff

// udp buf, around 1516 bytes
#[derive(Copy, Clone)]
pub struct UDPBuf {
    pub vpcId: u32,
    pub srcIPAddr: u32,
    pub dstIPAddr: u32,
    pub srcPort: u16,
    pub dstPort: u16,

    pub data: [u8; MTU],
}

impl Default for UDPBuf {
    fn default() -> UDPBuf {
        UDPBuf {
            vpcId: 0,
            srcIPAddr: 0,
            dstIPAddr: 0,
            srcPort: 0,
            dstPort: 0,
            data: [0; MTU],
        }
    }
}

#[repr(align(4096))]
pub struct ShareRegion {
    pub srvBitmap: AtomicU64, // whether server is sleeping
    pub bitmap: TriggerBitmap,
    //pub udpBufs: [UDPBuf; UDP_BUF_COUNT]
}

// impl Default for ShareRegion {
//     fn default() -> ShareRegion {
//         ShareRegion {
//             srvBitmap: AtomicU64::new(0),
//             bitmap: TriggerBitmap {
//                 l1bitmap: [AtomicU64::new(0); 8],
//                 l2bitmap: [AtomicU64::new(0); BITMAP_COUNT],
//             },
//             //udpBufs: [UDPBuf::default(); UDP_BUF_COUNT]
//         }
//     }
// }

impl ShareRegion {
    pub fn updateBitmap(&mut self, agentId: u32) {
        let l2idx = agentId as usize / 64;
        let l2pos = agentId as usize % 64;
        let l1idx = l2idx / 64;
        let l1pos = l2idx % 64;

        self.bitmap.l1bitmap[l1idx].fetch_or(1 << l1pos, Ordering::SeqCst);
        self.bitmap.l2bitmap[l2idx].fetch_or(1 << l2pos, Ordering::SeqCst);
    }

    pub fn getAgentIds(&self) -> Vec<u32> {
        let mut agentIds: Vec<u32> = Vec::with_capacity(32192);
        for l1idx in 0..8 {
            // println!("l1idx: {}", l1idx);
            let mut l1 = self.bitmap.l1bitmap[l1idx].swap(0, Ordering::SeqCst);
            // println!("l1: {:x}", l1);
            for l1pos in 0..64 {
                if l1 == 0 {
                    // println!("break for l1idx: {}", l1idx);
                    break;
                }
                if l1 % 2 == 1 {
                    let l2idx = l1idx * 64 + l1pos;
                    // println!("l2idx: {}", l2idx);
                    if l2idx > 502 {
                        break;
                    }
                    let mut l2 = self.bitmap.l2bitmap[l2idx as usize].swap(0, Ordering::SeqCst);
                    // println!("l2: {:x}", l2);
                    for l2pos in 0..64 {
                        if l2 == 0 {
                            // println!("before break, l2pos: {}", l2pos);
                            break;
                        }
                        if l2 % 2 == 1 {
                            agentIds.push((l2idx * 64 + l2pos) as u32);
                        }
                        l2 >>= 1;
                    }
                }
                l1 >>= 1
            }
        }

        return agentIds;
    }
}

#[derive(Clone, Copy)]
pub enum SockStatus {
    CLOSE_WAIT,
    CLOSING,
    ESTABLISHED,
    SYN_RECEIVED,
    LISTENING,
    CONNECTING,
    BINDED,
    // ... to simulate TCP status
}

#[derive(Clone, Copy)]
pub enum DuplexMode {
    SHUTDOWN_RD,
    SHUTDOWN_WR,
    SHUTDOWN_RDWR,
}

#[derive(Eq, Hash, PartialEq)]
pub struct Endpoint {
    // same as vpcId
    pub ipAddr: u32,
    pub port: u16,
}

pub struct IdMgr {
    pub set: HashSet<u32>,
    pub len: u32,
    pub start: u32,
}

impl IdMgr {
    pub fn Init(start: u32, len: u32) -> Self {
        return IdMgr {
            set: HashSet::new(),
            len: len,
            start: start,
        };
    }

    pub fn AllocId(&mut self) -> Result<u32> {
        if self.set.len() == self.len as usize {
            return Err(Error::NoEnoughSpace);
        }
        for i in self.start..(self.len + self.start) {
            if !self.set.contains(&i) {
                self.set.insert(i);
                return Ok(i);
            }
        }
        return Err(Error::NoData);
    }

    pub fn Remove(&mut self, i: u32) {
        self.set.remove(&i);
    }

    pub fn AddCapacity(&mut self, i: u32) {
        self.len += i;
    }
}
