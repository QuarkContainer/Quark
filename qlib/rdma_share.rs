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

pub const COUNT: usize = 128;
pub struct RingQueue <T: 'static + Default> {
    pub data: [T; COUNT],
    pub ringMask: AtomicU32,
    pub head: AtomicU32,
    pub tail: AtomicU32,
}

impl <T: 'static + Default + Copy> RingQueue <T> {
    pub fn Init(&self) {
        self.ringMask.store(COUNT as u32 -1, Ordering::Release);
        self.head.store(0, Ordering::Release);
        self.tail.store(0, Ordering::Release);
    }

    #[inline]
    pub fn RingMask(&self) -> u32 {
        return self.ringMask.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn Count(&self) -> usize {
        return self.ringMask.load(Ordering::Relaxed) as usize + 1
    }

    // pop
    pub fn Pop(&self) -> Option<T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let available = tail.wrapping_sub(head) as usize;
        if available == 0 {
            return None
        }

        let idx = head & self.RingMask();
        let data = self.data[idx as usize];
        self.head.store(head.wrapping_add(1),  Ordering::Release);
        return Some(data);
    }

    pub fn DataCount(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let available = tail.wrapping_sub(head) as usize;
        return available
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
        assert!(available < self.Count());

        let idx = tail & self.RingMask();
        self.data[idx as usize] = data;
        self.tail.store(tail.wrapping_add(1),  Ordering::Release);
    }
}

#[derive(Default)]
pub struct RDMAReq {}

#[derive(Default)]
pub struct RDMAResp {}

pub struct MemRegion {
    pub addr: u64,
    pub len: u64,
}

pub const SOCKET_BUF_SIZE : usize = 64 * 1024; // 64KB

// todo: caculate this to fit ClientShareRegion in 1GB
pub const IO_BUF_COUNT : usize = 16 * 1024 - 128; // ~16K

pub struct IOBuf {
    pub read: [u8; SOCKET_BUF_SIZE],
    pub write: [u8; SOCKET_BUF_SIZE],
}

pub struct IOMetas {
    pub readBufAtoms: [AtomicU32; 2],
    pub writeBufAtoms: [AtomicU32; 2],
    pub consumeReadData: AtomicU32
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
    pub iobufs:  [IOBuf; IO_BUF_COUNT]
}

// total 4096 x 8 = 32KB or 8 pages
// can index about 32K x 8 = 256K containers, hope it is enough
pub const BITMAP_COUNT : usize = 4096 - 4;

pub struct TriggerBitmap {
    // one bit map to one l2 bitmap to expedite the notification search
    pub l1bitmap: [u64; 4],

    // one bit map to one Quark Container
    pub l2bitmap: [u64; BITMAP_COUNT],
}

pub const MTU: usize = 1500;
pub const UDP_BUF_COUNT: usize = 512 * 1024; // 512K udp buff

// udp buf, around 1516 bytes
pub struct UDPBuf {
    pub vpcId: u32,
    pub srcIPAddr: u32,
    pub dstIPAddr: u32,
    pub srcPort: u16,
    pub dstPort: u16,

    pub data: [u8; MTU],
}

pub struct ShareRegion {
    pub srvBitmap: AtomicU64, // whether server is sleeping
    pub bitmap: TriggerBitmap,
    pub udpBufs: [UDPBuf; UDP_BUF_COUNT]
}