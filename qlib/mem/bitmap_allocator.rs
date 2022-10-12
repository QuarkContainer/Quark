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

use core::{ops::Deref};
use core::sync::atomic;
use core::mem::size_of;
use core::cmp::max;
use core::alloc::{GlobalAlloc, Layout};

use crate::qlib::mutex::*;


// handle total 64 * 64 = 4K blocks
// for 2MB block, (4K x 512B Block ~ 256 x 16 KB block)
// 
// total 8 + 64 * 8 = 8 + 512 = 520 bytes
pub struct BitmapSet<const COUNT_ORDER: usize> {
    pub l1bitmap: u64,           
    pub total: u16,
    pub free: u16,  
    pub l2bitmap: [u64; 64],
}

impl <const COUNT_ORDER: usize> BitmapSet <COUNT_ORDER> {
    pub fn Init(&mut self) {
        assert!(8<=COUNT_ORDER && COUNT_ORDER<=12);
        let totalCount = 1 << COUNT_ORDER;
        let left = totalCount % 64;
        let l2Count = totalCount / 64;

        for i in 0..l2Count {
            self.l1bitmap |= 1 << i;
            self.l2bitmap[i] = u64::MAX;
        }

        for i in 0..left {
            self.l1bitmap |= 1 << l2Count;
            self.l2bitmap[l2Count] |= 1 << i;
        }

        self.total = totalCount as u16;
        self.free = totalCount as u16;
    }

    pub fn Reserve(&mut self, count: usize) {
        for _i in 0..count {
            self.Alloc();
            self.total -= count as u16;
        }
    }

    // return the free block idx
    pub fn Alloc(&mut self) -> Option<usize> {
        let l1idx = self.l1bitmap.trailing_zeros() as usize;
        if l1idx == 64 {
            return None
        }
        let l2idx = self.l2bitmap[l1idx].trailing_zeros() as usize;
        self.l2bitmap[l1idx] &= !(1<<l2idx);
        if self.l2bitmap[l1idx] == 0 {
            self.l1bitmap &= !(1<<l1idx);
        }

        self.free -= 1;

        return Some(l1idx * 64 + l2idx);
    }

    pub fn Free(&mut self, idx: usize) {
        let l1idx = idx / 64;
        let l2idx = idx % 64;
        
        assert!(self.l2bitmap[l1idx] & (1<<l2idx) == 0);
        if self.l2bitmap[l1idx] == 0 {
            self.l1bitmap |= 1<<l1idx;
        }
        self.l2bitmap[l1idx] |= 1 <<l2idx;

        self.free += 1;
    }
}

#[derive(Default, Clone, Copy)]
pub struct FreeListNode {
    pub next: u32,
    pub addr: u64,
}

#[derive(Default)]
pub struct HeadNodeIntern {
    pub currAddr: u64,
    pub head: u32,
    pub tail: u32,
    pub totalFree: usize,
}

impl HeadNodeIntern {
    // move head node to tail. and head-> next as head
    pub fn Rotate(&mut self, fl: &FreeListNodeAllocator) {
        let mut fll = fl.lock();

        if self.head == self.tail { // only one node or not node
            return
        }

        let head = self.head;
        let tail = self.tail;

        {
            let headNode = fll.GetMut(head);

            headNode.next = tail;
            self.head = headNode.next;
        }
        
        self.tail = head;
        self.currAddr = fll.GetMut(self.head).addr;
    }

    pub fn PushFront(&mut self, addr: u64, freeCount: usize, fl: &FreeListNodeAllocator) {
        let mut fll = fl.lock();
        let idx = fll.Alloc();
        let node = fll.GetMut(idx);
        node.next = self.head;
        node.addr = addr;
        self.currAddr = addr;
        self.head = idx;
        self.totalFree += freeCount;
        if self.tail == 0 {
            self.tail = idx;
        }
    }
}

#[repr(align(128))]
#[derive(Default)]
pub struct HeadNode(QMutex<HeadNodeIntern>);

impl Deref for HeadNode {
    type Target = QMutex<HeadNodeIntern>;

    fn deref(&self) -> &QMutex<HeadNodeIntern> {
        &self.0
    }
}

pub enum BlockType {
    Block8s(&'static mut BitmapSetBlock<3, 12>),
    Block16s(&'static mut BitmapSetBlock<4, 12>),
    Block32s(&'static mut BitmapSetBlock<5, 12>),
    Block64s(&'static mut BitmapSetBlock<6, 12>),
    Block128s(&'static mut BitmapSetBlock<7, 12>),
    Block256s(&'static mut BitmapSetBlock<8, 12>),
    Block512s(&'static mut BitmapSetBlock<9, 12>),
    Block1ks(&'static mut BitmapSetBlock<10, 11>),
    Block2ks(&'static mut BitmapSetBlock<11, 10>),
    Block4ks(&'static mut BitmapSetBlock<12, 9>),
    Block8ks(&'static mut BitmapSetBlock<13, 8>),
    Block16ks(&'static mut BitmapBlock<14>),
    Block32ks(&'static mut BitmapBlock<15>),
    Block64ks(&'static mut BitmapBlock<16>),
    Block128ks(&'static mut BitmapBlock<17>),
    Block256ks(&'static mut BitmapBlock<18>),
    Block512ks(&'static mut BitmapBlock<19>),
    Block1Ms(&'static mut BitmapBlock<20>),
    Block2Ms(&'static mut Block2Ms),
}

impl BlockType {
    pub fn GetMut(order: usize, addr: u64) -> Self { 
        match order {
            3 => return Self::Block8s(Block8s::GetMut(addr)),
            4 => return Self::Block16s(Block16s::GetMut(addr)),
            5 => return Self::Block32s(Block32s::GetMut(addr)),
            6 => return Self::Block64s(Block64s::GetMut(addr)),
            7 => return Self::Block128s(Block128s::GetMut(addr)),
            8 => return Self::Block256s(Block256s::GetMut(addr)),
            9 => return Self::Block512s(Block512s::GetMut(addr)),
            10 => return Self::Block1ks(Block1ks::GetMut(addr)),
            11 => return Self::Block2ks(Block2ks::GetMut(addr)),
            12 => return Self::Block4ks(Block4ks::GetMut(addr)),
            13 => return Self::Block8ks(Block8ks::GetMut(addr)),
            14 => return Self::Block16ks(Block16ks::GetMut(addr)),
            15 => return Self::Block32ks(Block32ks::GetMut(addr)),
            16 => return Self::Block64ks(Block64ks::GetMut(addr)),
            17 => return Self::Block128ks(Block128ks::GetMut(addr)),
            18 => return Self::Block256ks(Block256ks::GetMut(addr)),
            19 => return Self::Block512ks(Block512ks::GetMut(addr)),
            20 => return Self::Block1Ms(Block1Ms::GetMut(addr)),
            21 => return Self::Block2Ms(Block2Ms::GetMut(addr)),
            _ => panic!("BlockType::GetMut"),
        }
    }

    pub fn GetMutDirect(order: usize, addr: u64) -> Self { 
        match order {
            3 => return Self::Block8s(Block8s::GetMutDirect(addr)),
            4 => return Self::Block16s(Block16s::GetMutDirect(addr)),
            5 => return Self::Block32s(Block32s::GetMutDirect(addr)),
            6 => return Self::Block64s(Block64s::GetMutDirect(addr)),
            7 => return Self::Block128s(Block128s::GetMutDirect(addr)),
            8 => return Self::Block256s(Block256s::GetMutDirect(addr)),
            9 => return Self::Block512s(Block512s::GetMutDirect(addr)),
            10 => return Self::Block1ks(Block1ks::GetMutDirect(addr)),
            11 => return Self::Block2ks(Block2ks::GetMutDirect(addr)),
            12 => return Self::Block4ks(Block4ks::GetMutDirect(addr)),
            13 => return Self::Block8ks(Block8ks::GetMutDirect(addr)),
            14 => return Self::Block16ks(Block16ks::GetMutDirect(addr)),
            15 => return Self::Block32ks(Block32ks::GetMutDirect(addr)),
            16 => return Self::Block64ks(Block64ks::GetMutDirect(addr)),
            17 => return Self::Block128ks(Block128ks::GetMutDirect(addr)),
            18 => return Self::Block256ks(Block256ks::GetMutDirect(addr)),
            19 => return Self::Block512ks(Block512ks::GetMutDirect(addr)),
            20 => return Self::Block1Ms(Block1Ms::GetMutDirect(addr)),
            21 => return Self::Block2Ms(Block2Ms::GetMutDirect(addr)),
            _ => panic!("BlockType::GetMutDirect"),
        }
    }

    pub fn Init(&mut self) { 
        match self {
            Self::Block8s(inner) => return inner.Init(),
            Self::Block16s(inner) => return inner.Init(),
            Self::Block32s(inner) => return inner.Init(),
            Self::Block64s(inner) => return inner.Init(),
            Self::Block128s(inner) => return inner.Init(),
            Self::Block256s(inner) => return inner.Init(),
            Self::Block512s(inner) => return inner.Init(),
            Self::Block1ks(inner) => return inner.Init(),
            Self::Block2ks(inner) => return inner.Init(),
            Self::Block4ks(inner) => return inner.Init(),
            Self::Block8ks(inner) => return inner.Init(),
            Self::Block16ks(inner) => return inner.Init(),
            Self::Block32ks(inner) => return inner.Init(),
            Self::Block64ks(inner) => return inner.Init(),
            Self::Block128ks(inner) => return inner.Init(),
            Self::Block256ks(inner) => return inner.Init(),
            Self::Block512ks(inner) => return inner.Init(),
            Self::Block1Ms(inner) => return inner.Init(),
            Self::Block2Ms(inner) => return inner.Init(),
        }
    }

    pub fn Alloc(&mut self) -> Option<u64> {
        match self {
            Self::Block8s(inner) => return (*inner).Alloc(),
            Self::Block16s(inner) => return (*inner).Alloc(),
            Self::Block32s(inner) => return (*inner).Alloc(),
            Self::Block64s(inner) => return (*inner).Alloc(),
            Self::Block128s(inner) => return (*inner).Alloc(),
            Self::Block256s(inner) => return (*inner).Alloc(),
            Self::Block512s(inner) => return (*inner).Alloc(),
            Self::Block1ks(inner) => return (*inner).Alloc(),
            Self::Block2ks(inner) => return (*inner).Alloc(),
            Self::Block4ks(inner) => return (*inner).Alloc(),
            Self::Block8ks(inner) => return (*inner).Alloc(),
            Self::Block16ks(inner) => return (*inner).Alloc(),
            Self::Block32ks(inner) => return (*inner).Alloc(),
            Self::Block64ks(inner) => return (*inner).Alloc(),
            Self::Block128ks(inner) => return (*inner).Alloc(),
            Self::Block256ks(inner) => return (*inner).Alloc(),
            Self::Block512ks(inner) => return (*inner).Alloc(),
            Self::Block1Ms(inner) => return (*inner).Alloc(),
            Self::Block2Ms(inner) => return (*inner).Alloc(),
        }
    }

    pub fn Free(order: usize, addr: u64) -> bool {
        let block = Self::GetMut(order, addr) ;
        match block {
            Self::Block8s(inner) => return (*inner).Free(addr),
            Self::Block16s(inner) => return (*inner).Free(addr),
            Self::Block32s(inner) => return (*inner).Free(addr),
            Self::Block64s(inner) => return (*inner).Free(addr),
            Self::Block128s(inner) => return (*inner).Free(addr),
            Self::Block256s(inner) => return (*inner).Free(addr),
            Self::Block512s(inner) => return (*inner).Free(addr),
            Self::Block1ks(inner) => return (*inner).Free(addr),
            Self::Block2ks(inner) => return (*inner).Free(addr),
            Self::Block4ks(inner) => return (*inner).Free(addr),
            Self::Block8ks(inner) => return (*inner).Free(addr),
            Self::Block16ks(inner) => return (*inner).Free(addr),
            Self::Block32ks(inner) => return (*inner).Free(addr),
            Self::Block64ks(inner) => return (*inner).Free(addr),
            Self::Block128ks(inner) => return (*inner).Free(addr),
            Self::Block256ks(inner) => return (*inner).Free(addr),
            Self::Block512ks(inner) => return (*inner).Free(addr),
            Self::Block1Ms(inner) => return (*inner).Free(addr),
            Self::Block2Ms(inner) => return (*inner).Free(addr),
        }
    } 

    pub fn FreeCount(&self) -> usize {
        match self {
            Self::Block8s(inner) => return (*inner).FreeCount(),
            Self::Block16s(inner) => return (*inner).FreeCount(),
            Self::Block32s(inner) => return (*inner).FreeCount(),
            Self::Block64s(inner) => return (*inner).FreeCount(),
            Self::Block128s(inner) => return (*inner).FreeCount(),
            Self::Block256s(inner) => return (*inner).FreeCount(),
            Self::Block512s(inner) => return (*inner).FreeCount(),
            Self::Block1ks(inner) => return (*inner).FreeCount(),
            Self::Block2ks(inner) => return (*inner).FreeCount(),
            Self::Block4ks(inner) => return (*inner).FreeCount(),
            Self::Block8ks(inner) => return (*inner).FreeCount(),
            Self::Block16ks(inner) => return (*inner).FreeCount(),
            Self::Block32ks(inner) => return (*inner).FreeCount(),
            Self::Block64ks(inner) => return (*inner).FreeCount(),
            Self::Block128ks(inner) => return (*inner).FreeCount(),
            Self::Block256ks(inner) => return (*inner).FreeCount(),
            Self::Block512ks(inner) => return (*inner).FreeCount(),
            Self::Block1Ms(inner) => return (*inner).FreeCount(),
            Self::Block2Ms(inner) => return (*inner).FreeCount(),
        }
    }

    pub fn TotalCount(&self) -> usize {
        match self {
            Self::Block8s(inner) => return (*inner).TotalCount(),
            Self::Block16s(inner) => return (*inner).TotalCount(),
            Self::Block32s(inner) => return (*inner).TotalCount(),
            Self::Block64s(inner) => return (*inner).TotalCount(),
            Self::Block128s(inner) => return (*inner).TotalCount(),
            Self::Block256s(inner) => return (*inner).TotalCount(),
            Self::Block512s(inner) => return (*inner).TotalCount(),
            Self::Block1ks(inner) => return (*inner).TotalCount(),
            Self::Block2ks(inner) => return (*inner).TotalCount(),
            Self::Block4ks(inner) => return (*inner).TotalCount(),
            Self::Block8ks(inner) => return (*inner).TotalCount(),
            Self::Block16ks(inner) => return (*inner).TotalCount(),
            Self::Block32ks(inner) => return (*inner).TotalCount(),
            Self::Block64ks(inner) => return (*inner).TotalCount(),
            Self::Block128ks(inner) => return (*inner).TotalCount(),
            Self::Block256ks(inner) => return (*inner).TotalCount(),
            Self::Block512ks(inner) => return (*inner).TotalCount(),
            Self::Block1Ms(inner) => return (*inner).TotalCount(),
            Self::Block2Ms(inner) => return (*inner).TotalCount(),
        }
    }
}

pub type Block8s     = BitmapSetBlock<3, 12>;
pub type Block16s    = BitmapSetBlock<4, 12>;
pub type Block32s    = BitmapSetBlock<5, 12>;
pub type Block64s    = BitmapSetBlock<6, 12>;
pub type Block128s   = BitmapSetBlock<7, 12>;
pub type Block256s   = BitmapSetBlock<8, 12>;
pub type Block512s   = BitmapSetBlock<9, 12>;
pub type Block1ks    = BitmapSetBlock<10, 11>;
pub type Block2ks    = BitmapSetBlock<11, 10>;
pub type Block4ks    = BitmapSetBlock<12, 9>;
pub type Block8ks    = BitmapSetBlock<13, 8>;
pub type Block16ks   = BitmapBlock<14>;
pub type Block32ks   = BitmapBlock<15>;
pub type Block64ks   = BitmapBlock<16>;
pub type Block128ks  = BitmapBlock<17>;
pub type Block256ks  = BitmapBlock<18>;
pub type Block512ks  = BitmapBlock<19>;
pub type Block1Ms    = BitmapBlock<20>;

// total 8GB/4096 * 2MB blocks
pub struct Block2Ms {
    pub bitmaps: BitmapSetBlock<21, 12>,
    pub bitmap2M: [QMutex<u128>; 4096], // (128 + 8) x 4k = 544KB
}

impl Block2Ms {
    pub fn Bootstrap(&mut self) {
        // the first 2MB is the Block2Ms metadata
        self.Init();

        /*let bitmapAllocatorAddr = self.Alloc().unwrap();
        let bitmapAllocator = BitmapAllocator::GetMut(bitmapAllocatorAddr);


        bitmapAllocator.block8GAlllocator.BootstrapBlock8G(self.Addr());
        return bitmapAllocatorAddr;*/
    }

    pub fn Init(&mut self) {
        self.bitmaps.Init();
        // reserve first 2M for metadata
        self.bitmaps.Reserve(1); 
    }

    pub fn BlockMask() -> u64 {
        return (1 << 33) - 1;
    }

    pub fn GetMut(addr: u64) -> &'static mut Self {
        let alignedAddr = addr & !Self::BlockMask();
        return unsafe {
            &mut *(alignedAddr as * mut Self)
        }
    }

    pub fn GetMutDirect(addr: u64) -> &'static mut Self {
        return unsafe {
            &mut *(addr as * mut Self)
        }
    }

    pub fn Alloc(&mut self) -> Option<u64> {
        let addr = match self.bitmaps.Alloc() {
            None => return None,
            Some(addr) => addr,
        };

        return Some(addr)
    }

    pub fn Free(&mut self, addr: u64) -> bool {
        return self.bitmaps.Free(addr);
    }

    pub fn Idx(&self, addr: u64) -> u64 {
        assert!(addr & ((1<<21) - 1) == 0);
        let idx = (addr - self.Addr()) >> 21;
        assert!(idx<4096);
        return idx;
    } 

    pub fn Addr(&self) -> u64 {
        return self as * const Self as u64;
    }

    pub fn SubBlockAddr(&self, idx: u64) -> u64 {
        assert!(idx < 4096);
        return self.Addr() + idx << 21;
    }

    pub fn FreeCount(&self) -> usize {
        return self.bitmaps.FreeCount();
    }

    pub fn TotalCount(&self) -> usize {
        return self.bitmaps.TotalCount();
    }
}

pub struct BitmapBlock<const SUB_BLOCK_ORDER: usize> {}

impl <const SUB_BLOCK_ORDER: usize> BitmapBlock <SUB_BLOCK_ORDER> {
    pub const COUNT_ORDER : usize = 21 - SUB_BLOCK_ORDER;

    pub fn Init(&self) {
        let count: usize = 1 << Self::COUNT_ORDER;
        let block2Ms = self.Parent();
        let idx = self.BlockIdx() as usize;

        let mut bitmap = block2Ms.bitmap2M[idx].lock();
        for i in 0..count {
            *bitmap |= 1 << i;
        }
    }

    pub fn Alloc(&mut self) -> Option<u64> {
        let block2Ms = self.Parent();
        let idx = self.BlockIdx() as usize;

        let mut bitmap = block2Ms.bitmap2M[idx].lock();
        let subBlockidx = bitmap.trailing_zeros();
        if subBlockidx == 128 {
            return None;
        }

        let tmp = *bitmap;
        *bitmap = tmp & !(1u128 << subBlockidx);

        return Some(self.SubBlockAddr(subBlockidx as u64));
    }

    // return true iff the pre-condition is empty
    pub fn Free(&mut self, addr: u64) -> bool {
        let block2Ms = self.Parent();
        let idx = self.BlockIdx() as usize;

        let mut bitmap = block2Ms.bitmap2M[idx].lock();
        let tmp = *bitmap;
        let empty = tmp == 0;
        let idx = self.SubBlocIndex(addr);
        *bitmap = tmp | 1 << idx;
        return empty;
    } 

    pub fn Parent(&self) -> &'static mut Block2Ms {
        let parentMask = (1 << 33) - 1;
        let parentAddr = self.Addr() & !parentMask;
        return Block2Ms::GetMut(parentAddr)
    }

    // the idx in parent 8GB block
    pub fn BlockIdx(&self) -> u64 {
        let parentMask = (1 << 33) - 1;
        let parentAddr = self.Addr() & !parentMask;
        return (self.Addr() - parentAddr) / Self::BlockSize()
    } 

    pub fn BlockSize() -> u64 {
        return 1 << 21;
    }

    pub fn BlockMask() -> u64 {
        return (1 << 21) - 1;
    }

    pub fn GetMut(addr: u64) -> &'static mut Self {
        let alignedAddr = addr & !Self::BlockMask();
        return unsafe {
            &mut *(alignedAddr as * mut Self)
        }
    }

    pub fn GetMutDirect(addr: u64) -> &'static mut Self {
        return unsafe {
            &mut *(addr as * mut Self)
        }
    }

    pub fn BlockOrder() -> usize {
        return 21;
    }

    pub fn Addr(&self) -> u64 {
        return self as * const Self as u64;
    }

    pub fn SubBlockSize(&self) -> u64 {
        return 1 << SUB_BLOCK_ORDER
    }

    pub fn SubBlockMask(&self) -> u64 {
        return self.SubBlockSize() - 1;
    }

    pub fn SubBlocIndex(&self, addr: u64) -> u64 {
        assert!(addr & self.SubBlockMask() == 0);
        let offset = addr - self.Addr();
        return offset / self.SubBlockSize()
    }

    pub fn SubBlockAddr(&self, idx: u64) -> u64 {
        assert!(idx < (1<<SUB_BLOCK_ORDER));
        return self.Addr() + (idx << SUB_BLOCK_ORDER);
    }

    pub fn FreeCount(&self) -> usize {
        let block2Ms = self.Parent();
        let idx = self.BlockIdx() as usize;

        let bitmap = block2Ms.bitmap2M[idx].lock();
        return bitmap.count_ones() as usize;
    }

    pub fn TotalCount(&self) -> usize {
        return 1 << Self::COUNT_ORDER;
    }
}

// handle total 64 * 64 sub blocks, from 8 Bytes to 512B and 2MB sub block size
// the ORDER is the block size, it is sub_block_size << 12

pub struct BitmapSetBlock<const SUB_BLOCK_ORDER: usize, const COUNT_ORDER: usize> {
    pub lock: QMutex<()>, // we need lock this before access following fields
    pub bitmapSet: BitmapSet<COUNT_ORDER>, // < 4KB
}

impl <const SUB_BLOCK_ORDER: usize, const COUNT_ORDER: usize> 
    InlineStruct<SUB_BLOCK_ORDER, COUNT_ORDER> 
    for BitmapSetBlock<SUB_BLOCK_ORDER, COUNT_ORDER> {}

impl <const SUB_BLOCK_ORDER: usize, const COUNT_ORDER: usize> 
    BitmapSetBlock<SUB_BLOCK_ORDER, COUNT_ORDER> {
    // Preconditions: the block is initalized to zero
    pub fn Init(&mut self) {
        self.bitmapSet.Init();
    }

    pub fn Reserve(&mut self, count: usize) {
        self.bitmapSet.Reserve(count);
    }

    pub fn FreeBlockCount(&self) -> usize {
        self.lock.lock();
        return self.bitmapSet.free as usize;
    }

    pub fn Alloc(&mut self) -> Option<u64> {
        let _l = self.lock.lock();
        let idx = match self.bitmapSet.Alloc() {
            None => return None,
            Some(i) => i
        } as u64;

        let addr = self.Addr() + idx * self.SubBlockSize();
        return Some(addr)
    }

    // return true iff the pre-condition is empty
    pub fn Free(&mut self, addr: u64) -> bool {
        let empty = self.bitmapSet.free == 0;
        let idx = self.SubBlocIndex(addr);
        self.bitmapSet.Free(idx as usize);
        return empty;
    }

    pub fn FreeCount(&self) -> usize {
        return self.bitmapSet.free as usize;
    }

    pub fn TotalCount(&self) -> usize {
        return self.bitmapSet.total as usize;
    }
}


pub trait InlineStruct<const SUB_BLOCK_ORDER: usize, const COUNT_ORDER: usize> : Sized {
    fn BlockOrder() -> usize {
        return SUB_BLOCK_ORDER + COUNT_ORDER;
    }

    fn GetMut(addr: u64) -> &'static mut Self {
        let mask = !((1 << Self::BlockOrder()) as u64 - 1);
        let alignedAddr = addr & mask;
        return Self::GetMutDirect(alignedAddr);
    }

    fn GetMutDirect(addr: u64) -> &'static mut Self {
        return unsafe {
            &mut *(addr as * mut Self)
        }
    }

    fn Addr(&self) -> u64 {
        return self as * const Self as u64;
    }

    fn SubBlockSize(&self) -> u64 {
        return 1 << SUB_BLOCK_ORDER
    }

    fn SubBlockMask(&self) -> u64 {
        return self.SubBlockSize() - 1;
    }

    fn SubBlocIndex(&self, addr: u64) -> u64 {
        assert!(addr & self.SubBlockMask() == 0);
        let offset = addr - self.Addr();
        return offset / self.SubBlockSize()
    }
}

pub struct Block8GAllocatorIntern { // <4KB
    pub availableBlockCnt: u32, // next free 8GB block idx
    pub bitmap: u128, // maximum 128 x 8 GB = 1024 GB 
    pub BlockArr: [u64; 128], 
    pub current8GAddr: u64,
}

pub struct Block8GAllocator(QMutex<Block8GAllocatorIntern>);

impl Deref for Block8GAllocator {
    type Target = QMutex<Block8GAllocatorIntern>;

    fn deref(&self) -> &QMutex<Block8GAllocatorIntern> {
        &self.0
    }
}

impl Block8GAllocator {
    /*pub fn Init(addr: u64) -> &self {
        assert!(addr.trailing_zeros() >= 23); // the address must be 8GB alligned
        let b = Self::GetMut(addr);


    }*/

    pub fn BootstrapBlock8G(&self, addr: u64) {
        let mut intern = self.lock();
        let idx = intern.availableBlockCnt as usize;
        intern.bitmap |= 1 << intern.availableBlockCnt;
        intern.BlockArr[idx] = addr;
        intern.availableBlockCnt += 1;
        intern.current8GAddr = addr;

        /*let block2Ms = Block2Ms::GetMut(addr);
        block2Ms.Init();*/
    }

    // placeholder, implement later
    pub fn Alloc8G(&mut self) {}

    pub fn Alloc2M(&self) -> Option<u64> {
        let addr8G = self.lock().current8GAddr;
        let block2Ms = Block2Ms::GetMut(addr8G);
        return block2Ms.Alloc();
    }
}

pub struct FreeListNodeAllocatorIntern {
    pub nextIdx: u32,
    pub freeList: u32,
    pub freeNodeArr: [FreeListNode; FREE_NODE_COUNT]
}

impl FreeListNodeAllocatorIntern {
    pub fn Alloc(&mut self) -> u32 {
        assert!(self.nextIdx < self.freeNodeArr.len() as u32);
        self.nextIdx += 1;
        return self.nextIdx - 1;
    }

    pub fn GetMut<'a>(&'a mut self, idx: u32) -> &'a mut FreeListNode {
        return &mut self.freeNodeArr[idx as usize];
    }
}

pub struct FreeListNodeAllocator(QMutex<FreeListNodeAllocatorIntern>);

impl Deref for FreeListNodeAllocator {
    type Target = QMutex<FreeListNodeAllocatorIntern>;

    fn deref(&self) -> &QMutex<FreeListNodeAllocatorIntern> {
        &self.0
    }
}

pub const MAX_ORDER : usize = 20; // 1MB
pub const FREE_NODE_COUNT: usize = 4096;
#[repr(C)]
pub struct BitmapAllocator {
    pub block2M: Block2Ms,
    // first order 3 start with Lists[0]
    // the max order is 20 at indx Lists[17]
    // so total 18 records
    pub Lists: [HeadNode; MAX_ORDER - 3 + 1], 
    
    pub block8GAlllocator: Block8GAllocator, 
    pub freeListAllocator: FreeListNodeAllocator,
}

impl BitmapAllocator {
    pub fn Bootstrap(&mut self, addr: u64) {
        self.block2M.Bootstrap();
        self.block8GAlllocator.BootstrapBlock8G(addr);
        //let first2M = self.block8GAlllocator.Alloc2M().unwrap();
        //assert!(first2M == addr);

        for i in (0..self.Lists.len()).rev() {
            let order = i + 3;
            let parentOrder = if order < 10 {
                order + 12
            } else {
                21
            };

            let newAddr = self.Alloc(parentOrder).unwrap();
            let mut headnode = self.Lists[order-3].lock();
            let mut newBlock = BlockType::GetMut(order,newAddr);
            newBlock.Init();
            headnode.PushFront(newAddr, newBlock.TotalCount(), &self.freeListAllocator)
        }
    }

    pub fn GetMut(addr: u64) -> &'static mut Self {
        return unsafe {
            &mut *(addr as * mut Self)
        }
    }

    pub fn Free(order: usize, addr: u64) -> bool {
        return BlockType::Free(order, addr);
    } 

    pub fn Alloc(&self, order: usize) -> Option<u64> {
        if order < 21 {
            let mut headnode = self.Lists[order-3].lock();
            loop {
                let addr = headnode.currAddr;
                let mut bs = BlockType::GetMut(order, addr);
                match bs.Alloc() {
                    Some(allocAddr) => {
                        headnode.totalFree -= 1;
                        return Some(allocAddr)
                    }
                    None => (),
                }
    
                if headnode.totalFree < 256 {
                    let parentOrder = if order < 10 {
                        order + 12
                    } else {
                        21
                    };
    
                    let newAddr = match self.Alloc(parentOrder) {
                        None => return None,
                        Some(addr) => addr,
                    };
    
                    let mut newBlock = BlockType::GetMut(order,newAddr);
                    newBlock.Init();
                    headnode.PushFront(newAddr, newBlock.TotalCount(), &self.freeListAllocator);
                } else {
                    headnode.Rotate(&self.freeListAllocator);
                }                   
            }
        } else if order == 21 {
            return self.block8GAlllocator.Alloc2M();
        } else {
            panic!("BitmapAllocator wrong order {}", order)
        }
    }
}


unsafe impl GlobalAlloc for BitmapAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );

        let class = size.trailing_zeros() as usize;

        return self.Alloc(class).unwrap() as _;
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );
        let class = size.trailing_zeros() as usize;

        Self::Free(class, ptr as _);
    }
}

#[derive(Debug)]
pub struct BitmapAllocatorWrapper {
    pub addr: atomic::AtomicU64,
}

impl BitmapAllocatorWrapper {
    pub fn Allocator(&self) -> &mut BitmapAllocator {
        let mut addr = self.addr.load(atomic::Ordering::Relaxed);
        if addr == 0 {
            self.Init();
            addr = self.addr.load(atomic::Ordering::Relaxed);
            let allocator = unsafe { &mut *(addr as *mut BitmapAllocator) };
            allocator.Bootstrap(addr);
            eturn allocator
        }

        return unsafe { &mut *(addr as *mut BitmapAllocator) };
    }

    pub fn Clear(&self) {}
    pub fn Initializated(&self) {}
}

unsafe impl GlobalAlloc for BitmapAllocatorWrapper {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        return self.Allocator().alloc(layout);
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        return self.Allocator().dealloc(ptr, layout);
    }
}