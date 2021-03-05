// Copyright (c) 2021 QuarkSoft LLC
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
use alloc::vec::Vec;
use core::ops::Deref;
use spin::Mutex;

//use alloc::string::String;

use super::addr::*;
use super::common::*;

fn ZeroPage(pageStart: u64) {
    unsafe {
        let arr = slice::from_raw_parts_mut(pageStart as *mut u64, 512);
        for i in 0..512 {
            arr[i] = 0
        }
    }
}

#[derive(PartialEq, Copy, Clone, Default)]
pub struct MemAllocatorInternal {
    ba: BuddyAllocator,
    baseAddr: u64,
}

impl MemAllocatorInternal {
    pub fn New() -> Self {
        return Self {
            ba: BuddyAllocator::New(0, 0),
            baseAddr: 0,
        }
    }

    //baseAddr: is the base memory address
    //ord: the memory size is 2^ord pages
    //memory layout: the Buddy Allocator's memory is also allocated by itself.
    pub fn Init(baseAddr: u64, ord: u64) -> Self {
        let mut ba = BuddyAllocator::New(ord, baseAddr);
        let baSize = 1 << (ord + 1);
        let mut baPages = baSize >> PAGE_SHIFT;
        if (baSize & PAGE_MASK) != 0 {
            baPages += 1;
        }

        //alloc baPages for the BuddyAllocator
        let addr = ba.allocate(baPages) as u64;
        assert_eq!(addr, 0);

        return Self {
            ba,
            baseAddr,
        }
    }

    pub fn Load(&mut self, baseAddr: u64, ord: u64) {
        self.ba.Load(ord, baseAddr);
        self.baseAddr = baseAddr;
    }

    pub fn Alloc(&mut self, pages: u64) -> Result<u64> {
        let pageOff = self.ba.allocate(pages);
        if pageOff == -1 {
            //super::KernelMsg(4);
            info!("buddyalloc ...");
            Err(Error::NoEnoughMemory)
        } else {
            Ok(self.baseAddr + ((pageOff as u64) << PAGE_SHIFT))
        }
    }

    pub fn Free(&mut self, addr: u64, pages: u64) -> Result<()> {
        let pageOff = (addr - self.baseAddr) as u64 >> PAGE_SHIFT;

        let ret = self.ba.free(pageOff, pages);
        if ret {
            Ok(())
        } else {
            Err(Error::InvalidInput)
        }
    }
}

pub struct MemAllocator(Mutex<MemAllocatorInternal>);

impl Deref for MemAllocator {
    type Target = Mutex<MemAllocatorInternal>;

    fn deref(&self) -> &Mutex<MemAllocatorInternal> {
        &self.0
    }
}

impl RefMgr for MemAllocator {
    fn Ref(&self, _addr: u64) -> Result<u64> {
        return Ok(1)
    }

    fn Deref(&self, addr: u64) -> Result<u64> {
        self.FreePage(addr).unwrap();
        Ok(0)
    }

    fn GetRef(&self, _addr: u64) -> Result<u64> {
        Ok(1)
    }
}

impl Allocator for MemAllocator {
    fn AllocPage(&self, _incrRef: bool) -> Result<u64> {
        let res = self.lock().Alloc(1);
        return res
    }

    fn FreePage(&self, addr: u64) -> Result<()> {
        ZeroPage(addr);
        return self.lock().Free(addr, 1)
    }

    fn ZeroPage(&self) -> u64 {
        panic!("MemAllocator doesn't support zeropage");
    }
}

impl MemAllocator {
    pub fn New() -> Self {
        return Self(Mutex::new(MemAllocatorInternal::New()))
    }

    //baseAddr: is the base memory address
    //ord: the memory size is 2^ord pages
    //memory layout: the Buddy Allocator's memory is also allocated by itself.
    pub fn Init(baseAddr: u64, ord: u64) -> Self {
        return Self(Mutex::new(MemAllocatorInternal::Init(baseAddr, ord)))
    }

    pub fn Load(&self, baseAddr: u64, ord: u64) {
        self.lock().Load(baseAddr, ord);
    }

    pub fn Alloc(&self, pages: u64) -> Result<u64> {
        return self.lock().Alloc(pages);
    }

    pub fn Free(&self, addr: u64, pages: u64) -> Result<()> {
        return self.lock().Free(addr, pages);
    }
}

#[repr(u8)]
#[derive(PartialEq, Copy, Clone, Debug)]
enum Node {
    Unused = 0,
    Used,
    Split,
    Full,
}

#[derive(PartialEq, Copy, Clone, Default)]
pub struct BuddyAllocator {
    levels: u64,
    size: u64,
    root: u64,
}

impl BuddyAllocator {
    pub fn New(levels: u64, addr: u64) -> BuddyAllocator {
        let size: u64 = (1 << (levels + 1)) - 1;

        return BuddyAllocator {
            levels: levels,
            size: size,
            root: addr,
        };
    }

    pub fn Load(&mut self, levels: u64, addr: u64) {
        let size: u64 = (1 << (levels + 1)) - 1;

        self.levels = levels;
        self.size = size;
        self.root = addr;
    }

    fn tree(&self) -> &mut [Node] {
        unsafe {
            slice::from_raw_parts_mut(self.root as *mut Node, self.size as usize)
        }
    }

    fn alloc(&mut self, idx: u64, t_level: u64, c_level: u64) -> isize {
        if c_level == t_level {
            if self.tree()[idx as usize] == Node::Unused {
                self.tree()[idx as usize] = Node::Used;
                let current_level_offset = (1 << self.levels - c_level) - 1;
                return (idx - current_level_offset) as isize * (1 << c_level);
            } else {
                return -1;
            }
        }

        let left_child = idx * 2 + 1;
        let right_child = idx * 2 + 2;

        match self.tree()[idx as usize] {
            Node::Used | Node::Full => {
                return -1
            }
            Node::Unused => {
                self.tree()[idx as usize] = Node::Split;
                return self.alloc(left_child, t_level, c_level - 1);
            }
            Node::Split => {
                let mut res = self.alloc(left_child, t_level, c_level - 1);
                if res == -1 {
                    res = self.alloc(right_child, t_level, c_level - 1);
                }

                self.CheckParentFull(idx);

                return res;
            }
        }
    }

    fn alloc1(&mut self, t_level: u64) -> isize {
        let mut stack: Vec<(u32, u32)> = Vec::with_capacity(self.levels as usize + 1);

        stack.push((0, self.levels as u32));

        while stack.len() > 0 {
            let (idx, c_level) = stack.pop().unwrap();
            if c_level as u64 == t_level {
                if self.tree()[idx as usize] == Node::Unused {
                    self.tree()[idx as usize] = Node::Used;
                    let current_level_offset = (1 << self.levels - c_level as u64) - 1;

                    if idx != 0 {
                        let mut parent = (idx + 1) / 2 - 1;
                        'inner: loop {
                            self.CheckParentFull(parent as u64);
                            if parent == 0 {
                                break 'inner;
                            }

                            parent = (parent + 1) / 2 - 1;
                        }
                    }

                    return (idx - current_level_offset) as isize * (1 << c_level);
                } else {
                    continue
                }
            }

            let left_child = idx * 2 + 1;
            let right_child = idx * 2 + 2;

            match self.tree()[idx as usize] {
                Node::Used | Node::Full => {
                    continue
                }
                Node::Unused => {
                    self.tree()[idx as usize] = Node::Split;
                    stack.push((left_child, c_level - 1));
                }
                Node::Split => {
                    stack.push((right_child, c_level - 1));
                    stack.push((left_child, c_level - 1));
                }
            }
        }

        return -1;
    }

    const STACK_LEN: usize = 28; // 2^(28+12 - 1) => 512GB

    fn alloc2(&mut self, t_level: u64) -> isize {
        //let mut stack : Vec<(u32, u32)> = Vec::with_capacity(self.levels as usize + 1);
        //assert!(self.levels < Self::STACK_LEN as u64);
        let mut stack: [(u32, u32); Self::STACK_LEN] = [(0, 0); Self::STACK_LEN];
        let mut top = 0;

        stack[top] = (0, self.levels as u32);
        top += 1;

        while top > 0 {
            let (idx, c_level) = stack[top - 1];
            top -= 1;
            //info!("idx is {}, c_level is {}", idx, c_level);
            if c_level as u64 == t_level {
                if self.tree()[idx as usize] == Node::Unused {
                    self.tree()[idx as usize] = Node::Used;
                    let current_level_offset = (1 << self.levels - c_level as u64) - 1;

                    if idx != 0 {
                        let mut parent = (idx + 1) / 2 - 1;
                        'inner: loop {
                            self.CheckParentFull(parent as u64);
                            if parent == 0 {
                                break 'inner;
                            }

                            parent = (parent + 1) / 2 - 1;
                        }
                    }

                    return (idx - current_level_offset) as isize * (1 << c_level);
                } else {
                    continue
                }
            }

            let left_child = idx * 2 + 1;
            let right_child = idx * 2 + 2;

            match self.tree()[idx as usize] {
                Node::Used | Node::Full => {
                    continue
                }
                Node::Unused => {
                    self.tree()[idx as usize] = Node::Split;
                    stack[top] = (left_child, c_level - 1);
                    top += 1;
                }
                Node::Split => {
                    stack[top] = (right_child, c_level - 1);
                    top += 1;
                    stack[top] = (left_child, c_level - 1);
                    top += 1;
                }
            }
        }

        return -1;
    }

    pub fn CheckParentFull(&mut self, idx: u64) {
        let mut idx = idx;

        while idx != 0 {
            let left_child = idx * 2 + 1;
            let right_child = idx * 2 + 2;
            let left_child_used_or_full = self.tree()[left_child as usize] == Node::Full || self.tree()[left_child as usize] == Node::Used;
            let right_child_used_or_full = self.tree()[right_child as usize] == Node::Full || self.tree()[right_child as usize] == Node::Used;
            if left_child_used_or_full && right_child_used_or_full {
                self.tree()[idx as usize] = Node::Full;
            }

            idx = (idx + 1) / 2 - 1;
        }
    }

    pub fn allocate(&mut self, num_pages: u64) -> isize {
        // Get the requested level from number of pages requested

        let requested_level = self.get_level_from_num_pages(num_pages);
        if requested_level > self.levels {
            return -1;
        }

        //let c_level = self.levels;
        //return self.alloc(0, requested_level, c_level)
        //todo: move to alloc2 later to use stack on stack
        return self.alloc2(requested_level)
    }

    pub fn free(&mut self, page_offset: u64, num_pages: u64) -> bool {
        //check whether the page is allocated from the pool
        //when doing the pagetable free, some final page might just mmap instead of being allocated from the page pool
        if self.root <= page_offset && page_offset <= self.root + self.size {
            return false;
        }

        let requested_level = self.get_level_from_num_pages(num_pages);
        // infer index offset from page_offset
        let level_offset = page_offset / (1 << requested_level);
        let current_level_offset = (1 << self.levels - requested_level) - 1;
        let mut idx = current_level_offset + level_offset;

        if idx as usize > self.tree().len() - 1 {
            panic!("offset {} is > length of tree() {}", idx, self.tree().len());
        }

        if self.tree()[idx as usize] != Node::Used {
            return false; //len doesn't match
        }

        self.tree()[idx as usize] = Node::Unused;
        while idx != 0 {
            let parent = (idx + 1) / 2 - 1;

            let left_child = parent * 2 + 1;
            let right_child = parent * 2 + 2;

            if self.tree()[left_child as usize] == Node::Unused && self.tree()[right_child as usize] == Node::Unused {
                self.tree()[parent as usize] = Node::Unused;
            } else {
                self.tree()[parent as usize] = Node::Split;
            }

            idx = parent
        }

        return true;
    }

    fn get_level_from_num_pages(&self, num_pages: u64) -> u64 {
        // Get the number of pages requested
        let requested_pages;
        if num_pages == 0 {
            requested_pages = 1;
        } else {
            requested_pages = num_pages.next_power_of_two();
        }
        let requested_level = self.log_base_2(requested_pages);
        requested_level
    }

    // Finds the position of the most signifcant bit
    fn log_base_2(&self, requested_pages: u64) -> u64 {
        let mut exp = 0;
        let mut find_msb_bit = requested_pages;
        find_msb_bit >>= 1;
        while find_msb_bit > 0 {
            find_msb_bit >>= 1;
            exp += 1;
        }
        return exp;
    }

    /*pub fn dump(&self) -> String {
        let mut out = "".to_string();
        let mut row = "".to_string();
        let mut level = 0;
        let mut index = 0;
        loop {
            if index == self.tree().len() {
                break
            }
            match self.tree()[index] {
                Node::Used => row += "U",
                Node::Unused => row += "O",
                Node::Split => row += "S",
                Node::Full => row += "F",
            }
            if row.len() == 1 << level {
                out += &(row + "\n");
                row = "".to_string();
                level += 1;
            }
            index += 1;
        }
        return out;
    }*/
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_alloc() {
        let mem = [0 as u8; 15];
        let mut alloc = buddyallocator::New(3, &mem[0] as *const _ as u64);

        //assert_eq!(alloc.allocate(8), 0);
        assert_eq!(alloc.allocate(9), -1);

        let offset1 = alloc.allocate(1);
        assert_eq!(offset1, 0);

        let offset2 = alloc.allocate(3);
        assert_eq!(offset2, 4);

        alloc.free(offset2 as u64, 3);
        alloc.free(offset1 as u64, 1);

        let offset3 = alloc.allocate(8);
        assert_eq!(offset3, 0);

        alloc.free(offset3 as u64, 8);

        let offset4 = alloc.allocate(9);
        assert_eq!(offset4, -1);
    }

    #[test]
    fn test_alloc1() {
        let mut alloc = buddyallocator::New(0, 0);

        assert_eq!(alloc.allocate(9), -1);
    }
}