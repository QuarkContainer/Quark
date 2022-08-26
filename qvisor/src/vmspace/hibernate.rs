// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
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

use spin::Mutex;
use std::collections::HashMap;
use alloc::collections::BTreeSet;
use std::collections::hash_map::Entry;
use std::fs::OpenOptions;
//use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::IntoRawFd;
use userfaultfd::UffdBuilder;
use userfaultfd::Uffd;
use std::os::unix::io::AsRawFd;
use libc::*;

use crate::qlib::*;
use crate::qlib::common::*;
use crate::qlib::linux_def::*;
use crate::qlib::hiber_mgr::*;
use crate::qlib::mem::block_allocator::*;
use crate::SWAP_FILE;
use crate::SHARE_SPACE;
use crate::qlib::mem::list_allocator::GLOBAL_ALLOCATOR;

impl HiberMgr {
    pub fn SwapOut(&self, start: u64, len: u64) -> Result<()> {
		let mut intern = self.lock();
		let mut map = BTreeSet::new();
		for (_, mm) in &intern.memmgrs {
			let mm = mm.Upgrade();
			mm.pagetable.write().pt.SwapOutPages(start, len, &mut map).unwrap();
		}

        let mut insertCount = 0;
        for page in map.iter() {
            let offset = SWAP_FILE.lock().SwapOutPage(*page)?;
            intern.pageMap.entry(*page).or_insert_with(||{
                insertCount += 1;
                offset
            } );
        }

        info!("swapout {} pages, new pages {} pages", map.len(), insertCount);

        let cnt = SHARE_SPACE.pageMgr.pagepool.DontneedFreePages()?;

        let allocated1 = GLOBAL_ALLOCATOR.Allocator().heap.lock().allocated;
        GLOBAL_ALLOCATOR.Allocator().FreeAll();
        let allocated2 = GLOBAL_ALLOCATOR.Allocator().heap.lock().allocated;
        info!("free pagepool {} pages, total allocated1 {} allocated2 {}", cnt, allocated1, allocated2);
        info!("heap usage1 is {:?}", &GLOBAL_ALLOCATOR.Allocator().counts);
        for i in 3..20 {
            info!("heap usage2 is {}/{:x}/{:?}/{:?}", i, 1<<i, GLOBAL_ALLOCATOR.Allocator().counts[i], GLOBAL_ALLOCATOR.Allocator().maxnum[i]);
        }
        info!("heap usage3 is {:?}", &GLOBAL_ALLOCATOR.Allocator().maxnum);

        return Ok(())
	}

    pub fn SwapIn(&self, phyAddr: u64) -> Result<()> {
        let mut intern = self.lock();

		match intern.pageMap.remove(&phyAddr) {
            None => {
                return Err(Error::SysError(SysErr::EINVAL))
            }
            Some(offset) => {
                return SWAP_FILE.lock().SwapInPage(phyAddr, offset)
            }
        }
	}
}


pub struct SwapFile {
    pub fd: i32,            // the file fd  
    pub size: u64,              //total allocated file size
    pub nextAllocOffset: u64, // the last allocated slot
    pub freeSlots: Vec<u64>, // free page slot
    pub mmapAddr: u64, // the file mmap start address
}

pub const SWAP_FILE_NAME : &str = "./swapfile.data";
pub const INIT_FILE_SIZE: u64 = MemoryDef::PAGE_SIZE_2M;
pub const EXTEND_FILE_SIZE: u64 = MemoryDef::PAGE_SIZE_2M;

impl SwapFile {
    pub fn Init() -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            //.custom_flags(O_DIRECT)
            .create(true)
            .open(SWAP_FILE_NAME)
            .unwrap();
        let fd = file.into_raw_fd();

        let ret = unsafe {
            libc::ftruncate(fd, INIT_FILE_SIZE as _)
        };
        
        GetRet(ret as _)?;
        
        /*let ret = unsafe {
            libc::mmap(
                ptr::null_mut(),
                INIT_FILE_SIZE as _,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };

        let addr = GetRet(ret as _)?;*/
        let addr = 0;

        return Ok(Self {
            fd: fd,
            size: INIT_FILE_SIZE,
            nextAllocOffset: 0,
            freeSlots: Vec::new(),
            mmapAddr: addr,
        })
    }

    pub fn ExtendSize(&mut self) -> Result<()> {
        let newSize = self.size + EXTEND_FILE_SIZE;
        let ret = unsafe {
            libc::ftruncate(self.fd, newSize as _) 
        };

        GetRet(ret as _)?;

        let ret = unsafe {
            libc::mmap(
                (self.mmapAddr + self.size) as _,
                EXTEND_FILE_SIZE as _,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                self.fd,
                0,
            )
        };

        let addr = GetRet(ret as _)?;
        assert!(addr == self.mmapAddr + self.size);
        
        self.size = newSize;
        return Ok(())
    }

    pub fn AllocateSlot(&mut self) -> Result<u64> {
        match self.freeSlots.pop() {
            Some(offset) => return Ok(offset),
            None => (),
        }

        let offset = self.nextAllocOffset;
        if offset == self.size {
            self.ExtendSize()?;
        }

        self.nextAllocOffset += MemoryDef::PAGE_SIZE_4K;
        return Ok(offset);
    }

    pub fn FreeSlot(&mut self, offset: u64) -> Result<()> {
        self.freeSlots.push(offset);
        return Ok(())
    } 

    // input: memory page address
    // ret: file offset
    pub fn SwapOutPage(&mut self, addr: u64) -> Result<u64> {
        let offset = self.AllocateSlot()?;
        let ret = unsafe {
            libc::pwrite(self.fd, addr as _, MemoryDef::PAGE_SIZE_4K as _, offset as _)
        };

        let count = GetRet(ret as _)?;
        assert!(count == MemoryDef::PAGE_SIZE_4K);

        let ret = unsafe {
            libc::madvise(addr as _, MemoryDef::PAGE_SIZE_4K as _, libc::MADV_DONTNEED)
        };

        GetRet(ret as _)?;

        return Ok(offset)
    }

    // input: memory page address, file offset
    // ret: file offset
    pub fn SwapInPage(&mut self, addr: u64, offset: u64) -> Result<()> {
        let ret = unsafe {
            libc::pread(self.fd, addr as _, MemoryDef::PAGE_SIZE_4K as _, offset as _)
        };

        let count = GetRet(ret as _)?;
        assert!(count == MemoryDef::PAGE_SIZE_4K);
        self.FreeSlot(offset)?;
        
        return Ok(())
    }

    pub fn OffsetToAddr(&self, offset: u64) -> u64 {
        assert!(offset < self.size);
        return self.mmapAddr + offset;
    }
}

impl PageBlock {
    // madvise MADV_DONTNEED free pages
    pub fn DontneedFreePages(&self) -> Result<u64> {
        let alloc = self.allocator.lock();
        if alloc.freePageList.totalFreeCount == 0 {
            return Ok(0)
        }

        for idx in 1..BLOCK_PAGE_COUNT+1 {
            if alloc.freePageList.IsFree(idx as _) {
                let addr = self.IdxToAddr(idx as _)
;                let ret = unsafe {
                    libc::madvise(addr as _, MemoryDef::PAGE_SIZE_4K as _, libc::MADV_DONTNEED)
                };
        
                GetRet(ret as _)?;
            }
        }

        return Ok(alloc.freePageList.totalFreeCount)
    }
}

impl PageBlockAlloc {
    // madvise MADV_DONTNEED free pages
    pub fn DontneedFreePages(&self) -> Result<u64> {
        let mut total = 0;
        let intern = self.data.lock();
        for addr in &intern.pageBlocks {
            let pb = PageBlock::FromAddr(*addr);
            total = pb.DontneedFreePages()?;
        }

        return Ok(total)
    }
}

pub struct HiberMap {
    pub blockRef: HashMap<u64, u32>, // blockAddr --> refcnt
    pub pageMap: HashMap<u64, u64>, // pageAddr --> file offset
}
pub struct HiberMgr1 {
    pub epfd: i32,
    pub eventfd: i32,
    pub map: Mutex<HiberMap>,
    pub swapFile: Mutex<SwapFile>,
    pub uffd: Uffd,
}

impl HiberMgr1 {
    pub fn New() -> Result<Self> {
        let epfd = unsafe { epoll_create1(0) };

        if epfd == -1 {
            panic!(
                "HiberMgr::Init create epollfd fail, error is {}",
                errno::errno().0
            );
        }

        let eventfd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) };

        if eventfd < 0 {
            panic!("Vcpu::Init fail...");
        }

        super::VMSpace::UnblockFd(eventfd);

        let mut ev = epoll_event {
            events: EVENT_READ as u32 | EPOLLET as u32,
            u64: eventfd as u64,
        };

        let ret = unsafe { epoll_ctl(epfd, EPOLL_CTL_ADD, eventfd, &mut ev as *mut libc::epoll_event) };

        if ret == -1 {
            panic!(
                "HiberMgr::Init add eventfd fail, error is {}",
                errno::errno().0
            );
        }

        let uffd = UffdBuilder::new()
            .close_on_exec(true)
            .non_blocking(true)
            .user_mode_only(true)
            .create()
            .expect("uffd creation");

        let mut ev = epoll_event {
            events: EVENT_READ as u32 | EPOLLET as u32,
            u64: uffd.as_raw_fd() as u64,
        };

        let ret = unsafe { epoll_ctl(epfd, EPOLL_CTL_ADD, uffd.as_raw_fd(), &mut ev as *mut libc::epoll_event) };

        if ret == -1 {
            panic!(
                "HiberMgr::Init add uffd fail, error is {}",
                errno::errno().0
            );
        }

        let swapFile = SwapFile::Init()?;

        return Ok(Self {
            epfd: epfd,
            eventfd: eventfd,
            uffd: uffd,
            swapFile: Mutex::new(swapFile),
            map: Mutex::new(HiberMap{
                blockRef: HashMap::new(),
                pageMap: HashMap::new()
            })
        })
    }

    pub fn Process(&self, sharespace: &ShareSpace) {
        let mut events = [epoll_event { events: 0, u64: 0 }; 2];
        let mut data: u64 = 0;
        while !sharespace.Shutdown() {
            self.ProcessUffdEvent().expect("HiberMgr::wakeup fail ...");

            let ret =
                unsafe { libc::read(self.eventfd, &mut data as *mut _ as *mut libc::c_void, 8) };

            if ret < 0 && errno::errno().0 != SysErr::EAGAIN {
                panic!(
                    "HiberMgr::Wakeup fail... eventfd is {}, errno is {}",
                    self.eventfd,
                    errno::errno().0
                );
            }
            let _nfds = unsafe { epoll_wait(self.epfd, &mut events[0], 2, -1) };
        }
    }

    pub fn Wakeup(&self) {
        let val: u64 = 8;
        let ret = unsafe { libc::write(self.eventfd, &val as *const _ as *const libc::c_void, 8) };
        if ret < 0 {
            panic!("HiberMgr::Wakeup fail...");
        }
    }

    pub fn UpdateBlockAllocation(&self, alloctor: &PageBlockAlloc) -> Result<()> {
        let alloctor = alloctor.data.lock();
        let pages = alloctor.GetPages()?;
        let mut map = self.map.lock();
        let mut newBlocks = Vec::new();
        for addr in pages {
            if map.pageMap.contains_key(&addr) {
                continue;
            }
            let offset = self.swapFile.lock().SwapOutPage(addr)?;
            map.pageMap.insert(addr, offset);

            let blockAddr = addr & MemoryDef::PAGE_SIZE_2M_MASK;

            let entry = map
                .blockRef
                .entry(blockAddr);

            match entry {
                Entry::Vacant(e) => {
                    e.insert(1);
                    newBlocks.push(blockAddr);
                }
                Entry::Occupied(mut e) => {
                    *e.get_mut() += 1;
                }
            }
        }

        for blockAddr in newBlocks {
            /*let ret = unsafe {
                libc::madvise(blockAddr as _, MemoryDef::PAGE_SIZE_2M as _, libc::MADV_DONTNEED)
            };

            GetRet(ret as _)?;*/

            // swapout the block idx page
            let offset = self.swapFile.lock().SwapOutPage(blockAddr)?;
            map.pageMap.insert(blockAddr, offset);

            unsafe {
                libc::munmap(blockAddr as _, MemoryDef::PAGE_SIZE_2M as _);
                let ret = libc::mmap(
                    blockAddr as _,
                    MemoryDef::PAGE_SIZE_2M as _,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_FIXED | libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                    -1,
                    0,
                );
                let addr = GetRet(ret as _)?;
                assert!(addr ==  blockAddr);
            }

            self.RegisterRange(blockAddr as _, MemoryDef::PAGE_SIZE_2M as _)?;
        }

        return Ok(())
    }

    pub fn ProcessUffdEvent(&self) -> Result<()> {
        let mut map = self.map.lock();
        loop {
            match self.uffd.read_event().map_err(|e| Error::Common(format!("read_event {:?}", e)))? {
                None => break,
                Some(event) => {
                    match event {
                        userfaultfd::Event::Pagefault { kind: _, rw: _, addr } => {
                            let pageAddr = addr as u64 & !MemoryDef::PAGE_MASK;
                            match map.pageMap.remove(&pageAddr) {
                                None => {
                                    unsafe {
                                        self.
                                        uffd.
                                        zeropage(pageAddr as _, MemoryDef::PAGE_SIZE_4K as _, true).
                                        map_err(|e| Error::Common(format!("zeropage {:?}", e)))?;
                                    }
                                }
                                Some(offset) => {
                                    unsafe {
                                        self.
                                        uffd.
                                        copy(self.swapFile.lock().OffsetToAddr(offset) as _, pageAddr as _, MemoryDef::PAGE_SIZE_4K as _, true).
                                        map_err(|e| Error::Common(format!("zeropage {:?}", e)))?;
                                    }

                                    self.swapFile.lock().FreeSlot(offset)?;

                                    let blockAddr = pageAddr & MemoryDef::PAGE_SIZE_2M_MASK;

                                    let entry = map
                                        .blockRef
                                        .entry(blockAddr);

                                    match entry {
                                        Entry::Vacant(_) => {
                                            panic!("ProcessUffdEvent get none block");
                                        }
                                        Entry::Occupied(mut e) => {
                                            let count = *e.get();
                                            *e.get_mut() = count - 1;
                                            if count == 1 {
                                                e.remove();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        e => {
                            panic!("ProcessUffdEvent get unexpected event {:?}", e);
                        }
                    }
                }
            }
        }

        return Ok(())
    }

    pub fn RegisterRange(&self, start: u64, len: u64) -> Result<()> {
        self.
        uffd.
        register(start as _, len as _).
        map_err(|e| Error::Common(format!("RegisterRange {:?}", e)))?;
        return Ok(())
    }


}

pub fn GetRet(ret: i64) -> Result<u64> {
    if ret == -1 {
        //info!("get error, errno is {}", errno::errno().0);
        return Err(Error::SysError(-errno::errno().0));
    }

    return Ok(ret as u64);
}