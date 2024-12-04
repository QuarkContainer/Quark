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

use alloc::collections::BTreeSet;
use core::hint::spin_loop;
use libc::*;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::IntoRawFd;
use std::sync::atomic::Ordering;

use crate::qlib::common::*;
use crate::qlib::hiber_mgr::*;
use crate::qlib::linux_def::IoVec;
use crate::qlib::linux_def::*;
use crate::qlib::mem::block_allocator::*;
use crate::qlib::mem::buddy_allocator::*;
use crate::vmspace::kernel::Timestamp;
use crate::vmspace::kernel::SHARESPACE;
use crate::GLOBAL_ALLOCATOR;
use crate::SHARE_SPACE;
use crate::SWAP_FILE;

impl<const ORDER: usize> Heap<ORDER> {
    pub fn DontNeed(&self) {
        // DO NOT ADD any debug print, it require heap lock
        let mut arr: [usize; 1024] = [0; 1024];
        // let mut total = 0;
        for i in 13..ORDER {
            let count = self.free_list[i].GetVals(&mut arr);
            // error!("heap order {} addresses {:x?}", i, &arr[0..count]);

            for addr in &arr[0..count] {
                let pageCount = 1 << (i - 12);
                let _ret = unsafe {
                    libc::madvise(
                        (*addr + MemoryDef::PAGE_SIZE_4K as usize) as _,
                        (pageCount - 1) * MemoryDef::PAGE_SIZE_4K as usize,
                        libc::MADV_DONTNEED,
                    )
                };
            }
            // total += count;
        }

        // error!("DontNeed total {}", total);
    }
}

impl HiberMgr {
    pub fn GetAllPagetablePages(&self) -> Result<BTreeSet<u64>> {
        let intern = self.lock();
        let mut map = BTreeSet::new();
        for (_, mm) in &intern.memmgrs {
            let mm = mm.Upgrade();
            mm.pagetable
                .read()
                .pt
                .GetAllPagetablePages(&mut map)
                .unwrap();
        }

        //info!("GetAllPagetablePages {} pages, {:x?}", map.len(), map);
        return Ok(map);
    }

    pub fn SwapOutUserPages(&self, start: u64, len: u64) -> Result<()> {
        let mut intern = self.lock();
        let mut map = BTreeSet::new();
        for (_, mm) in &intern.memmgrs {
            let mm = mm.Upgrade();
            mm.pagetable
                .write()
                .pt
                .SwapOutPages(start, len, &mut map, true)
                .unwrap();
        }

        let mut insertCount = 0;
        for page in map.iter() {
            let offset = SWAP_FILE.lock().SwapOutPage(*page)?;
            match intern.pageMap.insert(*page, offset) {
                None => {
                    insertCount += 1;
                }
                Some(offset) => {
                    // the page has been freed when it is swapped out
                    SWAP_FILE.lock().DropPage(offset)?;
                }
            }
            intern.pageMap.entry(*page).or_insert_with(|| {
                insertCount += 1;
                offset
            });
        }

        info!(
            "swapout {} pages, new pages {} pages",
            map.len(),
            insertCount
        );
        return Ok(());
    }

    pub fn ReapSwapOut(&self, start: u64, len: u64, map: &mut BTreeSet<u64>) -> Result<()> {
        let mut intern = self.lock();
        for (_, mm) in &intern.memmgrs {
            let mm = mm.Upgrade();
            mm.pagetable
                .write()
                .pt
                .SwapOutPages(start, len, map, false)
                .unwrap();
        }

        for page in map.iter() {
            intern.reapSwapFile.PushAddr(*page);
        }

        intern.reapSwapFile.SwapOut();

        SHARE_SPACE.reapFileAvaiable.store(true, Ordering::SeqCst);
        info!("ReapSwapOut {} pages", map.len());
        return Ok(());
    }

    pub fn ReapSwapIn(&self) -> Result<()> {
        let mut intern = self.lock();
        let now = Timestamp();
        let count = intern.reapSwapFile.iovs.len();
        intern.reapSwapFile.SwapIn();
        error!("ReapSwapIn pages {} in {}", count, Timestamp() - now);
        SHARE_SPACE.reapFileAvaiable.store(false, Ordering::SeqCst);

        return Ok(());
    }

    pub fn SwapOut(&self, start: u64, len: u64) -> Result<()> {
        // error!(
        //     "SwapOut 1 {} {} {:b}",
        //     SHARE_SPACE.scheduler.HaltVcpuCnt(),
        //     SHARE_SPACE.scheduler.vcpuCnt,
        //     SHARE_SPACE.scheduler.vcpuWaitMask.load(Ordering::SeqCst)
        // );
        while !SHARE_SPACE.scheduler.ReadyForHibernate() {
            // wait until all other vcpu halt
            for _ in 0..100 {
                spin_loop();
            }
        }

        if !self.lock().reap {
            self.SwapOutUserPages(start, len)?;
            self.lock().reap = true
        } else {
            let mut map = self.GetAllPagetablePages()?;
            self.ReapSwapOut(start, len, &mut map)?;
        }

        let cnt = SHARE_SPACE.pageMgr.pagepool.DontneedFreePages()?;

        crate::PMA_KEEPER.DontNeed()?;

        //SHARE_SPACE.scheduler.CleanVcpPageCache();
        let allocated1 = GLOBAL_ALLOCATOR.GuestHostSharedAllocator().heap.lock().allocated;

        GLOBAL_ALLOCATOR.GuestHostSharedAllocator().FreeAll();

        let allocated2 = GLOBAL_ALLOCATOR.GuestHostSharedAllocator().heap.lock().allocated;

        info!(
            "free pagepool {} pages, total allocated1 {} allocated2 {} free bytes {}",
            cnt,
            allocated1,
            allocated2,
            allocated1 - allocated2
        );
        //info!("heap usage1 is {:?}", &GLOBAL_ALLOCATOR.Allocator().counts);
        /*for i in 3..20 {
            info!("heap usage2 is {}/{:x}/{:?}/{:?}", i, 1<<i, GLOBAL_ALLOCATOR.Allocator().counts[i], GLOBAL_ALLOCATOR.Allocator().maxnum[i]);
        }
        info!("heap usage3 is {:?}", &GLOBAL_ALLOCATOR.Allocator().maxnum);*/

        GLOBAL_ALLOCATOR.GuestHostSharedAllocator().heap.lock().DontNeed();
        error!("swap out done ...");
        return Ok(());
    }

    pub fn SwapIn(&self, phyAddr: u64) -> Result<()> {
        let mut intern = self.lock();

        match intern.pageMap.remove(&phyAddr) {
            None => return Err(Error::SysError(SysErr::EINVAL)),
            Some(offset) => return SWAP_FILE.lock().SwapInPage(phyAddr, offset),
        }
    }
}

pub const REAP_SWAP_FILE_NAME: &str = "./reap_swapfile.data";

impl ReapSwapFile {
    pub fn Init(&mut self) {
        let direct = SHARESPACE.config.read().HiberODirect;

        let file = if direct {
            OpenOptions::new()
                .read(true)
                .write(true)
                .custom_flags(O_DIRECT)
                .create(true)
                .open(REAP_SWAP_FILE_NAME)
                .unwrap()
        } else {
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(REAP_SWAP_FILE_NAME)
                .unwrap()
        };
        let fd = file.into_raw_fd();

        self.fd = fd;
    }

    pub fn PushAddr(&mut self, addr: u64) {
        self.iovs.push(IoVec {
            start: addr,
            len: 4096,
        });
    }

    pub fn SwapOut(&mut self) {
        if self.fd == 0 {
            self.Init();
        }

        let mut idx = 0;
        while idx < self.iovs.len() {
            let count = if self.iovs.len() - idx <= 1024 {
                self.iovs.len() - idx
            } else {
                1024
            };

            let ret = unsafe {
                libc::pwritev(
                    self.fd,
                    &self.iovs[idx] as *const _ as u64 as _,
                    count as _,
                    idx as i64 * 4096,
                )
            };

            assert!(
                ret as usize == count * 4096,
                "ret is {:?}, count is {}",
                GetRet(ret as _),
                count
            );

            idx += count;
        }

        for iov in &self.iovs {
            let ret = unsafe {
                libc::madvise(
                    iov.start as _,
                    MemoryDef::PAGE_SIZE_4K as _,
                    libc::MADV_DONTNEED,
                )
            };

            assert!(ret == 0);
        }
    }

    pub fn SwapIn(&mut self) {
        let mut idx = 0;
        while idx < self.iovs.len() {
            let count = if self.iovs.len() - idx <= 1024 {
                self.iovs.len() - idx
            } else {
                1024
            };

            let ret = unsafe {
                libc::preadv(
                    self.fd,
                    &self.iovs[idx] as *const _ as u64 as _,
                    count as _,
                    idx as i64 * 4096,
                )
            };

            assert!(ret as usize == count * 4096);

            idx += count;
        }

        self.iovs.clear();
    }
}

pub struct SwapFile {
    pub fd: i32,              // the file fd
    pub size: u64,            //total allocated file size
    pub nextAllocOffset: u64, // the last allocated slot
    pub freeSlots: Vec<u64>,  // free page slot
    pub mmapAddr: u64,        // the file mmap start address
}

pub const SWAP_FILE_NAME: &str = "./swapfile.data";
pub const INIT_FILE_SIZE: u64 = MemoryDef::PAGE_SIZE_2M;
pub const EXTEND_FILE_SIZE: u64 = MemoryDef::PAGE_SIZE_2M;

impl SwapFile {
    pub fn Init() -> Result<Self> {
        let direct = SHARESPACE.config.read().HiberODirect;

        let file = if direct {
            OpenOptions::new()
                .read(true)
                .write(true)
                .custom_flags(O_DIRECT)
                .create(true)
                .open(SWAP_FILE_NAME)
                .unwrap()
        } else {
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(SWAP_FILE_NAME)
                .unwrap()
        };
        let fd = file.into_raw_fd();

        let ret = unsafe { libc::ftruncate(fd, INIT_FILE_SIZE as _) };

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
        });
    }

    pub fn ReadAhead(&self) {
        let _ret = unsafe {
            libc::readahead(
                self.fd,
                0,
                (self.nextAllocOffset * MemoryDef::PAGE_SIZE_4K) as usize,
            )
        };
    }

    pub fn ExtendSize(&mut self) -> Result<()> {
        let newSize = self.size + EXTEND_FILE_SIZE;
        let ret = unsafe { libc::ftruncate(self.fd, newSize as _) };

        GetRet(ret as _)?;

        /*let ret = unsafe {
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
        assert!(addr == self.mmapAddr + self.size);*/

        self.size = newSize;
        return Ok(());
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
        return Ok(());
    }

    // input: memory page address
    // ret: file offset
    pub fn SwapOutPage(&mut self, addr: u64) -> Result<u64> {
        let offset = self.AllocateSlot()?;
        let ret = unsafe {
            libc::pwrite(
                self.fd,
                addr as _,
                MemoryDef::PAGE_SIZE_4K as _,
                offset as _,
            )
        };

        let count = GetRet(ret as _)?;
        assert!(count == MemoryDef::PAGE_SIZE_4K);

        let ret =
            unsafe { libc::madvise(addr as _, MemoryDef::PAGE_SIZE_4K as _, libc::MADV_DONTNEED) };

        GetRet(ret as _)?;

        return Ok(offset);
    }

    pub fn DropPage(&mut self, offset: u64) -> Result<()> {
        self.FreeSlot(offset)?;

        return Ok(());
    }

    // input: memory page address, file offset
    // ret: file offset
    pub fn SwapInPage(&mut self, addr: u64, offset: u64) -> Result<()> {
        let ret = unsafe {
            libc::pread(
                self.fd,
                addr as _,
                MemoryDef::PAGE_SIZE_4K as _,
                offset as _,
            )
        };

        let count = GetRet(ret as _)?;
        assert!(count == MemoryDef::PAGE_SIZE_4K);
        self.FreeSlot(offset)?;

        return Ok(());
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
            return Ok(0);
        }

        for idx in 1..BLOCK_PAGE_COUNT + 1 {
            if alloc.freePageList.IsFree(idx as _) {
                let addr = self.IdxToAddr(idx as _);
                let ret = unsafe {
                    libc::madvise(addr as _, MemoryDef::PAGE_SIZE_4K as _, libc::MADV_DONTNEED)
                };

                GetRet(ret as _)?;
            }
        }

        return Ok(alloc.freePageList.totalFreeCount);
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

        return Ok(total);
    }
}

pub struct HiberMap {
    pub blockRef: HashMap<u64, u32>, // blockAddr --> refcnt
    pub pageMap: HashMap<u64, u64>,  // pageAddr --> file offset
}

pub fn GetRet(ret: i64) -> Result<u64> {
    if ret == -1 {
        //info!("get error, errno is {}", errno::errno().0);
        return Err(Error::SysError(-errno::errno().0));
    }

    return Ok(ret as u64);
}
