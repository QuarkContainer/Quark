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

use alloc::collections::BTreeSet;
use spin::Mutex;
use std::collections::VecDeque;

use super::super::heap_alloc::ENABLE_HUGEPAGE;
use super::super::memmgr::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::mem::areaset::*;
use super::super::qlib::range::*;

#[derive(Clone, Default)]
pub struct HostSegment {}

impl AreaValue for HostSegment {
    fn Merge(&self, _r1: &Range, _r2: &Range, _vma2: &HostSegment) -> Option<HostSegment> {
        return Some(HostSegment {});
    }

    fn Split(&self, _r: &Range, _split: u64) -> (HostSegment, HostSegment) {
        return (HostSegment {}, HostSegment {});
    }
}

//pub struct HostPMAKeeper (Mutex<AreaSet<HostSegment>>);

pub struct HostPMAKeeper {
    pub ranges: Mutex<AreaSet<HostSegment>>,
    pub hugePages: Mutex<VecDeque<u64>>,
    pub allocPages: Mutex<BTreeSet<u64>>,
}

impl HostPMAKeeper {
    pub fn New() -> Self {
        return Self {
            ranges: Mutex::new(AreaSet::New(0, 0)),
            hugePages: Mutex::new(VecDeque::with_capacity(1000)),
            allocPages: Mutex::new(BTreeSet::new()),
        };
    }

    pub fn FreeHugePage(&self, addr: u64) {
        self.hugePages.lock().push_front(addr);
        self.allocPages.lock().remove(&addr);
    }

    pub fn AllocHugePage(&self) -> Option<u64> {
        let ret = self.hugePages.lock().pop_back();
        match ret {
            None => return None,
            Some(addr) => {
                self.allocPages.lock().insert(addr);
                return Some(addr);
            }
        }
    }

    pub fn DontNeed(&self) -> Result<()> {
        let alloced = self.allocPages.lock();
        for page in alloced.iter() {
            let ret = unsafe {
                libc::madvise(
                    (*page) as _,
                    MemoryDef::PAGE_SIZE_2M as _,
                    libc::MADV_DONTNEED,
                )
            };

            if ret == -1 {
                info!(
                    "DontNeed get error, address is {:x} errno is {}",
                    *page,
                    errno::errno().0
                );
                //return Err(Error::SysError(-errno::errno().0));
            }
        }

        return Ok(());
    }

    pub fn Init(&self, start: u64, len: u64) {
        self.ranges.lock().Reset(start, len);
    }

    pub fn InitHugePages(&self) {
        let hugeLen = (self.ranges.lock().range.Len() / MemoryDef::ONE_GB - 2) * MemoryDef::ONE_GB;
        //error!("InitHugePages is {:x}", self.ranges.lock().range.Len() / MemoryDef::ONE_GB - 2);
        let hugePageStart = self
            .RangeAllocate(hugeLen, MemoryDef::PAGE_SIZE_2M)
            .unwrap();
        let mut addr = hugePageStart;
        while addr < hugePageStart + hugeLen {
            self.FreeHugePage(addr);
            addr += MemoryDef::PAGE_SIZE_2M;
        }
    }

    fn Map(&self, mo: &mut MapOption, r: &Range) -> Result<u64> {
        match mo.MMap() {
            Err(e) => {
                self.RemoveSeg(r);
                return Err(e);
            }
            Ok(addr) => {
                if addr != r.Start() {
                    panic!(
                        "AreaSet <HostSegment>:: memmap fail to alloc fix address at {:x}",
                        r.Start()
                    );
                }

                return Ok(r.Start());
            }
        }
    }

    pub fn MapHugePage(&self) -> Result<u64> {
        let mut mo = &mut MapOption::New();
        let prot = libc::PROT_READ | libc::PROT_WRITE;
        let len = MemoryDef::PAGE_SIZE_2M;
        mo = mo.MapAnan().Proto(prot).Len(len);
        if ENABLE_HUGEPAGE {
            mo.MapHugeTLB();
        }

        let start = self.Allocate(len, MemoryDef::PAGE_SIZE_2M)?;
        mo.Addr(start);
        return self.Map(&mut mo, &Range::New(start, len));
    }

    pub fn MapAnon(&self, len: u64, prot: i32) -> Result<u64> {
        let mut mo = &mut MapOption::New();
        mo = mo.MapAnan().Proto(prot).Len(len);
        mo.MapShare();

        let start = self.Allocate(len, MemoryDef::PAGE_SIZE)?;
        mo.Addr(start);
        return self.Map(&mut mo, &Range::New(start, len));
    }

    pub fn MapFile(&self, len: u64, prot: i32, fd: i32, offset: u64) -> Result<u64> {
        let osfd = fd;
        let mut mo = &mut MapOption::New();

        mo = mo
            .Proto(prot)
            .FileOffset(offset)
            .FileId(osfd)
            .Len(len)
            .MapFixed();

        mo.MapShare();
        // mo.MapPrecommit();

        //mo.MapLocked();

        let start = self.Allocate(len, MemoryDef::PMD_SIZE)?;
        mo.Addr(start);
        let ret = self.Map(&mut mo, &Range::New(start, len));
        return ret;
    }

    fn RangeAllocate(&self, len: u64, alignment: u64) -> Result<u64> {
        let mut ranges = self.ranges.lock();
        let start = ranges.FindAvailable(len, alignment)?;

        let r = Range::New(start, len);
        let gap = ranges.FindGap(start);
        let seg = ranges.Insert(&gap, &r, HostSegment {});
        assert!(seg.Ok(), "AreaSet <HostSegment>:: insert fail");

        return Ok(start);
    }

    fn Allocate(&self, len: u64, alignment: u64) -> Result<u64> {
        assert!(len == MemoryDef::PAGE_SIZE_2M);
        assert!(alignment == MemoryDef::PAGE_SIZE_2M);
        let addr = match self.AllocHugePage() {
            None => {
                error!("AllocHugePage fail...");
                panic!("AllocHugePage fail...");
            }
            Some(addr) => addr,
        };
        assert!(addr & (MemoryDef::PAGE_SIZE_2M - 1) == 0);
        return Ok(addr);
    }

    pub fn RemoveSeg(&self, r: &Range) {
        if r.Len() <= MemoryDef::PAGE_SIZE_2M {
            self.FreeHugePage(r.Start());
            return;
        }

        let mut ranges = self.ranges.lock();
        let (seg, _gap) = ranges.Find(r.Start());

        if !seg.Ok() || !seg.Range().IsSupersetOf(r) {
            panic!(
                "AreaSet <HostSegment>::Unmap invalid, remove range {:?} from range {:?}",
                r,
                seg.Range()
            );
        }

        let seg = ranges.Isolate(&seg, r);

        ranges.Remove(&seg);
    }

    pub fn Unmap(&self, r: &Range) -> Result<()> {
        //self.RemoveSeg(r);

        assert!(r.Start() % MemoryDef::PAGE_SIZE_2M == 0);
        assert!(r.Len() == MemoryDef::PAGE_SIZE_2M);

        self.FreeHugePage(r.Start());

        let res = MapOption::MUnmap(r.Start(), r.Len());
        return res;
    }
}

impl AreaSet<HostSegment> {
    fn FindAvailable(&mut self, len: u64, alignment: u64) -> Result<u64> {
        let mut gap = self.FirstGap();

        while gap.Ok() {
            let gr = gap.Range();
            if gr.Len() >= len {
                let offset = gr.Start() % alignment;
                if offset != 0 {
                    if gr.Len() >= len + alignment - offset {
                        return Ok(gr.Start() + alignment - offset);
                    }
                } else {
                    return Ok(gr.Start());
                }
            }

            gap = gap.NextGap();
        }

        return Err(Error::SysError(SysErr::ENOMEM));
    }
}
