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
use std::collections::VecDeque;

use super::super::qlib::mem::areaset::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::range::*;
use super::super::memmgr::*;
use super::super::IO_MGR;
use super::super::heap_alloc::ENABLE_HUGEPAGE;

#[derive(Clone, Default)]
pub struct HostSegment {}

impl AreaValue for HostSegment {
    fn Merge(&self, _r1: &Range, _r2: &Range, _vma2: &HostSegment) -> Option<HostSegment> {
        return Some(HostSegment {})
    }

    fn Split(&self, _r: &Range, _split: u64) -> (HostSegment, HostSegment) {
        return (HostSegment {}, HostSegment {})
    }
}

//pub struct HostPMAKeeper (Mutex<AreaSet<HostSegment>>);

pub struct HostPMAKeeper {
    pub ranges: Mutex<AreaSet<HostSegment>>,
    pub hugePages: Mutex<VecDeque<u64>>,
}

impl HostPMAKeeper {
    pub fn New() -> Self {
        return Self {
            ranges: Mutex::new(AreaSet::New(0,0)),
            hugePages: Mutex::new(VecDeque::with_capacity(1000)),
        }
    }

    pub fn FreeHugePage(&self, addr: u64) {
        self.hugePages.lock().push_front(addr);
    }

    pub fn AllocHugePage(&self) -> Option<u64> {
        let ret = self.hugePages.lock().pop_back();
        return ret;
    }

    pub fn Init(&self, start: u64, len: u64) {
        self.ranges.lock().Reset(start, len);
    }

    pub fn InitHugePages(&self) {
        let hugeLen = (self.ranges.lock().range.Len() / MemoryDef::ONE_GB - 2) * MemoryDef::ONE_GB;
        //error!("InitHugePages is {:x}", self.ranges.lock().range.Len() / MemoryDef::ONE_GB - 2);
        let hugePageStart = self.RangeAllocate(hugeLen, MemoryDef::PAGE_SIZE_2M).unwrap();
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
                    panic!("AreaSet <HostSegment>:: memmap fail to alloc fix address at {:x}", r.Start());
                }

                return Ok(r.Start())
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
        let osfd = IO_MGR.GetFdByHost(fd).expect("MapFile: Getosfd fail");
        let mut mo = &mut MapOption::New();

        //let prot = prot | MmapProt::PROT_WRITE as i32;

        mo = mo.Proto(prot).FileOffset(offset).FileId(osfd).Len(len).MapFixed();
        //mo.MapPrivate();
        mo.MapShare();
        mo.MapLocked();

        let start = self.Allocate(len, MemoryDef::PMD_SIZE)?;
        mo.Addr(start);
        return self.Map(&mut mo, &Range::New(start, len));
    }

    fn RangeAllocate(&self, len: u64, alignment: u64) -> Result<u64> {
        let mut ranges = self.ranges.lock();
        let start = ranges.FindAvailable(len, alignment)?;

        let r = Range::New(start, len);
        let gap = ranges.FindGap(start);
        let seg = ranges.Insert(&gap, &r, HostSegment {});
        assert!(seg.Ok(), "AreaSet <HostSegment>:: insert fail");

        return Ok(start)
    }

    fn Allocate(&self, len: u64, alignment: u64) -> Result<u64> {
        if len != MemoryDef::PAGE_SIZE_2M {
            error!("Allocate len is {:x} alignment {:x}", len, alignment);
        }

        if len <= MemoryDef::PAGE_SIZE_2M {
            assert!(alignment <= MemoryDef::PAGE_SIZE_2M, "Allocate fail .... {:x}/{:x}", len, alignment);
            let addr = match self.AllocHugePage() {
                None => {
                    error!("AllocHugePage fail...");
                    panic!("AllocHugePage fail...");
                }
                Some(addr) => addr
            };
            return Ok(addr)
        }

        return self.RangeAllocate(len, alignment);
    }

    pub fn RemoveSeg(&self, r: &Range) {
        if r.Len() <= MemoryDef::PAGE_SIZE_2M {
            self.FreeHugePage(r.Start());
            return;
        }

        let mut ranges = self.ranges.lock();
        let (seg, _gap) = ranges.Find(r.Start());

        if !seg.Ok() || !seg.Range().IsSupersetOf(r) {
            panic!("AreaSet <HostSegment>::Unmap invalid, remove range {:?} from range {:?}",
                   r, seg.Range());
        }

        let seg = ranges.Isolate(&seg, r);

        ranges.Remove(&seg);
    }

    pub fn Unmap(&self, r: &Range) -> Result<()> {
        self.RemoveSeg(r);

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
                        return Ok(gr.Start() + alignment - offset)
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
