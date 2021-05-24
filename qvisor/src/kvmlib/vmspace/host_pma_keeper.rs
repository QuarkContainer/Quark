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
use core::ops::Deref;

use super::super::qlib::mem::areaset::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::range::*;
use super::super::memmgr::*;
use super::super::IO_MGR;

pub struct HostPMAKeeper (Mutex<AreaSet<HostSegment>>);

impl Deref for HostPMAKeeper {
    type Target = Mutex<AreaSet<HostSegment>>;

    fn deref(&self) -> &Mutex<AreaSet<HostSegment>> {
        &self.0
    }
}

impl HostPMAKeeper {
    pub fn New() -> Self {
        return Self(Mutex::new(AreaSet::New(0,0)))
    }
}

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

impl AreaSet<HostSegment> {
    fn Map(&mut self, mo: &mut MapOption, r: &Range) -> Result<u64> {
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

    pub fn MapAnon(&mut self, len: u64, prot: i32) -> Result<u64> {
        let mut mo = &mut MapOption::New();
        mo = mo.MapAnan().Proto(prot).Len(len);
        mo.MapShare();

        let start = self.Allocate(len, MemoryDef::PAGE_SIZE)?;
        mo.Addr(start);
        return self.Map(&mut mo, &Range::New(start, len));
    }

    pub fn MapFile(&mut self, len: u64, prot: i32, fd: i32, offset: u64) -> Result<u64> {
        let osfd = IO_MGR.lock().GetFdByHost(fd).expect("MapFile: Getosfd fail");
        let mut mo = &mut MapOption::New();

        //let prot = prot | MmapProt::PROT_WRITE as i32;

        mo = mo.Proto(prot).FileOffset(offset).FileId(osfd).Len(len).MapFixed();
        //mo.MapPrivate();
        mo.MapShare();
        mo.MapLocked();

        let start = self.Allocate(len, MemoryDef::PAGE_SIZE)?;
        mo.Addr(start);
        return self.Map(&mut mo, &Range::New(start, len));
    }

    fn Allocate(&mut self, len: u64, alignment: u64) -> Result<u64> {
        let start = self.FindAvailable(len, alignment)?;

        let r = Range::New(start, len);
        let gap = self.FindGap(start);
        let seg = self.Insert(&gap, &r, HostSegment {});
        assert!(seg.Ok(), "AreaSet <HostSegment>:: insert fail");

        return Ok(start)
    }

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

    pub fn RemoveSeg(&mut self, r: &Range) {
        let (seg, _gap) = self.Find(r.Start());

        if !seg.Ok() || !seg.Range().IsSupersetOf(r) {
            panic!("AreaSet <HostSegment>::Unmap invalid, remove range {:?} from range {:?}",
                   r, seg.Range());
        }

        let seg = self.Isolate(&seg, r);

        self.Remove(&seg);
    }

    pub fn Unmap(&mut self, r: &Range) -> Result<()> {
        self.RemoveSeg(r);

        let res = MapOption::MUnmap(r.Start(), r.Len());
        return res;
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn TestPMA() {
        let mut pmaKeeper: AreaSet<HostSegment> = AreaSet::New(0, 64);

        assert_eq!(pmaKeeper.Allocate(2, 16).unwrap(), 0);

        let (seg, _gap) = pmaKeeper.Find(0);
        assert_eq!(seg.Range(), Range::New(0, 2));
        let gap = seg.NextGap();
        assert_eq!(gap.Range(), Range::New(2, 62));

        assert_eq!(pmaKeeper.Allocate(8, 16).unwrap(), 16);

        pmaKeeper.RemoveSeg(&Range::New(20, 2));
        let seg = pmaKeeper.FindSeg(0);
        assert_eq!(seg.Range(), Range::New(0, 2));
        let seg = seg.NextSeg();
        assert_eq!(seg.Range(), Range::New(16, 4));
        let seg = seg.NextSeg();
        assert_eq!(seg.Range(), Range::New(22, 2));

        assert_eq!(pmaKeeper.Allocate(14, 1).unwrap(), 2);
        let seg = pmaKeeper.FindSeg(0);
        assert_eq!(seg.Range(), Range::New(0, 20));
        let seg = seg.NextSeg();
        assert_eq!(seg.Range(), Range::New(22, 2));
    }
}