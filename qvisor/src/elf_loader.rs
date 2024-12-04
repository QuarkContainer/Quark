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

use std::fs::File;
use std::slice;
use xmas_elf::program::ProgramHeader::Ph64;
use xmas_elf::program::Type;
//use xmas_elf::program::{ProgramIter, SegmentData, Type};
//use xmas_elf::sections::SectionData;
use memmap::Mmap;
use std::os::unix::io::AsRawFd;
use xmas_elf::*;

pub use xmas_elf::header::HeaderPt2;
pub use xmas_elf::program::{Flags, ProgramHeader, ProgramHeader64};
pub use xmas_elf::sections::Rela;
pub use xmas_elf::symbol_table::{Entry, Entry64};
pub use xmas_elf::{P32, P64};

use super::addr::Addr;

//use xmas_elf::dynamic::Tag;
//use xmas_elf::header;
use super::qlib::common::Error;
use super::qlib::common::Result;

use super::memmgr::{MapOption, MappedRegion};
use crate::qlib::kernel::Kernel::IDENTICAL_MAPPING;
pub struct KernelELF {
    pub startAddr: Addr,
    pub endAddr: Addr,
    pub mrs: Vec<MappedRegion>,

    pub vdsoStart: u64,
    pub vdsoLen: u64,
    pub vdsomr: Option<MappedRegion>,
}

impl KernelELF {
    pub fn New() -> Result<Self> {
        return Ok(KernelELF {
            startAddr: Addr(0),
            endAddr: Addr(0),
            mrs: Vec::new(),
            vdsoStart: 0,
            vdsoLen: 0,
            vdsomr: None,
        });
    }

    pub fn StartAddr(&self) -> Addr {
        return self.startAddr;
    }

    pub fn EndAddr(&self) -> Addr {
        return self.endAddr;
    }

    pub fn LoadKernel(&mut self, fileName: &str) -> Result<u64> {
        let f: File =
            File::open(fileName).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        let mmap =
            unsafe { Mmap::map(&f).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))? };
        let fd = f.as_raw_fd();

        let mut startAddr: Addr = Addr(0xfffff_fffff_fffff);
        let mut endAddr: Addr = Addr(0);

        let elfFile = ElfFile::new(&mmap).map_err(Error::ELFLoadError)?;
        info!("elfFile header {:?}", elfFile.header);
        let entry = match &elfFile.header.pt2 {
            HeaderPt2::Header64(pt2) => pt2.entry_point,
            _ => return Err(Error::WrongELFFormat),
        };

        for p in elfFile.program_iter() {
            //todo : add more check
            if let Ph64(header) = p {
                info!("program header: {:?}", header);
                if header.get_type().map_err(Error::ELFLoadError)? == Type::Load {
                    let header_host_virtual_addr = if IDENTICAL_MAPPING.load(std::sync::atomic::Ordering::Acquire) {
                        header.virtual_addr
                    } else {
                        header.virtual_addr + crate::MemoryDef::UNIDENTICAL_MAPPING_OFFSET
                    };
                    let startMem = Addr(header_host_virtual_addr).RoundDown()?;
                    let endMem = Addr(header_host_virtual_addr)
                        .AddLen(header.file_size)?
                        .RoundUp()?;
                    let pageOffset =
                        Addr(header_host_virtual_addr).0 - Addr(header_host_virtual_addr).RoundDown()?.0;
                    let len = Addr(header.file_size).RoundUp()?.0;

                    if startMem.0 < startAddr.0 {
                        startAddr = startMem;
                    }

                    let end = Addr(header_host_virtual_addr)
                        .AddLen(header.mem_size)?
                        .RoundUp()?;
                    if endAddr.0 < endMem.0 {
                        endAddr = end;
                    }
                    info!("Elf: Kernel - start mem {:#x}, len {:#x}, offset {:#x}", startMem.0, len, Addr(header.offset).RoundDown()?.0);
                    let mut option = &mut MapOption::New();
                    option = option
                        .Addr(startMem.0)
                        .Len(len)
                        .FileId(fd)
                        .MapFixed()
                        .FileOffset(Addr(header.offset).RoundDown()?.0)
                        .MapPrivate()
                        .ProtoRead()
                        .ProtoWrite();

                    let mr = option.Map()?;
                    assert!(mr.ptr == startMem.0 + pageOffset);
                    self.mrs.push(mr);

                    let adjust = header_host_virtual_addr - startMem.0;

                    if adjust + header.file_size < endMem.0 - startMem.0 {
                        let cnt = (endMem.0 - startMem.0 - (adjust + header.file_size)) as usize;
                        let target = unsafe {
                            slice::from_raw_parts_mut(
                                (startMem.0 + adjust + header.file_size) as *mut u8,
                                cnt,
                            )
                        };

                        for i in 0..cnt {
                            target[i] = 0;
                        }
                    }

                    if header.mem_size > header.file_size {
                        let bssEnd = Addr(header_host_virtual_addr + header.mem_size).RoundUp()?;
                        if bssEnd.0 != endMem.0 {
                            info!("Elf: Kernel - bss - start mem {:#x}, len {:#x}",
                                endMem.0, bssEnd.0 - endMem.0);
                            let mut option = &mut MapOption::New();
                            option = option
                                .Addr(endMem.0)
                                .Len(bssEnd.0 - endMem.0)
                                .MapAnan()
                                .MapPrivate()
                                .ProtoRead()
                                .ProtoWrite();

                            let mr = option.Map()?;
                            assert!(mr.ptr == endMem.0);
                            self.mrs.push(mr);
                        }
                    }
                }
            }
        }

        self.startAddr = startAddr;
        self.endAddr = endAddr;

        return Ok(entry);
    }

    pub fn LoadVDSO(&mut self, fileName: &str) -> Result<()> {
        let f =
            File::open(fileName).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        let mmap =
            unsafe { Mmap::map(&f).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))? };
        let len = mmap.len();

        assert!(Addr(len as u64).RoundUp()?.0 == 2 * 4096);

        let mut option = &mut MapOption::New();
        option = option
            .Addr(self.EndAddr().0)
            .Len(3 * 4096)
            .MapAnan()
            .MapPrivate()
            .ProtoRead()
            .ProtoWrite()
            .ProtoExec();
        let mr = option.Map()?;
        //let mr = MappedRegion::Init(self.startAddr, self.endAddr.0 - self.startAddr.0, false, libc::PROT_READ |  libc::PROT_WRITE |  libc::PROT_EXEC)?;
        let hostAddr = mr.ptr as u64;
        if hostAddr != self.EndAddr().0 {
            return Err(Error::AddressDoesMatch);
        }

        let target = unsafe { slice::from_raw_parts_mut((hostAddr + 4096) as *mut u8, len) };
        let source = &mmap[..];
        target.clone_from_slice(source);

        self.vdsoStart = hostAddr;
        self.vdsoLen = 3 * 4096;
        self.vdsomr = Some(mr);

        return Ok(());
    }
}
