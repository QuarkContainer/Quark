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

use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
pub use xmas_elf::header::HeaderPt2;
use xmas_elf::program::ProgramHeader::Ph64;
use xmas_elf::program::ProgramHeader64;
use xmas_elf::program::Type;
use xmas_elf::*;

use super::super::super::addr::*;
use super::super::super::auxv::*;
use super::super::super::common::*;
use super::super::super::limits::*;
use super::super::super::linux_def::*;
use super::super::super::platform::defs_impl::*;
use super::super::arch::__arch::context::Context64;
use super::super::fs::file::*;
use super::super::memmgr::*;
use super::super::task::*;
use super::super::util::cstring::*;

pub const ELF_MAGIC: &str = "\x7fELF";
pub const INTERPRETER_SCRIPT_MAGIC: &str = "#!";

pub type OS = i32;
pub const LINUX_OS: OS = 0;

pub type Arch = i32;
pub const AMD64: Arch = 0;

// elfInfo contains the metadata needed to load an ELF binary.
pub struct ElfHeadersInfo {
    // os is the target OS of the ELF.
    pub os: OS,

    // arch is the target architecture of the ELF.
    pub arch: Arch,

    // entry is the program entry point.
    pub entry: u64,

    // phdrAddr is the offset of the program headers in the file.
    pub phdrAddr: u64,

    // phdrSize is the size of a single program header in the ELF.
    pub phdrSize: usize,

    // phdrNum is the program header count
    pub phdrNum: usize,

    // phdrs are the program headers.
    pub phdrs: Vec<ProgramHeader64>,

    // sharedObject is true if the ELF represents a shared object.
    pub sharedObject: bool,
}

// parseHeader parse the ELF header, verifying that this is a supported ELF
// file and returning the ELF program headers.
pub fn ParseHeader(task: &mut Task, file: &File) -> Result<ElfHeadersInfo> {
    let mut buf = DataBuff::New(2 * 0x1000);
    let n = match ReadAll(task, &file, &mut buf.buf, 0) {
        Err(e) => {
            print!("Error ParseHeader {:?}", e);
            return Err(Error::SysError(SysErr::ENOEXEC));
        }
        Ok(n) => n,
    };

    let elfFile = ElfFile::new(&buf.buf[0..n])
        .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

    let phdrAddr;
    let phdrSize;
    let phdrNum;

    let isSharedObject = match &elfFile.header.pt2 {
        HeaderPt2::Header64(pt2) => {
            phdrAddr = pt2.ph_offset;
            phdrSize = pt2.ph_entry_size;
            phdrNum = pt2.ph_count;
            match pt2.type_.as_type() {
                xmas_elf::header::Type::SharedObject => true,
                xmas_elf::header::Type::Executable => false,
                _ => return Err(Error::WrongELFFormat),
            }
        }
        _ => return Err(Error::WrongELFFormat),
    };

    let entry = match &elfFile.header.pt2 {
        HeaderPt2::Header64(pt2) => pt2.entry_point,
        _ => return Err(Error::WrongELFFormat),
    };

    let mut phdrs = Vec::new();

    for p in elfFile.program_iter() {
        if let Ph64(header) = p {
            let headerType = header.get_type().map_err(Error::ELFLoadError)?;
            if headerType == Type::Interp || headerType == Type::Load {
                phdrs.push(*header);
            }
        }
    }

    return Ok(ElfHeadersInfo {
        os: LINUX_OS,
        arch: AMD64,
        entry: entry,
        phdrAddr: phdrAddr,
        phdrSize: phdrSize as usize,
        phdrNum: phdrNum as usize,
        phdrs: phdrs,
        sharedObject: isSharedObject,
    });
}

pub fn PHFlagsAsPerms(header: &ProgramHeader64) -> AccessType {
    let flags = header.flags;
    let mut perms = 0;

    if flags.is_execute() {
        perms |= MmapProt::PROT_EXEC;
    }

    if flags.is_write() {
        perms |= MmapProt::PROT_WRITE;
    }

    if flags.is_read() {
        perms |= MmapProt::PROT_READ;
    }

    return AccessType(perms);
}

pub fn MapSegment(
    task: &Task,
    file: &File,
    header: &ProgramHeader64,
    offset: u64,
    filesize: i64,
) -> Result<()> {
    let size = header.file_size;
    let startMem = Addr(header.virtual_addr).RoundDown()?;
    let endMem = Addr(header.virtual_addr)
        .AddLen(header.file_size)?
        .RoundUp()?;

    let fileOffset = Addr(header.offset).RoundDown()?;
    //info!("MapSegment fileoffset is {:x}, size is {:x}, filesize is {:x}", fileOffset.0, size, filesize);
    if fileOffset.0 + size > filesize as u64 {
        return Err(Error::SysError(SysErr::ENOEXEC));
    }

    let addr = if endMem.0 - startMem.0 == 0 {
        offset + startMem.0
    } else {
        //info!("virtual address is {:x}, fileoffset is {:x}, delta is {:x}, offset is {:x}, len is {:x}",
        //    header.virtual_addr, header.offset, header.virtual_addr - header.offset, offset, endMem.0 - startMem.0);
        let mut moptions = MMapOpts::NewFileOptions(file)?;
        moptions.Length = endMem.0 - startMem.0;
        moptions.Addr = offset + startMem.0;
        moptions.Fixed = true;
        moptions.Perms = AccessType(PHFlagsAsPerms(header).0);
        moptions.MaxPerms = AccessType::AnyAccess();
        moptions.Private = true;
        moptions.Offset = fileOffset.0;

        task.mm.MMap(task, &mut moptions)?
    };

    let adjust = header.virtual_addr - startMem.0;

    if adjust + header.file_size < endMem.0 - startMem.0 {
        let cnt = (endMem.0 - startMem.0 - (adjust + header.file_size)) as usize;
        let buf: [u8; 4096] = [0; 4096];
        let vaddr = addr + adjust + header.file_size;
        task.mm
            .MProtect(
                Addr(vaddr).RoundDown()?.0,
                4096,
                &AccessType::AnyAccess(),
                false,
            )
            .unwrap();
        task.CopyOutSlice(&buf[0..cnt], vaddr, cnt)?;
    }

    if header.mem_size > size {
        let bssEnd = Addr(header.virtual_addr + header.mem_size).RoundUp()?;
        if bssEnd.0 != endMem.0 {
            let mut moptions = MMapOpts::NewAnonOptions("[elf:static]".to_string())?;
            moptions.Length = bssEnd.0 - endMem.0;
            moptions.Addr = offset + endMem.0;
            moptions.Fixed = true;
            moptions.Perms = AccessType::ReadWrite();
            moptions.MaxPerms = AccessType::ReadWrite();
            moptions.Private = true;

            let _addr = task.mm.MMap(task, &mut moptions)?;
        }
    }

    return Ok(());
}

// loadedELF describes an ELF that has been successfully loaded.
#[derive(Default)]
pub struct LoadedElf {
    // os is the target OS of the ELF.
    pub os: OS,

    // arch is the target architecture of the ELF.
    pub arch: Arch,

    // entry is the entry point of the ELF.
    pub entry: u64,

    // start is the start of the ELF.
    pub start: u64,

    // start is the end of the ELF.
    pub end: u64,

    // interpter is the path to the ELF interpreter.
    pub interpreter: String,

    // phdrAddr is the address of the ELF program headers.
    pub phdrAddr: u64,

    // phdrSize is the size of a single program header in the ELF.
    pub phdrSize: usize,

    // phdrNum is the number of program headers.
    pub phdrNum: usize,

    // auxv contains a subset of ELF-specific auxiliary vector entries:
    // * AT_PHDR
    // * AT_PHENT
    // * AT_PHNUM
    // * AT_BASE
    // * AT_ENTRY
    pub auxv: Vec<AuxEntry>,
}

pub fn ReadAll(task: &mut Task, file: &File, data: &mut [u8], offset: u64) -> Result<usize> {
    let mut data = data;
    let mut offset = offset;
    let mut cnt = 0;

    while data.len() > 0 {
        let mut iovecs: [IoVec; 1] = [IoVec {
            start: &data[0] as *const _ as u64,
            len: data.len(),
        }];

        let l = file.Preadv(task, &mut iovecs, offset as i64)? as usize;
        cnt += l;

        if l == data.len() || l == 0 {
            return Ok(cnt);
        }

        data = &mut data[l..];
        offset += l as u64;
    }

    return Ok(cnt);
}

pub fn LoadParseElf(
    task: &mut Task,
    file: &File,
    info: &mut ElfHeadersInfo,
    sharedLoadOffset: u64,
) -> Result<LoadedElf> {
    let mut first = true;
    let mut start = 0;
    let mut end = 0;
    let mut interpreter = "".to_string();

    let filesize = file.UnstableAttr(task)?.Size;
    for header in &info.phdrs {
        let headerType = header.get_type().map_err(Error::ELFLoadError)?;
        match headerType {
            Type::Interp => {
                if header.file_size < 2 {
                    info!("Error: PT_INTERP path too small");
                    return Err(Error::SysError(SysErr::ENOEXEC));
                }

                if header.file_size > 4096
                /*PATH_MAX*/
                {
                    info!("Error: PT_INTERP path too big");
                    return Err(Error::SysError(SysErr::ENOEXEC));
                }

                let mut fileName: Vec<u8> = vec![0; header.file_size as usize]; //remove last '/0'

                match ReadAll(task, file, &mut fileName, header.offset as u64) {
                    Err(e) => {
                        info!("Error: reading PT_INTERP path {:?}", e);
                        return Err(Error::SysError(SysErr::ENOEXEC));
                    }
                    Ok(_) => (),
                }

                if fileName[fileName.len() - 1] != 0 {
                    info!("Error: PT_INTERP path not NUL-terminated");
                    return Err(Error::SysError(SysErr::ENOEXEC));
                }

                interpreter = match String::from_utf8(fileName[0..fileName.len() - 1].to_vec()) {
                    Err(_) => {
                        info!("interpreter name can't covert to utf8");
                        return Err(Error::SysError(SysErr::ENOEXEC));
                    }
                    Ok(s) => s,
                };

                info!("the interpreter is {}", interpreter);
            }
            Type::Load => {
                let vaddr = header.virtual_addr;

                if first {
                    first = false;
                    start = vaddr;
                }

                if vaddr < end {
                    info!("PT_LOAD headers out-of-order. {:x} < {:x}", vaddr, end);
                    return Err(Error::SysError(SysErr::ENOEXEC));
                }

                end = match Addr(vaddr).AddLen(header.mem_size) {
                    Err(_) => {
                        info!(
                            "PT_LOAD header size overflows. {:x} + {:x}",
                            vaddr, header.mem_size
                        );
                        return Err(Error::SysError(SysErr::ENOEXEC));
                    }
                    Ok(a) => a.0,
                };
            }
            t => {
                panic!("find unexpect type {:?}", t)
            }
        }
    }

    // Shared objects don't have fixed load addresses. We need to pick a
    // base address big enough to fit all segments, so we first create a
    // mapping for the total size just to find a region that is big enough.
    let mut offset = 0;
    if info.sharedObject {
        let totalSize = match Addr(end - start).RoundUp() {
            Err(_) => {
                info!("ELF PT_LOAD segments too big");
                return Err(Error::SysError(SysErr::ENOEXEC));
            }
            Ok(s) => s.0,
        };

        offset = match task.mm.FindAvailableSeg(task, sharedLoadOffset, totalSize) {
            Err(e) => {
                info!("Error allocating address space for shared object: {:?}", e);
                return Err(Error::SysError(SysErr::ENOEXEC));
            }
            Ok(s) => s,
        };

        start += offset;
        end += offset;
        info.entry += offset;
    }

    // Map PT_LOAD segments.
    for header in &info.phdrs {
        let headerType = header.get_type().map_err(Error::ELFLoadError)?;
        if headerType == Type::Load {
            if header.mem_size == 0 {
                // No need to load segments with size 0, but
                // they exist in some binaries.
                continue;
            }

            match MapSegment(task, file, header, offset, filesize) {
                Err(e) => {
                    info!("Failed to map PT_LOAD segment: {:?}", e);
                    return Err(Error::SysError(SysErr::ENOEXEC));
                }
                Ok(()) => (),
            }
        }
    }

    let phdrAddr = match Addr(start).AddLen(info.phdrAddr) {
        Err(_) => {
            info!(
                "ELF start address {:x} + phdr offset {:x} overflows",
                start, info.phdrAddr
            );
            0
        }
        Ok(a) => a.0,
    };

    return Ok(LoadedElf {
        os: info.os,
        arch: info.arch,
        entry: info.entry,
        start: start,
        end: end,
        interpreter: interpreter,
        phdrAddr: phdrAddr,
        phdrSize: info.phdrSize,
        phdrNum: info.phdrNum,
        auxv: Vec::new(),
    });
}

// loadInitialELF loads f into mm.
pub fn LoadInitalElf(task: &mut Task, file: &File) -> Result<LoadedElf> {
    let mut info = ParseHeader(task, file)?;

    let l = task
        .mm
        .SetMmapLayout(MIN_USER_ADDR, MAX_USER_ADDR, &LimitSet::default())?;
    *task.mm.layout.lock() = l;

    let loadAddr = Context64::PIELoadAddress(&l)?;

    let le = LoadParseElf(task, file, &mut info, loadAddr)?;
    return Ok(le);
}

// loadInterpreterELF loads f into mm.
//
// The interpreter must be for the same OS/Arch as the initial ELF.
//
// It does not return any auxv entries.
//
// Preconditions:
//  * f is an ELF file
pub fn loadInterpreterELF(task: &mut Task, file: &File, initial: &LoadedElf) -> Result<LoadedElf> {
    let mut info = match ParseHeader(task, file) {
        Err(e) => {
            if e == Error::SysError(SysErr::ENOEXEC) {
                return Err(Error::SysError(SysErr::ELIBBAD));
            }

            return Err(e);
        }
        Ok(i) => i,
    };

    if info.os != initial.os {
        info!(
            "Initial ELF OS {} and interpreter ELF OS {} differ",
            initial.os, info.os
        );
        return Err(Error::SysError(SysErr::ELIBBAD));
    }

    if info.arch != initial.arch {
        info!(
            "Initial ELF arch {} and interpreter ELF arch {} differ",
            initial.arch, info.arch
        );
        return Err(Error::SysError(SysErr::ELIBBAD));
    }

    // The interpreter is not given a load offset, as its location does not
    // affect brk.
    return LoadParseElf(task, file, &mut info, 0);
}

pub fn LoadElf(task: &mut Task, file: &File) -> Result<LoadedElf> {
    let mut bin = match LoadInitalElf(task, file) {
        Err(e) => {
            info!("Error loading binary: {:?}", e);
            return Err(e);
        }
        Ok(b) => b,
    };

    let mut interp = LoadedElf::default();
    if bin.interpreter.as_str() != "" {
        let fileName = CString::New(&bin.interpreter);
        let fd = task.Open(fileName.Ptr(), OpenFlags::O_RDONLY as u64, 0); //kernel address is same as phy address
        if (fd as i64) < 0 {
            info!(
                "LoadElf Error opening interpreter {} with error {}",
                &bin.interpreter, -fd as i64
            );
        }

        let interpFile = task.GetFile(fd as i32)?;
        interp = loadInterpreterELF(task, &interpFile, &bin)?;

        if interp.interpreter.as_str() != "" {
            info!(
                "Interpreter requires an interpreter {}",
                &interp.interpreter
            );
            return Err(Error::SysError(SysErr::ENOEXEC));
        }
    }

    bin.auxv.push(AuxEntry {
        Key: AuxVec::AT_PHDR,
        Val: bin.phdrAddr,
    });
    bin.auxv.push(AuxEntry {
        Key: AuxVec::AT_PHENT,
        Val: bin.phdrSize as u64,
    });
    bin.auxv.push(AuxEntry {
        Key: AuxVec::AT_PHNUM,
        Val: bin.phdrNum as u64,
    });
    bin.auxv.push(AuxEntry {
        Key: AuxVec::AT_ENTRY,
        Val: bin.entry,
    });

    if bin.interpreter.as_str() != "" {
        bin.auxv.push(AuxEntry {
            Key: AuxVec::AT_BASE,
            Val: interp.start,
        });
        bin.entry = interp.entry;
    } else {
        bin.auxv.push(AuxEntry {
            Key: AuxVec::AT_BASE,
            Val: 0,
        });
    }

    return Ok(bin);
}
