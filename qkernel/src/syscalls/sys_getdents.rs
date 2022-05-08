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

use alloc::vec::Vec;

use super::super::fd::*;
use super::super::fs::dentry::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::mem::io::*;
use super::super::syscalls::syscalls::*;
use super::super::task::*;

pub fn SysGetDents(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let size = args.arg2 as i32;

    return GetDents(task, fd, addr, size);
}

pub fn GetDents(task: &Task, fd: i32, addr: u64, size: i32) -> Result<i64> {
    let minSize = Dirent::SmallestDirent() as i32;
    if minSize > size {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let n = getDents(task, fd, addr, size, Serialize)?;

    return Ok(n);
}

pub fn SysGetDents64(task: &mut Task, args: &SyscallArguments) -> Result<i64> {
    let fd = args.arg0 as i32;
    let addr = args.arg1 as u64;
    let size = args.arg2 as i32;

    return GetDents64(task, fd, addr, size);
}

pub fn GetDents64(task: &Task, fd: i32, addr: u64, size: i32) -> Result<i64> {
    let minSize = Dirent::SmallestDirent64() as i32;
    if minSize > size {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let n = getDents(task, fd, addr, size, Serialize64)?;

    return Ok(n);
}

const WIDTH: u32 = 8;

fn getDents(
    task: &Task,
    fd: i32,
    addr: u64,
    size: i32,
    f: fn(&Task, &Dirent, &mut IOWriter) -> Result<i32>,
) -> Result<i64> {
    let dir = task.GetFile(fd)?;

    let size = task.CheckPermission(addr, size as u64, true, true)? as i32;

    let mut writer: MemBuf = MemBuf::New(size as usize);

    let len = size; // writer.Len() as i32;
    let mut ds = HostDirentSerializer::New(f, &mut writer, WIDTH, len);
    let err = dir.ReadDir(task, &mut ds);
    match err {
        Ok(()) => {
            let buf = &writer.data;
            task.CopyOutSlice(buf, addr, size as usize)?;
            return Ok(buf.len() as i64);
        }
        Err(Error::EOF) => return Ok(0),
        Err(e) => return Err(e),
    }
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
struct OldDirentHdr {
    pub Ino: u64,
    pub Off: u64,
    pub Reclen: u16,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
struct DirentHdr {
    pub OldHdr: OldDirentHdr,
    pub Type: u8,
}

#[repr(C)]
#[derive(Debug, Default, Clone)]
struct Dirent {
    pub Hdr: DirentHdr,
    pub Name: Vec<u8>,
}

impl Dirent {
    pub fn New(width: u32, name: &str, attr: &DentAttr, offset: u64) -> Self {
        let mut d = Dirent {
            Hdr: DirentHdr {
                OldHdr: OldDirentHdr {
                    Ino: attr.InodeId,
                    Off: offset,
                    Reclen: 0,
                },
                Type: attr.Type.ToType(),
            },
            Name: name.as_bytes().to_vec(),
        };

        d.Name.push(0);
        d.Hdr.OldHdr.Reclen = d.padRec(width as usize);

        return d;
    }

    fn padRec(&mut self, width: usize) -> u16 {
        //let a = mem::size_of::<DirentHdr>() + self.Name.len();
        let a = 19 + self.Name.len();
        let r = (a + width) & !(width - 1);
        let padding = r - a;
        self.Name.append(&mut vec![0; padding]);
        return r as u16;
    }

    pub fn SmallestDirent() -> u32 {
        return 18 + WIDTH + 1;
    }

    pub fn SmallestDirent64() -> u32 {
        return 19 + WIDTH;
    }
}

fn Serialize64(task: &Task, dir: &Dirent, w: &mut IOWriter) -> Result<i32> {
    let addr = &dir.Hdr as *const _ as u64;
    let size = 18; //mem::size_of::<DirentHdr>();
                   //let slice = task.GetSlice::<u8>(addr, size)?;
    let buf = task.CopyInVec::<u8>(addr, size)?;

    let n1 = w.Write(&buf)?;
    let n3 = w.Write(&[dir.Hdr.Type; 1])?;
    let n2 = w.Write(&dir.Name)?;
    return Ok((n1 + n2 + n3) as i32);
}

fn Serialize(task: &Task, dir: &Dirent, w: &mut IOWriter) -> Result<i32> {
    let addr = &dir.Hdr as *const _ as u64;
    let size = 18; //mem::size_of::<OldDirentHdr>();
                   //let slice = task.GetSlice::<u8>(addr, size)?;
    let buf = task.CopyInVec::<u8>(addr, size)?;

    let n1 = w.Write(&buf)?;
    let n2 = w.Write(&dir.Name)?;
    let n3 = w.Write(&[dir.Hdr.Type; 1])?;
    return Ok((n1 + n2 + n3) as i32);
}

struct HostDirentSerializer<'a> {
    pub serialize: fn(&Task, &Dirent, &mut IOWriter) -> Result<i32>,
    pub w: &'a mut IOWriter,
    pub width: u32,
    pub offset: u64,
    pub written: i32,
    pub size: i32,
}

impl<'a> HostDirentSerializer<'a> {
    pub fn New(
        f: fn(&Task, &Dirent, &mut IOWriter) -> Result<i32>,
        w: &'a mut IOWriter,
        width: u32,
        size: i32,
    ) -> Self {
        return Self {
            serialize: f,
            w: w,
            width: width,
            offset: 0,
            written: 0,
            size: size,
        };
    }
}

impl<'a> DentrySerializer for HostDirentSerializer<'a> {
    fn CopyOut(&mut self, task: &Task, name: &str, attr: &DentAttr) -> Result<()> {
        self.offset += 1;
        let d = Dirent::New(self.width, name, attr, self.offset);

        let mut writer = MemBuf::New(1024);
        let res = (self.serialize)(task, &d, &mut writer);
        let n = match res {
            Ok(n) => n,
            Err(e) => {
                self.offset -= 1;
                return Err(e);
            }
        };

        if n as i32 > self.size - self.written {
            self.offset -= 1;
            return Err(Error::EOF);
        }

        let b = &writer.data;
        match self.w.Write(&b[..n as usize]) {
            Ok(_) => (),
            Err(e) => {
                self.offset -= 1;
                return Err(e);
            }
        }

        self.written += n as i32;
        return Ok(());
    }

    fn Written(&self) -> usize {
        return self.written as usize;
    }
}
