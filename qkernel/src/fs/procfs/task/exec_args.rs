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

use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use super::super::super::super::qlib::common::*;
use super::super::super::super::qlib::linux_def::*;
use super::super::super::super::qlib::addr::*;
use super::super::super::super::qlib::auth::*;
use super::super::super::super::qlib::mem::seq::*;
use super::super::super::fsutil::file::readonly_file::*;
use super::super::super::super::task::*;
use super::super::super::file::*;
use super::super::super::attr::*;
use super::super::super::super::threadmgr::thread::*;
use super::super::super::flags::*;
use super::super::super::dirent::*;
use super::super::super::mount::*;
use super::super::super::inode::*;
use super::super::super::fsutil::inode::simple_file_inode::*;
use super::super::inode::*;

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum ExecArgType {
    CmdlineExecArg,
    EnvironExecArg,
}

pub fn NewExecArg(task: &Task, thread: &Thread, msrc: &Arc<Mutex<MountSource>>, typ: ExecArgType) -> Inode {
    let v = NewExecArgSimpleFileInode(task, thread, &ROOT_OWNER, &FilePermissions::FromMode(FileMode(0o400)), FSMagic::PROC_SUPER_MAGIC, typ);
    return NewProcInode(&Arc::new(v), msrc, InodeType::SpecialFile, Some(thread.clone()))
}

pub fn NewExecArgSimpleFileInode(task: &Task,
                                 thread: &Thread,
                                 owner: &FileOwner,
                                 perms: &FilePermissions,
                                 typ: u64,
                                 execArgType: ExecArgType)
                                -> SimpleFileInode<ExecArgSimpleFileTrait> {
    return SimpleFileInode::New(task, owner, perms, typ, false, ExecArgSimpleFileTrait{
        typ: execArgType,
        thread: thread.clone(),
    })
}

pub struct ExecArgSimpleFileTrait {
    pub typ: ExecArgType,
    pub thread: Thread,
}

impl SimpleFileTrait for ExecArgSimpleFileTrait {
    fn GetFile(&self, _task: &Task, _dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        let fops = NewExecArgReadonlyFileNodeFileOperations(self.typ, &self.thread);
        let file = File::New(dirent, &flags, fops);
        return Ok(file);
    }
}

pub fn NewExecArgReadonlyFileNodeFileOperations(typ: ExecArgType, thread: &Thread) -> ReadonlyFileOperations<ExecArgReadonlyFileNode> {
    return ReadonlyFileOperations {
        node: ExecArgReadonlyFileNode {
            thread: thread.clone(),
            typ: typ,
        }
    }
}

pub struct ExecArgReadonlyFileNode {
    pub typ: ExecArgType,
    pub thread: Thread,
}

impl ReadonlyFileNode for ExecArgReadonlyFileNode {
    fn ReadAt(&self, task: &Task, _f: &File, dsts: &mut [IoVec], offset: i64, _blocking: bool) -> Result<i64> {
        if offset < 0 {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        let mm = self.thread.lock().memoryMgr.clone();

        let range = match self.typ {
            ExecArgType::CmdlineExecArg => {
                mm.metadata.lock().argv
            }
            ExecArgType::EnvironExecArg => {
                mm.metadata.lock().envv
            }
        };

        info!("ExecArgReadonlyFileNode range is {:x?}", &range);
        if range.Start() == 0 || range.End() == 0 {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        let start = match Addr(range.Start()).AddLen(offset as u64) {
            Err(_) => return Ok(0),
            Ok(v) => v.0,
        };

        let end = range.End();
        if start >= end {
            return Ok(0)
        }

        let mut length = end - start;

        let blocks = BlockSeq::ToBlocks(dsts);
        let dsts = BlockSeq::NewFromSlice(&blocks);
        if length > dsts.NumBytes() {
            length = dsts.NumBytes();
        }

        let data : Vec<u8> = task.CopyIn(start, length as usize)?;
        let mut buf = &data[..];

        // On Linux, if the NUL byte at the end of the argument vector has been
        // overwritten, it continues reading the environment vector as part of
        // the argument vector.
        if self.typ == ExecArgType::CmdlineExecArg && buf[buf.len() - 1] != 0 {
            let mut copyN = buf.len();
            for i in 0..buf.len() {
                if buf[i] == 0 {
                    copyN = i;
                    break;
                }
            }

            if copyN < buf.len() {
                buf = &buf[..copyN]
            } else {
                let envv = mm.metadata.lock().envv;
                let mut lengthEnvv = envv.Len() as usize;

                if lengthEnvv > MemoryDef::PAGE_SIZE as usize - buf.len() {
                    lengthEnvv = MemoryDef::PAGE_SIZE as usize - buf.len();
                }

                let envvData = task.CopyIn(envv.Start(), lengthEnvv as usize)?;
                let mut copyNE = envvData.len();
                for i in 0..envvData.len() {
                    if envvData[i] == 0{
                        copyNE = i;
                        break;
                    }
                }

                let mut ret = Vec::with_capacity(buf.len() + copyNE);
                for b in buf {
                    ret.push(*b)
                }
                for b in &envvData[..copyNE]  {
                    ret.push(*b)
                }
                buf = &ret[..];

                let n = dsts.CopyOut(buf);
                return Ok(n as i64)
            }
        }

        let n = dsts.CopyOut(buf);
        return Ok(n as i64)
    }
}
