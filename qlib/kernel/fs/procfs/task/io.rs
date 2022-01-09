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
use crate::qlib::mutex::*;
use alloc::string::ToString;
use core::sync::atomic::Ordering;
use alloc::vec::Vec;

use super::super::super::super::super::common::*;
use super::super::super::super::super::linux_def::*;
use super::super::super::super::super::auth::*;
use super::super::super::super::super::usage::io::*;
use super::super::super::fsutil::file::readonly_file::*;
use super::super::super::fsutil::inode::simple_file_inode::*;
use super::super::super::super::task::*;
use super::super::super::attr::*;
use super::super::super::file::*;
use super::super::super::flags::*;
use super::super::super::dirent::*;
use super::super::super::mount::*;
use super::super::super::inode::*;
use super::super::super::super::threadmgr::thread::*;
use super::super::super::super::threadmgr::thread_group::*;
use super::super::super::super::threadmgr::task_acct::*;
use super::super::inode::*;

pub fn NewIO(task: &Task, thread: &Thread, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let v = NewIOSimpleFileInode(task, thread, &ROOT_OWNER, &FilePermissions::FromMode(FileMode(0o400)), FSMagic::PROC_SUPER_MAGIC);
    return NewProcInode(&Arc::new(v), msrc, InodeType::SpecialFile, Some(thread.clone()))

}

pub fn NewIOSimpleFileInode(task: &Task,
                              thread: &Thread,
                              owner: &FileOwner,
                              perms: &FilePermissions,
                              typ: u64)
                              -> SimpleFileInode<IOData> {
    let tg = thread.ThreadGroup();
    let io = IOData{tg: tg};

    return SimpleFileInode::New(task, owner, perms, typ, false, io)
}

pub struct IOData {
    tg: ThreadGroup,
}

impl IOData {
    pub fn GenSnapshot(&self) -> Vec<u8> {
        let io = IO::default();
        io.Accumulate(&self.tg.IOUsage());

        let mut buf = "".to_string();
        buf += &format!("char: {}\n", io.CharsRead.load(Ordering::SeqCst));
        buf += &format!("wchar: {}\n", io.CharsWritten.load(Ordering::SeqCst));
        buf += &format!("syscr: {}\n", io.ReadSyscalls.load(Ordering::SeqCst));
        buf += &format!("syscw: {}\n", io.WriteSyscalls.load(Ordering::SeqCst));
        buf += &format!("read_bytes: {}\n", io.BytesRead.load(Ordering::SeqCst));
        buf += &format!("write_bytes: {}\n", io.BytesWritten.load(Ordering::SeqCst));
        buf += &format!("cancelled_write_bytes: {}\n", io.BytesWriteCancelled.load(Ordering::SeqCst));

        return buf.as_bytes().to_vec();
    }
}

impl SimpleFileTrait for IOData {
    fn GetFile(&self, _task: &Task, _dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        let fops = NewSnapshotReadonlyFileOperations(self.GenSnapshot());
        let file = File::New(dirent, &flags, fops);
        return Ok(file);
    }
}

/*use alloc::sync::Arc;
use crate::qlib::mutex::*;
use alloc::vec::Vec;
use core::sync::atomic::Ordering;
use alloc::string::ToString;

use super::super::super::super::super::usage::io::*;
use super::super::super::super::task::*;
use super::super::super::attr::*;
use super::super::super::mount::*;
use super::super::super::inode::*;
use super::super::super::super::threadmgr::thread_group::*;
use super::super::super::super::threadmgr::task_acct::*;
use super::super::inode::*;
use super::super::seqfile::*;


pub fn NewIO(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let tg = task.Thread().ThreadGroup();
    let seqFile = SeqFile::New(task, Arc::new(QMutex::new(IOData{tg: tg})));

    return NewProcInode(&Arc::new(seqFile), msrc, InodeType::SpecialFile, Some(task))
}

pub struct IOData {
    tg: ThreadGroup,
}

impl SeqSource for IOData {
    fn NeedsUpdate(&mut self, generation: i64) -> bool {
        return generation == 0;
    }

    fn ReadSeqFileData(&mut self, _task: &Task, handle: SeqHandle) -> (Vec<SeqData>, i64) {
        info!("IOData ReadSeqFileData...");

        match handle {
            SeqHandle::None => (),
            _ => return (Vec::new(), 0),
        }

        let io = IO::default();
        io.Accumulate(&self.tg.IOUsage());

        let mut buf = "".to_string();
        buf += &format!("char: {}\n", io.CharsRead.load(Ordering::SeqCst));
        buf += &format!("wchar: {}\n", io.CharsWritten.load(Ordering::SeqCst));
        buf += &format!("syscr: {}\n", io.ReadSyscalls.load(Ordering::SeqCst));
        buf += &format!("syscw: {}\n", io.WriteSyscalls.load(Ordering::SeqCst));
        buf += &format!("read_bytes: {}\n", io.BytesRead.load(Ordering::SeqCst));
        buf += &format!("write_bytes: {}\n", io.BytesWritten.load(Ordering::SeqCst));
        buf += &format!("cancelled_write_bytes: {}\n", io.BytesWriteCancelled.load(Ordering::SeqCst));

        info!("IOData ReadSeqFileData... buf is {}", &buf);
        return (vec!(SeqData {
            Buf: buf.as_bytes().to_vec(),
            Handle: SeqHandle::None,
        }), 0)
    }
}*/