// Copyright (c) 2021 QuarkSoft LLC
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
use spin::Mutex;
use alloc::vec::Vec;
use alloc::string::ToString;

use super::super::super::super::qlib::common::*;
use super::super::super::super::qlib::linux_def::*;
use super::super::super::super::qlib::auth::*;
use super::super::super::super::kernel::kernel::*;
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
use super::super::super::super::threadmgr::pid_namespace::*;
use super::super::inode::*;

pub fn NewStatus(task: &Task, thread: &Thread, msrc: &Arc<Mutex<MountSource>>) -> Inode {
    let v = NewStatusSimpleFileInode(task, thread, &ROOT_OWNER, &FilePermissions::FromMode(FileMode(0o400)), FSMagic::PROC_SUPER_MAGIC);
    return NewProcInode(&Arc::new(v), msrc, InodeType::SpecialFile, Some(thread.clone()))

}

pub fn NewStatusSimpleFileInode(task: &Task,
                                thread: &Thread,
                               owner: &FileOwner,
                               perms: &FilePermissions,
                               typ: u64)
                               -> SimpleFileInode<StatusData> {
    let kernel = GetKernel();
    let pidns = kernel.RootPIDNamespace();

    let status = StatusData {
        thread: thread.clone(),
        pidns: pidns,
    };

    return SimpleFileInode::New(task, owner, perms, typ, false, status)
}

pub struct StatusData {
    thread: Thread,
    pidns: PIDNamespace,
}

impl StatusData {
    pub fn GenSnapshot(&self, _task: &Task) -> Vec<u8> {
        let mut ret = "".to_string();

        ret += &format!("Name:\t{}\n", self.thread.Name());
        // todo: handle thread state
        //ret += &format!("State:\t{}\n", self.thread.Name());

        let tg = self.thread.ThreadGroup();
        ret += &format!("Tgid:\t{}\n", self.pidns.IDOfThreadGroup(&tg));
        ret += &format!("Pid:\t{}\n", self.pidns.IDOfTask(&self.thread));

        let ppid = match self.thread.Parent() {
            None => 0,
            Some(parent) => {
                let tg = parent.ThreadGroup();
                self.pidns.IDOfThreadGroup(&tg)
            },
        };
        ret += &format!("PPid:\t{}\n", ppid);
        ret += &format!("TracerPid:\t{}\n", 0);

        let fds = self.thread.lock().fdTbl.Count();
        ret += &format!("FDSize:\t{}\n", fds);

        let mm = self.thread.lock().memoryMgr.clone();
        let vss = mm.VirtualMemorySize();
        let rss = mm.ResidentSetSize();
        ret += &format!("VmSize:\t{} kB\n", vss>>10);
        ret += &format!("VmRSS:\t{} kB\n", rss>>10);
        ret += &format!("Threads:\t{}\n", tg.Count());

        let creds = self.thread.Credentials();
        ret += &format!("CapInh:\t{:016x}\n", creds.lock().InheritableCaps.0);
        ret += &format!("CapPrm:\t{:016x}\n", creds.lock().PermittedCaps.0);
        ret += &format!("CapEff:\t{:016x}\n", creds.lock().EffectiveCaps.0);
        ret += &format!("CapBnd:\t{:016x}\n", creds.lock().BoundingCaps.0);
        ret += &format!("Seccomp:\t{}\n", 0);

        ret += &format!("Mems_allowed:\t{}\n",
                        "00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000001");

        return ret.as_bytes().to_vec();
    }
}

impl SimpleFileTrait for StatusData {
    fn GetFile(&self, task: &Task, _dir: &Inode, dirent: &Dirent, flags: FileFlags) -> Result<File> {
        let fops = NewSnapshotReadonlyFileOperations(self.GenSnapshot(task));
        let file = File::New(dirent, &flags, fops);
        return Ok(file);
    }
}