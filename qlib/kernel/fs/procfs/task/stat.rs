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

use crate::qlib::mutex::*;
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;

use super::super::super::super::super::auth::*;
use super::super::super::super::super::common::*;
use super::super::super::super::super::limits::*;
use super::super::super::super::super::linux::time::*;
use super::super::super::super::super::linux_def::*;
use super::super::super::super::super::kernel::*;
use super::super::super::super::task::*;
use super::super::super::super::threadmgr::pid_namespace::*;
use super::super::super::super::threadmgr::thread::*;
use super::super::super::attr::*;
use super::super::super::dirent::*;
use super::super::super::file::*;
use super::super::super::flags::*;
use super::super::super::fsutil::file::readonly_file::*;
use super::super::super::fsutil::inode::simple_file_inode::*;
use super::super::super::inode::*;
use super::super::super::mount::*;
use super::super::inode::*;

pub fn NewStat(
    task: &Task,
    thread: &Thread,
    showSubtasks: bool,
    pidns: PIDNamespace,
    msrc: &Arc<QMutex<MountSource>>,
) -> Inode {
    let v = NewStatSimpleFileInode(
        task,
        thread,
        showSubtasks,
        pidns,
        &ROOT_OWNER,
        &FilePermissions::FromMode(FileMode(0o400)),
        FSMagic::PROC_SUPER_MAGIC,
    );
    return NewProcInode(
        &Arc::new(v),
        msrc,
        InodeType::SpecialFile,
        Some(thread.clone()),
    );
}

pub fn NewStatSimpleFileInode(
    task: &Task,
    thread: &Thread,
    showSubtasks: bool,
    pidns: PIDNamespace,
    owner: &FileOwner,
    perms: &FilePermissions,
    typ: u64,
) -> SimpleFileInode {
    let io = TaskStatData {
        t: thread.clone(),
        tgstats: showSubtasks,
        pidns: pidns,
    };

    return SimpleFileInode::New(task, owner, perms, typ, false, io.into());
}

pub struct TaskStatData {
    pub t: Thread,

    // If tgstats is true, accumulate fault stats (not implemented) and CPU
    // time across all tasks in t's thread group.
    pub tgstats: bool,

    // pidns is the PID namespace associated with the proc filesystem that
    // includes the file using this statData.
    pub pidns: PIDNamespace,
}

impl TaskStatData {
    pub fn GenSnapshot(&self, _task: &Task) -> Vec<u8> {
        let mut output: String = "".to_string();
        output += &format!("{} ", self.pidns.IDOfTask(&self.t));
        output += &format!("({}) ", self.t.Name());
        output += &format!("{} ", self.t.lock().StateStatus().as_bytes()[0] as char);

        let ppid = match self.t.Parent() {
            None => 0,
            Some(parent) => self.pidns.IDOfThreadGroup(&parent.ThreadGroup()),
        };
        output += &format!("{} ", ppid);
        output += &format!(
            "{} ",
            self.pidns
                .IDOfProcessGroup(&self.t.ThreadGroup().ProcessGroup().unwrap())
        );
        output += &format!(
            "{} ",
            self.pidns
                .IDOfSession(&self.t.ThreadGroup().Session().unwrap())
        );
        output += &format!("0 0 " /* tty_nr tpgid */);
        output += &format!("0 " /* flags */);
        output += &format!("0 0 0 0 " /* minflt cminflt majflt cmajflt */);

        let cputime = if self.tgstats {
            self.t.ThreadGroup().CPUStats()
        } else {
            self.t.CPUStats()
        };
        output += &format!(
            "{} {} ",
            ClockTFromDuration(Tsc::Scale(cputime.UserTime) * 1000),
            ClockTFromDuration(Tsc::Scale(cputime.SysTime) * 1000)
        );

        let cputime = self.t.ThreadGroup().JoinedChildCPUStats();
        output += &format!(
            "{} {} ",
            ClockTFromDuration(Tsc::Scale(cputime.UserTime) * 1000),
            ClockTFromDuration(Tsc::Scale(cputime.SysTime) * 1000)
        );

        output += &format!("{} {} ", self.t.Priority(), self.t.Niceness());
        output += &format!("{} ", self.t.ThreadGroup().Count());

        // itrealvalue. Since kernel 2.6.17, this field is no longer
        // maintained, and is hard coded as 0.
        output += &format!("{} ", 0);

        // Start time is relative to boot time, expressed in clock ticks.
        output += &format!(
            "{} ",
            ClockTFromDuration(
                self.t
                    .StartTime()
                    .Sub(self.t.Kernel().TimeKeeper().BootTime())
            )
        );

        let (vss, rss) = {
            let mm = self.t.lock().memoryMgr.clone();
            let _ml = mm.MappingReadLock();
            let vs = mm.VirtualMemorySizeLocked();
            let rs = mm.ResidentSetSizeLocked();
            (vs, rs)
        };
        output += &format!("{} {} ", vss, rss / MemoryDef::PAGE_SIZE);

        // rsslim.
        output += &format!("{} ", self.t.ThreadGroup().Limits().Get(LimitType::Rss).Cur);

        output += &format!("0 0 0 0 0 " /* startcode endcode startstack kstkesp kstkeip */);
        output += &format!("0 0 0 0 0 " /* signal blocked sigignore sigcatch wchan */);
        output += &format!("0 0 " /* nswap cnswap */);

        let terminationSignal = if Some(self.t.clone()) == self.t.ThreadGroup().Leader() {
            self.t.ThreadGroup().TerminationSignal()
        } else {
            Signal(0)
        };
        output += &format!("{} ", terminationSignal.0);
        output += &format!("0 0 0 " /* processor rt_priority policy */);
        output += &format!("0 0 0 " /* delayacct_blkio_ticks guest_time cguest_time */);
        output += &format!(
            "0 0 0 0 0 0 0 " /* start_data end_data start_brk arg_start arg_end env_start env_end */
        );
        output += &format!("0\n" /* exit_code */);

        return output.as_bytes().to_vec();
    }
}

impl SimpleFileTrait for TaskStatData {
    fn GetFile(
        &self,
        task: &Task,
        _dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        let fops = NewSnapshotReadonlyFileOperations(self.GenSnapshot(task));
        let file = File::New(dirent, &flags, fops.into());
        return Ok(file);
    }
}
