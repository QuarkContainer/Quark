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

use super::super::super::super::auth::*;
use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::kernel::kernel::*;
use super::super::super::task::*;
use super::super::attr::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::flags::*;
use super::super::fsutil::file::readonly_file::*;
use super::super::fsutil::inode::simple_file_inode::*;
use super::super::inode::*;
use super::super::mount::*;
use super::inode::*;

// cpuStats contains the breakdown of CPU time for /proc/stat.
#[derive(Default, Debug)]
pub struct CpuStats {
    // user is time spent in userspace tasks with non-positive niceness.
    pub user: u64,

    // nice is time spent in userspace tasks with positive niceness.
    pub nice: u64,

    // system is time spent in non-interrupt kernel context.
    pub system: u64,

    // idle is time spent idle.
    pub idle: u64,

    // ioWait is time spent waiting for IO.
    pub ioWait: u64,

    // irq is time spent in interrupt context.
    pub irq: u64,

    // softirq is time spent in software interrupt context.
    pub softirq: u64,

    // steal is involuntary wait time.
    pub steal: u64,

    // guest is time spent in guests with non-positive niceness.
    pub guest: u64,

    // guestNice is time spent in guests with positive niceness.
    pub guestNice: u64,
}

impl CpuStats {
    pub fn ToString(&self) -> String {
        let c = self;
        return format!(
            "{} {} {} {} {} {} {} {} {} {}",
            c.user,
            c.nice,
            c.system,
            c.idle,
            c.ioWait,
            c.irq,
            c.softirq,
            c.steal,
            c.guest,
            c.guestNice
        );
    }
}

pub fn NewStatData(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let v = NewStatDataSimpleFileInode(
        task,
        &ROOT_OWNER,
        &FilePermissions::FromMode(FileMode(0o400)),
        FSMagic::PROC_SUPER_MAGIC,
    );
    return NewProcInode(&Arc::new(v), msrc, InodeType::SpecialFile, None);
}

pub fn NewStatDataSimpleFileInode(
    task: &Task,
    owner: &FileOwner,
    perms: &FilePermissions,
    typ: u64,
) -> SimpleFileInode {
    let fs = StatData { k: GetKernel() };
    return SimpleFileInode::New(task, owner, perms, typ, false, fs.into());
}

pub struct StatData {
    pub k: Kernel,
}

impl StatData {
    pub fn GenSnapshot(&self, _task: &Task) -> Vec<u8> {
        let mut buf = "".to_string();

        // We currently export only zero CPU stats. We could
        // at least provide some aggregate stats.
        let cpu = CpuStats::default();
        buf += &format!("cpu {}\n", cpu.ToString());

        info!(
            "todo: fix self.k.ApplicationCores() is {}",
            self.k.ApplicationCores()
        );
        let cores = self.k.applicationCores;
        for i in 0..cores as usize {
            buf += &format!("cpu{} {}\n", i, cpu.ToString());
        }

        // The total number of interrupts is dependent on the CPUs and PCI
        // devices on the system. See arch_probe_nr_irqs.
        //
        // Since we don't report real interrupt stats, just choose an arbitrary
        // value from a representative VM.
        let numInterrupts = 256;

        // The Kernel doesn't handle real interrupts, so report all zeroes.
        buf += &format!("intr 0");
        for _i in 0..numInterrupts {
            buf += &format!(" 0");
        }
        buf += &format!("\n");

        // Total number of context switches.
        buf += &format!("ctxt 0\n");

        // CLOCK_REALTIME timestamp from boot, in seconds.
        buf += &format!("btime {}\n", self.k.TimeKeeper().BootTime().Seconds());

        // Total number of clones.
        buf += &format!("processes 0\n");

        // Number of runnable tasks.
        buf += &format!("procs_running 0\n");

        // Number of tasks waiting on IO.
        buf += &format!("procs_blocked 0\n");

        // Number of each softirq handled.
        let NumSoftIRQ = 10;
        buf += &format!("softirq 0"); // total
        for _i in 0..NumSoftIRQ {
            buf += &format!(" 0");
        }
        buf += &format!("\n");

        //info!("procstat is {}", &buf);
        return buf.as_bytes().to_vec();
    }
}

impl SimpleFileTrait for StatData {
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
