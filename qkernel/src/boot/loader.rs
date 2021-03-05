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

use spin::Mutex;
use alloc::sync::Arc;
use alloc::vec::Vec;
use alloc::string::String;
use alloc::string::ToString;
use core::cmp::Ordering;
use alloc::collections::btree_map::BTreeMap;
use core::ops::Deref;

use super::super::qlib::auth::cap_set::*;
use super::super::qlib::common::*;
use super::super::qlib::cpuid::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::limits::*;
use super::super::qlib::loader::*;
use super::super::qlib::auth::*;
use super::super::qlib::auth::userns::*;
use super::super::qlib::auth::id::*;
use super::super::SignalDef::*;
use super::super::task::*;
use super::super::kernel::kernel::*;
use super::super::kernel::ipc_namespace::*;
use super::super::kernel::uts_namespace::*;
use super::super::threadmgr::thread::*;
use super::super::threadmgr::thread_group::*;
use super::super::fs::host::tty::*;
use super::super::fs::mount::*;
use super::super::kernel::waiter::qlock::*;
use super::fs::*;

pub fn InitRef1<T>() -> &'static T {
    let ptr = unsafe {
        &*(0 as *const T)
    };

    return ptr;
}

impl Process {
    pub fn TaskCaps(&self) -> TaskCaps {
        return TaskCaps {
            PermittedCaps: CapSet(self.Caps.PermittedCaps.0),
            InheritableCaps: CapSet(self.Caps.InheritableCaps.0),
            EffectiveCaps: CapSet(self.Caps.EffectiveCaps.0),
            BoundingCaps: CapSet(self.Caps.BoundingCaps.0),
            AmbientCaps: CapSet(self.Caps.AmbientCaps.0),
        }
    }
}

#[derive(Eq)]
pub struct ExecID {
    pub cid: String,
    pub pid: ThreadID,
}

impl Ord for ExecID {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.cid != other.cid {
            return self.cid.cmp(&other.cid)
        }

        return self.pid.cmp(&other.pid)
    }
}

impl PartialOrd for ExecID {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ExecID {
    fn eq(&self, other: &Self) -> bool {
        self.cid == other.cid && self.pid == other.pid
    }
}

#[derive(Default, Clone)]
pub struct ExecProcess {
    pub tg: ThreadGroup,
    pub tty: Option<TTYFileOps>,
}

#[derive(Default)]
pub struct Loader(QLock<LoaderInternal>);

impl Deref for Loader {
    type Target = QLock<LoaderInternal>;

    fn deref(&self) -> &QLock<LoaderInternal> {
        &self.0
    }
}

impl Loader {
    pub fn WaitContainer(&self) -> Result<u32> {
        let task = Task::Current();
        let (tg, _) = self.Lock(task)?.ThreadGroupFromID(0).expect("WaitContainer: there is no root container");

        let task = Task::Current();
        tg.WaitExited(task);

        return Ok(tg.ExitStatus().Status())
    }

    pub fn WaitPID(&self, pid: ThreadID, clearStatus: bool) -> Result<u32> {
        let task = Task::Current();
        let tg =  match self.Lock(task)?.ThreadGroupFromID(pid) {
            None => {
                return Err(Error::Common(format!("Loader::WaitPID pid {} doesn't exist", pid)))
            },
            Some((tg, _)) => tg,
        };

        if clearStatus {
            self.Lock(task)?.processes.remove(&pid);
        }

        tg.WaitExited(task);

        return Ok(tg.ExitStatus().Status())
    }

    //Exec a new process in current sandbox, it supports 'runc exec'
    pub fn ExecProcess(&self, process: Process) -> Result<(i32, u64, u64, u64)> {
        let task = Task::Current();

        let kernel = self.Lock(task)?.kernel.clone();
        let userns = kernel.rootUserNamespace.clone();

        let mut gids = Vec::with_capacity(process.AdditionalGids.len());
        for gid in &process.AdditionalGids {
            gids.push(KGID(*gid))
        }

        let creds = Credentials::NewUserCredentials(
            KUID(process.UID),
            KGID(process.GID),
            &gids[..],
            Some(&process.TaskCaps()),
            &userns,
        );

        let mut procArgs = NewProcess(process, &creds, &kernel);

        let (tg, tid) = kernel.CreateProcess(&mut procArgs)?;

        let mut ttyFileOps = None;
        if procArgs.Terminal {
            let file = task.NewFileFromHostFd(0, procArgs.Stdiofds[0], true).expect("Task: create std fds");
            file.flags.lock().0.NonBlocking = false; //need to clean the stdio nonblocking

            assert!(task.Dup2(0, 1)==1);
            assert!(task.Dup2(0, 2)==2);

            let fileops = file.FileOp.clone();
            let ttyops = fileops.as_any().downcast_ref::<TTYFileOps>()
                .expect("TTYFileOps convert fail").clone();

            ttyops.InitForegroundProcessGroup(&tg.ProcessGroup().unwrap());
            ttyFileOps = Some(ttyops);
        } else {
            task.NewStdFds(&procArgs.Stdiofds[..], false).expect("Task: create std fds");
        }

        let execProc = ExecProcess {
            tg : tg,
            tty: ttyFileOps,
        };

        self.Lock(task)?.processes.insert(tid, execProc);

        let (entry, userStackAddr, kernelStackAddr) = kernel.LoadProcess(&procArgs.Filename, &procArgs.Envv, &mut procArgs.Argv)?;
        return Ok((tid, entry, userStackAddr, kernelStackAddr))
    }

    pub fn LoadRootProcess(&self, procArgs: &mut CreateProcessArgs) -> Result<(i32, u64, u64, u64)>  {
        let task = Task::Current();
        task.creds = procArgs.Credentials.clone();
        let kernel = self.Lock(task)?.kernel.clone();
        let (tg, tid) = kernel.CreateProcess(procArgs)?;
        let paths = GetPath(&procArgs.Envv);
        procArgs.Filename = task.mountNS.ResolveExecutablePath(task, &procArgs.WorkingDirectory, &procArgs.Filename, &paths)?;

        let mut ttyFileOps = None;
        if procArgs.Terminal {
            let file = task.NewFileFromHostFd(0, procArgs.Stdiofds[0], true)
                .expect("Task: create std fds");
            file.flags.lock().0.NonBlocking = false; //need to clean the stdio nonblocking
            assert!(task.Dup2(0, 1)==1);
            assert!(task.Dup2(0, 2)==2);

            let fileops = file.FileOp.clone();
            let ttyops = fileops.as_any().downcast_ref::<TTYFileOps>()
                .expect("TTYFileOps convert fail").clone();

            ttyops.InitForegroundProcessGroup(&tg.ProcessGroup().unwrap());
            ttyFileOps = Some(ttyops);
        } else {
            task.NewStdFds(&procArgs.Stdiofds[..], false).expect("Task: create std fds");
        }

        //task.NewStdFds(&procArgs.Stdiofds[..], procArgs.Terminal).expect("Task: create std fds");

        let execProc = ExecProcess {
            tg : tg,
            tty: ttyFileOps,
        };

        //self.processes.insert(ExecID{cid: procArgs.ContainerID.to_string(), pid: tid}, execProc);
        //for the root container, the tid is always 0,
        self.Lock(task)?.processes.insert(0, execProc);

        let (entry, userStackAddr, kernelStackAddr) = kernel.LoadProcess(&procArgs.Filename, &procArgs.Envv, &mut procArgs.Argv)?;
        return Ok((tid, entry, userStackAddr, kernelStackAddr))
    }

}

#[derive(Default)]
pub struct LoaderInternal {
    // k is the kernel.
    pub kernel: Kernel,

    // rootProcArgs refers to the root sandbox init task.
    pub rootProcArgs: CreateProcessArgs,

    // sandboxID is the ID for the whole sandbox.
    pub sandboxID: String,

    // console is set to true if terminal is enabled.
    pub console: bool,

    // processes maps containers init process and invocation of exec. Root
    // processes are keyed with container ID and pid=0, while exec invocations
    // have the corresponding pid set.
    pub processes: BTreeMap<ThreadID, ExecProcess>,

    //whether the root container will auto started without StartRootContainer Ucall
    pub autoStart: bool,
}

impl LoaderInternal {
    //init the root process
    pub fn Init(&mut self, process: Process) -> CreateProcessArgs {
        let console = process.Terminal;
        let sandboxID = process.ID.to_string();

        let mut gids = Vec::with_capacity(process.AdditionalGids.len());
        for gid in &process.AdditionalGids {
            gids.push(KGID(*gid))
        }

        let userns = UserNameSpace::NewRootUserNamespace();
        let creds = Credentials::NewUserCredentials(
            KUID(process.UID),
            KGID(process.GID),
            &gids[..],
            Some(&process.TaskCaps()),
            &userns,
        );

        let hostName = process.HostName.to_string();

        let utsns = UTSNamespace::New(hostName.to_string(), hostName.to_string(), userns.clone());
        let ipcns = IPCNamespace::New(&userns);

        let kernalArgs = InitKernalArgs {
            FeatureSet: Arc::new(Mutex::new(HostFeatureSet())),
            RootUserNamespace: userns.clone(),
            ApplicationCores: process.NumCpu,
            ExtraAuxv: Vec::new(),
            RootUTSNamespace: utsns,
            RootIPCNamespace: ipcns,
         };

        let kernel = Kernel::Init(kernalArgs);
        *KERNEL.lock() = Some(kernel.clone());

        let rootMounts = BootInitRootFs(Task::Current(), &process.Root).expect("in loader::New, InitRootfs fail");
        *kernel.mounts.write() = Some(rootMounts);

        let processArgs = NewProcess(process, &creds, &kernel);
        self.kernel = kernel;
        self.console = console;
        self.sandboxID = sandboxID;
        return processArgs
    }

    pub fn ThreadGroupFromID(&self, pid: ThreadID) -> Option<(ThreadGroup, Option<TTYFileOps>)> {
        match self.processes.get(&pid) {
            None => (),
            Some(ep) => return Some((ep.tg.clone(), ep.tty.clone())),
        };

        return None;
    }

    pub fn SignalForegroundProcessGroup(&self, tgid: ThreadID, signo: i32) -> Result<()> {
        let (tg, tty) = match self.ThreadGroupFromID(tgid) {
            None => {
                return Err(Error::Common(format!("no thread group found for {}", tgid)))
            }
            Some(r) => r,
        };
        let tty = match tty {
            None => return Err(Error::Common("no tty attached".to_string())),
            Some(t) => t,
        };

        let pg = tty.ForegroundProcessGroup();
        if pg.is_none() {
            // No foreground process group has been set. Signal the
            // original thread group.
            info!("No foreground process group PID {}. Sending signal directly to PID {}.",
                tgid, tgid);
            return tg.SendSignal(&SignalInfo{
                Signo: signo,
                ..Default::default()
            })
        }

        // Send the signal to all processes in the process group.
        let kernel = self.kernel.clone();
        kernel.extMu.lock();
        let tasks = kernel.tasks.read();

        let root = tasks.root.as_ref().unwrap().clone();
        let mut lastErr = Ok(());

        let tgs = root.ThreadGroups();
        for tg in &tgs {
            if tg.ProcessGroup() != pg {
                continue
            }

            match tg.SendSignal(&SignalInfo{
                Signo: signo,
                ..Default::default()
            }) {
                Err(e) => lastErr = Err(e),
                Ok(()) => (),
            }
        }

        return lastErr
    }

    pub fn SignalProcess(&self, tgid: ThreadID, signo: i32) -> Result<()> {
        match self.ThreadGroupFromID(tgid) {
            None => (),
            Some((execTG, _)) => {
                return execTG.SendSignal(&SignalInfo{
                    Signo: signo,
                    ..Default::default()
                });
            }
        }

        // The caller may be signaling a process not started directly via exec.
        // In this case, find the process in the container's PID namespace and
        // signal it.
        let (initTG, _) = self.ThreadGroupFromID(0).unwrap();
        let tg = match initTG.PIDNamespace().ThreadGroupWithID(tgid) {
            None => return Err(Error::Common(format!("no such process with PID {}", tgid))),
            Some(tg) => tg,
        };

        return tg.SendSignal(&SignalInfo{
            Signo: signo,
            ..Default::default()
        });
    }

    pub fn SignalAll(&self, signo: i32) -> Result<()> {
        self.kernel.Pause();
        match self.kernel.SignalAll(&SignalInfo{
            Signo: signo,
            ..Default::default()}) {
            Err(e) => {
                self.kernel.Unpause();
                return Err(e)
            }
            Ok(()) => {
                self.kernel.Unpause();
                return Ok(())
            }
        }
    }

    pub fn ThreadGroupFromIDLocked(&self, key: ThreadID) -> Result<(ThreadGroup, Option<TTYFileOps>)> {
        let ep = match self.processes.get(&key) {
            None => return Err(Error::Common("container not found".to_string())),
            Some(ep) => ep,
        };

        return Ok((ep.tg.clone(), ep.tty.clone()))
    }

    pub fn SignalAllProcesses(&self, cid: &str, signo: i32) -> Result<()> {
        self.kernel.Pause();
        let res = self.kernel.SendContainerSignal(cid, &SignalInfo{
            Signo: signo,
            ..Default::default()
        });

        self.kernel.Unpause();
        match res {
            Err(e) => return Err(e),
            Ok(()) => (),
        }

        return Ok(())
    }

    pub fn DestroyContainer(&mut self) -> Result<()> {
        let l = self;

        match l.ThreadGroupFromIDLocked(0) {
            Ok(_) => {
                l.SignalAll(Signal::SIGKILL as i32)
                    .map_err(|e| Error::Common(format!("sending SIGKILL to all container processes: {:?}", e)))?;
            }
            Err(_e) => (),
        }

        l.processes.clear();

        info!("Container destroyed");
        return Ok(())
    }
}

pub fn NewProcess(process: Process, creds: &Credentials, k: &Kernel) -> CreateProcessArgs {
    let utsns = k.rootUTSNamespace.clone();
    let ipcns = k.rootIPCNamespace.clone();
    let mut stdiofds : [i32; 3] = [0; 3];
    for i in 0..3 {
        stdiofds[i] = process.Stdiofds[i];
    }

    return CreateProcessArgs {
        Filename: process.Args[0].to_string(),
        Argv: process.Args,
        Envv: process.Envs,
        WorkingDirectory: process.Cwd,
        Credentials: creds.clone(),
        Umask: 0o22,
        Limits: LimitSet(Arc::new(Mutex::new(process.limitSet))),
        MaxSymlinkTraversals: MAX_SYMLINK_TRAVERSALS,
        UTSNamespace: utsns,
        IPCNamespace: ipcns,
        ContainerID: process.ID,
        Stdiofds: stdiofds,
        Terminal: process.Terminal,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_ser() {
        let mut buf: [u8; 4096] = [0; 4096];
        let mut pk = ProcessKernel::New(&mut buf);

        let process = Process {
            UID: 123,
            GID: 456,
            AdditionalGids: vec![1, 2, 3],
            Terminal: true,
            Args: vec!["str1".to_string(), "str2".to_string()],
            Commandline: "Commnandline".to_string(),
            Envs: vec!["env1".to_string(), "env2".to_string()],
            Cwd: "cwd".to_string(),
            PermittedCaps: 1,
            InheritableCaps: 2,
            EffectiveCaps: 3,
            BoundingCaps: 4,
            AmbientCaps: 5,
            NoNewPrivileges: false,
            NumCpu: 4,
            HostName: "asdf".to_string(),
            limitSet: LimitSetInternal::default(),
            ID: "containerID".to_string()
        };

        pk.Ser(&process).unwrap();
        let newProcess = pk.DeSer();

        print!("new process is {:?}", newProcess);
        assert!(process == newProcess)
    }
}
