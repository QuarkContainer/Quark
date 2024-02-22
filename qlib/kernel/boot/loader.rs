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

use alloc::collections::btree_map::BTreeMap;
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::ops::Deref;

use crate::qlib::mutex::*;

use super::super::super::auth;
use super::super::super::auth::cap_set::*;
use super::super::super::auth::id::*;
use super::super::super::auth::userns::*;
use super::super::super::common::*;
use super::super::super::cpuid::*;
use super::super::super::limits::*;
use super::super::super::linux_def::*;
use super::super::super::loader::*;
use super::super::fs::file::*;
use super::super::fs::host::tty::*;
use super::super::fs::mount::*;
use super::super::kernel::ipc_namespace::*;
use super::super::kernel::kernel::*;
use super::super::kernel::uts_namespace::*;
use super::super::kernel::waiter::qlock::*;
use super::super::task::*;
use super::super::threadmgr::thread::*;
use super::super::threadmgr::thread_group::*;
use super::super::SignalDef::*;
use crate::GUEST_KERNEL;
use super::fs::*;

impl Process {
    pub fn TaskCaps(&self) -> TaskCaps {
        return TaskCaps {
            PermittedCaps: CapSet(self.Caps.PermittedCaps.0),
            InheritableCaps: CapSet(self.Caps.InheritableCaps.0),
            EffectiveCaps: CapSet(self.Caps.EffectiveCaps.0),
            BoundingCaps: CapSet(self.Caps.BoundingCaps.0),
            AmbientCaps: CapSet(self.Caps.AmbientCaps.0),
        };
    }
}

#[derive(Eq, Debug)]
pub struct ExecID {
    pub cid: String,
    pub pid: ThreadID,
}

impl Ord for ExecID {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.cid != other.cid {
            return self.cid.cmp(&other.cid);
        }

        return self.pid.cmp(&other.pid);
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
    pub hostTTY: i32,
    pub stdios: [i32; 3],
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
    pub fn InitKernel(&self, process: Process) -> Result<()> {
        let mut gids = Vec::with_capacity(process.AdditionalGids.len());
        for gid in &process.AdditionalGids {
            gids.push(KGID(*gid))
        }

        let userns = UserNameSpace::NewRootUserNamespace();
        let hostName = process.HostName.to_string();

        let utsns = UTSNamespace::New(hostName.to_string(), hostName.to_string(), userns.clone());
        let ipcns = IPCNamespace::New(&userns);

        let kernelArgs = InitKernelArgs {
            FeatureSet: Arc::new(QMutex::new(HostFeatureSet())),
            RootUserNamespace: userns.clone(),
            ApplicationCores: process.NumCpu,
            ExtraAuxv: Vec::new(),
            RootUTSNamespace: utsns,
            RootIPCNamespace: ipcns,
        };

        let kernel = Kernel::Init(kernelArgs);
        *GUEST_KERNEL.lock() = Some(kernel.clone());
        let task = Task::Current();
        self.Lock(task)?.kernel = kernel;
        Ok(())
    }

    pub fn WaitContainer(&self, cid: String) -> Result<u32> {
        let task = Task::Current();
        let (tg, _) = self
            .Lock(task)?
            .ThreadGroupFromID(&ExecID { cid: cid, pid: 0 })
            .expect("WaitContainer: there is no root container");

        let task = Task::Current();
        tg.WaitExited(task);

        return Ok(tg.ExitStatus().Status());
    }

    pub fn WaitPID(&self, cid: String, pid: ThreadID, clearStatus: bool) -> Result<u32> {
        let task = Task::Current();
        let tg = match self.Lock(task)?.ThreadGroupFromID(&ExecID {
            cid: cid.clone(),
            pid: pid,
        }) {
            None => {
                return Err(Error::Common(format!(
                    "Loader::WaitPID pid {} doesn't exist",
                    pid
                )));
            }
            Some((tg, _)) => tg,
        };

        if clearStatus {
            self.Lock(task)?.processes.remove(&ExecID { cid, pid });
        }

        tg.WaitExited(task);

        return Ok(tg.ExitStatus().Status());
    }

    //Exec a new process in current sandbox, it supports 'runc exec'
    pub fn ExecProcess(&self, process: Process) -> Result<(i32, u64, u64, u64)> {
        let task = Task::Current();
        let kernel = self.Lock(task)?.kernel.clone();
        let userns = kernel.RootUserNamespace();
        let mut gids = Vec::with_capacity(process.AdditionalGids.len());
        for gid in &process.AdditionalGids {
            gids.push(KGID(*gid))
        }
        let cid = process.ID.clone();
        let creds = auth::Credentials::NewUserCredentials(
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
            let file = task
                .NewFileFromHostStdioFd(0, procArgs.Stdiofds[0], true)
                .expect("Task: create std fds");
            file.flags.lock().0.NonBlocking = false; //need to clean the stdio nonblocking

            assert!(task.Dup2(0, 1) == 1);
            assert!(task.Dup2(0, 2) == 2);

            let fileops = file.FileOp.clone();
            let ttyops = fileops
                .as_any()
                .downcast_ref::<TTYFileOps>()
                .expect("TTYFileOps convert fail")
                .clone();

            ttyops.InitForegroundProcessGroup(&tg.ProcessGroup().unwrap());
            ttyFileOps = Some(ttyops);
        } else {
            task.NewStdFds(&procArgs.Stdiofds[..], false)
                .expect("Task: create std fds");
        }

        let execProc = ExecProcess {
            tg: tg,
            tty: ttyFileOps,
            ..Default::default()
        };
        self.Lock(task)?
            .processes
            .insert(ExecID { cid: cid, pid: tid }, execProc);
        let paths = GetPath(&procArgs.Envv);
        procArgs.Filename = task.mountNS.ResolveExecutablePath(
            task,
            &procArgs.WorkingDirectory,
            &procArgs.Filename,
            &paths,
        )?;
        let (entry, userStackAddr, kernelStackAddr) =
            kernel.LoadProcess(&procArgs.Filename, &procArgs.Envv, &mut procArgs.Argv)?;
        return Ok((tid, entry, userStackAddr, kernelStackAddr));
    }

    pub fn LoadRootProcess(
        &self,
        procArgs: &mut CreateProcessArgs,
    ) -> Result<(i32, u64, u64, u64)> {
        let task = Task::Current();
        task.creds = procArgs.Credentials.clone();
        let kernel = self.Lock(task)?.kernel.clone();
        let (tg, tid) = kernel.CreateProcess(procArgs)?;
        let paths = GetPath(&procArgs.Envv);
        procArgs.Filename = task.mountNS.ResolveExecutablePath(
            task,
            &procArgs.WorkingDirectory,
            &procArgs.Filename,
            &paths,
        )?;

        let mut ttyFileOps = None;
        if procArgs.Terminal {
            let file = task
                .NewFileFromHostStdioFd(0, procArgs.Stdiofds[0], true)
                .expect("Task: create std fds");
            file.flags.lock().0.NonBlocking = false; //need to clean the stdio nonblocking
            assert!(task.Dup2(0, 1) == 1);
            assert!(task.Dup2(0, 2) == 2);

            let fileops = file.FileOp.clone();
            let ttyops = fileops
                .as_any()
                .downcast_ref::<TTYFileOps>()
                .expect("TTYFileOps convert fail")
                .clone();

            ttyops.InitForegroundProcessGroup(&tg.ProcessGroup().unwrap());
            ttyFileOps = Some(ttyops);
        } else {
            task.NewStdFds(&procArgs.Stdiofds[..], false)
                .expect("Task: create std fds");
        }

        GetKernel().Start()?;

        //task.NewStdFds(&procArgs.Stdiofds[..], procArgs.Terminal).expect("Task: create std fds");

        let execProc = ExecProcess {
            tg: tg,
            tty: ttyFileOps,
            ..Default::default()
        };

        //self.processes.insert(ExecID{cid: procArgs.ContainerID.to_string(), pid: tid}, execProc);
        //for the root container, the tid is always 0,
        self.Lock(task)?.processes.insert(
            ExecID {
                cid: procArgs.ContainerID.to_string(),
                pid: 0,
            },
            execProc,
        );

        let (entry, userStackAddr, kernelStackAddr) =
            kernel.LoadProcess(&procArgs.Filename, &procArgs.Envv, &mut procArgs.Argv)?;
        return Ok((tid, entry, userStackAddr, kernelStackAddr));
    }

    pub fn CreateSubContainer(&self, cid: String, fds: Vec<i32>) -> Result<()> {
        let tty = if fds.len() == 1 { fds[0] } else { -1 };
        let stdios = if fds.len() == 3 {
            [fds[0], fds[1], fds[2]]
        } else {
            [-1, -1, -1]
        };
        let task = Task::Current();
        let execId = ExecID { cid: cid, pid: 0 };

        if self.lock().processes.contains_key(&execId) {
            error!("Subcontainer {} already exists", &execId.cid);
            return Err(Error::Common("Subcontainer already exists".to_string()));
        }

        self.Lock(task)?.processes.insert(
            execId,
            ExecProcess {
                hostTTY: tty,
                stdios: stdios,
                ..Default::default()
            },
        );
        return Ok(());
    }

    pub fn StartSubContainer(&self, processSpec: Process) -> Result<(i32, u64, u64, u64)> {
        let task = Task::Current();
        let mut lockedLoader = self.Lock(task)?;
        let kernel = lockedLoader.kernel.clone();
        let cid = processSpec.ID.clone();
        let execId = ExecID { cid: cid, pid: 0 };
        let process = match lockedLoader.processes.get_mut(&execId) {
            None => {
                return Err(Error::Common(format!(
                    "trying to start a deleted container {}",
                    &execId.cid
                )));
            }
            Some(p) => p,
        };

        let mut gids = Vec::with_capacity(processSpec.AdditionalGids.len());
        for gid in &processSpec.AdditionalGids {
            gids.push(KGID(*gid))
        }

        let userns = kernel.RootUserNamespace();

        let creds = auth::Credentials::NewUserCredentials(
            KUID(processSpec.UID),
            KGID(processSpec.GID),
            &gids[..],
            Some(&processSpec.TaskCaps()),
            &userns,
        );
        let rootMounts = InitRootFs(Task::Current(), &processSpec.Root)
            .expect("in loader::StartSubContainer, InitRootfs fail");
        kernel
            .mounts
            .write()
            .insert(processSpec.ID.clone(), rootMounts);

        //todo: investigate PID namespace and whether we need it.
        let mut createProcessArgs = NewProcess(processSpec, &creds, &kernel);
        let (tg, tid) = kernel.CreateProcess(&mut createProcessArgs)?;

        let mut ttyFileOps = None;
        if createProcessArgs.Terminal {
            if process.hostTTY == -1 {
                error!("terminal fd not provided for terminal mode");
                return Err(Error::Common(
                    "missing terminal fd for subcontainer".to_string(),
                ));
            }
            let file = task
                .NewFileFromHostStdioFd(0, process.hostTTY, true)
                .expect("Task: create std fds");
            file.flags.lock().0.NonBlocking = false;

            assert!(task.Dup2(0, 1) == 1);
            assert!(task.Dup2(0, 2) == 2);

            let fileops = file.FileOp.clone();

            let ttyops = fileops.TTYFileOps().unwrap();

            /*let ttyops = fileops
            .as_any()
            .downcast_ref::<TTYFileOps>()
            .expect("TTYFileOps convert fail")
            .clone();*/

            ttyops.InitForegroundProcessGroup(&tg.ProcessGroup().unwrap());
            ttyFileOps = Some(ttyops)
        } else {
            if process.stdios[0] == -1 {
                error!("stdio fds not provided for non-terminal mode");
                return Err(Error::Common(
                    "missing stdio fds for subcontainer".to_string(),
                ));
            }
            debug!(
                "using stdios to start subcontainer: {:?}",
                &process.stdios[..]
            );
            task.NewStdFds(&process.stdios[..], false)
                .expect("Task: create std fds");
        }

        process.tg = tg;
        process.tty = ttyFileOps;

        let paths = GetPath(&createProcessArgs.Envv);
        createProcessArgs.Filename = task.mountNS.ResolveExecutablePath(
            task,
            &createProcessArgs.WorkingDirectory,
            &createProcessArgs.Filename,
            &paths,
        )?;
        let (entry, userStackAddr, kernelStackAddr) = kernel.LoadProcess(
            &createProcessArgs.Filename,
            &createProcessArgs.Envv,
            &mut createProcessArgs.Argv,
        )?;
        return Ok((tid, entry, userStackAddr, kernelStackAddr));
    }
}

#[derive(Default)]
pub struct LoaderInternal {
    // kernel is a copy of the kernel info
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
    pub processes: BTreeMap<ExecID, ExecProcess>,

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
        let creds = auth::Credentials::NewUserCredentials(
            KUID(process.UID),
            KGID(process.GID),
            &gids[..],
            Some(&process.TaskCaps()),
            &userns,
        );

        let hostName = process.HostName.to_string();

        let utsns = UTSNamespace::New(hostName.to_string(), hostName.to_string(), userns.clone());
        let ipcns = IPCNamespace::New(&userns);

        let kernelArgs = InitKernelArgs {
            FeatureSet: Arc::new(QMutex::new(HostFeatureSet())),
            RootUserNamespace: userns.clone(),
            ApplicationCores: process.NumCpu,
            ExtraAuxv: Vec::new(),
            RootUTSNamespace: utsns,
            RootIPCNamespace: ipcns,
        };

        let kernel = Kernel::Init(kernelArgs);
        *GUEST_KERNEL.lock() = Some(kernel.clone());

        let rootMounts =
            InitRootFs(Task::Current(), &process.Root).expect("in loader::New, InitRootfs fail");
        kernel.mounts.write().insert(sandboxID.clone(), rootMounts);

        let processArgs = NewProcess(process, &creds, &kernel);
        self.kernel = kernel;
        self.console = console;
        self.sandboxID = sandboxID;
        return processArgs;
    }

    pub fn ThreadGroupFromID(&self, key: &ExecID) -> Option<(ThreadGroup, Option<TTYFileOps>)> {
        match self.processes.get(key) {
            None => (),
            Some(ep) => return Some((ep.tg.clone(), ep.tty.clone())),
        };

        return None;
    }

    pub fn SignalForegroundProcessGroup(
        &self,
        cid: String,
        tgid: ThreadID,
        signo: i32,
    ) -> Result<()> {
        let (tg, tty) = match self.ThreadGroupFromID(&ExecID {
            cid: cid,
            pid: tgid,
        }) {
            None => {
                info!(
                    "SignalForegroundProcessGroup: no thread group found for {}",
                    tgid
                );
                let err = Err(Error::Common(format!("no thread group found for {}", tgid)));
                return err;
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
            info!(
                "No foreground process group PID {}. Sending signal directly to PID {}.",
                tgid, tgid
            );
            return tg.SendSignal(&SignalInfo {
                Signo: signo,
                ..Default::default()
            });
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
                continue;
            }

            match tg.SendSignal(&SignalInfo {
                Signo: signo,
                ..Default::default()
            }) {
                Err(e) => lastErr = Err(e),
                Ok(()) => (),
            }
        }

        return lastErr;
    }

    pub fn SignalProcess(&self, mut cid: String, tgid: ThreadID, signo: i32) -> Result<()> {
        // if the cid string is empty, by default send to process of the root container
        if cid.is_empty() {
            cid = self.sandboxID.clone();
        }
        match self.ThreadGroupFromID(&ExecID {
            cid: cid.clone(),
            pid: tgid,
        }) {
            None => (),
            Some((execTG, _)) => {
                return execTG.SendSignal(&SignalInfo {
                    Signo: signo,
                    ..Default::default()
                });
            }
        }

        // The caller may be signaling a process not started directly via exec.
        // In this case, find the process in the container's PID namespace and
        // signal it.
        let (initTG, _) = match self.ThreadGroupFromID(&ExecID {
            cid: cid.clone(),
            pid: 0,
        }) {
            None => return Err(Error::SysError(SysErr::ENOENT)),
            Some(d) => d,
        };

        let tg = match initTG.PIDNamespace().ThreadGroupWithID(tgid) {
            None => return Err(Error::Common(format!("no such process with PID {}", tgid))),
            Some(tg) => tg,
        };

        return tg.SendSignal(&SignalInfo {
            Signo: signo,
            ..Default::default()
        });
    }

    pub fn SignalAll(&self, signo: i32) -> Result<()> {
        self.kernel.Pause();
        match self.kernel.SignalAll(&SignalInfo {
            Signo: signo,
            ..Default::default()
        }) {
            Err(e) => {
                self.kernel.Unpause();
                return Err(e);
            }
            Ok(()) => {
                self.kernel.Unpause();
                return Ok(());
            }
        }
    }

    pub fn ThreadGroupFromIDLocked(
        &self,
        key: &ExecID,
    ) -> Result<(ThreadGroup, Option<TTYFileOps>)> {
        let ep = match self.processes.get(key) {
            None => return Err(Error::Common("container not found".to_string())),
            Some(ep) => ep,
        };

        return Ok((ep.tg.clone(), ep.tty.clone()));
    }

    pub fn SignalAllProcesses(&self, cid: &str, signo: i32) -> Result<()> {
        self.kernel.Pause();
        let res = self.kernel.SendContainerSignal(
            cid,
            &SignalInfo {
                Signo: signo,
                ..Default::default()
            },
        );

        self.kernel.Unpause();
        match res {
            Err(e) => return Err(e),
            Ok(()) => (),
        }

        return Ok(());
    }

    pub fn DestroyContainer(&mut self, cid: String) -> Result<()> {
        let l = self;
        let execId = ExecID {
            cid: cid.clone(),
            pid: 0,
        };
        match l.ThreadGroupFromIDLocked(&execId) {
            Ok(_) => match l.SignalProcess(cid.clone(), 0, 9) {
                Ok(()) => (),
                Err(Error::NotExist) => (),
                Err(e) => {
                    return Err(Error::Common(format!(
                        "sending SIGKILL to container processes: {:?}",
                        e
                    )));
                }
            },
            Err(_e) => (),
        }

        l.processes.remove(&execId);

        info!("Container {} destroyed", &cid);
        return Ok(());
    }
}

pub fn NewProcess(process: Process, creds: &auth::Credentials, k: &Kernel) -> CreateProcessArgs {
    let utsns = k.rootUTSNamespace.clone();
    let ipcns = k.rootIPCNamespace.clone();
    let mut stdiofds: [i32; 3] = [0; 3];
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
        Limits: LimitSet(Arc::new(QMutex::new(process.limitSet))),
        MaxSymlinkTraversals: MAX_SYMLINK_TRAVERSALS,
        UTSNamespace: utsns,
        IPCNamespace: ipcns,
        ContainerID: process.ID,
        Stdiofds: stdiofds,
        Terminal: process.Terminal,
        ExecId: process.ExecId.clone(),
        ..Default::default()
    };
}
