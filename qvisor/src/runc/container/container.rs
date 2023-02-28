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
use alloc::vec::Vec;
use std::env;
use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use fs2::FileExt;
use regex::Regex;

use super::hook::*;
use super::status::*;
//use super::super::super::qlib::util::*;
use super::super::super::qlib::auth::cap_set::*;
use super::super::super::qlib::auth::id::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::control_msg::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::path::*;
use super::super::super::ucall::ucall::*;
use super::super::cgroup::cgroup::*;
use super::super::cmd::config::*;
use super::super::cmd::exec::*;
use super::super::oci::serialize::*;
use super::super::oci::*;
use super::super::runtime::fs::FsImageMounter;
use super::super::sandbox::sandbox::*;
use super::super::shim::container_io::*;
use super::super::specutils::specutils::*;

// metadataFilename is the name of the metadata file relative to the
// container root directory that holds sandbox metadata.
const METADATA_FILENAME: &str = "meta.json";

// metadataLockFilename is the name of a lock file in the container
// root directory that is used to prevent concurrent modifications to
// the container state and metadata.
const METADATA_LOCK_FILENAME: &str = "meta.lock";

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Container {
    // ID is the container ID.
    #[serde(default)]
    pub ID: String,

    // Spec is the OCI runtime spec that configures this container.
    #[serde(default)]
    pub Spec: Spec,

    // BundleDir is the directory containing the container bundle.
    #[serde(default)]
    pub BundleDir: String,

    // Root is the directory containing the container metadata file. If this
    // container is the root container, Root and RootContainerDir will be the
    // same.
    #[serde(default)]
    pub Root: String,

    // CreatedAt is the time the container was created.
    #[serde(default)]
    pub CreateAt: u64,

    // Owner is the container owner.
    #[serde(default)]
    pub Owner: String,

    // ConsoleSocket is the path to a unix domain socket that will receive
    // the console FD.
    #[serde(default)]
    pub ConsoleSocket: String,

    // Status is the current container Status.
    #[serde(default)]
    pub Status: Status,

    // Sandbox is the sandbox this container is running in. It's set when the
    // container is created and reset when the sandbox is destroyed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub Sandbox: Option<Sandbox>,

    // RootContainerDir is the root directory containing the metadata file of the
    // sandbox root container. It's used to lock in order to serialize creating
    // and deleting this Container's metadata directory. If this container is the
    // root container, this is the same as Root.
    #[serde(default)]
    pub RootContainerDir: String,

    #[serde(default)]
    pub sandboxed: bool,
}

// List returns all container ids in the given root directory.
pub fn ContainerList(rootDir: &str) -> Result<Vec<String>> {
    info!("List containers {}", rootDir);
    let path = Path::new(rootDir);
    let mut ret = Vec::new();
    for entry in path
        .read_dir()
        .map_err(|e| Error::IOError(format!("List io::error is {:?}", e)))?
    {
        if let Ok(entry) = entry {
            ret.push(entry.file_name().to_str().unwrap().to_string());
        }
    }

    return Ok(ret);
}

pub fn findContainerRoot(rootDir: &str, partialID: &str) -> Result<String> {
    // Check whether the id fully specifies an existing container.
    let cRoot = Join(rootDir, partialID);
    if Path::new(&cRoot).exists() {
        return Ok(cRoot);
    }

    // Now see whether id could be an abbreviation of exactly 1 of the
    // container ids. If id is ambigious (it could match more than 1
    // container), it is an error.
    let mut cRoot = "".to_string();
    let ids = ContainerList(&cRoot)?;

    for id in &ids {
        if HasPrefix(id, partialID) {
            if cRoot.len() == 0 {
                return Err(Error::Common(format!(
                    "id {} is ambiguous and could refer to multiple containers: {}, {}",
                    partialID, cRoot, id
                )));
            }

            cRoot = id.to_string();
        }
    }

    if cRoot.len() == 0 {
        return Err(Error::SysError(SysErr::EEXIST));
    }

    info!("abbreviated id {} resolves to full id {}", partialID, cRoot);
    return Ok(Join(rootDir, &cRoot));
}

pub fn ValidateID(id: &str) -> Result<()> {
    let re = Regex::new(r"^[\w+-\.]+$").unwrap();
    if !re.is_match(id) {
        return Err(Error::Common(format!("id {} is not valid", id)));
    }

    return Ok(());
}

// maybeLockRootContainer locks the sandbox root container. It is used to
// prevent races to create and delete child container sandboxes.
pub fn maybeLockRootContainer(
    bundleDir: &str,
    spec: &Spec,
    rootDir: &str,
) -> Result<FileLockCleanup> {
    if IsRoot(spec) {
        return Ok(FileLockCleanup::default());
    }

    let sbid = match SandboxID(spec) {
        None => {
            return Err(Error::Common(
                "no sandbox ID found when locking root container".to_string(),
            ));
        }
        Some(id) => id,
    };

    let sandBoxRootDir = std::path::Path::new(bundleDir)
        .parent()
        .unwrap()
        .join(&sbid)
        .join(rootDir)
        .to_str()
        .unwrap()
        .to_string();
    let sb = Container::Load(&sandBoxRootDir, &sbid)?;

    return sb.Lock();
}

pub fn IsRoot(spec: &Spec) -> bool {
    return ShouldCreateSandbox(spec);
}

pub fn lockContainerMetadata(containerRootDir: &str) -> Result<FileLockCleanup> {
    fs::create_dir_all(containerRootDir).map_err(|e| {
        Error::IOError(format!(
            "lockContainerMetadata create_dir_all io::error is {:?}",
            e
        ))
    })?;
    let f = Join(containerRootDir, METADATA_LOCK_FILENAME);

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&f)
        .map_err(|e| Error::IOError(format!("lockContainerMetadata open io::error is {:?}", e)))?;

    // block until this process can lock the file
    file.lock_exclusive().map_err(|e| {
        Error::IOError(format!(
            "lockContainerMetadata lock_exclusive io::error is {:?}",
            e
        ))
    })?;
    return Ok(FileLockCleanup { file: Some(file) });
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Copy)]
pub enum RunAction {
    Create,
    Run,
}

#[derive(Default)]
pub struct FileLockCleanup {
    file: Option<File>,
}

impl Drop for FileLockCleanup {
    fn drop(&mut self) {
        match self.file {
            None => (),
            Some(ref file) => file.unlock().unwrap(),
        }
    }
}

impl Container {
    // Load loads a container with the given id from a metadata file. id may be an
    // abbreviation of the full container id, in which case Load loads the
    // container to which id unambiguously refers to.
    // Returns ErrNotExist if container doesn't exist.
    pub fn Load(rootDir: &str, id: &str) -> Result<Self> {
        info!("Load metadata for container {} {}", rootDir, id);
        ValidateID(id)?;

        let cRoot = findContainerRoot(rootDir, id)?;

        // Lock the container metadata to prevent other runsc instances from
        // writing to it while we are reading it.
        let _unlock = lockContainerMetadata(&cRoot)?;

        let metafile = Join(&cRoot, METADATA_FILENAME);
        info!("metadatafile is {}", &metafile);
        let mut c: Container = deserialize(&metafile)
            .map_err(|e| Error::Common(format!("Container::Load error is {:?}", e)))?;

        // If the status is "Running" or "Created", check that the sandbox
        // process still exists, and set it to Stopped if it does not.
        //
        // This is inherently racey.
        if c.Status == Status::Running || c.Status == Status::Created {
            // Check if the sandbox process is still running.
            if !c.isSandboxRunning() {
                info!("sandbox is not running, marking container as stopped...");
                c.changeStatus(Status::Stopped);
            } else if c.Status == Status::Running {
                match c.SignalContainer(0, false) {
                    Err(_e) => {
                        c.changeStatus(Status::Stopped);
                    }
                    Ok(_) => (),
                }
            }
        }

        Ok(c)
    }

    pub fn CreateTime(&self) -> SystemTime {
        return UNIX_EPOCH
            .checked_add(Duration::from_secs(self.CreateAt))
            .unwrap();
    }

    pub fn Lock(&self) -> Result<FileLockCleanup> {
        return lockContainerMetadata(&Join(&self.Root, &self.ID));
    }

    fn RequireStatus(&self, action: &str, statuses: &[Status]) -> Result<()> {
        for s in statuses {
            if self.Status == *s {
                return Ok(());
            }
        }

        return Err(Error::Common(format!(
            "cannot {} container {} in state {:?}",
            action, self.ID, self.Status
        )));
    }

    // SignalContainer sends the signal to the container. If all is true and signal
    // is SIGKILL, then waits for all processes to exit before returning.
    // SignalContainer returns an error if the container is already stopped.
    pub fn SignalContainer(&self, sig: i32, all: bool) -> Result<()> {
        info!("Signal container {:?} with signal: {:?}", self.ID, sig);
        // Signaling container in Stopped state is allowed. When all=false,
        // an error will be returned anyway; when all=true, this allows
        // sending signal to other processes inside the container even
        // after the init process exits. This is especially useful for
        // container cleanup.
        self.RequireStatus("signal", &[Status::Running, Status::Stopped])?;
        if !self.isSandboxRunning() {
            return Err(Error::Common("sandbox is not running".to_string()));
        }

        return self
            .Sandbox
            .as_ref()
            .unwrap()
            .SignalContainer(&self.ID, sig, all);
    }

    pub fn ForwardSignals(&self, pid: i32) {
        self.Sandbox.as_ref().unwrap().ForwardSignals(pid)
    }

    pub fn StopSignal(&self) {
        self.Sandbox.as_ref().unwrap().StopSignal();
    }

    pub fn SignalProcess(&self, sig: i32, pid: i32) -> Result<()> {
        info!(
            "Signal process {} in container {:?}: {:?}",
            pid, self.ID, sig
        );
        self.RequireStatus("signal", &[Status::Running])?;
        if !self.isSandboxRunning() {
            return Err(Error::Common("sandbox is not running".to_string()));
        }

        return self
            .Sandbox
            .as_ref()
            .unwrap()
            .SignalProcess(&self.ID, pid, sig, false);
    }

    fn changeStatus(&mut self, s: Status) {
        match s {
            Status::Creating => {
                panic!("invalid state transition: {:?} => {:?}", self.Status, s)
            }
            Status::Created => {
                if self.Status != Status::Creating {
                    panic!("invalid state transition: {:?} => {:?}", self.Status, s)
                }
                if self.Sandbox.is_none() {
                    panic!("sandbox cannot be nil")
                }
            }
            Status::Paused => {
                if self.Status != Status::Running {
                    panic!("invalid state transition: {:?} => {:?}", self.Status, s)
                }
                if self.Sandbox.is_none() {
                    panic!("sandbox cannot be nil")
                }
            }
            Status::Running => {
                if self.Status != Status::Created && self.Status != Status::Paused {
                    panic!("invalid state transition: {:?} => {:?}", self.Status, s)
                }
                if self.Sandbox.is_none() {
                    panic!("sandbox cannot be nil")
                }
            }
            Status::Stopped => {
                if self.Status != Status::Creating
                    && self.Status != Status::Created
                    && self.Status != Status::Running
                    && self.Status != Status::Stopped
                {
                    panic!("invalid state transition: {:?} => {:?}", self.Status, s)
                }
            }
        }

        self.Status = s;
    }

    fn isSandboxRunning(&self) -> bool {
        return self.Sandbox.is_some() && self.Sandbox.as_ref().unwrap().IsRunning();
    }

    pub fn CheckTerminal(
        action: RunAction,
        terminal: bool,
        consoleSocket: &str,
        detach: bool,
    ) -> Result<()> {
        let detach = detach || action == RunAction::Create;

        if detach && terminal && consoleSocket.len() == 0 {
            return Err(Error::Common(
                "cannot allocate tty if runc will detach without setting console socket"
                    .to_string(),
            ));
        }

        if (!detach || !terminal) && consoleSocket.len() > 0 {
            return Err(Error::Common(
                "annot use console socket if runc will not detach or allocate tty".to_string(),
            ));
        }

        return Ok(());
    }

    //pub fn maybeLockRootContainer(spec: &Spec, rootDir: &str)

    // Create creates the container in a new Sandbox process, unless the metadata
    // indicates that an existing Sandbox should be used. The caller must call
    // Destroy() on the container.
    pub fn Create(
        id: &str,
        action: RunAction,
        spec: Spec,
        conf: &GlobalConfig,
        bundleDir: &str,
        consoleSocket: &str,
        // pid is only used by shim, so not needed after switching to new shim
        pidFile: &str,
        userlog: &str,
        detach: bool,
        pivot: bool,
    ) -> Result<Self> {
        info!("Create container {} in root dir: {}", id, &conf.RootDir);
        debug!("spec for creating container: {:#?}", &spec);
        //debug!("container spec is {:?}", &spec);
        ValidateID(id)?;

        Self::CheckTerminal(action, spec.process.terminal, consoleSocket, detach)?;

        let _unlockRoot = maybeLockRootContainer(bundleDir, &spec, &conf.RootDir)?;

        // Lock the container metadata file to prevent concurrent creations of
        // containers with the same id.
        let containerRoot = Join(&conf.RootDir, id);

        let c = {
            let _unlock = lockContainerMetadata(&containerRoot)?;

            // Check if the container already exists by looking for the metadata
            // file.
            if Path::new(&Join(&containerRoot, METADATA_FILENAME)).exists() {
                return Err(Error::Common(format!(
                    "container with id {} already exists",
                    id
                )));
            }

            let user = match env::var("USER") {
                Err(_) => "".to_string(),
                Ok(s) => s.to_string(),
            };

            let mut c = Self {
                ID: id.to_string(),
                Spec: spec,
                ConsoleSocket: consoleSocket.to_string(),
                BundleDir: bundleDir.to_string(),
                Root: containerRoot,
                Status: Status::Creating,
                CreateAt: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                Owner: user,
                Sandbox: None,
                RootContainerDir: conf.RootDir.to_string(),
                sandboxed: false,
            };

            // If the metadata annotations indicate that this container should be
            // started in an existing sandbox, we must do so. The metadata will
            // indicate the ID of the sandbox, which is the same as the ID of the
            // init container in the sandbox.
            let isRoot = IsRoot(&c.Spec);
            if isRoot {
                debug!("Creating new sandbox for container {}", id);

                // Create and join cgroup before processes are created to ensure they are
                // part of the cgroup from the start (and all children processes).

                let mut cg: Option<Cgroup> = if crate::QUARK_CONFIG.lock().DisableCgroup {
                    None
                } else {
                    match Cgroup::New(&c.Spec) {
                        Err(e) => {
                            c.Destroy()?;
                            return Err(e);
                        }
                        Ok(cg) => cg,
                    }
                };
                if cg.is_some() {
                    let ret = cg
                        .as_mut()
                        .unwrap()
                        .Install(&c.Spec.linux.as_ref().unwrap().resources);
                    match ret {
                        Err(e) => {
                            c.Destroy()?;
                            return Err(e);
                        }
                        Ok(_) => (),
                    }
                }

                let restore = match &cg {
                    Some(ref cgroup) => {
                        let restore = cgroup.Join()?;
                        Some(restore)
                    }
                    None => None,
                };

                let ret = Sandbox::New(
                    id,
                    action,
                    &c.Spec,
                    conf,
                    bundleDir,
                    consoleSocket,
                    userlog,
                    cg,
                    detach,
                    pivot,
                );

                c.Sandbox = match ret {
                    Err(e) => {
                        c.Destroy()?;
                        return Err(e);
                    }
                    Ok(s) => Some(s),
                };

                match restore {
                    None => (),
                    Some(restore) => restore(),
                }
            } else {
                panic!("non CRI-compliant runtime should never call subcontainer");
                /* this is for non oci-compliant (to be specific), and should not call subcontainer
                let sandboxId = match SandboxID(&c.Spec) {
                    Some(sid) => sid,
                    None => {
                        error!("No sandbox ID found in spec when creating container inside sandbox");
                        return Err(Error::InvalidInput);
                    }
                };

                debug!("Creating new container {} inside exisitng sandbox {}", id, &sandboxId);
                let rootContainer = match Container::Load(&c.RootContainerDir, &sandboxId) {
                    Ok(container) => container,
                    Err(e) => {
                        error!("failed to load root container: {:?}", &e);
                        return Err(e);
                    }
                };
                c.Sandbox = rootContainer.Sandbox;

                // TODO: create placeholder cgroup paths for subcontainers,
                // althought it won't take effect, some tools use this for reporting and discovery

                // If the console control socket file is provided, then create a new
                // pty master/slave pair and send the TTY to the sandbox process.
                let tty = if c.ConsoleSocket.len() > 0 {
                    let (master, replicas) = NewPty()?;
                    let client = UnixSocket::NewClient(&c.ConsoleSocket)?;
                    client.SendFd(master.as_raw_fd())?;
                    replicas.as_raw_fd()
                } else {
                    -1
                };

                if let Err(e) = c.Sandbox.as_ref().unwrap().CreateSubContainer(conf, id, tty) {
                    error!("failed to create subcontainer: {:?}", e);
                }
                */
            }

            c.changeStatus(Status::Created);

            // Save the metadata file.
            let ret = c.Save();
            match ret {
                Err(e) => {
                    c.Destroy()?;
                    return Err(e);
                }
                Ok(_) => (),
            }

            // Write the PID file. Containerd considers the create complete after
            // this file is created, so it must be the last thing we do.
            if pidFile.len() != 0 {
                let id = format!("{}", c.SandboxPid());
                let ret = Self::WriteStr(pidFile, &id);
                match ret {
                    Err(e) => {
                        c.Destroy()?;
                        return Err(e);
                    }
                    Ok(_) => (),
                }
            }

            c
        };

        return Ok(c);
    }

    pub fn Create1(
        id: &str,
        action: RunAction,
        spec: Spec,
        conf: &GlobalConfig,
        bundleDir: &str,
        userlog: &str,
        io: &ContainerIO,
        pivot: bool,
    ) -> Result<Self> {
        info!(
            "Create container {} in root dir: {}, bundleDir {}",
            id, &conf.RootDir, bundleDir
        );
        //debug!("container spec is {:?}", &spec);
        ValidateID(id)?;

        let _unlockRoot = if !crate::QUARK_CONFIG.lock().Sandboxed {
            Some(maybeLockRootContainer(bundleDir, &spec, &conf.RootDir)?)
        } else {
            None
        };

        // Lock the container metadata file to prevent concurrent creations of
        // containers with the same id.
        let containerRoot = Join(&conf.RootDir, id);

        let c = {
            let _unlock = lockContainerMetadata(&containerRoot)?;

            // Check if the container already exists by looking for the metadata
            // file.
            if Path::new(&Join(&containerRoot, METADATA_FILENAME)).exists() {
                return Err(Error::Common(format!(
                    "container with id {} already exists",
                    id
                )));
            }

            let user = match env::var("USER") {
                Err(_) => "".to_string(),
                Ok(s) => s.to_string(),
            };

            let mut c = Self {
                ID: id.to_string(),
                Spec: spec,
                ConsoleSocket: "".to_string(),
                BundleDir: bundleDir.to_string(),
                Root: containerRoot,
                Status: Status::Creating,
                CreateAt: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                Owner: user,
                Sandbox: None,
                RootContainerDir: conf.RootDir.to_string(),
                sandboxed: false,
            };

            if crate::QUARK_CONFIG.lock().Sandboxed {
                let sandbox = crate::SANDBOX.lock();
                c.Sandbox = Some(Sandbox {
                    ID: sandbox.ID.to_string(),
                    Pid: sandbox.Pid,
                    ..Default::default()
                });
                c.sandboxed = true;
                c.Sandbox
                    .as_ref()
                    .unwrap()
                    .CreateSubContainer(conf, id, io)?;
            } else if IsRoot(&c.Spec) {
                // If the metadata annotations indicate that this container should be
                // started in an existing sandbox, we must do so. The metadata will
                // indicate the ID of the sandbox, which is the same as the ID of the
                // init container in the sandbox.
                debug!("Creating new sandbox for container {}", id);

                // Create and join cgroup before processes are created to ensure they are
                // part of the cgroup from the start (and all children processes).
                let mut cg: Option<Cgroup> = if crate::QUARK_CONFIG.lock().DisableCgroup {
                    None
                } else {
                    match Cgroup::New(&c.Spec) {
                        Err(e) => {
                            c.Destroy()?;
                            return Err(e);
                        }
                        Ok(cg) => cg,
                    }
                };
                if cg.is_some() {
                    let ret = cg
                        .as_mut()
                        .unwrap()
                        .Install(&c.Spec.linux.as_ref().unwrap().resources);
                    match ret {
                        Err(e) => {
                            c.Destroy()?;
                            return Err(e);
                        }
                        Ok(_) => (),
                    }
                }

                let restore = match &cg {
                    Some(ref cgroup) => {
                        let restore = cgroup.Join()?;
                        Some(restore)
                    }
                    None => None,
                };

                let ret = Sandbox::New1(id, action, conf, bundleDir, io, userlog, cg, pivot);

                c.Sandbox = match ret {
                    Err(e) => {
                        c.Destroy()?;
                        return Err(e);
                    }
                    Ok(s) => Some(s),
                };

                match restore {
                    None => (),
                    Some(restore) => restore(),
                }
            } else {
                let sandboxId = match SandboxID(&c.Spec) {
                    Some(sid) => sid,
                    None => {
                        error!(
                            "No sandbox ID found in spec when creating container inside sandbox"
                        );
                        return Err(Error::InvalidInput);
                    }
                };

                debug!(
                    "Creating new container {} inside exisitng sandbox {}",
                    id, &sandboxId
                );
                let rootContainer = match Container::Load(&c.RootContainerDir, &sandboxId) {
                    Ok(container) => container,
                    Err(e) => {
                        error!("failed to load root container: {:?}", &e);
                        return Err(e);
                    }
                };
                c.Sandbox = rootContainer.Sandbox;

                // TODO: create placeholder cgroup paths for subcontainers,
                // althought it won't take effect, some tools use this for reporting and discovery

                if let Err(e) = c.Sandbox.as_ref().unwrap().CreateSubContainer(conf, id, io) {
                    error!("failed to create subcontainer: {:?}", e);
                }
            }

            c.changeStatus(Status::Created);

            // Save the metadata file.
            let ret = c.Save();
            match ret {
                Err(e) => {
                    c.Destroy()?;
                    return Err(e);
                }
                Ok(_) => (),
            }

            c
        };

        return Ok(c);
    }

    // Run is a helper that calls Create + Start + Wait.
    pub fn Run(
        id: &str,
        spec: Spec,
        conf: &GlobalConfig,
        bundleDir: &str,
        consoleSocket: &str,
        pidFile: &str,
        userlog: &str,
        detach: bool,
        pivot: bool,
    ) -> Result<u32> {
        info!("Run container {} in root dir: {}", id, &conf.RootDir);

        let mut c = Self::Create(
            id,
            RunAction::Run,
            spec,
            conf,
            bundleDir,
            consoleSocket,
            pidFile,
            userlog,
            detach,
            pivot,
        )?;
        c.changeStatus(Status::Running);

        return c.Wait();
    }

    // Wait waits for the container to exit, and returns its WaitStatus.
    // Call to wait on a stopped container is needed to retrieve the exit status
    // and wait returns immediately.
    pub fn Wait(&mut self) -> Result<u32> {
        info!("Wait on container {}", &self.ID);
        let id = self.ID.to_string();
        let res = self.Sandbox.as_mut().unwrap().Wait(&id);
        return res;
    }

    pub fn WaitRootPID(&mut self, pid: i32, clearStatus: bool) -> Result<u32> {
        info!("Wait on pid {} container {}", pid, &self.ID);
        if !self.isSandboxRunning() {
            return Err(Error::Common("sandbox is not running".to_string()));
        }

        let id = self.Sandbox.as_ref().unwrap().ID.to_string();
        return self
            .Sandbox
            .as_mut()
            .unwrap()
            .WaitPID(&id, pid, clearStatus);
    }

    pub fn WaitPid(&mut self, pid: i32, clearStatus: bool) -> Result<u32> {
        let id = self.ID.to_string();

        return self
            .Sandbox
            .as_mut()
            .unwrap()
            .WaitPID(&id, pid, clearStatus);
    }

    pub fn Pause(&mut self) -> Result<()> {
        info!("Checkpoint container {}", self.ID);

        let _unlock = self.Lock()?;

        self.RequireStatus("Pause", &[Status::Running])?;

        self.Sandbox.as_ref().unwrap().Pause(&self.ID)?;
        self.changeStatus(Status::Paused);
        return self.Save();
    }

    pub fn Resume(&mut self) -> Result<()> {
        info!("Resume container {}", self.ID);

        let _unlock = self.Lock()?;

        self.RequireStatus("Resume", &[Status::Paused])?;

        self.Sandbox.as_ref().unwrap().Unpause(&self.ID)?;
        self.changeStatus(Status::Running);
        return self.Save();
    }

    pub fn Processes(&self) -> Result<Vec<ProcessInfo>> {
        self.RequireStatus("get processes of", &[Status::Running, Status::Paused])?;
        return self.Sandbox.as_ref().unwrap().Processes(&self.ID);
    }

    // Start starts running the containerized process inside the sandbox.
    pub fn Start(&mut self) -> Result<()> {
        info!("Start container {}", &self.ID);

        let _unlockRoot = if !self.sandboxed {
            Some(maybeLockRootContainer(
                &self.BundleDir,
                &self.Spec,
                &self.RootContainerDir,
            )?)
        } else {
            None
        };

        let _unlock = self.Lock()?;

        self.RequireStatus("start", &[Status::Created])?;
        // "If any prestart hook fails, the runtime MUST generate an error,
        // stop and destroy the container" -OCI spec.
        if self.Spec.hooks.is_some() {
            executeHooks(&self.Spec.hooks.as_ref().unwrap().prestart, &self.State())?;
        }

        if IsRoot(&self.Spec) {
            self.Sandbox.as_ref().unwrap().StartRootContainer()?;
        } else {
            if let Err(e) = self.Sandbox.as_ref().unwrap().StartSubContainer(
                &self.Spec,
                &self.ID[..],
                &self.BundleDir,
            ) {
                error!("Failed to start subcontainer, error : {:?}", &e);
                return Err(e);
            }
        }

        if self.Spec.hooks.is_some() {
            executeHooksBestEffort(&self.Spec.hooks.as_ref().unwrap().poststart, &self.State());
        }

        self.changeStatus(Status::Running);
        return self.Save();
    }

    pub fn WriteStr(file: &str, data: &str) -> Result<()> {
        let mut file = File::create(file)
            .map_err(|e| Error::Common(format!("Container::Create error is {:?}", e)))?;
        file.write_all(data.as_bytes())
            .map_err(|e| Error::Common(format!("Container::Create error is {:?}", e)))?;
        return Ok(());
    }

    pub fn Save(&self) -> Result<()> {
        info!("Save container {}", &self.ID);
        let metafile = Join(&self.Root, METADATA_FILENAME);
        serialize(self, &metafile)
            .map_err(|e| Error::Common(format!("Container::Save error is {:?}", e)))?;
        return Ok(());
    }

    pub fn Destroy(&mut self) -> Result<()> {
        info!("Destroy container {}", &self.ID);
        // We must perform the following cleanup steps:
        // * stop the container,
        // * remove the container filesystem on the host, and
        // * delete the container metadata directory.
        //
        // It's possible for one or more of these steps to fail, but we should
        // do our best to perform all of the cleanups. Hence, we keep a slice
        // of errors return their concatenation.
        info!("Find sandbox id for container {}", &self.ID);
        let mut sandboxId = self.ID.clone();
        if self.Sandbox.is_some() {
            sandboxId = self.Sandbox.as_mut().unwrap().ID.clone()
        }

        let mut errs = Vec::new();
        let _unlockRoot = if !self.sandboxed {
            Some(maybeLockRootContainer(
                &self.BundleDir,
                &self.Spec,
                &self.RootContainerDir,
            )?)
        } else {
            None
        };

        match self.Stop() {
            Err(e) => {
                info!(
                    "fail to stop container and uninstall cgroup: {}: {:?}",
                    &self.Root, &e
                );
                errs.push(format!(
                    "fail to stop container and uninstall cgroup: {} {:?}",
                    &self.Root, &e
                ));
            }
            Ok(_) => {
                info!("container process stopped");
            }
        }

        // Clean up rootfs mounts in the sandbox root directory.
        // This is a workaround to fix the issue that sandbox directory mounts
        // were not cleaned up after container removal. Ideally the mounts
        // should be done in a separate mount namespace, invisible to the host.
        let fsMounter = FsImageMounter::New(&sandboxId);
        let ret = fsMounter.UnmountContainerFs(&self.Spec, &self.ID);
        if ret.is_err() {
            info!(
                "umount fs for container {}, err: {}",
                self.ID,
                ret.err().unwrap()
            );
        }

        self.changeStatus(Status::Stopped);
        if self.Spec.hooks.is_some() {
            executeHooksBestEffort(&self.Spec.hooks.as_ref().unwrap().poststop, &self.State());
        }

        if errs.len() == 0 {
            return Ok(());
        }

        let mut errstr = "".to_string();
        for e in errs {
            errstr += &e;
            errstr += "\n";
        }

        return Err(Error::Common(errstr));
    }

    pub fn State(&self) -> State {
        return State {
            version: Version(),
            id: self.ID.to_string(),
            status: self.Status.String(),
            pid: self.SandboxPid(),
            bundle: self.BundleDir.to_string(),
            ..Default::default()
        };
    }

    pub fn SandboxPid(&self) -> i32 {
        match self.RequireStatus(
            "get PID",
            &[Status::Created, Status::Running, Status::Paused],
        ) {
            Err(_) => return -1,
            Ok(_) => return self.Sandbox.as_ref().unwrap().Pid,
        }
    }

    // stop stops the container (for regular containers) or the sandbox (for
    // root containers), and waits for the container or sandbox
    // to stop. If any of them doesn't stop before timeout, an error is returned.
    pub fn Stop(&mut self) -> Result<()> {
        let mut cgroup: Option<Cgroup> = None;

        if self.Sandbox.is_some() {
            info!("Destroying container {}", &self.ID);
            let sandbox = self.Sandbox.as_mut().unwrap();

            sandbox.DestroyContainer(&self.ID)?;

            // Only uninstall cgroup for sandbox stop.
            if sandbox.IsRootContainer(&self.ID) {
                let destroyed = sandbox.Destroy();
                cgroup = self.Sandbox.as_mut().unwrap().Cgroup.take();
                match destroyed {
                    Ok(()) => (),
                    Err(e) => return Err(e),
                }
            }

            // Only set sandbox to none after it has been told to destroy the container.
            self.Sandbox = None;
        }

        self.WaitforStopped()?;

        if cgroup.is_some() {
            cgroup.as_ref().unwrap().Uninstall();
        }

        return Ok(());
    }

    pub fn WaitforStopped(&self) -> Result<()> {
        if self.isSandboxRunning() {
            return self.SignalContainer(0, false);
        }

        return Ok(());
    }

    pub fn Execute(&mut self, mut args: ExecArgs, execCmd: &mut ExecCmd) -> Result<u32> {
        info!("Execute in container {}, args {:?}", &self.ID, args);

        self.RequireStatus("execute in", &[Status::Created, Status::Running])?;

        let terminal = args.Terminal;
        args.ContainerID = self.ID.to_string();

        if !execCmd.clearStatus {
            args.Detach = true;
        }

        let pid = self.Sandbox.as_ref().unwrap().Execute(args)?;

        if terminal {
            self.ForwardSignals(pid);
        }

        if execCmd.internalPidFile.len() > 0 {
            let pidStr = format!("{}", pid);
            Self::WriteStr(&execCmd.internalPidFile, &pidStr)?;
        }

        if execCmd.pid.len() > 0 {
            let currPid = unsafe { libc::getpid() };

            assert!(
                currPid > 0,
                "Container execute get current pid fail with error {}",
                errno::errno().0
            );
            let currPidStr = format!("{}", currPid);
            Self::WriteStr(&execCmd.pid, &currPidStr)?;
        }

        let ret = self.WaitPid(pid, execCmd.clearStatus);

        if terminal {
            self.StopSignal();
        }
        return ret;
    }
}

pub fn runInCgroup(cg: &Option<Cgroup>, mut f: impl FnMut() -> Result<()>) -> Result<()> {
    if cg.is_none() {
        return f();
    }

    let restore = cg.as_ref().unwrap().Join()?;
    f()?;
    restore();
    return Ok(());
}

#[derive(Serialize, Deserialize, Default, Debug)]
pub struct ExecArgs {
    pub Argv: Vec<String>,
    pub Envv: Vec<String>,
    pub Root: String,
    pub WorkDir: String,
    pub KUID: KUID,
    pub KGID: KGID,
    pub ExtraKGIDs: Vec<KGID>,
    pub Capabilities: TaskCaps,
    pub Terminal: bool,
    pub ContainerID: String,
    pub Detach: bool,
    pub ConsoleSocket: String,
    pub ExecId: String,

    #[serde(default, skip_serializing, skip_deserializing)]
    pub Fds: Vec<i32>,
}

impl FileDescriptors for ExecArgs {
    fn GetFds(&self) -> Option<&[i32]> {
        return Some(&self.Fds);
    }

    fn SetFds(&mut self, fds: &[i32]) {
        for fd in fds {
            self.Fds.push(*fd)
        }
    }
}
