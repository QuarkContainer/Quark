/*
   Copyright The containerd Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

use std::fs::File;
use std::sync::mpsc::SyncSender;
//use std::fs::OpenOptions;
use std::path::Path;
//use std::os::unix::io::AsRawFd;
//use std::os::unix::io::FromRawFd;
use core::convert::TryFrom;
use nix::ioctl_write_ptr_bad;
use nix::sys::termios::Termios;
//use nix::sys::termios::tcgetattr;

use containerd_shim::api::ExecProcessRequest;
use containerd_shim::api::StateResponse;
use containerd_shim::protos::protobuf::well_known_types::Timestamp;
use containerd_shim::protos::types::task::Status;
use containerd_shim::util::read_pid_from_file;
use containerd_shim::Error;
use containerd_shim::Result;
use time::OffsetDateTime;

ioctl_write_ptr_bad!(ioctl_set_winsz, libc::TIOCSWINSZ, libc::winsize);

use super::super::super::qlib::path::*;
use super::super::cmd::config::*;
use super::super::container::container::*;
use super::super::oci;
use super::super::oci::Spec;
use super::container_io::*;

pub struct Console {
    pub file: File,
    pub termios: Termios,
}

#[derive(Default)]
pub struct CommonProcess {
    pub state: Status,
    pub id: String,
    pub stdio: ContainerStdio,
    pub pid: i32,
    pub exit_code: i32,
    pub exited_at: Option<OffsetDateTime>,
    pub wait_chan_tx: Vec<SyncSender<i8>>,
    pub containerIO: ContainerIO,
}

impl Drop for CommonProcess {
    fn drop(&mut self) {
        self.stdio = ContainerStdio::default();
        self.containerIO = ContainerIO::default();
        error!("CommonProcess drop 3 {}", self.id);
    }
}

impl CommonProcess {
    pub fn CopyIO(&self, cid: &str, pid: i32, sandboxId: String) -> Result<()> {
        return self
            .containerIO
            .CopyIO(&self.stdio, cid, pid, sandboxId)
            .map_err(|e| Error::Other(format!("{:?}", e)));
    }

    pub fn set_exited(&mut self, exit_code: i32) {
        self.state = Status::STOPPED;
        self.exit_code = exit_code;
        self.exited_at = Some(OffsetDateTime::now_utc());
        // set wait_chan_tx to empty, to trigger the drop of the initialized Receiver.
        self.wait_chan_tx = vec![];
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn status(&self) -> Status {
        self.state
    }

    pub fn set_status(&mut self, status: Status) {
        self.state = status;
    }

    pub fn pid(&self) -> i32 {
        self.pid
    }

    pub fn terminal(&self) -> bool {
        self.stdio.terminal
    }

    pub fn stdin(&self) -> String {
        self.stdio.stdin.to_string()
    }

    pub fn stdout(&self) -> String {
        self.stdio.stdout.to_string()
    }

    pub fn stderr(&self) -> String {
        self.stdio.stderr.to_string()
    }

    pub fn state(&self) -> StateResponse {
        let mut resp = StateResponse::new();
        resp.id = self.id.to_string();
        resp.status = self.state;
        resp.pid = self.pid as u32;
        resp.terminal = self.stdio.terminal;
        resp.stdin = self.stdio.stdin.to_string();
        resp.stdout = self.stdio.stdout.to_string();
        resp.stderr = self.stdio.stderr.to_string();
        resp.exit_status = self.exit_code as u32;
        if let Some(exit_at) = self.exited_at {
            let mut time_stamp = Timestamp::new();
            time_stamp.set_seconds(exit_at.unix_timestamp());
            time_stamp.set_nanos(exit_at.nanosecond() as i32);
            resp.set_exited_at(time_stamp);
        }
        resp
    }

    pub fn add_wait(&mut self, tx: SyncSender<i8>) {
        self.wait_chan_tx.push(tx)
    }

    pub fn exit_code(&self) -> i32 {
        self.exit_code
    }

    pub fn exited_at(&self) -> Option<OffsetDateTime> {
        self.exited_at
    }

    pub fn set_pid_from_file(&mut self, pid_path: &Path) -> Result<()> {
        let pid = read_pid_from_file(pid_path)?;
        self.pid = pid;
        Ok(())
    }

    pub fn resize_pty(&mut self, height: u32, width: u32) -> Result<()> {
        return self
            .containerIO
            .ResizePty(height, width)
            .map_err(|e| Error::Other(format!("resize_pty {:?}", e)));
    }
}

pub enum Process<'a> {
    InitProcess(&'a InitProcess),
    ExecProcess(&'a ExecProcess),
}

pub struct ExecProcess {
    pub common: CommonProcess,
    pub spec: oci::Process,
}

impl ExecProcess {
    pub fn set_exited(&mut self, exit_code: i32) {
        self.common.set_exited(exit_code);
    }

    pub fn id(&self) -> &str {
        self.common.id()
    }

    pub fn status(&self) -> Status {
        self.common.status()
    }

    pub fn set_status(&mut self, status: Status) {
        self.common.set_status(status)
    }

    pub fn pid(&self) -> i32 {
        self.common.pid()
    }

    pub fn terminal(&self) -> bool {
        self.common.terminal()
    }

    pub fn stdin(&self) -> String {
        self.common.stdin()
    }

    pub fn stdout(&self) -> String {
        self.common.stdout()
    }

    pub fn stderr(&self) -> String {
        self.common.stderr()
    }

    pub fn state(&self) -> StateResponse {
        self.common.state()
    }

    pub fn add_wait(&mut self, tx: SyncSender<i8>) {
        self.common.add_wait(tx)
    }

    pub fn exit_code(&self) -> i32 {
        self.common.exit_code()
    }

    pub fn exited_at(&self) -> Option<OffsetDateTime> {
        self.common.exited_at()
    }

    pub fn set_pid_from_file(&mut self, pid_path: &Path) -> Result<()> {
        self.common.set_pid_from_file(pid_path)
    }

    pub fn resize_pty(&mut self, height: u32, width: u32) -> Result<()> {
        self.common.resize_pty(height, width)
    }
}

impl TryFrom<ExecProcessRequest> for ExecProcess {
    type Error = Error;
    fn try_from(req: ExecProcessRequest) -> Result<Self> {
        let p = get_spec_from_request(&req)?;
        let exec_process = ExecProcess {
            common: CommonProcess {
                state: Status::CREATED,
                id: req.exec_id,
                stdio: ContainerStdio {
                    stdin: req.stdin,
                    stdout: req.stdout,
                    stderr: req.stderr,
                    terminal: req.terminal,
                },
                pid: 0,
                exit_code: 0,
                exited_at: None,
                wait_chan_tx: vec![],
                containerIO: ContainerIO::None,
            },
            spec: p,
        };
        Ok(exec_process)
    }
}

pub fn get_spec_from_request(req: &ExecProcessRequest) -> Result<oci::Process> {
    if let Some(val) = req.spec.as_ref() {
        let mut p = serde_json::from_slice::<oci::Process>(val.get_value())
            .map_err(|e| Error::Other(format!("{:?}", e)))?;
        p.terminal = req.terminal;
        Ok(p)
    } else {
        Err(Error::InvalidArgument("no spec in request".to_string()))
    }
}

#[derive(Default)]
pub struct InitProcess {
    pub common: CommonProcess,
    pub bundle: String,
    pub rootfs: String,
    pub work_dir: String,
    pub io_uid: u32,
    pub io_gid: u32,
    pub no_pivot_root: bool,
    pub no_new_key_ring: bool,
    pub criu_work_path: String,
}

impl InitProcess {
    pub fn New(id: &str, bundle: &str, stdio: ContainerStdio) -> Self {
        let containerIO = stdio.CreateIO().unwrap();
        InitProcess {
            common: CommonProcess {
                state: Status::CREATED,
                id: id.to_string(),
                stdio,
                pid: 0,
                exit_code: 0,
                exited_at: None,
                wait_chan_tx: vec![],
                containerIO: containerIO,
            },
            bundle: bundle.to_string(),
            rootfs: "".to_string(),
            work_dir: "".to_string(),
            io_uid: 0,
            io_gid: 0,
            no_pivot_root: false,
            no_new_key_ring: false,
            criu_work_path: "".to_string(),
        }
    }

    pub fn Create(&self, config: &GlobalConfig) -> Result<Container> {
        let specfile = Join(&self.bundle, "config.json");
        let spec = Spec::load(&specfile).unwrap();
        let container = Container::Create1(
            &self.common.id,
            RunAction::Create,
            spec,
            config,
            &self.bundle,
            "",
            &self.common.containerIO,
            !self.no_pivot_root,
        )
        .map_err(|e| Error::Other(format!("{:?}", e)))?;
        let sandboxId = container.Sandbox.as_ref().unwrap().ID.clone();
        self.common.CopyIO(&*container.ID, 0, sandboxId)?;
        return Ok(container);
    }

    pub fn pid(&self) -> i32 {
        self.common.pid()
    }
}
